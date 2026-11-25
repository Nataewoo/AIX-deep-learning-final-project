"""Rebuild numeric sleep features from the source records selected by the team package."""
import argparse
import csv
import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "past3_latency_median_min", "past3_latency_n", "prev_sleep_interval_min",
    "past3_sleep_interval_median_min", "past3_sleep_interval_n",
]


@lru_cache(None)
def records(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        delimiter = ";" if file.readline().count(";") > 5 else ","
        file.seek(0)
        return list(csv.DictReader(file, delimiter=delimiter))


def get(row, key):
    value = str(row.get(key) or row.get(key + "_v2") or "").strip()
    return "" if value.lower() in {"na", "n/a", "nan", "null"} else value


def timestamp(value):
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S",
                "%d.%m.%Y %H:%M", "%d.%m.%Y %H:%M:%S",
                "%m/%d/%Y %H:%M", "%m/%d/%Y %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    return None


def number(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


def clock(value):
    stamp = datetime.strptime(value, "%H:%M:%S")
    return stamp.hour * 60 + stamp.minute + stamp.second / 60


def extract(raw_dir, split_file, name):
    index = pd.read_csv(split_file)
    rows = []
    for item in index.itertuples(index=False):
        source_path = raw_dir / Path(item.source_file).relative_to("raw")
        raw = records(source_path)[item.source_record_number - 1]
        quality = np.nan
        latency = duration = np.nan
        date = item.night_date
        if name == "test_korea":
            assert date == get(raw, "date")
            assert item.participant_id == "korea:" + get(raw, "userId")
            bed, onset, wake = (clock(get(raw, key)) for key in ("go2bed", "asleep", "wakeup"))
            latency, duration = (onset - bed) % 1440, (wake - onset) % 1440
            assert np.isclose(latency, 60 * number(raw["sleep_latency"]))
            assert np.isclose(duration, 60 * number(raw["sleep_duration"]))
            assert np.isclose(latency, item.bed_to_sleep_min)
        else:
            bed, attempt, wake = (timestamp(get(raw, key)) for key in ("bedtime", "sleep", "offset"))
            for key, parsed in zip(("bedtime", "sleep", "offset"), (bed, attempt, wake)):
                if get(raw, key) and parsed is None:
                    raise ValueError(f"Unparsed {key}={get(raw, key)!r}: {item.source_file}:{item.source_record_number}")
            if wake is not None:
                assert str(wake.date()) == date, (item.source_file, date, wake)
            delay = number(get(raw, "sleepdelay"))
            if bed and attempt and attempt >= bed and np.isfinite(delay) and delay >= 0:
                latency = (attempt - bed).total_seconds() / 60 + delay
            if pd.notna(item.bed_to_sleep_min):
                assert np.isclose(latency, item.bed_to_sleep_min), (item.source_file, item.source_record_number, latency, item.bed_to_sleep_min)
            if bed and attempt and wake and np.isfinite(latency):
                total = (wake - bed).total_seconds() / 60
                if 0 < total < 1440 and latency < total:
                    duration = total - latency
                else:
                    latency = np.nan
            if pd.isna(date):
                latency = duration = np.nan
            quality = number(get(raw, "sleepquality"))
            if not np.isfinite(quality):
                marks = [i for i in range(1, 6) if number(get(raw, f"sleepquality___{i}")) == 1]
                quality = marks[0] if len(marks) == 1 else np.nan
            if quality not in range(1, 6):
                quality = np.nan
            assert np.isclose(quality, item.sleep_quality, equal_nan=True), (item.source_file, item.source_record_number, quality, item.sleep_quality)
        rows.append({
            "participant_id": item.participant_id, "site_code": item.site_code,
            "night_date": date, "source_file": item.source_file,
            "source_record_number": item.source_record_number,
            "bed_to_sleep_min": latency, "sleep_interval_min": duration,
            "sleep_quality": quality, "is_korea_final7": item.is_korea_final7,
        })
    df = pd.DataFrame(rows).sort_values(["participant_id", "night_date"]).reset_index(drop=True)
    dated = df[df.night_date.notna()]
    assert not dated.duplicated(["participant_id", "night_date"]).any()
    for col in FEATURES:
        df[col] = np.nan
    for _, group in dated.groupby("participant_id", sort=False):
        latency_history, duration_history = [], []
        previous_duration = np.nan
        for row in group.itertuples():
            latency_window = latency_history[-3:]
            duration_window = duration_history[-3:]
            df.loc[row.Index, FEATURES] = [
                np.median(latency_window) if latency_window else np.nan, len(latency_window),
                previous_duration, np.median(duration_window) if duration_window else np.nan,
                len(duration_window),
            ]
            if pd.notna(row.bed_to_sleep_min):
                latency_history.append(row.bed_to_sleep_min)
            if pd.notna(row.sleep_interval_min):
                duration_history.append(row.sleep_interval_min)
            previous_duration = row.sleep_interval_min
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--repo-layout", action="store_true",
                        help="Read the repository's 데이터(Dataset) directory instead of source_data")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()
    raw_dir, splits_dir = ("원본", "분할") if args.repo_layout else ("raw", "splits")
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {}
    all_ids = set()
    for name in ("train_melidos", "validation_melidos", "test_korea"):
        split_name = "test_melidos" if args.repo_layout and name == "validation_melidos" else name
        df = extract(args.package / raw_dir, args.package / splits_dir / f"{split_name}.csv", name)
        ids = set(df.participant_id)
        assert not all_ids & ids, "Participant appears in more than one split"
        all_ids.update(ids)
        df.to_csv(args.output / (name + ".csv"), index=False)
        summary[name] = {
            "rows": len(df), "participants": len(ids),
            "A_eligible": int(df.bed_to_sleep_min.notna().sum()),
            "B_eligible": int(df[["sleep_interval_min", "sleep_quality"]].notna().all(axis=1).sum()),
            "missing": {col: int(df[col].isna().sum()) for col in FEATURES + ["sleep_interval_min", "bed_to_sleep_min", "sleep_quality"]},
        }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
