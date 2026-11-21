"""Run the supplied final MLP weights on an illustrative sleep history."""
import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from paths import REPO, model_file


def positive_minutes(value, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Time values must be numeric minutes.")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (result == 0 and not allow_zero):
        raise ValueError("Time values must be finite and positive (latency may be zero).")
    if result > 1440:
        raise ValueError("A sleep interval or latency cannot exceed 1440 minutes in this demo.")
    return result


def history_features(history):
    if not isinstance(history, list):
        raise ValueError("history must list records in chronological order.")
    latency, duration = [], []
    previous_duration = np.nan
    for record in history:
        if not isinstance(record, dict):
            raise ValueError("Each history record must be an object.")
        a, d = record.get("bed_to_sleep_min"), record.get("sleep_interval_min")
        if a is not None:
            latency.append(positive_minutes(a, allow_zero=True))
        previous_duration = np.nan if d is None else positive_minutes(d)
        if d is not None:
            duration.append(previous_duration)
    return {
        "past3_latency_median_min": float(np.median(latency[-3:])) if latency else np.nan,
        "past3_latency_n": min(len(latency), 3),
        "prev_sleep_interval_min": previous_duration,
        "past3_sleep_interval_median_min": float(np.median(duration[-3:])) if duration else np.nan,
        "past3_sleep_interval_n": min(len(duration), 3),
    }


def predict_saved(saved, frame):
    """The same float32 inference and seed averaging as mlp.py, without torch."""
    if saved["model"] != "mlp":
        raise ValueError("This demonstration expects the published final MLP models.")
    x = saved["scaler"].transform(frame[saved["features"]]).astype(np.float32)
    outputs = []
    for weights in saved["weights"]:
        h = x
        for layer in ("0", "3"):
            h = np.maximum(h @ weights[layer + ".weight"].T + weights[layer + ".bias"], 0)
        h = (h @ weights["5.weight"].T + weights["5.bias"]).ravel()
        outputs.append(np.logaddexp(np.float32(0), h) if saved["task"] == "A"
                       else 1 / (1 + np.exp(-h)))
    average = np.mean(outputs, axis=0)
    return average * 30 if saved["task"] == "A" else 1 + average * 4


def run_example(example):
    features = history_features(example["history"])
    wake = datetime.strptime(example["wake_time"], "%H:%M")
    minimum = positive_minutes(example["minimum_sleep_interval_minutes"])
    intervals = example["candidate_sleep_intervals_minutes"]
    if not isinstance(intervals, list) or not intervals:
        raise ValueError("Provide at least one candidate sleep interval.")
    intervals = [positive_minutes(value) for value in intervals]
    model_a = joblib.load(model_file("A", "selected"))
    model_b = joblib.load(model_file("B", "selected"))
    latency = float(predict_saved(model_a, pd.DataFrame([features]))[0])
    if not math.isfinite(latency):
        raise ValueError("Model A returned a non-finite value.")
    candidates = []
    b_inputs = pd.DataFrame([{**features, "sleep_interval_min": minutes} for minutes in intervals])
    scores = predict_saved(model_b, b_inputs)
    for minutes, score in zip(intervals, scores):
        bedtime_total = round(wake.hour * 60 + wake.minute - minutes - latency)
        day, clock_minute = divmod(bedtime_total, 1440)
        candidates.append({
            "sleep_interval_minutes": minutes,
            "predicted_quality": float(score),
            "bedtime": f"{clock_minute // 60:02}:{clock_minute % 60:02}",
            "bedtime_day_relative_to_wake": day,
            "meets_minimum_interval": minutes >= minimum,
        })
    eligible = [candidate for candidate in candidates if candidate["meets_minimum_interval"]]
    if not eligible:
        raise ValueError("No candidate meets minimum_sleep_interval_minutes.")
    return {
        "input_type": "illustrative_history",
        "models": {task: model_file(task, "selected").relative_to(REPO).as_posix() for task in ("A", "B")},
        "features": {key: None if not math.isfinite(value) else value for key, value in features.items()},
        "wake_time": example["wake_time"],
        "predicted_latency_minutes": latency,
        "candidates": candidates,
        "selected_candidate": max(eligible, key=lambda item: item["predicted_quality"]),
        "note": "Model behavior example. Sleep interval includes nocturnal awakenings; efficacy of changing bedtime was not tested.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=REPO / "예측시연" / "입력예시.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        example = json.loads(args.input.read_text(encoding="utf-8-sig"))
        result = run_example(example)
    except (KeyError, ValueError, TypeError) as error:
        parser.error(str(error))
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
