"""Regenerate the published five participant folds without drawing new groups."""
import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from paths import CV_DIR, REPO, SPLIT_DIR


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / ".recheck" / "splits")
    args = parser.parse_args()
    fields, pool = read_csv(SPLIT_DIR / "train_melidos.csv")
    groups_path = CV_DIR / "cv_groups.csv"
    _, groups = read_csv(groups_path)
    assignments = {row["participant_id"]: int(row["fold"]) for row in groups}
    people = {row["participant_id"] for row in pool}
    assert len(assignments) == len(groups) == 147
    assert set(assignments) == people
    assert set(assignments.values()) == set(range(1, 6))
    args.output.mkdir(parents=True, exist_ok=True)
    validation_count = Counter()
    counts = {}
    for fold in range(1, 6):
        for kind in ("train", "validation"):
            rows = [row for row in pool
                    if (assignments[row["participant_id"]] == fold) == (kind == "validation")]
            ids = {row["participant_id"] for row in rows}
            if kind == "validation":
                validation_count.update(ids)
            name = f"{kind}_fold_{fold}.csv"
            with (args.output / name).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            counts[name] = {"participants": len(ids), "rows": len(rows)}
    assert set(validation_count) == people
    assert set(validation_count.values()) == {1}
    for name in ("train_melidos.csv", "test_melidos.csv", "test_korea.csv"):
        source, dest = SPLIT_DIR / name, args.output / name
        if source.resolve() != dest.resolve():
            shutil.copyfile(source, dest)
        _, rows = read_csv(source)
        counts[name] = {"participants": len({row["participant_id"] for row in rows}), "rows": len(rows)}
    metadata = json.loads((SPLIT_DIR / "split_info.json").read_text(encoding="utf-8-sig"))
    metadata.update(fold_assignment_source=groups_path.relative_to(REPO).as_posix(),
                    fold_assignment_sha256=hashlib.sha256(groups_path.read_bytes()).hexdigest(),
                    participant_folds=assignments)
    (args.output / "split_info.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
