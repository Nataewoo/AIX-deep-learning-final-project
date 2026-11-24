"""Check saved model predictions and compute paired participant-bootstrap intervals."""
import json

import joblib
import numpy as np
import pandas as pd
import torch

from evaluate import DATA_DIR, ROOT, predict
from paths import TEST_DIR, model_file
from mlp import network


def verify_history(data_dir):
    for name in ("train_melidos_all", "test_melidos", "test_korea"):
        rows = pd.read_csv(data_dir / f"{name}.csv").dropna(subset=["night_date"])
        for _, group in rows.groupby("participant_id"):
            group = group.sort_values("night_date")
            for row in group.itertuples():
                past = group[group.night_date < row.night_date]
                latency = past.bed_to_sleep_min.dropna().tail(3)
                duration = past.sleep_interval_min.dropna().tail(3)
                actual = [row.past3_latency_n, row.past3_latency_median_min,
                          row.past3_sleep_interval_n, row.past3_sleep_interval_median_min,
                          row.prev_sleep_interval_min]
                expected = [len(latency), latency.median(), len(duration), duration.median(),
                            past.sleep_interval_min.iloc[-1] if len(past) else np.nan]
                np.testing.assert_allclose(actual, expected, equal_nan=True)


def verify_saved_models(data_dir, results_dir, scored):
    files = {
        "melidos_holdout": ("test_melidos.csv", None),
        "korea_last7": ("test_korea.csv", "is_korea_final7"),
        "korea_all": ("test_korea.csv", None),
    }
    for task in ("A", "B"):
        for role in ("selected", "mlp"):
            saved = joblib.load(model_file(task, role))
            model = saved.get("estimator")
            if saved["model"] == "mlp":
                nets = []
                for weights in saved["weights"]:
                    net = network(task, len(saved["features"]), saved["params"]["width"])
                    net.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()})
                    nets.append(net.eval())
                model = saved["scaler"], nets
            rows = scored[(scored.task == task) & (scored.role == role)]
            keys = ["participant_id", "site_code", "night_date"]
            for dataset, expected in rows.groupby("dataset"):
                filename, marker = files[dataset]
                data = pd.read_csv(data_dir / filename)
                if marker:
                    data = data[data[marker] == 1]
                inputs = expected[keys].merge(data, on=keys, validate="one_to_one", how="left")
                actual = predict(task, saved["model"], model, inputs, saved["features"])
                np.testing.assert_allclose(actual, expected.predicted, atol=1e-4)


def paired_intervals(scored):
    checks = [
        ("A", "melidos_holdout", "baseline"),
        ("A", "korea_last7", "baseline"),
        ("A", "korea_all", "baseline"),
        ("B", "melidos_holdout", "baseline"),
        ("B", "melidos_holdout", "without_duration"),
    ]
    rng = np.random.default_rng(42)
    out = []
    for task, dataset, ref in checks:
        d = scored.loc[(scored.task == task) & (scored.dataset == dataset)].copy()
        d["ae"] = d.error.abs()
        scores = d.groupby(["participant_id", "role"]).ae.mean().unstack()
        gap = (scores["selected"] - scores[ref]).to_numpy()
        samples = rng.choice(gap, (10000, len(gap)), replace=True).mean(axis=1)
        lower, upper = np.quantile(samples, [0.025, 0.975])
        out.append({"task": task, "dataset": dataset, "comparison": ref,
                    "delta_macro_mae": float(gap.mean()), "lower_95": float(lower),
                    "upper_95": float(upper), "people": len(gap)})
    return out


def main():
    torch.set_num_threads(1)
    out = TEST_DIR
    scored = pd.read_csv(out / "holdout_predictions.csv")
    verify_history(DATA_DIR)
    verify_saved_models(DATA_DIR, out, scored)
    intervals = paired_intervals(scored)
    pd.DataFrame(intervals).to_csv(out / "paired_intervals.csv", index=False)
    print("Past-only features and saved model predictions verified.")
    print(json.dumps(intervals, indent=2))


if __name__ == "__main__":
    main()
