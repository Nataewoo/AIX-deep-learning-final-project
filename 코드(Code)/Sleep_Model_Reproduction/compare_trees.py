"""Compare Random Forest and XGBoost using the original folds and evaluation rows."""
import json

import numpy as np
import pandas as pd
from joblib import load

import xgboost_regressor
from evaluate import FEATURES, GRID, ROOT, TARGET, eligible, fit, metrics, predict, save_model


def train_model(task, kind, params, rows, features):
    if kind == "random_forest":
        return fit(task, kind, params, rows, features)[0]
    return xgboost_regressor.fit(task, params, rows[features], rows[TARGET[task]])


def search(task, kind, grid, rows, features):
    results = []
    for params in grid:
        oof = pd.Series(index=rows.index, dtype=float)
        for fold in range(1, 6):
            train = rows[rows.fold != fold]
            val = rows[rows.fold == fold]
            assert not set(train.participant_id) & set(val.participant_id)
            model = train_model(task, kind, params, train, features)
            oof.loc[val.index] = predict(task, kind, model, val, features)
        assert oof.notna().all()
        results.append({"task": task, "model": kind, "params": json.dumps(params),
                        **metrics(rows, task, oof.to_numpy())})
    return sorted(results, key=lambda row: row["macro_mae"])


def main():
    original = ROOT / "results"
    out = ROOT / "tree_comparison"
    out.mkdir(exist_ok=True)
    train_all = pd.read_csv(ROOT / "data/train_melidos.csv").merge(
        pd.read_csv(original / "cv_groups.csv"), on="participant_id", validate="many_to_one"
    )
    candidates = {"random_forest": GRID["random_forest"]}
    try:
        import xgboost
    except ImportError:
        print("XGBoost unavailable; running Random Forest only.")
    else:
        candidates["xgboost"] = xgboost_regressor.GRID
        print("XGBoost", xgboost.__version__)
    cv, choices = [], []
    original_cv = pd.read_csv(original / "cv_summary.csv")
    for task in ("A", "B"):
        features = FEATURES + (["sleep_interval_min"] if task == "B" else [])
        for kind, grid in candidates.items():
            configurations = search(task, kind, grid, eligible(train_all, task), features)
            cv.extend(configurations)
            best = configurations[0]
            if kind == "random_forest":
                prior = original_cv.query("task == @task and variant == 'full' and model == @kind").iloc[0]
                np.testing.assert_allclose(best["macro_mae"], prior.macro_mae, atol=1e-9)
            choices.append({"task": task, "model": kind, "params": json.loads(best["params"]),
                            "features": features, "cv_macro_mae": best["macro_mae"]})
    pd.DataFrame(cv).to_csv(out / "cv_summary.csv", index=False)
    (out / "selection.json").write_text(json.dumps(choices, indent=2), encoding="utf-8")

    melidos = pd.read_csv(ROOT / "data/validation_melidos.csv")
    korea = pd.read_csv(ROOT / "data/test_korea.csv")
    previous = pd.read_csv(original / "holdout_predictions.csv")
    previous = previous[previous.role.isin(["baseline", "selected"])]
    scores, predictions, intervals = [], [], []
    rng = np.random.default_rng(42)
    for choice in choices:
        task, kind, params, features = (choice[key] for key in ("task", "model", "params", "features"))
        final = train_model(task, kind, params, eligible(train_all, task), features)
        path = out / f"model_{task}_{kind}.joblib"
        save_model(path, task, choice, final)
        saved = load(path)["estimator"]
        datasets = {"melidos_holdout": eligible(melidos, task)}
        if task == "A":
            datasets.update(korea_last7=korea[korea.is_korea_final7 == 1], korea_all=korea)
        for dataset, subset in datasets.items():
            pred = predict(task, kind, final, subset, features)
            np.testing.assert_allclose(pred, predict(task, kind, saved, subset, features))
            scores.append({"task": task, "dataset": dataset, "model": kind, **metrics(subset, task, pred)})
            keys = ["participant_id", "night_date"]
            detail = subset[keys + [TARGET[task]]].rename(columns={TARGET[task]: "actual"}).copy()
            detail["prediction"] = pred
            detail["error"] = pred - detail.actual
            predictions.append(detail.assign(task=task, dataset=dataset, model=kind))
            for role in ("baseline", "selected"):
                ref = previous.query("task == @task and dataset == @dataset and role == @role")
                paired = detail.merge(ref[keys + ["actual", "error"]], on=keys, validate="one_to_one")
                assert len(paired) == len(detail) == len(ref)
                np.testing.assert_allclose(paired.actual_x, paired.actual_y)
                delta = (paired.error_x.abs() - paired.error_y.abs()).groupby(paired.participant_id).mean()
                draws = rng.choice(delta.to_numpy(), size=(10000, len(delta)), replace=True).mean(axis=1)
                lo, hi = np.quantile(draws, [0.025, 0.975])
                intervals.append({"task": task, "dataset": dataset, "model": kind,
                                  "reference": "mlp" if role == "selected" else role,
                                  "delta_macro_mae": delta.mean(), "lower_95": lo, "upper_95": hi})
    pd.DataFrame(scores).to_csv(out / "holdout_metrics.csv", index=False)
    pd.concat(predictions).to_csv(out / "holdout_predictions.csv", index=False)
    pd.DataFrame(intervals).to_csv(out / "paired_intervals.csv", index=False)
    print(pd.DataFrame(scores).to_string(index=False))


if __name__ == "__main__":
    main()
