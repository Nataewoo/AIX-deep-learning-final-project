"""Select A/B models by grouped CV, then score the frozen models on holdouts."""
import argparse
import json
import platform
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
import torch
from matplotlib import pyplot as plt

import baseline
import gradient_boosting
import mlp
import random_forest
import ridge_regression
from paths import (REPO, DATA_DIR, RESULTS_DIR, CV_DIR, TEST_DIR, FINAL_MODELS,
                   COMPARISON_MODELS, FIGURE_DIR, ENV_DIR)
from prepare import FEATURES
ROOT = REPO
TARGET = {"A": "bed_to_sleep_min", "B": "sleep_quality"}
SEEDS = (42, 7, 17)
GRID = {
    "baseline": [{}],
    "ridge": [{"alpha": a} for a in (0.1, 1, 10, 100)],
    "random_forest": [{"max_depth": d, "min_samples_leaf": n} for d in (3, 6) for n in (5, 15)],
    "gradient_boosting": [{"max_depth": d, "n_estimators": n} for d in (1, 2) for n in (60, 150)],
    "mlp": [{"width": w, "epochs": e} for w in (16, 32) for e in (60, 160)],
}


def fit(task, kind, params, rows, features, seeds=SEEDS, monitor=None):
    inputs = rows[features]
    targets = rows[TARGET[task]].to_numpy()
    if kind == "baseline":
        return baseline.fit(task, targets), []
    if kind == "ridge":
        return ridge_regression.fit(params, inputs, targets), []
    if kind == "random_forest":
        return random_forest.fit(params, inputs, targets), []
    if kind == "gradient_boosting":
        return gradient_boosting.fit(task, params, inputs, targets), []
    if kind == "mlp":
        if monitor is None:
            return mlp.fit(task, params, inputs, targets, seeds)
        return mlp.fit(
            task, params, inputs, targets, seeds, monitor_inputs=monitor[features],
            validation_score=lambda pred: metrics(monitor, task, pred)["macro_mae"],
        )
    raise ValueError(kind)


def predict(task, kind, fitted, rows, features):
    if kind == "baseline":
        return baseline.predict(task, fitted, rows)
    if kind == "mlp":
        return mlp.predict(task, fitted, rows[features])
    pred = fitted.predict(rows[features])
    return np.clip(pred, 0, None) if task == "A" else np.clip(pred, 1, 5)


def metrics(rows, task, pred):
    error = np.asarray(pred) - rows[TARGET[task]].to_numpy()
    absolute = pd.Series(np.abs(error), index=rows.index)
    scores = {
        "n": len(rows), "people": rows.participant_id.nunique(),
        "macro_mae": float(absolute.groupby(rows.participant_id).mean().mean()),
        "mae": float(absolute.mean()), "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bias_pred_minus_actual": float(error.mean()),
        "within_tolerance": float((absolute <= (10 if task == "A" else 1)).mean()),
    }
    return scores


def folds_for(train_all, data_dir=DATA_DIR):
    fold = pd.Series(index=train_all.index, dtype="int64")
    all_ids = set(train_all.participant_id)
    seen = set()
    for i in range(1, 6):
        train = pd.read_csv(data_dir / f"train_fold_{i}.csv")
        valid = pd.read_csv(data_dir / f"validation_fold_{i}.csv")
        train_ids, valid_ids = set(train.participant_id), set(valid.participant_id)
        assert not train_ids & valid_ids
        assert train_ids | valid_ids == all_ids
        assert not seen & valid_ids
        mask = train_all.participant_id.isin(valid_ids)
        assert len(valid) == int(mask.sum()) and len(train) == int((~mask).sum())
        fold.loc[mask] = i
        seen.update(valid_ids)
    assert seen == all_ids and fold.notna().all()
    return fold.astype(int)


def select(task, rows, features, spec, label, fold_results):
    summaries = []
    for kind, grid in spec.items():
        for params in grid:
            predictions = pd.Series(index=rows.index, dtype=float)
            for fold in range(1, 6):
                train = rows[rows.fold != fold]
                val = rows[rows.fold == fold]
                assert not set(train.participant_id) & set(val.participant_id)
                model, _ = fit(task, kind, params, train, features)
                pred = predict(task, kind, model, val, features)
                predictions.loc[val.index] = pred
                fold_results.append({"task": task, "variant": label, "model": kind,
                                     "params": json.dumps(params), "fold": fold, **metrics(val, task, pred)})
            assert predictions.notna().all()
            scores = metrics(rows, task, predictions.to_numpy())
            summaries.append({"task": task, "variant": label, "model": kind,
                              "params": json.dumps(params), **scores})
    return pd.DataFrame(summaries).sort_values("macro_mae", kind="stable")


def eligible(data, task):
    needed = [TARGET[task]] + (["sleep_interval_min"] if task == "B" else [])
    return data.dropna(subset=needed).copy()


def model_info(row, features):
    return {"model": row["model"], "params": json.loads(row["params"]), "features": features}


def save_model(path, task, choice, fitted):
    if choice["model"] == "mlp":
        scaler, models = fitted
        joblib.dump({"task": task, **choice, "scaler": scaler,
                     "weights": [{k: v.numpy() for k, v in model.state_dict().items()} for model in models]}, path)
    else:
        joblib.dump({"task": task, **choice, "estimator": fitted}, path)


def plot_learning(history, FIGURE_DIR):
    fig, axes = plt.subplots(2, 2, figsize=(9, 5), layout="constrained")
    for col, task in enumerate(("A", "B")):
        df = history[history.task == task]
        for row, (feature, title) in enumerate((
            ("train_loss_scaled", "Training loss (scaled)"),
            ("validation_macro_mae", "Validation participant MAE"),
        )):
            for seed, run in df.groupby("seed"):
                axes[row, col].plot(run.epoch, run[feature], label=str(seed), alpha=0.8)
            axes[row, col].set(title=f"{task} | {title}", xlabel="Epoch")
    axes[0, 0].legend(title="Seed")
    fig.savefig(out / "learning_curves.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    out = args.output
    cv_out, test_out = out / "교차검증", out / "테스트"
    for directory in (cv_out, test_out, FINAL_MODELS, COMPARISON_MODELS, FIGURE_DIR, ENV_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    all_train = pd.read_csv(args.data / "train_melidos_all.csv")
    all_train["fold"] = folds_for(all_train, args.data)
    all_train[["participant_id", "fold"]].drop_duplicates().to_csv(cv_out / "cv_groups.csv", index=False)
    fold_results, cv_tables, choices, histories = [], [], {}, []
    for task in ("A", "B"):
        print(f"Selecting {task}", flush=True)
        rows = eligible(all_train, task)
        features = FEATURES + (["sleep_interval_min"] if task == "B" else [])
        cv = select(task, rows, features, GRID, "full", fold_results)
        cv_tables.append(cv)
        choices[task] = {
            "selected": model_info(cv.iloc[0], features),
            "baseline": model_info(cv[cv.model == "baseline"].iloc[0], features),
            "mlp": model_info(cv[cv.model == "mlp"].iloc[0], features),
        }
        if task == "B":
            ablation = select(task, rows, FEATURES, GRID, "without_duration", fold_results)
            cv_tables.append(ablation)
            choices[task]["without_duration"] = model_info(ablation.iloc[0], FEATURES)
        mlp = choices[task]["mlp"]
        _, history = fit(task, "mlp", mlp["params"], rows[rows.fold != 1],
                         features, monitor=rows[rows.fold == 1])
        histories.extend({"task": task, **record} for record in history)
    pd.concat(cv_tables).to_csv(cv_out / "cv_summary.csv", index=False)
    pd.DataFrame(fold_results).to_csv(cv_out / "cv_by_fold.csv", index=False)
    history = pd.DataFrame(histories)
    history.to_csv(cv_out / "learning_curves.csv", index=False)
    plot_learning(history, FIGURE_DIR)
    (cv_out / "selection.json").write_text(json.dumps(choices, indent=2), encoding="utf-8")
    print("Selection frozen. Scoring holdout data.", flush=True)
    melidos = pd.read_csv(args.data / "test_melidos.csv")
    korea = pd.read_csv(args.data / "test_korea.csv")
    id_sets = [set(frame.participant_id) for frame in (all_train, melidos, korea)]
    assert len(set.union(*id_sets)) == sum(map(len, id_sets))
    assert korea.sleep_quality.isna().all()
    assert len(korea[korea.is_korea_final7 == 1]) == 7 * korea.participant_id.nunique()
    results, predictions = [], []
    for task, options in choices.items():
        rows = eligible(all_train, task)
        datasets = {"melidos_holdout": eligible(melidos, task)}
        if task == "A":
            datasets.update(korea_last7=korea[korea.is_korea_final7 == 1], korea_all=korea)
        for role, info in options.items():
            kind, params, features = info["model"], info["params"], info["features"]
            fitted, _ = fit(task, kind, params, rows, features)
            if role in {"selected", "mlp"}:
                save_model((FINAL_MODELS if role == "selected" else COMPARISON_MODELS) / f"model_{task}_{role}.joblib", task, info, fitted)
            for dataset, subset in datasets.items():
                pred = predict(task, kind, fitted, subset, features)
                assert np.isfinite(pred).all()
                result = {"task": task, "dataset": dataset, "role": role, "model": kind}
                results.append({**result, **metrics(subset, task, pred)})
                export = subset[["participant_id", "site_code", "night_date", TARGET[task]]].copy()
                export = export.rename(columns={TARGET[task]: "actual"})
                export["predicted"] = pred
                export["error"] = export.predicted - export.actual
                predictions.append(export.assign(**result))
    pd.DataFrame(results).to_csv(test_out / "holdout_metrics.csv", index=False)
    pd.concat(predictions).to_csv(test_out / "holdout_predictions.csv", index=False)
    versions = {"python": platform.python_version(), "torch": torch.__version__,
                "sklearn": sklearn.__version__, "pandas": pd.__version__,
                "numpy": np.__version__, "matplotlib": matplotlib.__version__, "joblib": joblib.__version__}
    (ENV_DIR / "versions.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")
    print(pd.DataFrame(results).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
