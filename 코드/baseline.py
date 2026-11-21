import numpy as np


def fit(task, targets):
    return float(np.median(targets) if task == "A" else np.mean(targets))


def predict(task, fitted, rows):
    if task == "A":
        return rows.past3_latency_median_min.fillna(fitted).to_numpy()
    return np.full(len(rows), fitted)
