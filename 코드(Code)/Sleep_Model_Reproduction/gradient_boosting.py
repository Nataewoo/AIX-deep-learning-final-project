from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


def fit(task, params, inputs, targets):
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        GradientBoostingRegressor(
            learning_rate=0.05, min_samples_leaf=15, random_state=42,
            loss="absolute_error" if task == "A" else "squared_error", **params,
        ),
    ).fit(inputs, targets)
