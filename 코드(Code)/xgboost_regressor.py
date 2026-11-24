from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


GRID = [
    {"max_depth": depth, "n_estimators": trees, "learning_rate": rate}
    for depth in (1, 2, 3) for trees, rate in ((80, 0.05), (180, 0.03))
]


def fit(task, params, inputs, targets):
    from xgboost import XGBRegressor

    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        XGBRegressor(
            objective="reg:absoluteerror" if task == "A" else "reg:squarederror",
            tree_method="hist", min_child_weight=10, reg_lambda=5,
            n_jobs=1, random_state=42, **params,
        ),
    ).fit(inputs, targets)
