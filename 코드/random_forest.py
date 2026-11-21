from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


def fit(params, inputs, targets):
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        RandomForestRegressor(n_estimators=300, random_state=42, **params),
    ).fit(inputs, targets)
