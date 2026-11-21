import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

DROP_COLS = ["project_id", "hours_spent", "log_hours", "hours_bucket"]
MAX_HOURS = 140.7


def _num(X, name):
    if name not in X.columns:
        return pd.Series(np.nan, index=X.index)
    col = X[name]
    if col.dtype == object or not pd.api.types.is_numeric_dtype(col):
        col = col.map({True: 1, False: 0, "True": 1, "False": 0, "true": 1, "false": 0}).where(
            col.notna(), np.nan) if col.map(type).isin([bool, str]).any() else col
    return pd.to_numeric(col, errors="coerce")


def _ratio(X, a, b):
    return _num(X, a) / _num(X, b).replace(0, np.nan)


def clean_features(df):
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns]).copy()

    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)

    X["commits_per_active_day"] = _ratio(X, "num_commits", "active_commit_days")
    X["lines_per_active_day"] = _ratio(X, "lines_added_total", "active_commit_days")
    X["lines_per_commit_avg"] = _ratio(X, "lines_added_total", "num_commits")
    X["active_day_share"] = _ratio(X, "active_commit_days", "project_span_days")
    X["commits_per_span_day"] = _ratio(X, "num_commits", "project_span_days")
    X["lines_per_span_day"] = _ratio(X, "lines_added_total", "project_span_days")
    X["weekend_commits"] = _num(X, "pct_commits_weekend") * _num(X, "num_commits")
    X["late_night_commits"] = _num(X, "pct_commits_late_night") * _num(X, "num_commits")
    X["msg_effort"] = _num(X, "avg_commit_msg_length") * (1 - _num(X, "pct_generic_commit_msgs"))
    X["code_per_directory"] = _ratio(X, "lines_of_code", "num_directories")
    X["asset_mb_per_image"] = _ratio(X, "asset_size_mb", "num_image_files")
    X["readme_words_per_heading"] = _ratio(X, "readme_word_count", "readme_num_headings")
    X["popularity"] = pd.concat([_num(X, "stars"), _num(X, "forks")], axis=1).sum(axis=1, min_count=1)
    flags = ["has_tests", "has_ci_config", "has_dockerfile", "has_license", "has_docs_folder", "has_database", "has_backend"]
    X["good_practice_count"] = pd.concat([_num(X, f) for f in flags], axis=1).sum(axis=1, min_count=1)
    return X.replace([np.inf, -np.inf], np.nan)


def find_text_columns(X):
    return [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]


def text_to_str(X, text_cols):
    X = X.copy()
    for c in text_cols:
        X[c] = X[c].map(lambda v: np.nan if pd.isna(v) else str(v)).astype(object)
    return X


def build_pipeline(text_cols):
    encode_text = ColumnTransformer(
        [("text", OrdinalEncoder(handle_unknown="use_encoded_value",
                                 unknown_value=np.nan,
                                 encoded_missing_value=np.nan), text_cols)],
        remainder="passthrough",
    )
    brain = LGBMRegressor(
        n_estimators=400, learning_rate=0.03, num_leaves=15,
        min_child_samples=20, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0, random_state=0, verbose=-1,
    )
    return Pipeline([("encode", encode_text), ("model", brain)])


def fit_bundle(train_df):
    """Train on a full training dataframe. Returns everything needed to predict."""
    X = clean_features(train_df)
    text_cols = find_text_columns(X)
    X = text_to_str(X, text_cols)
    y_log = np.log1p(train_df["hours_spent"].clip(lower=0))
    model = build_pipeline(text_cols).fit(X, y_log)
    return {"model": model, "feature_cols": list(X.columns), "text_cols": text_cols}


def predict_hours(bundle, df):
    X = clean_features(df).reindex(columns=bundle["feature_cols"])
    X = text_to_str(X, bundle["text_cols"])
    return np.clip(np.expm1(bundle["model"].predict(X)), 0, MAX_HOURS)
