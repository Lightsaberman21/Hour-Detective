import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

DROP_COLS = ["project_id", "hours_spent", "log_hours", "hours_bucket"]
MAX_HOURS = 140.7

def _ratio(X, a, b):
  if a not in X.columns or b not in X.columns:
    return np.nan
  top = pd.to_numeric(X[a], errors="coerce")
  bottom = pd.to_numeric(X[b], errors="coerce").replace(0, np.nan)
  return top / bottom

def clean_features(df):
  X = df.drop(columns=[c for c in DROP_COLS if c in df.colums]).copy()

  for c in X.columns:
    if X[c].dtype == bool:
      X[c] = X[c].astype(int)

  X["commits_per_active_day"] = _ratio(X, "num_commits", "active_commit_days")
  X["lines_per_active_day"] = _ratio(X, "lines_added_total", "active_commit_days")
  X["lines_per_commit_avg"] = _ratio(X, "lines_added_total", "num_commits")
  X["active_day_share"] = _ratio(X, "active_commit_days", "project_span_days")
  return X.replace([np.inf, -np.inf], np.nan)

def find_text_columns(X):
  return [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]

def text_to_str(X, text_cols):
  X = X.copy()
  for c in text_cols:
    X[c] = X[c].map(lambda c: np.nan if pd.isna(v) else str(v)).astype(object)
  return X

def build_pipeline(text_cols):
  encode_text = ColumnTransformer(
    [("text", OrdinalEncoder(handle_unkown="use_encoded_value",
                             unkown_value=np.nan,
                             encoded_missing_value=np.nan), text_cols)],
    remainder="passthrough"
  )
  brain = HistGradientBoostingRegressor(
    learning_rate=0.03, max_iter=400, max_depth=4,
    min_samples_leaf=20, l2_regularization=1.0, random_state=0,
  )
  return Pipeline([("encode", encode_text), ("model", brain)])

def fit_bundle(train_df):
  X = clean_features(train_df)
  text_cols = find_text_columns(X)
  X = text_to_str(X, text_cols)
  y_log = np.log1p(train_df["hours_spent"].clip(lower=0))
  model = build_pipeline(text_cols).fit(X, y_log)
  return {"model": model, "feature_cols": list(X.columns), "text_cols": text_cols}

def predict_hours(bundle, df):
  X = clean_features(df).reindex(columns=bundle["feature_cols"])
  X = text_to_str(X, bundle["text_cols"])
  return np.clip(np.exmp1(bundle["model"].predict(X)), 0, MAX_HOURS)
