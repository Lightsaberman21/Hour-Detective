import time 
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer

from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer

from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, QuantileTransformer, StandardScaler

from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping
from xgboost import XGBRegressor

from catboost import CatBoostRegressor


DROP_COLS = ["project_id", "hours_spent", "log_hours", "hours_bucket"]
MAX_HOURS = 140.7
MAX_LOG = float(np.log1p(MAX_HOURS))

CAT_COLS = [
  "commit_cadence",
  "primary_language",
  "framework",
  "hosting_type",
  "project_category",
  "author_prior_projects",
  "author_account_age_years",
]

BOOL_COLS = [
  "uses_game_engine",
  "has_tests",
  "has_ci_config",
  "has_dockerfile",
  "has_backend",
  "has_database",
  "has_custom_art",
  "readme_has_install_instructions",
  "has_license",
  "has_docs_folder",
  "url_is_live",
  "is_mobile_responsive",
  "has_custom_domain",
  "is_mobile_responsive",
  "has_custom_domain",
  "is_multiplayer",
  "mentions_ai_api",
  "is_clone_or_tutorial",
  "is_forked_from_template",
]

PAIR_COLS = [
  ("primary_language", "project_category"),
  ("framework", "project_category"),
  ("project_category", "commit_cadence"),
  ("author_prior_projects", "author_account_age_years"),
  ("use_game_engine", "project_category"),
  ("is_clone_or_tutorial", "is_forked_from_template"),
]

GROUP_COLS = ["primary_language", "project_category", "framework"]

GROUP_VAL_COLS = [
  "log1p_lines_of_code",
  "log1p_num_commits",
  "log1p_active_commit_days",
  "log1p_lines_added_total",
  "log1p_avg_file_length",
  "log1p_asset_size_mb",
]

GIT_CHECK_COLS = [
  "num_commits",
  "active_commit_days",
  "project_span_days",
  "commit_cadence",
  "lines_added_total",
  "lines_deleted_total",
  "pct_code_in_first_commit",
  "num_commit_authors",
  "avg_commit_msg_length",
]

CODE_CHECK_COLS = [
  "lines_of_code",
  "num_source_files",
  "num_directories",
  "primary_language",
  "avg_file_length",
  "comment_line_ratio",
]

ART_CHECK_COLS = [
  "num_image_files",
  "num_audio_files",
  "num_3d_model_files",
  "num_font_files",
  "asset_size_mb",
]

README_CHECK_COLS = [
  "readme_word_count",
  "readme_num_headings",
  "readme_num_images",
  "readme_num_code_blocks",
]

LIVE_CHECK_COLS = [
  "hosting_type",
  "page_weight_kb",
  "num_js_files_loaded",
  "num_internal_pages",
  "load_time_ms",
]

PROJECT_CHECK_COLS = [
  "description_word_count",
  "stars",
  "forks",
  "num_issues",
  "num_pull_requests",
  "num_releases",
]

LOG1P_COLS = [
  "num_commits",
  "active_commit_days",
  "project_span_days",
  "lines_added_total",
  "lines_deleted_total",
  "median_lines_per_commit",
  "max_lines_in_one_commit",
  "lines_of_code",
  "num_source_files",
  "num_directories",
  "num_dependencies",
  "avg_file_length",
  "asset_size_mb",
  "total_assets",
  "readme_word_count",
  "description_word_count",
  "page_weight_kb",
  "load_time_ms",
  "num_internal_pages",
  "stars",
  "forks",
  "num_issues",
  "num_pull_requests",
  "num_releases",
  "non_first_added",
  "non_first_loc",
  "meaningful_commits",
  "effective_commits",
  "total_lines_touched",
  "typical_commits_total",
  "burst_excess_lines",
]

BOOL_MAP = {
  True: 1.0,
  False: 0.0,
  "True": 1.0,
  "False": 0.0,
  "true": 1.0,
  "false": 0.0,
  "Yes": 1.0,
  "No": 0.0,
  "yes": 1.0,
  "no": 0.0,
}

def _num(X, name):
  if name not in X.columns:
    return pd.Series(np.nan, index=X.index, dtype=float)
  col = X[name]
  if pd.api.types.is_bool_dtype(col):
    return col.astype(float)
  if pd.api.types.is_numeric_dtype(col):
    return pd.to_numeric(col, errors="coerce").astype(float)
  s = col.astype(object)
  mapped = s.map(BOOL_MAP)

  s = s.where(mapped.isna(), mapped)
  return pd.to_numeric(s, errors="coerce").astype(float)
def _str_col(X, name):
  if name not in X.columns:
    return pd.Series(np.nan, index=X.index, dtype=object)
  col = X[name]
  out = col.astype(object).where(col.notna(), np.nan)
  return out.where(out.isna(), out.astype(str))
def _ratio(X, a, b):
  return _num(X, a) / _num(X, b).replace(0, np.nan)
def _div(a, b):
  return a / b.replace(0, np.nan)

def _safe_log1p(s):
  return np.log1p(s.clip(lower=0.0))
def _signed_log1p(s):
  return np.sign(s) * np.log1p(np.abs(s))
def _missing_count(X, cols):
  present = [X[c].isna().astype(float) if c in X.columns else pd.Series(1.0, index=X.index) for c in cols]
  return pd.concat(present, axis=1).sum(axis=1)
def clean_features(df):
  X = df.drop(columns=[c for c in DROP_COLS if c in df.columns]).copy()

  for c in list(X.columns):
    if c in BOOL_COLS or pd.api.types.is_bool_dtype(X[c]):
      X[c] = _num(X, c)
    elif c not in CAT_COLS:
      if not pd.api.types.is_numeric_dtype(X[c]):
        if X[c].isna().all():
          X[c] = X[c].astype(float)
        else:
          converted = _num(X, c)
          if converted.notna().sum() >= X[c].notna().sum() * 0.5 and X[c].notna().sum() > 0:
            X[c] = converted

  for c in CAT_COLS:
    if c in X.columns:
      X[c] = _str_col(X, c)
  num_commits



