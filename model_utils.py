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
  num_commits =_num(X, "num_commits")
  active_days =_num(X, "active_commit_days")
  span_days =_num(X, "project_span_days")
  lines_added =_num(X, "lines_added_total")
  lines_deleted =_num(X, "lines_deleted_total")
  median_commit =_num(X, "median_lines_per_commit")
  max_commit =_num(X, "max_lines_in_one_commit")
  pct_first =_num(X, "pct_code_in_first_commit")
  num_authors =_num(X, "num_commit_authors")
  avg_msg_len =_num(X, "avg_commit_msg_len")
  pct_generic =_num(X, "pct_generic_commit_msgs")
  num_branches =_num(X, "num_branches")
  num_merges =_num(X, "num_merge_commits")
  pct_weeked =_num(X, "pct_commits_weekend")
  pct_late =_num(X, "pct_commits_late_night")
  loc =_num(X, "lines_of_code")
  num_files =_num(X, "num_source_files")
  num_dirs =_num(X, "num_directories")
  max_depth =_num(X, "max_dir_depth")
  num_langs =_num(X, "num_languages")
  pct_prim_lang =_num(X, "pct_primary_language")
  num_deps =_num(X, "num_dependencies")
  num_tests =_num(X, "num_test_files")
  avg_file_len =_num(X, "avg_file_length")
  comment_ratio =_num(X, "comment_line_ratio")
  num_imgs =_num(X, "num_images_files")
  num_audio =_num(X, "num_audio_files")
  num_3d =_num(X, "num_3d_model_files")
  num_fonts =_num(X, "num_font_files")
  asset_mb =_num(X, "assest_size_mb")
  readme_words =_num(X, "readme_word_count")
  readme_heads =_num(X, "readme_num_headings")
  readme_imgs =_num(X, "readme_num_images")
  readme_code =_num(X, "readme_num_code_blocks")
  page_kb =_num(X, "page_weight_kb")
  num_js =_num(X, "num_js_files_loaded")
  num_pages =_num(X, "num_internal_pages")
  load_ms =_num(X, "load_time_ms")
  desc_words =_num(X, "description_word_count")
  stars =_num(X, "stars")
  forks =_num(X, "forks")
  num_issues =_num(X, "num_issues")
  num_prs =_num(X, "num_pull_requests")
  num_rels =_num(X, "num_releases")
  is_clone =_num(X, "is_clone_or_tutorial")
  is_fork_tpl =_num(X, "is_forked_from_template")
  cadence_str = _str_col(X, "commit_cadence")
  prior_str = _str_col(X, "author_prior_projects")
  age_str = _str_col(X, "age_account_age_years")
  fw_str = _str_col(X, "framework")
  cadence_ord = cadence_str.map({"spordic": 0.0, "daily": 1.0, "frequent": 2.0})
  prior_ord = prior_str.map({"0": 0.0, "1-2": 1.0, "3-5": 2.0, "6+": 3.0})
  age_ord = age_str.map({"<1": 0.0, "1-2": 1.0, "3-5": 2.0, "6+": 3.0})
  prior_mid = prior_str.map({"0": 0.0, "1-2": 1.5, "3-5": 4.0, "6+": 8.0})
  age_mid = age_str.map({"<1": 0.5, "1-2": 1.5, "3-5": 4.0, "6+": 7.0})
  non_first_share = (1.0 - pct_first).clip(lower=0.0)
  meaningful_share = (1.0 - pct_generic).clip(lower=0.0)
  effective_commits = (num_commits - num_merges.fillna(0.0)).clip(lower=0.0)
  net_lines_git = lines_added - lines_deleted
  total_lines_touched = lines_added + lines_deleted
  typical_commits_total = median_commit * num_commits
  burst_excess_lines = (lines_added - typical_commits_total).clip(lower=0.0)
  total_assets = pd.concat([num_imgs, num_audio, num_3d, num_fonts], axis=1).sum(axis=1, min_count=1)
  weighted_assets = (
    num_imgs.fillna(0.0)
    + 2.0 * num_audio.fillna(0.0)
    +0.5 * num_fonts.fillna(0.0)
  ).where(total_assets.notna(), np.nan)
  non_first_added = lines_added * non_first_share
  non_first_loc = loc * non_first_share
  meaningful_commits = num_commits * meaningful_share
  lines_per_commit_avg = _div(lines_added, num_commits)
  est_commit_hours = 0.5 * effective_commits
  good_flags = [
    "has_tests",
    "has_ci_config",
    "has_dockerfile",
    "has_license",
    "has_docs_folder",
    "has_database",
    "has_backend",
  ]
  all_quality_flags = good_flags + [
    "has_custom_art",
    "readme_has_install_instructions",
    "url_is_live",
    "is_mobile_responsive",
    "has_custom_domain",
    "is_multiplayer",
  ]
  good_practice_count = pd.concat([_num(X, f) for f in good_flags], axis=1).sum(axis=1, min_count=1)
  all_quality_count = pd.concat([_num(X, f) for f in all_quality_flags], axis=1).sum(axis=1, min_count=1)
  clone_sum = is_clone.fillna(0.0) + is_fork_tpl.fillna(0.0)
  readme_richness = pd.concat([readme_heads, 2.0 * readme_imgs, 2.0 * readme_code], axis=1).sum(axis=1, min_count=1)
  total_written_words = pd.concat([readme_words, desc_words], axis=1).sum(axis=1, min_count=1)
  new_cols = {
    "commits_per_active_day": _div(num_commits, active_days),
    "lines_per_active_day": _div(lines_added, active_days),
    "lines_per_commit_avg": lines_per_commit_avg,
    "active_day_share": _div(active_days, span_days),
    "active_day_density": _div(active_days, span_days + 1.0),
    "commits_per_span_day": _div(num_commits, span_days),
    "commits_per_span_day_s": _div(num_commits, span_days + 1.0),
    "lines_per_span_day": _div(lines_added, span_days),
    "lines_per_span_day_s": _div(lines_added, span_days + 1.0),
    "idle_span_days": span_days - active_days,
    "weekend_commits": pct_weekend * num_commits,
    "late_night_commits": pct_late * num_commits,
    "weekday_commits": (1.0 - pct_weekend).clip(lower=0.0) * num_commits,
    "daytime_commits": (1.0 - pct_late).clip(lower=0.0) * num_commits,
    "msg_effort": avg_msg_len * meaningful_share,
    "total_msg_chars": num_commits * avg_msg_len,
    "meaningful_commits": meaningful_commits,
    "meaningful_msg_chars": meaningful_commits * avg_msg_len,
    "generic_commits": num_commits * pct_generic,
    "effective_commits": effective_commits,
    "effective_commits_per_day": _div(effective_commits, active_days),
    "merge_commit_share": _div(num_merges, num_commits),
    "est_commit_hours": est_commit_hours,
    "loc_per_est_hour": _div(loc, est_commit_hours + 1.0),
    "net_lines_git": net_lines_git,
    "total_lines_touched": total_lines_touched,
    "touched_per_active_day": _div(total_lines_touched, active_days),
    "touched_per_commit": _div(total_lines_touched, num_commits),
    "deleted_per_commit": _div(lines_deleted, num_commits),
    "deleted_per_active_day": _div(lines_deleted, active_days),
    "loc_to_added_ratio": _div(loc, lines_added + 1.0),
    "loc_to_net_ratio": _div(loc, net_lines_git.abs() + 1.0),
    "loc_to_touched_ratio": _div(loc, total_lines_touched + 1.0),
    "net_vs_loc_gap": loc - net_lines_git,
    "non_first_commit_share": non_first_share,
    "non_first_added": non_first_added,
    "non_first_loc": non_first_loc,
    "first_commit_loc_est": loc * pct_first,
    "max_to_median_commit": _div(max_commit, median_commit + 1.0),
    "max_commit_share_added": _div(max_commit, lines_added + 1.0),
    "median_to_mean_commit": _div(median_commit, lines_per_commit_avg + 1.0),
    "typical_commits_total": typical_commits_total,
    "typical_to_added_share": _div(typical_commits_total, lines_added + 1.0),
    "burst_excess_lines": burst_excess_lines,
    "commits_per_author": _div(num_commits, num_authors),
    "active_days_per_author": _div(active_days, num_authors),
    "loc_per_author": _div(loc, num_authors),
    "added_per_author": _div(lines_added, num_authors),
    "commits_per_branch": _div(num_commits, num_branches + 1.0),
    "branches_per_author": _div(num_branches, num_authors),
    "non_comment_loc": loc * (1.0 - comment_ratio).clip(lower=0.0),
    "comment_loc_est": loc * comment_ratio,
    "primary_lang_loc": loc * pct_prim_lang,
    "secondary_lang_loc": loc * (1.0 - pct_prim_lang).clip(lower=0.0),
    "code_per_directory": _div(loc, num_dirs),
    "files_per_directory": _div(num_files, num_dirs),
    "loc_per_source_file": _div(loc, num_files),
    "loc_per_language": _div(loc, num_langs),
    "prod_source_files": (num_files - num_tests.fillna(0.0)).clip(lower=0.0),
    "test_file_share": _div(num_tests, num_files + 1.0),
    "loc_per_dependency": _div(loc, num_deps + 1.0),
    "deps_per_source_file": _div(num_deps, num_files + 1.0),
    "commits_per_source_file": _div(num_commits, num_files + 1.0),
    "loc_per_active_day": _div(loc, active_days),
    "loc_per_commit": _div(loc, num_commits),
    "loc_per_span_day": _div(loc, span_days + 1.0),
    "dir_complexity": _safe_log1p(num_dirs) * max_depth,
    "total_assets": total_assets,
    "weighted_assets": weighted_assets,
    "asset_mb_per_image": _div(asset_mb, num_imgs),
    "asset_mb_per_total": _div(asset_mb, total_assets + 1.0),
    "assets_per_active_day": _div(total_assets, active_days),
    "readme_words_per_heading": _div(readme_words, readme_heads),
    "readme_richness": readme_richness,
    "readme_words_per_loc": _div(readme_words, loc + 1.0),
    "total_written_words": total_written_words,
    "kb_per_internal_page": _div(page_kb, num_pages + 1.0),
    "kb_per_js_file": _div(page_kb, num_js + 1.0),
    "js_per_internal_page": _div(num_js, num_pages + 1.0),
    "popularity": pd.concat([stars, forks], axis=1).sum(axis=1, min_count=1),
    "repo_activity": pd.concat([stars, forks, num_issues, 2.0 * num_prs, 2.0 * num_rels], axis=1).sum(axis=1, min_count=1),
    "pr_per_commit": _div(num_prs, num_commits + 1.0),
    "release_per_commit": _div(num_rels, num_commits + 1.0),
    "capped_stars": (stars >= 50.0).astype(float).where(stars.notna(), np.nan),
    "capped_forks": (forks >= 50.0).astype(float).where(forks.notna(), np.nan),
    "capped_pages": (num_pages >= 21.0).astype(float).where(num_pages.notna(), np.nan),
    "good_practice_count": good_practice_count,
    "all_quality_count": all_quality_count,
    "clone_or_template_sum": clone_sum,
    "net_quality_minus_clone": all_quality_count - 2.0 * clone_sum,
    "non_clone_loc": loc * (1.0 - is_clone.fillna(0.0)).clip(lower=0.0),
    "non_clone_commits": num_commits * (1.0 - is_clone.fillna(0.0)).clip(lower=0.0),
    "non_template_loc": loc * (1.0 - is_fork_tpl.fillna(0.0)).clip(lower=0.0),
    "commit_cadence_ord": cadence_ord,
    "author_prior_ord": prior_ord,
    "author_age_ord": age_ord,
    "author_prior_mid": prior_mid,
    "author_age_mid": age_mid,
    "author_projects_per_year": _div(prior_mid, age_mid + 0.5),
    "author_exp_sum": prior_ord + age_ord,
    "git_missing_cnt": _missing_count(X, GIT_CHECK_COLS),
    "code_missing_cnt": _missing_count(X, CODE_CHECK_COLS),
    "art_missing_cnt": _missing_count(X, ART_CHECK_COLS),
    "readme_missing_cnt": _missing_count(X, README_CHECK_COLS),
    "live_missing_cnt": _missing_count(X, LIVE_CHECK_COLS),
    "project_missing_cnt": _missing_count(X, PROJECT_CHECK_COLS),
    "total_missing_cnt": X.isna().sum(axis=1).astype(float),
    "has_no_git": num_commits.isna().astype(float),
    "has_no_code": loc.isna().astype(float),
    "has_no_live_metrics": page_kb.isna().astype(float),
    "has_no_readme_metrics": (readme_words.isna() | (readme_words == 0.0)).astype(float),
    "has_no_load_time": load_ms.isna().astype(float),
    "has_no_framework": (fw_str.isna() | (fw_str == "none")).astype(float),
    "is_single_commit": (num_commits == 1.0).astype(float).where(num_commits.notna(), np.nan),
    "is_single_day": (active_days == 1.0).astype(float).where(active_days.notna(), np.nan),
    "is_single_upload": (pct_first >= 0.95).astype(float).where(pct_first.notna(), np.nan),
    "is_multi_author": (num_authors > 1.0).astype(float).where(num_authors.notna(), np.nan),
    "signed_log_net_lines": _signed_log1p(net_lines_git),
    "signed_log_idle_days": _signed_log1p(span_days - active_days),
  }
  log_sources = {
    "num_commits": num_commits,
    "active_commit_days": active_days,
    "project_span_days": span_days,
    "lines_added_total": lines_added,
    "lines_deleted_total": lines_deleted,
    "median_lines_per_commit": median_commit,
    "max_lines_in_one_commit": max_commit,
    "lines_of_code": loc,
    "num_source_files": num_files,
    "num_directories": num_dirs,
    "num_dependencies": num_deps,
    "avg_file_length": avg_file_len,
    "asset_size_mb": asset_mb,
    "total_assets": total_assets,
    "readme_word_count": readme_words,
    "description_word_count": desc_words,
    "page_weight_kb": page_kb,
    "load_time_ms": load_ms,
    "num_internal_pages": num_pages,
    "stars": stars,
    "forks": forks,
    "num_issues": num_issues,
    "num_pull_requests": num_prs,
    "num_releases": num_rels,
    "non_first_added": non_first_added,
    "non_first_loc": non_first_loc,
    "meaningful_commits": meaningful_commits,
    "effective_commits": effective_commits,
    "total_lines_touched": total_lines_touched,
    "typical_commits_total": typical_commits_total,
    "burst_excess_lines": burst_excess_lines,
  }
  for col_name in LOG1P_COLS:
    new_cols["log1p_" + col_name] = _safe_log1p(log_sources[col_name])
    new_cols["log_diff_commits_active"] = new_cols["log1p_num_commits"] - new_cols["log1p_active_commit_days"]
    new_cols["log_diff_active_span"] = new_cols["log1p_active_commit_days"] - new_cols["log1p_project_span_days"]
    new_cols["log_diff_loc_commits"] = new_cols["log1p_lines_of_code"] - new_cols["log1p_num_commits"]
    new_cols["log_diff_loc_added"] = new_cols["log1p_lines_of_code"] - new_cols["log1p_lines_added_total"]
    new_cols["log_diff_added_deleted"] = new_cols["log1p_lines_added_total"] - new_cols["log1p_lines_deleted_total"]
    new_cols["log_diff_max_median_commit"] = new_cols["log1p_max_lines_in_one_commit"] - new_cols["log1p_median_lines_per_commit"]
    new_cols["log_commits_x_log_active"] = new_cols["log1p_num_commits"] * new_cols["log1p_active_commit_days"]
    new_cols["log_commits_x_log_loc"] = new_cols["log1p_num_commits"] * new_cols["log1p_lines_of_code"]
    new_cols["effective_work_score"] = (
      new_cols["log1p_effective_commits"].fillna(0.0)
      + new_cols["log1p_active_commit_days"].fillna(0.0)
      + 0.5 * new_cols["log1p_non_first_loc"].fillna(0.0)
      + 0.5 * new_cols["log1p_total_assets"].fillna(0.0)
    )
    pair_df_cols = {}
  for a, b in PAIR_COLS:
    sa = _str_col(X, a).fillna("_NA_")
    sb = _str_col(X, b).fillna("_NA_")
    pair_df_cols[a + "__" + b] = sa + "__" + sb
    out = pd.concat([X, pd.DataFrame(new_cols, index=X.index), pd.DataFrame(pair_df_cols, index=X.index)], axis=1)
    return out.replace([np.inf, -np.inf], np.nan)
  
