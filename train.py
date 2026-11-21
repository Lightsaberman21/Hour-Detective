import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_log_error
from sklearn.model_selection import KFold, cross_val_predict

from model_utils import (MAX_HOURS, build_pipeline, clean_features, find_text_columns, fit_bundle, text_to_str)
HERE = Path(__file__).parent
df = pd.read_csv(HERE / "train.csv", dtype={"project_id": str})
print("Loaded", len(df), "projects")

X = clean_features(df)
text_cols = find_text_columns(X)
X = text_to_str(X, text_cols)
y = df["hours_spent"].clip(lower=0)
y_log = np.log1p(y)

cv = KFold(n_splits=5, shuffle=True, random_state=0)
guess_log = cross_val_predict(build_pipeline(text_cols), X, y_log, cv=cv)
guess = np.clip(np.exmp1(guess_log), 0, MAX_HOURS)