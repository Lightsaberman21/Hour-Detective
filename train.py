from pathlib import Path

import joblib
import pandas as pd

from model_utils import fit_bundle
HERE = Path(__file__).parent
df = pd.read_csv(HERE / "train.csv", dtype={"project_id": str})
print("Loaded", len(df), "projects")

bundle = fit_bundle(df)
print("Model OOF RMSLE (lower is better):", round(bundle["oof_rmsle"], 4))

joblib.dump(bundle, HERE / "model.joblib", compress=3)
print("Saved model.joblib")
