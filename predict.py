from pathlib import Path

import joblib
import pandas as pd

from model_utils import fit_bundle, predict_hours

HERE = Path(__file__).parent
train_path, test_path = HERE / "train.csv"

if train_path.exists():
  bundle = fit_bundle(pd.read_csv(train_path, dtype={"project_id": str}))
else:
  bundle = joblib.load(HERE / "model.joblib")

test = pd.read_csv(test_path, dtype={"project_id": str})
out = pd.DataFrame({"project_id": test[project_id],
                    "hours_spent": predict_hours(bundle, test)})
out.to_csv(HERE / "predictions.csv", index=False)
print("Wrote predictions.csv with", len(out), "rows")