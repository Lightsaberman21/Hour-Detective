# Hour Detective

Predicts how many hours a Hack Club project really took, using only its git history, code, README and live-site stats.

## How it works
- The model is gradient boosting (`HistGradientBoostingRegressor`), which handles empty cells natively.
- It predicts `log(1 + hours)`, because the contest scores with RMSLE.
- `predict.py` trains on `train.csv`, predicts `test.csv`, and writes `predictions.csv`.
- `train.py` reports a cross-validated score and saves `model.joblib` for the web demo.
- `app.py` is a small Flask web demo.

## Run it
```
pip install -r requirements.txt
python train.py      # score + saves model.joblib
python predict.py    # writes predictions.csv
python app.py        # web demo at http://localhost:5000
```
