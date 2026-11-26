# Hour Detective

Hour Detective estimates how many hours a Hack Club project took from its Git history, code, README, and live-site stats.

## Model

The model uses an ensemble of regressors trained on `log(1 + hours_spent)`. `train.py` reports the ensemble's out-of-fold (OOF) RMSLE, a cross-validated estimate where lower is better. The latest run reported **0.6543**; scores can vary when the data or model changes.

## Files

- `train.csv` contains the labeled training data.
- `test.csv` contains projects to predict.
- `model_utils.py` contains feature preparation, training, and prediction code.
- `train.py` trains the model, prints its OOF RMSLE, and saves a compressed `model.joblib`.
- `predict.py` predicts hours for `test.csv` and writes `predictions.csv`. If `train.csv` is present, it trains a fresh model before predicting.
- `app.py` runs the Flask web demo using the saved `model.joblib`.

## Setup and usage

Install the dependencies:

```bash
pip install -r requirments.txt
```

Train and save the model:

```bash
python train.py
```

Generate predictions for `test.csv`:

```bash
python predict.py
```

Run the web demo after training:

```bash
python app.py
```

Then open <http://localhost:5000>.
