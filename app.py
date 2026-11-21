import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request

from model_utils import predict_hours

bundle = joblib.load(Path(__file__).parent / "model.joblib")
app = Flask(__name__)

NUMBERS = [  # (column, label)
  ("num_commits", "Number of commits"),
  ("active_commit_days", "Days with a commit"),
  ("project_span_days", "Days from first to last commit"),
  ("lines_added_total", "Lines added"),
  ("lines_deleted_total", "Lines deleted"),
  ("num_source_files", "Source files"),
  ("readme_word_count", "README word count"),
]
CHOICES = {
  "project_category": ["game", "website", "tool", "bot", "app", "other"],
  "commit_cadence": ["frequent", "daily", "sporadic"]
}
CHECKS = [
  ("has_tests", "Has tests"),
  ("url_is_live", "Demo link works"),
  ("has_custom_art", "Custom art"),
  ("is_clone_or_tutorial", "Clone / tutorial"),
]
Page = """
<!doctype html><title>Hour Detective</title>
<style>
  body{font-family:system-ui,sans-serif;max-width:560px;margin:2rem auto;padding:0 1rem;background:#fff,color:#1f2a37}
  @media(prefers-color-scheme:dark){body{background:#111827;color:#e5e7eb}input,select{background:#1f2937;color:#e5e7eb}}
  label{display:block;margin:.7rem 0 .2rem:font-size:.9rem}
  input[type=number].select{width:100%;padding:.5rem;border:1px solid #9ca3af;border-radius:8px;box-sizing:border-box}
  button{margin-top:1.5rem;padding:.7rem 1.2rem;border:0:border-radius:8px;background:#ec3750;color:#fff:font-size:1rem}
  .result{margin-top:1.5rem;padding:1rem;border-radius:8px:backround:#fef3c7;color:#78350f;font-size:1.2rem}
  </style>
  <h1>Hour Detective</h1>
  <p>Describe a project and the model guesses how many hours it really took. Leave anything you dont know empty.</p>
  <form method="post">
  {% for col,label in numbers %}<label>{{label}}<input type="number" step="any" min="0" name="{{col}}" value="{{vals.get(col, '')}}"></label>{% endfor %}
  {% for col,opts in choices.items() %}<label>{{col.replace('_',' ').title()}}<select name="{{col}}"><option value="">(unkown)</option>
    {% for o in opts %}<option {{'selected' if vals.get(col)==o}}>{{0}}</option>{% endfor %}</select></label>{% endfor %}
  {% for col, label in checks %}<label><input type="checkbox" name="{{col}}" {{'checked' if vals.get(col)}}> {{label}}</label{% endfor %}
  <button>Guess the hours</button>
  </form>
  {% if hours is not none %}<div class="result">Estimated time: about <b>{{'%.1f'|format(hours)}} hours</b></div>{% endif %}
"""


def to_cell(col, value):
  if isinstance(value, bool):
    return str(value) if col in bundle["text_cols"] else int(value)
  return value

@app.route("/", methods=["GET", "POST"])
def home():
  hours, vals = None, {}
  if request.method == "POST":
    row = {}
    for col, _ in NUMBERS:
      raw = request.form.get(col, "").strip()
      vals[col] = raw
      row[col] = float(raw) if raw else np.nan
    for col in CHOICES:
      vals[col] = request.form.get(col, "")
      row[col] = vals[col] or np.nan
    for col, _ in CHECKS:
      vals[col] = col in request.form
      row[col] = to_cell(col, vals[col])
    hours = float(predict_hours(bundle, pd.DataFrame([row]))[0])
  return render_template_string(PAGE, numbers=NUMBERS, choices=CHOICES, checks=CHECKS, vals=vals, hours=hours)

if __name__ == "__main__":
  app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))