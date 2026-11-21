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
    {% for 
"""