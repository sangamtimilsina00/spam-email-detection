# Email Spam Classifier

Simple Flask web application that classifies pasted email text as "Spam" or "Ham". The app will load an optional serialized ML model + vectorizer (joblib/pickle). If none are found or loading fails, it falls back to a lightweight rule-based predictor.

## Repository structure
- app.py — Flask app and prediction logic (loads model/vectorizer if present; fallback rule).
- templates/index.html — Frontend: responsive UI, CSS, and JS (auto-resize textarea, prevents double submits, responsive "Classify" button and spinner).
- .gitignore — ignores virtual envs, cache, joblib outputs, etc.
- (optional) model.joblib, vectorizer.joblib or .pkl — trained classifier and vectorizer placed in project root.
- README.md — this file.

## Quick features
- Loads serialized model + vectorizer when available.
- Robust fallback: keyword-based rule if model unavailable or errors occur.
- Modern responsive UI with animated classify button and spinner; button becomes full-width on small screens.
- Prevents double submission and auto-resizes textarea.

## Requirements
- Python 3.8+
- Recommended packages:
  - Flask
  - scikit-learn (if using a saved model)
  - joblib

Example install (after creating venv):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install flask scikit-learn joblib
```

## How to run
1. Activate virtual environment.
2. (Optional) Place trained model and vectorizer in project root as `model.joblib` and `vectorizer.joblib` (or use `.pkl` names listed in app.py).
3. Start the app:
```bash
python3 app.py
```
4. Open http://127.0.0.1:5000/ in a browser.

## How prediction works (app.py)
- At startup, app.py tries to load model and vectorizer from candidate filenames using joblib.
- model_predict(text):
  - If both model and vectorizer available: vectorizer.transform([text]) → model.predict(...) and maps common label formats to "Spam"/"Ham".
  - If model only: attempts model.predict([text]).
  - On any exception or missing artifacts: falls back to simple_rule_predict.
- simple_rule_predict(text): counts presence of spam-indicative keywords (e.g., "win", "free", "click") and returns "Spam" if threshold exceeded, otherwise "Ham".

## Frontend behavior (templates/index.html)
- Large textarea for pasting email; auto-resizes on input.
- Classify button:
  - Shows spinner and "Classifying..." when submitting.
  - Prevents double submits.
  - Responsive: full-width on narrow screens (improved tap target).
- Sidebar displays result and a sample quick message.

## Training and saving your own model
Train a text classification pipeline (vectorizer + classifier), then save them with joblib:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Example training flow (replace with real data)
vect = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X = vect.fit_transform(texts)         # texts: list[str]
model = LogisticRegression(max_iter=1000).fit(X, labels)  # labels: 0/1 or 'spam'/'ham'

joblib.dump(vect, 'vectorizer.joblib')
joblib.dump(model, 'model.joblib')
```

Place `vectorizer.joblib` and `model.joblib` in the project root and restart the app.

## Troubleshooting
- "Model not loaded": ensure files exist and are accessible; check app console/logs for errors.
- Dependency issues: ensure venv is activated and required packages installed.
- If prediction fails unexpectedly, app will log the exception and use the rule-based fallback.

## Notes & Next steps
- Replace rule-based fallback with a trained model for production accuracy.
- Add unit tests for app.py prediction logic and UI behavior.
- Consider adding Dockerfile and CI for reproducible deployment.

License: provided as-is for learning and development. Update license as needed.
