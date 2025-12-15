from flask import Flask, render_template, request
import os
import logging

app = Flask(__name__, template_folder='templates')

# Try to load a serialized model/vectorizer (joblib/pickle). If not found, use a simple fallback.
model = None
vectorizer = None
base = os.path.dirname(__file__)
candidates_model = ('model.joblib', 'model.pkl', 'spam_model.joblib', 'spam_model.pkl')
candidates_vect = ('vectorizer.joblib', 'vectorizer.pkl', 'vect.joblib', 'vect.pkl')

try:
    import joblib
except Exception:
    joblib = None

for fname in candidates_model:
    path = os.path.join(base, fname)
    if os.path.exists(path) and joblib:
        try:
            model = joblib.load(path)
            logging.info("Loaded model from %s", path)
            break
        except Exception:
            logging.exception("Failed loading model %s", path)

for fname in candidates_vect:
    path = os.path.join(base, fname)
    if os.path.exists(path) and joblib:
        try:
            vectorizer = joblib.load(path)
            logging.info("Loaded vectorizer from %s", path)
            break
        except Exception:
            logging.exception("Failed loading vectorizer %s", path)


def simple_rule_predict(text: str) -> str:
    t = (text or "").lower()
    indicators = ['win', 'prize', 'free', 'click', 'buy now', 'unsubscribe', 'claim', 'urgent', 'lottery', 'congrat']
    score = sum(1 for w in indicators if w in t)
    return 'Spam' if score >= 2 else 'Ham'


def model_predict(text: str) -> str:
    if model and vectorizer:
        try:
            X = vectorizer.transform([text])
            y = model.predict(X)
            label = y[0]
            # handle common label types
            if isinstance(label, (int, float)):
                return 'Spam' if int(label) == 1 else 'Ham'
            s = str(label).strip().lower()
            return 'Spam' if s in ('1', 'spam', 'true', 'yes') else 'Ham'
        except Exception:
            logging.exception("Model prediction failed, falling back to rule-based")
            return simple_rule_predict(text)
    if model and not vectorizer:
        try:
            y = model.predict([text])
            label = y[0]
            s = str(label).strip().lower()
            return 'Spam' if s in ('1', 'spam', 'true', 'yes') else 'Ham'
        except Exception:
            logging.exception("Model-only prediction failed, falling back to rule-based")
            return simple_rule_predict(text)
    return simple_rule_predict(text)


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        if email:
            prediction = model_predict(email)
    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
