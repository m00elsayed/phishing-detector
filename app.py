# ==============================
# 📦 Imports
# ==============================
from flask import Flask, render_template, request
import joblib
import pandas as pd
import re
from urllib.parse import urlparse

# ==============================
# 🚀 Initialize App
# ==============================
app = Flask(__name__)

# ==============================
# 📂 Load Model
# ==============================
model = joblib.load("model.pkl")

# ==============================
# 🧠 Feature Extraction
# ==============================
def has_ip(url):
    return 1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0

def has_at(url):
    return 1 if '@' in url else 0

def count_dots(url):
    return url.count('.')

def url_length(url):
    return len(url)

def has_https(url):
    return 1 if url.startswith("https") else 0

def count_hyphens(url):
    return url.count('-')

def count_digits(url):
    return sum(c.isdigit() for c in url)

def has_suspicious_words(url):
    keywords = ['login', 'secure', 'verify', 'update', 'bank', 'account']
    return 1 if any(k in url.lower() for k in keywords) else 0

def get_domain_length(url):
    try:
        return len(urlparse(url).netloc)
    except:
        return 0

# ==============================
# 🏠 Home Route
# ==============================
@app.route('/')
def home():
    return render_template('index.html')

# ==============================
# 🔮 Prediction Route
# ==============================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        url = request.form['url']

        if not url or len(url.strip()) < 5:
            return render_template(
                'index.html',
                prediction_text="⚠️ Please enter a valid URL",
                confidence=0,
                level="Invalid Input"
            )

        # ======================
        # Extract Features
        # ======================
        features = pd.DataFrame([{
            'url_length': url_length(url),
            'has_ip': has_ip(url),
            'has_at': has_at(url),
            'dots': count_dots(url),
            'has_https': has_https(url),
            'hyphens': count_hyphens(url),
            'digits': count_digits(url),
            'suspicious': has_suspicious_words(url),
            'domain_length': get_domain_length(url)
        }])

        # ======================
        # Prediction
        # ======================
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        confidence = round(max(probabilities) * 100, 2)

        # ======================
        # Result
        # ======================
        if prediction == 1:
            result = "⚠️ Phishing URL"
        else:
            result = "✅ Safe URL"

        # ======================
        # Confidence Level
        # ======================
        if confidence >= 90:
            level = "Very High Confidence 🔥"
        elif confidence >= 75:
            level = "High Confidence 💪"
        elif confidence >= 60:
            level = "Medium Confidence ⚠️"
        else:
            level = "Low Confidence 🤔"

        return render_template(
            'index.html',
            prediction_text=result,
            confidence=confidence,
            level=level
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text="⚠️ Error occurred",
            confidence=0,
            level=str(e)
        )

# ==============================
# ▶️ Run App
# ==============================
if __name__ == "__main__":
    app.run(debug=True)