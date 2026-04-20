# ==============================
# 📦 Imports
# ==============================
import pandas as pd
import matplotlib.pyplot as plt
import re
import joblib
from urllib.parse import urlparse

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==============================
# 📂 Load Dataset
# ==============================
df = pd.read_csv(
    "phishing_urls.csv",
    encoding='latin-1',
    on_bad_lines='skip',
    low_memory=False
)

print("✅ Dataset Loaded Successfully")
print(df.head())

# ==============================
# 🧹 Data Cleaning
# ==============================
df = df.dropna()
df['label'] = df['label'].astype(int)

print("\n✅ Data Cleaned")
print("Shape:", df.shape)

# ==============================
# 🧠 Feature Extraction Functions
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
# ⚙️ Apply Features
# ==============================
df['url_length'] = df['domain'].apply(url_length)
df['has_ip'] = df['domain'].apply(has_ip)
df['has_at'] = df['domain'].apply(has_at)
df['dots'] = df['domain'].apply(count_dots)
df['has_https'] = df['domain'].apply(has_https)
df['hyphens'] = df['domain'].apply(count_hyphens)
df['digits'] = df['domain'].apply(count_digits)
df['suspicious'] = df['domain'].apply(has_suspicious_words)
df['domain_length'] = df['domain'].apply(get_domain_length)

# ==============================
# 🎯 Features & Target
# ==============================
X = df[
    ['url_length','has_ip','has_at','dots','has_https',
     'hyphens','digits','suspicious','domain_length']
]
y = df['label']

# ==============================
# ✂️ Train Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 🤖 Train Model
# ==============================
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

print("\n✅ Model Trained")

# ==============================
# 📊 Evaluation
# ==============================
y_pred = model.predict(X_test)

print("\n📊 Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ==============================
# 🔍 Feature Importance
# ==============================
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(10,6))
plt.barh(features, importances)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# ==============================
# 💾 Save Model
# ==============================
joblib.dump(model, "model.pkl")
print("\n✅ Model saved as model.pkl")