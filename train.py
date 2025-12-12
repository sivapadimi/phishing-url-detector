# train.py
import zipfile
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# --- CONFIG ---
DATA_ZIP = "phishing_dataset/PhiUSIIL_Phishing_URL_Dataset.csv.zip"  # optional
DATA_CSV = "phishing_dataset/PhiUSIIL_Phishing_URL_Dataset.csv"
MODEL_OUT = "phishing_url_model.pkl"
VECT_OUT = "tfidf_vectorizer.pkl"
RANDOM_STATE = 42

# --- Unzip if needed ---
if os.path.exists(DATA_ZIP) and not os.path.exists(DATA_CSV):
    with zipfile.ZipFile(DATA_ZIP, 'r') as zip_ref:
        zip_ref.extractall("phishing_dataset")
    print("Unzipped dataset.")

# --- Load CSV ---
if not os.path.exists(DATA_CSV):
    raise FileNotFoundError(f"Dataset not found at {DATA_CSV}. Put your CSV there.")

df = pd.read_csv(DATA_CSV)
print("Columns:", df.columns.tolist())

# Normalize column names
df.columns = df.columns.str.strip().str.lower()

# Ensure required columns exist
if 'url' not in df.columns or 'label' not in df.columns:
    raise ValueError("Dataset must contain 'url' and 'label' columns (case-insensitive).")

# --- Preprocessing & vectorization ---
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3,5))
X = vectorizer.fit_transform(df['url'].astype(str))
y = df['label'].astype(str)

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# --- Train model ---
model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
model.fit(X_train, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# --- Save ---
joblib.dump(model, MODEL_OUT)
joblib.dump(vectorizer, VECT_OUT)
print(f"Saved model to {MODEL_OUT} and vectorizer to {VECT_OUT}")
