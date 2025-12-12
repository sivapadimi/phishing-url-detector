# app.py
import joblib
from flask import Flask, request, render_template, jsonify, abort
import os

app = Flask(__name__)

MODEL_PATH = "phishing_url_model.pkl"
VECT_PATH = "tfidf_vectorizer.pkl"

# Load model & vectorizer once at startup
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECT_PATH):
    raise FileNotFoundError("Model or vectorizer file not found. Run train.py first to generate them.")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECT_PATH)

@app.route("/")
def index():
    return render_template("index.html")

# Prediction API: accepts GET ?url=...  (simple) or JSON POST {"url": "..."}
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        url = request.args.get("url", "")
    else:
        data = request.get_json(silent=True) or {}
        url = data.get("url", "")

    if not url:
        return jsonify({"error": "No URL provided."}), 400

    try:
        X_new = vectorizer.transform([url])
        prediction = model.predict(X_new)[0]
        # Optionally map common labels to 'safe'/'phishing'
        label = str(prediction)
        # Normalize: assume labels like 'phishing' or 'benign' or 'safe'
        if label.lower() in ("phishing", "malicious", "bad"):
            status = "phishing"
        elif label.lower() in ("benign", "safe", "legit", "good"):
            status = "safe"
        else:
            # fallback: if label is numeric (0/1) assume 1=phishing if training used so
            try:
                val = float(label)
                status = "phishing" if val == 1 else "safe"
            except:
                status = label.lower()

        return jsonify({"url": url, "prediction": label, "status": status}), 200
    except Exception as e:
        return jsonify({"error": "Prediction error", "detail": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
