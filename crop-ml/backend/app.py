"""
app.py
------
Flask backend that loads the trained model and exposes a /predict API
for the frontend (crop-advisor.html) to call.

Run with: python3 backend/app.py
Then it's live at: http://127.0.0.1:5000

Endpoints:
  GET  /              -> health check
  POST /predict        -> takes N, P, K, temperature, humidity, ph, rainfall
                          returns the predicted crop + top 3 probabilities
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load the trained model, scaler, and label encoder once at startup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")

model = joblib.load(os.path.join(MODEL_DIR, "crop_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


# ---------------------------------------------------------------------------
# Allow the frontend (opened as a local HTML file or hosted elsewhere)
# to call this API from the browser. Manual CORS headers since flask-cors
# isn't required for a small student project.
# ---------------------------------------------------------------------------
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "message": "Crop recommendation API is running.",
        "crops_known": len(encoder.classes_),
    })


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    # Browsers send an OPTIONS preflight request before POST; respond empty.
    if request.method == "OPTIONS":
        return "", 204

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Send a JSON body with N, P, K, temperature, humidity, ph, rainfall."}), 400

    # ---------------------------------------------------------------------
    # Validate that every required field is present and numeric
    # ---------------------------------------------------------------------
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": "Missing fields: " + ", ".join(missing)}), 400

    try:
        values = [float(data[f]) for f in FEATURES]
    except (TypeError, ValueError):
        return jsonify({"error": "All fields must be numbers."}), 400

    # ---------------------------------------------------------------------
    # Scale input the same way the training data was scaled, then predict
    # ---------------------------------------------------------------------
    X = pd.DataFrame([values], columns=FEATURES)
    X_scaled = scaler.transform(X)

    probabilities = model.predict_proba(X_scaled)[0]
    top3_idx = np.argsort(probabilities)[::-1][:3]

    top3 = [
        {
            "crop": encoder.inverse_transform([idx])[0],
            "confidence": round(float(probabilities[idx]) * 100, 2),
        }
        for idx in top3_idx
    ]

    return jsonify({
        "input": dict(zip(FEATURES, values)),
        "recommended_crop": top3[0]["crop"],
        "top_3": top3,
    })


if __name__ == "__main__":
    # debug=True auto-reloads on code changes — turn off for any real deployment
    app.run(debug=True, port=5000)
