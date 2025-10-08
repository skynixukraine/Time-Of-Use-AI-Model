"""
AI TOU Optimization Model API
-----------------------------
This Flask-based API trains a personalized linear regression model
for each user (UID) and predicts optimal Time-of-Use (TOU) settings
based on historical and forecasted solar generation data.

Developed by Skynix Team — https://skynix.co/about-skynix
"""

from flask import Flask, request, jsonify
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Directory where user-specific AI models will be stored
MODEL_DIR = "user_models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_or_create_model(uid):
    """
    Loads the existing user model if available, otherwise creates a new one.
    Each user has their own model_<uid>.pkl file for independent training.
    """
    path = os.path.join(MODEL_DIR, f"model_{uid}.pkl")
    if os.path.exists(path):
        model = joblib.load(path)
    else:
        model = LinearRegression()
    return model, path


@app.route('/run', methods=['POST'])
def run_model():
    """
    Main API endpoint:
    - Accepts user ID, 14-day historical data, and 24-hour forecast
    - Trains (auto-feedback) the user-specific model
    - Predicts next 24h generation
    - Returns 6 optimized TOU slots with recommended target SOC levels
    """

    req = request.get_json()
    uid = req.get("uid")
    historical = req.get("historical", [])
    forecast = req.get("forecast", [])

    # Validate input data
    if not uid or len(historical) < 24 or len(forecast) != 24:
        return jsonify({"error": "Invalid input"}), 400

    # Load or initialize model
    model, path = load_or_create_model(uid)

    # ---- Auto-feedback training ----
    df_hist = pd.DataFrame(historical)
    X_hist = df_hist[["month", "hour", "clouds", "soc", "price"]]
    y_hist = df_hist["generation"]
    model.fit(X_hist, y_hist)
    joblib.dump(model, path)

    # ---- Forecast prediction ----
    df_forecast = pd.DataFrame(forecast)
    df_forecast["predicted_generation"] = model.predict(
        df_forecast[["month", "hour", "clouds", "soc", "price"]]
    )

    # ---- TOU Slot Optimization ----
    # Split the day into 6 slots of 4 hours each
    slots = []
    gen = df_forecast["predicted_generation"].values
    prices = df_forecast["price"].values
    slot_hours = np.array_split(range(24), 6)

    for i, hrs in enumerate(slot_hours):
        avg_gen = np.mean(gen[hrs])
        avg_price = np.mean(prices[hrs])

        # Example heuristic: higher price or lower generation → higher SOC
        target_soc = np.clip(100 - avg_gen * 5 + avg_price * 2, 20, 100)

        slots.append({
            "slot": i + 1,
            "start_hour": int(hrs[0]),
            "end_hour": int(hrs[-1]),
            "target_soc": round(target_soc, 1)
        })

    # Return response
    return jsonify({
        "uid": uid,
        "slots": slots,
        "message": "Prediction complete with auto-feedback training"
    })


if __name__ == '__main__':
    # Run Flask server
    app.run(host='0.0.0.0', port=8080)
