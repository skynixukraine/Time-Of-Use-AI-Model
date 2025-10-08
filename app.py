"""
AI TOU Optimization Model API with Cost-Aware TensorBoard Logging
-----------------------------------------------------------------
Flask API that:
1. Trains a user-specific linear regression model on historical data
2. Predicts solar generation for the next 24h
3. Optimizes 6 Time-of-Use slots to minimize electricity costs
4. Logs metrics to TensorBoard for monitoring model behavior

Developed by Skynix Team — https://skynix.co/about-skynix
"""

from flask import Flask, request, jsonify
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from torch.utils.tensorboard import SummaryWriter

app = Flask(__name__)

# ---- Configuration ----
MODEL_DIR = "user_models"
LOG_DIR = "logs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def load_or_create_model(uid):
    """
    Load or initialize a user-specific model.
    Each user has a separate model_<uid>.pkl file for independent training.
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
    - Accepts user ID, historical data (14 days), and forecast (24 hours)
    - Trains the model (auto-feedback)
    - Predicts next 24h generation
    - Computes 6 TOU slots to minimize cost
    - Logs cost, generation, SOC, and price to TensorBoard
    """

    req = request.get_json()
    uid = req.get("uid")
    historical = req.get("historical", [])
    forecast = req.get("forecast", [])

    # Validate inputs
    if not uid or len(historical) < 24 or len(forecast) != 24:
        return jsonify({"error": "Invalid input"}), 400

    # Load or create user-specific model
    model, model_path = load_or_create_model(uid)

    # ---- TensorBoard Setup ----
    log_path = os.path.join(LOG_DIR, f"user_{uid}")
    writer = SummaryWriter(log_dir=log_path)

    # ---- Train model (Auto-feedback) ----
    df_hist = pd.DataFrame(historical)
    X_hist = df_hist[["month", "hour", "clouds", "soc", "price"]]
    y_hist = df_hist["generation"]

    model.fit(X_hist, y_hist)
    joblib.dump(model, model_path)

    # ---- TensorBoard Training Metrics ----
    writer.add_scalar("training/num_samples", len(X_hist), 0)
    writer.add_scalar("training/mean_generation", y_hist.mean(), 0)
    writer.add_scalar("training/mean_soc", df_hist["soc"].mean(), 0)
    writer.flush()

    # ---- Predict Next 24 Hours ----
    df_forecast = pd.DataFrame(forecast)
    df_forecast["predicted_generation"] = model.predict(
        df_forecast[["month", "hour", "clouds", "soc", "price"]]
    )

    # ---- TOU Optimization (6 Slots) ----
    slots = []
    gen = df_forecast["predicted_generation"].values
    prices = df_forecast["price"].values
    soc_current = df_forecast["soc"].values
    slot_hours = np.array_split(range(24), 6)

    for i, hrs in enumerate(slot_hours):
        avg_gen = np.mean(gen[hrs])
        avg_price = np.mean(prices[hrs])
        avg_soc = np.mean(soc_current[hrs])

        # Simple heuristic: minimize cost by adjusting SOC
        target_soc = np.clip(100 - avg_gen * 5 + avg_price * 2, 20, 100)

        # Estimated electricity cost for the slot
        # Assume battery charges to reach target SOC, any deficit is bought from grid
        slot_cost = max(0, target_soc - avg_soc) * avg_price

        slots.append({
            "slot": i + 1,
            "start_hour": int(hrs[0]),
            "end_hour": int(hrs[-1]),
            "target_soc": round(target_soc, 1)
        })

        # ---- TensorBoard Logging per slot ----
        writer.add_scalar(f"slots/slot_{i+1}_avg_generation", avg_gen, 0)
        writer.add_scalar(f"slots/slot_{i+1}_avg_price", avg_price, 0)
        writer.add_scalar(f"slots/slot_{i+1}_avg_soc", avg_soc, 0)
        writer.add_scalar(f"slots/slot_{i+1}_target_soc", target_soc, 0)
        writer.add_scalar(f"slots/slot_{i+1}_estimated_cost", slot_cost, 0)
        # Logging cost allows monitoring how well the model minimizes expenses

    writer.flush()
    writer.close()

    # ---- Return Response ----
    return jsonify({
        "uid": uid,
        "slots": slots,
        "message": "Prediction complete. Model trained and cost metrics logged to TensorBoard."
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
