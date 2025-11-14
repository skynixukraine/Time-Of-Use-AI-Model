# ⚡ AI Energy Optimization Model

An intelligent Python-based model for optimizing daily and monthly electricity usage and costs using AI-assisted analysis of power consumption, day/night tariffs, and local weather conditions.

---

## 🧠 Overview

This model predicts and optimizes energy costs by combining:
- Real-time **electricity pricing** (day & night rates)
- **Power consumption** data
- **Battery capacity** for storage systems
- **Weather-based generation forecasts** (for solar or hybrid systems)

It produces JSON-based analytical outputs and cost forecasts for both short-term and monthly usage scenarios.

---

## 🔍 Example Parameters

```python
power = 10            # Device power (kW)
price_day = 4.6       # Daytime electricity rate
price_night = 2.3     # Nighttime electricity rate
capacity = 45         # Battery capacity (kWh)
```

### 🧾 Example Output

```json
{
  "optimized_usage_pattern": "Use stored energy from 19:00 to 07:00",
  "total_day_cost": 120.75,
  "total_night_cost": 68.25,
  "average_daily_cost": 6.3,
  "monthly_savings_percent": 18.4
}
```

---

## ⚙️ Installation

```bash
git clone git@github.com:skynixukraine/Time-Of-Use-AI-Model.git
cd Time-Of-Use-AI-Model
pip install -r requirements.txt
```

---

## ▶️ Usage

Start the FastAPI service with Uvicorn:

```bash
uvicorn predict:app --app-dir /workspace --host 0.0.0.0 --port 8000 --workers 2
```

The API is served at `http://0.0.0.0:8000` (or `http://127.0.0.1:8000` locally) and exposes two JSON endpoints: `/train` and `/predict`.

---

## 🧪 Example API Requests (for Postman or cURL)

### Train (`POST /train`)

Endpoint

```http
POST http://localhost:8000/train
Content-Type: application/json
```

Body (JSON)

```jsonc
{
  "uid": "user123",            // Unique user identifier (required)
  "train_window_days": 14,      // optional: truncate history to the last N days
  "historical": [               // Time-ordered hourly history (required)
    {
      "month": 10,
      "day": 6,
      "hour": 14,
      "clouds": 35,
      "power": 10.0,
      "generation": 7.5,
      "consumption": 4.2,
      "battery_capacity": 45.0,
      "soc": 60.0,
      "price": 4.32,
      "temperature": 12.3,      // optional (default 0.0)
      "irradiance": 320.0       // optional (default 0.0)
    }
    // ... more records ...
  ]
}
```

ℹ️ Provide at least 48 hourly records so the model can assemble one training sequence (24 for context + 24 for prediction).

Response (JSON)

```json
{
  "status": "trained",
  "uid": "user123",
  "last_trained": "2024-05-01 10:22:48",
  "train_records": 336,
  "train_window_days": 14,
  "metrics": {
    "forecaster_train_loss": 0.0021,
    "forecaster_val_loss": 0.0024,
    "soc_train_loss": 0.0018,
    "soc_val_loss": 0.0020,
    "avg_slot_costs": [1.82, 1.76, 1.71]
  }
}
```

Validation errors return `{ "error": "message" }` with HTTP status `400`; unexpected failures return status `500`.

---

### Predict (`POST /predict`)

Endpoint

```http
POST http://localhost:8000/predict
Content-Type: application/json
```

Body (JSON)

```jsonc
{
  "uid": "user123",               // required
  "soc_current": 58.0,            // optional: current battery SOC (%). Defaults to 0 or current.soc
  "power": 10.0,                  // optional: max charge/discharge power (kW)
  "battery_capacity": 45.0,       // optional: battery capacity (kWh)
  "current": {                    // optional: latest measured point, used as fallback for missing values
    "month": 10,
    "day": 6,
    "hour": 23,
    "clouds": 48,
    "power": 10.0,
    "generation": 8.0,
    "consumption": 5.1,
    "battery_capacity": 45.0,
    "soc": 58.0,
    "price": 4.18
  },
  "forecast_24h": [               // list of 24+ hourly forecast records (required)
    {
      "month": 10,
      "day": 7,
      "hour": 8,
      "clouds": 25,
      "temperature": 9.4,
      "irradiance": 240.0,
      "generation": 6.1,         // optional (defaults to 0.0)
      "consumption": 3.8,        // optional (defaults to 0.0)
      "price": 4.32,             // optional (defaults to 0.0)
      "power": 10.0              // optional (falls back to request-level power)
    }
    // ... 23 more entries ...
  ]
}
```

Response (JSON)

```json
{
  "generation_pred": [6.0, 6.1],
  "consumption_pred": [3.5, 3.8],
  "slots": [
    {
      "start_hour": 0,
      "end_hour": 4,
      "soc_target_pct": 62.5,
      "estimated_cost": 1.72
    }
  ],
  "strategy_cost": {
    "slot_costs": [1.72, 1.68],
    "total_cost": 10.21
  },
  "model_soc_targets": [61.8, 63.1]
}
```

If the payload is invalid or the user has not been trained, the service responds with `{ "error": "message" }` and HTTP status `400` or `500`.

---

## 🧩 Tech Stack

- Python 3.9+
- Pandas — data processing
- Matplotlib — visualization
- Requests — API integration
- FastAPI — REST API
- Uvicorn — ASGI server
- Numpy, scikit-learn, joblib — ML logic

---

**Skynix Team**
[https://skynix.co/about-skynix](https://skynix.co/about-skynix)

A professional software development company specializing in advanced AI automation and energy efficiency systems.

---

## 📜 License

This project is **NOT open source**.  
Any use, copying, distribution, or modification of this code is **prohibited without explicit written permission from Skynix Team**.

© Skynix Team. All rights reserved.
