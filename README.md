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

```bash
python app.py
```

---

## 🧪 Example API Request (for Postman or cURL)

### Endpoint

```
POST http://localhost:8080/run
Content-Type: application/json
```

### Body (JSON)

```json
{
  "uid": "user123",
  "historical": [
    {
      "month": 10,
      "day": 6,
      "hour": 14,
      "clouds": 35,
      "power": 10,
      "generation": 7.5,
      "consumption": 4.2,
      "purchase": 0.0,
      "battery_capacity": 45,
      "soc": 60,
      "price": 4.32
    }
    // ...
  ],
  "forecast": [
    {
      "month": 10,
      "day": 7,
      "hour": 8,
      "clouds": 25,
      "soc": 55,
      "price": 4.32
    }
    // ...
  ]
}
```

💡 In a real use case:
- `historical` should contain at least several days (e.g., 14 × 24 = 336 records)
- `forecast` must include exactly 24 records (one per hour for the next day)

---

## 🧩 Tech Stack

- Python 3.9+
- Pandas — data processing
- Matplotlib — visualization
- Requests — API integration
- Flask — REST API
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