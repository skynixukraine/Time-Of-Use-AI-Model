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

🧾 Example Output
```{
  "optimized_usage_pattern": "Use stored energy from 19:00 to 07:00",
  "total_day_cost": 120.75,
  "total_night_cost": 68.25,
  "average_daily_cost": 6.3,
  "monthly_savings_percent": 18.4
}
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

## 📊 Visualization

The model can display:

- Hourly and daily consumption charts
- Cost optimization trends
- Battery charge/discharge simulation

---

## 🧩 Tech Stack

- Python 3.9+
- Pandas — data processing
- Matplotlib — visualization
- Requests — API integration
- AI-assisted logic for optimization

---

## 🧑‍💻 Developed by

**Skynix Team**  
[https://skynix.co/about-skynix](https://skynix.co/about-skynix)

A professional software development company specializing in advanced AI automation and energy efficiency systems.

---

## 📜 License

This project is **NOT open source**.  
Any use, copying, distribution, or modification of this code is **prohibited without explicit written permission from Skynix Team**.

© Skynix Team. All rights reserved.