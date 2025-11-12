"""
TOU LSTM Forecaster + SOC Controller service.
- PyTorch models (Forecaster, SOCController)
- Dataset builder from historical JSON
- LP label generator (PuLP)
- Training loops with TensorBoard logging
- Flask endpoints: /train and /predict
- Per-user model structure under user_models/{uid}/

Developed by Skynix Team — https://skynix.co/about-skynix
"""

import os
import json
import time
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
import pulp
from flask import Flask, request, jsonify

# ---------------------------
# Config / Hyperparameters
# ---------------------------
DEFAULT_SEQ_LEN = 24
DEFAULT_PRED_LEN = 24
FORECASTER_HIDDEN = 64
FORECASTER_LAYERS = 2
SOC_HIDDEN = 64
BATCH_SIZE = 64
EPOCHS_FORECASTER = 20
EPOCHS_SOC = 30
LEARNING_RATE_FORECASTER = 1e-3
LEARNING_RATE_SOC = 5e-4
WG = 1.0
WC = 1.0
LAMBDA_SMOOTH = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USER_MODELS_DIR = "user_models"
os.makedirs(USER_MODELS_DIR, exist_ok=True)

# ---------------------------
# Utilities
# ---------------------------
def hour_sin_cos(hour: np.ndarray) -> np.ndarray:
    radians = 2 * np.pi * (hour / 24.0)
    return np.stack([np.sin(radians), np.cos(radians)], axis=-1)

def month_sin_cos(month: np.ndarray) -> np.ndarray:
    radians = 2 * np.pi * ((month - 1) / 12.0)
    return np.stack([np.sin(radians), np.cos(radians)], axis=-1)

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    with open(path) as f:
        return json.load(f)

# ---------------------------
# Dataset
# ---------------------------
class TOUDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len=24, pred_len=24, uid: str = "global"):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.df = df.reset_index(drop=True)
        self.uid = uid
        self.samples = []
        n = len(self.df)
        for start in range(0, n - seq_len - pred_len + 1):
            self.samples.append((slice(start, start + seq_len), slice(start + seq_len, start + seq_len + pred_len)))
        self.scalers = {}

    def set_scalers(self, scaler_dict: Dict[str, Dict[str, float]]):
        self.scalers = scaler_dict

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_slice, target_slice = self.samples[idx]
        seq = self.df.iloc[seq_slice]
        target = self.df.iloc[target_slice]

        hours = seq['hour'].to_numpy()
        months = seq['month'].to_numpy()
        hour_cc = hour_sin_cos(hours)
        month_cc = month_sin_cos(months)
        clouds = seq['clouds'].to_numpy().reshape(-1, 1)
        temp = seq['temperature'].to_numpy().reshape(-1, 1)
        irr = seq['irradiance'].to_numpy().reshape(-1, 1)
        power = seq['power'].to_numpy().reshape(-1, 1)
        gen = seq['generation'].to_numpy().reshape(-1, 1)
        cons = seq['consumption'].to_numpy().reshape(-1, 1)
        price = seq['price'].to_numpy().reshape(-1, 1)

        def apply_scaler(arr, name):
            if name in self.scalers:
                m = self.scalers[name]['mean']
                s = self.scalers[name]['std'] if self.scalers[name]['std'] != 0 else 1.0
                return (arr - m) / s
            return arr

        gen_s = apply_scaler(gen, 'generation')
        cons_s = apply_scaler(cons, 'consumption')
        price_s = apply_scaler(price, 'price')
        power_s = apply_scaler(power, 'power')

        X_seq = np.concatenate([hour_cc, month_cc, clouds, temp, irr, power_s, gen_s, cons_s, price_s], axis=-1)
        y_g = self.df.iloc[target_slice]['generation'].to_numpy()
        y_c = self.df.iloc[target_slice]['consumption'].to_numpy()

        meta = {
            'price': self.df.iloc[target_slice]['price'].to_numpy(),
            'generation': self.df.iloc[target_slice]['generation'].to_numpy(),
            'consumption': self.df.iloc[target_slice]['consumption'].to_numpy(),
            'battery_capacity': float(self.df.iloc[seq_slice]['battery_capacity'].to_numpy()[-1]),
            'soc_start': float(self.df.iloc[seq_slice]['soc'].to_numpy()[-1]),
            'power': float(self.df.iloc[seq_slice]['power'].to_numpy()[-1]),
        }

        return {'X': X_seq.astype(np.float32), 'y_g': y_g.astype(np.float32),
                'y_c': y_c.astype(np.float32), 'meta': meta}

# ---------------------------
# Models
# ---------------------------
class Forecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size=FORECASTER_HIDDEN, n_layers=FORECASTER_LAYERS,
                 dropout=0.2, pred_len=24):
        super().__init__()
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                               num_layers=n_layers, batch_first=True, bidirectional=True,
                               dropout=dropout if n_layers > 1 else 0.0)
        self.fc = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.ReLU(), nn.Dropout(dropout))
        self.g_head = nn.Linear(hidden_size, pred_len)
        self.c_head = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        out, _ = self.encoder(x)
        h_last = out[:, -1, :]
        h = self.fc(h_last)
        return self.g_head(h), self.c_head(h)

class SOCController(nn.Module):
    def __init__(self, input_size: int, hidden_size=SOC_HIDDEN, n_layers=1, dropout=0.2,
                 n_targets=6, min_reserve=10.0, max_soc=100.0):
        super().__init__()
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                               num_layers=n_layers, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_targets)
        )
        self.min_reserve = min_reserve
        self.max_soc = max_soc

    def forward(self, x):
        out, _ = self.encoder(x)
        raw = self.mlp(out[:, -1, :])
        return torch.sigmoid(raw) * (self.max_soc - self.min_reserve) + self.min_reserve

# ---------------------------
# LP target generator
# ---------------------------
def build_lp_soc_targets(gen, cons, price, soc_start, capacity, power_limit, slots=None):
    T = 24
    if slots is None:
        slots = [3, 7, 11, 15, 19, 23]
    prob = pulp.LpProblem("soc_lp", pulp.LpMinimize)
    p = [pulp.LpVariable(f"p_{t}", lowBound=0, upBound=power_limit) for t in range(T)]
    E = [pulp.LpVariable(f"E_{t}", lowBound=0, upBound=capacity) for t in range(T + 1)]
    E0 = soc_start / 100.0 * capacity
    prob += E[0] == E0
    for t in range(T):
        prob += E[t + 1] == E[t] + float(gen[t]) - float(cons[t]) + p[t]
    prob += pulp.lpSum([float(price[t]) * p[t] for t in range(T)])
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[prob.status] != 'Optimal':
        return np.array([soc_start] * len(slots))
    E_vals = np.array([pulp.value(E[t]) for t in range(T + 1)])
    socs = [100.0 * E_vals[i + 1] / capacity for i in slots]
    return np.clip(socs, 0, 100)

# ---------------------------
# Training pipeline
# ---------------------------
def prepare_dataset_from_json(historical: List[Dict[str, Any]], uid: str):
    df = pd.DataFrame(historical).fillna(method='ffill').fillna(method='bfill').reset_index(drop=True)
    scaler_info = {}
    for col in ['generation', 'consumption', 'price', 'power']:
        arr = df[col].to_numpy().astype(np.float64)
        scaler_info[col] = {'mean': float(np.mean(arr)), 'std': float(np.std(arr))}
    ds = TOUDataset(df, seq_len=DEFAULT_SEQ_LEN, pred_len=DEFAULT_PRED_LEN, uid=uid)
    ds.set_scalers(scaler_info)
    return ds, scaler_info

def full_train_pipeline(uid: str, historical: List[Dict[str, Any]], train_window_days: int = 14):
    user_dir = os.path.join(USER_MODELS_DIR, uid)
    os.makedirs(user_dir, exist_ok=True)
    os.makedirs(os.path.join(user_dir, "logs"), exist_ok=True)

    hist_path = os.path.join(user_dir, f"history_{uid}.json")
    forecaster_path = os.path.join(user_dir, f"forecaster_{uid}.pt")
    soc_path = os.path.join(user_dir, f"soc_controller_{uid}.pt")
    scaler_path = os.path.join(user_dir, f"scaler_{uid}.json")
    meta_path = os.path.join(user_dir, f"meta_{uid}.json")

    save_json(historical, hist_path)
    ds, scalers = prepare_dataset_from_json(historical, uid)
    save_json(scalers, scaler_path)

    # padding short data
    if len(historical) < 48:
        print(f"[WARN] Padding history for {uid}")
        last = historical[-1]
        while len(historical) < 48:
            new = dict(last)
            new["hour"] = (new["hour"] + 1) % 24
            historical.append(new)
        ds, scalers = prepare_dataset_from_json(historical, uid)
        save_json(scalers, scaler_path)

    sample0 = ds[0]
    input_size = sample0['X'].shape[1]
    train_forecaster(ds, uid, input_size)
    train_soc_controller([s['X'] for s in ds], [build_lp_soc_targets(
        s['meta']['generation'], s['meta']['consumption'], s['meta']['price'],
        s['meta']['soc_start'], s['meta']['battery_capacity'], s['meta']['power']
    ) for s in ds], uid, input_size)

    meta = {
        "uid": uid,
        "last_trained": time.strftime("%Y-%m-%d %H:%M:%S"),
        "forecaster_path": forecaster_path,
        "soc_path": soc_path,
        "scaler_path": scaler_path,
        "train_records": len(historical)
    }
    save_json(meta, meta_path)
    return {"status": "trained", "meta_path": meta_path}

# ---------------------------
# Train helpers
# ---------------------------
def train_forecaster(dataset, uid, input_size):
    model = Forecaster(input_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_FORECASTER)
    criterion = nn.MSELoss()
    for epoch in range(EPOCHS_FORECASTER):
        total = 0
        for item in dataset:
            X = torch.tensor(item['X']).unsqueeze(0).to(DEVICE)
            y_g = torch.tensor(item['y_g']).unsqueeze(0).to(DEVICE)
            y_c = torch.tensor(item['y_c']).unsqueeze(0).to(DEVICE)
            optimizer.zero_grad()
            g_pred, c_pred = model(X)
            loss = criterion(g_pred, y_g) + criterion(c_pred, y_c)
            loss.backward(); optimizer.step()
            total += loss.item()
        print(f"[Forecaster][{uid}] Epoch {epoch+1}/{EPOCHS_FORECASTER} Loss {total/len(dataset):.5f}")
    torch.save(model.state_dict(), os.path.join(USER_MODELS_DIR, uid, f"forecaster_{uid}.pt"))

def train_soc_controller(Xs, Ys, uid, input_size):
    model = SOCController(input_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_SOC)
    criterion = nn.MSELoss()
    Xs = np.stack(Xs); Ys = np.stack(Ys)
    for epoch in range(EPOCHS_SOC):
        idx = np.random.permutation(len(Xs))
        total = 0
        for i in idx:
            X = torch.tensor(Xs[i]).unsqueeze(0).to(DEVICE)
            y = torch.tensor(Ys[i]).unsqueeze(0).to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward(); optimizer.step()
            total += loss.item()
        print(f"[SOC][{uid}] Epoch {epoch+1}/{EPOCHS_SOC} Loss {total/len(Xs):.5f}")
    torch.save(model.state_dict(), os.path.join(USER_MODELS_DIR, uid, f"soc_controller_{uid}.pt"))

# ---------------------------
# Prediction
# ---------------------------
def load_models_for_uid(uid, input_size):
    user_dir = os.path.join(USER_MODELS_DIR, uid)
    forecaster_state = os.path.join(user_dir, f"forecaster_{uid}.pt")
    soc_state = os.path.join(user_dir, f"soc_controller_{uid}.pt")
    scaler_path = os.path.join(user_dir, f"scaler_{uid}.json")
    scalers = load_json(scaler_path)
    fore = Forecaster(input_size).to(DEVICE)
    fore.load_state_dict(torch.load(forecaster_state, map_location=DEVICE)); fore.eval()
    soc = SOCController(input_size).to(DEVICE)
    soc.load_state_dict(torch.load(soc_state, map_location=DEVICE)); soc.eval()
    return fore, soc, scalers

def predict_for_uid(uid, forecast_24h, soc_current, power, capacity):
    df = pd.DataFrame(forecast_24h)
    if 'price' not in df.columns: df['price'] = 0
    if 'power' not in df.columns: df['power'] = power
    hours = df['hour'].to_numpy(); months = df['month'].to_numpy()
    hour_cc = hour_sin_cos(hours); month_cc = month_sin_cos(months)
    X_seq = np.concatenate([
        hour_cc, month_cc,
        df[['clouds','temperature','irradiance','power']].to_numpy(),
        np.zeros((24,3))
    ], axis=1).astype(np.float32)
    input_size = X_seq.shape[1]
    fore, soc_model, _ = load_models_for_uid(uid, input_size)
    with torch.no_grad():
        g, c = fore(torch.tensor(X_seq).unsqueeze(0).to(DEVICE))
    g = g.cpu().numpy().ravel(); c = c.cpu().numpy().ravel()
    with torch.no_grad():
        socs = soc_model(torch.tensor(X_seq).unsqueeze(0).to(DEVICE)).cpu().numpy().ravel()
    socs = np.maximum(socs, soc_current - 1.0)
    slots = [{"start_hour": i*4, "end_hour": (i+1)*4, "soc_target_pct": float(round(v,3))} for i,v in enumerate(socs)]
    return {"generation_pred": g.tolist(), "consumption_pred": c.tolist(), "slots": slots}

# ---------------------------
# Flask
# ---------------------------
app = Flask(__name__)

@app.route("/train", methods=["POST"])
def train_endpoint():
    data = request.get_json()
    uid = data.get("uid"); hist = data.get("historical")
    if not uid or not hist:
        return jsonify({"error":"uid and historical required"}),400
    try:
        result = full_train_pipeline(uid, hist)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}),500

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    data = request.get_json()
    uid = data.get("uid"); forecast = data.get("forecast_24h")
    soc = float(data.get("soc_current",0)); power=float(data.get("power",0)); cap=float(data.get("battery_capacity",1))
    if not uid or not forecast:
        return jsonify({"error":"uid and forecast_24h required"}),400
    try:
        return jsonify(predict_for_uid(uid, forecast, soc, power, cap))
    except Exception as e:
        return jsonify({"error":str(e)}),500

if __name__ == "__main__":
    print("Starting TOU service on http://127.0.0.1:5001")
    app.run(host="0.0.0.0", port=5001, debug=True)
