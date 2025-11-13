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

from __future__ import annotations

import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:  # pragma: no cover - executed when PuLP is unavailable
    pulp = None
    PULP_AVAILABLE = False
from flask import Flask, request, jsonify
import shutil

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset, default_collate
    from torch.utils.tensorboard import SummaryWriter
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - executed when torch is unavailable
    torch = None
    nn = None
    TORCH_AVAILABLE = False

    class Dataset:  # minimal stub so our Dataset subclass can still initialise
        def __init__(self, *args, **kwargs):
            pass

        def __len__(self):  # pragma: no cover - torch-free mode avoids DataLoader usage
            return 0

        def __getitem__(self, item):  # pragma: no cover
            raise NotImplementedError("Dataset iteration requires PyTorch to be installed")

    class DataLoader:  # pragma: no cover - we don't rely on DataLoader without torch
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for DataLoader operations")

    class Subset:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for Dataset subset operations")

    class TensorDataset:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for TensorDataset operations")

    def default_collate(*args, **kwargs):  # pragma: no cover
        raise RuntimeError("PyTorch is required for default_collate operations")

    class SummaryWriter:  # pragma: no cover - noop writer when torch is unavailable
        def __init__(self, *args, **kwargs):
            self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def add_scalar(self, *args, **kwargs):
            pass

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
DEVICE = torch.device("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else "cpu"

USER_MODELS_DIR = "user_models"
os.makedirs(USER_MODELS_DIR, exist_ok=True)

HISTORICAL_REQUIRED_FIELDS = {
    "month",
    "hour",
    "clouds",
    "power",
    "generation",
    "consumption",
    "price",
    "battery_capacity",
    "soc",
}

HISTORICAL_OPTIONAL_DEFAULTS = {
    "temperature": 0.0,
    "irradiance": 0.0,
}

FORECAST_REQUIRED_FIELDS = {
    "month",
    "hour",
    "clouds",
    "temperature",
    "irradiance",
}

FORECAST_OPTIONAL_DEFAULTS = {
    "price": 0.0,
    "generation": 0.0,
    "consumption": 0.0,
}

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


def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def validate_records(
    records: List[Dict[str, Any]],
    required_fields: set,
    context: str,
    optional_defaults: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if not isinstance(records, list) or not records:
        raise ValueError(f"{context} must be a non-empty list of records")
    normalised: List[Dict[str, Any]] = []
    for idx, row in enumerate(records):
        if not isinstance(row, dict):
            raise ValueError(f"{context} entry at index {idx} must be an object")
        item = dict(row)
        if optional_defaults:
            for key, default_value in optional_defaults.items():
                if item.get(key) is None:
                    item[key] = default_value() if callable(default_value) else default_value
        missing = required_fields - item.keys()
        if missing:
            raise ValueError(
                f"{context} entry at index {idx} is missing fields: {sorted(missing)}"
            )
        for key in required_fields:
            value = item.get(key)
            if value is None:
                raise ValueError(f"{context} entry at index {idx} has null value for '{key}'")
            if key in {"month", "hour"}:
                if not isinstance(value, (int, np.integer)):
                    raise ValueError(f"{context} entry at index {idx} has non-integer '{key}'")
            else:
                if not isinstance(value, (int, float, np.integer, np.floating)):
                    raise ValueError(f"{context} entry at index {idx} has non-numeric '{key}'")
        normalised.append(item)
    return normalised


def backup_existing_file(path: str) -> None:
    if os.path.exists(path):
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        backup_path = f"{path}.bak-{timestamp}"
        shutil.copy2(path, backup_path)


def safe_model_save(model_obj, path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        ensure_directory(directory)
    backup_existing_file(path)
    if TORCH_AVAILABLE and isinstance(model_obj, nn.Module):
        torch.save(model_obj.state_dict(), path)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(model_obj, f, indent=2)

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


def collate_sequences(batch: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("collate_sequences requires PyTorch to be installed")
    X = torch.from_numpy(np.stack([item['X'] for item in batch])).float()
    y_g = torch.from_numpy(np.stack([item['y_g'] for item in batch])).float()
    y_c = torch.from_numpy(np.stack([item['y_c'] for item in batch])).float()
    return X, y_g, y_c


def split_dataloaders(
    dataset: Dataset,
    batch_size: int,
    val_ratio: float = 0.2,
    collate_fn=None,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("split_dataloaders requires PyTorch to be installed")
    n_samples = len(dataset)
    if n_samples == 0:
        raise ValueError("Dataset is empty after preprocessing")
    indices = np.arange(n_samples)
    rng = np.random.default_rng()
    rng.shuffle(indices)
    val_count = int(max(1, round(n_samples * val_ratio))) if n_samples > 1 else 0
    val_count = min(val_count, n_samples - 1) if n_samples > 1 else 0
    if val_count <= 0:
        train_indices = indices.tolist()
        val_loader = None
    else:
        val_indices = indices[:val_count].tolist()
        train_indices = indices[val_count:].tolist()
        val_subset = Subset(dataset, val_indices)
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn or collate_sequences,
        )
    train_subset = Subset(dataset, train_indices)
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn or collate_sequences,
    )
    return train_loader, val_loader

# ---------------------------
# Models
# ---------------------------
if TORCH_AVAILABLE:

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

else:

    class Forecaster:  # pragma: no cover - stub used when torch is unavailable
        pass


    class SOCController:  # pragma: no cover - stub used when torch is unavailable
        pass


class TorchForecasterWrapper:
    def __init__(self, model: "Forecaster"):
        self.model = model.to(DEVICE)
        self.model.eval()

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        tensor_input = torch.tensor(features).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            generation, consumption = self.model(tensor_input)
        return generation.cpu().numpy().ravel(), consumption.cpu().numpy().ravel()


class SimpleForecasterArtifact:
    def __init__(self, mean_generation: float, mean_consumption: float, pred_len: int):
        self.mean_generation = float(mean_generation)
        self.mean_consumption = float(mean_consumption)
        self.pred_len = int(pred_len)

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        length = self.pred_len or features.shape[0]
        gen = np.full(length, self.mean_generation, dtype=np.float32)
        cons = np.full(length, self.mean_consumption, dtype=np.float32)
        return gen, cons

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "simple_forecaster",
            "mean_generation": self.mean_generation,
            "mean_consumption": self.mean_consumption,
            "pred_len": self.pred_len,
            "backend": "numpy",
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimpleForecasterArtifact":
        if data.get("type") != "simple_forecaster":
            raise ValueError("Invalid forecaster artifact")
        return cls(
            mean_generation=data.get("mean_generation", 0.0),
            mean_consumption=data.get("mean_consumption", 0.0),
            pred_len=int(data.get("pred_len", DEFAULT_PRED_LEN)),
        )


class TorchSOCControllerWrapper:
    def __init__(self, model: "SOCController"):
        self.model = model.to(DEVICE)
        self.model.eval()

    def predict(self, features: np.ndarray) -> np.ndarray:
        tensor_input = torch.tensor(features).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            targets = self.model(tensor_input)
        return targets.cpu().numpy().ravel()


class SimpleSOCControllerArtifact:
    def __init__(self, mean_targets: List[float]):
        arr = np.asarray(mean_targets, dtype=np.float32)
        if arr.size == 0:
            arr = np.zeros(6, dtype=np.float32)
        self.mean_targets = arr

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self.mean_targets, dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "simple_soc_controller",
            "mean_targets": self.mean_targets.tolist(),
            "backend": "numpy",
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimpleSOCControllerArtifact":
        if data.get("type") != "simple_soc_controller":
            raise ValueError("Invalid SOC controller artifact")
        return cls(mean_targets=data.get("mean_targets", []))

# ---------------------------
# LP target generator
# ---------------------------
def build_lp_soc_targets(
    gen: np.ndarray,
    cons: np.ndarray,
    price: np.ndarray,
    soc_start: float,
    capacity: float,
    power_limit: float,
    slots: Optional[List[int]] = None,
    min_reserve_pct: float = 10.0,
) -> Dict[str, Any]:
    T = 24
    if slots is None:
        slots = [3, 7, 11, 15, 19, 23]
    if not PULP_AVAILABLE:
        fallback = np.array([soc_start] * len(slots), dtype=np.float32)
        return {"soc_targets": fallback, "slot_costs": [0.0] * len(slots), "total_cost": 0.0}
    if capacity <= 0:
        fallback = np.array([soc_start] * len(slots), dtype=np.float32)
        return {"soc_targets": fallback, "slot_costs": [0.0] * len(slots), "total_cost": 0.0}
    if len(gen) < T or len(cons) < T or len(price) < T:
        raise ValueError("LP target generation requires 24 hourly points")
    prob = pulp.LpProblem("soc_lp", pulp.LpMinimize)
    max_power = max(power_limit, 0.0)
    charge = [pulp.LpVariable(f"charge_{t}", lowBound=0, upBound=max_power) for t in range(T)]
    discharge = [pulp.LpVariable(f"discharge_{t}", lowBound=0, upBound=max_power) for t in range(T)]
    grid = [pulp.LpVariable(f"grid_{t}", lowBound=-max_power, upBound=max_power) for t in range(T)]
    energy = [pulp.LpVariable(f"E_{t}", lowBound=capacity * min_reserve_pct / 100.0, upBound=capacity) for t in range(T + 1)]

    initial_energy = soc_start / 100.0 * capacity
    prob += energy[0] == initial_energy

    for t in range(T):
        net_load = float(cons[t]) - float(gen[t]) + charge[t] - discharge[t]
        prob += grid[t] == net_load
        prob += energy[t + 1] == energy[t] + float(gen[t]) - float(cons[t]) + charge[t] - discharge[t]

    # Objective: minimise energy cost and encourage smooth SOC trajectory
    smooth_abs = [pulp.LpVariable(f"smooth_{t}", lowBound=0) for t in range(T)]
    for t in range(T):
        prob += discharge[t] <= energy[t]  # cannot discharge more than available energy
        delta = energy[t + 1] - energy[t]
        prob += smooth_abs[t] >= delta
        prob += smooth_abs[t] >= -delta

    cost_expr = pulp.lpSum(float(price[t]) * grid[t] for t in range(T))
    smooth_penalty = pulp.lpSum(smooth_abs[t] for t in range(T)) * (
        LAMBDA_SMOOTH / (capacity if capacity else 1.0)
    )
    prob += cost_expr + smooth_penalty

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[prob.status] != 'Optimal':
        fallback = np.array([soc_start] * len(slots), dtype=np.float32)
        return {"soc_targets": fallback, "slot_costs": [0.0] * len(slots), "total_cost": 0.0}

    energy_values = np.array([pulp.value(energy[t]) for t in range(T + 1)])
    grid_values = np.array([pulp.value(grid[t]) for t in range(T)])
    soc_targets = [100.0 * energy_values[i + 1] / capacity for i in slots]

    slot_costs = []
    for slot_idx, slot_end in enumerate(slots):
        slot_start = 0 if slot_idx == 0 else slots[slot_idx - 1] + 1
        hours = range(slot_start, slot_end + 1)
        slot_cost = sum(float(price[h]) * grid_values[h] for h in hours)
        slot_costs.append(float(slot_cost))

    total_cost = float(np.sum(float(price[t]) * grid_values[t] for t in range(T)))
    return {
        "soc_targets": np.clip(soc_targets, min_reserve_pct, 100.0),
        "slot_costs": slot_costs,
        "total_cost": total_cost,
    }

# ---------------------------
# Training pipeline
# ---------------------------
def prepare_dataset_from_json(historical: List[Dict[str, Any]], uid: str):
    df = pd.DataFrame(historical).ffill().bfill().reset_index(drop=True)
    for key, default in HISTORICAL_OPTIONAL_DEFAULTS.items():
        if key not in df:
            df[key] = default
    df.fillna(value=HISTORICAL_OPTIONAL_DEFAULTS, inplace=True)
    scaler_info = {}
    for col in ['generation', 'consumption', 'price', 'power']:
        arr = df[col].to_numpy().astype(np.float64)
        scaler_info[col] = {'mean': float(np.mean(arr)), 'std': float(np.std(arr))}
    ds = TOUDataset(df, seq_len=DEFAULT_SEQ_LEN, pred_len=DEFAULT_PRED_LEN, uid=uid)
    ds.set_scalers(scaler_info)
    return ds, scaler_info

def full_train_pipeline(uid: str, historical: List[Dict[str, Any]], train_window_days: int = 14):
    historical = validate_records(
        list(historical),
        HISTORICAL_REQUIRED_FIELDS,
        "historical",
        optional_defaults=HISTORICAL_OPTIONAL_DEFAULTS,
    )
    if train_window_days and train_window_days > 0:
        horizon = int(train_window_days * 24)
        if len(historical) > horizon:
            historical = historical[-horizon:]
    user_dir = os.path.join(USER_MODELS_DIR, uid)
    ensure_directory(user_dir)
    ensure_directory(os.path.join(user_dir, "logs"))

    hist_path = os.path.join(user_dir, f"history_{uid}.json")
    forecaster_path = os.path.join(user_dir, f"forecaster_{uid}.pt")
    soc_path = os.path.join(user_dir, f"soc_controller_{uid}.pt")
    scaler_path = os.path.join(user_dir, f"scaler_{uid}.json")
    meta_path = os.path.join(user_dir, f"meta_{uid}.json")

    save_json(historical, hist_path)
    ds, scalers = prepare_dataset_from_json(historical, uid)
    save_json(scalers, scaler_path)

    # padding short data
    if len(historical) < DEFAULT_SEQ_LEN + DEFAULT_PRED_LEN:
        print(f"[WARN] Padding history for {uid}")
        last = historical[-1]
        while len(historical) < DEFAULT_SEQ_LEN + DEFAULT_PRED_LEN:
            new = dict(last)
            new["hour"] = (new["hour"] + 1) % 24
            historical.append(new)
        ds, scalers = prepare_dataset_from_json(historical, uid)
        save_json(scalers, scaler_path)

    if len(ds) == 0:
        raise ValueError(
            "Not enough historical records to create at least one training sample"
        )

    sample0 = ds[0]
    input_size = sample0['X'].shape[1]
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    forecaster_metrics = train_forecaster(
        ds,
        uid,
        input_size,
        log_dir=os.path.join(user_dir, "logs", f"forecaster_{timestamp}"),
        model_path=forecaster_path,
    )

    dataset_items = [ds[i] for i in range(len(ds))]
    Xs = np.stack([item['X'] for item in dataset_items])
    lp_results = []
    for item in dataset_items:
        meta = item['meta']
        power_limit = float(abs(meta['power'])) or 1.0
        lp_result = build_lp_soc_targets(
            np.asarray(meta['generation']),
            np.asarray(meta['consumption']),
            np.asarray(meta['price']),
            float(meta['soc_start']),
            float(meta['battery_capacity']),
            power_limit,
        )
        lp_results.append(lp_result)
    Ys = np.stack([res['soc_targets'] for res in lp_results]).astype(np.float32)
    soc_metrics = train_soc_controller(
        Xs,
        Ys,
        uid,
        input_size,
        log_dir=os.path.join(user_dir, "logs", f"soc_{timestamp}"),
        model_path=soc_path,
    )

    aggregate_slot_costs = (
        np.mean([res['slot_costs'] for res in lp_results], axis=0).tolist()
        if lp_results and lp_results[0]['slot_costs']
        else []
    )

    last_trained = time.strftime("%Y-%m-%d %H:%M:%S")
    meta = {
        "uid": uid,
        "last_trained": last_trained,
        "forecaster_path": forecaster_path,
        "soc_path": soc_path,
        "scaler_path": scaler_path,
        "train_records": len(historical),
        "train_window_days": train_window_days,
        "forecaster_metrics": forecaster_metrics,
        "soc_metrics": soc_metrics,
        "forecaster_backend": forecaster_metrics.get("backend"),
        "soc_backend": soc_metrics.get("backend"),
        "avg_slot_costs": aggregate_slot_costs,
    }
    save_json(meta, meta_path)
    response_metrics = {
        "forecaster_train_loss": forecaster_metrics.get("train_loss"),
        "forecaster_val_loss": forecaster_metrics.get("val_loss"),
        "soc_train_loss": soc_metrics.get("train_loss"),
        "soc_val_loss": soc_metrics.get("val_loss"),
        "avg_slot_costs": aggregate_slot_costs,
    }
    return {
        "status": "trained",
        "uid": uid,
        "last_trained": last_trained,
        "train_records": len(historical),
        "train_window_days": train_window_days,
        "metrics": response_metrics,
    }

# ---------------------------
# Train helpers
# ---------------------------
def _train_forecaster_numpy(dataset: TOUDataset, model_path: str) -> Dict[str, float]:
    start_time = time.time()
    items = [dataset[i] for i in range(len(dataset))]
    if not items:
        raise ValueError("Dataset is empty; cannot train forecaster")
    y_g = np.stack([item['y_g'] for item in items])
    y_c = np.stack([item['y_c'] for item in items])
    mean_g = float(np.mean(y_g))
    mean_c = float(np.mean(y_c))
    pred_len = int(getattr(dataset, 'pred_len', y_g.shape[1] if y_g.ndim > 1 else y_g.shape[0]))
    preds_g = np.full_like(y_g, mean_g)
    preds_c = np.full_like(y_c, mean_c)
    mse = float(np.mean((y_g - preds_g) ** 2 + (y_c - preds_c) ** 2))
    artifact = SimpleForecasterArtifact(mean_g, mean_c, pred_len).to_dict()
    safe_model_save(artifact, model_path)
    duration = time.time() - start_time
    return {
        "train_loss": mse,
        "val_loss": mse,
        "training_time": duration,
        "backend": "numpy",
    }


def train_forecaster(
    dataset: TOUDataset,
    uid: str,
    input_size: int,
    log_dir: str,
    model_path: str,
) -> Dict[str, float]:
    if not TORCH_AVAILABLE:
        return _train_forecaster_numpy(dataset, model_path)
    model = Forecaster(input_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_FORECASTER)
    criterion = nn.MSELoss()
    train_loader, val_loader = split_dataloaders(dataset, BATCH_SIZE)
    best_state = None
    best_val = float("inf")
    last_train_loss = float("inf")
    global_step = 0
    start_time = time.time()
    ensure_directory(log_dir)
    with SummaryWriter(log_dir=log_dir) as writer:
        for epoch in range(EPOCHS_FORECASTER):
            model.train()
            running_loss = 0.0
            batch_count = 0
            for X, y_g, y_c in train_loader:
                X = X.to(DEVICE)
                y_g = y_g.to(DEVICE)
                y_c = y_c.to(DEVICE)
                optimizer.zero_grad()
                g_pred, c_pred = model(X)
                loss = criterion(g_pred, y_g) + criterion(c_pred, y_c)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                batch_count += 1
                writer.add_scalar("forecaster/train_step_loss", loss.item(), global_step)
                global_step += 1
            if batch_count:
                last_train_loss = running_loss / batch_count
            writer.add_scalar("forecaster/epoch_train_loss", last_train_loss, epoch)

            if val_loader is not None:
                model.eval()
                val_loss = 0.0
                val_batches = 0
                with torch.no_grad():
                    for X_val, y_g_val, y_c_val in val_loader:
                        X_val = X_val.to(DEVICE)
                        y_g_val = y_g_val.to(DEVICE)
                        y_c_val = y_c_val.to(DEVICE)
                        g_val, c_val = model(X_val)
                        loss_val = criterion(g_val, y_g_val) + criterion(c_val, y_c_val)
                        val_loss += loss_val.item()
                        val_batches += 1
                if val_batches:
                    val_loss /= val_batches
                writer.add_scalar("forecaster/val_loss", val_loss, epoch)
            else:
                val_loss = last_train_loss

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            print(
                f"[Forecaster][{uid}] Epoch {epoch + 1}/{EPOCHS_FORECASTER} "
                f"train={last_train_loss:.5f} val={val_loss:.5f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    safe_model_save(model, model_path)
    training_time = time.time() - start_time
    return {
        "train_loss": float(last_train_loss),
        "val_loss": float(best_val if best_state is not None else last_train_loss),
        "training_time": float(training_time),
        "backend": "torch",
    }


def _train_soc_controller_numpy(Ys: np.ndarray, model_path: str) -> Dict[str, float]:
    start_time = time.time()
    if Ys.size == 0:
        raise ValueError("SOC target labels are empty; cannot train controller")
    mean_targets = np.mean(Ys, axis=0)
    preds = np.broadcast_to(mean_targets, Ys.shape)
    mse = float(np.mean((Ys - preds) ** 2))
    artifact = SimpleSOCControllerArtifact(mean_targets.tolist()).to_dict()
    safe_model_save(artifact, model_path)
    duration = time.time() - start_time
    return {
        "train_loss": mse,
        "val_loss": mse,
        "training_time": duration,
        "backend": "numpy",
    }


def train_soc_controller(
    Xs: np.ndarray,
    Ys: np.ndarray,
    uid: str,
    input_size: int,
    log_dir: str,
    model_path: str,
) -> Dict[str, float]:
    if not TORCH_AVAILABLE:
        return _train_soc_controller_numpy(Ys, model_path)
    model = SOCController(input_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_SOC)
    criterion = nn.MSELoss()
    dataset = TensorDataset(
        torch.from_numpy(Xs).float(), torch.from_numpy(Ys).float()
    )
    train_loader, val_loader = split_dataloaders(dataset, BATCH_SIZE, collate_fn=default_collate)
    best_state = None
    best_val = float("inf")
    global_step = 0
    last_train_loss = float("inf")
    start_time = time.time()
    ensure_directory(log_dir)
    with SummaryWriter(log_dir=log_dir) as writer:
        for epoch in range(EPOCHS_SOC):
            model.train()
            running_loss = 0.0
            batches = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                optimizer.zero_grad()
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                batches += 1
                writer.add_scalar("soc/train_step_loss", loss.item(), global_step)
                global_step += 1
            if batches:
                last_train_loss = running_loss / batches
            writer.add_scalar("soc/epoch_train_loss", last_train_loss, epoch)

            if val_loader is not None:
                model.eval()
                val_loss = 0.0
                val_batches = 0
                with torch.no_grad():
                    for X_val, y_val in val_loader:
                        X_val = X_val.to(DEVICE)
                        y_val = y_val.to(DEVICE)
                        preds = model(X_val)
                        loss_val = criterion(preds, y_val)
                        val_loss += loss_val.item()
                        val_batches += 1
                if val_batches:
                    val_loss /= val_batches
                writer.add_scalar("soc/val_loss", val_loss, epoch)
            else:
                val_loss = last_train_loss

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            print(
                f"[SOC][{uid}] Epoch {epoch + 1}/{EPOCHS_SOC} "
                f"train={last_train_loss:.5f} val={val_loss:.5f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    safe_model_save(model, model_path)
    training_time = time.time() - start_time
    return {
        "train_loss": float(last_train_loss),
        "val_loss": float(best_val if best_state is not None else last_train_loss),
        "training_time": float(training_time),
        "backend": "torch",
    }

# ---------------------------
# Prediction
# ---------------------------
def _load_json_artifact(path: str) -> Dict[str, Any]:
    try:
        return load_json(path)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Unable to parse model artifact at {path}. Install PyTorch and retrain the model."
        ) from exc


def load_models_for_uid(uid, input_size, scalers: Optional[Dict[str, Any]] = None):
    user_dir = os.path.join(USER_MODELS_DIR, uid)
    forecaster_state = os.path.join(user_dir, f"forecaster_{uid}.pt")
    soc_state_path = os.path.join(user_dir, f"soc_controller_{uid}.pt")
    scaler_path = os.path.join(user_dir, f"scaler_{uid}.json")
    if scalers is None:
        scalers = load_json(scaler_path)

    if TORCH_AVAILABLE:
        try:
            fore_state = torch.load(forecaster_state, map_location=DEVICE)
            fore_model = Forecaster(input_size)
            fore_model.load_state_dict(fore_state)
            fore_wrapper = TorchForecasterWrapper(fore_model)
        except FileNotFoundError:
            raise
        except Exception:
            artifact = _load_json_artifact(forecaster_state)
            fore_wrapper = SimpleForecasterArtifact.from_dict(artifact)
    else:
        artifact = _load_json_artifact(forecaster_state)
        fore_wrapper = SimpleForecasterArtifact.from_dict(artifact)

    if TORCH_AVAILABLE:
        try:
            soc_state = torch.load(soc_state_path, map_location=DEVICE)
            soc_model = SOCController(input_size)
            soc_model.load_state_dict(soc_state)
            soc_wrapper = TorchSOCControllerWrapper(soc_model)
        except FileNotFoundError:
            raise
        except Exception:
            artifact = _load_json_artifact(soc_state_path)
            soc_wrapper = SimpleSOCControllerArtifact.from_dict(artifact)
    else:
        artifact = _load_json_artifact(soc_state_path)
        soc_wrapper = SimpleSOCControllerArtifact.from_dict(artifact)

    return fore_wrapper, soc_wrapper, scalers

def resolve_prediction_payload(data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], float, float, float]:
    forecast = data.get("forecast_24h")
    if forecast is None:
        forecast = data.get("weather_forecast")
    if forecast is None:
        raise ValueError("forecast_24h or weather_forecast is required")
    if not isinstance(forecast, list) or not forecast:
        raise ValueError("forecast_24h must be a non-empty list of records")

    current_payload = data.get("current")
    current_record: Optional[Dict[str, Any]] = None
    if current_payload is not None:
        current_record = validate_records(
            [current_payload],
            HISTORICAL_REQUIRED_FIELDS,
            "current",
            optional_defaults=HISTORICAL_OPTIONAL_DEFAULTS,
        )[0]

    def _coalesce_numeric(key: str, fallback_key: str, default: float) -> float:
        value = data.get(key)
        if value is None and current_record is not None:
            value = current_record.get(fallback_key)
        if value is None:
            value = default
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be numeric")

    soc_current = _coalesce_numeric("soc_current", "soc", 0.0)
    power = _coalesce_numeric("power", "power", 0.0)
    capacity = _coalesce_numeric("battery_capacity", "battery_capacity", 1.0)

    return forecast, soc_current, power, capacity


def predict_for_uid(uid, forecast_24h, soc_current, power, capacity):
    forecast_records = validate_records(
        list(forecast_24h),
        FORECAST_REQUIRED_FIELDS,
        "forecast_24h",
        optional_defaults=FORECAST_OPTIONAL_DEFAULTS,
    )
    df = pd.DataFrame(forecast_records)
    if len(df) < DEFAULT_PRED_LEN:
        raise ValueError("forecast_24h must contain at least 24 records")
    if capacity <= 0:
        raise ValueError("battery_capacity must be greater than zero")
    df = df.sort_values(["month", "hour"]).reset_index(drop=True)
    for key, default in FORECAST_OPTIONAL_DEFAULTS.items():
        if key not in df:
            df[key] = default
    df.fillna(value=FORECAST_OPTIONAL_DEFAULTS, inplace=True)
    if 'power' not in df.columns:
        df['power'] = power
    else:
        df['power'] = df['power'].fillna(power)

    user_dir = os.path.join(USER_MODELS_DIR, uid)
    scaler_path = os.path.join(user_dir, f"scaler_{uid}.json")
    scalers = load_json(scaler_path)

    hours = df['hour'].to_numpy()
    months = df['month'].to_numpy()
    hour_cc = hour_sin_cos(hours)
    month_cc = month_sin_cos(months)

    def apply_scaler(arr: np.ndarray, name: str) -> np.ndarray:
        info = scalers.get(name)
        if not info:
            return arr
        std = info.get('std', 1.0) or 1.0
        return (arr - info.get('mean', 0.0)) / std

    features = df[['clouds', 'temperature', 'irradiance', 'power', 'generation', 'consumption', 'price']].to_numpy(dtype=np.float32)
    clouds = features[:, 0:1]
    temperature = features[:, 1:2]
    irradiance = features[:, 2:3]
    power_arr = apply_scaler(features[:, 3], 'power').reshape(-1, 1)
    gen_arr = apply_scaler(features[:, 4], 'generation').reshape(-1, 1)
    cons_arr = apply_scaler(features[:, 5], 'consumption').reshape(-1, 1)
    price_arr = apply_scaler(features[:, 6], 'price').reshape(-1, 1)

    X_seq = np.concatenate(
        [
            hour_cc,
            month_cc,
            clouds,
            temperature,
            irradiance,
            power_arr,
            gen_arr,
            cons_arr,
            price_arr,
        ],
        axis=-1,
    ).astype(np.float32)

    input_size = X_seq.shape[1]
    fore_model, soc_model, _ = load_models_for_uid(uid, input_size, scalers=scalers)
    g, c = fore_model.predict(X_seq)
    soc_pred = soc_model.predict(X_seq)
    g = np.asarray(g, dtype=np.float32).ravel()
    c = np.asarray(c, dtype=np.float32).ravel()
    soc_pred = np.asarray(soc_pred, dtype=np.float32).ravel()

    soc_current_clipped = float(np.clip(soc_current, 0.0, 100.0))
    power_limit = max(abs(power), 1.0)
    lp_result = build_lp_soc_targets(
        g,
        c,
        df['price'].to_numpy(),
        soc_current_clipped,
        capacity,
        power_limit,
    )
    lp_targets = np.asarray(lp_result['soc_targets'], dtype=np.float32)
    if lp_targets.size:
        soc_targets = lp_targets
    else:
        soc_targets = soc_pred
    soc_targets = np.clip(soc_targets, 0.0, 100.0)

    slots = []
    for idx, target in enumerate(soc_targets):
        start_hour = idx * 4
        end_hour = (idx + 1) * 4
        slot_info = {
            "start_hour": start_hour,
            "end_hour": end_hour,
            "soc_target_pct": float(round(target, 3)),
        }
        if idx < len(lp_result['slot_costs']):
            slot_info["estimated_cost"] = float(round(lp_result['slot_costs'][idx], 4))
        slots.append(slot_info)

    return {
        "generation_pred": g.tolist(),
        "consumption_pred": c.tolist(),
        "slots": slots,
        "strategy_cost": {
            "slot_costs": lp_result['slot_costs'],
            "total_cost": lp_result['total_cost'],
        },
        "model_soc_targets": soc_pred.tolist(),
    }

# ---------------------------
# Flask
# ---------------------------
app = Flask(__name__)

@app.route("/train", methods=["POST"])
def train_endpoint():
    data = request.get_json() or {}
    if not isinstance(data, dict):
        return jsonify({"error": "invalid JSON payload"}), 400
    uid = data.get("uid"); hist = data.get("historical")
    if not uid or not hist:
        return jsonify({"error": "uid and historical required"}), 400
    try:
        result = full_train_pipeline(uid, hist)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    data = request.get_json() or {}
    if not isinstance(data, dict):
        return jsonify({"error": "invalid JSON payload"}), 400
    uid = data.get("uid")
    if not uid:
        return jsonify({"error": "uid is required"}), 400
    try:
        forecast, soc, power, cap = resolve_prediction_payload(data)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    try:
        return jsonify(predict_for_uid(uid, forecast, soc, power, cap))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting TOU service on http://127.0.0.1:5001")
    app.run(host="0.0.0.0", port=5001, debug=True)
