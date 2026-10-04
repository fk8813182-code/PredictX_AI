"""Data loading + feature engineering. Target/failure-mode columns are NEVER model inputs."""
import os, numpy as np, pandas as pd
DATA_PATH = os.environ.get("PREDICTX_DATA", os.path.join(os.path.dirname(__file__), "..", "data", "ai4i2020.csv"))
TARGET = "Machine failure"
LEAK_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]          # failure-mode labels (leakage)
ID_COLS = ["UDI", "Product ID"]                           # identifiers (no signal)
RAW_NUM = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
NUM = RAW_NUM + ["Temp diff [K]", "Power [W]"]
CAT = ["Type"]
FEATURES = CAT + NUM

def load_data(path=None):
    path = path or DATA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found. Place the UCI AI4I 2020 CSV at: {os.path.abspath(path)}")
    df = pd.read_csv(path)
    missing = [c for c in [TARGET, "Type"] + RAW_NUM if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    return df

def add_features(df):
    d = df.copy()
    d["Temp diff [K]"] = d["Process temperature [K]"] - d["Air temperature [K]"]
    d["Power [W]"] = d["Torque [Nm]"] * d["Rotational speed [rpm]"] * 2 * np.pi / 60
    return d
