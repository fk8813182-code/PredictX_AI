"""Prediction engine: validated input -> class, probability, risk level (uses the saved training pipeline)."""
import json, os, joblib, numpy as np, pandas as pd
from data_loader import add_features, FEATURES, RAW_NUM
M = os.path.join(os.path.dirname(__file__), "..", "models")

class ModelNotAvailable(Exception): pass

def load_model():
    try:
        return joblib.load(os.path.join(M, "final_model.joblib")), json.load(open(os.path.join(M, "model_metadata.json")))
    except Exception as e:
        raise ModelNotAvailable("Model files not found. Run: python src/train.py") from e

def validate(inp, meta):
    row = {}
    for c in RAW_NUM:
        try: v = float(inp[c])
        except Exception: raise ValueError(f"Invalid or missing value for '{c}'")
        lo, hi, _ = meta["feature_ranges"][c]; span = hi - lo
        if not np.isfinite(v) or v < lo - 0.25 * span or v > hi + 0.25 * span:
            raise ValueError(f"'{c}' = {v} is outside the plausible range [{lo:.1f}, {hi:.1f}]")
        row[c] = v
    if inp.get("Type") not in meta["types"]: raise ValueError(f"Type must be one of {meta['types']}")
    row["Type"] = inp["Type"]
    return row

def risk_level(p, th):
    return "HIGH" if p >= th["high"] else "MEDIUM" if p >= th["medium"] else "LOW"

RECS = {"HIGH": "Prioritize inspection before the next production cycle.",
        "MEDIUM": "Continue monitoring and schedule preventive inspection.",
        "LOW": "No immediate intervention indicated. Continue normal monitoring."}

def predict(inp, model=None, meta=None):
    if model is None: model, meta = load_model()
    row = validate(inp, meta)
    X = add_features(pd.DataFrame([row]))[FEATURES]
    p = float(model.predict_proba(X)[0, 1]); lvl = risk_level(p, meta["risk_thresholds"])
    return {"probability": p, "risk_level": lvl, "predicted_class": int(p >= 0.5),
            "label": "FAILURE RISK DETECTED" if p >= 0.5 else "NORMAL OPERATION", "recommendation": RECS[lvl]}
