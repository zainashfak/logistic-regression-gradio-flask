import joblib
import pandas as pd

def load_bundle(key):
    return joblib.load(f"models/{key}.joblib")

def predict(bundle, values):
    """values: dict {feature: raw value}. Returns (label, {label: prob})."""
    df = pd.DataFrame([values], columns=bundle["features"])
    for c in bundle["numeric"]:
        df[c] = pd.to_numeric(df[c])
    proba = bundle["model"].predict_proba(df)[0]
    names = bundle["class_names"]
    probs = {names[i]: float(p) for i, p in enumerate(proba)}
    return max(probs, key=probs.get), probs
