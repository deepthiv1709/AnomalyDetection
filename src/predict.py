# src/predict.py

import pandas as pd
import joblib
import os
from src.config import Config

# --------------------------
# Paths
# --------------------------
# MODEL_DIR = "models"

ISO_MODEL_PATH = os.path.join(Config.MODEL_DIR, "iso_model.pkl")
XGB_MODEL_PATH = os.path.join(Config.MODEL_DIR, "xgb_model.pkl")
SCALER_PATH = os.path.join(Config.MODEL_DIR, "scaler.pkl")
TRANSFORMER_PATH = os.path.join(Config.MODEL_DIR, "transformer.pkl")
FEATURES_PATH = os.path.join(Config.MODEL_DIR, "features.pkl")

# --------------------------
# Load Artifacts (once)
# --------------------------
iso_model = joblib.load(ISO_MODEL_PATH)
xgb_model = joblib.load(XGB_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
transformer = joblib.load(TRANSFORMER_PATH)
feature_names = joblib.load(FEATURES_PATH)


# --------------------------
# Core Prediction Function
# --------------------------
def predict_transaction(input_data: dict):
    """
    Predict fraud risk for a single transaction.
    
    Args:
        input_data (dict): transaction data
        
    Returns:
        dict: prediction, risk_score, decision
    """

    try:
        # Convert input to DataFrame
        df = pd.DataFrame([input_data])

        # Ensure correct column order
        df = df[feature_names]

        # --------------------------
        # Apply transformations
        # --------------------------
        X_transformed = transformer.transform(df)
        X_scaled = scaler.transform(X_transformed)


        # --------------------------
        # Isolation Forest - Model prediction
        # --------------------------
        iso_pred = iso_model.predict(X_scaled)[0]   # -1 or 1
        iso_score = iso_model.decision_function(X_scaled)[0]

        iso_fraud = 1 if iso_pred == -1 else 0

        # --------------------------
        # XGBoost - Model prediction
        # --------------------------
        xgb_prob = xgb_model.predict_proba(X_scaled)[0][1]
        xgb_pred = 1 if xgb_prob > 0.5 else 0

        # --------------------------
        # Ensemble Logic
        # --------------------------
        final_score = (abs(iso_score) + xgb_prob) / 2

        final_prediction = 1 if (iso_fraud == 1 or xgb_pred == 1) else 0

        decision = "Review" if final_prediction == 1 else "Approve"

        return {
            "prediction": final_prediction,
            "risk_score": float(final_score),
            "iso_score": float(iso_score),
            "xgb_probability": float(xgb_prob),
            "decision": decision
        }

    except Exception as e:
        return {
            "error": str(e)
        }


# --------------------------
# Batch Prediction
# --------------------------
def predict_batch(df: pd.DataFrame):
    
    df = df[feature_names]

    X_transformed = transformer.transform(df)
    X_scaled = scaler.transform(X_transformed)

    iso_preds = iso_model.predict(X_scaled)
    iso_scores = iso_model.decision_function(X_scaled)

    xgb_probs = xgb_model.predict_proba(X_scaled)[:, 1]

    df["iso_fraud"] = [1 if p == -1 else 0 for p in iso_preds]
    df["xgb_prob"] = xgb_probs

    df["final_prediction"] = (
        (df["iso_fraud"] == 1) | (df["xgb_prob"] > 0.5)
    ).astype(int)

    df["decision"] = df["final_prediction"].map({1: "Review", 0: "Approve"})

    return df
