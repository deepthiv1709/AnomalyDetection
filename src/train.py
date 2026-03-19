from src.data_loader import load_data, validate_data
from src.preprocessing import preprocess
from src.feature_engineering import create_features
from src.models.isolation_forest import train_isolation_forest
from src.models.xgboost_model import train_xgboost
from src.config import Config
import joblib
import os
import pandas as pd

def train():
    # --------------------------
    # Load Data
    # --------------------------
    df = load_data(Config.DATA_PATH)
    validate_data(df)

    # --------------------------
    # feature engineering
    # --------------------------
    # df = create_features(df)
    data = preprocess(df)
    X = data["X_train"]
    y = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    scaler = data["scaler"]
    pt = data["transformer"]
    feature_names = data["feature_names"]
    print(feature_names)
    print(df.head(5))

    # --------------------------
    # Train models
    # --------------------------
    iso_model = train_isolation_forest(X, Config.ISOLATION_FOREST_PARAMS)
    xgb_model = train_xgboost(X, y)

    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    joblib.dump(iso_model, Config.MODEL_DIR + "iso_model.pkl")
    joblib.dump(xgb_model, Config.MODEL_DIR + "xgb_model.pkl")
    joblib.dump(scaler, Config.MODEL_DIR + "scaler.pkl")
    joblib.dump(pt, Config.MODEL_DIR + "transformer.pkl")
    joblib.dump(feature_names, Config.MODEL_DIR + "features.pkl")

    print("Training complete")
    return X_val, y_val, feature_names