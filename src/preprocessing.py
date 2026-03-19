# src/preprocessing.py

import pandas as pd
import numpy as np

from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.model_selection import train_test_split
from src.config import Config


def preprocess(df: pd.DataFrame, test_size=0.2, random_state=42):
    """
    Full preprocessing pipeline:
    - Split features and target
    - Apply Yeo-Johnson transformation
    - Standard scaling
    - Train-test split
    """

    if "Class" not in df.columns:
        raise ValueError("Target column 'Class' not found")
    # --------------------------
    # Separate features and target
    # --------------------------
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Keep feature names
    feature_names = X.columns.tolist()
    print(feature_names)


    # --------------------------
    # Train-Test Split
    # --------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df["Class"] = y_test
    # Save to CSV for later use in FastAPI
    test_df.to_csv(Config.DATA_DIR + "test_data.csv", index=False)
     # --------------------------
    # Transformation (handles skew)
    # --------------------------
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    # X_numeric = X.drop(columns=["Amount_bin"])
    X_transformed = pt.fit_transform(X_train)

    # --------------------------
    # Scaling
    # --------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_transformed)
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_train, test_size=test_size,
        random_state=random_state,
        stratify=y_train)
    
    print("Preprocessing complete")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}, Val shape: {X_val.shape}")

    return {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "scaler": scaler,
        "transformer": pt,
        "feature_names": feature_names
    }


def transform_new_data(df: pd.DataFrame, scaler, transformer, feature_names):
    """
    Apply same preprocessing to new/unseen data (for inference).
    """

    df = df.copy()

    # Ensure correct column order
    df = df[feature_names]

    # Apply transformations
    X_numeric = df.drop(columns=["Amount_bin"])
    X_transformed = transformer.transform(X_numeric)
    
    X_scaled = scaler.transform(X_transformed)

    return X_scaled