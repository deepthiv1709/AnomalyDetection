# src/evaluate.py

import pandas as pd
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a single model's performance.
    
    Returns:
        dict of metrics
    """
    # Predict
    if hasattr(model, "predict_proba"):
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_prob > 0.5).astype(int)
    else:
        y_pred = model.predict(X_test)
        # For Isolation Forest, -1 = anomaly
        y_pred = np.where(y_pred == -1, 1, 0)

    # Metrics
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    print(f"\n{model_name} Classification Report:\n")
    print(classification_report(y_test, y_pred))

    return {
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm
    }


def evaluate_ensemble(iso_model, xgb_model, X_test, y_test):
    """
    Evaluate ensemble prediction of Isolation Forest + XGBoost
    """
    # Isolation Forest
    iso_pred = iso_model.predict(X_test)
    iso_pred = np.where(iso_pred == -1, 1, 0)
    iso_score = np.abs(iso_model.decision_function(X_test))

    # XGBoost
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    xgb_pred = (xgb_prob > 0.5).astype(int)

    # Ensemble: predict fraud if either model flags
    ensemble_pred = ((iso_pred == 1) | (xgb_pred == 1)).astype(int)
    ensemble_score = (iso_score + xgb_prob) / 2

    # Metrics
    f1 = f1_score(y_test, ensemble_pred)
    roc_auc = roc_auc_score(y_test, ensemble_pred)
    cm = confusion_matrix(y_test, ensemble_pred)

    # Plot
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
    plt.title("Ensemble Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    print("\nEnsemble Classification Report:\n")
    from sklearn.metrics import classification_report
    print(classification_report(y_test, ensemble_pred))

    return {
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm
    }