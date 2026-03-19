from src.evaluate import evaluate_ensemble, evaluate_model
from src.explainability.shap_explainer import shap_summary, shap_force_plot
from src.train import train
from src.config import Config
import joblib
import os

X_val, y_val, feature_names = train()
ISO_MODEL_PATH = os.path.join(Config.MODEL_DIR, "iso_model.pkl")
XGB_MODEL_PATH = os.path.join(Config.MODEL_DIR, "xgb_model.pkl")
iso_model = joblib.load(ISO_MODEL_PATH)
xgb_model = joblib.load(XGB_MODEL_PATH)

# Evaluate models
metrics_iso = evaluate_model(iso_model, X_val, y_val, model_name="Isolation Forest")
metrics_xgb = evaluate_model(xgb_model, X_val, y_val, model_name="XGBoost")
metrics_ensemble = evaluate_ensemble(iso_model, xgb_model, X_val, y_val)

# SHAP explanation (XGBoost only for simplicity)
shap_df = shap_summary(xgb_model, X_val, feature_names=feature_names, model_name="XGBoost")
shap_force_plot(xgb_model, X_val, instance_index=0, feature_names=feature_names)