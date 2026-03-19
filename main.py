from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer
from pydantic import create_model
app = FastAPI(title="Anomaly Detection API")


model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
pt = joblib.load("models/pt.pkl")
FEATURE_ORDER = joblib.load("models/feature_order.pkl")


@app.post("/predict")
def predict(data: list[dict]):
    try:
        df = pd.DataFrame(data)
        df = df[FEATURE_ORDER]
        
        # Apply the same preprocessing - Yeo-Johnson and StandardScaler
        # scaler = StandardScaler() - obtained from models
        # pt = PowerTransformer(method='yeo-johnson', standardize=True) - obtained from models  
        df_transformed = pt.transform(df)
        df_transformed = pd.DataFrame(df_transformed, columns=FEATURE_ORDER)    
        df_scaled = scaler.transform(df_transformed)
        pred = model.predict(df_scaled)  # -1 anomalous, 1 normal
        # Convert to 1=anomaly, 0=normal
        # Convert: -1 → 1 (anomaly), 1 → 0 (normal)
        results = [1 if p == -1 else 0 for p in pred]

        return {"anomalies": results}
    except KeyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Missing feature: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
