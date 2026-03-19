from fastapi import FastAPI
from src.predict import predict_transaction

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}

@app.post("/predict")
def predict(data: dict):
    result = predict_transaction(data)
    return result