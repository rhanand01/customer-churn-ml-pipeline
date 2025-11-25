from fastapi import FastAPI

from src.schemas import CustomerFeatures
from src.inference import ChurnModel

# Create FastAPI app instance
app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
    description="API that predicts telecom customer churn probability."
)

# Load model once at startup
model = ChurnModel()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_churn(features: CustomerFeatures):
    data = features.dict()
    proba = model.predict_proba(data)
    label = model.predict_label(data)

    return {
        "churn_probability": proba,
        "churn_prediction": label  # 1 = churn, 0 = not churn
    }
