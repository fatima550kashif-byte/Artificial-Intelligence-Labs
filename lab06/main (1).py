import pickle
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Diabetes Predictor", version="1.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# Load the artefact ONCE at start-up, not on every request
MODEL_PATH = Path(__file__).parent / "diabetes_logistic_model.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None

FEATURES = list(model.feature_names_in_) if model is not None else []


class PredictRequest(BaseModel):
    Glucose: float = Field(..., ge=0, le=300, description="Plasma glucose (mg/dL)")
    BloodPressure: float = Field(..., ge=0, le=200, description="Diastolic BP (mm Hg)")
    Insulin: float = Field(..., ge=0, le=900, description="Serum insulin (mu U/ml)")
    BMI: float = Field(..., ge=0, le=80, description="Body mass index")
    DiabetesPedigreeFunction: float = Field(..., ge=0, le=3.5)
    Age: float = Field(..., ge=1, le=120, description="Age in years")


class PredictResponse(BaseModel):
    predicted_label: int
    prediction: str
    probability: float


@app.get("/")
def home():
    return {"message": "Diabetes API is running. See /docs for documentation."}


@app.get("/health")
def health():
    return {"status": "ok" if model is not None else "model_not_loaded"}


@app.get("/metadata")
def metadata():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_type": type(model).__name__,
        "features": FEATURES,
        "n_features": len(FEATURES),
        "classes": {"0": "Not Diabetic", "1": "Diabetic"},
        "coefficients": dict(zip(FEATURES, map(float, model.coef_[0]))),
        "intercept": float(model.intercept_[0]),
        "sklearn_version_trained": "1.7.2",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        df = pd.DataFrame([req.model_dump()])[FEATURES]
        pred = int(model.predict(df)[0])
        prob = float(model.predict_proba(df)[0][1])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
    return PredictResponse(predicted_label=pred,
                           prediction="Diabetic" if pred == 1 else "Not Diabetic",
                           probability=round(prob, 3))
