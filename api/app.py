"""
api/app.py
───────────
FastAPI prediction service.

Design decisions:
  - Single /predict endpoint (can extend to /predict/batch for bulk scoring)
  - Model loaded ONCE at startup (not per-request) — critical for latency
  - Pydantic v2 schema for input validation (type coercion + clear errors)
  - Returns probability + binary prediction + risk tier for downstream systems
  - /health endpoint for k8s liveness/readiness probes
  - /model/info endpoint for monitoring dashboards

In production: wrap in Docker, deploy behind nginx, use gunicorn -k uvicorn.workers.UvicornWorker
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
import pandas as pd
import numpy as np
import logging
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Literal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.model_registry import ModelRegistry
from src.features.feature_engineer import add_engineered_features
from configs.config import DEFAULT_THRESHOLD, LOGS_DIR

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "api.log"),
    ],
)
logger = logging.getLogger(__name__)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production ML service for churn probability scoring",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Global model state (loaded once at startup) ───────────────────────────────
MODEL_STATE: dict = {}


@app.on_event("startup")
async def load_model():
    """
    Load model at startup. If loading fails, app refuses to start.
    This prevents silent failures where requests hit an unloaded model.
    """
    try:
        registry = ModelRegistry()
        pipeline, winsorizer, metadata = registry.load_champion()
        MODEL_STATE["pipeline"] = pipeline
        MODEL_STATE["winsorizer"] = winsorizer
        MODEL_STATE["metadata"] = metadata
        MODEL_STATE["threshold"] = metadata.get("threshold", DEFAULT_THRESHOLD)
        MODEL_STATE["loaded_at"] = datetime.utcnow().isoformat()
        logger.info(
            f"Model loaded: {metadata['version']} | "
            f"ROC-AUC: {metadata['metrics'].get('roc_auc', 'N/A'):.4f}"
        )
    except FileNotFoundError:
        logger.error("No champion model found. Run training pipeline first.")
        raise


# ── Request/Response schemas ──────────────────────────────────────────────────

class CustomerFeatures(BaseModel):
    """
    Input schema for a single customer prediction request.
    All fields mirror the raw data schema — the API handles feature engineering internally.
    """
    customer_id: str = Field(..., description="Unique customer identifier")
    gender: Literal["Male", "Female"]
    is_senior_citizen: int = Field(..., ge=0, le=1)
    tenure_months: int = Field(..., ge=0, le=120, description="Months as customer")
    has_phone_service: Literal["Yes", "No"]
    internet_service: Literal["DSL", "Fiber Optic", "No"]
    has_online_backup: Literal["Yes", "No"]
    has_tech_support: Literal["Yes", "No"]
    contract_type: Literal["Month-to-Month", "One Year", "Two Year"]
    payment_method: Literal["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"]
    monthly_charges: float = Field(..., ge=0, le=500)
    total_charges: Optional[float] = Field(None, ge=0, description="Can be null for new customers")
    num_products: int = Field(..., ge=1, le=20)
    num_support_calls: int = Field(..., ge=0, le=50)
    days_since_last_login: int = Field(..., ge=0, le=365)
    avg_session_duration_mins: Optional[float] = Field(None, ge=0, le=1440)

    model_config = {"json_schema_extra": {
        "example": {
            "customer_id": "CUST_000042",
            "gender": "Female",
            "is_senior_citizen": 0,
            "tenure_months": 3,
            "has_phone_service": "Yes",
            "internet_service": "Fiber Optic",
            "has_online_backup": "No",
            "has_tech_support": "No",
            "contract_type": "Month-to-Month",
            "payment_method": "Electronic Check",
            "monthly_charges": 89.50,
            "total_charges": None,
            "num_products": 2,
            "num_support_calls": 4,
            "days_since_last_login": 28,
            "avg_session_duration_mins": 12.5,
        }
    }}


class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float = Field(..., description="Probability of churning [0, 1]")
    churn_prediction: bool = Field(..., description="Binary prediction at configured threshold")
    risk_tier: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        ..., description="LOW (<30%), MEDIUM (30-60%), HIGH (>60%)"
    )
    threshold_used: float
    model_version: str
    prediction_timestamp: str


class BatchPredictionRequest(BaseModel):
    customers: list[CustomerFeatures]


# ── Helper functions ──────────────────────────────────────────────────────────

def _get_risk_tier(prob: float) -> str:
    if prob < 0.30:
        return "LOW"
    elif prob < 0.60:
        return "MEDIUM"
    return "HIGH"


def _customer_to_dataframe(customer: CustomerFeatures) -> pd.DataFrame:
    """Convert Pydantic model to DataFrame for the preprocessing pipeline."""
    return pd.DataFrame([customer.model_dump()])


def _log_prediction(customer_id: str, prob: float, latency_ms: float):
    """
    Append prediction to prediction log for monitoring.
    In production: publish to Kafka/Kinesis for real-time monitoring.
    """
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "customer_id": customer_id,
        "churn_probability": prob,
        "latency_ms": latency_ms,
    }
    pred_log_path = LOGS_DIR / "predictions.jsonl"
    with open(pred_log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Kubernetes liveness probe. Returns 200 if model is loaded."""
    if "pipeline" not in MODEL_STATE:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_version": MODEL_STATE["metadata"].get("version"),
        "loaded_at": MODEL_STATE.get("loaded_at"),
    }


@app.get("/model/info")
async def model_info():
    """Returns model metadata for monitoring dashboards."""
    if "metadata" not in MODEL_STATE:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_info": MODEL_STATE["metadata"],
        "threshold": MODEL_STATE["threshold"],
        "loaded_at": MODEL_STATE["loaded_at"],
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerFeatures):
    """
    Single-customer churn prediction.

    Pipeline:
        1. Validate input (Pydantic)
        2. Convert to DataFrame
        3. Feature engineering
        4. Winsorization (outlier capping)
        5. Preprocessing (impute + scale + encode)
        6. Classifier → probability
        7. Threshold → binary prediction
        8. Log prediction for monitoring
    """
    if "pipeline" not in MODEL_STATE:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        df = _customer_to_dataframe(customer)
        df = add_engineered_features(df)

        # Apply winsorization (fitted on training data)
        winsorizer = MODEL_STATE["winsorizer"]
        df = winsorizer.transform(df)

        pipeline = MODEL_STATE["pipeline"]
        churn_prob = float(pipeline.predict_proba(df)[0, 1])
        threshold = MODEL_STATE["threshold"]

        latency_ms = (time.time() - start_time) * 1000
        _log_prediction(customer.customer_id, churn_prob, latency_ms)

        response = PredictionResponse(
            customer_id=customer.customer_id,
            churn_probability=round(churn_prob, 4),
            churn_prediction=churn_prob >= threshold,
            risk_tier=_get_risk_tier(churn_prob),
            threshold_used=threshold,
            model_version=MODEL_STATE["metadata"].get("version", "unknown"),
            prediction_timestamp=datetime.utcnow().isoformat(),
        )

        logger.info(
            f"Predicted {customer.customer_id}: prob={churn_prob:.4f} "
            f"tier={response.risk_tier} latency={latency_ms:.1f}ms"
        )
        return response

    except Exception as e:
        logger.error(f"Prediction failed for {customer.customer_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """
    Batch prediction endpoint.
    Processes all customers in a single model call (more efficient than N single calls).
    """
    if "pipeline" not in MODEL_STATE:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(request.customers) > 1000:
        raise HTTPException(status_code=400, detail="Batch size limit: 1000 customers")

    start_time = time.time()

    try:
        df = pd.DataFrame([c.model_dump() for c in request.customers])
        df = add_engineered_features(df)
        df = MODEL_STATE["winsorizer"].transform(df)

        probs = MODEL_STATE["pipeline"].predict_proba(df)[:, 1]
        threshold = MODEL_STATE["threshold"]

        results = []
        for customer, prob in zip(request.customers, probs):
            results.append({
                "customer_id": customer.customer_id,
                "churn_probability": round(float(prob), 4),
                "churn_prediction": float(prob) >= threshold,
                "risk_tier": _get_risk_tier(float(prob)),
            })

        latency_ms = (time.time() - start_time) * 1000
        logger.info(f"Batch prediction: {len(results)} customers in {latency_ms:.1f}ms")

        return {
            "predictions": results,
            "batch_size": len(results),
            "latency_ms": round(latency_ms, 1),
            "model_version": MODEL_STATE["metadata"].get("version"),
        }

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Run locally ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False, workers=1)
