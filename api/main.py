"""FastAPI inference service for episode risk prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "models" / "artifacts" / "best_model.joblib"

app = FastAPI(
    title="Urban Air-Quality Episode Early Warning API",
    version="0.1.0",
    description="Predicts probability of an elevated PM2.5 episode in the next 24 hours.",
)


class Measurement(BaseModel):
    """
    Feature vector aligned to training columns.

    For production you would accept raw sensors and run the same feature pipeline.
    This endpoint accepts a dict of engineered feature_name → value for simplicity.
    """

    features: dict[str, float] = Field(
        ...,
        description="Mapping of feature column name to numeric value.",
        examples=[{"PM2.5": 80.0, "PM25_lag_1": 70.0, "TEMP": 5.0}],
    )


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    prediction_window: str
    model_name: str


def _load_artifact() -> dict[str, Any]:
    if not ARTIFACT.exists():
        raise FileNotFoundError(
            f"Missing {ARTIFACT}. Run: python scripts/train_evaluate.py"
        )
    return joblib.load(ARTIFACT)


def _risk_level(p: float) -> str:
    if p < 0.33:
        return "LOW"
    if p < 0.66:
        return "MODERATE"
    return "HIGH"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": ARTIFACT.exists(),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(body: Measurement):
    try:
        art = _load_artifact()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    feature_cols: list[str] = art["feature_cols"]
    missing = [c for c in feature_cols if c not in body.features]
    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Missing required features.",
                "missing_count": len(missing),
                "missing_example": missing[:10],
                "n_required": len(feature_cols),
            },
        )

    x = np.array([[body.features[c] for c in feature_cols]], dtype=float)
    model = art["model"]
    raw_p = float(model.predict_proba(x)[0, 1])
    calibrator = art.get("calibrator")
    if calibrator is not None:
        from src.evaluation.calibration import apply_calibrator

        p = float(apply_calibrator(calibrator, np.array([raw_p]))[0])
    else:
        p = raw_p

    decision_thr = float(art.get("decision_threshold", 0.5))
    pred = int(p >= decision_thr)
    return PredictionResponse(
        prediction=pred,
        probability=round(p, 6),
        risk_level=_risk_level(p),
        prediction_window="next_24_hours",
        model_name=str(art.get("model_name", "unknown")),
    )
