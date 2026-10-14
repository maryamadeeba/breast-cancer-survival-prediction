"""FastAPI backend for the Breast Cancer Survival Prediction app.

Run from the backend/ folder:   uvicorn app.main:app --reload --port 8000
Docs (Swagger UI):              http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .ml import SurvivalModel, read_metrics_json
from .schemas import FeatureContribution, ModelInfo, PatientData, PredictionResponse, to_model_row

logging.basicConfig(level=logging.INFO)
STATE: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    STATE["model"] = SurvivalModel.load()
    yield
    STATE.clear()


app = FastAPI(
    title="Breast Cancer Survival Prediction API",
    description="Estimates the chance of surviving 10 years after diagnosis from 15 clinical inputs. "
                "For research and education only; not a medical device.",
    version="4.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok", "modelLoaded": "model" in STATE}


@app.get("/api/metrics")
def metrics():
    """Full cross-validation and calibration report written by train_model.py."""
    return read_metrics_json()


@app.post("/api/predict", response_model=PredictionResponse)
def predict(patient: PatientData):
    model: SurvivalModel = STATE["model"]
    row, warnings = to_model_row(patient, model.ranges)
    out = model.predict(row)
    contributions = sorted(out["contributions"], key=lambda c: abs(c["value"]), reverse=True)[:6]
    return PredictionResponse(
        survivalProbability=out["survival_probability"],
        deathProbability=out["death_probability"],
        riskLevel=out["risk_level"],
        riskPercentile=out["risk_percentile"],
        horizonYears=model.horizon_years,
        milestones=out["milestones"],
        survivalCurve=out["curve"],
        medianSurvivalYears=out["median_survival_years"],
        featureContributions=[
            FeatureContribution(
                feature=c["feature"],
                contribution=abs(c["value"]),
                direction="increases" if c["value"] > 0 else "decreases",
            )
            for c in contributions
        ],
        modelInfo=ModelInfo(**model.info()),
        warnings=warnings,
    )


# Optional: after `npm run build`, serve the built frontend from the same server (http://127.0.0.1:8000)
_DIST = Path(__file__).resolve().parents[2] / "dist"
if _DIST.exists():
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="frontend")
