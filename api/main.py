"""
RAIN-X Operational REST API Backend
FastAPI server providing endpoints for:
  - Held-out verification scorecard and regime stratification
  - District & point-level post-processed forecasts
  - Forecast Battle Mode benchmark cases (Cyclone Michaung, Wayanad, etc.)
  - AI Forecast Analyst Agent briefings
Strictly bound to 127.0.0.1 for secure local execution.
"""

import os
import sys
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure root path in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import RainXEngine
from src.spatial.domain import DISTRICT_CATALOG

# Security Note: Server will bind to 127.0.0.1
app = FastAPI(
    title="RAIN-X Operational API",
    description="Regime-Aware Neural Post-Processing Engine for NWP Rainfall Forecasts (SIH 26080)",
    version="1.0.0"
)

# CORS Policy: Restrict to localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize Engine singleton
engine = RainXEngine()


class DistrictForecastRequest(BaseModel):
    state: str = Field(..., example="Andhra Pradesh")
    district: str = Field(..., example="Visakhapatnam")
    station: Optional[str] = Field(None, example="Anakapalli")
    lead_time_hrs: int = Field(24, ge=6, le=120, example=24)


class PointForecastRequest(BaseModel):
    lat: float = Field(..., ge=8.0, le=38.0, example=17.68)
    lon: float = Field(..., ge=68.0, le=98.0, example=83.00)
    raw_nwp_mm: float = Field(..., ge=0.0, le=1000.0, example=84.0)
    lead_time_hrs: int = Field(24, ge=6, le=120, example=24)
    location_name: Optional[str] = Field("Custom Location", example="Visakhapatnam")


@app.on_event("startup")
def startup_event():
    # Warm up models
    print("[API] Initializing and checking RAIN-X models...")
    if not engine.is_trained:
        engine.load_or_train_models()
    print("[API] RAIN-X engine ready.")


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "engine": "RAIN-X Regime-Aware Neural Post-Processing Engine",
        "version": "1.0.0",
        "sih_problem": "SIH 26080 (NCMRWF / MoES)",
        "models_trained": engine.is_trained
    }


@app.get("/api/districts")
def get_district_hierarchy():
    """Returns catalog of states, districts, and stations with coordinates."""
    return DISTRICT_CATALOG


@app.get("/api/scorecard")
def get_verification_scorecard():
    """Returns official verification scorecard and regime-stratified evaluation."""
    if engine.verification_cache is None:
        engine.run_held_out_verification()
    return engine.verification_cache


@app.get("/api/benchmarks")
def get_benchmark_cases():
    """Returns manifest of benchmark extreme events for Forecast Battle mode."""
    manifest_path = os.path.join(engine.data_dir, "benchmarks", "manifest.json")
    if not os.path.exists(manifest_path):
        raise HTTPException(status_code=404, detail="Benchmark manifest not found. Run dataset_generator.py.")
    
    import json
    with open(manifest_path) as f:
        data = json.load(f)
    return list(data.values())


@app.get("/api/benchmarks/{event_id}")
def get_benchmark_case_details(event_id: str):
    """Returns gridded forecast battle data (Raw NWP vs RAIN-X vs Observed)."""
    # Defensive input sanitization: allow only alphanumeric and underscore
    if not event_id.replace("_", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid event_id parameter")
    try:
        return engine.get_benchmark_event_case(event_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Benchmark case {event_id} not found.")


@app.post("/api/predict/district")
def predict_district(req: DistrictForecastRequest):
    """Generates complete forecast, regime intelligence, and XAI for a district/station."""
    # Sanitize inputs
    state = req.state.strip()
    district = req.district.strip()
    station = req.station.strip() if req.station else None
    
    if state not in DISTRICT_CATALOG:
        # Fallback to first available state
        state = list(DISTRICT_CATALOG.keys())[0]

    return engine.predict_district(
        state=state,
        district=district,
        station=station,
        lead_time_hrs=req.lead_time_hrs
    )


@app.post("/api/predict/point")
def predict_point(req: PointForecastRequest):
    """Generates complete forecast, regime intelligence, and XAI for arbitrary coordinates."""
    return engine.predict_point(
        lat=req.lat,
        lon=req.lon,
        raw_nwp_mm=req.raw_nwp_mm,
        lead_time_hrs=req.lead_time_hrs,
        location_name=req.location_name
    )


# Serve Static Web Frontend
web_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "web"))
if os.path.exists(web_dir):
    app.mount("/", StaticFiles(directory=web_dir, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    # Enforce strict 127.0.0.1 binding as required by security guidelines
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False)
