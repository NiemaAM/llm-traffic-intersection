"""
app.py
------
FastAPI service for LLM-based intersection conflict resolution.
Milestone 5: Production model serving with on-demand inference.
"""

import os
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.llm_model import IntersectionLLM

# ─── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="🚦 Intersection Conflict Resolver API",
    description=(
        "LLM-driven vehicle conflict detection and resolution at intersections. "
        "Accepts vehicle scenarios, returns conflict analysis and control decisions."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Model singleton ──────────────────────────────────────────────────────────

_llm: IntersectionLLM | None = None

def get_llm() -> IntersectionLLM:
    global _llm
    if _llm is None:
        _llm = IntersectionLLM(
            model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
            few_shot=os.getenv("FEW_SHOT", "true").lower() == "true",
            fine_tuned_model_id=os.getenv("FINE_TUNED_MODEL_ID"),
        )
    return _llm


# ─── Request / Response schemas ───────────────────────────────────────────────

class Vehicle(BaseModel):
    vehicle_id: str = Field(..., json_schema_extra={"example": "V1234"})
    lane: int = Field(..., ge=1, le=10, json_schema_extra={"example": 1})
    speed: float = Field(..., ge=0, le=200, json_schema_extra={"example": 62.5})
    distance_to_intersection: float = Field(..., ge=0, le=2000, json_schema_extra={"example": 45.0})
    direction: str = Field(..., pattern="^(north|south|east|west)$", json_schema_extra={"example": "north"})
    destination: str = Field(..., json_schema_extra={"example": "A"})


class ScenarioRequest(BaseModel):
    vehicles: list[Vehicle] = Field(..., min_length=2, max_length=20)


class ConflictPair(BaseModel):
    vehicle1_id: str
    vehicle2_id: str


class PredictionResponse(BaseModel):
    is_conflict: str
    number_of_conflicts: int
    conflict_vehicles: list[ConflictPair]
    decisions: list[str]
    priority_order: dict[str, Any]
    waiting_times: dict[str, Any]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model: str
    few_shot: bool
    version: str


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Health check endpoint."""
    llm = get_llm()
    return {
        "status": "ok",
        "model": llm.model,
        "few_shot": llm.few_shot,
        "version": "1.0.0",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(request: ScenarioRequest):
    """
    Analyze an intersection scenario and return conflict decisions.

    **Serving mode:** On-demand inference (request-response).
    """
    llm = get_llm()
    scenario = {"vehicles": [v.model_dump() for v in request.vehicles]}

    t0 = time.time()
    try:
        result = llm.predict(scenario)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    latency_ms = (time.time() - t0) * 1000

    # Parse conflict_vehicles if returned as string
    cv = result.get("conflict_vehicles", [])
    if isinstance(cv, str):
        import ast
        cv = ast.literal_eval(cv)

    return {
        **result,
        "conflict_vehicles": cv,
        "latency_ms": round(latency_ms, 2),
    }


@app.post("/predict/batch", tags=["Inference"])
def predict_batch(requests: list[ScenarioRequest]):
    """
    Batch inference endpoint.
    **Serving mode:** Batch processing.
    """
    llm = get_llm()
    results = []
    for req in requests:
        scenario = {"vehicles": [v.model_dump() for v in req.vehicles]}
        try:
            results.append(llm.predict(scenario))
        except Exception as e:
            results.append({"error": str(e)})
    return {"results": results, "count": len(results)}


@app.get("/", tags=["System"])
def root():
    return {
        "message": "🚦 Intersection Conflict Resolver API",
        "docs": "/docs",
        "health": "/health",
    }


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
