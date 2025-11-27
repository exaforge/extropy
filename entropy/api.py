"""FastAPI app for Entropy web UI.

This provides a REST API for the Entropy functionality.
Currently a stub for future web UI development.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import db
from .models import Population, Scenario, SimulationResult

app = FastAPI(
    title="Entropy API",
    description="Simulate how populations respond to scenarios",
    version="0.1.0",
)


# =============================================================================
# Request/Response Models
# =============================================================================


class CreatePopulationRequest(BaseModel):
    context: str
    name: str
    seed: int | None = None


class CreateScenarioRequest(BaseModel):
    description: str
    population_name: str
    name: str


class RunSimulationRequest(BaseModel):
    population_name: str
    scenario_name: str
    mode: str = "single"
    duration: str | None = None


# =============================================================================
# Population Endpoints
# =============================================================================


@app.post("/populations")
async def create_population(request: CreatePopulationRequest):
    """Create a new population."""
    # TODO: Implement async population creation
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/populations")
async def list_populations():
    """List all populations."""
    return db.list_populations()


@app.get("/populations/{name}")
async def get_population(name: str):
    """Get a population by name."""
    population = db.load_population(name, include_agents=False)
    if not population:
        raise HTTPException(status_code=404, detail=f"Population '{name}' not found")
    return population.model_dump()


@app.delete("/populations/{name}")
async def delete_population(name: str):
    """Delete a population."""
    if not db.population_exists(name):
        raise HTTPException(status_code=404, detail=f"Population '{name}' not found")
    db.delete_population(name)
    return {"status": "deleted", "name": name}


# =============================================================================
# Scenario Endpoints (Phase 2)
# =============================================================================


@app.post("/scenarios")
async def create_scenario(request: CreateScenarioRequest):
    """Create a scenario for a population."""
    raise HTTPException(status_code=501, detail="Phase 2 not implemented")


@app.get("/scenarios")
async def list_scenarios():
    """List all scenarios."""
    raise HTTPException(status_code=501, detail="Phase 2 not implemented")


# =============================================================================
# Simulation Endpoints (Phase 3)
# =============================================================================


@app.post("/simulations")
async def run_simulation(request: RunSimulationRequest):
    """Run a simulation."""
    raise HTTPException(status_code=501, detail="Phase 3 not implemented")


@app.get("/simulations/{id}")
async def get_simulation_result(id: str):
    """Get simulation results."""
    raise HTTPException(status_code=501, detail="Phase 3 not implemented")


@app.get("/simulations/{id}/timeline")
async def get_simulation_timeline(id: str):
    """Get simulation timeline (continuous mode)."""
    raise HTTPException(status_code=501, detail="Phase 3 not implemented")


# =============================================================================
# Health Check
# =============================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}

