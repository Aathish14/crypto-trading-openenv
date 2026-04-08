"""
OpenEnv server for the cryptocurrency trading environment.
Implements the full OpenEnv API specification with root-level compatibility.
"""

import os
import yaml
import numpy as np
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.encoders import jsonable_encoder

from crypto_trading_env.crypto_trading_env import CryptoTradingEnv

app = FastAPI(title="OpenEnv Crypto Trading API")

# Global storage for environment instances (stateful)
# { "env_id": env_instance }
env_instances: Dict[str, CryptoTradingEnv] = {}

# Load openenv.yaml for task metadata
with open("openenv.yaml", "r") as f:
    config = yaml.safe_load(f)

# Models for the API
class ActionModel(BaseModel):
    action: List[int]

class ObservationModel(BaseModel):
    observation: Any
    info: Dict[str, Any]

class StepResultModel(BaseModel):
    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]

class SimulationRequest(BaseModel):
    steps: int = 100
    strategy: str = "random"

class SimulationResponse(BaseModel):
    steps: List[int]
    rewards: List[float]
    actions: List[List[int]]
    balance: List[float]

def numpy_to_python(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(i) for i in obj]
    return obj

def get_env_kwargs(env_id: str) -> Dict[str, Any]:
    """Get the environment configuration from openenv.yaml."""
    # Find the environment or task ID
    for env in config.get("environments", []):
        if env["id"] == env_id:
            return env.get("kwargs", {})
    
    # Check task list as well
    for task in config.get("task_list", []):
        if task["id"] == env_id:
            target_env_id = task["environment_id"]
            for env in config.get("environments", []):
                if env["id"] == target_env_id:
                    return env.get("kwargs", {})
    
    return {}

# --- OpenEnv API Endpoints (Root & Versioned) ---

@app.post("/reset")
@app.post("/v1/envs/{env_id}/reset")
async def reset(env_id: Optional[str] = None):
    """Reset the specified environment (or default) and return the initial observation."""
    target_id = env_id or "crypto_trading_basic"
    kwargs = get_env_kwargs(target_id)
    if not kwargs:
        # Fallback to the first environment in the list
        kwargs = config["environments"][0]["kwargs"]
    
    env = CryptoTradingEnv(**kwargs)
    obs, info = env.reset()
    env_instances[target_id] = env
    
    return numpy_to_python({
        "observation": obs,
        "info": info
    })

@app.post("/step")
@app.post("/v1/envs/{env_id}/step")
async def step(action_data: ActionModel, env_id: Optional[str] = None):
    """Take a step in the specified environment (or default)."""
    target_id = env_id or "crypto_trading_basic"
    if target_id not in env_instances:
        # Auto-reset if not found (required for root-level pings)
        await reset(target_id)
    
    env = env_instances[target_id]
    action = np.array(action_data.action)
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    return numpy_to_python({
        "observation": obs,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "info": info
    })

@app.get("/v1/envs")
async def list_envs():
    """List all available environment tasks."""
    tasks = []
    for task in config.get("task_list", []):
        tasks.append({
            "id": task["id"],
            "name": task["name"],
            "difficulty": task["difficulty"]
        })
    return {"environments": tasks}

@app.get("/v1/envs/{env_id}/state")
@app.get("/state")
async def get_state(env_id: Optional[str] = None):
    """Return the current state of the environment."""
    target_id = env_id or "crypto_trading_basic"
    if target_id not in env_instances:
        raise HTTPException(status_code=404, detail="Environment instance not found.")
    
    env = env_instances[target_id]
    return numpy_to_python(env._get_current_prices())

# --- Legacy & Internal Endpoints ---

@app.post("/v1/simulate", response_model=SimulationResponse)
async def simulate(request: SimulationRequest):
    """Run a high-speed simulation for the dashboard."""
    # Use default kwargs
    kwargs = config["environments"][0]["kwargs"]
    sim_env = CryptoTradingEnv(**kwargs)
    
    obs, info = sim_env.reset()
    results = {"steps": [], "rewards": [], "actions": [], "balance": []}
    
    for i in range(request.steps):
        action = sim_env.action_space.sample()
        obs, reward, terminated, truncated, info = sim_env.step(action)
        
        results["steps"].append(i)
        results["rewards"].append(float(reward))
        results["actions"].append(action.tolist())
        results["balance"].append(float(info["portfolio_value"]))
        
        if terminated or truncated:
            break
            
    return results

@app.get("/health")
async def health():
    return {"status": "healthy"}

# --- Static Dashboard ---

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
