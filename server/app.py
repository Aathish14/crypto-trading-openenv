"""
OpenEnv server for the cryptocurrency trading environment.
Implements the full OpenEnv API specification with root-level compatibility.
"""

import os
import yaml
import numpy as np
import uvicorn
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.encoders import jsonable_encoder

from crypto_trading_env.crypto_trading_env import CryptoTradingEnv

app = FastAPI(title="OpenEnv Crypto Trading API")

# Global storage for environment instances (stateful)
env_instances: Dict[str, CryptoTradingEnv] = {}

# Load config once at start
config_path = "openenv.yaml"
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
else:
    config = {"environments": [], "task_list": []}

# Models
class ActionModel(BaseModel):
    action: List[int]

def numpy_to_python(obj):
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
    for env in config.get("environments", []):
        if env["id"] == env_id: return env.get("kwargs", {})
    for task in config.get("task_list", []):
        if task["id"] == env_id:
            target_env_id = task["environment_id"]
            for env in config.get("environments", []):
                if env["id"] == target_env_id: return env.get("kwargs", {})
    return {}

@app.post("/reset")
@app.post("/v1/envs/{env_id}/reset")
async def reset(env_id: Optional[str] = None):
    target_id = env_id or "crypto_trading_basic"
    kwargs = get_env_kwargs(target_id)
    if not kwargs and config.get("environments"):
        kwargs = config["environments"][0]["kwargs"]
    env = CryptoTradingEnv(**kwargs)
    obs, info = env.reset()
    env_instances[target_id] = env
    return numpy_to_python({"observation": obs, "info": info})

@app.post("/step")
@app.post("/v1/envs/{env_id}/step")
async def step(action_data: ActionModel, env_id: Optional[str] = None):
    target_id = env_id or "crypto_trading_basic"
    if target_id not in env_instances: await reset(target_id)
    env = env_instances[target_id]
    obs, reward, terminated, truncated, info = env.step(np.array(action_data.action))
    return numpy_to_python({"observation": obs, "reward": reward, "terminated": terminated, "truncated": truncated, "info": info})

@app.get("/v1/envs")
async def list_envs():
    tasks = []
    for task in config.get("task_list", []):
        tasks.append({"id": task["id"], "name": task["name"], "difficulty": task["difficulty"]})
    return {"environments": tasks}

@app.get("/health")
async def health(): return {"status": "healthy"}

app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/")
async def read_index(): return FileResponse("static/index.html")

def main():
    """Entry point for the server as defined in pyproject.toml."""
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    main()
