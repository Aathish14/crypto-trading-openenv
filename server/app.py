"""
OpenEnv server for the cryptocurrency trading environment.
"""

import os
from typing import List, Dict, Any
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from openenv.core import create_app, Action, Observation
from crypto_trading_env.crypto_trading_env import CryptoTradingEnv


def main():
    """Main function to run the OpenEnv server."""
    # Create the app using OpenEnv's factory function
    app = create_app(
        env=lambda: CryptoTradingEnv(
            symbols=["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "LTC-USD"],
            initial_balance=10000,
            transaction_fee=0.001,
            lookback_window=10,
        ),
        action_cls=Action,
        observation_cls=Observation,
        env_name="crypto-trading-env",
    )

    # Models for simulation
    class SimulationRequest(BaseModel):
        steps: int = 100
        strategy: str = "random"

    class SimulationResponse(BaseModel):
        steps: List[int]
        rewards: List[float]
        actions: List[List[int]]
        balance: List[float]

    # Mount static files
    app.mount("/static", StaticFiles(directory="static"), name="static")

    # Serve the dashboard at the root URL
    @app.get("/")
    async def read_index():
        return FileResponse("static/index.html")

    @app.post("/v1/simulate", response_model=SimulationResponse)
    async def simulate(request: SimulationRequest):
        # Create a fresh environment for simulation
        sim_env = CryptoTradingEnv(
            symbols=["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "LTC-USD"],
            initial_balance=10000,
            transaction_fee=0.001,
            lookback_window=10,
        )
        
        obs, info = sim_env.reset()
        
        results = {
            "steps": [],
            "rewards": [],
            "actions": [],
            "balance": []
        }
        
        for i in range(request.steps):
            # Take random action
            action = sim_env.action_space.sample()
            
            # Step the environment
            obs, reward, terminated, truncated, info = sim_env.step(action)
            
            # Collect results
            results["steps"].append(i)
            results["rewards"].append(float(reward))
            results["actions"].append(action.tolist())
            results["balance"].append(float(info["portfolio_value"]))
            
            if terminated or truncated:
                break
                
        return results

    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
