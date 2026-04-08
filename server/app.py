"""
OpenEnv server for the cryptocurrency trading environment.
"""

import os
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

    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
