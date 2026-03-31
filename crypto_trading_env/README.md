# Cryptocurrency Trading Environment

A reinforcement learning environment for simulating cryptocurrency trading with multiple assets, following OpenAI Gymnasium standards.

## Overview

This environment simulates a real-world cryptocurrency trading scenario where an intelligent agent acts as an automated trader managing a financial portfolio. The environment is designed to replicate realistic market conditions using historical price data, enabling the agent to make sequential decisions such as buying, selling, or holding assets.

## Features

- **Multi-asset trading**: Trade BTC, ETH, BNB, XRP, and LTC against USD
- **5-minute candle data**: Uses 5-minute OHLCV (Open, High, Low, Close, Volume) data
- **Discrete position sizing**: Agent can choose to hold, buy (25%, 50%, 75%, 100% of balance), or sell (25%, 50%, 75%, 100% of holdings) for each asset
- **Transaction costs**: 0.1% fee applied to each trade
- **Realistic simulation**: Synthetic price data generation with realistic volatility patterns
- **Gymnasium compatibility**: Follows OpenAI Gymnasium interface standards

## Installation

```bash
pip install numpy pandas gymnasium
```

## Usage

```python
from crypto_trading_env import CryptoTradingEnv

# Create environment
env = CryptoTradingEnv(
    symbols=['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'LTC-USD'],
    initial_balance=10000,
    transaction_fee=0.001,
    lookback_window=10
)

# Reset environment
observation, info = env.reset()

# Run episode
for step in range(1000):
    # Sample random action (replace with your agent's policy)
    action = env.action_space.sample()
    
    # Take step
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Optional: render environment state
    env.render()
    
    if terminated or truncated:
        break

# Get final portfolio value
final_value = info['portfolio_value']
return_pct = (final_value - env.initial_balance) / env.initial_balance * 100
```

## Environment Details

### Action Space
- Type: `MultiDiscrete([9, 9, 9, 9, 9])` (9 actions per asset)
- Actions per asset:
  - 0: Hold (no action)
  - 1: Buy 25% of available balance
  - 2: Buy 50% of available balance
  - 3: Buy 75% of available balance
  - 4: Buy 100% of available balance
  - 5: Sell 25% of current holdings
  - 6: Sell 50% of current holdings
  - 7: Sell 75% of current holdings
  - 8: Sell 100% of current holdings

### Observation Space
- Type: `Box(low=-inf, high=inf, shape=(lookback_window * 5 * n_assets + 1 + n_assets,))`
- Components:
  - OHLCV data for each asset over lookback_window periods
  - Current USD balance
  - Current holdings amount for each asset

### Reward Function
- Reward = change in portfolio value from previous step
- Encourages profitable trading while accounting for transaction costs

### Episode Termination
- Episode ends when all available data has been processed
- No early termination conditions

## Files

- `crypto_trading_env.py`: Main environment implementation
- `test_env.py`: Test script to verify environment functionality
- `README.md`: This documentation

## Customization

You can modify the following parameters when creating the environment:
- `symbols`: List of cryptocurrency pairs to trade (default: ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'LTC-USD'])
- `initial_balance`: Starting portfolio balance in USD (default: 10000)
- `transaction_fee`: Fee percentage per trade (default: 0.001)
- `lookback_window`: Number of historical periods to include in state (default: 10)

## Notes

- This implementation uses synthetic data for demonstration purposes
- For production use, replace the `_load_data()` method with actual historical data loading
- The environment is designed to be compatible with stable-baselines3 and other RL libraries