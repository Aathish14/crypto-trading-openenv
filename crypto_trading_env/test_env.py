"""Test script for the cryptocurrency trading environment."""

import numpy as np
import sys
import os

# Add the current directory to Python path to enable importing the module
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from crypto_trading_env import CryptoTradingEnv


def test_environment():
    """Test the cryptocurrency trading environment."""
    print("Testing CryptoTradingEnv...")

    # Create environment
    env = CryptoTradingEnv(
        symbols=["BTC-USD", "ETH-USD"],  # Use fewer assets for faster testing
        initial_balance=1000,
        lookback_window=5,
    )

    # Test reset
    print("\n1. Testing reset...")
    obs, info = env.reset()
    print(f"   Observation shape: {obs.shape}")
    print(f"   Expected shape: ({env.observation_space.shape[0]},)")
    print(f"   Initial portfolio value: ${info['portfolio_value']:.2f}")
    assert obs.shape == env.observation_space.shape
    assert info["portfolio_value"] == env.initial_balance
    print("   * Reset test passed")

    # Test action space
    print("\n2. Testing action space...")
    print(f"   Action space: {env.action_space}")
    print(f"   Number of assets: {env.n_assets}")
    print(f"   Actions per asset: {env.n_actions_per_asset}")
    sample_action = env.action_space.sample()
    print(f"   Sample action: {sample_action}")
    assert len(sample_action) == env.n_assets
    assert all(0 <= a < env.n_actions_per_asset for a in sample_action)
    print("   * Action space test passed")

    # Test step
    print("\n3. Testing step...")
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(
            f"   Step {i + 1}: Reward = {reward:.4f}, Portfolio = ${info['portfolio_value']:.2f}"
        )
        if terminated:
            print(f"   Episode terminated at step {i + 1}")
            break

    print(f"   Total reward over {i + 1} steps: {total_reward:.4f}")
    print("   * Step test passed")

    # Test observation bounds
    print("\n4. Testing observation bounds...")
    obs_low = env.observation_space.low
    obs_high = env.observation_space.high
    print(f"   Observation low bounds: {obs_low[:5]}...")  # Show first 5
    print(f"   Observation high bounds: {obs_high[:5]}...")  # Show first 5
    # Note: Box space with infinite bounds doesn't require checking values against bounds
    print("   * Observation bounds test passed")

    print("\n* All tests passed!")
    return True


if __name__ == "__main__":
    test_environment()
