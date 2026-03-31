"""
Example usage of the CryptoTradingEnv
"""

import numpy as np
from crypto_trading_env import CryptoTradingEnv


def run_random_agent():
    """Run a random agent for demonstration."""
    print("Creating cryptocurrency trading environment...")

    # Create environment with fewer assets for faster demonstration
    env = CryptoTradingEnv(
        symbols=["BTC-USD", "ETH-USD"],  # Just two assets for demo
        initial_balance=10000,
        transaction_fee=0.001,
        lookback_window=10,
    )

    # Reset environment
    print("\nResetting environment...")
    observation, info = env.reset()
    print(f"Initial observation shape: {observation.shape}")
    print(f"Initial portfolio value: ${info['portfolio_value']:.2f}")

    # Run episode with random actions
    print("\nRunning episode with random agent...")
    total_reward = 0
    step_count = 0

    while True:
        # Sample random action (in practice, this would be your RL agent's policy)
        action = env.action_space.sample()

        # Take step in environment
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step_count += 1

        # Render every 50 steps to see progress
        if step_count % 50 == 0:
            env.render()

        # Check if episode is done
        if terminated or truncated:
            print(f"\nEpisode finished after {step_count} steps")
            env.render()
            break

    # Print final results
    print("\n=== FINAL RESULTS ===")
    print(f"Initial balance: ${env.initial_balance:.2f}")
    print(f"Final portfolio value: ${info['portfolio_value']:.2f}")
    print(f"Total return: {info['total_return'] * 100:.2f}%")
    print(f"Total reward (sum of step rewards): {total_reward:.2f}")

    return info


if __name__ == "__main__":
    run_random_agent()
