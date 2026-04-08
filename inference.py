"""
Baseline inference script for the cryptocurrency trading environment.
Uses OpenAI API to run a model against the environment.
"""

import os
import numpy as np
import sys

# Try to import openai, if not available, install it
try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not found. Please ensure it is installed.")
    raise

from crypto_trading_env.crypto_trading_env import CryptoTradingEnv


def run_baseline_inference():
    """Run baseline inference using OpenAI-compatible API (NVIDIA Nemotron)."""
    # Get API credentials from environment variables
    # Defaulting to NVIDIA's endpoint and Nemotron-3 model for high performance
    api_base_url = os.environ.get("API_BASE_URL", "https://integrate.api.nvidia.com/v1")
    api_key = os.environ.get("OPENAI_API_KEY")  # The spec requires this variable name
    model_name = os.environ.get("MODEL_NAME", "nvidia/nemotron-3-super-120b-a12b")

    if not api_key:
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("Using random actions as fallback.")
        use_llm = False
        client = None
    else:
        # Initialize OpenAI client
        if api_base_url == "https://api.openai.com/v1":
            client = OpenAI(api_key=api_key)
        else:
            client = OpenAI(api_key=api_key, base_url=api_base_url)
        use_llm = True

    print(f"Using API base URL: {api_base_url}")
    print(f"Using model: {model_name}")

    # Create environment
    print("Creating environment...")
    env = CryptoTradingEnv(
        symbols=["BTC-USD", "ETH-USD"],  # Simplified for faster inference
        initial_balance=1000,
        transaction_fee=0.001,
        lookback_window=5,
    )

    # Reset environment
    print("Resetting environment...")
    observation, info = env.reset()
    print(f"Initial observation shape: {observation.shape}")
    print(f"Initial portfolio value: ${info['portfolio_value']:.2f}")

    # Run episode
    total_reward = 0
    step_count = 0
    max_steps = 50  # Limit steps for demo

    print("Running episode...")
    while step_count < max_steps:
        # Get current state information for description
        current_step = (
            env.current_step - env.lookback_window
        )  # Adjust for lookback window
        balance = env.balance
        holdings = env.holdings.copy()

        # Get current prices
        try:
            current_prices = env._get_current_prices()
        except Exception as e:
            print(f"Error getting current prices: {e}")
            current_prices = np.zeros(len(env.symbols))

        # Create description for LLM
        obs_description = f"""
Step: {step_count}
Balance: ${balance:.2f}
Holdings: {dict(zip(env.symbols, holdings))}
Current Prices: {dict(zip(env.symbols, current_prices))}
Portfolio Value: ${info["portfolio_value"]:.2f}
"""

        # Choose action
        if use_llm:
            try:
                action = get_llm_action(
                    obs_description, model_name, env.symbols, client
                )
            except Exception as e:
                print(f"Error getting LLM action: {e}")
                # Fallback to random action
                action = env.action_space.sample()
        else:
            # Random action
            action = env.action_space.sample()

        print(f"Step {step_count + 1}: Taking action {action}")

        # Take step in environment
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        print(f"  Reward: {reward:.4f}, Total Reward: {total_reward:.4f}")
        print(f"  Portfolio Value: ${info['portfolio_value']:.2f}")

        if terminated or truncated:
            print("Episode finished early!")
            break

    # Final results
    print("\n=== BASELINE INFERENCE RESULTS ===")
    print(f"Total steps: {step_count}")
    print(f"Total reward: {total_reward:.4f}")
    print(f"Final portfolio value: ${info['portfolio_value']:.2f}")
    print(f"Initial balance: ${env.initial_balance:.2f}")
    if info["portfolio_value"] > 0:
        return_pct = (
            (info["portfolio_value"] - env.initial_balance) / env.initial_balance * 100
        )
        print(f"Return: {return_pct:.2f}%")

    return {
        "total_reward": total_reward,
        "final_portfolio_value": info["portfolio_value"],
        "steps": step_count,
        "initial_balance": env.initial_balance,
    }


def get_llm_action(obs_description, model_name, symbols, client):
    """Get action from LLM based on observation description."""
    prompt = f"""
You are a cryptocurrency trading agent. Your goal is to maximize portfolio value through strategic trading.

Current observation:
{obs_description}

Available actions for each asset ({", ".join(symbols)}):
0: Hold
1: Buy 25% of balance
2: Buy 50% of balance
3: Buy 75% of balance
4: Buy 100% of balance
5: Sell 25% of holdings
6: Sell 50% of holdings
7: Sell 75% of holdings
8: Sell 100% of holdings

Based on the current market conditions and your portfolio state, choose actions for each asset.
Provide your answer as a comma-separated list of integers (one per asset, in the same order as the assets listed above).
For example, if there are 2 assets (BTC-USD, ETH-USD), you might return "1,3" for Buy 25% BTC, Buy 75% ETH.

Your action (comma-separated integers, one per asset):
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a cryptocurrency trading agent."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=10,
        temperature=0.7,
    )

    action_text = response.choices[0].message.content.strip()
    print(f"LLM raw response: '{action_text}'")

    # Parse the response to get actions
    try:
        actions = [int(x.strip()) for x in action_text.split(",")]
        # Ensure we have the correct number of actions
        if len(actions) != len(symbols):
            # If wrong number, repeat the first action or truncate
            if len(actions) < len(symbols):
                actions = actions + [actions[-1]] * (len(symbols) - len(actions))
            else:
                actions = actions[: len(symbols)]
        # Ensure actions are in valid range
        actions = [max(0, min(8, a)) for a in actions]
    except (ValueError, AttributeError):
        # Fallback to hold actions if parsing fails
        actions = [0] * len(symbols)

    # Convert to the format expected by our environment (MultiDiscrete)
    # Our environment expects a numpy array of actions for each asset
    return np.array(actions)


if __name__ == "__main__":
    run_baseline_inference()
