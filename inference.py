"""
Baseline inference script for the cryptocurrency trading environment.
Uses OpenAI API to run a model against all 3 defined environments.
"""

import os
import yaml
import numpy as np
import sys

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not found. Please ensure it is installed.")
    sys.exit(1)

from crypto_trading_env.crypto_trading_env import CryptoTradingEnv

def load_tasks():
    with open("openenv.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    tasks = []
    for task in config.get("task_list", []):
        env_def = None
        for env in config.get("environments", []):
            if env["id"] == task["environment_id"]:
                env_def = env
                break
        
        if env_def:
            kwargs = env_def.get("kwargs", {}).copy()
            # Feed the task's success threshold as the goal_return to the environment
            kwargs["goal_return"] = task.get("success_threshold", 0.1)
            tasks.append({
                "id": task["id"],
                "name": task["name"],
                "difficulty": task["difficulty"],
                "kwargs": kwargs,
                "max_steps": min(15, task.get("max_episode_steps", 50)) # Cap for fast demo
            })
    return tasks

def run_baseline_inference():
    """Run baseline inference using OpenAI-compatible API (NVIDIA Nemotron)."""
    api_base_url = os.getenv("API_BASE_URL", "https://integrate.api.nvidia.com/v1")
    model_name = os.getenv("MODEL_NAME", "nvidia/nemotron-3-super-120b-a12b")
    hf_token = os.getenv("HF_TOKEN")
    
    api_key = os.getenv("OPENAI_API_KEY") or hf_token

    if not api_key:
        print("Warning: Neither OPENAI_API_KEY nor HF_TOKEN environment variables set.")
        use_llm = False
        client = None
    else:
        client = OpenAI(api_key=api_key, base_url=api_base_url)
        use_llm = True

    print(f"Using API base URL: {api_base_url}")
    print(f"Using model: {model_name}\n")

    tasks = load_tasks()
    if not tasks:
        print("Error: No tasks loaded from openenv.yaml")
        sys.exit(1)

    print(f"Loaded {len(tasks)} tasks for evaluation.\n")
    
    all_results = {}

    for t_idx, task in enumerate(tasks):
        print("="*50)
        print(f"TASK {t_idx+1}/{len(tasks)}: {task['name']} ({task['difficulty']})")
        print("="*50)
        
        env = CryptoTradingEnv(**task["kwargs"])
        observation, info = env.reset()
        
        total_reward = 0
        step_count = 0
        
        print("[START]")
        
        while step_count < task["max_steps"]:
            balance = env.balance
            holdings = env.holdings.copy()
            current_prices = env._get_current_prices()

            obs_description = f"""
Step: {step_count}
Balance: ${balance:.2f}
Holdings: {dict(zip(env.symbols, holdings))}
Current Prices: {dict(zip(env.symbols, current_prices))}
Portfolio Value: ${info["portfolio_value"]:.2f}
Goal Return: {task['kwargs'].get('goal_return', 0.1) * 100}%
"""

            if use_llm:
                try:
                    action = get_llm_action(obs_description, model_name, env.symbols, client)
                except Exception as e:
                    print(f"Error getting LLM action: {e}")
                    action = env.action_space.sample()
            else:
                action = env.action_space.sample()

            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            print(f"[STEP] step={step_count} reward={reward:.4f} portfolio_value={info['portfolio_value']:.2f}")

            if terminated or truncated:
                break
                
        print(f"[END] total_reward={total_reward:.4f} steps={step_count} final_balance={info['portfolio_value']:.2f}")
        
        # Extract the score from info (which we mapped strictly to 0.01 - 0.99)
        score = info.get("score", 0.01)
        print(f"\n[SCORE] task_id={task['id']} score={score:.4f}\n")
        all_results[task['id']] = score
        
    print("=== FINAL BASELINE EVALUATION SCORES ===")
    for task_id, score in all_results.items():
        print(f"- {task_id}: {score:.4f}")

def get_llm_action(obs_description, model_name, symbols, client):
    """Get action from LLM based on observation description."""
    prompt = f"""
You are a cryptocurrency trading agent.
Current state:
{obs_description}

Provide a comma-separated list of integers (0-8) for each asset ({", ".join(symbols)}).
0=Hold, 1-4=Buy(25%-100%), 5-8=Sell(25%-100%).
Your action:
"""
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0.7,
    )
    action_text = response.choices[0].message.content.strip()
    try:
        actions = [int(x.strip()) for x in action_text.split(",")]
        if len(actions) < len(symbols):
            actions = actions + [0] * (len(symbols) - len(actions))
        actions = actions[:len(symbols)]
        return np.array([max(0, min(8, a)) for a in actions])
    except:
        return np.array([0] * len(symbols))

if __name__ == "__main__":
    run_baseline_inference()
