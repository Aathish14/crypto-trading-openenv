import requests
import json

BASE_URL = "http://localhost:7860"
ENV_ID = "crypto_trading_basic"

def test_api():
    print(f"Testing OpenEnv API at {BASE_URL}...")
    
    try:
        # 1. List Envs
        print("\n1. GET /v1/envs")
        resp = requests.get(f"{BASE_URL}/v1/envs")
        print(f"Status: {resp.status_code}")
        print(f"Data: {json.dumps(resp.json(), indent=2)}")
        
        # 2. Reset
        print(f"\n2. POST /v1/envs/{ENV_ID}/reset")
        resp = requests.post(f"{BASE_URL}/v1/envs/{ENV_ID}/reset")
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            print("Success! Initial observation received.")
        else:
            print(f"Error: {resp.text}")
            return

        # 3. Step
        print(f"\n3. POST /v1/envs/{ENV_ID}/step")
        payload = {"action": [0, 0]}
        resp = requests.post(f"{BASE_URL}/v1/envs/{ENV_ID}/step", json=payload)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"Success! Reward: {data['reward']}")
            print(f"Done: {data['terminated'] or data['truncated']}")
        else:
            print(f"Error: {resp.text}")

        # 4. State
        print(f"\n4. GET /v1/envs/{ENV_ID}/state")
        resp = requests.get(f"{BASE_URL}/v1/envs/{ENV_ID}/state")
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            print(f"Success! State: {resp.json()}")
            
    except Exception as e:
        print(f"Connection error: {e}")

if __name__ == "__main__":
    test_api()
