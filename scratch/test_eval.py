import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../rl_agent")))

import numpy as np
from stable_baselines3 import DQN, PPO
from rl_agent.uav_env import UAVRoutingEnv
from double_dqn import DoubleDQN

def test_models():
    model_dir = "models"
    dummy_env = UAVRoutingEnv(grid_size=15, obstacle_density=0.20, no_fly_density=0.0, fixed_grid=True)
    
    models_to_test = [
        "dqn_static_uav.zip",
        "best/best_model.zip",
        "dqn_static_uav_bfs_1m.zip",
        "dqn_static_uav_reldisp_500k.zip"
    ]
    
    for m in models_to_test:
        path = os.path.join(model_dir, m)
        if not os.path.exists(path):
            print(f"Skipping {path} - does not exist")
            continue
            
        print(f"\n--- Testing {path} ---")
        loaded = False
        
        try:
            model = DQN.load(path, env=dummy_env)
            print("Loaded successfully as DQN!")
            loaded = True
        except Exception as e:
            print(f"Failed to load as DQN: {e}")
            
        if not loaded:
            try:
                model = DoubleDQN.load(path, env=dummy_env)
                print("Loaded successfully as DoubleDQN!")
                loaded = True
            except Exception as e:
                print(f"Failed to load as DoubleDQN: {e}")
                
        if not loaded:
            try:
                model = PPO.load(path, env=dummy_env)
                print("Loaded successfully as PPO!")
                loaded = True
            except Exception as e:
                print(f"Failed to load as PPO: {e}")

if __name__ == "__main__":
    test_models()
