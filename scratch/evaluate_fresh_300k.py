import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../rl_agent")))

import numpy as np
from rl_agent.uav_env import UAVRoutingEnv
from double_dqn import DoubleDQN

def main():
    model_path = r"C:\Users\Fast Computer\Downloads\dqn_her_300k_final (1).zip"
    print("=" * 60)
    print(f"Evaluating fresh 300k model: {model_path}")
    print("=" * 60)
    
    dummy_env = UAVRoutingEnv(
        grid_size=15,
        obstacle_density=0.20,
        no_fly_density=0.0,
        fixed_grid=True
    )
    
    model = DoubleDQN.load(model_path, env=dummy_env)
    print("Model loaded successfully!")
    
    eval_env = UAVRoutingEnv(
        grid_size=15,
        obstacle_density=0.20,
        no_fly_density=0.0,
        fixed_grid=True,
        seed=42
    )
    
    N_EPISODES = 20
    goals, crashes, timeouts = 0, 0, 0
    
    for ep in range(1, N_EPISODES + 1):
        obs, info = eval_env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(int(action))
            total_reward += reward
            steps += 1
            done = terminated or truncated
            
        crashed = info.get("crashed", False)
        ag = np.round(obs["achieved_goal"]).astype(np.int32)
        dg = np.round(obs["desired_goal"]).astype(np.int32)
        
        if crashed:
            outcome = "CRASH"
            crashes += 1
        elif np.array_equal(ag[:2], dg[:2]):  # 4D goals comparison with ag[:2]
            outcome = "GOAL"
            goals += 1
        else:
            outcome = "TIMEOUT"
            timeouts += 1
            
        print(f"  Ep {ep:>2d} | {outcome:<7} | Steps: {steps:>4d} | Reward: {total_reward:>8.3f}")
        
    print()
    print("=" * 60)
    print(f"  GOAL:    {goals}/{N_EPISODES}")
    print(f"  CRASH:   {crashes}/{N_EPISODES}")
    print(f"  TIMEOUT: {timeouts}/{N_EPISODES}")
    print("=" * 60)
    
    eval_env.close()
    dummy_env.close()

if __name__ == "__main__":
    main()
