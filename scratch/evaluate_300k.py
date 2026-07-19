import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../rl_agent")))

import numpy as np
from gymnasium import spaces
from rl_agent.uav_env import UAVRoutingEnv
from double_dqn import DoubleDQN

class LegacyUAVEnv(UAVRoutingEnv):
    """
    Subclass of UAVRoutingEnv that matches the older 2D achieved_goal/desired_goal
    representation used by the downloaded 300k Colab model.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Override observation space to use (2,) shapes
        obs_dim = 4 + 49 + 8  # 61
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(
                low=-1.0,
                high=2.0,
                shape=(obs_dim,),
                dtype=np.float32,
            ),
            "achieved_goal": spaces.Box(
                low=0.0, high=float(self.grid_size - 1), shape=(2,), dtype=np.float32
            ),
            "desired_goal": spaces.Box(
                low=0.0, high=float(self.grid_size - 1), shape=(2,), dtype=np.float32
            ),
        })

    def _build_observation(self) -> dict:
        # Get standard dict
        obs = super()._build_observation()
        # Override to 2D goals
        obs["achieved_goal"] = self.uav_pos.astype(np.float32)
        obs["desired_goal"] = self.goal_pos.astype(np.float32)
        return obs

def main():
    model_path = r"C:\Users\Fast Computer\Downloads\dqn_her_300k_final.zip"
    print("=" * 60)
    print(f"Evaluating downloaded model: {model_path}")
    print("=" * 60)
    
    # Instantiate the env using the legacy 2D shape subclass
    dummy_env = LegacyUAVEnv(
        grid_size=15,
        obstacle_density=0.20,
        no_fly_density=0.05,
        fixed_grid=True
    )
    
    model = DoubleDQN.load(model_path, env=dummy_env)
    print("Model loaded successfully!")
    
    eval_env = LegacyUAVEnv(
        grid_size=15,
        obstacle_density=0.20,
        no_fly_density=0.05,
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
        elif np.array_equal(ag, dg):  # 2D goals comparison works perfectly with simple array_equal
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
