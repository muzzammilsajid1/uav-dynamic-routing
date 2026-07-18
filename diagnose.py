import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import DQN
from rl_agent.uav_env import UAVRoutingEnv, CELL_OBSTACLE

def main():
    model_path = PROJECT_ROOT / "models" / "dqn_static_uav_lastaction_500k.zip"
    model = DQN.load(model_path)
    
    print("=== Shape Verification ===")
    print("Model expected observation space:", model.observation_space)
    print("Model network input shape (features_dim):", model.q_net.features_extractor.features_dim)
    
    env = UAVRoutingEnv(grid_size=15, obstacle_density=0.15, no_fly_density=0.05, curriculum_enabled=False)
    print("Environment actual observation space:", env.observation_space)
    
    print("\n=== Running 5 Fresh Episodes ===")
    for ep in range(5):
        obs, info = env.reset(seed=ep+2000)
        done = False
        step = 0
        total_rew = 0
        print(f"\n--- Episode {ep+1} --- (Start: {env.uav_pos}, Goal: {env.goal_pos})")
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action_idx = int(action)
            
            # Pre-calculate to know what specifically ended the episode
            curr_pos = env.uav_pos.copy()
            next_pos = curr_pos + env.ACTION_DELTAS[action_idx]
            
            obs, reward, terminated, truncated, info = env.step(action_idx)
            step += 1
            total_rew += reward
            
            outcome = ""
            if terminated:
                if not env._in_bounds(next_pos):
                    outcome = f"COLLISION (Out of bounds at {next_pos})"
                elif env.grid[next_pos[0], next_pos[1]] == CELL_OBSTACLE:
                    outcome = f"COLLISION (Hit obstacle at {next_pos})"
                elif np.array_equal(curr_pos, env.goal_pos) or np.array_equal(next_pos, env.goal_pos):
                    # goal logic updates uav_pos so we check next_pos
                    outcome = "SUCCESS (Goal Reached)"
                else:
                    outcome = "TERMINATED (Unknown)"
            elif truncated:
                outcome = "TIMEOUT"
                
            print(f"Step {step:3d} | Pos (after) {env.uav_pos} | Act {action_idx} | Rew {reward:7.2f} | {outcome}")
            done = terminated or truncated
            
        print(f"Episode {ep+1} Total Reward: {total_rew:.2f}")

if __name__ == '__main__':
    main()
