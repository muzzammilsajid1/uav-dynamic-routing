import os
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import DQN
from rl_agent.uav_env import UAVRoutingEnv, CELL_FREE, CELL_NO_FLY, CELL_OBSTACLE

# Hyperparameters copied from train_static_diagnostic.py
HYPERPARAMS = dict(
    policy="MlpPolicy",
    learning_rate=1e-3,
    buffer_size=100_000,
    learning_starts=1_000,
    batch_size=128,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=1_000,
    exploration_fraction=0.20,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    policy_kwargs=dict(net_arch=[128, 128]),
    verbose=1,
)

TOTAL_TIMESTEPS = 75_000

class ToyDetourEnv(UAVRoutingEnv):
    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        
        # Override the grid to create a fixed detour scenario
        self.grid.fill(CELL_FREE)
        
        # Set fixed start and goal (opposite sides)
        self.uav_pos = np.array([1, 1], dtype=np.int32)
        self.goal_pos = np.array([7, 7], dtype=np.int32)
        
        # Build a large wall to block the direct path
        # A horizontal wall stretching across the middle
        self.grid[4, 1:8] = CELL_OBSTACLE
        
        self.current_step = 0
        # Recompute distance using the newly placed obstacles
        self.previous_distance = self._bfs_distance(self.uav_pos, self.goal_pos)
        
        obs = self._build_observation()
        info = self._build_info()
        return obs, info

def main():
    print("Initializing Toy Detour Environment...")
    env = ToyDetourEnv(
        grid_size=9,
        obstacle_density=0.0,
        no_fly_density=0.0,
        curriculum_enabled=False,
    )
    
    print(f"Training DQN for {TOTAL_TIMESTEPS} timesteps...")
    model = DQN(
        env=env,
        **HYPERPARAMS,
        seed=42,
        device="auto"
    )
    
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
    
    model_dir = PROJECT_ROOT / "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = model_dir / "toy_detour_test_weighted"
    model.save(str(model_path))
    print(f"\nModel saved to {model_path}.zip\n")
    
    print("Evaluating for 10 deterministic episodes on the fixed detour scenario...")
    eval_env = ToyDetourEnv(
        grid_size=9,
        obstacle_density=0.0,
        no_fly_density=0.0,
        curriculum_enabled=False,
    )
    
    for ep in range(1, 11):
        obs, info = eval_env.reset(seed=100+ep)
        done = False
        total_ep_reward = 0.0
        
        step_count = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action_idx = int(action)
            
            pos_before = eval_env.uav_pos.copy()
            dist_before = eval_env._bfs_distance(pos_before, eval_env.goal_pos)
            
            obs, reward, terminated, truncated, info = eval_env.step(action_idx)
            
            dist_after = eval_env._bfs_distance(eval_env.uav_pos, eval_env.goal_pos)
            
            step_count += 1
            if ep == 1:
                print(f"  Step {step_count:2d} | Pos {eval_env.uav_pos} | Action {action_idx} | DistBefore {dist_before:7.4f} | DistAfter {dist_after:7.4f} | Reward {reward:.4f}")
            
            total_ep_reward += reward
            done = terminated or truncated
            
        if total_ep_reward > 0:
            outcome = "SUCCESS"
        elif terminated:
            outcome = "COLLISION"
        else:
            outcome = "TIMEOUT"
            
        print(f"Episode {ep:>2} | Outcome: {outcome:<9} | Total Reward: {total_ep_reward:.4f}")

if __name__ == "__main__":
    main()
