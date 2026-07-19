import sys
import os
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.her import HerReplayBuffer

sys.path.insert(0, os.path.abspath('rl_agent'))
from uav_env import UAVRoutingEnv
from double_dqn import DoubleDQN

class SnapshotCallback(BaseCallback):
    def __init__(self, eval_env, snapshots_dict, model_name):
        super().__init__(verbose=0)
        self.eval_env = eval_env
        self.snapshots_dict = snapshots_dict
        self.model_name = model_name
        self.states_to_eval = [
            ([7, 4], [7, 5]),  # Terminal 1
            ([7, 3], [7, 5]),  # Pre-Terminal 1 (Action W from [7,4])
            ([2, 2], [2, 3]),  # Terminal 2
            ([2, 2], [7, 7])   # Non-terminal
        ]
        self.snapshots_dict[self.model_name] = {
            f"UAV={u} Goal={g}": {} for u, g in self.states_to_eval
        }

    def _on_step(self) -> bool:
        if self.num_timesteps > 0 and self.num_timesteps % 10000 == 0:
            for u, g in self.states_to_eval:
                self.eval_env.reset()
                self.eval_env.unwrapped.uav_pos = np.array(u)
                self.eval_env.unwrapped.goal_pos = np.array(g)
                self.eval_env.unwrapped.last_action = 0
                
                obs = self.eval_env.unwrapped._build_observation()
                obs_tensor, _ = self.model.policy.obs_to_tensor(obs)
                
                with torch.no_grad():
                    q_values = self.model.q_net(obs_tensor)
                
                q_np = q_values.cpu().numpy()[0]
                key = f"UAV={u} Goal={g}"
                self.snapshots_dict[self.model_name][key][self.num_timesteps] = q_np
        return True

def train_and_snapshot(model_class, model_name, snapshots_dict):
    print(f"Training {model_name}...")
    env = UAVRoutingEnv(grid_size=15, obstacle_density=0.20, no_fly_density=0.05, fixed_grid=True, seed=42)
    eval_env = UAVRoutingEnv(grid_size=15, obstacle_density=0.20, no_fly_density=0.05, fixed_grid=True, seed=42)

    from rl_agent.safe_her_buffer import SafeHerReplayBuffer

    model = model_class(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=256,
        tau=1.0,
        gamma=0.99,
        train_freq=2,
        gradient_steps=1,
        target_update_interval=1_000,
        exploration_fraction=0.50,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[128, 128]),
        replay_buffer_class=SafeHerReplayBuffer,
        replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
        verbose=1,
        seed=42,
        device="auto",
    )

    cb = SnapshotCallback(eval_env, snapshots_dict, model_name)
    model.learn(total_timesteps=200_000, callback=cb, progress_bar=False)
    
    env.close()
    eval_env.close()

def main():
    snapshots = {}
    train_and_snapshot(DoubleDQN, "DoubleDQN (Normalized)", snapshots)
    
    action_names = ["N", "S", "W", "E", "NW", "NE", "SW", "SE"]
    steps = [i * 10000 for i in range(1, 21)]
    
    model_name = "DoubleDQN (Normalized)"
    print("\n" + "=" * 120)
    print(f"  {model_name} Multi-State Evaluation (200k)")
    print("=" * 120)
    
    for state_key, data in snapshots[model_name].items():
        print(f"\n--- State: {state_key} ---")
        
        # Header
        header = f"{'Action':<8}"
        for step in steps:
            header += f"| {step//1000}k     "
        print(header)
        print("-" * 100)
        
        # Values
        for i, name in enumerate(action_names):
            row = f"{i} ({name:<2})  "
            for step in steps:
                if step in data:
                    val = data[step][i]
                    row += f"| {val:>6.2f} "
                else:
                    row += f"| {'N/A':>6} "
            print(row)
            
        print("-" * 100)
        # Argmax
        argmax_row = f"{'Argmax':<8}"
        for step in steps:
            if step in data:
                best_action = int(np.argmax(data[step]))
                argmax_row += f"| {best_action:<6}"
            else:
                argmax_row += f"| {'N/A':>6}"
        print(argmax_row)

if __name__ == "__main__":
    main()
