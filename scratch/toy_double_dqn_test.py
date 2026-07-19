import sys
import os
sys.path.insert(0, os.path.abspath('rl_agent'))

import numpy as np
import torch
from stable_baselines3.her import HerReplayBuffer
from uav_env import UAVRoutingEnv
from double_dqn import DoubleDQN

def main():
    print("=" * 60)
    print("  Running DoubleDQN Toy Test (5,000 steps)")
    print("=" * 60)
    
    env = UAVRoutingEnv(
        grid_size=15, obstacle_density=0.20, no_fly_density=0.05, fixed_grid=True, seed=42
    )

    model = DoubleDQN(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=10_000,          # Reduced for memory
        learning_starts=100,         # Reduced to actually train
        batch_size=256,
        tau=1.0,
        gamma=0.99,
        train_freq=2,
        gradient_steps=1,
        target_update_interval=500,  # Scaled down
        exploration_fraction=0.50,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[128, 128]),
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
        verbose=0,
        seed=42,
        device="auto",
    )

    model.learn(total_timesteps=5_000, progress_bar=False)
    
    print("\nTraining complete. Evaluating Q-values at state UAV=[7,4], Goal=[7,5]\n")
    
    # Evaluate
    eval_env = UAVRoutingEnv(
        grid_size=15, obstacle_density=0.20, no_fly_density=0.05, fixed_grid=True, seed=42
    )
    eval_env.reset()
    
    eval_env.unwrapped.uav_pos = np.array([7, 4])
    eval_env.unwrapped.goal_pos = np.array([7, 5])
    eval_env.unwrapped.last_action = 0  # Assuming arrived via N, as per trace
    
    obs = eval_env.unwrapped._build_observation()
    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    
    with torch.no_grad():
        q_values = model.q_net(obs_tensor)
    
    q_values_np = q_values.cpu().numpy()[0]
    action_names = ["N", "S", "W", "E", "NW", "NE", "SW", "SE"]
    
    print("=" * 60)
    print(f"  DoubleDQN Q-Values for UAV at [7, 4] with Goal at [7, 5]")
    print("=" * 60)
    
    for i, (name, q_val) in enumerate(zip(action_names, q_values_np)):
        print(f"  Action {i} ({name:<2}) : {q_val:>8.3f}")
    
    best_action = int(np.argmax(q_values_np))
    print("-" * 60)
    print(f"  Greedy Choice : Action {best_action} ({action_names[best_action]})")
    print("=" * 60)

    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
