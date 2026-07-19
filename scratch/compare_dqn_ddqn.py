import sys
import os
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.her import HerReplayBuffer

sys.path.insert(0, os.path.abspath('rl_agent'))
from uav_env import UAVRoutingEnv
from double_dqn import DoubleDQN

def get_q_values(model_class, model_name):
    print(f"\nTraining {model_name}...")
    env = UAVRoutingEnv(
        grid_size=15, obstacle_density=0.20, no_fly_density=0.05, fixed_grid=True, seed=42
    )

    model = model_class(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=10_000,
        learning_starts=100,
        batch_size=256,
        tau=1.0,
        gamma=0.99,
        train_freq=2,
        gradient_steps=1,
        target_update_interval=500,
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
    
    eval_env = UAVRoutingEnv(
        grid_size=15, obstacle_density=0.20, no_fly_density=0.05, fixed_grid=True, seed=42
    )
    eval_env.reset()
    
    eval_env.unwrapped.uav_pos = np.array([7, 4])
    eval_env.unwrapped.goal_pos = np.array([7, 5])
    eval_env.unwrapped.last_action = 0
    
    obs = eval_env.unwrapped._build_observation()
    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    
    with torch.no_grad():
        q_values = model.q_net(obs_tensor)
    
    env.close()
    eval_env.close()
    
    return q_values.cpu().numpy()[0]

def main():
    q_dqn = get_q_values(DQN, "Vanilla DQN")
    q_ddqn = get_q_values(DoubleDQN, "DoubleDQN")
    
    action_names = ["N", "S", "W", "E", "NW", "NE", "SW", "SE"]
    
    print("\n" + "=" * 60)
    print("  Q-Values for UAV at [7, 4] with Goal at [7, 5]")
    print("=" * 60)
    
    print(f"{'Action':<10} | {'Vanilla DQN':<15} | {'DoubleDQN':<15}")
    print("-" * 60)
    for i, name in enumerate(action_names):
        print(f"{i} ({name:<2})    | {q_dqn[i]:>15.3f} | {q_ddqn[i]:>15.3f}")
        
    print("-" * 60)
    
    argmax_dqn = int(np.argmax(q_dqn))
    argmax_ddqn = int(np.argmax(q_ddqn))
    
    print(f"Argmax Vanilla DQN : {argmax_dqn} ({action_names[argmax_dqn]})")
    print(f"Argmax DoubleDQN   : {argmax_ddqn} ({action_names[argmax_ddqn]})")
    print()
    print(f"Vanilla DQN argmax == 3 (E)? : {argmax_dqn == 3}")
    print(f"DoubleDQN argmax == 3 (E)?   : {argmax_ddqn == 3}")
    print("=" * 60)

if __name__ == "__main__":
    main()
