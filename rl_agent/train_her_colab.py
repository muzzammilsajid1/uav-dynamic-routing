import os
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.her import HerReplayBuffer
from uav_env import UAVRoutingEnv

def main():
    # ---------------------------------------------------------
    # 1. Environment Setup
    # ---------------------------------------------------------
    env = UAVRoutingEnv(
        grid_size=15,
        obstacle_density=0.20,
        no_fly_density=0.05,
        fixed_grid=True,
        seed=42
    )

    # ---------------------------------------------------------
    # 2. Model Configuration
    # ---------------------------------------------------------
    model = DQN(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=100_000,
        learning_starts=5000,
        batch_size=256,
        tau=1.0,
        gamma=0.99,
        train_freq=2,
        target_update_interval=1000,
        exploration_fraction=0.20,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[256, 256]),
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        ),
        verbose=1,
        seed=42,
    )

    # ---------------------------------------------------------
    # 3. Training
    # ---------------------------------------------------------
    print("\nStarting Training...")
    model.learn(total_timesteps=300_000, progress_bar=True)

    # ---------------------------------------------------------
    # 4. Save Model
    # ---------------------------------------------------------
    save_dir = "/content/models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "dqn_her_300k.zip")
    model.save(save_path)
    print(f"\nModel saved to: {save_path}")

    # ---------------------------------------------------------
    # 5. Evaluation
    # ---------------------------------------------------------
    # We use a fixed seed for deterministic evaluation on the same layout
    eval_env = UAVRoutingEnv(
        grid_size=15,
        obstacle_density=0.20,
        no_fly_density=0.05,
        fixed_grid=True,
        seed=42
    )

    print("\n" + "=" * 50)
    print("  Evaluation (20 Episodes)")
    print("=" * 50)

    n_episodes = 20
    goals = 0
    crashes = 0
    timeouts = 0

    for ep in range(1, n_episodes + 1):
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

        if terminated and reward > 500:
            outcome = "GOAL"
            goals += 1
        elif terminated:
            outcome = "CRASH"
            crashes += 1
        else:
            outcome = "TIMEOUT"
            timeouts += 1

        print(f"Episode {ep:>2d} | Outcome: {outcome:<7} | Steps: {steps:>4d} | Reward: {total_reward:>8.2f}")

    print("\n" + "=" * 50)
    print("  Final Summary")
    print("=" * 50)
    print(f"{goals} GOAL")
    print(f"{crashes} CRASH")
    print(f"{timeouts} TIMEOUT")

    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
