"""
speed_test_her.py -- 5,000-step speed test for HER + distance table.
Run from the project root: python speed_test_her.py
"""
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rl_agent"))

from stable_baselines3 import DQN
from stable_baselines3.her import HerReplayBuffer
from uav_env import UAVRoutingEnv

TOTAL_STEPS = 5_000

env = UAVRoutingEnv(
    grid_size=15,
    obstacle_density=0.20,
    no_fly_density=0.05,
    fixed_grid=True,
    seed=42,
)

model = DQN(
    policy="MultiInputPolicy",
    env=env,
    learning_rate=1e-3,
    buffer_size=100_000,
    learning_starts=1_000,   # lower so learning starts within 5k steps
    batch_size=256,
    tau=1.0,
    gamma=0.99,
    train_freq=2,
    target_update_interval=1_000,
    exploration_fraction=0.50,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    policy_kwargs=dict(net_arch=[128, 128]),
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
    verbose=0,
    seed=42,
)

print(f"Running {TOTAL_STEPS:,}-step speed test...")
t0 = time.perf_counter()
model.learn(total_timesteps=TOTAL_STEPS, progress_bar=False)
elapsed = time.perf_counter() - t0

fps = TOTAL_STEPS / elapsed
print(f"\nCompleted {TOTAL_STEPS:,} steps in {elapsed:.1f}s")
print(f"Speed: {fps:.1f} fps")

projected_300k_hours = (300_000 / fps) / 3600
print(f"Projected time for 300k steps: {projected_300k_hours:.2f} hours "
      f"({projected_300k_hours * 60:.0f} minutes)")

env.close()
