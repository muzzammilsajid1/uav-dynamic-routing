"""
verify_checkpoint_name.py
Runs a short training run with CheckpointCallback (save_freq=500)
and prints the exact filenames it produces, so we can confirm whether
DriveCheckpointCallback is looking for the right name.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rl_agent"))

from stable_baselines3 import DQN
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback
from uav_env import UAVRoutingEnv

CKPT_DIR = "verify_ckpt_test"
os.makedirs(CKPT_DIR, exist_ok=True)

env = UAVRoutingEnv(grid_size=15, obstacle_density=0.20, no_fly_density=0.05,
                   fixed_grid=True, seed=42)

model = DQN(
    policy="MultiInputPolicy",
    env=env,
    learning_rate=1e-3,
    buffer_size=10_000,
    learning_starts=100,
    batch_size=64,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
    verbose=0,
    seed=42,
)

checkpoint_cb = CheckpointCallback(
    save_freq=500,
    save_path=CKPT_DIR,
    name_prefix="rl_model",
    verbose=1,
)

model.learn(total_timesteps=1_500, callback=checkpoint_cb)
env.close()

print("\nActual files written by CheckpointCallback:")
for fname in sorted(os.listdir(CKPT_DIR)):
    print(f"  {fname}")

# Also confirm what DriveCheckpointCallback would look for
print()
print("DriveCheckpointCallback looks for: rl_model_{n}_steps.zip")
print("e.g. at save_freq=500:  rl_model_500_steps.zip, rl_model_1000_steps.zip, rl_model_1500_steps.zip")

# Cleanup
import shutil
shutil.rmtree(CKPT_DIR)
