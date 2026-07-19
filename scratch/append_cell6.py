with open("train_her_colab_v2.py", "a", encoding="utf-8") as f:
    f.write("""
# %% ---------------------------------------------------------------------------
# CELL 6: Q-Value Inspection for specific state
# ------------------------------------------------------------------------------

import numpy as np
import torch
from stable_baselines3 import DQN
from uav_env import UAVRoutingEnv

# Load from local (or Drive path if session restarted)
MODEL_PATH = "/content/dqn_her_300k_final.zip"

dummy_env = UAVRoutingEnv(
    grid_size=15, obstacle_density=0.20, no_fly_density=0.05, fixed_grid=True
)
model = DQN.load(MODEL_PATH, env=dummy_env)

eval_env = UAVRoutingEnv(
    grid_size=15, obstacle_density=0.20, no_fly_density=0.05,
    fixed_grid=True, seed=42
)
eval_env.reset()

# Override positions
eval_env.unwrapped.uav_pos = np.array([7, 4])
eval_env.unwrapped.goal_pos = np.array([7, 5])

# Get the observation dict the network would see
obs = eval_env.unwrapped._build_observation()

# Convert to tensor using SB3's policy helper
obs_tensor, _ = model.policy.obs_to_tensor(obs)

# Pass through the Q-network
with torch.no_grad():
    q_values = model.q_net(obs_tensor)

q_values_np = q_values.cpu().numpy()[0]
action_names = ["N", "S", "W", "E", "NW", "NE", "SW", "SE"]

print("=" * 60)
print(f"  Q-Values for UAV at [7, 4] with Goal at [7, 5]")
print("=" * 60)

for i, (name, q_val) in enumerate(zip(action_names, q_values_np)):
    print(f"  Action {i} ({name:<2}) : {q_val:>8.3f}")

best_action = np.argmax(q_values_np)
print("-" * 60)
print(f"  Greedy Choice : Action {best_action} ({action_names[best_action]})")
print("=" * 60)

eval_env.close()
dummy_env.close()
""")
