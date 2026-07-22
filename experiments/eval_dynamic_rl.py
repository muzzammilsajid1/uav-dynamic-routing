"""Week 3 RL evaluation: run UAVRoutingEnv with dynamic_obstacles_enabled=True
on an obstacle_density=0.0 grid (only the three shared DynamicObstacle cells
differ from empty), so the comparison against run_dynamic_baseline.py is
apples-to-apples — no static obstacle divergence between the two sides.

This is an evaluation-only script. Training obstacle_density is unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from envs.grid_environment import default_dynamic_obstacles
from rl_agent.uav_env import UAVRoutingEnv

N_EPISODES = 20


def main() -> None:
    obs_list = default_dynamic_obstacles()
    dyn_cells = [obs.cell for obs in obs_list]

    env = UAVRoutingEnv(
        grid_size=15,
        obstacle_density=0.0,   # matches run_dynamic_baseline.py — no static obstacles
        dynamic_obstacles_enabled=True,
        dynamic_obstacles=obs_list,
        fixed_grid=True,
        seed=42,
    )

    print("Week 3 RL evaluation (UAVRoutingEnv, dynamic_obstacles_enabled=True)")
    print(f"Grid: 15x15, obstacle_density=0.0")
    print(f"Dynamic obstacle cells: {dyn_cells}")
    print(f"Episodes: {N_EPISODES}")
    print()

    successes = 0
    crashes = 0
    timeouts = 0
    total_steps = 0
    total_reward = 0.0

    for ep in range(N_EPISODES):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0

        while not done:
            # Random policy — placeholder until a trained dynamic model exists
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps += 1
            done = terminated or truncated

        total_steps += ep_steps
        total_reward += ep_reward

        if info.get("is_success"):
            successes += 1
        elif info.get("crashed"):
            crashes += 1
        else:
            timeouts += 1

    env.close()

    print(f"Success:  {successes}/{N_EPISODES}")
    print(f"Crashes:  {crashes}/{N_EPISODES}")
    print(f"Timeouts: {timeouts}/{N_EPISODES}")
    print(f"Avg steps/episode:  {total_steps / N_EPISODES:.1f}")
    print(f"Avg reward/episode: {total_reward / N_EPISODES:.3f}")
    print()
    print("Note: random policy used (no trained dynamic model yet).")
    print("Grid confirmed: 15x15, obstacle_density=0.0 + 3 dynamic cells only.")


if __name__ == "__main__":
    main()
