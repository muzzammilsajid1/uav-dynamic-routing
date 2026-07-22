"""
Run the trained RL model on each of Simra's 50 Week 3 scenarios and output
a CSV with matching scenario_id for row-by-row comparison.
"""
from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import DQN

from envs.grid_environment import default_dynamic_obstacles
import rl_agent.safe_her_buffer as safe_her_buffer_module
from rl_agent.uav_env import UAVRoutingEnv

sys.modules["safe_her_buffer"] = safe_her_buffer_module

# ---- Double DQN (must match training) ----------------------------------------
class DoubleDQN(DQN):
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma
            with torch.no_grad():
                next_q_values_online = self.q_net(replay_data.next_observations)
                next_actions = next_q_values_online.argmax(dim=1, keepdim=True)
                next_q_values_target = self.q_net_target(replay_data.next_observations)
                next_q_values = torch.gather(next_q_values_target, dim=1, index=next_actions)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values
            current_q_values = self.q_net(replay_data.observations)
            current_q_values = torch.gather(current_q_values, dim=1, index=replay_data.actions.long())
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())
            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))


MODEL_PATH = str(PROJECT_ROOT / "models" / "dqn_her_phase3_100k.zip")
BASELINE_CSV = str(PROJECT_ROOT / "evaluation" / "week3_dynamic_baseline_results.csv")
OUTPUT_CSV = str(PROJECT_ROOT / "evaluation" / "week3_rl_results.csv")


def parse_coord(s: str) -> tuple[int, int]:
    """Parse 'row,col' string into (row, col) tuple."""
    parts = s.strip().split(",")
    return int(parts[0]), int(parts[1])


def main() -> None:
    # ---- Load model -----------------------------------------------------------
    dummy_env = UAVRoutingEnv(
        grid_size=15, obstacle_density=0.0, no_fly_density=0.0,
        dynamic_obstacles_enabled=True,
        dynamic_obstacles=default_dynamic_obstacles(),
        fixed_grid=True,
    )
    model = DoubleDQN.load(MODEL_PATH, env=dummy_env)
    dummy_env.close()

    # ---- Load scenarios -------------------------------------------------------
    scenarios = []
    with open(BASELINE_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scenarios.append({
                "scenario_id": int(row["scenario_id"]),
                "start": parse_coord(row["start"]),
                "goal": parse_coord(row["goal"]),
            })

    # ---- Evaluate each scenario -----------------------------------------------
    eval_env = UAVRoutingEnv(
        grid_size=15, obstacle_density=0.0, no_fly_density=0.0,
        dynamic_obstacles_enabled=True,
        dynamic_obstacles=default_dynamic_obstacles(),
        fixed_grid=True, seed=42,
    )
    eval_env.reset()  # initialise grid

    results = []
    for sc in scenarios:
        start = sc["start"]
        goal = sc["goal"]

        # Reset episode state without re-randomising positions
        eval_env.reset()
        eval_env.uav_pos = np.array(start, dtype=np.int32)
        eval_env.goal_pos = np.array(goal, dtype=np.int32)
        eval_env.current_step = 0
        eval_env._elapsed_steps = 0
        eval_env.previous_distance = eval_env._bfs_distance(eval_env.uav_pos, eval_env.goal_pos)
        eval_env.last_action = None
        eval_env.visited_cells.clear()
        eval_env.visited_cells.add(tuple(start))

        obs = eval_env._build_observation()
        done = False
        path_cost = 0.0
        steps = 0
        t0 = time.perf_counter()

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action_idx = int(action)

            # Compute move cost (matches Dijkstra's edge weights)
            delta = eval_env.ACTION_DELTAS[action_idx]
            move_cost = 1.0 if delta[0] == 0 or delta[1] == 0 else math.sqrt(2)

            obs, reward, terminated, truncated, info = eval_env.step(action_idx)
            steps += 1

            if not info.get("crashed", False):
                path_cost += move_cost

            done = terminated or truncated

        elapsed_ms = (time.perf_counter() - t0) * 1000
        success = info.get("is_success", False)

        results.append({
            "scenario_id": sc["scenario_id"],
            "start": f"{start[0]},{start[1]}",
            "goal": f"{goal[0]},{goal[1]}",
            "success": success,
            "steps_taken": steps,
            "rl_path_cost": f"{path_cost:.6f}" if success else "N/A",
            "compute_time_ms": f"{elapsed_ms:.3f}",
        })

    eval_env.close()

    # ---- Write output CSV -----------------------------------------------------
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # ---- Summary --------------------------------------------------------------
    n_success = sum(1 for r in results if r["success"])
    print(f"Model:          {MODEL_PATH}")
    print(f"Scenarios:      {len(results)}")
    print(f"Success rate:   {n_success}/{len(results)} ({100*n_success/len(results):.0f}%)")
    print(f"Output CSV:     {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
