"""
Timed re-run of the static 40-scenario DQN vs Dijkstra comparison.
Adds route-level compute-time measurement (matching the methodology
described in the paper: one warm-up predict() call discarded, cumulative
wall-clock time for dijkstra() vs accumulated .predict() calls).
"""
import os
import sys
import csv
import time
import math
import random
import argparse
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from envs.grid_environment import GridEnvironment
from baselines.dijkstra import dijkstra
from rl_agent.uav_env import UAVRoutingEnv
from stable_baselines3 import DQN
import rl_agent.safe_her_buffer as safe_her_buffer
sys.modules['safe_her_buffer'] = safe_her_buffer


class DoubleDQN(DQN):
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        pass


def generate_test_pairs(grid_env, n_pairs, min_dist=5.0, seed=42):
    random.seed(seed)
    valid_nodes = grid_env.nodes
    pairs = []
    attempts = 0
    while len(pairs) < n_pairs and attempts < 10000:
        attempts += 1
        s = random.choice(valid_nodes)
        g = random.choice(valid_nodes)
        if s == g:
            continue
        dij_result = dijkstra(s, g, grid_env.get_neighbors)
        if dij_result.found and dij_result.cost >= min_dist:
            pairs.append((s, g, dij_result.cost))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_pairs', type=int, default=40)
    parser.add_argument('--min_dist', type=float, default=5.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=str, default='logs/eval_dqn_vs_dijkstra_timed.csv')
    args = parser.parse_args()

    output_dir = os.path.dirname(args.out)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    simra_env = GridEnvironment(size=15, obstacle_density=0.2, seed=args.seed, diagonal=True)
    pairs = generate_test_pairs(simra_env, n_pairs=args.n_pairs, min_dist=args.min_dist, seed=args.seed + 100)

    model_path = str(PROJECT_ROOT / 'models' / 'dqn_her_300k_final.zip')
    dummy_env = UAVRoutingEnv(grid_size=15, obstacle_density=0.20, no_fly_density=0.0, fixed_grid=True)
    model = DoubleDQN.load(model_path, env=dummy_env)

    eval_env = UAVRoutingEnv(grid_size=simra_env.size, obstacle_density=simra_env.obstacle_density,
                              no_fly_density=0.0, fixed_grid=True, seed=simra_env.seed)
    eval_env.reset()
    eval_env.unwrapped.grid.fill(0)
    for (r, c) in simra_env.blocked:
        eval_env.unwrapped.grid[r, c] = 1
    eval_env.unwrapped._distance_table = eval_env.unwrapped._build_distance_table()

    # ---- warm-up predict() call, discarded from timing (framework init cost) ----
    warm_obs = eval_env.unwrapped._build_observation()
    _ = model.predict(warm_obs, deterministic=True)

    rows = []
    for (s, g, dij_cost_known) in pairs:
        # ---- Dijkstra: single timed graph search ----
        t0 = time.perf_counter()
        dij_result = dijkstra(s, g, simra_env.get_neighbors)
        dij_time_ms = (time.perf_counter() - t0) * 1000.0

        # ---- DQN: accumulated .predict() time over the episode ----
        start_pos = np.array([s[0], s[1]])
        goal_pos = np.array([g[0], g[1]])
        eval_env.unwrapped.uav_pos = start_pos.copy()
        eval_env.unwrapped.goal_pos = goal_pos.copy()
        eval_env.unwrapped.previous_distance = eval_env.unwrapped._bfs_distance(start_pos, goal_pos)
        eval_env.unwrapped.last_action = None
        eval_env.unwrapped.visited_cells.clear()
        eval_env.unwrapped.visited_cells.add(tuple(start_pos))
        eval_env.unwrapped.current_step = 0

        obs = eval_env.unwrapped._build_observation()
        done = False
        dqn_cost = 0.0
        dqn_time_ms = 0.0

        while not done:
            t0 = time.perf_counter()
            action, _ = model.predict(obs, deterministic=True)
            dqn_time_ms += (time.perf_counter() - t0) * 1000.0
            action_idx = int(action)
            move_cost = 1.0 if action_idx < 4 else math.sqrt(2)
            dqn_cost += move_cost
            obs, reward, terminated, truncated, info = eval_env.step(action_idx)
            done = terminated or truncated

        crashed = info.get("crashed", False)
        ag = np.round(obs["achieved_goal"]).astype(np.int32)
        dg = np.round(obs["desired_goal"]).astype(np.int32)
        success = not crashed and np.array_equal(ag[:2], dg[:2])

        rows.append({
            "start": f"{s[0]},{s[1]}", "goal": f"{g[0]},{g[1]}",
            "dijkstra_cost": dij_result.cost, "dijkstra_time_ms": dij_time_ms,
            "dqn_success": success, "dqn_cost": dqn_cost if success else None,
            "dqn_time_ms": dqn_time_ms,
        })

    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    dij_times = [r["dijkstra_time_ms"] for r in rows]
    dqn_times_success = [r["dqn_time_ms"] for r in rows if r["dqn_success"]]
    dqn_times_all = [r["dqn_time_ms"] for r in rows]

    print("=" * 60)
    print(f"Mean Dijkstra time (all {len(rows)}):      {np.mean(dij_times):.4f} ms")
    print(f"Mean DQN time (successes only, n={len(dqn_times_success)}): {np.mean(dqn_times_success):.4f} ms")
    print(f"Mean DQN time (all {len(dqn_times_all)} episodes):     {np.mean(dqn_times_all):.4f} ms")
    print(f"Ratio (DQN/Dijkstra, successes):  {np.mean(dqn_times_success)/np.mean(dij_times):.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
