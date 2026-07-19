import os
import sys
import csv
import time
import math
import random
import argparse
import numpy as np
from pathlib import Path

# Add project root to sys.path
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

def generate_test_pairs(grid_env: GridEnvironment, n_pairs: int, min_dist: float = 5.0, seed: int = 42) -> list[tuple[tuple[int, int], tuple[int, int], float]]:
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

def evaluate_dqn_vs_dijkstra(
    grid_env: GridEnvironment, 
    model: DQN, 
    pairs: list[tuple[tuple[int, int], tuple[int, int], float]],
    output_csv: str
):
    # Setup matching DQN environment
    eval_env = UAVRoutingEnv(
        grid_size=grid_env.size, 
        obstacle_density=grid_env.obstacle_density, 
        no_fly_density=0.0, 
        fixed_grid=True, 
        seed=grid_env.seed
    )
    eval_env.reset()
    
    # Inject exact obstacles
    eval_env.unwrapped.grid.fill(0)
    for (r, c) in grid_env.blocked:
        eval_env.unwrapped.grid[r, c] = 1 # CELL_OBSTACLE
        
    eval_env.unwrapped._distance_table = eval_env.unwrapped._build_distance_table()
    
    results = []
    
    print(f"Evaluating {len(pairs)} pairs. Writing results to {output_csv}...")
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['start', 'goal', 'dijkstra_cost', 'dqn_cost', 'dqn_success', 'gap'])
        
        for i, (s, g, dij_cost) in enumerate(pairs):
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

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                action_idx = int(action)
                move_cost = 1.0 if action_idx < 4 else math.sqrt(2)
                dqn_cost += move_cost
                
                obs, reward, terminated, truncated, info = eval_env.step(action_idx)
                done = terminated or truncated

            crashed = info.get("crashed", False)
            ag = np.round(obs["achieved_goal"]).astype(np.int32)
            dg = np.round(obs["desired_goal"]).astype(np.int32)

            success = not crashed and np.array_equal(ag[:2], dg[:2])
            gap = dqn_cost - dij_cost if success else float('inf')
            
            writer.writerow([f"{s[0]},{s[1]}", f"{g[0]},{g[1]}", f"{dij_cost:.4f}", f"{dqn_cost:.4f}" if success else "N/A", success, f"{gap:.4f}" if success else "N/A"])
            results.append((success, gap))

    eval_env.close()
    
    # Compute stats
    successes = [r for r in results if r[0]]
    success_rate = len(successes) / len(results) if results else 0
    gaps = [r[1] for r in successes]
    mean_gap = sum(gaps) / len(gaps) if gaps else 0
    max_gap = max(gaps) if gaps else 0
    
    print("\n" + "=" * 60)
    print("  Summary Statistics")
    print("=" * 60)
    print(f"  Total Pairs Evaluated:  {len(results)}")
    print(f"  Success Rate:           {success_rate*100:.1f}% ({len(successes)}/{len(results)})")
    if successes:
        print(f"  Mean Suboptimality Gap: +{mean_gap:.4f}")
        print(f"  Max Suboptimality Gap:  +{max_gap:.4f}")
    print("=" * 60)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DQN vs Dijkstra on identical grid.")
    parser.add_argument('--n_pairs', type=int, default=40, help='Number of pairs to evaluate')
    parser.add_argument('--min_dist', type=float, default=5.0, help='Minimum Dijkstra distance')
    parser.add_argument('--seed', type=int, default=42, help='Seed for grid generation')
    parser.add_argument('--out', type=str, default='logs/eval_dqn_vs_dijkstra.csv', help='Output CSV file path')
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    print(f"Generating base GridEnvironment (seed={args.seed}, size=15, obs=0.2)...")
    simra_env = GridEnvironment(size=15, obstacle_density=0.2, seed=args.seed, diagonal=True)
    
    print(f"Generating {args.n_pairs} valid start/goal pairs (dist >= {args.min_dist})...")
    pairs = generate_test_pairs(simra_env, n_pairs=args.n_pairs, min_dist=args.min_dist, seed=args.seed + 100)
    
    model_path = os.path.join(PROJECT_ROOT, 'models/dqn_her_300k_final.zip')
    print(f"Loading model from {model_path}...")
    dummy_env = UAVRoutingEnv(grid_size=15, obstacle_density=0.20, no_fly_density=0.0, fixed_grid=True)
    model = DoubleDQN.load(model_path, env=dummy_env)
    
    evaluate_dqn_vs_dijkstra(simra_env, model, pairs, args.out)
