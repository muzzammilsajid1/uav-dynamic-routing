from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.dijkstra import dijkstra
from envs.grid_environment import GridEnvironment
from evaluation.metrics import path_cost
from evaluation.visualize import render_ascii
from rl_agent.q_learning import GridRoutingEnv, evaluate_agent, train_q_learning


def main() -> None:
    grid = GridEnvironment(size=15, obstacle_density=0.2, seed=42, diagonal=True)
    rl_env = GridRoutingEnv(grid=grid, max_steps=100)

    dijkstra_result = dijkstra(grid.start, grid.goal, grid.get_neighbors)
    agent, history = train_q_learning(rl_env, episodes=5000, seed=42)
    results = evaluate_agent(rl_env, agent, episodes=20)

    train_successes = sum(result.success for result in history[-100:])
    eval_successes = sum(result.success for result in results)
    eval_crashes = sum(result.crashed for result in results)
    eval_timeouts = sum(result.timed_out for result in results)
    successful_paths = [result for result in results if result.success]
    best_result = min(successful_paths, key=lambda result: result.steps) if successful_paths else results[0]
    best_path_cost = path_cost(grid, best_result.path)

    print("Static Q-learning run")
    print(f"Grid size: {grid.size}x{grid.size}")
    print(f"Obstacles: {len(grid.blocked)}")
    print(f"Start: {grid.start}")
    print(f"Goal: {grid.goal}")
    print(f"Dijkstra path cost: {dijkstra_result.cost:.3f}")
    print(f"Dijkstra path nodes: {len(dijkstra_result.path)}")
    print(f"Training success in last 100 episodes: {train_successes}/100")
    print(f"Evaluation success: {eval_successes}/{len(results)}")
    print(f"Evaluation crashes: {eval_crashes}/{len(results)}")
    print(f"Evaluation timeouts: {eval_timeouts}/{len(results)}")
    print(f"Best evaluation path steps: {best_result.steps}")
    print(f"Best evaluation path cost: {best_path_cost:.3f}")
    print(f"Cost above Dijkstra: {best_path_cost - dijkstra_result.cost:.3f}")
    print(f"Best evaluation total reward: {best_result.total_reward:.3f}")
    print(f"Best evaluation path: {best_result.path}")
    print("\nGrid legend: S=start, G=goal, #=obstacle, *=RL path")
    print(render_ascii(grid, best_result.path))


if __name__ == "__main__":
    main()
