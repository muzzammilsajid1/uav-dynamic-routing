from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.dijkstra import dijkstra
from envs.grid_environment import GridEnvironment
from evaluation.visualize import render_ascii


def main() -> None:
    env = GridEnvironment(size=15, obstacle_density=0.2, seed=42, diagonal=True)

    start_time = time.perf_counter()
    result = dijkstra(env.start, env.goal, env.get_neighbors)
    elapsed = time.perf_counter() - start_time

    print("Static Dijkstra baseline")
    print(f"Grid size: {env.size}x{env.size}")
    print(f"Obstacles: {len(env.blocked)}")
    print(f"Start: {env.start}")
    print(f"Goal: {env.goal}")
    print(f"Path found: {result.found}")
    print(f"Path cost: {result.cost:.3f}")
    print(f"Path nodes: {len(result.path)}")
    print(f"Visited nodes: {result.visited_count}")
    print(f"Runtime seconds: {elapsed:.6f}")
    print(f"Path: {result.path}")
    print("\nGrid legend: S=start, G=goal, #=obstacle, *=path")
    print(render_ascii(env, result.path))


if __name__ == "__main__":
    main()

