from __future__ import annotations

import random

from baselines.dijkstra import dijkstra
from envs.grid_environment import DynamicObstacle, GridEnvironment, Node


def generate_week3_pairs(
    *,
    n_pairs: int = 50,
    size: int = 15,
    seed: int = 42,
    min_static_cost: float = 5.0,
    dynamic_obstacles: list[DynamicObstacle] | None = None,
) -> list[tuple[Node, Node]]:
    """Generate deterministic Week 3 start/goal pairs.

    Week 3 comparisons must use the same start/goal pairs on both sides.
    Dynamic-obstacle cells are excluded as endpoints, and obstacle_density
    is fixed at 0.0 so only the shared toggling cells can affect the route.
    """
    rng = random.Random(seed)
    dyn_cells = {obs.cell for obs in dynamic_obstacles or []}
    candidates = [
        (row, col)
        for row in range(size)
        for col in range(size)
        if (row, col) not in dyn_cells
    ]

    pairs: list[tuple[Node, Node]] = []
    seen: set[tuple[Node, Node]] = set()
    env = GridEnvironment(
        size=size,
        obstacle_density=0.0,
        dynamic_obstacles=dynamic_obstacles or [],
    )

    attempts = 0
    max_attempts = max(10000, n_pairs * 500)
    while len(pairs) < n_pairs and attempts < max_attempts:
        attempts += 1
        start = rng.choice(candidates)
        goal = rng.choice(candidates)
        if start == goal or (start, goal) in seen:
            continue

        env.start = start
        env.goal = goal
        result = dijkstra(start, goal, env.get_neighbors)
        if result.found and result.cost >= min_static_cost:
            pairs.append((start, goal))
            seen.add((start, goal))

    if len(pairs) < n_pairs:
        raise RuntimeError(f"Could only generate {len(pairs)} valid pairs out of {n_pairs}.")

    return pairs
