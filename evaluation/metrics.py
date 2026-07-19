from __future__ import annotations

from envs.grid_environment import GridEnvironment, Node


def path_cost(env: GridEnvironment, path: list[Node]) -> float:
    if len(path) < 2:
        return 0.0

    total = 0.0
    for current, next_node in zip(path, path[1:]):
        neighbor_costs = dict(env.get_neighbors(current))
        if next_node not in neighbor_costs:
            return float("inf")
        total += neighbor_costs[next_node]

    return total


def success_rate(successes: int, total: int) -> float:
    if total == 0:
        return 0.0
    return successes / total

