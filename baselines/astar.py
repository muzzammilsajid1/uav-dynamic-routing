"""A* shortest-path search for the shared grid-environment contract."""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Callable

from baselines.dijkstra import NeighborFunction, Node


@dataclass(frozen=True)
class AStarResult:
    path: list[Node]
    cost: float
    visited_count: int

    @property
    def found(self) -> bool:
        return bool(self.path)


def octile_distance(node: Node, goal: Node) -> float:
    """Admissible 8-connected-grid heuristic for this project's move costs."""
    row_delta = abs(node[0] - goal[0])
    col_delta = abs(node[1] - goal[1])
    diagonal_steps = min(row_delta, col_delta)
    straight_steps = max(row_delta, col_delta) - diagonal_steps
    return diagonal_steps * math.sqrt(2) + straight_steps


def astar(
    start: Node,
    goal: Node,
    get_neighbors: NeighborFunction,
    heuristic: Callable[[Node, Node], float] = octile_distance,
) -> AStarResult:
    """Find a minimum-cost path while using ``get_neighbors`` as the graph contract."""
    distances: dict[Node, float] = {start: 0.0}
    parents: dict[Node, Node | None] = {start: None}
    queue: list[tuple[float, float, Node]] = [(heuristic(start, goal), 0.0, start)]
    visited: set[Node] = set()

    while queue:
        _, current_cost, current = heapq.heappop(queue)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            return AStarResult(_reconstruct_path(parents, goal), current_cost, len(visited))

        for neighbor, edge_cost in get_neighbors(current):
            if edge_cost < 0:
                raise ValueError("A* requires non-negative edge weights")
            new_cost = current_cost + edge_cost
            if new_cost < distances.get(neighbor, float("inf")):
                distances[neighbor] = new_cost
                parents[neighbor] = current
                heapq.heappush(queue, (new_cost + heuristic(neighbor, goal), new_cost, neighbor))

    return AStarResult(path=[], cost=float("inf"), visited_count=len(visited))


def _reconstruct_path(parents: dict[Node, Node | None], goal: Node) -> list[Node]:
    path: list[Node] = []
    current: Node | None = goal
    while current is not None:
        path.append(current)
        current = parents[current]
    return list(reversed(path))
