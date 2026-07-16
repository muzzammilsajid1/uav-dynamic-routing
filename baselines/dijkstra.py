from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Callable

Node = tuple[int, int]
NeighborFunction = Callable[[Node], list[tuple[Node, float]]]


@dataclass(frozen=True)
class DijkstraResult:
    path: list[Node]
    cost: float
    visited_count: int

    @property
    def found(self) -> bool:
        return bool(self.path)


def dijkstra(start: Node, goal: Node, get_neighbors: NeighborFunction) -> DijkstraResult:
    distances: dict[Node, float] = {start: 0.0}
    parents: dict[Node, Node | None] = {start: None}
    queue: list[tuple[float, Node]] = [(0.0, start)]
    visited: set[Node] = set()

    while queue:
        current_cost, current = heapq.heappop(queue)

        if current in visited:
            continue

        visited.add(current)

        if current == goal:
            return DijkstraResult(
                path=_reconstruct_path(parents, goal),
                cost=current_cost,
                visited_count=len(visited),
            )

        for neighbor, edge_cost in get_neighbors(current):
            if edge_cost < 0:
                raise ValueError("Dijkstra requires non-negative edge weights")

            new_cost = current_cost + edge_cost
            if new_cost < distances.get(neighbor, float("inf")):
                distances[neighbor] = new_cost
                parents[neighbor] = current
                heapq.heappush(queue, (new_cost, neighbor))

    return DijkstraResult(path=[], cost=float("inf"), visited_count=len(visited))


def _reconstruct_path(parents: dict[Node, Node | None], goal: Node) -> list[Node]:
    path: list[Node] = []
    current: Node | None = goal

    while current is not None:
        path.append(current)
        current = parents[current]

    path.reverse()
    return path

