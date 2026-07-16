from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable

Node = tuple[int, int]


@dataclass
class GridEnvironment:
    size: int = 15
    obstacle_density: float = 0.2
    seed: int | None = 42
    diagonal: bool = True
    start: Node = (0, 0)
    goal: Node | None = None
    blocked: set[Node] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.size <= 1:
            raise ValueError("size must be greater than 1")
        if not 0 <= self.obstacle_density < 1:
            raise ValueError("obstacle_density must be in [0, 1)")

        if self.goal is None:
            self.goal = (self.size - 1, self.size - 1)

        if not self.in_bounds(self.start):
            raise ValueError("start must be inside the grid")
        if not self.in_bounds(self.goal):
            raise ValueError("goal must be inside the grid")

        if not self.blocked:
            self.blocked = self._generate_obstacles()

        self.blocked.discard(self.start)
        self.blocked.discard(self.goal)

    def _generate_obstacles(self) -> set[Node]:
        rng = random.Random(self.seed)
        blocked: set[Node] = set()

        for row in range(self.size):
            for col in range(self.size):
                node = (row, col)
                if node in {self.start, self.goal}:
                    continue
                if rng.random() < self.obstacle_density:
                    blocked.add(node)

        return blocked

    @property
    def nodes(self) -> list[Node]:
        return [
            (row, col)
            for row in range(self.size)
            for col in range(self.size)
            if (row, col) not in self.blocked
        ]

    def in_bounds(self, node: Node) -> bool:
        row, col = node
        return 0 <= row < self.size and 0 <= col < self.size

    def is_blocked(self, node: Node) -> bool:
        return node in self.blocked

    def movement_offsets(self) -> Iterable[tuple[int, int, float]]:
        orthogonal = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
        ]
        diagonal = [
            (-1, -1, math.sqrt(2)),
            (-1, 1, math.sqrt(2)),
            (1, -1, math.sqrt(2)),
            (1, 1, math.sqrt(2)),
        ]
        return orthogonal + diagonal if self.diagonal else orthogonal

    def get_neighbors(self, node: Node) -> list[tuple[Node, float]]:
        if not self.in_bounds(node) or self.is_blocked(node):
            return []

        row, col = node
        neighbors: list[tuple[Node, float]] = []

        for d_row, d_col, cost in self.movement_offsets():
            neighbor = (row + d_row, col + d_col)
            if self.in_bounds(neighbor) and not self.is_blocked(neighbor):
                neighbors.append((neighbor, cost))

        return neighbors

    def to_networkx_graph(self):
        import networkx as nx

        graph = nx.Graph()
        for node in self.nodes:
            graph.add_node(node)
            for neighbor, cost in self.get_neighbors(node):
                graph.add_edge(node, neighbor, weight=cost)
        return graph

    def step_dynamics(self) -> None:
        return None

