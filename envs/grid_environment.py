from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable

Node = tuple[int, int]


@dataclass
class DynamicObstacle:
    """A single pre-positioned cell that toggles blocked/passable on a
    fixed period. Deliberately NOT randomly generated: both the Dijkstra
    replanner and the RL env must read the exact same list of these, so
    hand-picked positions (shared via one config) are the source of truth
    instead of two independently-seeded RNGs that can silently drift.
    """

    cell: Node
    period: int
    initial_state: str = "blocked"  # "blocked" or "passable"

    def __post_init__(self) -> None:
        if self.period <= 0:
            raise ValueError("period must be a positive integer")
        if self.initial_state not in ("blocked", "passable"):
            raise ValueError("initial_state must be 'blocked' or 'passable'")


@dataclass
class StochasticObstacle:
    """A cell that toggles independently with a fixed per-step probability."""

    cell: Node
    toggle_probability: float
    initial_state: str = "passable"

    def __post_init__(self) -> None:
        if not 0.0 <= self.toggle_probability <= 1.0:
            raise ValueError("toggle_probability must be in [0, 1]")
        if self.initial_state not in ("blocked", "passable"):
            raise ValueError("initial_state must be 'blocked' or 'passable'")


@dataclass
class MovingObstacle:
    """An obstacle following a known cyclic path at a fixed step period."""

    path: list[Node]
    period: int = 1
    initial_index: int = 0

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("moving-obstacle path must not be empty")
        if self.period <= 0:
            raise ValueError("period must be a positive integer")
        if not 0 <= self.initial_index < len(self.path):
            raise ValueError("initial_index must reference the path")


def default_dynamic_obstacles() -> list[DynamicObstacle]:
    """Shared, hand-picked dynamic obstacle set for the default 15x15,
    seed=42 environment (start (0,0), goal (14,14)).

    Positions are chosen ON the static Dijkstra shortest path (cost
    20.385, via (0,0)->(1,1)->...->(8,8)->(9,8)->...->(14,14)) so toggles
    actually force a detour instead of flipping a cell nobody's route
    ever touches. This is the single source of truth both the Dijkstra
    replanner and the RL env should import, so obstacle placement can't
    silently drift between the two sides again.
    """
    return [
        DynamicObstacle(cell=(4, 4), period=5, initial_state="passable"),
        DynamicObstacle(cell=(8, 8), period=5, initial_state="blocked"),
        DynamicObstacle(cell=(12, 11), period=5, initial_state="passable"),
    ]


@dataclass
class GridEnvironment:
    size: int = 15
    obstacle_density: float = 0.2
    seed: int | None = 42
    diagonal: bool = True
    start: Node = (0, 0)
    goal: Node | None = None
    blocked: set[Node] = field(default_factory=set)
    dynamic_obstacles: list[DynamicObstacle] = field(default_factory=list)
    stochastic_obstacles: list[StochasticObstacle] = field(default_factory=list)
    moving_obstacles: list[MovingObstacle] = field(default_factory=list)
    traversal_penalties: dict[Node, float] = field(default_factory=dict)
    dynamics_seed: int | None = None
    _elapsed_steps: int = field(default=0, init=False, repr=False)
    _dynamics_rng: random.Random = field(init=False, repr=False)
    _moving_indices: list[int] = field(default_factory=list, init=False, repr=False)

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

        for node, penalty in self.traversal_penalties.items():
            if not self.in_bounds(node):
                raise ValueError(f"traversal-penalty cell {node} is outside the grid")
            if penalty < 0:
                raise ValueError("traversal penalties must be non-negative")

        self._dynamics_rng = random.Random(
            self.seed if self.dynamics_seed is None else self.dynamics_seed
        )
        self._moving_indices = [
            obstacle.initial_index for obstacle in self.moving_obstacles
        ]
        self._apply_dynamic_obstacle_initial_states()
        self._apply_stochastic_obstacle_initial_states()
        self._apply_moving_obstacle_initial_states()

    def _apply_dynamic_obstacle_initial_states(self) -> None:
        for obstacle in self.dynamic_obstacles:
            if not self.in_bounds(obstacle.cell):
                raise ValueError(f"dynamic obstacle {obstacle.cell} is outside the grid")
            if obstacle.cell in (self.start, self.goal):
                raise ValueError(
                    f"dynamic obstacle {obstacle.cell} cannot be the start or goal cell"
                )
            if obstacle.initial_state == "blocked":
                self.blocked.add(obstacle.cell)
            else:
                self.blocked.discard(obstacle.cell)

    def _validate_dynamic_cell(self, cell: Node, label: str) -> None:
        if not self.in_bounds(cell):
            raise ValueError(f"{label} {cell} is outside the grid")
        if cell in (self.start, self.goal):
            raise ValueError(f"{label} {cell} cannot be the start or goal cell")

    def _apply_stochastic_obstacle_initial_states(self) -> None:
        for obstacle in self.stochastic_obstacles:
            self._validate_dynamic_cell(obstacle.cell, "stochastic obstacle")
            if obstacle.initial_state == "blocked":
                self.blocked.add(obstacle.cell)
            else:
                self.blocked.discard(obstacle.cell)

    def _apply_moving_obstacle_initial_states(self) -> None:
        self._moving_indices = [
            obstacle.initial_index for obstacle in self.moving_obstacles
        ]
        for obstacle, index in zip(self.moving_obstacles, self._moving_indices):
            for cell in obstacle.path:
                self._validate_dynamic_cell(cell, "moving obstacle")
            self.blocked.add(obstacle.path[index])

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
        if not self.in_bounds(node):
            return []

        row, col = node
        neighbors: list[tuple[Node, float]] = []

        for d_row, d_col, cost in self.movement_offsets():
            neighbor = (row + d_row, col + d_col)
            if self.in_bounds(neighbor) and not self.is_blocked(neighbor):
                neighbors.append(
                    (neighbor, cost + self.traversal_penalties.get(neighbor, 0.0))
                )

        return neighbors

    def to_networkx_graph(self):
        import networkx as nx

        graph = nx.Graph()
        for node in self.nodes:
            graph.add_node(node)
            for neighbor, cost in self.get_neighbors(node):
                graph.add_edge(node, neighbor, weight=cost)
        return graph

    def step_dynamics(self) -> set[Node]:
        """Advance dynamic obstacles by one timestep.

        No advance warning: this only reports cells that changed on
        *this* step. Neither the Dijkstra replanner nor the RL agent get
        any information about future toggles ahead of time (agreed with
        Muzzammil, Week 3 kickoff) — both sides only ever see the grid's
        current state via get_neighbors() / is_blocked(), reacting after
        a change lands rather than anticipating it.

        Returns the set of cells that toggled this step (empty if none
        did), so a caller like the naive replanner can cheaply check
        "did anything change?" without diffing the whole grid every step.
        """
        self._elapsed_steps += 1
        changed: set[Node] = set()

        for obstacle in self.dynamic_obstacles:
            if self._elapsed_steps % obstacle.period == 0:
                if obstacle.cell in self.blocked:
                    self.blocked.discard(obstacle.cell)
                else:
                    self.blocked.add(obstacle.cell)
                changed.add(obstacle.cell)

        for obstacle in self.stochastic_obstacles:
            if self._dynamics_rng.random() < obstacle.toggle_probability:
                if obstacle.cell in self.blocked:
                    self.blocked.discard(obstacle.cell)
                else:
                    self.blocked.add(obstacle.cell)
                changed.add(obstacle.cell)

        for index, obstacle in enumerate(self.moving_obstacles):
            if self._elapsed_steps % obstacle.period != 0:
                continue
            old_index = self._moving_indices[index]
            new_index = (old_index + 1) % len(obstacle.path)
            old_cell = obstacle.path[old_index]
            new_cell = obstacle.path[new_index]
            self.blocked.discard(old_cell)
            self.blocked.add(new_cell)
            self._moving_indices[index] = new_index
            changed.update((old_cell, new_cell))

        return changed

    def reset_dynamics(self) -> None:
        """Reset the toggle clock and restore every dynamic obstacle to
        its initial_state. Use this between episodes/runs so repeated
        evaluation scenarios start from the same known state."""
        self._elapsed_steps = 0
        self._dynamics_rng = random.Random(
            self.seed if self.dynamics_seed is None else self.dynamics_seed
        )
        self._apply_dynamic_obstacle_initial_states()
        self._apply_stochastic_obstacle_initial_states()
        for obstacle in self.moving_obstacles:
            for cell in obstacle.path:
                self.blocked.discard(cell)
        self._apply_moving_obstacle_initial_states()

    @property
    def elapsed_steps(self) -> int:
        return self._elapsed_steps

