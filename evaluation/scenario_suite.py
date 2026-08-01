"""Shared schema and constructors for persisted benchmark scenarios."""
from __future__ import annotations

import json
from pathlib import Path

from envs.grid_environment import (
    DynamicObstacle,
    GridEnvironment,
    MovingObstacle,
    StochasticObstacle,
)


def load_suite(path: Path) -> dict:
    suite = json.loads(path.read_text(encoding="utf-8"))
    if suite.get("schema_version") != 2:
        raise ValueError("Unsupported benchmark-suite schema")
    ids = [scenario["scenario_id"] for scenario in suite["scenarios"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Scenario IDs must be unique")
    return suite


def dynamic_obstacles(scenario: dict) -> list[DynamicObstacle]:
    return [
        DynamicObstacle(
            cell=tuple(item["cell"]),
            period=int(item["period"]),
            initial_state=str(item["initial_state"]),
        )
        for item in scenario["dynamic_obstacles"]
    ]


def stochastic_obstacles(scenario: dict) -> list[StochasticObstacle]:
    return [
        StochasticObstacle(
            cell=tuple(item["cell"]),
            toggle_probability=float(item["toggle_probability"]),
            initial_state=str(item["initial_state"]),
        )
        for item in scenario.get("stochastic_obstacles", [])
    ]


def moving_obstacles(scenario: dict) -> list[MovingObstacle]:
    return [
        MovingObstacle(
            path=[tuple(cell) for cell in item["path"]],
            period=int(item["period"]),
            initial_index=int(item.get("initial_index", 0)),
        )
        for item in scenario.get("moving_obstacles", [])
    ]


def traversal_penalties(scenario: dict) -> dict[tuple[int, int], float]:
    penalties = {
        tuple(item["cell"]): float(item["penalty"])
        for item in scenario.get("traversal_penalties", [])
    }
    if scenario.get("no_fly_mode") == "penalty":
        penalty = float(scenario.get("no_fly_penalty", 5.0))
        for cell in scenario.get("no_fly_cells", []):
            penalties[tuple(cell)] = penalties.get(tuple(cell), 0.0) + penalty
    return penalties


def grid_environment(scenario: dict) -> GridEnvironment:
    blocked = {tuple(cell) for cell in scenario["blocked"]}
    if scenario.get("no_fly_mode") == "hard":
        blocked.update(tuple(cell) for cell in scenario.get("no_fly_cells", []))
    return GridEnvironment(
        size=int(scenario["grid_size"]),
        obstacle_density=0.0,
        diagonal=True,
        start=tuple(scenario["start"]),
        goal=tuple(scenario["goal"]),
        blocked=blocked,
        dynamic_obstacles=dynamic_obstacles(scenario),
        stochastic_obstacles=stochastic_obstacles(scenario),
        moving_obstacles=moving_obstacles(scenario),
        traversal_penalties=traversal_penalties(scenario),
        dynamics_seed=scenario.get("dynamics_seed"),
    )
