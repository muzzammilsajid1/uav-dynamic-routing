"""Build V2 environments from persisted or diagnostic scenarios."""
from __future__ import annotations

import numpy as np

from evaluation.scenario_suite import (
    dynamic_obstacles,
    moving_obstacles,
    stochastic_obstacles,
    traversal_penalties,
)
from rl_agent.uav_env import CELL_FREE, CELL_NO_FLY, CELL_OBSTACLE, UAVRoutingEnv


def env_from_scenario(
    scenario: dict,
    *,
    potential_shaping_enabled: bool = True,
    observation_mode: str = "local",
    dynamics_timing: str = "post_move_observed",
    max_steps: int | None = None,
) -> tuple[UAVRoutingEnv, dict]:
    obstacles = dynamic_obstacles(scenario)
    stochastic = stochastic_obstacles(scenario)
    moving = moving_obstacles(scenario)
    penalties = traversal_penalties(scenario)
    env = UAVRoutingEnv(
        grid_size=int(scenario["grid_size"]),
        obstacle_density=0.0,
        no_fly_density=0.0,
        fixed_grid=True,
        dynamic_obstacles_enabled=bool(obstacles),
        dynamic_obstacles=obstacles,
        stochastic_obstacles=stochastic,
        moving_obstacles=moving,
        dynamics_seed=scenario.get("dynamics_seed"),
        traversal_penalties=penalties,
        no_fly_mode=str(scenario.get("no_fly_mode") or "penalty"),
        no_fly_penalty=float(scenario.get("no_fly_penalty", 5.0)),
        sensor_noise_probability=float(scenario.get("sensor_noise_probability", 0.0)),
        seed=0,
        potential_shaping_enabled=potential_shaping_enabled,
        observation_mode=observation_mode,
        dynamics_timing=dynamics_timing,
        max_steps=max_steps,
    )
    env.reset(seed=0)
    env.grid = np.full((int(scenario["grid_size"]), int(scenario["grid_size"])), CELL_FREE, dtype=np.int32)
    for cell in scenario.get("blocked", []):
        env.grid[tuple(cell)] = CELL_OBSTACLE
    for cell in scenario.get("no_fly_cells", []):
        env.grid[tuple(cell)] = CELL_NO_FLY
    for obstacle in obstacles:
        env.grid[obstacle.cell] = CELL_OBSTACLE if obstacle.initial_state == "blocked" else CELL_FREE
    for obstacle in stochastic:
        env.grid[obstacle.cell] = CELL_OBSTACLE if obstacle.initial_state == "blocked" else CELL_FREE
    for obstacle in moving:
        env.grid[obstacle.path[obstacle.initial_index]] = CELL_OBSTACLE
    env._initial_grid = env.grid.copy()
    env._distance_table = None
    env.uav_pos = np.asarray(scenario["start"], dtype=np.int32)
    env.goal_pos = np.asarray(scenario["goal"], dtype=np.int32)
    env.current_step = 0
    env._elapsed_steps = 0
    env._dynamics_rng = np.random.default_rng(scenario.get("dynamics_seed"))
    env._moving_indices = [obstacle.initial_index for obstacle in moving]
    env._last_dynamic_changes = []
    env.last_action = None
    env.visited_cells.clear()
    env.visited_cells.add(tuple(env.uav_pos))
    env._last_prev_pos = env.uav_pos.copy()
    return env, env._build_observation()


def empty_map_scenario(
    *,
    scenario_id: str,
    grid_size: int,
    distance_ratio: float,
    orientation: str,
) -> dict:
    max_index = grid_size - 1
    distance = max(1, min(max_index, int(round(max_index * distance_ratio))))
    mid = grid_size // 2
    if orientation == "horizontal":
        start = (mid, max(0, (grid_size - distance) // 2))
        goal = (mid, min(max_index, start[1] + distance))
    elif orientation == "vertical":
        start = (max(0, (grid_size - distance) // 2), mid)
        goal = (min(max_index, start[0] + distance), mid)
    elif orientation == "diagonal":
        start = (max(0, (grid_size - distance) // 2), max(0, (grid_size - distance) // 2))
        goal = (min(max_index, start[0] + distance), min(max_index, start[1] + distance))
    elif orientation == "goal_opposing":
        # Goal lies northeast of start. A wrong first move to the southwest remains legal,
        # so this probes horizon and direction normalization rather than immediate legality.
        start = (min(max_index - 1, max(1, distance)), min(max_index - 1, max(1, distance)))
        goal = (max(0, start[0] - distance), max(0, start[1] - distance))
    else:
        raise ValueError(f"unknown orientation: {orientation}")
    return {
        "scenario_id": scenario_id,
        "split": "empty_map_diagnostic",
        "grid_size": grid_size,
        "obstacle_density": 0.0,
        "blocked": [],
        "dynamic_obstacles": [],
        "start": list(start),
        "goal": list(goal),
        "stochastic_obstacles": [],
        "moving_obstacles": [],
        "traversal_penalties": [],
        "dynamics_seed": None,
        "no_fly_cells": [],
        "no_fly_mode": None,
        "no_fly_penalty": 5.0,
        "sensor_noise_probability": 0.0,
        "route_bucket": "empty",
    }
