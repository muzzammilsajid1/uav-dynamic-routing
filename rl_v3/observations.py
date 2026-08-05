"""RL V3 observation construction without changing V2 transition semantics."""
from __future__ import annotations

import math

import numpy as np
from gymnasium import spaces

from rl_agent.uav_env import CELL_NO_FLY, CELL_OBSTACLE, UAVRoutingEnv


LOCAL_CHANNELS = 8
GLOBAL_CHANNELS = 8
OBSERVATION_FAMILIES = {"local_only", "global_local", "global_local_recency"}


def observation_space(
    local_size: int = 11,
    global_size: int = 24,
    max_grid_size: int = 100,
) -> spaces.Dict:
    return spaces.Dict(
        {
            "local_map": spaces.Box(
                0.0, 1.0, shape=(LOCAL_CHANNELS, local_size, local_size), dtype=np.float32
            ),
            "global_map": spaces.Box(
                0.0,
                1.0,
                shape=(GLOBAL_CHANNELS, global_size, global_size),
                dtype=np.float32,
            ),
            "scalars": spaces.Box(
                low=np.array([-1.0, -1.0, 0.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0, 7.0], dtype=np.float32),
                shape=(4,),
                dtype=np.float32,
            ),
        }
    )


def phase_b_observation_space(
    family: str, local_size: int = 11, global_size: int = 32
) -> spaces.Dict:
    """Observation space for one of the frozen Phase B information families."""
    if family not in OBSERVATION_FAMILIES:
        raise ValueError(f"unknown observation family: {family}")
    items: dict[str, spaces.Space] = {
        "local_map": spaces.Box(
            0.0, 1.0, shape=(LOCAL_CHANNELS, local_size, local_size), dtype=np.float32
        ),
        "scalars": spaces.Box(
            low=np.array([-1.0, -1.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 7.0], dtype=np.float32),
            shape=(4,),
            dtype=np.float32,
        ),
    }
    if family != "local_only":
        items["global_map"] = spaces.Box(
            0.0, 1.0, shape=(GLOBAL_CHANNELS, global_size, global_size), dtype=np.float32
        )
    return spaces.Dict(items)


def build_phase_b_observation(
    env: UAVRoutingEnv,
    *,
    family: str,
    local_size: int = 11,
    global_size: int = 32,
    visited: dict[tuple[int, int], int] | None = None,
) -> dict[str, np.ndarray]:
    """Build an encoder-specific observation without planner-derived inputs."""
    if family not in OBSERVATION_FAMILIES:
        raise ValueError(f"unknown observation family: {family}")
    use_recency = family == "global_local_recency"
    result = {
        "local_map": local_crop(
            env, local_size=local_size, visited=visited if use_recency else None
        ),
        "scalars": scalar_features(env),
    }
    if family != "local_only":
        result["global_map"] = coarse_global_map(
            env,
            output_size=global_size,
            visited=visited if use_recency else None,
        )
    return result


def build_v3_observation(
    env: UAVRoutingEnv,
    *,
    local_size: int = 11,
    global_size: int = 24,
    visited: dict[tuple[int, int], int] | None = None,
) -> dict[str, np.ndarray]:
    if env.grid is None or env.uav_pos is None or env.goal_pos is None:
        raise RuntimeError("environment must be reset before building observations")
    return {
        "local_map": local_crop(env, local_size=local_size, visited=visited),
        "global_map": coarse_global_map(env, output_size=global_size, visited=visited),
        "scalars": scalar_features(env),
    }


def scalar_features(env: UAVRoutingEnv, max_grid_size: int = 100) -> np.ndarray:
    norm = max(float(env.grid_size - 1), 1.0)
    previous_action = -1.0 if env.last_action is None else float(env.last_action)
    return np.array(
        [
            (float(env.goal_pos[0]) - float(env.uav_pos[0])) / norm,
            (float(env.goal_pos[1]) - float(env.uav_pos[1])) / norm,
            float(env.grid_size) / float(max_grid_size),
            previous_action,
        ],
        dtype=np.float32,
    )


def local_crop(
    env: UAVRoutingEnv,
    *,
    local_size: int = 11,
    visited: dict[tuple[int, int], int] | None = None,
) -> np.ndarray:
    if local_size % 2 != 1:
        raise ValueError("local_size must be odd")
    radius = local_size // 2
    channels = np.zeros((LOCAL_CHANNELS, local_size, local_size), dtype=np.float32)
    for out_r, dr in enumerate(range(-radius, radius + 1)):
        for out_c, dc in enumerate(range(-radius, radius + 1)):
            row = int(env.uav_pos[0]) + dr
            col = int(env.uav_pos[1]) + dc
            if row < 0 or row >= env.grid_size or col < 0 or col >= env.grid_size:
                channels[0, out_r, out_c] = 1.0
                continue
            value = int(env.grid[row, col])
            if value == CELL_OBSTACLE:
                channels[0, out_r, out_c] = 1.0
            if value == CELL_NO_FLY:
                channels[1, out_r, out_c] = 1.0
            if value != CELL_OBSTACLE:
                channels[6, out_r, out_c] = 1.0
            channels[2, out_r, out_c] = _penalty_value(env, (row, col))
            if row == int(env.uav_pos[0]) and col == int(env.uav_pos[1]):
                channels[3, out_r, out_c] = 1.0
            if row == int(env.goal_pos[0]) and col == int(env.goal_pos[1]):
                channels[4, out_r, out_c] = 1.0
            if (row, col) in {tuple(cell) for cell in env._last_dynamic_changes}:
                channels[5, out_r, out_c] = 1.0
            channels[7, out_r, out_c] = _visited_value(visited, (row, col))
    return channels


def coarse_global_map(
    env: UAVRoutingEnv,
    *,
    output_size: int = 24,
    visited: dict[tuple[int, int], int] | None = None,
) -> np.ndarray:
    """Downsample with channel-specific aggregation, never average occupancy."""
    channels = np.zeros((GLOBAL_CHANNELS, output_size, output_size), dtype=np.float32)
    # Vectorized exact equivalent of per-cell maximum aggregation. This is
    # performance-critical at 100x100 and does not use image interpolation.
    row_bins = np.minimum(
        output_size - 1,
        np.arange(env.grid_size, dtype=np.int32) * output_size // env.grid_size,
    )
    col_bins = row_bins.copy()
    out_rows = np.repeat(row_bins, env.grid_size)
    out_cols = np.tile(col_bins, env.grid_size)
    flat_grid = env.grid.reshape(-1)
    np.maximum.at(channels[0], (out_rows, out_cols), (flat_grid == CELL_OBSTACLE).astype(np.float32))
    np.maximum.at(channels[1], (out_rows, out_cols), (flat_grid == CELL_NO_FLY).astype(np.float32))
    np.maximum.at(channels[6], (out_rows, out_cols), (flat_grid != CELL_OBSTACLE).astype(np.float32))
    for cell in env.traversal_penalties:
        out_r, out_c = _bin(cell, env.grid_size, output_size)
        channels[2, out_r, out_c] = max(channels[2, out_r, out_c], _penalty_value(env, cell))
    if visited:
        for cell in visited:
            out_r, out_c = _bin(cell, env.grid_size, output_size)
            channels[7, out_r, out_c] = max(channels[7, out_r, out_c], _visited_value(visited, cell))
    agent_bin = _bin(tuple(int(value) for value in env.uav_pos), env.grid_size, output_size)
    goal_bin = _bin(tuple(int(value) for value in env.goal_pos), env.grid_size, output_size)
    channels[3, agent_bin[0], agent_bin[1]] = 1.0
    channels[4, goal_bin[0], goal_bin[1]] = 1.0
    for cell in env._last_dynamic_changes:
        changed_bin = _bin(tuple(int(value) for value in cell), env.grid_size, output_size)
        channels[5, changed_bin[0], changed_bin[1]] = 1.0
    return channels


def _bin(cell: tuple[int, int], grid_size: int, output_size: int) -> tuple[int, int]:
    return (
        min(output_size - 1, int(cell[0] * output_size / grid_size)),
        min(output_size - 1, int(cell[1] * output_size / grid_size)),
    )


def _penalty_value(env: UAVRoutingEnv, cell: tuple[int, int]) -> float:
    value = float(env.traversal_penalties.get(cell, 0.0))
    if value <= 0.0:
        return 0.0
    return min(1.0, math.log1p(value) / math.log1p(10.0))


def _visited_value(
    visited: dict[tuple[int, int], int] | None, cell: tuple[int, int]
) -> float:
    if not visited:
        return 0.0
    count = visited.get(cell, 0)
    return min(1.0, float(count) / 5.0)
