"""Fixed single-scenario Gymnasium environment for Phase C0 overfitting test.

Wraps the authoritative V2 UAVRoutingEnv with the global_local Phase B
observation and R1 sparse reward.  Every call to reset() returns the same
deterministic start, goal and empty 15x15 grid -- there is no curriculum,
no randomization and no dynamics.
"""
from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import gymnasium as gym

from rl_agent.uav_env import CELL_FREE, UAVRoutingEnv
from rl_v3.action_masking import legal_action_mask
from rl_v3.observations import build_phase_b_observation, phase_b_observation_space


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _astar_cost(start: tuple[int,int], goal: tuple[int,int]) -> float:
    """Octile-distance lower bound on an empty grid (exact for empty grids)."""
    dr = abs(goal[0] - start[0])
    dc = abs(goal[1] - start[1])
    diag = min(dr, dc)
    straight = max(dr, dc) - diag
    return diag * math.sqrt(2.0) + straight


class PhaseC0Env(gym.Env):
    """Deterministic single-scenario environment.

    Always resets to the same start/goal on a 15x15 empty grid.
    Observation: global_local (local 11x11 + global 32x32 + 4 scalars).
    Reward:      R1 sparse (+1 goal, -1 crash, small step penalty).
    """

    metadata = {"render_modes": []}

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        sc = config["scenario"]
        obs_cfg = config["observation"]
        rew_cfg = config["reward"]
        tr_cfg = config["training"]

        self._grid_size = int(sc["grid_size"])
        self._start = tuple(int(v) for v in sc["start"])
        self._goal = tuple(int(v) for v in sc["goal"])
        self._local_size = int(obs_cfg["local_size"])
        self._global_size = int(obs_cfg["global_size"])
        self._step_penalty_total = float(rew_cfg["step_penalty_total_per_episode_budget"])

        astar = _astar_cost(self._start, self._goal)
        multiplier = float(tr_cfg["episode_budget_astar_multiplier"])
        minimum = int(tr_cfg["minimum_episode_budget"])
        self._max_steps = max(minimum, int(math.ceil(multiplier * astar)))

        self.observation_space = phase_b_observation_space(
            "global_local", self._local_size, self._global_size
        )
        self.action_space = gym.spaces.Discrete(8)

        self._v2: UAVRoutingEnv | None = None
        self._visit_counts: dict[tuple[int,int], int] = {}
        self._episode_count = 0

    # ------------------------------------------------------------------
    def _make_v2(self) -> UAVRoutingEnv:
        env = UAVRoutingEnv(
            grid_size=self._grid_size,
            obstacle_density=0.0,
            no_fly_density=0.0,
            fixed_grid=True,
            dynamic_obstacles_enabled=False,
            potential_shaping_enabled=False,
            observation_mode="local",
            dynamics_timing="post_move_observed",
            max_steps=self._max_steps,
            seed=0,
        )
        env.reset(seed=0)
        env.grid = np.full(
            (self._grid_size, self._grid_size), CELL_FREE, dtype=np.int32
        )
        env._initial_grid = env.grid.copy()
        env._distance_table = None
        env.uav_pos = np.asarray(self._start, dtype=np.int32)
        env.goal_pos = np.asarray(self._goal, dtype=np.int32)
        env.current_step = 0
        env._elapsed_steps = 0
        env._dynamics_rng = np.random.default_rng(None)
        env._moving_indices = []
        env._last_dynamic_changes = []
        env.last_action = None
        env.visited_cells.clear()
        env.visited_cells.add(self._start)
        env._last_prev_pos = env.uav_pos.copy()
        return env

    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._v2 = self._make_v2()
        self._visit_counts = {self._start: 1}
        self._episode_count += 1
        return self._obs(), self._info()

    def step(self, action: int):
        assert self._v2 is not None, "call reset() before step()"
        prev = self._v2.uav_pos.copy()
        _, _, terminated, truncated, raw_info = self._v2.step(action)
        cell = tuple(int(v) for v in self._v2.uav_pos)
        self._visit_counts[cell] = self._visit_counts.get(cell, 0) + 1
        reward = self._reward(prev, raw_info)
        info = self._info(raw_info)
        return self._obs(), reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        assert self._v2 is not None
        return legal_action_mask(self._v2)

    # ------------------------------------------------------------------
    def _obs(self) -> dict[str, np.ndarray]:
        return build_phase_b_observation(
            self._v2,
            family="global_local",
            local_size=self._local_size,
            global_size=self._global_size,
            visited=self._visit_counts,
        )

    def _reward(self, previous: np.ndarray, raw_info: dict) -> float:
        if raw_info.get("crashed", False):
            return -1.0
        if raw_info.get("is_success", False):
            return 1.0
        step_penalty = self._step_penalty_total / max(1.0, float(self._v2.max_steps))
        return float(step_penalty)

    def _info(self, raw_info: dict | None = None) -> dict:
        result = dict(raw_info) if raw_info else {}
        result["action_mask"] = self.action_masks().astype(int).tolist()
        result["episode_count"] = self._episode_count
        result["grid_size"] = self._grid_size
        result["start"] = list(self._start)
        result["goal"] = list(self._goal)
        return result

    def close(self):
        if self._v2 is not None:
            self._v2.close()
