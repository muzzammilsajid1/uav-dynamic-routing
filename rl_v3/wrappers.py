"""V3 wrappers around the authoritative V2 Gymnasium environment."""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from rl_agent.uav_env import UAVRoutingEnv
from rl_v3.action_masking import legal_action_mask
from rl_v3.observations import (
    build_phase_b_observation,
    build_v3_observation,
    observation_space,
    phase_b_observation_space,
)


class V3UAVRoutingWrapper(gym.Wrapper):
    """Add V3 observations and masks while delegating transitions to V2."""

    def __init__(
        self,
        env: UAVRoutingEnv,
        *,
        local_size: int = 11,
        global_size: int = 24,
    ) -> None:
        super().__init__(env)
        self.v2_env = env
        self.local_size = local_size
        self.global_size = global_size
        self.observation_space = observation_space(local_size, global_size)
        self.action_space = env.action_space
        self.visit_counts: dict[tuple[int, int], int] = {}

    def reset(self, **kwargs):
        _, info = self.v2_env.reset(**kwargs)
        self.visit_counts = {tuple(int(value) for value in self.v2_env.uav_pos): 1}
        return self._observation(), self._info(info)

    def step(self, action: int):
        _, reward, terminated, truncated, info = self.v2_env.step(action)
        cell = tuple(int(value) for value in self.v2_env.uav_pos)
        self.visit_counts[cell] = self.visit_counts.get(cell, 0) + 1
        return self._observation(), reward, terminated, truncated, self._info(info)

    def action_masks(self) -> np.ndarray:
        return legal_action_mask(self.v2_env)

    def _observation(self) -> dict[str, np.ndarray]:
        return build_v3_observation(
            self.v2_env,
            local_size=self.local_size,
            global_size=self.global_size,
            visited=self.visit_counts,
        )

    def _info(self, info: dict) -> dict:
        enriched = dict(info)
        enriched["action_mask"] = self.action_masks().astype(int).tolist()
        return enriched


class PhaseBUAVRoutingWrapper(gym.Wrapper):
    """Phase B observation and reward contract over the authoritative V2 transition."""

    def __init__(
        self,
        env: UAVRoutingEnv,
        *,
        observation_family: str,
        reward_family: str,
        local_size: int = 11,
        global_size: int = 32,
        step_penalty_total: float = -0.20,
        progress_coefficient: float = 0.10,
        gamma: float = 0.99,
    ) -> None:
        super().__init__(env)
        if reward_family not in {"R1", "R2"}:
            raise ValueError("reward_family must be R1 or R2")
        self.v2_env = env
        self.observation_family = observation_family
        self.reward_family = reward_family
        self.local_size = local_size
        self.global_size = global_size
        self.step_penalty_total = float(step_penalty_total)
        self.progress_coefficient = float(progress_coefficient)
        self.gamma = float(gamma)
        self.observation_space = phase_b_observation_space(
            observation_family, local_size, global_size
        )
        self.action_space = env.action_space
        self.visit_counts: dict[tuple[int, int], int] = {}

    def reset(self, **kwargs):
        _, info = self.v2_env.reset(**kwargs)
        self.visit_counts = {tuple(int(v) for v in self.v2_env.uav_pos): 1}
        return self._observation(), self._info(info)

    def step(self, action: int):
        previous = self.v2_env.uav_pos.copy()
        _, _, terminated, truncated, info = self.v2_env.step(action)
        cell = tuple(int(v) for v in self.v2_env.uav_pos)
        self.visit_counts[cell] = self.visit_counts.get(cell, 0) + 1
        reward = self._reward(previous, info)
        return self._observation(), reward, terminated, truncated, self._info(info)

    def action_masks(self) -> np.ndarray:
        return legal_action_mask(self.v2_env)

    def _observation(self) -> dict[str, np.ndarray]:
        return build_phase_b_observation(
            self.v2_env,
            family=self.observation_family,
            local_size=self.local_size,
            global_size=self.global_size,
            visited=self.visit_counts,
        )

    def _reward(self, previous: np.ndarray, info: dict) -> float:
        if info.get("crashed", False):
            return -1.0
        if info.get("is_success", False):
            return 1.0
        reward = self.step_penalty_total / max(1.0, float(self.v2_env.max_steps))
        if self.reward_family == "R2":
            max_distance = max(1.0, (self.v2_env.grid_size - 1) * np.sqrt(2.0))
            phi_previous = -_octile(previous, self.v2_env.goal_pos) / max_distance
            phi_current = -_octile(self.v2_env.uav_pos, self.v2_env.goal_pos) / max_distance
            reward += self.progress_coefficient * (
                self.gamma * phi_current - phi_previous
            )
        return float(reward)

    def _info(self, info: dict) -> dict:
        result = dict(info)
        result["action_mask"] = self.action_masks().astype(int).tolist()
        return result


def _octile(first: np.ndarray, second: np.ndarray) -> float:
    delta = np.abs(first.astype(float) - second.astype(float))
    diagonal = min(float(delta[0]), float(delta[1]))
    straight = max(float(delta[0]), float(delta[1])) - diagonal
    return diagonal * np.sqrt(2.0) + straight


def clone_v2_env_state(source: UAVRoutingEnv, target: UAVRoutingEnv) -> None:
    """Copy deterministic state for parity tests without changing semantics."""
    target.grid = source.grid.copy()
    target._initial_grid = source._initial_grid.copy() if source._initial_grid is not None else None
    target.uav_pos = source.uav_pos.copy()
    target.goal_pos = source.goal_pos.copy()
    target.current_step = int(source.current_step)
    target._elapsed_steps = int(source._elapsed_steps)
    target._last_dynamic_changes = list(source._last_dynamic_changes)
    target.last_action = source.last_action
    target.visited_cells = set(source.visited_cells)
    target._moving_indices = list(source._moving_indices)
    target._dynamics_rng = np.random.default_rng(source.dynamics_seed)
    target._last_prev_pos = source._last_prev_pos.copy() if hasattr(source, "_last_prev_pos") else source.uav_pos.copy()
    target._last_crashed = getattr(source, "_last_crashed", False)
    target._last_energy_penalty = getattr(source, "_last_energy_penalty", 0.0)
