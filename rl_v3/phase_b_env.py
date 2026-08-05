"""Curriculum environment that delegates every transition to V2."""
from __future__ import annotations

import math

import gymnasium as gym

from rl_v3.env_builders import env_from_scenario
from rl_v3.phase_b_scenarios import generate_curriculum_scenario
from rl_v3.wrappers import PhaseBUAVRoutingWrapper


class PhaseBCurriculumEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: dict, observation_family: str, reward_family: str):
        super().__init__()
        self.config = config
        self.observation_family = observation_family
        self.reward_family = reward_family
        obs = config["observation"]
        reward = config["reward"]
        self._wrapper_kwargs = {
            "observation_family": observation_family,
            "reward_family": reward_family,
            "local_size": int(obs["local_size"]),
            "global_size": int(obs["global_size"]),
            "step_penalty_total": float(reward["step_penalty_total_per_episode_budget"]),
            "progress_coefficient": float(reward["R2_progress_coefficient"]),
            "gamma": float(reward["R2_discount"]),
        }
        from rl_v3.observations import phase_b_observation_space
        self.observation_space = phase_b_observation_space(
            observation_family, int(obs["local_size"]), int(obs["global_size"])
        )
        self.action_space = gym.spaces.Discrete(8)
        self.episode_index = 0
        self.total_interactions = 0
        self.current: PhaseBUAVRoutingWrapper | None = None
        self.current_scenario: dict | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        scenario = generate_curriculum_scenario(
            self.config, self.episode_index, self.total_interactions
        )
        self.episode_index += 1
        self.current_scenario = scenario
        multiplier = float(self.config["training"]["episode_budget_astar_multiplier"])
        minimum = int(self.config["training"]["minimum_episode_budget"])
        budget = max(minimum, int(math.ceil(multiplier * scenario["initial_astar_cost"])))
        v2, _ = env_from_scenario(
            scenario,
            potential_shaping_enabled=False,
            dynamics_timing="post_move_observed",
            max_steps=budget,
        )
        self.current = PhaseBUAVRoutingWrapper(v2, **self._wrapper_kwargs)
        start = tuple(int(v) for v in v2.uav_pos)
        self.current.visit_counts = {start: 1}
        return self.current._observation(), self.current._info(v2._build_info())

    def step(self, action: int):
        if self.current is None:
            raise RuntimeError("reset must be called before step")
        result = self.current.step(action)
        self.total_interactions += 1
        return result

    def action_masks(self):
        if self.current is None:
            raise RuntimeError("reset must be called before action_masks")
        return self.current.action_masks()

    def get_generator_state(self) -> dict:
        return {
            "episode_index": self.episode_index,
            "total_interactions": self.total_interactions,
        }

    def set_generator_state(self, state: dict) -> None:
        self.episode_index = int(state["episode_index"])
        self.total_interactions = int(state["total_interactions"])

    def close(self):
        if self.current is not None:
            self.current.close()
