import json

import numpy as np

from rl_v3.env_builders import empty_map_scenario, env_from_scenario
from rl_v3.observations import (
    build_phase_b_observation, coarse_global_map, phase_b_observation_space,
    _penalty_value, _visited_value,
)
from rl_agent.uav_env import CELL_NO_FLY, CELL_OBSTACLE, UAVRoutingEnv
from rl_v3.phase_b_env import PhaseBCurriculumEnv
from rl_v3.phase_b_scenarios import generate_validation_v2, manifest_balance
from rl_v3.wrappers import PhaseBUAVRoutingWrapper


def _config():
    return json.loads(open("configs/rl_v3_phase_b.json", encoding="utf-8").read())


def test_validation_v2_is_exactly_balanced_and_not_final():
    manifest = generate_validation_v2(_config())
    balance = manifest_balance(manifest)
    assert len(manifest["scenarios"]) == 96
    assert set(balance["by_scale"].values()) == {24}
    assert set(balance["by_route_bucket"].values()) == {32}
    assert set(balance["by_family"].values()) == {24}
    assert set(balance["scale_family_bucket"].values()) == {2}
    assert not manifest["final_test"]
    assert all(not 910000 <= row["episode_seed"] <= 919999 for row in manifest["scenarios"])


def test_validation_v2_generation_is_deterministic():
    assert generate_validation_v2(_config()) == generate_validation_v2(_config())


def test_phase_b_observation_families_have_only_approved_keys():
    scenario = empty_map_scenario(scenario_id="test", grid_size=15, distance_ratio=.5, orientation="diagonal")
    env, _ = env_from_scenario(scenario, potential_shaping_enabled=False)
    visited = {tuple(env.uav_pos): 2}
    local = build_phase_b_observation(env, family="local_only", visited=visited)
    global_local = build_phase_b_observation(env, family="global_local", visited=visited)
    recency = build_phase_b_observation(env, family="global_local_recency", visited=visited)
    assert set(local) == {"local_map", "scalars"}
    assert set(global_local) == {"local_map", "global_map", "scalars"}
    assert global_local["local_map"][7].sum() == 0
    assert global_local["global_map"][7].sum() == 0
    assert recency["local_map"][7].sum() > 0
    assert recency["global_map"][7].sum() > 0
    assert recency["global_map"].shape == (8, 32, 32)


def test_r1_and_r2_terminal_contract_and_gentle_shaping():
    scenario = empty_map_scenario(scenario_id="reward", grid_size=15, distance_ratio=.25, orientation="horizontal")
    env, _ = env_from_scenario(scenario, potential_shaping_enabled=False, max_steps=30)
    r1 = PhaseBUAVRoutingWrapper(env, observation_family="local_only", reward_family="R1")
    r1.visit_counts = {tuple(env.uav_pos): 1}
    legal = np.flatnonzero(r1.action_masks())
    _, reward, _, _, _ = r1.step(int(legal[0]))
    assert abs(reward) < .02

    env2, _ = env_from_scenario(scenario, potential_shaping_enabled=False, max_steps=30)
    r2 = PhaseBUAVRoutingWrapper(env2, observation_family="local_only", reward_family="R2")
    r2.visit_counts = {tuple(env2.uav_pos): 1}
    legal2 = np.flatnonzero(r2.action_masks())
    _, shaped, _, _, _ = r2.step(int(legal2[0]))
    assert abs(shaped) < .05


def test_curriculum_environment_masks_and_state_are_resumable():
    env = PhaseBCurriculumEnv(_config(), "global_local", "R1")
    observation, _ = env.reset(seed=7)
    assert phase_b_observation_space("global_local").contains(observation)
    mask = env.action_masks()
    assert mask.shape == (8,) and mask.any()
    state = env.get_generator_state()
    clone = PhaseBCurriculumEnv(_config(), "global_local", "R1")
    clone.set_generator_state(state)
    assert clone.get_generator_state() == state


def test_vectorized_coarse_map_is_bit_identical_to_cellwise_reference():
    rng = np.random.default_rng(20260805)
    for size in (15, 31, 50, 73, 100):
        env = UAVRoutingEnv(grid_size=size, obstacle_density=0.0, fixed_grid=True, seed=1)
        env.reset(seed=1)
        env.grid = rng.choice([0, CELL_OBSTACLE, CELL_NO_FLY], size=(size, size), p=[.82, .15, .03]).astype(np.int32)
        env.uav_pos = np.array([1, 1], dtype=np.int32)
        env.goal_pos = np.array([min(2, size - 1), min(2, size - 1)], dtype=np.int32)
        env._last_dynamic_changes = [(size // 2, size // 2)]
        env.traversal_penalties = {(3, 4): 2.0, (size - 2, size - 3): 7.0}
        visited = {(1, 1): 3, (size // 2, size // 2): 5}
        actual = coarse_global_map(env, output_size=32, visited=visited)
        expected = _cellwise_reference(env, 32, visited)
        assert np.array_equal(actual, expected)


def _cellwise_reference(env, output_size, visited):
    channels = np.zeros((8, output_size, output_size), dtype=np.float32)
    for row in range(env.grid_size):
        for col in range(env.grid_size):
            out_r = min(output_size - 1, int(row * output_size / env.grid_size))
            out_c = min(output_size - 1, int(col * output_size / env.grid_size))
            value = int(env.grid[row, col])
            channels[0, out_r, out_c] = max(channels[0, out_r, out_c], float(value == CELL_OBSTACLE))
            channels[1, out_r, out_c] = max(channels[1, out_r, out_c], float(value == CELL_NO_FLY))
            channels[6, out_r, out_c] = max(channels[6, out_r, out_c], float(value != CELL_OBSTACLE))
            channels[2, out_r, out_c] = max(channels[2, out_r, out_c], _penalty_value(env, (row, col)))
            channels[7, out_r, out_c] = max(channels[7, out_r, out_c], _visited_value(visited, (row, col)))
    agent_bin = tuple(np.minimum(output_size - 1, env.uav_pos * output_size // env.grid_size))
    goal_bin = tuple(np.minimum(output_size - 1, env.goal_pos * output_size // env.grid_size))
    channels[3, agent_bin[0], agent_bin[1]] = 1
    channels[4, goal_bin[0], goal_bin[1]] = 1
    for row, col in env._last_dynamic_changes:
        channels[5, min(output_size - 1, row * output_size // env.grid_size), min(output_size - 1, col * output_size // env.grid_size)] = 1
    return channels
