import numpy as np
import pytest

from rl_agent.uav_env import REWARD_STEP, UAVRoutingEnv


def test_full_grid_observation_shape_scales_with_grid():
    env = UAVRoutingEnv(
        grid_size=9,
        obstacle_density=0.0,
        observation_mode="full",
        fixed_grid=True,
        seed=1,
    )
    observation, _ = env.reset()
    assert observation["observation"].shape == (4 + 9 * 9 + 8,)


def test_default_episode_limit_scales_with_grid_size():
    env = UAVRoutingEnv(grid_size=30, obstacle_density=0.0)
    assert env.max_steps == 900


def test_sparse_reward_ablation_disables_potential_term():
    env = UAVRoutingEnv(
        grid_size=5,
        obstacle_density=0.0,
        potential_shaping_enabled=False,
        seed=2,
    )
    env.reset()
    env.uav_pos = np.array([2, 2], dtype=np.int32)
    env.goal_pos = np.array([4, 4], dtype=np.int32)
    _, reward, terminated, _, _ = env.step(3)
    assert not terminated
    assert reward == REWARD_STEP


def test_dynamic_her_rejects_obstacle_dependent_potential():
    with pytest.raises(ValueError, match="incompatible with HER"):
        UAVRoutingEnv(
            dynamic_obstacles_enabled=True,
            potential_metric="shortest_path",
        )
