import numpy as np

from rl_agent.uav_env import CELL_NO_FLY, CELL_OBSTACLE, UAVRoutingEnv
from rl_v3.observations import coarse_global_map, scalar_features


def _env(size):
    env = UAVRoutingEnv(grid_size=size, obstacle_density=0.0, fixed_grid=True, seed=1)
    env.reset(seed=1)
    env.grid = np.zeros((size, size), dtype=np.int32)
    env.uav_pos = np.array([1, 1], dtype=np.int32)
    env.goal_pos = np.array([size - 2, size - 2], dtype=np.int32)
    return env


def test_coarse_map_preserves_one_cell_wall_and_corridor():
    env = _env(17)
    env.grid[:, 8] = CELL_OBSTACLE
    env.grid[8, 8] = 0
    coarse = coarse_global_map(env, output_size=8)
    assert coarse[0].sum() >= 7
    corridor_bin = int(8 * 8 / 17)
    assert coarse[6, corridor_bin, corridor_bin] == 1.0


def test_coarse_map_preserves_isolated_obstacle_and_u_shape():
    env = _env(19)
    env.grid[5, 5] = CELL_OBSTACLE
    env.grid[10, 10:15] = CELL_OBSTACLE
    env.grid[10:15, 10] = CELL_OBSTACLE
    env.grid[10:15, 14] = CELL_OBSTACLE
    coarse = coarse_global_map(env, output_size=10)
    assert coarse[0].sum() >= 5


def test_agent_and_goal_can_share_same_coarse_bin():
    env = _env(100)
    env.uav_pos = np.array([1, 1], dtype=np.int32)
    env.goal_pos = np.array([2, 2], dtype=np.int32)
    coarse = coarse_global_map(env, output_size=24)
    assert coarse[3].sum() == 1.0
    assert coarse[4].sum() == 1.0
    assert np.argwhere(coarse[3] == 1.0).tolist() == np.argwhere(coarse[4] == 1.0).tolist()


def test_non_divisible_dimensions_and_hard_no_fly_channel():
    env = _env(17)
    env.grid[16, 16] = CELL_NO_FLY
    coarse = coarse_global_map(env, output_size=6)
    assert coarse.shape == (8, 6, 6)
    assert coarse[1, 5, 5] == 1.0


def test_initial_scalar_features_are_limited_to_approved_set():
    env = _env(15)
    values = scalar_features(env)
    assert values.shape == (4,)
    assert values[2] == 0.15
