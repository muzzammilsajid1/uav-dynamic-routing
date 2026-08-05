import numpy as np

from envs.grid_environment import DynamicObstacle
from rl_agent.uav_env import CELL_FREE, CELL_NO_FLY, CELL_OBSTACLE, UAVRoutingEnv
from rl_v3.action_masking import legal_action_mask


def _empty_env(size=5):
    env = UAVRoutingEnv(grid_size=size, obstacle_density=0.0, fixed_grid=True, seed=1)
    env.reset(seed=1)
    env.grid = np.zeros((size, size), dtype=np.int32)
    env.uav_pos = np.array([size // 2, size // 2], dtype=np.int32)
    env.goal_pos = np.array([size - 1, size - 1], dtype=np.int32)
    return env


def test_masks_all_boundaries_and_corners():
    env = _empty_env(5)
    cases = {
        (0, 2): {0, 4, 5},
        (4, 2): {1, 6, 7},
        (2, 0): {2, 4, 6},
        (2, 4): {3, 5, 7},
        (0, 0): {0, 2, 4, 5, 6},
        (0, 4): {0, 3, 4, 5, 7},
        (4, 0): {1, 2, 4, 6, 7},
        (4, 4): {1, 3, 5, 6, 7},
    }
    for pos, illegal in cases.items():
        env.uav_pos = np.array(pos, dtype=np.int32)
        mask = legal_action_mask(env)
        assert not any(mask[action] for action in illegal)


def test_mask_blocks_static_obstacles_and_hard_no_fly():
    env = _empty_env(5)
    env.uav_pos = np.array([2, 2], dtype=np.int32)
    env.grid[1, 2] = CELL_OBSTACLE
    env.grid[2, 3] = CELL_NO_FLY
    env.no_fly_mode = "hard"
    mask = legal_action_mask(env)
    assert not mask[0]
    assert not mask[3]
    env.no_fly_mode = "penalty"
    assert legal_action_mask(env)[3]


def test_mask_updates_after_dynamic_change():
    env = UAVRoutingEnv(
        grid_size=5,
        obstacle_density=0.0,
        fixed_grid=True,
        seed=1,
        dynamic_obstacles_enabled=True,
        dynamic_obstacles=[DynamicObstacle((1, 2), period=1, initial_state="passable")],
        potential_shaping_enabled=False,
    )
    env.reset(seed=1)
    env.grid = np.zeros((5, 5), dtype=np.int32)
    env.uav_pos = np.array([2, 2], dtype=np.int32)
    env.goal_pos = np.array([4, 4], dtype=np.int32)
    assert legal_action_mask(env)[0]
    env._toggle_dynamic_obstacles()
    assert not legal_action_mask(env)[0]


def test_occupied_blocked_cell_can_escape_under_v2_contract():
    env = _empty_env(5)
    env.uav_pos = np.array([2, 2], dtype=np.int32)
    env.grid[2, 2] = CELL_OBSTACLE
    mask = legal_action_mask(env)
    assert mask.any()
    assert mask[0]


def test_diagonal_corner_cutting_is_preserved():
    env = _empty_env(5)
    env.uav_pos = np.array([2, 2], dtype=np.int32)
    env.grid[1, 2] = CELL_OBSTACLE
    env.grid[2, 1] = CELL_OBSTACLE
    # V2 allows diagonal corner cutting, so NW remains legal if target is free.
    assert legal_action_mask(env)[4]


def test_only_one_legal_action_case():
    env = _empty_env(5)
    env.uav_pos = np.array([2, 2], dtype=np.int32)
    for action, delta in enumerate(env.ACTION_DELTAS):
        dest = env.uav_pos + delta
        env.grid[tuple(dest)] = CELL_OBSTACLE
    env.grid[1, 2] = CELL_FREE
    mask = legal_action_mask(env)
    assert mask.tolist().count(True) == 1
    assert mask[0]
