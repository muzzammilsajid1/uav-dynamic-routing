import numpy as np

from envs.grid_environment import DynamicObstacle
from rl_agent.uav_env import UAVRoutingEnv
from rl_v3.wrappers import V3UAVRoutingWrapper, clone_v2_env_state


def _env():
    env = UAVRoutingEnv(
        grid_size=7,
        obstacle_density=0.0,
        fixed_grid=True,
        seed=5,
        dynamic_obstacles_enabled=True,
        dynamic_obstacles=[DynamicObstacle((3, 3), period=2, initial_state="passable")],
        potential_shaping_enabled=False,
    )
    env.reset(seed=5)
    env.grid = np.zeros((7, 7), dtype=np.int32)
    env.uav_pos = np.array([2, 2], dtype=np.int32)
    env.goal_pos = np.array([6, 6], dtype=np.int32)
    env._initial_grid = env.grid.copy()
    env._last_prev_pos = env.uav_pos.copy()
    return env


def test_v3_wrapper_delegates_transition_to_v2_step():
    v2 = _env()
    wrapped_core = _env()
    clone_v2_env_state(v2, wrapped_core)
    v3 = V3UAVRoutingWrapper(wrapped_core)

    for action in [7, 3, 1, 7]:
        _, reward_v2, terminated_v2, truncated_v2, info_v2 = v2.step(action)
        _, reward_v3, terminated_v3, truncated_v3, info_v3 = v3.step(action)
        assert reward_v3 == reward_v2
        assert terminated_v3 == terminated_v2
        assert truncated_v3 == truncated_v2
        assert info_v3["uav_pos"] == info_v2["uav_pos"]
        assert info_v3["dynamic_changes"] == info_v2["dynamic_changes"]
        assert np.array_equal(v3.v2_env.grid, v2.grid)
        if terminated_v2 or truncated_v2:
            break
