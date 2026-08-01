import math

import numpy as np

from baselines.dijkstra import dijkstra
from envs.grid_environment import (
    GridEnvironment,
    MovingObstacle,
    StochasticObstacle,
)
from rl_agent.uav_env import CELL_FREE, CELL_NO_FLY, CELL_OBSTACLE, UAVRoutingEnv


def test_traversal_penalty_changes_route_choice():
    env = GridEnvironment(
        size=3,
        obstacle_density=0.0,
        start=(1, 0),
        goal=(1, 2),
        traversal_penalties={(1, 1): 10.0},
    )
    result = dijkstra(env.start, env.goal, env.get_neighbors)
    assert (1, 1) not in result.path
    assert math.isclose(result.cost, 2 * math.sqrt(2))


def test_stochastic_obstacle_is_seeded_and_resettable():
    obstacle = StochasticObstacle((2, 2), toggle_probability=0.5)
    first = GridEnvironment(
        size=5,
        obstacle_density=0.0,
        stochastic_obstacles=[obstacle],
        dynamics_seed=9,
    )
    second = GridEnvironment(
        size=5,
        obstacle_density=0.0,
        stochastic_obstacles=[obstacle],
        dynamics_seed=9,
    )
    assert [first.step_dynamics() for _ in range(12)] == [
        second.step_dynamics() for _ in range(12)
    ]
    first.reset_dynamics()
    second.reset_dynamics()
    assert [first.step_dynamics() for _ in range(12)] == [
        second.step_dynamics() for _ in range(12)
    ]


def test_moving_obstacle_advances_on_period():
    env = GridEnvironment(
        size=6,
        obstacle_density=0.0,
        moving_obstacles=[
            MovingObstacle(path=[(2, 2), (2, 3), (3, 3)], period=2)
        ],
    )
    assert env.is_blocked((2, 2))
    assert env.step_dynamics() == set()
    assert env.step_dynamics() == {(2, 2), (2, 3)}
    assert not env.is_blocked((2, 2))
    assert env.is_blocked((2, 3))


def test_rl_moving_obstacle_uses_same_period_rule():
    env = UAVRoutingEnv(
        grid_size=6,
        obstacle_density=0.0,
        moving_obstacles=[
            MovingObstacle(path=[(2, 2), (2, 3), (3, 3)], period=2)
        ],
        seed=5,
    )
    env.reset()
    assert env.grid[2, 2] == CELL_OBSTACLE
    assert env._toggle_dynamic_obstacles() == []
    assert env._toggle_dynamic_obstacles() == [(2, 2), (2, 3)]
    assert env.grid[2, 3] == CELL_OBSTACLE


def test_sensor_noise_does_not_change_underlying_grid():
    env = UAVRoutingEnv(
        grid_size=7,
        obstacle_density=0.0,
        sensor_noise_probability=1.0,
        seed=6,
    )
    env.reset()
    env.uav_pos = np.array([3, 3])
    underlying = env.grid.copy()
    observed = env._get_local_observation()
    assert (observed == CELL_OBSTACLE).all()
    assert (env.grid == underlying).all()


def test_hard_no_fly_zone_is_not_a_neighbor():
    env = UAVRoutingEnv(
        grid_size=5,
        obstacle_density=0.0,
        no_fly_mode="hard",
        seed=7,
    )
    env.reset()
    env.grid.fill(CELL_FREE)
    env.grid[2, 3] = CELL_NO_FLY
    neighbors = {
        tuple(position) for position, _ in env.get_neighbors(np.array([2, 2]))
    }
    assert (2, 3) not in neighbors
