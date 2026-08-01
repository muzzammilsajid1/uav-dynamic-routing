import math

from baselines.dijkstra import dijkstra
from baselines.dstar_lite import DStarLite, run_dstar_lite_replanning
from baselines.replanning import run_naive_replanning
from envs.grid_environment import DynamicObstacle, GridEnvironment


def test_initial_plan_matches_dijkstra():
    env = GridEnvironment(size=12, obstacle_density=0.0)
    reference = dijkstra(env.start, env.goal, env.get_neighbors)
    planner = DStarLite(env, env.start, env.goal)
    assert planner.compute_shortest_path().found == reference.found
    current = env.start
    cost = 0.0
    while current != env.goal:
        move = planner.next_step(current)
        assert move is not None
        current, edge_cost = move
        planner.start = current
        cost += edge_cost
    assert math.isclose(cost, reference.cost, abs_tol=1e-9)


def test_dynamic_run_matches_full_replanning_cost():
    dynamic = [
        DynamicObstacle((4, 4), period=3, initial_state="passable"),
        DynamicObstacle((7, 7), period=5, initial_state="blocked"),
    ]
    full = run_naive_replanning(
        GridEnvironment(size=10, obstacle_density=0.0, dynamic_obstacles=dynamic)
    )
    incremental = run_dstar_lite_replanning(
        GridEnvironment(size=10, obstacle_density=0.0, dynamic_obstacles=dynamic)
    )
    assert incremental.success == full.success
    assert math.isclose(incremental.total_cost, full.total_cost, abs_tol=1e-9)
    assert incremental.node_expansions < full.node_expansions


def test_change_after_move_uses_same_information_boundary():
    dynamic = [
        DynamicObstacle((2, 2), period=1, initial_state="passable"),
    ]
    full = run_naive_replanning(
        GridEnvironment(
            size=5,
            obstacle_density=0.0,
            start=(2, 1),
            goal=(2, 3),
            dynamic_obstacles=dynamic,
        )
    )
    incremental = run_dstar_lite_replanning(
        GridEnvironment(
            size=5,
            obstacle_density=0.0,
            start=(2, 1),
            goal=(2, 3),
            dynamic_obstacles=dynamic,
        )
    )

    assert full.success
    assert incremental.success
    assert full.realized_path == [(2, 1), (2, 2), (2, 3)]
    assert incremental.realized_path == full.realized_path
    assert full.replan_events[1]["step"] == 1
    assert incremental.replan_events[1]["step"] == 1
