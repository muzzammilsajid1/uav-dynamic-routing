import math
import unittest

from baselines.astar import astar
from baselines.dijkstra import dijkstra
from baselines.replanning import run_astar_replanning, run_naive_replanning
from envs.grid_environment import DynamicObstacle, GridEnvironment


class TestAStar(unittest.TestCase):
    def test_matches_dijkstra_cost_on_seeded_grids(self):
        for seed in range(5):
            env = GridEnvironment(size=10, obstacle_density=0.15, seed=seed)
            expected = dijkstra(env.start, env.goal, env.get_neighbors)
            actual = astar(env.start, env.goal, env.get_neighbors)
            self.assertEqual(actual.found, expected.found)
            self.assertTrue(math.isclose(actual.cost, expected.cost, abs_tol=1e-9))

    def test_uses_fewer_expansions_on_open_grid(self):
        env = GridEnvironment(size=15, obstacle_density=0.0)
        self.assertLess(
            astar(env.start, env.goal, env.get_neighbors).visited_count,
            dijkstra(env.start, env.goal, env.get_neighbors).visited_count,
        )

    def test_matched_dynamic_replanning_has_optimal_cost(self):
        dynamic = [DynamicObstacle(cell=(4, 4), period=3, initial_state="passable")]
        dijkstra_result = run_naive_replanning(
            GridEnvironment(size=8, obstacle_density=0.0, dynamic_obstacles=dynamic)
        )
        astar_result = run_astar_replanning(
            GridEnvironment(size=8, obstacle_density=0.0, dynamic_obstacles=dynamic)
        )
        self.assertTrue(astar_result.success)
        self.assertAlmostEqual(astar_result.total_cost, dijkstra_result.total_cost)
        self.assertLess(astar_result.node_expansions, dijkstra_result.node_expansions)
