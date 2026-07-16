import math
import unittest

try:
    import networkx as nx
except ImportError:
    nx = None

from baselines.dijkstra import dijkstra
from envs.grid_environment import GridEnvironment


class TestDijkstra(unittest.TestCase):
    def test_finds_path_on_empty_3_by_3_grid(self):
        env = GridEnvironment(size=3, obstacle_density=0.0, diagonal=True)

        result = dijkstra(env.start, env.goal, env.get_neighbors)

        self.assertTrue(result.found)
        self.assertEqual(result.path[0], env.start)
        self.assertEqual(result.path[-1], env.goal)
        self.assertAlmostEqual(result.cost, 2 * math.sqrt(2))

    def test_returns_no_path_when_goal_is_enclosed(self):
        env = GridEnvironment(
            size=3,
            obstacle_density=0.0,
            start=(0, 0),
            goal=(2, 2),
            diagonal=False,
            blocked={(1, 2), (2, 1)},
        )

        result = dijkstra(env.start, env.goal, env.get_neighbors)

        self.assertFalse(result.found)
        self.assertEqual(result.cost, float("inf"))

    def test_weighted_graph_prefers_lower_cost_not_fewer_edges(self):
        graph = {
            (0, 0): [((0, 1), 10.0), ((1, 0), 1.0)],
            (1, 0): [((1, 1), 1.0)],
            (1, 1): [((0, 1), 1.0)],
            (0, 1): [],
        }

        result = dijkstra((0, 0), (0, 1), lambda node: graph.get(node, []))

        self.assertEqual(result.path, [(0, 0), (1, 0), (1, 1), (0, 1)])
        self.assertEqual(result.cost, 3.0)

    def test_rejects_negative_edge_weights(self):
        graph = {
            (0, 0): [((0, 1), -1.0)],
            (0, 1): [],
        }

        with self.assertRaises(ValueError):
            dijkstra((0, 0), (0, 1), lambda node: graph.get(node, []))

    def test_matches_networkx_cost_on_seeded_random_grids(self):
        if nx is None:
            self.skipTest("networkx is not installed")

        for seed in range(5):
            env = GridEnvironment(size=8, obstacle_density=0.15, seed=seed)
            graph = env.to_networkx_graph()
            result = dijkstra(env.start, env.goal, env.get_neighbors)

            try:
                expected_cost = nx.dijkstra_path_length(
                    graph,
                    env.start,
                    env.goal,
                    weight="weight",
                )
            except nx.NetworkXNoPath:
                expected_cost = float("inf")

            self.assertAlmostEqual(result.cost, expected_cost)


if __name__ == "__main__":
    unittest.main()
