import unittest

from envs.grid_environment import GridEnvironment
from evaluation.metrics import path_cost
from rl_agent.q_learning import ACTION_DELTAS, GridRoutingEnv, evaluate_agent, train_q_learning


class TestQlearning(unittest.TestCase):
    def test_action_mapping_matches_handoff_contract(self):
        self.assertEqual(
            ACTION_DELTAS,
            (
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ),
        )

    def test_invalid_move_crashes_episode(self):
        grid = GridEnvironment(size=3, obstacle_density=0.0, diagonal=True)
        env = GridRoutingEnv(grid=grid)

        state = env.reset()
        next_state, reward, done, info = env.step(0)

        self.assertEqual(state, next_state)
        self.assertLess(reward, 0)
        self.assertTrue(done)
        self.assertTrue(info["crashed"])

    def test_agent_learns_empty_grid(self):
        grid = GridEnvironment(size=5, obstacle_density=0.0, diagonal=True)
        env = GridRoutingEnv(grid=grid, max_steps=20)

        agent, _history = train_q_learning(env, episodes=500, seed=7)
        results = evaluate_agent(env, agent, episodes=5)

        self.assertTrue(all(result.success for result in results))

    def test_path_cost_uses_shared_environment_costs(self):
        grid = GridEnvironment(size=3, obstacle_density=0.0, diagonal=True)
        path = [(0, 0), (1, 1), (2, 2)]

        self.assertAlmostEqual(path_cost(grid, path), 2.8284271247461903)


if __name__ == "__main__":
    unittest.main()
