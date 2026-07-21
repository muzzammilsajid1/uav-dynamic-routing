import unittest

from baselines.replanning import run_naive_replanning
from envs.grid_environment import DynamicObstacle, GridEnvironment


class TestDynamicObstacles(unittest.TestCase):
    def test_initial_state_blocked_applies_immediately(self):
        env = GridEnvironment(
            size=5,
            obstacle_density=0.0,
            dynamic_obstacles=[DynamicObstacle(cell=(2, 2), period=3, initial_state="blocked")],
        )
        self.assertTrue(env.is_blocked((2, 2)))

    def test_initial_state_passable_is_not_blocked(self):
        env = GridEnvironment(
            size=5,
            obstacle_density=0.0,
            dynamic_obstacles=[DynamicObstacle(cell=(2, 2), period=3, initial_state="passable")],
        )
        self.assertFalse(env.is_blocked((2, 2)))

    def test_toggles_only_on_period_boundary(self):
        env = GridEnvironment(
            size=5,
            obstacle_density=0.0,
            dynamic_obstacles=[DynamicObstacle(cell=(2, 2), period=3, initial_state="passable")],
        )
        for step in (1, 2):
            changed = env.step_dynamics()
            self.assertEqual(changed, set())
            self.assertFalse(env.is_blocked((2, 2)))

        changed = env.step_dynamics()  # step 3
        self.assertEqual(changed, {(2, 2)})
        self.assertTrue(env.is_blocked((2, 2)))

    def test_toggles_back_on_next_period_boundary(self):
        env = GridEnvironment(
            size=5,
            obstacle_density=0.0,
            dynamic_obstacles=[DynamicObstacle(cell=(2, 2), period=2, initial_state="passable")],
        )
        env.step_dynamics()  # step 1
        env.step_dynamics()  # step 2 -> blocked
        self.assertTrue(env.is_blocked((2, 2)))
        env.step_dynamics()  # step 3
        env.step_dynamics()  # step 4 -> passable again
        self.assertFalse(env.is_blocked((2, 2)))

    def test_reset_dynamics_restores_initial_state_and_clock(self):
        env = GridEnvironment(
            size=5,
            obstacle_density=0.0,
            dynamic_obstacles=[DynamicObstacle(cell=(2, 2), period=2, initial_state="passable")],
        )
        env.step_dynamics()
        env.step_dynamics()
        self.assertTrue(env.is_blocked((2, 2)))
        self.assertEqual(env.elapsed_steps, 2)

        env.reset_dynamics()
        self.assertFalse(env.is_blocked((2, 2)))
        self.assertEqual(env.elapsed_steps, 0)

    def test_dynamic_obstacle_cannot_be_start_or_goal(self):
        with self.assertRaises(ValueError):
            GridEnvironment(
                size=5,
                obstacle_density=0.0,
                start=(0, 0),
                goal=(4, 4),
                dynamic_obstacles=[DynamicObstacle(cell=(0, 0), period=2)],
            )


class TestNaiveReplanning(unittest.TestCase):
    def test_succeeds_with_no_dynamic_obstacles_matches_static_dijkstra(self):
        env = GridEnvironment(size=8, obstacle_density=0.1, seed=1)
        result = run_naive_replanning(env)
        self.assertTrue(result.success)
        self.assertEqual(result.replans, 1)  # only the initial plan, nothing ever changes
        self.assertEqual(result.realized_path[0], env.start)
        self.assertEqual(result.realized_path[-1], env.goal)

    def test_replans_when_toggle_lands_ahead_on_current_plan(self):
        env = GridEnvironment(
            size=8,
            obstacle_density=0.0,
            dynamic_obstacles=[DynamicObstacle(cell=(4, 4), period=2, initial_state="passable")],
        )
        result = run_naive_replanning(env)
        self.assertTrue(result.success)
        self.assertGreaterEqual(result.replans, 2)

    def test_does_not_replan_for_toggles_behind_the_uav(self):
        # A toggle far from the direct path, with a long period, should
        # not force any extra replanning beyond the initial plan.
        env = GridEnvironment(
            size=8,
            obstacle_density=0.0,
            start=(0, 0),
            goal=(7, 7),
            dynamic_obstacles=[DynamicObstacle(cell=(7, 0), period=100, initial_state="passable")],
        )
        result = run_naive_replanning(env)
        self.assertTrue(result.success)
        self.assertEqual(result.replans, 1)

    def test_realized_path_cost_matches_manual_edge_sum(self):
        env = GridEnvironment(size=6, obstacle_density=0.0)
        result = run_naive_replanning(env)
        # Recompute independently using a fresh, un-stepped env of the
        # same static layout to sanity-check the cost bookkeeping.
        fresh = GridEnvironment(size=6, obstacle_density=0.0)
        manual_total = 0.0
        for a, b in zip(result.realized_path, result.realized_path[1:]):
            manual_total += dict(fresh.get_neighbors(a))[b]
        self.assertAlmostEqual(manual_total, result.total_cost, places=6)


if __name__ == "__main__":
    unittest.main()
