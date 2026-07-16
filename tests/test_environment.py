import unittest

from envs.grid_environment import GridEnvironment


class TestGridEnvironment(unittest.TestCase):
    def test_start_and_goal_are_never_blocked(self):
        env = GridEnvironment(size=15, obstacle_density=0.9, seed=1)

        self.assertNotIn(env.start, env.blocked)
        self.assertNotIn(env.goal, env.blocked)

    def test_neighbors_are_inside_bounds_and_not_blocked(self):
        env = GridEnvironment(size=5, obstacle_density=0.0, blocked={(0, 1)})

        for neighbor, _ in env.get_neighbors((0, 0)):
            self.assertTrue(env.in_bounds(neighbor))
            self.assertFalse(env.is_blocked(neighbor))

    def test_diagonal_movement_has_eight_neighbors_from_center(self):
        env = GridEnvironment(size=3, obstacle_density=0.0, diagonal=True)

        self.assertEqual(len(env.get_neighbors((1, 1))), 8)

    def test_four_direction_movement_has_four_neighbors_from_center(self):
        env = GridEnvironment(size=3, obstacle_density=0.0, diagonal=False)

        self.assertEqual(len(env.get_neighbors((1, 1))), 4)


if __name__ == "__main__":
    unittest.main()

