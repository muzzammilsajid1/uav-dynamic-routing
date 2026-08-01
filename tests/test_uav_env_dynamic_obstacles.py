import numpy as np
from rl_agent.uav_env import UAVRoutingEnv, CELL_FREE, CELL_OBSTACLE, LOCAL_VIEW_RADIUS
from envs.grid_environment import DynamicObstacle

def test_dynamic_obstacle_toggles_and_updates_observation():
    env = UAVRoutingEnv(
        grid_size=15,
        dynamic_obstacles_enabled=True,
        dynamic_obstacles=[DynamicObstacle(cell=(5, 5), period=1, initial_state="passable")],
        fixed_grid=True,
        obstacle_density=0.0  # Crucial: prevent random static obstacles from blocking our test movements
    )
    
    # We must reset to initialize the internal grid
    env.reset()
    
    # Force UAV to a known position adjacent to the dynamic cell
    env.uav_pos = np.array([5, 4])
    # Keep the tested move non-terminal.  reset() chooses a random goal, and
    # occasionally selected (5, 3), causing the westward move below to reach
    # the goal.  Terminal transitions intentionally suppress post-move
    # dynamics because that state would never be observed by the policy.
    env.goal_pos = np.array([14, 14])
    
    # helper for checking if (5,5) is in get_neighbors of (5,4)
    def check_neighbor_free():
        neighbors = env.get_neighbors(np.array([5, 4]))
        return (5, 5) in [tuple(n[0]) for n in neighbors]

    # INITIAL: OFF (FREE)
    obs = env._get_local_observation()
    east_idx = LOCAL_VIEW_RADIUS * (2 * LOCAL_VIEW_RADIUS + 1) + (LOCAL_VIEW_RADIUS + 1)
    assert obs[east_idx] == CELL_FREE, f"Expected cell (5,5) to be FREE initially, got {obs[east_idx]}"
    assert check_neighbor_free() == True, "Cell (5,5) should be a valid neighbor when FREE"
    
    # STEP 1: ON (OBSTACLE)
    # Action 2 is West (moves away to (5, 3))
    _, _, _, _, info = env.step(2)
    assert info["dynamic_changes"] == [[5, 5]]
    next_local_grid = env._get_local_observation()
    # Now UAV is at (5, 3), so (5, 5) is 2 cells East
    east_east_idx = LOCAL_VIEW_RADIUS * (2 * LOCAL_VIEW_RADIUS + 1) + (LOCAL_VIEW_RADIUS + 2)
    assert next_local_grid[east_east_idx] == CELL_OBSTACLE, f"Expected cell (5,5) to toggle to OBSTACLE, got {next_local_grid[east_east_idx]}"
    assert check_neighbor_free() == False, "Cell (5,5) should NOT be a valid neighbor when OBSTACLE"
    
    # STEP 2: OFF (FREE)
    env.step(2)
    next_local_grid_2 = env._get_local_observation()
    # Now UAV is at (5, 2), so (5, 5) is 3 cells East
    east_east_east_idx = LOCAL_VIEW_RADIUS * (2 * LOCAL_VIEW_RADIUS + 1) + (LOCAL_VIEW_RADIUS + 3)
    assert next_local_grid_2[east_east_east_idx] == CELL_FREE, f"Expected cell (5,5) to toggle back to FREE, got {next_local_grid_2[east_east_east_idx]}"
    assert check_neighbor_free() == True, "Cell (5,5) should be a valid neighbor again when FREE"

    # STEP 3: ON (OBSTACLE) again - full cycle
    env.step(3) # Action 3 is East, move to (5, 3)
    next_local_grid_3 = env._get_local_observation()
    assert next_local_grid_3[east_east_idx] == CELL_OBSTACLE, f"Expected cell (5,5) to toggle back to OBSTACLE, got {next_local_grid_3[east_east_idx]}"
    assert check_neighbor_free() == False, "Cell (5,5) should NOT be a valid neighbor when OBSTACLE"


def test_reset_raises_runtime_error_when_sealed():
    # Make all cells EXCEPT (0,0) and (14,14) dynamic toggle cells.
    toggle_cells = []
    for r in range(15):
        for c in range(15):
            if (r, c) not in [(0, 0), (14, 14)]:
                toggle_cells.append(DynamicObstacle(cell=(r, c), period=5, initial_state="passable"))
                
    env = UAVRoutingEnv(
        grid_size=15,
        dynamic_obstacles_enabled=True,
        dynamic_obstacles=toggle_cells,
        fixed_grid=True,
        obstacle_density=0.0
    )
    
    # reset() will attempt to place the UAV and Goal, verify reachability, fail, retry 50 times, and raise.
    try:
        env.reset()
    except RuntimeError as exc:
        assert "Could not find a start/goal pair not sealed off by dynamic obstacles." in str(exc)
    else:
        raise AssertionError("Expected reset() to raise RuntimeError for a sealed grid")


def test_multiple_obstacles_toggle_independently():
    env = UAVRoutingEnv(
        grid_size=15,
        dynamic_obstacles_enabled=True,
        dynamic_obstacles=[
            DynamicObstacle(cell=(1, 1), period=2, initial_state="passable"),
            DynamicObstacle(cell=(2, 2), period=3, initial_state="passable")
        ],
        fixed_grid=True,
        obstacle_density=0.0
    )
    env.reset()
    # Exercise four non-terminal transitions deterministically.  Leaving the
    # random reset positions in place made an eastward step occasionally hit
    # the goal or grid boundary, where post-terminal dynamics are correctly
    # suppressed.
    env.uav_pos = np.array([7, 5])
    env.goal_pos = np.array([14, 14])
    
    # Step 1: elapsed=1 (neither toggles)
    env.step(3) # Move somewhere
    assert env.grid[1, 1] == CELL_FREE
    assert env.grid[2, 2] == CELL_FREE
    
    # Step 2: elapsed=2 ((1,1) toggles)
    env.step(3)
    assert env.grid[1, 1] == CELL_OBSTACLE
    assert env.grid[2, 2] == CELL_FREE
    
    # Step 3: elapsed=3 ((2,2) toggles)
    env.step(3)
    assert env.grid[1, 1] == CELL_OBSTACLE  # (1,1) is on period 2, so it stays OBSTACLE (next toggle at 4)
    assert env.grid[2, 2] == CELL_OBSTACLE
    
    # Step 4: elapsed=4 ((1,1) toggles back)
    env.step(3)
    assert env.grid[1, 1] == CELL_FREE
    assert env.grid[2, 2] == CELL_OBSTACLE


def test_post_move_change_is_observed_before_next_decision():
    env = UAVRoutingEnv(
        grid_size=5,
        dynamic_obstacles_enabled=True,
        dynamic_obstacles=[
            DynamicObstacle(cell=(2, 2), period=1, initial_state="passable")
        ],
        fixed_grid=True,
        obstacle_density=0.0,
        dynamics_timing="post_move_observed",
    )
    env.reset(seed=7)
    env.uav_pos = np.array([2, 1], dtype=np.int32)
    env.goal_pos = np.array([2, 3], dtype=np.int32)

    observation, _, terminated, truncated, info = env.step(3)

    assert not terminated
    assert not truncated
    assert tuple(env.uav_pos) == (2, 2)
    assert env.grid[2, 2] == CELL_OBSTACLE
    assert info["dynamic_changes"] == [[2, 2]]
    center = LOCAL_VIEW_RADIUS * (2 * LOCAL_VIEW_RADIUS + 1) + LOCAL_VIEW_RADIUS
    assert observation["observation"][4 + center] == CELL_OBSTACLE

    _, _, terminated, _, info = env.step(3)
    assert terminated
    assert info["is_success"]
    assert tuple(env.uav_pos) == (2, 3)
    assert info["dynamic_changes"] == []
