"""
sanity_check_her.py

1. Verifies that _bfs_distance() table lookup (O(1)) exactly matches
   the live Dijkstra result for 10 random (start, goal) pairs.
2. Verifies that step() and compute_reward() produce identical rewards
   on real (non-relabeled) transitions.
3. Spot-checks a relabeled goal for eyeball-plausibility.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rl_agent"))
from uav_env import UAVRoutingEnv

# -----------------------------------------------------------------------
# 1. Distance-table accuracy check
# -----------------------------------------------------------------------
print("=" * 60)
print("  1. Distance Table vs Live Dijkstra (10 random pairs)")
print("=" * 60)

env = UAVRoutingEnv(grid_size=15, obstacle_density=0.20, no_fly_density=0.05,
                   fixed_grid=True, seed=42)
env.reset()

assert env._distance_table is not None, "Distance table was not built!"
print(f"Table built: {len(env._distance_table)} source cells precomputed.\n")

rng = np.random.default_rng(seed=0)
free_cells = np.argwhere(env.grid == 0)  # CELL_FREE == 0

all_match = True
for i in range(10):
    pair = rng.choice(len(free_cells), size=2, replace=False)
    start = free_cells[pair[0]]
    goal  = free_cells[pair[1]]

    # Force live search by temporarily hiding the table
    saved_table = env._distance_table
    env._distance_table = None
    live_dist = env._bfs_distance(start, goal)
    env._distance_table = saved_table

    # Now use the table
    table_dist = env._bfs_distance(start, goal)

    match = np.isclose(live_dist, table_dist)
    all_match = all_match and match
    status = "OK  [MATCH]" if match else "MISMATCH [FAIL]"
    print(f"  Pair {i+1:>2d}: ({start[0]:>2d},{start[1]:>2d}) -> ({goal[0]:>2d},{goal[1]:>2d}) | "
          f"live={live_dist:.4f}  table={table_dist:.4f}  {status}")

print()
if all_match:
    print("  ALL 10 pairs match exactly. [PASS]")
else:
    print("  WARNING: mismatches detected! [FAIL]")

# -----------------------------------------------------------------------
# 2. step() vs compute_reward() identity check (real transitions)
# -----------------------------------------------------------------------
print()
print("=" * 60)
print("  2. step() reward vs compute_reward() on real transitions")
print("=" * 60)

obs, info = env.reset()
print(f"\nUAV pos: {info['uav_pos']}, Goal pos: {info['goal_pos']}\n")

all_rewards_match = True
for i in range(5):
    action = i % 8

    obs, reward, terminated, truncated, info = env.step(action)
    achieved_goal = obs["achieved_goal"]
    desired_goal  = obs["desired_goal"]

    cr_reward = env.compute_reward(achieved_goal, desired_goal, info)

    match = np.isclose(reward, cr_reward)
    all_rewards_match = all_rewards_match and match
    status = "OK  [MATCH]" if match else "MISMATCH [FAIL]"
    print(f"  Step {i+1} action={action} | crashed={info['crashed']} | "
          f"step={reward:.6f}  compute_reward={cr_reward:.6f}  {status}")

    # Also show relabeled reward for plausibility
    relabeled_goal = np.array([0.0, 0.0], dtype=np.float32)
    if np.array_equal(np.round(relabeled_goal), np.round(desired_goal)):
        relabeled_goal = np.array([14.0, 14.0], dtype=np.float32)
    cr_relab = env.compute_reward(achieved_goal, relabeled_goal, info)
    print(f"           relabeled goal={relabeled_goal} -> compute_reward={cr_relab:.6f}")

    if terminated or truncated:
        break

print()
if all_rewards_match:
    print("  All step/compute_reward pairs match exactly. [PASS]")
else:
    print("  WARNING: mismatches detected! [FAIL]")
