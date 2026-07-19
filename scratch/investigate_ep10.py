import numpy as np
from rl_agent.train_static_diagnostic import make_env

def main():
    env = make_env(seed=999, is_eval=True)
    base_env = env.unwrapped
    
    # Run to Episode 10
    for _ in range(10):
        obs, info = env.reset()
        
    base_env.goal_pos = np.array([11, 1], dtype=np.int32)
    # The UAV happens to be placed at [12, 2] by the random placement in ep 10?
    # Let's see where it actually is:
    print(f"Episode 10 Actual UAV Pos: {base_env.uav_pos}")
    
    pos = np.array([12, 2], dtype=np.int32)
    
    print(f"Grid at {pos} (UAV): {base_env.grid[pos[0], pos[1]]}")
    print(f"Grid at [12, 3] (East): {base_env.grid[12, 3]}")
    print(f"Grid at [11, 2] (North): {base_env.grid[11, 2]}")
    print(f"Grid at [11, 3] (NE): {base_env.grid[11, 3]}")
    
    neighbors = base_env.get_neighbors(pos)
    print("\nNeighbors of [12, 2]:")
    for n, cost in neighbors:
        print(f"  Pos: {n}, Cost: {cost}")
        
    ne_in_neighbors = any(np.array_equal(n, [11, 3]) for n, cost in neighbors)
    print(f"\nIs [11, 3] (NE) in neighbors? {ne_in_neighbors}")
    
    # Also dump local grid obs just to see what the agent saw in ep 10
    local_obs = base_env._get_local_observation().reshape((7, 7))
    print("\nLocal Observation in Episode 10 (centered at 12, 2):")
    for r in range(7):
        for c in range(7):
            print(f" {int(local_obs[r, c])} ", end="")
        print()

if __name__ == '__main__':
    main()
