import numpy as np
from rl_agent.uav_env import UAVRoutingEnv

def main():
    env = UAVRoutingEnv(fixed_grid=True, seed=42)
    env.reset()
    
    print("Sanity Check: get_neighbors")
    start_pos = np.array([7, 7])
    neighbors = env.get_neighbors(start_pos)
    print(f"Neighbors for {start_pos}:")
    for n, c in neighbors:
        print(f"  Pos: {n}, Cost: {c:.4f}")
        
    print("\nSanity Check: Dijkstra Distance")
    # Clear grid to ensure a clear path for the test
    env.grid.fill(0)
    
    # 3 hops straight
    cost_straight = env._bfs_distance(np.array([0, 0]), np.array([0, 3]))
    # 3 hops diagonal
    cost_diag = env._bfs_distance(np.array([0, 0]), np.array([3, 3]))
    
    print(f"Distance [0,0] to [0,3] (3 straight): {cost_straight:.4f} (Expected: 3.0000)")
    print(f"Distance [0,0] to [3,3] (3 diagonal): {cost_diag:.4f} (Expected: 4.2426)")

if __name__ == '__main__':
    main()
