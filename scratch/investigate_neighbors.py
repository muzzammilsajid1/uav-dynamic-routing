import numpy as np
from rl_agent.train_static_diagnostic import make_env

def main():
    env = make_env(seed=999, is_eval=True)
    base_env = env.unwrapped
    
    obs, info = env.reset()
    
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

if __name__ == '__main__':
    main()
