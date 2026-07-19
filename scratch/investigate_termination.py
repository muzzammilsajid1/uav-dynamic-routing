import numpy as np
from rl_agent.train_static_diagnostic import make_env

def main():
    env = make_env(seed=999, is_eval=True)
    base_env = env.unwrapped
    
    # Run to Episode 10
    for _ in range(10):
        obs, info = env.reset()
        
    base_env.goal_pos = np.array([11, 1], dtype=np.int32)
    
    print("==================================================")
    print("1. ACTUAL GRID CONTENTS 5x5 AROUND [11, 1] IN EP 10")
    print("==================================================")
    
    r, c = 11, 1
    radius = 2
    
    for dr in range(-radius, radius + 1):
        gr = r + dr
        print(f"r={gr:<2}", end=" ")
        for dc in range(-radius, radius + 1):
            gc = c + dc
            if 0 <= gr < base_env.grid_size and 0 <= gc < base_env.grid_size:
                cell_val = base_env.grid[gr, gc]
            else:
                cell_val = 1 # boundary is obstacle
            
            if cell_val == 1:
                mark = "X"
            elif cell_val == 2:
                mark = "N"
            else:
                mark = "."
                
            print(f" [{gr},{gc}]={mark} ", end="")
        print()
        
    print("\nSpecific cells requested:")
    for query_pos in [[10,1], [11,2], [9,0], [11,0], [12,1], [10,0], [11,1]]:
        qr, qc = query_pos
        if 0 <= qr < base_env.grid_size and 0 <= qc < base_env.grid_size:
            val = base_env.grid[qr, qc]
        else:
            val = 1
        print(f"  [{qr},{qc}]: {val}")

    print("\n==================================================")
    print("2. TESTING ACTION 'S' FROM [10, 1]")
    print("==================================================")
    
    base_env.uav_pos = np.array([10, 1], dtype=np.int32)
    # Action 'S' is index 1
    action_s = 1
    
    print(f"UAV Pos: {base_env.uav_pos}")
    print(f"Goal Pos: {base_env.goal_pos}")
    print(f"Action S (index 1) delta: {base_env.ACTION_DELTAS[action_s]}")
    
    obs, reward, terminated, truncated, info = env.step(action_s)
    
    print(f"\nAfter Step:")
    print(f"UAV Pos: {base_env.uav_pos}")
    print(f"Terminated: {terminated}")
    print(f"Reward: {reward}")

if __name__ == '__main__':
    main()
