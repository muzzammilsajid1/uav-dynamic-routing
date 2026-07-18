import numpy as np
from stable_baselines3 import PPO
from rl_agent.train_static_diagnostic import make_env, PROJECT_ROOT

def main():
    env = make_env(seed=999, is_eval=True)
    base_env = env.unwrapped
    
    obs, info = env.reset()
    
    # Force state to reproduce the crash
    base_env.uav_pos = np.array([12, 2], dtype=np.int32)
    base_env.goal_pos = np.array([11, 1], dtype=np.int32)
    
    print("==================================================")
    print("1. ACTUAL GRID CONTENTS AROUND [12, 2]")
    print("==================================================")
    
    r, c = base_env.uav_pos
    radius = 3
    print("    ", end="")
    for dc in range(-radius, radius + 1):
        gc = c + dc
        print(f" c={gc:<2}", end="")
    print()
    
    for dr in range(-radius, radius + 1):
        gr = r + dr
        print(f"r={gr:<2}", end=" ")
        for dc in range(-radius, radius + 1):
            gc = c + dc
            if 0 <= gr < base_env.grid_size and 0 <= gc < base_env.grid_size:
                cell_val = base_env.grid[gr, gc]
            else:
                cell_val = 1 # boundary is obstacle
            
            # Highlight UAV, Goal, and Obstacles
            if gr == r and gc == c:
                mark = "[U]"
            elif gr == 11 and gc == 1:
                mark = "[G]"
            elif cell_val == 1:
                mark = " X "
            elif cell_val == 2:
                mark = " N "
            else:
                mark = " . "
            print(f"{mark:>4}", end="")
        print()
        
    print("\n==================================================")
    print("2. _build_observation() LOCAL GRID OUTPUT")
    print("==================================================")
    
    # Get just the local grid portion
    # obs = [6 positional] + [49 local grid] + [8 one-hot actions]
    local_obs = base_env._get_local_observation()
    
    print("Flattened array length:", len(local_obs))
    
    # Reconstruct back to 7x7 to verify visually
    local_2d = local_obs.reshape((7, 7))
    
    print("\nReshaped 7x7 observation (should match actual grid):")
    for row in range(7):
        for col in range(7):
            val = int(local_2d[row, col])
            if val == 1:
                mark = " X "
            elif val == 2:
                mark = " N "
            elif row == 3 and col == 3:
                mark = "[U]"
            elif row == 3 - 1 and col == 3 - 1: # NW is -1, -1
                # Wait, goal might not be marked specially in local_obs since it just shows cell values
                if val == 0:
                   mark = " G "
                else:
                   mark = f" {val} " 
            else:
                mark = " . "
            print(f"{mark:>4}", end="")
        print()
        
    print("\nMapping of indices:")
    print("Index 0  -> dr=-3, dc=-3 -> row={}, col={}".format(r-3, c-3))
    print("Index 6  -> dr=-3, dc=+3 -> row={}, col={}".format(r-3, c+3))
    print("Index 24 -> dr=0, dc=0   -> row={}, col={} (UAV)".format(r, c))
    print("Index 48 -> dr=+3, dc=+3 -> row={}, col={}".format(r+3, c+3))
    
    print("\nGoal cell [11, 1] is at index 16 (dr=-1, dc=-1)")
    print(f"Goal cell value in obs: {local_obs[16]}")
    
    print("NE cell [11, 3] is at index 18 (dr=-1, dc=+1)")
    print(f"NE cell value in obs: {local_obs[18]}")
    
    print("\n==================================================")
    print("3. DIAGNOSIS")
    print("==================================================")
    print(f"Actual grid at [11, 1] (Goal/NW): {base_env.grid[11, 1]}")
    print(f"Actual grid at [11, 3] (NE): {base_env.grid[11, 3]}")

if __name__ == '__main__':
    main()
