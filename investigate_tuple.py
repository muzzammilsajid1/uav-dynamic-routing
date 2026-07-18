import numpy as np
from rl_agent.train_static_diagnostic import make_env

def main():
    env = make_env(seed=999, is_eval=True)
    base_env = env.unwrapped
    
    base_env.reset()
    base_env.uav_pos = np.array([12, 2], dtype=np.int32)
    
    # Simulate step for Action 1 (NE)
    action = 1
    delta = base_env.ACTION_DELTAS[action]
    next_pos = base_env.uav_pos + delta
    
    neighbors = base_env.get_neighbors(base_env.uav_pos)
    valid_next_positions = [tuple(n[0]) for n in neighbors]
    
    print(f"next_pos: {next_pos}, type: {type(next_pos)}, dtype: {next_pos.dtype}")
    print(f"tuple(next_pos): {tuple(next_pos)}, types: {[type(x) for x in tuple(next_pos)]}")
    
    print(f"valid_next_positions: {valid_next_positions}")
    print(f"types in first valid: {[type(x) for x in valid_next_positions[0]]}")
    
    if tuple(next_pos) not in valid_next_positions:
        print("RESULT: NOT IN VALID_NEXT_POSITIONS (Collision)")
    else:
        print("RESULT: IT IS VALID! No collision.")

if __name__ == '__main__':
    main()
