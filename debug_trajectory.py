import numpy as np
from stable_baselines3 import DQN
from rl_agent.train_static_diagnostic import make_env, PROJECT_ROOT
from rl_agent.uav_env import REWARD_STEP, REWARD_GOAL, REWARD_COLLISION
import argparse

def main():
    parser = argparse.ArgumentParser(description="Debug UAV routing trajectory.")
    parser.add_argument("model", type=str, nargs="?", 
                        default=str(PROJECT_ROOT / "models" / "dqn_static_uav_1_5M_final.zip"),
                        help="Path to the model file.")
    args = parser.parse_args()
    
    model_path = args.model
    print(f"Loading model from {model_path}...")
    model = DQN.load(model_path)

    # Use a fixed seed to ensure reproducible debugging and pass is_eval=True to disable curriculum
    env = make_env(seed=999, is_eval=True)
    base_env = env.unwrapped
    
    action_names = ["N", "S", "W", "E", "NW", "NE", "SW", "SE"]
    
    print("=" * 105)
    print("  Trajectory Debugging (10 Episodes)")
    print("=" * 105)
    
    for ep in range(1, 11):
        obs, info = env.reset()
        
        # Force goal to [11, 1] and recompute distance
        base_env.goal_pos = np.array([11, 1], dtype=np.int32)
        base_env.previous_distance = base_env._bfs_distance(base_env.uav_pos, base_env.goal_pos)
        
        done = False
        step_count = 0
        total_ep_reward = 0.0
        
        initial_uav_pos = base_env.uav_pos.copy()
        initial_goal_pos = base_env.goal_pos.copy()
        initial_manhattan = int(np.sum(np.abs(initial_uav_pos - initial_goal_pos)))
        
        trace_lines = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action_idx = int(action)
            
            pos_before = base_env.uav_pos.copy()
            dist_before = base_env.previous_distance
            
            obs, reward, terminated, truncated, info = env.step(action_idx)
            
            pos_after = base_env.uav_pos.copy()
            dist_after = base_env._bfs_distance(pos_after, base_env.goal_pos)
            
            step_penalty = REWARD_STEP
            dense_rew = reward - step_penalty
            
            if terminated and reward >= REWARD_GOAL + step_penalty:
                dense_rew -= REWARD_GOAL
            elif terminated and reward <= REWARD_COLLISION + step_penalty:
                dense_rew -= REWARD_COLLISION
                
            trace_line = f"{step_count+1:>4} | {str(pos_before):>12} | {action_names[action_idx]:>6} | {dist_before:>11.4f} | {dist_after:>10.4f} | {dense_rew:>9.4f} | {step_penalty:>8.1f} | {reward:>9.4f}"
            trace_lines.append(trace_line)
            
            total_ep_reward += reward
            step_count += 1
            done = terminated or truncated

        if total_ep_reward > 0:
            outcome = "SUCCESS"
        elif terminated:
            outcome = "COLLISION"
        else:
            outcome = "TIMEOUT"
            
        print(f"\nEpisode {ep} | Outcome: {outcome} | Initial Manhattan Distance: {initial_manhattan} | Total Reward: {total_ep_reward:.4f}")
        
        print("-" * 105)
        print(f"Goal Position: {initial_goal_pos}")
        print(f"{'Step':>4} | {'Pos (r, c)':>12} | {'Action':>6} | {'Dist Before':>11} | {'Dist After':>10} | {'Dense Rew':>9} | {'Step Pen':>8} | {'Total Rew':>9}")
        print("-" * 105)
        for line in trace_lines:
            print(line)
        print("-" * 105)

if __name__ == '__main__':
    main()
