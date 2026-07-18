import argparse
import numpy as np
from stable_baselines3 import DQN
from rl_agent.train_static_diagnostic import make_env, PROJECT_ROOT
from rl_agent.uav_env import UAVRoutingEnv

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN UAV routing model.")
    parser.add_argument(
        "--model", 
        type=str, 
        default="models/dqn_static_uav_1_5M_final.zip",
        help="Path to the trained model zip file."
    )
    args = parser.parse_args()

    model_path = PROJECT_ROOT / args.model
    print(f"Loading model from {model_path}...")
    
    dummy_env = UAVRoutingEnv(grid_size=15, obstacle_density=0.20, no_fly_density=0.05, fixed_grid=True)
    model = DQN.load(model_path, env=dummy_env)

    env = make_env(seed=999)
    n_episodes = 20

    episode_rewards = []
    episode_lengths = []
    outcomes = {"SUCCESS": 0, "COLLISION": 0, "TIMEOUT": 0}
    success_lengths = []

    print("Evaluating...")
    for ep in range(1, n_episodes + 1):
        obs, info = env.reset()
        total_reward = 0.0
        done = False
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            steps += 1
            done = terminated or truncated

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        crashed = info.get("crashed", False)
        ag = np.round(obs["achieved_goal"]).astype(np.int32)
        dg = np.round(obs["desired_goal"]).astype(np.int32)
        
        if crashed:
            outcomes["COLLISION"] += 1
        elif np.array_equal(ag, dg):
            outcomes["SUCCESS"] += 1
            success_lengths.append(steps)
        else:
            outcomes["TIMEOUT"] += 1

    print("\n--- Evaluation Results ---")
    print(f"Success Rate: {outcomes['SUCCESS']}/{n_episodes} ({(outcomes['SUCCESS']/n_episodes)*100:.1f}%)")
    print(f"Average Episode Reward: {np.mean(episode_rewards):.2f}")
    if outcomes["SUCCESS"] > 0:
        print(f"Average Path Length (Successful): {np.mean(success_lengths):.2f} steps")
    else:
        print("Average Path Length (Successful): N/A (0 successes)")

    print("\nOutcome Breakdown:")
    print(f"  Success (Goal Reached): {outcomes['SUCCESS']}")
    print(f"  Collision (Crash):      {outcomes['COLLISION']}")
    print(f"  Timeout (Max Steps):    {outcomes['TIMEOUT']}")

if __name__ == '__main__':
    main()
