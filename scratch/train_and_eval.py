import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../rl_agent")))

import time
import numpy as np
from stable_baselines3 import DQN
from rl_agent.uav_env import UAVRoutingEnv
from rl_agent.safe_her_buffer import SafeHerReplayBuffer
from double_dqn import DoubleDQN

def main():
    print("=" * 60)
    # 1. Train the model for 40,000 steps
    print("Training DoubleDQN + HER on no_fly_density=0.0 for 40,000 steps...")
    print("=" * 60)
    
    env = UAVRoutingEnv(
        grid_size=15,
        obstacle_density=0.20,
        no_fly_density=0.0,
        fixed_grid=True,
        seed=42
    )
    
    model = DoubleDQN(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=256,
        tau=1.0,
        gamma=0.99,
        train_freq=2,
        gradient_steps=1,
        target_update_interval=1_000,
        exploration_fraction=0.50,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[128, 128]),
        replay_buffer_class=SafeHerReplayBuffer,
        replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
        verbose=1,
        seed=42,
        device="cpu"
    )
    
    t0 = time.time()
    model.learn(total_timesteps=40_000, progress_bar=False)
    t1 = time.time()
    print(f"Training took {t1 - t0:.1f} seconds.")
    
    # 2. Run Cell 4's Evaluation Loop (20 episodes, deterministic)
    print("\n" + "=" * 60)
    print("Running Cell 4's Evaluation (20 episodes, deterministic, corrected classification)")
    print("=" * 60)
    
    eval_env = UAVRoutingEnv(
        grid_size=15,
        obstacle_density=0.20,
        no_fly_density=0.0,
        fixed_grid=True,
        seed=42
    )
    
    N_EPISODES = 20
    goals, crashes, timeouts = 0, 0, 0
    
    for ep in range(1, N_EPISODES + 1):
        obs, info = eval_env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(int(action))
            total_reward += reward
            steps += 1
            done = terminated or truncated
            
        crashed = info.get("crashed", False)
        ag = np.round(obs["achieved_goal"]).astype(np.int32)
        dg = np.round(obs["desired_goal"]).astype(np.int32)
        
        if crashed:
            outcome = "CRASH"
            crashes += 1
        elif np.array_equal(ag[:2], dg[:2]):  # corrected match
            outcome = "GOAL"
            goals += 1
        else:
            outcome = "TIMEOUT"
            timeouts += 1
            
        print(f"  Ep {ep:>2d} | {outcome:<7} | Steps: {steps:>4d} | Reward: {total_reward:>8.3f}")
        
    print()
    print("=" * 60)
    print(f"  GOAL:    {goals}/{N_EPISODES}")
    print(f"  CRASH:   {crashes}/{N_EPISODES}")
    print(f"  TIMEOUT: {timeouts}/{N_EPISODES}")
    print("=" * 60)
    
    eval_env.close()
    env.close()

if __name__ == "__main__":
    main()
