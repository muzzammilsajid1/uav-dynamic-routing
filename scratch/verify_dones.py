import numpy as np
from rl_agent.uav_env import UAVRoutingEnv
from stable_baselines3 import DQN
from stable_baselines3.her import HerReplayBuffer
import torch

def test_step():
    env = UAVRoutingEnv(fixed_grid=True, seed=42)
    obs, info = env.reset()
    env.unwrapped.uav_pos = np.array([7, 4])
    env.unwrapped.goal_pos = np.array([7, 5])
    
    # Sync internal state
    obs = env.unwrapped._build_observation()
    
    print("\n" + "="*60)
    print(" 1. EXACT OUTPUT FROM env.step(3) [East] from [7,4] -> [7,5]")
    print("="*60)
    next_obs, reward, terminated, truncated, info = env.step(3)
    print(f"Action        : 3 (East)")
    print(f"Next UAV Pos  : {env.unwrapped.uav_pos}")
    print(f"Goal Pos      : {env.unwrapped.goal_pos}")
    print(f"Reward        : {reward}")
    print(f"Terminated    : {terminated}")
    print(f"Truncated     : {truncated}")

def test_buffer():
    from rl_agent.safe_her_buffer import SafeHerReplayBuffer
    env = UAVRoutingEnv(fixed_grid=True, seed=42)
    model = DQN(
        "MultiInputPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=10_000,
        learning_starts=100,
        batch_size=256,
        replay_buffer_class=SafeHerReplayBuffer,
        replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
        seed=42,
        device="cpu"
    )
    print("\n" + "="*60)
    print(" 2. REPLAY BUFFER SAMPLED DONES (Goal-Reaching Transitions)")
    print("="*60)
    # Train enough to populate the buffer
    model.learn(total_timesteps=1000, progress_bar=False)
    
    found_count = 0
    # Sample from the buffer repeatedly to find transitions where the goal was reached
    for _ in range(50):
        samples = model.replay_buffer.sample(256)
        ag_next = samples.next_observations["achieved_goal"].numpy()
        dg = samples.observations["desired_goal"].numpy()
        dones = samples.dones.numpy()
        rewards = samples.rewards.numpy()
        
        for i in range(256):
            if np.allclose(ag_next[i], dg[i]):
                print(f"Found goal-reaching transition:")
                print(f"  Next Achieved Goal : {ag_next[i]}")
                print(f"  Desired Goal       : {dg[i]}")
                print(f"  Replay 'dones' flag: {dones[i].item()}")
                print(f"  Replay reward      : {rewards[i].item()}")
                print("-" * 40)
                found_count += 1
                if found_count >= 3:
                    return

if __name__ == "__main__":
    test_step()
    test_buffer()
