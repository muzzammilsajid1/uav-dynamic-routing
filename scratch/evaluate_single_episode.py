import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../rl_agent")))

import time
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import DQN
from rl_agent.uav_env import UAVRoutingEnv

class DoubleDQN(DQN):
    """
    Subclass of SB3's DQN that implements Double DQN.
    """
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with torch.no_grad():
                next_q_values_online = self.q_net(replay_data.next_observations)
                next_actions = next_q_values_online.argmax(dim=1, keepdim=True)
                
                next_q_values_target = self.q_net_target(replay_data.next_observations)
                next_q_values = torch.gather(next_q_values_target, dim=1, index=next_actions)
                
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = torch.gather(current_q_values, dim=1, index=replay_data.actions.long())

            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

def main():
    # 1. Reconstruct environment exactly as used for training (seed=42, fixed_grid=True)
    env = UAVRoutingEnv(
        grid_size=15,
        obstacle_density=0.20,
        no_fly_density=0.0,
        fixed_grid=True,
        seed=42
    )
    
    # 2. Reset once and get initial positions
    obs, info = env.reset()
    start_pos = info["uav_pos"]
    goal_pos = info["goal_pos"]
    
    print("=" * 60)
    print(f"Initial positions for seed=42:")
    print(f"  Start (uav_pos): {start_pos}")
    print(f"  Goal (goal_pos): {goal_pos}")
    print("=" * 60)
    
    # 3. Load the verified 300k model
    model_path = os.path.join("models", "dqn_her_300k_final.zip")
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found!")
        return
        
    model = DoubleDQN.load(model_path, env=env)
    print("Model loaded successfully!")
    print("=" * 60)
    
    # 4. Roll out one episode and collect path details
    done = False
    path = [list(start_pos)]
    actions_taken = []
    total_reward = 0.0
    steps = 0
    
    # Loop variables
    current_obs = obs
    current_info = info
    
    while not done:
        action, _ = model.predict(current_obs, deterministic=True)
        actions_taken.append(int(action))
        
        # Step the environment
        next_obs, reward, terminated, truncated, next_info = env.step(int(action))
        total_reward += reward
        steps += 1
        done = terminated or truncated
        
        path.append(list(next_info["uav_pos"]))
        current_obs = next_obs
        current_info = next_info

    # 5. Compute path cost using straight (1.0) and diagonal (sqrt(2)) rules
    path_cost = 0.0
    for act in actions_taken:
        if act in [0, 1, 2, 3]:
            path_cost += 1.0
        elif act in [4, 5, 6, 7]:
            path_cost += np.sqrt(2)
            
    crashed = current_info.get("crashed", False)
    ag = np.round(current_obs["achieved_goal"]).astype(np.int32)
    dg = np.round(current_obs["desired_goal"]).astype(np.int32)
    
    reached_goal = not crashed and np.array_equal(ag[:2], dg[:2])
    
    print("\nEpisode Trajectory details:")
    print(f"  Actions taken: {actions_taken}")
    print(f"  Path coordinates: {path}")
    print(f"  Steps taken: {steps}")
    print(f"  Reached goal: {reached_goal}")
    print(f"  Calculated path cost (straight=1.0, diag=sqrt(2)): {path_cost:.4f}")
    
    # 6. Benchmark time per query (predict call)
    # Run 100 predictions on the initial observation to get a stable microsecond estimate
    obs_for_timing = obs
    
    # Warmup
    for _ in range(10):
        _ = model.predict(obs_for_timing, deterministic=True)
        
    t_start = time.perf_counter()
    for _ in range(100):
        _ = model.predict(obs_for_timing, deterministic=True)
    t_end = time.perf_counter()
    
    time_per_query_ms = ((t_end - t_start) / 100.0) * 1000.0
    time_per_query_us = time_per_query_ms * 1000.0
    
    print(f"  Time per predict() query: {time_per_query_ms:.4f} ms ({time_per_query_us:.2f} us)")
    print("=" * 60)
    
    env.close()

if __name__ == "__main__":
    main()
