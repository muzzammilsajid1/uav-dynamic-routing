import torch
import numpy as np
from stable_baselines3.her import HerReplayBuffer

class SafeHerReplayBuffer(HerReplayBuffer):
    def sample(self, batch_size, env=None):
        samples = super().sample(batch_size, env)
        
        # Force dones=1.0 for transitions where the achieved goal matches the desired goal
        ag = np.round(samples.next_observations["achieved_goal"].cpu().numpy()).astype(np.int32)
        dg = np.round(samples.observations["desired_goal"].cpu().numpy()).astype(np.int32)
        
        # Check spatial equivalence (first 2 elements)
        goal_reached = np.all(ag[:, 0:2] == dg[:, 0:2], axis=1)
        
        # Update dones inplace
        if np.any(goal_reached):
            # Since samples.dones is a PyTorch tensor, we modify it in-place
            samples.dones[goal_reached] = 1.0
            
        return samples
