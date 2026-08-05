import gymnasium as gym
import numpy as np

class PotentialShapingWrapper(gym.Wrapper):
    """
    R2-PB Potential-Based Reward Shaping Wrapper.
    
    Phi(s) = - (octile_distance(s, goal) / max_grid_distance)
    F(s, a, s') = lambda_ * (gamma * Phi(s') - Phi(s))
    
    Terminal handling:
    If the state is terminal (success or collision), the episode ends. 
    Traditionally, Phi(terminal_state) = 0.
    So F(s, a, s') = lambda_ * (gamma * 0 - Phi(s)) = -lambda_ * Phi(s).
    """
    def __init__(self, env, gamma=0.99, lambda_=1.0):
        super().__init__(env)
        self.gamma = gamma
        self.lambda_ = lambda_
        self.max_dist = self.env.unwrapped._grid_size * np.sqrt(2)
        self.prev_phi = 0.0

    def octile_distance(self, pos, goal):
        dx = abs(pos[0] - goal[0])
        dy = abs(pos[1] - goal[1])
        return max(dx, dy) + (np.sqrt(2) - 1) * min(dx, dy)
        
    def get_phi(self, pos, goal):
        # Phi(s) is negative normalized distance
        dist = self.octile_distance(pos, goal)
        return -(dist / self.max_dist)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        pos = self.env.unwrapped._v2.uav_pos
        goal = self.env.unwrapped._v2.goal_pos
        
        self.prev_phi = self.get_phi(pos, goal)
        return obs, info
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        goal = self.env.unwrapped._v2.goal_pos
        
        if terminated:
            # Reached goal or crashed -> terminal state potential is 0
            next_phi = 0.0
        else:
            pos = self.env.unwrapped._v2.uav_pos
            next_phi = self.get_phi(pos, goal)
            
        shaping = self.lambda_ * (self.gamma * next_phi - self.prev_phi)
        self.prev_phi = next_phi
        
        # Add shaping to original R1 reward
        total_reward = reward + shaping
        
        return obs, total_reward, terminated, truncated, info
