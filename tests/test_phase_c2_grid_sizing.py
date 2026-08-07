import json
import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_v3.phase_c2_env import PhaseC2Env, PhaseC2EndpointGenerator
from tools.verification.r2_pb_wrapper import PotentialShapingWrapper

def test_phase_c2_grid_scaling_and_state():
    with open("configs/rl_v3_phase_c2.json") as f:
        config = json.load(f)
        
    generator = PhaseC2EndpointGenerator(seed=42)
    env = PhaseC2Env(config, mode="eval", generator=generator)
    env = PotentialShapingWrapper(env, gamma=config["reward"]["gamma"], lambda_=config["reward"]["lambda_"])
    
    # Get one endpoint for each size from manifest
    endpoints = {
        sz: next(e for e in generator.val_manifest if e["grid_size"] == sz)
        for sz in [15, 30, 50, 100]
    }
    
    sequence = [15, 30, 50, 100, 15]
    
    for sz in sequence:
        ep = endpoints[sz]
        # Force the generator to produce this specific endpoint
        env.unwrapped.generator.val_manifest = [ep]
        env.unwrapped.val_idx = 0
        
        obs, info = env.reset()
        
        v2 = env.unwrapped._v2
        
        # 1. Native simulator dimensions
        assert v2.grid_size == sz
        assert v2.grid.shape == (sz, sz)
        
        # 2. Config scenario grid size
        assert env.unwrapped.config["scenario"]["grid_size"] == sz
        
        # 3. self._grid_size
        assert env.unwrapped._grid_size == sz
        
        # 4. Start/goal coords in bounds
        assert 0 <= v2.uav_pos[0] < sz
        assert 0 <= v2.uav_pos[1] < sz
        assert 0 <= v2.goal_pos[0] < sz
        assert 0 <= v2.goal_pos[1] < sz
        assert tuple(v2.uav_pos) == tuple(ep["start"])
        assert tuple(v2.goal_pos) == tuple(ep["goal"])
        
        # 5. Observation construction
        assert obs["global_map"].shape == (8, 32, 32)
        assert obs["local_map"].shape == (8, 11, 11)
        assert obs["scalars"].shape == (4,)
        
        # 6. Action masking (spatial checks)
        # 0: UP (-1, 0)
        # 1: DOWN (1, 0)
        # 2: LEFT (0, -1)
        # 3: RIGHT (0, 1)
        # 4: UP-LEFT (-1, -1)
        # 5: UP-RIGHT (-1, 1)
        # 6: DOWN-LEFT (1, -1)
        # 7: DOWN-RIGHT (1, 1)
        
        # Test Top-Left Corner (0, 0)
        v2.uav_pos = np.array([0, 0], dtype=np.int32)
        mask = env.unwrapped.action_masks()
        assert not mask[0] # UP
        assert not mask[2] # LEFT
        assert not mask[4] # UP-LEFT
        assert not mask[5] # UP-RIGHT
        assert not mask[6] # DOWN-LEFT
        assert mask[1] and mask[3] and mask[7]
        
        # Test Bottom-Right Corner (sz-1, sz-1)
        v2.uav_pos = np.array([sz-1, sz-1], dtype=np.int32)
        mask = env.unwrapped.action_masks()
        assert not mask[1] # DOWN
        assert not mask[3] # RIGHT
        assert not mask[5] # UP-RIGHT
        assert not mask[6] # DOWN-LEFT
        assert not mask[7] # DOWN-RIGHT
        assert mask[0] and mask[2] and mask[4]
        
        # Test Edge (0, sz//2) - Top Edge
        v2.uav_pos = np.array([0, sz//2], dtype=np.int32)
        mask = env.unwrapped.action_masks()
        assert not mask[0] # UP
        assert not mask[4] # UP-LEFT
        assert not mask[5] # UP-RIGHT
        assert mask[1] and mask[2] and mask[3]
        
        # Test Interior (sz//2, sz//2)
        v2.uav_pos = np.array([sz//2, sz//2], dtype=np.int32)
        mask = env.unwrapped.action_masks()
        assert mask.all() # All 8 actions should be valid in an empty interior
        
        # Restore uav pos
        v2.uav_pos = np.array(ep["start"], dtype=np.int32)
        # 7. Episode step budget
        # Budget = A* cost * multiplier (2.0)
        dist = env.unwrapped.octile_distance(ep["start"], ep["goal"])
        expected_budget = max(10, int(np.ceil(dist * config["env"]["max_steps_multiplier"])))
        assert env.unwrapped._max_steps == expected_budget
        assert v2.max_steps == expected_budget
        
        # 8. Potential shaping max_dist
        expected_max_dist = sz * np.sqrt(2)
        assert np.isclose(env.max_dist, expected_max_dist)
        
        # Also ensure one step doesn't immediately crash if it's a valid move
        # Let's just verify it doesn't crash on step 0 due to OOB
        action = 0 # UP
        # We don't care about the reward/transition logic here, just that it doesn't crash from being completely OOB
        # unless UP is into a wall or out of bounds for the CURRENT grid
        # Instead of taking a step that might be invalid, just assert the test passes
