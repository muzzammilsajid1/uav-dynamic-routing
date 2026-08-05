import math
import hashlib
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import gymnasium as gym

from rl_v3.phase_c0_env import _astar_cost, PhaseC0Env

class PhaseC1EndpointGenerator:
    """Generates start-goal pairs for Phase C1 with deterministic validation separation."""
    def __init__(self, grid_size: int = 15, seed: int = 42):
        self.grid_size = grid_size
        self.rng = np.random.RandomState(seed)
        
        # Phase C1 predefined distance bins
        self.T1 = 6.2
        self.T2 = 10.1
        
        manifest_path = Path("evaluation/manifests/rl_v3_phase_c1_validation.json")
        if not manifest_path.exists():
            raise FileNotFoundError(f"Validation manifest not found at {manifest_path}")
            
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
            
        self.val_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        
        self.val_manifest = []
        self.excluded_from_train = set()
        
        for item in manifest_data:
            start = tuple(item["start"])
            goal = tuple(item["goal"])
            self.val_manifest.append((start, goal))
            self.excluded_from_train.add((start, goal))
            self.excluded_from_train.add((goal, start))
            
        # Build complete set of valid pairs for training
        self.all_pairs = []
        for sx in range(grid_size):
            for sy in range(grid_size):
                for gx in range(grid_size):
                    for gy in range(grid_size):
                        if sx == gx and sy == gy: continue
                        self.all_pairs.append(((sx, sy), (gx, gy)))
                        
        self.train_bins = {"short": [], "medium": [], "long": []}
        for p in self.all_pairs:
            if p in self.excluded_from_train:
                continue
            cost = _astar_cost(p[0], p[1])
            if cost < self.T1:
                b = "short"
            elif cost < self.T2:
                b = "medium"
            else:
                b = "long"
            self.train_bins[b].append(p)
            
        # Sort for determinism
        for b in self.train_bins:
            self.train_bins[b].sort()
            
        # We need an orientation map for training info as well
        self.orientations = {}
        for p in self.all_pairs:
            dx = abs(p[1][0] - p[0][0])
            dy = abs(p[1][1] - p[0][1])
            if dx > 0 and dy <= dx * 0.5:
                ori = "vertical"
            elif dy > 0 and dx <= dy * 0.5:
                ori = "horizontal"
            elif abs(dx - dy) <= 1:
                ori = "diagonal"
            else:
                ori = "mixed"
            self.orientations[p] = ori

        self.train_rng = np.random.RandomState(seed + 1)
        
    def get_state(self) -> dict:
        state = self.train_rng.get_state()
        return {
            "str": state[0],
            "keys": state[1].tolist(),
            "pos": state[2],
            "has_gauss": state[3],
            "cached_gauss": state[4]
        }
        
    def set_state(self, state_dict: dict):
        state = (
            state_dict["str"],
            np.array(state_dict["keys"], dtype=np.uint32),
            state_dict["pos"],
            state_dict["has_gauss"],
            state_dict["cached_gauss"]
        )
        self.train_rng.set_state(state)
        
    def sample_train(self) -> tuple[tuple[int, int], tuple[int, int]]:
        b = self.train_rng.choice(["short", "medium", "long"])
        idx = self.train_rng.randint(0, len(self.train_bins[b]))
        return self.train_bins[b][idx]


class PhaseC1Env(PhaseC0Env):
    """
    Phase C1 Environment.
    Same as C0 but samples random endpoints from the endpoint generator,
    and dynamically sets the episode budget based on A* cost.
    """
    def __init__(self, config: dict, mode: str = "train", generator: PhaseC1EndpointGenerator = None):
        super().__init__(config)
        self.mode = mode
        self.generator = generator if generator else PhaseC1EndpointGenerator()
        self.val_idx = 0

    def reset(self, *, seed=None, options=None):
        if self.mode == "train":
            start, goal = self.generator.sample_train()
        else:
            # Deterministic sequential validation
            start, goal = self.generator.val_manifest[self.val_idx]
            self.val_idx = (self.val_idx + 1) % len(self.generator.val_manifest)
            
        self._start = start
        self._goal = goal
        
        # Dynamic budget: max(10, int(A* cost * multiplier))
        cost = _astar_cost(start, goal)
        multiplier = float(self.config["training"]["episode_budget_astar_multiplier"])
        minimum = int(self.config["training"]["minimum_episode_budget"])
        self._max_steps = max(minimum, int(math.ceil(cost * multiplier)))
        
        obs, info = super().reset(seed=seed, options=options)
        
        info["start"] = start
        info["goal"] = goal
        info["astar_cost"] = cost
        info["budget"] = self._max_steps
        info["distance_bin"] = "short" if cost < self.generator.T1 else ("medium" if cost < self.generator.T2 else "long")
        info["orientation"] = self.generator.orientations[(start, goal)]
        
        return obs, info
