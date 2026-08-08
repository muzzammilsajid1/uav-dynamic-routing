import math
import hashlib
import json
import numpy as np
from pathlib import Path
import gymnasium as gym

from rl_v3.phase_c0_env import PhaseC0Env

PROJECT_ROOT = Path(__file__).resolve().parents[1]

class PhaseC2EndpointGenerator:
    """Generates start-goal pairs for Phase C2 with multi-scale empty grids."""
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        
        manifest_path = PROJECT_ROOT / "evaluation/manifests/rl_v3_phase_c2_validation.json"
        train_path = PROJECT_ROOT / "evaluation/manifests/rl_v3_phase_c2_train_generator.json"
        if not manifest_path.exists() or not train_path.exists():
            raise FileNotFoundError("Phase C2 manifests not found.")
            
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.val_manifest = json.load(f)
            
        with open(train_path, "r", encoding="utf-8") as f:
            self.train_pool = json.load(f)
            
        self.val_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        self.train_hash = hashlib.sha256(train_path.read_bytes()).hexdigest()
        
        self.active_sizes = [15, 30] # Updated by curriculum
        
    def set_active_sizes(self, sizes):
        normalized = [int(size) for size in sizes]
        unknown = sorted(set(normalized) - {15, 30, 50, 100})
        if not normalized or unknown:
            raise ValueError(f"invalid active grid sizes: {normalized}")
        self.active_sizes = normalized

    def sample_train(self) -> dict:
        sz = int(self.rng.choice(self.active_sizes))
        b = self.rng.choice(["short", "medium", "long"])
        idx = self.rng.randint(0, len(self.train_pool[str(sz)][b]))
        pair = self.train_pool[str(sz)][b][idx]
        return {
            "grid_size": sz,
            "start": tuple(pair[0]),
            "goal": tuple(pair[1])
        }

    def get_state(self) -> dict:
        state = self.rng.get_state()
        return {
            "str": state[0],
            "keys": state[1].tolist(),
            "pos": state[2],
            "has_gauss": state[3],
            "cached_gauss": state[4],
            "active_sizes": [int(size) for size in self.active_sizes],
        }
        
    def set_state(self, state_dict: dict):
        state = (
            state_dict["str"],
            np.array(state_dict["keys"], dtype=np.uint32),
            state_dict["pos"],
            state_dict["has_gauss"],
            state_dict["cached_gauss"]
        )
        self.rng.set_state(state)
        if "active_sizes" not in state_dict:
            raise ValueError("generator state is missing active_sizes")
        self.set_active_sizes([int(size) for size in state_dict["active_sizes"]])


class PhaseC2Env(PhaseC0Env):
    """
    Phase C2 Environment.
    Supports multi-scale empty grids, dynamic sizing per episode.
    Uses octile distance for budget to avoid O(N) A* computations on large grids.
    """
    def __init__(self, config: dict, mode: str = "train", generator: PhaseC2EndpointGenerator = None):
        super().__init__(config)
        self.mode = mode
        self.generator = generator if generator else PhaseC2EndpointGenerator()
        self.val_idx = 0

    def octile_distance(self, s, g):
        dx = abs(s[0] - g[0])
        dy = abs(s[1] - g[1])
        return max(dx, dy) + (np.sqrt(2) - 1) * min(dx, dy)

    def reset(self, *, seed=None, options=None):
        if self.mode == "train":
            spec = self.generator.sample_train()
        else:
            spec = self.generator.val_manifest[self.val_idx]
            self.val_idx = (self.val_idx + 1) % len(self.generator.val_manifest)
            
        self.config["scenario"]["grid_size"] = spec["grid_size"]
        self._grid_size = spec["grid_size"]
        
        self._start = tuple(spec["start"])
        self._goal = tuple(spec["goal"])
        
        # We pass dummy astar cost since empty grid octile distance is A* cost.
        cost = self.octile_distance(self._start, self._goal)
        multiplier = float(self.config["env"].get("max_steps_multiplier", 2.0))
        minimum = int(self.config["training"]["minimum_episode_budget"])
        self._max_steps = max(minimum, int(math.ceil(cost * multiplier)))
        
        # PhaseC0Env.reset() constructs the native environment from the updated
        # runtime fields. Do not construct it here as well: that would double
        # every episode-reset cost without changing the resulting state.
        obs, info = super().reset(seed=seed, options=options)
        
        info["start"] = self._start
        info["goal"] = self._goal
        info["octile_cost"] = cost
        info["budget"] = self._max_steps
        info["grid_size"] = spec["grid_size"]
        
        return obs, info
