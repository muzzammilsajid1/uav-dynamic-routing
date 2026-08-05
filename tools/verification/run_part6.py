import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from rl_v3.phase_c1_env import PhaseC1EndpointGenerator, PhaseC1Env
from rl_v3.phase_b_policy import PhaseBFeatureExtractor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium.spaces as spaces

from tools.verification.r2_pb_wrapper import PotentialShapingWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class ScalarOnlyFeatureExtractor(BaseFeaturesExtractor):
    """Simple feature extractor using only scalar inputs."""
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(4, 32), torch.nn.Tanh(), torch.nn.Linear(32, features_dim), torch.nn.Tanh()
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.net(observations["scalars"])


class DiagnosticEvalCallback(BaseCallback):
    def __init__(self, val_env, save_dir):
        super().__init__()
        self.val_env = val_env
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.history = {}

    def _evaluate(self, env, n_episodes):
        successes = 0
        collisions = 0
        timeouts = 0
        
        for _ in range(n_episodes):
            obs = env.reset()
            done = False
            while not done:
                mask = env.get_attr("action_masks")[0]()
                action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
                obs, reward, done_list, info_list = env.step(action)
                done = done_list[0]
                info = info_list[0]
                
            if info.get("is_success"):
                successes += 1
            elif info.get("crashed"):
                collisions += 1
            else:
                timeouts += 1
                
        return successes / n_episodes, collisions, timeouts

    def _on_step(self) -> bool:
        ts = self.num_timesteps
        if ts % 5000 == 0:
            val_sr, val_coll, val_time = self._evaluate(self.val_env, 120)
            logger.info(f"[{ts}] Val SR: {val_sr:.3f}, Coll: {val_coll}, Timeouts: {val_time}")
            self.history[ts] = {
                "val_sr": val_sr,
                "val_coll": val_coll,
                "val_time": val_time
            }
        return True

def make_env(config, mode):
    def _init():
        if mode == "train":
            gen = PhaseC1EndpointGenerator(seed=42)
        else:
            gen = PhaseC1EndpointGenerator(seed=999)
        env = PhaseC1Env(config, mode=mode, generator=gen)
        # Wrap with R2-PB
        return PotentialShapingWrapper(env, gamma=0.99, lambda_=2.0)
    return _init

def run_pilots():
    config_path = ROOT / "configs" / "rl_v3_phase_c1.json"
    with open(config_path) as f:
        config = json.load(f)
        
    out_dir = ROOT / "runs" / "rl_v3" / "phase_c1_diagnostic" / "part6"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for pilot, extractor, dim in [
        ("P1_GlobalLocal", PhaseBFeatureExtractor, 256),
        ("P2_ScalarOnly", ScalarOnlyFeatureExtractor, 64)
    ]:
        logger.info(f"--- Training {pilot} ---")
        
        train_env = make_vec_env(make_env(config, "train"), n_envs=1)
        eval_val_env = make_vec_env(make_env(config, "eval"), n_envs=1)
        
        policy_kwargs = {
            "features_extractor_class": extractor,
            "features_extractor_kwargs": {"features_dim": dim},
            "net_arch": [128, 128],
        }
        
        model = MaskablePPO(
            "MultiInputPolicy",
            train_env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=42,
        )
        
        cb = DiagnosticEvalCallback(eval_val_env, out_dir / pilot)
        
        model.learn(total_timesteps=50000, callback=cb)
        
        results[pilot] = cb.history
        
        with open(out_dir / "r2_pb_results.json", "w") as f:
            json.dump(results, f, indent=2)
            
if __name__ == "__main__":
    run_pilots()
