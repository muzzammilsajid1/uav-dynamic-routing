import json
import logging
import os
import sys
from pathlib import Path
from typing import Any
import hashlib
from datetime import datetime, timezone
import copy

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from rl_v3.phase_c1_env import PhaseC1EndpointGenerator, PhaseC1Env
from rl_v3.phase_b_policy import PhaseBFeatureExtractor
from tools.verification.r2_pb_wrapper import PotentialShapingWrapper
from tools.verification.run_part6 import evaluate_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class MultiseedEvalCallback(BaseCallback):
    def __init__(self, val_env, config, seed, model_name, save_dir):
        super().__init__()
        self.val_env = val_env
        self.config = config
        self.model_seed = seed
        self.model_name = model_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.history = {}
        self.consecutive_passes = 0
        self.checkpoints = [5000, 10000, 25000, 50000]

    def _on_step(self) -> bool:
        ts = self.num_timesteps
        if ts in self.checkpoints:
            logger.info(f"Evaluating {self.model_name} seed {self.model_seed} at {ts} steps...")
            metrics = evaluate_checkpoint(self.model, self.val_env, self.config, seed=self.model_seed)
            self.history[ts] = metrics
            
            # Check gate
            sr = metrics["success_rate"]
            bin_sr = metrics["success_by_bin"]
            colls = metrics["collisions"]
            loops = metrics["oscillations"] + metrics["longer_loops"]
            
            logger.info(f"[{ts}] SR: {sr:.3f}, bins: {bin_sr}, colls: {colls}, loops: {loops}")
            
            gate_passed = False
            if sr >= 0.95 and colls == 0 and loops <= 6: # 120 * 0.05 = 6
                if all(v >= 0.90 for v in bin_sr.values()):
                    gate_passed = True
            
            if gate_passed:
                self.consecutive_passes += 1
            else:
                self.consecutive_passes = 0
                
            if self.consecutive_passes >= 2:
                logger.info(f"Seed {self.model_seed} passed gate twice! Early stopping.")
                return False
                
        return True

def make_env(config, mode, reward_type):
    def _init():
        if mode == "train": gen = PhaseC1EndpointGenerator(seed=42)
        else: gen = PhaseC1EndpointGenerator(seed=999)
        env = PhaseC1Env(config, mode=mode, generator=gen)
        
        if reward_type == "R2-PB":
            return PotentialShapingWrapper(env, gamma=0.99, lambda_=2.0)
        return env
    return _init

def run_multiseed():
    config_path = ROOT / "configs" / "rl_v3_phase_c1.json"
    with open(config_path) as f:
        config = json.load(f)
        
    out_dir = ROOT / "runs" / "rl_v3" / "phase_c1_diagnostic" / "multiseed"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    seeds = [42, 43, 44]
    variants = ["R1", "R2-PB"]
    results = {}
    
    for var in variants:
        results[var] = {}
        for s in seeds:
            logger.info(f"--- Training {var} - Seed {s} ---")
            
            train_env = make_vec_env(make_env(config, "train", var), n_envs=1)
            eval_val_env = make_vec_env(make_env(config, "eval", var), n_envs=1)
            
            policy_kwargs = {
                "features_extractor_class": PhaseBFeatureExtractor,
                "features_extractor_kwargs": {"features_dim": 256},
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
                seed=s,
            )
            
            save_dir = out_dir / f"{var}_{s}"
            cb = MultiseedEvalCallback(eval_val_env, config, s, var, save_dir)
            model.learn(total_timesteps=50000, callback=cb)
            
            results[var][s] = cb.history
            
            with open(out_dir / "multiseed_results.json", "w") as f:
                json.dump(results, f, indent=2)
                
    # Compile statistics
    stats = {}
    for var in variants:
        final_srs = []
        for s in seeds:
            hist = results[var][s]
            final_ts = max(hist.keys())
            final_srs.append(hist[final_ts]["success_rate"])
        stats[var] = {
            "mean_sr": float(np.mean(final_srs)),
            "std_sr": float(np.std(final_srs)),
            "median_sr": float(np.median(final_srs)),
            "min_sr": float(np.min(final_srs)),
            "max_sr": float(np.max(final_srs)),
        }
        
    results["summary_statistics"] = stats
    with open(out_dir / "multiseed_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_multiseed()
