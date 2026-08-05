import json
import logging
import sys
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from rl_v3.phase_c2_env import PhaseC2Env, PhaseC2EndpointGenerator
from tools.verification.r2_pb_wrapper import PotentialShapingWrapper

logger = logging.getLogger(__name__)

class CurriculumCallback(BaseCallback):
    def __init__(self, stages, generator, runner, verbose=0):
        super().__init__(verbose)
        self.stages = stages
        self.generator = generator
        self.runner = runner

    def _on_step(self):
        ts = self.num_timesteps
        active_sizes = None
        for stage in self.stages:
            if ts <= stage["max_interactions"]:
                active_sizes = stage["active_sizes"]
                break
        
        if active_sizes is None:
            active_sizes = self.stages[-1]["active_sizes"]
            
        if self.generator.active_sizes != active_sizes:
            logger.info(f"[{ts}] Curriculum shift: Active sizes now {active_sizes}")
            self.generator.set_active_sizes(active_sizes)
            
        if ts in self.runner.checkpoints and ts not in self.runner.history:
            self.runner.evaluate_and_save(ts)
            
        return True

def preflight(config_path, out_dir):
    logger.info("=== PHASE C2 PREFLIGHT ===")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    gen = PhaseC2EndpointGenerator(seed=42)
    
    errors = []
    
    # Check val manifest
    if len(gen.val_manifest) != 240:
        errors.append(f"Validation routes count != 240: {len(gen.val_manifest)}")
        
    counts = {}
    for item in gen.val_manifest:
        sz = item["grid_size"]
        b = item["distance_bin"]
        if sz not in counts: counts[sz] = {"short": 0, "medium": 0, "long": 0}
        counts[sz][b] += 1
        
    for sz in [15, 30, 50, 100]:
        if counts.get(sz, {}).get("short") != 20: errors.append(f"Size {sz} short != 20")
        if counts.get(sz, {}).get("medium") != 20: errors.append(f"Size {sz} medium != 20")
        if counts.get(sz, {}).get("long") != 20: errors.append(f"Size {sz} long != 20")
        
    # Check duplicates and reverse
    seen = set()
    for item in gen.val_manifest:
        s = tuple(item["start"])
        g = tuple(item["goal"])
        sz = item["grid_size"]
        if (sz, s, g) in seen or (sz, g, s) in seen:
            errors.append(f"Duplicate or reverse leakage found: {sz} {s} {g}")
        seen.add((sz, s, g))
        
    with open(config_path) as f:
        config = json.load(f)
        
    # Test env properties
    try:
        env = PhaseC2Env(config, mode="eval", generator=gen)
        env = PotentialShapingWrapper(env, gamma=config["reward"]["gamma"], lambda_=config["reward"]["lambda_"])
        obs, info = env.reset()
        
        if obs["global_map"].shape != (8, 32, 32):
            errors.append(f"Global map shape != (8, 32, 32): {obs['global_map'].shape}")
            
        # Check NaNs
        if np.isnan(obs["global_map"]).any() or np.isnan(obs["scalars"]).any():
            errors.append("NaN detected in observations")
            
        obs, reward, term, trunc, info = env.step(0)
        if not np.isfinite(reward):
            errors.append(f"R2-PB reward is not finite: {reward}")
            
    except Exception as e:
        errors.append(f"Environment initialization/step failed: {e}")
        
    res = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors
    }
    
    with open(out_dir / "preflight_verification.json", "w") as f:
        json.dump(res, f, indent=2)
        
    if errors:
        for e in errors: logger.error(e)
        sys.exit(1)
        
    logger.info("Preflight PASSED.")
    sys.exit(0)


class PhaseC2Runner:
    def __init__(self, config_path, out_dir, model_type="M1", resume=False):
        with open(config_path) as f:
            self.config = json.load(f)
            
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.model_type = model_type
        self.resume = resume
        
        self.checkpoints = [25000, 50000, 75000, 100000, 150000]
        self.history = {}
        
        self.generator = PhaseC2EndpointGenerator(seed=42)
        
        self.generator.set_active_sizes(self.config["curriculum"]["stages"][0]["active_sizes"])
        
        self.train_env = PhaseC2Env(self.config, mode="train", generator=self.generator)
        if self.config["reward"]["type"] == "R2-PB-empty-v1":
            self.train_env = PotentialShapingWrapper(
                self.train_env,
                gamma=self.config["reward"]["gamma"],
                lambda_=self.config["reward"]["lambda_"]
            )
            
        if not resume:
            if model_type == "M1":
                from rl_v3.phase_b_policy import PhaseBFeatureExtractor
                self.model = MaskablePPO(
                    "MultiInputPolicy",
                    self.train_env,
                    policy_kwargs={
                        "features_extractor_class": PhaseBFeatureExtractor,
                        "features_extractor_kwargs": {"features_dim": 256}
                    },
                    **self.config["model"]
                )
            else:
                self.model = MaskablePPO(
                    "MultiInputPolicy",
                    self.train_env,
                    **self.config["model"]
                )
        else:
            # We will handle resume manually by loading the model file.
            pass
            
    def evaluate_and_save(self, ts):
        logger.info(f"[{ts}] Evaluating 240 validation routes...")
        val_env = PhaseC2Env(self.config, mode="eval", generator=self.generator)
        if self.config["reward"]["type"] == "R2-PB-empty-v1":
            val_env = PotentialShapingWrapper(
                val_env,
                gamma=self.config["reward"]["gamma"],
                lambda_=self.config["reward"]["lambda_"]
            )
            
        successes, collisions, timeouts = 0, 0, 0
        n_eval = len(self.generator.val_manifest)
        
        for i in range(n_eval):
            obs, info = val_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, term, trunc, info = val_env.step(int(action))
                done = term or trunc
                
            if info.get("is_success", False): successes += 1
            elif info.get("is_collision", False): collisions += 1
            else: timeouts += 1
            
        val_sr = successes / max(1, n_eval)
        logger.info(f"[{ts}] Val SR: {val_sr:.3f}")
        
        self.history[ts] = {
            "success_rate": val_sr,
            "collisions": collisions,
            "timeouts": timeouts
        }
        
        self.model.save(str(self.out_dir / f"model_{ts:06d}.zip"))
        
        gen_state = self.generator.get_state()
        with open(self.out_dir / f"generator_{ts:06d}.json", "w") as f:
            json.dump(gen_state, f)
            
        with open(self.out_dir / "status.json", "w") as f:
            json.dump({"history": self.history}, f, indent=2)

    def run(self, max_interactions=150000):
        cb = CurriculumCallback(self.config["curriculum"]["stages"], self.generator, self)
        logger.info(f"Starting Phase C2 training for {max_interactions} interactions")
        # In resume, total_timesteps should be adjusted if needed, but learn allows resetting num_timesteps=False
        reset_num = not self.resume
        self.model.learn(total_timesteps=max_interactions, callback=cb, reset_num_timesteps=reset_num)
        
        if max_interactions not in self.history:
            self.evaluate_and_save(max_interactions)

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'preflight':
        config_path = "configs/rl_v3_phase_c2.json"
        out_dir = "runs/uav_phase_c2_preflight"
        if len(sys.argv) > 2:
            out_dir = sys.argv[2]
        preflight(config_path, out_dir)
