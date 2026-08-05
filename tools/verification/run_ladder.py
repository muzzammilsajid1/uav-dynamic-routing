import json
import logging
import os
import sys
from pathlib import Path
import hashlib
from datetime import datetime, timezone

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from rl_v3.phase_c1_env import PhaseC1EndpointGenerator, PhaseC1Env
from rl_v3.phase_b_policy import PhaseBFeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class LadderEvalCallback(BaseCallback):
    def __init__(self, train_env, val_env, train_len, save_dir):
        super().__init__()
        self.train_env = train_env
        self.val_env = val_env
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.train_len = train_len
        self.checkpoints = [5000, 10000, 25000]
        self.history = {}
        self.episodes_completed = 0
        self.terminal_goal_rewards = 0

    def _on_rollout_end(self):
        # Count terminal rewards in the rollout buffer
        for i in range(len(self.model.rollout_buffer.rewards)):
            r = self.model.rollout_buffer.rewards[i][0]
            if r >= 1.0:  # In Phase C1 R1, goal reward is 1.0
                self.terminal_goal_rewards += 1
            if self.model.rollout_buffer.episode_starts[i][0]:
                self.episodes_completed += 1

    def _evaluate(self, env, n_episodes):
        successes = 0
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
        return successes / n_episodes

    def _on_step(self) -> bool:
        ts = self.num_timesteps
        if ts in self.checkpoints or ts % 5000 == 0:
            train_sr = self._evaluate(self.train_env, self.train_len)
            val_sr = self._evaluate(self.val_env, 120)
            
            logger.info(f"[{ts}] Train SR: {train_sr:.3f}, Val SR: {val_sr:.3f}")
            self.history[ts] = {
                "train_sr": train_sr,
                "val_sr": val_sr,
                "episodes_completed": self.episodes_completed,
                "terminal_goal_rewards": self.terminal_goal_rewards,
                "terminal_goal_reward_freq": self.terminal_goal_rewards / max(1, self.episodes_completed)
            }
            if ts in self.checkpoints:
                self.model.save(str(self.save_dir / f"model_{ts:06d}.zip"))
        return True

def make_env(config, manifest_name, mode):
    def _init():
        manifest_path = ROOT / "evaluation" / "manifests" / manifest_name
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
            ladder_pairs = [(tuple(d["start"]), tuple(d["goal"])) for d in manifest_data]
            
        gen = PhaseC1EndpointGenerator(seed=42)
        gen.val_manifest = ladder_pairs
        gen.train_bins = {"short": [], "medium": [], "long": []}
        from rl_v3.phase_c0_env import _astar_cost
        for p in ladder_pairs:
            cost = _astar_cost(p[0], p[1])
            if cost < gen.T1: b = "short"
            elif cost < gen.T2: b = "medium"
            else: b = "long"
            gen.train_bins[b].append(p)
            
        for b in gen.train_bins:
            gen.train_bins[b].sort()
            
        gen.orientations = {}
        for p in ladder_pairs:
            dx = abs(p[1][0] - p[0][0])
            dy = abs(p[1][1] - p[0][1])
            if dx > 0 and dy <= dx * 0.5: ori = "vertical"
            elif dy > 0 and dx <= dy * 0.5: ori = "horizontal"
            elif abs(dx - dy) <= 1: ori = "diagonal"
            else: ori = "mixed"
            gen.orientations[p] = ori
            
        return PhaseC1Env(config, mode=mode, generator=gen)
    return _init

def train_ladder():
    config_path = ROOT / "configs" / "rl_v3_phase_c1.json"
    with open(config_path) as f:
        config = json.load(f)
        
    out_dir = ROOT / "runs" / "rl_v3" / "phase_c1_diagnostic" / "ladder"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ladder_results = {}
    
    for name, size in [("D1", 4), ("D2", 16), ("D3", 64), ("D4", 256)]:
        logger.info(f"--- Training {name} (size {size}) ---")
        
        manifest_path = ROOT / "evaluation" / "manifests" / f"rl_v3_ladder_{name}.json"
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        
        train_env = make_vec_env(make_env(config, f"rl_v3_ladder_{name}.json", "train"), n_envs=1)
        eval_train_env = make_vec_env(make_env(config, f"rl_v3_ladder_{name}.json", "eval"), n_envs=1)
        eval_val_env = make_vec_env(make_env(config, "rl_v3_phase_c1_validation.json", "eval"), n_envs=1)
        
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
            seed=42,
        )
        
        save_dir = out_dir / name
        cb = LadderEvalCallback(eval_train_env, eval_val_env, size, save_dir)
        
        model.learn(total_timesteps=25000, callback=cb)
        
        # Pull final stats from cb
        final_ts = max(cb.history.keys())
        episodes = cb.history[final_ts]["episodes_completed"]
        
        ladder_results[name] = {
            "metadata": {
                "manifest_name": f"rl_v3_ladder_{name}.json",
                "manifest_hash": manifest_hash,
                "interactions": 25000,
                "completed_episodes": episodes,
                "interactions_per_endpoint": 25000 / size,
                "episodes_per_endpoint": episodes / size if size > 0 else 0,
                "seed": 42,
                "architecture": "PhaseBFeatureExtractor(256) -> MLP([128, 128])",
                "checkpoints_evaluated": cb.checkpoints
            },
            "history": cb.history
        }
        
        with open(out_dir / "ladder_results.json", "w") as f:
            json.dump(ladder_results, f, indent=2)
            
if __name__ == "__main__":
    train_ladder()
