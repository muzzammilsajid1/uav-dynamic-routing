import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_v3.phase_c0_env import _astar_cost
from rl_v3.phase_b_policy import PhaseBFeatureExtractor, model_parameter_count
from rl_v3.phase_c1_env import PhaseC1Env, PhaseC1EndpointGenerator
from rl_v3.scenario_generation import write_stable_json

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "rl_v3_phase_c1.json"
RUN_ROOT = ROOT / "runs" / "rl_v3" / "phase_c1"
ACTION_NAMES = ["N", "S", "W", "E", "NW", "NE", "SW", "SE"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, float) and (obj != obj or obj == float("inf") or obj == float("-inf")):
        return None
    return obj

def _write_json(path: Path, data: dict) -> None:
    safe = _json_safe(data)
    path.write_text(json.dumps(safe, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def _source_hash(files: list[str]) -> str:
    agg = hashlib.sha256()
    for rel in files:
        p = ROOT / rel
        if p.exists():
            d = hashlib.sha256(p.read_bytes()).hexdigest()
            agg.update(rel.encode()); agg.update(d.encode())
    return agg.hexdigest()

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def run_preflight() -> dict:
    print("Running Phase C1 Preflight Verification...")
    config = load_config()
    generator = PhaseC1EndpointGenerator(seed=42)
    env = PhaseC1Env(config, mode="val", generator=generator)
    
    # Add gae_lambda check to verify we matched C0
    assert "gae_lambda" in config["training"], "Missing gae_lambda in config"
    
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(CONFIG_PATH.relative_to(ROOT)),
        "generator_hash": generator.val_hash
    }
    
    # 1. Validation pair uniqueness and A*
    manifest = generator.val_manifest
    assert len(manifest) == 120, f"Expected 120 val pairs, got {len(manifest)}"
    assert len(set(manifest)) == 120, "Val manifest contains duplicates"
    
    # 2. Train/val disjoint check
    for p in manifest:
        assert p not in generator.train_bins["short"]
        assert p not in generator.train_bins["medium"]
        assert p not in generator.train_bins["long"]
        rev = (p[1], p[0])
        assert rev not in generator.train_bins["short"]
        assert rev not in generator.train_bins["medium"]
        assert rev not in generator.train_bins["long"]
        
    result["train_val_disjoint"] = True
    
    # 3. Bin balance
    counts = {"short": 0, "medium": 0, "long": 0}
    ori_counts = defaultdict(int)
    for p in manifest:
        cost = _astar_cost(p[0], p[1])
        if cost < generator.T1: counts["short"] += 1
        elif cost < generator.T2: counts["medium"] += 1
        else: counts["long"] += 1
        ori_counts[generator.orientations[p]] += 1
        
    assert counts["short"] == 40 and counts["medium"] == 40 and counts["long"] == 40, "Bins not balanced 40/40/40"
    result["bin_balance"] = counts
    result["orientations"] = dict(ori_counts)
    
    # 4. Deterministic reset and channels
    # 4. Deterministic reset and channels
    env.val_idx = 0
    obs, info = env.reset()
    assert info["start"] == manifest[0][0] and info["goal"] == manifest[0][1], f"First val pair mismatch: {info['start']} != {manifest[0][0]}"
    assert tuple(env._v2.uav_pos) == manifest[0][0], "Actual V2 uav_pos mismatch"
    assert tuple(env._v2.goal_pos) == manifest[0][1], "Actual V2 goal_pos mismatch"
    
    # 5. Scalar feature correct
    uav_r, uav_c = env._v2.uav_pos
    goal_r, goal_c = env._v2.goal_pos
    assert np.isclose(obs["scalars"][0], (goal_r - uav_r) / 14.0), "Scalar goal displacement R mismatch"
    assert np.isclose(obs["scalars"][1], (goal_c - uav_c) / 14.0), "Scalar goal displacement C mismatch"
    
    # Check that budget is properly set
    budget = int(math.ceil(_astar_cost(manifest[0][0], manifest[0][1]) * config["training"]["episode_budget_astar_multiplier"]))
    budget = max(int(config["training"]["minimum_episode_budget"]), budget)
    assert env._max_steps == budget, "Episode budget mismatch"
    assert info["budget"] == budget, "Info budget mismatch"
    
    assert obs["local_map"][3, 5, 5] == 1.0, "Agent not centered in local map"
    
    # 5. Masking
    mask = env.action_masks()
    assert len(mask) == 8
    
    env.close()
    
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(RUN_ROOT / "preflight_verification.json", result)
    print("Preflight passed.")
    return result

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_phase_c1(model: MaskablePPO, config: dict, generator: PhaseC1EndpointGenerator) -> dict:
    """Evaluate deterministically on the full validation manifest."""
    env = PhaseC1Env(config, mode="val", generator=generator)
    env.val_idx = 0  # reset to start of manifest
    
    n_episodes = len(generator.val_manifest)
    
    # Aggregate metrics
    overall = {"successes": 0, "collisions": 0, "timeouts": 0, "oscillations_2cell": 0, "longer_loops": 0}
    bin_metrics = {b: {"episodes": 0, "successes": 0, "route_steps": [], "realized_costs": [], "success_astar_costs": [], "failed_astar_costs": [], "path_cost_gaps": [], "normalized_cost_ratios": []} for b in ["short", "medium", "long"]}
    returns = []
    action_counts = Counter()
    trajectories = {"success": [], "failure": []}
    
    for ep in range(n_episodes):
        obs, info_reset = env.reset()
        ep_return = 0.0
        ep_steps = 0
        ep_cost = 0.0
        traj = []
        pos_history = [tuple(int(v) for v in env._v2.uav_pos)]
        done = False
        
        while not done:
            masks = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=masks)
            action = int(action)
            edge_cost = math.sqrt(2.0) if action >= 4 else 1.0
            obs, rew, term, trunc, info = env.step(action)
            ep_return += rew
            ep_steps += 1
            ep_cost += edge_cost
            action_counts[ACTION_NAMES[action]] += 1
            pos = tuple(int(v) for v in env._v2.uav_pos)
            pos_history.append(pos)
            
            traj.append({
                "step": ep_steps, "action": action, "action_name": ACTION_NAMES[action],
                "pos": list(pos), "reward": float(rew),
                "terminated": bool(term), "truncated": bool(trunc),
                "crashed": bool(info.get("crashed", False)),
                "is_success": bool(info.get("is_success", False)),
            })
            done = term or trunc
            
            if info.get("crashed"):
                overall["collisions"] += 1
            if trunc:
                overall["timeouts"] += 1
                
        # Analyze outcome
        b = info_reset["distance_bin"]
        bin_metrics[b]["episodes"] += 1
        
        if info.get("is_success"):
            overall["successes"] += 1
            bin_metrics[b]["successes"] += 1
            bin_metrics[b]["route_steps"].append(ep_steps)
            bin_metrics[b]["realized_costs"].append(ep_cost)
            bin_metrics[b]["success_astar_costs"].append(info_reset["astar_cost"])
            
            gap = ep_cost - info_reset["astar_cost"]
            ratio = ep_cost / info_reset["astar_cost"] if info_reset["astar_cost"] > 0 else 1.0
            bin_metrics[b]["path_cost_gaps"].append(gap)
            bin_metrics[b]["normalized_cost_ratios"].append(ratio)
            
            if len(trajectories["success"]) < 5:
                trajectories["success"].append(traj)
        else:
            bin_metrics[b]["failed_astar_costs"].append(info_reset["astar_cost"])
            # Failure categorization
            osc = False
            if len(pos_history) >= 4:
                osc = any(pos_history[i] == pos_history[i+2] for i in range(len(pos_history)-2))
                if osc: overall["oscillations_2cell"] += 1
                
            pos_set = Counter(pos_history)
            loop = any(v >= 3 for v in pos_set.values())
            if loop and not osc: overall["longer_loops"] += 1
            
            if len(trajectories["failure"]) < 5:
                trajectories["failure"].append(traj)
                
        returns.append(ep_return)
        
    env.close()
    
    # Calculate bin summaries
    bin_summaries = {}
    for b in ["short", "medium", "long"]:
        bm = bin_metrics[b]
        mean_steps = float(np.mean(bm["route_steps"])) if bm["route_steps"] else float("nan")
        mean_realized = float(np.mean(bm["realized_costs"])) if bm["realized_costs"] else float("nan")
        mean_astar_success = float(np.mean(bm["success_astar_costs"])) if bm["success_astar_costs"] else float("nan")
        mean_astar_failed = float(np.mean(bm["failed_astar_costs"])) if bm["failed_astar_costs"] else float("nan")
        mean_gap = float(np.mean(bm["path_cost_gaps"])) if bm["path_cost_gaps"] else float("nan")
        mean_ratio = float(np.mean(bm["normalized_cost_ratios"])) if bm["normalized_cost_ratios"] else float("nan")
        
        bin_summaries[b] = {
            "success_rate": bm["successes"] / bm["episodes"] if bm["episodes"] else 0.0,
            "mean_route_steps": mean_steps,
            "mean_realized_cost": mean_realized,
            "mean_astar_cost_success": mean_astar_success,
            "mean_astar_cost_failed": mean_astar_failed,
            "mean_path_cost_gap": mean_gap,
            "mean_normalized_cost_ratio": mean_ratio
        }
        
    return {
        "n_episodes": n_episodes,
        "success_rate": overall["successes"] / n_episodes,
        "overall": overall,
        "bin_summaries": bin_summaries,
        "mean_return": float(np.mean(returns)),
        "action_distribution": dict(action_counts),
        "trajectories": trajectories
    }

# ---------------------------------------------------------------------------
# Training Callbacks
# ---------------------------------------------------------------------------

class PhaseC1EvalCallback(BaseCallback):
    def __init__(self, config: dict, generator: PhaseC1EndpointGenerator):
        super().__init__(verbose=1)
        self.config = config
        self.generator = generator
        self.checkpoints = set(config["training"]["checkpoints"])
        self.eval_results = []
        self.wall_start = time.perf_counter()
        
    def _on_step(self) -> bool:
        if self.num_timesteps in self.checkpoints:
            print(f"\n--- Evaluating at {self.num_timesteps} interactions ---")
            res = evaluate_phase_c1(self.model, self.config, self.generator)
            
            wall = time.perf_counter() - self.wall_start
            sr = res["success_rate"]
            b_sr = {b: res["bin_summaries"][b]["success_rate"] for b in ["short", "medium", "long"]}
            
            print(f"Overall SR: {sr*100:.1f}% | Short: {b_sr['short']*100:.0f}% | Med: {b_sr['medium']*100:.0f}% | Long: {b_sr['long']*100:.0f}%")
            print(f"Crashes: {res['overall']['collisions']} | Timeouts: {res['overall']['timeouts']} | Loops: {res['overall']['longer_loops']}")
            
            self.eval_results.append({
                "interactions": self.num_timesteps,
                "wall_seconds": round(wall, 1),
                "success_rate": sr,
                "bin_success_rates": b_sr,
                "overall_metrics": res["overall"],
                "bin_summaries": res["bin_summaries"],
                "mean_return": res["mean_return"],
                "generator_state": self.generator.get_state()
            })
            
            # Save model
            model_path = RUN_ROOT / f"model_{self.num_timesteps:06d}.zip"
            self.model.save(str(model_path))
            
            # Save status
            _write_json(RUN_ROOT / "status.json", {"history": self.eval_results})
            
            # Early stop logic
            # >= 95% overall, >= 90% per bin, 0 crashes, <5% loops/osc
            consecutive = 0
            for r in reversed(self.eval_results):
                fail_rate = (r["overall_metrics"]["oscillations_2cell"] + r["overall_metrics"]["longer_loops"]) / 120.0
                if (r["success_rate"] >= 0.95 and 
                    all(v >= 0.90 for v in r["bin_success_rates"].values()) and 
                    r["overall_metrics"]["collisions"] == 0 and 
                    fail_rate < 0.05):
                    consecutive += 1
                else:
                    break
                    
            if consecutive >= 2:
                print("\n[C1] Early stopping criteria met (2 consecutive checkpoints >= 95%).")
                return False
                
            if self.num_timesteps == 100000 and sr < 0.50:
                print("\n[C1] Early stopping: Validation success < 50% at 100k.")
                return False
                
            import gc
            gc.collect()
            
        return True

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_training():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    config = load_config()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    
    if not (RUN_ROOT / "preflight_verification.json").exists():
        run_preflight()
        
    generator = PhaseC1EndpointGenerator(seed=42)
    env = PhaseC1Env(config, mode="train", generator=generator)
    vec_env = DummyVecEnv([lambda: env])
    
    policy_kwargs = {
        "features_extractor_class": PhaseBFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": [128, 64]
    }
    
    status_path = RUN_ROOT / "status.json"
    start_timestep = 0
    if status_path.exists():
        status_data = json.loads(status_path.read_text())
        if "history" in status_data and len(status_data["history"]) > 0:
            start_timestep = status_data["history"][-1]["interactions"]
            print(f"Found existing training state, resuming from {start_timestep} interactions")
            
    if start_timestep > 0:
        model_path = RUN_ROOT / f"model_{start_timestep:06d}.zip"
        if not model_path.exists():
            print(f"Warning: model_{start_timestep:06d}.zip not found, starting from scratch")
            start_timestep = 0
            model = MaskablePPO(
                "MultiInputPolicy",
                vec_env,
                learning_rate=config["training"]["learning_rate"],
                n_steps=config["training"]["n_steps"],
                batch_size=config["training"]["batch_size"],
                n_epochs=config["training"]["n_epochs"],
                gamma=config["training"]["gamma"],
                gae_lambda=config.get("training", {}).get("gae_lambda", 0.95),
                ent_coef=config["training"]["ent_coef"],
                policy_kwargs=policy_kwargs,
                seed=config.get("seed", 42),
                verbose=1,
            )
        else:
            model = MaskablePPO.load(str(model_path), env=vec_env)
            if "generator_state" in status_data["history"][-1]:
                generator.set_state(status_data["history"][-1]["generator_state"])
                print("Restored generator state from checkpoint.")
    else:
        model = MaskablePPO(
            "MultiInputPolicy",
            vec_env,
            learning_rate=config["training"]["learning_rate"],
            n_steps=config["training"]["n_steps"],
            batch_size=config["training"]["batch_size"],
            n_epochs=config["training"]["n_epochs"],
            gamma=config["training"]["gamma"],
            gae_lambda=config.get("training", {}).get("gae_lambda", 0.95),
            ent_coef=config["training"]["ent_coef"],
            policy_kwargs=policy_kwargs,
            seed=config.get("seed", 42),
            verbose=1,
        )
    
    cb = PhaseC1EvalCallback(config, generator)
    if start_timestep > 0:
        cb.eval_results = status_data.get("history", [])
        
    max_steps = max(config["training"]["checkpoints"])
    remaining_steps = max_steps - start_timestep
    
    if remaining_steps <= 0:
        print("Training already finished according to checkpoints.")
        return
        
    # Hack the base callback to not reset num_timesteps
    model.num_timesteps = start_timestep
    
    print(f"Starting Phase C1 Training for {remaining_steps} remaining steps...")
    model.learn(total_timesteps=remaining_steps, callback=cb, reset_num_timesteps=False)
    env.close()
    
    print("Training finished. Summary available in status.json.")
    return cb.eval_results

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight","train"))
    args = parser.parse_args()
    
    if args.command == "preflight":
        run_preflight()
    elif args.command == "train":
        run_training()

if __name__ == "__main__":
    main()
