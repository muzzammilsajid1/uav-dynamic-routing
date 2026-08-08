import json
import logging
import math
import random
import sys
import time
import numpy as np
import torch
from pathlib import Path
from sb3_contrib import MaskablePPO

from rl_v3.phase_c2_env import PhaseC2Env, PhaseC2EndpointGenerator
from tools.verification.r2_pb_wrapper import PotentialShapingWrapper
import gymnasium as gym
from rl_v3.diagnostics import has_longer_loop, has_two_cell_oscillation


def _octile_distance(start, goal):
    dr = abs(int(start[0]) - int(goal[0]))
    dc = abs(int(start[1]) - int(goal[1]))
    return max(dr, dc) + (math.sqrt(2.0) - 1.0) * min(dr, dc)


def _mean(values):
    return float(sum(values) / len(values)) if values else None


def _write_json_atomic(path, payload):
    path = Path(path)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _aggregate_validation(rows):
    groups = {"all": rows}
    for row in rows:
        groups.setdefault(f"scale/{row['grid_size']}", []).append(row)
        groups.setdefault(f"distance/{row['distance_bin']}", []).append(row)
        groups.setdefault(
            f"scale_distance/{row['grid_size']}/{row['distance_bin']}", []
        ).append(row)
    aggregates = {}
    for name, group in sorted(groups.items()):
        count = len(group)
        successes = [row for row in group if row["is_success"]]
        aggregates[name] = {
            "episodes": count,
            "successes": len(successes),
            "collisions": sum(bool(row["crashed"]) for row in group),
            "timeouts": sum(bool(row["is_timeout"]) for row in group),
            "success_rate": len(successes) / max(1, count),
            "two_cell_oscillation_rate": sum(bool(row["two_cell_oscillation"]) for row in group) / max(1, count),
            "longer_loop_rate": sum(bool(row["longer_loop"]) for row in group) / max(1, count),
            "mean_decisions": _mean([row["decisions"] for row in group]),
            "mean_episode_return": _mean([row["episode_return"] for row in group]),
            "mean_final_octile_distance": _mean([row["final_octile_distance"] for row in group]),
            "mean_success_path_cost_gap": _mean([row["path_cost_gap"] for row in successes]),
            "mean_success_path_cost_ratio": _mean([row["path_cost_ratio"] for row in successes]),
            "mean_policy_inference_latency_ms": _mean(
                [row["mean_policy_inference_latency_ms"] for row in group]
            ),
            "mean_masked_action_latency_ms": _mean(
                [row["mean_masked_action_latency_ms"] for row in group]
            ),
        }
    return aggregates

class M2ScalarWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Re-define observation space to just the scalars
        self.observation_space = gym.spaces.Dict({
            "scalars": env.observation_space.spaces["scalars"]
        })
        
    def observation(self, obs):
        return {"scalars": obs["scalars"]}

logger = logging.getLogger(__name__)

def preflight(config_path, out_dir):
    logger.info("=== PHASE C2 PREFLIGHT ===")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    gen = PhaseC2EndpointGenerator(seed=42)
    
    errors = []
    oracle_success_sizes = []
    
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

    # Validation must remain disjoint from every training pair, including the
    # reverse direction of a training route.
    training_pairs = set()
    for size, buckets in gen.train_pool.items():
        for pairs in buckets.values():
            for start, goal in pairs:
                key = (int(size), tuple(start), tuple(goal))
                reverse = (int(size), tuple(goal), tuple(start))
                training_pairs.add(key)
                training_pairs.add(reverse)
    overlap = seen & training_pairs
    if overlap:
        errors.append(f"Training/validation endpoint overlap: {len(overlap)}")
        
    with open(config_path) as f:
        config = json.load(f)
        
    # Test env properties across all grid sizes
    from tools.verification.oracle import solve_astar
    
    test_manifest = [
        {"grid_size": 15, "start": [10, 7], "goal": [12, 4]},
        {"grid_size": 30, "start": [29, 26], "goal": [24, 16]},
        {"grid_size": 50, "start": [5, 16], "goal": [24, 27]},
        {"grid_size": 100, "start": [36, 48], "goal": [12, 33]}
    ]
    gen.val_manifest = test_manifest
    
    try:
        env = PhaseC2Env(config, mode="eval", generator=gen)
        env = PotentialShapingWrapper(env, gamma=config["reward"]["gamma"], lambda_=config["reward"]["lambda_"])
        
        for i in range(4):
            obs, info = env.reset()
            sz = test_manifest[i]["grid_size"]
            
            if info["grid_size"] != sz:
                errors.append(f"Expected grid size {sz}, got {info['grid_size']}")
                
            native_shape = env.unwrapped._v2.grid.shape
            if native_shape != (sz, sz):
                errors.append(f"Expected native shape ({sz}, {sz}), got {native_shape}")
                
            expected_max_dist = sz * np.sqrt(2)
            if not np.isclose(env.max_dist, expected_max_dist):
                errors.append(f"Expected max_dist {expected_max_dist}, got {env.max_dist}")
                
            if obs["global_map"].shape != (8, 32, 32):
                errors.append(f"Global map shape != (8, 32, 32): {obs['global_map'].shape}")
                
            if np.isnan(obs["global_map"]).any() or np.isnan(obs["scalars"]).any():
                errors.append("NaN detected in observations")
                
            # Quick A* oracle test
            path = solve_astar((sz, sz), info["start"], info["goal"], set())
            if not path:
                errors.append(f"No path found for size {sz}")
                continue

            action_by_delta = {
                tuple(int(value) for value in delta): action
                for action, delta in enumerate(env.unwrapped._v2.ACTION_DELTAS)
            }
            step_info = {}

            for step_idx in range(1, len(path)):
                s = path[step_idx-1]
                nxt = path[step_idx]
                dx, dy = nxt[0]-s[0], nxt[1]-s[1]
                a = action_by_delta.get((dx, dy))
                if a is None:
                    errors.append(f"No native action for oracle delta {(dx, dy)} at size {sz}")
                    break
                mask = env.unwrapped.action_masks()
                if not mask[a]:
                    errors.append(f"Oracle selected an illegal action at size {sz}")
                    break
                obs, reward, term, trunc, step_info = env.step(a)
                if step_info.get("crashed", False):
                    errors.append(f"Collision in preflight oracle for size {sz}")
                    break
            if step_info.get("is_success", False):
                oracle_success_sizes.append(sz)
            else:
                errors.append(f"Oracle did not reach the goal for size {sz}")

    except Exception as e:
        errors.append(f"Environment initialization/step failed: {e}")
    finally:
        if "env" in locals():
            env.close()
        
    res = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "oracle_success_sizes": oracle_success_sizes,
    }
    
    with open(out_dir / "preflight_verification.json", "w") as f:
        json.dump(res, f, indent=2)
        
    if errors:
        for e in errors: logger.error(e)
        sys.exit(1)
        
    logger.info("Preflight PASSED.")
    sys.exit(0)


class PhaseC2Runner:
    def __init__(self, config_path, out_dir, model_type="M1", resume=False, device="auto", seed=None, deterministic_cuda=False):
        with open(config_path) as f:
            self.config = json.load(f)
            
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.model_type = model_type
        self.resume = resume
        
        self.seed = seed if seed is not None else self.config.get("training", {}).get("seed", 42)
        self.deterministic_cuda = deterministic_cuda

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        if self.deterministic_cuda:
            # Note: This reduces a major source of nondeterminism in cuDNN convolutions,
            # but does NOT guarantee strict bit-for-bit determinism across differing
            # GPU architectures (e.g. Kaggle T4 vs local RTX), Driver versions, or CUDA toolkits.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        if device not in {"cpu", "cuda", "auto"}:
            raise ValueError(f"Invalid device: {device}. Must be cpu, cuda, or auto.")
        self.device = device
        
        self.checkpoints = [
            25000, 50000, 75000, 100000, 150000, 200000, 250000, 300000
        ]
        self.history = {}
        
        self.generator = PhaseC2EndpointGenerator(seed=self.seed)
        
        self.generator.set_active_sizes(self.config["curriculum"]["stages"][0]["active_sizes"])
        
        self.train_env = PhaseC2Env(self.config, mode="train", generator=self.generator)
        if self.config["reward"]["type"] == "R2-PB-empty-v1":
            self.train_env = PotentialShapingWrapper(
                self.train_env,
                gamma=self.config["reward"]["gamma"],
                lambda_=self.config["reward"]["lambda_"]
            )
        if model_type == "M2":
            self.train_env = M2ScalarWrapper(self.train_env)
            
        if not resume:
            if model_type == "M1":
                from rl_v3.phase_b_policy import PhaseBFeatureExtractor
                self.model = MaskablePPO(
                    "MultiInputPolicy",
                    self.train_env,
                    device=self.device,
                    policy_kwargs={
                        "features_extractor_class": PhaseBFeatureExtractor,
                        "features_extractor_kwargs": {"features_dim": 256}
                    },
                    seed=self.seed,
                    **self.config["model"]
                )
            else:
                self.model = MaskablePPO(
                    "MultiInputPolicy",
                    self.train_env,
                    device=self.device,
                    seed=self.seed,
                    **self.config["model"]
                )
        else:
            # We will handle resume manually by loading the model file.
            pass
            
    def rollout_size(self):
        return int(self.model.n_steps) * int(self.model.n_envs)

    def aligned_interactions(self, requested_interactions):
        requested = int(requested_interactions)
        if requested <= 0:
            raise ValueError("requested interactions must be positive")
        rollout = self.rollout_size()
        return int(math.ceil(requested / rollout) * rollout)

    def checkpoint_plan(self, max_interactions):
        requested = [value for value in self.checkpoints if value < max_interactions]
        requested.append(int(max_interactions))
        plan = []
        for target in requested:
            actual = self.aligned_interactions(target)
            if plan and plan[-1][1] == actual:
                plan[-1] = (target, actual)
            else:
                plan.append((target, actual))
        return plan

    def active_sizes_for_interaction(self, interaction):
        for stage in self.config["curriculum"]["stages"]:
            if interaction <= int(stage["max_interactions"]):
                return [int(size) for size in stage["active_sizes"]]
        return [
            int(size)
            for size in self.config["curriculum"]["stages"][-1]["active_sizes"]
        ]

    def evaluate_and_save(self, ts, requested_ts=None):
        requested_ts = int(ts if requested_ts is None else requested_ts)
        logger.info(f"[{ts}] Evaluating 240 validation routes...")
        val_env = PhaseC2Env(self.config, mode="eval", generator=self.generator)
        if self.config["reward"]["type"] == "R2-PB-empty-v1":
            val_env = PotentialShapingWrapper(
                val_env,
                gamma=self.config["reward"]["gamma"],
                lambda_=self.config["reward"]["lambda_"]
            )
        if self.model_type == "M2":
            val_env = M2ScalarWrapper(val_env)
            
        n_eval = len(self.generator.val_manifest)
        rows = []
        validation_started = time.perf_counter()

        for i in range(n_eval):
            obs, info = val_env.reset()
            scenario = self.generator.val_manifest[i]
            start = tuple(int(value) for value in info["start"])
            goal = tuple(int(value) for value in info["goal"])
            initial_cost = _octile_distance(start, goal)
            trajectory = [start]
            actions = []
            masks = []
            inference_seconds = 0.0
            masked_action_seconds = 0.0
            path_cost = 0.0
            episode_return = 0.0
            invalid_action_count = 0
            episode_budget = int(info["budget"])
            done = False
            while not done:
                masked_action_started = time.perf_counter()
                mask = np.asarray(val_env.unwrapped.action_masks(), dtype=bool)
                inference_started = time.perf_counter()
                action, _ = self.model.predict(
                    obs, deterministic=True, action_masks=mask
                )
                inference_seconds += time.perf_counter() - inference_started
                masked_action_seconds += time.perf_counter() - masked_action_started
                action = int(action)
                if not mask[action]:
                    invalid_action_count += 1
                delta = val_env.unwrapped._v2.ACTION_DELTAS[action]
                path_cost += 1.0 if delta[0] == 0 or delta[1] == 0 else math.sqrt(2.0)
                obs, reward, term, trunc, info = val_env.step(int(action))
                episode_return += float(reward)
                trajectory.append(tuple(int(value) for value in val_env.unwrapped._v2.uav_pos))
                actions.append(action)
                masks.append(mask.astype(int).tolist())
                done = term or trunc
            success = bool(info.get("is_success", False))
            crashed = bool(info.get("crashed", False))
            oscillation = has_two_cell_oscillation(trajectory)
            longer_loop = has_longer_loop(trajectory)
            if success:
                failure_label = "success"
            elif crashed:
                failure_label = "collision"
            elif oscillation:
                failure_label = "two_cell_oscillation"
            elif longer_loop:
                failure_label = "longer_repeated_loop"
            else:
                failure_label = "step_limit_timeout"
            final_distance = _octile_distance(trajectory[-1], goal)
            rows.append({
                "episode": i,
                "grid_size": int(info["grid_size"]),
                "distance_bin": scenario["distance_bin"],
                "start": list(start),
                "goal": list(goal),
                "is_success": success,
                "crashed": crashed,
                "is_timeout": not success and not crashed,
                "failure_label": failure_label,
                "two_cell_oscillation": oscillation,
                "longer_loop": longer_loop,
                "decisions": len(actions),
                "episode_budget": episode_budget,
                "episode_return": episode_return,
                "initial_octile_cost": initial_cost,
                "path_cost": path_cost if success else None,
                "path_cost_gap": path_cost - initial_cost if success else None,
                "path_cost_ratio": path_cost / initial_cost if success else None,
                "final_octile_distance": final_distance,
                "invalid_action_count": invalid_action_count,
                "mean_policy_inference_latency_ms": 1000.0 * inference_seconds / max(1, len(actions)),
                "mean_masked_action_latency_ms": 1000.0 * masked_action_seconds / max(1, len(actions)),
                "trajectory": [list(cell) for cell in trajectory],
                "actions": actions,
                "legal_action_masks": masks,
            })

        aggregates = _aggregate_validation(rows)
        overall = aggregates["all"]
        successes = int(overall["successes"])
        collisions = int(overall["collisions"])
        timeouts = int(overall["timeouts"])
        val_sr = float(overall["success_rate"])
        logger.info(f"[{ts}] Val SR: {val_sr:.3f}")

        evaluation = {
            "schema_version": 1,
            "requested_interactions": requested_ts,
            "completed_interactions": int(ts),
            "checkpoint_is_ppo_update_aligned": int(ts) % self.rollout_size() == 0,
            "model_type": self.model_type,
            "validation_manifest_sha256": self.generator.val_hash,
            "action_masking_applied": True,
            "collision_field": "crashed",
            "invalid_action_count": sum(row["invalid_action_count"] for row in rows),
            "validation_wall_seconds": time.perf_counter() - validation_started,
            "aggregates": aggregates,
            "episodes": rows,
        }
        evaluation_path = self.out_dir / f"evaluation_{ts:06d}.json"
        _write_json_atomic(evaluation_path, evaluation)

        self.history[ts] = {
            "success_rate": val_sr,
            "collisions": collisions,
            "timeouts": timeouts,
            "evaluation_file": evaluation_path.name,
            "action_masking_applied": True,
            "collision_field": "crashed",
            "requested_interactions": requested_ts,
            "completed_interactions": int(ts),
        }
        
        model_path = self.out_dir / f"model_{ts:06d}.zip"
        temporary_model = model_path.with_name(f"{model_path.stem}.tmp.zip")
        self.model.save(str(temporary_model))
        temporary_model.replace(model_path)
        
        gen_state = self.generator.get_state()
        _write_json_atomic(self.out_dir / f"generator_{ts:06d}.json", gen_state)

        self.save_rng_state(self.out_dir / f"rng_{ts:06d}.pt")
            
        status_info = {
            "history": self.history,
            "provenance": {
                "seed": self.seed,
                "deterministic_cuda": self.deterministic_cuda,
                "resume_semantics": "statistically_equivalent",
            },
            "completed_interactions": int(ts),
            "requested_interactions": requested_ts,
            "rollout_size": self.rollout_size(),
        }
        _write_json_atomic(self.out_dir / "status.json", status_info)

        val_env.close()

    def save_rng_state(self, path):
        payload = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }
        path = Path(path)
        temporary = path.with_name(f"{path.name}.tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    @staticmethod
    def restore_rng_state(path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        random.setstate(payload["python"])
        np.random.set_state(payload["numpy"])
        torch.random.set_rng_state(payload["torch_cpu"])
        if torch.cuda.is_available() and payload.get("torch_cuda"):
            torch.cuda.set_rng_state_all(payload["torch_cuda"])

    def run(self, max_interactions=150000):
        plan = self.checkpoint_plan(int(max_interactions))
        self.runtime_checkpoint_plan = [
            {"requested_interactions": requested, "completed_interactions": actual}
            for requested, actual in plan
        ]
        logger.info(
            "Starting Phase C2 training: requested=%s, update-aligned=%s, rollout=%s",
            max_interactions,
            plan[-1][1],
            self.rollout_size(),
        )
        current = int(self.model.num_timesteps)
        if self.resume:
            if current <= 0 or current % self.rollout_size() != 0:
                raise ValueError(
                    f"Resume interaction {current} is not a positive complete PPO "
                    f"rollout boundary (rollout size {self.rollout_size()})"
                )
            if current > plan[-1][1]:
                raise ValueError(
                    f"Resume interaction {current} exceeds the requested update-aligned "
                    f"target {plan[-1][1]}"
                )

        first_learn = not self.resume and current == 0
        for requested_target, actual_target in plan:
            if actual_target < current:
                continue
            if actual_target == current:
                if actual_target not in self.history:
                    self.evaluate_and_save(actual_target, requested_target)
                continue

            active_sizes = self.active_sizes_for_interaction(current + 1)
            self.generator.set_active_sizes(active_sizes)
            remaining = actual_target - current
            logger.info(
                "Training segment %s -> %s interactions (requested checkpoint %s); sizes=%s",
                current,
                actual_target,
                requested_target,
                active_sizes,
            )
            self.model.learn(
                total_timesteps=remaining,
                callback=None,
                reset_num_timesteps=first_learn,
            )
            first_learn = False
            current = int(self.model.num_timesteps)
            if current != actual_target:
                raise RuntimeError(
                    f"PPO interaction accounting mismatch: expected {actual_target}, got {current}"
                )
            self.evaluate_and_save(actual_target, requested_target)

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'preflight':
        config_path = "configs/rl_v3_phase_c2.json"
        out_dir = "runs/uav_phase_c2_preflight"
        if len(sys.argv) > 2:
            out_dir = sys.argv[2]
        preflight(config_path, out_dir)
