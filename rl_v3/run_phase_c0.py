"""Phase C0: single-scenario overfitting test for RL V3 MaskablePPO.

Scientific question: Can the pipeline master one trivial fixed navigation task?

Usage:
    python -m rl_v3.run_phase_c0 preflight
    python -m rl_v3.run_phase_c0 train
    python -m rl_v3.run_phase_c0 report
    python -m rl_v3.run_phase_c0 all
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
from sb3_contrib import MaskablePPO

from rl_v3.phase_b_policy import PhaseBFeatureExtractor, model_parameter_count
from rl_v3.phase_c0_env import PhaseC0Env, _astar_cost
from rl_v3.scenario_generation import write_stable_json  # kept for potential future use

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "rl_v3_phase_c0.json"
RUN_ROOT = ROOT / "runs" / "rl_v3" / "phase_c0"

ACTION_NAMES = {
    0: "N", 1: "S", 2: "W", 3: "E",
    4: "NW", 5: "NE", 6: "SW", 7: "SE",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _json_safe(obj):
    """Recursively convert numpy scalars to native Python types for JSON."""
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


def _hardware() -> dict:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "logical_cpus": psutil.cpu_count(),
        "memory_gib": round(psutil.virtual_memory().total / 2**30, 2),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }


# ---------------------------------------------------------------------------
# Preflight verification
# ---------------------------------------------------------------------------

def run_preflight() -> dict:
    """Validate the scenario before any training."""
    from rl_agent.uav_env import UAVRoutingEnv
    config = load_config()
    sc = config["scenario"]
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    start = tuple(int(v) for v in sc["start"])
    goal  = tuple(int(v) for v in sc["goal"])
    grid_size = int(sc["grid_size"])

    # 1. A* cost
    octile = _astar_cost(start, goal)

    # 2. Build environment
    env = PhaseC0Env(config)
    obs, info = env.reset()

    # 3. Verify reset consistency -- second reset must return same start/goal
    obs2, info2 = env.reset()
    start_consistent = (info2["start"] == list(start))
    goal_consistent  = (info2["goal"]  == list(goal))
    obs_consistent   = all(
        np.allclose(obs[k], obs2[k]) for k in obs
    )

    # 4. Verify observation channels
    local_map  = obs["local_map"]    # shape (8, 11, 11)
    global_map = obs["global_map"]   # shape (8, 32, 32)
    scalars    = obs["scalars"]

    local_center = local_map.shape[1] // 2
    agent_in_local_center = float(local_map[3, local_center, local_center])

    # Goal visible in local if within 5 cells
    dr = abs(goal[0] - start[0]); dc = abs(goal[1] - start[1])
    goal_in_local_view = dr <= 5 and dc <= 5
    if goal_in_local_view:
        goal_local_r = local_center + (goal[0] - start[0])
        goal_local_c = local_center + (goal[1] - start[1])
        goal_in_local = float(local_map[4, goal_local_r, goal_local_c])
    else:
        goal_in_local = None

    # Agent in global map
    gs = grid_size
    gmap_sz = global_map.shape[1]
    agent_global_r = min(gmap_sz-1, int(start[0] * gmap_sz / gs))
    agent_global_c = min(gmap_sz-1, int(start[1] * gmap_sz / gs))
    agent_in_global = float(global_map[3, agent_global_r, agent_global_c])
    goal_global_r = min(gmap_sz-1, int(goal[0] * gmap_sz / gs))
    goal_global_c = min(gmap_sz-1, int(goal[1] * gmap_sz / gs))
    goal_in_global = float(global_map[4, goal_global_r, goal_global_c])

    channels_not_swapped = (agent_in_local_center == 1.0 and
                            agent_in_global == 1.0 and
                            goal_in_global == 1.0)

    # 5. Verify action masks
    mask = env.action_masks()
    mask_check = {}
    v2 = env._v2
    for action_idx, delta in enumerate(v2.ACTION_DELTAS):
        dest = tuple(int(v) for v in (np.array(start) + delta))
        in_bounds = 0 <= dest[0] < grid_size and 0 <= dest[1] < grid_size
        expected = in_bounds  # empty grid, all in-bounds moves legal
        actual = bool(mask[action_idx])
        mask_check[ACTION_NAMES[action_idx]] = {
            "action": action_idx,
            "delta": [int(v) for v in delta],
            "destination": [int(v) for v in dest],
            "expected_legal": expected,
            "actual_legal": actual,
            "correct": expected == actual,
        }
    all_masks_correct = all(v["correct"] for v in mask_check.values())

    # 6. Manual step trace: walk 3 legal steps, verify transitions and rewards
    legal_actions = [i for i, m in enumerate(mask) if m]
    step_trace = []
    env.reset()
    for step_i, act in enumerate(legal_actions[:3]):
        obs_before = env._obs()
        obs_step, rew, term, trunc, info_step = env.step(act)
        step_trace.append({
            "step": step_i + 1,
            "action": act,
            "action_name": ACTION_NAMES[act],
            "reward": float(rew),
            "terminated": bool(term),
            "truncated": bool(trunc),
            "crashed": bool(info_step.get("crashed", False)),
            "is_success": bool(info_step.get("is_success", False)),
            "uav_pos": list(int(v) for v in env._v2.uav_pos),
            "obs_changed": not np.allclose(obs_before["local_map"], obs_step["local_map"]),
        })
        if term or trunc:
            break

    obs_changes_on_move = all(s["obs_changed"] for s in step_trace)

    # NaN / Inf checks
    no_nan = all(not np.any(np.isnan(obs[k])) for k in obs)
    no_inf = all(not np.any(np.isinf(obs[k])) for k in obs)

    # Scalars sanity: goal direction
    scalar_dr = float(scalars[0])  # (goal_r - uav_r) / norm
    scalar_dc = float(scalars[1])  # (goal_c - uav_c) / norm
    expected_dr = (goal[0] - start[0]) / max(grid_size - 1, 1)
    expected_dc = (goal[1] - start[1]) / max(grid_size - 1, 1)

    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": {
            "scenario_id": sc["scenario_id"],
            "grid_size": grid_size,
            "start": list(start),
            "goal": list(goal),
            "max_steps": env._max_steps,
        },
        "astar": {
            "octile_cost": round(octile, 6),
            "optimal_route_description": f"5 diagonal SE + 5 straight S (or equivalent)",
        },
        "reset_consistency": {
            "start_consistent": start_consistent,
            "goal_consistent": goal_consistent,
            "obs_consistent": obs_consistent,
            "verdict": "PASS" if (start_consistent and goal_consistent and obs_consistent) else "FAIL",
        },
        "observation_channels": {
            "agent_in_local_center": agent_in_local_center,
            "agent_in_global": agent_in_global,
            "goal_in_global": goal_in_global,
            "goal_in_local": goal_in_local,
            "channels_not_swapped": channels_not_swapped,
            "verdict": "PASS" if channels_not_swapped else "FAIL",
        },
        "action_mask_check": {
            "all_correct": all_masks_correct,
            "details": mask_check,
            "verdict": "PASS" if all_masks_correct else "FAIL",
        },
        "manual_step_trace": step_trace,
        "observations_change_on_move": {
            "result": obs_changes_on_move,
            "verdict": "PASS" if obs_changes_on_move else "FAIL",
        },
        "nan_inf_check": {
            "no_nan": no_nan,
            "no_inf": no_inf,
            "verdict": "PASS" if (no_nan and no_inf) else "FAIL",
        },
        "scalars_sanity": {
            "scalar_dr": scalar_dr,
            "expected_dr": expected_dr,
            "scalar_dc": scalar_dc,
            "expected_dc": expected_dc,
            "match": abs(scalar_dr - expected_dr) < 1e-5 and abs(scalar_dc - expected_dc) < 1e-5,
        },
        "hardware": _hardware(),
    }

    all_pass = all(
        result[k]["verdict"] == "PASS"
        for k in ["reset_consistency","observation_channels",
                  "action_mask_check","observations_change_on_move","nan_inf_check"]
    )
    result["overall_verdict"] = "PASS" if all_pass else "FAIL"

    preflight_path = RUN_ROOT / "preflight_verification.json"
    _write_json(preflight_path, result)
    env.close()
    print(f"Preflight: {result['overall_verdict']}")
    print(json.dumps({k: result[k]["verdict"] for k in
        ["reset_consistency","observation_channels","action_mask_check",
         "observations_change_on_move","nan_inf_check"]}, indent=2))
    return result


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def build_model(config: dict, env: PhaseC0Env) -> MaskablePPO:
    tr = config["training"]
    pol = config["policy"]
    return MaskablePPO(
        "MultiInputPolicy", env,
        learning_rate=float(tr["learning_rate"]),
        n_steps=int(tr["n_steps"]),
        batch_size=int(tr["batch_size"]),
        n_epochs=int(tr["n_epochs"]),
        gamma=float(tr["gamma"]),
        gae_lambda=float(tr["gae_lambda"]),
        ent_coef=float(tr["ent_coef"]),
        seed=int(config["seed"]),
        device="cpu",
        verbose=1,
        policy_kwargs={
            "features_extractor_class": PhaseBFeatureExtractor,
            "features_extractor_kwargs": {"features_dim": int(pol["features_dim"])},
            "net_arch": {"pi": list(pol["net_arch_pi"]), "vf": list(pol["net_arch_vf"])},
            "activation_fn": torch.nn.ReLU,
        },
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_single_scenario(
    model: MaskablePPO, config: dict, n_episodes: int = 100
) -> dict:
    """Run n_episodes deterministically on the fixed scenario and collect metrics."""
    env = PhaseC0Env(config)
    successes = 0
    collisions = 0
    timeouts = 0
    oscillations_2 = 0  # 2-cell oscillation count
    longer_loops = 0
    route_steps_list: list[int] = []
    realized_costs: list[float] = []
    returns: list[float] = []
    trajectories_success: list[list[dict]] = []
    trajectories_failure: list[list[dict]] = []
    action_counts: Counter = Counter()
    failure_patterns: set[str] = set()

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_return = 0.0
        ep_steps = 0
        ep_cost = 0.0
        traj: list[dict] = []
        pos_history: list[tuple] = [tuple(int(v) for v in env._v2.uav_pos)]
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
                collisions += 1
                failure_patterns.add("crash")
            if trunc:
                timeouts += 1
                failure_patterns.add("timeout")

        if info.get("is_success"):
            successes += 1
            route_steps_list.append(ep_steps)
            realized_costs.append(ep_cost)
            if not trajectories_success:
                trajectories_success.append(traj)
        else:
            # Detect oscillation
            if len(pos_history) >= 4:
                osc = any(
                    pos_history[i] == pos_history[i+2]
                    for i in range(len(pos_history) - 2)
                )
                if osc:
                    oscillations_2 += 1
                    failure_patterns.add("oscillation_2")
            # Detect longer loops
            pos_set = Counter(pos_history)
            if any(v >= 3 for v in pos_set.values()):
                longer_loops += 1
                failure_patterns.add("longer_loop")
            if len(trajectories_failure) < 3:
                pattern = "crash" if info.get("crashed") else ("timeout" if trunc else "other")
                if pattern not in {t[0]["action_name"] if t else "" for t in trajectories_failure}:
                    trajectories_failure.append(traj)

        returns.append(ep_return)

    env.close()

    astar_cost = _astar_cost(
        tuple(config["scenario"]["start"]), tuple(config["scenario"]["goal"])
    )
    mean_route_steps = float(np.mean(route_steps_list)) if route_steps_list else float("nan")
    mean_realized_cost = float(np.mean(realized_costs)) if realized_costs else float("nan")

    return {
        "n_episodes": n_episodes,
        "successes": successes,
        "success_rate": successes / n_episodes,
        "collisions": collisions,
        "timeouts": timeouts,
        "oscillations_2cell": oscillations_2,
        "longer_loops": longer_loops,
        "failure_patterns": sorted(failure_patterns),
        "mean_route_steps": round(mean_route_steps, 3),
        "max_route_steps": int(max(route_steps_list)) if route_steps_list else None,
        "mean_realized_cost": round(mean_realized_cost, 6),
        "astar_cost": round(astar_cost, 6),
        "path_cost_gap": round(mean_realized_cost - astar_cost, 6) if realized_costs else None,
        "mean_return": round(float(np.mean(returns)), 6),
        "action_distribution": dict(action_counts),
        "trajectory_success": trajectories_success[:1],
        "trajectories_failure": trajectories_failure[:3],
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_training() -> dict:
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    config = load_config()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    # Verify preflight was done
    preflight_path = RUN_ROOT / "preflight_verification.json"
    if not preflight_path.exists():
        print("Running preflight first...")
        preflight = run_preflight()
    else:
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))

    if preflight.get("overall_verdict") != "PASS":
        raise RuntimeError(f"Preflight FAILED: {preflight}")

    tr = config["training"]
    checkpoints = [int(v) for v in tr["checkpoints"]]
    eval_episodes = int(tr["eval_episodes"])
    early_threshold = int(tr["early_stop_success_threshold"])
    early_consecutive = int(tr["early_stop_consecutive_checkpoints"])

    status_path = RUN_ROOT / "status.json"
    completed = 0
    history: list[dict] = []
    checkpoint_path = None

    # Resume if status exists
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        completed = int(status.get("completed_interactions", 0))
        history = list(status.get("history", []))
        candidate = RUN_ROOT / f"model_{completed:06d}.zip"
        if completed and candidate.exists():
            checkpoint_path = candidate

    env = PhaseC0Env(config)

    if checkpoint_path:
        print(f"Resuming from {checkpoint_path}")
        model = MaskablePPO.load(checkpoint_path, env=env, device="cpu")
    else:
        model = build_model(config, env)

    # Capture initial parameters for change verification
    initial_params = {
        name: param.data.clone()
        for name, param in model.policy.named_parameters()
    }
    params_changed_confirmed = False
    params_changed_at: int | None = None

    param_count = model_parameter_count(model.policy)
    process = psutil.Process()

    consecutive_good = 0
    earliest_mastery: int | None = None

    for target in checkpoints:
        if target <= completed:
            continue
        delta = target - completed

        cpu_start = process.cpu_times()
        wall_start = time.perf_counter()
        model.learn(total_timesteps=delta, reset_num_timesteps=False, progress_bar=False)
        wall = time.perf_counter() - wall_start
        cpu_end = process.cpu_times()
        cpu_seconds = (cpu_end.user + cpu_end.system) - (cpu_start.user + cpu_start.system)
        completed = target

        # Save checkpoint
        model_path = RUN_ROOT / f"model_{target:06d}"
        model.save(model_path)

        # Check parameters changed from init
        if not params_changed_confirmed:
            for name, param in model.policy.named_parameters():
                if name in initial_params and not torch.equal(param.data, initial_params[name]):
                    params_changed_confirmed = True
                    params_changed_at = target
                    break

        # Collect SB3 training stats from logger
        sb3_stats = {}
        try:
            sb3_stats = {
                "value_loss": model.logger.name_to_value.get("train/value_loss"),
                "policy_gradient_loss": model.logger.name_to_value.get("train/policy_gradient_loss"),
                "entropy_loss": model.logger.name_to_value.get("train/entropy_loss"),
                "approx_kl": model.logger.name_to_value.get("train/approx_kl"),
                "clip_fraction": model.logger.name_to_value.get("train/clip_fraction"),
                "clip_range": model.logger.name_to_value.get("train/clip_range"),
                "explained_variance": model.logger.name_to_value.get("train/explained_variance"),
                "learning_rate": model.logger.name_to_value.get("train/learning_rate"),
                "n_updates": model.logger.name_to_value.get("train/n_updates"),
                "fps": model.logger.name_to_value.get("time/fps"),
            }
        except Exception:
            pass

        # Evaluate
        eval_result = evaluate_single_scenario(model, config, n_episodes=eval_episodes)

        # Save trajectory files
        traj_dir = RUN_ROOT / "trajectories" / f"step_{target:06d}"
        traj_dir.mkdir(parents=True, exist_ok=True)
        if eval_result["trajectory_success"]:
            (traj_dir / "success.json").write_text(
                json.dumps(eval_result["trajectory_success"][0], indent=2), encoding="utf-8"
            )
        for fi, ftraj in enumerate(eval_result.get("trajectories_failure", [])):
            pattern = eval_result["failure_patterns"][fi] if fi < len(eval_result["failure_patterns"]) else f"failure_{fi}"
            (traj_dir / f"failure_{pattern}.json").write_text(
                json.dumps(ftraj, indent=2), encoding="utf-8"
            )

        successes = eval_result["successes"]
        item = {
            "interactions": target,
            "training_wall_seconds": round(wall, 2),
            "environment_steps_per_second": round(delta / wall, 1),
            "cpu_utilization_percent": round(100.0 * cpu_seconds / max(wall, 1e-9) / max(1, psutil.cpu_count()), 2),
            "gpu_utilization_percent": None,
            "gpu_note": "CPU build",
            "eval_successes": successes,
            "eval_n_episodes": eval_episodes,
            "eval_success_rate": eval_result["success_rate"],
            "eval_mean_route_steps": eval_result["mean_route_steps"],
            "eval_max_route_steps": eval_result["max_route_steps"],
            "eval_mean_realized_cost": eval_result["mean_realized_cost"],
            "eval_path_cost_gap": eval_result["path_cost_gap"],
            "eval_collisions": eval_result["collisions"],
            "eval_timeouts": eval_result["timeouts"],
            "eval_oscillations_2cell": eval_result["oscillations_2cell"],
            "eval_longer_loops": eval_result["longer_loops"],
            "eval_failure_patterns": eval_result["failure_patterns"],
            "eval_mean_return": eval_result["mean_return"],
            "eval_action_distribution": eval_result["action_distribution"],
            "sb3_stats": sb3_stats,
            "params_changed_confirmed": params_changed_confirmed,
            "params_changed_at": params_changed_at,
        }
        history.append(item)
        print(f"\n[C0] interactions={target:6d}  success={successes}/{eval_episodes}  "
              f"wall={wall:.1f}s  fps={round(delta/wall,0)}")

        # Early stop check
        if successes >= early_threshold:
            consecutive_good += 1
            if earliest_mastery is None:
                earliest_mastery = target
        else:
            consecutive_good = 0
            earliest_mastery = None  # reset if it drops

        # Persist status
        status_payload = {
            "schema_version": 1,
            "completed_interactions": completed,
            "target_interactions": max(checkpoints),
            "parameter_count": param_count,
            "seed": config["seed"],
            "scenario": config["scenario"],
            "history": history,
            "params_changed_confirmed": params_changed_confirmed,
            "params_changed_at": params_changed_at,
            "consecutive_good": consecutive_good,
            "earliest_mastery_interactions": earliest_mastery,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(status_path, status_payload)

        if consecutive_good >= early_consecutive:
            print(f"\n[C0] Early stop: {consecutive_good} consecutive checkpoints >= {early_threshold}/100")
            break

    env.close()

    # Plot learning curve
    _plot_learning_curve(history)

    # Final summary
    summary = _build_summary(config, history, earliest_mastery, param_count, preflight)
    _write_json(RUN_ROOT / "phase_c0_summary.json", summary)
    return summary


def _plot_learning_curve(history: list[dict]) -> None:
    if not history:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle("Phase C0 — Single-Scenario Overfitting Test", fontweight="bold")

    steps = [h["interactions"] for h in history]
    success_rates = [h["eval_success_rate"] for h in history]
    returns = [h["eval_mean_return"] for h in history]
    losses_v = [h["sb3_stats"].get("value_loss") for h in history]
    losses_p = [h["sb3_stats"].get("policy_gradient_loss") for h in history]
    entropy = [h["sb3_stats"].get("entropy_loss") for h in history]
    kl = [h["sb3_stats"].get("approx_kl") for h in history]

    ax = axes[0, 0]
    ax.plot(steps, success_rates, "o-", color="#2196F3")
    ax.axhline(0.99, color="green", linestyle="--", alpha=0.5, label="Target 99%")
    ax.set(xlabel="Interactions", ylabel="Success rate (100 eps)", title="Success Rate")
    ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ax.plot(steps, returns, "s-", color="#FF5722")
    ax.set(xlabel="Interactions", ylabel="Mean episode return", title="Mean Return")
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    valid_v = [(s, v) for s, v in zip(steps, losses_v) if v is not None]
    valid_p = [(s, v) for s, v in zip(steps, losses_p) if v is not None]
    if valid_v: ax.plot(*zip(*valid_v), "o-", label="Value loss", color="#9C27B0")
    if valid_p: ax.plot(*zip(*valid_p), "s-", label="Policy loss", color="#FF9800")
    ax.set(xlabel="Interactions", title="Losses"); ax.legend(); ax.grid(alpha=0.25)

    ax = axes[1, 1]
    valid_e = [(s, v) for s, v in zip(steps, entropy) if v is not None]
    valid_k = [(s, v) for s, v in zip(steps, kl) if v is not None]
    if valid_e: ax.plot(*zip(*valid_e), "o-", label="Entropy loss", color="#4CAF50")
    if valid_k: ax.plot(*zip(*valid_k), "s-", label="Approx KL", color="#F44336")
    ax.set(xlabel="Interactions", title="Entropy & KL"); ax.legend(); ax.grid(alpha=0.25)

    fig.savefig(RUN_ROOT / "learning_curve.png", dpi=180)
    plt.close(fig)
    print(f"Learning curve saved -> {RUN_ROOT / 'learning_curve.png'}")


def _build_summary(config: dict, history: list[dict], earliest_mastery: int | None,
                   param_count: int, preflight: dict) -> dict:
    if history:
        best = max(history, key=lambda h: h["eval_success_rate"])
        final = history[-1]
    else:
        best = final = {}

    mastered = earliest_mastery is not None
    return {
        "schema_version": 1,
        "suite_id": config["suite_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": config["scenario"],
        "seed": config["seed"],
        "parameter_count": param_count,
        "training_checkpoints": len(history),
        "total_interactions_run": final.get("interactions", 0),
        "earliest_mastery_interactions": earliest_mastery,
        "mastered": mastered,
        "best_success_rate": best.get("eval_success_rate"),
        "best_success_at_interactions": best.get("interactions"),
        "final_success_rate": final.get("eval_success_rate"),
        "final_collisions": final.get("eval_collisions"),
        "final_timeouts": final.get("eval_timeouts"),
        "final_oscillations_2cell": final.get("eval_oscillations_2cell"),
        "final_mean_return": final.get("eval_mean_return"),
        "history": history,
        "preflight_verdict": preflight.get("overall_verdict"),
        "hardware": _hardware(),
        "pipeline_diagnosis": _diagnose(history, mastered),
    }


def _diagnose(history: list[dict], mastered: bool) -> dict:
    if not history:
        return {"status": "no_data"}
    final = history[-1]
    rate = final.get("eval_success_rate", 0.0)
    crashes = final.get("eval_collisions", 0)
    timeouts = final.get("eval_timeouts", 0)
    osc = final.get("eval_oscillations_2cell", 0)
    sb3 = final.get("sb3_stats", {})
    ev = sb3.get("explained_variance")
    kl = sb3.get("approx_kl")

    if mastered:
        return {"status": "mastered", "verdict": "Pipeline works correctly on the trivial task."}

    clues = []
    if crashes > 50:
        clues.append("HIGH_CRASH_RATE: mask may not be applied during training, or crash reward too weak")
    if timeouts > 50:
        clues.append("HIGH_TIMEOUT_RATE: agent looping or step budget too tight")
    if osc > 30:
        clues.append("OSCILLATION: policy stuck in 2-cell loop; step penalty or entropy may need adjustment")
    if ev is not None and ev < 0:
        clues.append("NEGATIVE_EXPLAINED_VARIANCE: value function worse than baseline; reward scale issue")
    if kl is not None and kl > 0.5:
        clues.append("HIGH_KL: policy changing too fast; reduce learning rate or n_epochs")
    if rate < 0.10 and len(history) >= 3:
        clues.append("NEAR_ZERO_SUCCESS: possible observation bug, reward never reaches goal, or reset inconsistency")

    # Check params changed
    params_ok = any(h.get("params_changed_confirmed") for h in history)
    if not params_ok:
        clues.append("PARAMS_NEVER_CHANGED: gradient updates may not be occurring")

    return {
        "status": "failed",
        "final_success_rate": rate,
        "likely_defects": clues,
        "verdict": (
            "Pipeline defect more likely than task difficulty — "
            "the trivial 15x15 empty scenario should be solvable." if clues
            else "Unknown failure; collect more diagnostics."
        ),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight","train","report","all"))
    args = parser.parse_args()

    if args.command == "preflight" or args.command == "all":
        result = run_preflight()
        print(json.dumps(result, indent=2))
    if args.command == "train" or args.command == "all":
        summary = run_training()
        print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))
    if args.command == "report":
        status_path = RUN_ROOT / "status.json"
        summary_path = RUN_ROOT / "phase_c0_summary.json"
        if summary_path.exists():
            print(summary_path.read_text(encoding="utf-8"))
        elif status_path.exists():
            print(status_path.read_text(encoding="utf-8"))
        else:
            print("No results found. Run 'train' first.")


if __name__ == "__main__":
    main()
