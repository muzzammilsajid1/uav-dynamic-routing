"""Fixed-manifest evaluation and failure evidence for Phase B."""
from __future__ import annotations

import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from rl_v3.diagnostics import astar_cost, has_longer_loop, has_two_cell_oscillation
from rl_v3.env_builders import env_from_scenario
from rl_v3.wrappers import PhaseBUAVRoutingWrapper


def evaluate_manifest(model, manifest: dict, pilot: dict, config: dict, output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir = output_dir / "trajectories"
    success_strata: set[tuple] = set()
    rows: list[dict] = []
    for scenario in manifest["scenarios"]:
        row, evidence = evaluate_scenario(model, scenario, pilot, config)
        rows.append(row)
        stratum = (row["grid_size"], row["scenario_family"], row["route_bucket"])
        save = not row["success"] or stratum not in success_strata
        if row["success"]:
            success_strata.add(stratum)
        if save:
            trajectories_dir.mkdir(parents=True, exist_ok=True)
            (trajectories_dir / f"{scenario['scenario_id']}.json").write_text(
                json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
            )
    _write_csv(output_dir / "episodes.csv", rows)
    (output_dir / "aggregates.json").write_text(
        json.dumps(aggregate_metrics(rows), indent=2) + "\n", encoding="utf-8"
    )
    return rows


def evaluate_scenario(model, scenario: dict, pilot: dict, config: dict) -> tuple[dict, dict]:
    training = config["training"]
    budget = max(
        int(training["minimum_episode_budget"]),
        int(math.ceil(float(training["episode_budget_astar_multiplier"]) * scenario["initial_astar_cost"])),
    )
    v2, _ = env_from_scenario(
        scenario, potential_shaping_enabled=False,
        dynamics_timing="post_move_observed", max_steps=budget,
    )
    wrapper = PhaseBUAVRoutingWrapper(
        v2,
        observation_family=pilot["observation"],
        reward_family=pilot["reward"],
        local_size=int(config["observation"]["local_size"]),
        global_size=int(config["observation"]["global_size"]),
        step_penalty_total=float(config["reward"]["step_penalty_total_per_episode_budget"]),
        progress_coefficient=float(config["reward"]["R2_progress_coefficient"]),
        gamma=float(config["reward"]["R2_discount"]),
    )
    wrapper.visit_counts = {tuple(int(v) for v in v2.uav_pos): 1}
    observation = wrapper._observation()
    trajectory = [tuple(int(v) for v in v2.uav_pos)]
    rewards: list[float] = []
    actions: list[int] = []
    masks: list[list[int]] = []
    decision_seconds = 0.0
    path_cost = 0.0
    dynamic_events = 0
    fresh_costs = [float(scenario["initial_astar_cost"])]
    done = False
    info: dict = {}
    episode_started = time.perf_counter()
    while not done:
        mask = wrapper.action_masks()
        started = time.perf_counter()
        action, _ = model.predict(observation, deterministic=True, action_masks=mask)
        decision_seconds += time.perf_counter() - started
        action = int(action)
        delta = v2.ACTION_DELTAS[action]
        observation, reward, terminated, truncated, info = wrapper.step(action)
        if not info.get("crashed", False):
            path_cost += 1.0 if delta[0] == 0 or delta[1] == 0 else math.sqrt(2.0)
            path_cost += float(info.get("energy_penalty", 0.0))
        trajectory.append(tuple(int(v) for v in v2.uav_pos))
        rewards.append(float(reward))
        actions.append(action)
        masks.append(mask.astype(int).tolist())
        if info.get("dynamic_changes"):
            dynamic_events += 1
            cost, _ = astar_cost(v2, trajectory[-1])
            if math.isfinite(cost):
                fresh_costs.append(float(cost))
        done = bool(terminated or truncated)
    episode_seconds = time.perf_counter() - episode_started
    success = bool(info.get("is_success", False))
    collision = bool(info.get("crashed", False))
    oscillation = has_two_cell_oscillation(trajectory)
    loop = has_longer_loop(trajectory)
    excessive = bool(path_cost > 2.0 * float(scenario["initial_astar_cost"]))
    if success:
        failure = "success"
    elif collision:
        failure = "collision"
    elif oscillation:
        failure = "two_cell_oscillation"
    elif loop:
        failure = "longer_repeated_loop"
    elif excessive:
        failure = "excessive_detour"
    else:
        failure = "step_limit_timeout"
    row = {
        "scenario_id": scenario["scenario_id"],
        "grid_size": int(scenario["grid_size"]),
        "route_bucket": scenario["route_bucket"],
        "scenario_family": scenario["scenario_family"],
        "success": success,
        "collision": collision,
        "timeout": not success and not collision,
        "two_cell_oscillation": oscillation,
        "longer_loop": loop,
        "excessive_detour": excessive,
        "failure_label": failure,
        "post_change_completion": success if dynamic_events else "",
        "dynamic_event_count": dynamic_events,
        "path_cost": path_cost if success else "",
        "fresh_astar_cost": fresh_costs[0],
        "path_cost_gap": path_cost - fresh_costs[0] if success else "",
        "decisions": len(actions),
        "mean_decision_latency_ms": 1000.0 * decision_seconds / max(1, len(actions)),
        "decision_computation_seconds": decision_seconds,
        "episode_computation_seconds": episode_seconds,
    }
    evidence = {
        "summary": row,
        "trajectory": [list(cell) for cell in trajectory],
        "actions": actions,
        "legal_action_masks": masks,
        "rewards": rewards,
        "fresh_astar_costs_at_initial_and_dynamic_events": fresh_costs,
    }
    wrapper.close()
    return row, evidence


def aggregate_metrics(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups["all"].append(row)
        groups[f"scale/{row['grid_size']}"] .append(row)
        groups[f"family/{row['scenario_family']}"] .append(row)
        groups[f"route/{row['route_bucket']}"] .append(row)
        groups[f"scale_family/{row['grid_size']}/{row['scenario_family']}"] .append(row)
    output = {}
    for name, group in sorted(groups.items()):
        n = len(group)
        successful = [row for row in group if row["success"]]
        output[name] = {
            "episodes": n,
            "success_rate": sum(bool(r["success"]) for r in group) / n,
            "collision_rate": sum(bool(r["collision"]) for r in group) / n,
            "timeout_rate": sum(bool(r["timeout"]) for r in group) / n,
            "two_cell_oscillation_rate": sum(bool(r["two_cell_oscillation"]) for r in group) / n,
            "longer_loop_rate": sum(bool(r["longer_loop"]) for r in group) / n,
            "excessive_detour_rate": sum(bool(r["excessive_detour"]) for r in group) / n,
            "mean_path_cost_gap_successes": _mean([float(r["path_cost_gap"]) for r in successful]),
            "mean_decisions": _mean([float(r["decisions"]) for r in group]),
            "mean_decision_latency_ms": _mean([float(r["mean_decision_latency_ms"]) for r in group]),
            "total_attempted_episode_computation_seconds": sum(float(r["episode_computation_seconds"]) for r in group),
            "successful_route_computation_seconds": sum(float(r["episode_computation_seconds"]) for r in successful),
        }
    return output


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
