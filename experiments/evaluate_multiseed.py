"""Evaluate every trained seed on the persisted benchmark-v2 scenarios."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from stable_baselines3 import DQN

from baselines.dijkstra import dijkstra
from evaluation.experiment_metadata import collect_environment_metadata
from evaluation.scenario_suite import (
    dynamic_obstacles,
    load_suite,
    moving_obstacles,
    stochastic_obstacles,
    traversal_penalties,
)
from rl_agent.double_dqn import DoubleDQN
from rl_agent.uav_env import CELL_FREE, CELL_NO_FLY, CELL_OBSTACLE, UAVRoutingEnv


def _search_cost(env: UAVRoutingEnv, start: tuple[int, int]) -> float:
    result = dijkstra(
        start,
        tuple(int(value) for value in env.goal_pos),
        lambda node: [
            (tuple(int(value) for value in neighbor), cost)
            for neighbor, cost in env.get_neighbors(np.asarray(node))
        ],
    )
    return result.cost


def _prepare_env(
    scenario: dict,
    variant: dict,
    dynamics_timing: str = "post_move_observed",
) -> tuple[UAVRoutingEnv, dict]:
    obstacles = dynamic_obstacles(scenario)
    stochastic = stochastic_obstacles(scenario)
    moving = moving_obstacles(scenario)
    penalties = traversal_penalties(scenario)
    env = UAVRoutingEnv(
        grid_size=int(scenario["grid_size"]),
        obstacle_density=0.0,
        no_fly_density=0.0,
        fixed_grid=True,
        dynamic_obstacles_enabled=bool(obstacles),
        dynamic_obstacles=obstacles,
        stochastic_obstacles=stochastic,
        moving_obstacles=moving,
        dynamics_seed=scenario.get("dynamics_seed"),
        traversal_penalties=penalties,
        no_fly_mode=str(scenario.get("no_fly_mode") or "penalty"),
        no_fly_penalty=float(scenario.get("no_fly_penalty", 5.0)),
        sensor_noise_probability=float(
            scenario.get("sensor_noise_probability", 0.0)
        ),
        seed=0,
        potential_shaping_enabled=bool(variant["potential_shaping"]),
        observation_mode=str(variant["observation_mode"]),
        dynamics_timing=dynamics_timing,
    )
    env.reset(seed=0)
    env.grid = np.full(
        (int(scenario["grid_size"]), int(scenario["grid_size"])),
        CELL_FREE,
        dtype=np.int32,
    )
    for cell in scenario["blocked"]:
        env.grid[tuple(cell)] = CELL_OBSTACLE
    for cell in scenario.get("no_fly_cells", []):
        env.grid[tuple(cell)] = CELL_NO_FLY
    for obstacle in obstacles:
        env.grid[obstacle.cell] = (
            CELL_OBSTACLE if obstacle.initial_state == "blocked" else CELL_FREE
        )
    for obstacle in stochastic:
        env.grid[obstacle.cell] = (
            CELL_OBSTACLE if obstacle.initial_state == "blocked" else CELL_FREE
        )
    for obstacle in moving:
        env.grid[obstacle.path[obstacle.initial_index]] = CELL_OBSTACLE
    env._initial_grid = env.grid.copy()
    env._distance_table = None
    env.uav_pos = np.asarray(scenario["start"], dtype=np.int32)
    env.goal_pos = np.asarray(scenario["goal"], dtype=np.int32)
    env.current_step = 0
    env._elapsed_steps = 0
    env._dynamics_rng = np.random.default_rng(scenario.get("dynamics_seed"))
    env._moving_indices = [
        obstacle.initial_index for obstacle in moving
    ]
    env._last_dynamic_changes = []
    env.last_action = None
    env.visited_cells.clear()
    env.visited_cells.add(tuple(env.uav_pos))
    env._last_prev_pos = env.uav_pos.copy()
    return env, env._build_observation()


def _evaluate_route(
    model: DQN,
    scenario: dict,
    variant: dict,
    dynamics_timing: str = "post_move_observed",
) -> tuple[dict[str, object], list[dict[str, object]]]:
    env, observation = _prepare_env(scenario, variant, dynamics_timing)
    initial_optimal_cost = _search_cost(
        env, tuple(int(value) for value in env.uav_pos)
    )
    path_cost = 0.0
    predict_time = 0.0
    steps = 0
    events: list[dict[str, object]] = []
    open_events: list[dict[str, object]] = []
    done = False
    info: dict = {}

    while not done:
        grid_before_action = env.grid.copy()
        started = time.perf_counter()
        action, _ = model.predict(observation, deterministic=True)
        decision_duration = time.perf_counter() - started
        predict_time += decision_duration
        for event in events:
            if (
                event.get("reaction_time_ms") is None
                and steps >= int(event["change_step"])
            ):
                event["reaction_time_ms"] = decision_duration * 1000
        action_index = int(action)
        delta = env.ACTION_DELTAS[action_index]
        move_cost = 1.0 if delta[0] == 0 or delta[1] == 0 else math.sqrt(2)
        observation, _, terminated, truncated, info = env.step(action_index)
        steps += 1
        if not info.get("crashed", False):
            path_cost += move_cost + float(info.get("energy_penalty", 0.0))
        done = terminated or truncated

        if info["dynamic_changes"]:
            event_position = tuple(int(value) for value in env.uav_pos)
            post_change_cost = _search_cost(env, event_position)
            post_change_grid = env.grid
            try:
                env.grid = grid_before_action
                pre_change_cost = _search_cost(env, event_position)
            finally:
                env.grid = post_change_grid
            event = {
                "event_index": len(events) + 1,
                "change_step": steps,
                "changed_cells_json": json.dumps(info["dynamic_changes"]),
                "pre_change_optimal_cost": pre_change_cost,
                "post_change_optimal_cost": post_change_cost,
                "optimal_cost_delta": (
                    post_change_cost - pre_change_cost
                    if (
                        math.isfinite(post_change_cost)
                        and math.isfinite(pre_change_cost)
                    )
                    else float("inf")
                ),
                "extra_optimal_cost": (
                    max(0.0, post_change_cost - pre_change_cost)
                    if (
                        math.isfinite(post_change_cost)
                        and math.isfinite(pre_change_cost)
                    )
                    else float("inf")
                ),
                "recovery_steps": None,
                "reaction_time_ms": None,
            }
            events.append(event)
            if post_change_cost <= pre_change_cost:
                event["recovery_steps"] = 0
            else:
                open_events.append(event)

        if open_events:
            remaining_cost = _search_cost(
                env, tuple(int(value) for value in env.uav_pos)
            )
            for event in list(open_events):
                if remaining_cost <= float(event["pre_change_optimal_cost"]):
                    event["recovery_steps"] = steps - int(event["change_step"])
                    open_events.remove(event)

    success = bool(info.get("is_success", False))
    env.close()
    route = {
        "success": success,
        "timed_out": not success and not bool(info.get("crashed", False)),
        "crashed": bool(info.get("crashed", False)),
        "steps_taken": steps,
        "route_path_cost": path_cost if success else "NA",
        "initial_optimal_cost": initial_optimal_cost,
        "path_cost_gap": (
            path_cost - initial_optimal_cost if success else "NA"
        ),
        "route_compute_time_ms": predict_time * 1000,
        "mean_decision_latency_ms": predict_time * 1000 / max(steps, 1),
        "dynamic_event_count": len(events),
    }
    for event in events:
        event["post_change_success"] = success
        event["route_steps"] = steps
    return route, events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "research_experiments.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "manifests" / "benchmark_v2.json",
    )
    parser.add_argument("--variant", default="full")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--checkpoint-stage", default="02_dynamic_full_final.zip"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "results" / "rl_suite_raw.csv",
    )
    parser.add_argument(
        "--events-out",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "results" / "rl_adaptability_events.csv",
    )
    args = parser.parse_args()
    if args.repetitions < 2:
        raise ValueError("At least two timing repetitions are required")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    variant = config["variants"][args.variant]
    seeds = args.seeds or config["policy_seeds"]
    suite = load_suite(args.manifest)
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    model_class = DoubleDQN if variant["double_dqn"] else DQN
    rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []

    for seed in seeds:
        training_metadata_path = (
            PROJECT_ROOT
            / "evaluation"
            / "results"
            / f"training_{args.variant}_seed_{int(seed):03d}.json"
        )
        if not training_metadata_path.exists():
            raise FileNotFoundError(
                f"Missing completed-training metadata: {training_metadata_path}"
            )
        training_metadata = json.loads(
            training_metadata_path.read_text(encoding="utf-8")
        )
        training_record = training_metadata["training"]
        if training_record.get("smoke_test"):
            raise RuntimeError(
                f"Refusing to evaluate smoke-test checkpoint for seed {seed}"
            )
        if not training_record.get("stages"):
            raise RuntimeError(f"No completed training stages for seed {seed}")
        source_snapshot = training_metadata.get("training_source_snapshot")
        if not source_snapshot or not source_snapshot.get("sha256"):
            raise RuntimeError(
                f"Missing training-source provenance for seed {seed}; run "
                "scripts/capture_training_provenance.py"
            )
        checkpoint = (
            PROJECT_ROOT
            / "models"
            / "research"
            / args.variant
            / f"seed_{int(seed):03d}"
            / args.checkpoint_stage
        )
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing trained checkpoint: {checkpoint}")
        model_env = UAVRoutingEnv(
            grid_size=int(config["grid_size"]),
            obstacle_density=0.0,
            fixed_grid=True,
            seed=int(config["layout_seed"]),
            potential_shaping_enabled=bool(variant["potential_shaping"]),
            observation_mode=str(variant["observation_mode"]),
            dynamics_timing=str(config["dynamics_timing"]),
        )
        model = model_class.load(checkpoint, env=model_env, device="auto")

        for scenario in suite["scenarios"]:
            if (
                variant["observation_mode"] == "full"
                and int(scenario["grid_size"]) != int(config["grid_size"])
            ):
                continue
            for repetition in range(1, args.repetitions + 1):
                route, events = _evaluate_route(
                    model,
                    scenario,
                    variant,
                    str(config["dynamics_timing"]),
                )
                run_id = (
                    f"{args.variant}:seed{int(seed):03d}:"
                    f"{scenario['scenario_id']}:r{repetition:02d}"
                )
                rows.append(
                    {
                        "run_id": run_id,
                        "manifest_sha256": manifest_hash,
                        "variant": args.variant,
                        "policy_seed": seed,
                        "scenario_id": scenario["scenario_id"],
                        "split": scenario["split"],
                        "grid_size": scenario["grid_size"],
                        "repetition": repetition,
                        **route,
                    }
                )
                for event in events:
                    event_rows.append(
                        {
                            "run_id": run_id,
                            "variant": args.variant,
                            "policy_seed": seed,
                            "scenario_id": scenario["scenario_id"],
                            "split": scenario["split"],
                            "grid_size": scenario["grid_size"],
                            "repetition": repetition,
                            **event,
                        }
                    )
        model.get_env().close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with args.events_out.open("w", newline="", encoding="utf-8") as handle:
        if event_rows:
            writer = csv.DictWriter(handle, fieldnames=list(event_rows[0]))
            writer.writeheader()
            writer.writerows(event_rows)

    metadata = collect_environment_metadata(PROJECT_ROOT)
    metadata["experiment"] = {
        "variant": args.variant,
        "policy_seeds": seeds,
        "checkpoint_stage": args.checkpoint_stage,
        "manifest_sha256": manifest_hash,
        "repetitions": args.repetitions,
    }
    metadata_text = json.dumps(metadata, indent=2) + "\n"
    variant_manifest = args.out.with_name(
        f"rl_{args.variant}_environment.json"
    )
    variant_manifest.write_text(
        metadata_text, encoding="utf-8", newline="\n"
    )
    # Keep this compatibility alias for existing artifact consumers while the
    # variant-specific file preserves each run's machine record permanently.
    args.out.with_name("rl_suite_environment.json").write_text(
        metadata_text, encoding="utf-8", newline="\n"
    )
    print(f"Wrote {len(rows)} RL route runs to {args.out}")
    print(f"Wrote {len(event_rows)} adaptability events to {args.events_out}")


if __name__ == "__main__":
    main()
