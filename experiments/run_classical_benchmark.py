"""Run repeated, matched Dijkstra and A* dynamic-routing evaluations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.dijkstra import dijkstra
from baselines.replanning import run_astar_replanning, run_naive_replanning
from envs.grid_environment import DynamicObstacle, GridEnvironment
from evaluation.experiment_metadata import collect_environment_metadata


RUNNERS = {
    "dijkstra": run_naive_replanning,
    "astar": run_astar_replanning,
}


def _cell(value: list[int]) -> tuple[int, int]:
    return int(value[0]), int(value[1])


def _format_cell(value: tuple[int, int]) -> str:
    return f"{value[0]},{value[1]}"


def _load_manifest(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported scenario-manifest schema")
    return manifest, hashlib.sha256(raw).hexdigest()


def run_benchmark(
    manifest_path: Path,
    planners: list[str],
    repetitions: int,
    max_steps: int,
) -> tuple[list[dict[str, object]], str]:
    if repetitions < 2:
        raise ValueError("At least two repetitions are required for timing uncertainty")

    manifest, manifest_hash = _load_manifest(manifest_path)
    config = manifest["environment"]
    dynamic = [
        DynamicObstacle(
            cell=_cell(item["cell"]),
            period=int(item["period"]),
            initial_state=str(item["initial_state"]),
        )
        for item in config["dynamic_obstacles"]
    ]

    rows: list[dict[str, object]] = []
    for planner_name in planners:
        runner = RUNNERS[planner_name]
        for scenario in manifest["scenarios"]:
            start = _cell(scenario["start"])
            goal = _cell(scenario["goal"])
            initial_env = GridEnvironment(
                size=int(config["size"]),
                obstacle_density=float(config["obstacle_density"]),
                diagonal=bool(config["diagonal_movement"]),
                start=start,
                goal=goal,
                dynamic_obstacles=dynamic,
            )
            static_result = dijkstra(start, goal, initial_env.get_neighbors)

            for repetition in range(1, repetitions + 1):
                env = GridEnvironment(
                    size=int(config["size"]),
                    obstacle_density=float(config["obstacle_density"]),
                    diagonal=bool(config["diagonal_movement"]),
                    start=start,
                    goal=goal,
                    dynamic_obstacles=dynamic,
                )
                result = runner(env, max_steps=max_steps)
                event_times_ms = [
                    float(event["duration"]) * 1000
                    for event in result.replan_events
                ]
                rows.append(
                    {
                        "run_id": (
                            f"{manifest['benchmark_id']}:{scenario['scenario_id']}:"
                            f"{planner_name}:r{repetition:02d}"
                        ),
                        "manifest_sha256": manifest_hash,
                        "benchmark_id": manifest["benchmark_id"],
                        "scenario_id": scenario["scenario_id"],
                        "planner": planner_name,
                        "repetition": repetition,
                        "start": _format_cell(start),
                        "goal": _format_cell(goal),
                        "success": result.success,
                        "timed_out": result.timed_out,
                        "initial_path_cost": f"{static_result.cost:.9f}",
                        "route_path_cost": f"{result.total_cost:.9f}",
                        "adaptability_extra_cost": (
                            f"{result.total_cost - static_result.cost:.9f}"
                            if result.success and static_result.found
                            else "NA"
                        ),
                        "steps_taken": result.steps_taken,
                        "replans": result.replans,
                        "node_expansions": result.node_expansions,
                        "route_compute_time_ms": f"{result.total_planning_time * 1000:.9f}",
                        "median_decision_latency_ms": (
                            f"{statistics.median(event_times_ms):.9f}"
                            if event_times_ms
                            else "NA"
                        ),
                        "replan_events_json": json.dumps(
                            result.replan_events, separators=(",", ":")
                        ),
                    }
                )
    return rows, manifest_hash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "manifests" / "week3_dynamic_50.json",
    )
    parser.add_argument(
        "--planners",
        nargs="+",
        choices=sorted(RUNNERS),
        default=sorted(RUNNERS),
    )
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "results" / "classical_dynamic_raw.csv",
    )
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "results" / "classical_dynamic_environment.json",
    )
    args = parser.parse_args()

    rows, manifest_hash = run_benchmark(
        args.manifest, args.planners, args.repetitions, args.max_steps
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metadata = collect_environment_metadata(PROJECT_ROOT)
    metadata["experiment"] = {
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_hash,
        "planners": args.planners,
        "repetitions": args.repetitions,
        "max_steps": args.max_steps,
        "timing_protocol": (
            "perf_counter around each planner call; route time is their cumulative sum"
        ),
        "raw_results": str(args.out),
    }
    args.metadata_out.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(rows)} raw runs to {args.out}")
    print(f"Wrote environment metadata to {args.metadata_out}")


if __name__ == "__main__":
    main()
