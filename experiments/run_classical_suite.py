"""Evaluate classical planners on every persisted benchmark-v2 scenario."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.dijkstra import dijkstra
from baselines.dstar_lite import run_dstar_lite_replanning
from baselines.replanning import run_astar_replanning, run_naive_replanning
from evaluation.experiment_metadata import collect_environment_metadata
from evaluation.scenario_suite import grid_environment, load_suite

RUNNERS = {
    "astar": run_astar_replanning,
    "dijkstra": run_naive_replanning,
    "dstar_lite": run_dstar_lite_replanning,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "manifests" / "benchmark_v2.json",
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "results" / "classical_suite_raw.csv",
    )
    args = parser.parse_args()
    if args.repetitions < 2:
        raise ValueError("At least two timing repetitions are required")

    raw_manifest = args.manifest.read_bytes()
    manifest_hash = hashlib.sha256(raw_manifest).hexdigest()
    suite = load_suite(args.manifest)
    rows: list[dict[str, object]] = []
    for scenario in suite["scenarios"]:
        initial = grid_environment(scenario)
        reference = dijkstra(initial.start, initial.goal, initial.get_neighbors)
        for planner, runner in RUNNERS.items():
            for repetition in range(1, args.repetitions + 1):
                result = runner(
                    grid_environment(scenario),
                    max_steps=int(scenario["grid_size"]) ** 2,
                )
                rows.append(
                    {
                        "run_id": (
                            f"{scenario['scenario_id']}:{planner}:r{repetition:02d}"
                        ),
                        "manifest_sha256": manifest_hash,
                        "scenario_id": scenario["scenario_id"],
                        "split": scenario["split"],
                        "grid_size": scenario["grid_size"],
                        "planner": planner,
                        "repetition": repetition,
                        "success": result.success,
                        "timed_out": result.timed_out,
                        "initial_path_cost": f"{reference.cost:.9f}",
                        "route_path_cost": f"{result.total_cost:.9f}",
                        "steps_taken": result.steps_taken,
                        "replans": result.replans,
                        "node_expansions": result.node_expansions,
                        "route_compute_time_ms": (
                            f"{result.total_planning_time * 1000:.9f}"
                        ),
                        "mean_decision_latency_ms": (
                            f"{result.total_planning_time * 1000 / max(result.replans, 1):.9f}"
                        ),
                        "replan_events_json": json.dumps(
                            result.replan_events, separators=(",", ":")
                        ),
                    }
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metadata = collect_environment_metadata(PROJECT_ROOT)
    metadata["experiment"] = {
        "suite_id": suite["suite_id"],
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_hash,
        "repetitions": args.repetitions,
        "raw_results": str(args.out),
    }
    metadata_path = args.out.with_name("classical_suite_environment.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} runs to {args.out}")


if __name__ == "__main__":
    main()
