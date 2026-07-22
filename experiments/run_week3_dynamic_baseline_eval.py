"""Run Simra's Week 3 naive-replanning baseline over a paired scenario set.

This script writes the Dijkstra-side CSV Muzzammil can pair with the RL
Phase 3 results. It intentionally uses obstacle_density=0.0 so the only
changing cells are the three shared default_dynamic_obstacles().
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.dijkstra import dijkstra
from baselines.replanning import run_naive_replanning
from envs.grid_environment import GridEnvironment, default_dynamic_obstacles
from evaluation.week3_scenarios import generate_week3_pairs


def _fmt_cell(cell: tuple[int, int]) -> str:
    return f"{cell[0]},{cell[1]}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the Week 3 naive Dijkstra replanning baseline."
    )
    parser.add_argument("--n-pairs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-static-cost", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "week3_dynamic_baseline_results.csv",
    )
    args = parser.parse_args()

    dynamic_obstacles = default_dynamic_obstacles()
    pairs = generate_week3_pairs(
        n_pairs=args.n_pairs,
        seed=args.seed,
        min_static_cost=args.min_static_cost,
        dynamic_obstacles=dynamic_obstacles,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for scenario_id, (start, goal) in enumerate(pairs, start=1):
        env = GridEnvironment(
            obstacle_density=0.0,
            start=start,
            goal=goal,
            dynamic_obstacles=default_dynamic_obstacles(),
        )
        initial = dijkstra(start, goal, env.get_neighbors)
        result = run_naive_replanning(env, max_steps=args.max_steps)

        rows.append(
            {
                "scenario_id": scenario_id,
                "start": _fmt_cell(start),
                "goal": _fmt_cell(goal),
                "success": result.success,
                "timed_out": result.timed_out,
                "initial_dijkstra_cost": f"{initial.cost:.6f}",
                "dynamic_path_cost": f"{result.total_cost:.6f}",
                "adaptability_extra_cost": (
                    f"{result.total_cost - initial.cost:.6f}"
                    if result.success and initial.found
                    else "NA"
                ),
                "steps_taken": result.steps_taken,
                "replans": result.replans,
                "total_planning_time_ms": f"{result.total_planning_time * 1000:.6f}",
                "replan_events": json.dumps(result.replan_events),
            }
        )

    fieldnames = list(rows[0].keys()) if rows else []
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    successes = [row for row in rows if row["success"] is True]
    costs = [float(row["dynamic_path_cost"]) for row in successes]
    replans = [int(row["replans"]) for row in rows]
    planning_ms = [float(row["total_planning_time_ms"]) for row in rows]
    extra_costs = [
        float(row["adaptability_extra_cost"])
        for row in successes
        if row["adaptability_extra_cost"] != "NA"
    ]

    print(f"Wrote {len(rows)} Week 3 baseline rows to {args.out}")
    print(f"Success rate: {len(successes)}/{len(rows)}")
    if costs:
        print(f"Avg dynamic path cost: {statistics.mean(costs):.3f}")
        print(f"Avg adaptability extra cost: {statistics.mean(extra_costs):.3f}")
    print(f"Avg replans: {statistics.mean(replans):.2f}")
    print(f"Avg total planning time: {statistics.mean(planning_ms):.3f} ms")


if __name__ == "__main__":
    main()
