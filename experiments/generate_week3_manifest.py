"""Persist the deterministic Week 3 scenario set with stable IDs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from envs.grid_environment import default_dynamic_obstacles
from evaluation.week3_scenarios import generate_week3_pairs


def build_manifest(n_pairs: int, seed: int, min_static_cost: float) -> dict:
    dynamic = default_dynamic_obstacles()
    pairs = generate_week3_pairs(
        n_pairs=n_pairs,
        size=15,
        seed=seed,
        min_static_cost=min_static_cost,
        dynamic_obstacles=dynamic,
    )
    return {
        "schema_version": 2,
        "benchmark_id": "week3-dynamic-v1",
        "description": "Paired dynamic 15x15 benchmark used by all planning methods.",
        "dynamics_timing": "post_move_observed",
        "generation": {
            "seed": seed,
            "minimum_static_path_cost": min_static_cost,
        },
        "environment": {
            "size": 15,
            "obstacle_density": 0.0,
            "diagonal_movement": True,
            "orthogonal_cost": 1.0,
            "diagonal_cost": "sqrt(2)",
            "dynamic_obstacles": [
                {
                    "cell": list(obstacle.cell),
                    "period": obstacle.period,
                    "initial_state": obstacle.initial_state,
                }
                for obstacle in dynamic
            ],
        },
        "scenarios": [
            {
                "scenario_id": f"W3D-{index:03d}",
                "start": list(start),
                "goal": list(goal),
            }
            for index, (start, goal) in enumerate(pairs, start=1)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-pairs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-static-cost", type=float, default=5.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "manifests" / "week3_dynamic_50.json",
    )
    args = parser.parse_args()

    manifest = build_manifest(args.n_pairs, args.seed, args.min_static_cost)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(manifest['scenarios'])} scenarios to {args.out}")


if __name__ == "__main__":
    main()
