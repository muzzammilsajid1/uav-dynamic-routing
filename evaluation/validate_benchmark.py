"""Fail fast if a persisted scenario cannot be reconstructed consistently."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.astar import astar
from baselines.dijkstra import dijkstra
from baselines.dstar_lite import DStarLite
from evaluation.scenario_suite import grid_environment, load_suite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "manifests" / "benchmark_v2.json",
    )
    args = parser.parse_args()
    suite = load_suite(args.manifest)

    for scenario in suite["scenarios"]:
        env = grid_environment(scenario)
        reference = dijkstra(env.start, env.goal, env.get_neighbors)
        heuristic = astar(env.start, env.goal, env.get_neighbors)
        if not reference.found:
            raise RuntimeError(f"No initial path in {scenario['scenario_id']}")
        if not math.isclose(reference.cost, heuristic.cost, abs_tol=1e-9):
            raise RuntimeError(f"A* mismatch in {scenario['scenario_id']}")

        incremental = DStarLite(env, env.start, env.goal)
        state = incremental.compute_shortest_path()
        if not state.found or not math.isclose(
            incremental._g(env.start), reference.cost, abs_tol=1e-9
        ):
            raise RuntimeError(f"D* Lite mismatch in {scenario['scenario_id']}")

    print(
        f"Validated {len(suite['scenarios'])} scenarios across "
        "Dijkstra, A*, and D* Lite"
    )


if __name__ == "__main__":
    main()
