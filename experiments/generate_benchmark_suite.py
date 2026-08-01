"""Generate persisted generalization and scaling scenarios with stable IDs."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.dijkstra import dijkstra
from envs.grid_environment import (
    DynamicObstacle,
    GridEnvironment,
    MovingObstacle,
    StochasticObstacle,
)
import numpy as np


def _pairs(
    *,
    size: int,
    blocked: set[tuple[int, int]],
    dynamic: list[DynamicObstacle],
    count: int,
    seed: int,
    minimum_cost: float,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    rng = random.Random(seed)
    dynamic_cells = {obstacle.cell for obstacle in dynamic}
    candidates = [
        (row, col)
        for row in range(size)
        for col in range(size)
        if (row, col) not in blocked and (row, col) not in dynamic_cells
    ]
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for _ in range(max(10_000, count * 1_000)):
        if len(pairs) == count:
            break
        start, goal = rng.sample(candidates, 2)
        if (start, goal) in seen:
            continue
        env = GridEnvironment(
            size=size,
            obstacle_density=0.0,
            start=start,
            goal=goal,
            blocked=set(blocked),
            dynamic_obstacles=dynamic,
        )
        result = dijkstra(start, goal, env.get_neighbors)
        if result.found and result.cost >= minimum_cost:
            pairs.append((start, goal))
            seen.add((start, goal))
    if len(pairs) != count:
        raise RuntimeError(f"Generated {len(pairs)}/{count} valid pairs")
    return pairs


def _layout(size: int, density: float, seed: int) -> set[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    count = int(size * size * density)
    indices = rng.choice(size * size, size=count, replace=False)
    rows, columns = np.unravel_index(indices, (size, size))
    return {
        (int(row), int(column))
        for row, column in zip(rows.tolist(), columns.tolist())
    }


def _scenario(
    *,
    scenario_id: str,
    split: str,
    size: int,
    density: float,
    layout_seed: int | None,
    blocked: set[tuple[int, int]],
    dynamic: list[DynamicObstacle],
    start: tuple[int, int],
    goal: tuple[int, int],
    stochastic: list[StochasticObstacle] | None = None,
    moving: list[MovingObstacle] | None = None,
    traversal_penalties: dict[tuple[int, int], float] | None = None,
    dynamics_seed: int | None = None,
    no_fly_cells: set[tuple[int, int]] | None = None,
    no_fly_mode: str | None = None,
    sensor_noise_probability: float = 0.0,
) -> dict:
    return {
        "scenario_id": scenario_id,
        "split": split,
        "grid_size": size,
        "obstacle_density": density,
        "layout_seed": layout_seed,
        "blocked": [list(cell) for cell in sorted(blocked)],
        "dynamic_obstacles": [
            {
                "cell": list(obstacle.cell),
                "period": obstacle.period,
                "initial_state": obstacle.initial_state,
            }
            for obstacle in dynamic
        ],
        "start": list(start),
        "goal": list(goal),
        "stochastic_obstacles": [
            {
                "cell": list(obstacle.cell),
                "toggle_probability": obstacle.toggle_probability,
                "initial_state": obstacle.initial_state,
            }
            for obstacle in (stochastic or [])
        ],
        "moving_obstacles": [
            {
                "path": [list(cell) for cell in obstacle.path],
                "period": obstacle.period,
                "initial_index": obstacle.initial_index,
            }
            for obstacle in (moving or [])
        ],
        "traversal_penalties": [
            {"cell": list(cell), "penalty": penalty}
            for cell, penalty in sorted((traversal_penalties or {}).items())
        ],
        "dynamics_seed": dynamics_seed,
        "no_fly_cells": [list(cell) for cell in sorted(no_fly_cells or set())],
        "no_fly_mode": no_fly_mode,
        "no_fly_penalty": 5.0,
        "sensor_noise_probability": sensor_noise_probability,
    }


def _append_layout_split(
    scenarios: list[dict],
    *,
    prefix: str,
    split: str,
    size: int,
    density: float,
    layout_seeds: list[int],
    pairs_per_layout: int,
    pair_seed: int,
) -> None:
    index = 1
    for layout_seed in layout_seeds:
        blocked = _layout(size, density, layout_seed)
        pairs = _pairs(
            size=size,
            blocked=blocked,
            dynamic=[],
            count=pairs_per_layout,
            seed=pair_seed + layout_seed,
            minimum_cost=max(5.0, size / 3),
        )
        for start, goal in pairs:
            scenarios.append(
                _scenario(
                    scenario_id=f"{prefix}-{index:03d}",
                    split=split,
                    size=size,
                    density=density,
                    layout_seed=layout_seed,
                    blocked=blocked,
                    dynamic=[],
                    start=start,
                    goal=goal,
                )
            )
            index += 1


def build_suite() -> dict:
    scenarios: list[dict] = []
    _append_layout_split(
        scenarios,
        prefix="ID-PAIR",
        split="seen_layout_unseen_pairs",
        size=15,
        density=0.2,
        layout_seeds=[42],
        pairs_per_layout=30,
        pair_seed=1_000,
    )
    _append_layout_split(
        scenarios,
        prefix="OOD-LAYOUT",
        split="unseen_layout_same_density",
        size=15,
        density=0.2,
        layout_seeds=[101, 102, 103],
        pairs_per_layout=10,
        pair_seed=2_000,
    )
    _append_layout_split(
        scenarios,
        prefix="OOD-DENSE",
        split="denser_unseen_layout",
        size=15,
        density=0.3,
        layout_seeds=[201, 202, 203],
        pairs_per_layout=10,
        pair_seed=3_000,
    )

    empty: set[tuple[int, int]] = set()
    rng = random.Random(4_000)
    dynamic_location_index = 1
    for layout_index in range(3):
        cells = rng.sample(
            [(row, col) for row in range(1, 14) for col in range(1, 14)],
            3,
        )
        dynamic = [
            DynamicObstacle(cells[0], 5, "passable"),
            DynamicObstacle(cells[1], 5, "blocked"),
            DynamicObstacle(cells[2], 5, "passable"),
        ]
        pairs = _pairs(
            size=15,
            blocked=empty,
            dynamic=dynamic,
            count=10,
            seed=4_100 + layout_index,
            minimum_cost=5.0,
        )
        for start, goal in pairs:
            scenarios.append(
                _scenario(
                    scenario_id=f"OOD-DYNLOC-{dynamic_location_index:03d}",
                    split="new_dynamic_obstacle_locations",
                    size=15,
                    density=0.0,
                    layout_seed=None,
                    blocked=empty,
                    dynamic=dynamic,
                    start=start,
                    goal=goal,
                )
            )
            dynamic_location_index += 1

    period_sets = [(3, 7, 11), (4, 6, 9), (7, 8, 13)]
    period_index = 1
    default_cells = [(4, 4), (8, 8), (12, 11)]
    for set_index, periods in enumerate(period_sets):
        dynamic = [
            DynamicObstacle(default_cells[0], periods[0], "passable"),
            DynamicObstacle(default_cells[1], periods[1], "blocked"),
            DynamicObstacle(default_cells[2], periods[2], "passable"),
        ]
        pairs = _pairs(
            size=15,
            blocked=empty,
            dynamic=dynamic,
            count=10,
            seed=5_100 + set_index,
            minimum_cost=5.0,
        )
        for start, goal in pairs:
            scenarios.append(
                _scenario(
                    scenario_id=f"OOD-PERIOD-{period_index:03d}",
                    split="changed_toggle_periods",
                    size=15,
                    density=0.0,
                    layout_seed=None,
                    blocked=empty,
                    dynamic=dynamic,
                    start=start,
                    goal=goal,
                )
            )
            period_index += 1

    for size in [15, 30, 50, 100]:
        _append_layout_split(
            scenarios,
            prefix=f"SCALE-{size:03d}",
            split=f"scale_{size}",
            size=size,
            density=0.2,
            layout_seeds=[6_000 + size],
            pairs_per_layout=10,
            pair_seed=7_000,
        )

    for density, label, seed in [(0.1, "10", 8_010), (0.4, "40", 8_040)]:
        _append_layout_split(
            scenarios,
            prefix=f"REAL-DENSITY-{label}",
            split=f"obstacle_density_{label}",
            size=15,
            density=density,
            layout_seeds=[seed],
            pairs_per_layout=10,
            pair_seed=8_100,
        )

    stochastic_pairs = _pairs(
        size=15,
        blocked=empty,
        dynamic=[],
        count=20,
        seed=9_000,
        minimum_cost=5.0,
    )
    stochastic_rng = random.Random(9_100)
    for index, (start, goal) in enumerate(stochastic_pairs, start=1):
        available = [
            (row, col)
            for row in range(15)
            for col in range(15)
            if (row, col) not in {start, goal}
        ]
        cells = stochastic_rng.sample(available, 3)
        probability = 0.05 if index <= 10 else 0.20
        stochastic = [
            StochasticObstacle(cells[0], probability, "passable"),
            StochasticObstacle(cells[1], probability, "blocked"),
            StochasticObstacle(cells[2], probability, "passable"),
        ]
        scenarios.append(
            _scenario(
                scenario_id=f"REAL-STOCH-{index:03d}",
                split=f"stochastic_obstacles_p{int(probability * 100):02d}",
                size=15,
                density=0.0,
                layout_seed=None,
                blocked=empty,
                dynamic=[],
                stochastic=stochastic,
                dynamics_seed=9_200 + index,
                start=start,
                goal=goal,
            )
        )

    moving_paths = [
        [(3, 3), (3, 4), (3, 5), (3, 6)],
        [(7, 10), (8, 10), (9, 10), (10, 10)],
    ]
    excluded = [
        DynamicObstacle(cell, 999, "passable")
        for path in moving_paths
        for cell in path
    ]
    moving_pairs = _pairs(
        size=15,
        blocked=empty,
        dynamic=excluded,
        count=20,
        seed=10_000,
        minimum_cost=5.0,
    )
    for index, (start, goal) in enumerate(moving_pairs, start=1):
        period = 2 if index <= 10 else 4
        moving = [
            MovingObstacle(path=list(path), period=period)
            for path in moving_paths
        ]
        scenarios.append(
            _scenario(
                scenario_id=f"REAL-MOVING-{index:03d}",
                split=f"moving_obstacles_period_{period}",
                size=15,
                density=0.0,
                layout_seed=None,
                blocked=empty,
                dynamic=[],
                moving=moving,
                start=start,
                goal=goal,
            )
        )

    wind_pairs = _pairs(
        size=15,
        blocked=empty,
        dynamic=[],
        count=20,
        seed=11_000,
        minimum_cost=5.0,
    )
    wind_rng = random.Random(11_100)
    for index, (start, goal) in enumerate(wind_pairs, start=1):
        penalty_level = 0.25 if index <= 10 else 1.0
        cells = wind_rng.sample(
            [(row, col) for row in range(15) for col in range(15)],
            34,
        )
        penalties = {cell: penalty_level for cell in cells}
        scenarios.append(
            _scenario(
                scenario_id=f"REAL-WIND-{index:03d}",
                split=f"wind_energy_penalty_{str(penalty_level).replace('.', '_')}",
                size=15,
                density=0.0,
                layout_seed=None,
                blocked=empty,
                dynamic=[],
                traversal_penalties=penalties,
                start=start,
                goal=goal,
            )
        )

    for mode, seed in [("hard", 12_001), ("penalty", 12_002)]:
        no_fly = _layout(15, 0.1, seed)
        no_fly_pairs = _pairs(
            size=15,
            blocked=no_fly,
            dynamic=[],
            count=10,
            seed=seed + 100,
            minimum_cost=5.0,
        )
        for index, (start, goal) in enumerate(no_fly_pairs, start=1):
            scenarios.append(
                _scenario(
                    scenario_id=f"REAL-NOFLY-{mode.upper()}-{index:03d}",
                    split=f"no_fly_{mode}",
                    size=15,
                    density=0.0,
                    layout_seed=seed,
                    blocked=empty,
                    dynamic=[],
                    no_fly_cells=no_fly,
                    no_fly_mode=mode,
                    start=start,
                    goal=goal,
                )
            )

    sensor_pairs = _pairs(
        size=15,
        blocked=empty,
        dynamic=[],
        count=20,
        seed=13_000,
        minimum_cost=5.0,
    )
    for index, (start, goal) in enumerate(sensor_pairs, start=1):
        probability = 0.05 if index <= 10 else 0.10
        scenarios.append(
            _scenario(
                scenario_id=f"REAL-SENSOR-{index:03d}",
                split=f"sensor_noise_p{int(probability * 100):02d}",
                size=15,
                density=0.0,
                layout_seed=None,
                blocked=empty,
                dynamic=[],
                sensor_noise_probability=probability,
                start=start,
                goal=goal,
            )
        )

    return {
        "schema_version": 2,
        "suite_id": "uav-routing-benchmark-v2",
        "dynamics_timing": "post_move_observed",
        "movement": {
            "connectivity": 8,
            "orthogonal_cost": 1.0,
            "diagonal_cost": "sqrt(2)",
            "corner_cutting": True,
        },
        "split_definitions": {
            "seen_layout_unseen_pairs": "Training layout seed 42, held-out endpoints.",
            "unseen_layout_same_density": "New layouts at density 0.20.",
            "denser_unseen_layout": "New layouts at density 0.30.",
            "new_dynamic_obstacle_locations": "Empty static grid with unseen toggle cells.",
            "changed_toggle_periods": "Known toggle cells with unseen periods.",
            "scale_*": "New density-0.20 layout at the named grid size.",
            "obstacle_density_*": "Controlled static obstacle-density sweep.",
            "stochastic_obstacles_*": "Seeded independent obstacle toggles.",
            "moving_obstacles_*": "Cyclic moving obstacles at controlled periods.",
            "wind_energy_penalty_*": "Spatially varying additive energy costs.",
            "no_fly_*": "Hard or penalized regulatory exclusion cells.",
            "sensor_noise_*": "Independent observation corruption for RL.",
        },
        "scenarios": scenarios,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "manifests" / "benchmark_v2.json",
    )
    args = parser.parse_args()
    suite = build_suite()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Keep the persisted manifest byte-identical across Linux and Windows.
    # Route artifacts record the manifest SHA-256, so platform-native newline
    # translation would otherwise invalidate an equivalent benchmark locally.
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(suite, indent=2) + "\n")
    split_counts: dict[str, int] = {}
    for scenario in suite["scenarios"]:
        split_counts[scenario["split"]] = split_counts.get(scenario["split"], 0) + 1
    print(f"Wrote {len(suite['scenarios'])} scenarios to {args.out}")
    for split, count in split_counts.items():
        print(f"  {split}: {count}")


if __name__ == "__main__":
    main()
