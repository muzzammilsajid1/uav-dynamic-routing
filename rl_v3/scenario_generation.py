"""Deterministic RL V3 scenario generation and validation assets."""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from baselines.astar import astar
from envs.grid_environment import DynamicObstacle, GridEnvironment

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RouteBucket:
    name: str
    min_ratio: float
    max_ratio: float


def stable_json(data: object) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def sha256_json(data: object) -> str:
    return hashlib.sha256(stable_json(data).encode("utf-8")).hexdigest()


def write_stable_json(path: Path, data: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = stable_json(data)
    path.write_text(text, encoding="utf-8", newline="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_distribution_config(config: dict) -> dict:
    return {
        "schema_version": config["schema_version"],
        "suite_id": config["suite_id"],
        "movement_contract": config["movement_contract"],
        "training_generator": config["training_generator"],
        "validation_manifest": config["validation_manifest"],
    }


def build_training_generator_asset(config: dict) -> dict:
    distribution = source_distribution_config(config)
    return {
        "schema_version": 1,
        "asset_type": "rl_v3_training_generator",
        "description": (
            "This is a deterministic distribution definition, not a finite "
            "training manifest. Training episodes are generated from the "
            "master seed and episode index."
        ),
        "distribution": distribution,
        "distribution_sha256": sha256_json(distribution),
    }


def generate_validation_manifest(config: dict) -> dict:
    validation = config["validation_manifest"]
    generator_cfg = config["training_generator"]
    buckets = _route_buckets(generator_cfg)
    scenarios: list[dict] = []
    for grid_size in validation["grid_sizes"]:
        for bucket_name in validation["route_length_buckets"]:
            bucket = buckets[bucket_name]
            for index in range(validation["scenarios_per_grid_size_per_bucket"]):
                episode_id = (
                    f"VAL-G{grid_size:03d}-{bucket_name.upper()}-{index + 1:02d}"
                )
                seed = int(validation["seed"]) + grid_size * 10 + index * 17
                seed += {"short": 1, "medium": 2, "long": 3}[bucket_name] * 1000
                scenarios.append(
                    generate_scenario(
                        generator_cfg,
                        episode_id=episode_id,
                        grid_size=int(grid_size),
                        bucket=bucket,
                        episode_seed=seed,
                    )
                )
    manifest = {
        "schema_version": 1,
        "suite_id": "rl-v3-validation-v1",
        "generator_version": generator_cfg["generator_version"],
        "seed": validation["seed"],
        "movement_contract": config["movement_contract"],
        "final_test": False,
        "scenarios": scenarios,
    }
    manifest["manifest_sha256"] = sha256_json(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    return manifest


def generate_scenario(
    generator_cfg: dict,
    *,
    episode_id: str,
    grid_size: int,
    bucket: RouteBucket,
    episode_seed: int,
) -> dict:
    rng = random.Random(episode_seed)
    attempts = int(generator_cfg["max_generation_attempts"])
    density_range = generator_cfg["obstacle_density"]
    for attempt in range(attempts):
        density = rng.uniform(float(density_range["min"]), float(density_range["max"]))
        blocked = _layout(grid_size, density, rng.randrange(1_000_000_000))
        dynamic = _dynamic_obstacles(
            grid_size,
            blocked,
            rng,
            min_count=int(generator_cfg["dynamic_obstacle_count"]["min"]),
            max_count=int(generator_cfg["dynamic_obstacle_count"]["max"]),
            periods=[int(value) for value in generator_cfg["dynamic_periods"]],
        )
        candidates = [
            (row, col)
            for row in range(grid_size)
            for col in range(grid_size)
            if (row, col) not in blocked and (row, col) not in {item.cell for item in dynamic}
        ]
        if len(candidates) < 2:
            continue
        start, goal = rng.sample(candidates, 2)
        env = GridEnvironment(
            size=grid_size,
            obstacle_density=0.0,
            start=start,
            goal=goal,
            blocked=set(blocked),
            dynamic_obstacles=dynamic,
        )
        result = astar(start, goal, env.get_neighbors)
        ratio = result.cost / float(grid_size)
        if result.found and bucket.min_ratio <= ratio < bucket.max_ratio:
            return _scenario(
                episode_id=episode_id,
                split="validation",
                grid_size=grid_size,
                obstacle_density=density,
                blocked=blocked,
                dynamic=dynamic,
                start=start,
                goal=goal,
                episode_seed=episode_seed,
                optimal_cost=result.cost,
                route_bucket=bucket.name,
                attempts=attempt + 1,
            )
    raise RuntimeError(f"could not generate {episode_id} after {attempts} attempts")


def episode_seed(master_seed: int, episode_index: int) -> int:
    digest = hashlib.sha256(f"{master_seed}:{episode_index}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def generate_training_episode(config: dict, episode_index: int) -> dict:
    generator_cfg = config["training_generator"]
    buckets = list(_route_buckets(generator_cfg).values())
    seed = episode_seed(int(generator_cfg["master_seed"]), episode_index)
    rng = random.Random(seed)
    grid_size = int(rng.choice(generator_cfg["grid_sizes"]))
    bucket = buckets[episode_index % len(buckets)]
    return generate_scenario(
        generator_cfg,
        episode_id=f"TRAIN-{episode_index:012d}",
        grid_size=grid_size,
        bucket=bucket,
        episode_seed=seed,
    )


def verify_manifest_separation(config: dict, validation_manifest: dict) -> dict:
    generator_cfg = config["training_generator"]
    reserved = {
        item["name"]: (int(item["start"]), int(item["end"]))
        for item in generator_cfg["forbidden_seed_ranges"]
    }
    validation_seeds = {int(scenario["episode_seed"]) for scenario in validation_manifest["scenarios"]}
    train_master = int(generator_cfg["master_seed"])
    train_preview = {
        episode_seed(train_master, index)
        for index in range(max(1000, len(validation_seeds) * 10))
    }
    overlap = sorted(validation_seeds & train_preview)
    final_range = reserved["final_test_private"]
    final_seed_overlap = [
        seed for seed in validation_seeds if final_range[0] <= seed <= final_range[1]
    ]
    return {
        "validation_scenarios": len(validation_manifest["scenarios"]),
        "validation_sha256": validation_manifest["manifest_sha256"],
        "training_preview_count": len(train_preview),
        "train_validation_seed_overlap": overlap,
        "validation_final_private_seed_overlap": final_seed_overlap,
        "passed": not overlap and not final_seed_overlap,
    }


def _route_buckets(generator_cfg: dict) -> dict[str, RouteBucket]:
    return {
        name: RouteBucket(name, float(values["min_ratio"]), float(values["max_ratio"]))
        for name, values in generator_cfg["route_length_buckets"].items()
    }


def _layout(size: int, density: float, seed: int) -> set[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    count = int(size * size * density)
    if count <= 0:
        return set()
    indices = rng.choice(size * size, size=count, replace=False)
    rows, cols = np.unravel_index(indices, (size, size))
    return {(int(row), int(col)) for row, col in zip(rows.tolist(), cols.tolist())}


def _dynamic_obstacles(
    size: int,
    blocked: set[tuple[int, int]],
    rng: random.Random,
    *,
    min_count: int,
    max_count: int,
    periods: Iterable[int],
) -> list[DynamicObstacle]:
    count = rng.randint(min_count, max_count)
    candidates = [
        (row, col)
        for row in range(1, size - 1)
        for col in range(1, size - 1)
        if (row, col) not in blocked
    ]
    if not candidates or count == 0:
        return []
    cells = rng.sample(candidates, min(count, len(candidates)))
    period_values = list(periods)
    return [
        DynamicObstacle(
            cell=cell,
            period=int(rng.choice(period_values)),
            initial_state=rng.choice(["blocked", "passable"]),
        )
        for cell in cells
    ]


def _scenario(
    *,
    episode_id: str,
    split: str,
    grid_size: int,
    obstacle_density: float,
    blocked: set[tuple[int, int]],
    dynamic: list[DynamicObstacle],
    start: tuple[int, int],
    goal: tuple[int, int],
    episode_seed: int,
    optimal_cost: float,
    route_bucket: str,
    attempts: int,
) -> dict:
    return {
        "scenario_id": episode_id,
        "episode_seed": episode_seed,
        "split": split,
        "grid_size": grid_size,
        "obstacle_density": round(float(obstacle_density), 6),
        "route_bucket": route_bucket,
        "initial_astar_cost": optimal_cost,
        "generation_attempts": attempts,
        "blocked": [list(cell) for cell in sorted(blocked)],
        "dynamic_obstacles": [
            {
                "cell": list(item.cell),
                "period": item.period,
                "initial_state": item.initial_state,
            }
            for item in dynamic
        ],
        "start": list(start),
        "goal": list(goal),
        "stochastic_obstacles": [],
        "moving_obstacles": [],
        "traversal_penalties": [],
        "dynamics_seed": episode_seed + 10_000,
        "no_fly_cells": [],
        "no_fly_mode": None,
        "no_fly_penalty": 5.0,
        "sensor_noise_probability": 0.0,
    }
