"""Frozen Phase B development validation suite and curriculum scenarios."""
from __future__ import annotations

import math
import random
from typing import Iterable

import numpy as np

from baselines.astar import astar
from envs.grid_environment import DynamicObstacle, GridEnvironment
from rl_v3.scenario_generation import episode_seed, sha256_json

ROUTE_RATIOS = {
    "short": (0.20, 0.40),
    "medium": (0.40, 0.70),
    "long": (0.70, 1.35),
}
STRUCTURED_TYPES = ("long_wall", "u_shape", "corridor", "dead_end")


def generate_validation_v2(config: dict) -> dict:
    cfg = config["validation_v2"]
    scenarios: list[dict] = []
    seed = int(cfg["seed_start"])
    per = int(cfg["scenarios_per_scale_family_bucket"])
    for size in cfg["grid_sizes"]:
        for family in cfg["scenario_families"]:
            for bucket in cfg["route_buckets"]:
                for replicate in range(per):
                    subtype = (
                        STRUCTURED_TYPES[
                            (int(size) + cfg["route_buckets"].index(bucket) * per + replicate)
                            % len(STRUCTURED_TYPES)
                        ]
                        if family == "structured"
                        else family
                    )
                    scenario_id = (
                        f"VAL2-G{int(size):03d}-{family.upper()}-"
                        f"{bucket.upper()}-{replicate + 1:02d}"
                    )
                    scenarios.append(
                        generate_family_scenario(
                            scenario_id=scenario_id,
                            grid_size=int(size),
                            family=str(family),
                            route_bucket=str(bucket),
                            seed=seed,
                            structured_type=subtype,
                            split="validation_v2",
                        )
                    )
                    seed += 1
    body = {
        "schema_version": 1,
        "suite_id": "rl-v3-development-validation-v2",
        "movement_contract": config["movement_contract"],
        "final_test": False,
        "scenarios": scenarios,
    }
    body["manifest_sha256"] = sha256_json(body)
    return body


def generate_curriculum_scenario(config: dict, episode_index: int, timestep: int) -> dict:
    training = config["training"]
    stage_index, stage = curriculum_stage(training["curriculum"], timestep)
    seed = episode_seed(int(training["master_seed"]), int(episode_index))
    rng = random.Random(seed)
    family = str(rng.choice(stage["families"]))
    size = int(rng.choice(stage["sizes"]))
    bucket = ("short", "medium", "long")[episode_index % 3]
    subtype = STRUCTURED_TYPES[episode_index % len(STRUCTURED_TYPES)]
    return generate_family_scenario(
        scenario_id=f"TRAIN-S{stage_index + 1}-{episode_index:012d}",
        grid_size=size,
        family=family,
        route_bucket=bucket,
        seed=seed,
        structured_type=subtype,
        split="training",
    )


def curriculum_stage(stages: list[dict], timestep: int) -> tuple[int, dict]:
    for index, stage in enumerate(stages):
        if timestep < int(stage["until"]):
            return index, stage
    return len(stages) - 1, stages[-1]


def generate_family_scenario(
    *,
    scenario_id: str,
    grid_size: int,
    family: str,
    route_bucket: str,
    seed: int,
    structured_type: str,
    split: str,
) -> dict:
    rng = random.Random(seed)
    low, high = ROUTE_RATIOS[route_bucket]
    for attempt in range(2500):
        blocked: set[tuple[int, int]] = set()
        dynamic: list[DynamicObstacle] = []
        if family in {"random_static", "dynamic"}:
            density = rng.uniform(0.03, 0.16 if family == "random_static" else 0.10)
            blocked = _random_layout(grid_size, density, rng.randrange(2**31 - 1))
        elif family == "structured":
            blocked = _structured_layout(grid_size, structured_type, rng)
            density = len(blocked) / float(grid_size * grid_size)
        else:
            density = 0.0 if attempt % 2 == 0 else 0.01
            blocked = _random_layout(grid_size, density, rng.randrange(2**31 - 1))
        free = [
            (r, c)
            for r in range(grid_size)
            for c in range(grid_size)
            if (r, c) not in blocked
        ]
        if len(free) < 2:
            continue
        start, goal = rng.sample(free, 2)
        if family == "dynamic":
            candidates = [cell for cell in free if cell not in {start, goal}]
            for cell in rng.sample(candidates, min(2, len(candidates))):
                dynamic.append(
                    DynamicObstacle(cell, rng.choice((5, 7, 9)), rng.choice(("blocked", "passable")))
                )
        env = GridEnvironment(
            size=grid_size,
            obstacle_density=0.0,
            start=start,
            goal=goal,
            blocked=blocked,
            dynamic_obstacles=dynamic,
        )
        result = astar(start, goal, env.get_neighbors)
        ratio = float(result.cost) / float(grid_size)
        if result.found and low <= ratio < high:
            return _record(
                scenario_id, split, grid_size, family, structured_type,
                route_bucket, seed, attempt + 1, density, blocked, dynamic,
                start, goal, float(result.cost),
            )
    raise RuntimeError(f"could not generate {scenario_id}")


def manifest_balance(manifest: dict) -> dict:
    rows = manifest["scenarios"]
    def counts(keys: Iterable[str]) -> dict[str, int]:
        output: dict[str, int] = {}
        for row in rows:
            label = "/".join(str(row[key]) for key in keys)
            output[label] = output.get(label, 0) + 1
        return dict(sorted(output.items()))
    return {
        "total": len(rows),
        "by_scale": counts(("grid_size",)),
        "by_route_bucket": counts(("route_bucket",)),
        "by_family": counts(("scenario_family",)),
        "scale_family_bucket": counts(("grid_size", "scenario_family", "route_bucket")),
    }


def _random_layout(size: int, density: float, seed: int) -> set[tuple[int, int]]:
    count = int(size * size * density)
    if count == 0:
        return set()
    rng = np.random.default_rng(seed)
    indices = rng.choice(size * size, size=count, replace=False)
    rows, cols = np.unravel_index(indices, (size, size))
    return set(zip(rows.astype(int).tolist(), cols.astype(int).tolist()))


def _structured_layout(size: int, subtype: str, rng: random.Random) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()
    margin = max(2, size // 8)
    mid = size // 2
    if subtype == "long_wall":
        gap = rng.randrange(margin, size - margin)
        blocked = {(r, mid) for r in range(margin, size - margin) if abs(r - gap) > 1}
    elif subtype == "u_shape":
        lo, hi = margin, size - margin - 1
        blocked |= {(hi, c) for c in range(lo, hi + 1)}
        blocked |= {(r, lo) for r in range(mid, hi + 1)}
        blocked |= {(r, hi) for r in range(mid, hi + 1)}
    elif subtype == "corridor":
        offset = max(2, size // 6)
        blocked |= {(r, mid - 1) for r in range(offset, size - offset)}
        blocked |= {(r, mid + 2) for r in range(offset, size - offset)}
    elif subtype == "dead_end":
        length = max(4, size // 3)
        blocked |= {(mid - 2, c) for c in range(margin, min(size - margin, margin + length))}
        blocked |= {(mid + 2, c) for c in range(margin, min(size - margin, margin + length))}
        blocked.add((mid - 1, margin + length - 1))
        blocked.add((mid, margin + length - 1))
        blocked.add((mid + 1, margin + length - 1))
    else:
        raise ValueError(subtype)
    return blocked


def _record(
    scenario_id: str, split: str, size: int, family: str, subtype: str,
    bucket: str, seed: int, attempts: int, density: float,
    blocked: set[tuple[int, int]], dynamic: list[DynamicObstacle],
    start: tuple[int, int], goal: tuple[int, int], cost: float,
) -> dict:
    return {
        "scenario_id": scenario_id,
        "episode_seed": seed,
        "split": split,
        "grid_size": size,
        "scenario_family": family,
        "structured_type": subtype if family == "structured" else None,
        "obstacle_density": round(float(density), 6),
        "route_bucket": bucket,
        "initial_astar_cost": cost,
        "generation_attempts": attempts,
        "blocked": [list(cell) for cell in sorted(blocked)],
        "dynamic_obstacles": [
            {"cell": list(item.cell), "period": item.period, "initial_state": item.initial_state}
            for item in dynamic
        ],
        "dynamics_timing": "post_move_observed",
        "start": list(start),
        "goal": list(goal),
        "stochastic_obstacles": [],
        "moving_obstacles": [],
        "traversal_penalties": [],
        "dynamics_seed": seed + 10000,
        "no_fly_cells": [],
        "no_fly_mode": None,
        "no_fly_penalty": 5.0,
        "sensor_noise_probability": 0.0,
    }
