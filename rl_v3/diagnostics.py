"""Phase A diagnostic evaluation and deterministic failure labels."""
from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DQN

from baselines.astar import astar
from rl_agent.double_dqn import DoubleDQN
from rl_agent.uav_env import UAVRoutingEnv
from rl_v3.action_masking import highest_q_legal_action, legal_action_mask
from rl_v3.env_builders import empty_map_scenario, env_from_scenario

PROJECT_ROOT = Path(__file__).resolve().parents[1]


FAILURE_PRECEDENCE = [
    "immediate_invalid_action_crash",
    "collision",
    "step_limit_timeout",
    "two_cell_oscillation",
    "longer_repeated_loop",
    "dead_end_failure",
    "excessive_detour",
    "post_change_failure",
    "long_horizon_goal_following_failure",
    "obstacle_navigation_failure",
    "scale_normalization_failure",
    "unknown_failure",
]


def astar_cost(env: UAVRoutingEnv, start: tuple[int, int]) -> tuple[float, list[tuple[int, int]]]:
    result = astar(
        start,
        tuple(int(value) for value in env.goal_pos),
        lambda node: [
            (tuple(int(value) for value in neighbor), cost)
            for neighbor, cost in env.get_neighbors(np.asarray(node))
        ],
    )
    return result.cost, result.path


def q_values(model: DQN, observation: dict) -> np.ndarray:
    obs_tensor, _ = model.policy.obs_to_tensor(observation)
    with torch.no_grad():
        values = model.q_net(obs_tensor).detach().cpu().numpy()[0]
    return values.astype(float)


def evaluate_episode(
    model: DQN,
    scenario: dict,
    *,
    mode: str,
    variant_config: dict,
    max_steps: int | None = None,
    diagnostic_budget_to_astar_cost: float | None = None,
    minimum_diagnostic_budget: int = 20,
    save_trajectory_path: Path | None = None,
) -> dict:
    if mode not in {"original", "posthoc_masked"}:
        raise ValueError("mode must be original or posthoc_masked")
    env, observation = env_from_scenario(
        scenario,
        potential_shaping_enabled=bool(variant_config["potential_shaping"]),
        observation_mode=str(variant_config["observation_mode"]),
        max_steps=max_steps,
    )
    initial_cost, initial_astar_path = astar_cost(env, tuple(int(value) for value in env.uav_pos))
    if diagnostic_budget_to_astar_cost is not None and math.isfinite(initial_cost):
        env.max_steps = min(
            int(env.max_steps),
            max(
                int(minimum_diagnostic_budget),
                int(math.ceil(float(diagnostic_budget_to_astar_cost) * initial_cost)),
            ),
        )
    budget_ratio = float(env.max_steps) / max(float(initial_cost), 1e-9)
    trajectory = [tuple(int(value) for value in env.uav_pos)]
    selected_actions: list[int] = []
    masks: list[list[int]] = []
    q_history: list[list[float]] = []
    rewards: list[float] = []
    dynamic_events: list[dict] = []
    event_astar_paths: list[dict] = []
    path_cost = 0.0
    decision_time = 0.0
    invalid_action_crash_step: int | None = None
    done = False
    info: dict = {}

    while not done:
        mask = legal_action_mask(env)
        values = q_values(model, observation)
        started = time.perf_counter()
        if mode == "posthoc_masked":
            action = highest_q_legal_action(values, mask)
        else:
            action, _ = model.predict(observation, deterministic=True)
            action = int(action)
        decision_time += time.perf_counter() - started
        selected_actions.append(int(action))
        masks.append(mask.astype(int).tolist())
        q_history.append(values.tolist())
        if not mask[int(action)] and invalid_action_crash_step is None:
            invalid_action_crash_step = len(selected_actions)
        delta = env.ACTION_DELTAS[int(action)]
        move_cost = 1.0 if delta[0] == 0 or delta[1] == 0 else math.sqrt(2)
        observation, reward, terminated, truncated, info = env.step(int(action))
        rewards.append(float(reward))
        if not info.get("crashed", False):
            path_cost += move_cost + float(info.get("energy_penalty", 0.0))
        trajectory.append(tuple(int(value) for value in env.uav_pos))
        if info.get("dynamic_changes"):
            current = tuple(int(value) for value in env.uav_pos)
            event_cost, event_path = astar_cost(env, current)
            event_record = {
                "step": len(selected_actions),
                "dynamic_changes": info["dynamic_changes"],
                "event_astar_cost": event_cost,
            }
            dynamic_events.append(event_record)
            event_astar_paths.append({"step": len(selected_actions), "path": event_path})
        done = bool(terminated or truncated)

    success = bool(info.get("is_success", False))
    repeated = Counter(trajectory)
    label = classify_failure(
        success=success,
        crashed=bool(info.get("crashed", False)),
        timed_out=not success and not bool(info.get("crashed", False)),
        trajectory=trajectory,
        invalid_action_crash_step=invalid_action_crash_step,
        path_cost=path_cost,
        initial_cost=initial_cost,
        scenario=scenario,
        dynamic_events=dynamic_events,
    )
    route = {
        "scenario_id": scenario["scenario_id"],
        "mode": mode,
        "grid_size": int(scenario["grid_size"]),
        "route_bucket": scenario.get("route_bucket", ""),
        "success": success,
        "failure_label": "success" if success else label,
        "immediate_invalid_action_crash": invalid_action_crash_step == 1,
        "invalid_action_crash": invalid_action_crash_step is not None,
        "collision": bool(info.get("crashed", False)),
        "timeout": not success and not bool(info.get("crashed", False)),
        "two_cell_oscillation": has_two_cell_oscillation(trajectory),
        "longer_loop": has_longer_loop(trajectory),
        "excessive_detour": bool(path_cost > 2.0 * initial_cost) if math.isfinite(initial_cost) else False,
        "decisions": len(selected_actions),
        "path_cost": path_cost if success else "",
        "initial_astar_cost": initial_cost,
        "path_cost_gap": path_cost - initial_cost if success else "",
        "mean_decision_latency_ms": 1000.0 * decision_time / max(1, len(selected_actions)),
        "episode_budget": env.max_steps,
        "budget_to_astar_cost_ratio": budget_ratio,
        "repeated_cell_count": sum(count - 1 for count in repeated.values() if count > 1),
        "dynamic_event_count": len(dynamic_events),
    }
    if save_trajectory_path is not None:
        save_trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        save_trajectory_path.write_text(
            json.dumps(
                {
                    "route": route,
                    "trajectory": [list(cell) for cell in trajectory],
                    "selected_actions": selected_actions,
                    "q_values": q_history,
                    "legal_action_masks": masks,
                    "rewards": rewards,
                    "dynamic_events": dynamic_events,
                    "initial_astar_path": [list(cell) for cell in initial_astar_path],
                    "event_astar_paths": [
                        {"step": item["step"], "path": [list(cell) for cell in item["path"]]}
                        for item in event_astar_paths
                    ],
                    "failure_precedence": FAILURE_PRECEDENCE,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    env.close()
    return route


def classify_failure(
    *,
    success: bool,
    crashed: bool,
    timed_out: bool,
    trajectory: list[tuple[int, int]],
    invalid_action_crash_step: int | None,
    path_cost: float,
    initial_cost: float,
    scenario: dict,
    dynamic_events: list[dict],
) -> str:
    if success:
        return "success"
    if invalid_action_crash_step == 1:
        return "immediate_invalid_action_crash"
    if crashed:
        return "collision"
    if timed_out:
        if has_two_cell_oscillation(trajectory):
            return "two_cell_oscillation"
        if has_longer_loop(trajectory):
            return "longer_repeated_loop"
        if dynamic_events:
            return "post_change_failure"
        if scenario.get("obstacle_density", 0.0) == 0.0:
            if int(scenario["grid_size"]) > 15:
                return "scale_normalization_failure"
            return "long_horizon_goal_following_failure"
        return "obstacle_navigation_failure"
    if math.isfinite(initial_cost) and path_cost > 2.0 * initial_cost:
        return "excessive_detour"
    return "unknown_failure"


def has_two_cell_oscillation(trajectory: list[tuple[int, int]], window: int = 6) -> bool:
    if len(trajectory) < window:
        return False
    tail = trajectory[-window:]
    return len(set(tail)) == 2 and all(tail[i] == tail[i - 2] for i in range(2, len(tail)))


def has_longer_loop(trajectory: list[tuple[int, int]], window: int = 12) -> bool:
    if len(trajectory) < window:
        return False
    tail = trajectory[-window:]
    return len(set(tail)) <= window // 2


def load_frozen_ddqn(variant: str, seed: int, checkpoint_stage: str, variant_config: dict) -> DQN:
    model_class = DoubleDQN if variant_config["double_dqn"] else DQN
    model_env = UAVRoutingEnv(
        grid_size=15,
        obstacle_density=0.0,
        fixed_grid=True,
        seed=42,
        potential_shaping_enabled=bool(variant_config["potential_shaping"]),
        observation_mode=str(variant_config["observation_mode"]),
    )
    checkpoint = (
        PROJECT_ROOT
        / "models"
        / "research"
        / variant
        / f"seed_{int(seed):03d}"
        / checkpoint_stage
    )
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    return model_class.load(checkpoint, env=model_env, device="auto")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def empty_map_scenarios(config: dict) -> list[dict]:
    diag = config["empty_map_diagnostics"]
    scenarios = []
    for grid_size in diag["grid_sizes"]:
        for ratio in diag["distance_ratios"]:
            for orientation in diag["orientations"]:
                scenarios.append(
                    empty_map_scenario(
                        scenario_id=f"EMPTY-G{grid_size:03d}-R{int(ratio * 100):03d}-{orientation}",
                        grid_size=int(grid_size),
                        distance_ratio=float(ratio),
                        orientation=str(orientation),
                    )
                )
    return scenarios
