from __future__ import annotations

import time as _time
from dataclasses import dataclass, field

from baselines.dijkstra import DijkstraResult, dijkstra
from envs.grid_environment import GridEnvironment, Node


@dataclass
class ReplanningResult:
    """Outcome of one full naive-replanning run from start to goal."""

    realized_path: list[Node]
    total_cost: float
    replans: int
    total_planning_time: float  # seconds, wall-clock, cumulative across all dijkstra() calls
    steps_taken: int
    success: bool
    timed_out: bool
    replan_events: list[dict] = field(default_factory=list)


def run_naive_replanning(env: GridEnvironment, max_steps: int = 1000) -> ReplanningResult:
    """Naive replanning baseline (plan Section 4.1): whenever a dynamic
    environment changes, recompute a full Dijkstra path from the UAV's
    CURRENT position to the goal. No anticipation, no incremental replanning (e.g. D*
    Lite) — intentionally the "dumb but correct" comparison point against
    the RL agent, which reacts via a forward pass instead of a full
    graph search.

    No advance obstacle visibility (Week 3 agreement with Muzzammil):
    this only reacts to env.step_dynamics() results *after* they happen,
    never before.
    """
    current = env.start
    realized_path: list[Node] = [current]
    total_cost = 0.0
    replans = 0
    total_planning_time = 0.0
    replan_events: list[dict] = []

    def _replan(step: int, reason: str) -> DijkstraResult:
        nonlocal replans, total_planning_time
        t0 = _time.perf_counter()
        result = dijkstra(current, env.goal, env.get_neighbors)
        elapsed = _time.perf_counter() - t0
        total_planning_time += elapsed
        replans += 1
        replan_events.append(
            {"step": step, "reason": reason, "duration": elapsed, "found": result.found}
        )
        return result

    plan = _replan(step=0, reason="initial_plan")
    if not plan.found:
        return ReplanningResult(
            realized_path, total_cost, replans, total_planning_time, 0, False, False, replan_events
        )

    plan_path = plan.path
    plan_index = 0

    for step in range(1, max_steps + 1):
        changed = env.step_dynamics()

        if changed:
            plan = _replan(step=step, reason=f"dynamic_change:{sorted(changed)}")
            if not plan.found:
                return ReplanningResult(
                    realized_path, total_cost, replans, total_planning_time,
                    step, False, False, replan_events,
                )
            plan_path = plan.path
            plan_index = 0

        neighbor_costs = dict(env.get_neighbors(current))
        next_node = plan_path[plan_index + 1]

        # Defensive fallback: should be rare (only if a change we didn't
        # think was "ahead" still broke the immediate next step), but
        # replan rather than crash if the edge no longer exists.
        if next_node not in neighbor_costs:
            plan = _replan(step=step, reason="stale_plan_fallback")
            if not plan.found:
                return ReplanningResult(
                    realized_path, total_cost, replans, total_planning_time,
                    step, False, False, replan_events,
                )
            plan_path = plan.path
            plan_index = 0
            neighbor_costs = dict(env.get_neighbors(current))
            next_node = plan_path[plan_index + 1]

        total_cost += neighbor_costs[next_node]
        current = next_node
        realized_path.append(current)
        plan_index += 1

        if current == env.goal:
            return ReplanningResult(
                realized_path, total_cost, replans, total_planning_time,
                step, True, False, replan_events,
            )

    return ReplanningResult(
        realized_path, total_cost, replans, total_planning_time,
        max_steps, False, True, replan_events,
    )
