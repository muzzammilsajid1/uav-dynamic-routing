from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Callable, Protocol

from baselines.astar import AStarResult, astar
from baselines.dijkstra import DijkstraResult, dijkstra
from envs.grid_environment import GridEnvironment, Node


@dataclass
class ReplanningResult:
    """Outcome of one full matched dynamic-replanning run."""

    realized_path: list[Node]
    total_cost: float
    replans: int
    total_planning_time: float
    steps_taken: int
    success: bool
    timed_out: bool
    replan_events: list[dict] = field(default_factory=list)
    node_expansions: int = 0


class Planner(Protocol):
    def __call__(
        self, start: Node, goal: Node, get_neighbors: Callable
    ) -> DijkstraResult | AStarResult: ...


def run_replanning(
    env: GridEnvironment, planner: Planner, max_steps: int = 1000
) -> ReplanningResult:
    """Move, apply dynamics, then replan before the next decision.

    All classical methods use this runner, so they share the same graph,
    obstacle schedule, movement costs, triggers, and route-level timing
    definition. A move is executed under the state used to select it; dynamics
    then advance and any changed state is visible before the next move is
    selected. The runner never inspects future toggle times.
    """
    current = env.start
    realized_path: list[Node] = [current]
    total_cost = 0.0
    replans = 0
    total_planning_time = 0.0
    replan_events: list[dict] = []
    node_expansions = 0

    def _replan(
        step: int,
        reason: str,
        pre_change_cost: float | None = None,
    ) -> DijkstraResult | AStarResult:
        nonlocal replans, total_planning_time, node_expansions
        started = _time.perf_counter()
        result = planner(current, env.goal, env.get_neighbors)
        elapsed = _time.perf_counter() - started
        total_planning_time += elapsed
        replans += 1
        node_expansions += result.visited_count
        event = {
            "step": step,
            "reason": reason,
            "duration": elapsed,
            "found": result.found,
            "node_expansions": result.visited_count,
            "plan_cost": result.cost,
            "pre_change_optimal_cost": pre_change_cost,
            "optimal_cost_delta": (
                result.cost - pre_change_cost
                if pre_change_cost is not None and result.found
                else None
            ),
            "recovery_steps": None,
        }
        replan_events.append(event)
        return result

    def _result(
        *, steps: int, success: bool, timed_out: bool
    ) -> ReplanningResult:
        for event in replan_events:
            event["post_change_success"] = success
        return ReplanningResult(
            realized_path=realized_path,
            total_cost=total_cost,
            replans=replans,
            total_planning_time=total_planning_time,
            steps_taken=steps,
            success=success,
            timed_out=timed_out,
            replan_events=replan_events,
            node_expansions=node_expansions,
        )

    plan = _replan(step=0, reason="initial_plan")
    if not plan.found:
        return _result(steps=0, success=False, timed_out=False)

    plan_path = plan.path
    plan_index = 0
    open_events: list[dict] = []

    for step in range(1, max_steps + 1):
        neighbor_costs = dict(env.get_neighbors(current))
        next_node = plan_path[plan_index + 1]

        if next_node not in neighbor_costs:
            plan = _replan(step=step, reason="stale_plan_fallback")
            if not plan.found:
                return _result(steps=step, success=False, timed_out=False)
            plan_path = plan.path
            plan_index = 0
            neighbor_costs = dict(env.get_neighbors(current))
            next_node = plan_path[plan_index + 1]

        total_cost += neighbor_costs[next_node]
        current = next_node
        realized_path.append(current)
        plan_index += 1

        if open_events:
            current_cost = dijkstra(current, env.goal, env.get_neighbors).cost
            for event in list(open_events):
                threshold = event["pre_change_optimal_cost"]
                if threshold is not None and current_cost <= threshold:
                    event["recovery_steps"] = step - int(event["step"])
                    open_events.remove(event)

        if current == env.goal:
            return _result(steps=step, success=True, timed_out=False)

        pre_change_cost = 0.0
        for source, target in zip(
            plan_path[plan_index:], plan_path[plan_index + 1 :]
        ):
            edge_cost = dict(env.get_neighbors(source)).get(target)
            if edge_cost is None:
                pre_change_cost = float("inf")
                break
            pre_change_cost += edge_cost
        changed = env.step_dynamics()
        if changed:
            plan = _replan(
                step=step,
                reason=f"dynamic_change:{sorted(changed)}",
                pre_change_cost=pre_change_cost,
            )
            event = replan_events[-1]
            if (
                plan.found
                and pre_change_cost is not None
                and plan.cost <= pre_change_cost
            ):
                event["recovery_steps"] = 0
            else:
                open_events.append(event)
            if not plan.found:
                return _result(steps=step, success=False, timed_out=False)
            plan_path = plan.path
            plan_index = 0

    return _result(steps=max_steps, success=False, timed_out=True)


def run_naive_replanning(
    env: GridEnvironment, max_steps: int = 1000
) -> ReplanningResult:
    """Full Dijkstra replanning reference baseline."""
    return run_replanning(env, dijkstra, max_steps)


def run_astar_replanning(
    env: GridEnvironment, max_steps: int = 1000
) -> ReplanningResult:
    """Heuristic-pruned A* replanning on the same graph and triggers."""
    return run_replanning(env, astar, max_steps)
