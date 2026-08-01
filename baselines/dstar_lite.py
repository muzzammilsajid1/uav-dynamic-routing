"""D* Lite incremental replanning for the shared dynamic grid."""
from __future__ import annotations

import heapq
import itertools
import math
import time
from dataclasses import dataclass

from baselines.astar import octile_distance
from baselines.dijkstra import dijkstra
from baselines.replanning import ReplanningResult
from envs.grid_environment import GridEnvironment, Node

INF = float("inf")


@dataclass(frozen=True)
class DStarPlanState:
    found: bool
    expanded: int


class DStarLite:
    """Koenig and Likhachev's incremental search for an undirected grid."""

    def __init__(self, env: GridEnvironment, start: Node, goal: Node) -> None:
        self.env = env
        self.start = start
        self.goal = goal
        self.last_start = start
        self.km = 0.0
        self.g: dict[Node, float] = {}
        self.rhs: dict[Node, float] = {goal: 0.0}
        self._queue: list[tuple[float, float, int, Node]] = []
        self._queued_keys: dict[Node, tuple[float, float]] = {}
        self._counter = itertools.count()
        self.total_expanded = 0
        self._push(goal)

    def _g(self, node: Node) -> float:
        return self.g.get(node, INF)

    def _rhs(self, node: Node) -> float:
        return self.rhs.get(node, INF)

    def _consistent(self, node: Node) -> bool:
        return math.isclose(
            self._g(node), self._rhs(node), rel_tol=0.0, abs_tol=1e-12
        )

    def _key(self, node: Node) -> tuple[float, float]:
        best = min(self._g(node), self._rhs(node))
        return best + octile_distance(self.start, node) + self.km, best

    def _push(self, node: Node) -> None:
        key = self._key(node)
        self._queued_keys[node] = key
        heapq.heappush(self._queue, (*key, next(self._counter), node))

    def _discard_stale_top(self) -> None:
        while self._queue:
            key = self._queue[0][:2]
            node = self._queue[0][3]
            if self._queued_keys.get(node) == key:
                return
            heapq.heappop(self._queue)

    def _top_key(self) -> tuple[float, float]:
        self._discard_stale_top()
        return self._queue[0][:2] if self._queue else (INF, INF)

    def _pop(self) -> tuple[tuple[float, float], Node]:
        self._discard_stale_top()
        key1, key2, _, node = heapq.heappop(self._queue)
        self._queued_keys.pop(node, None)
        return (key1, key2), node

    def _predecessors(self, node: Node) -> list[Node]:
        if not self.env.in_bounds(node) or self.env.is_blocked(node):
            return []
        predecessors: list[Node] = []
        row, col = node
        for d_row, d_col, _ in self.env.movement_offsets():
            candidate = (row - d_row, col - d_col)
            if (
                self.env.in_bounds(candidate)
                and node in dict(self.env.get_neighbors(candidate))
            ):
                predecessors.append(candidate)
        return predecessors

    def update_vertex(self, node: Node) -> None:
        if node != self.goal:
            successors = self.env.get_neighbors(node)
            self.rhs[node] = min(
                (cost + self._g(successor) for successor, cost in successors),
                default=INF,
            )
        self._queued_keys.pop(node, None)
        if not self._consistent(node):
            self._push(node)

    def compute_shortest_path(self) -> DStarPlanState:
        expanded_before = self.total_expanded
        while (
            self._top_key() < self._key(self.start)
            or not self._consistent(self.start)
        ):
            if self._top_key() == (INF, INF):
                break
            old_key, node = self._pop()
            new_key = self._key(node)
            self.total_expanded += 1
            if old_key < new_key:
                self._push(node)
            elif self._g(node) > self._rhs(node):
                self.g[node] = self._rhs(node)
                for predecessor in self._predecessors(node):
                    self.update_vertex(predecessor)
            else:
                self.g[node] = INF
                self.update_vertex(node)
                for predecessor in self._predecessors(node):
                    self.update_vertex(predecessor)
        return DStarPlanState(
            found=not math.isinf(self._g(self.start)),
            expanded=self.total_expanded - expanded_before,
        )

    def notify_changes(self, current: Node, changed: set[Node]) -> DStarPlanState:
        self.km += octile_distance(self.last_start, current)
        self.start = current
        self.last_start = current
        affected: set[Node] = set(changed)
        for row, col in changed:
            for d_row, d_col, _ in self.env.movement_offsets():
                candidate = (row + d_row, col + d_col)
                if self.env.in_bounds(candidate):
                    affected.add(candidate)
        for node in affected:
            self.update_vertex(node)
        return self.compute_shortest_path()

    def next_step(self, current: Node) -> tuple[Node, float] | None:
        candidates = [
            (cost + self._g(neighbor), neighbor, cost)
            for neighbor, cost in self.env.get_neighbors(current)
            if not math.isinf(self._g(neighbor))
        ]
        if not candidates:
            return None
        _, neighbor, cost = min(candidates)
        return neighbor, cost


def run_dstar_lite_replanning(
    env: GridEnvironment, max_steps: int = 1000
) -> ReplanningResult:
    """Run D* Lite under the shared move-then-observe timing contract."""
    current = env.start
    realized_path = [current]
    total_cost = 0.0
    total_planning_time = 0.0
    replans = 0
    events: list[dict] = []
    planner = DStarLite(env, current, env.goal)

    def _result(
        *, steps: int, success: bool, timed_out: bool
    ) -> ReplanningResult:
        for event in events:
            event["post_change_success"] = success
        return ReplanningResult(
            realized_path,
            total_cost,
            replans,
            total_planning_time,
            steps,
            success,
            timed_out,
            events,
            planner.total_expanded,
        )

    started = time.perf_counter()
    state = planner.compute_shortest_path()
    elapsed = time.perf_counter() - started
    total_planning_time += elapsed
    replans += 1
    events.append(
        {
            "step": 0,
            "reason": "initial_plan",
            "duration": elapsed,
            "found": state.found,
            "node_expansions": state.expanded,
            "plan_cost": planner._g(current),
            "pre_change_optimal_cost": None,
            "optimal_cost_delta": None,
            "recovery_steps": None,
        }
    )
    if not state.found:
        return _result(steps=0, success=False, timed_out=False)

    open_events: list[dict] = []
    for step in range(1, max_steps + 1):
        next_move = planner.next_step(current)
        if next_move is None:
            return _result(steps=step, success=False, timed_out=False)
        current, move_cost = next_move
        planner.start = current
        total_cost += move_cost
        realized_path.append(current)
        if open_events:
            current_cost = dijkstra(current, env.goal, env.get_neighbors).cost
            for event in list(open_events):
                threshold = event["pre_change_optimal_cost"]
                if threshold is not None and current_cost <= threshold:
                    event["recovery_steps"] = step - int(event["step"])
                    open_events.remove(event)
        if current == env.goal:
            return _result(steps=step, success=True, timed_out=False)

        pre_change_cost = dijkstra(current, env.goal, env.get_neighbors).cost
        changed = env.step_dynamics()
        if changed:
            started = time.perf_counter()
            state = planner.notify_changes(current, changed)
            elapsed = time.perf_counter() - started
            total_planning_time += elapsed
            replans += 1
            post_change_cost = planner._g(current)
            event = {
                "step": step,
                "reason": f"dynamic_change:{sorted(changed)}",
                "duration": elapsed,
                "found": state.found,
                "node_expansions": state.expanded,
                "plan_cost": post_change_cost,
                "pre_change_optimal_cost": pre_change_cost,
                "optimal_cost_delta": (
                    post_change_cost - pre_change_cost
                    if state.found and math.isfinite(pre_change_cost)
                    else None
                ),
                "recovery_steps": None,
            }
            events.append(event)
            if state.found and post_change_cost <= pre_change_cost:
                event["recovery_steps"] = 0
            else:
                open_events.append(event)
            if not state.found:
                return _result(steps=step, success=False, timed_out=False)

    return _result(steps=max_steps, success=False, timed_out=True)
