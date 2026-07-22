"""Run the naive Dijkstra replanning baseline against the shared,
pre-positioned dynamic obstacle set (Week 3, no advance visibility).
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.replanning import run_naive_replanning
from envs.grid_environment import GridEnvironment, default_dynamic_obstacles
from evaluation.visualize import render_ascii


def main() -> None:
    env = GridEnvironment(
        obstacle_density=0.0,
        dynamic_obstacles=default_dynamic_obstacles(),
    )
    result = run_naive_replanning(env)

    print(f"Success:              {result.success}")
    print(f"Timed out:            {result.timed_out}")
    print(f"Steps taken:          {result.steps_taken}")
    print(f"Realized path cost:   {result.total_cost:.3f}")
    print(f"Replans triggered:    {result.replans}")
    print(f"Total planning time:  {result.total_planning_time * 1000:.3f} ms")
    print()
    print("Replan events:")
    for event in result.replan_events:
        print(f"  step={event['step']:<4} reason={event['reason']}")
    print()
    print(render_ascii(env, result.realized_path))


if __name__ == "__main__":
    main()
