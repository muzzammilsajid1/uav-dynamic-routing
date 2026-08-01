"""Extract one auditable row per classical dynamic-change event."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "results" / "classical_suite_raw.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=(
            PROJECT_ROOT
            / "evaluation"
            / "results"
            / "classical_adaptability_events.csv"
        ),
    )
    args = parser.parse_args()

    event_rows: list[dict[str, object]] = []
    with args.raw.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            events = json.loads(row["replan_events_json"])
            dynamic_events = [
                event
                for event in events
                if str(event["reason"]).startswith("dynamic_change:")
            ]
            for event_index, event in enumerate(dynamic_events, start=1):
                event_rows.append(
                    {
                        "run_id": row["run_id"],
                        "scenario_id": row["scenario_id"],
                        "split": row["split"],
                        "grid_size": row["grid_size"],
                        "planner": row["planner"],
                        "repetition": row["repetition"],
                        "event_index": event_index,
                        "change_step": event["step"],
                        "change_reason": event["reason"],
                        "planning_time_ms": float(event["duration"]) * 1000,
                        "reaction_time_ms": float(event["duration"]) * 1000,
                        "node_expansions": event["node_expansions"],
                        "pre_change_optimal_cost": event.get(
                            "pre_change_optimal_cost"
                        ),
                        "post_change_optimal_cost": event.get("plan_cost"),
                        "optimal_cost_delta": event.get("optimal_cost_delta"),
                        "extra_optimal_cost": (
                            max(0.0, float(event["optimal_cost_delta"]))
                            if event.get("optimal_cost_delta") is not None
                            else None
                        ),
                        "recovery_steps": event.get("recovery_steps"),
                        "post_change_success": event.get(
                            "post_change_success", row["success"]
                        ),
                    }
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        if event_rows:
            writer = csv.DictWriter(handle, fieldnames=list(event_rows[0]))
            writer.writeheader()
            writer.writerows(event_rows)
    print(f"Wrote {len(event_rows)} event rows to {args.out}")


if __name__ == "__main__":
    main()
