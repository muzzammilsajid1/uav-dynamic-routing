"""Create an auditable summary table from raw repeated benchmark runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _ci95(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["planner"]].append(row)

    summaries: list[dict[str, object]] = []
    for planner, group in sorted(grouped.items()):
        times = [float(row["route_compute_time_ms"]) for row in group]
        costs = [
            float(row["route_path_cost"])
            for row in group
            if row["success"].lower() == "true"
        ]
        expansions = [int(row["node_expansions"]) for row in group]
        summaries.append(
            {
                "planner": planner,
                "runs": len(group),
                "scenarios": len({row["scenario_id"] for row in group}),
                "repetitions": len({row["repetition"] for row in group}),
                "success_rate": sum(
                    row["success"].lower() == "true" for row in group
                )
                / len(group),
                "path_cost_mean": statistics.mean(costs),
                "route_time_mean_ms": statistics.mean(times),
                "route_time_median_ms": statistics.median(times),
                "route_time_std_ms": statistics.stdev(times),
                "route_time_95ci_half_width_ms": _ci95(times),
                "node_expansions_mean": statistics.mean(expansions),
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "results" / "classical_dynamic_raw.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "results" / "classical_dynamic_summary.json",
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "results" / "classical_dynamic_summary.md",
    )
    args = parser.parse_args()

    with args.raw.open(newline="", encoding="utf-8") as handle:
        summaries = summarize(list(csv.DictReader(handle)))
    args.out.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Repeated classical dynamic benchmark",
        "",
        "| Planner | Runs | Success | Path cost mean | Route time mean (ms) | "
        "Median (ms) | SD (ms) | 95% CI half-width (ms) | Expansions mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['planner']} | {row['runs']} | {row['success_rate']:.3f} | "
            f"{row['path_cost_mean']:.4f} | {row['route_time_mean_ms']:.4f} | "
            f"{row['route_time_median_ms']:.4f} | {row['route_time_std_ms']:.4f} | "
            f"{row['route_time_95ci_half_width_ms']:.4f} | "
            f"{row['node_expansions_mean']:.1f} |"
        )
    lines.extend(
        [
            "",
            "The confidence interval summarizes raw route-run timings. Rows are retained",
            "in `classical_dynamic_raw.csv`; environment details are in",
            "`classical_dynamic_environment.json`.",
            "",
        ]
    )
    args.markdown_out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote summaries to {args.out} and {args.markdown_out}")


if __name__ == "__main__":
    main()
