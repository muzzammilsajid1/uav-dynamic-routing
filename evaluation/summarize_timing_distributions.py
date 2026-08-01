"""Report auditable route and decision timing distributions from raw runs."""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _method_rows(
    classical_path: Path,
    rl_paths: list[Path],
) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    classical = _read(classical_path)
    for row in classical:
        output.append(
            {
                **row,
                "method": row["planner"],
                "policy_seed": "classical",
            }
        )
    for path in rl_paths:
        rows = _read(path)
        for row in rows:
            output.append(
                {
                    **row,
                    "method": f"rl_{row['variant']}",
                }
            )
    return output


def _distribution(values: list[float]) -> dict[str, float | int]:
    return {
        "measurements": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else float("nan"),
        "minimum": min(values),
        "maximum": max(values),
    }


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    expanded: list[dict[str, str]] = []
    for row in rows:
        expanded.append(row)
        if row["policy_seed"] != "classical":
            expanded.append({**row, "policy_seed": "all_seeds"})

    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in expanded:
        grouped[(row["method"], row["split"], row["policy_seed"])].append(row)

    output: list[dict[str, object]] = []
    for (method, split, seed), group in sorted(grouped.items()):
        route = _distribution(
            [float(row["route_compute_time_ms"]) for row in group]
        )
        decision = _distribution(
            [float(row["mean_decision_latency_ms"]) for row in group]
        )
        repetitions = sorted({int(row["repetition"]) for row in group})
        output.append(
            {
                "method": method,
                "split": split,
                "policy_seed": seed,
                "scenarios": len({row["scenario_id"] for row in group}),
                "repetitions_per_scenario": len(repetitions),
                "route_measurements": route["measurements"],
                "route_mean_ms": route["mean"],
                "route_median_ms": route["median"],
                "route_sample_std_ms": route["sample_std"],
                "route_min_ms": route["minimum"],
                "route_max_ms": route["maximum"],
                "decision_measurements": decision["measurements"],
                "decision_mean_ms": decision["mean"],
                "decision_median_ms": decision["median"],
                "decision_sample_std_ms": decision["sample_std"],
                "decision_min_ms": decision["minimum"],
                "decision_max_ms": decision["maximum"],
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classical",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "results" / "classical_suite_raw.csv",
    )
    parser.add_argument("--rl", nargs="*", type=Path, default=[])
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "results" / "timing_distributions.csv",
    )
    args = parser.parse_args()
    rows = summarize(_method_rows(args.classical, args.rl))
    if not rows:
        raise RuntimeError("No timing rows were available")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} timing-distribution rows to {args.out}")


if __name__ == "__main__":
    main()
