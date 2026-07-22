"""Paired Week 3 statistics for Dijkstra-vs-RL result CSVs."""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


def _read_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as f:
        return {row["scenario_id"]: row for row in csv.DictReader(f)}


def _as_float(row: dict[str, str], candidates: list[str]) -> float | None:
    for name in candidates:
        value = row.get(name)
        if value is None or value in {"", "NA", "N/A", "inf", "Infinity"}:
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return None


def _paired_t_test(diffs: list[float]) -> tuple[float, float]:
    if len(diffs) < 2:
        return float("nan"), float("nan")

    mean = sum(diffs) / len(diffs)
    variance = sum((x - mean) ** 2 for x in diffs) / (len(diffs) - 1)
    std = math.sqrt(variance)
    if std == 0:
        return 0.0, 1.0

    t_stat = mean / (std / math.sqrt(len(diffs)))
    try:
        from scipy import stats

        p_value = float(stats.ttest_1samp(diffs, popmean=0.0).pvalue)
    except Exception:
        # Normal approximation fallback. This keeps the script usable on a
        # minimal install; install scipy for the exact Student-t p-value.
        p_value = math.erfc(abs(t_stat) / math.sqrt(2))
    return t_stat, p_value


def _rank_absolute_values(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: abs(item[1]))
    ranks = [0.0] * len(values)
    position = 0

    while position < len(indexed):
        end = position + 1
        while (
            end < len(indexed)
            and math.isclose(
                abs(indexed[end][1]),
                abs(indexed[position][1]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            end += 1

        average_rank = (position + 1 + end) / 2.0
        for idx in range(position, end):
            ranks[indexed[idx][0]] = average_rank
        position = end

    return ranks


def _wilcoxon_signed_rank(diffs: list[float]) -> tuple[int, float, float, float, float, str]:
    nonzero_diffs = [diff for diff in diffs if abs(diff) > 1e-12]
    n = len(nonzero_diffs)
    if n == 0:
        return 0, 0.0, 0.0, 0.0, 1.0, "all-zero"

    ranks = _rank_absolute_values(nonzero_diffs)
    w_plus = sum(rank for rank, diff in zip(ranks, nonzero_diffs) if diff > 0)
    w_minus = sum(rank for rank, diff in zip(ranks, nonzero_diffs) if diff < 0)
    w_stat = min(w_plus, w_minus)

    if n <= 25:
        total_rank = sum(ranks)
        extreme_count = 0
        assignment_count = 1 << n
        for mask in range(assignment_count):
            assigned_plus = 0.0
            for idx, rank in enumerate(ranks):
                if mask & (1 << idx):
                    assigned_plus += rank
            assigned_minus = total_rank - assigned_plus
            if min(assigned_plus, assigned_minus) <= w_stat + 1e-12:
                extreme_count += 1
        return n, w_plus, w_minus, w_stat, extreme_count / assignment_count, "exact two-sided"

    mean = n * (n + 1) / 4
    variance = n * (n + 1) * (2 * n + 1) / 24
    z_score = (w_stat - mean + 0.5) / math.sqrt(variance)
    p_value = math.erfc(abs(z_score) / math.sqrt(2))
    return n, w_plus, w_minus, w_stat, p_value, "normal approximation two-sided"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired Week 3 stats.")
    parser.add_argument("--dijkstra-csv", type=Path, required=True)
    parser.add_argument("--rl-csv", type=Path, required=True)
    parser.add_argument(
        "--dijkstra-metric",
        nargs="+",
        default=["dynamic_path_cost", "dijkstra_cost", "path_cost"],
        help="Candidate metric columns in priority order.",
    )
    parser.add_argument(
        "--rl-metric",
        nargs="+",
        default=["dynamic_path_cost", "rl_path_cost", "dqn_cost", "path_cost"],
        help="Candidate metric columns in priority order.",
    )
    args = parser.parse_args()

    dijkstra_rows = _read_csv(args.dijkstra_csv)
    rl_rows = _read_csv(args.rl_csv)
    shared_ids = sorted(set(dijkstra_rows) & set(rl_rows), key=lambda x: int(x))

    diffs: list[float] = []
    skipped = 0
    mismatches = 0
    for scenario_id in shared_ids:
        dijkstra_row = dijkstra_rows[scenario_id]
        rl_row = rl_rows[scenario_id]
        if (dijkstra_row.get("start"), dijkstra_row.get("goal")) != (
            rl_row.get("start"),
            rl_row.get("goal"),
        ):
            mismatches += 1
            skipped += 1
            continue

        dijkstra_value = _as_float(dijkstra_row, args.dijkstra_metric)
        rl_value = _as_float(rl_row, args.rl_metric)
        if dijkstra_value is None or rl_value is None:
            skipped += 1
            continue
        diffs.append(rl_value - dijkstra_value)

    if not diffs:
        raise RuntimeError("No paired numeric rows found. Check scenario_id and metric columns.")

    t_stat, p_value = _paired_t_test(diffs)
    mean_diff = sum(diffs) / len(diffs)
    median_diff = statistics.median(diffs)
    wins = sum(1 for diff in diffs if diff < 0)
    ties = sum(1 for diff in diffs if abs(diff) <= 1e-12)
    losses = sum(1 for diff in diffs if diff > 0)
    wilcoxon_n, w_plus, w_minus, w_stat, wilcoxon_p, wilcoxon_method = (
        _wilcoxon_signed_rank(diffs)
    )

    print(f"Dijkstra rows: {len(dijkstra_rows)}")
    print(f"RL rows: {len(rl_rows)}")
    print(f"Shared scenario IDs: {len(shared_ids)}")
    print(f"Start/goal mismatches: {mismatches}")
    print(f"Paired scenarios used: {len(diffs)}")
    print(f"Skipped shared scenarios: {skipped}")
    print(f"Mean RL - Dijkstra difference: {mean_diff:.6f}")
    print(f"Median RL - Dijkstra difference: {median_diff:.6f}")
    print(f"RL lower/equal/higher count: {wins}/{ties}/{losses}")
    print(f"Wilcoxon nonzero pairs: {wilcoxon_n}")
    print(f"Wilcoxon W+: {w_plus:.6f}")
    print(f"Wilcoxon W-: {w_minus:.6f}")
    print(f"Wilcoxon statistic W: {w_stat:.6f}")
    print(f"Wilcoxon p-value: {wilcoxon_p:.6g}")
    print(f"Wilcoxon method: {wilcoxon_method}")
    print(f"Paired t-test t-statistic: {t_stat:.6f}")
    print(f"Paired t-test p-value: {p_value:.6g}")


if __name__ == "__main__":
    main()
