"""Generate LaTeX tables from final aggregate CSVs without manual transcription."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHOD_ORDER = ["astar", "dijkstra", "dstar_lite", "rl_full"]
CORE_METHODS = set(METHOD_ORDER)
METHOD_LABELS = {
    "astar": "A*",
    "dijkstra": "Dijkstra",
    "dstar_lite": "D* Lite",
    "rl_full": "DDQN+HER",
    "rl_dqn": "DQN+HER",
    "rl_no_her": "No HER",
    "rl_no_shaping": "No shaping",
    "rl_no_curriculum": "No curriculum",
    "rl_full_observation": "Full-grid obs.",
    "rl_dynamic_from_scratch": "Dynamic-only train",
}
SPLIT_LABELS = {
    "seen_layout_unseen_pairs": "Held-out pairs",
    "unseen_layout_same_density": "Unseen layout",
    "denser_unseen_layout": "Denser layout",
    "new_dynamic_obstacle_locations": "New obstacle sites",
    "changed_toggle_periods": "New toggle periods",
    "obstacle_density_40": "Static density 0.40",
    "stochastic_obstacles_p20": "Stochastic toggle 0.20",
    "moving_obstacles_period_2": "Moving obstacle, $T=2$",
    "wind_energy_penalty_1_0": "Energy penalty 1.0",
    "no_fly_hard": "Hard no-fly",
    "no_fly_penalty": "Penalized no-fly",
    "sensor_noise_p10": "Sensor noise 0.10",
    "scale_15": "$15\\times15$",
    "scale_30": "$30\\times30$",
    "scale_50": "$50\\times50$",
    "scale_100": "$100\\times100$",
}
GENERALIZATION_ORDER = [
    "seen_layout_unseen_pairs",
    "unseen_layout_same_density",
    "denser_unseen_layout",
    "new_dynamic_obstacle_locations",
    "changed_toggle_periods",
]


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _escape(value: str) -> str:
    return value.replace("_", r"\_")


def _estimate(row: dict[str, str], metric: str, digits: int = 3) -> str:
    mean = float(row[f"{metric}_mean"])
    ci = float(row[f"{metric}_95ci"])
    if math.isnan(mean):
        return "--"
    if math.isnan(ci):
        return f"{mean:.{digits}f}"
    return f"${mean:.{digits}f} \\pm {ci:.{digits}f}$"


def _percent_estimate(row: dict[str, str], metric: str) -> str:
    mean = 100 * float(row[f"{metric}_mean"])
    ci = 100 * float(row[f"{metric}_95ci"])
    if math.isnan(mean):
        return "--"
    if math.isnan(ci):
        return f"{mean:.1f}"
    return f"${mean:.1f} \\pm {ci:.1f}$"


def _method_key(row: dict[str, str]) -> int:
    try:
        return METHOD_ORDER.index(row["method"])
    except ValueError:
        return len(METHOD_ORDER)


def _table(
    rows: list[dict[str, str]],
    *,
    caption: str,
    label: str,
) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Method & Condition & $n$ & Success (\%) & Cost gap & Route ms \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{METHOD_LABELS.get(row['method'], _escape(row['method']))} & "
            f"{SPLIT_LABELS.get(row['split'], _escape(row['split']))} & "
            f"{row['independent_units']} & {_percent_estimate(row, 'success_rate')} & "
            f"{_estimate(row, 'path_cost_gap', digits=2)} & "
            f"{_estimate(row, 'compute_time_ms', digits=2)} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\label{{{label}}}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def _compact_comparison_table(
    rows: list[dict[str, str]],
    *,
    split_order: list[str],
    caption: str,
    label: str,
) -> str:
    by_key = {(row["method"], row["split"]): row for row in rows}
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.4pt}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Condition & C succ. (\%) & RL succ. (\%) & RL gap & A* ms & RL ms \\",
        r"\midrule",
    ]
    for split in split_order:
        classical = [
            by_key[(method, split)]
            for method in METHOD_ORDER[:-1]
            if (method, split) in by_key
        ]
        rl = by_key[("rl_full", split)]
        astar = by_key[("astar", split)]
        classical_success = min(
            100 * float(row["success_rate_mean"]) for row in classical
        )
        lines.append(
            f"{SPLIT_LABELS.get(split, _escape(split))} & "
            f"{classical_success:.1f} & {_percent_estimate(rl, 'success_rate')} & "
            f"{_estimate(rl, 'path_cost_gap', digits=2)} & "
            f"{_estimate(astar, 'compute_time_ms', digits=2)} & "
            f"{_estimate(rl, 'compute_time_ms', digits=2)} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            rf"\label{{{label}}}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def _scaling_table(rows: list[dict[str, str]]) -> str:
    return _compact_comparison_table(
        rows,
        split_order=["scale_15", "scale_30", "scale_50", "scale_100"],
        caption=(
            "Scaling summary (mean $\\pm$ 95\\% CI). C succ. is the minimum "
            "success among A*, Dijkstra, and D* Lite; RL estimates use five "
            "policies and gap uses successful routes."
        ),
        label="tab:scaling-results",
    )


def _adaptability_table(rows: list[dict[str, str]]) -> str:
    aggregate = sorted(
        [row for row in rows if row["split"] == "all_dynamic"],
        key=_method_key,
    )
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Event-level adaptability across all dynamic scenarios. "
        r"Values are means and 95\% confidence intervals over independent "
        r"scenarios for classical planners and policy seeds for RL.}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.3pt}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Method & $n$ & Post-success (\%) & Recovery (\%) & Cost shock & Reaction ms \\",
        r"\midrule",
    ]
    for row in aggregate:
        lines.append(
            f"{METHOD_LABELS.get(row['method'], _escape(row['method']))} & {row['independent_units']} & "
            f"{_percent_estimate(row, 'post_change_success')} & "
            f"{_percent_estimate(row, 'recovery_rate')} & "
            f"{_estimate(row, 'extra_optimal_cost')} & "
            f"{_estimate(row, 'reaction_time_ms')} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\label{tab:adaptability-results}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def _ablation_table(rows: list[dict[str, str]]) -> str:
    ablation_order = [
        "rl_dqn",
        "rl_dynamic_from_scratch",
        "rl_full",
        "rl_full_observation",
        "rl_no_curriculum",
        "rl_no_her",
        "rl_no_shaping",
    ]
    by_method = {row["method"]: row for row in rows}
    ordered = [by_method[method] for method in ablation_order if method in by_method]
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Held-out-pair ablations across five policy seeds (mean "
        r"$\pm$ 95\% CI); an interval is omitted when fewer than two finite "
        r"seed estimates are available.}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.7pt}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Variant & Success (\%) & Cost gap & Route ms \\",
        r"\midrule",
    ]
    for row in ordered:
        lines.append(
            f"{METHOD_LABELS[row['method']]} & "
            f"{_percent_estimate(row, 'success_rate')} & "
            f"{_estimate(row, 'path_cost_gap', digits=2)} & "
            f"{_estimate(row, 'compute_time_ms', digits=2)} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\label{tab:ablation-results}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "results" / "research_summary.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "paper_latex_v2" / "generated",
    )
    parser.add_argument(
        "--adaptability-summary",
        type=Path,
        default=(
            PROJECT_ROOT
            / "evaluation"
            / "results"
            / "adaptability_summary.csv"
        ),
    )
    args = parser.parse_args()
    rows = _read(args.summary)
    adaptability_rows = _read(args.adaptability_summary)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    generalization_splits = set(GENERALIZATION_ORDER)
    scaling_splits = {"scale_15", "scale_30", "scale_50", "scale_100"}
    manuscript_realism_splits = {
        "obstacle_density_40",
        "stochastic_obstacles_p20",
        "moving_obstacles_period_2",
        "wind_energy_penalty_1_0",
        "no_fly_hard",
        "no_fly_penalty",
        "sensor_noise_p10",
    }

    outputs = {
        "generalization_table.tex": _compact_comparison_table(
            [
                row
                for row in rows
                if row["split"] in generalization_splits
                and row["method"] in CORE_METHODS
            ],
            split_order=GENERALIZATION_ORDER,
            caption=(
                "Generalization summary (mean $\\pm$ 95\\% CI). C succ. is "
                "the minimum success among A*, Dijkstra, and D* Lite; RL "
                "estimates use five policies and gap uses successful routes."
            ),
            label="tab:generalization-results",
        ),
        "scaling_table.tex": _scaling_table(
            sorted([
                row
                for row in rows
                if row["split"] in scaling_splits
                and row["method"] in CORE_METHODS
            ], key=lambda row: (int(row["split"].split("_")[1]), _method_key(row))),
        ),
        "realism_table.tex": _compact_comparison_table(
            [
                row
                for row in rows
                if row["split"] in manuscript_realism_splits
                and row["method"] in CORE_METHODS
            ],
            split_order=[
                "obstacle_density_40",
                "stochastic_obstacles_p20",
                "moving_obstacles_period_2",
                "wind_energy_penalty_1_0",
                "no_fly_hard",
                "no_fly_penalty",
                "sensor_noise_p10",
            ],
            caption=(
                "Controlled realism summary (mean $\\pm$ 95\\% CI). C succ. "
                "is the minimum success among the three classical planners; "
                "RL gap uses successful routes."
            ),
            label="tab:realism-results",
        ),
        "ablation_table.tex": _ablation_table(
            [
                row
                for row in rows
                if row["method"].startswith("rl_")
                and row["split"] == "seen_layout_unseen_pairs"
            ],
        ),
        "adaptability_table.tex": _adaptability_table(
            [
                row
                for row in adaptability_rows
                if row["method"] in CORE_METHODS
            ]
        ),
    }
    for filename, content in outputs.items():
        (args.out_dir / filename).write_text(content, encoding="utf-8")
    print(f"Generated {len(outputs)} LaTeX tables in {args.out_dir}")


if __name__ == "__main__":
    main()
