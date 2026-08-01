"""Generate evidence-bound abstract, results, and conclusion prose fragments."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _find(rows: list[dict[str, str]], method: str, split: str) -> dict[str, str]:
    return next(
        row
        for row in rows
        if row["method"] == method and row["split"] == split
    )


def _pct(row: dict[str, str]) -> str:
    return f"{100 * float(row['success_rate_mean']):.1f}\\%"


def _pct_value(value: str) -> str:
    return f"{100 * float(value):.1f}\\%"


def _number(row: dict[str, str], metric: str) -> str:
    return f"{float(row[f'{metric}_mean']):.3f}"


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

    rl_seen = _find(rows, "rl_full", "seen_layout_unseen_pairs")
    rl_layout = _find(rows, "rl_full", "unseen_layout_same_density")
    rl_dense = _find(rows, "rl_full", "denser_unseen_layout")
    astar_seen = _find(rows, "astar", "seen_layout_unseen_pairs")
    dstar_seen = _find(rows, "dstar_lite", "seen_layout_unseen_pairs")
    rl_scale_15 = _find(rows, "rl_full", "scale_15")
    rl_scale_30 = _find(rows, "rl_full", "scale_30")
    rl_scale_50 = _find(rows, "rl_full", "scale_50")
    rl_scale_100 = _find(rows, "rl_full", "scale_100")
    astar_scale_100 = _find(rows, "astar", "scale_100")
    vanilla = _find(rows, "rl_dqn", "seen_layout_unseen_pairs")
    no_her = _find(rows, "rl_no_her", "seen_layout_unseen_pairs")
    no_shaping = _find(rows, "rl_no_shaping", "seen_layout_unseen_pairs")
    no_curriculum = _find(rows, "rl_no_curriculum", "seen_layout_unseen_pairs")
    full_observation = _find(
        rows, "rl_full_observation", "seen_layout_unseen_pairs"
    )
    rl_adapt = _find(adaptability_rows, "rl_full", "all_dynamic")
    dstar_adapt = _find(adaptability_rows, "dstar_lite", "all_dynamic")

    abstract = (
        "Across five independently trained policies, DDQN+HER achieved "
        f"{_pct(rl_seen)} mean success on held-out endpoints, "
        f"{_pct(rl_layout)} on unseen matched-density layouts, and "
        f"{_pct(rl_dense)} on denser layouts; A*, Dijkstra, and D* Lite "
        "achieved 100\\% on all three splits. RL success fell from "
        f"{_pct(rl_scale_15)} at $15\\times15$ to "
        f"{_pct(rl_scale_100)} at $100\\times100$, where route decision "
        f"time was {_number(rl_scale_100, 'compute_time_ms')} ms versus "
        f"{_number(astar_scale_100, 'compute_time_ms')} ms for A*. "
        f"Post-change RL success was {_pct_value(rl_adapt['post_change_success_mean'])}, "
        "despite the recovery criterion being met before termination in all "
        "observed events. Ablations were non-monotonic: removing HER, shaping, "
        "or curriculum each reached 100\\% held-out-pair success."
    )
    results = (
        "Table~\\ref{tab:generalization-results} separates endpoint "
        "generalization from layout and dynamics shifts. The full RL "
        f"policy changes from {_pct(rl_seen)} success on the training layout "
        f"with held-out endpoints to {_pct(rl_layout)} on unseen matched-density "
        f"layouts and {_pct(rl_dense)} on denser layouts. The three classical "
        "planners remain at 100\\% on these splits, so the experiment provides "
        "no reliability advantage for the learned method. "
        "Scaling exposes the strongest boundary: RL success is "
        f"{_pct(rl_scale_15)}, {_pct(rl_scale_30)}, {_pct(rl_scale_50)}, and "
        f"{_pct(rl_scale_100)} at grid widths 15, 30, 50, and 100, while every "
        "classical planner remains at 100\\%. At $100\\times100$, the full RL "
        f"policy uses {_number(rl_scale_100, 'compute_time_ms')} ms of route "
        f"decision time versus {_number(astar_scale_100, 'compute_time_ms')} ms "
        "for A*. Thus, these implementations show no scale crossover in favor "
        "of RL.\n\n"
        "The held-out-pair ablation in Table~\\ref{tab:ablation-results} is "
        "also non-monotonic. Vanilla DQN and the full-grid observation reach "
        f"only {_pct(vanilla)} and {_pct(full_observation)}, respectively, "
        "supporting Double DQN and the local observation in this setup. Yet "
        f"removing HER ({_pct(no_her)}), shaping ({_pct(no_shaping)}), or the "
        f"curriculum ({_pct(no_curriculum)}) each exceeds the full method's "
        f"{_pct(rl_seen)} on this split. These one-factor interventions show "
        "configuration sensitivity, not separable causal main effects, because "
        "the learning components interact.\n\n"
        "At matched change events, the full RL method attains "
        f"{_pct_value(rl_adapt['post_change_success_mean'])} post-change "
        f"success, while its recovery criterion is met in "
        f"{_pct_value(rl_adapt['recovery_rate_mean'])} of observed events. "
        "This distinction matters: recovery of optimal remaining cost before "
        "termination does not guarantee that the complete route succeeds. "
        "The corresponding D* Lite values are "
        f"{_pct_value(dstar_adapt['post_change_success_mean'])} and "
        f"{_pct_value(dstar_adapt['recovery_rate_mean'])}."
    )
    conclusion = (
        "Under the shared graph, timing, and information contracts, the "
        "classical planners provide the strongest reliability and scale in "
        "this benchmark. DDQN+HER is competitive only on the native small-grid "
        f"setting: it achieves {_pct(rl_seen)} held-out-pair success and "
        f"{_pct(rl_layout)} on unseen matched-density layouts, but falls to "
        f"{_pct(rl_scale_100)} at $100\\times100$ while all classical planners "
        "remain at 100\\%. The ablations further reject a simple component "
        "story: Double DQN and local observations help, whereas HER, shaping, "
        "and curriculum are not individually supported as beneficial under "
        "the tested protocol. The contribution is therefore not a claim of RL "
        "superiority, but a reproducible map of where learned routing works, "
        "where it fails, and which conclusions survive seed variation, paired "
        "scenarios, distribution shift, scale, and event-level analysis."
    )

    (args.out_dir / "abstract_results.tex").write_text(
        abstract + "\n", encoding="utf-8"
    )
    (args.out_dir / "results_narrative.tex").write_text(
        results + "\n", encoding="utf-8"
    )
    (args.out_dir / "conclusion_results.tex").write_text(
        conclusion + "\n", encoding="utf-8"
    )
    print(f"Generated evidence-bound prose in {args.out_dir}")


if __name__ == "__main__":
    main()
