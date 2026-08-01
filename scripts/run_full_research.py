"""Run the complete experiment and artifact-generation pipeline."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VARIANTS = [
    "full",
    "dqn",
    "no_her",
    "no_shaping",
    "no_curriculum",
    "full_observation",
    "dynamic_from_scratch",
]


def _run(arguments: list[str]) -> None:
    print(f"\n> {' '.join(arguments)}", flush=True)
    subprocess.run(arguments, cwd=PROJECT_ROOT, check=True)


def _completed_pair(route_path: Path, event_path: Path) -> bool:
    return (
        route_path.exists()
        and route_path.stat().st_size > 0
        and event_path.exists()
        and event_path.stat().st_size > 0
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run checkpointed training before evaluation.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=DEFAULT_VARIANTS,
        help="Full method and/or configured ablation variants.",
    )
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed classical and RL result pairs, then run final gates.",
    )
    args = parser.parse_args()
    python = sys.executable

    _run([python, "-m", "pytest", "-q"])
    _run([python, "experiments/generate_week3_manifest.py"])
    _run([python, "experiments/generate_benchmark_suite.py"])
    _run([python, "evaluation/validate_benchmark.py"])

    if args.train:
        for variant in args.variants:
            command = [
                python,
                "experiments/train_multiseed.py",
                "--variant",
                variant,
            ]
            if variant == "dynamic_from_scratch":
                command.extend(["--stage", "dynamic_full"])
            _run(command)

    _run([python, "scripts/capture_training_provenance.py"])

    classical_path = (
        PROJECT_ROOT / "evaluation" / "results" / "classical_suite_raw.csv"
    )
    classical_event_path = (
        PROJECT_ROOT
        / "evaluation"
        / "results"
        / "classical_adaptability_events.csv"
    )
    if args.resume and _completed_pair(classical_path, classical_event_path):
        print(
            f"\n> resume: keeping {classical_path.name} and "
            f"{classical_event_path.name}",
            flush=True,
        )
    else:
        _run(
            [
                python,
                "experiments/run_classical_suite.py",
                "--repetitions",
                str(args.repetitions),
            ]
        )
        _run([python, "evaluation/extract_classical_adaptability.py"])

    rl_paths: list[str] = []
    rl_event_paths: list[str] = []
    for variant in args.variants:
        route_path = (
            PROJECT_ROOT / "evaluation" / "results" / f"rl_{variant}_raw.csv"
        )
        event_path = (
            PROJECT_ROOT
            / "evaluation"
            / "results"
            / f"rl_{variant}_adaptability_events.csv"
        )
        if args.resume and _completed_pair(route_path, event_path):
            print(
                f"\n> resume: keeping {route_path.name} and {event_path.name}",
                flush=True,
            )
        else:
            _run(
                [
                    python,
                    "experiments/evaluate_multiseed.py",
                    "--variant",
                    variant,
                    "--repetitions",
                    str(args.repetitions),
                    "--out",
                    str(route_path),
                    "--events-out",
                    str(event_path),
                ]
            )
        rl_paths.append(str(route_path))
        rl_event_paths.append(str(event_path))

    _run(
        [
            python,
            "evaluation/check_artifact_integrity.py",
            "--repetitions",
            str(args.repetitions),
            "--rl",
            *rl_paths,
            "--rl-events",
            *rl_event_paths,
        ]
    )
    _run(
        [
            python,
            "evaluation/analyze_research_results.py",
            "--rl",
            *rl_paths,
        ]
    )
    _run(
        [
            python,
            "evaluation/summarize_timing_distributions.py",
            "--rl",
            *rl_paths,
        ]
    )
    _run(
        [
            python,
            "evaluation/statistical_tests.py",
            "--rl",
            *rl_paths,
        ]
    )
    _run(
        [
            python,
            "evaluation/analyze_adaptability.py",
            "--rl",
            *rl_event_paths,
        ]
    )
    _run([python, "evaluation/generate_research_plots.py"])
    _run([python, "evaluation/generate_paper_tables.py"])
    _run([python, "evaluation/generate_paper_narrative.py"])
    _run([python, "scripts/check_latex_sources.py"])
    _run([python, "scripts/install_tectonic.py"])
    _run([python, "scripts/compile_paper.py"])
    _run([python, "scripts/finalize_research_status.py"])
    print("\nFull research artifacts regenerated successfully.")


if __name__ == "__main__":
    main()
