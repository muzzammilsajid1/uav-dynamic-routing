"""Rebuild the reproducible classical-evaluation artifacts from a clean checkout."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(arguments: list[str]) -> None:
    print(f"\n> {' '.join(arguments)}", flush=True)
    subprocess.run(arguments, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip unit tests when only regenerating measurement artifacts.",
    )
    args = parser.parse_args()

    python = sys.executable
    if not args.skip_tests:
        _run([python, "-m", "pytest", "-q"])
    _run([python, "experiments/generate_week3_manifest.py"])
    _run(
        [
            python,
            "experiments/run_classical_benchmark.py",
            "--repetitions",
            str(args.repetitions),
        ]
    )
    _run([python, "evaluation/summarize_classical_benchmark.py"])
    print("\nClassical reproducibility artifacts regenerated successfully.")


if __name__ == "__main__":
    main()
