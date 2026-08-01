"""Wait for the full method, then train every configured ablation sequentially."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "research_experiments.json"
STATUS_PATH = PROJECT_ROOT / "runs" / "research" / "ablation_queue_status.json"
ABLATIONS = [
    "dqn",
    "no_her",
    "no_shaping",
    "no_curriculum",
    "full_observation",
    "dynamic_from_scratch",
]


def _write_status(state: str, **details: object) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(
        json.dumps(
            {
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "state": state,
                **details,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _full_training_complete(seeds: list[int]) -> bool:
    for seed in seeds:
        path = (
            PROJECT_ROOT
            / "evaluation"
            / "results"
            / f"training_full_seed_{seed:03d}.json"
        )
        if not path.exists():
            return False
        record = json.loads(path.read_text(encoding="utf-8"))["training"]
        if record.get("smoke_test") or len(record.get("stages", [])) != 3:
            return False
    return True


def _completed_variant_seeds(variant: str, seeds: list[int]) -> list[int]:
    expected_stages = (
        {"dynamic_full"}
        if variant == "dynamic_from_scratch"
        else {"static", "dynamic_mild", "dynamic_full"}
    )
    completed: list[int] = []
    for seed in seeds:
        path = (
            PROJECT_ROOT
            / "evaluation"
            / "results"
            / f"training_{variant}_seed_{seed:03d}.json"
        )
        if not path.exists():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))["training"]
        stages = {stage["stage"] for stage in record.get("stages", [])}
        if not record.get("smoke_test") and stages == expected_stages:
            completed.append(seed)
    return completed


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in config["policy_seeds"]]
    deadline = time.monotonic() + 24 * 60 * 60
    _write_status("waiting_for_full_baseline", seeds=seeds)
    while not _full_training_complete(seeds):
        if time.monotonic() >= deadline:
            _write_status("failed", reason="baseline wait exceeded 24 hours")
            raise TimeoutError("Full baseline training did not finish within 24 hours")
        time.sleep(60)

    completed: list[str] = []
    for variant in ABLATIONS:
        completed_seeds = _completed_variant_seeds(variant, seeds)
        pending_seeds = [seed for seed in seeds if seed not in completed_seeds]
        if not pending_seeds:
            completed.append(variant)
            continue
        _write_status(
            "training",
            variant=variant,
            completed=completed,
            completed_seeds=completed_seeds,
            pending_seeds=pending_seeds,
        )
        command = [
            sys.executable,
            "experiments/train_multiseed.py",
            "--variant",
            variant,
            "--seeds",
            *[str(seed) for seed in pending_seeds],
        ]
        if variant == "dynamic_from_scratch":
            command.extend(["--stage", "dynamic_full"])
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        completed.append(variant)

    variants = ["full", *ABLATIONS]
    _write_status("generating_final_artifacts", completed=completed)
    subprocess.run(
        [
            sys.executable,
            "scripts/run_full_research.py",
            "--variants",
            *variants,
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    _write_status("complete", completed=completed, variants=variants)


if __name__ == "__main__":
    main()
