"""Run selected research seeds in Colab and persist each completion to Drive."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "research_experiments.json"


def _copy_replace(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        shutil.copy2(source, destination)


def _metadata_path(root: Path, variant: str, seed: int) -> Path:
    return (
        root
        / "evaluation"
        / "results"
        / f"training_{variant}_seed_{seed:03d}.json"
    )


def _is_complete(path: Path, variant: str) -> bool:
    if not path.exists():
        return False
    record = json.loads(path.read_text(encoding="utf-8"))["training"]
    expected = (
        {"dynamic_full"}
        if variant == "dynamic_from_scratch"
        else {"static", "dynamic_mild", "dynamic_full"}
    )
    completed = {stage["stage"] for stage in record.get("stages", [])}
    return not record.get("smoke_test") and completed == expected


def _sync_seed(backup: Path, variant: str, seed: int) -> None:
    relative_paths = [
        Path("models") / "research" / variant / f"seed_{seed:03d}",
        Path("runs") / "research" / variant / f"seed_{seed:03d}",
        Path("evaluation")
        / "results"
        / f"training_{variant}_seed_{seed:03d}.json",
        Path("evaluation") / "results" / "training_source_snapshot.json",
    ]
    for relative in relative_paths:
        _copy_replace(PROJECT_ROOT / relative, backup / relative)


def _restore_seed(backup: Path, variant: str, seed: int) -> None:
    relative_paths = [
        Path("models") / "research" / variant / f"seed_{seed:03d}",
        Path("runs") / "research" / variant / f"seed_{seed:03d}",
        Path("evaluation")
        / "results"
        / f"training_{variant}_seed_{seed:03d}.json",
    ]
    for relative in relative_paths:
        _copy_replace(backup / relative, PROJECT_ROOT / relative)


def _write_status(backup: Path, **values: object) -> None:
    path = backup / "colab_worker_status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                **values,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument(
        "--max-hours",
        type=float,
        default=20.0,
        help="Stop between seeds before exceeding this runtime budget.",
    )
    args = parser.parse_args()

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    seeds = args.seeds or [int(seed) for seed in config["policy_seeds"]]
    for variant in args.variants:
        if variant not in config["variants"]:
            raise ValueError(f"Unknown variant: {variant}")

    snapshot = (
        PROJECT_ROOT / "evaluation" / "results" / "training_source_snapshot.json"
    )
    backup_snapshot = (
        args.backup_root
        / "evaluation"
        / "results"
        / "training_source_snapshot.json"
    )
    if backup_snapshot.exists():
        if snapshot.read_bytes() != backup_snapshot.read_bytes():
            raise RuntimeError("Drive backup uses a different training-source snapshot")
    else:
        _copy_replace(snapshot, backup_snapshot)

    started = time.monotonic()
    completed: list[str] = []
    for variant in args.variants:
        for seed in seeds:
            key = f"{variant}:seed{seed:03d}"
            _restore_seed(args.backup_root, variant, seed)
            if _is_complete(_metadata_path(PROJECT_ROOT, variant, seed), variant):
                completed.append(key)
                continue
            if (time.monotonic() - started) / 3600 >= args.max_hours:
                _write_status(
                    args.backup_root,
                    state="time_budget_reached",
                    completed=completed,
                    next_task=key,
                )
                return
            _write_status(
                args.backup_root,
                state="training",
                task=key,
                completed=completed,
            )
            command = [
                sys.executable,
                "experiments/train_multiseed.py",
                "--variant",
                variant,
                "--seeds",
                str(seed),
            ]
            if variant == "dynamic_from_scratch":
                command.extend(["--stage", "dynamic_full"])
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            subprocess.run(
                [sys.executable, "scripts/capture_training_provenance.py"],
                cwd=PROJECT_ROOT,
                check=True,
            )
            _sync_seed(args.backup_root, variant, seed)
            completed.append(key)

    _write_status(args.backup_root, state="complete", completed=completed)


if __name__ == "__main__":
    main()
