"""Capture and attach an immutable digest of all training-relevant sources."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = (
    PROJECT_ROOT / "evaluation" / "results" / "training_source_snapshot.json"
)
TRAINING_FILES = (
    "configs/research_experiments.json",
    "envs/grid_environment.py",
    "experiments/train_multiseed.py",
    "requirements.txt",
    "rl_agent/double_dqn.py",
    "rl_agent/safe_her_buffer.py",
    "rl_agent/uav_env.py",
)


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot() -> dict[str, object]:
    files = {
        relative: _file_digest(PROJECT_ROOT / relative)
        for relative in TRAINING_FILES
    }
    aggregate = hashlib.sha256()
    for relative, digest in files.items():
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(digest.encode("ascii"))
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": aggregate.hexdigest(),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Replace the existing snapshot before annotating metadata.",
    )
    args = parser.parse_args()

    current = _snapshot()
    if args.refresh or not SNAPSHOT_PATH.exists():
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT_PATH.write_text(
            json.dumps(current, indent=2) + "\n",
            encoding="utf-8",
        )
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    if snapshot.get("sha256") != current.get("sha256"):
        changed = sorted(
            relative
            for relative in set(snapshot.get("files", {})) | set(current["files"])
            if snapshot.get("files", {}).get(relative)
            != current["files"].get(relative)
        )
        raise RuntimeError(
            "Training-relevant sources changed after the locked snapshot: "
            f"{changed}. Finish or supersede the active run before using "
            "--refresh."
        )

    annotated = 0
    for path in sorted(
        (PROJECT_ROOT / "evaluation" / "results").glob("training_*_seed_*.json")
    ):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata.get("training", {}).get("smoke_test"):
            continue
        metadata["training_source_snapshot"] = snapshot
        path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        annotated += 1
    print(
        f"Training source sha256={snapshot['sha256']}; "
        f"annotated {annotated} completed metadata files"
    )


if __name__ == "__main__":
    main()
