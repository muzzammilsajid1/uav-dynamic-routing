"""Run the frozen Phase C2 M2 confirmatory seeds sequentially and resumably."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (11, 22, 33, 44, 55)
ROLLOUT_SIZE = 2048


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def aligned_target(requested: int) -> int:
    return int(math.ceil(int(requested) / ROLLOUT_SIZE) * ROLLOUT_SIZE)


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def require_clean_tracked_tree() -> None:
    for command in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"]):
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode != 0:
            raise RuntimeError("Tracked repository files must be clean before confirmatory training")


def verify_bundle(bundle_path: Path) -> dict:
    with zipfile.ZipFile(bundle_path) as archive:
        inventory = {}
        for line in archive.read("inventory.txt").decode("utf-8").splitlines():
            digest, name = line.split("  ", 1)
            inventory[name] = digest
        names = set(archive.namelist()) - {"inventory.txt"}
        missing = set(inventory) - names
        extra = names - set(inventory)
        mismatched = [
            name
            for name, digest in inventory.items()
            if hashlib.sha256(archive.read(name)).hexdigest() != digest
        ]
    if missing or extra or mismatched:
        raise RuntimeError(
            f"Invalid bundle {bundle_path}: missing={sorted(missing)}, "
            f"extra={sorted(extra)}, mismatched={mismatched}"
        )
    return {"sha256": sha256(bundle_path), "entries": len(inventory)}


def core_training_sources_match(provenance: dict) -> tuple[bool, list[str]]:
    """Compare every hashed source except this orchestration-only queue."""
    from cloud.kaggle.phase_c2_kaggle_runner import hash_source_file, source_hash_files

    mismatches = []
    for name, path in source_hash_files().items():
        if name == "confirmatory_queue":
            continue
        if provenance.get("hashes", {}).get(name) != hash_source_file(path):
            mismatches.append(name)
    return not mismatches, mismatches


def inspect_seed(out_dir: Path, seed: int, requested: int, expected_commit: str) -> dict:
    status_path = out_dir / "status.json"
    provenance_path = out_dir / "provenance.json"
    bundle_path = out_dir / "latest_checkpoint_bundle.zip"
    if not (status_path.exists() and provenance_path.exists() and bundle_path.exists()):
        return {
            "complete": False,
            "bundle_path": str(bundle_path) if bundle_path.exists() else None,
        }

    status = json.loads(status_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance["model_type"] != "M2":
        raise RuntimeError(f"{out_dir}: expected M2, found {provenance['model_type']}")
    if int(provenance["seed"]) != seed:
        raise RuntimeError(f"{out_dir}: expected seed {seed}, found {provenance['seed']}")
    prior_queue_only_commit = provenance["git_commit"] != expected_commit
    if prior_queue_only_commit:
        sources_match, mismatches = core_training_sources_match(provenance)
        if not sources_match:
            raise RuntimeError(
                f"{out_dir}: commit {provenance['git_commit']} != queue commit "
                f"{expected_commit}; changed core sources={mismatches}"
            )
    bundle = verify_bundle(bundle_path)
    completed = int(status["completed_interactions"])
    target = aligned_target(requested)
    final_evaluation = out_dir / f"evaluation_{target:06d}.json"
    final_model = out_dir / f"model_{target:06d}.zip"
    complete = completed == target and final_evaluation.exists() and final_model.exists()
    if completed > target:
        raise RuntimeError(f"{out_dir}: completed {completed} exceeds frozen target {target}")
    return {
        "complete": complete,
        "completed_interactions": completed,
        "bundle_path": str(bundle_path),
        "bundle_sha256": bundle["sha256"],
        "bundle_entries": bundle["entries"],
        "prior_queue_only_commit": prior_queue_only_commit,
    }


def run_seed(output_root: Path, seed: int, requested: int, device: str, commit: str) -> dict:
    out_dir = output_root / f"seed_{seed:03d}" / "artifacts"
    state = inspect_seed(out_dir, seed, requested, commit) if out_dir.exists() else {
        "complete": False,
        "bundle_path": None,
    }
    if state["complete"]:
        return {"seed": seed, "status": "already_complete", **state}

    if out_dir.exists() and any(out_dir.iterdir()) and state["bundle_path"] is None:
        raise RuntimeError(f"{out_dir}: partial output exists without a resumable bundle")

    environment = os.environ.copy()
    environment["KAGGLE_TEST_OUT_DIR"] = str(out_dir)
    command = [
        sys.executable,
        str(ROOT / "cloud" / "kaggle" / "phase_c2_kaggle_runner.py"),
        "--model",
        "M2",
        "--interactions",
        str(requested),
        "--device",
        device,
        "--seed",
        str(seed),
    ]
    if state["bundle_path"] is not None:
        command.extend(["--resume", "--bundle-path", state["bundle_path"]])
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    verified = inspect_seed(out_dir, seed, requested, commit)
    if not verified["complete"]:
        raise RuntimeError(f"Seed {seed} returned without reaching the frozen target")
    return {"seed": seed, "status": "completed", **verified}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--interactions", type=int, default=150000)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--expected-commit")
    args = parser.parse_args()

    if tuple(args.seeds) != DEFAULT_SEEDS:
        raise ValueError(f"Confirmatory seeds are frozen as {DEFAULT_SEEDS}")
    if args.interactions != 150000:
        raise ValueError("Confirmatory requested interactions are frozen at 150000")
    require_clean_tracked_tree()
    commit = git_commit()
    if args.expected_commit and commit != args.expected_commit:
        raise RuntimeError(f"HEAD {commit} != expected commit {args.expected_commit}")

    queue_path = args.output_root / "confirmatory_queue_status.json"
    queue = {
        "schema_version": 1,
        "classification": "confirmatory_validation",
        "model_type": "M2",
        "git_commit": commit,
        "seeds": list(DEFAULT_SEEDS),
        "requested_interactions_per_seed": 150000,
        "completed_interactions_per_seed": aligned_target(150000),
        "started_unix_time": time.time(),
        "runs": {},
        "status": "running",
    }
    write_json_atomic(queue_path, queue)
    try:
        for seed in DEFAULT_SEEDS:
            queue["active_seed"] = seed
            write_json_atomic(queue_path, queue)
            queue["runs"][str(seed)] = run_seed(
                args.output_root, seed, args.interactions, args.device, commit
            )
            write_json_atomic(queue_path, queue)
    except Exception as error:
        queue["status"] = "failed"
        queue["error"] = f"{type(error).__name__}: {error}"
        write_json_atomic(queue_path, queue)
        raise

    queue["status"] = "complete"
    queue["active_seed"] = None
    queue["finished_unix_time"] = time.time()
    write_json_atomic(queue_path, queue)
    print(queue_path)


if __name__ == "__main__":
    main()
