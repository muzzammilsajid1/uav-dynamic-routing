"""Capture an auditable environment manifest for experiment outputs."""
from __future__ import annotations

import importlib.metadata
import hashlib
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _source_tree_digest(project_root: Path) -> tuple[str, int]:
    roots = [
        "baselines",
        "configs",
        "envs",
        "evaluation",
        "experiments",
        "rl_agent",
        "scripts",
    ]
    files = [project_root / "requirements.txt"]
    for root_name in roots:
        root = project_root / root_name
        if root.exists():
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file()
                and path.suffix.lower() in {".py", ".json", ".toml", ".yaml", ".yml"}
                and "__pycache__" not in path.parts
            )
    digest = hashlib.sha256()
    unique_files = sorted(set(files))
    for path in unique_files:
        relative = path.relative_to(project_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest(), len(unique_files)


def collect_environment_metadata(project_root: Path) -> dict[str, object]:
    package_names = [
        "gymnasium",
        "matplotlib",
        "networkx",
        "numpy",
        "pytest",
        "scipy",
        "stable-baselines3",
        "torch",
    ]
    packages: dict[str, str] = {}
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"

    metadata: dict[str, object] = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "operating_system": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "cpu": {
            "processor": platform.processor() or "unknown",
            "machine": platform.machine(),
            "logical_count": os.cpu_count(),
        },
        "packages": packages,
    }
    source_digest, source_count = _source_tree_digest(project_root)
    metadata["source_tree"] = {
        "sha256": source_digest,
        "files_hashed": source_count,
    }

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        metadata["git_commit"] = completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        metadata["git_commit"] = "unavailable"

    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        status_lines = [
            line for line in completed.stdout.splitlines() if line.strip()
        ]
        metadata["git_worktree"] = {
            "dirty": bool(status_lines),
            "changed_paths": status_lines,
        }
    except (OSError, subprocess.CalledProcessError):
        metadata["git_worktree"] = {"dirty": "unavailable"}

    try:
        import torch

        metadata["accelerator"] = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "gpu_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        }
    except ImportError:
        metadata["accelerator"] = {"torch": "not-installed"}

    return metadata
