"""Launch the long ablation-and-artifact queue as a detached local process."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = PROJECT_ROOT / "runs" / "research"


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    stdout_path = RUN_DIR / "ablation_queue_post_move.stdout.log"
    stderr_path = RUN_DIR / "ablation_queue_post_move.stderr.log"
    command = [sys.executable, "scripts/train_ablation_queue.py"]
    options: dict[str, object] = {
        "cwd": PROJECT_ROOT,
        "stdin": subprocess.DEVNULL,
        "close_fds": True,
    }
    if os.name == "nt":
        options["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NO_WINDOW
        )
    else:
        options["start_new_session"] = True

    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            command,
            stdout=stdout,
            stderr=stderr,
            **options,
        )

    time.sleep(1.0)
    return_code = process.poll()
    if return_code is not None:
        error_tail = stderr_path.read_text(encoding="utf-8")[-2000:]
        raise RuntimeError(
            f"Research queue exited during startup with code {return_code}:\n"
            f"{error_tail}"
        )
    print(f"Launched research queue as PID {process.pid}")
    print(f"stdout: {stdout_path}")
    print(f"stderr: {stderr_path}")


if __name__ == "__main__":
    main()
