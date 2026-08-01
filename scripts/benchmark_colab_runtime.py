"""Measure short training throughput without creating research checkpoints."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.train_multiseed import _make_env, _new_model, _seed_everything
from rl_agent.double_dqn import DoubleDQN


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20_000)
    args = parser.parse_args()
    config = json.loads(
        (PROJECT_ROOT / "configs" / "research_experiments.json").read_text(
            encoding="utf-8"
        )
    )
    variant = config["variants"]["full"]
    stage = config["stages"][0]
    seed = 999
    _seed_everything(seed)
    env = _make_env(config, stage, variant, int(config["layout_seed"]))
    model = _new_model(
        DoubleDQN,
        env,
        config,
        variant,
        seed,
        args.steps,
    )
    started = time.perf_counter()
    model.learn(total_timesteps=args.steps, progress_bar=False)
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "steps": args.steps,
                "elapsed_seconds": elapsed,
                "steps_per_second": args.steps / elapsed,
                "device": str(model.device),
            },
            indent=2,
        )
    )
    env.close()


if __name__ == "__main__":
    main()
