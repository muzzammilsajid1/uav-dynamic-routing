"""Mark the research roadmap complete only after all release gates pass."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "evaluation" / "results"
STATUS_PATH = PROJECT_ROOT / "docs" / "RESEARCH_EXECUTION_STATUS.md"
VARIANTS = (
    "full",
    "dqn",
    "no_her",
    "no_shaping",
    "no_curriculum",
    "full_observation",
    "dynamic_from_scratch",
)
SEEDS = (11, 22, 33, 44, 55)


def main() -> None:
    integrity = json.loads(
        (RESULTS / "integrity_report.json").read_text(encoding="utf-8")
    )
    build = json.loads(
        (PROJECT_ROOT / "output" / "pdf" / "build_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if integrity.get("status") != "passed":
        raise RuntimeError("Cannot finalize: artifact integrity did not pass")
    if build.get("artifact_integrity_status") != "passed":
        raise RuntimeError("Cannot finalize: PDF was not built from passed evidence")

    missing: list[str] = []
    for variant in VARIANTS:
        for seed in SEEDS:
            path = RESULTS / f"training_{variant}_seed_{seed:03d}.json"
            if not path.exists():
                missing.append(str(path))
                continue
            metadata = json.loads(path.read_text(encoding="utf-8"))
            if metadata.get("training", {}).get("smoke_test"):
                missing.append(f"{path} (smoke test)")
            if not metadata.get("training_source_snapshot", {}).get("sha256"):
                missing.append(f"{path} (missing source provenance)")
    if missing:
        raise RuntimeError(f"Cannot finalize; incomplete training evidence: {missing}")

    timestamp = datetime.now(timezone.utc).isoformat()
    content = f"""# Research execution status

Finalized at `{timestamp}` after the integrity-gated artifact and PDF pipeline.

| Workstream | Final evidence | Status |
|---|---|---|
| Reproducibility | Stable manifests, raw repetitions, environment/source provenance, one-command rebuild | Complete |
| Measurement rigor | Route and decision timings, repeated raw measurements, uncertainty estimates | Complete |
| Multi-seed RL | Five independently trained checkpoints and seed-level outputs for every variant | Complete |
| Generalization | Held-out pairs, unseen layouts/densities, locations, and periods | Complete |
| Scaling | Matched 15, 30, 50, and 100 cell grids | Complete |
| Strong baselines | Dijkstra, A*, and incremental D* Lite | Complete |
| Realistic scenarios | Density, stochastic/moving obstacles, energy, no-fly, and sensing sweeps | Complete |
| RL ablations | DQN, HER, shaping, curriculum, observation, and dynamic-scratch controls | Complete |
| Adaptability | Event CSVs, seed-aware summaries, plots, and paired tests | Complete |
| Paper revision | Generated tables/figures/prose and verified release PDF | Complete |

## Release gates

- Artifact integrity: `{integrity['status']}`
- Benchmark scenarios: `{integrity['scenario_count']}`
- Timing repetitions: `{integrity['repetitions']}`
- PDF pages: `{build['page_count']}`
- PDF SHA-256: `{build['pdf_sha256']}`
- Compiler: `{build['engine_version']}`
"""
    STATUS_PATH.write_text(content, encoding="utf-8")
    print(f"Finalized research status at {STATUS_PATH}")


if __name__ == "__main__":
    main()
