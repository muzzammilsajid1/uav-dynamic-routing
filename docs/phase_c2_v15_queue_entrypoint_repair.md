# Phase C2 v15 Queue Entrypoint Repair

**Incident date:** 2026-08-08

After the v14 JSON-serialization repair, the queue was restarted from the
repository root. It stopped before launching seed 22 because Python initializes
`sys.path` with the invoked script directory (`scripts/`), not necessarily the
repository root. The queue's cross-version source-equivalence check therefore
could not import `cloud.kaggle.phase_c2_kaggle_runner`.

No training started and no seed artifacts changed. Seed 11 remained complete;
seeds 22, 33, 44, and 55 remained unstarted.

The queue now inserts its resolved repository root into `sys.path` before any
project import. It also provides `--verify-only`, which executes the complete
existing-artifact and cross-version source verification path without launching
training. This allows the exact published entrypoint to be checked safely before
resuming the queue.

This is another orchestration-only repair. The ten environment, learning,
reward, observation, masking, manifest, configuration, and runner sources are
unchanged from v13 and v14. The exact queue hash and commit remain recorded.
