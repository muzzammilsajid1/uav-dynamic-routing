# Phase C2 v13 Confirmatory Release Audit

**Evidence cutoff:** 2026-08-08

**Branch:** `rl-v3-c2-empty-multiscale`

**Purpose:** freeze explicit-seed execution and the five-seed M2 confirmatory
queue after model selection on development seed 42.

## Release Contract

- Confirmatory model: M2.
- Confirmatory seeds: 11, 22, 33, 44, 55.
- Requested interactions per seed: 150,000.
- Primary completed checkpoint per seed: 151,552.
- Seed 42 is development-only and excluded from the confirmatory aggregate.
- The runner accepts an explicit seed, records it in provenance, and rejects a
  resume bundle whose seed differs from the requested seed.
- Eleven critical source files are hashed with
  `lf_normalized_sha256_v1`, including the durable confirmatory queue.
- The local queue refuses changed tracked files, enforces the exact Git commit,
  validates recovery-bundle inventories, resumes incomplete seeds, and skips
  only independently verified completed seeds.

## Verification Record

- Complete automated suite: **163 passed, 0 failed** in 922.98 seconds.
- Native multiscale preflight: **PASS**, including oracle goal completion at 15,
  30, 50, and 100.
- Isolated explicit-seed smoke: M2 seed 11, 2,048 update-aligned interactions,
  240 validation routes, all 12 scale-distance cells, masked actions,
  authoritative `crashed` collision field, and zero invalid selected actions.
- Smoke provenance: seed 11, M2, CPU,
  `source_hash_mode=lf_normalized_sha256_v1`, eleven source hashes, and an exact
  2,048 requested/completed checkpoint schedule.
- Smoke bundle: six inventoried payloads, zero missing/extra entries, and zero
  digest mismatches.
- Smoke raw and complete archives: eight entries each, final inventory present,
  and no nested recovery/raw/complete container archives.

| Smoke artifact | SHA-256 |
|---|---|
| `latest_checkpoint_bundle.zip` | `9e0191d246914baa0e8d3de44d1916082b95711c750446b101386938f0a2845d` |
| `rl_v3_phase_c2_M2_smoke_raw_artifacts.zip` | `f04d5f4e47bab73b9d7746558f53214ae74eafdf1828f6519098635dc84cd934` |
| `phase_c2_M2_SMOKE_COMPLETE.zip` | `f04d5f4e47bab73b9d7746558f53214ae74eafdf1828f6519098635dc84cd934` |

The smoke success rate is a pipeline observation only and has no scientific
performance status.

## Release Decision

Commit and annotate this exact source as `rl-v3-c2-kaggle-v13`, verify the
remote peeled tag, and then run the frozen local confirmatory queue sequentially
from interaction zero. Do not change any hashed source while the queue is
active. Do not access the final test set.
