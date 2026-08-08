# Phase C2 v12 Release Audit

**Evidence cutoff:** 2026-08-08

**Branch:** `rl-v3-c2-empty-multiscale`

**Purpose:** freeze a reproducible corrected M2 control after the first corrected
M1 development run.

## Release Changes

- Critical-source provenance now uses `lf_normalized_sha256_v1`, so Windows
  CRLF and Linux LF checkouts produce the same source identity.
- Resume remains compatible with legacy v11 raw hashes by comparing the exact
  expected digest against raw, LF-normalized, and CRLF-normalized candidates.
  Changed content is still rejected.
- A resume checkpoint may be any positive, complete 2,048-interaction PPO
  rollout boundary at or below the requested final target. It no longer has to
  coincide with a named reporting checkpoint.
- Provenance records the runtime requested-to-completed checkpoint schedule.
- The default 300k plan includes 200k and 250k checkpoints in addition to the
  original 25k-150k series.
- Final raw and complete archives are built from an explicit direct-evidence
  whitelist. Repeated resume/archive operations exclude prior recovery bundles,
  raw archives, complete archives, and temporary files while retaining model
  checkpoint ZIPs.
- The Kaggle notebook is pinned to v12 and predeclares the M2 seed-42 control
  through 300k interactions.

## Verification Record

- Complete automated suite: **154 passed, 0 failed** in 1,059.14 seconds.
- Native preflight: **PASS**, with successful oracle traversal at 15, 30, 50,
  and 100.
- Isolated M2 smoke: 2,048 update-aligned interactions followed by all 240 fixed
  validation routes and all 12 scale-distance cells.
- Smoke validation: action masking applied, collision field `crashed`, and zero
  invalid selected actions.
- Smoke provenance: `source_hash_mode=lf_normalized_sha256_v1`, M2, CPU, and an
  explicit 2,048-to-2,048 requested/completed checkpoint schedule.
- Smoke recovery bundle: six inventoried payloads, zero missing/extra entries,
  and zero digest mismatches.
- Raw and complete smoke archives: eight entries each, `final_inventory.txt`
  present, and zero nested recovery/raw/complete container archives.

Smoke artifact hashes:

| Artifact | SHA-256 |
|---|---|
| `latest_checkpoint_bundle.zip` | `7afb1bbbdb38c1fa74b7a2c39e9f55fd9d614a7dd1399ed4b1e6e92abf2bfd2b` |
| `rl_v3_phase_c2_M2_smoke_raw_artifacts.zip` | `cb2958f4c4c84131c66059889ddfa0bd6fb732aea120577baab2d130560062df` |
| `phase_c2_M2_SMOKE_COMPLETE.zip` | `cb2958f4c4c84131c66059889ddfa0bd6fb732aea120577baab2d130560062df` |

The smoke success rate is only a pipeline observation and has no scientific
performance status.

## Release Decision

After committing and tagging this exact tree, run M2 seed 42 from interaction
zero through the frozen 300k checkpoint plan. Do not resume the engineering
smoke. Do not inspect the final test set. Compare the full M1/M2 development
curves and failure strata before freezing the confirmatory multi-seed method.
