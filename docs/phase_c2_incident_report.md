# Phase C2 Grid-Size Incident Note

**Date**: 2026-08-08
**Component**: Phase C2 Curriculum & Environment Scaling

## Summary of Findings
During the Phase C2 reproducibility audit, it was discovered that the curriculum's dynamic grid-sizing logic was not propagating correctly to the native simulator (`UAVRoutingEnv`). The RL wrapper (`PhaseC2Env`) successfully parsed the requested grid sizes (15, 30, 50, 100), but the native simulator state was reconstructed from a stale runtime field and remained locked at `15x15` for all episodes.

Furthermore, the `PotentialShapingWrapper` calculated `max_dist` using the stale `15x15` geometry, resulting in improper reward scaling for all episodes that were intended to be on larger grids.

## Impact on Previous Runs
The preserved buggy C2 runs are invalid as evidence of true 30/50/100-grid performance.

Under the exact buggy 15x15 simulator plus the current 240-case validation manifest, at most 62/240 = 25.8333% of cases are geometrically compatible with the accidental 15x15 world.

Therefore, the previously reported lost ~29.6% result cannot have been produced by that exact combination of code behavior and validation manifest. Because its raw artifacts were lost, its exact provenance remains unresolved. We do not speculate that it used another branch or manifest unless evidence exists.

Exclude this 29.6% run from scientific performance claims while preserving it as a historical observation.

## Artifact Preservation
The artifacts and checkpoints from the invalid v9 runs (including the Kaggle bundle and logs) have been preserved separately for debugging and historical provenance. Local extracted copies are not release inputs and must not be committed accidentally.

## Corrective Actions
1. **Grid-Size Repair**: `PhaseC2Env.reset()` now correctly pushes `_grid_size` prior to `PhaseC0Env` reconstructing the native simulator.
2. **Potential Normalizer Repair**: `PotentialShapingWrapper` now dynamically computes `max_dist` on every `reset()` according to the active grid dimensions.
3. **Regression Tests**: A rigorous regression suite (`test_phase_c2_grid_sizing.py`) was introduced to enforce precise spatial consistency, shape generation, and boundary masking across `15->30->50->100` sequence shifts.

## V10 Release Audit

The corrected full M1 experiment was not started from v10. A second independent
audit found additional release-blocking defects before compute was spent:

1. validation called `MaskablePPO.predict()` without an action mask;
2. validation and forensic scripts read `is_collision`, while the environment's
   authoritative field is `crashed`;
3. checkpoint bundles did not preserve Python, NumPy, PyTorch CPU, and CUDA RNG
   states or the active curriculum sizes;
4. resume verified only a subset of training-relevant source hashes;
5. aggregate-only validation output was insufficient to audit scale-specific
   failures, route behavior, action legality, and path efficiency;
6. nominal checkpoints could be saved mid-rollout before the corresponding PPO
   update, producing misleading interaction labels; and
7. every episode reset constructed the native environment twice.

These findings mean v10 remains useful as a repaired-grid release record but is
not approved for the corrected expensive experiment. The v11 candidate repairs
these issues, emits per-route evidence, aligns checkpoints to completed PPO
updates, and adds a true isolated end-to-end Kaggle smoke gate.
