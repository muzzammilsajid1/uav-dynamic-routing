# Phase C2 Grid-Size Incident Note

**Date**: 2026-08-08
**Component**: Phase C2 Curriculum & Environment Scaling

## Summary of Findings
During the Phase C2 reproducibility audit, it was discovered that the curriculum's dynamic grid-sizing logic was not propagating correctly to the native C++ simulator (`UAVRoutingEnv`). The RL wrapper (`PhaseC2Env`) successfully parsed the requested grid sizes (15, 30, 50, 100), but the native simulator state was cached prematurely and remained locked at `15x15` for all episodes.

Furthermore, the `PotentialShapingWrapper` calculated `max_dist` using the stale `15x15` geometry, resulting in improper reward scaling for all episodes that were intended to be on larger grids.

## Impact on Previous Runs
The preserved buggy C2 runs are invalid as evidence of true 30/50/100-grid performance.

Under the exact buggy 15x15 simulator plus the current 240-case validation manifest, at most 62/240 = 25.8333% of cases are geometrically compatible with the accidental 15x15 world.

Therefore, the previously reported lost ~29.6% result cannot have been produced by that exact combination of code behavior and validation manifest. Because its raw artifacts were lost, its exact provenance remains unresolved. We do not speculate that it used another branch or manifest unless evidence exists.

Exclude this 29.6% run from scientific performance claims while preserving it as a historical observation.

## Artifact Preservation
The artifacts and checkpoints from the invalid v9 runs (including the Kaggle bundle and logs) have been **preserved** in the repository structure for debugging and historical provenance. Do not delete them.

## Corrective Actions
1. **Grid-Size Repair**: `PhaseC2Env.reset()` now correctly pushes `_grid_size` prior to `PhaseC0Env` reconstructing the native simulator.
2. **Potential Normalizer Repair**: `PotentialShapingWrapper` now dynamically computes `max_dist` on every `reset()` according to the active grid dimensions.
3. **Regression Tests**: A rigorous regression suite (`test_phase_c2_grid_sizing.py`) was introduced to enforce precise spatial consistency, shape generation, and boundary masking across `15->30->50->100` sequence shifts.
