# Phase C1 Audit and Repair Report

## 1. Exact Bug Findings

1. **Path-Cost Analysis Skew**: The previous evaluation logic computed the path-cost gap by taking `mean(successful_realized_costs) - mean(all_attempted_astar_costs)`. Because failed routes weren't contributing to the realized cost average, but *were* contributing to the A* cost average, the resulting gap was meaningless. We now correctly compute `realized_i - astar_i` per route and average the differences, tracking failed A* costs separately.
2. **Orientation Classification Inversion**: The original `PhaseC1EndpointGenerator` categorized row-dominant movement (`dx > 0, dy <= dx*0.5`) as "horizontal" and column-dominant movement as "vertical". This has been corrected.
3. **Configuration Drift**: The C1 configuration had drifted from the frozen C0 settings. It was using `ent_coef=0.0`, no `gae_lambda`, and a lower episode budget multiplier (`2.0` vs `5.0`). These have been aligned.
4. **Test Suite Flaws**: The previous tests had erroneous scalar assertions (checking index 2 and dividing by 15.0 instead of index 0/1 and dividing by 14.0). We replaced the weak tests with robust regression tests covering validation manifests, exclusions, orientation, and true deterministic application.

## 2. Training Application Check

**Finding:** The code and saved aggregate evidence strongly support varied-endpoint training, but the preserved artifacts are insufficient for an independent per-reset reconstruction.
*Reasoning:* `PhaseC1Env.reset()` mutated `self._start` and `self._goal` before invoking `super().reset()`, passing them cleanly into `_make_v2()`. The invalid attempt's preserved `status.json` metrics file reflects A* costs tracking dynamically with distance bins (`short: 4.12`, `medium: 8.29`, `long: 13.24`). However, without full traces recording the exact endpoints initialized for each episode, we cannot perform a perfect post-hoc per-reset reconstruction of the run.

## 3. Files Changed
- **`rl_v3/generate_phase_c1_manifest.py`**: [NEW] Script to enforce immutable validation pair sampling with rigorous region stratification.
- **`rl_v3/phase_c1_env.py`**: [MODIFY] Generator updated to load fixed manifest; orientation bug fixed; state tracking added.
- **`rl_v3/run_phase_c1.py`**: [MODIFY] Preflight checks repaired; path-cost gap correctly derived; checkpoint resumption fixed; generator config loaded.
- **`configs/rl_v3_phase_c1.json`**: [MODIFY] Synced exact frozen params with `configs/rl_v3_phase_c0.json`.
- **`tests/test_rl_v3_phase_c1.py`**: [MODIFY] Replaced useless dummy tests with actual assertions.

## 4. Persisted Validation Manifest

- **Path:** `evaluation/manifests/rl_v3_phase_c1_validation.json`
- **Hash (SHA-256):** `49ffc2d987e5ba7f4f59b483ff043822cd61676edf1145fa46fd2604f44582fa`

## 5. Balance Tables

We sampled exactly 120 pairs balancing both distance bin and route orientation. We also implemented deterministic start/goal region stratification to maximize spatial coverage. 

**Note on Short Bin "Mixed" Routes:** We mathematically verified that any route classified as "mixed" (`dx > 0`, `dy > 0`, `abs(dx-dy) > 1`, and neither dominates by `<= 0.5`) requires a minimum weighted A* cost of ~`6.242`. The short bin threshold is strictly `< 6.2`. Therefore, "mixed" routes in the short bin are mathematically impossible. We allocated the 40 short validation pairs as `13 / 13 / 14` across vertical, horizontal, and diagonal orientations respectively.

### Orientation Distribution
| Bin | Vertical | Horizontal | Diagonal | Mixed | Total |
|---|---|---|---|---|---|
| **Short** | 13 | 13 | 14 | 0 | 40 |
| **Medium** | 10 | 10 | 10 | 10 | 40 |
| **Long** | 10 | 10 | 10 | 10 | 40 |
| **Total** | 33 | 33 | 34 | 20 | 120 |

### Region Distribution (Top 3)
| Bin | Start Regions (Top 3) | Goal Regions (Top 3) |
|---|---|---|
| **Short** | Region 3 (6x), Region 0 (5x), Region 4 (5x) | Region 4 (8x), Region 1 (7x), Region 3 (7x) |
| **Medium** | Region 3 (11x), Region 1 (7x), Region 4 (5x) | Region 8 (7x), Region 4 (7x), Region 1 (6x) |
| **Long** | Region 6 (7x), Region 2 (6x), Region 0 (5x) | Region 1 (7x), Region 5 (7x), Region 0 (5x) |

*All 120 pairs and their reverse counterparts are strictly excluded from the training generator.*

## 6. Frozen Configuration Comparison Against C0

| Parameter | Phase C0 (Frozen) | Phase C1 (Now Corrected) |
|---|---|---|
| **Model Seed** | 42 | 42 |
| **MaskablePPO Seed** | 42 | 42 |
| **Learning Rate** | 3e-4 | 3e-4 |
| **n_steps** | 256 | 256 |
| **batch_size** | 64 | 64 |
| **n_epochs** | 10 | 10 |
| **gamma** | 0.99 | 0.99 |
| **gae_lambda** | 0.95 | 0.95 |
| **ent_coef** | 0.01 | 0.01 |
| **features_dim** | 256 | 256 |
| **net_arch (pi/vf)** | [128, 64] | [128, 64] |
| **Activation** | ReLU | ReLU |

## 7. Preflight & Test Results

- **Unit Tests:** `pytest tests/test_rl_v3_phase_c1.py` completes fully, passing all tests.
- **Preflight Check:** `run_phase_c1.py preflight` succeeds, validating actual underlying environment application, disjoint split integrity, exact orientation balance, and deterministic hash matches.

## 8. Command Proposed for Fresh Phase C1 Run

To kick off the repaired Phase C1 training correctly from scratch, run:
```bash
.venv\Scripts\python.exe -m rl_v3.run_phase_c1 train
```
*(Checkpoints will now safely resume if the process is interrupted.)*
