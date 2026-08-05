# Phase C1 Execution Report

## Overview
Phase C1 training was initiated following a fully deterministic and structurally verified preflight check. The training generator and validation manifest were perfectly intact with exact hash matches. 

The run executed successfully without interruption but was terminated early by the frozen protocol stopping rules due to extremely low success rates.

## Preflight Status
- **Validation Manifest Hash Match:** `49ffc2d987e5ba7f4f59b483ff043822cd61676edf1145fa46fd2604f44582fa` (PASS)
- **Training Generator Hash Match:** `dd3170b20c47e895331f64fab1bba4a087daf820ed8613c86b42462bf9d2136e` (PASS)
- **Endpoint-State Mismatches:** 0
- **Validation Routes:** Exactly 120 (40 short, 40 medium, 40 long)
- **Overlap:** No training/validation overlap; no reverse-pair overlap
- **Status:** PASS (Training launched)

## Training Run Characteristics
- **Run Type:** Uninterrupted
- **Stopping Reason:** Early Stopping Triggered (`Validation success < 50% at 100k interactions`)
- **Total Timesteps Computed:** 100,000

## Checkpoint-by-Checkpoint Results

| Interactions | Overall Success | Short Bin | Medium Bin | Long Bin | Collisions | Timeouts | Oscillations (2-cell) | Longer Loops |
|--------------|-----------------|-----------|------------|----------|------------|----------|-----------------------|--------------|
| **10,000**   | 7.5%            | 15.0%     | 0.0%       | 7.5%     | 0          | 111      | 109                   | 2            |
| **25,000**   | 9.2%            | 12.5%     | 7.5%       | 7.5%     | 0          | 109      | 104                   | 5            |
| **50,000**   | 8.3%            | 17.5%     | 5.0%       | 2.5%     | 0          | 110      | 96                    | 14           |
| **100,000**  | 5.8%            | 12.5%     | 0.0%       | 5.0%     | 0          | 113      | 80                    | 33           |

## Cost and Efficiency (Final Checkpoint - 100k)
Because success rate was exceptionally low (5.8% / 7 routes total out of 120), path cost gaps are heavily skewed.
- **Short Bin (12.5% success):**
  - Mean A* Cost (Success): `2.531`
  - Mean Realized Cost: `5.076`
  - Mean Normalized Cost Ratio: `2.175`
  - Mean Path Cost Gap: `2.545`

*Metrics for Medium/Long bins lack statistical power (0-5% success).*

## Failure Taxonomy & Learning Diagnostics
- **Dominant Failure Mode:** Timeouts (averaging ~110 out of 120 per eval).
- **Oscillations/Loops:** The agent predominantly exhibited heavy oscillatory behavior (e.g. 109 out of 120 routes exhibiting 2-cell oscillation at 10k, gradually shifting to "longer loops" by 100k, moving from 2 at 10k to 33 at 100k). 
- **Collisions:** 0 collisions throughout. The safety masking successfully prevents all grid constraint violations, but the agent completely fails to navigate efficiently to the goal, opting to oscillate in place rather than progressing.

## Test Suite Result
- `109 passed in 24.59s` (No failures or warnings)

## Archival Data
- **Archive Path:** `runs/rl_v3_phase_c1_archive.zip`
- **Archive SHA-256:** `F064EEC8368D930C3DD45889982FCBC238F5B8266B8E1759326DA593F729800F`
- **Inventory:** Contains `status.json`, `preflight_verification.json`, and PyTorch model checkpoints at 10k, 25k, 50k, and 100k interactions.
