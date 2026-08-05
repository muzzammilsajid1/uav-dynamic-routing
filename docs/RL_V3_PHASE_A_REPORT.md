# RL V3 Phase A Report

Generated on 2026-08-05 on branch `rl-v3-development`.

Phase A added scientific foundations and diagnostics for RL V3 without
modifying Version 2 evidence, changing classical planners, exposing final-test
scenarios, or launching substantive PPO training.

## Files Created Or Modified

Created:

- `configs/rl_v3_phase_a.json` - frozen Phase A distribution, diagnostic, and
  observation configuration.
- `docs/RL_V3_PHASE_A_PROTOCOL.md` - V2 authority map and final-test protocol.
- `evaluation/manifests/rl_v3_training_generator.json` - deterministic
  training generator asset, not a finite training manifest.
- `evaluation/manifests/rl_v3_validation.json` - fixed, hashed validation
  manifest with 36 scenarios.
- `rl_v3/` - additive V3 package for masks, observations, scenario generation,
  diagnostics, checkpoint smoke contract, and V2 wrapper.
- `tests/test_rl_v3_*.py` - Phase A contract tests.

Modified:

- `requirements.txt` - added pinned `sb3-contrib==2.9.0`.

Generated local diagnostic artifacts under ignored `runs/rl_v3/phase_a/`.

## Test Results

Full suite after installing `sb3-contrib==2.9.0`:

```powershell
.venv\Scripts\python.exe -m pytest -q --basetemp=tmp\pytest-full-phase-a-final
```

Result: `73 passed in 11.90s`.

## V2/V3 Transition Parity

`tests/test_rl_v3_parity.py` proves the V3 wrapper delegates transition
semantics to `UAVRoutingEnv.step()`. The test copies identical state into a V2
environment and the V3-wrapped core, executes the same action sequence, and
checks identical reward, termination, truncation, UAV position, dynamic-change
events, and grid state.

Authoritative V2 functions are documented in `docs/RL_V3_PHASE_A_PROTOCOL.md`.

## Manifest And Generator Separation

Training uses a deterministic generator, not one small finite manifest.

- Training generator hash:
  `c2df7b7417c2ba18f930e37e9e5659294f2544091d52a651770250f65e7eed80`
- Validation manifest hash:
  `36222fa85f726f57be1cc8094234f5151934f3f9e83c11b725e5af5b9b4bae80`
- Validation scenarios: 36
- Training-preview seeds checked: 1000
- Train/validation seed overlap: none
- Validation/private-final seed overlap: none
- Final-test manifest exposed: no

The final-test generation protocol is frozen in
`docs/RL_V3_PHASE_A_PROTOCOL.md`, but no readable final-test scenario manifest
is stored in the development repository.

## Original Versus Post-Hoc Masked DDQN

Diagnostic set: 24 validation episodes from the RL V3 validation manifest,
covering 15x15, 30x30, 50x50, and 100x100. The post-hoc masked condition is a
diagnostic shield over the frozen DDQN; it was not trained with masks.

| Grid | Original | Post-hoc masked | Main failures |
|---|---:|---:|---|
| 15 | 6/6 | 6/6 | none |
| 30 | 2/6 | 2/6 | two-cell oscillation; one original collision |
| 50 | 2/6 | 2/6 | oscillation; original collisions |
| 100 | 0/6 | 0/6 | oscillation and longer loops; original collisions |

Aggregate:

- Original: 10/24 success, 6 collisions, 7 two-cell oscillations, 1 longer loop.
- Post-hoc masked: 10/24 success, 0 collisions, 13 two-cell oscillations,
  1 longer loop.

Masking removes immediate illegal/collision behavior in this diagnostic set but
does not improve success. It converts many crashes into loops.

## Empty-Map And Route-Horizon Diagnostics

Diagnostic set: 112 empty-map episodes per mode across grid sizes 15, 20, 30,
40, 50, 75, and 100; orientations horizontal, vertical, diagonal, and
goal-opposing; distance ratios 0.25, 0.50, 0.75, and 1.00. Episode budgets were
capped at `max(20, 4 * initial A* cost)` for Phase A diagnostics and each row
records the budget/A* ratio.

| Grid | Original | Post-hoc masked | Main failures |
|---|---:|---:|---|
| 15 | 16/16 | 16/16 | none |
| 20 | 16/16 | 16/16 | none |
| 30 | 13/16 | 13/16 | two-cell oscillation |
| 40 | 7/16 | 7/16 | oscillation, loops, one original collision |
| 50 | 6/16 | 7/16 | oscillation, loops, original invalid/collision |
| 75 | 5/16 | 6/16 | loops, oscillation, original collisions |
| 100 | 5/16 | 6/16 | longer loops, original collisions |

Aggregate:

- Original: 68/112 success, 19 two-cell oscillations, 13 longer loops,
  11 collisions, 1 immediate invalid-action crash.
- Post-hoc masked: 71/112 success, 21 two-cell oscillations, 20 longer loops,
  0 collisions.

## Representative Plots

Representative plots are under:

- `runs/rl_v3/phase_a/plots/empty_success_by_grid_original.png`
- `runs/rl_v3/phase_a/plots/empty_success_by_grid_posthoc_masked.png`
- `runs/rl_v3/phase_a/plots/trajectories/`

Interpretation: the frozen local DDQN solves native and near-native empty maps,
but failure grows with route horizon and grid scale even without obstacles.
Masking improves safety but exposes persistent policy-level looping.

## Dominant Failure Interpretation

Immediate illegal actions are not the dominant root cause. They are real in the
unmasked policy, but post-hoc masking removes collisions while success barely
changes.

The dominant Phase A issue is long-horizon and scale generalization under local
observation. Empty-map failures at 40x40 and above show the policy can fail
without obstacle-navigation complexity. Obstacle navigation and reward design
may still matter, but they are downstream of a more basic horizon/scale problem.

## PPO Pilot Recommendation

The first PPO pilot should use the global-local observation, not a purely local
observation:

- local 11x11 crop;
- fixed-size coarse global map;
- explicit obstacle, no-fly, free/mixed-bin, agent, goal, changed-cell, penalty,
  and visitation/recency channels;
- scalars limited to normalized relative goal row, normalized relative goal
  column, normalized grid size, and previous action;
- one-step action masking through the V2 neighbor contract.

The global-local Maskable PPO design remains appropriate, with one amendment:
coarse-map downsampling must include both max occupancy and a free-cell/mixed
signal so one-cell walls and narrow corridors are not destroyed by aggregation.

Do not add A*-derived distance features yet. A normalized geometric octile
distance feature may be considered later, but it should be justified as a
scale/horizon cue rather than a planner-derived leakage path.

## Checkpoint And Resume Foundation

Smoke checkpoint path:

- `runs/rl_v3/phase_a/checkpoint_smoke/checkpoint.pt`
- `runs/rl_v3/phase_a/checkpoint_smoke/status.json`

The smoke checkpoint stores model and optimizer state, timestep,
configuration, policy seed, generator state, Python/NumPy/Torch RNG states,
normalization stats, source revision, manifest/generator hashes, software and
hardware metadata, and run status.

Resume contract verification: passed.

For future vectorized or multi-process training, this report does not claim
bit-identical resume. Until separately verified, vectorized resume should be
described as statistically equivalent.
