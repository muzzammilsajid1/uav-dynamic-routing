# Phase C2 v11 Release Audit and Experimental Decision

**Evidence cutoff:** 2026-08-08

**Branch:** `rl-v3-c2-empty-multiscale`

**Status:** locally verified release candidate; no corrected full C2 run has yet
been claimed.

## Local Verification Record

- Complete suite: **150 passed, 0 failed** in 1,045.29 seconds.
- Native preflight: **PASS**, with verified oracle goal completion at 15, 30,
  50, and 100.
- Real M1 smoke: 2,048 update-aligned interactions, 240 validation routes,
  all 12 scale-distance cells present, and zero illegal selected actions.
- Recovery artifact: checkpoint bundle inventory independently rehashed with
  zero mismatches; model, generator/curriculum state, RNG state, status,
  evaluation, and provenance were present.

The smoke's observed success rate is only a pipeline sanity result and is not a
corrected C2 performance claim.

## Decision

Do not run the corrected M1 experiment from v10. Release v11 only after its
targeted tests, complete suite, native preflight, and isolated end-to-end smoke
pass. Then run a fresh M1 pilot from interaction zero. Do not resume the invalid
v9 network.

This is not architecture churn. It is a measurement-integrity repair. The
current scientific question—whether M1 can learn empty-map multiscale routing—
remains unanswered because every preserved full C2 result predates a valid
multiscale training and evaluation pipeline.

## Independently Verified State

- The v10 annotated tag resolves to commit `d8d1fcf`.
- Corrected `PhaseC2Env.reset()` propagates the selected size to the runtime
  field used to build `UAVRoutingEnv`.
- The native grid changes across 15, 30, 50, and 100.
- Potential-shaping distance normalization is recalculated after each reset.
- The validation manifest contains 240 unique routes: 20 per distance bin at
  each of four scales.
- Training and validation endpoint pairs, including reverse directions, are
  disjoint.
- M1 has 428,937 trainable parameters in the current environment.
- The invalid preserved v9 result and the unresolved lost historical result do
  not support corrected C2 performance claims.

## V11 Measurement Contract

Validation is deterministic and masked. Each route records the start, goal,
scale, distance bin, actions, legal masks, trajectory, terminal classification,
decision count, episode budget and return, initial octile cost, successful path
cost/gap/ratio, final goal distance, policy-forward latency, and masked-action
latency. Aggregates are emitted overall, by scale,
by distance bin, and by scale-distance cell. `crashed` is the only authoritative
collision field.

PPO is trained and checkpointed in complete 2,048-interaction rollout/update
blocks. Nominal checkpoints map to actual completed-update boundaries:

| Nominal target | Actual saved model |
|---:|---:|
| 25,000 | 26,624 |
| 50,000 | 51,200 |
| 75,000 | 75,776 |
| 100,000 | 100,352 |
| 150,000 | 151,552 |

Artifacts always record both values. This is preferable to saving a model
mid-rollout before the corresponding samples have been used in a PPO update.

Checkpoint bundles contain the model, endpoint-generator state and active
curriculum sizes, Python/NumPy/PyTorch RNG states, status, provenance, raw
evaluations, and a SHA-256 inventory. Resume verifies all ten critical source
files. Resume is described as *statistically equivalent*, not bit-identical,
because SB3 does not serialize a partially active episode; checkpoints are now
after completed rollout updates, which removes the more serious partial-rollout
ambiguity.

## Compute Gate

The Kaggle notebook must complete, in order:

1. checkout of the exact v11 tag and ten-file hash verification;
2. full pytest suite;
3. native 15/30/50/100 preflight and oracle traversal;
4. bounded CPU/CUDA benchmark;
5. isolated 2,048-interaction M1 training, full 240-route validation, provenance,
   RNG checkpoint, bundle, and complete-archive verification; and
6. fresh production M1 training from interaction zero.

Failure of any gate halts the notebook before the full run.

## Research Direction After M1

The first corrected M1 run is a pilot and learning-curve diagnostic, not final
multi-seed evidence. Its raw checkpoint curves should determine the next action:

- if learning is healthy but unconverged, extend the frozen method before
  redesigning it;
- if performance differs sharply by scale or distance, diagnose the failing
  cells and representation behavior;
- if M1 remains weak, run the scalar-only M2 control to separate spatial-model
  failure from reward/curriculum failure;
- freeze the method before confirmatory multi-seed evaluation; and
- keep the final test set sealed until development decisions are complete.

Only after C2 competence is established should C3 introduce obstacle and
collision reward semantics. The existing empty-map R2-PB result does not by
itself justify its unchanged use in obstacle-rich phases.
