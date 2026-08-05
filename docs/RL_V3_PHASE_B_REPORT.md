# RL V3 Phase B: Controlled Maskable PPO Pilot Report

**Status:** complete; stopped after the four approved 100k pilots  
**Development seed:** 314159  
**Validation manifest:** `rl_v3_validation_v2.json`  
**Manifest SHA-256:** `3d2e18667de96516ae71250b2d2c5ce28b9027fce1cf5f79eaff052afe9d1574`  
**Private final test:** not generated, inspected, or used  
**Regression result:** 79 tests passed

## Executive conclusion

None of the four feed-forward Maskable PPO configurations passes the Phase B
engineering gates. The 250k two-seed stage must not begin.

One-step action masking was effective at its narrow purpose: no validation
episode collided. It did not solve long-horizon routing. At 100k interactions,
P1 succeeded on 2/96 scenarios, P2 on 3/96, P3 on 0/96, and P4 on 1/96. No
model succeeded on any 100x100 scenario. Only P2 succeeded on an empty 50x50
route, and no model showed credible empty-route scaling.

Global information gave the largest defensible structural improvement: P2
reduced two-cell oscillation from 88 episodes (P1) to 60 and achieved the best
final success count, but it converted many failures into longer repeated loops
rather than reliable completion. Recency alone was harmful: P3 ended with zero
successes and 64 longer-loop failures. R2 gave the best early checkpoint
(4/96 at 25k) but regressed to 1/96 at 100k, so it is not retained.

## Deliverables and implementation

Phase B added a frozen configuration, deterministic 96-scenario development
manifest, three observation contracts, two reward contracts, a four-stage
curriculum, a moderate multi-branch CNN/MLP policy, resumable training,
fixed-suite evaluation, full trajectory evidence, computation accounting,
coarse-map debug figures, comparison figures, and tests.

The policy continues to delegate movement, collision, dynamics timing and
termination to the authoritative Version 2 environment. The policy receives no
A* route distance, planner direction, current path, future schedule, or oracle
feature. The same Version 2-derived one-step legality mask is used for every
pilot.

The complete contracts and reward equations are in
`docs/RL_V3_PHASE_B_PROTOCOL.md`. Generated run artifacts are under
`runs/rl_v3/phase_b/` and the machine-readable comparison is
`runs/rl_v3/phase_b/phase_b_summary.json`.

## Validation_v2 design and integrity

The original Phase A manifest was preserved byte-for-byte with SHA-256
`36222fa85f726f57be1cc8094234f5151934f3f9e83c11b725e5af5b9b4bae80`.
Validation_v2 contains exactly:

- 96 scenarios total;
- 24 scenarios at each grid size 15, 30, 50 and 100;
- 32 scenarios in each short, medium and long initial-route bin;
- 24 scenarios in each empty/nearly-empty, random-static, structured and
  dynamic family;
- two scenarios in every scale x family x route-length cell.

Structured scenarios include long walls, U-shapes, corridors and dead ends.
Dynamic cases use `post_move_observed`. Seeds do not overlap the training
generator preview or private final-test range. The manifest explicitly declares
`final_test: false`.

## Observation and architecture specifications

`local_only` uses an 11x11 eight-channel crop and four approved scalars.
`global_local` adds a fixed 32x32 eight-channel coarse map.
`global_local_recency` activates visitation/recency in the eighth channel of
both maps. P1 has 159,929 parameters; P2-P4 each have 470,025.

The global map distinguishes blocked presence and free presence, so fully free,
fully blocked and mixed bins are separable. It also represents hard no-fly
presence, maximum traversal penalty, UAV, goal, recent changes and optional
recency. Channel-specific maximum aggregation preserves one-cell walls and
narrow corridors, including non-divisible dimensions and same-bin agent/goal
positions. Six side-by-side debug renders are in
`runs/rl_v3/phase_b/coarse_debug/`.

The initial cellwise implementation was performance-limited. P2 was paused at
its durable 50k checkpoint, the aggregation was vectorized, and randomized
tests proved exact array equality at grid sizes 15, 31, 50, 73 and 100 with
obstacles, no-fly cells, penalties, recency and dynamic changes. P2 then resumed
under the documented statistically-equivalent checkpoint contract. Its status
retains both source provenance hashes. This changed computation speed, not the
observation contract or values.

## Reward specifications

For episode budget `B`, R1 uses goal `+1`, collision `-1`, and non-terminal
step reward `-0.20/B`. It has no progress shaping.

R2 adds `0.10 * (0.99 Phi(next) - Phi(current))` on non-terminal transitions,
where `Phi = -normalized_octile_distance`. This is geometric information
derived only from current agent and goal coordinates. It is obstacle-independent
and small relative to the terminal objective.

## Learning curves

| Pilot | Observation | Reward | 25k | 50k | 100k |
|---|---|---:|---:|---:|---:|
| P1 | local only | R1 | 2/96 (2.08%) | 1/96 (1.04%) | 2/96 (2.08%) |
| P2 | global + local | R1 | 2/96 (2.08%) | 4/96 (4.17%) | 3/96 (3.13%) |
| P3 | global + local + recency | R1 | 1/96 (1.04%) | 2/96 (2.08%) | 0/96 (0%) |
| P4 | global + local + recency | R2 | 4/96 (4.17%) | 3/96 (3.13%) | 1/96 (1.04%) |

No curve shows reliable improvement with budget. P2 and P4 peaked before 100k;
P3 collapsed to zero. Training return was not used for model selection.

## Final validation by scale and family

| Pilot | 15x15 | 30x30 | 50x50 | 100x100 | Empty | Random | Structured | Dynamic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P1 | 1/24 | 1/24 | 0/24 | 0/24 | 1/24 | 0/24 | 1/24 | 0/24 |
| P2 | 2/24 | 0/24 | 1/24 | 0/24 | 2/24 | 0/24 | 1/24 | 0/24 |
| P3 | 0/24 | 0/24 | 0/24 | 0/24 | 0/24 | 0/24 | 0/24 | 0/24 |
| P4 | 1/24 | 0/24 | 0/24 | 0/24 | 0/24 | 0/24 | 1/24 | 0/24 |

All models failed every dynamic scenario. No model solved nearly all empty
15x15/30x30 routes. Only P2 showed non-zero empty 50x50 success (1/6), and no
model solved an empty 100x100 route. These directly fail the Phase B gates.

## Failure taxonomy at 100k

The label uses deterministic precedence, so each episode appears once.

| Pilot | Success | Two-cell oscillation | Longer repeated loop | Excessive detour |
|---|---:|---:|---:|---:|
| P1 | 2 | 88 | 2 | 4 |
| P2 | 3 | 60 | 30 | 3 |
| P3 | 0 | 22 | 64 | 10 |
| P4 | 1 | 70 | 7 | 18 |

Collision count is zero for every pilot. Nearly all unsuccessful episodes time
out. Recency reduced the narrow two-cell label but increased longer loops and
did not improve success; it changed failure topology rather than solving it.
R2 increased excessive-detour failures and did not reliably prevent
goal-greedy behavior around barriers.

## Computation and latency

PyTorch was CPU-only on an Intel i7-8650U (8 logical CPUs, 15.86 GiB RAM).
CUDA was unavailable, so GPU utilization is recorded as unavailable rather
than fabricated as zero. Average process CPU utilization was approximately
40-48% of total logical capacity.

| Pilot | Parameters | Training throughput range | Policy microbenchmark | Observed validation decision latency |
|---|---:|---:|---:|---:|
| P1 | 159,929 | 145-175 steps/s | 2.20 ms | 3.28 ms |
| P2 | 470,025 | 57-98 steps/s | 2.58 ms | 2.99 ms |
| P3 | 470,025 | 101-124 steps/s | 2.11 ms | 3.26 ms |
| P4 | 470,025 | 96-118 steps/s | 1.87 ms | 4.12 ms |

P2's early throughput includes the pre-optimization cellwise map builder and
must not be interpreted as a pure architecture penalty. Final 100k attempted
validation computation was 64.7 s (P1), 69.1 s (P2), 74.2 s (P3), and 90.4 s
(P4). Successful-route computation was small because successes were rare.
Full per-episode decision counts, path-cost gaps and timing are stored in each
`evaluation/step_*/episodes.csv`.

## Decision and next research question

Recommended configurations for the 250k two-seed stage: **none**. Increasing
the same pilots' budgets would violate the gate that explicitly forbids scaling
models that still oscillate on empty long routes.

Global information was the largest useful change, but insufficient. Recency is
not retained. R2 is not retained. These data do not prove recurrence is
necessary: loop-dominated failures make a controlled recurrent experiment
reasonable, but optimization stability, curriculum difficulty, episode budget,
and reward scaling remain alternative explanations. A future phase should first
redesign and approve a smaller diagnostic study that separates those causes.
It must not simply spend 250k interactions on P1-P4.

If an unchanged configuration were nevertheless run for 250k on this hardware,
the observed post-optimization throughput implies roughly 29-43 minutes of
training per seed, plus fixed-suite evaluations. Two configurations with two
seeds would require roughly 2-3 hours of training plus analysis. This is a
compute estimate, not a recommendation or authorization.

## Artifact index

- Configuration: `configs/rl_v3_phase_b.json`
- Validation: `evaluation/manifests/rl_v3_validation_v2.json`
- Protocol: `docs/RL_V3_PHASE_B_PROTOCOL.md`
- Machine summary: `runs/rl_v3/phase_b/phase_b_summary.json`
- Checkpoint proof: `runs/rl_v3/phase_b/resume_verification/result.json`
- Per-pilot status/checkpoints/evaluations: `runs/rl_v3/phase_b/P1` through `P4`
- Coarse-map visual checks: `runs/rl_v3/phase_b/coarse_debug/`
- Learning, scale, taxonomy and trajectory figures:
  `runs/rl_v3/phase_b/comparison_figures/`

Every failure trajectory and a stratified sample of successes are preserved.
Version 2 and Phase A artifacts remain unchanged. No final multi-seed training,
recurrence, behavior cloning, extra RL algorithm, or planner tuning was started.
