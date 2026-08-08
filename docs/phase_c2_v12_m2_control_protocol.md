# Phase C2 v12 M2 Control Protocol

**Declared:** 2026-08-08, before corrected M2 training

**Classification:** development experiment

## Question

Does the 428,937-parameter M1 spatial architecture provide useful information
for empty-map multiscale navigation, or can the approximately 9,545-parameter
M2 scalar policy learn the same task from relative goal position, grid size,
and previous action alone?

This is a diagnostic control, not an attempt to replace M1 after seeing a
single favorable or unfavorable checkpoint. Empty C2 maps contain no obstacle
geometry, so a scalar policy may be sufficient. Conversely, a persistent M2
failure at large scale would strengthen the case that M1's spatial/global-local
representation contributes beyond simple directional features.

## Frozen Comparison

- model: M2 scalar-only Maskable PPO;
- training seed: 42;
- training curriculum, reward, action masking, optimizer settings, validation
  manifest, deterministic evaluation, and route budgets: identical to M1;
- requested checkpoints: 25k, 50k, 75k, 100k, 150k, 200k, 250k, and 300k;
- saved checkpoints: the corresponding complete 2,048-interaction PPO update
  boundaries;
- primary development metric: overall success on the fixed 240-route validation
  manifest;
- diagnostic strata: scale, distance, scale-distance cell, path efficiency,
  failure label, collision/invalid-action counts, and inference latency; and
- no final-test access.

The M2 run may stop early only for a pipeline/integrity failure or a clearly
documented compute failure. Poor validation performance is not a stopping
condition. No M2-specific hyperparameter tuning is allowed before this control
finishes.

## Interpretation Rule

M1 and M2 will first be compared as seed-42 development trajectories, including
instability and scale-specific behavior rather than only their best aggregate
checkpoint. This comparison cannot establish population-level superiority.

After M2, the C2 method and checkpoint-selection rule will be frozen. The chosen
method must then be retrained with multiple independent seeds before any final
test or paper-level performance claim. If the comparison is ambiguous, the
scientifically conservative choice is the simpler M2 unless M1 shows a
repeatable, practically important large-scale advantage.

## V12 Engineering Gate

Before training, v12 must pass:

1. line-ending-independent hashes for all ten critical source files;
2. legacy-v11 resume compatibility without accepting changed content;
3. resume from any positive, complete PPO rollout boundary not exceeding the
   requested target;
4. repeated non-nesting raw and complete archive creation;
5. native 15/30/50/100 oracle preflight;
6. the complete automated test suite; and
7. an isolated 2,048-interaction M2 end-to-end smoke with bundle and archive
   verification.
