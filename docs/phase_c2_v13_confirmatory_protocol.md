# Phase C2 v13 Confirmatory Protocol

**Declared:** 2026-08-08, before confirmatory seed training

**Selected method:** scalar-only M2 Maskable PPO

**Development seed excluded:** 42

**Confirmatory seeds:** 11, 22, 33, 44, 55

## Frozen Training Rule

Every seed starts from interaction zero and trains for 150,000 requested
interactions, producing a primary update-aligned checkpoint at 151,552. The
same configuration, curriculum, reward, action masks, validation routes, route
budgets, and deterministic evaluation are used for every seed.

Intermediate 25k, 50k, 75k, and 100k checkpoints are retained for learning-curve
diagnosis. They are not used to select a different final checkpoint for each
seed. A poor seed is not discarded, extended, or retuned. The only valid early
termination is an engineering/integrity failure, in which case the exact saved
bundle is resumed after source verification.

The 151,552 checkpoint was selected during development because seed 42 had
explicit all-scale curriculum exposure and perfect validation at 100,352 and
151,552; additional training through 301,056 added no success-rate benefit. The
choice also preserves the original 150k C2 compute budget.

## Measurement and Claims

The primary confirmatory estimate is the across-training-seed distribution of
overall validation success at 151,552 interactions. The report will include all
five seed values, mean, standard deviation, median, range, and an uncertainty
interval appropriate for five independent training runs. Scale, distance,
scale-distance, path efficiency, collision/invalid-action counts, and failure
labels are secondary diagnostics.

Route-level episodes are not independent substitutes for training seeds.
Pooling 1,200 routes and reporting a narrow route-level interval as if there
were 1,200 independent trained policies is prohibited. Development seed 42 is
reported separately and is not included in the confirmatory aggregate.

The final test set remains sealed. Confirmatory validation may establish C2
robustness and freeze the method, but it is not final-test evidence.

The single-seed M1-M2 paired comparison remains a development ablation. Unless
M1 is also trained across confirmatory seeds, the paper must not claim
population-level statistical superiority of M2 over M1.

## Execution Contract

`scripts/run_phase_c2_confirmatory.py` runs seeds sequentially to avoid CPU
contention and writes an atomic queue status file. Each seed has its own output
root, provenance, update-aligned models, generator state, Python/NumPy/PyTorch
RNG state, evaluations, SHA-256 recovery bundle, raw archive, and complete
archive.

The queue:

- refuses a dirty tracked source tree;
- records and enforces one exact Git commit;
- refuses any seed list or budget other than the frozen protocol;
- passes the seed explicitly into Python, NumPy, PyTorch, MaskablePPO, and the
  endpoint generator through the existing runner plumbing;
- validates model type, seed, commit, completed interactions, final artifacts,
  and every recovery-bundle payload before marking a seed complete; and
- resumes only from an existing verified bundle without rerunning completed
  seeds.

Before launch, the exact v13 tree must pass the complete test suite, native
multiscale preflight, isolated explicit-seed M2 smoke, canonical eleven-source
hash gate, and non-nesting archive checks.
