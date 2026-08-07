# Independent Validation and Reproducibility Lead

**Owner:** Simra
**Branch:** `rl-v3-independent-validation`
**Branched from:** `4a2c78c` (Add RL V3 Phase A diagnostics and Phase B PPO pilots)
**Execution environment:** Anti-Gravity
**Status:** Active — Phase 2 kickoff

## Why this role exists

Simra verifies the experiments; she does not modify the model. Keeping the
validator and the model-modifier as different people is a standard
credibility control — it means a claim in the paper ("classical planners
outperform RL," "P4 shows this failure mode") has been checked by someone
with no incentive to make the result look better than it is. This is not
secondary work; it is a load-bearing part of the Phase 2 methodology and
should be described as such in the paper.

She works exclusively on `rl-v3-independent-validation`, branched from the
preserved commit `4a2c78c`, so that her audit trail and the active
development branch (`rl-v3-development`) never collide or overwrite each
other.

## 1. Independently reproduce the project state

- [ ] Full test suite passes (note: prior audits showed 57/58 vs. 58/58 —
      confirm current pass count and reconcile the discrepancy)
- [ ] Phase A and Phase B reports match the saved JSON/CSV results
      (`runs/rl_v3/phase_a/phase_a_summary.json`,
      `runs/rl_v3/phase_b/phase_b_summary.json`, and the associated CSVs)
- [ ] Validation manifest hash is correct — `manifest_separation.json`
      currently records `validation_sha256` for **36** validation
      scenarios, while `phase_b/setup.json` records **96** balanced
      scenarios (24 × 4 families/scales). Reconcile which manifest this
      hash actually covers before treating either count as final.
- [ ] No private final-test manifest exists anywhere in the repo or run
      directories
- [ ] The four Phase B pilot numbers (P1–P4) can be regenerated from the
      stored summaries without rerunning training

## 2. Independently audit the validation suite

Check whether the 96 validation scenarios are genuinely balanced across:

- [ ] Scale: 15×15, 30×30, 50×50, 100×100
- [ ] Route length: short, medium, long
- [ ] Scenario type: empty, random, structured, dynamic
- [ ] Obstacle densities
- [ ] Scenario seeds (no unintended overlap with training or final-test seeds)

Flag any category that is disproportionately difficult or malformed —
this could quietly bias the headline numbers.

## 3. Manually inspect failure trajectories

The automatic failure classifier may mislabel episodes. Pull a stratified
sample across P1–P4 and hand-classify each into:

- Two-cell oscillation
- Longer loop
- Aimless movement
- Goal-directed movement followed by failure
- Blocked corridor
- Poor global route choice
- Timeout despite progress
- Observation or action anomaly

Compare manual labels against `failure_taxonomy_counts.csv`. Disagreement
here is informative: it can distinguish "genuine long-horizon planning
failure" from something narrower and more fixable.

## 4. Independently inspect the Phase B learning curves

Answer, per checkpoint/run, not just at the final checkpoint:

- [ ] Did training reward improve?
- [ ] Did episode length reduce?
- [ ] Did success appear during training but disappear at deterministic eval?
- [ ] Did entropy collapse?
- [ ] Did the models stop exploring too early?
- [ ] Did any intermediate checkpoint outperform the final one?
- [ ] Did P4 improve temporarily and then degrade?

This is a factual audit, not a retuning exercise — no architecture or
hyperparameter changes come out of this step.

## 5. Private final-test custodian

Simra privately holds the final-test seed / manifest-generation secret.
It is not shared with or accessible to the development side until **all**
of the following are frozen:

- [ ] Final architecture
- [ ] Hyperparameters
- [ ] Training seeds
- [ ] Primary metrics
- [ ] Statistical analysis plan

Only after that freeze does she use Anti-Gravity to generate and run the
final test — once. She must not generate or inspect it before the freeze
is confirmed in writing (this document, updated, is the freeze record).

## 6. Paper-ready documentation

Simra maintains, on her branch:

- Experiment ledger
- Artifact inventory
- Model/configuration table
- Failure-taxonomy table
- Record of completed and rejected experiments
- Reproducibility checklist
- Independent-results summary

## Explicit boundaries — not yet

Simra should not, until instructed otherwise:

- Modify the PPO architecture
- Alter the reward function
- Change the environment
- Start another 100k training campaign
- Generate the final test
- Tune anything based on validation results
- Modify the classical algorithms (A*, Dijkstra, D* Lite)
- Commit to `rl-v3-development` or any other development branch

Her mandate is verification and organization, run in parallel with —
never inside — the active modeling work.

## Ledger Entry � 2026-08-07
- Full test suite: 79/79 passed (using --basetemp override).
- Default pytest temp dir (%LOCALAPPDATA%\Temp\pytest-of-User) is blocked by Windows permissions on this machine � environment issue, not a code defect.
- Note: earlier project memory referenced 57/58 and 58/58 test counts; actual current suite is 79 tests total. Worth flagging to Bug/Muzzammil � could be suite growth over time or a different subset being run previously.


## Ledger Entry -- 2026-08-07 -- Phase B regeneration check
- Attempted to regenerate phase_b_summary.json via rl_v3/summarize_phase_b.py.
- Script requires runs/rl_v3/phase_b/<pilot>/evaluation/step_100000/aggregates.json and episodes.csv for each pilot.
- Confirmed via git log --all that these paths have never existed in git history on any branch.
- Root cause: .gitignore excludes runs/ (all training artifacts) and *.csv (local data/results) globally. Explicit exceptions exist for evaluation/manifests/, evaluation/results/, and two week3_*.csv files, but NOT for runs/rl_v3/phase_b/*/evaluation/step_100000/.
- status.json and learning_curve.png under each pilot folder appear to have been force-added, bypassing the ignore rule; the raw per-episode data was not.
- Conclusion: phase_b_summary.json can currently be verified for internal consistency (cross-checked against the report) but CANNOT be regenerated from source using only what is committed to the repo. Raw evaluation data must be sourced separately (likely from Muzzammil's local training environment) to complete a true regeneration check.
- Action needed: ask Muzzammil whether raw step_100000 evaluation data exists locally and can be shared, or whether it should be committed going forward (with .gitignore exception added) for future reproducibility.

