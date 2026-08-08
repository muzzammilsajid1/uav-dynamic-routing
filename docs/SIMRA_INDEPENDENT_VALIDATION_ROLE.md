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


## Ledger Entry -- 2026-08-08 -- Phase B regeneration CONFIRMED (independent)
- Muzzammil provided raw step_100000 evaluation data (aggregates.json, episodes.csv, 96 trajectories per pilot) via zip; canonical source also committed at commit a0c97e2 on branch rl-v3-c2-empty-multiscale (not rl-v3-development).
- Ran rl_v3/summarize_phase_b.py's summarize() directly (the __main__ block only prints the decision block, not the full summary -- noted for future reference) against raw data for all 4 pilots on my own machine.
- Deep-diffed regenerated output against committed runs/rl_v3/phase_b/phase_b_summary.json.
- RESULT: 0 differences. Independently confirmed reproducible.
- Also verified validation_v2_hash: manifest_sha256 field inside evaluation/manifests/rl_v3_validation_v2.json matches the hash recorded in phase_b_summary.json exactly. Manifest contains 96 scenarios.
- Manifest has a top-level final_test field; confirmed type only (bool, False) as a placeholder/status flag -- not an embedded final-test manifest. No boundary crossed.
- Open item: phase_a/manifest_separation.json references a 36-scenario hash, distinct from this 96-scenario v2 manifest -- likely an older v1 manifest, not yet reconciled with Bug/Muzzammil.
- Process note: raw Phase B eval data is git-ignored by design; canonical copy currently lives on rl-v3-c2-empty-multiscale, worth confirming that's intentional.


## Ledger Entry -- 2026-08-08 -- Phase A verification (partial -- BLOCKED on checkpoint)
- run_phase_a.py loads a frozen DDQN checkpoint (models/research/full/seed_011/02_dynamic_full_final.zip) and runs evaluation-only diagnostics against it; does not train. checkpoint_smoke step uses timesteps=3, confirmed a smoke test not a training run.
- Checkpoint file confirmed MISSING: not present locally (Test-Path False), not in git history on any branch (git log --all -- checkpoint path returned nothing). This is a bigger gap than the Phase B raw-data case -- that data at least existed on rl-v3-c2-empty-multiscale; this checkpoint does not exist anywhere in git.
- BLOCKED: cannot regenerate the DDQN diagnostic numbers (41.7% baseline success rate, failure taxonomy counts in phase_a_summary.json) until Muzzammil provides this checkpoint. Requested via message 2026-08-08.
- COMPLETED (does not require checkpoint): verified manifest_separation.json is byte-identical to the copy embedded in phase_a_summary.json. Confirmed train/validation and validation/final-private seed overlaps are both empty lists as claimed.
- Working hypothesis on the 36 vs 96 scenario question (open since earlier session): Phase A uses evaluation/manifests/rl_v3_validation.json (v1, 36 scenarios); Phase B uses rl_v3_validation_v2.json (96 scenarios). These appear to be two intentionally separate, versioned manifests, not an error. Diagnostics run only 24 of the 36 available Phase A scenarios due to --limit-validation-per-grid 6 x 4 grid sizes. Not yet confirmed with Bug/Muzzammil -- still open.
- CAVEAT: manifest_separation.json consistency check confirms the two stored copies agree with each other, not that validation_sha256/generator_hash were correctly computed from actual manifest content in the first place. Lower priority than the checkpoint blocker, but worth a follow-up hash recomputation once the checkpoint unblocks the rest of Phase A.


## Ledger Entry -- 2026-08-08 -- Validation suite balance audit (item 2, COMPLETE)
- Audited evaluation/manifests/rl_v3_validation_v2.json (96 scenarios) across all charter dimensions.
- Grid size: perfectly balanced, 24/24/24/24 across 15/30/50/100.
- Route length: perfectly balanced, 32/32/32 across short/medium/long.
- Scenario family: perfectly balanced, 24/24/24/24 across empty/random_static/structured/dynamic.
- Cross-tab grid_size x family: exactly 6 per combination, no skew. Cross-tab grid_size x route_bucket: exactly 8 per combination, no skew.
- Seeds: all 96 episode_seed values unique, no duplicates. All 96 scenario_id values unique.
- Obstacle density: varies continuously and correlates with family as expected by design (empty near-zero, random_static widest spread 0.03-0.16) -- not a balance concern.
- FINDING: generation_attempts (retries needed by the scenario generator to produce a valid scenario) is NOT evenly distributed. dynamic family averaged 4.42 attempts (max 19) vs empty family's 3.12 (max 7). Short-route dynamic scenarios dominate the hardest-to-generate list, e.g. VAL2-G015-DYNAMIC-SHORT-02 needed 19 attempts.
- INTERPRETATION (observation, not conclusion): this suggests dynamic+short-route scenarios sit closer to the edge of what the generator's validity constraints allow, meaning the accepted scenarios in that cell may be systematically different/harder in ways unrelated to the family label itself. Worth cross-referencing against P1-P4 failure rates for dynamic-short specifically once failure trajectory work (item 3) begins.
- Overall verdict: category counts are genuinely balanced; scenario construction difficulty is not, concentrated in dynamic+short-route. Flag to Bug/Muzzammil as a modeling-relevant note, not a defect.


## Ledger Entry -- 2026-08-08 -- Manual failure trajectory review (item 3, COMPLETE for this batch)
- Stratified sample: 30 of planned 32 trajectories reviewed, 2/pilot/family average, drawn from Muzzammil's raw evaluation data (step_100000).
- Methodology note: first 3 scenarios were manually inspected by reading raw trajectory JSON directly. Remaining 27 were analyzed via a computational heuristic (cycle-length detection + distance-to-goal) run by Claude, with results reviewed and typed in by Simra rather than independently re-derived by eye for each one. Logged honestly as a weaker form of review than the first 3.
- RAW RESULT: 15/30 flagged as disagreeing with automated failure_label.
- CORRECTION: 8 of those 15 are a false disagreement from a naming mismatch -- automated "longer_repeated_loop" and manual "longer_loop" are the same category, named differently. Not a real disagreement.
- TRUE disagreement rate: 7/30 = 23.3%.
- PATTERN (high confidence): all 7 real disagreements are cases where the classifier said "two_cell_oscillation" (6) or "excessive_detour" (1), but manual review found a longer/messier cycle or diffuse aimless movement. Direction consistent across all 7 -- classifier appears to over-label ambiguous stuck patterns as clean two-cell oscillations.
- SECONDARY PATTERN (low confidence, small n): 6 of 7 real disagreements occur on 15x15 grids. Hypothesis only, needs larger sample.
- By pilot (true disagreements): P3=3, P4=3, P1=1, P2=0.
- SEPARATE FINDING: 28/30 reviewed episodes show progress-then-stuck; only 2/30 stuck-from-step-1. Most Phase B failures involve real initial progress that breaks down, not a failure to initiate any plan.
- ACTION FOR PHASE 2: recommend reviewing/tightening the two_cell_oscillation detector given the consistent one-directional bias found. Cross-referenced against item 2's dynamic+short-route generation-difficulty finding -- NOT correlated in this sample (only 2/7 disagreements were dynamic family).

## Ledger Entry -- 2026-08-08 -- Phase B learning curve inspection (item 4)
- Source: runs/rl_v3/phase_b/phase_b_summary.json learning_curve field (3 checkpoints per pilot: 25k/50k/100k interactions).
- SCOPE GAP: learning_curve only contains CPU/GPU utilization, environment_steps_per_second, interactions, training_wall_seconds, validation_episodes, validation_success_rate. NO reward, NO episode length, NO entropy fields exist in committed data. logs/ (TensorBoard) is git-ignored, not present in git history on any branch checked.
- Confirmed via grep that model.predict(..., deterministic=True) is used everywhere validation_success_rate is computed -- these are genuine deterministic-eval snapshots.
- ANSWERABLE:
  - Did any checkpoint outperform final? YES for 3 of 4 pilots.
    P1: 2/96 -> 1/96 -> 2/96 (final). Flat/noisy.
    P2: 2/96 -> 4/96 (peak@50k) -> 3/96 (final).
    P3: 1/96 -> 2/96 (peak@50k) -> 0/96 (final). Collapsed to zero.
    P4: 4/96 (peak@25k) -> 3/96 -> 1/96 (final). Monotonic decline.
  - Did P4 improve then degrade? YES, confirmed with exact numbers above.
- NOT ANSWERABLE from current data: training reward improvement, episode length reduction, success-in-training-vs-deterministic-eval gap, entropy collapse, early-stopped exploration. All require data not present in committed artifacts.
- ACTION: request TensorBoard logs / reward-episode-length CSVs from Muzzammil, same pattern as prior blockers.
- Headline finding for Bug/Muzzammil: 3 of 4 pilots peaked before their final checkpoint; P3 collapsed from 2/96 to 0/96. Worth asking whether best-checkpoint, not final-checkpoint, should be each pilot's reported result.

## Ledger Entry -- 2026-08-08 -- Phase A DDQN checkpoint received and hash-verified
- Muzzammil provided models/research/full/seed_011/02_dynamic_full_final.zip via commit d8d1fcf on branch rl-v3-c2-empty-multiscale.
- Independently pulled the file bytes from that commit and computed SHA-256 myself: 8ea28bdfe6d68c138de0128f80cd862064a71a0b6af56cf3b3c38f4cd2d13ad7.
- MATCHES exactly the hash Muzzammil reported.
- Caveat: checked evaluation/results/training_full_seed_011.json (this checkpoint's own training provenance record) for a pre-existing recorded hash of the checkpoint file itself. None exists -- only source_tree/sha256 and training_source_snapshot/sha256, which hash the codebase, not model weights. This verification confirms no corruption/tampering in transfer and matches Muzzammil's claim, but there was no independent pre-existing record to check the claim against. Recommend Phase 2 checkpoint releases record a hash at training time going forward.
- File size 452,270 bytes.
- UNBLOCKS: Phase A diagnostic regeneration.

## Ledger Entry -- 2026-08-08 -- Phase A regeneration CONFIRMED (item 1 now fully COMPLETE)
- Ran python -m rl_v3.run_phase_a with the checkpoint Muzzammil provided.
- git status showed 5 files 'modified'. Investigated with git diff -w (plain git diff is unreliable on this repo -- Windows CRLF vs committed LF makes every line appear changed even when identical).
- Both manifest JSONs: byte-identical with -w. Manifest generation is deterministic.
- Both diagnostic CSVs: every row pair differs ONLY in mean_decision_latency_ms (expected wall-clock variance, not research-relevant). ALL substantive fields identical: success, failure_label, decisions, path_cost, initial_astar_cost, path_cost_gap, repeated_cell_count, dynamic_event_count, collision/timeout/oscillation flags.
- phase_a_summary.json: git status confirmed clean, no change at all.
- CONCLUSION: Phase A diagnostic results are CONFIRMED fully reproducible. Item 1 (reproduce project state) is now COMPLETE for both Phase A and Phase B.
