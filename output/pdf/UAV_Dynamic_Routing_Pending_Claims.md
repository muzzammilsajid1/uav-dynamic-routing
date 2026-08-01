# UAV Dynamic Routing Documentary - Claims Requiring Final Evidence

Evidence cutoff: 2026-07-27 22:03 PKT (UTC+05:00)

This file is the update checklist for the definitive edition. Nothing below
should be converted into a completed claim until the listed artifact exists,
passes integrity checks, and agrees with the manuscript tables and figures.

## Training and provenance

- Complete and restore the final metadata/checkpoint set for `dqn:seed055`.
- Complete all five seeds for `no_her`, `no_shaping`, `no_curriculum`,
  `full_observation`, and `dynamic_from_scratch`.
- Confirm every non-smoke training metadata file contains the locked digest
  `00a4ff215b3d31f7be2a42d62f7467d19beacd3f3e78c8684efe2d233412f39d`.
- Confirm each standard variant contains static, dynamic-mild, and
  dynamic-full stages, while dynamic-from-scratch contains the configured
  500,000-step dynamic-full stage.
- Reconcile Drive backups and local restored artifacts without mixing
  superseded pre-observation-fix checkpoints.

## Final route-level evidence

- Generate `classical_suite_raw.csv` for Dijkstra, A*, and D* Lite on all
  310 benchmark scenarios and every configured timing repetition.
- Generate one `rl_<variant>_raw.csv` file for the complete method and each
  ablation across all five policy seeds, 310 scenarios, and all repetitions.
- Verify exact run IDs, row counts, uniqueness, benchmark hashes, and
  seed-by-scenario-by-repetition Cartesian coverage.
- Determine final success, cost, route-level compute, per-decision latency,
  and classical node-expansion results for every benchmark split.

## Research questions that remain unanswered

- Whether the full Double DQN+HER method is more reliable than any controlled
  ablation across independent training seeds.
- Whether Double DQN materially improves over vanilla DQN.
- Whether HER, potential shaping, and curriculum each contribute measurable
  gains under the matched final protocol.
- Whether local observation is more robust or efficient than full-grid
  observation on the 15x15 conditions.
- Whether dynamic fine-tuning through the curriculum outperforms equal-budget
  training on the full dynamic condition from scratch.
- Whether RL route-level decision cost crosses below Dijkstra, A*, or D* Lite
  as grid size grows from 15 to 100.
- Whether any speed advantage, if observed, survives the accompanying changes
  in success and path quality.
- How policy performance shifts on held-out endpoints, unseen layouts, denser
  layouts, new obstacle locations, and changed toggle periods.
- How all methods respond to stochastic obstacles, moving obstacles, wind or
  energy penalties, hard and penalized no-fly zones, and sensor noise.
- Which method recovers most often and most quickly after a matched change
  event.

## Statistical outputs

- Generate seed-level and aggregate confidence intervals without treating
  scenarios or timing repetitions as independent trained policies.
- Run exact paired McNemar tests for success.
- Run paired Wilcoxon signed-rank tests for cost and compute time.
- Report rank-biserial effect sizes and Holm-adjusted p-values.
- Restrict paired path-cost tests to jointly successful routes while retaining
  all routes for success and computation analyses.
- Generate the matched change-event statistical artifact using scenario,
  change step, and changed-cell signature.
- Confirm that missing recovery times remain missing and are not silently
  dropped.

## Adaptability outputs

- Generate classical and RL event files with exact parent-route coverage.
- Report optimal-cost shock, positive extra optimal cost, recovery steps,
  unrecovered-event counts, post-change success, reaction time, and classical
  node expansions.
- Replace every provisional description of adaptability with measured values.

## Paper and artifact release

- Produce a passing `evaluation/results/integrity_report.json`.
- Regenerate `research_summary.csv`, timing distributions, statistical tests,
  adaptability summaries, and all paper-ready tables and figures.
- Replace every placeholder in `paper_latex_v2/generated/`.
- Regenerate abstract, result, discussion, conclusion, and contribution claims
  only from the final generated artifacts.
- Compile the integrity-gated release PDF to
  `output/pdf/uav_dynamic_routing_research_paper.pdf`.
- Render and visually inspect every release-paper page.
- Verify PDF hash, compiler version, integrity-report hash, and page count in
  the build manifest.
- Update this documentary: change its subtitle from interim to definitive,
  replace the cutoff snapshot, move completed items out of this list, insert
  final plots and tables, and revise conditional contribution language.

## Statements that must remain conditional until then

- Any assertion that RL "outperforms" Dijkstra, A*, or D* Lite.
- Any assertion of a scaling crossover.
- Any claim that a particular RL component is necessary or beneficial.
- Any claim of robustness to distribution shift or realistic disturbances.
- Any claim of statistically significant superiority.
- Any generalization from the discrete grid to physical UAV flight.
