# Research Improvement Plan

This roadmap expands the current UAV dynamic-routing study without changing its
research question. The work is ordered to first make the existing findings
fully reproducible, then test their robustness, scope, and practical relevance.

## 1. Reproducibility

Add the artifacts needed to independently reproduce every reported result.

- Add the A* implementation and its matched dynamic-replanning runner.
- Commit raw per-scenario A* evaluation results.
- Add scripts to generate every paper table and figure from the raw data.
- Provide one documented command (or Makefile target) that runs the full
  evaluation and regenerates the paper artifacts from a clean checkout.

**Completion criterion:** every numerical claim, plot, and table in the paper
can be regenerated from committed code and data.

## 2. Measurement Rigor

Make the computational measurements auditable and robust.

- Record hardware, operating system, Python version, package versions, and
  CPU/GPU configuration with every experiment.
- Repeat timing experiments multiple times and report mean, median, standard
  deviation, and number of repetitions.
- Preserve per-run timing data instead of only aggregate summaries.
- Keep route-level compute time as the primary measure, while also reporting
  per-decision latency where useful.

**Completion criterion:** timing results include a documented protocol,
environment manifest, repeated measurements, and uncertainty estimates.

## 3. Multi-Seed Reinforcement Learning Evaluation

Measure training variance rather than relying on one policy seed.

- Train the complete DQN+HER curriculum with 5--10 independent seeds.
- Save the checkpoint and evaluation outputs for every seed.
- Evaluate every seed on the same static, dynamic, and generalization scenario
  sets.
- Report seed-level values plus means and confidence intervals for success,
  path cost, adaptability, and compute time.

**Completion criterion:** the paper reports performance distributions over
independent trained policies, not only one seed-42 policy.

## 4. Generalization Benchmark

Test whether the learned policy generalizes beyond its fixed training layout.

- Define benchmark splits for seen layouts with unseen start/goal pairs,
  unseen layouts at the same obstacle density, denser unseen layouts, new
  dynamic-obstacle locations, and changed toggle periods.
- Generate and persist scenario manifests with stable scenario IDs.
- Run all planning methods on the identical scenario IDs within every split.
- Separate in-distribution performance from out-of-distribution performance in
  the results and discussion.

**Completion criterion:** generalization is measured across multiple unseen
layouts and dynamics, rather than with a single informal probe.

## 5. Scaling Study

Determine how the methods behave as routing problems grow.

- Repeat the matched protocol on 15x15, 30x30, 50x50, and, if resources
  permit, 100x100 grids.
- Keep movement rules and scenario-generation principles consistent across
  scales.
- Report success, path-cost gap, compute time, replans, and node expansions at
  each scale.
- Include plots showing how the relative performance of RL and each classical
  method changes with grid size.

**Completion criterion:** the paper can support or reject the claim that RL's
deployment-time trade-off improves at larger scale.

## 6. Stronger Classical Baselines

Compare RL with both simple and modern replanning methods.

- Retain naive Dijkstra as an interpretable full-replanning reference.
- Retain A* as the heuristic-pruned shortest-path baseline.
- Add D* Lite or Lifelong Planning A* as an incremental dynamic-replanning
  baseline.
- Ensure every planner shares the same graph, obstacle schedule, scenario IDs,
  movement costs, replan triggers, and timing definition.

**Completion criterion:** the study distinguishes naive search, heuristic
search, and incremental replanning rather than treating Dijkstra as the sole
classical alternative.

## 7. More Realistic Scenario Families

Expand the controlled grid benchmark toward realistic UAV-routing conditions.

- Sweep static-obstacle density and dynamic-obstacle frequency.
- Add stochastic and moving obstacles.
- Add wind or spatial energy-cost maps.
- Add no-fly zones as configurable traversal penalties or hard restrictions.
- Add partial observability and optional sensor noise.
- Make each factor independently configurable for controlled experiments.

**Completion criterion:** the benchmark supports targeted experiments that
isolate the effect of each realism factor.

## 8. Reinforcement Learning Ablations

Establish which parts of the RL design actually improve performance.

- Compare DQN and Double DQN.
- Compare HER enabled versus disabled.
- Compare potential-based shaping enabled versus disabled.
- Compare curriculum learning enabled versus disabled.
- Compare local observation against full-grid observation where feasible.
- Compare dynamic fine-tuning against training on dynamic conditions from
  scratch.

**Completion criterion:** every major RL design choice in the methodology has
controlled experimental evidence.

## 9. Adaptability Analysis

Turn adaptability into a measured outcome instead of a qualitative claim.

- Define recovery time after an obstacle change.
- Measure extra path cost attributable to each change event.
- Measure post-change success rate and route-disruption duration.
- Store event-level metrics in a dedicated CSV.
- Plot and statistically compare adaptability metrics across methods.

**Completion criterion:** adaptability has formal definitions, raw data,
plots, and statistical comparisons.

## 10. Paper Revision and Artifact Release

Revise the paper only after the expanded experiment suite is complete.

- Regenerate every figure and table from final raw data.
- Update methodology with exact benchmark, seed, and timing protocols.
- Report uncertainty, generalization splits, ablations, and scale results.
- Preserve explicit limitations where evidence remains incomplete.
- Add an artifact appendix mapping every result to its command, source file,
  raw data, and generated figure.

**Completion criterion:** the final paper is traceable, reproducible, and
clear about both supported findings and limitations.

## Suggested Timeline

| Weeks | Focus |
|---|---|
| 1--2 | Reproducibility and measurement rigor |
| 3--4 | Multi-seed RL training and evaluation |
| 5--6 | Generalization, scale study, and stronger baselines |
| 7--8 | Realistic scenarios, RL ablations, and adaptability metrics |
| 9--10 | Paper revision, artifact release, and final replication pass |

## Recommended Execution Order

Complete reproducibility, measurement rigor, and multi-seed evaluation before
expanding the benchmark. Those three steps establish confidence in the current
findings; the later studies determine where those findings do and do not
generalize.

## Final Study Outcome

The ten-workstream expansion has been executed in the current repository and is
summarized in `docs/RESEARCH_EXECUTION_STATUS.md`. The result is intentionally
not an RL-superiority claim: A*, Dijkstra, and D* Lite remain stronger overall
under the shared graph-routing contract. The final contribution is a
reproducible, multi-seed boundary study that shows where learned routing works,
where it fails, and which conclusions survive stronger baselines, distribution
shift, scaling, ablations, and event-level adaptability analysis.

Remaining loopholes are tracked in `docs/LOOPHOLE_REGISTER.md`. Items marked
"future experiment required" are outside the current artifact set and should be
treated as the next research phase rather than as missing analysis from the
completed benchmark.
