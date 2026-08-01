# UAV Dynamic Routing Documentary - Evidence Manifest

Evidence cutoff: 2026-07-27 22:03 PKT (UTC+05:00)

Repository represented: `D:\UAV Dynamic Routing\uav-dynamic-routing`

Git HEAD at cutoff: `f57ae150068fbd7a00c1b37a8280cf60a5dd2ddc`

Working-tree state at cutoff: dirty, with 63 status entries. The uncommitted
expanded-study implementation is therefore part of the represented state and
cannot be reconstructed from the Git commit alone.

This manifest identifies the principal primary materials used by the interim
documentary. It is selective rather than exhaustive: the workspace contains
many generated checkpoints, progress files, tests, and manuscript build files.

| ID | Project material | What it establishes | Evidentiary status |
|---|---|---|---|
| E01 | `uav-routing-project-plan.md` | Original motivation, research question, planned grid, algorithm comparison, metrics, four-week division of work, and intended paper structure | Historical planning record |
| E02 | Git history from `9da9099` through `f57ae15` | Dated implementation sequence from initial structure through the v1/v2 paper split and evaluation fixes | Version-control record |
| E03 | `README.md` | Current project scope, environment contract, v1/v2 distinction, and full-pipeline entry points | Current documentation |
| E04 | `envs/ENVIRONMENT_SPEC.md` | Original shared coordinate, movement, cost, obstacle, and Week 3 dynamic-obstacle agreement | Historical/current contract record |
| E05 | `envs/grid_environment.py` | Shared classical graph, fixed/stochastic/moving obstacles, traversal penalties, reset behavior, and neighbor costs | Locked training-relevant implementation |
| E06 | `baselines/dijkstra.py` | Dijkstra implementation and visited-node accounting | Current implementation |
| E07 | `baselines/astar.py` | A* implementation and octile heuristic | Current expanded-study implementation |
| E08 | `baselines/dstar_lite.py` | Incremental D* Lite implementation and repair accounting | Current expanded-study implementation |
| E09 | `baselines/replanning.py` | Shared classical move-then-observe runner, replanning triggers, timing boundary, and adaptability events | Current expanded-study implementation |
| E10 | `rl_agent/q_learning.py` and `experiments/train_q_learning_static.py` | Early tabular Q-learning baseline, reward design, and static evaluation route | Historical implementation |
| E11 | `rl_agent/uav_env.py` | Current Gymnasium environment, 61-dimensional local observation, actions, rewards, potential shaping, dynamics timing, realism factors, and HER reward function | Locked training-relevant implementation |
| E12 | `rl_agent/double_dqn.py` | Double DQN online action selection and target-network evaluation | Locked training-relevant implementation |
| E13 | `rl_agent/safe_her_buffer.py` | Relabeled-goal terminal-flag correction used by HER training | Locked training-relevant implementation |
| E14 | `training_summary.md` | Detailed record of the early DQN+HER tuning and diagnostic phase, including computational bottlenecks, reward/Q-value failures, the 200k toy result, and seed-999 diagnostic | Historical diagnostic record; not final expanded-study evidence |
| E15 | `methodology_draft.md`, `discussion_limitations.md`, `paper_outline.md`, and `references_plan.md` | How the early Week 3 result was interpreted and converted into the first paper narrative | Historical manuscript-planning record |
| E16 | `paper_latex_v1/` | Archived original single-baseline paper, including the 40-scenario static and 50-scenario dynamic results | Historical paper artifact; not the final expanded study |
| E17 | `research_improvement_plan.md` | Identified weaknesses and the ten-workstream expansion toward a multi-seed, multi-baseline, reproducible study | Improvement decision record |
| E18 | `configs/research_experiments.json` | Exact five policy seeds, layout seed, model hyperparameters, three-stage training budget, and six controlled ablations | Locked experimental configuration |
| E19 | `experiments/train_multiseed.py` | Checkpointed training implementation, stage transitions, seed control, model selection, metadata, and resumability | Locked training-relevant implementation |
| E20 | `evaluation/manifests/week3_dynamic_50.json` | Stable 50-scenario pilot with three fixed toggle cells and the corrected `post_move_observed` label | Persisted pilot manifest |
| E21 | `evaluation/manifests/benchmark_v2.json` | The 310-scenario expanded benchmark across 21 named splits and four grid scales | Persisted expanded-study manifest |
| E22 | `experiments/generate_benchmark_suite.py` and `evaluation/scenario_suite.py` | How exact blocked cells, starts/goals, dynamics, costs, no-fly modes, and noise are generated and reconstructed | Current benchmark implementation |
| E23 | `docs/EXPERIMENT_PROTOCOL.md` | Matching, timing, statistics, adaptability, reproducibility, and reward-shaping rules | Current protocol |
| E24 | `docs/ARTIFACT_MAP.md` | Command-to-manifest-to-raw-data-to-generated-output mapping | Current reproducibility map |
| E25 | `docs/RESEARCH_EXECUTION_STATUS.md` | Workstream status and the explicit distinction between implemented features and generated evidence | Interim status record |
| E26 | `evaluation/results/classical_dynamic_raw.csv`, `classical_dynamic_summary.json`, `classical_dynamic_environment.json`, and `classical_dynamic_summary.md` | Verified repeated Dijkstra/A* pilot: 50 scenarios, 10 repetitions, success, cost, timing, and node expansions | Completed pilot evidence |
| E27 | `evaluation/week3_dynamic_baseline_results.csv`, `evaluation/week3_rl_results.csv`, and `evaluation/week3_paired_stats_summary.md` | Early single-policy Week 3 comparison: 50/50 Dijkstra success, 49/50 RL success, path-cost and timing tests | Historical result; excluded from final expanded-study claims |
| E28 | `logs/eval_dqn_vs_dijkstra.csv` and `paper_latex_v1/sections/results.tex` | Early 40-pair static comparison and its paper interpretation | Historical result |
| E29 | `evaluation/results/training_full_seed_011.json` through `training_full_seed_055.json` | Five complete corrected full-method seeds, 500,000 steps each, stage timings, environment metadata, package versions, and provenance digest | Completed and provenance-verified training evidence |
| E30 | `evaluation/results/training_dqn_seed_011.json` through `training_dqn_seed_044.json` | Four complete corrected vanilla-DQN ablation seeds at the cutoff | Completed and provenance-verified training evidence |
| E31 | `evaluation/results/superseded_pre_observation_fix/` and `models/research/superseded_pre_observation_fix/` | Earlier research runs intentionally quarantined after the information-timing correction | Superseded evidence; not admissible for final claims |
| E32 | `evaluation/results/training_source_snapshot.json` | Locked seven-file training source digest `00a4ff...f39d` | Current provenance control |
| E33 | `scripts/capture_training_provenance.py` | Exact list of training-relevant files and the hard failure when their hashes change | Current provenance implementation |
| E34 | `evaluation/check_artifact_integrity.py` | Exact Cartesian-product, uniqueness, parent-route, smoke-checkpoint, and manifest-hash gates | Implemented final integrity gate; final report not yet present |
| E35 | `evaluation/statistical_tests.py`, `evaluation/analyze_research_results.py`, and `evaluation/analyze_adaptability.py` | Seed-aware summaries, McNemar and Wilcoxon tests, effect sizes, Holm correction, and event-level analysis | Implemented analysis; final outputs pending |
| E36 | `scripts/run_full_research.py` | Ordered one-command pipeline from tests and manifests through evaluation, statistics, plots, tables, manuscript fragments, compilation, and status finalization | Current orchestration implementation |
| E37 | `paper_latex_v2/` | Current expanded manuscript design and placeholder-gated generated fragments | Interim manuscript source; existing PDF is explicitly non-final |
| E38 | `runs/research/multiseed_full_post_move.stdout.log` | Completion timings for the corrected five-seed full method | Completed training log |
| E39 | `runs/research/ablation_queue_status.json` | Local queue migrated to cloud; four DQN seeds complete; seed 55 listed as in progress; local compute stopped | Cutoff status snapshot |
| E40 | `runs/research/cloud_dispatch.json` | Free Google Colab T4 selection, benchmark throughput, Drive paths, source digest, cloud task, and local-process stop record | Cutoff cloud dispatch record |
| E41 | `notebooks/cloud_research_runner.ipynb` and `scripts/cloud_colab_worker.py` | Drive mount, bundle extraction, free-runtime benchmark, resumable seed loop, restore/sync behavior, status writes, and time-budget exit | Current cloud workflow implementation |
| E42 | `tests/` plus the 2026-07-27 local test run | Unit/integration coverage for planners, dynamics, realism, timing, metadata, adaptability, and integrity; 58 tests passed | Verified implementation evidence at cutoff |
| E43 | `paper_latex_v2/references.bib` | Bibliographic metadata already selected by the project for graph search, RL, reward shaping, statistics, and UAV context | Current literature record |

## Hashes and exact identifiers

- Training-source aggregate SHA-256:
  `00a4ff215b3d31f7be2a42d62f7467d19beacd3f3e78c8684efe2d233412f39d`
- Expanded benchmark SHA-256:
  `296196e774e5aa7f45b988f0321355a85f4dcbe21d1c727842cc818ebde0e0b5`
- Week 3 pilot manifest file SHA-256:
  `0c6dee96a925d62498f7be63b050a5912b663d1c43f8fc9ba515a25ee3ea12f7`
- Classical pilot embedded manifest SHA-256 recorded by its environment metadata:
  `d60264a1ed0cc3674c7cbbb772f1d110adcdfa0448a932b167bb47d75888475e`
- Cloud completed-artifacts bundle SHA-256 recorded at dispatch:
  `7dd5032e7d3ddd1d9ea9f68b1ef52a0103a49c00b66fa16d0ade276bc63a85ab`
- Git commit:
  `f57ae150068fbd7a00c1b37a8280cf60a5dd2ddc`

## Important evidence boundary

The repository intentionally contains results from different methodological
eras. The documentary uses those early results to explain learning and project
evolution, but it does not combine them numerically with the corrected
expanded study. Final claims require the final v2 raw Cartesian products and a
passing `evaluation/results/integrity_report.json`.
