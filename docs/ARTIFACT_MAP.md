# Artifact map

| Result family | Command | Scenario source | Raw data | Generated artifact |
|---|---|---|---|---|
| Classical dynamic pilot | `python scripts/reproduce_classical.py` | `evaluation/manifests/week3_dynamic_50.json` | `evaluation/results/classical_dynamic_raw.csv` | `evaluation/results/classical_dynamic_summary.md` |
| Expanded classical suite | `python experiments/run_classical_suite.py` | `evaluation/manifests/benchmark_v2.json` | `evaluation/results/classical_suite_raw.csv` | `evaluation/results/research_summary.csv` |
| Multi-seed RL | `python experiments/evaluate_multiseed.py` | `evaluation/manifests/benchmark_v2.json` | `evaluation/results/rl_<variant>_raw.csv` | `evaluation/results/research_summary.csv` |
| Timing distributions | `python evaluation/summarize_timing_distributions.py --rl ...` | All expanded-suite splits | Classical and RL raw repetitions | `evaluation/results/timing_distributions.csv` |
| Adaptability | `python evaluation/analyze_adaptability.py --rl ...` | Dynamic splits in benchmark v2 | `evaluation/results/*adaptability_events.csv` | `adaptability_summary.csv`, paired tests, paper table, and `paper_latex_v2/figures/adaptability_*.png` |
| Scaling | `python scripts/run_full_research.py` | `scale_15`, `scale_30`, `scale_50`, `scale_100` splits | Classical and RL suite CSVs | `paper_latex_v2/figures/scaling_*.png` |
| Generalization | `python scripts/run_full_research.py` | ID/OOD splits in benchmark v2 | Classical and RL suite CSVs | `paper_latex_v2/figures/generalization_success.png` |
| Ablations | `python scripts/run_full_research.py --train --variants ...` | Benchmark v2 | One RL CSV per variant | Final ablation table/plot |

The final paper references this evidence chain through its artifact appendix.
See `docs/RESEARCH_EXECUTION_STATUS.md` for the final release gate summary and
`docs/LOOPHOLE_REGISTER.md` for remaining claim boundaries.
