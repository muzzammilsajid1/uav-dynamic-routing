# LaTeX Paper Package — v2 (current)

This is the **current version** of the paper. For the original submitted version, see [`paper_latex_v1/`](../paper_latex_v1).

## What changed from v1

- **Added A\* as a second classical baseline**, alongside naive Dijkstra replanning. A\* and Dijkstra are both optimal on this graph, so they produce identical path costs (confirmed empirically); A\* resolves the same replanning events roughly 16× faster than Dijkstra (Wilcoxon p = 3.6×10⁻¹⁵). This revises the original compute-time conclusion: RL's per-step compute-time cost was not competitive with either classical planner once a properly pruned one was included.
- **Added a full DQN + HER hyperparameter table** (network architecture, learning rate, buffer size, discount factor, curriculum phases, reward shaping) for reproducibility.
- **Re-measured static-evaluation compute time directly** (the original run did not persist raw per-scenario timing data) and independently reconfirmed the original success-rate and path-cost figures.
- **Added 95% Wilson confidence intervals** on the small-sample success-rate comparisons (50 scenarios).
- **Added an informal generalization probe** (94% success on an unseen grid layout, seed 999) as a directional signal, clearly flagged as non-rigorous.
- **Made the benchmark's scale an explicit, stated methodological choice** (small grid, zero static obstacle density in the dynamic eval) rather than an unstated limitation.
- **Named single-seed RL training and the absence of ablations as open limitations**, rather than omitting them.

## Compile from this folder with:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Before submission

- Replace placeholder authors/institution.
- Complete BibTeX metadata in `references.bib`.
- Confirm target venue formatting requirements.

## Related files elsewhere in the repo

- `evaluation/compare_dqn_dijkstra_timed.py` — instrumented re-run of the static evaluation with compute-time measurement, used to produce Table II in this version.
- `evaluation/week3_astar_baseline_results.csv` — raw per-scenario A* results used for the new baseline comparisons.
- `logs/eval_dqn_vs_dijkstra_timed.csv`, `logs/eval_dqn_vs_dijkstra_v2.csv` — re-run static evaluation outputs.
