# LaTeX Paper Package — v1 (archived)

This is the **original submitted version** of the paper, kept for reference. It is no longer the current version — see [`paper_latex_v2/`](../paper_latex_v2) for the active draft.

**Scope of this version:** a single classical baseline (naive Dijkstra replanning) compared against the DQN+HER policy, on the static 40-scenario and dynamic 50-scenario benchmarks.

**Why it's archived rather than deleted:** v2 revised the compute-time conclusions after adding a second classical baseline (A*), so the two versions report meaningfully different findings on that axis. Keeping v1 makes that revision auditable rather than silently overwritten.

## Compile from this folder with:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Before submission (if this version is ever needed again)

- Replace placeholder authors/institution.
- Complete BibTeX metadata in `references.bib`.
- Confirm target venue formatting requirements.
