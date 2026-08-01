# Expanded reproducible paper package

This is the current manuscript. It supersedes the single-seed Dijkstra/A*/RL
revision in `paper_latex_v1/` and the earlier generated PDF in this directory.

The manuscript now describes:

- five independent RL training seeds;
- Dijkstra, A*, and D* Lite classical baselines;
- persisted ID/OOD, scaling, and realism benchmark splits;
- repeated route timings and machine manifests;
- controlled RL ablations;
- event-level adaptability metrics; and
- generated tables, figures, and evidence-bound prose.

The final release is now bounded by the repository status file
`docs/RESEARCH_EXECUTION_STATUS.md`. Known loopholes and follow-up experiments
are tracked in `docs/LOOPHOLE_REGISTER.md`; those boundaries should be kept in
sync with the discussion and conclusion before any new submission.

The expanded experiment queue has completed and
`evaluation/results/integrity_report.json` reports `status: passed`. Treat
`output/pdf/uav_dynamic_routing_research_paper.pdf` as the release PDF. The
in-directory `main.pdf` is a local build artifact and should be regenerated
before use if the LaTeX source or generated fragments change.

## Regenerate evidence

```bash
python scripts/run_full_research.py --variants full
```

Use `--train` on a clean checkout when checkpoints do not yet exist.

## Compile

From the repository root, the final pipeline uses the checksum-pinned
workspace-local Tectonic compiler
and refuses to publish a PDF unless artifact integrity passes and all generated
fragments have replaced their placeholders:

```bash
python scripts/compile_paper.py
```

For source/layout debugging when intentionally inspecting incomplete edits:

```bash
python scripts/compile_paper.py --allow-incomplete
```

Release output is written to
`output/pdf/uav_dynamic_routing_research_paper.pdf`, accompanied by a build
manifest containing the engine version, integrity-report digest, PDF digest,
and page count.
