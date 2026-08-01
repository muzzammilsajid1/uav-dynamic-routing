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

Do not treat the existing `main.pdf` as current until the expanded experiment
queue finishes and the source is recompiled. Generated fragments under
`generated/` are placeholders until `evaluation/results/integrity_report.json`
reports `status: passed`.

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

For source/layout debugging before the experiment queue finishes:

```bash
python scripts/compile_paper.py --allow-incomplete
```

Release output is written to
`output/pdf/uav_dynamic_routing_research_paper.pdf`, accompanied by a build
manifest containing the engine version, integrity-report digest, PDF digest,
and page count.
