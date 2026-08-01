# Research execution status

Finalized at `2026-08-01T09:47:49.211216+00:00` after the integrity-gated artifact and PDF pipeline.

| Workstream | Final evidence | Status |
|---|---|---|
| Reproducibility | Stable manifests, raw repetitions, environment/source provenance, one-command rebuild | Complete |
| Measurement rigor | Route and decision timings, repeated raw measurements, uncertainty estimates | Complete |
| Multi-seed RL | Five independently trained checkpoints and seed-level outputs for every variant | Complete |
| Generalization | Held-out pairs, unseen layouts/densities, locations, and periods | Complete |
| Scaling | Matched 15, 30, 50, and 100 cell grids | Complete |
| Strong baselines | Dijkstra, A*, and incremental D* Lite | Complete |
| Realistic scenarios | Density, stochastic/moving obstacles, energy, no-fly, and sensing sweeps | Complete |
| RL ablations | DQN, HER, shaping, curriculum, observation, and dynamic-scratch controls | Complete |
| Adaptability | Event CSVs, seed-aware summaries, plots, and paired tests | Complete |
| Paper revision | Generated tables/figures/prose and verified release PDF | Complete |

## Release gates

- Artifact integrity: `passed`
- Benchmark scenarios: `310`
- Timing repetitions: `10`
- PDF pages: `7`
- PDF SHA-256: `2c52ae6714504a9c88e04351a5bfa118d12182c048bf2509d39c9e8b2a02049c`
- Compiler: `Tectonic 0.16.9`
