# Loophole Register And Mitigation Plan

Last updated: 2026-08-01

This register tracks the main weaknesses a reviewer could use to challenge the
UAV dynamic-routing study. It separates issues already mitigated by the final
benchmark from issues that are only bounded by honest reporting and issues
that require new experiments.

## Status Key

| Status | Meaning |
|---|---|
| Closed in current study | The final repo contains evidence or protocol changes that address the issue for the current claim scope. |
| Bounded in paper | The issue is real, but the manuscript now limits claims so the weakness does not invalidate the stated contribution. |
| Future experiment required | The issue cannot be resolved from the existing artifacts. New data, models, environments, or hardware are required. |

## Critical Claim Boundaries

| Loophole | Current mitigation | Status | Next action |
|---|---|---|---|
| The work can be mistaken for a real flight-control study. | The paper frames the contribution as dynamic graph routing, not end-to-end UAV autonomy. The limitations state that vehicle dynamics, altitude, turn radius, acceleration, localization drift, communications, and multi-UAV interactions are outside scope. | Bounded in paper | Add a continuous-dynamics simulator study before making flight-control claims. |
| The title and UAV language may overreach the evidence. | The current title includes distribution shift and comparison language rather than superiority language. The abstract says all claims are traceable to raw artifacts. | Bounded in paper | Keep all external summaries tied to "grid routing" or "routing under controlled graph changes." |
| RL does not outperform classical planners overall. | The final conclusions explicitly say classical planners are strongest and that the contribution is a reproducible map of where learned routing works and fails. | Closed in current study | Do not claim RL superiority. Use the result as a negative or boundary-finding contribution. |
| The benchmark is synthetic. | The paper reports persisted synthetic splits and does not claim real-world deployment validity. | Future experiment required | Add GIS or airspace-map scenarios, recorded obstacle traces, or an external benchmark. |
| Real UAV constraints are simplified. | Wind, energy, no-fly zones, sensing, stochastic obstacles, and moving obstacles are included only as controlled graph factors. | Future experiment required | Add flight dynamics, battery model, altitude layers, clearance buffers, and vehicle limits. |
| Dynamic obstacles are simplified grid events. | The expanded suite includes fixed toggles, stochastic toggles, moving obstacles, changed periods, and new obstacle locations. | Bounded in paper | Add realistic moving-agent trajectories and perception-limited obstacle detection. |
| The occupied-cell toggle rule is debatable. | The methodology states that departure from a newly blocked occupied cell is allowed and names it as a legacy contract shared by all methods. | Bounded in paper | Compare escape, collision, and safety-buffer semantics in a new benchmark split. |
| Diagonal corner cutting may be unsafe near obstacles. | The README and methodology explicitly declare diagonal corner cutting as part of the environment contract. | Bounded in paper | Add a no-corner-cutting split and report whether conclusions change. |
| No real-world maps or operational data are used. | The paper does not generalize beyond controlled graph-routing environments. | Future experiment required | Add terrain, urban, corridor, or no-fly-zone map imports. |
| No multi-UAV interactions are modeled. | The limitations exclude multi-UAV interactions. | Future experiment required | Add traffic-aware and conflict-resolution scenarios. |

## Benchmark And Fairness Risks

| Loophole | Current mitigation | Status | Next action |
|---|---|---|---|
| Classical and RL methods process information differently. | All methods share the same movement graph, costs, scenario IDs, obstacle schedules, and post-move-observed information timing. | Closed in current study | For future work, add planners with partial-map assumptions to match local sensing more tightly. |
| Classical planners have an optimality advantage on known graphs. | The paper treats this as a structural property, not a bias to hide. A*, Dijkstra, and D* Lite are included as strong baselines. | Closed in current study | Keep classical optimality as the reference point for graph-routing claims. |
| The scenario suite is project-designed. | Scenarios are persisted with stable IDs and raw rows carry the benchmark digest. | Bounded in paper | Add an external benchmark or independent scenario generator. |
| The 310 scenarios are not a random sample of all UAV domains. | The discussion states the suite is broad within the grid model but not representative of all operational environments. | Bounded in paper | Define a sampling frame before making population-level claims. |
| The full-grid observation ablation cannot scale beyond 15x15. | The experimental setup records that this is a structural input-dimension exclusion, not a performance-based omission. | Closed in current study | Use convolutional or graph-network policies for variable-size full-map inputs. |
| Scaling may be unfair to a policy trained for small grids. | The results are framed as the boundary of the trained policy, not proof that all RL cannot scale. | Bounded in paper | Train scale-conditioned or variable-size policies across grid sizes. |
| RL may be undertuned relative to classical baselines. | The paper fixes a transparent DDQN+HER configuration and reports ablations, but does not claim best possible RL. | Bounded in paper | Run systematic hyperparameter optimization or compare PPO, SAC-discrete variants, GNN policies, and imitation-learning hybrids. |

## Evidence And Statistics Risks

| Loophole | Current mitigation | Status | Next action |
|---|---|---|---|
| One RL seed would be too weak. | The final study uses five independent policy seeds for each of seven RL configurations. | Closed in current study | Increase to 10-30 seeds for unstable variants if compute is available. |
| Five seeds still leave wide uncertainty. | Student-t confidence intervals are reported across independent policy seeds. | Bounded in paper | Add more seeds for final archival publication. |
| Timing could be implementation-specific. | The protocol records route-level and per-decision timings, repetitions, runtime class, package versions, and timing boundaries. | Bounded in paper | Re-run optimized C++/embedded planner and inference implementations on target hardware. |
| Cross-variant timing provenance was historically incomplete. | The paper discloses that historical per-variant machine manifests were not retained separately and says the current pipeline prevents recurrence. | Bounded in paper | Preserve immutable per-variant timing manifests for any new run. |
| Failed RL routes produce missing path-cost gaps. | Success is reported separately and path-cost tests are restricted to jointly successful routes. | Closed in current study | Add failure-penalized utility metrics for operational decision-making. |
| Recovery metrics can be misread as route success. | Results distinguish recovery criterion completion from post-change route success. | Closed in current study | Add safety and mission-risk metrics around each change event. |
| Scenario and timing repetitions could create pseudoreplication. | Timing repetitions are collapsed within seed/scenario pairs before statistical tests. | Closed in current study | Keep this rule in every future analysis script. |

## Reproducibility And Repository Risks

| Loophole | Current mitigation | Status | Next action |
|---|---|---|---|
| Checkpoints are not tracked in git. | Training metadata, source digests, raw results, manifests, and one-command rebuild scripts are tracked; checkpoints are regenerated by the long pipeline. | Bounded in paper | Publish checkpoint bundles separately with hashes if distribution size permits. |
| Stale interim documentary artifacts can confuse readers. | The final status doc identifies the integrity-gated release PDF as current. | Partially bounded | Replace the interim documentary with a definitive edition or mark output copies as historical. |
| Local test setup can fail if a virtual environment points at a missing base Python. | `requirements.txt` includes pytest and the README documents the standard test command. This local workspace currently has a broken `venv` launcher. | Partially bounded | Recreate the venv with an installed Python, then run `python -m pytest -q` before the next commit. |
| Generated output under `output/pdf/` is ignored by git. | The tracked source, generated fragments, raw results, and build manifest logic reproduce the release output. | Bounded in paper | Store release PDFs externally or attach them to tagged releases with recorded hashes. |

## Immediate Cleanup Checklist

- Keep `docs/RESEARCH_EXECUTION_STATUS.md` as the authoritative final status.
- Treat `output/pdf/UAV_Dynamic_Routing_Pending_Claims.md` and the interim
  documentary as historical unless regenerated from the final artifacts.
- Do not use the old July 2026 paper PDFs for final claims.
- Use the final paper hash
  `2c52ae6714504a9c88e04351a5bfa118d12182c048bf2509d39c9e8b2a02049c`
  when identifying the current release.
- Recreate the local virtual environment if `venv\Scripts\python.exe` reports
  a missing base interpreter.

