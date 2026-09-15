# PyVBMC 1.5: remaining work and scope

Updated 2026-09-14. These lists describe scope, not priority or execution
order; independent workstreams can be picked up in any order. Inclusion in
scope does not settle an implementation design or launch a campaign.
Completed 1.5 work is not listed here: the
[roadmap](plans/modernization-roadmap.md) retains it, and each item's plan
records its execution.

## In scope for 1.5

- [ ] **Efficiency of existing noisy acquisitions.** Improve sieve/search,
  importance sampling, criterion-evaluation costs and GP-update costs.
  Separate changes to search behavior from computational optimizations;
  assess them on noisy synthetic and real-data targets against the promoted
  reference. Choose seed coverage for the experiment; a full reference-sized
  campaign is unnecessary. **Designing new acquisition functions or criteria
  is excluded.** Start from the
  [investigation](2026-09-08-noisy-acquisitions.md) and
  [search analysis](results/2026-09-09-acquisition-search-analysis.md).
  The [efficiency plan](plans/noisy-acquisition-efficiency.md) tracks the
  guarded standard-VIQR sum optimization and the GP-kernel reuse, both
  integrated into `dev-next` and validated (the
  [production validation](results/2026-09-13-viqr-kernel-production.md)
  reports a 1.205x late-sieve speedup and 18/18 exact stored replays).
  Remaining possibilities: adaptive sieve size, integration budgets,
  multistart L-BFGS-B, more efficient integration of the same VIQR
  criterion (shared-weight Bayesian quadrature against mixture-stratified
  randomized quasi-Monte Carlo), and a controlled whole-VBMC timing
  comparison of the combined changes, which replay wall times do not
  establish; see the
  [pickup point](plans/noisy-acquisition-efficiency.md#pickup-point-and-deferred-measurements).
  Keep standard VIQR (`loss="iqr"`); the retained `iqr_reduction`
  formulation preserves the criterion for fixed samples and weights but
  can change numerical search behavior and needs end-to-end validation
  before any use. IMIQR cache reuse is optional cleanup.

- [ ] **S-VBMC ELBO debiasing (the optimism note's Phase 2).** Decide the
  reported estimator, fallback and interface from the cross-run honest
  estimator's results on the full run pools. The estimator
  (`dev/scripts/svbmc_honest_elbo.py`) scores every cell of the stacking
  comparison and is prototyped on the pilot pool: the
  [pilot report](results/2026-09-14-svbmc-honest-elbo-pilot.md) records
  the checks, a systematic cross-run under-prediction on two noisy
  conditions, and that at three runs the existing component-median cap is
  closer to the reference than the honest estimate on four of five
  conditions. The full pools are scored (2026-09-15; the
  [stage D report](results/2026-09-15-svbmc-pool-comparison.md),
  sections "Phase 2" and "Weight-aware caps"): the honest estimate is
  within 0.04 nats on the noiseless controls and Rosenbrock, under-
  predicts by 0.2 to 0.7 on noisy GMM and both multisensory conditions
  where the cap is closer, and on Student D8 is as low as the cap (both
  1 to 1.8 nats below the reference) while the raw value is within
  0.3; the pilot's under-prediction holds at scale, and no weight-aware
  variant of the cap separates the heavy-tailed case from the typical
  one. No single rule wins, so the decision on the reported estimator
  is open for the PI with that evidence. Next: the `M = 32` cells once
  they run, and, if a cross-run estimator is still wanted, the two
  variants the report names (a leave-one-run-out GP refit on the pooled
  evaluations of the other runs; a per-run offset correction). The
  stacking objective stays unchanged. See the
  [ELBO optimism note](2026-09-12-svbmc-elbo-optimism.md) and the
  completed [Phase 1 plan](plans/svbmc-elbo-reporting.md).

- [ ] **S-VBMC benchmark campaign against the original implementation.**
  Assess the combined effect of the 1.5 S-VBMC changes (debiasing,
  preparation and entropy speedups) against the original standalone
  S-VBMC on matched input-run groups: posterior quality, evidence accuracy
  scored by bias against the stack's own Monte Carlo ELBO, runtime. The
  [campaign plan](plans/svbmc-benchmark-campaign.md) owns the approved
  design: eight pool conditions at PyVBMC's default budget, 100 filtered
  runs per noisy condition and 50 per control, each run saved with its
  posterior and the GP behind its statistics so the same pools serve the
  Phase 2 estimator, gpyreg pinned to 1.2.1. The harness is implemented
  and reviewed and a three-seed pilot has run. Status: the pools were
  generated on the cluster per the
  [hand-off note](2026-09-14-svbmc-pool-handoff.md) (1100 runs, 700
  selected), handed back as the archive of the draft release
  `svbmc-pool-20260914` and PR #177 (Slurm scripts, a `verify`
  subcommand, the tracked records under
  `experiments/svbmc_pool/pool_20260914/`), reviewed and merged on
  2026-09-14. The archive is unpacked and verified on the analysis
  machine, whose BLAS does not recompute the stored statistics bit for
  bit; the recomputation gate now allows each artifact the rounding its
  own GP's condition number amplifies, which on the noisy Rosenbrock
  condition (GPs trained on far-tail evaluations) reaches a relative
  0.3 (the plan's worklog of 2026-09-14 has the numbers). The stacking
  harness reads a copied pool through `--gpyreg-source`. Stage D, the
  two-arm comparison on the decision-6 grid (both arms, 560 cells), ran
  on the analysis machine from `dev-next` `744864a` in 8.1 hours into
  `dev/scripts/runs/svbmc_pool_20260914_stack/`, its summaries tracked
  under `experiments/svbmc_pool/stack_20260914/`. Criteria 1, 2, 4 and
  5 hold on every condition; criterion 3's gate fails on Student D8 at
  every `M`, where the capped headline sits 0.9 to 1.8 nats below the
  stack's own Monte Carlo ELBO while both raw estimates are within 0.4
  (the plan's worklog of 2026-09-15 has every number). The PI decided
  (2026-09-15): the Rosenbrock noise-3 condition stays, with its
  numerically fragile GPs stated as a caveat; Phase 2 scores every
  condition and its own-run consistency check marks Rosenbrock's honest
  estimates as unreliable, so Phase 2's conclusions rest on the other
  five noisy conditions; and `M = 32` runs for the integrated arm alone
  (`svbmc_pool_stack.py --arms integrated`, merged 2026-09-15 together
  with `--summarize-only` over several `--from-results` files). The
  [stage D report](results/2026-09-15-svbmc-pool-comparison.md) covers
  the comparison, the Phase 2 scoring of its cells (2026-09-15) and the
  weight-aware-cap experiment. Next: the `M = 32` run for the
  integrated arm (80 cells, about 8 hours, on a night the PI chooses:
  `svbmc_pool_stack.py --pool ... --gpyreg-source ... --M 32
  --repetitions 10 --arms integrated`), its Phase 2 scoring, one merged
  summary (`--summarize-only` with both `--from-results` files), and
  the report's `M = 32` section. Whether the large per-cell files
  (stage D's 17.6 MB `results.json`, the scoring's 27.6 MB
  `results.json` and 17.4 MB `cells.jsonl`) are tracked or published
  as a release asset like the pool is open. See
  the
  [campaign requirement](plans/svbmc-integration.md#benchmark-campaign-required-for-15).

- [ ] **Robustness of the GP on noisy unbounded targets (investigate;
  algorithmic changes are outside 1.5 unless the cause is a bug).** In
  the S-VBMC pool's Rosenbrock condition at noise 3 (D = 2, unbounded,
  N(0, 3²) prior, VIQR), most runs evaluate points far in the tails, log
  densities of -1e5 to -1e6 and once -6.5e9, and the GP that fits them
  next to values near zero with the negative-quadratic mean ends with
  kernel-matrix condition numbers of 1e10 to 1e17, so its posterior
  factors are at the edge of double precision: another machine's BLAS
  recomputes the run's expected-log-joint statistics with relative
  deviations up to 0.3 (the campaign plan's worklog of 2026-09-14 has
  the numbers; the laptop pilot's runs show the same). The runs' own
  answers are fine (median evidence error 0.17, gsKL 0.11, MMTV 0.09),
  but a GP this ill-conditioned makes every consumer of its posterior
  fragile. To find out: why the acquisition samples such points on a
  noisy unbounded target and whether that is intended, whether noise
  shaping (`gaussian_process_train.py`) is doing what it should for
  them, and whether the acquisition's search or the plausible box
  should bound the candidates. The PI decided (2026-09-15) that an
  algorithmic fix belongs after 1.5 unless the investigation finds a
  bug.

- [ ] **PyMC integration: the target adapter with the tested scope.** The
  PI chose (2026-09-14), on the
  [feasibility report](results/2026-09-14-pymc-feasibility.md), to include
  a model-to-target adapter in 1.5 with the scope the prototype
  (`dev/scripts/pymc_feasibility.py`) exercised: continuous variables with
  none, log, log-odds or fixed-interval transforms, a variable bounded on
  one side keeping PyMC's transform and Jacobian, everything else rejected
  by name; a default starting point and plausible box, explicit values
  accepted; structured ArviZ export with the model's names, shapes and
  coordinates; deterministics and posterior predictions through PyMC's
  own functions. The PI's review of the plan set the default's cost: at
  most `20 + 5 D` evaluations (a gradient search and an exact Hessian of
  the log joint, fallbacks that cost none), all of them handed to VBMC.
  The
  [implementation plan](plans/pymc-target-adapter.md) is drafted and
  reviewed (module and extra, the `PyMCTarget` class, the `to_arviz`
  extension, the four guarded PyMC reaches, the `pymc >= 6.3` floor with
  the CI cell, save and load, tests, docs, Example 8). Next, before
  approval: the plan's two investigations, the setup probe (the budget,
  the stopping rule, the location check) and the route by which the setup
  evaluations reach VBMC (the existing `f_vals` option or an extension of
  `VBMC`'s interface, measured in short runs); then approval, Phase 0
  creates the branch `dev-pymc-adapter` and Opus sub-agents implement
  Phases 1 to 5. Automatic
  initialization and inference orchestration stay deferred. See the
  [PyMC proposal](2026-09-13-pymc-integration.md).

- [ ] **Slurm/HPC benchmark support.** Design reproducible submission,
  resource settings, resumption and result collection. The pool campaign's
  per-case worker and `cases` enumeration are the first cluster-ready
  pieces, and the sbatch scripts the cluster developer returns with the
  pool PR (under `dev/scripts/hpc/` or similar) are the seed of this item;
  the general design remains open. Needed before relying on that workflow
  for further cluster campaigns, not before local experiments. See
  [HPC support](plans/modernization-roadmap.md#benchmark-coverage-and-hpc-support).

- [ ] **Release documentation and validation.** Finish the API/tutorial
  review; run examples, check links, build Sphinx and inspect rendered pages.
  Check the [agent skill](../skills/pyvbmc/SKILL.md) against the release docs.
  Run final integrated tests, the required CI matrix and package checks;
  prepare the golden-trace release archive. Search the whole repository's
  documentation for references to the `dev-next` branch and remove those
  that are not historical: `dev-next` merges into `main` for the release,
  so nothing a user or contributor reads after it should point at
  `dev-next`; dated devlogs and plan worklogs may keep it as history.
  Documentation can proceed
  alongside implementation; final checks must cover settled release code.
  See the [documentation checklist](plans/modernization-roadmap.md#pre-release-documentation-review)
  and [reference record](golden/promotion_20260913/README.md).

## Outside 1.5 scope

- **New acquisition-function design.** Efficiency work uses existing criteria.
  The experimental VIQR losses and the EIG acquisition were removed from the
  package on 2026-09-14 and are retained on the branch
  `retain-experimental-acquisitions`; the
  [experiment records](results/2026-09-08-noisy-acquisition-experiments.md)
  stay.
- **Additional packaged regression tests using timing or multisensory.**
  These targets and datasets remain in the developer benchmark suite; their
  benchmark coverage is complete. See the
  [real-data plan](plans/benchmark-realistic-targets.md).
- **A full Torch solver port.** NumPy/SciPy remains the 1.5 solver; optional
  Torch and ArviZ exports remain available. See the
  [backend decision](plans/stage4-torch-feasibility.md).
- **Fixing existing occasional inference failures or redesigning convergence
  rules as a research project.** The accepted assessment found no convincing
  evidence of degradation. These questions do not block the release. See the
  [assessment](golden/promotion_20260913/README.md) and
  [deferred research](plans/modernization-roadmap.md#deferred-devlog-12).
- **The Goris neuronal-model benchmark.** It remains deferred; see the
  [target decisions](plans/benchmark-realistic-targets.md#decisions-pi-2026-09-11).

## After PyVBMC 1.5 is published

- [ ] **Standalone `svbmc` compatibility release.** It depends on
  `pyvbmc[torch]>=1.5` and forwards old `svbmc` imports to the integrated
  implementation. S-VBMC is already available through PyVBMC; this serves
  users of the old package. See the [integration plan](plans/svbmc-integration.md).
- [ ] **Respond to issue #138 about RNG control** with the released API/docs.
  See the [post-release follow-up](plans/modernization-roadmap.md#post-release-follow-up).

## Dependencies and working rules

- Evaluate numerical proposals before choosing defaults or reported
  estimators. If an accepted change moves default trajectories, update the
  affected golden references after assessment and preserve the old ones.
- The Phase 2 estimator and the stacking comparison both consume the run
  pools, which arrive from the cluster as a verified archive; nothing that
  needs them runs before the archive is in and every artifact re-verifies.
- At most one heavy computation runs at a time. Read-only investigation
  and documentation may proceed alongside it.
- Final release checks depend on included changes being settled.
  S-VBMC changes require the comparison campaign against original S-VBMC
  before release. Publication and post-release tasks remain separate
  actions.
- Use feature branches for implementation. Planning, proposal, handoff and
  status edits belong on `dev-next`. Leave unrelated work intact.

## Completed baseline and local artifacts

Core modernization, S-VBMC integration with its preparation and entropy
speedups and Phase 1 ELBO reporting, the real-data benchmark extension,
the population assessment and the removal of the experimental
acquisitions are complete; the [roadmap](plans/modernization-roadmap.md)
retains them. The active reference is **`reference_990_20260913`**: 870
candidate pairs plus 120 unchanged real-data pairs. The
[promotion record](golden/promotion_20260913/README.md) owns the
assessment, independent review, hashes and passed gates;
[the population plan](plans/final-population-benchmark.md) owns execution
history.

Raw traces, boost captures, run pools, captured states and frozen
worktrees are gitignored under `dev/scripts/runs/` and exist only on the
machines that produced them. A machine that holds such artifacts lists
them, with the commands that recreate its worktrees, in
`dev/scripts/runs/LOCAL.md`; that file is gitignored with the rest, so its
absence in a checkout means nothing is stored locally. The tracked side of
every campaign (sidecars, manifests, summaries, hashes, reports) is under
`dev/experiments/`, `dev/golden/` and `dev/results/`, and each plan names
the revisions needed to recreate a worktree, so a fresh clone can read every
result and revalidate raw artifacts copied from the holding machine. No
completed campaign or replay needs reattachment.
