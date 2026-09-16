# PyVBMC 1.5: remaining work and scope

Updated 2026-09-15. These lists describe scope, not priority or execution
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

- [ ] **S-VBMC ELBO debiasing (the optimism note's Phase 2).** Decide
  the headline `elbo` of a noisy stack. The candidate is the two-level
  empirical-Bayes shrinkage of the components' expected log joints by
  the estimation covariance saved with each posterior (PI,
  2026-09-15): the
  [headline note](2026-09-15-svbmc-headline-shrinkage.md) is the
  summary and decision record, the
  [stage D report](results/2026-09-15-svbmc-pool-comparison.md)
  (sections "Phase 2", "Weight-aware caps" and "Empirical-Bayes
  shrinkage") the evidence, the
  [tutorial note](2026-09-15-svbmc-shrinkage-explained.md) the
  explanation of the method with a worked example, and
  `python dev/scripts/svbmc_headline_numbers.py` regenerates every
  number they quote. What the run pools settled: the cross-run
  "honest" estimator (`dev/scripts/svbmc_honest_elbo.py`; the
  [pilot report](results/2026-09-14-svbmc-honest-elbo-pilot.md) is
  its first run) is not the answer, under-predicting by 0.2 to 0.7
  nats on three noisy conditions and matching the cap's error on
  Student D8; no weight-aware variant of the cap separates the
  heavy-tailed case from the typical one; the shrinkage is the first
  estimate acceptable on every condition, with no tuned constant. It
  reads only `I_sk`, `J_sjk` and each run's own weights, which
  `SVBMC` already requires of every input posterior, so the class's
  interface does not change and no GP or VBMC object is needed. The
  yardstick (PI, 2026-09-15; the campaign plan's decision 11): a
  stacked headline must not add to the optimism its input runs
  already carry, so it is judged by its bias minus the mean bias of
  its inputs, each scored against its own Monte Carlo ELBO
  (`svbmc_single_run_bias.py`; the report's section "The inputs' own
  bias, and what stacking adds"): the raw value adds 0.02 to 0.60
  nats growing with `M`, the cap removes more than the stacking added
  on five of the six noisy conditions and lands at the inputs' level
  on Rosenbrock, the two-level shrinkage adds nothing within 0.23
  nats, and the anchored variants (what the shrinkage removes at each
  run's own weights added back, so that only the stacking's selection
  is removed) leave −0.02 to +0.21 nats of the addition at `M` = 3 to
  5, so the two-level shrinkage stays the candidate; the untested
  refinement is a run-level error term that includes the run-to-run
  scatter of the runs' own optimism, which the stacking selects on.
  Pickup (end of 2026-09-15): nothing is running; the analyses and
  their write-ups were reviewed and corrected (the campaign plan's
  worklog records what changed); the next step is decision (1).
  Decisions to make: (1) whether the switch ships in 1.5 or 1.5 keeps
  the component-median cap; (2) the headline's user-facing caveat
  either way, stating the two contributions to the bias separately:
  VBMC's own, which each input run carries and S-VBMC inherits (0.1 to
  0.7 nats of optimism on the typical noisy targets, 0.2 of pessimism
  on the heavy-tailed one, within 0.03 on the noiseless ones; the
  separate item below), and what S-VBMC adds to it (the raw value
  +0.07 to +0.36 nats at `M` = 3 to 5 and +0.36 to +0.79 at `M = 32`
  on the noisy targets; the two-level shrinkage within 0.23 at every
  `M`; the cap nothing added but up to 0.4 nats of the inputs' own
  optimism removed on the typical noisy targets and 0.8 to 1.6 on the
  heavy-tailed one, a `cap_amount` large against `elbo_sd` being the
  sign; on noiseless targets up to about 0.1 nats at `M = 32`, within
  0.05 through `M = 16`); and that the raw value is not an upper bound
  on the truth. If it ships, the
  note's "Decision" section lists the change: a headline method in
  `pyvbmc/svbmc/svbmc.py` from the reference
  `dev/scripts/svbmc_shrink_elbo.py`, applied at the selected weights
  so the posterior does not move, an `elbo_details` key and the noise
  share as a diagnostic, the tests that pin the capped headline, and
  the user documentation. Rejected on the evidence: re-optimizing
  the weights on the shrunken estimates (`svbmc_shrink_optimize.py`,
  `M` = 3 to 5: the optimizer selects on the shrinkage's own errors,
  the headline is 0.05 to 0.40 nats more optimistic than the
  value-only shrinkage on four of six noisy conditions, and the
  posterior improves on the multisensory conditions and the ring and
  worsens on Rosenbrock and, in the KL gap, on GMM and Student). Not
  planned: the cross-run variants the report names (a
  leave-one-run-out GP refit, a per-run offset) unless a cross-run
  estimator is still wanted. The stacking
  objective stays unchanged. The campaign's `M = 32` cells (the
  report's section "The integrated arm at `M = 32`") keep the ordering
  of the estimates and add one observation for the caveat: on the
  noiseless multisensory control, stacking 32 runs adds 0.11 nats of
  optimism to the 0.03 its inputs carry, and no estimate removes it.
  See the
  [ELBO optimism note](2026-09-12-svbmc-elbo-optimism.md) and the
  completed [Phase 1 plan](plans/svbmc-elbo-reporting.md).

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
- **Debiasing the VBMC ELBO itself on noisy targets.** A single run's
  reported ELBO is optimistic on a noisy target because the
  variational optimization selects components on noisy GP estimates;
  the S-VBMC run pools measure it per run against the run's own Monte
  Carlo ELBO (`experiments/svbmc_pool/single_run_20260915/`, read in
  the [stage D report](results/2026-09-15-svbmc-pool-comparison.md)).
  The within-run empirical-Bayes shrinkage of the components' expected
  log joints (the [tutorial note](2026-09-15-svbmc-shrinkage-explained.md))
  reads only a run's own `I_sk`, `J_sjk` and weights and would apply
  to a single run's headline. The PI (2026-09-15): a question separate
  from S-VBMC, whose requirement is not to add bias to its inputs; not
  part of 1.5.

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
  pools, the asset of the draft release `svbmc-pool-20260914`; nothing
  that needs them runs on a machine before every artifact re-verifies
  there (`svbmc_pool_run.py verify`).
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
retains them. The S-VBMC benchmark campaign against the original
implementation is complete (2026-09-16): eight pool conditions at
`M` = 2, 4, 8 and 16 with both implementations and at `M` = 3, 5 and 32
with the port alone; criteria 1, 2, 4 and 5 hold on every condition,
criterion 3's gate fails on Student D8 because the capped headline
over-corrects on heavy tails, which the debiasing item owns. The
[stage D report](results/2026-09-15-svbmc-pool-comparison.md) reads
the numbers, the [campaign plan](plans/svbmc-benchmark-campaign.md)
owns the design, worklog and the PI's decisions, the
[experiments README](experiments/svbmc_pool/README.md) indexes the
tracked outputs, and the per-cell outputs are the three assets of the
draft release `svbmc-analyses-20260915` (the pool itself the asset of
`svbmc-pool-20260914`). The active reference is **`reference_990_20260913`**: 870
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
