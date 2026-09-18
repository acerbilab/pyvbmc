# PyVBMC 1.5: remaining work and scope

Updated 2026-09-16. These lists describe scope, not priority or execution
order; independent workstreams can be picked up in any order. Inclusion in
scope does not settle an implementation design or launch a campaign.
Completed 1.5 work is not listed here: the
[roadmap](plans/modernization-roadmap.md) retains it, and each item's plan
records its execution.

## In scope for 1.5

- [ ] **Efficiency of existing noisy acquisitions.** Improve sieve/search,
  integration accuracy and criterion-evaluation costs. GP fitting,
  initialization and retraining policy are outside this workstream
  (PI decision, 2026-09-16).
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
  The [integration and search experiment](plans/noisy-acquisition-efficiency.md#integration-and-search-experiment)
  compares ordinary and stratified Monte Carlo and randomized quasi-Monte
  Carlo, then fixed-budget
  search and conditional adaptive integration. It includes an independent
  integration judge and a bounded paired inference comparison. On
  `dev-noisy-acquisition-efficiency`, E2 closed without an integration
  finalist, E3 froze the S2 search arm with mixed holdout evidence, E4's
  entry gates are unmet, and E5 is complete: 120 paired S0/S2 fits over
  ten seeds and six configurations. S2 halves fit time at unchanged
  evaluation counts but degrades inference on high-noise Rosenbrock and
  the posterior-shape metrics of logistic regression while improving
  Student-t substantially, so the E6 assessment recommends keeping the
  production search and proposes frozen-state follow-ups (sieve size,
  matched-cost RQMC, a trajectory diagnosis) that await a user decision.
  The [experiment report](results/2026-09-16-noisy-acquisition-integration-search.md)
  records the results. The benefit
  criteria, allocation and screening gates are accepted; Bayesian
  quadrature is excluded following the scope review.
  A controlled whole-VBMC timing comparison of the completed
  arithmetic optimizations remains optional; replay wall times do not
  establish that speedup.
  Keep standard VIQR (`loss="iqr"`); the retained `iqr_reduction`
  formulation preserves the unregularized criterion for fixed samples and
  weights. The complete objective, including candidate-dependent penalties,
  needs an equivalence check before a search reformulation is used, followed
  by end-to-end validation. IMIQR cache reuse is optional cleanup.

- [ ] **S-VBMC ELBO headline selection.** The two-level shrinkage estimate
  is implemented as `elbo_details["shrunk_two_level"]` and integrated into
  `dev-next`; see the [integration plan](plans/svbmc-shrinkage-estimator.md).
  Validate it against the existing estimates on fresh release-campaign
  pools, judging the optimism added by stacking relative to its input runs.
  Promote it only if the campaign confirms it is the best estimator; decide
  separately whether noiseless stacks use shrinkage or retain the raw value.
  The stacking objective and selected posterior remain unchanged.
  Finalize the headline's qualitative caveat, distinguishing inherited VBMC
  bias from bias added by stacking and explaining that residual bias can
  remain; the raw estimate is not an upper bound on the truth. Update the
  `SVBMC` docstring, API reporting section, FAQ and existing Example 7
  explanation to match the decision, with no additional user step.
  The [headline note](2026-09-15-svbmc-headline-shrinkage.md) retains the
  evidence, rejected alternatives and open scientific questions; the
  [campaign plan](plans/svbmc-benchmark-campaign.md) owns the release gate.

- [ ] **Slurm/HPC benchmark support.** Design reproducible submission,
  resource settings, resumption and result collection. The pool campaign's
  per-case worker and `cases` enumeration are the first cluster-ready
  pieces, and the sbatch scripts the cluster developer returns with the
  pool PR (under `dev/scripts/hpc/` or similar) are the seed of this item;
  the general design remains open. Needed before relying on that workflow
  for further cluster campaigns, not before local experiments; the
  final large-scale check below is the first such campaign. See
  [HPC support](plans/modernization-roadmap.md#benchmark-coverage-and-hpc-support).

- [ ] **Independent codebase and MATLAB-port audit.** Before freezing the
  code for the final release benchmark, have several independent reviewers
  examine the PyVBMC codebase for errors and latent bugs. Split coverage
  between internal correctness and systematic comparison with the original
  MATLAB VBMC implementation, with overlapping review of critical numerical
  paths. Cover formulas and gradients, indexing and array shapes, defaults,
  control flow, random draws, state and caching, and cross-module behavior.
  Record the MATLAB revision used and distinguish intentional Python
  differences from porting mistakes and defects shared by both versions.
  The recently discovered long-standing acquisition-box sampling error
  motivates checking established code as well as recent changes; passing
  tests and stored oracles do not establish correctness of the original port.
  Reconcile findings against source and reproducible examples, add regression
  checks for confirmed fixes, and apply the existing numerical gates to any
  behavior changes. Resolve findings or document their disposition before the
  final benchmark so it measures the reviewed release candidate. This audit
  can start before the other release work is complete.

- [ ] **Final large-scale check before the release (the gate).** Once
  1.5 is consolidated and the code audit above is complete, regenerate the
  VBMC run pools on the test targets with the release code on the cluster
  (about 100 runs per condition as in the
  [campaign](plans/svbmc-benchmark-campaign.md),
  several hundred where `M = 32` is to be scored: the present pools of
  100 noisy and 50 noiseless runs reuse each run 1.6 to 3.2 times at
  `M = 16` and 3.2 to 6.4 times at `M = 32`, so ten disjoint subsets of
  32 need 320 runs), then run S-VBMC at several `M` and check that the
  campaign's results hold: the acceptance criteria of the comparison,
  the added-bias ranges of the headline note, and the switch of the
  headline to the two-level shrinkage (the debiasing item above). The
  gate tests PyVBMC 1.5 end to end, not S-VBMC alone: the pools are
  VBMC runs of the release code, and each run's record carries its
  convergence status and its posterior and evidence metrics against
  the ground truth. Uses the pool generator and stacking harness under
  `dev/scripts/` through the cluster workflow of the HPC item above.
  PI, 2026-09-16.

- [ ] **Release documentation and validation.** Finish the release-wide
  API/tutorial and compatibility review; run examples, check links, build
  Sphinx and inspect rendered pages against the settled release code. The
  focused Torch/JAX workflow and teaching consistency review are recorded
  in the [teaching-material plan](plans/teaching-material.md).
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

- **GP robustness on noisy unbounded targets.** Investigate controls on
  the influence of extreme low-density observations, including MATLAB's
  optional noise shaping. The [tail-acquisition report](results/2026-09-16-gp-tail-acquisition.md)
  records the VIQR/refit mechanism and precision checks at its onset;
  remedies and their effects on inference accuracy have not been compared.

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
  part of 1.5. No user-facing text states this optimism today (the FAQ
  and the noisy-target documentation are silent); a sentence there
  belongs to this item, and the S-VBMC caveat of the debiasing item
  above will point at it.

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
- **Experiments test what ships (PI decision, 2026-09-18).** The golden
  suite's noisy configurations pin the 2020 paper's budget of 50 (D + 2)
  evaluations, which suppresses the package's own adjustment for a
  specified-noise target (75 (D + 2)); every other option is the package
  default. The golden references and every campaign run on those labels,
  including the E5 inference comparison, therefore measured noisy targets
  on the noiseless budget. From now on the release checks and any new
  experiment run noisy targets at the package's production defaults, on
  the `production` suite of `benchmark_targets.py`: the golden suite with
  its noisy entries freed of the budget pin and given the `production`
  label tag. The noiseless golden runs are production runs already, so a
  production reference copies them and adds fresh runs only for the noisy
  labels: the seven noisy configurations with reference runs, 250 runs at
  their production budgets (`golden_trace.py run --suite production --only
  <noisy production labels>`), plus `lumpy_D10_noise3_production` if that
  configuration is to enter the reference (the current reference has no
  runs for it). The golden references stay the regression baseline for
  trajectory identity at the paper budget. The E5 report's F3 section
  records how the difference was found.
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
