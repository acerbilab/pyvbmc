# PyVBMC 1.5: remaining work and scope

Updated 2026-09-13. These lists describe scope, not priority or execution
order. Independent workstreams can be picked up in any order: PyMC
feasibility, for example, does not depend on acquisition-efficiency work.
Inclusion in scope does not settle an implementation design or launch a
campaign. The guarded-VIQR optimization's approved 18-run comparison passed;
see the [efficiency plan](plans/noisy-acquisition-efficiency.md) for validation
and subsequent possibilities.

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
  Historical replacement-criterion experiments do not expand 1.5 scope.
  The [efficiency plan](plans/noisy-acquisition-efficiency.md) tracks the
  guarded standard-VIQR sum optimization, its measured gains and validation.
  GP-kernel reuse has a positive
  [feasibility probe](results/2026-09-13-viqr-kernel-reuse.md); its gpyreg
  interface and temporary-memory policy remain to be implemented. Adaptive
  sieve size, integration budgets and multistart L-BFGS-B remain main search
  experiments. Another possibility is more efficient integration of the
  same VIQR criterion (shared-weight Bayesian quadrature,
  compared with mixture-stratified randomized quasi-Monte Carlo).
  IMIQR cache reuse is optional cleanup; it does not accelerate VIQR.
  Keep standard VIQR (`loss="iqr"`). The `iqr_reduction` formulation may
  be evaluated within search-efficiency work: it preserves the underlying
  criterion for fixed samples and weights, but can change numerical search
  behavior and needs end-to-end validation.

- [ ] **Remove experimental acquisition alternatives from the 1.5 public
  interface.** The decision covers VIQR's `var_reduction` and `sd_reduction`,
  scalar `AcqFcnEIG()` and per-component `AcqFcnEIG(components=True)`.
  The VIQR alternatives change the criterion. In the Rosenbrock D2
  experiments, scalar EIG underperformed at both tested noise levels;
  per-component EIG was competitive at low noise but worse at higher noise.
  Neither EIG variant demonstrated a robust advantage over default VIQR.
  Preserve their implementation on a retained experimental development
  branch when carrying out the cleanup, and keep the
  [experiment records](results/2026-09-08-noisy-acquisition-experiments.md).
  Code removal and branch preservation remain to be done.

- [ ] **S-VBMC ELBO debiasing.** Evaluate raw, capped and proposed cross-run
  GP estimates against ground truth, including noisy targets and different
  numbers of stacked runs. The stacking objective stays unchanged. Phase 1
  reporting/corrections are complete; Phase 2 is the remaining investigation.
  Decide the reported estimator, fallback and interface from its results.
  See the [ELBO optimism note](2026-09-12-svbmc-elbo-optimism.md) and
  completed [Phase 1 plan](plans/svbmc-elbo-reporting.md).

- [ ] **S-VBMC preparation and entropy speedups.** Reuse transforms and
  density calculations and vectorize entropy computations during stacking.
  These are implementation optimizations that preserve the method and sampled
  objective, separate from ELBO debiasing. Validate against the S-VBMC
  regression references and measure performance on representative workloads.
  See the [prototype measurements](results/2026-09-09-svbmc-numpy-prototype.md)
  and [integration decisions](plans/svbmc-integration.md).

- [ ] **S-VBMC benchmark campaign against the original implementation.**
  After the S-VBMC changes included in 1.5 are settled, assess their combined
  effect against the original standalone S-VBMC, covering debiasing and
  preparation/entropy speedups. Compare posterior quality, reported evidence
  accuracy and runtime on matched input-run groups, including noisy targets
  and different numbers of stacked runs. S-VBMC has numerical regression
  fixtures and integration parity checks, but lacks a population benchmark
  reference comparable to VBMC's golden campaign. Establish and preserve the
  original S-VBMC baseline and reproducible campaign results. Target/seed
  allocation and acceptance criteria remain to be designed. Reuse suitable
  VBMC numerical fixtures and retained runs, prioritizing hard targets such
  as noisy multisensory and Rosenbrock; check that the required posterior
  and GP state is available. See the
  [campaign requirement](plans/svbmc-integration.md#benchmark-campaign-required-for-15).

- [ ] **Slurm/HPC benchmark support.** Design reproducible submission,
  resource settings, resumption and result collection. The implementation
  design remains open. This is needed before relying on that workflow for
  cluster campaigns, not before local experiments or unrelated integration.
  See [HPC support](plans/modernization-roadmap.md#benchmark-coverage-and-hpc-support).

- [ ] **Release documentation and validation.** Finish the API/tutorial
  review; run examples, check links, build Sphinx and inspect rendered pages.
  Check the [agent skill](../skills/pyvbmc/SKILL.md) against the release docs.
  Run final integrated tests, the required CI matrix and package checks;
  prepare the golden-trace release archive. Documentation can proceed
  alongside implementation; final checks must cover settled release code.
  See the [documentation checklist](plans/modernization-roadmap.md#pre-release-documentation-review)
  and [reference record](golden/promotion_20260913/README.md).

## Scope or disposition still to be decided

- [ ] **PyMC integration: include what in 1.5?** A bounded feasibility check
  can be the next task. The proposal covers structured ArviZ export, a
  model-to-target adapter and a worked example with explicit initialization
  and plausible bounds. Decide the supported model scope and inclusion
  after the check. Automatic initialization and inference orchestration are
  deferred in the proposal. See the [PyMC proposal](2026-09-13-pymc-integration.md).

## Outside 1.5 scope

- **New acquisition-function design.** Efficiency work uses existing criteria.
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
- Cross-run S-VBMC evaluation needs posteriors with their corresponding GPs.
  Establish suitable retained inputs or an approved run allocation before
  the experiment. Cluster execution additionally needs the Slurm workflow.
- At most one heavy computation runs at a time. Read-only investigation
  and documentation may proceed alongside it.
- Final release checks depend on included changes being settled.
  S-VBMC changes require the comparison campaign against original S-VBMC
  before release; baseline preparation can proceed before those changes.
  Publication and post-release tasks remain separate actions.
- Use feature branches for implementation. Planning, proposal, handoff and
  status edits belong on `dev-next`. Leave unrelated work intact.

## Completed baseline and local artifacts

Core modernization, S-VBMC integration and Phase 1 reporting, the real-data
benchmark extension and the population assessment are complete. The active
reference is **`reference_990_20260913`**: 870 candidate pairs plus 120
unchanged real-data pairs. The [promotion record](golden/promotion_20260913/README.md)
owns the assessment, independent review, hashes and passed gates;
[the population plan](plans/final-population-benchmark.md) owns execution
history. The [roadmap](plans/modernization-roadmap.md) retains completed work.

Raw traces, boost captures, diagnostic scripts and frozen checkouts remain
local and gitignored under `dev/scripts/runs/` on the original Windows
benchmark machine: `population_overnight_20260910/`,
`population_extension_20260911/`, `population_completion_20260912/`,
`population_realdata_20260912/`, and `golden/reference_990_20260913/` with
its previous references. A fresh clone has tracked sidecars, assessments,
manifests and reports. Copy those directories to revalidate raw artifacts;
follow the population plan for the environment and repair or recreate copied
worktrees at their recorded commits.

Reproduce the historical comparison on that machine with
`.venv/Scripts/python.exe -u dev/scripts/runs/population_completion_20260912/assess.py`.
This archived wrapper selects the frozen classes and historical 870-case
baseline; the active baseline contains the promoted candidates. No completed
campaign or replay needs reattachment. The controller's final
`assessment_ready_for_review` state precedes the acceptance and publication
recorded in the promotion record.
