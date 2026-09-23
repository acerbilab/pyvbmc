# PyVBMC 1.5: remaining work and scope

Updated 2026-09-21. These lists describe scope, not priority or execution
order; independent workstreams can be picked up in any order. Inclusion in
scope does not settle an implementation design or launch a campaign.
Completed 1.5 work is not listed here: the
[roadmap](plans/modernization-roadmap.md) retains it, and each item's plan
records its execution.

## In scope for 1.5

- [ ] **Changelog for 1.5.** `CHANGELOG.md` (root, Keep a Changelog layout)
  covers the changes since `v1.0.4`; the worklog of the
  [port review plan](plans/port-correctness-review.md) records how it was
  written and checked (2026-09-20). Open before the release: the PI's
  reading of the entries added that day; the whole-run timings of "Runs are
  faster", measured on 2026-09-03 to 09-05, before the corrections that
  change how long a run takes, and the S-VBMC speed figure, both to be
  measured again on the release benchmark; the S-VBMC entry's account of the
  reported `elbo`, which the item below may change; and the "What's new in
  PyVBMC 1.5" blocks of `README.md` and `docsrc/source/index.rst`, which
  link to the changelog and are to be revised against it (the link of the
  docs resolves once the file is on `main`). A user-visible change is
  listed with the work that makes it (`AGENTS.md`), and its line in the
  changelog's "Upgrading from 1.0.4" lead is kept in step with its entry.

- [ ] **S-VBMC ELBO headline selection.** The two-level shrinkage estimate
  is implemented as `elbo_details["shrunk_two_level"]` and integrated into
  `dev-next`; see the [integration plan](plans/svbmc-shrinkage-estimator.md).
  Validate it against the existing estimates on fresh release-campaign
  pools, judging the optimism added by stacking relative to its input runs.
  Promote it only if the campaign confirms it is the best estimator; decide
  separately whether noiseless stacks use shrinkage or retain the raw value.
  Before that decision, compare two compositions of its stages on the
  existing pools with the campaign's added-bias measure. The implemented
  one adds the between-run change of a run's raw level to the
  within-shrunk components, so the run's final own-weighted level is the
  shrunk level plus whatever the within-run stage did to it; the
  alternative pins each run's level to its shrunk value. They coincide
  for uniform own weights and differ where a run's weights align with its
  component estimates (0.16 and 0.30 nats in the constructed case of the
  [port review's verification](experiments/port_review_20260919/verification/wave0.md)).
  The implemented form keeps the within-run correction of the level, which
  addresses the single-run optimism that regression across runs cannot
  see; the comparison is to confirm that on data (PI, 2026-09-19).
  The stacking objective and selected posterior remain unchanged.
  Finalize the headline's qualitative caveat, distinguishing inherited VBMC
  bias from bias added by stacking and explaining that residual bias can
  remain; the raw estimate is not an upper bound on the truth. Update the
  `SVBMC` docstring, API reporting section, FAQ and existing Example 7
  explanation to match the decision, with no additional user step.
  The [headline note](2026-09-15-svbmc-headline-shrinkage.md) retains the
  evidence, rejected alternatives and open scientific questions; the
  [campaign plan](plans/svbmc-benchmark-campaign.md) owns the release gate.

- [ ] **Saving and loading an S-VBMC object.** `SVBMC` has no `save` or
  `load`, unlike `VBMC` and `VariationalPosterior`, and neither the API
  page nor Example 7 says how to keep a stack; the original standalone
  package had no such methods either. The standard `pickle` fails on an
  `SVBMC` object, as it does on a posterior, because the parameter
  transformer holds local closures. `dill`, which the existing `save` and
  `load` methods use, serializes a fresh or an optimized stack with an
  exact round trip of the weights, the ELBO and the generator state, and
  the object holds no Torch state (checked 2026-09-19 during the
  [port correctness review](plans/port-correctness-review.md)). Add
  `SVBMC.save` and `SVBMC.load` mirroring the posterior's (`dill`, the
  overwrite guard), a round-trip test in the Torch CI cell, the API
  documentation, and a short saving step in Example 7.

- [ ] **Slurm/HPC benchmark support.** Design reproducible submission,
  resource settings, resumption and result collection. The pool campaign's
  per-case worker and `cases` enumeration are the first cluster-ready
  pieces, and the sbatch scripts the cluster developer returns with the
  pool PR (under `dev/scripts/hpc/` or similar) are the seed of this item;
  the general design remains open. Needed before relying on that workflow
  for further cluster campaigns, not before local experiments; the
  final large-scale check below is the first such campaign. See
  [HPC support](plans/modernization-roadmap.md#benchmark-coverage-and-hpc-support).

- [ ] **Independent codebase and MATLAB-port review.** Before freezing the
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
  final benchmark so it measures the reviewed release candidate. This review
  can start before the other release work is complete. Started 2026-09-19;
  the [review plan](plans/port-correctness-review.md) records the PI's
  decisions (gpyreg in scope, comparison against the latest MATLAB
  `master`, Opus reviewers, no MATLAB run unless a finding's disposition
  depends on one), the slice map, the reviewer brief and the worklog. Status
  on 2026-09-23: waves 0 to 6 (the subsystems without a MATLAB counterpart,
  the MATLAB changes since the port, every P slice on both tracks, and the
  two gpyreg slices G1 and G2) are reviewed, verified, ruled on and fixed,
  and merged into `dev-next`; wave 6's fixes to gpyreg shipped in gpyreg
  1.3.0, which PyVBMC requires. Several of the fixes of waves 1 to 3, one
  of wave 5 (the balanced draw of the variational posterior) and one of
  wave 6 (the GP mean's bounds per input dimension, W6-1) move default
  trajectories, and so does one fix of the check of wave 1 (the noise
  counts refreshed after every evaluation of active sampling, as in
  MATLAB), so the golden references and the production-reference pools
  describe the code from before them. The passes of waves 1 to 6 were each
  checked afterwards by fresh reviewers without the session's context, and
  what the checks found is fixed and merged as well. Wave 7, the third
  readers O1 to O4, is reviewed, verified, ruled on and fixed, and its pass
  checked by fresh reviewers (`verification/wave7.md`); none of its fixes
  moves a default trajectory, and its fixes to gpyreg shipped in gpyreg
  1.3.1, which PyVBMC requires; all of it is merged into `dev-next`. Next:
  the capped run of row W7-17 and the run that measures W7-2 on a resume,
  then the closing ledger under `results/` and the consolidation of the
  sheet's durable entries into the porting log. The plan's pickup point
  says where to resume.

- [ ] **An oracle state at uncertainty level 1.** No fixture under
  `pyvbmc/testing/oracles/fixtures/` holds a state of a run with
  `uncertainty_handling=True` and no `specify_target_noise`: nine are
  noiseless and two come from targets that provide their noise. At that
  level the GP has the noise model of MATLAB VBMC, a constant plus the
  recorded noise of each point scaled by a fitted multiplier, since the port
  review's wave-3 pass (row W3-1 of the
  [ledger](experiments/port_review_20260919/verification/wave3.md)), and the
  unit tests of the fit's inputs and of the candidate noise in active
  sampling are what pins it. Add one level-1 state: a new recipe of
  `dev/scripts/make_oracle_fixtures.py` on a level-1 variant of a benchmark
  target, with `max_repeated_observations` above 0 so that the recorded
  noise differs between points, generated alone (`--only`) so that the
  existing fixtures stay bit-identical. To be done once the port review's
  remaining fixes are in, since slices G1, G2 and O4 may still move the GP
  fit (PI, 2026-09-21).

- [ ] **A seeded gate run with a prior.** None of the four seeded runs that
  gate the port review's fix passes
  (`experiments/port_review_20260919/verification/scripts/wave2_fixpass_gate_runs.py`)
  is given a prior with `prior=`, so what a run does with a prior object
  rests on unit tests: the check of its support against the hard bounds,
  and the support that `SciPy` and `Product` read from their distribution
  at every call. Add two runs: one with the FAQ's list of `uniform`
  marginals built from the hard bounds, which reaches the slack of that
  check, and one with a `SplineTrapezoidal` on them. To be done when the
  golden references and the run pools are regenerated after the review's
  remaining fixes, which is when the gate's records are made anew (PI,
  2026-09-21; the wave-5
  [ledger](experiments/port_review_20260919/verification/wave5.md), "The
  independent check of the pass").

- [ ] **What the independent check of the wave-6 pass left for later.**
  None changes a number of a run at the shipped options; the raw reports of
  the check are on the machine that ran it (`scripts/runs/LOCAL.md`, "Port
  correctness review").
  - gpyreg: `RationalQuadraticARD.get_bounds_info` carries a full copy of
    the bounds helper of `covariance_functions.py`, and every change of the
    recommendations has had to patch both; it can call the helper and
    override the shape's entries. `GP.update` and `GP.quad` each recover the
    noise scale `sl` of a posterior pickled without it, in two copies of the
    same lines that a method of `Posterior` could hold.
  - gpyreg: a `fit` on a single training point ends with the `KeyError` of
    L-BFGS-B, since the width of every input column is zero and the
    length-scale bounds are `(-inf, -inf)`; gplite's bounds are finite
    there, its `max` and `min` running along the single row (read from its
    source, not run) (entry 49 of
    `experiments/port_review_20260919/matlab_side_defects.md`). A candidate
    for triage, as the constant targets of row W6-4 were. Row W7-17 of
    `experiments/port_review_20260919/verification/wave7.md` finds the same
    mechanism in a set of distinct points that share one coordinate, which
    PyVBMC reaches from such starting points; the PI rules on the two
    together after a capped run from such points.
  - gpyreg: `get_priors` returns `None` for a Student's t block with mixed
    degrees of freedom, such as `[0, nan]` or `[0, 3]`, which `set_priors`
    writes, so `set_priors(get_priors())` drops it; and for a smooth-box
    coordinate written directly into `hyper_priors` with finite `a` and `b`
    and a NaN `sigma` it can return a block that `set_priors` refuses.
  - gpyreg: `set_priors` sets `no_prior` to false before its checks, so a
    refused call leaves it false on a GP without priors; `np.abs(sigma)` in
    the prior code is dead, `set_priors` refusing a negative `sigma`; the
    `Raises` section of `fit` names neither the `ValueError` it passes on
    from `get_recommended_bounds` nor those from `update`; `set_bounds`
    takes an inverted pair, which only `fit` and `get_recommended_bounds`
    refuse.
  - gpyreg's tests: `test_split_update` fixes two hyperparameter samples
    where it drew one or two; the test of the documented prior densities
    runs without bounds, so the renormalization over them is not checked
    against the truncated densities.
  - PyVBMC: `load` refuses a construction-only option even when its value
    equals the stored one, so passing back the original options with a
    new `max_fun_evals` raises.
  - gpyreg: in the low-noise representation of the posterior (a smallest
    noise variance below 1e-6), `predict` and `predict_full` form the
    predictive covariance from the explicit negative inverse the posterior
    holds, as `gplite_pred.m:102-104` does (read, not run), so its
    rounding grows like the inverse of the noise: at a noise standard
    deviation of 1e-6 the variances inside the data are off by up to
    1.9e-3 where they are about 1e-12, and `predict_full`'s covariance has
    eigenvalues down to -1.8e-2. The low-noise rank-one update divides by
    such a variance. `random_function` forms it from a Cholesky factor
    instead. No PyVBMC run reaches the representation. A candidate for
    triage, and for the list of MATLAB-side defects.
  - gpyreg: `fit` with a whole float `thin`, 2.0 for one, raises
    `TypeError` from its own use of it, where `SliceSampler.sample` takes
    it.
  - PyVBMC: `ns_gp_max` is read at every fit, but `_init_optim_state` also
    derives `stop_sampling` from it, so raising it from zero at `load` is
    stored and leaves the hyperparameters unsampled.
  - PyVBMC: the three captures under
    `pyvbmc/testing/oracles/fixtures/gp_fit_history/` still hold the arrays
    `capture/ref/fit/hyp_dict_logp`, which no sidecar names since
    `hyp_dict` lost its `logp`; prune them the next time a capture is
    rewritten.

- [ ] **Final large-scale check before the release (the gate).** Once
  1.5 is consolidated and the code review above is complete, regenerate the
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
  prepare the golden-trace release archive. The package checks include what
  the sdist ships: setuptools_scm puts every tracked file in it, `dev/` and
  `papers/` included (36 MB when built on 2026-09-21), and of `MANIFEST.in`
  only `prune docsrc` has an effect, its `include` lines naming files that
  the sdist holds already. Decide which locally held
  artifacts attach to the release as archives rather than commits: the
  golden reference traces, the run pools, the captured frozen states and
  the raw campaign records that `dev/scripts/runs/LOCAL.md` lists on the
  holding machine. The draft releases `svbmc-pool-20260914` and
  `svbmc-analyses-20260915` already hold the pool and its analyses that
  way, so the decision is which of the remaining artifacts a reader of the
  release needs to revalidate its results. Search the whole repository's
  documentation for references to the `dev-next` branch and remove those
  that are not historical: `dev-next` merges into `main` for the release,
  so nothing a user or contributor reads after it should point at
  `dev-next`; dated devlogs and plan worklogs may keep it as history.
  Before the release, sweep every tracked document and record for two
  kinds of statement: an acknowledgment that a value or a defect in some
  file is wrong while that file itself carries neither the correction
  nor a flag, and a statement that was true when written and is stale
  now (branch names, counts, work described as continuing or remaining).
  The scope is the developer notes and plans, the results, the review
  copies under `dev/experiments/`, `dev/README.md`, `AGENTS.md`, the
  README, the Sphinx sources and the docstrings. The rule: an error in a
  record, document or code is fixed in that file; where a record cannot
  change, the flag goes into the record, or as close to it as its format
  allows; a note somewhere else is not a fix. Read-only reviewers work by
  area and the PI triages their findings. The rule was set on 2026-09-19
  after three cases surfaced in one day: a gpyreg version label wrong in
  252 run sidecars, upper medians quoted as medians in a report, and a
  publication index missing four entries, each acknowledged only in a
  note elsewhere.
  Documentation can proceed
  alongside implementation; final checks must cover settled release code.
  See the [documentation checklist](plans/modernization-roadmap.md#pre-release-documentation-review)
  and [reference record](golden/promotion_20260913/README.md).

- [ ] **The 3D animation of a PyVBMC run (`feat-3d-animation`).** An
  interactive three.js page that plays back a recorded two-dimensional run
  as a rotating landscape of the log density: the GP surrogate, the
  evaluations, the acquisition function while points are chosen, the
  variational mixture and, at the end, the true target. A second page ends
  with the final posterior as the V of the PyVBMC wordmark;
  `scripts/record.mjs` records either page as MP4, GIF or PNG frames, and
  `dev/scripts/export_animation_trace.py` writes the trace of a real run.
  The work is ongoing on the branch `feat-3d-animation`, cut from `dev-next`
  on 2026-09-20 and not merged back; its `README.md`, `NOTES.md` and
  `TODO.md` under `docsrc/source/_static/vbmc3d/` hold the design and the
  next actions. Bring it back into `dev-next` once it settles. Where the
  visualization is shown is undecided (the documentation, the project page,
  the README): on the branch it sits in the docs' static folder, which the
  Sphinx build does not publish, and no page links to it.

## Outside 1.5 scope

- **GP robustness on noisy unbounded targets.** Investigate controls on
  the influence of extreme low-density observations, including MATLAB's
  optional noise shaping. The [tail-acquisition report](results/2026-09-16-gp-tail-acquisition.md)
  records the VIQR/refit mechanism and precision checks at its onset;
  remedies and their effects on inference accuracy have not been compared.

- **New acquisition-function design.** The acquisition-efficiency work used
  existing criteria only.
  The experimental VIQR losses and the EIG acquisition were removed from the
  package on 2026-09-14 and are retained on the branch
  `retain/experimental-acquisitions`; the
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
history. The production reference's noisy runs have begun: 80 runs of six
noisy configurations at production budgets (logistic regression and
high-noise Rosenbrock at twenty seeds; low-noise Rosenbrock, Student-t,
timing and multisensory at ten) exist locally under
`golden/production_noisy_20260918/` (listed in `dev/scripts/runs/LOCAL.md`),
run on 2026-09-18 and 2026-09-19 as the baseline arm of the F2 comparison
and after its stop; the
[efficiency plan](plans/noisy-acquisition-efficiency.md#f2-stage-2-outcome-and-decision-2026-09-19)
records their provenance.

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
