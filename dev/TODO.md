# PyVBMC 1.5: remaining work and scope

Updated 2026-10-12. These lists describe scope, not priority or execution
order; independent workstreams can be picked up in any order. Inclusion in
scope does not settle an implementation design or launch a campaign.
Completed 1.5 work is not listed here, apart from the active reference and
the locally held artifacts that the last section describes: the
[roadmap](plans/modernization-roadmap.md) retains it, and each item's plan
records its execution.

PyVBMC 1.5.0 was released on 2026-10-12, on GitHub, PyPI and
conda-forge. The soft release circulates within the lab, and its checks
are completed before the public announcement, likely on 2026-10-13, a
date that does not bind (PI, 2026-10-11).

## In scope for 1.5

Nothing remains: the roadmap's
[pre-release checklist](plans/modernization-roadmap.md#pre-release-checklist)
records the release and the checks before it.

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
  rules as a research project.** The assessment accepted on 2026-09-13
  found no convincing evidence of degradation. These questions do not
  block the release. See the [assessment](golden/promotion_20260913/README.md) and
  [deferred research](plans/modernization-roadmap.md#deferred-devlog-12).
- **The acceptance of a rotoscaling.** The undo check keeps a warp whose
  refit ELBO exceeds the previous iteration's by `warp_tol_improvement`, as
  MATLAB VBMC does, and a refit posterior that spreads where the refit GP
  has no data can pass it on a gain of the surrogate: Example 2 keeps such
  a warp, reports an ELBO of −0.82 where the true log evidence is −1.836,
  and recovers two iterations later. One run in the 990 of
  `reference_990_20260913` does the same. A warp only reparameterizes the
  space, so a guard could undo a warp whose refit posterior moves far from the one
  before (a threshold on their sKL); untested, and it moves default
  trajectories. See the
  [report](results/2026-09-26-example-2-rotoscale-swing.md).
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
  belongs to this item, and the caveat of the S-VBMC headline on the
  `SVBMC` page will point at it.
- **A longer warm-up for noisy targets of few dimensions.** The release
  code ends warm-up as MATLAB does, which on the noisy two-dimensional
  Rosenbrock target is about three iterations earlier than the port did
  before the port review, and its runs there are usable 72 times in 100
  against the before arm's 87
  ([the note](results/2026-10-05-noisy-rosenbrock-warmup.md)).
  The port's end of warm-up, run as Arm 3 on every configuration of the
  release gate, did not establish the gain on fresh seeds and made the
  noisy Student target worse, so MATLAB's stays (PI, 2026-10-06;
  [the reading](results/2026-10-06-arm-3-reading.md)). A rule that
  lengthens warm-up only for noisy targets of few dimensions is untested
  and would move default trajectories; Arm 3's plan (decision 5) names this
  item. Arm 3's code is on the branch `retain/arm3-port-warmup`.

## After PyVBMC 1.5 is published

- [ ] **Deprecate the standalone `svbmc` package.** S-VBMC is part of
  PyVBMC from 1.5, and users are directed to `pyvbmc.SVBMC` (PI,
  2026-10-08). The last release of `svbmc` depends on `pyvbmc[torch]>=1.5`,
  forwards the old imports to the integrated implementation, so that
  existing scripts keep running, and warns on import that the package is
  deprecated, naming `pyvbmc.SVBMC` and the migration table of the `SVBMC`
  page. The repository `acerbilab/svbmc` stays, as the code of the S-VBMC
  paper, and its README opens with a note that S-VBMC is integrated in
  PyVBMC. See the [integration plan](plans/svbmc-integration.md).
- [ ] **Zenodo at the next release.** The repository's webhook to Zenodo
  archives each release as a new version of the record
  [10.5281/zenodo.10659431](https://doi.org/10.5281/zenodo.10659431),
  whose concept DOI `CITATION.cff` gives; it holds v1.0.2 to v1.0.4 and
  archived v1.0.4 six seconds after its publication. It archived no 1.5.0:
  the webhook's deliveries on 2026-10-12 got HTTP 409 for `published`
  ("The release has already been received") and 202 for `released`, after
  a 202 for `created` when the draft was saved on 2026-10-11. 1.5.0 is not
  added by hand (PI, 2026-10-12): the papers are what users cite. If the
  draft saved before publication is what kept Zenodo from archiving 1.5.0,
  a release prepared the same way meets the same; check Zenodo after the
  next release. A version's authors come from `CITATION.cff`, which lists
  Francesco Silvestrin, the first author of the S-VBMC paper, since
  2026-10-12 (PI).
- [ ] **The old PyPI token.** Until 2026-10-12, `PYPI_PASSWORD` held a
  token set on 2023-03-16 whose owner is not known: the PI's PyPI account
  holds no token for pyvbmc, and only the owner can revoke it. pyvbmc's
  collaborators page on PyPI lists the accounts that can upload. gpyreg's
  secret of the same name was set a minute after it and uploaded gpyreg
  1.4.0 on 2026-09-30, so it likely holds the same token: give it a token
  scoped to gpyreg before the old one is revoked. PyPI's Trusted
  Publishing would replace both secrets with credentials that each
  upload workflow obtains for itself.
- [ ] **gpyreg's upload trigger.** gpyreg's `release.yml` triggers on
  `created` as PyVBMC's did, so publishing a saved draft does not start its
  upload; [acerbilab/gpyreg#70](https://github.com/acerbilab/gpyreg/pull/70)
  changes it to `published`, for version tags.
- [ ] **The conda-forge repodata patch.** The patch
  `conda-forge/conda-forge-repodata-patches-feedstock` #1274 (open since
  2026-10-08) bounds every release from 0.9.0 to 1.0.4 at `gpyreg <1.3`
  and `numpy <2.4`, beyond which they fail; until it merges, conda can
  install 1.0.4 with a gpyreg or a NumPy that it fails with, in an
  environment whose Python or other packages exclude 1.5.0. Follow it up
  until it merges.
- [ ] **PR #73.** The pull request "pymc integration" (aloctavodia, open
  since 2022) is superseded by `pyvbmc.PyMCTarget`; close it with thanks
  and a pointer to the PyMC guide of the quickstart and Example 8.
- [ ] **The `vp.plot()` tip's anchor, for 1.5.1.** The startup tip about
  `vp.plot()` (`pyvbmc/vbmc/_tip_catalog.py`) links
  `variational_posterior.html#pyvbmc.VariationalPosterior.VariationalPosterior.plot`,
  but the page's anchor for the method is
  `pyvbmc.variational_posterior.VariationalPosterior.plot` since the page
  documents the methods with its class (`544224c5`, 2026-10-08), so the
  link opens the page at its top. 1.5.0 ships the tip so, and the stored
  output of Example 6 shows it. Correct the anchor, check the other links
  of both tip catalogs, anchors included, against the published
  documentation, and execute Example 6 again.
- [ ] **Respond to issue #138 about RNG control** with the released API/docs.
  See the [post-release follow-up](plans/modernization-roadmap.md#post-release-follow-up).
- [ ] **Packaging metadata.** A build of 1.5.0 warns of two parts of its
  configuration that setuptools will stop accepting: the `project.license`
  table, which setuptools stops supporting on 2027-02-18 (a `license` SPDX
  string with `license-files` needs setuptools 77), and the subpackages that
  `[tool.setuptools] packages` does not list, among them `pyvbmc.vbmc` and
  `pyvbmc.entropy`, which ship as package data and which a later setuptools
  would leave out of the wheel. Both hold since 1.0.4; fix them in the next
  release, checking that the wheel still holds every runtime file and no
  test.
- [ ] **A readable `print(target)`, for 1.5.1.** `PyMCTarget.__str__`
  (`pyvbmc/pymc/_target.py`) prints a variable's hard and plausible bounds
  as two NumPy arrays at full precision, the lower bounds of all its
  coordinates and then the upper ones, so that Example 8 shows the plausible
  bounds of `beta` as `[[-0.09952025  0.24172931], [0.52304277 1.28427527]]`,
  which reads as one interval for each coefficient. Lay the table out so that
  each coordinate's lower and upper bounds stand together, with fewer
  digits (one row per VBMC coordinate, say), re-execute Example 8
  (`dev/scripts/execute_notebooks.py --only 8`) and check the quickstart's
  description of the table. `test_mapping_validation_and_string_table`
  (`pyvbmc/testing/pymc/test_target.py`) asserts only the header and two
  coordinate names. Left out of 1.5.0, whose code is settled (PI,
  2026-10-12).

## Dependencies and working rules

- A release at least once a year: the old-release reminder tells a user of
  a release more than a year old that a newer version may exist, so a longer
  gap reminds users of the latest release (PI, 2026-09-30;
  [plans/version-check.md](plans/version-check.md), D1).
- "In scope for 1.5" is empty when 1.5 is done: each of its items is either
  done or decided not to be done, with the reason recorded where its work
  is recorded. A finding is not parked in this file to wait; it is fixed or
  ruled not to be fixed (PI, 2026-09-23).
- Evaluate numerical proposals before choosing defaults or reported
  estimators. If an accepted change moves default trajectories, update the
  affected references after assessment and preserve the old ones.
- Whatever consumes a run pool (the Phase 2 estimator, a stacking
  comparison) runs on a machine only after every artifact of that pool
  re-verifies there (`svbmc_pool_run.py verify`): the September pool, the
  asset of the draft release `svbmc-pool-20260914`, and the release pools
  alike.
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
  label tag. The earlier golden references remain the record of the
  golden suite; the reference that replaced them after the port review,
  `reference_2400_20261007`, runs the whole `production` suite.
  The E5 report's F3 section records how the difference was found.
- Use feature branches from `main` for implementation and planning
  changes, with pull requests into `main`. Leave unrelated work intact.

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
over-corrects on heavy tails; since 2026-10-06 the headline of a noisy
stack is the two-level shrinkage estimate
([the reading](results/2026-10-06-arm-3-reading.md)), chosen on the bias
that stacking adds to its inputs; under it, the gate still fails on
Student D8 at small `M` (the campaign plan says where). The
[stage D report](results/2026-09-15-svbmc-pool-comparison.md) reads
the numbers, the [campaign plan](plans/svbmc-benchmark-campaign.md)
owns the design, worklog and the PI's decisions, the
[experiments README](experiments/svbmc_pool/README.md) indexes the
tracked outputs, and the per-cell outputs are the three assets of the
draft release `svbmc-analyses-20260915` (the pool itself the asset of
`svbmc-pool-20260914`). The active reference is
**`reference_2400_20261007`**: the release gate's after arm, 2400 runs of
the `production` suite at seeds 0–99, made by the release code. Its
[promotion record](golden/promotion_20261007/README.md) owns the
assessment, hashes and passed gates, and the
[Slurm plan](plans/slurm-benchmark-support.md) the campaigns' execution.
The previous reference, `reference_990_20260913` of the `golden` suite, is
preserved with its [promotion record](golden/promotion_20260913/README.md).
80 runs of six noisy configurations at production budgets, made
with the code from before the port review's fixes (logistic regression and
high-noise Rosenbrock at twenty seeds; low-noise Rosenbrock, Student-t,
timing and multisensory at ten), exist locally under
`golden/production_noisy_20260918/` (listed in `dev/scripts/runs/LOCAL.md`),
run on 2026-09-18 and 2026-09-19 as the baseline arm of the F2 comparison
and after its stop; the active reference does not include them. The
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
