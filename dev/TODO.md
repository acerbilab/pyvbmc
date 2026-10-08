# PyVBMC 1.5: remaining work and scope

Updated 2026-10-07. These lists describe scope, not priority or execution
order; independent workstreams can be picked up in any order. Inclusion in
scope does not settle an implementation design or launch a campaign.
Completed 1.5 work is not listed here, apart from the active reference and
the locally held artifacts that the last section describes: the
[roadmap](plans/modernization-roadmap.md) retains it, and each item's plan
records its execution.

## In scope for 1.5

- [ ] **Review of the tips.** The tips that `VBMC` and `SVBMC` print when a
  run starts (`show_tips`; catalogs in `pyvbmc/vbmc/_tip_catalog.py` and
  `pyvbmc/svbmc/_tip_catalog.py`) are to be reviewed before the release (PI,
  2026-09-30). The [tips plan](plans/runtime-tips.md) records their policy,
  catalog, wording and acceptance checks. The tips are described in the
  section "Startup tips" of the `VBMC` API page, a paragraph of the
  quickstart, the `SVBMC` page and the description of `show_tips` in
  `basic_vbmc_options.ini`; a change to what the feature does also reaches
  the changelog's "Tips" entry and the "What's new" blocks. The stored
  outputs of Examples 4, 5 and 7 show tips, and are executed again at the
  release. The old-release reminder
  ([plans/version-check.md](plans/version-check.md)) shares the tips' slot
  and switches, and the review covers it: it is described in the same places
  and in the FAQ's entry on newer versions, the `check_for_updates` API page
  and the changelog's "Update reminders" entry. The PI's rulings so far,
  and what remains open (the reading of the wording of every tip), are in
  the tips plan's last section, "Review before the 1.5 release". The
  switch of the S-VBMC headline (2026-10-06) rewrote `noisy_elbo` and gave
  it a link to the `SVBMC` page's account of the headline.

- [ ] **Release documentation and validation.** The final pass on the
  settled release code; each step's procedure is in the roadmap's
  [pre-release checklist](plans/modernization-roadmap.md#pre-release-checklist):
  - re-execute the example notebooks at the release commit and commit their
    outputs, Example 7 with its account of the headline as the switch of
    2026-10-06 rewrote it;
  - the Sphinx build, `linkcheck`, the rendered pages and the agent skill on
    the settled code, with a `linkcheck` again after the merge into `main`
    (the API and tutorial review was done on 2026-10-07 in PR #186; the
    roadmap's checklist records it and that day's build);
  - a delta pass of the [release sweep](results/2026-09-27-release-sweep.md)
    over the diff since `46200293` at the settled release commit; the
    documentation findings of 2026-10-07 are corrected in their source
    files and recorded in the sweep's ledger;
  - the final tests, the CI matrix and the package checks;
  - the artifacts that attach to the release as archives (PI, 2026-09-28;
    the [Slurm plan](plans/slurm-benchmark-support.md), "Records and
    hand-back"): the release gate's after-arm population and its pools,
    each built apart from its draft release from the numeric files and the
    redacted copies by `scripts/hpc/campaign_public.sh`, in the operator's
    account after the redaction
    (the stacking's tracked copies hold every cell already);
    the draft releases of the campaigns stay drafts, since their raw
    archives hold the cluster's details and the operator's paths; the
    replay fingerprints and the earlier references' traces are backed up
    in a draft release of their own;
  - the references to `dev-next` that change with the release merge;
  - `RELEASE_DATE` in `pyvbmc/_release.py` set to the date of the
    changelog's release heading, in the release pull request (`AGENTS.md`,
    "Release date").

- [ ] **Publish and link the PyVBMC film.** The film is ready for upload:
  its seventh draft was approved and mastered on 2026-10-05, and it has
  not yet been uploaded to YouTube (PI, 2026-10-07). Upload it first,
  then follow PyBADS's treatment of its film: add the final YouTube
  link to `README.md` and `docsrc/source/index.rst`, in the introduction
  and alongside the explanation of the method (`How does it work?` in
  the README, `Example run` on the docs' index). Then add the video to
  the lab's [tools for fitting models to data](https://acerbilab.org/model-fitting/)
  page.

  The film's source is on `feat-3d-animation`, under
  `docsrc/source/_static/vbmc3d/`; its `README.md`, `NOTES.md` and
  `TODO.md` hold the design, production record and publishing steps.
  `dev/scripts/runs/LOCAL.md` lists the masters and the publishing kit.
  The film quotes the [matched MCMC budget](plans/matched-mcmc-budget.md).

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

- [ ] **Standalone `svbmc` compatibility release.** It depends on
  `pyvbmc[torch]>=1.5` and forwards old `svbmc` imports to the integrated
  implementation. S-VBMC is already available through PyVBMC; this serves
  users of the old package. See the [integration plan](plans/svbmc-integration.md).
- [ ] **conda-forge.** Once 1.5 is on PyPI, the version bot of the
  [feedstock](https://github.com/conda-forge/pyvbmc-feedstock) opens a PR
  that bumps the version and the source hash. Its PR 11 (open on
  2026-10-06) proposes `bot: {automerge: true, inspection:
  update-grayskull}` in the feedstock's `conda-forge.yml`, as gpyreg's and
  PyBADS's feedstocks have, so that the bot's PR takes the requirements
  from the PyPI metadata and merges itself once the feedstock's CI passes.
  Under those settings the bot's PR for gpyreg 1.4.0
  (`conda-forge/gpyreg-feedstock` #13) changed the version and the hash
  alone, though its analysis listed the run requirements that 1.4.0 had
  dropped, and merged itself; a second PR, under build number 1, removed
  them (#14). So the recipe's requirements, still those of 1.0.4, are
  updated by hand, in a PR opened before the bot's or pushed to it before
  it merges (removing `[bot-automerge]` from the bot's title keeps it from
  merging itself), as PyBADS 1.5.0 did
  (`conda-forge/pybads-feedstock` #12, merged 2026-10-06). Check the published package's requirements
  against `pyproject.toml`: 1.5 needs gpyreg 1.4.0 or later, on
  conda-forge since 2026-09-30, and adds `filelock`, `platformdirs` and
  `threadpoolctl`. conda-forge's
  `python_min`, 3.11 on 2026-09-25, is the conda package's Python floor,
  where PyPI's is 3.10.
- [ ] **Respond to issue #138 about RNG control** with the released API/docs.
  See the [post-release follow-up](plans/modernization-roadmap.md#post-release-follow-up).

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
- Use feature branches for implementation. Planning, proposal, handoff and
  status edits belong on `dev-next`. Leave unrelated work intact.
- Dependabot's PRs target `main`, whose `tests.yml` and `merge-tests.yml`
  `dev-next` has replaced (its tests run through `test-matrix.yml`, which
  Dependabot does not see from `main`). Until `dev-next` merges into
  `main`, a bump is applied on `dev-next`, in every workflow that uses the
  action, and its PR closed: merged into `main`, it makes those files
  conflict in the release merge.

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
