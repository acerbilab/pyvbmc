# PyVBMC 1.5: remaining work and scope

Updated 2026-10-06. These lists describe scope, not priority or execution
order; independent workstreams can be picked up in any order. Inclusion in
scope does not settle an implementation design or launch a campaign.
Completed 1.5 work is not listed here, apart from the active reference and
the locally held artifacts that the last section describes: the
[roadmap](plans/modernization-roadmap.md) retains it, and each item's plan
records its execution.

## In scope for 1.5

- [ ] **Changelog for 1.5.** `CHANGELOG.md` (root, Keep a Changelog layout)
  covers the changes since `v1.0.4`, written to the bar that `AGENTS.md`
  ("Changelog") sets (condensed to it on 2026-09-30 and read by the PI the
  same day; the worklog of the
  [port review plan](plans/port-correctness-review.md) records how the
  earlier, longer text was written and checked). "Runs are faster" gives
  the speed against 1.0.4 measured on 2026-10-05/06 (the Arm 0 item below;
  the PI's wording, 2026-10-06;
  [results](results/2026-10-06-speed-against-1.0.4.md)), as do the "What's
  new" blocks below. Open before the release: the S-VBMC speed figure, to be
  measured again on the release benchmark. The "What's new in PyVBMC 1.5"
  blocks of `README.md` and `docsrc/source/index.rst` summarize the
  changelog and move with it, its timings included; the docs' link to the
  changelog resolves once the file is on `main`.

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

- [ ] **Slurm/HPC benchmark support.** Reproducible submission, resource
  settings, resumption and result collection on the Turso cluster. The
  design is the [Slurm plan](plans/slurm-benchmark-support.md), with the
  PI's decisions of 2026-09-25: one campaign contract (`prepare`, `cases`,
  a `worker` that claims its case, `verify`) that the pool, population and
  stacking harnesses meet, and generic scripts beside the pool campaign's
  under `scripts/hpc/`. The PI reviewed the plan on 2026-09-25. Phases 2
  to 5, the redaction and guide of Phase 7, and the tools of the plan's
  Phase 9 and of the promotion of the new reference (the two items below)
  are on `dev-next`, from the branch `feat-slurm-campaigns`, reviewed
  twice and fixed. Phases 1b and 6 (the survey, the source trees, the
  frozen environment and its check, and the smoke campaigns, which
  measured every job's time and memory) ran on the cluster from
  2026-09-28 to 10-01. Phase 8, the release gate's campaigns, ran on
  2026-10-02 from the release commit `ff3ed014` (PI, 2026-10-01) and was
  handed back on 2026-10-04 (PR #181): every case of the four campaigns
  verified, and their redacted copies are under
  `experiments/release_gate_20261002/`. Phase 9, the replay fingerprints
  and the six gate runs of the two items below, ran on the developer's
  machine on 2026-10-04 and met its acceptance (the plan's worklog). What
  remains is the smaller fixes that the plan's worklog of 2026-09-30
  leaves for after the merge, which change nothing a fingerprint computes
  and land before the promotion's `prepare`, and the review's optional
  findings, which await the PI's ruling. The smaller fixes are done with
  the promotion's work (PI, 2026-10-05), which takes the after arm, since
  MATLAB's end of warm-up stays (PI, 2026-10-06): they reach no batch that
  runs from its named commits. An
  edit to the package's text, the tips' wording among them, lands after
  the promotion of the new reference, whose check compares the package's
  files with the after arm's (the plan's decision 13). See
  [HPC support](plans/modernization-roadmap.md#benchmark-coverage-and-hpc-support).

- [ ] **The golden references after the port review.** Several fixes of
  the [port correctness review](plans/port-correctness-review.md) move
  default trajectories; its
  [ledger](results/2026-09-23-port-correctness-review.md) lists them. The
  golden reference `reference_990_20260913` and the production-budget
  runs of 2026-09-18 and 09-19 (below, "Completed baseline and local
  artifacts") describe the code from before them, so
  `scripts/golden_replay.py` compares a run with trajectories the code no
  longer follows. The reference that replaces them (PI, 2026-09-25; the
  [Slurm plan](plans/slurm-benchmark-support.md), "What runs where") is
  to run the `production` suite alone at 100 seeds per configuration on
  Turso with the release code, beside a before arm of the same runs at
  `f91fdf0` with gpyreg 1.2.1; exact replay is to come from one seed per
  configuration generated on the developer's machine, with the six seeded
  gate runs of the item below. The moving fixes of the wave-1 and wave-2
  passes, W5-1 and W5-6 were not read for accuracy on the benchmark
  targets (the ledger, "The fixes that move default trajectories"); the
  comparison of the two arms is their measure. This item owns that
  assessment, by the method of the
  [population plan](plans/final-population-benchmark.md), and the
  promotion of the new reference with its fingerprints, preserving the old
  references (the working rule below). The two arms ran on 2026-10-02
  (the plan's Phase 8; `experiments/release_gate_20261002/`), and their
  comparison (`analyze_population_run.py --arms`) is
  `experiments/release_gate_20261002_assessment/`, for the PI's reading;
  the promotion takes the assessment the PI accepts, by its SHA-256
  (`assessment.json`'s, with LF line endings:
  `2625fc50cb035ccabe7e5ec50bc6ebc7743c515eec08240c15b361a14420898d`).
  It rejects none of the 96 confirmatory tests; its KS screen flags the
  evaluation counts of `cigar_D4` and `rosenbrock_D2_noise3_production`.
  The noisy Rosenbrock target is worse in the release code on every
  measure (usable 72 times against 87): two of the port review's fixes
  end its warm-up about three iterations earlier, as MATLAB does
  ([the note](results/2026-10-05-noisy-rosenbrock-warmup.md), with a
  paired experiment at 50 seeds).
  MATLAB's end of warm-up stays (PI, 2026-10-06), read on the results of
  Arm 3, the release code with the port's end of warm-up
  ([the reading](results/2026-10-06-arm-3-reading.md)): the promotion
  takes the after arm, and its record states the cost on the noisy
  Rosenbrock target. The PI's acceptance of the assessment remains. The
  promotion is
  `scripts/reference_promote.py` (2026-09-28), in
  the manner of `golden/promotion_20260913/promote.py`, since
  `scripts/reference_join.py` extends a reference and refuses any overlap
  with it: `fingerprints` makes the Phase 9 runs of seed 0 against the
  after arm's envelopes, `prepare` checks the previous reference, the after
  arm, the accepted assessment, the fingerprints and the gate runs and
  writes the record, `replay` replays the new defaults, and `publish`
  replaces the sidecars of `golden/baseline/` and changes
  `golden_replay.py`'s `DEFAULT_BASELINE` and `DEFAULT_CONFIGS` together
  with `AGENTS.md` ("Trajectories"), the current reference's section and
  the `golden_replay.py` entry of `dev/README.md`, and `golden/README.md`
  (its `dev/README.md` entry gives the steps). Its test module
  runs a whole promotion on small array-mode campaigns. It rewrites those
  passages only as they stood when it was written (a SHA-256 guard), so an
  edit to one before the promotion is carried into its template. The PI's
  rulings of 2026-09-28 (the Slurm plan, "The populations" and Phase 9):
  the 24 fingerprints are judged as a set, since a correct run lies
  outside its envelope now and then (4.2 % of the runs of
  `reference_990_20260913`), and more of them outside than the after arm's
  own rate makes plausible refuses; the code the promotion compares with
  the after arm's leaves out the package's tests and S-VBMC; and the
  script moves into its record at the promotion. Phase 9 ran on
  2026-10-04 (the Slurm plan's worklog): one fingerprint of 24,
  `corr_D5`'s, lies outside its envelope, within the after arm's own
  rate, and the gate runs of the item below reproduced.

- [ ] **A seeded gate run with a prior.** The four seeded runs that gate
  the port review's fix passes
  (`experiments/port_review_20260919/verification/scripts/wave2_fixpass_gate_runs.py`)
  pass no prior with `prior=`, so what a run does with a prior object
  rested on unit tests: the check of its support against the hard bounds,
  and the support that `SciPy` and `Product` read from their distribution
  at every call. The script holds two more runs, on a two-dimensional
  Rosenbrock likelihood with the hard bounds -2.9 and 3.3: the FAQ's list
  of `uniform` marginals built from the hard bounds, whose support ends
  one rounding short of the upper bound and so passes the check within its
  slack, and a `SplineTrapezoidal` on the same bounds. On 2026-09-25 each
  reproduced bit for bit in two recordings, and their ELBOs came out 0.34
  and 0.37 nats below the log evidence computed by quadrature, the gap the
  unbounded Rosenbrock run of the same script shows without a prior object
  (0.40). The script's six runs join the release gate's records (PI,
  2026-09-21; the wave-5
  [ledger](experiments/port_review_20260919/verification/wave5.md), "The
  independent check of the pass") as part of the replay fingerprints of
  the new reference (the item above): they are run again with the release
  code, wrapped so that their record carries the commit, the gpyreg
  source, the thread settings and the host, which the script does not
  record (PI, 2026-09-25; the
  [Slurm plan](plans/slurm-benchmark-support.md), Phase 9). The wrapper is
  `scripts/seeded_gate_runs.py` (2026-09-28): it
  records the six runs twice, each in a fresh process with one BLAS thread
  and `performance_calibration="off"`, compares them with the script's own
  `compare`, and writes each recording's identity (the commit, the gpyreg
  checkout, the thread settings, the host) beside them. On 2026-09-28 it
  recorded them at `4efee154` on `feat-slurm-campaigns` with gpyreg
  `d96d0d9`, identical
  in all 138 arrays; that run checked the wrapper and is no record. The
  runs that enter the records are Phase 9's, made with the after arm's
  code; they ran on 2026-10-04 at `dce4e18c`, whose package is the
  release commit's, and reproduced in all 138 arrays.

- [ ] **Final large-scale check before the release (the gate).** Once no
  algorithmic work on 1.5 remains, regenerate the VBMC run pools on the test
  targets with the release code on the cluster, the latest `dev-next` at the
  launch, after which only the documentation and the headline selection change;
  320 filtered runs per condition (PI, 2026-09-25; the [Slurm
  plan](plans/slurm-benchmark-support.md)): the present pools of 100 noisy and
  50 noiseless runs reuse each run 1.6 to 3.2 times at `M = 16` and 3.2 to 6.4
  times at `M = 32`, and ten disjoint subsets of 32 need 320 runs. Then stack
  them on the cluster in disjoint subsets at every `M`, with both arms of the
  [campaign](plans/svbmc-benchmark-campaign.md) at `M` = 2, 4, 8 and 16 and the
  integrated arm alone at 3, 5 and 32, and check that the campaign's results
  hold: the acceptance criteria of the comparison, the added-bias ranges of the
  headline note, and the switch of the headline to the two-level shrinkage. The
  gate tests PyVBMC 1.5 end to end, not S-VBMC
  alone: the pools are VBMC runs of the release code, and each run's record
  carries its convergence status and its posterior and evidence metrics against
  the ground truth. Uses the pool generator and stacking harness under
  `dev/scripts/` through the cluster workflow of the HPC item above. PI,
  2026-09-16. The pools, the stacking and the two analyses ran on 2026-10-02
  (`experiments/release_gate_20261002/pools/` and `stacking/`; the analyses'
  outputs are an asset of the draft release `release-gate-stacking-20261002`,
  since they hold the cluster's details). Read against the campaign's criteria
  on 2026-10-04 (the stacking's `summary.md`): the median over each condition's
  cells of the largest weight difference between the two arms is 0.007 to 0.027
  (criterion 1), no cell is flagged (criterion 2), and stacking takes 0.19 to
  0.53 of the original's time (criterion 4); criterion 3 fails on `student_D8`
  alone, at every `M` where both arms ran (2, 4, 8 and 16), where the capped
  headline's median bias is −0.71 to −1.39 nats against the original's −0.01 to
  +0.28, as on the stage D pools. The added bias of each estimate, on these
  pools and on those of Arm 3 (the release code with the port's end of
  warm-up), and the switch of the noisy headline to the two-level shrinkage
  that the PI made on them (2026-10-06), are in
  [the reading](results/2026-10-06-arm-3-reading.md).

- [ ] **1.5 against 1.0.4 (Arm 0).** The reference populations compare the
  release code with `f91fdf0`, which held the modernization already; a
  user upgrades from 1.0.4, so the release's claims of accuracy and speed
  are made against 1.0.4 (PI, 2026-10-04). Arm 0 runs 1.0.4 as released
  (`v1.0.4`, with gpyreg `v1.0.4`, each version on its own defaults) on
  the `production` suite at seeds 0–99 on the cluster, compared seed by
  seed with the after arm. Speed was measured apart on the developer's
  machine on 2026-10-05/06
  ([results](results/2026-10-06-speed-against-1.0.4.md)): the release
  code's runs take 3.7 times less time than 1.0.4's on the noiseless
  configurations and 2.2 times less on the noisy ones. The
  [plan](plans/arm-1.0.4-comparison.md) holds the decisions, the harness's
  legacy profile, the rescoring across harness commits and the checklist.
  It ran in one batch with Arm 3, the release code with the port's end of
  warm-up ([its plan](plans/arm-3-warmup-comparison.md)). MATLAB's end of
  warm-up stays (PI, 2026-10-06), so the after arm remains the release code
  that Arm 0 is compared with. The campaign ran on
  2026-10-05/06: 2398 cases verified and 2 failed in 1.0.4, which the
  comparison counts as runs that gave no usable posterior (the plan's
  decision 9). The changelog's, the README's and the release notes'
  statements of accuracy against 1.0.4 wait for the comparison; the
  changelog's statement of speed is in (2026-10-06).

- [ ] **Links to the lab.** PyBADS settled before its 1.5.0 release where
  its published texts link the lab (PI, 2026-10-06; the convention "Links
  to the lab" in PyBADS's `AGENTS.md`, applied in PyBADS's commit
  "docs: links to the lab's tools for fitting models to data, and to its
  group page"), and PyVBMC takes the same pass (PI, 2026-10-06):
  - a link that names Luigi Acerbi goes to his personal page,
    https://lacerbi.github.io/, and one that names the lab or its members
    to the group's page,
    https://www.helsinki.fi/en/researchgroups/machine-and-human-intelligence,
    not to its `/people` page, which `README.md`, `docsrc/source/index.rst`
    and `docsrc/source/about_us.rst` link now (`about_us.rst` for each
    developer as well);
  - the lab's page of its model-fitting tools,
    https://acerbilab.org/model-fitting/ (link text "tools for fitting
    models to data"), is linked under the title of `README.md` and
    `index.rst`, in the docs' footer (`extra_footer` of sphinx-book-theme),
    in a `[project.urls]` table of `pyproject.toml`, which has none, and in
    each paragraph that sends the reader to another of the lab's methods
    (PyBADS, PyIBS, MATLAB VBMC): the FAQ, the skill, the examples;
  - a tip that sends the reader to another method links the page too:
    the `pybads` tip links PyBADS's documentation alone.

  PyBADS's README opens with "PyBADS is one of the open-source tools for
  fitting models to data from Luigi Acerbi's group at the University of
  Helsinki. Check out our other tools, such as PyVBMC for the posterior
  and the model evidence and PyIBS for models that can only be simulated."
  A change to a tip reaches the review of the tips (above) and the stored
  outputs of the examples that show tips.

- [ ] **Release documentation and validation.** The final pass on the
  settled release code; each step's procedure is in the roadmap's
  [pre-release checklist](plans/modernization-roadmap.md#pre-release-checklist):
  - re-execute the example notebooks at the release commit and commit their
    outputs, Example 7 with its account of the headline as the switch of
    2026-10-06 rewrote it;
  - the API and tutorial review, the Sphinx build, `linkcheck`, the rendered
    pages and the agent skill, with a `linkcheck` again after the merge into
    `main`;
  - a delta pass of the [release sweep](results/2026-09-27-release-sweep.md)
    over the diff since `46200293`;
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

- [ ] **The 3D animation of a PyVBMC run (`feat-3d-animation`).** An
  interactive three.js page that plays back a recorded two-dimensional run
  as a rotating landscape of the log density: the GP surrogate, the
  evaluations, the acquisition function while points are chosen, the
  variational mixture and, at the end, the true target. A second page ends
  with the final posterior as the V of the PyVBMC wordmark;
  `scripts/record.mjs` records either page as MP4, GIF or PNG frames, and
  `dev/scripts/export_animation_trace.py` writes the trace of a real run
  (the exporter is on `dev-next` too, for the
  [matched MCMC budget](plans/matched-mcmc-budget.md), whose number the
  video quotes).
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
- **The acceptance of a rotoscaling.** The undo check keeps a warp whose
  refit ELBO exceeds the previous iteration's by `warp_tol_improvement`, as
  MATLAB VBMC does, and a refit posterior that spreads where the refit GP
  has no data can pass it on a gain of the surrogate: Example 2 keeps such
  a warp, reports an ELBO of −0.82 where the true log evidence is −1.836,
  and recovers two iterations later. One run in the 990 of the golden
  reference does the same. A warp only reparameterizes the space, so a
  guard could undo a warp whose refit posterior moves far from the one
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
  label tag. The existing golden references remain the record of the
  golden suite; the reference that replaces them after the port review is
  to run the whole `production` suite (the golden references item above).
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
([the reading](results/2026-10-06-arm-3-reading.md)). The
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
history. 80 runs of six noisy configurations at production budgets, made
with the code from before the port review's fixes (logistic regression and
high-noise Rosenbrock at twenty seeds; low-noise Rosenbrock, Student-t,
timing and multisensory at ten), exist locally under
`golden/production_noisy_20260918/` (listed in `dev/scripts/runs/LOCAL.md`),
run on 2026-09-18 and 2026-09-19 as the baseline arm of the F2 comparison
and after its stop; the new reference does not include them (the golden
references item above). The
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
