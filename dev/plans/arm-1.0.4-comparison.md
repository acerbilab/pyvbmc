# Arm 0: PyVBMC 1.5 against 1.0.4 on the release gate's populations

Created 2026-10-04. Status: **implemented, reviewed and merged into
`dev-next` (`9b84a1ae`); the local runs of 1.0.4 done; the brief and the
campaign open**.

## Purpose

The reference populations of the release gate
([Slurm plan](slurm-benchmark-support.md), Phase 8) compare the release
code with `f91fdf0`, `dev-next` on 2026-09-19, which already held the
modernization's speed-ups and most of its fixes: that comparison measures
the port correctness review and nothing earlier. A user upgrades from
PyVBMC 1.0.4, so the release's claims of accuracy and speed are made
against 1.0.4. Arm 0 is a third population, 1.0.4 on the same cases,
compared seed by seed with the after arm, the release code's population.

What exists against 1.0.4 before Arm 0: the first golden population,
`baseline_20260903` (on `5020879`, 1.0.4 with `seed=` and three fixes of
2026-09-02; twelve noiseless configurations and two noisy ones at the 2020
paper's budget, seeds 0–19), whose sidecars and traces the developer's
machine holds (`dev/scripts/runs/LOCAL.md`). On its twelve noiseless
configurations, the after arm's tracked copies at seeds 0–19 count 240
usable runs against its 239 (usable: evidence error below 1, gsKL below 1
and MMTV below 0.2, as the population summaries count them), and every
configuration's median evidence error, gsKL and MMTV is equal or a little
better in the release; the two sets of metrics come from two machines and
from the metric code of each. No 1.0.4 run exists of a noisy configuration
at production defaults.

## Decisions (PI, 2026-10-04)

1. **The code is 1.0.4 as released**, the tag `v1.0.4` (`0bb8b8f5`),
   unmodified. It has no `seed=`: every draw goes through NumPy's global
   state, which the harness seeds with the case's seed before it
   constructs `VBMC`. Should 1.0.4 fail in the campaign environment, the
   compatibility goes into the harness's shims for it, which change no
   numerics, and is recorded here; the 1.0.4 tree stays the tag, and the
   environment is not changed, so that the arms differ in their code
   alone.
2. **gpyreg `v1.0.4` (`3b1300a`)**, the release that 1.0.4's users had
   until September 2026.
3. **Each version on its own defaults.** 1.0.4's noisy defaults are the
   production ones (75 (D + 2) evaluations, a stability window of 90, VIQR,
   the in-loop updates), so the `production` suite measures what a user of
   each version gets. 1.0.4 has no guard on the final boost, which it
   always keeps, and no machine calibration.
4. **The whole `production` suite at seeds 0–99**, 2400 runs on the
   cluster, on the after arm's node family and environment.
5. **Speed is measured apart**, on the developer's machine, one process at
   a time: the cluster's packed nodes give no clean timing.
6. The release's statements of accuracy and speed against 1.0.4 (the
   changelog, the README and the release notes) wait for Arm 0; the
   promotion of the new reference, the S-VBMC headline and the
   documentation do not.
7. **Arm 0's data stay internal**: its archive and tracked copies, as for
   the before arm, and no public asset in the release.
8. **The speed measurement** is the one under "Speed" below.

## Constraints

**Nothing that builds a run changes.** From the launch of the release gate
to the promotion of the new reference, no file of `NUMERIC_PATHS`
(`dev/scripts/reference_promote.py`: the package but for its tests and
S-VBMC, `golden_trace.py`, `benchmark_targets.py`, `profile_run.py`,
`dev/scripts/data`) changes (the Slurm plan's decision 13). Every adaptation
to 1.0.4 is therefore in `population_run.py`, which installs it in the
worker's process, and in the comparison's tools.

**Arm 0's harness commit holds the after arm's run code.** `rescore-arms`
rescores the after arm with the package of Arm 0's harness checkout and
must reproduce the after arm's in-run metrics, so the commit that runs Arm
0 holds the after arm's package code (all of `pyvbmc/` but its tests and
S-VBMC), its files that build and score a run and its
`campaign_requirements.txt`. The promotion does not wait for Arm 0
(decision 6), and the package's text may change after the promotion (the
tips' wording; `pyvbmc/_release.py` in the release pull request), so the
commit is cut from `dev-next` before any such change lands, or from the
after arm's commit with the harness change merged onto it. `prepare
--compare` refuses any other before a case runs. The commit enters
`dev-next` by a merge, not a squash: the comparison compares the package
code of the two commits in the history of the checkout it runs in. The
campaign's harness checkout stays at the commit from its first
submission to its hand-back, so `dev-next` moves on once the commit is
named: the head of `dev-next` once the pairing that Arm 3 needs has
merged (the checklist), and the tips' wording and the release date wait
until then. The S-VBMC headline does not, since its code, docstring and
tip lie under `pyvbmc/svbmc/`, which the check leaves out. The one change
to the release code's numerics still open, the end of warm-up, is decided
on Arm 3, which runs in the same batch; Arm 0's runs stand whatever the
decision, and only their comparison changes if the release code does (PI,
2026-10-05; [Arm 3's plan](arm-3-warmup-comparison.md)).

## Design

### The legacy profile

`population_run.LEGACY_PACKAGES` names, by the commit of the package tree,
the releases that have no `seed=`, draw every random number from NumPy's
global state and refuse an option they do not define: `0bb8b8f5…` (1.0.4).
Its profile holds the release, the gpyreg release it runs with (`1.0.4`),
the options of `DEFAULT_OPTIONS` it does not define and the weight penalty
of its final boost. A campaign whose package tree is at such a commit is a
legacy campaign:

- `prepare` requires the tree clean, and takes the gpyreg release from the
  profile in place of the package tree's `pyproject.toml` (1.0.4's says
  `gpyreg >= 0.1.0`), exactly;
- its options are `DEFAULT_OPTIONS` without those the package does not
  define (`vectorized_target`, `performance_calibration`,
  `tol_elcbo_boost`), and `--options` may name only options that the
  package's option files define;
- `--pair` is refused, as for every campaign whose package is not the
  harness checkout's, and `--compare` names the campaign of the release
  code it will be compared with (a campaign directory or its tracked
  copies), and is required: `prepare` refuses there what the comparison
  would refuse after the runs (`compare_problems`: another allocation or
  confirmatory family, other options but those the package does not
  define, other files that build and score a run, other environment
  versions, another node family, a harness checkout whose package code is
  not the one that campaign ran);
- its tracked copies hold the rescoring that `rescore-arms` writes.

### The worker

In a legacy campaign the worker installs, before `golden_trace.run_task`
constructs the run:

- **construction**: `pyvbmc.VBMC`, which `run_task` imports at call time,
  becomes a factory that seeds NumPy's global state with the case's seed
  (`np.random.seed(seed)`, 1.0.4's own way of fixing a run), drops the
  options the package does not define (`run_task` adds
  `performance_calibration`), and constructs the package's `VBMC`; the
  sidecar's `requested_options` keep what `run_task` asked for and its
  `effective_options` what 1.0.4 applied;
- **deterministic metrics**: 1.0.4's posterior draws from the global state
  and ignores the `vp.rng` that `benchmark_targets.metrics` sets, so the
  metrics run with the global state seeded with
  `benchmark_targets.DIAG_SEED`, and the state is restored after them; the
  in-run metrics and those of the rebuilt posterior then agree exactly, as
  the worker requires;
- **the boost capture**: the generator states it records are NumPy's
  global state, the posteriors are copied without a generator, and the
  boost's options keep 1.0.4's weight penalty (`weight_penalty = 0.1`),
  which the checks of a legacy case expect in place of zero; the tolerance
  is none;
- **the rebuilt posterior**: constructed without the `rng` argument that
  1.0.4's `VariationalPosterior` lacks.

`verify` runs in the legacy campaign's own trees, as for any campaign.

### Rescoring across harness commits

A legacy campaign does not rescore itself. `population_run.py
rescore-arms --out ARM0` runs in a process of the release code
(`PYVBMC_SOURCE` unset, gpyreg the after arm's), from Arm 0's harness
checkout at its commit, which the comparison requires, and rescores the
campaign that `--compare` named (or `--campaign`) and then Arm 0, writing,
into Arm 0, `rescored/<name>.json` for both and `rescoring.json` with its
identity, as `rescore` does in a campaign of the release code; Arm 0's
tracked copies declare them, so that the redaction writes them for the
repository. A campaign whose scoring code is the process's
(`scoring_mismatches`: the same source identity, or else, by
`scoring_differences`, clean trees whose package code but its tests and
S-VBMC is equal between the two commits, gpyreg at one commit, and both
identities holding the files that build and score a run, equal, and the
same imported versions) must reproduce its in-run metrics exactly, so the
after arm's 2400 cases test that the rescoring scores as the release code
did, and one such campaign is required. It runs on the cluster, where both
raw campaigns are, as a batch job on the campaigns' node family, after Arm
0's finish, and again after any later finish of Arm 0, whose verification
report the rescoring names. In the second batch it also rescores Arm 3
([its plan](arm-3-warmup-comparison.md)), so that a comparison of Arm 0
with Arm 3, should the PI adopt Arm 3's end of warm-up, has both arms
scored by one process: `--campaign` names the after arm and then Arm 3,
and the job runs after Arm 3's finish too.

### The comparison

`analyze_population_run.py --arms ARM0 AFTER` compares a legacy reference
with the after arm, either as campaign directories or as their tracked
copies, with the rescoring that Arm 0 holds (the default of `--rescoring`
for a legacy reference). For such a pair it allows options that differ
only by those the legacy package does not define, and harness checkouts at
different commits whose run-building files and imported versions are equal
(`legacy_pair_differences`), and it requires a rescoring that scores as the
candidate's code, made from the legacy arm's harness checkout, rather than
one of the candidate's whole identity. It still requires the after arm's
rescored metrics to equal its in-run metrics, the same allocation and
confirmatory family, one node family, and a legacy profile in the
reference's manifest that its package commit names. Its boost reports
have no tolerance and are expected to keep every boost they made. The
report records the legacy profile (`legacy_reference`), a key that the
assessment of two arms of the harness's own code does not hold;
`reference_promote.py` refuses an assessment of a legacy reference.

### What the operator runs

The operator's guide (`dev/scripts/hpc/README.md`, "Arm 0: PyVBMC 1.0.4")
gives the commands: the trees (`v1.0.4` as a detached worktree of the
harness checkout, gpyreg `v1.0.4`), the harness checkout moved to Arm 0's
commit, the environment check there, the after arm's directory restored
from its archive where it is gone, the campaign as one arm with
`--compare` and `cigar_D15_exhaust` as its own subset, its finish,
`rescore-arms` and its archive part, and the hand-back, which is the second
batch's with Arm 3's campaigns ([the brief](release-gate-handoff-2.md)). A
brief in the
manner of `release-gate-handoff.md` gives the order, the commit and the
limits.

### Limits from the local runs

The local runs give each configuration's time and memory with 1.0.4 on the
developer's machine; the fingerprints of the Slurm plan's Phase 9 give the
release code's on the same machine, at the same seed. The ratio of the two
times, per configuration, scales the release gate's cluster limits
(`POP_TIME`, `CIGAR_TIME`, `POP_MEM`) into Arm 0's (`ARM0_TIME`,
`ARM0_CIGAR_TIME`, `ARM0_MEM`), with the largest ratio of the configurations
that each limit covers and the margin those limits already hold; the canary
checks them before the rest is submitted.

From the local runs of 2026-10-04/05 (the worklog): 1.0.4 took 1.5 to 5.2
times the release code's time per configuration, and the slowest cluster
run of each configuration in the after arm, so scaled, gives at most 33
minutes outside `cigar_D15_exhaust` (the noisy `logreg_D5`) and 141 minutes
for it. The limits, with a margin of about two for one seed's ratio:

| Setting | Value | Covers |
|---|---|---|
| `ARM0_TIME` | 2 h | every configuration but `cigar_D15_exhaust` |
| `ARM0_CIGAR_TIME` | 5 h | `cigar_D15_exhaust`, and the canary |
| `ARM0_MEM` | 3 GB | every task (1.0.4's peak, 685 MB, on `cigar_D15_exhaust`) |

The after arm's runs, so scaled, add up to about 500 CPU-hours, a third of
them `cigar_D15_exhaust`'s, against the after arm's 153.

### Speed

On the developer's machine, one process at a time with one BLAS thread, a
legacy campaign and a campaign of the release code of the population
harness on the same configurations and seeds (the `production` suite at a
few seeds), compared on each run's wall time (`wall_s` of the sidecars)
per configuration (PI, 2026-10-04); the changelog's "Runs are faster"
takes its figures from it.

## Live checklist

- [x] Decisions (PI, 2026-10-04).
- [x] Worktrees: `../pyvbmc-arm0` (`feat-arm0-v104`), `../pyvbmc-v1.0.4`
  (`v1.0.4`, detached), `../gpyreg-v1.0.4` (`v1.0.4`, detached).
- [x] The legacy profile, the worker's shims, the checks of a legacy case.
- [x] `rescore-arms`.
- [x] The comparison of a legacy reference.
- [x] Tests (the population and analysis modules; a legacy case on the
  real 1.0.4 trees, which the environment check runs).
- [x] The operator's guide: the trees, the harness commit, the environment
  check, the after arm's directory, the campaign, `rescore-arms` and its
  archive part, the hand-back, the limits' names.
- [x] Review (doublecheck, three reviewers) and its fixes.
- [x] Local runs: one case of each configuration with 1.0.4 through the
  worker, for compatibility and the limits; a compatibility patch for
  NumPy 2.4 and later (PI, 2026-10-05).
- [x] Merge into `dev-next` (`9b84a1ae`).
- [ ] The operator's brief, with Arm 0's commit and limits, for one batch
  with Arm 3 ([its plan](arm-3-warmup-comparison.md)). Arm 0's commit is
  the head of `dev-next` once the pairing across harness commits of Arm
  3's plan has merged, its package still the after arm's, so no edit of
  the package's text (a tip's wording, the release date) lands before it
  is named.
- [ ] The campaign on the cluster, the hand-back, the comparison.
- [ ] Speed against 1.0.4 on the developer's machine.

## Worklog

- 2026-10-04: plan written from the PI's decisions of the day. 1.0.4's
  interface, read at `v1.0.4`: no `seed=` and no generator anywhere (the
  global state; `iteration_history["random_state"]`); unknown options
  raise `ValueError`; `final_boost(vp, gp)` returns `(vp, elbo, elbo_sd,
  changed)` and keeps every boost, at the default weight penalty; its
  iteration history holds every key `golden_trace.run_task` reads; its
  transform types are numbered as the release's (`logit` 3, `probit` 12,
  `student4` 13); its `pyproject.toml` asks for `numpy >= 2.0.0`.
- 2026-10-04: the legacy profile, the shims, `rescore-arms` and the
  comparison of a legacy reference on `feat-arm0-v104` (`4d3df9a2`), with
  the operator's guide. On the developer's machine (Python 3.12.6, NumPy
  2.5.2, SciPy 1.18.1), the population module passed (71 tests, the two
  real runs of the release code and a real case of 1.0.4 among them) and
  the analysis module (55). The real case of 1.0.4, `normal_D2` at seed 0
  of the `smoke` suite, ran from the `v1.0.4` tree with gpyreg `v1.0.4` and
  no compatibility patch: 65 evaluations, stable, evidence error 3e-4,
  MMTV 0.007, its boost kept at weight penalty 0.1; it verified,
  `rescore-arms` rescored it beside the release code's real case, which
  reproduced its in-run metrics exactly, and the comparison read both. The
  first version of the scoring check required clean trees even of two
  processes of one source identity, which a development checkout with
  uncommitted changes is not, and refused the release code's own campaign;
  one source identity now scores alike whatever the trees' state.
- 2026-10-04: a doublecheck of `4d3df9a2` by three fresh reviewers,
  read-only (the harness against the 1.0.4 and gpyreg 1.0.4 sources; the
  comparison and the tests; the documents and the operator's path). The
  worker path held against the 1.0.4 source (its seeding, cma's,
  gpyreg's, the metrics shim, the boost capture, the rebuilt posterior).
  The fixes: `prepare --compare` checks before any case runs what the
  comparison checks after (the harness commit's package code above all,
  which a package edit after the promotion would have broken only at
  `rescore-arms`); options are checked against 1.0.4's option files and
  the 1.0.4 tree must be clean; `rescore-arms` refuses a harness checkout
  other than the legacy campaign's, takes the compared campaign from the
  manifest, and rescores the campaign of its own code first; both
  identities must hold every file that builds and scores a run; the
  comparison checks the reference's legacy profile against its commit,
  passes each arm's legacy state to the boost check, reads a legacy
  reference's rescoring by default, and writes `legacy_reference` only for
  a legacy reference, so that the assessment of two arms of the harness's
  own code is the one it was; `reference_promote.py` refuses an assessment
  of a legacy reference; the guide's rescoring job exports what it reads,
  and the guide gives the harness commit, the restore of the after arm, the
  archive part with its log, and the hand-back; tests of the comparison of
  a legacy arm's redacted copies, of its refusals, of the legacy boost
  check and of the node family. With the fixes the population module
  passes (75, the real case of 1.0.4 among them), the analysis module
  (57), the promotion's (11) and the contract's (139).
- 2026-10-04/05: the local runs of 1.0.4, seed 0 of every `production`
  configuration through the worker (the legacy campaign prepared on the
  release gate's allocation and compared with a campaign of the release
  code prepared here and never run, since the cluster's after arm ran in
  another environment and on a node family). 20 configurations ran and
  verified. The other four (`cigar_D15_exhaust` and the noisy `logreg_D5`,
  `student_D8` and `lumpy_D10`) failed in `_eval_full_elcbo` with
  `ValueError: setting an array element with a sequence`: once GP
  hyperparameter sampling stops (N >= 200 + 10 D), 1.0.4's `_gp_log_joint`
  keeps a sample axis on the ELBO's variance, which NumPy before 2.4 stored
  as its element and NumPy 2.4 and later, the campaign environment's 2.5.2
  among them, refuse (`CHANGELOG.md` records the fix among the crashes of
  1.0.4). The PI chose (2026-10-05) to measure 1.0.4's algorithm there,
  under decision 1: the legacy profile names a compatibility patch, the
  package's own fix (`6f3f0ba7`), which the shims apply to every module
  that holds the function (`7c344fa3`); a test applies it in a process of
  1.0.4. With it the four ran and verified, `cigar_D15_exhaust` to its 750
  evaluations (evidence error 0.006, MMTV 0.006). 1.0.4 took 1.5 to 5.2
  times the release code's time per configuration on the developer's
  machine (the release code's from the fingerprints of the Slurm plan's
  Phase 9, at the same seed): 3.1 to 5.2 on the noiseless targets but
  `lumpy_D10` (2.5) and `student_D4` (2.3), 1.5 to 4.1 on the noisy ones,
  5.1 on `cigar_D15_exhaust` (46 minutes against 9); its peak resident set
  685 MB, on `cigar_D15_exhaust`. "Limits from the local runs" sets the
  cluster limits from them. Two runs of the session were stopped by the
  workstation's memory pressure (Claude Code's reaper, not the runs), and
  the worker's claims let them continue where they stopped.
