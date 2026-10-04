# Arm 0: PyVBMC 1.5 against 1.0.4 on the release gate's populations

Created 2026-10-04. Status: **design written; implementation on
`feat-arm0-v104`**.

## Purpose

The reference populations of the release gate
([Slurm plan](slurm-benchmark-support.md), Phase 8) compare the release
code with `f91fdf0`, `dev-next` on 2026-09-19, which already held the
modernization's speed-ups and most of its fixes: that comparison measures
the port correctness review and nothing earlier. A user upgrades from
PyVBMC 1.0.4, so the release's claims of accuracy and speed are made
against 1.0.4. Arm 0 is a third population, 1.0.4 on the same cases,
compared seed by seed with the after arm, the release code's population.

What exists against 1.0.4 before Arm 0: the first golden population
(`baseline_20260903`, on `5020879`, 1.0.4 with `seed=` and three fixes of
2026-09-02; 20 seeds of twelve noiseless configurations, on the
developer's machine) gave 239 usable runs against the release's 240 at the
same seeds, every configuration's median evidence error, gsKL and MMTV
equal or a little better in the release. No 1.0.4 run exists of a noisy
configuration at production defaults.

## Decisions (PI, 2026-10-04)

1. **The code is 1.0.4 as released**, the tag `v1.0.4` (`0bb8b8f5`),
   unmodified. It has no `seed=`: every draw goes through NumPy's global
   state, which the harness seeds with the case's seed before it
   constructs `VBMC`. Should 1.0.4 fail in the campaign environment, a
   compatibility patch that changes no numerics is added and recorded
   here; the environment is not changed, so that the arms differ in their
   code alone.
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

## Constraint: nothing that builds a run changes

From the launch of the release gate to the promotion of the new reference,
no file of `NUMERIC_PATHS` (`dev/scripts/reference_promote.py`: the package
but for its tests and S-VBMC, `golden_trace.py`, `benchmark_targets.py`,
`profile_run.py`, `dev/scripts/data`) changes (the Slurm plan's decision
13). Every adaptation to 1.0.4 is therefore in `population_run.py`, which
installs it in the worker's process, and in the comparison's tools.

## Design

### The legacy profile

`population_run.LEGACY_PACKAGES` names the package commits the harness
runs as released code older than the contract: `0bb8b8f5…` (1.0.4), with
the gpyreg release it runs with (`1.0.4`) and the options it defines. A
campaign whose package tree is at such a commit is a legacy campaign:

- `prepare` takes the gpyreg release from the profile in place of the
  package tree's `pyproject.toml` (1.0.4's says `gpyreg >= 0.1.0`), exactly;
- its options are `DEFAULT_OPTIONS` without those the package does not
  define (`vectorized_target`, `performance_calibration`,
  `tol_elcbo_boost`), and `--options` naming one of those is refused;
- `prepare --pair` is refused: a legacy campaign is compared with an arm
  that already exists (below).

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
  metrics run with the global state seeded as the metric's own generators
  are, and the state is restored after them; the in-run metrics and those
  of the rebuilt posterior then agree exactly, as the worker requires;
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
rescore-arms --out ARM0 --campaign AFTER` runs in a process of the release
code (`PYVBMC_SOURCE` unset, gpyreg the after arm's) and writes, into the
legacy campaign, `rescored/<name>.json` for both campaigns and
`rescoring.json` with its identity, as `rescore` does in a campaign of the
release code; the legacy campaign's tracked copies declare them, so that
the redaction writes them for the repository. A campaign whose scoring
code is the process's (`scores_alike`: the same source identity, or else,
by `scoring_differences`, clean trees whose package code but its tests
and S-VBMC is equal between the two commits, gpyreg at one commit, and
the files that build and score a run and the imported versions equal) must
reproduce its in-run metrics exactly, so the after arm's 2400 cases test
that the rescoring scores as the release code did, and one such campaign
is required. It runs on the cluster, where both raw campaigns are, as a
batch job on the campaigns' node family, after Arm 0's finish.

### The comparison

`analyze_population_run.py --arms ARM0 AFTER --rescoring ARM0` compares a
legacy reference with the after arm, either as campaign directories or as
their tracked copies. For such a pair it allows options that differ only by
those the legacy package does not define, and harness checkouts at
different commits whose run-building files (`golden_trace.py`,
`profile_run.py`, `benchmark_targets.py`, `dev/scripts/data`) and imported
versions are equal (`legacy_pair_differences`), and it requires a
rescoring that scores as the candidate's code rather than one of its whole
identity. It still requires the after arm's rescored metrics to equal its
in-run metrics, the same allocation and confirmatory family, and one node
family. A boost report without a tolerance, a legacy package's, is
expected to keep every boost it made. The report records the legacy
profile (`legacy_reference`).

### What the operator runs

A brief in the manner of `release-gate-handoff.md`: the trees (`v1.0.4` as
a detached worktree of the harness checkout, gpyreg `v1.0.4`), the
environment check of the new harness commit, the campaign as one arm
without `--pair`, with `cigar_D15_exhaust` as its own subset, its finish,
`rescore-arms` over Arm 0 and the after arm, and the hand-back. Limits
come from the local smoke runs (1.0.4 is slower than the release on
noiseless targets).

## Live checklist

- [x] Decisions (PI, 2026-10-04).
- [x] Worktrees: `../pyvbmc-arm0` (`feat-arm0-v104`), `../pyvbmc-v1.0.4`
  (`v1.0.4`, detached), `../gpyreg-v1.0.4` (`v1.0.4`, detached).
- [x] The legacy profile, the worker's shims, the checks of a legacy case.
- [x] `rescore-arms`.
- [x] The comparison of a legacy reference.
- [x] Tests (the population and analysis modules; a legacy case on the
  real 1.0.4 trees, which the environment check runs).
- [x] The operator's guide: the trees, the environment check, the
  campaign, `rescore-arms` and its archive part, the limits' names.
- [ ] Local smoke: one case of each configuration with 1.0.4 through the
  worker, for compatibility and the limits (after the warm-up experiment
  frees the machine).
- [ ] Review (doublecheck), merge into `dev-next`.
- [ ] The operator's brief, with Arm 0's limits.
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
  comparison of a legacy reference on `feat-arm0-v104`, with the operator's
  guide. On the developer's machine (Python 3.12.6, NumPy 2.5.2, SciPy
  1.18.1), the population module passes (71 tests, the two real runs of
  the release code and a real case of 1.0.4 among them) and the analysis
  module (55). The real case of 1.0.4, `normal_D2` at seed 0 of the
  `smoke` suite, ran from the `v1.0.4` tree with gpyreg `v1.0.4` and no
  compatibility patch: 65 evaluations, stable, evidence error 3e-4, MMTV
  0.007, its boost kept at weight penalty 0.1; it verified, `rescore-arms`
  rescored it beside the release code's real case, which reproduced its
  in-run metrics exactly, and the comparison read both. The first version
  of the scoring check required clean trees even of two processes of one
  source identity, which a development checkout with uncommitted changes
  is not, and refused the release code's own campaign; `scores_alike`
  takes one source identity as scoring alike whatever the trees' state.
