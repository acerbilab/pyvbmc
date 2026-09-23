# Plan: independent correctness review of PyVBMC and its MATLAB port

Started 2026-09-19. Owner of the item: `TODO.md`, "Independent codebase and
MATLAB-port review". This file holds the design, the reviewer brief, the
working rules and the worklog. The consolidated findings go into a dated
ledger under `results/` when verification completes; the raw reviewer
reports, the reports of the fix agents, the known-differences sheet and the
verification scripts are kept under `experiments/port_review_20260919/`.

## Purpose

Before the code is frozen for the final release benchmark, several
independent reviewers read the PyVBMC codebase and its GP backend gpyreg
for errors and latent defects, half of them checking internal correctness
and half comparing the code with the original MATLAB VBMC implementation,
with a third reader on the critical numerical paths. The motivation is the
uniform box-sampling error found on 2026-09-16
([report](../results/2026-09-16-gp-box-sampler.md)): it entered with the
2021 port, survived every later refactor, and was invisible to a unit test
that mirrored the implementation instead of the specification. Passing
tests and the stored oracles pin the current behavior; they do not
establish that the port is right.

Coverage follows the categories of the TODO item: formulas and gradients,
indexing and array shapes, defaults, control flow, random draws, state and
caching, and cross-module behavior.

## Decisions (PI, 2026-09-19)

- gpyreg is in scope. It is the GP backend and itself a port of MATLAB's
  `gplite`; a defect there is a PyVBMC defect for users.
- The MATLAB comparison target is the latest `master` of `acerbilab/vbmc`
  (see revisions below), so that the review also finds what moved in MATLAB
  after the port and was never carried over.
- The MATLAB comparison covers only the modules that have a MATLAB
  counterpart. Modules without one (`svbmc/`, `pymc/`, the Torch and ArviZ
  exports, `calibration/`) get the internal-correctness track alone, at
  lower intensity: one reviewer, no third reader, the same brief.
- Reviewers are Opus agents. Fable is used only on request, for a critical
  or suspicious part where Opus is not confident, and each such use is
  asked for separately.
- At most four agents run at a time.
- Agents may run small checks (short scripts, single function calls,
  finite differences, one test function). They may not run test suites,
  `optimize()` runs, installs, or anything else that competes for the one
  heavy-compute slot. After every wave the orchestrator checks all three
  repositories for leftover files and removes them.
- No MATLAB check session is planned (2026-09-20; the design of
  2026-09-19 had one). A MATLAB run is worth its cost only when its result
  would change a decision. A session is written up, for another developer
  to run, only if a finding is classified **needs MATLAB** and its
  disposition depends on what MATLAB actually computes.
- Confirmed findings are brought to the PI for triage before any fix is
  made. PyVBMC 1.5 takes only bug fixes and unequivocal improvements.
- A fix may make an interface stricter (2026-09-20): a value that used to
  be accepted and now fails with a clear message costs a user one
  correction in one place. What a fix must not do is change behavior
  silently.

**Terminology.** The records and the agent prompts of this review use the
plain vocabulary of code review and debugging: review, reviewer, finding,
discrepancy, defect, reproduction, disposition. They avoid the vocabulary
of computer security, which automated content classifiers have misread in
earlier debugging sessions on this codebase.

## Reference revisions

| Code | Location | Revision |
| --- | --- | --- |
| PyVBMC under review | this repository, `dev-next` | `f91fdf0` at the start of the review |
| gpyreg under review | `../gpyreg`, `main` | `9e70e6b` (1.2.1 release notes dated) |
| MATLAB VBMC, comparison target | `../vbmc`, `master` | `396d649` (2023-05-03; equal to the remote `master` on 2026-09-19 by `git ls-remote`) |

The Python port has no single MATLAB source revision. Its first commit,
`d974e20`, is dated 2021-01-19; modules were ported over the following
months and MATLAB kept changing meanwhile. MATLAB has 68 commits since
2021-01-19. The algorithmic ones, which slice M checks one by one against
the Python side, are:

| MATLAB commit | Date | Change |
| --- | --- | --- |
| `f9c04bc`, `a5240d2` | 2021-02-02 | rotoscaling fix; warp robustness (plausible-range recomputation, `elbo_sd` check for undoing a warp) |
| `f3f8b80` | 2021-03-13 | option updates and fixes for `D = 1` |
| `11737f5` | 2021-03-27 | `vbmc_mode.m` starting points within bounds |
| `46b6f5e` | 2021-05-26 | `Xs = []` for the case `N <= N0` |
| `fcf2674` | 2021-06-07 | slice-sampler fix |
| `2044530` | 2021-06-18 | removal of unused and experimental features |
| `1b72896` | 2021-06-23 | gradient of the Monte Carlo entropy with respect to the mixture weights (`ent/entmc_vbmc.m`) |
| `2cb857d` | 2021-07-13 | `gsKL_true` saved in the per-iteration stats |
| `1d1f20d` | 2022-01-30 | `uuinv` fix in `gplite/private/fminfill.m` (the gpyreg lineage) |
| `bd38b48` | 2022-01-30 | `utils/fminfill.m` update (VBMC's own copy, no Python counterpart; `acq/acqbald_vbmc.m`, also touched, was deleted later) |
| `68a197b` | 2022-06-25 | GP log-density fix in `gplite/gplite_pred.m`; slice sampling in `private/activesample_vbmc.m` |
| `74b046e`, `f3e5d76` | 2022-06-25, 07-06 | bounded-transform fixes in `shared/warpvars_vbmc.m`, `misc/setupvars_vbmc.m`, `vbmc.m`; probit name |
| `a9615ba` | 2022-07-23 | VIQR speed changes in `acq/acqviqr_vbmc.m`, `private/activeimportancesampling_vbmc.m`, `misc/setupoptions_vbmc.m`, `vbmc.m` |
| `c387612`, `f3c05d6` | 2022-10-26 | new priors and their log-densities under `shared/` |
| `6bd4f2f` | 2022-10-30 | `vbmc.m` update |

`git log --format='%h %ad %s' --date=short --since=2021-01-19` in `../vbmc`
lists all 68; the rest touch the README, license, examples and tests.

A discrepancy is dated during verification, not by a fixed revision: the
verifier reads the history of the MATLAB lines (`git log -L` or `git log
-p -- <path>` in `../vbmc`) and of the Python lines (`git log -L` here) and
records whether the Python ever matched MATLAB, whether MATLAB changed
after the module was ported, or whether the two never agreed. Any of the
three may be a defect, an intentional difference, or a defect shared by
both implementations.

Two facts about the GP layer shape slices G1, G2 and M:

- gpyreg keeps a reference copy of `gplite` under `matlab/gplite/`. Ignoring
  whitespace, that copy differs from the comparison target in
  `private/gplite_core.m` (119 lines), `gplite_post.m` (32),
  `gplite_pred.m` (20), `private/slicesamplebnd.m` (11), `gplite_train.m`
  (4) and `private/fminfill.m` (2). Most of the difference is the
  integrated mean function (`intmeanfun`, `betabar`), a `gplite` feature
  gpyreg never ported; `gplite_intmeanfun.m` is absent from gpyreg's copy.
  Of the remaining lines, all but one are blank lines or comments; the
  one substantive line is the predictive log-density of `gplite_pred.m`,
  which `68a197b` changed to use the total predictive variance. Whether
  gpyreg's prediction reflects it is the first question for slices M and
  G2. gpyreg's copy of `fminfill.m` already carries the `uuinv` fix.
- The integrated mean function reaches VBMC as well: `acq/acqviqr_vbmc.m`,
  `misc/intkernel.m`, `misc/gpreupdate.m`,
  `misc/check_quadcoefficients_vbmc.m` and
  `private/activeimportancesampling_vbmc.m` all branch on it. PyVBMC has
  no counterpart (`gaussian_process_train.py` carries a "Missing port"
  comment for its hyperprior). It is the first entry of the
  known-differences sheet, so that no comparison reviewer reports it.

## Slices

Each PyVBMC and gpyreg slice gets two reviewers, one per track. The M slice
and the N slices get one. The O slices are third readers on the critical
numerical paths, added on top of the slice reviewers, and combine both
tracks: re-derive the formula independently, then compare with MATLAB.
MATLAB files marked *unported* have no Python counterpart; the comparison
reviewer confirms that and spends no more time on them.

| Slice | Python files | MATLAB counterparts |
| --- | --- | --- |
| **P1a** Main loop, warmup, termination, final boost | `vbmc/vbmc.py`: `optimize`, the warmup and termination checks, `final_boost`, `determine_best_vp`, `get_gp`, `_create_result_dict`; the call site of `update_K` (P6) and the commented-out `vptrain2real` calls (lines 1344 and 1536) | `vbmc.m`, `private/vbmc_warmup.m`, `private/vbmc_termination.m`, `private/recompute_lcbmax.m`, `private/vbmc_output.m`, `misc/finalboost_vbmc.m`, `misc/best_vbmc.m` |
| **P1b** Setup, options, defaults, bounds, log-joint construction, history, save/load | `vbmc/vbmc.py`: `__init__`, option and bounds setup, `_init_log_joint`/`_rebuild_log_joint`, save/load and everything not in P1a; `vbmc/options.py`, `vbmc/option_configs/*.ini`, `vbmc/_bounds.py`, `vbmc/iteration_history.py`, `pyvbmc/rng.py`, `pyvbmc/__init__.py` (eager imports, lazy `SVBMC` resolution) | `misc/setupoptions_vbmc.m`, `misc/setupvars_vbmc.m`, `misc/boundscheck_vbmc.m`, `misc/evaloption_vbmc.m`, `utils/evalbool.m`, `vbmc.m` (`save_stats`), `lpostfun.m`; `vbmc_diagnostics.m` (unported). Four PyVBMC options (`double_gp`, `empirical_gp_prior`, `integrate_gp_mean`, `gp_stochastic_step_size`) correspond to MATLAB options removed in `2044530`, and `Bandwidth` and `FeatureTest` exist only in MATLAB with no recorded reason: are they live? |
| **P2** Initial design and active-sampling search | `vbmc/active_sample.py` (including `_BatchedNoiseHandler`, removed by the wave-1 fixes) | `private/activesample_vbmc.m`, `misc/initdesign_vbmc.m`; `utils/cmaes_modded.m` (PyVBMC uses the `cma` package and binds its draws to the run's generator; until the wave-1 fixes it also subclassed cma's noise handler), `misc/proposal_vbmc.m` (unported; unused in MATLAB too), `private/acqhedge_vbmc.m`, `utils/fastkmeans.m`, `utils/covcma.m` (unported), `misc/check_quadcoefficients_vbmc.m` (unported, integrated mean function) |
| **P3** Acquisition functions. First question (PI, 2026-09-20): at uncertainty level 1 (`uncertainty_handling=True` without `specify_target_noise`) the GP has had MATLAB's noise model, a constant plus the recorded noise scaled by a fitted multiplier, only since the wave-3 fixes, so no earlier run exercised it: do the noisy acquisitions treat the noise of a candidate point at that level as MATLAB's do? | `acquisition_functions/*.py` (including `_real2int` and `_sq_dist` of the base class) | `acq/*.m`, `misc/real2int_vbmc.m`, `utils/sq_dist.m`; `misc/noiseshaping_vbmc.m` (unported, default off), `acq/acqeig_vbmc.m` and `misc/intkernel.m` (the EIG acquisition was removed from PyVBMC on 2026-09-14 and retained on `retain/experimental-acquisitions`) |
| **P4** Noisy importance sampling. First question: the same as for P3, for the importance sampling and the GP updates inside active sampling at uncertainty level 1 (`active_sample.py:337` evaluates the noise at the variance of a single observation, `S**2 * n_evals`, as `private/activesample_vbmc.m:165` does, which the multiplier now scales) | `vbmc/active_importance_sampling.py` (including `fess`) | `private/activeimportancesampling_vbmc.m`, `misc/fess_vbmc.m`, `gplite/private/eissample_lite.m` (MATLAB's MCMC sampler, duplicated as `utils/eissample_lite.m`; PyVBMC uses gpyreg's `SliceSampler`, a substitution the comparison reviewer examines) |
| **P5** GP training policy, hyperpriors, training data, GP re-update, lean GP records | `vbmc/gaussian_process_train.py` (including `_gp_hyp`, `_get_training_data`, `reupdate_gp`, `_lean_gp`) | `misc/gptrain_vbmc.m` (its local subfunction `vbmc_gphyp` at line 109; the tracked `misc/vbmc_gphyp.m` is an empty file), `misc/get_GPTrainOptions.m`, `misc/get_traindata_vbmc.m`, `misc/gpreupdate.m`, `misc/gpsample_vbmc.m`; `utils/slicelite.m` (unported, one of the hyperparameter samplers `get_GPTrainOptions.m` offers) |
| **P6** Variational optimization, the ELBO, the `K` update | `vbmc/variational_optimization.py` (including `update_K`), `vbmc/minimize_adam.py` | `private/updateK.m`, `misc/vpoptimize_vbmc.m`, `misc/vpsieve_vbmc.m`, `misc/negelcbo_vbmc.m`, `misc/gplogjoint.m`, `misc/gplogjoint_weights.m`, `misc/vpbndloss.m`, `misc/vpoptimizeweights_vbmc.m`, `misc/vbinit_vbmc.m`, `utils/fminadam.m`, `utils/softbndloss.m`; `misc/vpsample_vbmc.m`, `utils/slicesample_vbmc.m`, `utils/malasample_vbmc.m` (unported: sampling of variational parameters) |
| **P7** Variational posterior, entropies, VP statistics | `variational_posterior/variational_posterior.py`, `entropy/*.py`, `stats/*.py` | `vbmc_pdf.m`, `vbmc_rnd.m`, `vbmc_moments.m`, `vbmc_kldiv.m`, `vbmc_mtv.m`, `vbmc_mode.m`, `ent/entlb_vbmc.m`, `ent/entmc_vbmc.m`, `shared/mvnkl.m`, `shared/kde1d.m`, `misc/gethpd_vbmc.m`, `misc/vpbounds.m`, `misc/get_vptheta.m`, `misc/rescale_params.m`, `shared/qtrapz.m` (inlined in `mtv` through SciPy's trapezoid rule); `vbmc_power.m`, `ent/entub_vbmc.m`, `misc/vptrain2real.m` (unported; the identity at the default temperature) |
| **P8** Parameter transformer, warping, function logger | `parameter_transformer/parameter_transformer.py`, `whitening/whitening.py`, `function_logger/function_logger.py`, `decorators/handle_0D_1D_input.py` | `shared/warpvars_vbmc.m`, `misc/warp_input_vbmc.m`, `misc/warp_gpandvp_vbmc.m`, `utils/unscent_warp.m`, `misc/funlogger_vbmc.m` |
| **P9** Priors | `priors/*.py` | `shared/m*pdf.m`, `shared/m*logpdf.m`, `shared/m*rnd.m` |
| **G1** gpyreg: fit, objective, hyperpriors, posterior factors, samplers, generators | `gaussian_process.py`: `fit`, the objective and its gradient, `update`, `Posterior`, bounds and priors; `f_min_fill.py`, `slice_sample.py`, `gpyreg/rng.py` | `gplite/gplite_train.m`, `gplite/gplite_hypprior.m`, `gplite/gplite_nlZ.m`, `gplite/gplite_post.m`, `gplite/gplite_clean.m`, `gplite/private/gplite_core.m`, `gplite/private/fminfill.m` (carries the `uuinv` fix `1d1f20d`; does `f_min_fill.py`?), `gplite/private/slicesamplebnd.m` (duplicated as `utils/slicesamplebnd.m`) |
| **G2** gpyreg: prediction, quadrature, kernels, means, noise | `gaussian_process.py`: `predict`, `predict_full`, `quad`, sampling; `covariance_functions.py`, `isotropic_covariance_functions.py`, `mean_functions.py`, `noise_functions.py` | `gplite/gplite_pred.m`, `gplite/gplite_quad.m`, `gplite/gplite_qpred.m`, `gplite/gplite_covfun.m`, `gplite/gplite_meanfun.m`, `gplite/gplite_noisefun.m`, `gplite/gplite_rnd.m`, `gplite/private/sq_dist.m` (`scipy.spatial.distance.cdist` in gpyreg), `gplite/private/quantile1.m` (`np.quantile`, whose convention differs; the sheet has the entry); `gplite/gplite_sample.m`, `gplite/gplite_fmin.m`, `gplite/gplite_intmeanfun.m` and `gplite/outwarp_*.m` (unported). First question: since `68a197b`, `gplite_pred.m` computes the predictive log-density with the total predictive variance; does gpyreg's prediction have that quantity, and with which variance? |
| **M** MATLAB changes since the port began (comparison track) | whichever PyVBMC or gpyreg code corresponds | the algorithmic commits in the table above, checked one by one (`bd38b48` has no Python surface: `utils/fminfill.m` is VBMC's own copy), plus the one substantive non-`intmeanfun` line where gpyreg's `gplite` copy differs from the target, the predictive log-density of `gplite_pred.m` |
| **O1** Third reader: expected log joint and ELBO gradients | `_gp_log_joint`, `_neg_elcbo`, `_vp_bound_loss`, `_soft_bound_loss`, and the softmax Jacobian copies in `variational_optimization.py` and in the VP parameter path | `misc/gplogjoint.m`, `misc/negelcbo_vbmc.m`, `misc/vpbndloss.m` |
| **O2** Third reader: entropies and VP density gradients | `entropy/entlb_vbmc.py`, `entropy/entmc_vbmc.py` (each with its own softmax Jacobian copy), the `vp.pdf` gradient | `ent/entlb_vbmc.m`, `ent/entmc_vbmc.m`, `vbmc_pdf.m` |
| **O3** Third reader: transformations and their Jacobians | `parameter_transformer.py`: forward, inverse, `log_abs_det_jacobian`; `whitening.py`. MATLAB's `warpvars_vbmc.m` also provides the gradient of the log Jacobian, which PyVBMC lacks: the reader checks whether any Python path needs it | `shared/warpvars_vbmc.m`, `misc/warp_gpandvp_vbmc.m` |
| **O4** Third reader: GP marginal likelihood and its gradient | the negative log marginal likelihood and its gradient in `gaussian_process.py`; the derivative branches (`compute_grad=True`) of `covariance_functions.py`, `mean_functions.py`, `noise_functions.py` | `gplite/private/gplite_core.m`, `gplite/gplite_nlZ.m`, `gplite/gplite_covfun.m`, `gplite/gplite_meanfun.m`, `gplite/gplite_noisefun.m`, `gplite/private/derivcheck.m` |
| **N1** Internal only: S-VBMC | `svbmc/*.py` | none (the original standalone `svbmc` package is the reference) |
| **N2** Internal only: PyMC adapter and posterior exports | `pymc/*.py`, `variational_posterior/_torch.py`, `variational_posterior/_arviz.py` | none |
| **N3** Internal only: machine-local calibration | `calibration/*.py` | none |

Out of scope as non-numerical: `formatting/`, `timer/`,
`vbmc/_runtime_tips.py`, `vbmc/_tip_catalog.py`,
`vbmc/create_vbmc_animation.py`, `_user_hints.py`, `__main__.py`,
`gpyreg/formatting.py`, and the plotting code on both sides
(`VariationalPosterior.plot` and `GP.plot`; `vbmc_plot.m`, `gplite_plot.m`).

That is 33 reviewer runs: 12 two-track slices (24), M (1), O1 to O4 (4),
N1 to N3 (3) and the preparatory agent (1), plus verification agents as
findings accumulate. At four agents a wave, the review runs in about nine
waves.

## Preparatory agent: the known-differences sheet and the counterpart map

One agent, run before any comparison-track reviewer, writes
`experiments/port_review_20260919/known_differences.md`. Its sources are
the porting log `pyvbmc/vbmc/README.md`, `AGENTS.md`, §9 of
`dev/2026-09-02-modernization-discussion.md`,
`dev/plans/latent-bug-fixes.md`, `dev/plans/modernization-roadmap.md`,
the fix reports under `dev/results/` (box sampler, final-boost guard, eta
bound, weighted covariance, sampling termination), the retained-branch
records in `TODO.md`, the "Missing port" comments in the package, and
gpyreg's `AGENTS.md`. An entry is a *settled* deliberate difference from
MATLAB, in this form: Python location; MATLAB location; what differs; why,
with the record that decided it; kind (deliberate change, unported feature,
removed feature, substituted library). Open findings and undecided
questions do not go on the sheet. Every entry is a claim a reviewer may
challenge: the sheet tells reviewers what not to report as new, not what
is beyond question.

The same agent builds the counterpart map mechanically: every `*.m`
filename of `../vbmc` searched for across `pyvbmc/`, `../gpyreg/gpyreg/`
and the porting log, reconciled against the slice table above. Mismatches
(a MATLAB file with a Python counterpart in a different slice, a Python
file with no slice, a counterpart the table misses) go into its report and
are fixed in the table before the comparison-track waves start.

## Reviewer brief

Every reviewer is a fresh agent. It does not inherit this conversation, it
does not see another reviewer's report, and it does not open any file under
`dev/` except the known-differences sheet. `CLAUDE.md` and `AGENTS.md` are
loaded into every agent by the harness, so full independence from the
repository's own description of its behavior is not available; the brief
says that `AGENTS.md` describes intended behavior and is not the
specification, and that its pointers into `dev/` are not to be followed.
Each reviewer receives:

- the slice: the Python files and, on the comparison track, the MATLAB
  files, with the comparison revision and how to read a file's history
  (`git log -L<start>,<end>:<path>` in `../vbmc`);
- the track: *internal correctness* (does the code do what its docstrings,
  the porting log `pyvbmc/vbmc/README.md`, the algorithm papers in
  `papers/` and the mathematics require) or *MATLAB comparison* (line by
  line, does the Python do what the MATLAB does, and where not, is the
  difference on the known-differences sheet); O slices do both;
- the known-differences sheet;
- the seven categories and the finding format below;
- the terminology rule;
- the working rules: tracked files are read-only; scripts and outputs go
  only into the scratchpad directory given in the prompt; small checks are
  allowed, test suites, `optimize()` runs and installs are not; tests may
  be read to judge whether they would catch an error, but a test is not
  the specification.
- the interpreters for small checks, since no single environment holds
  every optional dependency: `.venv/Scripts/python.exe` (core, gpyreg);
  `dev/scripts/runs/pymc_feasibility_20260914/venv/Scripts/python.exe`
  (PyMC 6.3.2, PyTensor 3.3.1, ArviZ 1.3.0; documented in the gitignored
  `dev/scripts/runs/LOCAL.md`); `../pyvbmc-stage3/.venv/Scripts/python.exe`
  (Torch 2.7 CPU, ArviZ 1.3.0). The N2 reviewer of wave 0 was not told
  this and reviewed the PyMC adapter statically.
- two warnings about documents they receive: gpyreg's `AGENTS.md`
  misdescribes three MATLAB mappings (`gplite_quad.m` is ported as
  `GP.quad`; `slice_sample.py` ports `gplite/private/slicesamplebnd.m`, not
  `gplite_sample.m`; `f_min_fill.py` ports `gplite/private/fminfill.m`, not
  `gplite_fmin.m`), and the porting log names two functions that have
  since moved or been renamed (`reupdate_gp` in
  `gaussian_process_train.py`, `get_hpd` in `stats/get_hpd.py`).

The reviewer returns its report as its final message, and the orchestrator
saves that text: the harness refuses report files written by agents. The
report has three parts: **coverage**
(what was read completely, what was skimmed, what was not reached),
**findings** in the format below, and **test adequacy notes** (existing
tests that mirror the implementation rather than the specification, in the
reviewer's judgment). A reviewer with no findings says so; the coverage
section is then the deliverable.

Finding format:

```
### F<n>. <one-line title>
- Location: <python path>:<line>; MATLAB: <path>:<line> or "no counterpart"
- Category: formula/gradient | indexing/shape | defaults | control flow |
  random draws | state/caching | cross-module
- Proposed classification: port discrepancy | suspected defect in both |
  possibly intentional | unsure
- Confidence: high | medium | low
- History (comparison track): did the MATLAB lines change after
  2021-01-19? Cite the commit if so.
- What the code does, what it should do, and why (derivation, paper
  equation, or the MATLAB lines).
- Consequence if real: effect on results, when it triggers, how large.
- Suggested reproduction: the smallest check that would settle it.
- Test adequacy: would an existing test have caught it? Which one?
```

Reviewers do not propose or make fixes.

## Verification and the findings ledger

Every finding is verified before it enters the ledger. Verification is a
small script, a finite-difference check, or a second reading of the MATLAB
source by a different agent, run in the scratchpad directory and then
copied under `experiments/port_review_20260919/verification/`. It dates
the discrepancy from both histories (see the reference revisions) and
classifies the finding as one of:

- **confirmed port discrepancy** (PyVBMC or gpyreg differs from MATLAB and
  MATLAB is right, or the difference is unintended);
- **confirmed shared defect** (both implementations are wrong);
- **intentional difference** (missing from the known-differences sheet,
  which is then updated);
- **listed as intentional, but the justification does not hold** (the
  sheet entry is withdrawn and the finding proceeds as a discrepancy);
- **needs MATLAB** (only a MATLAB run can settle it; see "MATLAB runs"
  below);
- **not a defect** (the reviewer misread the code).

Findings that two independent reviewers made carry that fact in the
ledger; a finding made by one reviewer alone is verified with the same
care.

The ledger is `results/<date>-port-correctness-review.md`, dated on
completion. Per finding: identifier, slice, both locations, category,
classification, the dating from both histories, verification evidence
(path under `experiments/port_review_20260919/`), the PI's disposition,
and the fix commit if any. It also records the intentional differences
discovered during the review, the sheet entries withdrawn, and the
test-adequacy notes worth acting on. Durable entries of the
known-differences sheet are consolidated into the porting log
`pyvbmc/vbmc/README.md` during the fix phase, so that the catalogue of
deliberate differences outlives this review.

## Fixes and gates

Fixes wait for the PI's triage of the verified ledger. They go on the
review's branch, `dev-port-review`, one commit per finding, each with a
regression check that would have caught it. gpyreg fixes go to gpyreg on
its own branch and PR, and PyVBMC bumps `GPYREG_PIN` in
`.github/workflows/test-matrix.yml` once gpyreg `main` carries them.

The numerical gates are the existing ones, described in `AGENTS.md`
("Testing conventions and traps"); the commands are repeated here so the
fix phase can be run from this file:

- Before the first fix, `python dev/scripts/make_oracle_fixtures.py
  --dump-outputs DIR` records the current oracle outputs. After each fix,
  `--check --exact --against DIR` shows exactly which oracles moved; a
  fix must move only the oracles its finding predicts. On this machine the
  oracle commands run with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  MKL_NUM_THREADS=1`. `--rebaseline` is sanctioned for `active_sample_step`
  and `gp_fit` alone (with `--expect-moving` naming the second when a
  random-stream change moves both); any other reference is regenerated
  only to set a new baseline on purpose, with the reason recorded in the
  fixture metadata, never to make a change pass.
- `python dev/scripts/golden_replay.py` for the per-step trajectory check.
- The four short seeded runs of
  `experiments/port_review_20260919/verification/scripts/wave2_fixpass_gate_runs.py`,
  recorded on the starting commit and compared bit for bit after every
  batch that must leave default trajectories alone. They are a
  fingerprint of the trajectory and no measure of accuracy: their targets
  have no recorded truth, and the noiseless Rosenbrock among them is the
  unscaled function, far sharper than any problem of the benchmark, on
  which PyVBMC misses the log evidence by about 0.35 whatever the pass.
  A pass that moves default trajectories is read for accuracy on targets
  of `dev/scripts/benchmark_targets.py`, whose truth is known, over
  several seeds before and after
  (`verification/scripts/wave3_gate_benchmark_sweep.py`).
- The finite-difference gradient checks: `pyvbmc/testing/**/test_*_grad_fd.py`
  and the entropy tests `pyvbmc/testing/entropy/test_entlb_vbmc.py` and
  `test_entmc_vbmc.py`, which call `check_grad` directly.
- The focused tests of the changed module, and gpyreg's suite for a gpyreg
  fix.
- The tests that need Torch or PyMC, in the environments that have them
  (listed in the reviewer brief): the default environment skips them
  without a word, and the branch smoke, which has Torch, does not. After
  the wave-3 pass the smoke failed on an S-VBMC test that the local suite
  had skipped.
- A fix pass ends with its lines in `CHANGELOG.md` under `Unreleased`: what a
  user of the last release will notice, one sentence each, with what to do
  where a script may need it, and one line in the section's "Upgrading from"
  list for a change that can stop such a script or change what it returns. A
  fix to a feature that no release has shipped goes into that feature's
  entry. A fix agent proposes its sentences against the branch it worked
  on, so the orchestrator checks each against the tag of the last release
  before writing it (`git show v1.0.4:<path>`): a defect that an earlier
  commit of the same pass brought in never reached a user, and the
  sentence that reports its fix describes nothing a user can notice.
- After a fix pass, the full CI matrix on the branch, dispatched once the
  branch smoke is green (`gh workflow run tests.yml --ref dev-port-review`).
  The smoke that a push starts is one cell, Ubuntu with the newest Python, and
  the orchestrator's machine is one more; the matrix adds macOS and Python
  3.10 and 3.11 on every system. On the wave-2 pass it found what neither had:
  mock targets that Python 3.10 resolves differently, a tolerance set to one
  machine's result, and the bounded transforms pickled as bytecode. A push
  that touches `pyvbmc/` while a smoke run is going cancels it.

A fix whose gates fail is not merged; it returns to triage with the gate
output. A fix that moves default trajectories follows the working rule in
`TODO.md`: the affected golden references are updated after assessment and
the old ones preserved. The production-reference runs under way on
`dev-production-reference` are provisional for the same reason; the final
release gate regenerates the pools.

## MATLAB runs

The review runs no MATLAB (PI, 2026-09-20). Through wave 2 no verified
finding needed a MATLAB run for its disposition: reading both sources and
running the Python side settled every one. The four candidates collected
up to then would not have changed a decision: the duplicate GP training
row of MATLAB's rank-one path on a noiseless repeat and the population
call of MATLAB's search concern behavior PyVBMC has already been ruled to
keep as it is; `recompute_lcbmax` is ported whatever a run shows, and a
GP prediction followed by a cumulative maximum is testable against its
specification in Python; what MATLAB does on the empty warm-up window
changes a classification, not the fix.

Same-input comparisons of the core formulas are covered otherwise. The
test suite already pins, against stored MATLAB outputs, both entropies
with their gradients, the gradient of the expected log joint, the
variational posterior's functions, the fractional effective sample size
and the importance-sampling weights of VIQR and IMIQR
(`pyvbmc/testing/**/FIXTURES.md`). The acquisition values, the
transformer's log-abs-det Jacobian and gpyreg's marginal likelihood and
prediction are not pinned against MATLAB; for them the review relies on
the finite-difference gradient checks, on the comparison reviewers'
line-by-line reading (a Python transcription of the MATLAB function,
compared with the port on the same inputs, is the tool of choice), and on
the third readers of the O slices, who re-derive each formula.

If a later finding can only be settled by a MATLAB run and its disposition
depends on the outcome, the session is written up then, as
`dev/plans/port-review-matlab-checks.md` with scripts under
`dev/scripts/matlab_checks/`: a Python side that dumps states to `.mat`
files with `scipy.io.savemat`, a MATLAB side that loads them and calls the
counterpart functions, and a comparison step with stated tolerances, so
that a developer with MATLAB and the `../vbmc` checkout at `396d649` can
run it from the file alone. The `.m` scripts beside the fixtures in
`pyvbmc/testing/vbmc/compare_MATLAB/` are the existing instance of this
pattern.

Defects that the review finds on the MATLAB side are collected in
`experiments/port_review_20260919/matlab_side_defects.md`, from a reading
of the MATLAB source, as material for the MATLAB VBMC repository.

## Working rules

- At most four agents at a time; the orchestrator holds the heavy-compute
  slot and does not use it while the production-reference runs are active.
- Reviewers are fresh general-purpose agents, never forks.
- An agent that changes code works in its own git worktree. The harness
  creates that worktree at an old commit of `main`, so its brief starts
  with `git merge --ff-only dev-port-review`; its interpreters are the
  ones listed in the reviewer brief, run as `python -m pytest` from the
  worktree root so that the worktree's package is the one imported.
  It commits one finding at a time and does not push; the orchestrator
  reviews each diff, cherry-picks the commits onto `dev-port-review` and
  runs the module's whole test directory there. A `dev/scripts` script
  run from a worktree imports the main checkout's package unless it puts
  its own repository root on `sys.path`, as `make_oracle_fixtures.py` and
  `golden_replay.py` do, or `PYTHONPATH` names the worktree (`python -m
  pytest` is unaffected), so every script command in an agent's brief
  carries `PYTHONPATH=<worktree>` and the agent prints `pyvbmc.__file__`
  once; `AGENTS.md` records the trap.
- The review proceeds one wave at a time. After every wave the
  orchestrator stops and reports the wave's findings to the PI with
  enough context to judge them; the PI decides what follows: the next
  wave alone, the next wave alongside verification, tests or fixes of
  what came in, or a pause until something is sorted out. No wave starts
  without that decision.
- After every wave: `git status --porcelain --ignored` in `pyvbmc`,
  `../gpyreg` and `../vbmc`, ignoring `.venv/` and `docs/`. Anything a
  reviewer left is removed before the next wave. The orchestrator's own
  in-progress edits (this file, `TODO.md`, `dev/README.md`, the files under
  `experiments/port_review_20260919/`) are the only expected changes.
  Reviewer reports are saved as
  `experiments/port_review_20260919/reviews/<slice>_<track>.md` by the
  orchestrator, and the reports of fix agents as
  `experiments/port_review_20260919/fixes/<wave>_agent_<letter>.md`. The
  worktree and the branch of a fix agent are removed once `git cherry`
  shows every one of its commits on `dev-port-review`.
- An agent's report exists only as its final message, because the harness
  refuses report files written by agents. The orchestrator saves that
  message verbatim, never retyped, under a header that says what it is. A
  long message is kept by the harness as a text file in the session's
  `tool-results` directory under `~/.claude/projects/<project>/<session>/`;
  a shorter one arrives inline and is extracted from the agent's
  transcript, `subagents/agent-<id>.jsonl` in the same directory, as the
  last string of the agent's own output that holds the report's title.
  `experiments/port_review_20260919/extract_report.py` does that and writes
  the file under a given header (`<agent.jsonl> <title substring> <header
  file> <out file>`). An agent that hands its report back through a tool
  call and then closes with a short message under the same title leaves
  two such strings, the report first: the script lists every candidate
  with its length, and a fifth argument names the one to take (`0` for the
  first), so the length printed is checked against the report received.
  The `tasks/<id>.output` file that the harness names
  when it launches an agent stayed empty in wave 3; the transcript is the
  source.
- A ruling that leaves an item to a slice leaves it to a pass that exists:
  the worklog says whether that slice still has one ahead. An item left to
  a slice that has been taken gets its own line in `TODO.md`, or is fixed.
- The pre-commit hooks rewrite files (formatting, unused imports) and then
  abort the commit; the rewritten files are staged again and the commit is
  repeated.
- All of the review's work happens on the branch `dev-port-review`, cut
  from `dev-next` at `f91fdf0`: this plan's worklog, the files under
  `experiments/port_review_20260919/`, the ledger and the fixes. It
  merges into `dev-next`, with a merge commit as the other work branches
  do, after a fix pass whose gates have passed, the full CI matrix among
  them (first on 2026-09-20, after wave 2), and is then fast-forwarded onto
  `dev-next` so that the review goes on from the merged line. Only the
  `TODO.md` status line is updated on `dev-next` directly. The merge is made
  in whichever checkout can hold `dev-next`: `git worktree list` says where
  it is checked out, and the sibling checkout `../pyvbmc-stage3`, whose
  environment has Torch, may be on another branch that belongs to other
  work.
- A gate is evidence for a change only if it reaches the changed code. The
  oracle of an acquisition exercises the branch of the importance sampling
  that the acquisition takes and no other (VIQR samples from the variational
  posterior alone; IMIQR alone reaches the proposal density and the MCMC
  step), so a ledger that cites an unmoved oracle first checks that the
  oracle calls the function. Where two fixes meet in one oracle, a bitwise
  comparison of the old and the new function on a stored state tells them
  apart (`verification/scripts/wave4_A7_proposal_pdf_bitwise.py` reads the
  old module from git).
- The heavy gates run one after the other, unbuffered and straight into a
  log file (a pipe through `grep` holds the output back until the process
  ends), and each leaves a record that is checked before the next statement
  about it is written: the machine stops a background run when memory is
  short, and what the run had finished by then is in its records, not in
  its log.
- A script or a block of text that holds apostrophes is written to a file
  and run from there; the shell tool's here-documents fail on them.

## Worklog

- [x] 2026-09-19: design discussed and decided with the PI (decisions
  above). Reference revisions fixed. Slice map built from the file
  inventories of the three trees; gpyreg's `gplite` copy compared with the
  target. One independent Opus doublecheck of this plan before any
  reviewer ran: it corrected the port's start date and the dating rule
  (the port has no single MATLAB source revision), the G1/G2 paths, the
  P4 counterparts (MATLAB samples with `eissample_lite`), the placement of
  `gethpd`, the VP parameter helpers and `_lean_gp`, the unported
  markers, the gplite difference list and its cause (the integrated mean
  function), the agent arithmetic, the wave-0 dependency on the sheet,
  the missing `rng` modules and gpyreg gradient reader (O4), the
  finite-difference gate list and the oracle rebaseline rule.
- [x] 2026-09-19: wave 0 complete: preparatory agent (known-differences
  sheet, 67 entries, and the 124-row counterpart map); N1 (5 findings);
  N3 (10 findings); N2 (5 findings, static only: the brief named no
  interpreter and the PyMC environment lives under `dev/scripts/runs/`),
  rerun the same day with the interpreters named. Records under
  `experiments/port_review_20260919/`. The branch was fast-forwarded
  onto `dev-next` at `bb2d2f4` (closing of the acquisition-efficiency
  workstream; no change under `pyvbmc/`, so the code under review is
  still that of `f91fdf0`).
- [x] 2026-09-19: slice table reconciled against the counterpart map
  (`experiments/port_review_20260919/prep_report.md` lists the
  corrections applied).
- [x] 2026-09-19: wave 0 verified and triaged. All 24 findings are true
  of the code (`experiments/port_review_20260919/verification/wave0.md`
  has the method, outcome and class of each, with scripts and logs);
  the PI ruled on them the same day. Fixed on this branch by three
  Opus agents, one commit per finding with a test written against the
  contract, each module's whole test directory passing afterwards:
  calibration (summary speed-up of the selected setting only; the
  `calibrate` docstrings; the regression veto on its own constant and a
  dead helper removed; 86 tests), S-VBMC (one original-space box across
  runs; every `optimize()` from the runs' own weights, first call
  bit-identical; two docstrings, the API page and Example 7; 226 tests)
  and the PyMC adapter (a missing pullback treated as an absent
  derivative; the unsupported-model type kept through the mode search;
  the no-gradient warning's start point; a dead mask; the support of a
  log-transformed variable checked by a probe ladder on its own prior
  density; the arguments that avoid prior draws documented; a draw that
  rounds onto a support bound kept; 107 tests). Rulings without a code
  change: the two-level shrinkage stays as devised, with a comparison
  of the two compositions added to the headline-selection item of
  `TODO.md`; the prior draws at construction stay; saving an `SVBMC`
  object works through `dill` and gets `save`/`load` methods as a
  `TODO.md` item; the core transformer's loss of precision near a
  nonzero bound waits for slices P8 and O3 to say whether MATLAB shares
  it.
- [x] 2026-09-19: chunk-independent Monte Carlo entropy (PI decision).
  Every chunk budget reproduces the default budget's output bit for bit:
  the mixture densities and every sum over samples and components are
  taken on the blocks of the default budget, and the default path is
  the previous code call for call (`f3ba8d3`). The matrix-vector product
  that forms the mixture density turned out to depend on the number of
  rows of its block on this machine, so it runs on the canonical block
  too. The campaign accepts an entropy budget only on identical output
  (`7c7972f`); the kernel revision is `chunk-kernels-v2`. Gates in this
  checkout: `--check --exact` 11 of 11 with nothing re-baselined; a new
  equality test across budgets and shapes that fails on the old kernel;
  entropy, calibration, oracle and variational-optimization tests;
  the S-VBMC directory with Torch (226). Timings unchanged within noise.
  The equality test asserts bit equality of NumPy and BLAS results
  across block shapes, which holds here and is a property of the build:
  the CI matrix is where another platform would show otherwise.
- [x] 2026-09-19: `fun_eval_start` follows MATLAB's `10*ceil((D+1)/10)`
  (`cb2d513`, PI decision; found by slices M and P2). **Moves default
  trajectories** for targets of ten or more dimensions (`lumpy_D10`, its
  noisy variant, `cigar_D15_exhaust`): their golden references describe
  the previous default until they are regenerated, which is done once,
  after the review's other trajectory-moving fixes are decided.
- [x] 2026-09-19: wave 1 reported: M (5 findings), P6 internal (13),
  P6 comparison (10), P2 comparison (16); reports under
  `experiments/port_review_20260919/reviews/`. Found by two reviewers
  independently: the initial-design size, the rank-one GP update gated
  by `and` for MATLAB's `or`, the broadcast in the type-2 starting
  widths of `_vb_init`, the stale `eta` on the returned posterior, the
  soft bounds rebuilt every iteration, `_sieve` with no candidates,
  `minimize_adam` writing into its starting point, and the
  deterministic-entropy branch raising where MATLAB continues.
  Verification and triage pending.
- [x] 2026-09-19: wave 1 verified. P6: 23 ledger rows
  (`verification/wave1_P6.md`); M and P2: 21 rows and eight minor
  observations (`verification/wave1_M_P2.md`); every statement
  reproduced or read in both sources, and the reports' own errors
  recorded there (the reasoning behind P2 F15, four line citations). The
  ledger's correction of M's cause for the one-dimensional search
  override was itself wrong, and the ledger's header says why. A pass over
  the options surface that grew out of M F2
  (`verification/wave1_options.md`) found 25 of the 28 declared options
  that nothing read (wave 2 found the other three), three selection
  options that `determine_best_vp` took
  as arguments both call sites left out, a ranking branch that had
  never executed, and two iteration counts off by one against
  `best_vbmc`.
- [x] 2026-09-19: wave 1 fixed, 35 commits on `dev-port-review` after
  `6c151cb`, one of them the note in `AGENTS.md` on scripts run from
  another checkout (`0d24698`) (three Opus agents on worktrees, one
  finding per commit with its test, cherry-picked after review; their
  reports were not kept, the rule that saves them under `fixes/` dating
  from wave 2). Active sampling (13 plus one):
  the CMA-ES search starts at the per-coordinate scales (`CMA_stds`) and
  runs without cma's noise handler; the rank-one GP update takes a first
  noisy observation with its variance; an empty search cache no longer
  raises; `n_eff` is refreshed under the name its readers use; one
  fallback search bound per coordinate; a failing local search keeps
  the sieve's best candidate; the discarded repository write and the
  unbalanced GP timer removed; surplus starting points stay in the cache
  and an acquired cached point leaves it in every case; a
  one-dimensional acquisition is searched by a bounded scalar method over
  its interval; `warp_cov_reg` accepts a callable or any scalar.
  Variational optimization (11): type-2 starting widths divide each
  dimension's variance by its own length scale; the exact entropy of a
  one-component posterior; `eta` set alongside `w`; `_sieve` without
  candidates returns one-element arrays; `minimize_adam` copies its start
  and averages over the iterates performed; the deterministic optimizer's
  iterate is kept with a warning; `adaptive_k` evaluated on `K`; stable
  sorts and `nanargmin` in the candidate selection; `var_ss` documented;
  the Gaussian-mixture tests made deterministic; `get_bounds` computes
  the soft-bound box from the call's training inputs. Options and
  selection (5): inert options marked and warned about, `noise_shaping`
  rejected, the selection options wired, iteration counts corrected.
  Gates: every module suite touched; `--check --exact` 11 of 11 after
  one sanctioned re-baseline of `active_sample_step` (`dd89374`: the
  search reaches a different point on all seven states that hold an
  `active_sample_step` reference, the eighth, `rosenbrock_D2_noise1_viqr`,
  having none because the oracle does not apply to a run with the full
  update inside active sampling; every other reference bit-identical);
  the oracle tests; both flaky tests made
  deterministic. Records: `AGENTS.md`, the Stage 1 and Stage 2 plans and
  the oracle harness comments describe the search without its noise
  handler (`80204f7`); the known-differences sheet brought to the state
  of the code. **Moves default trajectories**: the per-coordinate search
  scales and the dropped noise handler (every search with `D > 1`), the
  ranking criterion (runs ending without a stable iteration), the exact
  one-component entropy (where `K = 1`: the reported ELBO, the ranking of
  the finished optimizations and the pruning decision, and the draws the
  Monte Carlo estimate took from the run's generator), the type-2
  starting widths (`D > 1` with two slow candidates), the bounded
  one-dimensional search (`D = 1`), the kept starting points (`x0`
  longer than `fun_eval_start`), the rank-one GP update of a fresh noisy
  observation (`510a493`: every acquired point of a noisy run that is not
  a repeat, equal to the full recomputation to rounding) and the refreshed
  `n_eff` (`118626e`: the GP refit inside active sampling on noisy
  targets), on top of `fun_eval_start` above. The
  golden references are regenerated once after the review's remaining
  trajectory-moving fixes are decided.
- [x] 2026-09-19: rulings without a code change (PI). The soft-bound box
  stays a function of the call's training inputs where MATLAB
  accumulates it: `verification/scripts/soft_bounds_trace.py` on two
  short seeded runs (Rosenbrock `D = 2`, two Gaussians `D = 3`) found
  the accumulated box wider than the rebuilt one on 16 of 34 and 6 of
  14 calls (up to 2.08 and 10.4 times its width), while on those calls
  the fitted posterior stayed within the rebuilt box up to a small
  overshoot (a component mean 0.008 box widths outside, a log scale
  0.007 above its bound; the largest overshoot of either run, a mean
  0.14 box widths outside, is at a warm-up call where the two boxes
  coincide), so the rebuilt bounds bind marginally at most and the
  accumulation code, which never had a stored box to widen, was removed
  instead of completed. The log is listed in `dev/scripts/runs/LOCAL.md`.
  `search_cmaes_best`
  stays inert: the acquisition is deterministic while the search runs,
  so cma's best-ever point is the best. The noise handler was dropped
  after a side-by-side (`verification/cmaes_side_by_side/`) found the
  two searches equivalent in value within noise and the handler costing
  about 2.4 acquisition evaluations per generation. Unported features
  found by the wave (`Bandwidth`, the `samples` output struct, the
  clustering of surplus starting points) become sheet entries.
- [x] `test_vp_optimize_1D_g_mixture` failed about one run in forty (PI:
  no flaky test is acceptable). Cause traced
  (`verification/scripts/flaky_vp_optimize_1d*.py`): `optimize_vp` gets a
  fresh, degenerate posterior, so its first slow optimization always
  ends in the merged solution and everything rides on the one candidate
  the sieve ranks first, which is sometimes a broad merged pair. Remedy
  (`e48fade`): a second `optimize_vp` call continuing from the returned
  posterior, a seed, and the one-dimensional KL from exact moments; ten
  of ten runs pass. `test_active_uncertainty_sampling` failed about one
  run in seven for a different reason: the CMA-ES search stops by its
  own tolerance (`tolfun = 1e-2` on a log acquisition) and on 6 of 40
  seeds halts on the Rosenbrock valley floor short of the minimum the
  test asserts; it is seeded (`39ad28a`).
- [x] 2026-09-19: rulings on the four `cma` settings the sheet lists
  as unmatched (PI). The two tolerances (`tolx` absolute against
  MATLAB's `1e-11*max(insigma)`; `tolfunhist` 1e-12 against 1e-13)
  stay at cma's defaults: both sit far below `tolfun`, which stops
  the search first. The starting point stays unevaluated inside cma:
  its acquisition value is the sieve's, the search result is kept only
  when it beats that value, and MATLAB spends two more acquisition
  evaluations per search for the same outcome. Recorded in the
  sheet's entry on the `cma` package.
- [x] 2026-09-20: wave 2 reported (PI decision of 2026-09-19: P1a and
  P1b, both tracks each, four Opus reviewers at once): P1a internal (10
  findings), P1a comparison (10), P1b internal (13), P1b comparison (10);
  reports under `experiments/port_review_20260919/reviews/`. About 28
  distinct findings. Found by two or more reviewers independently: the
  forced entropy switch reading a key nothing writes (all four), the
  running average of the moments behind a misplaced parenthesis, the
  minimum-iteration guard, the unimplemented recomputation of the LCB
  maxima, `results["problem_type"]` and `results["iterations"]`, the
  `temperature` and `diagnostics` options that reach no reader and the
  inert-option test that cannot see them, the stale
  `optim_state["max_fun_evals"]` after a load with a larger budget,
  `integer_vars` read as a mask, and scalar bounds. The P1b comparison
  report holds the complete defaults table: of MATLAB's 180 options, 159
  agree at every tested `D`, and the two defaults that differ in value are
  the bounded transform (on the sheet) and the search acquisition
  function (`AcqFcnLog` for MATLAB's `acqf_vbmc`, deliberate since
  `e20d081`, not on the sheet). Both P1a reviewers found no defect in the
  best-posterior selection corrected on 2026-09-19. The sweeps before and
  after the wave were clean.
- [x] 2026-09-20: the six wave-2 findings that fire at the default
  options verified and ruled on (`verification/wave2.md`, with
  `verification/wave2_warmup_history.md` for the history of three of
  them). PI: all six are fixed, no reason for differing from MATLAB being
  recorded anywhere: the warm-up improvement window, the recomputation of
  the LCB maxima (the function and the branch that consumes it), the
  warping clocks at the end of warm-up, the initial variational means in
  transformed coordinates, the minimum-iteration guard, and the transform
  that `warp_input` inverts the search bounds with (shared with MATLAB,
  latent). The first five **move default trajectories**.
- [x] 2026-09-20: the other wave-2 findings verified and ruled on
  (`verification/wave2.md`, part 2, rows W2-7 to W2-27, with
  `verification/wave2_B_loop.md` and `verification/wave2_C_setup.md`, two
  Opus verifiers, and the orchestrator's scripts and two short runs). No
  finding was refuted. Reclassified: `results["problem_type"]` and the
  empty slice of the warm-up end check are shared with MATLAB; the stale
  budget copy after a load is a regression of `6769a9a` (2026-09-16), in
  no release; the global-state draws of an unseeded construction and the
  `AcqFcnLog` default are recorded decisions. PI: every row is fixed,
  except the `AcqFcnLog` default (kept, sheet entry) and the unseeded
  construction (docstring only); `separate_search_gp` is removed and its
  option registered as inert; `integer_vars` and `uncertainty_handling`
  become type-strict; unknown option names raise on every route;
  `results["iterations"]` reports the number of iterations. The seven
  minor observations that the verifiers call defects are fixed as well.
  One observation is left to slice P8 and is not passed to its reviewers:
  the warp re-transforms only the active rows of the function logger,
  where MATLAB re-transforms every row (B-M11); the verification of P8
  checks whether they found it.
- [x] 2026-09-20: wave 2 fixed, 33 commits on `dev-port-review` after
  `2f2bc94`, on the PI's go-ahead of the same day: three Opus agents on
  worktrees, split by area so that their hunks of `vbmc.py` do not overlap,
  one finding per commit with a test written against the contract (the
  MATLAB lines, the docstring or the ruling), cherry-picked after review.
  Their reports are under `experiments/port_review_20260919/fixes/`, and
  `verification/wave2.md` lists the commit of every finding. The loop
  (agent A): the forced entropy switch read from the options at both sites;
  the running average of the variational moments; the warp refit's sieve
  sized at `Knew`; `warp_input` mapping the search state back with the
  transform of the current inference space; `results["problem_type"]` from
  the original bounds and `results["iterations"]` as the count; the warp
  branch copying `gp.X`; `final_boost` working on a copy of `optim_state`;
  `determine_best_vp` returning a copy. Options and inputs (agent B):
  `uncertainty_handling` a boolean; the noisy defaults applied whenever
  uncertainty handling is on and settled after the options file is read, an
  option set in that file counting as the user's; `integer_vars` a boolean
  mask or 0-based indices; scalar bounds replicated; an undeclared option
  name refused on every route; `max_fun_evals` and `max_iter` checked and
  `max_iter` raised to `min_iter`; the budget path following the values of
  its two arguments; `temperature`, `diagnostics` and `separate_search_gp`
  registered as inert, the search-GP branch removed, the guard test
  matching reads through an options mapping; whole option descriptions,
  nineteen of their texts corrected; the `seed` docstring. Construction
  state and load (agent C): the budget copy refreshed on load and on
  continuation; one transformer after `load`, the restored iteration's, and
  `x0_orig` for the summary; `__str__` reporting the GP, the prior and the
  log-density (the last found by the agent; PI: fix); `_init_logger`. By
  the orchestrator: a frozen options object refuses removal (C-M5, whose
  fix needed four test files outside the agent's area), and in the LCB
  recomputation the handling of `NaN` entries, found in review: a warm-up
  trim can drop every point an early iteration logged, `movmax` then leaves
  that iteration's entry `NaN`, and the maxima of the warm-up check pass
  over it as MATLAB's `max` does, where `np.amax` made the check raise.
  Last, the block that **moves default trajectories**: the empty stability
  window, the recent-improvement window, the recomputed LCB maxima with the
  branch that consumes them, the warping clocks at the end of warm-up, the
  minimum-iteration guard, and the initial variational means in transformed
  coordinates.

  Gates. Before the first cherry-pick: four short seeded runs at the
  default options (`verification/scripts/wave2_fixpass_gate_runs.py`:
  Rosenbrock `D = 2` unbounded, two Gaussians `D = 3` bounded with `x0` off
  the centre of the box, and a noisy variant of each through
  `specify_target_noise`, one of them with a user-set budget; each reaches
  the end of warm-up and at least one warp), which reproduce bit for bit on
  the starting commit, and `make_oracle_fixtures.py --check --exact`, 11 of
  11. After each batch of the first phase (C, B, A): the whole
  `pyvbmc/testing/vbmc` directory, the four runs bit for bit (92 arrays,
  the final generator state among them) and the exact oracle check, 11 of
  11. After the second phase: the exact oracle check, 11 of 11; the whole
  default suite (1521 passed, 58 skipped, no reruns); and the four runs for
  the record. Three of them moved: the noisy Rosenbrock run ends warm-up at
  iteration 4 where it ended at 8 and takes 165 evaluations for 180, the
  two bounded runs change from their first iteration, and every final ELBO
  is within its reported SD of the earlier one. The unbounded Rosenbrock
  run is bit-identical. The golden references are regenerated once, after
  the review's remaining trajectory-moving fixes;
  `dev/scripts/golden_replay.py` was no gate for this pass, its references
  describing the defaults from before wave 1's fixes.

  Records brought to the state of the code: the known-differences sheet
  (new entries on `results["problem_type"]`, the empty warm-up window, the
  diagnostic's isolated draws, the separate search GP, the `AcqFcnLog`
  default, `integer_vars`, the two MATLAB input-check defects PyVBMC does
  not reproduce, the `display` values and `warp_input`; corrections to the
  entries on the options, the randomness, `results["rng_state"]`,
  `OptimToolbox`, the inert options, the sieve's candidate count and
  posterior tempering; seven settled non-differences), the counterpart map
  (eight rows, `private/recompute_lcbmax.m` and `misc/setupvars_vbmc.m`
  among them), the list of MATLAB-side defects, the note in
  `verification/wave1_P6.md` on the warp branch, and `AGENTS.md` (the
  options and the transformer after `load`).
- [x] 2026-09-20: three follow-ups of the wave-2 fix pass, on the PI's
  rulings of the same day (36 commits after `2f2bc94` in all). The closing
  "finalize" line of the display (B-M9, `e9e7803`): the agent's fix follows
  `vbmc.m:884-913`, printing the line also for a posterior selected from an
  earlier iteration and measuring its divergence from the posterior the
  loop ended on; the divergence is estimated from 2 x 10^5 samples, which
  the PI ruled come from a copy of the run's generator unless the boost
  changed the posterior, so that the new case leaves the run's stream alone
  and a run stopped without a boost still continues as an uninterrupted one
  would (`test_vbmc_resume_optimization` pins that property; a new test
  pins the stream). A noisy run with `max_fun_evals = np.inf` no longer
  raises `OverflowError` (`verification/wave2.md`, W2-28, `1421759`, found
  by a fix agent: `update_defaults` computed every noisy default before
  checking which the user had set). The `final_boost` docstring types
  `elbo` and `elbo_sd` as floats (B-M4, `d5b2139`). Gates on `d5b2139`: the
  exact oracle check, 11 of 11; the whole default suite (1525 passed, 58
  skipped, no reruns); the PyMC adapter tests (107) and the S-VBMC tests
  with Torch (226) on the phase-2 head, each with its own interpreter; and
  the four seeded runs, of which every trajectory array is bit-identical to
  the phase-2 head, while the final generator state of the two noisy runs
  is not. The cause is the corrected reference of the closing line's
  divergence: the balanced sampler of `VariationalPosterior.sample` draws a
  number of values that depends on the mixture weights, so comparing with
  another posterior consumes another amount of the run's stream whenever
  the boost changed the posterior. Nothing but the generator state after
  the run depends on it. The worktree of agent A, which held the agent's
  version of B-M9, is removed.
- [x] 2026-09-20: the two questions the follow-ups left open, ruled on by
  the PI the same day (39 commits after `2f2bc94` in all, the notebook's
  included). The closing line's divergence is estimated on a copy of the
  run's generator in every case (`9ff44c5`): keeping the run's generator
  for the boosted case had been meant to leave existing streams alone, the
  corrected reference moved them all the same, and with the copy the
  display has no effect on the stream; a test pins the generator of a
  boosted run where the boost left it. The final boost of a posterior with
  fixed means (W2-29 of `verification/wave2.md`, found by a fix agent,
  reproduced by `verification/scripts/wave2_variable_means_boost.py`,
  `73d2a81`): with `variable_means=False` the boost asked for
  `min_final_components` components while its candidates kept the
  posterior's own means, and raised a broadcast error in a run that ended
  with fewer components than that, as one stopped during warm-up always
  does; MATLAB reads the same way (`misc/finalboost_vbmc.m:6`,
  `misc/vbinit_vbmc.m:132-136`; entry 14 of the list of MATLAB-side
  defects). The boost places the components at the training inputs of the
  GP it is handed and takes their number, as the main loop does after
  warm-up. Gates on `73d2a81`: the exact oracle check, 11 of 11; the whole
  default suite (1528 passed, 58 skipped, no reruns); the four seeded runs,
  of which every trajectory array is bit-identical to the phase-2 head and
  the generator state after the run differs in all four, the closing line
  no longer drawing from it. Both changes have their sheet entries, and the
  line citations of the sheet and of the counterpart map into the files the
  pass changed point at this head. The stored output of example notebook 3
  shows `results["iterations"]` as the count (`bf027a7`, PI: edit the two
  numbers, which the stored function counts confirm, without running the
  notebook again).
- [x] 2026-09-20: the full CI matrix on the fix pass (PI: dispatch it after
  the branch smoke, which runs Ubuntu and Python 3.12 alone, as the
  orchestrator's machine runs Windows and 3.12). It needed three runs, and
  what failed was in tests, two of them wave 1's, on cells that had not run
  since before wave 1. On the three Python 3.10 cells seven tests of
  `test_vbmc_active_sample.py` patched `pyvbmc.vbmc.active_sample.<name>`
  by its dotted name: inside the package that name is the function the
  package exports, which shadows the module, and Python 3.10's `mock`
  resolves the target to the function (`99fdb2f`: the `cma` patches name
  `cma.fmin`, the others patch the module object, as the file's older tests
  do). On Windows 3.12 the seeded one-dimensional mixture fit of
  `test_variational_optimization.py` ended in a good solution of variance
  4.62 where this machine reaches 5.06, against a moment-matched KL bound
  set at 1.5e-3, and the seed replays the same draws on every rerun
  (`8e1df32`: the bound is 5e-3, a variance within about 13% of the truth;
  a merged posterior fails the ELBO check of the same test). On the Python
  3.11 cells of Ubuntu and macOS the interpreter ended in a segmentation
  fault, in three different places over three runs: a new test of the pass
  saved again the run stored in `test_vbmc_save_static.pkl`, which holds
  its target function pickled by value, as bytecode of the Python 3.10 or
  older that wrote it in 2023, and pickling such a function makes dill
  disassemble it, which corrupts memory on 3.11 (`e3ccd0e`: the test saves
  a short run of its own, and `AGENTS.md` records the trap). The third run,
  on `e3ccd0e`, is green in all nine cells.

  The crash led to a defect of the package (W2-30 of
  `verification/wave2.md`; PI: fix). `ParameterTransformer` kept its
  bounded transforms as functions defined inside `_set_bounded_transforms`,
  so every saved posterior, saved run and pickled `SVBMC` object carried
  them as bytecode of the Python version that wrote the file, and for a
  bounded problem the first use of such a file under another minor version
  (`vp.sample`, `vp.pdf`, the construction of an `SVBMC`) ended the
  interpreter; an unbounded problem never calls them, which is why the two
  static fixtures never showed it. Established with two interpreters on the
  orchestrator's machine (3.11.9 in a throwaway environment, 3.12.6), one
  process per step, in both directions. The transformer pickles and copies
  without the functions and rebuilds them from `bounded_types` when
  restored, files written before included, whose stored functions are
  dropped unused (`e610479`); with it every step of the experiment works on
  the files saved before the fix, and so does the whole S-VBMC path: three
  converged bounded runs saved under 3.12 by the earlier code, stacked,
  optimized, sampled and saved under 3.11, and loaded back under 3.12 with
  the same ELBO. Two fixtures hold one bounded posterior written under 3.11
  and under 3.12 by the last commit that stored the functions, so that
  every CI cell uses a file with bytecode of another version. A saved
  `VBMC` instance still holds the target and the log-joint wrapper by
  value: under another version it can be inspected, not continued or saved
  again, which the `save` and `load` docstrings say (`1c0d0d3`). Gates: the
  exact oracle check, 11 of 11; the four seeded runs bit for bit against
  the head before, the generator state included; the whole default suite
  (1540 passed, 58 skipped); the S-VBMC tests with Torch (227); and the
  full matrix on `ae4e651`, green in all nine cells.
- [x] 2026-09-20: `dev-port-review` merged into `dev-next` (PI; merge
  commit `57cbfb4`, on `dev-next` at `bb2d2f4`, which had not moved since
  the review branch was last brought onto it), after the full CI matrix on
  `ae4e651` was green in all nine cells. The review's status in `TODO.md`
  was updated on `dev-next` (`0306d68`), and `dev-port-review` was
  fast-forwarded onto it; waves 3 onward continue on this branch.
- [x] 2026-09-20: `CHANGELOG.md` started (PI), with the changes of waves 0
  to 2 that a user of 1.0.4 will notice; fixes to features that are new in
  1.5 are left to those features' entries. The features of 1.5 (601 commits
  since `v1.0.4`) are to be backfilled from `dev/results/`, `dev/plans/`
  and the log, on the PI's word.
- [x] 2026-09-20: `CHANGELOG.md` rewritten and completed (PI: a session
  that did not see the fix pass rewrites the prose for a user of 1.0.4,
  keeping the facts, then adds what the review did not make). The PI
  approved the rewritten wording. For the rest of 1.5, four read-only Opus
  agents distilled `dev/results/`, `dev/plans/` and the log by area
  (reproducibility and the fixes made before the review; performance,
  memory, calibration and tips; the pipeline features, the PyMC adapter and
  packaging; S-VBMC), and the orchestrator checked every line against the
  code at the head and at `v1.0.4` before writing it. In the entries the
  first version already had, two things changed beyond wording: the
  scalar-bound entry moved from Changed to Fixed, and the initial-design
  entry spells out the count (the first version's "20 from ten dimensions
  on" holds only up to `D = 19`). Everything else is new: the Added
  section, an "Upgrading from 1.0.4" lead, and further Changed and Fixed
  entries. An independent doublecheck by five read-only Opus reviewers
  (fidelity of the rewrite; the new entries in three parts; a reader's pass
  with an inventory of user-visible differences taken from the code) found
  six statements to correct and a number of gaps, all applied the same day;
  two of its findings were refuted against the code (the crash of long runs
  does start with NumPy 2.4, where the conversion of a one-element array to
  a scalar stopped being a deprecation warning, and `get_bounds` never
  accumulated its box in 1.0.4 either, as the message of `7dd2e3f` says).
  The "What's new in PyVBMC 1.5" blocks of `README.md` and
  `docsrc/source/index.rst` link to the changelog. The file and the rule in
  `AGENTS.md` that goes with it stay on `dev-port-review`; the PI ruled that
  `dev-next` is not fast-forwarded for them.
- [x] 2026-09-20: wave 3 run, not yet reported to the PI: P8 internal (14
  findings), P8 comparison (10), P5 internal (7), P5 comparison (17); four
  fresh Opus reviewers by the brief, code at `96cded7`; reports saved
  verbatim under `experiments/port_review_20260919/reviews/`. The
  orchestrator started the wave on its reading of the pickup note, before
  the PI had read the completed changelog; the working rule stands that no
  wave starts without the PI's decision, and the PI let this one stand.
  The sweeps before and after the wave were clean in the three repositories.
  Nothing below is verified.

  Reported by both reviewers of a slice, independently. P5: at uncertainty
  level 1 (`uncertainty_handling` without `specify_target_noise`) the GP
  noise function comes out as level 0's, because gpyreg's `GaussianNoise`
  honours `scale_user_provided` only under `user_provided_add`, so the
  per-point noise and its multiplier hyperparameter are dropped (the
  orchestrator read both code sites and they are as described); the upper
  bound from `upper_gp_length_factor` overwritten a few statements later;
  the reset at the end of warm-up writing `hyp_dict["runcov"]` where every
  reader uses `run_cov`; `np.min` with two operands in the unreachable
  output-dependent-noise branch; the noise upper bound set to `+inf` where
  MATLAB leaves the recommended one. P8: the warp rewriting only the active
  rows of the function logger; the evaluation time of a row turned to NaN
  by a repeated `add`; a one-element array returned for a repeat and a
  scalar for a new point; a half-bounded variable taken as unbounded by the
  public `ParameterTransformer`. One disagreement to settle in
  verification: the window of past GPs that feeds the starting points of
  the hyperparameter fit, which the P5 comparison reviewer finds one
  iteration short for an even-length history (MATLAB has
  `ceil(numel(stats.gp)/2):numel(stats.gp)`, `misc/gptrain_vbmc.m:39`) and
  the internal reviewer took for a correct translation without reading
  MATLAB.

  The two open items that rode on P8. B-M11 was found unprompted by both
  reviewers (P8 comparison F4; P8 internal within F5); neither was told of
  it, and the sheet's P8 section does not mention it. For the wave-0
  question of the precision near a nonzero bound, the brief of the P8
  comparison reviewer alone carried a neutral first question on the
  floating-point evaluation of the bounded transforms near a hard bound. Its
  answer: against a Python transcription of `shared/warpvars_vbmc.m`, the
  forward transform and the log-Jacobian agree bit for bit on 20 000 random
  points per configuration, so MATLAB evaluates the same expressions and
  shares the loss of precision; the differences are Python's own nudge of a
  unit-interval image that rounds to 0 or 1 (MATLAB returns an infinity
  there), `nextafter` for MATLAB's `eps(bound)` in the clamp of the inverse,
  and the grouping of one product in the Student-t inverse. O3 has still to
  read the same code. Both comparison reports end with defects of the MATLAB
  side (eight in all, among them `misc/funlogger_vbmc.m:244` writing a
  repeat's value to the last row), for `matlab_side_defects.md` at
  verification.

  The reviewers' check scripts, the transcription of `warpvars_vbmc.m`
  among them, are kept on the orchestrator's machine only
  (`dev/scripts/runs/LOCAL.md`).
- [x] 2026-09-20: wave 3 reported to the PI and verified (PI: verify, the
  findings that can change what a run computes by the orchestrator and the
  rest by two read-only Opus verifiers, one per slice; the wave gives work
  enough, and no further wave is considered until it is dealt with). The
  ledger is `experiments/port_review_20260919/verification/wave3.md`: 36
  rows from the 48 findings, each with the orchestrator's proposed
  disposition and an empty column for the PI's; the verifiers' raw reports
  are `wave3_P5.md` and `wave3_P8.md` beside it, and the 21 scripts are
  under `verification/scripts/`. How often a finding fires was measured on
  the 21 stored runs with their history and on the 990 posteriors that the
  runs of the population campaigns ended with before their boost
  (`dev/scripts/runs/LOCAL.md`).

  Seven rows can change what a run computes (part 1 of the ledger). At
  uncertainty level 1 the GP has level 0's noise function, confirmed, never
  matched MATLAB, no reason recorded (W3-1). Five differences in the
  hyperparameter fit fire at the defaults and change its starting points
  or its sampler widths, not the model: the window of past GPs, one short
  for an even-length history in 170 of the 361 calls that collect past
  samples, which settles the reviewers' disagreement for the comparison
  reviewer (W3-2); `ceil` for MATLAB's `floor` in the subsample count
  (W3-3); `gp_hyp_full` recording the thinned chain where MATLAB records
  the chain before thinning, in all 732 sampling iterations (W3-4); and
  two bounds written as infinite where MATLAB leaves the recommendation,
  the lower bound of `mean_const`, which keeps the whole design of that
  coordinate inside the plausible box, and the upper bound of the noise
  (W3-5, W3-7). The thresholded covariance of the warp can be indefinite
  in principle, as in MATLAB and in the paper's recipe, and is in none of
  the 990 posteriors (W3-6). Of the other 29 rows none changes a number at
  the defaults: 13 small fixes, 6 cases for a stricter interface, 5
  intentional differences for the sheet, 5 to leave. The verifiers found
  errors in all four reports, which their own reports list; of the eight
  defects the reviewers ascribed to MATLAB five stand, one of them with
  another consequence than reported, and three do not hold, and the P5
  verifier added one.
- [x] 2026-09-20: wave 3 triaged and fixed. The PI ruled on all 36 rows the
  same day (`verification/wave3.md`, last column): the level-1 noise model
  and the five moving differences of the hyperparameter fit are fixed; the
  warp keeps the covariance when the thresholded one is not positive
  definite; `gp_mean_fun`, `gp_hyp_sampler`, `f_vals` with
  `specify_target_noise`, a half-bounded variable, a `scale` that is not
  positive and a logger whose flag contradicts its level are refused where
  they are given; five intentional differences go on the sheet and five
  rows are left. Three Opus agents on worktrees cut at `fc8e561`, split so
  that their hunks do not overlap (the inputs of the fit; its policy and
  option values; slice P8), made 32 commits, one finding each with a test
  written against the contract; the orchestrator reviewed and cherry-picked
  them, 31 on the branch since the guard of W3-2 for a history that holds no
  GP was squashed into the commit of its finding, and made three more (a
  test that combined `f_vals` with a noisy target, which construction now
  refuses; the dead-branch bound and the unreachable `npv` block; two
  docstrings): 34 commits. Their reports are
  `fixes/wave3_agent_A.md`, `_B.md` and `_C.md`, and the ledger lists the
  commit of every row, the gates and what was found on the way.

  Gates, in the ledger's words: the first phase, every commit but the five
  that move default trajectories, left the four seeded runs bit for bit and
  the oracles exact, 11 of 11; the second phase moved the four runs and
  three oracles, `gp_fit`, `gp_fit_history` and the log prior of `gp_nlZ`,
  each by what its fix predicts, and no other; the PI approved their new
  baseline, set from the stored states with the reason in every fixture,
  and the noisy one of the three authentic captures was replayed on its
  stored inputs through a mode added to the generator
  (`--rebaseline-gp-fit-history`). The level-1 run is healthy before and
  after the fix, and after it the multiplier carries the noise, as in
  MATLAB's model.

  One lesson for the gates. The noiseless Rosenbrock of the four seeded
  runs ended 0.054 lower in ELBO after the second phase, against a reported
  SD of 0.0006, and the PI asked whether results had got worse. That target
  is the unscaled Rosenbrock function, written into the gate script of the
  wave-2 pass, ten times sharper than the benchmark's `rosenbrock`; PyVBMC
  misses its log evidence (-1.3947) by about 0.35 before and after, and the
  reported SD reflects the GP alone. On the benchmark's `rosenbrock_D2` (ten
  seeds) and `cigar_D4` (six) the phase changes nothing measurable: median
  error of the ELBO 0.021 before and 0.018 after, and 0.006 and 0.006. The
  four runs are a fingerprint, not a measure of accuracy, and the section
  "Fixes and gates" now says so and names the sweep to run instead.

  The fix agents' commits carried a `Claude-Session:` trailer, which the PI
  does not want: it was removed from the unpushed commits, and `AGENTS.md`
  tells every session to leave it out.
- [x] 2026-09-20: the wave-3 pass through CI, and merged (PI: push, the
  smoke, the matrix, and the merge into `dev-next` if all is green). The
  branch smoke failed twice, each time on one test that the local gates had
  not shown. An S-VBMC test built a run bounded on one side in one variable,
  which the transformer refuses since W3-23 (PI: the refusal stays); it
  needs Torch, and the orchestrator's default environment skips it without
  a word, so the tests that need Torch or PyMC now belong to the gates of a
  pass and were run in their environments (739 and 107 passed). A test of
  the wave-2 pass needed the seeded run it drives to prune a component
  before its warp, which after the fixes of the hyperparameter fit the run
  on Ubuntu no longer did while the one on Windows did; its fixture's
  options now bring the pruning and the warp about on any platform. The
  third smoke and the full matrix, nine cells, are green on `92eb2cc`.

  Before the merge the PI asked for a wider check against a regression. On
  six more benchmark targets with known truth, a few seeds each, before
  the pass and after it, every run is a good solution and the differences
  go both ways: with the two targets of the first sweep, eight targets and
  32 seeded pairs (`verification/wave3.md`, "Gates"). The final release
  benchmark remains the measure of the pass.

  `dev-port-review` merged into `dev-next` (merge commit `82aee90`), the
  review's status in `TODO.md` updated there (`5e5fa18`), and
  `dev-port-review` fast-forwarded onto it. The worktrees and branches of
  the three fix agents and the orchestrator's own, which held the code from
  before the pass for the sweeps, are removed.
- [x] 2026-09-20: wave 4 run and reported to the PI (PI decision of the same
  day: P3 with P4, both tracks): P3 internal (9 findings), P3 comparison (4,
  and five minor observations), P4 internal (8), P4 comparison (8); four
  fresh Opus reviewers by the brief, code at `f873556`; reports saved
  verbatim under `experiments/port_review_20260919/reviews/`. Every brief
  carried the first question of its slice on uncertainty level 1, put to the
  comparison reviewers as "as MATLAB's do" and to the internal ones as
  "consistently with the GP's own noise model", and every report answers it
  under its own heading. The sweeps before and after the wave were clean in
  the three repositories. Nothing below is verified; the orchestrator read
  the code sites of the first two defects and they are as described.

  The first question. All four reviewers find the noise handled at level 1
  as MATLAB handles it and consistently with the GP's model. The function
  logger pools repeats to `S = 1/sqrt(n_evals)`, so `S**2 * n_evals` is 1 at
  every training point and `sn2_new = exp(2*h0) + exp(h1)`, the variance of
  one new observation, where the GP's own training rows carry the pooled
  variance; on a state at level 1 the P3 comparison reviewer's transcription
  of `private/activesample_vbmc.m:159-183` and of the acquisitions agrees
  with the port bit for bit in `sn2_new`, in the length scale and in the four
  pointwise acquisitions, and in VIQR and IMIQR up to the constant `u` below.
  The noise of a newly acquired point is MATLAB's at each level, and gpyreg's
  rank-one update with it equals a GP rebuilt from the enlarged training set
  to about 1e-15 at the three levels (both P4 reviewers). What the answers
  add: MATLAB never takes the rank-one path on a noisy target
  (`gplite/gplite_post.m:76-79` turns the request into a full recomputation
  whenever `s2` is given), so the sheet's entry on the rank-one update,
  written with wave 1's fix, misdescribes MATLAB, while the results agree to
  rounding. Three properties of both implementations weigh more at level 1
  than elsewhere: the candidate's noise is averaged over the hyperparameter
  samples before it enters each sample's formula (in one reviewer's level-1
  fit of 8 samples it ran from 1.24 to 2.91); `sn2_new` leaves out the
  `sn2_mult` of a retried Cholesky factorization, which the prediction and
  the update apply; and the MCMC target of IMIQR asks for a prediction with
  noise at a point that has no recorded noise, so at levels 1 and 2 it adds
  the constant term alone. The multiplier's hard bounds, `[1e-3, 1e3]` on
  the variance on both sides (`gplite/gplite_noisefun.m:114-115`), confine
  the noise SD of a level-1 target to about `[0.03, 32]`.

  Defects reported. In the MCMC step of the importance sampling, which
  IMIQR alone reaches, the resampling weights that choose the chain's
  starting point are built as a column and their maximum is taken along the
  axis of length one, so they come out uniform where MATLAB draws in
  proportion to the importance weights (`active_importance_sampling.py:240-251`
  against `private/activeimportancesampling_vbmc.m:206-214`; both P4
  reviewers, independently; never matched; the guard beside it raises when
  any one candidate has zero weight). The two disagree on what it costs: the
  comparison reviewer's chains in `D = 3` end in the same region after the
  burn-in of 50, the internal reviewer expects a poor start to persist in a
  chain of 100. `_real2int` indexes its input as two-dimensional and
  `active_sample.py:624` hands it the one-dimensional result of the local
  search, so a run with `integer_vars` set raises `IndexError` at the first
  search that improves on the sieve (P3 internal; reproduced on the bare
  call, no run made); `test_real2int` holds five comparisons and no `assert`
  (both P3 reviewers), and the one `active_sample` test with an integer
  variable sets `search_optimizer` to `none`. `string_to_acq` keeps the
  space after a comma in the name of every keyword argument but the first,
  so `"AcqFcnVIQR(quantile=0.9, loss='iqr')"` raises (P3 internal). Dormant:
  `fess` with a number of samples for `X`, its documented default, raises
  (both P4 reviewers; MATLAB's one caller of that form is not ported), and
  the retained `mcmc_importance_sampling` branch raises on its first call,
  gpyreg's sampler refusing the matrix of starting points that MATLAB's
  ensemble sampler takes (both P4 reviewers), against the reason the sheet
  gives for keeping it.

  Differences from MATLAB that no sheet entry records: `u =
  norm.ppf(quantile)` for MATLAB's literal `0.6745`, the whole of the
  residual between the port and the transcription in VIQR (1.7e-5) and IMIQR
  (8e-5), which falls to 1.6e-14 with the same constant (the port had the
  literal until `70325a3` added the `quantile` argument); the log weights
  normalized over the whole array, a constant shift that leaves every
  comparison between candidates alone (three reviewers), beside which the P4
  internal reviewer asks whether the rows of the MCMC branch, each drawn from
  its own unnormalized density, are comparable at all, a question that holds
  for MATLAB's raw weights alike; the search candidates drawn before the
  importance samples, where MATLAB draws them after; `np.around`, which
  rounds a half to even, for MATLAB's `round` in `_real2int`; an exception
  in `active_sample_proposal_pdf` where MATLAB's weight becomes zero. All
  four reviewers report that the sheet's entry on the removed VIQR losses is
  wrong: `loss="iqr_reduction"` is shipped, tested and in the changelog, as
  the message of the commit the entry cites says. Six defects of the MATLAB
  side are listed for `matlab_side_defects.md`, one of them already there.
  Test notes common to the reports: `test_active_importance_sampling`
  asserts shapes alone, and no test or stored state below the GP fit is at
  level 1.

  The reviewers' check scripts, the transcription of the acquisition
  functions among them, are kept on the orchestrator's machine only
  (`dev/scripts/runs/LOCAL.md`).
- [x] 2026-09-20: wave 4 verified and fixed (PI, on the orchestrator's
  recommendations given with the wave's report: one wave at a time, go ahead
  with verification and fixes). The ledger is
  `experiments/port_review_20260919/verification/wave4.md`, 22 rows from the
  29 findings, with the first question's answer checked on both sides; the
  orchestrator verified the three rows that stop a run or change what it
  computes, and two read-only Opus verifiers, one per slice, the others
  (`wave4_P3.md`, `wave4_P4.md`; 27 scripts under `verification/scripts/`).

  What the verification settled. The uniform starting point of IMIQR's
  chains is a port discrepancy from the file's first commit (`e38351a`,
  2022), and what it costs is within the Monte Carlo error of the
  acquisition: on the stored noisy state the error of the centred
  acquisition is 0.0187 nats against 0.0150 with MATLAB's weighted start
  (100 pairs, paired difference 0.0037, SE 0.0016), while the candidate
  ranked best is, on the reference, 0.0043 (SE 0.0021) better with the
  uniform start, both below the noise of either estimate there; on a
  six-dimensional state the two agree, the burn-in absorbing the start. A run with
  `integer_vars` and a search optimizer raises after its initial design, in
  `v1.0.4` as well: integer variables never worked. `string_to_acq` fails on
  any space around a keyword's name and drops a value that holds `=`. The
  tie of `_real2int`'s rounding is reachable, from a starting point at the
  midpoint of a box with an even number of integer levels. The spread of the
  total weights of IMIQR's rows, 17 nats on the stored state, which the P4
  internal reviewer read as an uneven average over the GP hyperparameter
  samples and the P4 verifier as the noise of the uniform start, is neither:
  it is as large with MATLAB's start, and every row's contribution to the
  acquisition sits within 0.05 nats of the others, a row estimating the
  relative reduction of its sample's interquantile range, as MATLAB's does
  (W4-10). Shared with MATLAB by design, and left: the candidate's noise
  averaged over the hyperparameter samples, `sn2_mult` left out of it, the
  constant noise term in the target of IMIQR's MCMC step, the bounds of the
  noise multiplier. Three sheet entries misdescribed the code or MATLAB (the
  removed VIQR losses, the rank-one update on a noisy target, the dormant
  MCMC branch) and are corrected; the verifiers found twelve errors in the
  reports of each slice.

  Fixes: one Opus agent on a worktree cut at `f873556`, ten commits, one row
  each with a test written against the contract (`fixes/wave4_agent.md`);
  the orchestrator reviewed and cherry-picked them and made two more, the
  rounding from the exact fractional part (`floor(abs(x) + 0.5)` sends the
  largest number below a half to one) and the re-baseline. `_real2int` takes
  a single point and rounds a half away from zero; `string_to_acq` parses a
  call; VIQR and IMIQR call the base constructor and refuse a quantile
  outside (0.5, 1); `fess` with a number of samples; weight zero where the
  proposal has density zero; an acquisition that sets
  `mcmc_importance_sampling` refused at construction and in the importance
  sampler, the branch removed (it had never run: gpyreg's sampler refuses
  the matrix of walkers) and `active_importance_sampling_fess_thresh` inert;
  the starting point of IMIQR's chains drawn by weight; a test of `sn2_new`
  at uncertainty level 1 with a repeated row, the first test below the GP
  fit at that level. Gates: the four seeded runs bit for bit against the
  record on `f873556`; the exact oracle check with `acq_AcqFcnIMIQR` moved
  and no other, re-baselined from the stored state with the reason in the
  fixture (`2dc98ce`), then 11 of 11; on `2dc98ce` the default suite (1662
  passed, 58 skipped, no reruns), the Torch environment (752 passed) and the
  PyMC environment (107 passed); the run with an integer variable completes.
  No default trajectory moves. Records brought to the state of the code: the
  sheet, the list of MATLAB-side defects (entries 21 to 24), `CHANGELOG.md`,
  `AGENTS.md` (the re-baseline). Not pushed; the CI matrix has not run.
- [x] 2026-09-20: wave 5 run and saved, neither reported nor verified (PI
  decision of the same day: P7 with P9, both tracks, launched while the
  wave-4 pass was under way, five agents at a time allowed for it; another
  session takes the wave from the saved reports): P7 internal (15 findings),
  P7 comparison (11), P9 internal (11), P9 comparison (7); four fresh Opus
  reviewers by the brief, reading `f873556` (the wave-4 pass changes no file
  of the two slices); reports saved verbatim under
  `experiments/port_review_20260919/reviews/`. The briefs carried first
  questions that waves 0 to 4 had raised, which the reports answer under
  their own heading. P7: where an underflowed `vp.pdf(log_flag=True)`, the
  log of a sum of component densities, reaches, and whether each reader
  treats it as its MATLAB counterpart does (comparison) or correctly
  (internal); and whether every caller of `vp.sample` unpacks the pair it
  returns, with the sampler itself against `vbmc_rnd.m` and against its
  specification. Both P7 briefs put the class first, the statistics second
  and the entropies last, which O2 reads again. P9: whether every sampler
  draws from exactly the density its `log_pdf` defines, family by family, on
  both sides. The sweep after the wave was clean in the three repositories.
  The reviewers' check scripts are kept on the orchestrator's machine only
  (`dev/scripts/runs/LOCAL.md`).
- [x] 2026-09-21: the wave-4 pass checked independently (PI: `/doublecheck`,
  then apply what it finds). Four fresh Opus reviewers, read-only, without
  the session's context: the commits of the acquisition side, those of the
  importance sampling, the ledger against the reports, the sources and the
  logs, and the user-facing and cross-document records. No defect in the
  twelve commits; one reviewer showed that the rounding from the exact
  fractional part is needed in a run, a coordinate saturated at the lower
  bound -0.5 of an integer variable coming back as the largest number below
  a half, negated. In the records: the ledger's evidence for W4-7 was the
  VIQR oracle, which never reaches the changed function (a bitwise
  comparison of the first step of the importance sampling before and after
  takes its place, `verification/scripts/wave4_A7_proposal_pdf_bitwise.py`);
  its statement on the best candidate under the weighted start went against
  its own log; `renormalize_weights` dates from the file's second commit; a
  disposition of W4-9 had not been carried out; and the changelog's sentence
  on `mcmc_importance_sampling` held for some acquisitions alone. All are
  corrected, with the sheet's stale line citations. Five commits follow
  (`76ed8e9` to `601555e`): the constant in the docstring of `AcqFcnVIQR`;
  the quantile taken as a real scalar of any type by one shared check;
  `search_acq_fcn` checked to be a list of acquisitions or strings, its
  description saying which strings are read; the state of the
  integer-variable search test seeded; three sentences of documentation.
  Each ran the focused tests of what it touches. A first run of the gates on
  the new head was stopped by the machine for lack of memory after the
  seeded runs; the second completed.
- [x] 2026-09-21: the wave-4 pass gated and through CI (PI: run the gates,
  push, the smoke, and the full matrix if it is green). On `11fb766`, the
  code of `601555e` with the records: the four seeded runs bit for bit those
  of `f873556` (92 arrays); the exact oracle check, 11 of 11; the default
  suite, 1687 passed and 58 skipped with no reruns; the Torch environment,
  777 passed; the PyMC environment, 107 passed. Pushed; the branch smoke and
  the full matrix, nine cells, are green at the first attempt.
- [x] 2026-09-21: `dev-port-review` merged into `dev-next` (PI; merge commit
  `9d9c01b`, on `dev-next` at `5e5fa18`, which had not moved since the merge
  of the wave-3 pass), after the full CI matrix on `11fb766` was green in all
  nine cells. The review's status in `TODO.md` was updated on `dev-next`
  (`0f3015b`), and `dev-port-review` was fast-forwarded onto it. The worktree
  and the branch of the fix agent had been removed once `git cherry` showed
  its ten commits on the branch.
- [x] 2026-09-21: the wave-3 pass checked independently (PI: `/doublecheck`,
  read-only while wave 4 ran in the same checkout; then, on the PI's word,
  apply what it found). Five fresh Opus reviewers without the session's
  context: the inputs of the hyperparameter fit; its policy, the recorded
  chain and the option checks; slice P8; the ledger, the user-facing records
  and the re-baselined fixtures; the consequences of the pass outside the
  lines it changed. No defect in what a run computes. Ten commits of code
  and tests follow (`2ff2dfe` to `095c29e`), each fix with a test that fails
  on the code before it: the stable order of equal values in the trim at the
  end of warm-up and in both rankings of `determine_best_vp`; `get_hpd` on
  integer values; the option checks in `load`; `batch_call` and a cached
  value at uncertainty level 2; the evaluation time of a repeat whose first
  time is unknown; an `f_vals` of NaN alone; the bound statements of
  `_gp_hyp`; two tests and two descriptions. PI rulings of the day: the
  checks run in `load`, the stable order goes into all three sites, and a
  stored oracle state at uncertainty level 1 is an item of `TODO.md` for
  after the review. Corrected in the records: the changelog (an entry
  missing under its "Upgrading" list, the sentence on the level-1 noise
  model, three lines of that list, the evaluation count of the design
  schedule), entry 18 of the MATLAB-side defects with a flag in the P5
  verifier's report, the counts of the ledger and of this worklog (six more
  targets, eight targets and 32 pairs; 34 commits), the ledger's list of
  sheet entries, the capture re-baseline mode in `dev/README.md` and the
  fixture plan, `AGENTS.md`, and the sheet: 83 of its 161 Python line
  citations carried to their present lines by
  `experiments/port_review_20260919/refresh_citations.py`, which takes each
  citation from the commit that last touched it through the history of the
  cited file. `verification/wave3.md`, "The independent check of the pass",
  has the findings, what was left and why, and the gates: the exact oracle
  check, 11 of 11 with nothing re-baselined; the four seeded runs bit for
  bit those of the wave-4 pass; the default suite, 1699 passed and 58
  skipped, with one rerun of `test_minimize_adam_matyas_with_noise`, an
  unseeded test that failed about one call in eight and that `b731fac` seeds
  and bounds by the directions of its target (PI: fix it before the push);
  the Torch environment, 785 passed and 19 skipped; the PyMC environment,
  107 passed.
- [x] 2026-09-21: the check of the wave-3 pass through CI, and merged (PI:
  fix the flaky test, then the push, the CI and the merge). Pushed on
  `f0be99b`; the branch smoke and the full matrix, nine cells, are green at
  the first attempt. `dev-port-review` merged into `dev-next` (merge commit
  `5acd382`, on `dev-next` at `0f3015b`, which had not moved), the review's
  status in `TODO.md` updated there (`24069e6`), and `dev-port-review`
  fast-forwarded onto it.
- [x] 2026-09-21: wave 5 reported to the PI, with the internal track of P2
  added to it (PI decision of the same day: the one reviewer that wave 1
  left out runs now, and its findings are verified, ruled on and fixed with
  those of P7 and P9 in one pass). P2 internal: one fresh Opus reviewer by
  the brief, reading `831acef`, the head of the branch after the fix passes
  of waves 1 to 4, without sight of the slice's comparison report; 10
  findings; saved verbatim as `reviews/P2_internal.md`. Its first question:
  with `integer_vars` set, whether every path by which a point reaches the
  target or the GP training set delivers it on the integer grid and inside
  the bounds, and whether the acquisition value that was compared belongs
  to the point that is then evaluated. The sweep after it was clean in the
  three repositories, and its check scripts are kept on the orchestrator's
  machine only (`dev/scripts/runs/LOCAL.md`). Since `f873556`, which the P7
  and P9 reviewers read, two files of their slices have changed, neither in
  the lines a finding cites: `stats/get_hpd.py` (`027e972`, the order of
  equal and of integer values; the rounding is now at line 38) and
  `priors/product.py` (`b693e42`, `Product._generic` alone). Nothing below
  is verified; the orchestrator read the balanced sampler, the rounding of
  `get_hpd` and the callers that set `balance_flag`, and they are as
  described.

  The first questions. P7, the underflowed log density: `pdf` takes the log
  of a linear sum, as `vbmc_pdf.m:107-110` does, exact down to about -744
  and `-inf` beyond. The acquisitions and `fess` floor it at the log of the
  smallest positive float as their MATLAB counterparts do, so a log-sum-exp
  evaluation changes nothing there while the floors stand; the mode search
  starts at samples and component means, never at such a point; `mtv` reads
  no density; the entropies take the log of their own sums with no floor on
  either side; at the default `kl_gauss=True` the main loop reads no
  density. `kl_div(gauss_flag=False)` replaces a zero density by the
  smallest positive float, which caps the divergence near 707 without a
  message. The one reader that raised, the proposal density of the
  importance sampling, is W4-7, fixed by the wave-4 pass. In original space
  the Jacobian has the right sign and the density integrates to one;
  PyVBMC sets the rows on or outside the bounds to zero where MATLAB
  returns complex values or NaN. P7, the pair of `vp.sample`: every caller
  unpacks it but `fess` (W4-5, `dd3d1d3`); the sampler has the mixture's law
  in every case tried (plain and balanced, `K = 1`, zero weights, both
  spaces, the `df` variant) and follows `vbmc_rnd.m` line by line, except
  in the remainder of the balanced draw. P9: every sampler draws from the
  density of its class, on both sides (Kolmogorov-Smirnov at 200 000 draws
  for the four families, and per dimension with different parameters), and
  the classes reproduce a transcription of the twelve MATLAB functions to
  7e-15 in the log density with the same `-inf` sets; the exception is
  MATLAB's `msmoothboxrnd`, which for `D > 1` puts the tails of every
  dimension at the pivots of the first. No run calls `prior.sample`, in
  MATLAB either, and `_init_log_joint` combines prior and likelihood as
  `lpostfun.m` does. P2: every branch of the search delivers a point on the
  grid, the acquisition snapping what it values and the snap being
  idempotent. Off the grid are the whole initial design (the provided `x0`,
  both designs, the cache rows the design consumes), by omission and
  undocumented, the FAQ saying that integer parameters are not supported at
  all, and a repeated observation, on purpose. The Nelder-Mead branch keeps
  no bounds. On a grid the search routinely lands on an input that is in
  the training set already, which costs an evaluation and adds no training
  point.

  What can change what a run computes. At defaults, one finding, made by
  both P7 reviewers independently. The remainder of the balanced draw
  (`variational_posterior.py:643`): the builtin `sum` of a `(1, K)` array
  returns the row, so the correction that `vbmc_rnd.m:68-72` applies with
  the total is applied element by element, and the remainder is drawn from
  other probabilities; never matched (`42a0deef`, 2021-03-14). Every
  active-sampling step with `K > 1` draws its posterior and heavy-tailed
  candidates that way (`active_sample.py:968`, `:1065`), and so do
  `moments` in original space, behind `sKL`, and `mtv`. With
  `w = [0.7, 0.2, 0.1]` and `N = 13` the frequencies are `[0.724, 0.183,
  0.093]`; on the `K = 50` weights of the MATLAB fixture at `N = 100` a
  quarter of the draws are remainder draws, at a total variation of 0.20
  from MATLAB's; the bias is about one sample per component, nothing at
  `N = 1e5`. A fix leaves the number of draws alone and changes an index
  now and then, so it moves default trajectories and the
  `active_sample_step` oracle (the orchestrator's reading). In extreme or
  non-default conditions, one reviewer each unless said. `get_hpd` rounds a
  half to even where `misc/gethpd_vbmc.m:10` rounds away from zero (both P7
  reviewers). The default fraction 0.8, which every default call passes,
  cannot produce a tie: `4N/5` is never a half-integer, and the internal
  report's `N = 5, 15, 25` is wrong. A tie needs `hpd_search_frac > 0`, zero
  by default, whose Gaussians take fractions that include the exact 0.1
  (`active_sample.py:983-1004`): one point fewer than MATLAB at `N = 5, 25,
  45` live points, and an empty subset at `N = 5`. `np.round` at
  `active_sample.py:994` sits in the same branch, and `70ce067` has the
  expression to reuse. `kl_div_mvn` tests and divides raw determinants, which
  underflow for a well-conditioned covariance of small scale, and returns
  `(inf, inf)` on the default route of `sKL` (`D = 40` at SD 1e-8; six
  digits left at `D = 20`), the formula being `mvnkl.m`'s; `entmc_vbmc`
  returns NaN when a weight is exactly zero and the components are
  separated, which needs `eta` more than 745 below the largest, the soft bound
  on `eta` being gone, and MATLAB has the same structure; the guard of
  `kl_div(gauss_flag=False)` against infinite densities never fires,
  `q == 0 | np.isinf(q)` binding as `q == (0 | isinf(q))` (both reviewers;
  correct in the first port, rewritten by `4a071d8c`, 2022-09-13), and
  `np.isinf` misses the NaN that MATLAB's `~isfinite` catches; nothing
  relates the support of a prior to the hard bounds, so a narrower prior
  stops the run in the function logger with a message that names neither,
  and a wider one leaves the evidence of a truncated prior that is not
  normalized again. In P2: the value of a cached starting point is recorded
  at the point the sieve has moved, by the clip to the search bounds (after
  a warp, which resets them) or by the snap to the grid (an off-grid `x0`
  with a value), and needs valued starting points beyond the initial
  design; an acquired point equal to a row that the trim of warm-up
  switched off is pooled into that row and reaches neither the GP nor
  `y_max`, the duplicate scan of the function logger covering the rows
  that are not live, within reach on a grid alone.

  Without effect on a default run. Both P7 reviewers: `kl_div(samples=,
  gauss_flag=True)` takes the mean of all entries for the mean vector (a
  divergence of 9 comes out as 3e-4; never matched `vbmc_kldiv.m:63`); the
  positivity check of `set_parameters(raw_flag=False)` slices from the
  wrong end, refusing most posteriors of `D >= 3`, `K >= 3` with a negative
  mean and accepting a negative `sigma` at `D = 2`, `K = 2` (Python-only
  check). P7 internal: `vp.mode()`, the documented default call, raises for
  every one-dimensional posterior; the box of the mode search is offset by
  an absolute `sqrt(eps)`; the mode cache ignores `n_opts` and outlives a
  direct assignment to `mu`; `y / exp(logJ)` loses about 36 nats of range;
  `get_parameters` rescales the posterior it reads; four shape and type
  inconsistencies of `sample` and `moments`; `kde_1d` integrates to
  `n/(n-1)`, which `mtv` normalizes away; the two entropies differ in the
  weight gradient of one component, which the softmax Jacobian sends to
  zero. P7 comparison: `vp.pdf` of an integer array truncates the
  transformed coordinates, the same trap as the `np.full_like` of three
  priors (both P9 reviewers: a truncated log density, density one with an
  even number of coordinates out of the support, float32 returned for
  float32). Both P9 reviewers: `SciPy.support()` reads the standardized
  `.a` and `.b` and ignores `loc` and `scale`, and `Product` copies the
  box; `Product.log_pdf` raises on a `UserFunction` marginal, a combination
  the documentation offers, whose test calls `sample` alone; `Trapezoidal`
  warns at its lower bound and the `np.seterr` pair of the spline is not
  exception-safe; documented shapes, a parameter named twice, a sign in a
  comment. P9 internal: a NaN coordinate has full density in `UniformBox`;
  NaN and infinite constructor arguments pass every check, and with a NaN
  pivot the sampler is uniform while the density is zero everywhere;
  `sample_prior` passes the guard of the PyMC target; `tile_inputs`
  reshapes an argument that disagrees with `size`; a frozen distribution
  with vector parameters is taken for one dimension. P9 comparison: the
  classes refuse `u == v`, the tent prior, and `v == b`, which MATLAB
  computes and normalizes correctly. P2 internal, all behind options that
  are off by default: `acq_hedge=True` raises `UnboundLocalError` at the
  first acquisition, the option being neither inert nor refused;
  the deletion of the acquired point from the search set is dead code, and
  the search cache keeps that point in first place (`search_cache_frac`);
  the Nelder-Mead branch has no bounds, ignores `search_max_fun_evals` and
  takes a tolerance of the value for one of the step; the evaluations of
  the initial design are not timed; `recompute_var_post` is saved and
  restored around a block that never writes it; two normalizations of the
  covariance and a literal 3 for `active_search_bound` in unreachable
  branches of `_get_search_points`; the snap after the clip can leave a
  candidate outside the search box.

  The sheet. No entry records the mode search, which since `ba8116fa`
  (2022-11-03) starts `ceil(sqrt(K))` optimizations at the best of 1e5
  draws where `vbmc_mode.m` starts one at each of up to 20 component means
  and draws nothing; the mask of `pdf` outside the bounds; or `kde_1d`,
  an independent implementation that bins on the centres of the grid where
  `kde1d.m` bins on left edges, half a bin low, and falls back on another
  rule. The entry on `qtrapz` is wrong: the two trapezoid rules are the
  same formula, to the last bit. The entry on the hedge says that
  `idx_acq` is never chosen, not that the run raises. Both entropies
  reproduce transcriptions of the MATLAB files to 9e-16 over eight
  configurations with the draws injected, the weight gradient of `1b72896`
  included; the comparison reviewer has no finding in them.

  Tests. The tests of the three defects that both P7 reviewers found are
  blind by construction (equal weights, for which the balanced error
  cancels; one mean for every coordinate; a parameter vector negative
  throughout), and so are those of the mode, in `D = 2` under an identity
  transform. No sampler
  of a prior has a test of its distribution, which MATLAB's
  `test_pdfs_vbmc.m` has for all four (both P9 reviewers). Six statements
  of `test_vbmc_init.py` compare the log joint with the likelihood plus the
  prior and lack their `assert`. `prepare_gp_for_acq` of the oracles copies
  the lines of `active_sample` that it stands for, so the oracles through it
  pin that block against itself; no test runs the Nelder-Mead branch, the
  hedge, a search cache through `active_sample`, or the initial design with
  an integer variable. Ten defects of the MATLAB side are listed, of which
  `msmoothboxrnd` changes what a MATLAB user draws.
- [x] 2026-09-21: wave 5 verified (PI: go ahead). The ledger is
  `experiments/port_review_20260919/verification/wave5.md`, 40 rows from the
  54 findings of the five reports, each with a proposed disposition and an
  empty column for the PI's; the orchestrator verified the eight rows that
  can change what a run computes (scripts `verification/scripts/wave5_A*`),
  and three read-only Opus verifiers, one per slice, the others
  (`wave5_P7.md`, `wave5_P9.md`, `wave5_P2.md`, saved verbatim, with their 43
  scripts). The reports describe the code as it is, with the corrections
  that the ledger's last section lists, and one row alone touches a run at
  the default options. The sweep after the verifiers was clean in the three
  repositories.

  What the verification settled. The remainder of the balanced draw (W5-1)
  is a port discrepancy from 2021 that biases the count of a component by up
  to about one draw at any `N`. The corrected line moves no oracle reference
  and the acquired point of no stored state; in the four seeded gate runs it
  moves `sKL` and `r_index` of all four in the fifth digit, leaves three
  otherwise bit for bit and moves the noisy Rosenbrock run altogether, so it
  is a moving fix without statistical effect. `kl_div_mvn` on raw
  determinants (W5-2) is shared with `mvnkl.m`: at `D = 20` a posterior SD
  below 8e-9 gives `sKL = inf` and a run that is never stable, one above 5e7
  gives NaN, which `max(0, nan)` turns into `sKL = 0`. Both entropies, not
  the Monte Carlo one alone, are NaN at a weight of exactly zero (W5-3), out
  of a run's reach. The guard of `kl_div(gauss_flag=False)` against infinite
  densities is a regression of 2022 (W5-4), and the two reviewers'
  disagreement on its reach is settled: under a probit transform no density
  is infinite up to an SD of 8 in the inference space, and they appear from
  10. The rounding of a half to even (W5-6) has no tie at the defaults; the
  same convention stands at twelve sites where MATLAB has `round`. Of the
  two P2 findings that can misplace an evaluation, the value of a cached
  starting point recorded at a moved point (W5-7) is shared with MATLAB, its
  clip out of reach, the search box holding the plausible box with a margin
  of 0.385 box widths through a warp; the point pooled into a row that is off
  (W5-8) is W3-20, which the PI ruled to leave, reached by a second route.
  Among the verifiers' rows: three properties that the internal P7 report
  read as Python's are MATLAB's; the mode search, the mask of `pdf` outside
  the bounds, `kde_1d` and the argument checks of the trapezoid priors are
  deliberate differences that the sheet lacks, and its entry on `qtrapz` is
  wrong; the MATLAB fix of the weight gradient of the Monte Carlo entropy is
  present, to 1.8e-15; most of the minor P2 findings are MATLAB's own lines;
  `acq_hedge=True` raises at the first acquisition and MATLAB's own default
  fails alike; `search_cache_frac` above 0.25 raises at the second
  active-sampling step, which the fix of wave 1 for the first step had left
  standing; the documentation of `integer_vars` contradicts itself, the FAQ
  saying that integer parameters are not supported. Twelve defects of the
  MATLAB side are new and five are shared; the one that changes what a
  MATLAB user draws, `msmoothboxrnd.m` for `D > 1`, is entry 1 of
  `matlab_side_defects.md` already, and is now measured on a transcription.

  Ruled so far (PI, 2026-09-21): the Nelder-Mead value of `search_optimizer`
  goes, with its code (W5-25). No default path has reached it since wave 1
  took the one-dimensional search off it. `v1.0.4` wrote the value into the
  options of every one-dimensional run, so `load` has to meet it in a saved
  object; the row has the orchestrator's proposal.
- [x] 2026-09-21: wave 5 ruled and fixed. The PI took the orchestrator's
  proposals as they stood, ruled W5-25 during the verification (the
  Nelder-Mead value of `search_optimizer` goes, with its code) and answered
  three questions: `vp.mode()` draws from a copy of the generator (W5-14),
  `integer_vars` is experimental and documented as such, with no refusal of
  an `x0` off the grid (W5-26), and the trapezoid priors keep their refusal
  of `u == v` and `v == b` (W5-40). The rulings are in the last column of
  `verification/wave5.md`. Where the proposal for W5-7 left the choice open,
  the orchestrator took the fix, which the PI can strike before the push.

  Fixes: three Opus agents on worktrees cut at `a2104e6`, thirty commits
  (`fixes/wave5_agent_A.md`, `wave5_agent_B.md`, `wave5_agent_C.md`):
  twenty-six fixes, each with a test written against the contract and seen
  to fail on the code before it, two commits of documentation and two of
  tests; the orchestrator reviewed each diff and cherry-picked
  them, without a conflict, and made two more (a docstring of the oracle
  harness, and the requirement of W5-5 in the FAQ and on the API page of the
  priors). The variational posterior: the balanced draw, the mean of the
  samples and the guards of `kl_div`, the check of `set_parameters`, `mode`
  in one dimension, its store and its generator, the shapes and counts of
  `sample` and `moments`, integer input and the density near the top of the
  range in `pdf`, a column `x0`; `kl_div_mvn` from log determinants. The
  priors: the support of a shifted SciPy distribution, float64 input,
  `UniformBox` at a NaN coordinate, arguments that are not finite, a
  `UserFunction` marginal of a `Product`, the shape check of `tile_inputs`,
  a frozen distribution with vector parameters, `np.errstate`, the
  docstrings, and in `VBMC` the refusal of a prior that does not cover the
  hard bounds. The active sampling and the options: the Nelder-Mead value
  removed, with `load` taking a stored one for a problem of one dimension,
  where release 1.0.4 wrote it into every run, and refusing it otherwise;
  `acq_hedge` refused; the fractions of the sieve checked and the training
  rows kept out of the search cache; the stored value of a cached starting
  point reused at that point alone; a half rounded away from zero at the
  twelve sites where MATLAB has `round`, through `pyvbmc/stats/_rounding.py`;
  the documentation of `integer_vars`. Tests beyond the fixes: a
  Kolmogorov-Smirnov test of each box sampler, and the twelve statements of
  `test_vbmc_init.py` that lacked their `assert`.

  Gates. On the seventeen commits without the sampler, the four seeded runs
  bit for bit the baseline (92 arrays) and the exact oracle check 11 of 11.
  On the thirty, the four runs bit for bit the record that the verification
  had made with the corrected sampler alone, so the pass moves what W5-1
  moves and no more (`sKL` and `r_index` of all four runs in the fifth
  digit, and the noisy Rosenbrock run from iteration 13 on); the exact
  oracle check 11 of 11 with nothing re-baselined. On `a093f2e`: the default
  suite, 1843 passed and 58 skipped with no reruns; the Torch environment,
  886 passed and 1 skipped, the seeded references of S-VBMC among them; the
  PyMC environment, 107 passed. W5-1 joins the moving fixes that the one
  regeneration of the golden references waits for. Records brought to the
  state of the code: the ledger (fix commits, gates, what the pass found),
  the sheet (eight new entries, five amended, three bullets under the
  settled non-differences, and its Python line citations carried to the
  present lines), the list of MATLAB-side defects (entries 25 to 40),
  `CHANGELOG.md`. Not pushed; the CI matrix has not run. The worktrees and
  branches of the three agents are removed, `git cherry` showing every one
  of their commits on the branch.
- [x] 2026-09-21: the wave-2 pass checked independently (PI: `/doublecheck`,
  read-only, by a second session while the wave-5 pass was closed in the
  main checkout; then, on the PI's word, fix what matters). Six fresh Opus
  reviewers without the session's context, on the head of the branch 144
  commits after the pass: the fixes of the warm-up, the termination and the
  initial posterior; the rest of the loop, the warp branch and the final
  boost; the options and inputs; `load`, the state and the pickling; the
  ledger and the developer records; the user-facing records and the
  consequences outside the changed lines. All 47 commits hold and no later
  pass undid one. What they found is fixed in 18 commits on the branch
  `dev-port-review-w2check`, cut at `0bf7963` in a worktree of its own so
  that the gates of the wave-5 pass ran on an untouched checkout, ten of
  them code or tests: `load` checks the limits on iterations and evaluations
  (three reviewers, independently; the check of the wave-3 pass had left
  them out of `load`); the `x0_orig` of a file saved without it comes back
  through the map of construction; the fixed-means boost refuses a GP with
  too few inputs; an option the user set keeps its description;
  `specify_target_noise` is read as a boolean, made under the PI's rule for
  input handling and not ruled on by itself; the titles of the final plots
  count the iterations; plausible bounds without `x0`; a test of the
  minimum-iteration guard that tested nothing; two back-fills. In the
  records: the note of `save` and `load` said that a saved run holds
  bytecode only for a target that cannot be pickled by name, where the nine
  options whose value is a function are stored by value in every file;
  `refresh_citations.py` had carried three citations of MATLAB and ini files
  through Python files that day (`9f6f0c7`) and is corrected, with an audit
  of the 112 citations it rewrote; the counterpart map, the header of the
  MATLAB-side defects, the changelog's "Upgrading" list, the FAQ on
  continuing a run and the stored output of example 2.
  `verification/wave2.md`, "The independent check of the pass", has the
  findings, what is left for the PI and the gates: the tests of every module
  touched, 413 passed; the oracle tests, 143 passed; the exact oracle check,
  11 of 11 with nothing re-baselined. The whole suite, the four seeded runs,
  the Torch and PyMC environments and the CI matrix have not run on the
  branch, which waits to be brought onto `dev-port-review`.
- [x] **2026-09-21, wave 5: the independent check of the pass, and its fix
  round.** On the PI's instruction (`/doublecheck`) five fresh read-only Opus
  reviewers read the pass on `0bf7963`: R1 the commits on the variational
  posterior, R2 the priors, R3 the active sampling and the options with the
  reach of the pass, R4 the ledger against its sources, R5 the user-facing
  texts and the records; 68 findings, their raw reports on the
  orchestrator's machine (`dev/scripts/runs/LOCAL.md`). Three had to be
  fixed. The check of a prior against the hard bounds compared exactly and
  refused the FAQ's own `uniform(loc=low, scale=high - low)` for about a
  quarter of decimal bounds. The sentence "with one variable a bounded
  scalar search is used whatever the option holds", which came from the
  orchestrator's brief, is false for `search_optimizer="none"`. And the
  ledger's W5-6 said that the rounding fires in no run at the defaults,
  where a starting cache reaches a tie, at which the rounded shares of the
  sieve could claim more than the whole and end the run with a `ValueError`.

  PI rulings: a slack of `1e-9 * (ub - lb)` in the support check, none at an
  infinite bound; the sieve caps each source at what is left, the check at
  construction staying; `mode(n_opts=k)` neither reads nor writes the stored
  mode; the sample count of `to_arviz` waits for slice N2; W5-7 stays; the
  should-fix items taken, with a list of optional ones.

  `dev-port-review-w2check` was merged first, as a fast-forward (`b1bab4d`;
  its worktree removed, and the branch, which was never pushed, removed
  later that day on the PI's word). Two Opus agents on worktrees
  cut there made thirteen commits (`fixes/wave5_check_agent_D.md`, priors;
  `wave5_check_agent_E.md`, active sampling, option checks and variational
  posterior), each fix with a test seen to fail on the code before it but
  for two commits of tests, two removals of dead code and the docstrings;
  the orchestrator reviewed the diffs, cherry-picked them without a conflict
  and wrote the texts: the FAQ, the API page of the priors and the README of
  the variational posterior, the changelog (against 1.0.4, which is why it
  does not follow the agents' sentences where those describe the branch),
  the sheet with its citations carried by the `refresh_citations.py` that
  w2check brought, entry 35 of `matlab_side_defects.md`, and the ledger:
  `verification/wave5.md`, "The independent check of the pass", has each
  finding with its commit, what is left and why, and the argument that the
  tie of the starting cache is the only one within reach of the shipped
  options. `extract_report.py` takes a chosen candidate, since an agent that
  hands its report back through a tool call and then closes under the same
  title leaves two.

  Gates on `f86d373`, once for the merge and the round: the four seeded runs
  bit for bit the record after the pass (92 arrays, 0 differ), as on the
  merged head before the round; the exact oracle check, 11 of 11 with
  nothing re-baselined; the default suite, 1961 passed and 58 skipped with
  no reruns; the Torch environment, 1009 passed and 1 skipped; the PyMC
  environment, 107 passed.

  One more fresh Opus reviewer, read-only (R6), read the round alone on
  `f86d373`, with the ledger's section: nothing that had to be fixed. It
  re-derived that the sieve returns the number of points asked for in every
  case and draws what it drew wherever no cap binds, that a point in the gap
  of the slack is never evaluated, that a property shadows the `a` and `b`
  left in the dictionary of an older pickle, the changelog's statements on
  1.0.4 at the tag, the sets of `Nrnd` at which MATLAB's rounded shares
  overshoot, and the argument on the other rounding sites, in which it found
  one error of the orchestrator's: the burn-in is the thinning times the
  count of samples after it is rounded, a whole number, and not
  `400 / sqrt(N)`, so its rounding meets no tie at all and the changelog no
  longer lists it. Its four other should-fix findings were records: this
  worklog's in-flight entry, stale once the commits were on the branch; the
  false sentence on the one-variable search in agent C's raw report, which
  carries a flag; a sentence of the sheet on what the search cache keeps;
  and R1-13, which the ledger's section named under R4-11 alone. Two
  docstrings followed (`5c4fc87`). The report and its scripts are with those
  of R1 to R5; the ledger's section has what is left.

  What the round left for the PI, and the decisions (2026-09-21). Two fixes
  before the merge,
  made by the orchestrator, each with a test seen to fail first: `moments`
  hands its count to `sample` as it is given, where its `int()` truncated a
  fractional one, in `vp.moments` and in `vp.kl_div(gauss_flag=True)`
  (`6d492a2`; the part of R1-3 that no ruling had covered), and `cache_frac`
  is checked to lie in [0, 1] with the five fractions, at construction and in
  `load` (`6dcd027`; above one it could overfill the search set, and a
  negative one took all but so many rows of the starting cache); the
  changelog, the sheet and the ledger have both. Gates on `6dcd027`: the four
  seeded runs bit for bit the record after the pass (92 arrays, 0 differ),
  the exact oracle check 11 of 11, the modules of the files touched. W5-7
  stays: with it the worst case is one evaluation that was not needed,
  without it a provided value recorded at a point that the grid or the
  search box moved. A `SciPy` or `Product` prior pickled from now on carries
  no `a` and `b` in its dictionary, which concerns a file written here and
  read by 1.0.4 alone: no action. None of the seeded gate runs is given a
  prior: one with `prior=` (the FAQ's list of `uniform` marginals built from
  the hard bounds, which reaches the slack, and a `SplineTrapezoidal` on
  them) joins the gate when the golden references and the run pools are
  regenerated, and `dev/TODO.md` gets the line at the merge into `dev-next`.
  The count of `to_arviz`, which the ruling had left to slice N2: N2 has no
  pass ahead of it, wave 0 having taken it, so the PI had it fixed after the
  merge (`f9ab814`). The exports of the variational posterior and of the PyMC
  target take the counts that `sample` takes, a whole float such as `1e3`
  among them, through one private helper that `sample` uses too; a boolean and
  a count below one stay refused, and a refused count draws nothing. Gates on
  `f9ab814`: the four seeded runs bit for bit the record after the pass (92
  arrays, 0 differ) and the exact oracle check 11 of 11, `sample` being on the
  path of every run; the tests of the variational posterior in the default
  environment (131 passed), of its directory in the environment that has
  ArviZ, where the tests of the export run and do not skip (273 passed, 1
  skipped), and of the adapter in the PyMC environment (110 passed).
- [x] 2026-09-21: wave 5 through CI, and merged (PI: go). Pushed on
  `63a0808`; the branch smoke and the full matrix, nine cells, are green at
  the first attempt. `dev-port-review` merged into `dev-next` (merge commit
  `03a8d46`, on `dev-next` at `831acef`, which had not moved), the
  review's status in `TODO.md` updated there with the item on a seeded gate
  run with a prior, and `dev-port-review` fast-forwarded onto it.
- [x] 2026-09-21: the count of the ArviZ exports through CI, and merged (PI:
  push). Pushed on `0c67e30`; the branch smoke, whose cell has the extras and
  so runs the tests of the exports, and the full matrix, nine cells, are
  green at the first attempt. `dev-port-review` merged into `dev-next` (merge
  commit `1563053`, on `dev-next` at `e71c667`, which holds the item of
  `TODO.md` on the branch `feat-3d-animation`), and `dev-port-review`
  fast-forwarded onto it.
- [x] 2026-09-21: wave 6 run and reported to the PI (PI: go, on the briefs
  as the session showed them): G1 internal (12 findings and 13 minor
  observations), G1 comparison (7 and 14), G2 internal (6 and 15), G2
  comparison (11 and 10); four fresh Opus reviewers by the brief, read-only,
  reading gpyreg at `fdbafdf`, whose code under `gpyreg/` is that of the pin
  `9e70e6b`, with PyVBMC at `8af5daac` as the caller and, on the comparison
  track, `gplite/` of MATLAB VBMC at `396d649`; reports saved verbatim under
  `experiments/port_review_20260919/reviews/`. What the briefs added to the
  section "Reviewer brief": the comparison reviewers were told that gpyreg's
  copy under `matlab/gplite/` is not the reference, and the internal ones
  not to open it; every brief listed how PyVBMC uses gpyreg, and every
  finding says whether a default run reaches it; a GP fit on a few dozen
  points counted as a small check, every script with BLAS single-threaded;
  the reading order put last what O4 reads again (the gradient of the
  marginal likelihood in G1, the `compute_grad=True` branches of the
  components in G2). First questions. G1: the plan's on `uuinv`, put to the
  comparison reviewer with `46b6f5e` and `fcf2674` beside `1d1f20d` and to
  the internal one as whether the inverse is right in every configuration of
  the four bounds; and one that the orchestrator added, whether the rank-one
  extension of `update` equals a full recomputation in every state that
  `update` accepts. G2: the plan's on the predictive log-density, spelled
  out over the noise parameterizations and over the noise-inclusive outputs
  of `predict`, `predict_full` and `quad`. The sweeps before and after the
  wave were clean in the three repositories, and the reviewers' check
  scripts are kept on the orchestrator's machine only
  (`dev/scripts/runs/LOCAL.md`). Nothing below is verified; where it says
  that the orchestrator read a code site, the site is as the report
  describes it. The line citations of both G1 reports into
  `gaussian_process.py` have drifted, the internal one's by up to 37 lines
  (`:1494` for `:1457`) and the comparison one's by up to 14; the G1
  verifier's report gives the present line for each (corrected on the day:
  the entry had the comparison report's citations holding, which was true of
  the five that the orchestrator had looked up).

  The first questions. `uuinv`: the comparison reviewer finds `1d1f20d`
  carried line for line in the three branches (`f_min_fill.py:192-255`),
  the out-of-range guard that sits in the general branch alone included,
  `46b6f5e` carried by construction and `fcf2674` carried
  (`slice_sample.py:452-453`). The internal reviewer finds the inverse exact
  to 2e-16 in every configuration of the bounds for the mixture that the
  body implements, which is not the one the docstring names: the body
  spreads `1 - w` over the two tails in proportion to their lengths, the
  docstring gives each tail `(1 - w)/2`. The docstring's sentence is the
  comment that `1d1f20d` added to `fminfill.m` (the orchestrator's reading
  of the commit), so the code agrees on both sides and the sentence is wrong
  on both. The rank-one update: both reviewers derived the two branches
  again and find a rebuild reproduced to 1e-15 (comparison) and 1e-13
  (internal) in `alpha` with `L_chol` true, at the three noise
  parameterizations, with and without `s2_new`, with several samples and
  over consecutive updates; with `hyp` passed `update` recomputes in full,
  bit for bit a rebuild. A stored `sn2_mult` of 10 to 1000, which the
  internal reviewer forced through a patched Cholesky factorization, is
  carried correctly (7e-14); the comparison reviewer could not produce one
  with any noise floor. With `L_chol` false the agreement is 2e-10 at a
  noise variance of 1e-7, and with an exact duplicate at a noise of `eps`
  `alpha` is wrong by O(1) with no warning, the guard `sqrt_arg <= 0`
  standing in the other branch alone (G1 internal F4). No PyVBMC run has
  `L_chol` false: `_gp_hyp` bounds the noise variance from below at 1e-5,
  against the threshold 1e-6 (both). The predictive log-density: both G2
  reviewers find the total variance, `max(s2, 0) + sn2_star * sn2_mult`
  (`gaussian_process.py:2032`), in every parameterization, with and without
  `s2_star` and whatever `add_noise`; it entered gpyreg in that form with
  `9b3d46b` (2022-06-02), 23 days before `68a197b`, so MATLAB's fix brought
  MATLAB to gpyreg. `predict` reproduces a transcription of `gplite_pred.m`
  to 8e-14 over the parameterizations, the flags, one and three samples and
  both representations of `L`, and the mean of `quad` that of
  `gplite_quad.m` bit for bit. `sn2_mult` multiplies the whole noise of a
  test point on both sides (checked at 10 and 1e7 by the internal reviewer),
  and `quad` adds no noise. One difference of kind: with
  `separate_samples=False` gpyreg returns the log density of the
  moment-matched Gaussian where MATLAB returns the matrix per sample; the
  docstring says so, the sheet does not, and no PyVBMC caller passes
  `return_lpd`.

  What a default run reaches. (1) Recommendations pooled over the
  dimensions, both G2 reviewers independently; the orchestrator read the
  sites on both sides. `covariance_functions.py:451` starts every length
  scale at the log of one standard deviation taken over all entries of `X`,
  where `width`, nine lines above, is taken with `axis=0` and
  `gplite_covfun.m:126` has `log(std(X))`, one per column; and
  `mean_functions.py:488`, `:511-524` take `w`, `min`, `max`, `median` and
  `std` over all entries for the bounds, the plausible bounds and the
  starting values of the location and the scale of the negative quadratic
  mean, where `gplite_meanfun.m:142`, `:220-230` has them per column (the
  comparison reviewer alone has the bounds). PyVBMC puts both `x0` into
  `hyp0` (`gaussian_process_train.py:334-365`) and leaves the bounds of the
  mean's location and scale to gpyreg, which fills them at every fit (no
  statement of `:371-416` sets them; the orchestrator's reading). On
  coordinates with SDs of 0.4, 1
  and 4 the comparison reviewer measures a starting length scale seven
  times MATLAB's in the narrow dimension and one box for the mean's
  location where MATLAB has three; the internal reviewer calls the effect on
  PyVBMC small, the transformed coordinates being comparably scaled, and
  limits the reach of `x0` to a fit with no earlier fit to start from. The
  reach and the size on a PyVBMC state are for the verification. Since the
  port on both counts; the sheet's entry on the quantile and
  standard-deviation conventions calls the difference small and dependent
  on `N`, which both reviewers contest: `ddof=1` is MATLAB's normalization
  already, and what differs is the axis. A fix moves the GP fit of a
  default run. (2) The window of the burn-in statistics of the slice
  sampler, G1 comparison F1; the orchestrator read both sites.
  `slicesamplebnd.m:362` accumulates over `ii > burn/2` and divides by
  `floor(burn/2)`, one term more than the divisor when the burn-in is odd,
  so that its variance estimate is negative wherever a coordinate's SD is
  small against its mean, the square root complex, and `:371` then discards
  the adapted widths of every coordinate; `slice_sample.py:562` accumulates
  `floor(burn/2)` terms and has a variance. PyVBMC's burn-in is odd in a
  large part of its fits (`thin * 3 = 15`, and `thin * gp_s_N` for an odd
  count of samples), as MATLAB's is (`get_GPTrainOptions.m:108`). For an
  even burn-in the reviewer's line-by-line transcription of
  `slicesamplebnd.m` and gpyreg's sampler are bit for bit the same, samples
  and widths, which settles every other line of the sampler; at `burn = 15`
  on a 34-point GP the adapted widths differ by up to a factor of 19. The
  defect is MATLAB's and gpyreg is the consistent side: a sheet entry and a
  line of `matlab_side_defects.md`, no fix, if the verification bears it
  out. Beside it, gpyreg floors a negative estimate at zero, which leaves a
  coordinate a width of zero (the reviewer's M6). (3) G1 internal F9: a
  training set whose targets are all equal gives the output scale the
  bounds `(-inf, -inf)`, which L-BFGS-B answers with a `KeyError`; a run
  reaches it with a log joint that is constant over the training inputs.
  (4) The variance over the hyperparameter samples has `ddof=1`
  (`gaussian_process.py:2049`, behind `lcb_max`), which the internal G2
  reviewer passes to the comparison track: `gplite_pred.m:157-160` divides
  by `Ns - 1` as well (the orchestrator's reading), so it is MATLAB's
  convention.

  Without effect on a default PyVBMC run, in gpyreg's public interface. All
  four reviewers: the eigenvalue fallback of `__robust_cholesky`
  (`gaussian_process.py:2620-2622`) flips single entries of the eigenvector
  matrix where `gplite_rnd.m:97-99` flips columns, so `T'T` is not the
  matrix (errors of the size of the matrix itself; two identical test
  points drawn with correlation -0.76), and with a negative eigenvalue of
  rounding size left, the ordinary case on a dense grid, it returns zeros,
  so that `random_function` returns the predictive mean without a word
  where MATLAB fails on the sizes (G2 comparison F6). Both G2 reviewers:
  `predict_full(add_noise=True)` adds a column broadcast over the rows, not
  a diagonal, for noise that varies by point (`:1852`); the variance of
  `quad` derives the scale of the factorization anew from the training
  noise where `Posterior.sl` stores it, and collapses to `eps` after a
  rank-one update with a point of lower noise (`:2188-2196`, in the fix of
  `cec1f85`); `RationalQuadraticARD.get_bounds_info` writes the plausible
  upper bound of the shape into the slot of the output scale
  (`covariance_functions.py:414`); the isotropic bounds take `min` and
  `max` of a scalar and the log of the mean width where
  `gplite_covfun.m:112-119` has the mean of the logs. Both G1 reviewers:
  `df == 0 | ~np.isfinite(df)` binds as `df == (0 | ...)` in the masks of
  the hyperpriors (`:1441`, `:1457`), so a prior with `df = inf`, the
  Gaussian of `gplite_nlZ.m`'s header, is dropped while its normalization
  constant is still subtracted, `__recompute_normalization_constants`
  having the test right (the slip of W5-4 again); `fit` writes its low-noise
  start through a view into the design and takes the default sampler widths
  from the design so altered (`:1243`, `:1259`, `:1262`), after the write
  where `gplite_train.m:206-207` takes them before, from a copy: gpyreg's
  own default options reach it, PyVBMC's never (`opts_N` of 0 or 1, or no
  samples); a single-point `update` on posteriors that `clean()` has
  emptied takes the rank-one path and raises. G1 internal alone: the
  normalization constant of a smooth-box prior is not masked to the
  coordinates it multiplies, so a smooth box over two or more
  hyperparameters gives twice the log prior or raises (F2); `fit` reads
  `options["sampler"]` and documents `sampler_name` (F5); `log_likelihood`
  and `log_posterior` raise on the dictionary their docstrings offer (F6);
  an infinite width passes the check of `SliceSampler` and returns after
  burn-in through `base_widths` (F10); a free parameter that never moved is
  flagged only if its constant value averages without rounding, 0 and 1,
  the two values the tests use (F11). G1 comparison alone: the test for no
  prior has `and` where `gplite_hypprior.m:35` and `fminfill.m:73` have
  `||`, which the smooth-box families need, and leaves MATLAB's documented
  flat prior, `sigma = Inf`, a log posterior of NaN (F4); the noise
  gradient for a constant total noise with more than one noise
  hyperparameter indexes `dsn2` along the wrong axis and raises, where
  `gplite_core.m:252-253` reads the wrong entry by linear indexing, no run
  reaching it since level 1 always has `s2` (F6); neither the design nor
  the optimizer loop catches an exception of the objective, where MATLAB
  catches in three places (F7). G2 comparison alone: the gradient of the
  Matern kernel of degree 1 is NaN on the diagonal on both sides, MATLAB
  carrying the repair as a commented-out line (F11). On the PyVBMC side,
  G1 comparison M2: `hyp_dict["logp"]` is given `res["log_priors"]`, an
  array of zeros, where `gptrain_vbmc.m:66` stores the thinned log
  posterior; nothing reads it on either side.

  The sheet. Contested: the entry on the quantile and standard-deviation
  conventions (above). The figure of 1e-15 in the entry on the rank-one
  update holds for `L_chol` true, the one representation a run has, and a
  second difference of state stands beside the `sn2_mult` it names: a new
  point whose noise falls below 1e-6 leaves `L_chol` true where a rebuild
  has false. Without an entry: the variance of `quad` since `cec1f85`,
  which is right where `gplite_quad.m:66-67` normalizes by the constant
  noise term alone and clamps to `eps` under heteroskedastic noise (2001
  grid points against both); the averaged log density; the three additions
  to the hyperprior (the smooth-box families, the renormalization over the
  bounds, the fixed prior) with the `and` they need; the window of the
  burn-in; the stability guard of the rank-one update and the zero fill of
  `s2`; the default `hyp0` of `fit`.

  Tests. `test_predict_lpd` and its isotropic twin zero their `s2_star`
  and hand `update`, which does not check the width, a row of 12
  hyperparameters for a GP of 11, so that the noise entry is read as the
  mean's constant: they pin the log density at a total noise of `eps`
  alone (both G2 reviewers). No test reads `PLB`, `PUB` or `x0` of any
  component, and `test_setting_bounds` compares the assembler with the
  components' own values; `random_function` is tested for shapes; the
  Matern gradient tests have degree 3 alone; no test of the sampler has an
  odd burn-in or compares a chain with a reference; the tests of the
  rank-one update assert `L_chol`, use one sample and `sn2_mult = 1`;
  `test_uuinv` asserts the length-weighted masses of the body; the two
  tests of a frozen chain pass on 0 and 1 alone; the fixture of the priors
  puts each smooth box on a single hyperparameter, the one case that F2
  leaves right. Well tested, against the specification: `quad` with noise
  against Gauss-Hermite quadrature of `predict_full` (on a fresh posterior
  only), the cross-covariance interface, `_solve_triangular`.

  The MATLAB side, for `matlab_side_defects.md` after verification: the
  burn-in window of `slicesamplebnd.m`; the variance of `gplite_quad.m`
  under heteroskedastic noise; the linear index of `gplite_core.m:252-253`;
  the Matern gradient of degree 1; the tolerance of `robustchol`, which
  refuses eigenvalues of rounding size; `gplite_post.m:250`, which leaves
  `gp.s2` shorter than `gp.X` after a rank-one update without `s2`;
  `gplite_train.m:298` with `Ninit == 0`; the comment of `uuinv`.
- [x] 2026-09-21: wave 6 verified (PI: go, on the orchestrator's plan: the
  verification alone, no wave beside it). The ledger is
  `experiments/port_review_20260919/verification/wave6.md`, 37 rows with a
  proposed disposition each and the PI's column empty. The orchestrator took
  the four rows that can change what a run computes
  (`verification/scripts/wave6_A*.py`, with `wave6_per_column_patch.py`), two
  read-only Opus verifiers the rest, one per slice (`verification/wave6_G1.md`
  and `wave6_G2.md`, saved verbatim, their scripts as
  `scripts/wave6_G1_*.py` and `wave6_G2_*.py`); the G1 verifier also took the
  candidate on `load(new_options=)`. The verifiers ran first and the
  orchestrator's seeded runs after them. The sweep after the verifiers was
  clean in gpyreg and in vbmc.

  Part 1. W6-1, the recommendations pooled over the dimensions, is larger
  than either reviewer had it and reaches every run: the pooled statistics
  mix the locations of the columns with their spreads (in the cigar states a
  pooled SD of 35.7 for 0.5 to 2.5 per column, starting length scales 15 to
  74 times gplite's, one hard box `[-107, 113]` for the mean's location in
  every dimension). `x0` reaches the first GP training of a run alone; the
  hard and the plausible bounds of the mean's location and scale, which
  `_gp_hyp` leaves to gpyreg, reach every fit. With the statistics per
  column, in the process alone, the `gp_fit` oracle gives other samples in
  the seven stored states that sample (it reproduces its reference bit for
  bit with gpyreg as it is), stored samples lie outside gplite's hard box in
  five of the eight states, and the four seeded runs of the gates differ from
  their record from the first evaluation after the initial design (82 of 92
  arrays). Never matched, no revision of gpyreg ever had the axis, no record
  of a reason. Proposed: fix in gpyreg, last and alone, read for accuracy on
  the benchmark targets before and after. W6-2, the window of the burn-in
  statistics: MATLAB's defect, as the reviewer has it; with MATLAB's window
  installed by an edit of the source, a burn-in of 16 gives gpyreg's samples
  bit for bit and the burn-in of 15 that every stored state records makes
  MATLAB's estimate negative in 5 to 11 coordinates in each of the seven
  states that sample, so MATLAB always discards its adapted widths there.
  No fix; a sheet entry and a line of the MATLAB-side list. W6-3, the floor
  at zero under that estimate: with a burn-in of 2 or 3 gpyreg's window
  holds one iteration, every width is zero and the chain returns one point
  (the G1 verifier's observation, run again by the orchestrator); a user's
  `gp_sample_thin = 1` makes the burn-in 3. W6-4, equal targets: the
  `KeyError` needs a problem without hard bounds, the log Jacobian making the
  targets differ otherwise, and there a log joint that is constant over the
  ten points of the initial design is enough; the pair `(-inf, -inf)` comes
  from formulas that gplite has as well.

  Part 2, 33 rows, none of which changes a number of a PyVBMC run at a
  shipped option. Of the findings of the four reports none is refuted
  outright; three that the internal G1 report takes for gpyreg's are MATLAB's
  as well (the variance clamp before the rank-one update, the order of the
  base widths of the sampler, the split of an odd number of samples), and
  its `update` without hyperparameters does not raise but returns NaN
  factors. The verifiers' own: `sp.linalg.eig` breaks the fallback of
  `__robust_cholesky` before any sign flip, so the fix takes `eigh` as well
  (both verifiers); `pL` is computed before the test for a failed
  factorization in the low-noise branch; MATLAB's own rank-one extension is
  wrong under noise that varies by point, by 0.116 relative, latent in VBMC;
  the fix of `uuinv` was gpyreg's first (`00d9406`, nine days before
  `1d1f20d`, same author), and the wrong sentence on the mixture with it;
  `load(new_options=)` passes by the checks of `_init_optim_state` beside
  taking no effect (`gp_mean_fun="nonsense"` is accepted). Thirteen new rows
  for the MATLAB-side list, one entry of the sheet replaced, two amended and
  six new, a seventh depending on the ruling on W6-19. The rows on which the orchestrator asks the PI and does not only
  propose: W6-1 (the moving fix), W6-4 (refuse equal targets, or take a range
  of one), W6-8 (a draw on a dense grid), W6-19 (exceptions of the objective:
  loud as now, or MATLAB's tolerance), W6-21 (the Metropolis step that never
  ran: repaired or removed).
- [x] 2026-09-22: wave 6 ruled (PI: the orchestrator's recommendations on
  the five open rows, and the proposals on the rest, written into the PI
  column of `verification/wave6.md`, `b7622e5`): W6-1 fixed, with the
  benchmark sweep as the gate that decides whether it is taken; W6-4 a range
  of one for equal targets, with a warning; W6-8 both parts; W6-19 left
  loud, with a sheet entry; W6-21 the Metropolis step removed. The thirteen
  rows of the MATLAB side are entries 41 to 53 of `matlab_side_defects.md`.
- [x] 2026-09-22: the wave-6 fix pass, part 2 (PI: go ahead, on the
  orchestrator's plan). How an agent works on gpyreg, settled with the PI:
  worktrees of `../gpyreg` made by hand on branches cut from gpyreg's `main`
  at `fdbafdf`, the code of the pin. Three Opus agents, one commit per row
  with a test seen to fail first, one test file at a time, no push: A on
  `../gpyreg-port-review-A` (`gaussian_process.py` and its tests, 19
  commits), B on `../gpyreg-port-review-B` (the other modules, their tests,
  `docsrc/`, 13 commits, the seeding of the sampler's unseeded tests among
  them), C on a harness worktree of PyVBMC (W6-23 and W6-37, 2 commits).
  Reports saved verbatim under `experiments/port_review_20260919/fixes/`.
  The orchestrator reviewed the diffs and cherry-picked: C's onto
  `dev-port-review` (`4d3c1515`, `e657eba8`, with the audit entries of the
  three fit-history captures through the generator's rebaseline mode, every
  replayed output moved by zero, `0a6682fc`); B's then A's onto gpyreg's
  `port-review-wave6` in `../gpyreg-port-review` (32 commits, no conflict,
  hashes in `verification/wave6.md`, "Fix commits"). W6-38, an inverted
  plausible pair in `fit` that B found on the way, is fixed as the last
  commit of that branch, unruled, for the PI to strike. gpyreg's suite on
  the assembled branch, 293 passed. Gate 1 against it, `PYTHONPATH` naming
  the worktree: the exact oracle check 11 of 11, the four seeded runs bit
  for bit the record after the wave-5 pass, so no part-2 row changes a
  number of a run. The benchmark sweep before W6-1 is recorded (eight
  targets; the log on the orchestrator's machine). C's worktree is removed;
  A's and B's stand until `git cherry` has been read once more before the
  pull request.
- [x] 2026-09-22: the three rulings, W6-1 made and taken, the re-baselines
  and the records (PI: go ahead on the orchestrator's plan, then "taken" on
  the sweep). The rulings: W6-38 kept; `test_fitting` seeded in this pass;
  the guard of W6-9 left as made. On gpyreg's `port-review-wave6`, after
  `caacbc1` and not pushed: `8dec795` (`test_fitting` seeded in both GP test
  files, the fit within 0.042 of the generating hyperparameters, a margin of
  0.46 to the tolerance of 0.5), `f76eca2`
  (W6-1: `axis=0` in the nine statistics of `X` of the two helpers and the
  rational quadratic copy, the wrong comment removed, four tests in
  `test_bounds_info.py` seen to fail first; gpyreg's suite 297 passed) and
  `dc2a930` (the release notes, a `1.3.0 (unreleased)` section from the
  agents' sentences, the number for the PI to confirm). Gate 2 against the
  branch, `PYTHONPATH` naming the worktree: the exact check moved `gp_fit`
  and `gp_fit_history` in the seven states that sample and the three
  fit-history captures, and `active_sample_step` and `gp_nlZ`, which the
  previous pickup point expected to move, not at all; the four seeded runs
  differ from the wave-5 record in 82 of 92 arrays and are bit for bit the
  verifier's prototype record. The after-sweep on the same eight targets
  and seeds as the before-sweep, the run after better on 15, 18 and 19 of
  the 34 seeds by the three metrics; the PI took W6-1 on the orchestrator's
  summary of it, which overstated the result (the next entry). The targeted
  re-baselines on the generating machine, `gp_fit` and `gp_fit_history` on
  the eight states and the three
  captures, each with its reason in the fixture (two captures moved in their
  default sampler widths alone), exact check 11 of 11, `3123135e`. The
  records: the ledger (W6-1 and W6-38 ruled, "Fix commits", "Gates"); the
  sheet, by `records_sheet.py` in the after-pass form (the quantile entry
  narrowed, the rank-one and isotropic entries amended, eight entries and
  two bullets of settled non-differences added; its citations into gpyreg
  at the pin, checked); the changelog (W6-37's entry and upgrading line, the
  gpyreg requirement with the GP-fit change the pin move brings, held back
  by the next entry until the pin moves, the `logp` key);
  `matlab_side_defects.md` entries 45 to 48 and
  53 checked against the branch, unchanged; `dev/scripts/runs/LOCAL.md`.
- [x] 2026-09-22: the independent check of the wave-6 pass, the reopened
  ruling on W6-1, and the fix round (PI: `/doublecheck`; then the eight
  decisions and the corrections as the orchestrator proposed them; W6-1
  reopened, then kept). Five fresh read-only Opus reviewers on gpyreg
  `dc2a930` and PyVBMC `a3d4a70d` (R1 the posterior, the rank-one update,
  the prediction, `quad` and the random draws; R2 the priors, the bounds of
  `fit` and its interface; R3 the other modules, W6-1 and the release notes;
  R4 PyVBMC's commits and its coupling with the branch; R5 the records)
  returned 7, 13, 11, 9 and 30 findings; `verification/wave6.md`, "The
  independent check of the pass", has the account, and the raw reports are
  on the orchestrator's machine (`dev/scripts/runs/LOCAL.md`). No gate of the
  pass had run PyVBMC's suite against the branch; run then, one test failed:
  `set_priors` refused the block prior with a coordinate without one that
  the dormant rectified-noise branch of `_gp_hyp` writes. The orchestrator's
  summary of the sweep, on which the PI had taken W6-1, was wrong (it said
  four targets better on every mean, where two are, and no target beyond
  its seed spread, where four are); the PI reopened the ruling, the four
  targets in question were run to ten seeds, and a probe of the mean's
  bounds (`verification/scripts/wave6_A1d_mean_bounds_probe.py`) found that
  the location of the mean leaves gplite's per-column box with a
  differently shaped quadratic and does not crowd the box's edge when
  confined, which the orchestrator reads as the box removing a separate
  mode of a misspecified mean rather than truncating a posterior that
  presses on it; the PI kept W6-1. The fix round: agent D on a worktree of gpyreg made by hand
  (`../gpyreg-port-review-D`, from `dc2a930`), 16 commits taken onto
  `port-review-wave6` as a fast-forward (`c4a441e` to `d93f03c`), its report
  `experiments/port_review_20260919/fixes/wave6_check_agent_D.md`; the
  orchestrator's `edc42739` (`load` gives a stored `integer_vars` that fails
  the check the mask the run was made with, as release 1.0.4 stored it) and
  `326c7676` (a scan test keeps `_CONSTRUCTION_ONLY_OPTIONS` in step with the
  code). Gate 3 against `d93f03c`, `PYTHONPATH` naming the worktree:
  gpyreg's suite 313 passed; PyVBMC's suite 1990 passed, 58 skipped; the
  exact oracle check 11 of 11; the four seeded runs bit for bit gate 2's record (92 arrays, 0 differ). The
  records: the ledger (rows W6-1, W6-16, W6-21 and W6-38, "Fix commits",
  "Gates", the section on the check), the sheet
  (`records_sheet_check.py`: citations and claims corrected, the history
  phrasing restated as facts, five entries added), the changelog (the gpyreg
  requirement held at 1.2.1 until the pin moves, `load` with the
  `integer_vars` of 1.0.4, the `logp` line of the upgrading list),
  `matlab_side_defects.md` entries 41 and 49 to 52, `TODO.md` (what the
  check left for later), the sheet's PyVBMC citations carried by
  `refresh_citations.py`, `dev/scripts/runs/LOCAL.md`.
- [x] 2026-09-22: the independent check of the fix round, and its fix
  round (PI: `/doublecheck`; then the four decisions and the corrections as
  the orchestrator proposed them). Three fresh read-only Opus reviewers (F1
  agent D's gpyreg commits and the release notes; F2 PyVBMC's `edc42739`
  and `326c7676`; F3 the records) returned 8, 8 and 27 findings;
  `verification/wave6.md`, "The independent check of the fix round", has
  the account, and the raw reports are on the orchestrator's machine. What
  had to be fixed: `random_function` still raised in the low-noise
  representation of the posterior, where a default fit on noiseless
  targets ends (the band of the round reached the Cholesky representation
  alone); and three records, the orchestrator's reading of the sweep among
  them, which still understated the losses (the ELBO error is worse after
  on 35 of the 59 seeds). Fix agent E on `../gpyreg-port-review-E` (from
  `d93f03c`): `5db2255` (the low-noise covariance from a Cholesky factor,
  through a helper that the posterior's computation shares, bit for bit),
  `2938581` and `dd12db3` (texts, release notes), with the orchestrator's
  `ec0f085` (texts); taken onto `port-review-wave6` as a fast-forward; its
  report `experiments/port_review_20260919/fixes/wave6_check_round_agent_E.md`.
  The orchestrator's `b2d7c6e8` (`load` gives the run's mask wherever the
  stored `integer_vars` reads otherwise) and `503562d7` (the scan test reads
  more forms and names its sites). Gate 4 against `ec0f085`: gpyreg's suite
  319 passed; PyVBMC's suite 1992 passed, 58 skipped; the exact oracle check
  11 of 11; the four seeded runs bit for bit gate 2's record `after_w6_1_f76eca2.npz` (92 arrays, 0 differ). The records: the ledger
  (the corrections of F3, the section on the check of the fix round), the
  sheet (the entries of W6-8 and W6-18, four corrections), the MATLAB-side
  list (entry 48), `TODO.md` (the findings left for later, E's two among
  them), the reader of the probe, which prints every number the ledger
  quotes, `dev/scripts/runs/LOCAL.md`.
- [x] **2026-09-22, the independent check of the wave-1 pass, and its fix
  round** (PI: `/doublecheck`, read-only, by a second session while wave 6
  was closed in the main checkout; then, on the PI's word, the fixes as the
  orchestrator proposed them). The pass of wave 1 was the one pass no fresh
  reviewer had read. Six fresh read-only Opus reviewers read it on
  `a3d4a70d`, and `verification/wave1.md` has what they read, what holds,
  what they found, the disposition of every finding of the wave (its two
  verifiers' ledgers have no column for it) and the gates. No later pass
  undid a wave-1 fix. In the code: `optimize_vp` could return a posterior
  of NaN without an error, the check of W5-7 dropped the stored value of a
  cached starting point after a warp, and `determine_best_vp` ranked NaN
  scores anywhere; in the records, the wave-1 ledgers' premise about
  MATLAB's rank-one update and their correction of the reason for the
  one-dimensional search were wrong, and the changelog lacked the
  "Upgrading" line for `noise_shaping`. The fix round runs on the branch
  `dev-port-review-w1check`, cut at `326c7676` in the worktree
  `../pyvbmc-w1check`, so that the main checkout stays wave 6's: two Opus
  fix agents on worktrees of their own (`fixes/wave1_check_agent_A.md` and
  `_B.md`), 19 commits cherry-picked after review, and a test, three
  option descriptions and the records by the orchestrator. One of them
  moves noisy trajectories: `N` and `n_eff` follow every evaluation of
  active sampling, as in MATLAB. On the PI's rulings of the same day on
  the items the round left, seven more commits by the orchestrator: the
  stable sort of the search set, the set of user options of options built
  from another run's, a CMA-ES start that is not finite, the warning of
  `get_parameters` on a zero weight, the check of `hpd_frac`, and two texts
  (`f_vals` are log-joint values; the design size of the PyMC setup
  probe). The raw reports of the six reviewers and the logs of the gates
  are on the orchestrator's machine (`dev/scripts/runs/LOCAL.md`). To do
  when the branch is brought onto `dev-port-review`: the whole suite, the
  Torch and PyMC environments, the refresh of the sheet's Python line
  citations against the merged code, the CI matrix, and the noisy half of
  the benchmark sweep for the refreshed counts; the worktrees and branches
  of the two fix agents are removed once `git cherry` shows their commits
  on `dev-port-review`.
- [x] 2026-09-23: a check of the last round for substantial errors alone
  (PI: `/doublecheck`, "only substantial errors"). Three fresh read-only
  Opus reviewers: P found none in PyVBMC's two commits; G found that the
  width check of `quad` (W6-34, `5ff9353`) refused a `sigma` of one column,
  which 1.2.1 and gplite take, and that the release notes misstated what
  1.2.1 did with a measure of the wrong width, fixed by the orchestrator in
  gpyreg's `286b595` with a test seen to fail (gpyreg's suite 320 passed;
  no PyVBMC code calls `quad`); R found that this pickup point left out the
  parallel branch `dev-port-review-w1check` and that row W6-16 missed the
  fixed coordinates, both corrected. `verification/wave6.md`, "The check
  for substantial errors of the last round".
- [x] 2026-09-23: gpyreg's pull request `#50` merged (PI's word; merged
  with admin rights at the PI's command, `main` asking for a review the
  author cannot give), as the merge commit `a4c2cc0`, so every commit the
  ledger cites is on gpyreg's `main`; gpyreg's build and docs workflows
  passed on it. `../gpyreg` brought to it. PyVBMC's `GPYREG_PIN` moved to
  it, with the sheet's header and its gpyreg line citations (`4196649a`).
- [x] 2026-09-23: gpyreg 1.3.0 released, and PyVBMC moved to it (PI: do
  it now). PyVBMC's full matrix against the pin at `a4c2cc0`, nine jobs,
  green, with the branch smoke. gpyreg's `#51` merged (`0186d89`), dating
  the release notes, the one text on `main` that said "unreleased"; the
  annotated tag `v1.3.0` on it and its GitHub release, whose workflow
  uploaded the wheel and the source archive to PyPI. `2dc2a6c3`:
  `GPYREG_PIN` at the tagged commit, `gpyreg >= 1.3.0` in `pyproject.toml`,
  the changelog's requirement, the sheet citing gpyreg at the tag. Locally,
  `../gpyreg` fetched with its tags and reinstalled editable (it reads
  1.3.0, and `pip check` finds no broken requirement).
- [x] 2026-09-23: wave 6 finished (PI: go). The full matrix on `647a886a`
  (the pin at `v1.3.0`, `gpyreg >= 1.3.0`), nine jobs, green, with its
  smoke; `dev-port-review` fast-forwarded into `dev-next`, with the status
  line of `TODO.md`. `AGENTS.md` states two traps this wave met: a raised
  gpyreg minimum puts the pin at the release's tagged commit, and a change
  to gpyreg is gated by PyVBMC's whole suite run against it.
- [ ] **Pickup point (2026-09-23, wave 6 finished): the wave-1 branch onto
  `dev-port-review`, on the PI's word.** The state: `dev-next` and
  `dev-port-review` are one commit, pushed; gpyreg 1.3.0 is on PyPI,
  required by `pyproject.toml` and pinned in CI at its tag; `../gpyreg`
  stands at `v1.3.0`, installed editable, so the oracles and the gates need
  no `PYTHONPATH`. gpyreg's worktrees `../gpyreg-port-review-A`, `-B`, `-D`
  and `-E` stand, every commit of theirs on gpyreg's `main`. The wave-1
  branch is `dev-port-review-w1check` (head `ab0a688b`, cut at `326c7676`),
  in `../pyvbmc-w1check`, with its agents' `../pyvbmc-w1check-A` (detached
  at `326c7676`) and `../pyvbmc-w1check-B` (branch `w1check-agent-B`) and
  the local branch `w1check-agent-A`, none pushed; its own worklog entries,
  on that branch, list its fixes, its rulings and its gates. In this order:
  1. The merge of `dev-port-review-w1check` into `dev-port-review`, in the
     main checkout. Seven files carry both sides' changes: `CHANGELOG.md`,
     the sheet, `matlab_side_defects.md`, this plan,
     `pyvbmc/vbmc/vbmc.py`, `pyvbmc/testing/vbmc/test_options.py` and
     `test_vbmc_option_names.py`. Both sides are kept: wave 6's back-fill
     of the `integer_vars` of release 1.0.4 in `load` and the scan test of
     `_CONSTRUCTION_ONLY_OPTIONS`, wave 1's changes, and both worklogs in
     the order of their dates.
  2. What the merge can break without a conflict, to be read after it: the
     scan test (`test_construction_only_options_are_the_options_only_
     construction_reads`) holds `_CONSTRUCTION_ONLY_OPTIONS` to the
     package's reads of the options, so where wave 1 adds or moves a read
     the tuple and its comment are corrected (the test's message names the
     sites), not the test, and the scan of `INERT_OPTIONS` likewise; the
     sheet's PyVBMC citations move with wave 1's code
     (`refresh_citations.py`, then by hand the ones it reports); a wave-1
     commit moves the noisy seeded runs, so gate 4's record
     `after_second_round_ec0f085.npz` no longer holds for them and a new
     record is made (`wave2_fixpass_gate_runs.py --out`), the wave-1
     entries saying which runs its commits move; and the exact oracle check
     on the merged head, where an oracle that a wave-1 change moves is
     re-baselined with the generator's targeted modes and a reason, never by
     rerunning the recipes.
  3. The gates of both on the merged head: PyVBMC's whole suite, the Torch
     and PyMC environments, the exact oracle check, the seeded runs, and
     wave 1's own gates from its entries (the noisy half of the benchmark
     sweep among them); then the push, the smoke, the full matrix (the
     `tests` workflow dispatched on `dev-port-review`), and the
     fast-forward into `dev-next` with the status line of `TODO.md`.
  4. The worktrees removed: gpyreg's A, B, D and E once `git cherry` has
     been read once more; wave 1's once its branch is merged.
  The session starts no other wave.
- [ ] After wave 6, on the PI's decision and not before: O1 to O4, the third
  readers, with O4 after the fixes of G1. With the internal track of P2,
  which wave 1 had left out and wave 5 took in, every P slice has both
  reports. Before comparison reviewers receive `known_differences.md`, its
  Python line citations are carried to the present lines again
  (`experiments/port_review_20260919/refresh_citations.py`, last run on
  2026-09-21): every fix pass moves them. The stored oracle state at
  uncertainty level 1 and the seeded gate run with a prior are items of
  `TODO.md` (PI, 2026-09-21).
- [x] A candidate from outside the slices, to be verified with the
  accumulated findings. `load(new_options=)` validates the names it is
  given, updates the options and checks single values
  (`_validate_option_values`); it runs neither `Options.update_defaults`
  nor `_init_optim_state`. An option that is read only at construction is
  therefore accepted and takes no coherent effect:
  `new_options={"uncertainty_handling": True}` leaves the noise model of
  the GP, which `_init_optim_state` fixes, and the noiseless values of the
  five noisy-target defaults as they were. Rows W2-7 and W2-8 of the wave-2
  ledger cover those defaults at construction only. Found on 2026-09-21
  while the statements of `AGENTS.md` were checked against the code, by
  reading `load`; not reproduced by a run. A disposition to consider: `load`
  refuses the options that are read only at construction. Verified in wave 6
  and fixed as W6-37 (`e657eba8`, `verification/wave6.md`): `load` refuses
  such an option, and the checks of `gp_mean_fun` and `integer_vars` run on
  both paths.
- [ ] Verification of the accumulated findings; ledger written.
- [ ] PI triage.
- [ ] Fixes on `dev-port-review` with gates; durable sheet entries
  consolidated into the porting log; the list of MATLAB-side defects
  (`experiments/port_review_20260919/matlab_side_defects.md`) brought up
  to date.
