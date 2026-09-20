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
| **P3** Acquisition functions | `acquisition_functions/*.py` (including `_real2int` and `_sq_dist` of the base class) | `acq/*.m`, `misc/real2int_vbmc.m`, `utils/sq_dist.m`; `misc/noiseshaping_vbmc.m` (unported, default off), `acq/acqeig_vbmc.m` and `misc/intkernel.m` (the EIG acquisition was removed from PyVBMC on 2026-09-14 and retained on `retain/experimental-acquisitions`) |
| **P4** Noisy importance sampling | `vbmc/active_importance_sampling.py` (including `fess`) | `private/activeimportancesampling_vbmc.m`, `misc/fess_vbmc.m`, `gplite/private/eissample_lite.m` (MATLAB's MCMC sampler, duplicated as `utils/eissample_lite.m`; PyVBMC uses gpyreg's `SliceSampler`, a substitution the comparison reviewer examines) |
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
- The finite-difference gradient checks: `pyvbmc/testing/**/test_*_grad_fd.py`
  and the entropy tests `pyvbmc/testing/entropy/test_entlb_vbmc.py` and
  `test_entmc_vbmc.py`, which call `check_grad` directly.
- The focused tests of the changed module, and gpyreg's suite for a gpyreg
  fix.

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
  run from a worktree imports the main checkout's package unless
  `PYTHONPATH` names the worktree (`python -m pytest` is unaffected), so
  every oracle command in an agent's brief carries `PYTHONPATH=<worktree>`
  and the agent prints `pyvbmc.__file__` once; `AGENTS.md` records the
  trap.
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
- The pre-commit hooks rewrite files (formatting, unused imports) and then
  abort the commit; the rewritten files are staged again and the commit is
  repeated.
- All of the review's work happens on the branch `dev-port-review`, cut
  from `dev-next` at `f91fdf0`: this plan's worklog, the files under
  `experiments/port_review_20260919/`, the ledger and the fixes. It
  merges into `dev-next` when the fixes have
  passed their gates. Only the `TODO.md` status line is updated on
  `dev-next` directly.

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
- [x] 2026-09-19: wave 1 verified. P6: 24 ledger rows
  (`verification/wave1_P6.md`); M and P2: 21 rows and seven minor
  observations (`verification/wave1_M_P2.md`); every statement
  reproduced or read in both sources, and the reports' own errors
  recorded there (the reasoning behind P2 F15, M's cause for the
  one-dimensional search override, three line citations). A pass over
  the options surface that grew out of M F2
  (`verification/wave1_options.md`) found 25 declared options that
  nothing reads, three selection options that `determine_best_vp` took
  as arguments both call sites left out, a ranking branch that had
  never executed, and two iteration counts off by one against
  `best_vbmc`.
- [x] 2026-09-19: wave 1 fixed, 34 commits on `dev-port-review` after
  `6c151cb` (three Opus agents on worktrees, one finding per commit with
  its test, cherry-picked after review). Active sampling (13 plus one):
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
  search reaches a different point on 7 of the 8 states, every other
  reference bit-identical); the oracle tests; both flaky tests made
  deterministic. Records: `AGENTS.md`, the Stage 1 and Stage 2 plans and
  the oracle harness comments describe the search without its noise
  handler (`80204f7`); the known-differences sheet brought to the state
  of the code. **Moves default trajectories**: the per-coordinate search
  scales and the dropped noise handler (every search with `D > 1`), the
  ranking criterion (runs ending without a stable iteration), the exact
  one-component entropy (reported values where `K = 1`), the type-2
  starting widths (`D > 1` with two slow candidates), the bounded
  one-dimensional search (`D = 1`) and the kept starting points (`x0`
  longer than `fun_eval_start`), on top of `fun_eval_start` above. The
  golden references are regenerated once after the review's remaining
  trajectory-moving fixes are decided.
- [x] 2026-09-19: rulings without a code change (PI). The soft-bound box
  stays a function of the call's training inputs where MATLAB
  accumulates it: `verification/scripts/soft_bounds_trace.py` on two
  short seeded runs (Rosenbrock `D = 2`, two Gaussians `D = 3`) found
  the accumulated box wider than the rebuilt one on about half the
  calls (up to 2.2 and 12 times its width) while the fitted posterior
  stayed within the rebuilt box up to a small overshoot (a component
  mean 0.07 box widths outside, a log scale 0.008 above its bound), so
  the rebuilt bounds bind marginally at most and the accumulation code
  was removed instead of completed. `search_cmaes_best`
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
  (`29c822d`): a second `optimize_vp` call continuing from the returned
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
- [ ] Waves 3 to 7: P3, P4, P5, P7, P8, P9, G1, G2, both tracks (16
  reviewers), and the internal track of P2, which wave 1 did not run
  (its four slots went to M, the P2 comparison and both P6 tracks);
  17 reviewers, so one wave has a free slot for it.
- [ ] Wave 8: O1 to O4.
- [ ] Verification of the accumulated findings; ledger written.
- [ ] PI triage.
- [ ] Fixes on `dev-port-review` with gates; durable sheet entries
  consolidated into the porting log; the list of MATLAB-side defects
  (`experiments/port_review_20260919/matlab_side_defects.md`) brought up
  to date.
