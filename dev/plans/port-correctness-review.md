# Plan: independent correctness review of PyVBMC and its MATLAB port

Started 2026-09-19. Owner of the item: `TODO.md`, "Independent codebase and
MATLAB-port review". This file holds the design, the reviewer brief, the
working rules and the worklog. The consolidated findings go into a dated
ledger under `results/` when verification completes; the raw reviewer
reports, the known-differences sheet and the verification scripts are kept
under `experiments/port_review_20260919/`.

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
- A later MATLAB session on another machine runs a short, selected list of
  fast checks. It is described in a self-contained plan file with scripts on
  the fix branch, written so that another developer can run it from that
  description alone.
- Confirmed findings are brought to the PI for triage before any fix is
  made. PyVBMC 1.5 takes only bug fixes and unequivocal improvements.

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
| **P2** Initial design and active-sampling search | `vbmc/active_sample.py` (including `_BatchedNoiseHandler`) | `private/activesample_vbmc.m`, `misc/initdesign_vbmc.m`; `utils/cmaes_modded.m` (PyVBMC uses the `cma` package, subclasses its noise handler and binds its draws to the run's generator), `misc/proposal_vbmc.m` (unported; unused in MATLAB too), `private/acqhedge_vbmc.m`, `utils/fastkmeans.m`, `utils/covcma.m` (unported), `misc/check_quadcoefficients_vbmc.m` (unported, integrated mean function) |
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

The reviewer writes `REPORT.md` in its scratchpad directory and returns the
same text as its final message. The report has three parts: **coverage**
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
- **needs MATLAB** (only a MATLAB run can settle it; collected for the
  session below);
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

## MATLAB check session

Findings classified **needs MATLAB**, and a short list of high-value
same-input comparisons chosen from the slices (the expected log joint, both
entropies, the VIQR and IMIQR acquisition values, the transformer's
log-abs-det Jacobian, the GP marginal likelihood and its gradient), are
settled by running both implementations on identical inputs. The plan for
that session lives on the fix branch as
`dev/plans/port-review-matlab-checks.md` with scripts under
`dev/scripts/matlab_checks/`: a Python side that dumps states to `.mat`
files with `scipy.io.savemat`, a MATLAB side that loads them and calls the
counterpart functions, and a comparison step with stated tolerances. The
`.m` scripts beside the fixtures in
`pyvbmc/testing/vbmc/compare_MATLAB/` are the existing instance of this
pattern. The plan is written so that a developer with MATLAB and the
`../vbmc` checkout at `396d649` can run it from the file alone.

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
  runs the module's whole test directory there.
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
  Reviewer reports are copied from the scratchpad into
  `experiments/port_review_20260919/reviews/<slice>_<track>.md` by the
  orchestrator.
- All of the review's work happens on the branch `dev-port-review`, cut
  from `dev-next` at `f91fdf0`: this plan's worklog, the files under
  `experiments/port_review_20260919/`, the ledger, the fixes and the
  MATLAB check session plan. It merges into `dev-next` when the fixes have
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
- [ ] Chunk-independent Monte Carlo entropy (PI decision, 2026-09-19):
  every chunk budget reproduces the default budget's output bit for bit
  through reductions over the default layout's blocks, the default path
  untouched (gate: `--check --exact` with no oracle moved), and the
  calibration campaign validates entropy budgets exactly.
- [ ] Wave 1 (PI decision, 2026-09-19): M, P6 both tracks, P2 comparison.
- [ ] Waves 1 to 7: P1a to P9, G1, G2, both tracks; M.
- [ ] Wave 8: O1 to O4.
- [ ] Verification of the accumulated findings; ledger written.
- [ ] PI triage.
- [ ] Fixes on `dev-port-review` with gates; MATLAB check session plan and
  scripts written on that branch; durable sheet entries consolidated into
  the porting log.
