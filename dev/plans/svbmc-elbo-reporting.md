# S-VBMC ELBO corrections and reporting

Created 2026-09-12. Status: implemented, verified and merged into `dev-next`
at `954677a` (implementation commit `483a8da`).

This is the execution record for Phase 1 of the
[ELBO optimism note](../2026-09-12-svbmc-elbo-optimism.md). It owns the
implementation contract, checks and worklog. The note retains the scientific
rationale and Phase 2 proposal. The PI accepted all five defaults on
2026-09-12 and authorized execution after the real-data reference campaign.

## Accepted decisions

1. `elbo` is a scalar headline, paired with `elbo_sd`. `elbo_details` holds
   raw, component-median capped, run-median capped and naive estimates, the
   headline method, cap amount and separate uncertainty contributions.
   Keep the run-median diagnostic and compute naive stacking by default,
   with equal weight per retained run and its original internal weights.
   Document migration from the dictionary API. Prefer names saying
   "capped": the correction does not eliminate all bias. The naive value
   is a diagnostic, not a guaranteed bound on the optimized ELBO.
2. Compute component Jacobian expectations once during construction:
   analytically for unbounded and probit coordinates, including whitening;
   one-dimensional Gauss-Hermite quadrature for logit and Student-t.
   Start with 32 nodes, compare 64, increase if necessary. Target an
   absolute difference of 1e-8 nats per component's total correction,
   subject to validation on broad and warped components. Construction
   consumes no correction draws; unconverged quadrature must be reported.
   The Gauss-Hermite ladder ends at 2048; remaining broad components use
   adaptive one-dimensional integration with an error check. Raise if
   neither deterministic method reaches the tolerance.
3. `show_tips=True`; guidance also requires INFO logging to be enabled.
   Suppressing tips leaves progress and applied-cap diagnostics available.
   Use the shared hint emitter and a small S-VBMC data-only catalog. Use
   the first eligible optimize call, then every third eligible call; one
   shuffle per session, each tip at most once. The noisy tip has priority
   when first eligible. Keep scheduling randomness outside numerical RNGs.
4. Use the component-median capped headline when any retained run is noisy;
   use raw otherwise. Record `uncertainty_handling_level` in VP stats for
   new runs. For older VPs infer noise from `elbo_sd > 0.1`, expose the
   inferred provenance, and permit an explicit noise-status override.
   Recorded noiseless status takes precedence over a high proxy SD.
5. Re-evaluate at returned weights with fresh draws and
   `n_samples_final=100` per component. `elbo_sd` combines the stratified
   entropy estimator's Monte Carlo variance with GP quadrature uncertainty.
   Preserve within-run component covariance and the between-hyperparameter
   variance from `I_sk`, following VBMC's total-variance calculation.
   Independent run blocks contribute at their global stacked weights.
   The SD describes the uncapped evaluation at selected weights; it does
   not quantify selection bias or propagate the median cap. Do not call
   the capped headline and this SD a calibrated confidence interval.

Implementation details: the keyword-only constructor override is `noisy`
(`None` infers per retained run; a boolean overrides the whole stack).
`noise_status_source` records `recorded`, `inferred` or `override` per run.
The detail keys are `raw`, `capped_I_median`, `capped_E_median`, `naive`,
`headline_method`, `cap_amount`, `entropy_sd`, `gp_sd`, `raw_sd`, `noisy`
and `noise_status_source`. `cap_amount` is the reduction applied to the
headline, zero for noiseless stacks even when a diagnostic cap would bind.
In naive mode, reuse the fresh final evaluation for `raw` and `naive`.

At global component weights `a_m`, each run contributes
`mean_s(a_m.T @ J_ms @ a_m) + var_s(a_m @ I_ms, ddof=1)`.
Use VBMC's diagonal/result floors and zero between-sample variance for
`Ns=1`. Sum these independent run contributions and add the entropy
variance `sum_k(w_k**2 * sample_var_k / n_samples_final)`.

Alternatives rejected: a primary dictionary (ambiguous reporting despite
compatibility); fixed Monte Carlo corrections (persistent sampling noise);
logger-only or switch-only tips (coupled or inconsistent quiet controls);
unconditional capping (unsupported noiseless downward shifts); entropy-only
SD (omits surrogate uncertainty), or adding the cap amount as a variance
(no statistical justification).

## Scope and sequencing

The optimization method and objective's mathematical definition stay fixed.
Deterministic corrections intentionally move S-VBMC weights and references.
Torch remains lazy and optional, numerical state remains float64, and input
posteriors and generators remain untouched. The VBMC core change only adds
noise metadata and must preserve every numerical output and random draw.
Phase 2 surrogate experiments, entropy speedups, the standalone forwarding
release and golden population completion are separate work.

Keep planning/status edits on `dev-next`; implement on a feature branch.
At most one heavy computation runs at a time, in the main thread.

## Execution and acceptance

### 1. Plan and decisions — Astra orchestrator

- [x] Record approval in the note, TODO and developer indexes; independently
  review this plan before implementation. No approval remains outstanding.

### 2. Deterministic corrections — Sol implementation subagent

- [x] Add private `pyvbmc/svbmc/_jacobian.py` with a function taking a VP and
  returning a float64 vector of expected log-Jacobians, one per component.
  Read `parameter_transformer.py` and use the Gaussian marginal after
  undoing scale, rotation and centering. Match constants and Jacobian sign.
  Support all shipped transform types and mixed bounded/unbounded inputs.
- [x] Add `pyvbmc/testing/svbmc/test_jacobian.py`: analytic probit and affine
  expectations, mixed and warped cases, comparison with independent
  integration/seeded Monte Carlo, broad components, RNG preservation, and
  quadrature nonconvergence handling. Write checks but leave execution to
  the main thread. Do not edit the tracker or other implementation files.

### 3. Reporting and guidance — Astra orchestrator

- [x] In `svbmc.py`, cache corrections at construction; make `stacked_ELBO`
  free of first-call mutation. Preserve the public entropy return shape.
  Add a private entropy path returning stratified sampling variance for
  final evaluation; reject insufficient final sample counts before drawing.
- [x] Add scalar headline, detailed diagnostics, fresh final evaluation,
  original naive weights, noise inference/override and uncertainty using
  both `I_sk` and `J_sjk`. Validate required sample axes. Check repeated
  optimize calls and all three optimization modes, including naive mode.
- [x] Record noise level in `optimize_vp`; add the S-VBMC tip catalog and
  scheduler using `emit_user_hint`, with quiet behavior and cadence tests.
- [x] Extend existing S-VBMC tests for API, covariance, fresh estimates,
  proxy provenance, input/RNG isolation and cap diagnostics. No new full
  VBMC optimize runs; use existing fixtures and tests for the metadata.
  Sol owns `test_elbo_reporting.py`; the main thread integrates and runs it.

### 4. Documentation and numerical gates — Astra orchestrator

- [x] Update class/method docstrings, `docsrc/source/api/classes/svbmc.rst`
  and Example 7; regenerate its script with the repository's notebook
  conversion workflow. State the migration, SD interpretation and noise
  fallback. Check all live dictionary-ELBO callers.
  Sol updates the API page and notebook sources; the main thread executes
  the notebook and generates the script.
- [x] Update `make_svbmc_fixtures.py` and reference readers for final sample
  count and reporting fields. Deliberately regenerate only S-VBMC reference
  outputs, record the reason/recipe in `FIXTURES.md`, and assess changes
  against the former Monte Carlo variability. Investigate material shifts
  instead of merely accepting new fixture numbers.
- [x] Run focused S-VBMC tests with Torch, relevant VBMC tests, exact core
  oracles and default golden replay (all identical). Use BLAS single-threaded
  environment on this machine. Run the full applicable suite and docs build
  serially. Inspect rendered Example 7 output and API documentation.
- [x] Independent Sol static review of the final diff, tests and this
  explicit plan path; main thread resolves findings and records outcomes.

## Risks and rollback

The dictionary-to-scalar migration is intentional. The SD is model-based,
and inference of noise for old posteriors is fallible. Quadrature convergence
must be tested rather than assumed. Independent run uncertainty is an
approximation when runs share evaluations or systematic surrogate errors.
Revert the feature changes and their regenerated S-VBMC references together
to roll back; never rebaseline the VBMC core or golden traces for this work.

## Worklog

- 2026-09-12: all five recommendations accepted; the reference campaign
  finished and the working tree was clean before preparation began.
- Plan review (Sol, static): no scientific must-fix. Clarified the precise
  total-variance calculation, naive-mode evaluation reuse and public
  diagnostic/override semantics before implementation.
- Initial validation: 97 existing reporting/filter tests passed. A broad
  logit component (effective Gaussian SD about 13.4) did not converge at
  2048 Hermite nodes. Added a deterministic adaptive-integration fallback
  while preserving the broad test and the 1e-8 accuracy target.
- Expanded focused suite: 157 passed, including the fallback, statistical
  reporting, tips and noise metadata through pruning. Exact core oracles:
  11/11 green. All five replay configurations have identical stored loop
  and final state against the current-treatment certification traces under
  `dev/scripts/runs/population_realdata_20260912/certification/replay_default5`.
  This baseline was itself certified against `68a43db`; the combined
  reference's older configurations predate the latent fixes. Historical
  traces omit the returned transformer, so that state is not certified by
  replay. Logs are under `dev/scripts/runs/svbmc_elbo_reporting_20260912/`;
  the replay traces/report are `dev/scripts/runs/golden/replay_20260912_165331`.
- S-VBMC references regenerated for all 18 group/mode cells; all 39 input
  posterior snapshots are unchanged. Plausibility comparison with the old
  code at `f93cf01`, seeds 0-4, same three-step recipe: unbounded weights
  differ only at rounding; bounded all-weights L1 change is 0.01005 against
  old seed-to-seed variation up to 0.12384, and bounded posterior-only
  changes by 1.27e-5 against 0.09167. Largest raw-ELBO shift is 0.02269 nats.
  Five fresh estimates lie just outside the five-seed range of the old
  best-iterate estimates (by at most 0.00457 nats); those are different
  estimators, and such finite-sample ranges are descriptive, not acceptance
  intervals. Tiny mode-order changes are assessed at the sampling-error
  scale rather than treated as deterministic rankings. The comparison
  script, full numbers and logs are in the ignored validation directory.
- Independent Sol implementation review: no scientific/numerical must-fix.
  Fixed the low-severity finding that invalid optimization arguments could
  consume a tip: shared validation precedes scheduling, with regression
  checks for the invalid version, step count and entropy sample count.
- Example 7 executed with the repository interpreter and CPU Torch overlay;
  headline 0.024, raw-evaluation SD 0.013, noiseless status recorded for all
  four inputs. Both posterior plots inspected: the stacked mixture covers
  both modes and the panels/legend are visible. The generated script was
  refreshed using the Makefile's nbconvert/isort/black recipe.
- Full package suite with CPU Torch and single-threaded BLAS: 1469 passed,
  35 skipped in 569.31 seconds. Three oracle density-gradient warnings arose
  from division by zero-density values; all oracle checks passed.
- Sphinx HTML build with copied example notebooks succeeded. The API's
  reporting table, migration table, uncertainty explanation and Example 7
  link render correctly. The sole warning concerns the existing Plotly MIME
  output in Example 2.
- Repository pre-commit hooks cover the changed Python files, notebook and
  documentation; the fixture writer uses the hook's Black formatting.
- Integration (2026-09-12): merged at `954677a` with no conflicts. The merge
  tree equals the verified feature tree at `483a8da`; subsequent integration
  edits update status documentation only.
