# S-VBMC ELBO corrections and reporting

Created 2026-09-12. Status: approved; implementation pending.

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

- [~] Record approval in the note, TODO and developer indexes; independently
  review this plan before implementation. No approval remains outstanding.

### 2. Deterministic corrections — Sol implementation subagent

- [ ] Add private `pyvbmc/svbmc/_jacobian.py` with a function taking a VP and
  returning a float64 vector of expected log-Jacobians, one per component.
  Read `parameter_transformer.py` and use the Gaussian marginal after
  undoing scale, rotation and centering. Match constants and Jacobian sign.
  Support all shipped transform types and mixed bounded/unbounded inputs.
- [ ] Add `pyvbmc/testing/svbmc/test_jacobian.py`: analytic probit and affine
  expectations, mixed and warped cases, comparison with independent
  integration/seeded Monte Carlo, broad components, RNG preservation, and
  quadrature nonconvergence handling. Write checks but leave execution to
  the main thread. Do not edit the tracker or other implementation files.

### 3. Reporting and guidance — Astra orchestrator

- [ ] In `svbmc.py`, cache corrections at construction; make `stacked_ELBO`
  free of first-call mutation. Preserve the public entropy return shape.
  Add a private entropy path returning stratified sampling variance for
  final evaluation; reject insufficient final sample counts before drawing.
- [ ] Add scalar headline, detailed diagnostics, fresh final evaluation,
  original naive weights, noise inference/override and uncertainty using
  both `I_sk` and `J_sjk`. Validate required sample axes. Check repeated
  optimize calls and all three optimization modes, including naive mode.
- [ ] Record noise level in `optimize_vp`; add the S-VBMC tip catalog and
  scheduler using `emit_user_hint`, with quiet behavior and cadence tests.
- [ ] Extend existing S-VBMC tests for API, covariance, fresh estimates,
  proxy provenance, input/RNG isolation and cap diagnostics. No new full
  VBMC optimize runs; use existing fixtures and tests for the metadata.

### 4. Documentation and numerical gates — Astra orchestrator

- [ ] Update class/method docstrings, `docsrc/source/api/classes/svbmc.rst`
  and Example 7; regenerate its script with the repository's notebook
  conversion workflow. State the migration, SD interpretation and noise
  fallback. Check all live dictionary-ELBO callers.
- [ ] Update `make_svbmc_fixtures.py` and reference readers for final sample
  count and reporting fields. Deliberately regenerate only S-VBMC reference
  outputs, record the reason/recipe in `FIXTURES.md`, and assess changes
  against the former Monte Carlo variability. Investigate material shifts
  instead of merely accepting new fixture numbers.
- [ ] Run focused S-VBMC tests with Torch, relevant VBMC tests, exact core
  oracles and default golden replay (all identical). Use BLAS single-threaded
  environment on this machine. Run the full applicable suite and docs build
  serially. Inspect rendered Example 7 output and API documentation.
- [ ] Independent Sol static review of the final diff, tests and this
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
