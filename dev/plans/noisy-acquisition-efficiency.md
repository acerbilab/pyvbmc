# Existing noisy-acquisition efficiency

Created 2026-09-13. Status: guarded-sinh implementation, validation and
independent review complete on `dev-noisy-viqr-sinh` (numerical revision
`6734817`), from `83692ac`. Evidence is archived below. The
[kernel-reuse implementation plan](#kernel-reuse-implementation-plan) is
approved and in progress. Search and quadrature experiments remain independent
possibilities in this workstream.

## Guarded-sinh execution checklist

- [x] Freeze before-change source and oracle outputs; implement guarded sum
  and focused tests.
- [x] Run numerical gates and classify changed outputs.
- [x] Capture matched states during the 18-run replay and assess results.
- [x] Measure complete acquisition calls against the frozen implementation.
- [x] Complete full tests, CI and independent review; record outcome.

## Goal and scope

Reduce the cost of evaluating existing noisy acquisitions while preserving
their criterion, importance samples, random draws and search settings.
The first implementation targets standard VIQR's sum over importance points.
Sieve size, importance-sample count, optimizer, frequent GP retraining and
default acquisition remain unchanged. New acquisitions and the planned
experimental-option removal are separate work.

This plan owns implementation decisions and validation for the efficiency
work. The [noisy-acquisition note](../2026-09-08-noisy-acquisitions.md) and
its linked reports retain the historical investigations. The
[TODO](../TODO.md) owns release scope and workstream status.

## Evidence and candidate changes

The historical measurements identify large-sieve acquisition evaluation and
frequent GP training as the main costs. Their proposed search changes need
separate validation: optimizing a small importance set more aggressively
can overfit its Monte Carlo error. They do not establish a new default.

Before this optimization, standard VIQR performed a log-sinh followed by
log-sum-exp over each candidate's importance points in
`pyvbmc/acquisition_functions/acq_fcn_viqr.py`. For finite nonnegative
`a = u * s_pred`, this computes `log(2 * sum(sinh(a)))`. Evaluating that
sum directly avoids two elementwise transcendental sweeps. An overflow
guard is required; the log-space calculation supplies the fallback for
large values.
The direct expression also retains extremely small positive contributions
that the old `1 - exp(-2a)` calculation can round to zero.

The bounded [probe](../scripts/probe_viqr_sum.py) uses the committed
`rosenbrock_D2_noise1_viqr` oracle state (D=2, N=25, eight GP samples,
8192 candidates, 100 importance points). In the initial probe, direct
evaluation took 91 ms versus 338 ms for the existing sum, with maximum
absolute difference 1.78e-15. A scaled exp/expm1 identity took 266 ms.
These are timings of the sum alone across the eight GP samples, on one
early state; they do not estimate complete-run speedup. Exact timings,
versions, hashes and limitations are in
[the probe output](../experiments/noisy-acquisition-efficiency/viqr_sum_probe.json).

Additional opportunities remain for subsequent steps:

- Optional IMIQR cleanup: it recomputes triangular solves in every acquisition call although
  `active_importance_sampling.py` already prepares `C_tmp`. Reusing it
  requires checking both GP factor representations and the placement of
  the noise scaling, which can change rounding. Keep IMIQR's weighted
  criterion and MCMC stream intact.
- The base acquisition calls `gp.predict`, then VIQR/IMIQR recompute its
  training-to-candidate kernel. The installed gpyreg predictor does not
  expose this intermediate. The
  [kernel-reuse probe](../results/2026-09-13-viqr-kernel-reuse.md) finds exact
  captured-state outputs and a 1.205x complete-sieve speedup on the late
  noisy state, with smaller early gains. A supported gpyreg interface is
  feasible within the lab's repositories. Production integration remains
  open, including a policy for the additional temporary matrix memory.
- Adaptive sieve/search remains a main efficiency experiment. Compare a
  smaller coarse sieve, more accurate re-scoring, and L-BFGS-B from one or
  several spatially separated candidates. Investigate sieve sizing and
  uncertainty-based integration budgets/refinement gates; the initial
  experimental sieve size is about 1024, not an established new default.
  Judge candidates on independent integration points to detect Monte Carlo
  overfitting, then validate inference end to end. The
  [search analysis](../results/2026-09-09-acquisition-search-analysis.md)
  records the mixed evidence on additional starts. Kernel reuse and
  quadrature are independent opportunities, not prerequisites for this
  experiment.
- More efficient integration of the existing VIQR criterion, including
  shared-weight Bayesian quadrature (PI, 2026-09-13). For each fixed VP and
  quadrature kernel, prepare nodes and weights once and reuse them across
  candidate locations. Gaussian-kernel means under the Gaussian-mixture VP
  are analytic, but the nonlinear VIQR integrand itself remains numerical.
  Compare with ordinary VP sampling and mixture-stratified randomized
  quasi-Monte Carlo on matched saved states, measuring total preparation
  and evaluation costs and candidate rankings under a larger independent
  integration set. Investigate quadrature-kernel choice, negative weights,
  positivity and difficult integrands before choosing an implementation.
  This is a possible integration-efficiency experiment within this work,
  preserving the VIQR criterion; it is not part of the guarded-sinh change.
  See the [VIQR definition](../../papers/acerbi2020variational_main.md#33-integrated-median--variational-interquantile-range-imiqr--viqr)
  and [Gaussian integration identities](../../papers/acerbi2019exploration_appendix.md#appendix-a-expected-log-joint-via-bayesian-quadrature).

## Phase 1: implement the VIQR sum optimization

Executor: Sol implementation agent, with Astra orchestrating design and
integration. Only the main thread runs heavy checks or benchmarks; reviewers
use static analysis.

1. Start a feature branch from the settled `dev-next` source. Read
   `dev/2026-09-02-modernization-discussion.md` and the acquisition note.
   Freeze the before-change revision and dump current oracle outputs using
   `python dev/scripts/make_oracle_fixtures.py --dump-outputs <before-dir>`.
   Set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` to 1
   for numerical checks on the fixture-generating machine.
2. Modify only standard VIQR's sum, preserving the hyperparameter-sample
   aggregation and normalization. Use a direct-sinh path on safe finite
   nonnegative arguments, with the existing log-space calculation as the
   fallback. Derive the overflow bound from float64 and the number of
   importance points, with margin for the sum. Keep the fallback's handling
   of unsupported/nonfinite arguments and old serialized VIQR instances.
   Do not apply the unweighted identity to IMIQR or reduction variants.
   Leave `is_log_added` and the importance proposal helpers unchanged:
   their values feed IMIQR's MCMC target and can alter its sampling stream.
3. Add focused tests in `pyvbmc/testing/acquisition_functions/` for small
   and large arguments, zero rows, mixed zero/positive entries, overflow
   guard boundaries, alternative quantiles, and one/multiple GP samples.
   Compare with independent analytic or high-precision values where the
   old expression loses precision; use the existing full-covariance
   reference in `_look_ahead.py` for acquisition-level checks.
4. Check public-call scalar/batch agreement, dtype, variance penalties and
   hard bounds; require unchanged RNG state during evaluation and unchanged
   importance samples/weights. Test reductions and IMIQR as unaffected
   paths. Avoid adding full `VBMC.optimize()` runs to the unit suite.

Acceptance: same mathematical criterion and all existing public behavior,
apart from documented floating-point rounding; no sampling or search-policy
change. If the fast path needs broader API/state changes, stop and revise
the design rather than expanding this implementation silently.

## Phase 2: numerical and performance checks

Executor: Astra main thread runs checks; Sol independently reviews source
and evidence. Model roles follow the repository's Astra/Fable and Sol/Opus
equivalences.

- Run `python -m pytest pyvbmc/testing/acquisition_functions
  pyvbmc/testing/vbmc/test_active_importance_sampling.py -q` and the oracle
  suite. Run the generator with `--check --exact --against <before-dir>`
  to classify every changed output, alongside the normal tolerance gate.
  Expect possible rounding in VIQR and movement in its stochastic search;
  investigate any other changes. Do not regenerate fixtures to make the
  optimization pass. Any sanctioned targeted rebaseline comes only after
  the acquisition value gates and assessment.
- Extend the probe to time complete public acquisition calls at candidate
  counts 1, a CMA-ES population, and 8192. Include setup costs separately.
  Use early and late noisy Rosenbrock, noisy Student D8 and noisy
  multisensory states where available, covering one/multiple GP samples.
  Inventory reconstructible saved states first: golden trace sidecars alone
  do not supply a GP/VP snapshot. Capture missing states during the approved
  replay allocation instead of launching additional exploratory runs.
  Add a developer-only capture wrapper around the VIQR sieve call in the
  replay runner, before acquisition evaluation. For seed 0 of the selected
  targets, capture the first sieve and the first sieve with either one GP
  sample or N at least 75% of the evaluation budget. Record an unreached
  late checkpoint as a coverage gap. Use `_state.snapshot_from_objects`
  and `save_snapshot`, adding the exact candidates through `_state.encode`;
  retain the current importance samples, weights and caches in `optim_state`.
  Snapshot copies must consume no random draws or mutate live inputs;
  assert RNG-state identity around capture. Serialize outside the measured
  acquisition interval. Record the live acquisition outputs for validation.
  Rebuild the same snapshots under frozen before/after checkouts using
  `build_state` and `prepare_gp_for_acq`, without redrawing importance points.
  Verify the after-code reconstruction against the captured candidate-run
  outputs. Then compare before and after on that same reconstructed state,
  classifying the expected rounding differences. Resolve any missing
  factor/state information before accepting timings.
- Compare before and after on identical states, candidates and importance
  samples in alternating timing order, single BLAS thread and one heavy
  process. Require a repeatable public-call gain beyond timing variation
  and no material small-batch slowdown. Record ranges, allocation overhead
  and fallback frequency; do not extrapolate the D2 sum-only ratio to runs.
- Approved bounded replay allocation: seeds 0-2 for
  `rosenbrock_D2_noise1`, `rosenbrock_D2_noise3`, `logreg_D5_noise3`,
  `student_D8_noise3`, `multisensory_s1_D6_noise1.3`, and
  `timing_D5_noise2.2` (18 candidate runs). Use
  `dev/scripts/golden_replay.py --configs <comma-separated-labels>
  --seeds 0-2 --out <candidate-dir>` against the promoted
  `reference_990_20260913`. Assess changed trajectories, initial designs,
  posterior/evidence errors, evaluations and completion/usable outcomes.
  These runs are a regression screen, not a powered equivalence study.
  Investigate flags before deciding whether a targeted extension is needed;
  a fresh 990-run campaign is not the default validation allocation.
  Acceptance requires no unresolved replay flags, identical initial designs,
  finite final metrics and population-envelope compliance for changed runs.
- Once the candidate is settled, run the required full suite and CI matrix
  for numerical changes, formatting hooks, and independent doublecheck.
  Existing occasional inference failures are not a new repair objective.

## Decisions and alternatives

- Start with the local VIQR sum: it has measured promise and a small
  implementation boundary. Kernel reuse could yield additional gains, but
  requires a gpyreg interface or duplicated prediction logic; it is a later
  design decision. Production monkey-patching is not an acceptable interface.
- Prefer a guarded direct sum with the current fallback. The scaled
  exp/expm1 alternative avoids overflow for finite nonnegative arguments,
  but was substantially slower in the probe. An unconditional direct sum
  would overflow on sufficiently uncertain GP states.
- Leave sample budgets and refinement unchanged in this step. A smaller
  sieve with better refinement is promising, but the historical experiments
  expose Monte Carlo overfitting and do not yet establish end-to-end gains.
- Treat IMIQR cache reuse as a separate subsequent change so its numerical
  and timing effects can be assessed independently.

## Documentation, review and rollback

Update this plan with accepted decisions and validation, link it from
`dev/TODO.md` and `dev/README.md`, and update the existing acquisition note
only when measurements warrant revised conclusions. No public option or
tutorial change is expected for the computational optimization.

Revert the isolated implementation commit to restore the previous sum;
preserve the before-change source and any accepted reference changes
together. The active 990-run baseline stays intact during investigation.

Independent Sol doublecheck found no must-fix mathematical or scope defect.
The plan incorporates its requested snapshot-capture procedure and explicit
replay acceptance criteria. The bounded probe, provenance hashes, relative
links and repository formatting hooks were checked during exploration.
Implementation and the 18-run allocation were subsequently approved by the
user. Execution status is tracked above.

## Validation record

The [validation artifact](../experiments/noisy-acquisition-efficiency/viqr_sinh_validation.json)
contains replay results, matched timing repeats, source hashes, coverage and
verification records. The [replay table](../experiments/noisy-acquisition-efficiency/viqr_sinh_replay.md)
shows each of the 18 comparisons.

- Before-change checkout: `dev/scripts/runs/viqr_sinh_20260913/before`,
  detached at `83692ac`; oracle dump in the adjacent `oracles_before/`.
- Acquisition and importance-sampling tests: 100 passed. Exact comparison
  against the dump changed only `acq_AcqFcnVIQR` on the noisy Rosenbrock
  fixture, by at most 8.88e-16. All other outputs, including the seeded
  active-sampling step and GP fits, matched exactly (10/11 fixture groups
  completely exact). The normal oracle suite passed: 143 passed, 15 skipped.
- The developer capture preflight rebuilt an acquisition output exactly
  after the live VP had been deliberately mutated, confirming snapshot
  isolation. The runner retains live GP factors and temporary data as well
  as the normal snapshot fields. Captured sets have per-file hashes;
  timing uses only checkpoints declared by the current status record.
- Independent Sol code review found no issues in the guarded calculation,
  its integration or focused tests. A complete-call preflight on the early
  noisy Rosenbrock state measured a 1.34x sieve speedup; single-point and
  six-point calls were 1-3% slower in that measurement. The broader timings
  below informed the final guard refinement. Full-suite execution initially
  encountered an inaccessible Windows temporary directory; a rerun used a dedicated
  workspace test directory and passed.
- Full suite passed on candidate `0b3c7b8`: 1254 passed, 39 skipped in
  399.50 seconds. The approved replay ran via
  `python -u dev/scripts/validate_viqr_sinh.py replay --out
  dev/scripts/runs/viqr_sinh_20260913/replay`; raw logs, traces and snapshots
  remain under the ignored run directory. The numerical source and runner
  were frozen at `0b3c7b8` throughout replay. Uncommitted status documentation
  accounts for the dirty-tree flag. The only intervening commit, `b2d7d72`,
  corrects a test mock and changes no numerical or replay source.
- Full-matrix run `34749673691` started after dispatch retries. Its three
  Python 3.10 cells exposed a pre-existing ambiguous mock target in
  `test_ns_gp_max_active_caps_the_in_loop_refits` (introduced in `5b857dde`):
  Python 3.10 resolves the dotted path through the exported `active_sample`
  function. Explicitly importing the module and using `patch.object`
  corrects the test (`b2d7d72`), with independent static review and formatting
  passed. Replacement full matrix `34750419101` passed all nine OS/Python
  cells; smoke `34750413117` also passed. The final full-suite run includes
  this corrected test.
- Static review of the capture runner led to v2 manifests that bind the
  runner, allocation, production/dependency sources, numerical-library
  versions and benchmark data. These were introduced after the campaign;
  its original v1 manifest remains intact.
- The 18-run replay completed in 49.0 minutes: 17 exact stored-loop/final
  matches and no flags. Every initial design matched, and all 18 runs
  terminated successfully with finite, consistent final outputs. The sole
  changed trajectory was noisy Rosenbrock (noise 3), seed 1: both versions
  used 180 evaluations; its evidence error, gsKL and MMTV remained inside
  the promoted population envelopes. This is a bounded regression screen.
  The historical archives omit the returned transformer's state, so exact
  identity applies to the recorded arrays and semantic final fields.
- Capture coverage is six early states and one late state (Rosenbrock,
  noise 3, N=150, six GP samples). The other five seed-0 runs converged
  before the late checkpoint. The existing `normal_D2_singlesample` oracle
  supplies a separate single-GP-sample timing check; it is not a noisy-run
  checkpoint. No extra target evaluations were used for timing.
- The first guard incurred single-point allocation overhead. Revision
  `6734817` uses scalar extrema for an entirely safe array and caches the
  logarithmic constants, retaining the same arithmetic and row fallback.
  All 11 oracle fixtures matched the first guard exactly, as did 162
  boundary/mixed/random helper arrays and every captured public output.
  Independent review confirmed this preserves the completed campaign's
  numerical behavior; a second campaign is unnecessary.
- Complete 8192-candidate calls on the seven noisy states improved by
  1.113-1.371x. The supplementary single-GP-sample sieve improved by 1.260x.
  Short timing blocks showed substantial variation with laptop load, so
  small calls were rechecked with nine rounds of 101 individually
  interleaved before/after pairs. Singleton speedups were 0.991-1.009x;
  CMA-population speedups were 1.035-1.060x. No material small-call slowdown
  remains. Peak traced allocations and all raw timing repeats are recorded;
  no fallback rows occurred in these measured states. These are acquisition
  timings, not estimates of whole-run speedup.
- The final full suite passed on `6734817`: 1254 passed, 39 skipped,
  three existing warnings in 567.46 seconds. The
  [full CI matrix](https://github.com/acerbilab/pyvbmc/actions/runs/34752259774)
  passed all nine cells, and the
  [smoke run](https://github.com/acerbilab/pyvbmc/actions/runs/34752255560)
  passed. The runner's v2 manifests bind source,
  allocation, environment and benchmark-data hashes. Existing v1 manifests
  remain intact; explicit `--equivalent-viqr` timing records both the
  capture and successor hashes and requires exact captured outputs.
- The completed replay's report-only check exited 0 with no flags. Metadata
  checks accepted an unchanged v2 resume and rejected eight changes covering
  source revision, VIQR/runner/production code, seed allocation, numerical
  dependency versions, GP sampler source and real-data archives. Default
  timing rejected the successor source until equivalence mode was explicit.
- Final independent Sol review of the source, recorded evidence and
  documentation found no remaining issues. The review checked the revision
  equivalence, timing claims, replay outcomes and stated coverage limits.

## Kernel-reuse implementation plan

Created 2026-09-13. **Status: APPROVED; IN PROGRESS.** This section owns the
cross-repository API, integration, validation and rollout decisions for
kernel reuse. The
[feasibility report](../results/2026-09-13-viqr-kernel-reuse.md) owns the
prototype measurements.

### Kernel-reuse execution checklist

- [x] K1: freeze before sources/oracles; implement and verify gpyreg API.
- [x] K2: implement bounded VIQR reuse and compatibility checks.
- [x] K3: numerical/performance gates, 18-run replay, full suites and CI.
- [~] K4: release validated gpyreg, update dependency/pin, integrate and archive.
- [ ] Final independent doublecheck and closeout.

Execution baseline: PyVBMC `15a14cc`, feature branch
`dev-noisy-viqr-kernel`; gpyreg `a2f8ddc`/v1.1.0, feature branch
`dev-predict-cross-covariance`. Isolated before checkouts and the gpyreg
candidate are under `dev/scripts/runs/viqr_kernel_20260913/`.
Before-source worktrees and oracle-output dumps are preserved. The candidate
pair is PyVBMC `9e8d2f2` with gpyreg `355d754`; the dependency change is in
[gpyreg PR #45](https://github.com/acerbilab/gpyreg/pull/45).

Initial validation: gpyreg's full suite passed (198 tests); the focused API
module then passed 50 tests after adding the independent review's missing-factor
coverage. API coverage is consolidated in `test_predict_cross_covariance.py`.
PyVBMC acquisition/importance checks passed (123 tests), the normal oracle
suite passed (143 tests, 15 skips), and all 11 fixture groups passed the exact
comparison against the frozen outputs. Independent code reviews found no
implementation defect. Production measurements pass: all 96 comparisons are
exact; the late sieve is 1.205x faster; no control or fallback exceeds the 5%
slowdown threshold. Full details and raw measurements are in the
[production report](../results/2026-09-13-viqr-kernel-production.md).
Both nine-cell candidate CI matrices passed. The full local PyVBMC suite
passed (1277 tests, 39 skips, 2 successful reruns). The bounded 18-run replay
passed in 45.9 minutes from `16dcc3f` (documentation successor of `9e8d2f2`)
and gpyreg `355d754`: all 18 stored trajectories and final results are exact,
all initial designs match, and no runs are flagged. Before/after source and
input hashes match. Tracker edits during replay affect documentation only.
Independent replay review confirmed all output/input hashes and reported no
findings. Gpyreg PR #45 is merged as `39536b0`, whose tree equals `355d754`;
both repositories' merged-source matrices passed all nine cells before release.
Gpyreg's standalone tests workflow was re-enabled after GitHub disabled it
for inactivity. Gpyreg v1.2.0 is published from the validated merge; its PyPI
wheel passes API/version/source checks. The final dependency floor is 1.2.0
and CI pin is `39536b0`; final PyVBMC CI and integration remain open. The integration is
[PyVBMC PR #174](https://github.com/acerbilab/pyvbmc/pull/174), targeting `dev-next`.

### Objective and boundaries

Compute the training-to-candidate covariance once per GP hyperparameter
sample and reuse it in standard VIQR. Preserve the criterion, integration
points and weights, random stream, candidate set, search settings, GP fits,
noise estimates, variance penalties, integer mapping and bound checks.
The accepted numerical scope is floating-point rounding; aim for exact
outputs on the generating platform, as in all 24 probe cases.

This change covers gpyreg's supported prediction API and the built-in
standard VIQR path. IMIQR cache reuse, experimental-loss removal, adaptive
sieve/search, multistart L-BFGS-B and BQ/QMC remain separate tasks. In
particular, none of those tasks depends on this change.

### Proposed API and integration

**gpyreg:** append a keyword-only `return_cross_covariance=False` argument
to `GP.predict`. When true, append a tuple with one entry per GP
hyperparameter sample to the existing return values:

- Ordinary prediction: `(mu, s2, cross_covariance)`.
- With `return_lpd=True`: `(mu, s2, lpd, cross_covariance)`.
- Each trained-GP entry contains the `(N_training, N_candidates)` latent
  kernel matrix `Ks = K(X, X_star)` already computed by the predictor.
  This is the unconditioned kernel, not posterior predictive covariance.
  Preserve sample order and values; do not stack or transpose for the return.
  Retain known unchanged bundled kernel implementations' fresh matrices
  directly. For custom or overridden covariance methods, snapshot each
  matrix immediately after computation: a custom kernel may reuse a scratch
  buffer across calls. This copying is confined to the requested extra
  output and does not change which matrix prediction uses. Do not impose
  new lifetime requirements on existing custom covariance implementations.
- `separate_samples=False` still averages `mu`, `s2` and any `lpd` as
  before; the kernel tuple remains per hyperparameter sample. There is no
  useful averaged kernel for the downstream calculation.
- For a prior-only GP with posterior hyperparameters but `self.y is None`,
  return one `None` entry per hyperparameter sample: the predictor has not
  computed a training cross-covariance. A GP with training data but missing
  factors after `clean()` or `compute_posterior=False` retains its existing
  unsupported prediction behavior; the new flag does not rebuild it.
- `add_noise`, `y_star`, `s2_star`, custom covariance implementations and
  both posterior factor representations retain their existing behavior.
  The extra output contains the kernel values used by that predictor.
- The default false path retains its return arity and does not retain
  matrices from earlier samples. No new GP attributes, persistent caches,
  serialization fields or random draws are introduced. Returned kernels
  are for read-only use by consumers; do not mutate flags on arrays owned
  by a custom covariance implementation. The returned tuple contains stable
  per-sample values; custom-kernel entries may be defensive copies.

**PyVBMC:** keep `_compute_acquisition_function`'s existing extension
contract. Introduce two protected hooks in `AbstractAcqFcn` for prediction
with an optional call-local context, and computation using that context.
The default hooks call the existing predictor and existing acquisition
method with their previous arguments. `__call__` retains its arithmetic
and pointwise checks, passes the context between the hooks, and releases
it after acquisition evaluation.

VIQR's hooks request and consume the kernel tuple only for the built-in
`AcqFcnVIQR` type with `getattr(self, "loss", "iqr") == "iqr"`, the standard
SE-ARD covariance implementation, and the unmodified gpyreg predictor.
Custom acquisition subclasses, custom/overridden covariance calculations
and GP prediction overrides use the original path. Recognize bound method
implementations, including instance overrides, rather than assuming
`isinstance` proves compatibility. This avoids replacing VIQR's hard-coded
SE calculation with a custom kernel's potentially different result.
Also require VIQR's `_compute_acquisition_function` to resolve to its
original implementation: an exact-type instance with an overridden method
must still call that override. Keep original method references outside
serialized instance state so later class or instance overrides are detected.

Move VIQR's common arithmetic into one private helper accepting an optional
kernel tuple. Its existing `_compute_acquisition_function` delegates with
no tuple; the fast hook delegates with the tuple. Both paths retain the
same loop and formulas. The fast path uses `cross_covariance[s].T` as a
view; it computes the candidate-to-integration-point kernel as before.
No output, context or matrix is stored on the acquisition, GP, VP or
`optim_state`. Legacy saved VIQR objects require no new attributes.

**Memory:** use an internal constant of **128 MiB** for the retained
cross-kernel payload. Before requesting the extra output, calculate
`8 * N_training * N_candidates * N_GP_samples` with Python integers on
the supported float64 path. Allow reuse at or below the cap; use ordinary
prediction and VIQR evaluation above it. Apply this check after candidate
normalization and integer mapping. This is a cap on retained matrix bytes,
not total process memory or the complete call's peak allocation. Empty or
untrained states follow the existing path. The check neither chunks nor
discards candidates and has no user-facing option.

### Phase K1: freeze the baseline and implement gpyreg

**Executor:** Astra orchestrator freezes provenance; Sol implementation
agent implements the API and focused tests. Only Astra runs heavy checks.

1. Inspect both repositories' instructions and Git state. Preserve unrelated
   changes. Start a PyVBMC feature branch from the settled guarded-sinh work
   (`a98dac0` at planning time), and a separate gpyreg feature branch from
   the checked source (`a2f8ddc`, v1.1.0, at planning time). Use isolated
   checkouts if either branch has concurrent work. Record exact source SHAs,
   installed module paths, library versions and hashes before measuring.
2. Read `dev/2026-09-02-modernization-discussion.md` before numerical edits.
   Dump before-change oracle outputs with
   `python dev/scripts/make_oracle_fixtures.py --dump-outputs <before-dir>`.
   Preserve the seven noisy captures, supplementary oracle capture, and
   18 guarded-sinh replay traces under
   `dev/scripts/runs/viqr_sinh_20260913/`. Verify their recorded hashes.
3. In gpyreg's `gpyreg/gaussian_process.py`, implement the optional output
   at the existing `Ks` assignment and return sites. Preserve numerical
   evaluation order and the false path. Do not use the feasibility probe's
   source substitutions in production.
   For zero-copy eligibility, snapshot the original compute methods of the
   five fresh-output bundled classes in a private module-level mapping:
   `SquaredExponential`, `Matern`, `RationalQuadraticARD`,
   `MaternIsotropic` and `SquaredExponentialIsotropic`. Require exact class
   identity and the original bound method on that instance. Unknown classes
   and class/instance overrides use
   `np.array(Ks, copy=True, order="K", subok=False)` for the returned entry.
   Evaluate eligibility only when the new output is requested.
4. Add focused checks in `gpyreg/testing/test_gaussian_process.py` for
   false/true output agreement; return arity
   with/without `lpd`; separate/averaged samples; observation-noise options;
   one/multiple samples; trained/prior-only GP; `(N,M)` orientation; custom
   kernels; and both `L_chol` representations. Count kernel evaluations to
   prove the option does not recompute `Ks`. Verify direct retention of
   known fresh built-in matrices, and independent snapshots from a custom
   kernel that deliberately reuses one scratch buffer across samples.
   Include an isotropic case in `test_gaussian_process_isotropic.py` and
   a Matern/RQ case to exercise the generic API. Check no matrices remain on
   the GP after a call. Use small explicit GP updates, not new fitting runs.
5. Update the `GP.predict` docstring, rendered by
   `docsrc/source/gaussian_process.rst` in gpyreg,
   with shapes, tuple order, lifetime and per-sample behavior. Correct the
   existing `add_noise` documentation mismatch while touching this
   docstring (the code default is false). Keep other GP APIs unchanged.

**Acceptance:** default prediction outputs and errors retain their previous
behavior; requested covariance entries equal the predictor's actual
matrices. Every supported return combination has a clear contract and a
focused check. Unexpected numerical differences require investigation.

### Phase K2: integrate bounded reuse in VIQR

**Executor:** Sol implementation agent; Astra reviews the cross-repository
boundary and runs checks. Depends on K1's API, not on a published release.

1. Implement the two protected hooks in
   `pyvbmc/acquisition_functions/abstract_acq_fcn.py`, preserving total
   variance aggregation, regularization and masking order.
2. Implement eligibility, memory fallback and shared arithmetic in
   `pyvbmc/acquisition_functions/acq_fcn_viqr.py`. Keep reduction losses
   and other acquisitions on their existing paths. Add no constructor
   arguments or serialized state.
3. Extend `test_abstract_acquisition_function.py` with an acquisition
   subclass implementing only the old method signature. In focused VIQR
   checks, cover custom VIQR subclasses and overridden acquisition, GP and
   kernel methods, including instance overrides on an exact-type VIQR
   object; verify those methods are still called.
4. Add fast/fallback equivalence checks at one, several and sieve-sized
   candidate counts; one/multiple GP samples; alternative quantiles; both
   factor representations; integer mapping, bounds, penalties and saved
   objects without `loss`. Use `_look_ahead.py` and the existing
   `test_acq_fcn_viqr_losses.py` setup for independent value checks.
5. Test the memory decision below, at and above the cap by lowering the
   private constant on small fixtures. Above the cap, assert prediction
   is called without requesting kernels, and the acquisition values and
   RNG state match ordinary evaluation. Exercise repeated calls after GP
   and candidate changes to catch accidental stale reuse. Check float64,
   input/state immutability, and absence of retained context after errors.
6. Update the existing acquisition API reference/docstrings as necessary
   to describe unchanged public behavior and custom-subclass compatibility;
   private hooks need developer docstrings, not new public API pages.

**Acceptance:** only eligible standard-VIQR calls request kernels. The
existing subclass signature and all fallback behavior remain usable.
Kernel matrices have one-call lifetime, and allocation of the retained
payload is guarded before prediction. No duplication of VIQR formulas.

### Phase K3: numerical, performance and replay gates

**Executor:** Astra main thread performs all compute. Independent Sol
reviewers use static analysis of code and recorded evidence.

1. Run focused acquisition/importance checks and the normal oracle suite:
   `python -m pytest pyvbmc/testing/acquisition_functions
   pyvbmc/testing/vbmc/test_active_importance_sampling.py -q`, then
   `python -m pytest pyvbmc/testing/oracles -q`. Run the generator with
   `--check --exact --against <before-dir>`. Require unchanged default
   GP predictions and classify every VIQR difference; do not regenerate
   references to make a change pass. Check unrelated oracle outputs too.
2. Keep `probe_viqr_kernel_reuse.py` and its recorded source hashes as the
   historical feasibility recipe. Add a production comparison mode or
   developer runner that evaluates the actual before/after packages on
   identical restored captures. It must support different gpyreg sources;
   do not disable the existing runner's dependency checks wholesale.
   Record both package revisions, hashes and imported module paths.
3. Use isolated, warmed before/after worker processes when comparing the
   two package pairs. Give each explicit source paths and the same numerical
   environment; time only complete public acquisition calls inside the
   workers. Alternate requests, keeping the other worker idle, to avoid
   concurrent heavy compute. Restore captured live factors and importance
   arrays without redrawing. Verify the baseline against captured outputs
   before accepting comparisons. Assert runtime BLAS thread counts of one.
4. Repeat the 24 captured cases and both fast/fallback paths. Record complete
   call timings, peak traced allocations, retained payload and fallback
   counts. Use nine alternating rounds, with at least 31 pairs per round
   for small calls. Include ordinary GP prediction with the flag false
   and representative non-VIQR acquisition calls to check hook overhead.
   Recheck only unresolved timing differences; no new target evaluations
   are needed for these measurements.
5. Performance acceptance: preserve a repeatable late-sieve gain (target
   at least 1.10x on the N=150 capture), with no repeatable slowdown above
   5% on small/default/fallback calls. Small early-state gains need not be
   significant. Account for timing variation and report individual states.
   Verify the payload cap; report total peak memory separately. A failure
   calls for revising the implementation, not silently raising tolerances
   or changing search settings.
6. Run the bounded 18-candidate replay, seeds 0-2 for the same six noisy
   configurations as the guarded-sinh campaign. Use
   `python dev/scripts/golden_replay.py --configs
   rosenbrock_D2_noise1,rosenbrock_D2_noise3,logreg_D5_noise3,student_D8_noise3,multisensory_s1_D6_noise1.3,timing_D5_noise2.2
   --seeds 0-2 --baseline dev/scripts/runs/viqr_sinh_20260913/replay
   --sidecars dev/golden/baseline --out <kernel-candidate-dir>`.
   The immediate trajectory baseline is the validated guarded-sinh
   campaign; accuracy fences remain the promoted 990-run population.
   Its use relies on the documented equivalence between the replayed
   guarded-sinh source and the final before-change numerical source.
7. Require identical initial designs, successful finite final results and
   no unresolved replay flags. Investigate trajectory changes against the
   immediate baseline and promoted population. Historical exact claims
   cover only stored state, not omitted returned-transformer information.
   If traces are missing, restore the preserved artifacts first; final-only
   comparisons do not replace the intended trajectory check. Any needed
   extension beyond this allocation requires a reason tied to a finding.
8. Run both repositories' full suites and CI matrices, repository formatting
   hooks, and independent doublecheck. On Windows use workspace-owned
   pytest temporary/cache directories when required. Record exact commands,
   versions, outcomes and unresolved limits. gpyreg's current matrix covers
   Python 3.9-3.11 on three OSes; PyVBMC's candidate matrix also covers
   Python 3.12. No fresh 990-run campaign or
   repair of historical inference failures is part of this change.

### Phase K4: dependency rollout and closeout

**Executor:** Sol prepares repository changes and reviewable PRs; Astra
coordinates verification, release sequencing and final integration.

1. During development, install the gpyreg candidate editable before running
   PyVBMC. Use its exact commit in development CI. Keep PyVBMC's current
   minimum requirement during this pre-release validation so dependency
   resolution does not request a nonexistent release. Do not merge or ship
   the new PyVBMC path against the old gpyreg pin.
2. After both-package validation and review, merge the gpyreg PR and record
   its merge SHA. Verify gpyreg CI and PyVBMC integration CI against that
   merged source are green; use the manual `gpyreg-ref` dispatch to check
   gpyreg main without waiting for the scheduled run. Fetch origin and tags
   and choose the next available additive release (proposed `v1.2.0`).
   Create the version tag and GitHub release at exactly the validated merge
   SHA, following gpyreg's release workflow. Verify publication and that
   the PyPI artifact exposes the new API before raising PyVBMC's minimum
   to that version and updating `GPYREG_PIN` in
   `.github/workflows/test-matrix.yml` to the released commit. Reinstall the
   editable sibling after fetching the release tag so setuptools_scm and
   the requirement agree.
3. Run the final PyVBMC nine-cell matrix with the final pin and dependency
   floor; verify installation resolves the intended gpyreg source/version.
   Keep gpyreg and PyVBMC commits separate and traceable. Integrate into the
   active 1.5 development branch after checks, without publishing PyVBMC 1.5.
4. Update this plan's execution status, `dev/TODO.md`, and the existing
   feasibility report with a link to production evidence. Store the
   consolidated raw results under
   `dev/experiments/noisy-acquisition-efficiency/`; use a new artifact for
   production validation so prototype measurements remain distinguishable.
   Update repository setup/pin notes where their minimum-version statement
   changes. Do not add a separate completion journal.

**Rollback:** revert the PyVBMC integration/pin change to the frozen
guarded-sinh source. The opt-in gpyreg API may remain released: ordinary
calls retain their old behavior. Preserve the before checkouts, captures,
oracle dumps and validation artifacts. Do not change the promoted golden
population as part of rollback.

### Decisions and approval

- **Optional gpyreg return, with a release-backed dependency floor:** a
  supported API avoids duplicated prediction logic and temporary monkey
  patches. Rejected permanent runtime feature detection against old gpyreg:
  it adds a second installed-dependency path for a capability both
  repositories can release together. Development validation precedes the
  floor bump so an unpublished dependency cannot block installation.
- **Transpose views, retained up to 128 MiB:** the probe's 56.25 MiB late
  kernel payload fits, preserving its measured gain. Rejected unconditional
  retention (unbounded extra memory), mandatory copies (lost much of the
  late gain), and streaming/chunking in this change (broader interfaces or
  changes to batching and arithmetic). The cap bounds kernel payload,
  not total memory, and remains an internal implementation choice.
- **Conservative fast-path eligibility:** built-in standard VIQR and
  standard GP/kernel implementations receive the optimization. Rejected
  automatically enabling it for arbitrary subclasses: VIQR's hard-coded
  SE calculation and custom prediction/kernel overrides may differ.
  At the public gpyreg boundary, unknown covariance implementations receive
  defensive output snapshots; changing the existing custom-kernel ownership
  contract would make an additive optimization unnecessarily disruptive.
- **Existing extension contract through private hooks:** avoids changing
  every acquisition method signature or retaining per-call state on objects.
- **Bounded 18-run validation against the immediate baseline:** separates
  this change from guarded-sinh trajectory differences and uses existing
  captures. Rejected a new reference-sized campaign without a specific
  unresolved regression question.

No scientific choice blocks this plan. API tuple semantics, the 128 MiB
cap, the release-backed dependency floor and the bounded replay allocation
were approved for execution on 2026-09-13. Exact release naming is confirmed
against gpyreg's tags at execution time. Plan approval covers implementation,
validation, the prerequisite gpyreg release and integration into the 1.5
development branch; PyVBMC 1.5 publication remains separate.

Independent Sol plan doublecheck completed on 2026-09-13 with no remaining
findings. Review corrections specify defensive snapshots for custom kernel
scratch buffers, fallback for acquisition-method overrides on exact-type
instances, and release from the merged, validated gpyreg commit. The review
covered source/API branches, probe evidence, tests, documentation and both
repositories' dependency/release workflows. This was a static plan review;
implementation tests and campaigns remain the work described above.
