# Existing noisy-acquisition efficiency

Created 2026-09-13. Status: guarded-sinh implementation, validation and
independent review complete on `dev-noisy-viqr-sinh` (numerical revision
`6734817`), from `83692ac`. Evidence is archived below. Quadrature remains
a possible subsequent experiment in this workstream.

## Execution checklist

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
