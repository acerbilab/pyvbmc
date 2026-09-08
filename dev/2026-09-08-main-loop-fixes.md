# Main-loop latent fixes

The PI parked the paired final-boost experiment on 2026-09-08 and authorized
continuing the independent main-loop repairs. Boost work is preserved at
local commit `764a177` on `dev-final-boost`; its
[restart record](2026-09-08-boost-penalty-pilot.md#parked-experiment-restart)
specifies the remaining experiment. No boost batch is scheduled.

Main-loop work starts from validated `dev-next` commit `3e879b6` on
`dev-main-loop-fixes`, without the experimental boost implementation.
The [live plan](plans/latent-bug-fixes.md) owns the implementation checklist.
Final Q1/Q4 choices and the existing PyTorch feasibility decisions are unchanged.

## Weighted GP covariance (Phase 3 complete; population assessment pending)

The correction treats each historical GP hyperparameter matrix as sample
rows, distributes an iteration's total weight across its samples, and uses
the unbiased weighted outer-product covariance. The current GP determines
the parameter dimension. Sample counts may vary between iterations;
incompatible model dimensions are skipped. History decay uses
`sKL / (tol_skl * fun_evals_per_iter)`. Undefined covariance falls back to
the existing default sampler widths.

The legacy `gp_fit` oracle explicitly disables history-derived widths
to preserve its previous numerical contract. Separate controlled-history
and authentic-history tests exercise the corrected estimator and the
widths actually supplied to the sampler.

Before edits, `make_oracle_fixtures.py --check --exact` passed all eight
fixtures. The interpreter was the original `.venv` (Python 3.12.6, NumPy
2.5.2, SciPy 1.18.1), with OMP/OpenBLAS/MKL threads set to 1. Actual imports
resolved to this checkout and sibling gpyreg at
`a2f8ddce867f502e29717959cf0ff3529f598618`. The pre-change fixture hashes and
environment are recorded in
`dev/scripts/runs/latent_fixes/gp_covariance_20260908/before.json`.

Verification completed:

- 146 focused tests passed, 15 skipped (`final_focused.log`). The subsequent
  addition of two raw sampler-width dtype assertions passed all eight tests
  in the authentic-history module (`width_dtype_tests.log`).
- All 11 fixtures pass exactly (`all_exact.log`): the eight original states,
  now with the additive controlled oracle, plus three authentic captures.
  Existing state arrays and reference outputs were preserved by
  `--add-oracle gp_fit_history`; no snapshot recipe was regenerated.
- Authentic captures are normal D2 iterations 3 and 6 and noisy Rosenbrock
  D2 iteration 3. All have finite proposed GP-fit widths, and respectively
  6, 6 and 3 effective sampler widths are below gpyreg's internal defaults.
  Source hashes, inputs, RNG, proposed/default/effective widths and fit
  outputs are stored. The later history contains changing sample counts.
- Independent Sol review found no remaining correctness or coverage issues.

The first capture attempt selected normal iteration 1 and noisy iteration 2,
whose infinite reliability indices produced infinite proposed widths.
gpyreg capped these to its defaults, so those captures did not exercise
history influence despite replaying exactly. All six initial files remain
under `initial_capture_diagnostics/`. The corrected selection requires
finite proposals and actual reduction of at least one sampler width. This
changes fixture selection only, not production behavior. The first focused
test's bit-exact symmetry assertion was also corrected for a 4.4e-16
off-diagonal rounding difference; covariance value checks remain tight.

The default five-case trajectory replay took 2.9 minutes. All initial
designs match exactly; main-loop differences first appear at iterations
3, 4, 3, 5 and 3 for normal, banana, halfnormal, cigar and noisy Rosenbrock.
All runs finish with finite results. Four pass the accuracy fences.
**Cigar D4 seed 0 is flagged**, and the command correctly exits 1:

| Metric | Reference | Corrected covariance | Reference fence |
|---|---:|---:|---:|
| Absolute log-evidence error | 0.00174 | 0.0547 | 0.0324 |
| gsKL | 0.000422 | 0.0124 | 0.00676 |
| MMTV | 0.00880 | 0.0296 | 0.0249 |
| Target evaluations | 130 | 120 | Not a replay gate |

This is an adverse accuracy change on that seed, though all three metrics
remain inside the benchmark's usability thresholds. A bounded follow-up
on Cigar seed 1 passes every fence: log-evidence error 0.00526 to 0.00559,
gsKL 0.00127 to 0.000167, MMTV 0.00883 to 0.00808, evaluations 140 to 130.
Its first difference is iteration 6 and its initial design is exact.
This diagnoses a second seed; it does not estimate a population effect or
erase the seed-0 flag. Correctness gates support the approved repair;
quality acceptance remains for the planned integrated population assessment.
Neither fences nor reference outcomes were changed. Full results are in
`replay/replay.md` and `cigar_followup/replay.md` under the evidence directory.

No reference population is regenerated. The Phase 3 checkpoint is `8f906d8`.
Commit hooks reformatted several source files after capture; fixture source
hashes retain the measured source's provenance. A fresh check of the committed
code passes all 11 fixtures exactly (`committed_exact.log`).

## Acquisition regularization (Phase 4 complete)

The isolated implementation `1ec320e` was applied as `90d08d3`. New state
uses the canonical key; old saves and snapshots activate the legacy alias
only when the canonical key is absent. Results are normalized to one value
per point before regularization and bounds masks. Zero variance uses the
limiting penalty without a new variance floor.

The first focused run passed 110 tests and found four failures in the new
VIQR/IMIQR test fixture: constant GP noise returned one value, whereas the
temporary nearest-training-point noise array needed one per training row.
The fixture now broadcasts that value. All eight real VIQR/IMIQR cases and
two actual old-save tests (latest state and iteration 0) pass. No production
change was needed for those failures. Independent Sol review cleared the
implementation and the two follow-up test changes.

All 34 stored acquisition checks agree with the penalty formula. With
regularization disabled, every result is exact against the old reference.
Only four acquisition names and `active_sample_step` change, on the seven
noiseless snapshots. Targeted updates preserve all other arrays, and the
full 11-fixture exact check passes. Evidence is under
`dev/scripts/runs/latent_fixes/acquisition_20260908/`.

The separate five-case replay uses the Phase 3 traces as its comparison
baseline and the unchanged golden population's accuracy fences. This
isolates the acquisition change without rerunning the preceding phase.
It completes in 3.1 minutes with zero flags and all initial designs exact.
Normal, banana and halfnormal first differ at iteration 3; Cigar at 4.
Noisy Rosenbrock retains its entire stored loop and final result exactly.
Cigar seed 0 is back within the population fences: evidence error 0.00775,
gsKL 0.000887, MMTV 0.0157, and 130 target evaluations. This cumulative
result does not erase the separately recorded covariance-only seed-0 flag.

## GP sampling termination (Phase 5 complete)

The implementation is integrated after the Phase 4 gate.
It restores the existing variance criterion, records logged distinct-location
counts in history, and backfills compatible old histories on load or direct
continuation. Missing counts or variances do not trigger this optional stop.
The existing hard N/K stops and configured threshold remain the contract.

The 38 focused loop-termination, save/load and legacy-acquisition tests pass,
as do all 11 exact fixtures, without reference changes. The first focused
run exposed a NumPy boolean return where the existing private method returned
a Python bool; the final result is explicitly converted. Five setup errors
were Windows permissions on pytest's default temporary directory, resolved
by using a workspace `--basetemp`. Logs are retained under
`dev/scripts/runs/latent_fixes/sampling_termination_20260908/`.

The five-case replay against Phase 4 takes 5.1 minutes and passes with zero
flags: all stored loops, initial designs and semantic final fields are exact.
The observer records 52 real criterion calls, all false. Independently
recomputing their weighted variances confirms that none crosses the default
`1e-4`; this check therefore does not demonstrate a default trajectory change
or a speed improvement. Missing historical returned transformers still limit
what the compact traces can certify.

A separate bounded normal D5 seed-0 run forces the criterion with a high
finite tolerance (`1e6`), disables the hard N/K stops, and caps the run at
11 iterations. The real loop sets `stop_sampling=50` at iteration 8; the
next GP fit switches from 11 samples to the configured stable count of zero,
and the following iteration also uses zero. Every recorded N equals that
iteration's `optim_state["N"]`. Full trace, options and transition assertions
are under `forced_crossing_11iters/`. This is an integration check with custom
options, not a quality benchmark or proposed threshold change. The initial
eight-iteration attempt ended in warmup before the criterion could run; its
trace and failed assertion remain under `forced_crossing/`.

Independent Sol review found a unit-test gap: manually setting the stopping
flag bypassed the actual loop assignment. The guarded transition is now a
small private method called by both the loop and test; the test asserts the
flag before checking the subsequent `_gp_hyp` sample count. Guard cases also
assert that state stays unchanged. Follow-up review has no remaining findings.

After that extraction and repository formatting, the combined focused gate
passes **296 tests, 15 skipped** in 72 seconds (`integrated_focused.log`).
It covers GP training, initialization, loop termination, save/load, the whole
acquisition test directory and numerical oracles. All 11 fixtures also pass
the separate exact comparison (`final_exact.log`). No Phase 5 references
were changed. The full pytest suite, CI and integrated population assessment
remain final-integration work; these focused gates do not replace them.

## Next pickup

Phases 3-5 are implemented and independently reviewed on `dev-main-loop-fixes`.
Keep the covariance-only Cigar seed-0 adverse result in the population analysis,
alongside its passing cumulative result after acquisition regularization.
Next unresolved work is the Phase 6 eta-bound comparison and PI choice,
the Phase 7 upstream gpyreg step-out repair and final integration gates,
and the explicitly parked paired boost campaign. Q1/Q4 and PyTorch decisions
have not been selected or revised by this work. No population campaign was
started, and no numerical job remains attached to this task.
