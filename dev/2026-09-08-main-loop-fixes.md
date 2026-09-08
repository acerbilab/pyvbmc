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

The legacy `gp_fit` oracle will explicitly disable history-derived widths
to preserve its previous numerical contract. Separate controlled-history
and authentic-history tests will exercise the corrected estimator and the
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

No reference population is regenerated. Acquisition regularization is
prepared independently at `1ec320e` in `dev-acq-regularization-stage`, under
`dev/scripts/runs/latent_fixes/acq_worktree`; its separate gate follows.
