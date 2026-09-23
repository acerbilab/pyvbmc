# Wave 7 fix round: gpyreg fix agent D

Raw report of fix agent D, wave 7 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-23. One Opus agent on a
worktree of gpyreg made by hand (`../gpyreg-w7`, branch `w7-fixes`, cut
from gpyreg's `main` at `0186d89`, the tag `v1.3.0`), made the fixes of
rows W7-14, W7-15 and W7-16 of `../verification/wave7.md` as the PI ruled
them, and the test notes of slice O4, one commit each, with the entries of
a section `1.3.1 (unreleased)` in gpyreg's release notes. It ran only the
test functions it added or changed, by node id, while another session held
the machine's heavy slot. The commits stay on gpyreg's branch until the
gates of the fix pass, PyVBMC's whole suite and the exact oracle check
against it among them, and gpyreg's pull request. The text below the rule
is the agent's final message, unedited.

---

# Wave 7 fix round: report of gpyreg fix agent D

All three rows (W7-14, W7-15, W7-16) are fixed, and the three O4 test notes each have their own commit. That makes six commits on `w7-fixes`, and the worktree is clean. Across 2304 GPs built the way PyVBMC builds them, the fixed and unfixed gpyreg give exactly the same numbers everywhere. Nothing was pushed or merged, and no file outside the gpyreg worktree and my scratch directory was changed.

## 1. Setup
- **Starting commit:** gpyreg `0186d89` (v1.3.0). Worktree `C:\Users\luigi\Documents\GitHub\gpyreg-w7`, branch `w7-fixes`.
- **Import paths:**
  - `python -m pytest` from the worktree root: gpyreg `C:\Users\luigi\Documents\GitHub\gpyreg-w7\gpyreg\__init__.py`.
  - Scripts run with `PYTHONPATH="…/gpyreg-w7;…/pyvbmc-wave7"`: gpyreg from `gpyreg-w7`, pyvbmc `C:\Users\luigi\Documents\GitHub\pyvbmc-wave7\pyvbmc\__init__.py`.
  - The unfixed comparison runs used `…\GitHub\gpyreg\gpyreg\__init__.py`.
- **Versions:** numpy 2.5.2, scipy 1.18.1, Python 3.12.6.
- **Formatting:** gpyreg has no pre-commit hooks installed. Before each commit I ran gpyreg's own hook config with `pre-commit run --files …` (black 22.3.0, isort 5.10.1, pycln 1.2.5, from the existing cache, so nothing was installed). All hooks passed.

## 2. Commits

| hash | row | what it does | test (node id) | failure before the fix | run after |
|---|---|---|---|---|---|
| `e5b7238` | W7-14 | Removes `dlp[f_idx] = np.nan` in `__compute_log_priors`. A coordinate with equal bounds now gets its own prior's gradient, or 0 where the prior leaves it unset (no prior, or a value inside a smooth box), as in `gplite_hypprior.m`. Adds the release-notes entry. | `gpyreg/testing/test_gaussian_process.py::test_fitting_with_fixed_bounds`, parametrized over no prior, a smooth box around the fixed value and a smooth-box Student's t around it. Adds a seeded `n_samples=0` fit that checks the prior contributes nothing to the fixed coordinate and that the free-coordinate gradient is below 1e-2. | All 3 cases: `assert gradient[3] == gradient_likelihood[3]` gave `nan == 0.5874…` (and `nan == 5.6623…` for the smooth box). The free-coordinate gradient at that point was up to 5.5, against about 1e-3 after the fix. | 3 passed |
| `5ae1c44` | W7-15 | In `__recompute_normalization_constants`, where `lb` is strictly above the prior's centre (`mu`, or `(a+b)/2` for a smooth box), the mass is `sf(lb) - sf(ub)`. Elsewhere it is the old `cdf(ub) - cdf(lb)`, computed by the same calls. Adds `smoothbox_sf` and `smoothbox_student_t_sf` to `f_min_fill.py`, and the release-notes entry. | `::test_prior_mass_in_the_upper_tail` (4 families): the upper-tail mass and log prior equal the mirrored lower-tail ones. `::test_prior_mass_from_the_centre_down`: bit-identity for `lb <= mu`. `test_smoothbox.py::test_sf`, `test_smoothbox_student_t.py::test_sf`. | `assert masses[0] > 0.0` gave `0.0 > 0.0` for all 4 families. The centre-down test passed before and after; it guards the unchanged path. The two `test_sf` tests import functions this commit adds, so they could not run on the unfixed code. | 6 + 2 passed |
| `5499b59` | W7-16 | `set_priors` refuses a location that is not finite beside a finite `sigma`: an infinite or NaN `mu`, or an infinite or NaN smooth-box end. The message names the hyperparameter and says what is wrong. A block coordinate whose location and `sigma` are both NaN still reads as having no prior. Updates the docstring and adds the release-notes entry. | `::test_set_priors_refuses_a_location_that_is_not_finite` (11 cases), `::test_set_priors_refuses_a_location_that_is_not_finite_in_a_block`. `::test_get_priors_returns_what_set_priors_reads_back` gains 5 round-trip cases. | `Failed: DID NOT RAISE ValueError` in all 12. The round-trip test passed before and after (10 cases). | 35 passed, together with the existing `set_priors` tests |
| `14736b5` | test note | Gradient of the log likelihood where the noise varies by point: user-provided variance with and without its scale, rectified noise with its threshold inside the target range (8 of 14 points active), D = 1, repeated inputs. | `::test_likelihood_gradient_with_a_noise_per_point` (4 cases) | No defect to see. Margin: error/max gradient about 1e-11 against a tolerance of 1e-6. A deliberately broken copy (per-point sum replaced by mean(dsn2)·trace) fails 3 of 4 cases; the fourth has no per-point noise hyperparameter, so it cannot see that change. | 4 passed |
| `dcaea44` | test note | Low-noise representation at sn2 = 0.9e-6 on a well-conditioned set (cond 1.01), with 1.1e-6 as the control: checks the representation flag, the negative log marginal likelihood against a dense evaluation (rtol 1e-12), and the gradient against finite differences. Constant and negative-quadratic means. | `::test_likelihood_in_the_low_noise_representation` (4 cases) | No defect to see. Margin about 2e-13 against 1e-8. Scaling the low-noise diagonal by 1.5 in a scratch copy fails both low-noise cases. | 4 passed |
| `8be6ee4` | test note | Seeds `test_gp_gradient_computations`: every draw now comes from `default_rng(seed)`, including `random_function(X, rng=rng)`. Seeds 0, 1, 2 (seed 2 draws one hyperparameter sample, the others two). | `::test_gp_gradient_computations[0-2]` | Seeds 0–29 all pass: largest error 5.5e-12 for the likelihood and 1.1e-12 for the prior, against 1e-6. | 3 passed |

The final run of the added, changed and closely related tests by node id: 71 passed. Four RuntimeWarnings (overflow in exp, `covariance_functions.py:312`) come from the existing `test_setting_bounds`; I did not investigate them.

## 3. Evidence that nothing moved (W7-14, W7-15, W7-16)
All files are in the scratch directory `…\scratchpad\wave7\fix_D`.

- **What `pyvbmc_gp_outputs.py` records:** GPs built by `_gp_hyp` on a live VBMC object, over 2304 configurations:
  - D = 1, 2, 3, 5; levels 0, 1, 2; zero, constant and negative-quadratic means;
  - `noise_size` unset, 0.2 or 5.0, plus the rectified-noise branch of `_gp_hyp`;
  - 8 kinds of training set: regular, clustered, wide, noisy, target range 1e-8, target range 1e-3, equal targets, a shared input coordinate;
  - N = 12 and 40.

  For each it records the normalization constants (after `set_priors`, after the recommended-bound fill that `fit` does, and after a fit), the priors, the bounds, the log posterior and its gradient at 4 points, and 1008 seeded fits with 3 slice samples each.
- **Comparison:** run once under `0186d89` and once under `8be6ee4`, then compared with `compare_pyvbmc_gp_outputs.py`. The largest difference is exactly 0.0 for every quantity, and no configuration differs (`compare_final.log`). An earlier run at the W7-15 state gave the same result.
- **What the runs reached** (`coverage_pyvbmc_gp_outputs.py`, `coverage_final.log`):
  - 8832 prior coordinates with a finite lower bound. `(lb - mu)/sigma` runs from -14.73 to exactly 0: at level 0 without `noise_size`, the noise prior's centre is its lower bound, which is why the switch is strict. None lies above the centre. Masses run from 0.0075 to 0.9967.
  - 1344 coordinates with equal bounds. 1152 of them carry a Student's t prior (the collapsed noise pair). The other 192 are all on shared-coordinate sets, where the length-scale pair is (-inf, -inf) and the likelihood gradient is NaN with or without the fix.
- **The verifier's F2 script rerun on the fixed gpyreg** (`verifier_F2_fixed_*.log`): all 10 stored masses now equal the true masses. The fit with bounds [9, 10] reaches the optimum 25.8909 of the finite objective; before the fix it ended at -85.21.
- **Still for the gate:** PyVBMC's whole suite and `make_oracle_fixtures.py --check --exact` with `PYTHONPATH` naming `gpyreg-w7`. I did not run them.

## 4. Tests written but not run
None. No test needs `optimize()`.

## 5. Release-notes lines (`docsrc/source/release_notes.rst`, new section `1.3.1 (unreleased)`)
The section opens with: "A point marked **Upgrading** says what a script written for 1.3.0 may have to change."

- "The gradient of the log prior is zero, not NaN, for a hyperparameter whose lower and upper bounds are equal and that has no prior, or a smooth-box prior whose box holds its value. The NaN reached `GP.log_posterior` with `compute_grad=True`, and, where another hyperparameter has a prior, the optimizer of `GP.fit`, which stopped within an iteration, short of the optimum."
- "The probability that a hyperprior puts inside the bounds of its hyperparameter, by which `GP.log_posterior` renormalizes the prior, is computed from the survival function where both bounds lie above the centre of the prior. As a difference of two values of the cumulative distribution function it was zero with both bounds far in the upper tail (beyond about 8.3 scales of a Gaussian prior), which made the log posterior infinite everywhere and sent `GP.fit` to a poor point. Where the lower bound is not above the centre, the value is the same as before, to the last bit. `gpyreg.f_min_fill` has the survival functions of the two smooth-box families, `smoothbox_sf` and `smoothbox_student_t_sf`."
- "`GP.set_priors` refuses, with a message that says what is wrong, a coordinate whose `sigma` is finite beside a location that is not: an infinite or NaN `mu` of a Gaussian or Student's t prior, or an infinite or NaN end `a` or `b` of a smooth box, as it refuses a `sigma` that is not finite and positive. 1.3.0 took such a prior, and the log posterior was NaN with the bounds that `GP.fit` fills. A coordinate of a block without a prior is written, as before, with NaN for both its location and its `sigma`. **Upgrading:** a script that wrote a non-finite `mu` for no prior, which gplite reads that way, sets the hyperparameter's prior to `None`."

PyVBMC's changelog needs no sentence for users, since no PyVBMC number moves. A line is needed only if you raise PyVBMC's minimum gpyreg or the CI pin to 1.3.1.

Records for you to update (I edited none):
- The sheet entry "The hyperprior is renormalized to the bounds…". The verifier's replacement text in `wave7_O4.md` §8 describes the behaviour before the fix (NaN gradient, zero mass) and needs rewriting for the fixed code.
- The "No prior" sentence in the same section says a non-finite `mu` is not refused; it now is.
- The `dev/TODO.md` item on a NaN location goes with W7-16.
- Row W6-24 of `wave6.md` gets its flag.

## 6. Noticed beyond my rows, not fixed
- **The space-filling design has the same upper-tail problem as the mass.** `f_min_fill` maps the design through `cdf(lb) + (cdf(ub) - cdf(lb)) * S`. For N(0,1) with bounds [9, 10], 63 of 64 design values of that coordinate are +inf. A fit with `n_samples > 0` then raises `ValueError: The widths vector needs to be all positive real numbers`, because the standard deviation of the design column is NaN. The mirrored bounds [-10, -9] work. This happens identically before and after my fixes; PyVBMC cannot reach it. The same survival-function switch in the design (with `isf`) would fix it. Script: `f_min_fill_upper_tail.py`.
- **`set_priors` accepts an inverted smooth box (`a > b`).** The normalizer drops below 1 or turns negative. The log prior comes out a wrong finite value (-0.064 for (1, -1, 0.7), -11.9 for (3, -3, 0.7)) or NaN (smooth-box Student's t (3, -3, 0.7, 4)). This is the same kind of input check as W7-16. Script: `inverted_box.py`.

## 7. Final state
- `git status --porcelain` of `gpyreg-w7`: empty.
- `git log --oneline 0186d89..HEAD`:
```
8be6ee4 test: seed test_gp_gradient_computations
dcaea44 test: the marginal likelihood just below the low-noise switch
14736b5 test: the likelihood gradient with a noise that varies by point
5499b59 fix: set_priors refuses a location that is not finite beside a finite sigma
5ae1c44 fix: the prior's mass inside the bounds survives the upper tail
e5b7238 fix: a hyperparameter with equal bounds gets no NaN in the prior's gradient
```
