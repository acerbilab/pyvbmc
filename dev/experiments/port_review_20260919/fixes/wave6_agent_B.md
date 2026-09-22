# Wave 6 fix pass: gpyreg fix agent B

Raw report of fix agent B, wave 6 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-22. One Opus agent on a
worktree of gpyreg made by hand (`../gpyreg-port-review-B`, branch
`port-review-wave6-B`, cut from gpyreg's `main` at `fdbafdf`), made the
fixes of the rows of `../verification/wave6.md` that live outside
`gaussian_process.py` (the slice sampler, the space-filling design, the
kernels, the mean and noise functions, the documentation), one commit per
row as the PI ruled them, and seeded the distributional tests of the slice
sampler on the orchestrator's instruction; the orchestrator reviewed the
diffs and cherry-picked the commits onto gpyreg's `port-review-wave6`, where
they have other hashes (`../verification/wave6.md`, "Fix commits"). The text
below the rule is the agent's final message, unedited.

---

# Wave 6 fix pass: report of gpyreg fix agent B

## Setup

Interpreter check, run from the worktree root (`C:\Users\luigi\Documents\GitHub\gpyreg-port-review-B`), with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` on every Python command of this pass:

```
C:\Users\luigi\Documents\GitHub\gpyreg-port-review-B\gpyreg\__init__.py
```

`git log -1` at the start:

```
commit fdbafdfc73b0e97270a92f3da4682fa857323c20
Author: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>
Date:   Mon Sep 21 15:05:13 2026 +0300

    chore(deps): bump pypa/gh-action-pypi-publish from 1.13.0 to 1.14.2 (#42)
```

`git status --porcelain` at the start: clean. Branch `port-review-wave6-B`. Nothing outside the worktree was changed except probe scripts in the scratchpad (`.../scratchpad/wave6/fix_B/`). No push.

## Commits, in order

| hash | row | what it does | test file :: test | failing assertion seen before the fix | test-file run after |
|---|---|---|---|---|---|
| 970f6c6 | W6-3 | a coordinate whose burn-in variance estimate is not positive keeps its width; a window of fewer than two iterations (burn-in 2 or 3) adapts nothing | `test_slice_sample.py::test_short_burn_in_keeps_the_chain_moving[2]`, `[3]`; `::test_burn_in_statistics_window_is_the_second_half` | `assert np.all(sampler.widths > 0)` → `array([0., 0.]) > 0` at both burn-ins (and 1 distinct sample of 6). The window test pins the existing `floor(burn/2)`-term window (W6-2's test note) and passes before and after, by design; its non-vacuity clause shows MATLAB's one-iteration-longer window gives other widths | 38 passed |
| 6bc0c23 | W6-14 | `base_widths` copied after an infinite width is replaced by 10 | `test_slice_sample.py::test_infinite_width_does_not_come_back_after_the_burn_in` | `assert np.all(np.isfinite(sampler.base_widths))` → `base_widths = [inf, 1.]`, and the run logged "Target density function returned NaN" with non-finite recorded samples | 39 passed |
| f420255 | W6-15 | a frozen free parameter is detected by `np.ptp(samples, axis=0) == 0`; `R` and `eff_N` are NaN for a parameter that did not move; the autocorrelation pairing kept and documented as starting at lag 0 | `test_slice_sample.py::test_constant_trace_in_a_free_parameter_fails_the_diagnostics[0.3, 0.001, log(10), …]`, `::test_fixed_parameter_is_left_out_of_the_diagnostics[0.3]` | `assert exit_flag == -3` → `-1 == -3` (constants 0.3, 1e-3); `assert np.all(np.isnan(R))` → `R = [inf]` (log 10); fixed-parameter test: `assert np.isnan(res["R"][1])` → `R[1] = inf` | 46 passed |
| 5650211 | W6-21 | `__metropolis_step`, its two call sites, the `"metopolis_rnd"`/`"metropolis_pdf"` reads and the three attributes removed; the `metropolis` parametrization of `test_serialized_sampler_continues_stream` and its proposal helper go with them | none needed (ruling); the options docstring never mentioned the keys, so nothing to correct there | — | 40 passed |
| 7aba3fe | W6-22 (my part) | `sample` refuses a `thin` or `burn` that is not a whole number (whole values of any type accepted and coerced); `gammaln` in both smooth-box-t normalizers of `f_min_fill.py` | `test_slice_sample.py::test_sample_sanity_checks`; `test_smoothbox_student_t.py::test_normalizer_survives_a_large_df`, `::test_normalizer_agrees_with_the_gamma_form` | sanity checks: `TypeError: 'float' object cannot be interpreted as an integer` escaped `pytest.raises(ValueError)` (from `for i in range(0, total_N + burn)`); normalizer: `assert np.isfinite(smoothbox_student_t_cdf(0.5, 400, 1.0, -1.0, 1.0))` → `nan` | 40 + 7 passed |
| a1fcd8c | W6-16 | `uuinv` docstring states the length-weighted mixture; the caller's comment says "half of the draws of each coordinate without a prior"; the two NaN lines moved above the branch so all three return paths mark a `p` outside [0, 1] | `test_smoothbox.py::test_uuinv_marks_values_outside_the_unit_interval` | `assert np.all(np.isnan(x[[0, 2]]))` → `[8., 32.]` for `B = [0, 10, 30, 66]`, `w = 1` | 7 + 7 passed. `test_uuinv` is unchanged and now tests the documented specification (said so in the commit message) |
| d649bca | W6-4 | equal targets take a range of one (and the spread of a unit range), with one `warnings.warn`, in the ARD helper, the isotropic helper, `RationalQuadraticARD`, `GaussianNoise` and the mean helper, through one `_target_spread` function | new file `test_bounds_info.py::test_equal_targets_give_usable_bounds[7 components]`, `::test_fit_on_equal_targets_completes` (+ an output-scale test for the rational quadratic, folded into the parametrized one in the next commit) | bounds: `Failed: DID NOT WARN. No warnings of type (UserWarning,) were emitted` (with four `RuntimeWarning: divide by zero encountered in log`); fit: `KeyError: (np.float64(-inf), np.float64(-inf))` from `scipy/optimize/_lbfgsb_py.py:388` | 9 passed; `test_gaussian_process.py` (A's, read-only) 57 passed |
| a5c4ccf | W6-28 | `plausible_upper_bounds[-1] = 5.0` for the shape | `test_covariance_functions.py::test_rational_quad_ard_plausible_upper_bounds` | `assert np.all(np.isfinite(info["PUB"]))` → `[0.657, 0.538, 0.688, 5.0, inf]`, i.e. 5.0 in the output scale's slot and inf in the shape's | 25 passed (with `test_bounds_info.py`, which now includes the kernel in the parametrized check) |
| ca683b0 | W6-29 | the isotropic helper takes `mean(log(width))` for the four length-scale bounds and `mean(log(std(X, axis=0, ddof=1)))` for `x0`; the dead `min_width`/`max_width` lines go | `test_isotropic_covariance_functions.py::test_isotropic_bounds_take_the_means_of_the_logs` | `assert info["LB"][0] == mean_log_width + np.log(tol)` → `-12.3898 == 1.0328 + (-13.8155)` | 12 passed; `test_gaussian_process_isotropic.py` (A's, read-only) 23 passed |
| 5bd1eea | W6-32 | the length-scale gradient of the degree-1 Matern is zero where two inputs share a coordinate, in both the ARD and the isotropic kernel; the comment calling the NaN acceptable replaced | `test_covariance_functions.py::test_matern_kernel_gradient[seed-1]`, `test_isotropic_covariance_functions.py::test_matern_isotropic_kernel_gradient[seed-1]`, `::test_matern_isotropic_against_anisotropic` (drops `equal_nan=True`, asserts finite) | `assert np.all(np.abs(finite_diff - dK) <= eps)` with NaN on the diagonal of `dK` (three seeds each), and `assert np.all(np.isfinite(dK2_iso))` | 40 passed (both kernel files). Probe: a `Matern(1)` and a `MaternIsotropic(1)` GP now give `dnlZ` matching central differences to 6.4e-10 and `fit` completes |
| 5ac2f95 | W6-34 (my part) | the five kernels refuse `compute_diag=True` with `compute_grad=True` (`ValueError`); the squared exponential's diagonal shortcut no longer tests the now-unreachable flag | `test_covariance_functions.py::test_kernel_refuses_a_gradient_of_the_diagonal[3 kernels]`, `test_isotropic_covariance_functions.py::test_isotropic_kernel_refuses_a_gradient_of_the_diagonal[2 kernels]` | `Failed: DID NOT RAISE ValueError` (five parametrizations) | 95 passed (both kernel files plus `test_predict_cross_covariance.py`, read-only). No caller needs the combination: the only production `compute_diag=True` call is `gaussian_process.py:1993`, without a gradient |
| dfd2d84 | W6-35 (my part) | `AbstractKernel.compute` documents the `(N, 1)` diagonal and the new `ValueError`; `GaussianNoise` documents the `np.spacing(1)` nugget of `constant_add=False`; new `docsrc/source/isotropic_covariance_functions.rst` with its toctree entry in `index.rst` | documentation only, no test (ruling) | — | Sphinx build of `docsrc/source` into the scratchpad succeeded, the new page among the built ones, with no warning of its own (the two `numpydoc` underline warnings are pre-existing, from `gaussian_process.py`) |
| 2546892 | orchestrator's addition | seeds the eight unseeded distributional tests of `test_slice_sample.py` | `test_normal` (0), `test_normal_step_out` (1), `test_normal_mixture` (2), `test_exponential` (3), `test_uniform` (4), `test_beta` (5), `test_multivariate_normal` (6), `test_multivariate_t` (7), each `rng=np.random.default_rng(seed)` | — (test-only) | 40 passed. Measured worst deviation against the 0.1 thresholds on those seeds: normal 0.023, mixture 0.007, exponential 0.032, uniform 0.003, beta 0.002, multivariate normal 0.033, multivariate t 0.041. No threshold was touched. `test_multiple_runs`, `test_generator_runs_are_reproducible_…` and `test_serialized_sampler_continues_stream` keep the global stream, which is what they test |

Final per-file runs, one file per process: `test_slice_sample` 40, `test_smoothbox` 7, `test_smoothbox_student_t` 7, `test_covariance_functions` 25, `test_isotropic_covariance_functions` 20, `test_mean_functions` 15, `test_noise_functions` 1, `test_bounds_info` 9 — all passed. A's files were run read-only only: `test_gaussian_process.py` 57 passed (again at the end), `test_gaussian_process_isotropic.py` 23 passed, `test_predict_cross_covariance.py` included in the 95 above. `git diff --stat main..HEAD` touches no file of A's.

## Proposed release notes (for a user of 1.2.1)

Numbers a 1.2.1 user can observe change in the five marked **[moves numbers]**; the rest are refusals, removals or documentation.

- **[moves numbers]** Slice sampling with a burn-in of two or three iterations no longer returns the same point once per requested sample: a coordinate whose burn-in variance estimate is not positive keeps the width it has instead of being floored to zero, and a window of fewer than two iterations adapts nothing (in PyVBMC this is `gp_sample_thin = 1`).
- **[moves numbers]** An infinite entry of `SliceSampler(widths=...)`, which the widths documentation allows for an unbounded coordinate, no longer comes back after the burn-in and fills the chain with NaN.
- **[moves numbers]** The recommended bounds and starting length scale of the isotropic kernels take the means of the logs of the per-dimension widths and standard deviations, as MATLAB's gplite does, where they took the log of the mean width and the standard deviation pooled over all entries of `X`; a fit with `SquaredExponentialIsotropic` or `MaternIsotropic` starts from a different design and feasible set than in 1.2.1.
- **[moves numbers]** `Matern(1)` and `MaternIsotropic(1)` can be fitted: the gradient with respect to the length scale is zero, not NaN, where two inputs coincide, so the gradient of the marginal likelihood is finite.
- **[moves numbers]** A training set whose targets are all equal is given a range of one, with a warning, instead of bounds of `-inf` that ended `GP.fit` with `KeyError: (-inf, -inf)`; such a fit now completes.
- The plausible upper bound of the rational quadratic kernel's shape parameter is its own (5) and the output scale keeps the range of the targets, which changes the space-filling design of a fit with that kernel.
- The convergence diagnostics of `SliceSampler` report a parameter whose recorded chain did not move as undefined — `exit_flag` -3, `R` and `eff_N` NaN, and the "did not move" message — whatever value it is frozen at; before, this held only for a constant whose mean is exact.
- The cumulative distribution and quantile functions of the smooth-box Student's t distribution are finite for degrees of freedom above about 340, where the normalization constant used to overflow to NaN.
- `SliceSampler.sample` refuses a `thin` or `burn` that is not a whole number with a `ValueError` instead of raising `TypeError` from `range`. **Upgrading:** a script that passes a fractional value (for instance `thin=1.5`) must round it; a whole number of any type is still accepted.
- The covariance kernels refuse `compute(compute_diag=True, compute_grad=True)` with a `ValueError`. **Upgrading:** that combination returned the diagonal beside the gradient of the full matrix, a pair that meant nothing; ask for the two separately.
- The undocumented Metropolis step of `SliceSampler` is removed. **Upgrading:** it never ran (its option key was misspelled), and the attributes `metropolis_pdf`, `metropolis_rnd` and `metropolis_flag` no longer exist, so a script that set them must drop them.
- `uuinv` returns NaN for a `p` outside [0, 1] in every case, not only in the general one, and its documentation now states the mixture it implements: the weight outside the plausible box is spread over the two tails in proportion to their lengths.
- The isotropic kernels have a documentation page, and `AbstractKernel.compute` documents the `(N, 1)` shape of the diagonal it returns.

## Found on the way, not in the ledger (mine)

1. **W6-4 needs the standard deviation of the targets, not only their range** (this is the open detail I chose). With the range alone set to one and `np.std(y, ddof=1)` left as it is (~2.5e-16 for equal targets, exactly 0 for some values), the noise's plausible upper bound `log(std(y))` ≈ −36 lies below its lower bound `log(1e-6)` = −13.82; `GP.fit` clips it up to the lower bound, the plausible pair comes out inverted (`PLB` −6.91 > `PUB` −13.82) and the design stops at `uuinv`'s `assert B[0] <= B[1] <= B[2] <= B[3]`. So "the bounds are finite" is not enough for the ruling's second test. I therefore take the spread of a unit range for both the range and the standard deviation where the targets are all equal, and leave every other statistic of `y` (minimum, maximum, median, quantiles) untouched, as the ruling asks.
2. **The same inverted plausible pair is reachable without equal targets** (mine, and not in the ledger): any training set whose target standard deviation is below the tolerance 1e-6 — targets spanning 1e-8, say — gives `PUB_noise = log(std(y)) < LB_noise = log(1e-6)`, the clip in `GP.fit` (`PUB = max(min(PUB, UB), LB)`) raises it above `PLB`, and the design fails the same assertion. The clip is in `gaussian_process.py`, so a repair (order the plausible pair after clipping, or floor the noise's plausible upper bound at its lower bound) belongs to that file; it is worth a row of its own.
3. **W6-15's pairing claim does not hold of the references, so I took the ruling's alternative** (kept the pairing, documented that the pairs start at lag 0 — stated here as asked). BDA3 (11.8) and Stan pair `(rho_0, rho_1), (rho_2, rho_3), …` with `rho_0 = 1`, which is gpyreg's pairing; they differ from gpyreg only in taking the first pair unconditionally and testing from `(rho_2, rho_3)` on. Where that difference bites (`rho_0 + rho_1 <= 0`, i.e. `rho_1 <= -1`) both conventions leave a non-positive integrated time that the same `1 / log10(m n)` floor replaces, so the estimate is identical; pairing from lag 1 instead would change `eff_N` for every chain and would make `eff_N` above the number of draws unreachable, breaking the premise of `test_anticorrelated_trace_has_a_large_but_bounded_effective_n` (that trace would give exactly `m n`). The docstring now says the convention and why it agrees with the references.
4. **Setting `R` and `eff_N` to NaN for a chain that did not move** is my choice within W6-15: it is what the documented contract of both keys already promises, it makes the frozen-chain detection independent of the rounding of the constant's mean, and it is what lets the second of the two ruled tests (`test_fixed_parameter_is_left_out_of_the_diagnostics`, parametrized over constants) fail before the fix and pass after. The `R` docstring sentence was reworded from "not finite for a parameter whose chain is constant within each half" to "NaN for a parameter whose chain did not move", which is what now holds exactly.
5. **W6-3 also changes the width of a fixed coordinate** (`LB == UB`) after the burn-in from 0 to the 1 it is given at construction, since such a coordinate's window variance is zero. The width of a fixed coordinate is never used: the coordinate sweep skips it.
6. **`test_covariance_functions.py` calls `test_simple_rational_quad_ard()` at module level** (line 227 of the file as it stands), so that test body also runs at import. Pre-existing, not touched.
7. **The installed black is 26.5.1 while `.pre-commit-config.yaml` pins black 22.3.0**, and the two differ on two pre-existing spots in `slice_sample.py` (the `self.widths[self.LB == self.UB] = 1` line and one `%`-format continuation). I ran black and isort before every commit and reverted those two untouched-line reformats, so my commits carry only my changes; a run of the repo's own pinned black leaves the files as they are. `pycln` is not installed in this interpreter; I checked by hand that no import became unused (the Metropolis removal left none).
8. `_isotropic_bounds_info_helper` still unpacks `_, D = X.shape` and never uses `D` (pre-existing; left alone, since W6-29 is about the statistics).
9. `SquaredExponential.compute`'s shortcut for the diagonal tested `and not compute_grad`, which the W6-34 refusal makes unreachable; I removed that clause with it rather than leave a condition that can never be false.

## For the orchestrator's W6-1 commit (line numbers moved)

`covariance_functions.py` now has `_target_spread` at the top of the module, so everything below it is about 40 lines further down; the ARD helper's `np.log(np.std(X, ddof=1))` (was `:451`) and the rational quadratic copy's (was `:401`) are **untouched** and still pooled, as are the four reductions of `mean_functions._bounds_info_helper`. What changed in those helpers is only the target-side statistics (`height`, and `np.std(y, ddof=1)` renamed to `y_std`). One coupling: the isotropic helper's `x0` is already per column after W6-29 (`np.mean(np.log(np.std(X, axis=0, ddof=1)))`), so the W6-1 patch must not touch that line again.

## Left undone, and why

- Everything of W6-22, W6-34 and W6-35 that lives in `gaussian_process.py` is A's: the unknown-name and non-positive-`sigma` refusals of `set_priors`/`set_bounds`, the inverted bound pair, `df_base` filling a copy, `nll` starting at `+inf`, the NaN-hyperparameter refusal, the **third** copy of the smooth-box-t normalizer (`__prior_masks`, which still uses `sp.special.gamma` and overflows above a `df` of about 340 — A's `gammaln` edit), `quad`'s guards, `_convert_shapes`, the `update(hyp=...)` width check with the repair of `test_predict_lpd` and its isotropic twin, and the docstrings of `predict`/`predict_full`/`quad` (the `add_noise` default, the noise in the summary and in `lpd`, the unclipped diagonal, the "conputed" typo and the stale `quad` comment). I made my parts and said so in the commit messages.
- W6-1 (the pooled statistics of the ARD covariance and mean bound helpers) is deliberately untouched, as instructed.
- `docsrc/source/release_notes.rst` is untouched; the sentences above are the proposal.

## End state

`git status --porcelain`: empty (clean).

`git log --oneline main..HEAD`:

```
2546892 test: seed the distributional tests of the slice sampler
dfd2d84 docs: the diagonal's shape, the noise nugget, the isotropic kernels' page
5ac2f95 fix: the kernels refuse a gradient of the diagonal
5bd1eea fix: the degree-1 Matern length-scale gradient is zero on the diagonal
ca683b0 fix: the isotropic kernels' bounds take the means of the logs of the widths
a5c4ccf fix: the rational quadratic kernel's shape keeps its own plausible bound
d649bca fix: equal training targets take a range of one, with a warning
a1fcd8c docs: uuinv states the mixture it implements, and marks p out of range
7aba3fe fix: refuse a fractional thin or burn, and fix the smooth box t normalizer
5650211 fix: remove the slice sampler's dead Metropolis step
f420255 fix: a slice sampler chain that did not move is recognized by its range
6bc0c23 fix: an infinite slice sampler width no longer returns after the burn-in
970f6c6 fix: a short burn-in no longer stops the slice sampler's coordinates
```

Files changed against `main`: `docsrc/source/index.rst`, `docsrc/source/isotropic_covariance_functions.rst` (new), `gpyreg/covariance_functions.py`, `gpyreg/f_min_fill.py`, `gpyreg/isotropic_covariance_functions.py`, `gpyreg/mean_functions.py`, `gpyreg/noise_functions.py`, `gpyreg/slice_sample.py`, `gpyreg/testing/test_bounds_info.py` (new), `test_covariance_functions.py`, `test_isotropic_covariance_functions.py`, `test_slice_sample.py`, `test_smoothbox.py`, `test_smoothbox_student_t.py`.
