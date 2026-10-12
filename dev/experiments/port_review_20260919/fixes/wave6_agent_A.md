# Wave 6 fix pass: gpyreg fix agent A

Raw report of fix agent A, wave 6 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-22. One Opus agent on a
worktree of gpyreg made by hand (`../gpyreg-port-review-A`, branch
`port-review-wave6-A`, cut from gpyreg's `main` at `fdbafdf`), made the
fixes of the rows of `../verification/wave6.md` that live in
`gaussian_process.py`, one commit per row as the PI ruled them, and, on the
orchestrator's instruction, the row W6-38 that agent B found on the way,
which the PI had not ruled on when it was made; the orchestrator reviewed
the diffs and cherry-picked the commits onto gpyreg's `port-review-wave6`,
where they have other hashes (`../verification/wave6.md`, "Fix commits").
The text below the rule is the agent's final message, unedited.

---

# Wave 6 fix pass: report of gpyreg fix agent A

## Setup

```
$ python -c "import gpyreg; print(gpyreg.__file__)"      # from the worktree root
C:\Users\luigi\Documents\GitHub\gpyreg-port-review-A\gpyreg\__init__.py

$ git log -1            # at the start
commit fdbafdfc73b0e97270a92f3da4682fa857323c20
Author: dependabot[bot]
Date:   Mon Sep 21 15:05:13 2026 +0300
    chore(deps): bump pypa/gh-action-pypi-publish from 1.13.0 to 1.14.2 (#42)
```

Every command ran from the worktree root with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`. The import line above was re-checked at the end and is unchanged.

**Formatting note.** The installed black is 26.5.1 while gpyreg's pre-commit pins 22.3.0, and the newer one reformats three lines of `gaussian_process.py` that this branch does not touch. I ran black and isort before every commit and then reverted that unrelated churn, so the committed file keeps the form the pinned hook produces; one such hunk (the `nu_k` expression in `quad`) still shows under `black --diff` with the installed version, as expected.

## Commits, in order

| hash | row | what it does | test | failing assertion seen before the fix | test-file run after |
|---|---|---|---|---|---|
| `829bb50` | W6-26 | `predict_full(add_noise=True)` adds `np.diag(ravel(sn2_star)) * sn2_mult` (a scalar noise broadcast to one entry per point) | `test_gaussian_process.py::test_predict_full_add_noise_per_point` | `assert np.allclose(C, C.T, rtol=0, atol=1e-14)` → False: the added term was the `(M,1)` column replicated across every column | 58 passed |
| `8fc46b2` | W6-27 | `quad`'s variance reads `Posterior.sl` (with `update`'s `getattr`/`sW` fallback) instead of recomputing `min(sn2)*sn2_mult` | `test_gaussian_process.py::test_rank_one_update_with_heteroskedastic_noise[provided_without_s2_new]` | `assert np.allclose(F_var, F_var_ref, rtol=1e-8, atol=1e-12)` → `2.220446e-16` against `0.11647096` | 58 passed |
| `f79a3b2` | W6-33 | the `if L is None: raise LinAlgError` guard runs after the retries of both branches, before either forms `pL` | `test_gaussian_process.py::test_failed_factorization_raises_linalgerror` | `TypeError: bad operand type for unary -: 'NoneType'` in place of the `LinAlgError` (`pytest.raises(LinAlgError)` failed) | 59 passed |
| `e16cd12` | W6-7 | `__robust_cholesky` flips eigenvector signs per column and uses `sp.linalg.eigh` (the `np.real` calls go) | `test_gaussian_process.py::test_robust_cholesky_factors_matrices_cholesky_refuses` | `assert np.allclose(T.T @ T, sigma, atol=1e-10*max|sigma|)` → False on the rank-deficient PSD matrix | 60 passed |
| `88f40f2` | W6-8 | a surviving negative eigenvalue within a relative rounding band (`10 * n * eps * max|D|`) counts as zero; beyond it, `LinAlgError` with the value. MATLAB's tolerance still decides which eigenvalues carry the matrix | `test_random_function_on_a_dense_grid`, `test_robust_cholesky_refuses_an_indefinite_matrix` | `assert not np.array_equal(f_1, f_2)` → the two draws were bit-identical (and equal to `predict`'s mean); `Failed: DID NOT RAISE LinAlgError` | 62 passed |
| `79b05b9` | W6-9 | the low-noise rank-one branch warns and recomputes in full where `v_star <= sn2_eff` | `test_rank_one_update_low_noise_duplicate_recomputes` (and `test_rank_one_update_low_noise_branch`) | `Failed: DID NOT WARN. No warnings of type (UserWarning,) were emitted` (and the predictive variance was 1.6e-3 from a rebuild) | 63 passed |
| `16e74dd` | W6-13 | the rank-one path also asks `posteriors[0].alpha is not None` | `test_single_point_update_without_posterior_factors[cleaned]` and `[no_posterior]` | `TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'` | 66 passed |
| `61db728` | W6-5 | `((df == 0) | ~np.isfinite(df))` in both masks; the "is set to inf later" comments corrected | `test_log_prior_matches_the_documented_densities` | `assert np.isclose(log_prior, expected, rtol=1e-12)` → `-2.3719788` against `-4.8101201` with `df = inf` | 67 passed |
| `d39937f` | W6-6 | `C_sb`/`C_sb_t` subscripted with the same mask as their `sigma`; `get_priors` uses `np.all` | `test_smooth_box_prior_over_a_block[2|3-smoothbox|smoothbox_student_t]` | `-5.8021735` against `-2.9010867` (D=2, exactly twice) and `ValueError: operands could not be broadcast together with shapes (3,) (2,)` (D=3) | 71 passed |
| `1f879ec` | W6-17 + W6-22 | input checks: a prior's `sigma` finite and positive (message names `None`); unknown names in `set_priors`/`set_bounds`; `get_recommended_bounds` takes any array_like, names `upper_bounds`, refuses an inverted user pair; `update` refuses unset hyperparameters, naming them; `fit` fills `df_base` into a copy | five tests (`…refuses_a_scale_that_is_not_positive`, `…refuse_an_unknown_hyperparameter`, `test_get_recommended_bounds_input_checks`, `test_update_without_hyperparameters_raises`, `test_fit_leaves_the_prior_degrees_of_freedom_alone`) | eight × `Failed: DID NOT RAISE ValueError`, one `TypeError: only integer scalar arrays can be converted to a scalar index` (tuple bounds), and `array_equal([7.,7.,7.,7.], [nan,nan,nan,nan])` → False | 80 passed |
| `60ddd1d` | W6-22 (rest) | `gammaln` in the smooth-box-t normalizer; `nll` starts at `+inf` with a comment | `test_smooth_box_student_t_prior_with_many_degrees_of_freedom[400.0]` | `assert np.isfinite(nan)` (`df = 5` passed already) | 82 passed |
| `9a6ba38` | W6-10 | `sampler_name` read first, `sampler` second; the dead `"laplace"` clause goes | `test_fit_reads_the_documented_sampler_option[sampler_name]` | `Failed: DID NOT RAISE ValueError` (the `[sampler]` case passed already) | 84 passed |
| `b83b025` | W6-11 | `hyperparameters_from_dict(hyp)[0]` in both public wrappers | `test_log_likelihood_and_posterior_take_a_dictionary` | `ValueError: Expected 5 mean function hyperparameters, 0 passed instead.` | 85 passed |
| `7bbca70` | W6-12 | `hyp = X0[0 : max(opts_N, 1), :].copy()` | `test_fit_does_not_write_into_the_space_filling_design` | `assert np.array_equal(records["design"], records["as_returned"])` → False (row 1 of the design overwritten) | 86 passed |
| `b233eb8` | W6-18 | the scalar-`sn2` noise gradient takes `np.atleast_2d(dsn2)[0, i]` | `test_noise_gradient_with_a_constant_total_noise` | `ValueError: can only convert an array of size 1 to a Python scalar` | 87 passed |
| `6a1633f` | W6-34 | `quad` refuses a GP without data, without factors and with an unsupported mean, and reads a 1-D measure as one measure of `D` dimensions (documented); `_convert_shapes` takes any number and 0-d arrays, checks the row count of `s2`, raises `ValueError` with the space restored; `update(hyp=)` checks the row width; `test_predict_lpd` and its isotropic twin repaired | `test_quad_input_checks`, `test_convert_shapes_input_checks`, `test_update_checks_the_hyperparameter_width` | `AttributeError: 'NoneType' object has no attribute 'shape'` (quad without data), `TypeError: s2 type need to be …` for `np.int64(3)`, `Failed: DID NOT RAISE ValueError` (the width) | 163 passed over my four files |
| `744655b` | W6-35 (+ W6-31 docstring) | documentation only: `predict_full`'s `add_noise` default and unclamped diagonal; `predict`'s summary and `s2`; the log density always carrying the noise and being the moment-matched Gaussian's; `log_posterior`'s renormalization; "conputed"; the stale `gplite_quad` comment | none (documentation) | — | 164 passed |
| `595d907` | test notes | `test_split_update` seeded (`default_rng(17)`, fixed `N_s`, generator passed to `random_function`); `test_fitting_options` asserts the row count, the two result objects, the sampler's shape and the stored posteriors for each of its eight option sets | the two tests themselves | — (they asserted nothing before) | 2 passed |
| `b312417` | W6-38 | after the two clips in `fit`, an inverted plausible pair collapses onto its clipped upper bound | `test_fit_with_targets_of_a_tiny_range` | `assert B[0] <= B[1] <= B[2] <= B[3]` — the `AssertionError` from `uuinv` | 164 passed |

Final run of my four files together (`test_gaussian_process.py`, `test_gaussian_process_isotropic.py`, `test_predict_cross_covariance.py`, `test_utils.py`): **164 passed**.

### Choices where a ruling left a detail open

- **W6-8, the tolerance.** I first used one tolerance, `10 * n * eps * max|D|`, for both "which eigenvalues carry the matrix" and "which negative eigenvalue is rounding noise". That widened MATLAB's drop tolerance by up to twenty times and so moved every `random_function` draw that takes the fallback — and it made the isotropic `test_fitting` fail 2 of 6 runs (0 of 8 on the base revision). The committed version therefore keeps MATLAB's `|spacing(max(D))| * n` for the drop decision, so the factor of every draw that works today is unchanged, and adds the relative band only for a *surviving negative* eigenvalue. The commit `88f40f2` was amended and the eleven later commits replayed onto it, so no fixup commit is on the branch.
- **W6-8, the factor of ten.** The band is `10 * n * eps * max|D|`, documented at the line as the backward-error scale of the symmetric solver with room for the constant that bound hides. In the ledger's configuration `n * eps * max|D|` alone would leave the worst negative at 0.96 of the tolerance — too close to call.
- **W6-34, `mu` of length `D`.** Taken as one measure of `D` dimensions (`np.atleast_2d`), as `gplite_quad.m:26` reads a row vector, and documented in the `mu`/`sigma` entries. A measure whose width is not `D` raises.
- **W6-17, `get_recommended_bounds`.** `np.array(x, dtype=float)` rather than `np.asarray(...)`: it takes any array_like *and* copies, which the function needs — it fills NaNs through views of the argument.
- **W6-17, the inverted pair.** The check applies to the bounds the *call* carries, before the NaN fill; a recommended pair that comes out inverted on a degenerate training set is still collapsed by `ub = np.maximum(lb, ub)`, as `gplite_train.m:142` collapses it. Raising there would have turned W6-4's case into an error.
- **W6-22, `fit` and `df_base`.** The objectives read `self.hyper_priors`, so a local copy is not enough: `fit` installs the filled copy and restores the caller's array (and recomputes the normalization constants) in a `finally`, which meant indenting the body of `fit`. `git diff -w` shows only the intended lines.
- **W6-22, the `gammaln` site.** The row names `__recompute_normalization_constants`; the `sp.special.gamma` pair in my file is in `__prior_masks` (`C_sb_t`), which is the smooth-box-t normalizer the row's evidence (G1-28) cites. That is the one I changed.
- **W6-38, the message.** `PLB[inverted] = PUB[inverted]`, i.e. both become the clipped `PUB`, which the clip left inside `[LB, UB]`.

### `test_predict_lpd` and its isotropic twin: before and after

Before, both files asserted: `lpd == norm.logpdf(y_star, loc=f_mu, scale=sqrt(np.pi * s2_star + f_s2))` — vacuous, because `s2_star = np.arange(-3, 3)` was overwritten by `np.zeros((6, 1))` two lines later, so the total noise at the test points was `np.spacing(1)` and the factor `np.pi` never mattered; `lpd2 == lpd` for `add_noise=True`; and `lpd3[:, 0:1] == lpd`, `lpd3[:, 1:2] == lpd`, which held only because the two hyperparameter rows were identical and the noise was `eps`. Each row carried 12 entries for a GP of 11 (of 9, the twin), so `predict` read `log(pi)` as the constant of the mean and never read the last entry (the last three, in the twin).

After: a row of the right width (11 and 9; this noise object has no hyperparameter of its own), two rows that differ, `s2_star = np.linspace(0.1, 0.7, 6)`, and the assertions are `lpd == norm.logpdf(y_star, loc=f_mu, scale=sqrt(s2_star + f_s2))`; `s2` with `add_noise=True` equal to `f_s2 + s2_star`; `lpd2 == lpd`; per sample `lpd3 == norm.logpdf(y_star, loc=f_mu_s, scale=sqrt(y_s2_s))` with the two columns *not* equal; and, averaged, `lpd` equal to the density of the Gaussian carrying the mixture's mean and variance (`mean(y_s2) + var(mu, ddof=1)`) and *not* to the average of the per-sample densities.

## Release notes, one sentence per user-visible change (for a user of 1.2.1)

Numbers a 1.2.1 user can observe are marked **[moves]**.

- **[moves, at gpyreg's defaults]** `GP.fit` takes its starting points as a copy of the space-filling design, so the default widths of the slice sampler are the standard deviation of the design as it was drawn; a script that calls `GP.fit` at the default `opts_N = 3` will see its sampled hyperparameters move.
- **[moves, in the case it repairs]** `GP.quad(compute_var=True)` normalizes by the noise scale the stored Cholesky factor carries, so the variance of an integral is right after a rank-one `GP.update` that appended a point of lower total noise, where it used to be clamped to machine epsilon.
- **[moves, in the cases they repair]** `GP.random_function` now draws a sample where the predictive covariance is numerically singular, as it is on a dense one-dimensional grid: eigenvalues that are negative but of rounding size count as the zeros they are, where the draw used to collapse onto the predictive mean without a word, and a negative eigenvalue beyond the rounding band raises `LinAlgError` instead. The eigenvalue fallback also fixes the signs of whole eigenvectors and uses the symmetric eigensolver, so the factor it builds is a factor of the matrix it was given; a draw at closely spaced or duplicated test points came from the wrong covariance before. Draws through this path change for a given generator whether or not the old factor happened to be right.
- **[moves, in the case it repairs]** `GP.predict_full(add_noise=True)` adds the observation noise on the diagonal; with a noise that varies from point to point the returned matrix was neither symmetric nor a covariance matrix, while its diagonal was already right, so a constant noise function is unaffected.
- **[moves, in the cases they repair]** A hyperprior whose degrees of freedom are infinite or NaN is the Gaussian family it names, which is MATLAB's `HPRIOR.nu = Inf` convention, where it used to contribute no prior at all; and a smooth-box prior set on a block of several hyperparameters has one normalization constant per coordinate of the block, where it doubled the log density with two coordinates on different sides of the box and raised the shapes with three. `GP.get_priors` returns such a prior instead of raising.
- **[moves, in the case it repairs]** A single-observation `GP.update` of a posterior in the low-noise representation falls through to a full recomputation, with a warning, where the predictive variance of the new point is at or below what the variance clamp can produce — an observation at an existing training input, for one — as the Cholesky representation already did for its own stability test.
- `GP.fit` no longer leaves the plausible bounds inverted where the plausible box lies outside the hard box, which made the space-filling design fail an internal assertion for training targets whose standard deviation falls below the lower bound of the noise.
- The normalization constant of a smooth-box Student's t prior is taken through the logarithms of the gamma functions, so a prior with more than about 340 degrees of freedom no longer makes the log posterior NaN.
- `GP.log_likelihood` and `GP.log_posterior` accept the dictionary of hyperparameters their docstrings document.
- `GP.fit` reads the sampler under `sampler_name`, the name it documents, as well as under `sampler`, and fills `df_base` into a copy of the priors, so a fitted GP keeps the priors the caller set and a second fit with another value uses it.
- A single-observation `GP.update` on a GP that carries no posterior factors — after `GP.clean`, or after an update with `compute_posterior=False` — recomputes the posterior in full instead of raising `TypeError`.
- Inputs that were taken in silence are refused with a message: a prior needs a finite, positive `sigma` (no prior is written as `None`, not as an infinite scale); `set_priors` and `set_bounds` reject a hyperparameter name the model has not; `get_recommended_bounds` takes any array_like and rejects a bound pair given inverted; `GP.update(hyp=...)` checks the width of the hyperparameter row, and `GP.update` with `compute_posterior=True` refuses hyperparameters that were never set, naming them; `GP.quad` refuses a GP without training data or posterior factors, a mean function it cannot place, and a measure that has not one column per input dimension; and a noise variance may be any number or 0-d array while an array whose row count is not that of the inputs is refused. A failed Cholesky decomposition reports `LinAlgError` in both noise representations, where the low-noise one raised `TypeError`.
- The gradient of the marginal likelihood no longer raises where the noise function carries a scale for a user-provided variance that is never given.
- Documentation: the default of `predict_full`'s `add_noise`, that its diagonal is not clamped and can be negative on a nearly singular posterior, which variance `predict` returns by default, that the log predictive density always carries the observation noise and over several hyperparameter samples is the density of the moment-matched Gaussian, and that `log_posterior` renormalizes each prior over its bounds.

## Found on the way, not in the ledger (mine)

1. **W6-38's mechanism is not the one the message describes.** The inversion needs `std(y) < 1e-6 < max(y) - min(y)`: the noise's hard upper bound `log(max(y)-min(y))` must stay *above* its hard lower bound `log(1e-6)`, so that the plausible *lower* bound is clipped down to `UB` while `PUB = log(std(y))` stays below it. With targets spanning 1e-8 the hard pair collapses in `get_recommended_bounds` (`ub = np.maximum(lb, ub)`) and the plausible pair collapses with it, so nothing inverts and the fit completes; twenty targets spanning **1e-4** do invert, and that is what my test uses. Skewed sets (nineteen equal values and one outlier 2e-6 to 1e-5 away) invert too.
2. **`test_fitting` is an unseeded statistical test in both of my test files** (`|hyp - hyp2| < 0.5` on data drawn from NumPy's global stream through `random_function`), and it flakes on the base revision: running the two files in one process, `fdbafdf` failed 2 of 6 runs (once each test) and my HEAD 1 of 6. The ledger's test notes do not list it. It is also the test that caught my first W6-8 version, so any later change to `random_function`'s draws will disturb it; seeding it is a test note worth adding to a later wave (I left it alone, being outside my rows).
3. **A relative tolerance against the largest eigenvalue does not always absorb the rounding-size negatives.** On a grid that stays inside the training range the predictive covariance can have a largest eigenvalue of 1e-6 while its negative eigenvalues are 1e-15, so no band of the order `n * eps * max|D|` reaches them; the ledger's configuration (a grid extending past the data, largest eigenvalue 0.163) is absorbed. W6-8 repairs the case the ledger names, not every near-singular covariance whose scale collapses; such a matrix now raises with its smallest eigenvalue in the message instead of returning the mean in silence.
4. **W6-9's guard fires only where the clamp bit.** With `GaussianNoise(constant_add=False)` and a duplicated input at `log sf = 3`, the latent variance comes out 5.7e-14 > 0, so `v_star > sn2_eff` and the rank-one extension proceeds although `alpha` is 74% from a rebuild. The ruling's condition is the analogue of the Cholesky branch's `sqrt_arg <= 0`, which is equally silent when `sqrt_arg` is positive but tiny; the low-noise branch stays inaccurate in that band.
5. **`_convert_shapes`' type annotation** still reads `s2: Union[np.ndarray, float, int, None]` while the body now takes any `numbers.Number` and a 0-d array. I left the annotation as it stands; widening it is a one-line follow-up if the PI wants it exact.

## Left undone, and what belongs to agent B

- Nothing from my list, including the optional test notes (item 18).
- **W6-35, B's part**: `AbstractKernel.compute`'s docstring says `(N,)` for the diagonal where every kernel returns `(N, 1)` (`covariance_functions.py`), the `np.spacing(1)` nugget of `GaussianNoise(constant_add=False)` is undocumented (`noise_functions.py`), and the isotropic kernels have no `docsrc` page. My commit covers the `gaussian_process.py` texts alone.
- **W6-22, B's part**: the `sp.special.gamma` pair of `f_min_fill.py:307` and `:370`.
- **W6-17, B's part**: `f_min_fill.py:119` carries the same `~isfinite(mu) & ~isfinite(sigma)` mask as `__prior_masks`. Refusing a non-finite `sigma` in `set_priors` closes the way in through the public interface, so the design can no longer be handed one; the mask itself is unchanged, and a matching guard there is B's call.
- **W6-34, B's part**: `compute(compute_diag=True, compute_grad=True)` returning a meaningless pair (G2-21) is in `covariance_functions.py`.

## Final state

```
$ git status --porcelain
(clean)

$ git log --oneline main..HEAD
b312417 fix: the plausible bounds of a fit are not left inverted by the clip
595d907 test: seed test_split_update and assert in test_fitting_options
744655b docs: what predict, predict_full, quad and log_posterior return
6a1633f fix: input checks of quad, of the converted shapes and of update
b233eb8 fix: the noise gradient where the total noise is a scalar
7bbca70 fix: fit takes its starting points as a copy of the design
b83b025 fix: log_likelihood and log_posterior take the documented dictionary
9a6ba38 fix: fit reads the sampler option under its documented name
60ddd1d fix: the normalizer of a smooth-box Student's t prior at a large df
1f879ec fix: input checks of the bounds, the priors, update and fit
d39937f fix: a smooth-box hyperprior over a block of hyperparameters
61db728 fix: a Gaussian hyperprior with an infinite df is a Gaussian hyperprior
16e74dd fix: the rank-one update asks the posteriors for their factors
79b05b9 fix: the low-noise rank-one update reverts to a full recomputation
88f40f2 fix: a rounding-size negative eigenvalue is a zero, a larger one an error
e16cd12 fix: the eigenvalue fallback of the Cholesky decomposition factors its input
f79a3b2 fix: report a failed factorization as a LinAlgError in both branches
8fc46b2 fix: quad normalizes by the noise scale its factor carries
829bb50 fix: predict_full adds the observation noise on the diagonal
```

Nothing was pushed, no file outside the worktree was changed (probe and patch scripts live in the scratchpad directory), and no sub-agent was started.
