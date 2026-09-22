# Wave 6, the fix round of the check of the fix round: gpyreg fix agent E

Raw report of fix agent E, wave 6 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-22. One Opus agent on a
worktree of gpyreg made by hand (`../gpyreg-port-review-E`, branch
`port-review-wave6-E`, cut from `d93f03c`, the head of `port-review-wave6`
after fix agent D's round), made the fixes in gpyreg that the PI ruled on
after the independent check of D's round (the check's raw reports are on
the orchestrator's machine, `dev/scripts/runs/LOCAL.md`, "Port correctness
review"; `../verification/wave6.md`, "The independent check of the fix
round"). The orchestrator reviewed the diffs before the commits were taken
onto `port-review-wave6`. The text below the rule is the agent's final
message, unedited.

---

# Fix agent E report

E1, E2 and E3 are done as three commits on `port-review-wave6-E` in `C:/Users/luigi/Documents/GitHub/gpyreg-port-review-E`, on top of `d93f03c`. Nothing is pushed, and no other checkout was modified: `gpyreg`, `-port-review`, `-A`, `-B` and `-D` are clean at their heads, and PyVBMC was only read. All runs used single-threaded BLAS.

## Commits

| hash | item | what changed | test seen to fail, and how | tests run, result |
|---|---|---|---|---|
| 5db2255 | E1 (F1-1) | **`random_function`, low-noise posterior:** it recomputes `K` and `sn2` at the training inputs with the sampled posterior's `hyp`. It factors `K + sn2_mult*diag(sn2)`, starting from the posterior's `sn2_mult`, then forms `V = L^-T Ks` and `C = K_star - V'V`.<br>**Shared factorization:** both Cholesky retry ladders, with their `LinAlgError`, move verbatim from `__core_computation` into a new static helper `__training_cholesky(K, sn2, L_chol, sn2_mult=1)` that both methods call.<br>**Raises:** the entry now names two cases: an eigenvalue beyond the rounding of the prior variance, and, for a low-noise posterior, a failed factorization after the retries.<br>**Tests:** the dense-grid test is parametrized over SD 2e-3 (asserts `L_chol`), 1e-4 and 1e-6 (assert `not L_chol`), each on a grid inside the data (`[-1,1]`) and one reaching past it (`[-2.5,2.5]`). It checks against a covariance computed in the test by `cho_solve`. A new test, `test_random_function_draws_from_one_hyperparameter_sample`, covers two samples with and without `add_noise` in both representations. | On `d93f03c` all 5 low-noise cases raised `LinAlgError`, and the 3 Cholesky-representation cases passed. Messages: "smallest eigenvalue is -2.68e-06 / -0.0180 / -5.43e-06 / -0.0183 / -0.00151, beyond the rounding tolerance 3.2e-13 / 6.7e-14". | `test_gaussian_process.py`: 111 passed. |
| 2938581 | E2 (F1-5, F1-7), docs | **`set_priors` docstring and the W6-5 test comment:** a Student's t read as a Gaussian is attributed to `gplite_hypprior.m`. A smooth-box Student's t read as a smooth box is stated as gpyreg's own, since gplite has no smooth-box priors.<br>**`f_min_fill` comment:** the weight on the plausible intervals of the `m` coordinates that go through `uuinv` is `0.5 ** (m / n_vars)`. It is one half only when all of them go through it. The comment also says what a fixed coordinate, one with an infinite bound and one with a prior take. | n/a | same file: 111 passed at the final head. `f_min_fill` has no test file of its own; `fit` in `test_gaussian_process.py` exercises it. |
| dd12db3 | E3, release notes | **`random_function` point:** both representations draw. It defines the low-noise representation and says what 1.2.1 did there (draws often came out as the mean), how the covariance is now formed, what still raises, and that low-noise draws change for a given generator.<br>**Inputs point:** it separates inputs 1.2.1 took in silence from those it failed on with an internal error (`quad` without data or factors, `quad` measures with the wrong column count, `thin`/`burn`). The noise-variance row-count clause is exact about 1.2.1. It says that `SliceSampler.sample` takes a whole `thin`/`burn` of any type, where 1.2.1 took integers only.<br>**Upgrading list:** the fractional `thin`/`burn` entry is gone. The `update`-on-unset entry leads with `gp.fit(X, y)`, and says 1.2.1's `fit()` found hyperparameters only with `n_samples=0` and a space-filling design, which is also where `compute_posterior=False` works. A new line covers the shape checks' `AssertionError` becoming `ValueError`.<br>**W6-1 and isotropic points:** both now say that nothing changes with one input dimension. | n/a | docutils: no warnings, 26 points, as before. |

I also ran `test_gaussian_process_isotropic.py` (untouched, but it exercises `random_function`): 23 passed.

**Formatting:** black 26.5.1 flags only the `quad` hunk that was already there. isort is clean, no added line exceeds 79 columns, and every file keeps its CRLF endings.

**Commit messages:** Co-Authored-By on all three, no Claude-Session trailer, no line over 72.

## Choices

- **Refactor, results bit-identical.** `__core_computation` now calls the extracted helper. I compared 51 arrays between `d93f03c` and the fix:
  - the posterior's `alpha`, `L`, `sW`, `sn2_mult` and `sl`, `nlZ` with and without its gradient, and predictions;
  - over constant noise at SD 1e-2 and 1e-5, no constant noise, a user-provided `s2` and rectified output-dependent noise;
  - plus a seeded `fit` with 10 samples.

  All 51 are identical.
- **Low-noise factor.** It is the unscaled factor that the low-noise branch builds, and the covariance uses the Gram form `V'V`. For a freshly computed posterior the first attempt reproduces the posterior's own factorization. The retry raises the multiplier only if that fails, which is possible only after rank-one updates have extended the posterior. `add_noise` still uses the posterior's `sn2_mult`.
- **Cost:** a low-noise draw now does one O(N³) factorization.
- **Stability on F1's setups.** The smallest eigenvalue of `C` is between -3e-15 and -6e-15 against a band of 3.2e-13, including a grid on `[-3.5,3.5]`.
- **Why not F1's `cho_solve` reference.** It reaches -2.2e-12 on the grid reaching past the data at SD 1e-6, which is beyond the band.
- **Dense-grid SD moved from 1e-3 to 2e-3** for the Cholesky cases. The variance is then 4e-6, well clear of the switch, instead of two ulps above it. Both premises still hold: a direct Cholesky refuses the covariance, and the band before `97d2b1a` would still refuse the inside grid (1.25e-17 against -5.8e-15).
- **Dense-grid assertions.**
  - Dropped `not np.allclose(f_1, mu)`: at SD 1e-6 inside the data, draws deviate by about 5e-6, which is within `allclose`'s default tolerance. The covariance check implies it anyway.
  - The mean tolerance is now 0.15 times the largest predictive SD, which is tighter than 0.05 on the inside grid.
  - Margins: norm ratios 0.026–0.079 against 0.2; mean deviation over largest SD 0.022–0.058 against 0.15.
- **Several-samples test.**
  - How it works: it classifies each draw's sample by its noise, since the two variances are at least 100 times apart. It then checks each class's noise variance, covariance and mean against that sample.
  - Class sizes are 475 and 525. Noise ratios are 0.98–0.99 against ±0.1. Covariance ratios are 0.025–0.10, against 0.44–262 when compared with the other sample.
  - The largest |log ratio| of noise variance is 0.88, against a threshold of log 5.
- **E2** covers only the sites named in the item (see below for the others).
- **Scope of the new Upgrading line.** Its lead names only the shape-check exception, as ruled. I kept "a row of `N` is given as a column" unchanged.
- **Release-note claims checked by probe on the pin and on the head:**
  - `thin`/`burn` of every type;
  - `quad` without data (1.2.1: `AttributeError`) and without factors (`TypeError`);
  - `quad` measures with the wrong columns (1.2.1: `IndexError` or broadcast `ValueError`, never silent);
  - an `s2` of 11 rows (1.2.1: reshape `ValueError`) and a row of 12 (accepted by 1.2.1);
  - the shape checks (`AssertionError` in 1.2.1);
  - update-then-fit (defaults fail in 1.2.1; `n_samples=0` works there and with `compute_posterior=False` on the head; `fit(X, y)` works on both);
  - at D=1, `get_recommended_bounds` and `get_bounds_info` are bit-identical between pin and head for SE and SE-isotropic, and they differ at D=2.

## Found on the way

- **`predict_full`'s low-noise branch (not changed)** forms the same explicit-inverse product, `K_star + Ks' L Ks`.
  - Setups: the dense-grid data (40 points of sin 2x on [-2,2], ell 0.7, sf 1.2). Test sets: 100-point grids on [-2.5,2.5], [-1.8,1.8] and [-1,1], 5 points in [-1,1], and the training inputs.
  - Baseline: at SD 2e-3 (Cholesky representation) it matches the stable covariance to 8e-13, with smallest eigenvalue -3.7e-15.

  | noise SD | smallest eigenvalue (stable: -1.6e-15 to -5.8e-15, or positive) | max \|C − C_stable\| | largest variance of C_stable |
  |---|---|---|---|
  | 9.9e-4 | -3.4e-11 to -9.7e-11 | 1.2e-9 to 2.8e-9 | 7.6e-7 to 5.8e-2 |
  | 1e-4 | -1.9e-7 to -5.4e-6 | 1.1e-7 to 2.1e-7 | 1.0e-8 to 9.5e-3 |
  | 1e-5 | -2.3e-5 to -4.5e-4 | 1.1e-5 to 2.7e-5 | 1.0e-10 to 2.6e-3 |
  | 1e-6 | -2.1e-3 to -1.8e-2 | 9.3e-4 to 1.9e-3 | 1.0e-12 to 6.8e-4 |

  For reference, the band is 3.2e-13 at 100 points, 1.3e-13 at 40 and 1.6e-14 at 5.
- **`predict` carries the same error in its variances.** It uses the same product, clamped at zero. The maximum |s2 − s2_stable| reaches 1.9e-3 at SD 1e-6, where the true variances inside the data are about 1e-12 (1.6e-3 against 1e-12 at the training inputs). At SD 9.9e-4 it is 2.8e-9 against about 1e-6.
- **Where else this reaches, and where it comes from.** `update`'s low-noise rank-one path divides by `v_star` from `predict` (`gaussian_process.py:959`); I did not measure the effect there. `gplite_pred.m:102-104` has the same formula, so the port is faithful here.
- **PyVBMC is not reached.** It bounds the GP noise SD below at `tol_gp_noise = sqrt(1e-5)` (`gaussian_process_train.py:377`), so `sn2 >= 1e-5` and its GPs stay in the Cholesky representation unless a user lowers that option.
- **`fit` with a whole float `thin`** (for example 2.0) still raises `TypeError` on the head. The cause is `eff_s_N = s_N * thin` and the slice `thin - 1 :: thin`; `burn=3.0` works. `fit` documents `thin` as an int, and the note speaks of `SliceSampler.sample` only.
- **Smooth-box readings still attributed to gplite, outside E2's named sites:**
  - `gaussian_process.py:655`, the code comment in `set_priors`: "no prior … as `gplite_hypprior.m` reads it", which covers the smooth-box location `[a, b]`.
  - The docstrings of `test_set_priors_takes_a_coordinate_without_a_prior` (`test_gaussian_process.py:2563`) and `test_get_priors_returns_what_set_priors_reads_back` (`:2604`, a NaN df "names the Gaussian family … as in gplite_hypprior.m" in a test parametrized with `smoothbox_student_t`).
  - Also, gplite reads any non-finite `mu` or `sigma` as no prior (`gplite_hypprior.m:35`), a superset of gpyreg's rule that both must be NaN.
- **Other exception-type changes with no Upgrading line** (same reasoning as F1-8; not ruled on):
  - `thin`/`burn`: `TypeError` → `ValueError`;
  - `quad` without data or factors: `AttributeError`/`TypeError` → `ValueError`;
  - `quad` measure columns: `IndexError` → `ValueError`;
  - a failed low-noise Cholesky: `TypeError` → `LinAlgError`.

Probes and logs are in `C:/Users/luigi/AppData/Local/Temp/claude/C--Users-luigi-Documents-GitHub-pyvbmc/3d072054-989b-4149-86b7-253c1fc64af1/scratchpad/doublecheck_w6_round/fix_E/`: `e1_probe1.py`, `e1_probe2.py`, `e1_bitid.py`, `e1_margins.py`, `e1_margins2.py`, `e1_predict_full.py`, `e3_notes.py` and `e3_quadcols.py`, with the logs under `logs/`.

## Final state

```
$ git -C C:/Users/luigi/Documents/GitHub/gpyreg-port-review-E log --oneline d93f03c..HEAD
dd12db3 docs: the 1.3.0 notes on low-noise draws, refusals and upgrades
2938581 docs: the NaN df reading gplite shares, and the weight of the box
5db2255 fix: random_function draws from a low-noise posterior

$ git -C C:/Users/luigi/Documents/GitHub/gpyreg-port-review-E status --short
(empty)
```
