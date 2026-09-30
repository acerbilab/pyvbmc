# Wave 5 fix pass: fix agent A

Raw report of fix agent A, wave 5 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-21. One Opus agent on its
own git worktree, cut at `a2104e6`, made the fixes of the rows of
`../verification/wave5.md` on the variational posterior and `kl_div_mvn`, one commit per
row as the PI ruled them; the orchestrator reviewed the diffs and
cherry-picked the commits onto `dev-port-review`, where they have other
hashes (`../verification/wave5.md`, "Fix commits"). The text below the rule
is the agent's final message, unedited.

---

# Wave 5 fix pass: report of fix agent A

Worktree: `C:\Users\luigi\Documents\GitHub\pyvbmc\.claude\worktrees\agent-a233f2f6ca4cdb8f0`
Branch: `worktree-agent-a233f2f6ca4cdb8f0` (cut from `b2c16ebf`, fast-forwarded to `a2104e61`; nothing pushed)

## Setup outputs

- `git merge --ff-only dev-port-review` → fast-forward `b2c16ebf..a2104e61`; `git log -1 --format=%h` printed `a2104e61`.
- From the worktree root, `python -c "import os, sys; sys.path.insert(0, os.getcwd()); import pyvbmc; print(pyvbmc.__file__)"` printed
  `C:\Users\luigi\Documents\GitHub\pyvbmc\.claude\worktrees\agent-a233f2f6ca4cdb8f0\pyvbmc\__init__.py`.
- Every Python command ran with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`. One standalone probe script (scratchpad `fix_wave5_A/probe_w5_4.py`, `PYTHONPATH=<worktree>`) checked the corrected original-space density and the guards of item 10; it printed the same `pyvbmc.__file__`.
- `git status --porcelain` is empty at the end (and `--ignored` shows nothing but `!!` entries); HEAD is `ad03a116`.

## `git log --oneline a2104e61..HEAD`

```
ad03a116 fix(variational_posterior): correct the remainder weights of a balanced draw by their total
00532881 fix(stats): take the log determinants of the two covariance matrices
139e6d46 fix(variational_posterior): ignore the densities the KL divergence cannot use
4086f67f fix(variational_posterior): take a single starting point laid out as a column
7be2538d fix(variational_posterior): evaluate the density at the coordinates it was given
a6f96e02 fix(variational_posterior): refuse a negative df in sample with the reason
b78e2a26 fix(variational_posterior): give the shapes and counts of sample and moments their documented form
98e9d84c fix(variational_posterior): draw the mode candidates from a copy of the generator
79103cc0 fix(variational_posterior): drop the stored mode when the parameters are normalized
d169df32 fix(variational_posterior): find the mode of a one-dimensional posterior
72c35b7e fix(variational_posterior): check the entries that hold the scales and the weights
c40d3862 fix(variational_posterior): take the mean of the given samples per coordinate
```

Diffstat over the range: 4 files, +604 / −69 (`variational_posterior.py`, `stats/kl_div_mvn.py`, `testing/variational_posterior/test_variational_posterior.py`, `testing/stats/test_kldiv_mvn.py`). Nothing outside my area.

## Commits in order

| # | hash | finding | what it does | files | tests added or changed |
|---|---|---|---|---|---|
| 1 | `c40d3862` | W5-9 | `kl_div(samples=…, gauss_flag=True)` takes `np.mean(samples, axis=0)`, the mean per coordinate, as `vbmc_kldiv.m:63` does | `variational_posterior.py`, test file | +`test_kl_div_samples_mean_is_taken_per_coordinate` (D=3, means 10/−5/2: samples of itself give ≈0; shifted samples give exactly `kl_div_mvn` of the posterior's moments against the per-coordinate sample moments) |
| 2 | `72c35b7e` | W5-10 | the positivity check counts the entries of `sigma`, `lambd` and the weights and reads that tail; it is skipped when the tail is empty | `variational_posterior.py`, test file | **changed** `test_set_parameters_not_raw_negative_error` (its means are now positive, so the refusal can only come from the constrained entries); +`test_set_parameters_not_raw_checks_the_constrained_entries`, 32 cases (16 flag combinations × `D=3,K=4` and `D=2,K=2`), each taking a vector with negative means and refusing one negative scale or weight at a time |
| 3 | `d169df32` | W5-11 + W5-12 | the search box is built with `reshape(-1)`, which holds at `D = 1`; `x0` is clamped to that box (the bounds shrunk by `sqrt(eps)`), not to the raw bounds | `variational_posterior.py`, test file | +`test_mode_one_dimensional[bounded/unbounded]` (shape `(1,)`, agrees with the grid maximum, equals the component mean when unbounded); +`test_mode_starts_inside_the_box_it_searches` (records the `x0` handed to `minimize` through `mocker.patch.object` on the module) |
| 4 | `79103cc0` | W5-13 | `get_parameters` sets `self._mode = None` (the comment at `:1021` becomes the line it described) and its docstring says what it normalizes in place; `mode(n_opts=…)` does not read the cache | `variational_posterior.py`, test file | +`test_mode_is_recomputed_when_n_opts_is_given`, +`test_get_parameters_clears_the_stored_mode` |
| 5 | `98e9d84c` | W5-14 | the candidates come from `copy.deepcopy(self)` carrying `copy.deepcopy(self.rng)`, the pattern of `_compute_true_diagnostic`; `sample` is untouched | `variational_posterior.py`, test file | +`test_mode_leaves_the_random_stream_alone` (generator state equal before and after, two calls identical, a later draw equal to the draw from a freshly seeded generator) |
| 6 | `b78e2a26` | W5-15 | `np.atleast_2d` on the Monte Carlo covariance; a whole float `N` accepted and a fractional one refused naming `N`; the index array `(N,)` of integer dtype in every branch, and the docstring brought to it | `variational_posterior.py`, test file | **changed** `test_sample_n_lower_1` (asserted `(0, 1)`, now the flat shape and integer dtype); +`test_sample_index_array_has_one_integer_per_row` (4 cases), +`test_sample_takes_a_whole_number_given_as_a_float`, +`test_kl_div_and_mtv_take_a_whole_number_given_as_a_float`, +`test_moments_orig_flag_one_dimensional_covariance` |
| 7 | `a6f96e02` | W5-16 | `sample` refuses a finite negative `df` with a message naming the product of univariate `t` densities; documented under `df` and in `Raises` | `variational_posterior.py`, test file | +`test_sample_refuses_a_negative_df` (the density of that family is evaluated in the same test) |
| 8 | `7be2538d` | W5-18 | the working copy of the points is `np.array(x, dtype=np.float64)` | `variational_posterior.py`, test file | +`test_pdf_whole_coordinates_given_as_integers` (array of ints, list of ints, 1-D int array, against the float spelling, under a logit transform on `[0, 10]^2`) |
| 9 | `4086f67f` | W5-21 | `np.tile(x0.reshape(-1, 1), (1, K))`: the dead reshape becomes the one the tiling needs; the docstring of `x0` gives the layouts | `variational_posterior.py`, test file | +`test_constructor_takes_one_starting_point_in_any_layout` (row, flat, column, same seed, identical `mu`) |
| 10 | `139e6d46` | W5-4 | the four guards read `(q == 0) | ~np.isfinite(q)`; the original-space density replaces the quotient by `exp(log y − logJ)` on exactly the rows where the quotient is not finite or the Jacobian has underflowed | `variational_posterior.py`, test file | +`test_kl_div_ignores_a_density_that_is_zero_or_not_finite` (a mocked `pdf` returning `[0, 1, inf, nan, 0.5]`, compared with the MATLAB rule), +`test_kl_div_is_a_number_when_a_density_overflows` (probit unit box, SD 30), +`test_pdf_original_space_density_holds_its_representable_values` (the script's `u = (-27.3, -27.3)`), +`test_pdf_original_space_density_is_the_quotient_where_that_holds` (49 ordinary points, bit for bit against the old expression written out) |
| 11 | `00532881` | W5-2 | `np.linalg.slogdet` for both matrices, `lndet = lndetq2 − lndetq1`; infinite for sign 0 (singular) or negative (negative determinant); the two `lstsq` solves untouched | `stats/kl_div_mvn.py`, `testing/stats/test_kldiv_mvn.py` | +`test_kl_div_does_not_depend_on_the_scale` (D=20, SD 1e-9/1e-8/1/1e9 against the closed form, `rtol=1e-9`), +`test_kl_div_infinite_without_a_positive_determinant` (3 matrices × both argument positions). The five existing tests are untouched and pass |
| 12 | `ad03a116` | W5-1 | `repeats_extra - np.sum(w_extra)`; nothing else in the method, same number of draws from the generator | `variational_posterior.py`, test file | +`test_sample_balance_remainder_follows_the_fractional_parts`, with a `_RecordingGenerator` subclass keeping the `p` of `choice`: `[0.1, 0.6, 0.3]` for `w = [0.7, 0.2, 0.1]`, `N = 13`, and `[1/3, 2/3]` for `w = [1/3, 2/3]`, `N = 10` |

Every fix was seen to fail first: the hunk was reverted (Edit in and out, or `git checkout HEAD -- <one file>` with the working copy saved to the scratchpad first — never a bare `git stash`), the new tests were run and failed, and the fix was put back. Counts of the failures seen on the old code: 1 (item 1), 27 of the 32 flag cases + the corrected negative-vector test unchanged (item 2), 3 (item 3), 2 (item 4), 1 (item 5), 6 (item 6), 1 (item 7), 1 (item 8), 1 (item 9), 3 of the 4 (item 10 — the fourth pins the bit-for-bit invariance and passes on both), 4 of the 5 new cases (item 11), 2 (item 12).

## Test runs, and results

From the worktree root, `-q -p no:cacheprovider`, with the three thread variables set:

- `pyvbmc/testing/variational_posterior/test_variational_posterior.py` — **113 passed** (94 before the pass)
- `pyvbmc/testing/variational_posterior/test_variational_posterior_grad_fd.py` + `pyvbmc/testing/stats/test_kldiv_mvn.py` — **14 passed** (2 + 12)
- Final combined run of the three files at HEAD: **127 passed in 7.6 s**

No oracle command, no golden replay, no `optimize()` run, no install, no static pickle saved. One slip to record: after item 8 I ran the whole `pyvbmc/testing/variational_posterior/` directory once (118 passed, 2 skipped), which includes `test_vp_save_and_load.py`; `git status --porcelain` right afterwards was clean, so `test_vp_save_dynamic.pkl` was rewritten byte-identical or not at all. Every later run named the files.

## Changelog sentences (for a user of release 1.0.4)

- **W5-9** (`c40d3862`) — "`vp.kl_div(samples=…, gauss_flag=True)` compares the posterior with the moments of the given samples taken coordinate by coordinate, instead of with a single average over the whole sample matrix broadcast into every coordinate." Changes what such a call returns, unless every coordinate of the samples happens to have the same mean, so it belongs in the "Upgrading from 1.0.4" list.
- **W5-10** (`72c35b7e`) — "`vp.set_parameters(theta, raw_flag=False)` requires the entries that hold `sigma`, `lambd` and the weights to be positive, and those alone: a negative scale or weight that used to pass is refused, and a negative component mean that used to be refused is taken." Can stop a script that passed a negative scale, and unblocks one that passed negative means, so it belongs in the "Upgrading from 1.0.4" list.
- **W5-11 and W5-12** (`d169df32`) — "`vp.mode()` returns the mode of a posterior of one parameter, where it used to raise an `AxisError`, and the starting point of the search is placed inside the box the search runs in." Cannot stop a script; the one-dimensional call used to fail outright.
- **W5-13** (`79103cc0`) — "`vp.get_parameters()` discards a mode stored by an earlier `vp.mode()` call, because the normalization it performs may move it, and `vp.mode(n_opts=…)` runs the search instead of returning the stored mode." Can change what a script returns (a mode that used to come from the store is recomputed), so it belongs in the "Upgrading from 1.0.4" list.
- **W5-14** (`98e9d84c`) — "`vp.mode()` draws its starting candidates from a copy of the posterior's random generator, so a call leaves the stream where it was: a run continued after a call of `vp.mode()` is no longer diverted, and two calls on one posterior return the same mode." Changes what a script returns after a `vp.mode()` call (every later draw from that posterior), so it belongs in the "Upgrading from 1.0.4" list.
- **W5-15** (`b78e2a26`) — "`vp.moments(cov_flag=True)` returns a 1-by-1 covariance matrix for a posterior of one parameter; `vp.sample`, and through it `vp.kl_div(gauss_flag=False)` and `vp.mtv`, take a number of samples written as a whole float such as the documented `1e5` and refuse one with a fractional part; and the component indices `vp.sample` returns are one integer per drawn row in every case, where they were floating-point for a single-component posterior and a `(0, 1)` array for a count below one." Can stop a script that read the one-dimensional covariance as a scalar or the index array as a column, so it belongs in the "Upgrading from 1.0.4" list.
- **W5-16** (`a6f96e02`) — "`vp.sample` refuses a negative `df` with a message saying that it stands for the product of univariate `t` densities, which `vp.pdf` evaluates and `vp.sample` cannot draw from, instead of surfacing NumPy's `shape < 0`." Cannot stop a script; such a call already failed.
- **W5-18** (`7be2538d`) — "`vp.pdf` and `vp.log_pdf` evaluate the density at the point they are given also when its coordinates are integers or a list of integers: in the original space the transformed coordinates used to be truncated to the caller's integer type and the density was reported at another point." Changes what such a call returns, so it belongs in the "Upgrading from 1.0.4" list.
- **W5-21** (`4086f67f`) — "`VariationalPosterior(D, K, x0)` takes a single starting point given as a `(D, 1)` column, as it takes a row and a flat array, instead of ending in a broadcast error." Cannot stop a script.
- **W5-4** (`139e6d46`) — "`vp.kl_div(gauss_flag=False)` sets aside a density that is infinite or NaN, as it already set aside a zero one, and returns a number where it used to return `nan`; and `vp.pdf` in the original space keeps a density near the top of the range of a double instead of reporting it as infinite when the Jacobian of the transform underflows." Changes what those calls return in those cases, so it belongs in the "Upgrading from 1.0.4" list.
- **W5-2** (`00532881`) — "`pyvbmc.stats.kl_div_mvn`, and with it the symmetrized KL divergence a run reports, is computed from the log determinants of the two covariance matrices, so it is right where the determinants themselves leave the range of a double: at ten to twenty parameters with a posterior standard deviation below about 8e-9 or above about 5e7 it used to report an infinite divergence, or zero, the value of a posterior that has stopped moving." Can change what a run reports at such parameter scales (and the last digits of the divergence at ordinary ones), so it belongs in the "Upgrading from 1.0.4" list.
- **W5-1** (`ad03a116`) — "A balanced draw from a variational posterior with unequal component weights draws its remaining samples in proportion to the fractional parts of `w * N`, as MATLAB VBMC does, so a component contributes `w_k N` draws on average instead of up to about one draw more or fewer. Runs give slightly different results." Changes what a run returns, so it belongs in the "Upgrading from 1.0.4" list.

Commits 1 to 11 change no number of a run at the default options: `kl_div(samples=)` and `mode` have no caller inside the package, `set_parameters(raw_flag=False)` is never used by it, the `moments` covariance was already promoted by `kl_div_mvn`, every internal `pdf` caller passes float arrays and an `int` count, and item 10's two changes touch only rows that were `nan` or `inf`. Commit 11 can move the last digits of the log-determinant term of `sKL`; commit 12 is the known moving fix.

## What I stopped on

Nothing. All twelve items are done as ruled.

## For the orchestrator

- **S-VBMC and the last commit.** `pyvbmc/svbmc/svbmc.py:932` calls `vp_copy.sample(int(counts[m]), balance_flag=balance_flag)` on a copy whose weights are the optimized component weights of one run, generally unequal. W5-1 therefore moves S-VBMC draws wherever that path runs with `balance_flag=True`, which the W5-1 ledger row does not name among what the corrected line moves. `pyvbmc/testing/svbmc/fixtures/references.npz` is the gate for that subpackage; worth checking with the other gates of the moving fix. I ran no S-VBMC test.
- **A genuinely overflowing density still is infinite.** Item 10's density fix recovers only what a double can hold. On the probit unit box with an inference-space SD of 30, 1047 of 1e4 draws still have a log density above `log(realmax)` and a density of `inf` after the fix; the corrected guard is what keeps `kl_div` finite there, and both halves of W5-4 are needed. The script's `D = 2` point now gives 2.42628749e306, equal to the exponential of its log density.
- **`mode` still stores unconditionally.** The ruling says an explicit `n_opts` bypasses the cache; I made it bypass the *read* and left the store as it was, so `mode(n_opts=k)` recomputes and then stores its result, which a later `mode()` returns. Say the word if a non-default `n_opts` should also leave the store alone.
- **Warnings around the Jacobian quotient.** Item 10 wraps the quotient and its recovery in `np.errstate(divide, over, invalid = "ignore")`, since every non-finite outcome is handled there. One consequence: a density that genuinely overflows no longer emits NumPy's overflow warning, where the old expression did.
- **A float32 input to `pdf`.** Item 8 makes the working copy float64, so a float32 array of points in the original space is transformed in full precision instead of being rounded back into float32; the returned array was already float64 in both cases. It is the same promotion agent B makes in `Prior.log_pdf` (W5-31).
- **`test_mode_starts_inside_the_box_it_searches`** patches `minimize` with `mocker.patch.object` on the module fetched by `importlib.import_module`, per the dotted-name trap in `AGENTS.md`. It is the only mock-based test I added besides the mocked `pdf` of item 10, which is patched on the class.
- **No line-ending change.** Two files were rewritten through the scratchpad during the fail-first checks; `git diff` shows only the intended hunks (git normalizes to LF on commit, and the working copy keeps CRLF).
