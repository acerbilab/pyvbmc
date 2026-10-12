# P8 — parameter transformer, warping, function logger, MATLAB-comparison track

Raw reviewer report, wave 3 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-20. One Opus agent, fresh
context, read-only on the repositories; it compared the Python code with
MATLAB VBMC at `396d649`. Its brief added one first question, on the
floating-point evaluation of the bounded transforms near a hard bound. Its
check scripts ran in the session scratchpad and are kept only on the
machine that ran them (`dev/scripts/runs/LOCAL.md`). The text below is the
agent's final message, unedited.

---

# P8 — parameter transformer, warping, function logger, input-shape decorator (MATLAB-comparison track)

## 1. Coverage

**Read completely (Python):** `pyvbmc/parameter_transformer/parameter_transformer.py`, `pyvbmc/whitening/whitening.py`, `pyvbmc/function_logger/function_logger.py`, `pyvbmc/decorators/handle_0D_1D_input.py`. Also read completely: `pyvbmc/vbmc/_bounds.py`, the warping/active-sampling/bookkeeping blocks of `pyvbmc/vbmc/vbmc.py` (`__init__` 380-480, `_init_optim_state` 916-965, the iteration head and warp block 1244-1470, warm-up trimming 2060-2106), `pyvbmc/testing/whitening/test_rotoscaling.py`, and the relevant parts of `pyvbmc/testing/parameter_transformer/test_parameter_transformer.py` and `pyvbmc/testing/function_logger/test_function_logger.py` (test names + the boundary/Jacobian/duplicate tests).

**Read completely (MATLAB):** `shared/warpvars_vbmc.m`, `misc/warp_input_vbmc.m`, `misc/warp_gpandvp_vbmc.m`, `utils/unscent_warp.m`, `misc/funlogger_vbmc.m`, `misc/setupvars_vbmc.m`, `misc/boundscheck_vbmc.m`; plus the warping/active-sampling blocks of `vbmc.m` (500-670) and the `'add'` call sites in `private/activesample_vbmc.m` and `misc/initdesign_vbmc.m`.

**Skimmed:** `pyvbmc/vbmc/active_sample.py` (only the logger/`optim_state["N"]` interaction and the consumption of the logger's return value), `pyvbmc/vbmc/gaussian_process_train.py` (`_get_training_data` only). **Not reached:** the acquisition functions, the VP internals (`vp.moments`, whose covariance `warp_input` consumes), gpyreg. Read the sheet's "Slice P8" and "Settled non-differences" sections, plus the P1a/P1b entries that touch this slice.

**Checks run** (scripts in the scratchpad `wave3_P8_comparison/`; `pyvbmc.__file__` confirmed to be the checkout):
- `matlab_warpvars.py` — a line-by-line Python transcription of `warpvars_vbmc.m` (create, `'d'`, `'i'`, `'logp'`) for types 0/3/12/13, keeping MATLAB's expression grouping and its `eps(a)` clamp.
- `compare_transforms.py` — port vs. transcription at points 0-3 ulp from each hard bound, for logit/probit/student4, with bounds `(0,1)`, `(0,3)`, `(1,3)` and `(-5,5)` with plausible bounds `(-1,2)`.
- `random_ulp.py` — 20 000 random points per configuration, D=3, mixed bounds, with and without rotation+scale; ulp-exact comparison of direct, inverse and log-Jacobian.
- `logger_checks.py` — return types, `fun_eval_time`, and the noisy duplicate-pooling arithmetic against the MATLAB expression.
- `warp_inactive.py` — a small `VBMC` object with a trimmed logger row, through `warp_input`.

**Answer to the first question.** The bounded forward map, the inverse and the log-Jacobian evaluate the *same expressions in the same order* as `warpvars_vbmc.m`, with exactly two exceptions, and one added branch:
- Over 20 000 random interior points per configuration (all three transforms, with and without rotation and rescaling), the **direct transform and the log-Jacobian are bit-identical** — 0 ulp, 0 % differing — in every configuration. The centering, the `log(b-a) + … + log(delta)` grouping, the rotation-then-rescale order and the `+log(scale)` density correction all match.
- The **student4 inverse** regroups one product: MATLAB computes `((3/8)*x)/sqrt(1+t²/4)`, PyVBMC computes `(3/8)*(x/sqrt(1+t²/4))` (finding F3). 22 % of points differ, by 1 ulp of the `0.5 + …` sum; in the lower tail the shared formula cancels, so the relative difference reaches 9.6e-12 (largest seen: `f = 1.1613183546144157e-05` vs `1.1613183546033135e-05` at `x = -22.47`). In absolute terms the error is `(ub-lb)·1.1e-16`.
- The **logit and probit inverses** are bit-identical except where the bound clamp binds, where they can differ by 1 ulp (F2).
- PyVBMC adds a **nudge in `_to_unit_interval`** that MATLAB has no counterpart for (F1).

*Points within a few ulp of a hard bound.* Both sides send `x == lb` and `x == ub` to ∓∞ (all three transforms, all bound positions) — identical. For a *distinct* point one ulp inside:
- **Bound away from zero, `lb=-5, ub=5, plb=-1, pub=2`, `x = nextafter(5,-∞)`:** `(x-lb)/(ub-lb)` rounds to exactly `1.0`, so MATLAB returns **`+Inf`** while PyVBMC nudges `z` to `1-eps/2` and returns a finite `29.148` (logit), `10.381` (probit), `15274.55` (student4). Same for `lb=1, ub=3` at `x=nextafter(1,∞)`: there `z` does not round, and both sides agree bit-for-bit (`-36.7`, `-8.22`, `-23.8`).
- **Bound at zero, `lb=0, ub=1`, `x = 5e-324`:** `z = 5e-324` exactly, no nudge, both sides identical (`-744.44` logit, `-38.47` probit).
- **Bound at zero, `lb=0, ub=3`, `x = 5e-324`:** `z` underflows to 0, so MATLAB returns **`-Inf`** and PyVBMC returns `-744.44` / `-38.47` / `-8.83e+80`.
- On the inverse, both sides clamp the result inside the bounds; the clamps coincide except when the binding bound is an exact power of two in magnitude and the clamp moves toward zero, where MATLAB lands one ulp further inside (`ub=1`: MATLAB `0.9999999999999998`, PyVBMC `0.9999999999999999`).

I also checked and found **no** difference in: the `unscent_warp` reshape (the two row orders differ, but `fun` is row-independent and each side's reshape-back is its own inverse), `std` with N-1 on both sides, the ±σ sigma-point indices (MATLAB `2d`/`2d+1` 1-based ↔ Python `2d+1`/`2d+2` 0-based), every hyperparameter slice in `warp_gp_and_vp` (`Ncov+Nnoise+1+(1:D)` ↔ `[Ncov+Nnoise+1 : Ncov+Nnoise+D+1]`), the `lambdaw`/`sigmaw`/`w` formulas and their reduction axes, the `hyp_warped` transpose (gpyreg is row-per-sample), the cache size (500 both sides), the 50 % cache growth rule, `nevals`, `ymax` (recomputed at the head of active sampling on both sides, `vbmc.py:1428` ↔ `vbmc.m:632`), `optim_state["iter"]`/`last_warping`/`warping_count` bookkeeping, the `w_reg` clamp, `sqrt(diag(S+eps))`, the `det(U)<0` sign flip, `delta_temp/9`, `Nrnd` values, the search-bound update and the search-cache update. Posterior tempering: PyVBMC never puts a `temperature` key in `optim_state`, so `T = 1` everywhere and the fact that `warp_gp_and_vp` reads it from `optim_state` where MATLAB reads `vp_old.temperature` (which `vpoptimize_vbmc.m:190` keeps equal to `optimState.temperature`) has no effect; tempering is a sheet-covered unported feature.

---

## 2. Findings

### F1. The direct transform nudges an interior point whose unit-interval image rounds to 0 or 1; MATLAB returns ±Inf
- Location: `pyvbmc/parameter_transformer/parameter_transformer.py:539-548` (`_to_unit_interval`, `safe=True`), used at `:313-322`, `:358-367`, `:398-407`; MATLAB: `shared/warpvars_vbmc.m:106-107`, `:256-257`, `:265-266` — no counterpart.
- Category: formula/gradient
- Proposed classification: possibly intentional (Python-only addition, absent from the sheet's Slice P8 section)
- Confidence: high (the behavior), medium (that it was meant to differ from MATLAB)
- History: added 2022-08-28 in commit `3e9d1c21` (PR #89), whose messages read "fix: Handle points close to boundary identically to MATLAB" and "fix: Nudge bounded points in proportion to UB-LB"; `_to_unit_interval` itself dates from `4ce90e25` (2022-07-28). The MATLAB lines have not changed since before the port began (`74b046e`, 2022-06-25, only threaded the `bounded_type` argument into the create branch). The Python code matched MATLAB from 2022-07-28 to 2022-08-28.
- What the code does: MATLAB computes `z = (x-a)/(b-a)` and feeds it straight to the logit / `erfcinv` / Student-t map, so an `x` strictly inside the box whose `z` rounds to 0 or 1 is sent to ∓∞. PyVBMC replaces such a `z` with the adjacent representable value (`5e-324` or `1-eps/2`), but only when `x != lb` / `x != ub`, so an `x` exactly on a bound still maps to ∓∞ on both sides. Reproduced above: `lb=-5, ub=5`, `x = nextafter(5,-inf)` gives `+Inf` in MATLAB and `29.148`/`10.381`/`15274.55` in PyVBMC; `lb=0, ub=3`, `x = 5e-324` gives `-Inf` against `-744.44`/`-38.47`/`-8.83e+80`.
- Consequence if real: PyVBMC keeps a finite (very large) transformed coordinate where MATLAB puts ±Inf into the GP training set, `vp.mu`, or `optim_state`. It triggers whenever the direct transform sees a point within about one ulp of a bound — which the inverse's own clamp produces (a point whose transformed coordinate exceeds ≈ ±37 for probit comes back clamped to `nextafter(bound)`), and `warp_input` re-transforms every stored `X_orig` at each warp. Rare, but when it fires the MATLAB path would produce ±Inf and NaN-poison the GP, so PyVBMC is the more robust of the two; the divergence is nonetheless real and unrecorded.
- Suggested reproduction: ran it — `compare_transforms.py` in the scratchpad; the smallest form is `ParameterTransformer(1, [[-5.]], [[5.]])(np.nextafter(5.0,-np.inf))` against `-sqrt(2)*erfcinv(2*(x+5)/10)`.
- Test adequacy: `test_boundary_edge_cases` (`test_parameter_transformer.py:836`) asserts exactly that near-bound points transform to *finite* values, i.e. it pins the Python behavior. No test compares this case with MATLAB, and no test states that the nudge is a departure.

### F2. The inverse clamp uses `nextafter`; MATLAB uses `eps(bound)`, which is one ulp further in at power-of-two bounds
- Location: `pyvbmc/parameter_transformer/parameter_transformer.py:551-556` (`_from_unit_interval`); MATLAB: `shared/warpvars_vbmc.m:457-459`.
- Category: formula/gradient
- Proposed classification: port discrepancy (benign)
- Confidence: high
- History: MATLAB unchanged since before the port. PyVBMC matched MATLAB exactly until `3e9d1c21` (2022-08-28): the removed code was `xNew = maximum(xNew, lb + abs(np.spacing(lb)))` / `minimum(xNew, ub - abs(np.spacing(ub)))`, which is `eps(a)` verbatim; it was replaced by `np.nextafter`. So this entered with a commit whose stated aim was to match MATLAB more closely.
- What the code does: MATLAB clamps to `[a+eps(a), b-eps(b)]`, where `eps(x)` is the ulp of `|x|`. When the clamp moves *toward zero* across a power-of-two boundary (`b = 1, 2, 4, …`, or `a = -1, -2, …`), the ulp of `|x|` is twice the local spacing, so MATLAB's clamp is one ulp further inside than `nextafter`. Verified: bound `1.0` → MATLAB `0.9999999999999998`, PyVBMC `0.9999999999999999`; bound `-2.0` lower clamp → MATLAB `-1.9999999999999996`, PyVBMC `-1.9999999999999998`. At every other bound (`3`, `5`, `11`, `0`, `-5`) the two clamps are identical. In the randomized run this is the sole source of the probit-inverse differences (max 1 ulp, 0.7-1.8 % of points, all at the `ub = 1.0` coordinate).
- Consequence if real: one ulp in a clamped original-space coordinate, only for points pushed onto a bound. Negligible numerically; worth knowing when a future replay is compared bit-for-bit against MATLAB.
- Suggested reproduction: ran it — `np.spacing(abs(b))` vs `np.nextafter(b, ∓inf)` over a list of bounds (script output quoted above).
- Test adequacy: `test_boundary_edge_cases` asserts round-trip equality with `np.nextafter`-built points, i.e. it mirrors the implementation's choice of `nextafter`. It would pass under either convention.

### F3. The student4 inverse groups `3/8 * x / sqrt(1+t²/4)` differently from MATLAB
- Location: `pyvbmc/parameter_transformer/parameter_transformer.py:604-608` (`_inverse_student4`); MATLAB: `shared/warpvars_vbmc.m:451`.
- Category: formula/gradient
- Proposed classification: port discrepancy (floating-point association only)
- Confidence: high
- History: MATLAB's type-13 branch predates the port and is unchanged. PyVBMC's student4 arrived with `4ce90e25` (2022-07-28, "Probit changes"), already in this grouping; it never matched MATLAB's grouping.
- What the code does: MATLAB evaluates `((3/8)*x) ./ sqrt(1+t2/4) .* (1 - …)`; PyVBMC evaluates `(3/8) * (x/sqrt(1+t2/4)) * (1 - …)`. `3/8` is not a power of two, so the two differ by up to 1 ulp of the product. Because the closed form is `0.5 + (a quantity approaching -0.5)`, the sum cancels in the lower tail, and the 1-ulp difference becomes up to 9.6e-12 relative there (measured over 200 000 random `x`; 24.8 % of values differ). Upper tail and centre agree bit-for-bit.
- Consequence if real: the recovered original-space coordinate differs by at most `(ub-lb)·1.1e-16` — invisible in absolute terms, but it makes any bit-exact comparison with MATLAB impossible for `bounded_transform="student4"`, and it can flip a near-tie in a search that ranks such coordinates.
- Suggested reproduction: ran it — `random_ulp.py` (last block prints both groupings side by side).
- Test adequacy: `test_inverse_type3_within` and friends compare against `sps.t.cdf`-equivalent values with `np.isclose` (rtol 1e-5); no test is near this tolerance. Nothing would catch it, and nothing should — the point is only that the expression order is not MATLAB's.

### F4. `warp_input` rewrites only the active rows of the function logger; MATLAB rewrites rows `1:Xn`
- Location: `pyvbmc/whitening/whitening.py:213-220`; MATLAB: `misc/warp_input_vbmc.m:112-119`.
- Category: indexing/shape
- Proposed classification: port discrepancy
- Confidence: high (the difference), medium (that it matters)
- History: MATLAB's `idx_n = 1:optimState.Xn` dates from the 2020 rotoscaling work and is unchanged. The Python line has used `X_flag` since whitening was ported (`4aa4ca97`, 2022-03-03); it never matched.
- What the code does: MATLAB takes `X_orig(1:Xn,:)` and `y_orig(1:Xn)` — every filled row, active or not — re-transforms them and writes both `optimState.X` and `optimState.y` back. PyVBMC selects `function_logger.X_flag`, so rows deactivated by warm-up data trimming (`vbmc.py:2104`, MATLAB `private/vbmc_warmup.m:126`) keep coordinates and values expressed in the *previous* inference space. Reproduced: with row 1 marked inactive, after `warp_input` the row still holds `[0.1, -0.2]` where the new space wants `[-1.376, -5.976]`; rows 0, 2, 3 were rewritten.
- Consequence if real: `_record`'s duplicate test (`function_logger.py:657`) compares a new point against *all* rows of `self.X`, stale ones included, so after a warp a re-evaluation at a trimmed point's location is no longer recognized as a repeat — PyVBMC appends a second row with the same `X_orig` where MATLAB pools into the existing one. Conversely a stale row could match a new point by coincidence and pool two different points (astronomically unlikely). Nothing else reads inactive rows (GP training, `y_max`, `n_eff`, `recompute_lcb_max` all index through `X_flag`), so the effect is confined to duplicate accounting and to what a user sees in `function_logger.X` after `finalize()` (rows in mixed coordinate spaces). Requires both a data-trimming event and a later warp.
- Suggested reproduction: ran it — `warp_inactive.py` in the scratchpad.
- Test adequacy: no. `test_warp_input*` construct a logger with no evaluations at all, so the stored-point update in `warp_input` is never exercised, let alone with an inactive row. `test_finalize_preserves_noisy_duplicate_and_inactive_rows_then_appends` covers inactive rows in `finalize` only.

### F5. `ParameterTransformer` has no log transform for half-bounded variables (MATLAB types 1 and 2)
- Location: `pyvbmc/parameter_transformer/parameter_transformer.py:135-142` (the type loop); MATLAB: `shared/warpvars_vbmc.m:897-901` (assignment), `:92-101`, `:302-312`, `:491-494` (the three branches).
- Category: control flow
- Proposed classification: possibly intentional (unported feature, absent from the sheet)
- Confidence: high (the code), medium (the consequence, because VBMC itself blocks the input)
- History: the Python code had types 1 and 2 until commit `6a247e59` (2021-02-25) "fix: delete everything associated with type1and2 in ParameterTransformer". MATLAB's branches are unchanged.
- What the code does: a coordinate with one finite and one infinite bound gets `type = 0` in PyVBMC, i.e. the identity (with centering), where MATLAB assigns type 1/2 and applies `log(x-a)` / `log(b-x)` with inverse `exp(y)+a` / `b-exp(y)` and log-Jacobian `y`. The Python inverse then returns points *outside* the declared support: with `lb=0, ub=inf, plb=1, pub=3`, `inverse([-2, 0, 2])` gives `[-2, 2, 6]`, and `-2 < lb`. Unreachable from `VBMC`, which rejects half-bounded variables (`_bounds.py:291-299`, matching `misc/boundscheck_vbmc.m:138-143`) — but `ParameterTransformer` is public, documented API, and its docstring does not say that a one-sided bound is unsupported; it silently produces an unbounded map instead of raising.
- Consequence if real: only for a caller who builds a `ParameterTransformer` directly (or a future PyVBMC path that relaxes the half-bounds check). Then the target is evaluated outside its support and the log-Jacobian is wrong (constant `log(delta)` instead of `y`).
- Suggested reproduction: ran it — `ParameterTransformer(1, [[0.]], [[inf]], [[1.]], [[3.]]).inverse([[-2.]])` returns `-2.0`. (Note: my MATLAB transcription does not implement types 1/2, so the MATLAB column of that particular run is not usable; the MATLAB values above are from reading `warpvars_vbmc.m:94, 305, 493`.)
- Test adequacy: no test constructs a half-bounded transformer. `test_transform_bounded_and_unbounded` mixes a fully bounded with a fully unbounded coordinate only.

### F6. The low-correlation mask inverts MATLAB's comparison, so a NaN correlation is kept instead of zeroed
- Location: `pyvbmc/whitening/whitening.py:154-159`; MATLAB: `misc/warp_input_vbmc.m:52-56`.
- Category: control flow
- Proposed classification: port discrepancy (edge case only)
- Confidence: medium (the code), low (that it is ever reachable)
- History: unchanged on both sides since the whitening port (`4aa4ca97`, 2022-03-03).
- What the code does: MATLAB keeps `mask_idx = abs(vp_corr) > thresh` and zeroes the complement, `vp_Sigma(~mask_idx) = 0`. PyVBMC zeroes `abs(vp_corr) <= thresh`. The two agree on every number but not on NaN: `abs(NaN) > t` is false, so MATLAB zeroes such an entry; `abs(NaN) <= t` is also false, so PyVBMC leaves it NaN, and the NaN then propagates through the regularization into the SVD. A NaN correlation needs a zero (or non-finite) diagonal entry of the variational covariance, which a well-formed posterior does not have.
- Consequence if real: a NaN covariance entry would make `np.linalg.svd` raise instead of being silently repaired. Never observed; reported for completeness because it is the classic negated-comparison trap.
- Suggested reproduction: `vp_cov` with a zero diagonal entry, passed through both mask expressions. Not run (I could not construct a realistic posterior with a zero-variance coordinate).
- Test adequacy: no. `test_warp_input_cov_reg` covers the regularization amount, not the correlation mask's edge cases.

### F7. `add()` leaves `fun_eval_time` at NaN where MATLAB records 0, and charges `total_fun_eval_time` on a path MATLAB does not
- Location: `pyvbmc/function_logger/function_logger.py:468-473` (default `fun_eval_time=np.nan`), `:719-723`, `:734-736`; MATLAB: `misc/funlogger_vbmc.m:187` (`record(..., t=0, ...)`), `:246`, `:273`, and `:151` (the only place `totalfunevaltime` grows).
- Category: state/caching
- Proposed classification: port discrepancy
- Confidence: high
- History: MATLAB unchanged since 2019. The Python default has been `np.nan` since `f585f394` (2021-02-18); it never matched.
- What the code does: MATLAB's `'add'` action passes `t = 0`, so a cached point's `funevaltime` is `0` and a later repeat averages `(N*0+0)/(N+1) = 0`. PyVBMC's `add` defaults to NaN, so the row's `fun_eval_time` stays NaN and a repeat turns the running average into NaN permanently (reproduced: `fun_eval_time[0] = nan` after two `add` calls, where MATLAB has `0`). In the other direction, PyVBMC adds `fun_eval_time` into `total_fun_eval_time` inside `_record`, so an `add(..., fun_eval_time=t)` charges the total, while MATLAB charges it only in the `'iter'`/`'single'` branch; PyVBMC's own call sites always use the NaN default, so this half is latent.
- Consequence if real: none numerically today — `_get_training_data` returns `t_train` and `gaussian_process_train.py:958` records "Missing port: gp.t = t_train", so nothing consumes it. It does corrupt `function_logger.fun_eval_time` as a reporting field, and `total_fun_eval_time` feeds `results["overhead"]`, which PyVBMC already returns as NaN (sheet, P1a).
- Suggested reproduction: ran it — `logger_checks.py`, section 2.
- Test adequacy: partly. `test_add_record_funtime` exists but pins the Python behavior (it passes a time explicitly); nothing checks the default or a repeated `add`.

### F8. `FunctionLogger` returns a one-element array for a repeated point and a float for a new one
- Location: `pyvbmc/function_logger/function_logger.py:715` (`f_val = self.y_orig[idx].copy()`) against `:741` (`f_val = f_val_orig`); MATLAB: `misc/funlogger_vbmc.m:243` and `:269`, both scalars.
- Category: indexing/shape
- Proposed classification: port discrepancy (cosmetic)
- Confidence: high
- History: present since duplicates were added (`2527c47f`, 2021-05-20); the `.copy()` of 2026 preserved the shape.
- What the code does: `self.y_orig[idx]` with integer `idx` is a `(1,)` array, so `__call__`, `add` and `_record` return `np.float64` on a first evaluation and `ndarray` of shape `(1,)` on a repeat. Reproduced: `float64 ()` then `ndarray (1,)`.
- Consequence if real: none inside PyVBMC — `active_sample.py:696` discards the value and uses `idx_new` — and `batch_call` normalizes with `.item()`. An external caller, or a future reader that puts the value into a scalar slot, would see a shape surprise.
- Suggested reproduction: ran it — `logger_checks.py`, section 1.
- Test adequacy: no. `test_record_duplicate*` compare values with `np.isclose`, which accepts both shapes.

### F9. The duplicate-pooling fallback for extreme SDs has no MATLAB counterpart
- Location: `pyvbmc/function_logger/function_logger.py:663-707`; MATLAB: `misc/funlogger_vbmc.m:234-238`.
- Category: formula/gradient
- Proposed classification: possibly intentional (Python-only robustness guard, absent from the sheet)
- Confidence: high (the code), low (that it should be reported as a difference)
- History: recent; the MATLAB three-line form is unchanged since 2019.
- What the code does: in the ordinary case PyVBMC computes MATLAB's `tau_n`, `tau_1`, the precision-weighted mean and `1/sqrt(tau_n+tau_1)` — verified identical to the last bit (`-2.0588235294117645`, `0.48507125007266594` for SDs 2.0 and 0.5). When any intermediate is non-finite or non-positive (an SD below ≈1e-154 makes `1/S²` overflow), PyVBMC falls back to a scale-invariant reweighting; MATLAB produces Inf/NaN there.
- Consequence if real: PyVBMC survives where MATLAB writes NaN into `y_orig`. No effect on any representable ordinary case.
- Suggested reproduction: two `add` calls at the same point with `f_sd = 1e-170` and `1e-160`; the MATLAB expression divides by zero (I hit `ZeroDivisionError` reproducing the MATLAB arithmetic in plain Python, which is the same overflow MATLAB turns into Inf).
- Test adequacy: `test_record_duplicate_extreme_finite_sds` covers the fallback and mirrors the implementation's intent; `test_record_duplicate_f_sd` covers the ordinary case against the closed form.

### F10. `warp_input`'s plausible-bound quantiles use NumPy's convention, not MATLAB's
- Location: `pyvbmc/whitening/whitening.py:198`; MATLAB: `misc/warp_input_vbmc.m:85-86`.
- Category: formula/gradient
- Proposed classification: possibly intentional (same substitution the sheet records for gpyreg's mean functions, but not recorded for this call site)
- Confidence: high (the difference), low (that it matters)
- History: unchanged on both sides since the whitening port.
- What the code does: MATLAB's `quantile` interpolates at `(i-0.5)/n`, `np.quantile` at `(i-1)/(n-1)`. With `n = 1e5` the two read the empirical distribution about half an order statistic apart.
- Consequence if real: the post-warp `plb_tran`/`pub_tran` differ by a Monte-Carlo-scale amount. Since the 1e5 draws themselves come from a different random stream (sheet: RNG entry), this can never be compared point by point anyway.
- Suggested reproduction: not run; it needs only the two quantile definitions on the same sample.
- Test adequacy: no test checks the post-warp plausible bounds against MATLAB; `test_warp_input` checks only `R_mat` and `scale`.

---

## 3. Test adequacy notes

- **The whitening tests never touch a bounded variable.** Every test in `test_rotoscaling.py` builds `VBMC` with `lb = -inf`, `ub = +inf`, so `parameter_transformer.type == [0, 0]` and the assertions say so explicitly (`:216`, `:503`). The interaction that carries the most risk in this slice — a bounded transform composed with a rotation and a rescaling, in the direct map, the inverse and the log-Jacobian — is exercised only by `test_parameter_transformer_log_abs_det`, which does test it against MATLAB values but at a single point.
- **`warp_input`'s stored-point update is never executed with data.** All four `test_warp_input*` tests call it on a freshly constructed `VBMC` whose function logger has no evaluations, so lines 213-220 of `whitening.py` (the `X`/`y`/Jacobian rewrite, and F4) run on empty arrays. The MATLAB-derived fixtures cover `R_mat`, `scale` and the `warp_gp_and_vp` outputs only.
- **`test_boundary_edge_cases` mirrors the implementation.** It asserts that near-bound points transform to finite values and round-trip to `np.nextafter`-built neighbours — which is precisely what F1's nudge and F2's clamp produce, and would fail against MATLAB's own behavior. It is a reasonable regression test for the Python contract, but it is not evidence that the boundary handling is MATLAB's, and the surrounding docstrings do not say which one it is.
- **`test_log_abs_det_jacobian_logit_extreme_tails` compares the implementation with itself** (`legacy = -u + 2*(-log1p(exp(-u)))` in the ordinary range, `-|u| - 2*log1p(exp(-|u|))` in the tails, both algebraically the branch under test). The values happen to be right, and the finite-difference module `test_parameter_transformer_jacobian_fd.py` is the real gate here; the tail test's own assertions could not detect a wrong tail formula.
- **The value tests are specification-based where it counts.** `test_direct_transform_type3_within*` and the inverse tests compare against `scipy.stats.norm.ppf` / `t.ppf` / `t.cdf` rather than stored outputs, and `test_bounded_log_abs_det_jacobian_numerically` checks the Jacobian against a finite volume ratio. That is the right shape of test, and it is why F3 (1 ulp, 1e-11 in the tail) is invisible to it — correctly so.
- **`test_unscent_warp` and `test_warp_gp_and_vp` carry real MATLAB reference values**, at `atol=1e-4` and `1e-5`. They are the strongest tests in the slice. They pin the unscented transform, the length-scale geometric mean, the quadratic-mean warp and the `mu`/`sigma`/`lambda`/`w` updates — all of which I read line by line and found to agree.
- **Nothing tests the logger against MATLAB.** `test_function_logger.py` is thorough on Python contracts (shapes, validation, the batch path, finalize) but contains no MATLAB-derived value and no test of the pooling formula against `funlogger_vbmc.m`'s expression. F7 and F8 survive because of that.

---

## 4. Defects on the MATLAB side

1. **`misc/funlogger_vbmc.m:244` writes the duplicate's value to the wrong row.** In the duplicate branch of `record`, `optimState.y(optimState.Xn) = fval;` should be `optimState.y(idx) = fval;`. It corrupts the most recently added row's transformed value with the repeated point's value and leaves `y(idx)` stale, which then feeds the GP through `get_traindata_vbmc.m` and `ymax`. It fires on every repeated observation, i.e. on every noisy run that re-evaluates a point. Last touched by `774327b` (2020-11-01, "fixed tempering bug"), so it predates the Python port; PyVBMC has used `idx` since `2527c47f` (2021-05-20) and never reproduced it.
2. **The `'add'` action crashes on a noisy run.** `misc/funlogger_vbmc.m:159-162` reads `fsd = varargin{2}` whenever `isfield(optimState,'S')`, but both call sites pass only the value: `private/activesample_vbmc.m:388` and `misc/initdesign_vbmc.m:56` call `funlogger_vbmc(fun,x,optimState,'add',y_orig)`. A noisy run that supplies cached function values (`options.Fvals`) therefore raises "Index exceeds the number of array elements" instead of defaulting the SD to 1. PyVBMC's `add(x, f_val_orig, f_sd=None)` defaults it, as the MATLAB line `if isempty(fsd); fsd = 1; end` intends.
3. **`misc/warp_input_vbmc.m` never refreshes `optimState.ymax` after rewriting `optimState.y`.** It is saved by `vbmc.m:632-634` recomputing it at the head of active sampling; PyVBMC has the same gap and the same rescue (`vbmc.py:1428`). Not a difference between the two, noted only because it looks like one.
4. **`misc/setupvars_vbmc.m:96` sets `vp.temperature = NaN`**, which `misc/warp_gpandvp_vbmc.m:8-10` would read as `T = NaN` (the guard tests `~isempty`, not `~isnan`). In practice `misc/vpoptimize_vbmc.m:190` overwrites it with `optimState.temperature` before any warp can occur, so it never fires; it is one `vpoptimize` call away from turning every warped hyperparameter and weight into NaN.
