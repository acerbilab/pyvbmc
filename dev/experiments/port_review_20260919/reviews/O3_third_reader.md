# O3 — third reader: the parameter transformations, their Jacobians and the warps

Raw reviewer report, wave 7 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-23. One Opus agent, fresh
context, read-only on the repositories, combining both tracks: it re-derived
each formula of the slice, checked it numerically, and then compared the
code with MATLAB VBMC at `396d649`. It read PyVBMC at `a65b96f4`, in a
worktree whose package code stayed at that commit while the merged waves
were gated in the main checkout, with gpyreg at `v1.3.0` (`0186d89`). Its
brief carried three first questions, which the report answers before its
coverage: whether any Python computation needs the gradient of the log
Jacobian that MATLAB's `warpvars_vbmc.m` provides; what precision the
bounded transforms keep near a hard bound, a nonzero bound in particular,
and whether MATLAB evaluates the same expressions; and whether a warp
re-expresses every quantity stored in the old transformed space rightly. Its
check scripts ran in the session scratchpad and are kept only on the machine
that ran them (`dev/scripts/runs/LOCAL.md`). The text below is the agent's
final message, unedited.

---

# O3 — third reader: the parameter transformations, their Jacobians and the warps

All scripts and logs are in `C:\Users\luigi\AppData\Local\Temp\claude\C--Users-luigi-Documents-GitHub-pyvbmc\30d85518-7e4d-42ed-9a16-74a7fb47ace3\scratchpad\wave7\O3`. Every script ran with the wave7 worktree on `PYTHONPATH` and BLAS single-threaded. I checked `pyvbmc.__file__` (under `pyvbmc-wave7`) and `gpyreg.__file__` (under `gpyreg`).

**Summary.** There are seven findings. Only one can stop a run: a run with `gp_mean_fun="zero"` fails at its first warp (F2). The rest are precision losses with negligible effect, a search-box approximation that gets worse with dimension, a resume issue, and two errors in the known-differences sheet.

**Answers to the three questions.**
- **Q1 (gradient of the log Jacobian): no path needs it, on either side** (F1).
  - MATLAB's `'g'` action does not return that gradient. It returns the per-coordinate derivatives of the inverse map.
  - Its only caller is dead code.
  - No Python computation needs d log|J|/du. The optional torch export gets it through autograd.
- **Q2 (precision near a bound):**
  - At a lower bound, logit and probit keep full relative precision, zero or nonzero bound alike.
  - At an upper bound, all three maps resolve b−x only to (b−a)·eps/2. This is a real loss when |b| ≪ b−a, including b = 0 with a < 0 (F4).
  - The student4 inverse loses precision at a zero lower bound (F4), and its forward map loses precision at the middle of the interval (F3).
  - log|J| is exact as a function of u.
  - MATLAB evaluates the same expressions.
- **Q3 (re-expression at a warp):**
  - For the rotoscaling warp, `warpfun` is affine and the change in log|J| is one constant C = −log|det A|.
  - Exact: the stored X and y (shift C), the mean function's m0 and x_m, the constant mean, the VP means, the weights, and the search cache.
  - Approximated as MATLAB approximates: length scales, ω and the VP scales use marginal standard deviations, so correlations are dropped.
  - The search box shrinks with D (F5).
  - A zero-mean GP is refused (F2).
  - The recorded state of a warp iteration keeps the pre-warp `hyp_dict` (F6).

## 1. Coverage

**Read completely (worktree):**
- `pyvbmc/parameter_transformer/parameter_transformer.py`
- `pyvbmc/whitening/whitening.py`
- `pyvbmc/vbmc/vbmc.py`, only where it touches this slice:
  - construction and transformer set-up: 540-640, 1040-1260
  - the warp block and what follows it: 1370-1660, 1680-1830
  - the warm-up check that uses `lcb_max`: 2090-2170
  - `load`: 3225-3375, `_optim_state_record`, `_validate_gp_mean_fun_option`
- `function_logger.py` `_record` (680-790)
- `variational_posterior.py`: `mode`, `kl_div`, `get_parameters`/`set_parameters`, `__init__`
- `variational_posterior/_torch.py`
- `svbmc/_jacobian.py`; `svbmc/_entropy.py` (50-112)
- `gaussian_process_train.py` (60-240)
- `active_sample.py` (360-380, 860-920)
- gpyreg `mean_functions.py` (`NegativeQuadratic` layout) and the hyperparameter ordering in `GP`
- Sheet: section P8, plus the entries it cross-references (quantile, temperature, `vp.pdf`, soft bounds, cached points, mean families)

**MATLAB, read completely:** `shared/warpvars_vbmc.m`, `misc/warp_input_vbmc.m`, `misc/warp_gpandvp_vbmc.m`, `utils/unscent_warp.m`, `vbmc_pdf.m`, `vbmc_mode.m`.

**MATLAB, read in part:** `vbmc.m` 525-640; `setupvars_vbmc.m` 30-70 and 280-300; `boundscheck_vbmc.m` 130-145; `gplite_meanfun.m` header and 50-80; `activesample_vbmc.m` 340-390. I also listed every MATLAB caller of `warpvars_vbmc`.

**Tests, read in part:** `test_rotoscaling.py`; `test_parameter_transformer.py` (600-760); the header and parametrization of `test_parameter_transformer_jacobian_fd.py`.

**Not reached:** MATLAB `funlogger_vbmc.m` (I relied on the sheet), the internals of `vp.pdf`, and the rest of the transformer tests.

## 2. Derivations

"FD" means a finite-difference check.

| Quantity | Independent derivation (key identity) | Numeric check | MATLAB | Verdict |
|---|---|---|---|---|
| Unbounded map | u=(x−μ)/δ; inverse uδ+μ | round trip in `check_transforms.py` | same | agrees |
| Logit map | u=(logit z−μ)/δ, z=(x−a)/(b−a); inverse a+(b−a)σ(uδ+μ) | round trip ≤3e-15 relative | same, plus the nudge/clamp in the sheet | agrees; precision F4 |
| Probit map | u=(Φ⁻¹(z)−μ)/δ, with Φ⁻¹(z)=−√2 erfcinv(2z) and Φ(y)=½erfc(−y/√2) | against scipy ndtri/ndtr, ≤9e-16 | same | agrees; precision F4 |
| Student-t(4) CDF | F(t)=½+(3r−r³)/4 with r=t/√(t²+4), algebraically equal to the code | against scipy, 2e-16 | same (one-ulp grouping, in the sheet) | agrees; lower tail F4 |
| Student-t(4) quantile | sign(z−½)·2√(q−1), q=cos(acos α/3)/α, α=√(4z(1−z)) (Shaw); stable form r=2 sin(asin(2z−1)/3), t=2r/√(1−r²) | against the stable form | same expression | F3 |
| log\|det J\| | Σ_i[log(b−a)+log g′(y_i)+log δ_i]+Σ log s_i, with \|det R\|=1; logit −y−2log1p(e^{−y}); probit −½log2π−y²/2; t4 log(3/8)−(5/2)log1p(y²/4) | FD of the full D×D Jacobian of `inverse`: all three types, with and without R/scale, max error 1.8e-8 at step 1e-6. The t4 pdf constant equals log(3/8) exactly | same; Python adds a tail branch (sheet) | agrees |
| Rotation and scale | u=vR/s, v=(u⊙s)Rᵀ | round trip | same (MATLAB skips scale when all equal 1: same values) | agrees |
| Centering | μ=½(g(plb)+g(pub)), δ=g(pub)−g(plb), taken before R/s | covered by the tests | matches the post-a5240d2 (2021-02-02) version | agrees |
| Unscented transform | points x±√Dσ_d e_d, equal-weight mean, ddof=1 std; exact mean and marginal sd for affine maps | ≤4e-16 | same (reshape order does not matter, the map is row-wise) | agrees |
| Whitening | Σ_raw=diag(δ)R diag(s)Σ_u diag(s)Rᵀdiag(δ); SVD UΣUᵀ; R'=U, s'=√(S+eps), μ'=0, δ'=1, so AΣAᵀ=I | AΣAᵀ=I to 2e-15, on a first warp and on a warp from an already rotated space; `vp.moments` checked against the mixture formula, 4e-16 | same, after f9c04bc (2021-02-02) | agrees |
| Change of log\|J\| in a warp | constant C=Σlog s'−Σlog s−Σlog δ=−log\|det A\| | spread 1e-9 (probit) / 7e-13 (logit) across points; equals the formula | same | agrees |
| Stored log joint | y_new=y_orig+logJ_new(X_new)=y_old+C | exact to 1e-11 | same | agrees |
| GP length scales | ℓ'=diag(A diag(ℓ²)Aᵀ)^{1/2}, a marginal approximation of a non-diagonal metric | matches the code to 1e-16 | same | agrees (approximation) |
| Negative-quadratic mean | m0'=m0+C exact; x_m'=Ax_m+c exact; ω' marginal, which underestimates curvature (used 1/diag(P⁻¹) ≤ exact diag(P); example 0.237 vs 0.851) | exact where exact | same | agrees (approximation) |
| Constant mean / zero mean | m0'=m0+C / nothing to re-express | const exact (4e-16); zero raises | MATLAB `case 0` mislabels | F2 |
| Noise, output scale | unchanged under a constant shift of y (S unchanged too) | checked | same | agrees |
| VP parameters | μ' exact; σ'λ' = marginal sd exactly; Σλ'²=D; correlations dropped (KL 0.18 in a D=3 example) | ≤5e-15 | same | agrees (approximation) |
| VP weights | component masses are invariant, so w'=w; the code's factor exp(ΔlogJ)=e^C is constant | \|w'−w\|≤1e-15 | same | agrees. MATLAB-side remark: for a nonlinear warp this factor would not preserve masses; unreachable on both sides |
| Plausible box after a warp | heuristic: 5–95% quantiles of the image of uniform draws, ±1/9 of the width | — | same (quantile convention in the sheet) | agrees |
| Search box after a warp | intended: bounding box of the affine image, half-width \|A\|h; code: 1000 draws | coverage vs D | same | F5 |
| Search cache | warpfun applied | exact | same | agrees |
| d log\|J\|/du (Q1) | per coordinate (1−2σ(y))δ, −yδ, −(5/4)yδ/(1+y²/4), then R | — | MATLAB's `'g'` is something else | F1 |

## 3. Findings

### F1. The sheet's entry on "the gradient of the log Jacobian" misdescribes MATLAB; neither side needs that gradient
- **Location:**
  - `dev/experiments/port_review_20260919/known_differences.md:2402-2413`
  - `pyvbmc/parameter_transformer/parameter_transformer.py:292`
  - `pyvbmc/variational_posterior/variational_posterior.py:869-872`
  - MATLAB: `shared/warpvars_vbmc.m:463-477, 763-768`; `vbmc_pdf.m:116-120`; `vbmc_mode.m:38-42`
- **Category:** cross-module
- **Proposed classification:** possibly intentional. Leaving it out is harmless; the entry's description is what is wrong.
- **Confidence:** high
- **History:**
  - The `'g'` branch has existed since before the port (present at abbf946, 2020-12-05) and has not changed.
  - The dead caller in `vbmc_pdf.m` dates from 2018 (b37b519) and was renamed in 2019 (1e17fb8).
  - The Python never had this action.
- **What the code does:**
  - `'g'` shares the `'p'/'l'` branch. It fills the per-coordinate log-Jacobian terms, skips `log(scale)` and the sum over coordinates, and exponentiates.
  - So it returns the N×D matrix dx_i/dy_i of the inverse map, in pre-rotation coordinates. That is not ∇log|J|.
  - Its only caller, `vbmc_pdf.m:119`, comes right after an unconditional `error` (117-118), so it never runs. Had it run, it would subtract Jacobian factors from a log-density gradient, in the wrong coordinates (a MATLAB-side defect, dead).
  - `vbmc_mode` optimizes with `GradObj` off.
- **Python side:**
  - The GP, the acquisitions and the ELBO all work in transformed space. There log|J| enters only as data inside y.
  - `vp.pdf` refuses original-space gradients.
  - `vp.mode(orig_flag=True)` uses `minimize(jac=False)`, which is gradient-free, as in MATLAB.
  - S-VBMC optimizes its weights over fixed draws, where log|J| is a constant.
  - The only differentiation of log|J| is autograd in the torch export (`_torch.py:184-203`), whose formulas match the NumPy ones.
- **Consequence if real:** none on results. The open question resolves to "not needed", and the entry's premise is inaccurate.
- **Suggested reproduction:** static reading of the lines above.
- **Test adequacy:** not applicable.

### F2. `warp_gp_and_vp` refuses a zero-mean GP, so a run with `gp_mean_fun="zero"` stops at its first warp; MATLAB's branch labels are off by one mean code
- **Location:** `pyvbmc/whitening/whitening.py:435-472` (the raise at 471-472). MATLAB: `misc/warp_gpandvp_vbmc.m:37-66` (`case 0` at 38, `otherwise` at 65); `gplite/gplite_meanfun.m:57-62`.
- **Category:** control flow
- **Proposed classification:** suspected defect in both
- **Confidence:** high
- **History:**
  - MATLAB's `case 0` has been there since 2a076c9 (2020-06-16) and was not changed after the port began.
  - The Python branches have been `isinstance` tests since the whitening port (4aa4ca97, 2022-03; a5f6effb, 2022-10). The Python never followed MATLAB's numbering; it attached the branch to `ConstantMean`, which is what MATLAB's comment says the branch is for.
- **What the code does and why it is wrong:**
  - `VBMC` accepts `"zero"`, `"const"` and `"negquad"` (`vbmc.py:3905`).
  - A zero-mean GP has nothing to re-express beyond its length scales, yet the Python raises "Unsupported GP mean function for input warping."
  - Nothing catches the exception. `doWarping` (`vbmc.py:1398-1415`) does not look at the mean function.
  - In MATLAB, `gp.meanfun` is numeric: 0 zero, 1 const, 4 negquad. The branch commented "Warp constant mean" is `case 0`, which is the zero mean. It reads `hyp(Ncov+Nnoise+1)`, one past the end of a zero-mean hyperparameter vector: an index error, or with an output warp it overwrites that warp's first hyperparameter.
  - In MATLAB, `'const'` falls to `otherwise` and errors.
  - Net effect: MATLAB stops for both `'zero'` and `'const'`; PyVBMC handles `'const'` exactly and stops for `'zero'`.
- **Consequence if real:** a run with `gp_mean_fun="zero"`, the default `warp_rotoscaling`, and D≥2 ends in a ValueError once a warp triggers. That needs K≥5, r_index<3, and more than 5 iterations past warm-up. It is not reached at default options, noiseless or noisy.
- **Suggested reproduction:** `check_warp.py` gives "ValueError: Unsupported GP mean function for input warping." for `ZeroMean`. The same state with `ConstantMean` gives m0'−m0−C = −4e-16.
- **Test adequacy:** `test_warp_gp_and_vp` covers `NegativeQuadratic` only. `test_gp_training_policy` parametrizes `"zero"` but never warps.

### F3. The student4 forward map loses precision near the middle of the interval
- **Location:** `parameter_transformer.py:636-643`; MATLAB `shared/warpvars_vbmc.m:265-269`
- **Category:** formula/gradient
- **Proposed classification:** suspected defect in both
- **Confidence:** high
- **History:** the MATLAB lines have not changed since 9da4afd (2020-05-01). The Python has been identical to them since 4ce90e25 (2022-07-28).
- **What the code does and why it is wrong:**
  - The code forms q−1 ≈ (16/9)(z−½)² by cancellation. So the absolute error in t is about eps/|t|.
  - α rounds to 1 for |z−½| ≲ 1e-8, and the code then returns t=0 instead of (8/3)(z−½).
  - A cancellation-free form, r=2 sin(asin(2z−1)/3) and t=2r/√(1−r²), follows from F−½=(3r−r³)/4. `_torch.py:23-39` already uses it.
- **Measured:**
  - Maximum absolute error 1.55e-8 over z∈½±1e-3.
  - At z−½=1e-6 the code gives 2.666767e-6 against 2.666667e-6.
  - The x→u→x round trip moves points near the midpoint by up to 9.5e-9·(b−a).
- **Consequence if real:** negligible. student4 is not the default. The error applies wherever a stored original-space point is transformed (a warp's re-transform of `X_orig`, `vp.pdf(orig_flag=True)`, S-VBMC), as a displacement ≤1.6e-8/δ in u.
- **Suggested reproduction:** `check_student4.py`, `check_center.py`.
- **Test adequacy:** the round-trip tests use z=0.5025 and 0.52 with rtol loosened to 1e-11.

### F4. Near a bound: all three maps resolve the distance to the upper bound only to (b−a)·eps/2; the student4 inverse also loses it at the lower bound (answer to Q2)
- **Location:** `parameter_transformer.py:581-598, 609-633, 646-650`. MATLAB: `warpvars_vbmc.m:106-108, 256-258, 265-269, 318-319, 442-443, 450-453, 457-459`.
- **Category:** formula/gradient
- **Proposed classification:** suspected defect in both (precision only)
- **Confidence:** high
- **History:** the expressions are the same on both sides since the port. The Python adds the nudge and the clamp (in the sheet).
- **What the code does:**
  - Every map goes through z=(x−a)/(b−a). That carries x−a with full relative precision (Sterbenz), but 1−z only to an absolute eps/2.
  - At the lower bound, logit and probit keep full precision, zero or nonzero bound alike. Forward error is ≤2e-16, inverse ≤1e-14, and the round trip is within ~150 ulps, which is just the conditioning of storing u.
  - At the upper bound, when |b| ≪ b−a, precision drops.
    - On [−1000,1] at (b−x)/(b−a)=1e-8, the forward relative error is 3e-10 and the inverse error in b−x is 1e-8.
    - On [−1,0], b−x=1e-12 gives forward 8e-7 and inverse 9e-5.
    - On [−1,0], b−x=1e-20 comes out as the nudged u=36.7 instead of 46.05. The inverse of the correct u returns −4.9e-324, i.e. the round trip moves the point to the bound's neighbour.
  - Non-issue for comparison: bounds with |b| ≳ b−a, e.g. [10,11], lose nothing.
  - The student4 inverse computes ½+(−½+small). With a=0, the relative error of x−a is 7e-10 at t=−100 and 1.5e-5 at t=−1e3. At 1e-18·(b−a), a round trip returns 1.1e-16·(b−a).
  - log|J| as a function of u is exact (≤1.4e-14). So y stays consistent with the u the GP sees; only which x a given u denotes is off.
  - The torch export computes distances to each bound separately and is accurate here, so the two implementations differ in this region.
- **Consequence if real:** negligible. Only points within ~1e-10·(b−a) of such an upper bound, or ~1e-8·(b−a) of a zero lower bound under student4, are affected.
- **Suggested reproduction:** `check_bounds.py` and `check_zero_upper.py` (plus their logs).
- **Test adequacy:** no test works near a bound (see §4).

### F5. The warped search box covers a D-dependent fraction of the image of the old box
- **Location:** `whitening.py:333-347`; MATLAB `warp_input_vbmc.m:133, 143-148`
- **Category:** state/caching
- **Proposed classification:** suspected defect in both
- **Confidence:** medium
- **History:** MATLAB since 2a076c9 (2020-06). Python since 4aa4ca97 (2022-03). The Python's choice of transformer changed in 7d93f60 (sheet).
- **What the code does:**
  - The exact image of the box under u'=Au+c has half-width |A|h.
  - The code takes the min and max of 1000 uniform draws and adds a margin of range/1000.
  - Median coverage per coordinate: 0.97 at D=2, 0.92 at D=3, 0.82 at D=5, 0.66 at D=10, 0.55 at D=15, 0.49 at D=20. The minimum at D=20 is 0.40 (`check_searchbox.py`).
- **Consequence if real:**
  - Every kept warp narrows the acquisition search, and the shrinkage compounds across warps.
  - The 5% edge-expansion rule (`active_sample.py:884-899`) only re-widens the box gradually.
  - The default box (plausible box ±2 widths) is wide, so the effect is probably small.
  - It is reached at default options whenever D>1 and a warp is kept.
- **Suggested reproduction:** `check_searchbox.py`.
- **Test adequacy:** no test checks coverage.

### F6. The `optim_state` recorded at a warp iteration keeps the pre-warp `hyp_dict`, so a run resumed there starts from pre-warp GP hyperparameters
- **Location:** `whitening.py:219`; `vbmc.py:1433-1447, 1529-1531, 1571, 1585, 1941, 3252-3253`
- **Category:** state/caching
- **Proposed classification:** unsure (Python-only mechanism)
- **Confidence:** medium (static reading; not run)
- **History:** Python-only; the resume feature is a5f6effb (2022-10).
- **What the code does:**
  - In ordinary iterations, `optim_state["hyp_dict"]` is the same object as `self.hyp_dict`.
  - `warp_input` deep-copies `optim_state`, so from then on the copy holds the previous iteration's old-space values. Meanwhile `self.hyp_dict` receives `hyp_warped` and then the new fit.
  - Active sampling, which would re-link the two, is skipped in the warp iteration.
  - `VBMC.load(file, iteration=k)` then pairs the old-space `hyp_dict` (hyperparameters and `run_cov`) with `get_gp(k)`, which is in the warped space.
  - An iteration whose warp was undone gets a same-space version of the same staleness, one fit behind.
- **Consequence if real:** resuming from such an iteration, even with `set_random_state=True`, does not continue the original trajectory. The GP fit's starting points and slice widths differ. The effect on quality is small.
- **Suggested reproduction:** save a run with a kept warp. Compare `load(..., iteration=k).hyp_dict["hyp"]` with `get_gp(k)`'s hyperparameters, then resume and compare the next ELBO.
- **Test adequacy:** no test resumes from a warp iteration.

### F7. The sheet's reason for the probit default cites text that AGENTS.md no longer holds
- **Location:** `known_differences.md:2383-2400`; `advanced_vbmc_options.ini:311`
- **Category:** defaults
- **Proposed classification:** possibly intentional (the default is deliberate; only its cited reason fails)
- **Confidence:** high
- **History:**
  - The "probit by default" sentence was in AGENTS.md from 44f0af68 until the rewrite in af8fb783 (2026-09-21), which removed it.
  - The actual origin of the default is 6cee9bb5 (2022-11-24): "feat: change default bounded variable transform to probit."
- **Consequence if real:** documentation only.
- **Suggested reproduction:** `git log -S"probit by default" -- AGENTS.md`.
- **Test adequacy:** not applicable.

## 4. Test adequacy notes
- `test_inverse_type3_max_space`: its probit/student4 loop copies the min-space case (Y=−500, expects −10). So the upper saturation of those two transforms is never tested.
- `test_transform_*_largeN`: `10 ^ 6` is XOR, which gives 12 rows, and all rows are identical.
- The round-trip tests use only interior points, with loosened rtol. Nothing tests near a bound (F4) or near the midpoint for student4 (F3).
- `test_warp_gp_and_vp`:
  - It is a MATLAB reference on an unbounded 2-D problem, with the negative-quadratic mean only.
  - No constant or zero mean (F2).
  - No check that the m0 shift equals the logger's y shift on a bounded problem.
- `test_warp_input_rewrites_every_filled_row_of_the_logger` follows the specification (the new transform and the new log|J|). It is good.
- No test covers search-box coverage (F5) or resuming from a warp iteration (F6).
- The Jacobian FD test is thorough (all types, centered, mixed, with warps) but draws its points "well inside" the space. So the logit tail branch beyond y=−709.78 is never tested.
