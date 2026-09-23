# Wave 7 verification, slice O3: the parameter transformations, their Jacobians and the warps

Raw report of a verifier, wave 7 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-23. One Opus agent,
read-only on the repositories, verified the findings and the test notes of
the third reader of slice O3 (`../reviews/O3_third_reader.md`) and its
answers to the three first questions of its brief, reading PyVBMC's
package code at `a65b96f4`, gpyreg at `v1.3.0` (`0186d89`) and MATLAB VBMC
at `396d649`, with no `optimize()` run; its check of F6 replays the
statements of `optimize` around a warp with the package's own functions.
Its scripts are kept as `scripts/wave7_O3_*.py`, under the names the report
gives them with `wave7_` in front; their logs are kept on the machine that
ran them (`dev/scripts/runs/LOCAL.md`, "Port correctness review"). The text
below the rule is the agent's final message, unedited; `wave7.md` holds the
consolidated ledger and the dispositions.

---

# Wave 7 verification, slice O3: the parameter transformations, their Jacobians and the warps

## 1. Revisions and interpreter

- PyVBMC: worktree `C:\Users\luigi\Documents\GitHub\pyvbmc-wave7`. HEAD is now `dd079de1`, because two docs-only commits (the O4 and O1 verifications) landed after `df0f502b`. `git diff a65b96f4 HEAD -- pyvbmc` is empty, so the package is the one the reviewer read.
- gpyreg: `0186d89` (`v1.3.0`).
- MATLAB VBMC: `396d649`.
- Interpreter: `C:\Users\luigi\Documents\GitHub\pyvbmc\.venv\Scripts\python.exe`, with BLAS single-threaded. Every script printed:
  - `pyvbmc: C:\Users\luigi\Documents\GitHub\pyvbmc-wave7\pyvbmc\__init__.py`
  - `gpyreg: C:\Users\luigi\Documents\GitHub\gpyreg\gpyreg\__init__.py`
- Torch is not installed in this interpreter, so statements about the Torch export rest on reading its code.
- MATLAB statements are read from the source; none was run.
- My scripts and their logs are in `C:\Users\luigi\AppData\Local\Temp\claude\C--Users-luigi-Documents-GitHub-pyvbmc\30d85518-7e4d-42ed-9a16-74a7fb47ace3\scratchpad\wave7\verify_O3\`:
  - `O3_F1_log_jacobian_gradient.py`
  - `O3_F2_zero_mean_warp.py`
  - `O3_F3_F4_precision.py`
  - `O3_F3_round_trip_fine.py`
  - `O3_F5_search_box_after_warp.py`
  - `O3_F5_two_warps.py`
  - `O3_F6_hyp_dict_at_warp.py`
  - `O3_Q3_warp_reexpression.py`

## 2. Headline

All seven findings reproduce as statements about the code.

| Finding | Classification | What changes against the reviewer's reading |
|---|---|---|
| F1 | documentation only | The sheet entry misdescribes MATLAB. There is also a dead MATLAB line. |
| F2 | confirmed shared defect for `gp_mean_fun="zero"`; confirmed MATLAB-side defect for `'const'` | Adds: the ValueError leaves the instance half-warped. |
| F3 | confirmed shared defect (precision, student4 only) | None. |
| F4 | confirmed shared defect (precision) | This is the open wave-0 question (N2 rerun F3), now answered. Loss beyond x's own representation happens only at the upper bound, by a factor of about (b−a)/(2\|b\|). A nonzero lower bound loses nothing. |
| F5 | confirmed shared defect, of no consequence | The extent figures hold, but the box loses at most about 2D/1001 of the old box's volume, all in far corners. The posterior stays at least 8.7 SD inside. The shrinkage does not compound across warps. |
| F6 | confirmed Python-only defect, narrower than stated | Only a **kept** warp leaves a stale record. The undone-warp case is wrong. |
| F7 | documentation only | None. |

## 3. Ledger

| id | report | statement | verdict (class) | fires at defaults? | how verified | dating and recorded reason | already recorded? | note for disposition |
|---|---|---|---|---|---|---|---|---|
| O3-1 | F1 | The sheet entry "The gradient of the log Jacobian is not ported" (`known_differences.md:2402-2413`) says `warpvars_vbmc.m` returns that gradient. It does not. The `'g'` action (`shared/warpvars_vbmc.m:463-477`, `:763-768`) shares the `'p'`/`'l'` branch, has `logpdf_flag` false, writes the per-coordinate log terms, skips `log(scale)` and the sum, then exponentiates. The result is the N×D matrix of dx_i/dv_i of the coordinate-wise inverse, at the pre-rotation coordinates. Its only caller, `vbmc_pdf.m:119`, sits after an unconditional `error` (`:117-118`). No PyVBMC path needs d log\|J\|/du: `vp.pdf` refuses original-space gradients (`variational_posterior.py:869-872`) and `mode(orig_flag=True)` supplies no analytic gradient. | documentation only (plus one dead MATLAB-side line) | n/a (no computation) | `O3_F1_*`: my transcription of `'g'` matches the FD dx_i/dv_i of the Python inverse to 4.9e-9, 6.5e-9 and 1.9e-8 relative (logit, probit, student4), on D=3 with R and scale. It differs from the FD ∇_u log\|J\| by up to 10.5. The report's closed-form ∇log\|J\| matches FD to 1.1e-9 to 4.3e-9, including a point 1e-6 from a bound. | `'g'` since `a1680e3` (2018-05-04, as `pdftrans.m`). The dead caller and its guarding `error` since `b37b519` (2018-05-09); renamed in `1e17fb8` (2019). Unchanged since the port began. Python never had it. The entry's premise comes from the plan's O3 row (`dev/plans/port-correctness-review.md:157`), entry `0e4e27a5`. | Entry only, marked as an open question. `matlab_side_defects.md` row 9 is the non-log branch of `vbmc_pdf.m`, not this line. | Rewrite the entry (text in §8) or withdraw it. Optionally list the dead MATLAB line. |
| O3-2 | F2 | `warp_gp_and_vp` (`whitening.py:435-472`) handles `ConstantMean` and `NegativeQuadratic` and raises `ValueError("Unsupported GP mean function for input warping.")` for `ZeroMean`, which `VBMC` accepts (`vbmc.py:3905`; `.ini:108` documents it). `doWarping` (`vbmc.py:1398-1415`) ignores the mean, and nothing in `optimize()` catches the error. In MATLAB, `gp.meanfun` is numeric: `gplite_meanfun.m:57-62` and `:347`, stored by `gplite_post.m:120-121`, with 0 zero, 1 const, 4 negquad. `warp_gpandvp_vbmc.m:38` `case 0`, commented "Warp constant mean", reads `hyp(Ncov+Nnoise+1)`, which is past the end for a zero mean. `'const'` falls to `otherwise` (`:65-66`) and errors. | **zero:** confirmed shared defect (both stop). **const:** confirmed MATLAB-side defect (Python right). | No: the default is `"negquad"`. Yes for `gp_mean_fun="zero"` with every other option at default, noiseless or noisy, on any D≥2 run that reaches a warp (`warp_rotoscaling` True, K≥5, r_index<3, iteration − last_warping > 5·max(1, warping_count)). With `warmup=False`, `last_warping` starts at −inf. | `O3_F2_*`: zero gives the ValueError. Const and negquad give m0'−m0 equal to the logger's y shift (−3.3586, spread 3.6e-15) to −4.4e-16. The MATLAB index arithmetic is transcribed: numel(hyp)=5, and `case 0` reads hyp(6). | MATLAB `case 0` since `2a076c9` (2020-06-16), touched by `774327b` (2020-11-01), both before the port. Python `isinstance` branches since `4aa4ca97` (2022-03-03); `gp_old.mean` since `a5f6effb` (2022-10-27). The Python never used MATLAB's numbering and attached the branch to the constant mean, as MATLAB's comment says. No recorded reason for refusing zero. | No. W3-12 / P5-8 narrowed the accepted names to three and kept `"zero"` as implemented. This finding goes further: `"zero"` also fails part-way, at the first warp. | Options: (a) skip the mean block for a zero mean and warp the length scales only (note that the stored log joint moves by C = −log\|det A\|, which a zero mean cannot follow); (b) refuse `"zero"` at construction (stricter interface), which also makes CHANGELOG `:476-480` true; (c) disable warping for a zero-mean run. MATLAB-side entry in §7. |
| O3-3 | F3 | `_student4` (`parameter_transformer.py:636-643`) forms q−1 by cancellation near z=½. It returns t=0 while α rounds to 1, measured for \|z−½\| up to 6.4e-9. MATLAB `warpvars_vbmc.m:265-269` has the same expression. `_torch.py:23-39` uses the stable form. | confirmed shared defect (precision) | No (student4 only) | `O3_F3_F4_*`: max abs error of t is 1.55e-8 over ½±1e-3, at z−½=−1e-8. At 1e-6 the code gives 2.6667667e-6 against 2.6666667e-6. `O3_F3_round_trip_fine`: the x→u→x round trip reaches 9.50e-9·(b−a). | MATLAB `9da4afd` (2020-05-01). Python `4ce90e25` (2022-07-28). The guard added in `9c71461d` (2023-07-02) changes no finite value. They never differed. | No (P8-15c is the inverse's grouping, a different line). | Adopt the Torch export's cancellation-free form (moves student4 values near the midpoint in the last digits), or leave it, since it is shared and negligible. |
| O3-4 | F4 | Every bounded map goes through z=(x−a)/(b−a), so b−x is resolved only to (b−a)·eps/2. That is a loss beyond the representation of x when \|b\| < b−a (worst at b=0). The lower bound keeps full precision. The student4 inverse `½+(−½+small)` also loses x−a at a zero lower bound. log\|J\| as a function of u is exact. | confirmed shared defect (precision) | Reachable at defaults (probit) on a bounded problem with stored points within about 1e-10·(b−a) of such an upper bound, noiseless or noisy. Visible only where a stored x is re-transformed (a warp; `vp.pdf(orig_flag=True)`; S-VBMC). | `O3_F3_F4_*`, all as reported:<br>• [−1000,1] at 1e-8: forward 3.3e-10, inverse 9.6e-9.<br>• [−1,0] at b−x=1e-12: logit 8.0e-7 / 8.9e-5.<br>• [−1,0] at b−x=1e-20: u=36.74 against 46.05; the inverse of the correct u gives x=−4.9e-324.<br>• [10,11]: exact.<br>• Lower bounds [0,1], [0,7], [3,5], [−5,2]: forward ≤2.2e-16, inverse within 1 ulp of x (1.7e-13 relative at 1e-200·(b−a), probit, which is tail conditioning).<br>• student4 at a=0: 6.9e-10 (t=−100), 1.5e-5 (t=−1e3); 1e-18 round-trips to 1.1e-16.<br>• log\|J\| against independent formulas: ≤1.4e-14. | MATLAB lines from `a1680e3` / `0b7adb91` / `9da4afd` (2018–2020). Python `4ce90e25` (2022-07-28); nudge and clamp `3e9d1c21` (2022-08-28, P8-15a/b). | **Yes:** `wave0.md:44` (N2 rerun F3, bounds (−2,3)), left open "until P8 and O3 say whether MATLAB shares it" (`plans/port-correctness-review.md:554-556`, `:999-1008`). P8 found the same expressions. This finding is the same one, sharpened, with the student4 lower tail added. | Answers the wave-0 question: shared. Options: leave it (list it among "left as they are in both"), or compute the distance to each bound separately, as the Torch export does. That second option moves bounded-problem numbers at the rounding level. |
| O3-5 | F5 | After a warp the search box is the min/max of 1000 uniform draws of the old box, mapped and widened by range/1000 (`whitening.py:333-347`; MATLAB `warp_input_vbmc.m:143-148`). Its per-coordinate extent falls short of the exact bounding box \|A\|h more as D grows. | confirmed shared defect, of no consequence | Yes, at every warp when D>1, noiseless or noisy. There is no measurable effect. | `O3_F5_search_box_after_warp` runs the real `warp_input` on the default box with a posterior-like covariance:<br>• Median extent 0.985 / 0.946 / 0.876 / 0.741 / 0.649 / 0.590 for D=2..20 (min 0.439). The reviewer's random-covariance figures are 0.97…0.49.<br>• Lost fraction of the old box's volume: 0.0028 to 0.037, against the order-statistics bound 1−(1−2/1001)^D.<br>• Posterior mass and image of the plausible box outside the new box: 0.<br>• Posterior mean ≥8.7 marginal SD from every face; half-width 19 to 38 SD.<br>`O3_F5_two_warps`: after two warps the lost volume of the original box is 0 to 1e-4, and the extent against its exact image has median 0.85 to 1.15. | MATLAB `2a076c9` (2020-06-16), unchanged. Python `4aa4ca97` (2022-03-03), then `385f9e97`, `d02c517e`, and `7d93f606` (the current-space inversion, W2-6). Always MATLAB's scheme. | No. W2-6 is a different defect in the same lines. | Leave it (MATLAB's scheme, nothing measurable), or use the exact \|A\|h. The exact box would widen the search box up to about 2.3× per coordinate at D=20 and move default trajectories for D>1. The widening rule (`active_sample.py:884-899`) only grows the box, and its clip at `lb_tran`/`ub_tran` is inert because those are ±inf in every transformed frame. |
| O3-6 | F6 | In a kept-warp iteration, `warp_input` deep-copies `optim_state` (`whitening.py:219`), breaking the link to `self.hyp_dict`. Active sampling, which would relink it (`vbmc.py:1571`, `:1586`), is skipped. The record (`:1941`) therefore holds the pre-warp `hyp_dict` of iteration k−1, and `VBMC.load(file, iteration=k)` (`:3252-3253`) pairs it with the warped `get_gp(k)`. | confirmed Python-only defect (**kept warps only**) | Only on resume from a kept-warp iteration. `load(file)` at the default last iteration is also affected when the run stopped on such an iteration (possible via `max_iter`; not via `max_fun_evals`, since a warp iteration makes no evaluation). An uninterrupted run is unaffected, and so is the continuation branch (`:1316-1323`). | `O3_F6_*` replays the statements of `optimize` with the real `warp_input`/`warp_gp_and_vp` and `IterationHistory`. **Kept:** the record equals h_{k−1} (ℓ 0.6/0.5/0.8 against warped 2.35/1.59/1.93; m0 0.70 against −2.66). **Undone:** the record equals this iteration's fit, because `optim_state_old` has `skip_active_sampling` False, so `:1571` relinks before the fit at `:1599`. | Deep copy `4aa4ca97` (2022-03-03); relink `4949c826` (2021-11-04); load restoring `hyp_dict` `a5f6effb` (2022-10-27, "Resume optimization process"). The defect dates from `a5f6effb`. No MATLAB counterpart. | No. P8-11 (pre-warp `run_cov`/`full`/recorded GPs, shared) is related but different. | Options: relink after the warp (`self.optim_state["hyp_dict"] = self.hyp_dict`), or restore `hyp_dict` in `load` from something other than the record. `optim_state["hyp_dict"]` is read only by `active_sample` after the relink and by `load`, so a fix moves no number of an uninterrupted run. The run that measures the effect is given in §9. |
| O3-7 | F7 | The sheet's "Why" for the probit default (`known_differences.md:2383-2400`) cites AGENTS.md "probit by default". AGENTS.md held that sentence from `44f0af68` to `af8fb783` (2026-09-21), and it was a statement of the default, not a reason. The default comes from `6cee9bb5` (2022-11-24, PR #119, "feat: change default bounded variable transform to probit"). | documentation only | n/a | `git log -S"probit by default" -- AGENTS.md`; `git show 6cee9bb5` (`.ini:311` logit→probit). `v1.0.0` (2023-03-16) ships probit. | Recorded reason: that one commit bullet only. Nothing in `papers/`, `docsrc/` or `pyvbmc/vbmc/README.md`. | No. | Correct the "Why" (text in §8). |
| O3-8 | §4 note 1 | `test_inverse_type3_max_space` (`test_parameter_transformer.py:641-650`): the probit/student4 loop repeats the min-space case (Y=−500, expects −10), so the upper saturation of those two transforms is never tested. | confirmed | n/a | read | since `4ce90e25` (2022-07-28) | no | Cheap: Y=+500 against `nextafter(10, −inf)`. |
| O3-9 | §4 note 2 | `test_transform_*_largeN` (`:707`, `:719`, `:731`, `:743`): `10 ^ 6` is XOR (12 rows), and all rows are identical, so the tests check nothing beyond the 10-row tests. | confirmed | n/a | read | since `a20e8529` (2021-02-26) | no | Trivial fix, or delete the tests. |
| O3-10 | §4 note 4 | `test_warp_gp_and_vp` (`test_rotoscaling.py:688`) is a MATLAB reference on an unbounded 2-D problem with `NegativeQuadratic` only. `test_construction_accepts_a_mean_function_the_package_builds` (`test_gp_training_policy.py:69`) accepts `"zero"` without a warp. So F2 is uncovered. | confirmed | n/a | read | — | no | A constant-mean and zero-mean warp on a bounded state, checking that m0' − m0 equals the logger's y shift. |
| O3-11 | §4 notes 3, 7 | The round-trip tests use interior points only (z=0.5025, 0.52; u=0.2, 0.11), with rtol 1e-11 for probit and student4. The Jacobian FD test draws points "well inside" (`test_parameter_transformer_jacobian_fd.py:21`). Neither reaches the F3/F4 regions. | confirmed | n/a | read | — | no | Worth acting on only together with a fix of F3/F4. The logit tail branch is pinned against a closed form (`test_log_abs_det_jacobian_logit_extreme_tails`), and FD cannot reach it in double precision. |

## 4. The check of the first-question answers

- **Q1 holds.**
  - `'g'` is not ∇log\|J\| (O3-1).
  - Its callers are `vbmc_pdf.m:119` (dead) and a commented line in `shared/warpvars_vbmc_test.m:88`.
  - PyVBMC reads log\|J\| only as data:
    - the logger (`function_logger.py:748`, `:780`);
    - the warp (`whitening.py:324`, `:438-499`);
    - `vp.pdf` (`:1008`);
    - S-VBMC (`svbmc/_entropy.py:93`), whose weights are optimized over fixed draws.
  - The report's closed-form gradient matches FD to about 1e-9 with R, scale, and a near-bound row, which the FD test leaves out.
  - The Torch export's logit term `−softplus(v)−softplus(−v)` equals the NumPy one algebraically. This was read, not run.
- **Q2 holds, with the framing sharpened (O3-4).**
  - Every figure the report gives reproduces.
  - The one soft spot is "inverse ≤1e-14" at a lower bound: at x−a=1e-200·(b−a) the probit inverse is 1.7e-13 relative off an `ndtr` reference. That is the tail's own conditioning (about \|u\|²·eps), not a defect.
- **Q3 holds, on a bounded state (`O3_Q3_*`, which the warp tests never use).**
  - warpfun is affine to 1.5e-11.
  - AΣAᵀ = I to 2.2e-15.
  - ℓ' = √(A²ℓ²) to 6.7e-16.
  - x_m' = Ax_m+c to 3.9e-16.
  - ω' marginal to 2.2e-16.
  - σ'λ' equals the marginal SD to 2.2e-16, and Σλ'² = 3 exactly.
  - μ' exact to 6.7e-16; weights unchanged to 5e-16.
  - Search cache exact; new transformer μ=0, δ=1, det R=1.
  - The logger's y shift is constant (spread 3.6e-15) and equals the m0 shift (O3-2).

## 5. Errors in the report

1. **F6, undone warps.** "An iteration whose warp was undone gets a same-space version of the same staleness, one fit behind" is wrong. The undo restores `optim_state_old`, whose `skip_active_sampling` is False, so `vbmc.py:1571` relinks before the fit at `:1599`, and the record holds that iteration's fit (`O3_F6_*`). A warp cannot fall on the iteration right after the warm-up ends, where `skip_active_sampling_after_warmup` could make the flag True.
2. **F5, consequence.**
   - "The shrinkage compounds across warps" is not borne out. After two warps no volume of the original box is lost (at most 1e-4), because the second warp maps the whole sampled box, whose bounding box already exceeds the first image.
   - "Every kept warp narrows the acquisition search" holds for extent only. In volume the loss is at most about 2D/1001, and the posterior is untouched.
3. **F2, reach (minor).** "More than 5 iterations past warm-up": with `warmup=False`, `last_warping` starts at −inf and the first warp needs no end of warm-up.
4. **F1, Python side (minor).** "`minimize(jac=False)`, which is gradient-free" is inaccurate: it is L-BFGS-B with finite-difference gradients. MATLAB's `fmincon` with `GradObj` off also estimates gradients by FD. "No analytic gradient on either side" is the accurate form.
5. **F6, location (minor).** `vbmc.py:1585` should be `:1586`.
6. **F1, history (imprecise, not wrong).** The `'g'` branch dates from the first commit `a1680e3` (2018-05-04), not merely "present at `abbf946`".

Everything else I checked held: every location in F1–F7, the sheet ranges, the commit citations (`2a076c9`, `4aa4ca97`, `a5f6effb`, `9da4afd`, `4ce90e25`, `7d93f60`, `6cee9bb5`, `44f0af68`, `af8fb783`, `b37b519`, `1e17fb8`, `a5240d2`, `f9c04bc`) and the test descriptions. The F5 extent figures reproduce in trend; their values depend on the covariance model.

## 6. Test notes worth acting on

- **Worth acting on:**
  - O3-8 (cheap).
  - O3-9 (trivial).
  - O3-10: a constant-mean and zero-mean warp test on a bounded state would have caught F2, and it pins the m0-equals-y-shift identity.
  - A direct test that the optim_state recorded at a kept warp holds the post-warp `hyp_dict`, together with any F6 fix.
- **Right but not worth acting on alone:** O3-11, and "no search-box coverage test" (F5 has no consequence).
- **Agreed:** `test_warp_input_rewrites_every_filled_row_of_the_logger` is good.

## 7. Defects on the MATLAB side

1. **`misc/warp_gpandvp_vbmc.m:37-66`**, with `gplite/gplite_meanfun.m:57-62`, `:347` and `gplite/gplite_post.m:120-121`.
   - What the code does: `case 0` ("Warp constant mean") catches the zero mean. The constant mean (code 1) falls to `otherwise`.
   - Consequence: a run with `gpMeanFun='const'` (accepted by `misc/setupvars_vbmc.m:286-291`) stops at its first warp. A run with `'zero'` stops on the out-of-range read `hyp(Ncov+Nnoise+1)`. With an output warp, it would instead take the first output-warp hyperparameter as m0 and write the warped value back into it.
   - Shared: not for const (PyVBMC warps it exactly); yes for zero (O3-2).
   - Source: read; the index error is inferred from MATLAB's documented indexing.
2. **`vbmc_pdf.m:119`** (dead, after `:117-118`): it would subtract per-coordinate inverse derivatives, not log-Jacobian gradients, in pre-rotation coordinates. No consequence.
3. **Shared, candidates for "left as they are in both" if the PI so rules:**
   - the student4 forward cancellation (`warpvars_vbmc.m:265-269`);
   - the upper-bound resolution (`:106-108`, `:256-258`, `:442-459`);
   - the sampled search box (`warp_input_vbmc.m:143-148`).

## 8. Sheet entries

**Replace "The gradient of the log Jacobian is not ported" with:**

> ### MATLAB's `'g'` action of `warpvars_vbmc.m` is not ported
> - Python: `pyvbmc/parameter_transformer/parameter_transformer.py` has no counterpart; `log_abs_det_jacobian` returns the log determinant only.
> - MATLAB: `shared/warpvars_vbmc.m:463-477`, `:763-768`. The `'g'` action shares the branch of `'p'` and `'l'`, leaves out `log(scale)` and the sum over coordinates, and exponentiates. It returns the N-by-D matrix of the derivatives of the coordinate-wise inverse map, at the coordinates before the rotation and the rescaling, which is not the gradient of the log Jacobian. Its one caller, `vbmc_pdf.m:119`, follows an unconditional `error` (`:117-118`) and never runs.
> - What differs: nothing that a computation reads. No PyVBMC computation needs the gradient of the log Jacobian. The GP, the acquisitions and the ELBO work in the transformed space, where the log Jacobian enters only as data in the stored log joint. `vp.pdf` refuses original-space gradients. The original-space mode search takes no analytic gradient on either side. The Torch export differentiates its own log Jacobian by autograd.
> - Why: the ruling on the wave-7 row that carries O3 F1.
> - Kind: unported feature (dead in MATLAB).

**Correct the "Why" of "The default bounded transform is probit, not logit" to:**

> - Why: commit `6cee9bb` (2022-11-24, pull request 119, "Pre public"), one of whose items reads "feat: change default bounded variable transform to probit"; every release from 1.0.0 (2023-03-16) ships it. No reason is recorded beyond that line.

No new sheet entries: F2 and F6 are defects, and F3, F4 and F5 are shared behaviour.

## 9. Anything beyond the report

- **The F2 exception leaves the instance half-warped.** `self.optim_state` and `self.function_logger` are already the warped copies (`vbmc.py:1433-1443`), while `self.vp`, `self.gp`, `self.hyp_dict` and `self.parameter_transformer` are still in the old space. The history up to k−1 is intact.
- **Zero mean.** A zero-mean GP has no mean hyperparameter to absorb the constant shift C of the stored log joints. Any fix that "just skips" the mean block leaves the refit to absorb it.
- **"zero" is presented as implemented.** `CHANGELOG.md:476-480` ("the mean functions … that PyVBMC implements") and the policy test do this, although it fails at the first warp.
- **Wave-0 wording.** "Loss of precision near a nonzero bound" should be restated: the criterion is an upper bound with \|b\| < b−a (worst at b=0), and a nonzero lower bound loses nothing.
- **Mode search.** `vp.mode(orig_flag=False)` passes the analytic gradient (`jac=True`, `variational_posterior.py:1425-1427`), where `vbmc_mode.m` uses `GradObj` off in both branches. This is outside this slice and not in P7-3d's list; one line for whoever writes that sheet entry.
- **Search bounds.** `warp_input` does not re-express `optim_state["lb_tran"]`/`["ub_tran"]` on either side. This is inert, because both are ±inf in any transformed frame.
- **The run that measures F6's effect,** for when the machine's heavy slot is free (not made here):
  1. A seeded D=3 run on a correlated Gaussian, long enough to reach one kept warp at iteration k.
  2. Save it, then `VBMC.load(file, iteration=k, set_random_state=True)`.
  3. Compare `hyp_dict["hyp"]` with `get_gp(k)`'s hyperparameters.
  4. Continue both from k (loaded) and from k−1 (control, an ordinary iteration). Compare iteration k+1's GP hyperparameters and ELBO with the original record.

  The control shows whether a resume otherwise replays exactly. The mechanism matters most in the main phase: with init_N=0 (r_index < 1) and opts_N=0, the slice sampler starts directly from the best pre-warp sample.

## 10. `git status --porcelain` at the end

- `pyvbmc-wave7` (HEAD `dd079de1`): empty
- `gpyreg` (`0186d89`): empty
- `vbmc` (`396d649`): empty
