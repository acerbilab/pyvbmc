# Wave 7 verification, slice O2: the entropies and the gradient of the variational density

Raw report of a verifier, wave 7 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-23. One Opus agent,
read-only on the repositories, verified the findings and the other
observations of the third reader of slice O2
(`../reviews/O2_third_reader.md`) and its answers to the first question of
its brief, reading PyVBMC's package code at `a65b96f4`, gpyreg at `v1.3.0`
(`0186d89`) and MATLAB VBMC at `396d649`, with no `optimize()` run. Its
scripts are kept as `scripts/wave7_O2_*.py`, under the names the report
gives them with `wave7_` in front; its helper `O2_common.py`, the
independent log-sum-exp reference, is kept as `scripts/wave7_O2_common.py`,
with the imports renamed to match. Their logs are kept on the machine that
ran them (`dev/scripts/runs/LOCAL.md`, "Port correctness review"). The text
below the rule is the agent's final message, unedited; `wave7.md` holds the
consolidated ledger and the dispositions.

---

# Wave 7 verification, slice O2: the entropies and the gradient of the variational density

## 1. Revisions and interpreter

Every script printed the following:
- `pyvbmc: C:\Users\luigi\Documents\GitHub\pyvbmc-wave7\pyvbmc\__init__.py`
- `gpyreg: C:\Users\luigi\Documents\GitHub\gpyreg\gpyreg\__init__.py`
- `numpy 2.5.2 scipy 1.18.1`

Revisions:
- The worktree is at HEAD `0570d92a`. `git diff a65b96f4 HEAD -- pyvbmc/` is empty, so the package code is that of `a65b96f4`.
- gpyreg is at `0186d89` (tag `v1.3.0`).
- MATLAB VBMC is at `396d649`.

Interpreter and environment:
- Interpreter: `C:\Users\luigi\Documents\GitHub\pyvbmc\.venv\Scripts\python.exe`.
- BLAS single-threaded.
- `PYTHONPATH` pointed at the worktree.

Scripts and logs are in `C:\Users\luigi\AppData\Local\Temp\claude\C--Users-luigi-Documents-GitHub-pyvbmc\30d85518-7e4d-42ed-9a16-74a7fb47ace3\scratchpad\wave7\verify_O2\`, as `O2_*.py` with a `.log` beside each. `O2_common.py` holds the independent log-sum-exp reference.

## 2. Headline

**F1 reproduces.** Where the mixture density underflows, `pdf(log_flag=True)` returns −inf for the log density and NaN for its gradient, and it does so exactly as `vbmc_pdf.m` does. My transcription of the MATLAB lines gives the same non-finite pattern, identical finite values, and gradients equal to 1 ulp. That part is a **confirmed shared defect**.

**The consequence for `vp.mode()` reproduces for both values of `orig_flag`.** The optimizer's first trial step lands where the density underflows, and the unrefined start is returned. Whether `vbmc_mode.m` suffers the same depends on how `fmincon`/`fminunc` treat a +Inf trial value, so that part **needs MATLAB**.

**No run reaches F1**, noiseless or noisy. I checked every caller.

Four points change the reviewer's reading:
- **A warning does reach the user.** The report says none does. Under Python's default filters a NumPy `RuntimeWarning` is printed: "invalid value encountered in subtract" from SciPy's `_numdiff.py:710` for the default `orig_flag=True`, and "invalid value encountered in divide" from `variational_posterior.py:996` for `orig_flag=False`. The warning is cryptic, but it is shown. The reviewer's scripts ran under `warnings.simplefilter("ignore")`.
- **BFGS does not always stop at nit=0.** For `orig_flag=False` its line search sometimes backtracks and converges: 2 of 3 runs at D=6 and 3 of 3 at D=10, at scale 1e-3.
- **The failure of the default call depends on the parameter's units.** `vp.mode()` skips its refinement when the posterior SD is below about 0.013 to 0.026 of a unit of the parameter itself (unbounded variables). For variables bounded on both sides the threshold is about 0.005 to 0.017, since there the first L-BFGS-B step is the gradient step clipped to the box. Rescaling a parameter changes whether `mode()` refines.
- **Earlier records ruled this out and should be corrected.** P7 internal Q1, the wave-5 check of it, and the plan's record of the P7 first question (`dev/plans/port-correctness-review.md:1397-1402`, "the mode search starts at samples and component means, never at such a point") all judged the mode search unable to reach the underflow. Its starts do not reach it; its trial steps do.

The other observations:
- **The D = 1 input shape is a Python-only defect.** The report says a flat array always fails. In fact `orig_flag=False` with a finite `df` returns one meaningless number without error.
- **The paper errata read in the transcription as the report says** (documentation only). I could check only the Markdown transcription, not the published PDF.
- **All five test notes are confirmed**, with two corrections: the `atol` note is right in substance but wrong in its count (11 of 12 entries, not all 12), and the Torch export test checks the log-pdf gradient at K = 1, D = 1 and 4 rows. Asserting the unasserted `check_grad` as written would fail in 3 of 4 configurations, although the code is right.

**The first-question answers hold.** Checked by complex step, and against an exact path-derivative reference for `entmc`, every gradient agrees to machine precision in every configuration the tests leave out, including production-sized `entmc` gradient calls whose canonical blocks split one component's samples.

## 3. Ledger

| id | report | statement | verdict (class) | fires at defaults? | how verified | dating and recorded reason | already recorded? | note for disposition |
|---|---|---|---|---|---|---|---|---|
| O2-1 | F1 | `_pdf` sums the component densities in the linear domain (`variational_posterior.py:919-928`) and takes the log afterwards (`:994-1000`). Where log q < about −745 (38.6 SD for a unit Gaussian), it returns log q = −inf and ∇log q = `dy / y` = 0/0 = NaN (`:996`), although both are finite. In the subnormal band just before that, accuracy is lost: at the last representable density the value is off by 7e-4 relative and the gradient by 2.6e-2. **Consequence in `mode`** (`:1371-1381`, `:1425-1427`): on a narrow posterior the first trial point of the optimizer (about one coordinate unit long) lands in that region. L-BFGS-B (`orig_flag=True`, the default) returns its start with `success=True` and "CONVERGENCE: RELATIVE REDUCTION OF F". BFGS (`orig_flag=False`) usually stops at nit=0 with "precision loss" and sometimes recovers. MATLAB has the same lines: `vbmc_pdf.m:58-66` and `:107-110` | **Confirmed shared defect** for the log density and its gradient: the transcription of `vbmc_pdf.m:41-66, 107-110` gives the same −inf/NaN pattern and identical finite log q. The `mode` consequence is **confirmed on the Python side**. For `vbmc_mode.m` it **needs MATLAB**: its objective is +Inf there, it requests no gradient (`GradObj` off), and recovery depends on `fmincon`/`fminunc` | **No run reaches it**, noiseless or noisy. The in-run readers are covered: `acq_fcn_log.py:42` and `active_importance_sampling.py:481` floor the value at log(realmin), as MATLAB does. `active_importance_sampling.py:362-364` does not floor it, but the −inf enters a log-sum-exp with finite box terms (defaults 100/100 samples); at `w_vp = 1` the proposal evaluates its own draws. The gradient is requested only by `mode`, and `mode` is called by nothing in `optimize()`, the result dictionary, the final statistics, S-VBMC (it has its own `torch.logsumexp`), the Torch or ArviZ exports, or the PyMC adapter. **Users reach it** through `vp.log_pdf` in the far tails and through `vp.mode()`, which the FAQ lists and example 2 calls (notebook and generated script); example 2's posterior has an SD of order 1 and is not affected. **Scale for the default `mode()`**: unbounded variables refine at σλ ≥ 0.016 and are skipped at σλ ≤ 0.013 (original units). In a [0, 10] box, one of two runs was skipped at SD 0.017/0.012 and both at 0.005. A [0, 1] variable is skipped at SD 0.007 and refined at 0.021. For `orig_flag=False`, unbounded variables refine at an SD of 1% of the plausible range and are skipped at 0.5%. **Size of the error**: the returned start is 0.3% to 8% of the smallest component SD from the mode (at most 0.01 nats) at D = 2 to 10; it is exact for K = 1 under a linear transform | `O2_F1_pdf_tail_underflow.py`: D = 2, K = 2 exact to t = 35; at t = 37, 45 and 100 the code gives (−inf, NaN) against −786.8, −1163.0 and −5735.8; the transcription matches. `O2_F1_mode_narrow.py` covers four families: identity transform, unbounded with a plausible box, a box, and mixed. `O2_F1_mode_bfgs_trace.py` shows the trial steps of 1.01 units, the infinite values and the halvings. `O2_F1_mode_lse_counterfactual.py`: the same SciPy calls from `mode`'s own starts leave the point unmoved on the code's density (2.8e-3 of the scale off) and converge on a log-sum-exp density (1.2e-5 and 6e-12). `O2_F1_mode_start_error.py` measures the error. `O2_F1_mode_warning_origin.py` shows the printed `RuntimeWarning` | The MATLAB lines date from `b37b519` (2018-05-09) and have not changed since; `1e17fb8` (2019-09-13) is the file's last commit and touches other lines. On the Python side, `c87019b9` (2021-03-16) transcribed the lines, and `e20d0818` (2022-09-21, PR 102) added the zero mask with the message "fix: Use log-pdf; Underflow in `vp.pdf` to be fixed later.". The Python has always matched MATLAB. `mode`'s optimizer calls come from `ba8116fa` (2022-11-03, PR 115) and `15fbe6d3` (2022-11-15, `jac` for the transformed space). `vbmc_mode.m` last changed in `48c82e0` (2021-03-27, after the port began; it clamps the starts, as PyVBMC does) | **Partly.** The underflow and the NaN gradient at x = 40 are in `reviews/P7_internal.md:48-51` ("unreachable in practice"), in `verification/wave5_P7.md:53` (Q1), and in the plan's P7 first-question record (`dev/plans/port-correctness-review.md:1397-1402`). A log-space rewrite of `vp.pdf` is a deferred idea in `dev/2026-09-02-modernization-discussion.md` §12. There is no ledger row, no `TODO.md` item and no entry in `matlab_side_defects.md`. This finding is wider: the trial steps of the mode search reach the underflow | Options: (a) compute the log density by log-sum-exp in `_pdf` under `log_flag`. That repairs the value, the gradient and `mode`, and moves no run, since the in-run readers floor the value or combine it in a log-sum-exp. The `vp_pdf` oracle must pass `make_oracle_fixtures.py --check --exact`. It goes beyond MATLAB. (b) Keep `pdf` and make `mode` robust: a log-sum-exp objective inside `mode`, or an optimization in coordinates scaled by the posterior's SD. (c) Leave it, and say in `mode`'s docstring that a narrow posterior may be returned unrefined. Whatever the choice: record the shared part in `matlab_side_defects.md`, and correct the plan's statement at `:1397-1402` |
| O2-2 | §1, "D = 1 input shape" | `handle_0D_1D_input` reads a 1-D array as one point. For a one-dimensional posterior, a flat array of N > 1 points is therefore one point of N coordinates. `pdf` and `log_pdf` then raise `IndexError` (`orig_flag=True`, at `parameter_transformer.py:227`/`:358`) or `ValueError` (`orig_flag=False`, at `variational_posterior.py:917`). The exception is `orig_flag=False` with a finite `df`: the t branches (`:937-992`) take D from `x.shape[1]` (`:877`) and return one meaningless number without error (0.0375 for three points). A column of points works | **Python-only defect** (interface). The report's "rather than returning wrong numbers" does not hold in the t branches. MATLAB has no decorator; my transcription shows that `vbmc_pdf.m` also silently returns one number for a 1-by-N row when D = 1, which MATLAB's row-per-point convention makes a user's error | No: every package caller passes 2-D arrays | `O2_D1_input_shape.py`: nine calls, both orig flags, both transforms, `df` of 3 and −3 | The 1-D reading of `pdf` has held since `98472e9a` (2021-03-26, "generalize 1D handling with a decorator"); the decorator was renamed in `fbfb41b1` (2022-11-01). The t branches are as ported. No reason recorded | No. The sheet entry "The 0-D/1-D input decorator is a Python-only device" does not say that a 1-D array is one point | Options: read a flat array as a column when `vp.D == 1` (unambiguous there); refuse any `x` whose width is not `vp.D`; or document the reading. Any of these closes the silent t-branch case |
| O2-3 | §2, "Paper errata" | In `papers/acerbi2018variational_appendix.md`: line 88, the general form of the entropy gradient, has the denominator (σ_k λ^(i))² inside the sum over l, where the derivative of N(ξ; μ_l, σ_l²Σ) gives (σ_l λ^(i))². Line 113, the σ_j gradient, carries w_j/(K² N_s) | **Documentation only** (the paper); the code is right. The transcription's lines read as the report says: σ_k appears at line 88 alone, and σ_l at lines 103, 113 and 123; K² appears only at line 113. The code uses σ_l and w_j/N_s (`entmc_vbmc.py:118`, `:236-238`), and so does MATLAB (`entmc_vbmc.m:77-93`). **Only the Markdown transcription was checked.** Whether the published PDF has the same typos, or the transcription introduced them, cannot be settled from the repository | n/a | Read, and checked against the derivation and the code; the code's gradient is confirmed exactly (§4) | The transcription was copied from `acerbilab/pubs-llms` (`papers/README.md`) | No | Whether to flag the two lines in `papers/` (the README, or a note beside the lines), following the records convention. The code is unaffected |
| O2-4 | §4, `test_entmc_vbmc.py:128` | In `test_entmc_vbmc_nonoverlapping_mixture` the result of `check_grad` is not asserted, so the gradients at D = 1, 2 and K = 2, 3 are checked only against the separated-component approximations (`:117`) | **Confirmed (test weakness).** If asserted as written, the check would **fail** in 3 of 4 configurations: D1K2 (2 of 7 entries outside the band), D1K3 (3 of 10), D2K2 (1 of 10); D2K3 passes. The cause is O2-6, not a code error | n/a | `O2_T_entmc_unasserted_check.py`, the same `check_grad` computation run outside pytest | `5ffdf634` (2021-06-22, "feat: add entropy functions") | No | Adding `assert` alone would break the test. Replace the check with the exact frozen-density check (O2-6), or give it a tolerance scaled to the Monte Carlo error |
| O2-5 | §4, `test_entmc_vbmc.py:185` | `np.allclose(dH, dHm, rtol=0.01, atol=0.01)` leaves the μ block nearly unchecked: the band 0.01 + 0.01·abs(dHm_i) covers 11 of the 12 MATLAB μ entries (largest abs(entry) 0.01253), so a zero μ block fails only on the entry −0.01253 | **Confirmed in substance; the report's wording is wrong** ("exceeds every one of the 12"; it is 11 of 12). PyVBMC's seed-to-seed SD of these entries is 1.5e-5 to 4.6e-4 over 24 seeds, so the band is 20 to 600 SDs wide. In distribution, PyVBMC's mean against MATLAB gives max abs(z) 1.72 over the 22 entries and z of H = 0.64 | n/a | `O2_T_entmc_matlab_fixture.py` | `5ffdf634` (2021-06-22) | No | Options: a tolerance scaled to the Monte Carlo SD (MATLAB's value is itself a single estimate), or leave it and rely on the exact check of O2-6 |
| O2-6 | §4, "`entmc` finite-difference tests compare different quantities" | `:84`, `:128` and `:150` compare the path-derivative estimator with the derivative of the estimate at fixed draws; the two agree only in expectation | **Confirmed.** At Ns = 1e5 they differ by up to 1.4% (median 0.5%) on the μ, σ and λ entries, and by 57% (median 39%) at Ns = 40. The weight entries agree to 1e-11, since that gradient is the full derivative. The exact check (density frozen, samples moved) agrees with the code to within 6e-15 absolute | n/a | `O2_T_entmc_total_vs_path.py`; `O2_Q1_entmc_path_exact.py` | `5ffdf634` (2021-06-22) | No | Add the exact check. It would catch errors far below the 1% to 3% tolerances the present tests need |
| O2-7 | §3 (F1 test adequacy); §4, `test_pdf_grad_default_no_orig_flag` | `test_pdf_grad_log_flag_no_orig_flag` asserts dlog_y = 9.58788/exp(log_y), that is ∇q/q, the implementation's own route. `test_pdf_grad_default_no_orig_flag` uses 20 identical rows and two coincident components. The FD tests (`test_pdf_grad_fd`, `test_log_pdf_grad_fd`) use one row at D = 3, K = 2. No test evaluates `pdf`/`log_pdf` in the tails, or `mode` on a narrow posterior | **Confirmed, with a correction to the report's list:** `test_vp_torch.py::test_density_and_gradient` checks the **log**-pdf gradient (`orig_flag=False`) against Torch autograd at (D, K) = (1, 1) and (3, 3), 4 rows each. It skips without Torch, and CI runs it in one cell only. The linear-density gradient has no such check | n/a | Read; `O2_Q1_pdf_grad_fd.py` covers the missing configurations | The test lines are from the first port (`c87019b9`, 2021-03-16) and `fbfb41b1` (2022-11-01) | No | A few-row FD test of the linear-density gradient at K = 1 and D = 1, independent of Torch. Tail and narrow-`mode` tests go with any fix of O2-1 |
| O2-8 | §4, "Block budget" and "Budget-independence test" | `test_entmc_vbmc_is_independent_of_the_chunk_budget` compares each flag set only with itself (`:246-256`). Every gradient case of `_BUDGET_CASES` has Ns·D·K ≤ 65536 (the largest is 42,000, for (15, 50, 56)), so no gradient call splits one component's samples across canonical blocks. Production gradient calls do: with `ns_ent = 100 K^(2/3)`, K = 50 and D = 10 give 679,000 | **Confirmed (test gap); the code is right.** Across all 16 flag subsets, with and without the Jacobian, 3 cases, the blocks and H are bit-identical. Production-sized split calls agree with the exact reference to within 6e-13 at 4 or 5 budgets each | n/a | `O2_T_grad_flag_subsets.py`; `O2_Q1_entmc_path_exact.py` | `f3ba8d35` (2026-09-19) | No | Add one gradient case whose canonical blocks split samples (for example D = 8, K = 20, Ns = 738), and one assertion of agreement across flag subsets |

## 4. The check of the first-question answers

The report's answer holds: every gradient holds in the configurations the tests leave out, apart from the tail case of F1. My own checks follow.

**`entlb_vbmc`.** The code reads the VP arrays as they are and uses only arithmetic, sqrt, exp and log, so assigning complex arrays gives the complex-step derivative of the code's own value; the step was scaled to each coordinate.
- Coverage: 15 configurations × 2 parameterizations, i.e. (μ, σ, λ, w free) without the Jacobian and (μ, log σ, log λ, η) with it. The configurations were:
  - K = 1 at D = 1 and at D = 4;
  - D = 1 at K = 2 and at K = 3;
  - random K = 4/D = 3, K = 6/D = 5 and K = 50/D = 10;
  - weights of 1e-10 (overlapping) and 1e-100 (40 SD apart);
  - near-coincident (1e-7) and exactly coincident components;
  - scales 1e-3..1e3, and 1e-4 against 1;
  - components 40 SD apart;
  - D = 20 with small scales.
- Result: max relative error ≤ 2e-12, absolute ≤ 5e-13.
- The value agrees with an independent log-sum-exp form of the bound to ≤ 1.8e-15.
- On the MATLAB fixture: abs(H − Hl) = 0 and max abs(dH − dHl) = 4.4e-16, as the report says.
- Scripts: `O2_Q1_entlb_complex_step.py`, `O2_Q1_entlb_matlab_fixture.py`.

**`entmc_vbmc`.** I reproduced the draws (one `standard_normal((K, Ns/2, D))`, antithetic) and wrote the estimator independently.
- The code's H equals my F(θ; θ) to ≤ 3.3e-16 relative.
- The μ, σ and λ gradients equal the exact frozen-density path derivative ∂F/∂θ_samples, and the weight gradient equals the full ∂F/∂w. Both were taken by complex step, and agree to ≤ 6e-15 absolute with and without the Jacobian.
- Configurations: K = 1/D = 1; D = 1/K = 3; K = 6/D = 5; a weight of 1e-10; near-coincident and coincident components; scales 1e-3..1e3; components 40 SD apart.
- Two production-sized gradient calls split samples across canonical blocks: K = 50, D = 10, Ns = 1358 and K = 20, D = 8, Ns = 738. Each was checked at 4 or 5 budgets, down to 300; the results are identical across budgets.
- The claim that one draw equals K successive draws holds, generator state included.
- The flag subsets are bit-identical.
- I did not repeat the reviewer's quadrature check of unbiasedness; the exact check shows the estimator is the stated one.
- Scripts: `O2_Q1_entmc_path_exact.py`, `O2_Q1_entmc_draw_order.py`, `O2_T_grad_flag_subsets.py`.

**`pdf` and its gradient.**
- Checked with Richardson central differences on the code's own value, and with the independent formula.
- Configurations: K = 1/D = 1, D = 1/K = 3, K = 1/D = 4, 50 rows at K = 5/D = 4, a weight of 1e-12, scales 1e-3..1e3, coincident components, and tails at 20 to 30 SD.
- Result for the linear and the log gradient alike: against FD ≤ 1.5e-9, and ≤ 7e-6 for the 1e-3 scale, where the FD step is the limit; against the formula ≤ 1e-13. The chunk sizes 1, 3, 17 and 1000 give bit-identical results.
- The log gradient fails only beyond the underflow (O2-1).
- Script: `O2_Q1_pdf_grad_fd.py`.

**The coverage gaps the report lists.** Reading the tests confirms each one, except that the Torch test covers the log-pdf gradient at K = 1, D = 1 and several rows (O2-7). The `entlb` path with the Jacobian is also exercised by `test_neg_elcbo_grad_fd_deterministic_entropy`, as the report says.

## 5. Errors in the report

1. F1: "Either way, no warning reaches the user." This is wrong. With default filters, `vp.mode()` prints `RuntimeWarning: invalid value encountered in subtract` (`scipy/optimize/_numdiff.py:710`) for `orig_flag=True`, and `... in divide` (`variational_posterior.py:996`) for `orig_flag=False`. The warning does not say the refinement was skipped (`O2_F1_mode_warning_origin.py`). The reviewer's mode scripts ran under `warnings.simplefilter("ignore")`.
2. F1: "BFGS (`orig_flag=False`) stops at nit=0" is not systematic. Its line search halves the step from the infinite value and sometimes recovers (D = 6: 2 of 3 runs; D = 10: 3 of 3; `O2_F1_mode_bfgs_trace.py`).
3. F1 "Units" covers only unbounded variables. When every variable is bounded, L-BFGS-B's first step is the gradient step clipped to the box, not a unit step; the thresholds measured there are 0.005 to 0.017 original units. The report's distance of the start from the mode "growing with D" is also not monotone: 3.7e-3, 7.5e-2, 5.0e-2 and 8.1e-2 at D = 2, 4, 6 and 10.
4. F1 consequence: "The log-density consumers clamp at log(realmin) (`acq_fcn_log.py:39-44`, `active_importance_sampling.py:478-481`)" leaves out `active_importance_sampling.py:362-364`, which does not clamp. That omission is harmless (O2-1 row), and the report does list those lines among the ones it skimmed.
5. F1 history: "The MATLAB lines have not changed since 2019 (last commit `1e17fb8`, 2019-09-13)". `1e17fb8` is the file's last commit and touches other lines; these lines date from `b37b519` (2018-05-09).
6. §1 D = 1: "fails ... rather than returning wrong numbers" is not true of `orig_flag=False` with a finite `df` (O2-2).
7. §4, `:185`: "`atol=0.01` exceeds every one of the 12 MATLAB μ-gradient entries (largest 0.0125)" contradicts itself; the band covers 11 of 12 (O2-5).
8. §4 "`pdf` gradient: no K=1, no D=1, no finite-difference check over several rows". The Torch test checks the log-pdf gradient at K = 1, D = 1 and 4 rows, by autodiff rather than FD, where Torch is installed (O2-7).

All other citations I checked are right: the file lines, the MATLAB lines, `c87019b9`, `e20d0818` and its message, `1b72896`, `5a8e1818`, `_neg_elcbo` at `:1136-1383`, and the paper's lines 88 and 113.

## 6. Test notes worth acting on

- Add the exact frozen-density check for `entmc`, which agrees with the code to within 6e-15. It could replace, or join, the fixed-draw checks at `:84`, `:128` and `:150`. Asserting `:128` as it stands would fail (O2-4).
- Add one gradient budget case whose canonical blocks split one component's samples, and one assertion that the blocks agree across flag subsets (O2-8).
- Add a small FD test of the linear-density gradient at K = 1, D = 1 and several rows that does not depend on Torch (O2-7).
- If O2-1 is fixed: add tests of the log density and its gradient in the tails, and a test of `mode` on a narrow posterior for both values of `orig_flag`.
- Tightening `:185` is optional once the exact check exists.
- The note that the oracles mirror the implementation is true by construction and calls for no action.

## 7. Defects on the MATLAB side

- **`vbmc_pdf.m:107-110`, shared with PyVBMC.** The log density is the log of a linear sum: −Inf beyond about 38.6 SD, and a NaN gradient from `dy./y` = 0/0. This was read from the source and measured on my transcription. It is a candidate line for the shared list of `matlab_side_defects.md` once ruled on.
- **`vbmc_mode.m:37-46`, needs MATLAB.** The source shows that `nlnpdf` returns +Inf where the density underflows, and that neither `fmincon` nor `fminunc` requests a gradient (`GradObj` off), so no NaN gradient reaches them. Whether they recover from a +Inf trial value is behaviour of the Optimization Toolbox, which is not in the repository. My recollection of MathWorks' documentation, not verified here, is that `fmincon`'s default interior-point algorithm takes a shorter step after an Inf or NaN objective; if so, MATLAB's default `origflag = 1` call would refine where PyVBMC's does not. Only a MATLAB run settles it.
- **`vbmc_pdf.m:41` (minor).** The width of `X` is not checked against `vp.D`. For a D = 1 posterior, a 1-by-N row returns one number, Σ_k w_k Π_n N_k(x_n) (my transcription: 0.0485, where the product of the three densities is 0.0608). This is user misuse under MATLAB's row-per-point convention.

## 8. Sheet entries

- **Correct "The mode search starts from draws of the posterior" in "What differs".** For `orig_flag=False` the call has no bounds and uses SciPy's default BFGS with the analytic gradient; it is not L-BFGS-B. Proposed text: "the optimizer (`scipy.optimize.minimize`: L-BFGS-B within the bounds with finite-difference gradients in the original space, for `fmincon`; BFGS with the analytic gradient of the log density in the transformed space, for `fminunc`)".
  - If O2-1 is left as it is, add: "Where the density underflows, the objective is infinite at the first trial point of either optimizer, and a posterior whose SD is below about 0.01 in the units of the space searched is returned unrefined, with a NumPy `RuntimeWarning` (`verification/wave7.md`, O2-1)."
- **If O2-2 is left, add to "The 0-D/1-D input decorator is a Python-only device":** "A 1-D array is one point, so for a one-dimensional posterior a flat array of N points is read as one point of N coordinates: `pdf` raises, or, in the transformed space with a finite `df`, returns one value; a column of points works."
- **If the underflow of O2-1 is left, add to the settled non-differences:** "**The log density is the log of a linear sum, on both sides.** `pdf(log_flag=True)` forms the mixture density before taking its log, as `vbmc_pdf.m:107-110` does: exact to about −745, −inf beyond with a NaN gradient. No run reads it unfloored where it underflows (O2-1)."

## 9. Anything found beyond the report

- The origin of the printed warning, the contingent recovery of BFGS, and the thresholds for bounded variables (sections 2 and 5).
- `_pdf` takes D from `x.shape[1]` (`:877`), not from `self.D`. That is what lets a width-mismatched input for D = 1 through the t branches silently (O2-2).
- The plan's record of the P7 first question (`dev/plans/port-correctness-review.md:1397-1402`) states that the mode search never reaches an underflowed point. Under the records convention it is corrected in that file once O2-1 is ruled on.
- The sheet's mode entry misnames the transformed-space optimizer (section 8).
- The log-sum-exp counterfactual shows that the underflow alone stops the refinement: the same SciPy calls, from the same starts, converge on a log-sum-exp density.

## 10. `git status --porcelain`

- `C:\Users\luigi\Documents\GitHub\pyvbmc-wave7`: (empty)
- `C:\Users\luigi\Documents\GitHub\gpyreg`: (empty)
- `C:\Users\luigi\Documents\GitHub\vbmc`: (empty)
