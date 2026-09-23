# O2 — third reader: the entropies and the gradient of the variational density

Raw reviewer report, wave 7 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-23. One Opus agent, fresh
context, read-only on the repositories, combining both tracks: it
re-derived each formula of the slice, checked it by finite differences,
and then compared the code with MATLAB VBMC at `396d649`. It read PyVBMC at
`a65b96f4`, in a worktree frozen at that commit while the merged waves were
gated in the main checkout, with gpyreg at `v1.3.0` (`0186d89`). Its brief
carried one first question: which configurations the entropy tests and the
finite-difference test of the density's gradient leave out, and whether
each gradient holds there (`K = 1`, `D = 1`, a weight at or near zero,
nearly coincident components, disparate scales, points in the tails, with
and without the softmax Jacobian, several block budgets of the Monte Carlo
entropy); the report answers it in its section 4. Its check scripts ran in
the session scratchpad and are kept only on the machine that ran them
(`dev/scripts/runs/LOCAL.md`). The text below is the agent's final message,
unedited.

---

# O2 — third reader: the entropies and the gradient of the variational density

All scripts and logs are in `C:\Users\luigi\AppData\Local\Temp\claude\C--Users-luigi-Documents-GitHub-pyvbmc\30d85518-7e4d-42ed-9a16-74a7fb47ace3\scratchpad\wave7\O2`. Every script printed `pyvbmc.__file__` under `pyvbmc-wave7` and `gpyreg.__file__` under `gpyreg`, and ran with BLAS single-threaded.

**Summary.** The entropy code and the density gradient agree with my derivations, with finite differences and with MATLAB in every configuration the brief lists, and at every block budget tested. There is one finding (F1): where the mixture density underflows, the log density comes back as `-inf` and its gradient as NaN. MATLAB does the same, and it makes `vp.mode()` silently skip its refinement step on narrow posteriors.

## 1. Coverage

**Read completely:**
- `pyvbmc/entropy/entlb_vbmc.py`, `pyvbmc/entropy/entmc_vbmc.py` (including `_block_layout` and the fill/split helpers), `pyvbmc/entropy/README.md`.
- `pyvbmc/variational_posterior/variational_posterior.py`:
  - `:1-360` (constructor, calibration, rng, deepcopy);
  - `:637-1092` (`sample`, `pdf`, `_pdf`, `log_pdf`);
  - `:1093-1440` (`get_parameters`, `set_parameters`, `moments`, `mode`).
- `pyvbmc/decorators/handle_0D_1D_input.py`; `_neg_elcbo` (`pyvbmc/vbmc/variational_optimization.py:1136-1383`).
- MATLAB: `ent/entlb_vbmc.m`, `ent/entmc_vbmc.m`, `vbmc_pdf.m`, `vbmc_mode.m`.
- Tests: `test_entlb_vbmc.py`, `test_entmc_vbmc.py`, `_check_grad.py`, `test_variational_posterior_grad_fd.py`, `test_variational_posterior.py:325-424`, `testing/entropy/FIXTURES.md`, the `entlb`/`entmc`/`vp_pdf` oracle stages.
- Paper: appendix A.1 of `papers/acerbi2018variational_appendix.md:65-132`.
- Known-differences sheet: slices P6 (end) and P7, and the settled non-differences.

**Skimmed:** `calibration/profile.py` (defaults), `CANDIDATE_BUDGETS`, `ParameterTransformer` (standardization of unbounded variables), the two consumers of the log density (`acq_fcn_log.py:39-44`, `active_importance_sampling.py:350-365, 478-481`), the `ns_ent*` options, the pre-vectorization `entmc_vbmc.py` (`5a8e1818^`).

**Not reached:** the Student-t `pdf` branches beyond reading them (no gradient by design), the original-space Jacobian correction of `pdf`, `sample`, the calibration cache.

**D = 1 input shape.** For `D = 1`, a flat array of N > 1 points is taken as one N-dimensional point. It then fails with a cryptic `ValueError`/`IndexError` rather than returning wrong numbers (`pdf_d1.py`). A column input works.

## 2. Derivations

Notation: γ_jk = N(μ_j; μ_k, s_jk²Λ) with s_jk² = σ_j²+σ_k²; g_j = Σ_k w_k γ_jk; N_k is the density of component k.

"FD agrees" for `entlb` means central differences with a Richardson step. Checks were in both parameterizations: (μ, log σ, log λ, η) with `jacobian_flag=True`, and (μ, σ, λ, free w) with `jacobian_flag=False`. There were 16 configurations (`entlb_fd.py`, `entlb_fd2.py`, `entlb_fd3.py`):
- K=1 with D=1 and D=4;
- D=1 with K=2 and K=3;
- random K=4/D=3 and K=6/D=5;
- weights ≈1e-10 and ≈1e-100, both overlapping and separated;
- near-coincident (1e-7) and exactly coincident components;
- scales 1e-3..1e3 and 1e-4 vs 1;
- components 40 sd apart;
- D=20 with small scales.

| Quantity | Independent derivation | FD / other check | vs MATLAB | Verdict |
|---|---|---|---|---|
| `entlb` H, K≥2 | Jensen: H ≥ −Σ_j w_j log g_j | Value vs my log-sum-exp implementation ≤1.8e-15 | Line by line; fixture D=4, K=3 exact (ΔH=0, max\|ΔdH\|=4.4e-16) | agrees |
| `entlb` H, K=1 | Exact entropy D/2(1+log 2π)+D log σ+Σ log λ (not the bound; jump of D(1−log2)/2 to the K=2 coincident bound, by design) | exact | same (`entlb_vbmc.m:32-47`) | agrees |
| ∂H_lb/∂μ_m | w_m[Σ_k w_kγ_mk(μ_m−μ_k)/(s²λ²)/g_m + Σ_j w_jγ_jm(μ_m−μ_j)/(s²λ²)/g_j] | FD ≤1.6e-9; near-coincident limited by FD, converges with the step (gradient ~2.5e-8) | same | agrees |
| ∂H_lb/∂σ_m | −w_mσ_m[Σ_k w_kγ_mk c_mk/g_m + Σ_j w_jγ_jm c_jm/g_j], c=−D/s²+\|Δ/λ\|²/s⁴ (the diagonal term enters once through each sum) | FD ≤1.7e-9 | same | agrees |
| ∂H_lb/∂λ_d | −Σ_j (w_j/g_j) Σ_k w_kγ_jk(Δ_d²/(s²λ_d²)−1)/λ_d | FD ≤2.5e-9 | MATLAB forms the log-λ gradient and divides by λ when the Jacobian is off; equivalent | agrees |
| ∂H_lb/∂w_m | −log g_m − Σ_j w_jγ_jm/g_j | FD; the separated 4e-11 weight checked analytically (26.012) | same | agrees |
| Jacobians (both entropies) | ×σ, ×λ; softmax J=diag(w)−wwᵀ taken from η | FD in (log σ, log λ, η) | same | agrees |
| `entmc` H | −Σ_j w_j mean_n log q(μ_j+σ_jλε_jn), antithetic | Same-draws recomputation ≤1.8e-15. Mean vs quadrature entropy, D=1 and D=2: \|z\|≤0.6 (`entmc_unbiased.py`) | Same algorithm, different draw order. In distribution on the MATLAB fixture (48 seeds × Ns=1e5): all 22 gradient \|z\|<1.3, H z=−1.1 (`entmc_matlab_fixture.py`) | agrees |
| `entmc` ∇ μ, σ, λ | Path derivative only; the score term is dropped because E_q[∂_φ log q]=0 (paper A.1.1) | Equals ∂F(θ_samples; θ_density frozen)/∂θ to ≤1.1e-10 in 11 configurations × 2 Jacobian modes (`entmc_path.py`). Unbiased vs quadrature, \|z\|<1.6. It is **not** the derivative of the estimate at fixed draws (differs by 4–53% at Ns=40) | Same as MATLAB | agrees |
| `entmc` ∇ w | Full derivative, including −Σ_j w_j mean_n N_m/q (total mass) | Equals ∂F/∂w to FD precision; unbiased | MATLAB fix `1b72896` is present | agrees |
| `entmc` block layout | Canonical blocks drive all sums | Bit-identical over 9205 cases (7 canonical sizes including 1, 5, 17; 7 shapes; 7 flag sets; ~25 budgets each; `entmc_budget.py`). Split canonical blocks match a single block to 1.1e-15 (`entmc_canon.py`). Single (K, Ns/2, D) draw equals K successive draws (`draw_order.py`) | no counterpart | agrees |
| grad_flags subsets | — | Every subset returns bit-identical blocks and H for both entropies (`flag_subsets.py`) | same | agrees |
| `pdf` value | Σ_k w_k nf σ_k^−D exp(−d²/2) | vs log-sum-exp ≤1.4e-15 (1.2e-14 at 20 sd) | same | agrees |
| `pdf` ∇ | −Σ_k w_k N_k (x−μ_k)/(σ_k²λ²) | FD ≤1.3e-9; vs independent ≤1e-14. Tested K=1, D=1, K=5/D=4 with 50 rows, tiny weight, scales 1e-3..1e3, coincident components (`pdf_fd.py`). Bit-identical over 8 chunk sizes | same | agrees |
| log `pdf` ∇ | ∇q/q, stable form −Σ r_k(x−μ_k)/(σ_k²λ²) | Correct up to ~38.5 standardized units (subnormal density), then −inf and NaN | same (`vbmc_pdf.m:107-110`) | **F1** |

**Paper errata (the code is right).** Appendix A.1.1 of the 2018 paper has two typos:
- line 88 divides by (σ_kλ)² where σ_l is meant;
- line 113 carries a spurious 1/K².

The code, the MATLAB and my derivation all use w_j/N_s and σ_l.

## 3. Findings

### F1. The log density and its gradient are `-inf` and NaN wherever the mixture density underflows, and this silently stops `vp.mode()` from refining narrow posteriors
- Location: `pyvbmc/variational_posterior/variational_posterior.py:994-1000` (division `dy / y` and the zero mask), density formed in the linear domain at `:919-928`; consequence in `mode` at `:1371-1381` and `:1425-1427`. MATLAB: `vbmc_pdf.m:107-110`, density loop at `:58-66`.
- Category: formula/gradient
- Proposed classification: suspected defect in both
- Confidence: high for the behavior; medium for how often a user meets it
- History:
  - The MATLAB lines have not changed since 2019 (last commit `1e17fb8`, 2019-09-13).
  - The first Python port (`c87019b9`, 2021-03-16) transcribed them (`dy = dy / y; y = np.log(y)`).
  - `e20d0818` (2022-09-21) added the zero mask only to avoid `log(0)`. Its message reads "Underflow in `vp.pdf` to be fixed later".
  - Python has always matched MATLAB here.
- What the code does and what it should do:
  - `pdf` forms q(x) as a sum of exponentials and takes the log afterwards.
  - Beyond about 38.5 standardized units from every component (log q < −744), q underflows to 0. The code then returns log q = −inf and ∇ log q = 0/0 = NaN.
  - The true values are finite: log q = logsumexp_k(log w_k − D log σ_k − d²_k/2) + const, and ∇ log q = −Σ_k r_k(x−μ_k)/(σ_k²λ²) with responsibilities r_k from that log-sum-exp.
  - Examples: at 45 units the code gives (−inf, NaN) where the truth is (−970.66, [−44, −0.5]); at 100 units, (−inf, NaN) against −4903.2.
  - Just below the threshold the density is subnormal, and ∇ log q loses digits gradually.
  - Without `log_flag`, the returned 0 and 0 are right to double precision.
- Consequence if real:
  - **Within a run:** none, noiseless or noisy. The only package caller of the gradient is `vp.mode(orig_flag=False)`, and `optimize()` never calls `mode`. The log-density consumers clamp at log(realmin) (`acq_fcn_log.py:39-44`, `active_importance_sampling.py:478-481`); MATLAB's acquisitions do the same.
  - **User-facing:** `vp.log_pdf` in the tails returns −inf.
  - **`vp.mode()`, both `orig_flag` values:** the optimizer's first trial step is about one coordinate unit long. When the posterior scale σλ is below roughly 0.003–0.026 of a unit, that step lands where log q underflows.
    - The objective becomes +inf (and the gradient NaN).
    - BFGS (`orig_flag=False`) stops at nit=0 with "precision loss".
    - L-BFGS-B (`orig_flag=True`) backs off and reports "CONVERGENCE" with `success=True`.
    - Either way, no warning reaches the user, and `mode()` returns its starting candidate (the best of 1e5 draws or a component mean) unrefined.
  - **Units:** for `orig_flag=False` a unit is a plausible-box width for unbounded variables, so a posterior SD below about 1% of the plausible range triggers it; for `orig_flag=True` it is an original-space unit. The error equals how far the start is from the mode: 4e-3 of the scale in my D=2 example, 3.3e-2 at D=6, growing with D.
  - **MATLAB:** `vbmc_mode.m` uses fmincon/fminunc on the same `-inf` log density, without an analytic gradient. Whether those optimizers recover from an infinite trial point is not settled (MATLAB could not be run).
- Suggested reproduction and my results:
  - `pdf_fd.py` (tails section): the table above.
  - `mode_trace.py`: `mode(orig_flag=False)` refines normally at scales 0.1, 0.03 and 0.01. At 0.003 and below it has nit=0, success=False, an infinite objective evaluation, and returns the start unchanged.
  - `mode_trace3.py`: `mode(orig_flag=True)` at scale 0.01 makes its first trial 98 sd away, gets f=inf there, returns x0, and reports success.
  - `mode_trace4.py`: the same L-BFGS-B call from the same start moves 0 units on the code's log density. On a log-sum-exp log density it moves 0.896 scale units to the mode, and f drops by 0.307 nats.
- Test adequacy: no test evaluates `pdf` or `log_pdf` in the tails, or `mode` on a narrow posterior. `test_pdf_grad_log_flag_no_orig_flag` checks the log gradient as `dy / exp(log_y)`, which mirrors the implementation.

## 4. Test adequacy notes

The brief's first question. The tests leave out several configurations. Every gradient holds in all of them (section 2), apart from the tail case of F1.
- **`entlb`:** no D=1 case, no finite-difference check at K≥3, no finite-difference check with the Jacobian (that path is covered only by the one MATLAB fixture and by the `_neg_elcbo` finite-difference tests), and no near-zero weights, coincident components or disparate scales.
- **`entmc`:** likewise no near-zero weights, coincident components or disparate scales, and D=1 only in the separated case, whose gradient check is not asserted.
- **`pdf` gradient:** no K=1, no D=1, no finite-difference check over several rows, no tails.
- **Block budget:** gradient calls with canonical blocks that split samples are never exercised. Every gradient case in `_BUDGET_CASES` fits one component into one canonical block; only the value-only cases (Ns=4096) split samples. My sweep covers it.

Other gaps and weaknesses:
- **`test_entmc_vbmc.py:128`:** the result of `check_grad` in `test_entmc_vbmc_nonoverlapping_mixture` is not asserted, so the D=1 and D=2, K=2 and K=3 gradients are checked only against the separated-component approximations.
- **`test_entmc_vbmc.py:185`:** `atol=0.01` exceeds every one of the 12 MATLAB μ-gradient entries (largest 0.0125), so the μ block is effectively unchecked. Python's seed-to-seed SD on those entries is 1e-5 to 6e-4, which would allow a much tighter tolerance.
- **`entmc` finite-difference tests compare different quantities.** They compare the path-derivative estimator with the finite difference of the estimate at fixed draws; the two agree only in expectation, which forces 1–3% tolerances. An exact check exists: freeze the density's parameters and move only the samples. It matches to 1e-10 (`entmc_path.py`) and would catch errors far below 1%.
- **Budget-independence test:** it compares each flag set only with itself. Agreement of blocks across flag subsets is not tested; it holds (`flag_subsets.py`).
- **`test_pdf_grad_default_no_orig_flag`:** it uses 20 identical points and two coincident components, so the sum over distinct components with distinct scales is untested there. The finite-difference test covers one point at D=3, K=2.
- **Oracles:** the `entlb`, `entmc` and `vp_pdf` oracles pin the current outputs. They mirror the implementation and would have pinned any formula error present when they were generated.
- **Settled entries verified:** the known-differences entries on this slice match the code. I confirmed two of them: NaN at an exactly zero weight with separated components, only for exact zero (a weight of 4e-305 is fine; `zero_weight.py`), and the total-mass term of the Monte Carlo weight gradient, which is unbiased.
