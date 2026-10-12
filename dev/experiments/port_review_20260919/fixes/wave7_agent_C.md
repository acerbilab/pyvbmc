# Wave 7 fix round: PyVBMC fix agent C

Raw report of fix agent C, wave 7 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-23. One Opus agent on a
worktree made by hand (`../pyvbmc-w7-C`, branch `w7-agent-C`, cut from
`dev-port-review-w7` at `ed9df90e`, whose package code is that of
`a65b96f4`), made the fix of row W7-8 of `../verification/wave7.md` as the
PI ruled it, the docstrings of rows W7-9 and W7-10, and the test notes of
slice O1, one commit per row or note. It ran only the test functions it added or
changed, by node id, while another session held the machine's heavy slot.
The orchestrator reviewed the diffs and cherry-picked the commits onto
`dev-port-review-w7`, where they have other hashes
(`../verification/wave7.md`, "Fix commits"). The text below the rule is the
agent's final message, unedited.

---

# Wave 7 fix round: report of fix agent C

## 1. Setup

- `pyvbmc.__file__`: `C:\Users\luigi\Documents\GitHub\pyvbmc-w7-C\pyvbmc\__init__.py`
- `gpyreg.__file__`: `C:\Users\luigi\Documents\GitHub\gpyreg\gpyreg\__init__.py`, at `0186d89` (tag `v1.3.0`).
- Worktree `C:\Users\luigi\Documents\GitHub\pyvbmc-w7-C`, branch `w7-agent-C`. I started from `ed9df90e`, and `git diff a65b96f4 HEAD -- pyvbmc` was empty there.
- Interpreter: `C:\Users\luigi\Documents\GitHub\pyvbmc\.venv\Scripts\python.exe`, with BLAS single-threaded.
- Scripts and logs: `C:\Users\luigi\AppData\Local\Temp\claude\C--Users-luigi-Documents-GitHub-pyvbmc\30d85518-7e4d-42ed-9a16-74a7fb47ace3\scratchpad\wave7\fix_C\`

## 2. Commits

Abbreviations: F = `pyvbmc/testing/vbmc/test_variational_optimization_grad_fd.py`, V = `pyvbmc/testing/vbmc/test_variational_optimization.py`, P = `pyvbmc/testing/variational_posterior/test_variational_posterior.py`.

| hash | row | what it does | tests (node id) | seen to fail before the fix | run after it |
|---|---|---|---|---|---|
| `db8b1c45` | W7-8 | `set_parameters` rescales (`lambd` to unit RMS, `sigma` by the same factor) only when `sigma` and `lambd` are both optimized. Otherwise both are only reshaped. Docstring states the rule. F's module docstring qualified to match. | P::`test_set_parameters_keeps_a_scale_that_is_not_optimized` (3 flag cases × `raw_flag`); F::`test_neg_elcbo_grad_fd_scale_not_optimized[sigma\|lambd]`; F::`test_neg_elcbo_is_a_function_of_theta_with_a_scale_not_optimized[sigma\|lambd]` | P, all 6 cases: `assert np.array_equal(vp.sigma, expected_sigma)` gave `[[1.1747, 0.7832]]` vs `[[0.6, 0.4]]`. F grad `[sigma]`: `assert check_grad(f, grad, theta, rtol=1e-5, atol=1e-8)` was False. F repeated calls: `assert F == F_first`, 6802.35 vs 180.64 for `[sigma]` and 103.10 vs 178.13 for `[lambd]`. F grad `[lambd]` passed before, as expected: the gradient is consistent with its wrong value on a fresh copy, and the mirror defect shows only on repeated calls. | 14 passed, including V::`test_vb_init_type3_preserves_fixed_sigma_without_extra_draws` (4 cases) as the brief asked. 9 existing set/get tests in P also passed, 44 cases. |
| `56499954` | W7-9 | `_neg_elcbo` docstring: left as `None`, the variance is computed if and only if `beta` is nonzero, and `varF` is 0.0 when it is not computed. `beta` entry: a value that is not finite is taken as 0. | V::`test_neg_elcbo_default_variance_follows_beta` | Cannot fail: the fix is documentation only. The test pins the documented default and passed before and after. | passed |
| `1e4fd8d6` | W7-10 | `_initialize_full_elcbo`: `D` is the length of θ, `Ns` the number of GP hyperparameter samples. `_neg_elcbo`: `Ns` is per component (0 means the lower bound); entry renamed to `_entropy_alpha`; `compute_var : int or bool` documented as 0/1/2, where 2 is refused as not implemented. | V::`test_neg_elcbo_refuses_the_diagonal_variance[False\|True]` | Documentation only; the test pins the "refused" claim. | 2 passed |
| `1fde69e6` | test note | `_gp_log_joint` with zero and constant means. `_random_gp` gains a `mean=` argument; its negquad draws are unchanged. | F::`test_gp_log_joint_grad_fd_zero_and_constant_mean` (zero/const × D = 1, 3; Ns = 3); V::`test_gp_log_joint_value_against_quadrature` (all 3 means × D = 1, 2): `I_sk` vs a 60-node-per-dimension Gauss–Hermite quadrature of `gp.predict` | — | 10 passed. F::`test_gp_log_joint_grad_fd_D_ne_K`, which uses the changed helper, also passed. |
| `af2ed317` | test note | FD check of `_neg_elcbo` at (D, K) = (2, 1), (1, 1), (1, 2), (2, 3). Mean and log-scale bounds active. For K ≥ 2, `w[0]` sits below the penalty threshold. | F::`test_neg_elcbo_grad_fd_one_component_one_dimension_weight_penalty` | — | 4 passed |
| `11d19ae6` | test note | `test_neg_elcbo_retains_capped_small_weight_penalty`: replaced `not np.allclose` with the exact `(diag(w) - w wᵀ) @ slope`. The error is 8e-19 on a value of 3.35e-5; tolerance is rtol 1e-8, atol 1e-12. | F::`test_neg_elcbo_retains_capped_small_weight_penalty` | — | passed |
| `90ad725e` | test note | `test_neg_elcbo_grad_fd_mc_entropy` parametrized over `inside_bounds`. The old point stays. A new point inside the bounds with σ = 0.25 and 0.35 uses atol 5e-2 (the measured MC discrepancy is 0.004–0.024 per entry). Docstring gives the real size of that discrepancy. | F::`test_neg_elcbo_grad_fd_mc_entropy[False\|True]` | Mutation check (`note_d_mutation.py`), η block of dH ×5: passes at the old point (reproduces §5.4) and fails at the new one. At the new point the whole dH ×1.1 fails and the η block alone ×2 fails. | 2 passed |
| `655b5ce0` | test note | Variance at K = 3 with separated components of different widths. `J_sjk`, per-sample `varG` and the averaged `varG` checked against a 401-point grid quadrature of `predict_full(add_noise=False)`, D = 1. Measured error 5e-14 relative. | V::`test_gp_log_joint_variance_against_quadrature` | — | passed |

At the end, all 36 test cases added or changed passed in one process. `--collect-only` of the four touched files collected 223 tests, so the new import between test modules works.

## 3. Evidence that nothing moved

- **`w78_bit_identity.py`** (log `w78_bit_identity.log`). It compares against `git show a65b96f4:` copies of `variational_posterior.py` and `variational_optimization.py`: the unfixed `_neg_elcbo` runs on unfixed-class posteriors, the fixed one on fixed-class posteriors, from identical states and seeds.
  - Setup: `sigma` and `lambd` both optimized, `lambd` stored away from unit RMS; D ∈ {1, 2, 3, 5}, K ∈ {1, 2, 3}; all four settings of `optimize_mu`/`optimize_weights`; both values of `raw_flag`; deterministic and MC entropy; bounds on and off; gradient or variance; Ns = 1 and 3.
  - Calls compared: 576 `set_parameters`, 288 `get_parameters`, 2304 `_neg_elcbo` (all outputs, plus the posterior's state afterwards).
  - **Largest difference: 0.0 for all three.**
  - A control with `sigma` not optimized shows a difference of 0.18, so the comparison is not vacuous.
- **`w78_optimize_vp_frozen_sigma.py`** (one `optimize_vp` call with K = 1 and `sigma` not optimized; not a VBMC `optimize()` run): the unfixed `set_parameters` returns σ = 22292 with ELBO -1.18e8, the fixed one returns σ = 0.7804, held through the optimization.

## 4. Tests written but not run

None. No test needs `optimize()`.

## 5. Proposed changelog text

I checked that v1.0.4 has the defect: its `set_parameters` rescales whatever the flags (lines 749-752), and its `_neg_elcbo` calls `vp.set_parameters(theta)`. W7-9, W7-10 and the test notes need no sentence.

- **Entry:** "`VariationalPosterior.set_parameters` rescales `sigma` and `lambd` against each other only when both are optimized; a scale whose `optimize_sigma` or `optimize_lambd` flag is off keeps its value, so that the variational optimization no longer inflates a fixed `sigma` at every evaluation of its objective."
- **"Upgrading from 1.0.4" line:** "`VariationalPosterior.set_parameters` leaves `sigma` or `lambd` as it is when its optimization flag is off, where 1.0.4 rescaled it by the root mean square of `lambd`." This only affects a script that sets one of those flags by hand; no option or production path does. Drop the line if you read the flags as internal state.

## 6. Noticed beyond my rows, not fixed

- **Type-2 starting points overwrite a σ that is not optimized.** `_vb_init` (`variational_optimization.py`, `sigma0 = sqrt(mean(V / lambd0**2) / K_new) * ...`) sets σ from the spread of the HPD points whatever `optimize_sigma` says. MATLAB does the same (`vbinit_vbmc.m:32`). So `optimize_vp` with σ fixed returns the type-2 start's σ (0.7804 above, not the 0.5 given). The sheet entry on frozen σ covers type 3 only.
- **A representation difference from MATLAB for the sheet.** `vpoptimize_vbmc.m:189` ends with `rescale_params`, which rescales whatever the flags. `optimize_vp:362` now keeps a fixed scale, so with one scale fixed the returned `sigma` and `lambd` differ from MATLAB's by the gauge factor, with the same density. `get_parameters` still normalizes in place whatever the flags, as `rescale_params.m` does.
- **Porting-log line** proposed for `pyvbmc/variational_posterior/README.md` (I did not edit it): "`set_parameters()` rescales `lambd` to unit root mean square and `sigma` by the same factor only when both are optimized, where `rescale_params.m` always rescales; it is also the assignment of θ inside the objective, which `negelcbo_vbmc.m` makes without rescaling (wave 7, row W7-8)."
- **`_gp_log_joint`'s docstring** has the same slip as W7-10: `compute_var : bool, defaults to False`, though it takes 2, which its Raises section does describe.
- **Parametrized node id.** `dev/plans/stage2-entmc.md:219` names `test_neg_elcbo_grad_fd_mc_entropy`, which is now parametrized. The name still selects both cases.

## 7. Final state

`git status --porcelain`: empty.

```
655b5ce0 test(vbmc): check the variance of the expected log joint at separated components
90ad725e test(vbmc): make the Monte Carlo gradient check of _neg_elcbo constrain the entropy
11d19ae6 test(vbmc): pin the exact gradient of the weight penalty
af2ed317 test(vbmc): check the gradient of _neg_elcbo at K = 1, at D = 1 and with the weight penalty active
1fde69e6 test(vbmc): check _gp_log_joint with the zero and constant means
1e4fd8d6 docs(vbmc): correct the parameter docstrings of _initialize_full_elcbo and _neg_elcbo
56499954 docs(vbmc): state when _neg_elcbo computes the variance by default
db8b1c45 fix(variational_posterior): rescale sigma and lambd only when both are optimized
```
