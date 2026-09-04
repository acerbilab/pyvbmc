# Stage 2 item 1: `_gp_log_joint` vectorized over hyperparameter samples and components

Created: 2026-09-04 23:10. Status: **IN PROGRESS**. Roadmap pickup point 2
(`plans/modernization-roadmap.md`, "Next: Stage 2 item 1"); rationale in
`dev/2026-09-02-modernization-discussion.md` §2, §3, §9 and §10 (Stage 2,
item 1: "`_gp_log_joint` `(s,k)` loop → `einsum` over `(Ns,K,D,N)`"), and
the profile in `plans/stage2-batched-acquisition.md` §Results: after item 3
the variational stage is 24–32 % of a D = 4 run under cProfile, of which
`_gp_log_joint` is 18–24 % (3.1k–6.7k `_neg_elcbo` calls per run, 5–17 ms
each, ~50 tiny NumPy calls per `(s, k)` loop body), and `final_boost`
10–17 % in one call at K = 50. Method as in item 3: fixed-state oracles on
every commit, the golden replay per step, the profile suite once. This
file is the plan now and the worklog afterwards.

## Summary

Same formulas, one pass. `_gp_log_joint` computes the expected log joint
of the GP posterior mean under the variational mixture and its gradient
with respect to `(mu, sigma, lambd, w)`, and optionally the variance of
that expectation. Today it loops in Python over the `Ns` hyperparameter
samples and the `K` components, with a third loop over component pairs
for the variance, two single-RHS triangular solves per pair. (1) The
`(s, k)` loop becomes broadcasting over an `(Ns, K, D, N)` tensor of
standardized distances and two batched matrix-vector products against
`alpha`; every per-sample and per-component scalar (`ell`, `sf2`, `m0`,
`xm`, `omega`, `tau`, `lnnf`) becomes an `(Ns, …)` array. (2) The pair
loop becomes, per hyperparameter sample, one multi-RHS triangular solve
`L^{-T} Z^T` and one `K × K` Gram product, which is roadmap item 2 done in
its natural home (the loop lives in `_gp_log_joint`, not in
`_eval_full_elcbo`); it is a separate commit so the oracle report and the
replay attribute the change. (3) Three latent defects of the function
recorded in devlog §9 are resolved on the way, because the loop that
contained them is gone: the dead `compute_vargrad` accumulators are
deleted (the two `NotImplementedError`s that make them unreachable stay,
same conditions, same messages), `separate_K` without variance returns
`J_sjk = None` instead of raising `UnboundLocalError`, and
`jacobian_flag=False` returns all four gradient blocks (the Jacobian
corrections become conditional, the blocks are not). Gates: the
`gp_log_joint` and `neg_elcbo` oracles (GP-solve class 1e-6, variance
class 1e-3), the finite-difference tests, the MATLAB-pinned test, the
replay, the full suite; the profile suite once.

## Scope

- **In**: `pyvbmc/vbmc/variational_optimization.py::_gp_log_joint` (whole
  body; signature, return tuple and every caller unchanged); a
  finite-difference test for the `jacobian_flag=False` path and a unit test
  for `separate_K` without variance
  (`pyvbmc/testing/vbmc/test_variational_optimization_grad_fd.py`,
  `..._single_sample.py` or a sibling); a speed probe option for
  `dev/scripts/profile_suite.py` (item 3 follow-up: a reference config run
  at the start and the end of a campaign); records (`dev/README.md` index,
  roadmap, devlog §2/§9/§10 dated addenda, this file).
- **Out**: `_neg_elcbo` (its duplicated grad/no-grad branches are a
  cleanup, not this item; it needs no change for the fixes above because
  its own fallback already sets `J_sjk = None`); the softmax Jacobian
  duplicated in four places; `entmc_vbmc` (item 5); the mean-function
  generality (`ZeroMean`, `ConstantMean`, `NegativeQuadratic` stay the
  supported set, same as today); porting MATLAB's diagonal variance
  approximation (`compute_var == 2`) and the variance gradient (a feature,
  not a refactor; Stage 0 still lists "finite-difference checks for
  `compute_vargrad`", which presupposes it, see Open questions); gpyreg
  (item 8); the 20-seed population (end of stage).

## Findings the plan rests on

Verified against the code on 2026-09-04 (`c48c025`).

- **Call sites.** `_neg_elcbo` (five call shapes: gradients without
  variance, averaged, as Adam sees it; no gradients with the full variance
  and `separate_K`, as `_eval_full_elcbo` sees it; no gradients with the
  full variance and `separate_K=False`, live through `active_sample.py:675`
  when `active_sample_full_update` is on, off by default, and through the
  single-sample regression test, the shape that must keep returning a
  *scalar* `varG` at `Ns = 1`; gradients with variance, which raises; the
  `separate_K`-without-variance fallback, dead because `skip_elbo_variance`
  is not a valid option), the `gp_log_joint` oracle (three shapes,
  including `avg_flag=False`), `active_sample.py:337`
  (`_gp_log_joint(vp, gp, 0, 0, 0, 1)`: no gradients, `jacobian_flag=0`,
  `compute_var=1`; dead, no acquisition sets `compute_var_log_joint`),
  and three test modules plus the oracle module. The only non-test caller
  with `jacobian_flag=False` is that dead call, and it requests no
  gradients, so fix 3 has no dependent caller. `grad_flags` arrives as a
  bool or a 4-tuple of the VP's `optimize_*` flags; **`optimize_vp` sets
  `optimize_weights = False` for every warm-up iteration**
  (`variational_optimization.py:147`), so `(True, True, True, False)` runs
  in every VBMC run and the per-block conditionals are load-bearing; the
  `normal_D2_warmup` and `rosenbrock_D2_noise1_viqr` snapshots carry
  `optimize_weights: False`, so the `neg_elcbo` oracle gates that shape.
- **Shapes.** `gp.posteriors[s]`: `hyp (hyp_N,)`, `alpha (N, 1)`,
  `L (N, N)`, `L_chol` bool, `sW (N, 1)` constant across rows
  (`sn2_eff = 1 / sW[0]**2`, a length-1 array in the current code). VP:
  `mu (D, K)`, `sigma (1, K)`, `lambd (D, 1)`, `w (1, K)`, `eta (1, K)`.
  Hyperparameter layout: `hyp[0:D]` = `ln ell`, `hyp[D]` = `ln sf`,
  noise block of `noise_N` entries, then the mean block `m0`, `xm (D)`,
  `ln omega (D)` for `NegativeQuadratic` (`cov_N = D + 1`). The function
  reads `m0` for any non-`ZeroMean` mean and `xm`, `omega` only for the
  quadratic mean.
- **Output layout to preserve.** `dG (D K + K + D + K, Ns)` before
  averaging, blocks in the order mu (Fortran-flattened `(D, K)`: `d`
  fastest), sigma, lambd, w; then `dG.sum(1) / Ns` when `avg_flag` and
  `Ns > 1`; the `Ns == 1` squeeze drops the sample axis of `G`, `dG`,
  `varG` (the 2026-09-02 fix). `I_sk (Ns, K)`, `J_sjk (Ns, K, K)`
  symmetric with the **unclamped** diagonal (only `varG` clamps `J_kk` at
  `eps`), `varG (Ns,)` then `mean + varG_ss` when averaging,
  `var_ss = varG_ss + std(varG, ddof=1)`. `_eval_full_elcbo` stores
  `I_sk` and `J_sjk` into `elbo_stats` and `optimize_vp` prunes columns of
  both and keeps them in `vp.stats`.
- **Per-loop-body arithmetic today** (`s`, `k`): `tau_k = sqrt(sigma_k²
  lambd² + ell²)` `(D, 1)`; `lnnf_k = 2 ln sf + Σ ln ell − Σ_d ln tau_k`;
  `delta_k = (mu_k − X^T) / tau_k` `(D, N)`; `z_k = exp(lnnf_k − ½ Σ_d
  delta_k²)` `(N,)`; `I_k = z_k · alpha + m0 + nu_k` with `nu_k = −½ Σ_d
  (mu_dk² + sigma_k² lambd_d² − 2 mu_dk xm_d + xm_d²) / omega_d²`;
  gradients `mu: −w_k Σ_n delta z alpha / tau − w_k (mu_k − xm) / omega²`,
  `sigma: w_k sigma_k Σ_n [Σ_d (lambd_d / tau_d)² (delta² − 1)] z alpha −
  w_k sigma_k Σ_d lambd_d² / omega_d²`, `lambd: lambd_d Σ_k w_k
  (sigma_k / tau_kd)² Σ_n (delta² − 1) z alpha − Σ_k w_k sigma_k² lambd_d
  / omega_d²`, `w: I_k`; then `sigma_grad *= sigma`, `lambd_grad *=
  lambd`, `w_grad = J_softmax w_grad` under `jacobian_flag`. Variance, for
  `j ≤ k`: `J_jk = exp(lnnf_jk − ½ Σ_d delta_jk²)` with `tau_jk² =
  (sigma_j² + sigma_k²) lambd² + ell²`, `delta_jk = (mu_j − mu_k) /
  tau_jk`, minus `z_k · L^{-1} L^{-T} z_j / sn2_eff` when `L_chol`, plus
  `z_k · L z_j` otherwise (`L = −(K + σ²I)^{-1}` in that branch);
  `varG_s = Σ_k w_k² max(eps, J_kk) + 2 Σ_{j<k} w_j w_k J_jk`. `z_j` is
  recomputed for every pair (the "recomputing `z_j`" waste of devlog §2).
- **Memory of the tensor.** `(Ns, K, D, N)` doubles: the oracle
  snapshots peak at `7 × 50 × 4 × 115 = 161k` (1.3 MB) on
  `cigar_D4_boosted`; the profile suite at about `6 × 35 × 10 × 185 =
  389k` (3 MB, `lumpy_D10`) and `1 × 31 × 15 × 750 = 349k` (2.8 MB,
  the exhaust run's tail, a single sample); `final_boost` at `K = 50`
  with the best iteration's GP, `Ns ≤ 8`, `N ≤ 250` at `D = 5` (sampling
  stops at `N ≥ 200 + 10 D`, so `Ns > 1` and `N = 350` cannot coincide):
  0.5M (4 MB). One such tensor lives at once (`delta`; with the `einsum`
  contractions `delta²` is never materialized), plus the `(K, D, N)` `Xt`
  that the current code already builds. `K` is capped by `k_fun_max =
  N^(2/3)`, not by 50, so the corner `D = 20`, `N = 1100` (the default
  budget), `Ns = 1`, `K ≈ 106` is 2.3M per tensor (19 MB), and `Xt` is
  as large: about 37 MB peak. No chunking is needed at any supported
  operating point; the item 3 lesson (a 33 MB chunk was memory-bound) says
  to keep the tensors in the low tens of MB, which they are (and the
  `einsum` form measured fastest there, see the tracker).
- **Where rounding can move** (corrected by the review and by
  measurement). Elementwise pieces (`tau`, `delta`, `exp`) are the same
  scalar operations and bit-identical. `Σ_d ln tau`, `Σ ln ell` and the
  `Σ_d` of the quadratic term are *already* pairwise today (a `(D, 1)`
  or `(D,)` reduce coalesces to a contiguous one) and the new last-axis
  reduce over `(Ns, K, D)` is bit-identical to them at every `D`
  (verified at D = 3, 12, 40). `Σ_d delta²` over the `(Ns, K, D, N)`
  array is bit-identical to today's `(D, N).sum(0)` with `np.sum`
  (sequential over `d` in both) but the `einsum` form actually used
  moves `z` by an ulp on large arrays. `Σ_k w_k I_k` becomes a pairwise
  last-axis sum instead of a sequential accumulation (identical for
  `K < 8`). The contractions over `n` change kernel (`ddot` on one `(N,)`
  row today, `einsum`'s own loop tomorrow: a few ulp per product and a
  different summation order). The sigma and lambd gradients swap the
  order of the `Σ_d` and `Σ_n` sums and use `B − zalpha` for
  `Σ_n (delta² − 1) z alpha` (see below); `w_k / omega²` becomes
  `w_k · (1 / omega²)` and the lambd quadratic term factors `Σ_k w_k
  sigma_k²` out of the per-`k` division: one extra rounding each. All
  of it is rounding-level *per operation*; through the GP solve's
  conditioning it becomes 1e-9–2e-7 per element on the cigar and corr
  snapshots (tracker), the same order as the cross-BLAS floors the
  oracle classes were set from (1e-6 GP-solve, 1e-3 variance). Because
  `G` feeds Adam and the sieve, an ulp in `G` changes the Adam path and
  thus the VP, so the replay is expected to *part* at iteration 0 on
  every config (item 3's six bit-identical trajectories will not recur);
  the gates are the oracles and the population envelope, as the replay
  was designed for.
- **`B − zalpha`.** `Σ_n (delta² − 1) z alpha = Σ_n delta² z alpha −
  Σ_n z alpha` exactly; in floating point the difference form has an
  absolute error bound `eps · Σ_n (1 + delta²) |z alpha|` against
  `eps · Σ_n |delta² − 1| |z alpha|` for the signed sum, i.e. larger
  where `delta² ≈ 1`, which is where the sigma/lambd gradient is small
  (near Adam's optimum). Both bounds carry the same `Σ_n |z alpha|`
  factor, and that factor, inflated by the GP's conditioning (`alpha`
  entries of alternating sign and large magnitude), is what the measured
  floors are made of: the extra term is `~eps · |I_k|`, 1e-14 absolute,
  against the measured 1e-9 absolute on corr. The oracle's per-element
  criterion protects near-zero entries with the lower-quartile floor.
  The exact form would need `delta² − 1` materialized (one more
  `(Ns, K, D, N)` pass and temporary), which the `einsum` design avoids
  on purpose; accepted trade, recorded here.
- **Item 2's arithmetic.** Today, for `j ≤ k`: `J_jk = Jbase_jk − z_k ·
  (L^{-1} (L^{-T} z_j)) / sn2_eff` (two single-RHS `solve_triangular`
  calls, `trans=1` then `trans=0`, on gpyreg's *upper* factor `L`, so the
  default `lower=False` must stay), then `J_sjk[s, j, k] = J_sjk[s, k, j]`
  from that one number. Two multi-RHS forms are available: the Gram form
  `V = L^{-T} Z^T`, `C = V^T V / sn2_eff` (one TRSM, one GEMM), and the
  **two-solve form** `Y = L^{-1} (L^{-T} Z^T) / sn2_eff` `(N, K)`, `C = Z
  Y` (two TRSMs, one GEMM), whose entry `(k, j)` has exactly today's
  summands `z_k · y_j` and differs only in the contraction order (GEMM
  vs `ddot`). The review pointed out that `J = Jbase − C` is a difference
  of nearly equal numbers (which is why the diagonal is clamped at `eps`),
  so the Gram form's re-expression could move `J_sjk` by more than the
  1e-8 absolute variance floor; the two-solve form costs `2 K N²` instead
  of `K N² + K² N` (both negligible against the `K (K + 1)` single-RHS
  solves of today) and is chosen. Its `C[k, j]` for `k ≥ j` is the lower
  triangle, which is mirrored onto the upper one so `J_sjk` stays exactly
  symmetric as today. The non-Cholesky branch (`L_chol=False`: `L =
  −(K + σ²)^{-1}`, today `J_jk += z_k · L z_j`) becomes `J += Z L Z^T`,
  symmetrized the same way; **it is unreachable in production** (the GP
  noise is bounded below by `tol_gp_noise² = 1e-5 > 1e-6`,
  `gaussian_process_train.py:368`, so `min(sn2) ≥ 1e-6` and gpyreg's
  `L_chol` is always `True`), untested, and hit by no oracle: the
  scratch bit-check adds a state with `ln sn = −8` to compare old and new
  on it, and a unit test builds an equivalent Cholesky-form posterior by
  hand and asserts the two branches agree. The diagonal of `Jbase` is the
  self-term `exp(lnnf_kk)` (`delta_kk = 0`). `Jbase` over all `(j, k)` is
  exactly symmetric since `sigma_j² + sigma_k²` and `(mu_j − mu_k)²` are
  commutative in floating point. The final `varG = max(varG, eps)` stays.
- **The step oracle does not depend on this function.** `active_sample`
  reaches `_gp_log_joint` only through the dead `compute_var_log_joint`
  path and through per-sample VP re-optimization, which is off on every
  snapshot the step oracle applies to; `acq_*` oracles do not touch it
  either. So the oracle gate is `gp_log_joint` and `neg_elcbo` (every
  snapshot, including `normal_D2_K1` at `K = 1`, `normal_D2_singlesample`
  at `Ns = 1`, `cigar_D4_boosted` at `K = 50` and the noisy one with
  `s2`) with no re-baseline expected.
- **Existing tests on the function.** `test_gp_log_joint` (MATLAB-pinned
  `G`, `varG`, `var_ss`, `dG` at `D = K = 2`, 8 samples, `np.isclose`
  1e-5; it passes a **1-D `lambd` and a 1-D `eta`**, so the rewrite must
  reshape rather than index), three single-sample regression tests, two
  finite-difference tests (`Ns = 8` and `Ns = 1`, raw parameterization,
  `rtol 1e-5`), the `_neg_elcbo` finite-difference tests through it, and
  the `vp_optimize` tests end to end.
- **A fifth latent defect, found by the review of the rewrite.** The
  softmax Jacobian in the current `_gp_log_joint` is `-np.exp(vp.eta).T *
  np.exp(vp.eta) / eta_sum**2 + diag(...)`, an outer product only when
  `vp.eta` is `(1, K)`; with the 1-D `eta` that `test_gp_log_joint` sets
  it broadcasts to an elementwise row and the off-diagonal entries are
  wrong (`−e_j² / S²` instead of `−e_i e_j / S²`). The test passes only
  because the default `eta` is uniform, where the two coincide; production
  always has a `(1, K)` `eta` (`_neg_elcbo` reshapes it), so no run is
  affected. The rewrite uses `np.outer` on the raveled `eta`, correct for
  both shapes and bit-identical for `(1, K)`. `_neg_elcbo`'s own copy of
  the Jacobian keeps the old form (safe there); `entlb_vbmc` and
  `entmc_vbmc` already use the explicit outer product. Recorded in
  devlog §9.
- **Profiling baseline.** `runs/profile_20260904/` (plain, D ≤ 10
  configs, code `f441172`) and `runs/profile_20260904_retry/` (the
  exhaust run and the cProfile pass on a cool machine, code `93dc29d`);
  `aggregate.md` in each. Before/after comparison needs the same seed
  (0), one BLAS thread, an idle machine, and a speed probe (`banana_D4`
  plain, 42.6–45.5 s cool, 56.5 s hot on this laptop).

## Design

### Step 1: mean and gradients over `(s, k)` (item 1 proper)

Private copies with fixed shapes: `mu (D, K)`, `mu_T = mu.T`, `sigma =
array(vp.sigma).reshape(-1)` `(K,)`, `lambd (D,)`, `w (K,)` (the
MATLAB-pinned test passes 1-D `lambd` and `eta`). Per-sample arrays from
`H = stack(ravel(p.hyp) for p in gp.posteriors)` `(Ns, hyp_N)`: `ell =
exp(H[:, :D])` `(Ns, D)`, `ln_sf2 = 2 H[:, D]`, `sum_lnell = H[:,
:D].sum(1)`, `m0 = zeros(Ns)` for `ZeroMean` else `H[:, cov_N +
noise_N]`, `xm`, `omega`, `inv_omega2 = 1 / omega²` `(Ns, D)` for the
quadratic mean; `alpha = stack(reshape(p.alpha, -1))` `(Ns, N)`. Then

```
tau    = sqrt(sigma[None, :, None]**2 * lambd[None, None, :]**2 + ell[:, None, :]**2)   # (Ns, K, D)
lnnf   = ln_sf2[:, None] + sum_lnell[:, None] - log(tau).sum(2)                         # (Ns, K)
Xt     = mu_T[:, :, None] - X.T[None]                                                    # (K, D, N)
delta  = Xt[None] / tau[..., None]                                                       # (Ns, K, D, N)
dsq_sum = einsum("skdn,skdn->skn", delta, delta)                                         # (Ns, K, N)
z      = exp(lnnf[..., None] - 0.5 * dsq_sum)                                            # (Ns, K, N)
zalpha = (z @ alpha[:, :, None])[..., 0]                                                 # (Ns, K), gemv per sample
nu     = -0.5 * (inv_omega2[:, None, :] * (mu_T[None]**2 + sigma²·lambd² − 2 mu_T[None] xm[:, None, :] + xm[:, None, :]**2)).sum(2)
I_sk   = zalpha + m0[:, None] + nu                                                       # (Ns, K)
G      = (w[None, :] * I_sk).sum(1)                                                      # (Ns,)
```

Gradients, with `za = z * alpha[:, None, :]` `(Ns, K, N)` and two
`einsum` contractions over `n` (one pass over `delta` each, `delta²`
never materialized; see the Decisions for why not `np.matmul`):

```
M = einsum("skdn,skn->skd", delta, za)              # Σ_n delta z alpha          (Ns, K, D)
B = einsum("skdn,skdn,skn->skd", delta, delta, za)  # Σ_n delta² z alpha
B -= zalpha[..., None]                              # = Σ_n (delta² − 1) z alpha
mu_grad[s, k, d]  = −w_k M / tau − (w_k inv_omega2_sd) (mu_dk − xm_sd)     -> reshape (Ns, K D).T = (D K, Ns), d fastest
sigma_grad[s, k]  = w_k sigma_k [ Σ_d (lambd_d / tau_skd)² B − Σ_d inv_omega2_sd lambd_d² ]   -> .T = (K, Ns)
lambd_grad[s, d]  = lambd_d [ Σ_k w_k (sigma_k / tau_skd)² B − inv_omega2_sd Σ_k w_k sigma_k² ]  -> .T = (D, Ns)
w_grad[k, s]      = I_sk[s, k]
```

Jacobian corrections (`sigma_grad *= sigma`, `lambd_grad *= lambd`,
`w_grad = J_softmax w_grad` with `J_softmax` from `np.outer` on the
raveled `eta`) apply only under `jacobian_flag`; the blocks are appended
regardless (fix 3). `compute_vargrad` code deleted, the `compute_var ==
2` raise hoisted out of the deleted loop with its condition, message and
precedence intact (fix 1); `J_sjk = None` when `separate_K` and not
`compute_var` (fix 2). The variance branch in this step keeps the pair
loop, reading `z[s, k]`, `z[s, j]`, `ell[s]`, `ln_sf2[s]`, `sum_lnell[s]`
from the arrays already computed, so the commit's variance outputs move
only by the rounding of `z` and the commit's oracle report isolates the
mean and gradient arithmetic.

Gates: `pytest pyvbmc/testing/oracles` (all green, no re-baseline
expected), `pytest pyvbmc/testing/vbmc/test_variational_optimization*.py`,
a scratch bit-check of the old function (kept in the scratchpad) against
the new on the eight snapshots and on random states at `D = 8, 12`
(where the last-axis sums switch to pairwise), reporting per-output max
relative differences (expected `≤ 1e-13`), the replay (7 min), the full
suite (10 min). Commit `perf(vbmc): vectorize _gp_log_joint over
hyperparameter samples and components`.

### Step 2: variance by one multi-RHS solve per sample (item 2)

```
tau_jk  = sqrt((sigma[:, None]**2 + sigma[None, :]**2)[None, ..., None] * lambd**2 + ell[:, None, None, :]**2)   # (Ns, K, K, D)
lnnf_jk = ln_sf2 + sum_lnell − log(tau_jk).sum(3)                           # (Ns, K, K)
J       = exp(lnnf_jk − 0.5 Σ_d ((mu_j − mu_k) / tau_jk)²)                  # Jbase, (Ns, K, K), exactly symmetric
for s:  if L_chol: Y = solve_triangular(L_s, solve_triangular(L_s, z[s].T, trans=1), trans=0) / sn2_eff_s   # (N, K)
                   J[s] -= z[s] @ Y                       # entry (k, j) = z_k · y_j, today's summands
        else:      J[s] += z[s] @ L_s @ z[s].T            # L = −(K + σ²)^{-1}
mirror the lower triangle of J onto the upper (k ≥ j is what the pair loop computed)
varG[s] = einsum("sjk,j,k->s", J with diag clamped at eps, w, w); varG = max(varG, eps)
J_sjk = J if separate_K
```

Two `solve_triangular` calls per sample (Ns BLAS TRSMs; SciPy ≥ 1.15 is
the floor in `pyproject.toml`, batched `solve_triangular` is not relied
on).
Gates: the `gp_log_joint` oracle's `varG`, `var_ss`, `J_sjk` (variance
class, 1e-3 + 1e-8) and the `neg_elcbo` oracle's `varF`, `varG`,
`varG_ss`, `F_full`; `test_gp_log_joint`'s `varG` and `var_ss`
(`np.isclose`); the single-sample tests; the replay; the full suite.
Commit `perf(vbmc): compute the log-joint variance with one multi-RHS
solve per hyperparameter sample`.

### Step 3: tests for the fixes

- `jacobian_flag=False`: finite-difference check of `dG` against `G` in
  the raw `(mu, sigma, lambd, w)` coordinates, with `w` free (the function
  uses `w` as given, so `∂G/∂w_k = I_k`): asserts the full length
  `D K + K + D + K` and agreement at `rtol 1e-5`. Update the module
  docstring's "Parameterization note", which documents the old
  limitation.
- `separate_K=True, compute_var=False`: returns `J_sjk is None`, `I_sk`
  of shape `(Ns, K)` with `I_sk @ w == G` per sample.
- The vargrad deletion is covered by the existing `dvarG is None`
  assertions and the two `NotImplementedError` conditions (add a
  one-line `pytest.raises` for each if none exists).

### Step 4: speed probe in `profile_suite.py`

`--probe <config>` runs the named config plain (own tag `probe_start_…`,
`probe_end_…`) before the first and after the last campaign run and
prints both walls with their ratio at the end; the aggregate lists them
as ordinary plain rows. Twenty lines; recorded in `dev/README.md`.

### Step 5: measure

`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u
dev/scripts/profile_suite.py --suite profile --mode plain --probe
banana_D4 --out dev/scripts/runs/profile_<stamp>` alone on a cool machine
(≈ 40 min after item 3), then `--mode cprof --only
banana_D4,cigar_D4,lumpy_D4,student_D4` (≈ 6 min). Compare per config
with `runs/profile_20260904/aggregate.md` (D ≤ 10 plain) and
`runs/profile_20260904_retry/aggregate.md` (exhaust plain, cProfile):
wall, the variational-fit share, `_gp_log_joint` / `_neg_elcbo` /
`_eval_full_elcbo` / `final_boost` buckets and call counts. The other
stages must be within a few percent (nothing in them changed) or the
machine state is suspect (probe). Record here, in the roadmap, and as a
dated addendum in devlog §2/§10.

## Steps

- [x] Plan reviewed (one read-only Opus agent: formulas vs the code,
      shapes, gates); findings folded in
- [~] Step 1 vectorized mean and gradients; bit-check; oracles; unit and
      FD tests; replay; full suite (running); commit
- [ ] Step 2 multi-RHS variance; oracles; tests; replay; full suite; commit
- [x] Step 3 tests for the fixes (land with Step 1): FD at `D ≠ K`, FD
      with `jacobian_flag=False`, `separate_K` without variance, the two
      raises, the `L_chol=False` branch against a hand-built Cholesky-form
      posterior
- [x] Step 4 `--probe` in `profile_suite.py`; `X_init` in the traces and
      the design certificate in the replay
- [ ] Step 5 profile campaign (plain with probes; cProfile on the D = 4
      set); write-up
- [ ] Records: `dev/README.md`, roadmap, devlog §2/§9/§10, this file;
      commit and push; CI smoke green
- [ ] Read-only Opus code review of the commits; `/doublecheck`

## Verification

- [ ] `pytest pyvbmc/testing/oracles` green after each step with no
      re-baseline
- [ ] Bit-check old vs new on the eight snapshots and random states:
      max relative difference per output recorded (expected ≤ 1e-13 on
      `G`, `dG`, `I_sk`; `Z` bit-identical)
- [ ] `test_variational_optimization*.py` green (MATLAB-pinned values,
      FD checks, single-sample regressions, the new tests)
- [ ] Replay after each step: iteration 0 identical on every config,
      finals inside the population envelope
- [ ] `pytest --reruns=5 -x` green after each step
- [ ] Profile: wall and variational-fit share per config against the
      2026-09-04 numbers; untouched stages within a few percent; probe
      start/end within ~5 %
- [ ] CI smoke green on the push

## Decisions

- **Vectorize over `s` as well as `k`**, not `k` only: the per-sample
  Python overhead (~30 NumPy calls × Ns ≤ 14) would remain 0.5–1 ms per
  call, i.e. 2–9 % of a D = 4 run at 3k–7k calls; the tensors are a few
  MB at every supported operating point (Findings), so nothing is gained
  by keeping the outer loop.
- **`einsum` for the contractions over `n`, broadcasting for the rest**
  (decided after measurement; the plan first said `np.matmul`). Batched
  matmul on the `(Ns K, D, N)` batch dispatches to BLAS per slice and is
  the fastest at small shapes (0.34 ms against 0.45 for the core at
  D = 4, K = 15, N = 95, Ns = 8), but once the `(Ns, K, D, N)` tensor
  leaves the cache every pass over it costs about 1 ms on this laptop and
  the matmul design needs five passes (write `delta`, write `delta²`, read
  it for `Σ_d`, read both for the two products): 6.1 ms at D = 15,
  K = 26, N = 345, Ns = 4. `einsum` with `optimize` left at its default
  runs its own C loops (no BLAS), needs one write and three reads, never
  materializes `delta²`, and takes 3.2 ms there; chunking the matmul form
  over `s` at 2^16 elements gives 0.32 / 3.7 ms and cannot help `Ns = 1`.
  At the whole-run level the difference is under 1 % either way; one
  code path, one large temporary and the best large-shape behaviour
  decided it. Consequence: `z`'s bit-identity with the old code rests on
  `einsum`'s reduction order (it moved by an ulp on large arrays in the
  bit-check), which is unspecified behaviour; the docstring promises no
  bit-identity, the oracle at 1e-6 / 1e-3 is the gate.
- **Item 2 folded in as a separate commit.** The pair loop is inside
  the loop being removed; leaving it as a per-pair loop over the new `Z`
  and rewriting it again later would mean writing scaffolding twice. The
  roadmap ordered 1 → 8 → 2 by weight, not by dependency; item 8 is a
  gpyreg PR and is unaffected. Recorded in the roadmap as done with
  item 1.
- **No chunking**: see Findings on memory. Revisit only if a profile
  shows the D = 15 exhaust run's variational fit memory-bound.
- **Rounding-level differences accepted, bit-identity not pursued**
  (item 3's rule). The one place where a free choice keeps identity, the
  `Σ_d` over the second-to-last axis, is kept sequential; the last-axis
  sums (`ln tau`, the quadratic term) are not contorted.
- **The three §9 fixes ride along** because the loop that contained them
  is deleted, plus the 1-D `eta` softmax Jacobian found by the review;
  each is covered by a test. Porting the diagonal variance approximation
  is not attempted.
- **Two-solve form for item 2, not the Gram form** (review): it keeps
  today's summands and only changes the final contraction, so `J_sjk`,
  the one variance-class output with a measured cross-BLAS floor of
  2e-11 absolute against a 1e-8 tolerance, moves as little as possible.
- **The replay's initial-design certificate is the design itself, not the
  iteration-0 ELBO** (found on the first Step 1 replay): the trace now
  stores `X_init`; against the 2026-09-03 baseline, which lacks it, a
  design point of the new run found live in the reference certifies the
  design, and where warm-up trimming removed the whole design (cigar) the
  trace cannot certify it and says so instead of flagging.

## Open questions (defaults in bold)

1. Should the dead `compute_vargrad` branch be deleted or ported? **Deleted
   here**, raises kept; porting MATLAB's `compute_var == 2` path is a
   feature for a later stage if anyone wants the ELCBO gradient with
   `beta ≠ 0` (no option enables it today). The roadmap's Stage 0 line
   "finite-difference checks for `compute_vargrad`" is then moot; the
   roadmap is updated to say so.
2. Symmetrize `J_sjk` by mirroring the lower triangle (exactly symmetric,
   as today) or leave the GEMM's rounding asymmetry? **Mirror**: costs
   `K²`, keeps the stored array's invariant.

## Risks

- A shape slip in the block layout of `dG` (d-fastest mu block, then
  sigma, lambd, w) would pass a value check and fail the FD tests and the
  oracle; the FD tests at `D ≠ K` do not exist for `_gp_log_joint` (only
  for `_vp_bound_loss`): add the `D = 3, K = 2` variant of the FD check.
- `I_sk` for `K = 1` and `Ns = 1` squeeze paths: covered by the
  `normal_D2_K1` and `normal_D2_singlesample` oracles.
- The replay parts early everywhere (G moves by an ulp on the first Adam
  step); a config's final can land outside the envelope by chance; a
  second seed decides, as in item 3.
- Profiling must run alone and cool (item 3's throttling lesson; the
  probe makes it visible).

## Results

(to be written)

## Execution tracker

Legend: `[ ]` not started, `[~]` in progress, `[x]` done, `[!]` needs
attention. Times are wall clock on 2026-09-04/05.

- [x] Plan written — 23:10; read-only Opus review dispatched 23:20
- [x] Step 1 code in place — 23:35 (first with `np.matmul` for the
  contractions; switched to `einsum` at 23:55 after timing, see below)
- [x] **Bit-check old vs new** (scratch `bitcheck_gp_log_joint.py`, the
  eight snapshots and 23 random states at D 2–15, N 20–750, Ns 1–8, three
  mean functions, noisy and noiseless): random states agree to 1e-15;
  on the ill-conditioned snapshots the per-element error under the oracle
  criterion is `dG` 2.4e-9 (cigar_boosted, averaged) / 6.3e-9 (per
  sample), 5.3e-8 / 1.9e-7 on corr_D5_warped (abs 1e-9), `I_sk` up to
  3e-9, `var_ss` up to 5e-6 (abs 1e-12; it inherits the ulp moves of the
  per-sample `G`), `J_sjk` bit-identical with the matmul version and
  2.7e-12 absolute (1.1e-5 under the criterion on cigar) with the einsum
  version (`einsum`'s `Σ_d` moves `z` by an ulp). Same order as the
  cross-BLAS floors the tolerance classes were set from (cigar `dG`
  1.2e-8 / 2.9e-8 on Ubuntu). `jacobian_flag=False` now returns 304
  instead of 200 entries on the K = 50 snapshot (the fix)
- [x] **Timing of the contractions** (scratch, one thread, ms per call):
  one-shot `matmul` on the `(Ns K, D, N)` batch is fastest at small shapes
  (0.34 at D = 4, K = 15, N = 95, Ns = 8) but memory-bound at large ones
  (6.1 at D = 15, K = 26, N = 345, Ns = 4: the 4 MB tensors leave the
  cache and every pass costs ~1 ms); `einsum` (`Σ_d delta²` as
  `'skdn,skdn->skn'`, the two contractions as `'skdn,skn->skd'` and the
  three-operand `'skdn,skdn,skn->skd'`, so `delta²` is never
  materialized) is 0.45 / 3.2 at those shapes; chunking over `s` at 2^16
  elements with matmul 0.32 / 3.7 and does nothing for `Ns = 1`.
  Decision: one-shot `einsum`, one large temporary, no chunking
- [x] Oracle gate — 23:40 and 23:58: **100 passed, 15 skipped** both
  versions, no re-baseline
- [x] `test_variational_optimization*.py` — 18 passed; the whole function
  old → new: D = 4, K = 15, N = 95, Ns = 8: 6.5 → 0.58 ms (11×); D = 5,
  K = 17: 9.7 → 0.70 (14×); D = 10, K = 25, N = 185: 9.2 → 2.2 (4.3×);
  D = 15, K = 26, N = 345, Ns = 4: 9.0 → 4.3 (2.1×); D = 15, K = 31,
  N = 750, Ns = 1: 4.3 → 3.0 (1.5×); the variance path (pair loop still in
  place) 1.4–2.3× from the reuse of `z`
- [x] Step 3 tests written — 00:05: FD at `D = 3, K = 2` on a random GP,
  FD with `jacobian_flag=False` (w free, `dG/dw = mean_s I_sk`),
  `separate_K` without variance (`J_sjk is None`, `I_sk @ w == G`), the
  two `NotImplementedError`s; 7 `gp_log_joint` tests pass
- [~] Step 1 replay — 00:00 → `runs/golden/replay_item1_step1/`. First
  rows: `normal_D5`, `banana_D2`, `halfnormal_D2` all **"ITERATION 0
  DIFFERS"** with the first 10–11 live points identical: the ELBO of
  iteration 0 is itself computed by `_gp_log_joint`, so an ulp change in
  `G` parts the ELBO path at iteration 0 *exactly* while agreeing to 1e-6
  for several iterations. The replay's iteration-0 criterion (exact ELBO
  at iteration 0, item 3's design, where the ELBO arithmetic was
  untouched) conflated the initial design with the numerics run on it.
  **Fixed in `golden_replay.py`**: `initial_design_ok` = first live point
  bit-identical and the iteration-0 ELBO within 1e-6 (a different design
  moves it by far more); the flag text says "beyond 1e-6"; docstring and
  legend updated. The finished run is re-rendered with `--report-only`
- [x] Step 1 replay re-rendered with the corrected criterion — 00:30
  (`--report-only`): `normal_D5` parted at iteration 0 (within 1e-6 to
  iteration 4), `banana_D2` at 1 (to 3), `halfnormal_D2` at 0 (to 8),
  `rosenbrock_D2_noise1` at 1 (to 10), all with 5–11 leading live points
  identical and finals inside the envelope (all three metrics improve or
  stay on normal, halfnormal, rosenbrock; banana ΔLML 0.042 → 0.091
  against a fence of 0.124, gsKL 0.20 → 0.23 against 0.45).
  **`cigar_D4`: parted at iteration 0 by 0.36 in the ELBO (−767.898 →
  −767.538), no live point shared with the baseline (0 of 108), finals
  inside the envelope** (ΔLML 0.0066 vs fence 0.045, gsKL 6.9e-5 vs
  8.1e-3, 125 evaluations vs [100, 180]). Still flagged by the "first
  live point identical" fallback: no design point is live in either
  cigar trace (the uniform design on a cigar has `y ≈ −700` and warm-up
  trimming removes all of it; the first live rows have `y ≈ 4–6`), so
  the stored trace cannot certify the design there
- [x] **Plan review (Opus, read-only) returned 21 findings, no blocker**
  — 00:35; folded into Findings/Design/Decisions: callers
  (`active_sample.py:337` passes `jacobian_flag=0`; warm-up runs with
  `optimize_weights=False`, oracle-gated by two snapshots; the live
  no-gradient full-variance `separate_K=False` shape), the rounding
  bullet (`Σ_d ln tau` already pairwise today, hence bit-identical; the
  sigma-gradient `Σ_d` is the one that moves), `B − zalpha`'s error
  bound stated honestly, the two-solve form for item 2, the unreachable
  and untested `L_chol=False` branch, the 1-D `eta` softmax Jacobian the
  rewrite silently fixed, the memory figures (`k_fun_max`, single tensor,
  37 MB corner), the einsum decision. Code changes from it: the defensive
  copies of the VP arrays restored
- [x] **Sensitivity experiment on `cigar_D4` seed 0** (scratch
  `perturb_experiment.py`: the old loop code monkey-patched in, run
  through the replay) — 00:40–00:50: *control* (old code unchanged)
  reproduces HEAD's trajectory exactly (parted from the pre-item-3
  baseline at iteration 2 as item 3's replay did, same finals 2e-5 /
  9.6e-4 / 7.1e-3 / 125 evaluations); *G × (1 + 2⁻⁵²)* parts at iteration
  0 exactly but stays within 1e-6 through iteration 1 and ends on the
  control's trajectory. Direct measurement on the in-process state
  (scratch `iter0_state.py`): on the iteration-0 GP (N = 10, `y` from
  −10674 to −108) old and new agree to 2e-15 on `dG` and 7e-15 on `G`
  (an ulp), on the iteration-1 GP (N = 15, `Σ|alpha| / |Σ alpha| = 2e6`)
  to 1.3e-8 on per-sample `dG`, on the iteration-2 GP (N = 20, 4e6) to
  2.9e-7 on per-sample `dG` and 3.5e-8 on `G`: the warm-up cigar GP is
  far worse conditioned than any oracle snapshot. **Confirmation, 00:55:
  the old code with `dG × (1 + 2⁻⁵²)` (one ulp on the gradient, value
  untouched) moves the iteration-0 ELBO by 0.44 (−767.898 → −767.462; the
  vectorized code: 0.36), iteration 1 by 34, and ends at ΔLML 0.039 /
  gsKL 0.017, the latter *outside* the population fence (0.0081)**; with
  `dG × (1 − 2⁻⁵²)` the iteration-0 ELBO moves by 0.62 (→ −768.520) and
  the run ends inside the envelope (ΔLML 0.0018, gsKL 0.0013, 135
  evaluations). So the warm-up variational optimization
  on cigar_D4 seed 0 is chaotic to a one-ulp gradient perturbation, the
  vectorized code behaves as any rounding change does there, and the
  envelope check on this config has a visible false-alarm rate even for
  ulp-level changes (item 3's "a second seed decides" rule stands). The
  Step 1 run's cigar finals are well inside the envelope
- [x] Replay tooling: `golden_trace.py` now stores the initial design
  (`X_init`, `y_init`: every evaluation before the first GP fit, live or
  trimmed) and `golden_replay.py` certifies the design from it (exactly
  when both traces have it; against the 2026-09-03 baseline by finding a
  design point of the new run among the reference's live rows; "not
  certifiable" and no flag where trimming removed the whole design, as
  on cigar: the `dG`-perturbed run shows "none of the 10 design points is
  live in the reference trace"). A missing `import numpy` in the new
  block crashed the first two runs after the traces were written (fixed;
  reports re-rendered with `--report-only`, traces intact)
- [x] `profile_suite.py --probe CONFIG` — 00:20 (item 3 follow-up): the
  named config runs plain before and after the campaign as
  `probe_start_<cfg>` / `probe_end_<cfg>`, the two walls and their ratio
  are printed, the aggregate lists them as rows under their tag
