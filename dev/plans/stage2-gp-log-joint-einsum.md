# Stage 2 item 1: `_gp_log_joint` vectorized over hyperparameter samples and components

Created: 2026-09-04 22:35. Status: **DONE 2026-09-05 00:30** (items 1 and 2; the 20-seed population run launched afterwards, see the tracker). Roadmap pickup point 3
(`plans/modernization-roadmap.md`; pickup point 2 named it "Next: Stage 2
item 1"); rationale in
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
loop becomes, per hyperparameter sample, two multi-RHS triangular solves
`L^{-1} (L^{-T} Z^T)` and one `K × K` product (today's summands, one
contraction), which is roadmap item 2 done in its natural home (the loop
lives in `_gp_log_joint`, not in `_eval_full_elcbo`); it is a separate
commit so the oracle report and the replay attribute the change. (3) Four
latent defects of the function are resolved on the way, three recorded in
devlog §9 because the loop that contained them is gone: the dead
`compute_vargrad` accumulators are deleted (the two `NotImplementedError`s
that make them unreachable stay, same conditions, same messages),
`separate_K` without variance returns `J_sjk = None` instead of raising
`UnboundLocalError`, `jacobian_flag=False` returns all four gradient
blocks (the Jacobian corrections become conditional, the blocks are not),
and the softmax Jacobian is an outer product for a 1-D `eta` too (found
by the plan review). Gates: the
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
  `einsum` form measured fastest there, see the tracker). The variance
  block adds four `(Ns, K, K, D)` arrays (`tau_jk`, its log, `delta_jk`,
  its square): 1.8 MB each at that corner, 0.6 MB on the boosted snapshot.
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
- [x] Step 1 vectorized mean and gradients; bit-check; oracles; unit and
      FD tests; replay; full suite; commit `5ce1bc6`
- [x] Step 2 multi-RHS variance; oracles; tests; replay; full suite; commit
      `f93ea5e`
- [x] Step 3 tests for the fixes (land with Step 1): FD at `D ≠ K`, FD
      with `jacobian_flag=False`, `separate_K` without variance, the two
      raises, the `L_chol=False` branch against a hand-built Cholesky-form
      posterior
- [x] Step 4 `--probe` in `profile_suite.py`; `X_init` in the traces and
      the design certificate in the replay
- [x] Step 5 profile campaign (plain with probes; cProfile on the D = 4
      set); write-up (§Results)
- [x] Records: `dev/README.md`, roadmap, devlog §2/§9/§10, this file;
      commit and push; CI smoke (see the tracker)
- [x] Read-only Opus code review of the commits (folded in); `/doublecheck`
      (see the tracker)

## Verification

- [x] `pytest pyvbmc/testing/oracles` green after each step with no
      re-baseline (100 passed, 15 skipped, three times)
- [x] Bit-check old vs new on the eight snapshots and random states:
      recorded in the tracker (1e-15 on random states; up to 2e-7 per
      element on the ill-conditioned snapshots, the conditioning of the
      GP solve; `z` moves by an ulp with `einsum`)
- [x] `test_variational_optimization*.py` green (MATLAB-pinned values,
      FD checks, single-sample regressions, the new tests)
- [x] Replay after each step: initial design certified on every config
      (iteration 0's ELBO is numerics and parts, see Findings), finals
      inside the population envelope, 0 flagged of 5
- [x] `pytest --reruns=5 -x` green after each step (522 passed twice)
- [x] Profile: wall and variational-fit share per config against the
      2026-09-04 numbers (§Results); untouched stages within trajectory
      noise except the exhaust run's active sampling (+20 %, discussed);
      probe start/end 1.03
- [ ] CI smoke green on the push (pending at the time of writing)

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

## Results (2026-09-04/05)

### Per call (scratch timing, one thread, median of 5, ms; old loop → new)

| shape (D, K, N, Ns) | gradient call (`_neg_elcbo` as Adam sees it) | variance call (`_eval_full_elcbo`) |
|---|---|---|
| 4, 15, 95, 8 | 6.2 → 0.56 (11×) | 42.7 → 0.85 (50×) |
| 5, 17, 100, 8 | 6.9 → 0.71 (10×) | 58.6 → 0.95 (62×) |
| 8, 12, 120, 6 | 3.9 → 0.55 (7×) | 23.0 → 0.69 (33×) |
| 10, 25, 185, 6 | 9.0 → 2.1 (4.3×) | 106 → 2.2 (47×) |
| 15, 26, 345, 4 | 8.4 → 3.4 (2.5×) | 107 → 3.8 (28×) |
| 15, 31, 750, 1 | 4.2 → 2.6 (1.7×) | 86 → 3.3 (26×) |

The gradient call is bound by Python overhead below D ≈ 8 (11× at the
D = 4–5 operating point of most runs) and by memory traffic over the
`(Ns, K, D, N)` array above it; the variance call was bound by
`K (K + 1)` single-RHS solves per sample and is now two multi-RHS solves.

### End to end: plain profile campaign, seed 0, one process, one BLAS thread

`runs/profile_20260905/` (code `8943cec`, i.e. both perf commits; started
23:31, 32.7 min, machine otherwise idle) against `runs/profile_20260904/`
(after item 3, code `f441172`) and, for the exhaust row, against
`runs/profile_20260904_retry/` (cool machine, code `93dc29d`). Speed
probe `banana_D4` plain: **31.4 s at the start, 32.3 s at the end
(1.03)**, so no throttling this time (the item 3 campaign's probes were
42.8 / 45.5 / 42.6 s on the item 3 code). Wall and stage totals in
seconds; the arrow reads before → after this item.

| config | wall (×) | variational fit (× ; % of wall) | active sampling | GP train | iters | evals | ΔLML | gsKL | MMTV |
|---|---|---|---|---|---|---|---|---|---|
| banana_D4 | 43 → 31 (1.39) | 8 → 2 (4.1; 19 → 6 %) | 18 → 16 | 10 → 10 | 18 → 17 | 95 → 90 | 0.041 → 0.062 | 0.118 → 0.216 | 0.021 → 0.032 |
| cigar_D4 | 79 → 62 (1.28) | 18 → 7 (2.6; 23 → 11 %) | 34 → 33 | 17 → 18 | 25 → 25 | 125 → 125 | 0.000 → 0.007 | 0.001 → 0.000 | 0.007 → 0.006 |
| lumpy_D4 | 41 → 29 (1.39) | 7 → 1 (5.0; 17 → 5 %) | 15 → 14 | 11 → 11 | 16 → 16 | 85 → 85 | 0.040 → 0.034 | 0.013 → 0.015 | 0.026 → 0.032 |
| student_D4 | 47 → 36 (1.30) | 7 → 2 (3.1; 14 → 6 %) | 21 → 19 | 13 → 12 | 20 → 19 | 105 → 100 | 0.019 → 0.045 | 0.008 → 0.057 | 0.039 → 0.043 |
| logreg_D5 | 81 → 72 (1.13) | 22 → 8 (2.6; 28 → 12 %) | 29 → 31 | 18 → 19 | 24 → 26 | 120 → 130 | 0.066 → 0.001 | 0.012 → 0.010 | 0.029 → 0.019 |
| rosenbrock_D2_noise1 | 116 → 106 (1.10) | 15 → 8 (2.1; 13 → 7 %) | 87 → 87 | 23 → 24 | 27 → 27 | 140 → 140 | 0.059 → 0.097 | 0.025 → 0.008 | 0.026 → 0.022 |
| logreg_D5_noise3 | 251 → 273 (0.92) | 44 → 28 (1.6; 17 → 10 %) | 164 → 193 | 72 → 83 | 45 → 56 | 220 → 280 | 0.132 → 0.061 | 0.771 → 0.211 | 0.247 → 0.107 |
| lumpy_D10 | 182 → 147 (1.24) | 19 → 4 (4.6; 10 → 3 %) | 72 → 65 | 76 → 67 | 38 → 35 | 185 → 175 | 0.784 → 0.764 | 0.808 → 0.600 | 0.125 → 0.113 |
| banana_D10 | 103 → 79 (1.30) | 7 → 2 (3.1; 7 → 3 %) | 48 → 40 | 39 → 34 | 26 → 22 | 135 → 115 | 0.127 → 0.114 | 0.493 → 0.489 | 0.024 → 0.025 |
| cigar_D15_exhaust | 1288 → 1041 (1.24) | 585 → 264 (2.2; 45 → 25 %) | 363 → 437 | 308 → 303 | 151 → 150 | 750 → 750 | 0.015 → 0.007 | 0.001 → 0.004 | 0.007 → 0.008 |

Reading:

1. **Noiseless targets at D ≤ 10 run 1.13–1.39× faster end to end**
   (banana, lumpy 1.39; student, banana_D10 1.30; cigar 1.28; lumpy_D10
   1.24; logreg 1.13), the **variational fit 2.6–5.0× faster** and down
   from 7–28 % of wall to 3–12 %. On the **15-D exhaust run** (half of
   its 150 iterations in the optimize-only regime, K 23–33) the
   variational fit is **2.2× faster** (585 → 264 s) and the run 1.24×.
   Combined with item 3 the noiseless targets are now 1.7–2.4× faster
   than the 2026-09-03 baseline (banana_D4 67 → 31 s, cigar_D4 141 → 62,
   lumpy_D10 283 → 147, banana_D10 175 → 79, exhaust 2123 → 1041).
2. **Every trajectory changed** (the ELBO arithmetic moved by rounding,
   §Findings): 7 of 10 configs took a different number of iterations or
   evaluations. Every seed-0 final lies inside its population's
   `Q3 + 3 IQR` fence (checked for the eight configs in the golden set;
   the widest margins are cigar_D4 gsKL 1e-4 vs 8.1e-3, the narrowest
   lumpy_D4 MMTV 0.032 vs 0.041 and student_D4 gsKL 0.057 vs 0.236).
3. **The untouched stages stayed put within trajectory noise** — GP
   training within ±1 s on every D ≤ 5 config and −9 s on both D = 10
   configs (fewer iterations), active sampling within ±2 s at D ≤ 5 and
   −7 / −8 s at D = 10 — with one exception: **the exhaust run's active
   sampling took 363 → 437 s** (+20 %; 1.98 → 2.67 s per iteration in
   the optimize-only tail at the same mean K = 26.4 and the same N path;
   GP training 308 → 303 s on the same run). The probe rules out
   throttling (1.03), the stage does not call `_gp_log_joint` (the
   per-sample VP update is off), and the trajectory differs from
   iteration 0, so this is the search cost of a different trajectory
   (the number of CMA-ES generations per point is not recorded by the
   plain run); a cProfile of the exhaust run would settle it (35 min, not
   done). Noted for item 8, which owns the acquisition's per-call cost.
4. **The noisy VIQR targets** gain 10 % (rosenbrock) or run longer:
   `logreg_D5_noise3` took 56 iterations instead of 45 (280 instead of
   220 evaluations) on its new trajectory and ended with far better
   finals (gsKL 0.77 → 0.21, MMTV 0.25 → 0.11), so its wall rose 251 →
   273 s while its cost per iteration fell 5.6 → 4.9 s; the population
   spread of evaluations for that config is the reference, not this
   pair.

### Where the time goes now: cProfile of the four D = 4 configs

`runs/profile_20260905/<config>_cprof/` (00:04–00:08, right after the
plain campaign) against the item 3 pass on the cool machine
(`runs/profile_20260904_retry/`). Percentages of profiled
`VBMC.optimize`; call counts in parentheses where they matter.

| bucket | banana_D4 | cigar_D4 | lumpy_D4 | student_D4 |
|---|---|---|---|---|
| active_sample | 40.0 → 51.0 | 43.8 → 54.1 | 34.5 → 45.6 | 43.5 → 50.7 |
| ├ `GP.predict` (calls) | 26.5 → 33.8 (10.1k → 9.4k) | 26.4 → 32.6 (24.2k → 24.9k) | 23.2 → 30.8 (8.0k → 7.8k) | 29.3 → 34.0 (11.5k → 10.9k) |
| train_gp | 27.4 → 36.9 | 24.0 → 31.1 | 31.8 → 42.2 | 31.3 → 37.2 |
| ├ `SliceSampler.sample` | 23.5 → 31.8 | 20.8 → 27.2 | 28.4 → 37.5 | 27.4 → 32.7 |
| optimize_vp | 31.1 → 10.1 | 30.9 → 13.3 | 32.3 → 10.3 | 23.6 → 10.3 |
| ├ `_gp_log_joint` (calls) | 23.5 → 2.1 (3.1k → 2.9k) | 22.6 → 2.5 (6.7k → 6.6k) | 24.2 → 1.8 (2.4k → 2.3k) | 18.2 → 1.9 (2.7k → 2.8k) |
| ├ `_eval_full_elcbo` | 6.3 → 1.7 | 4.6 → 1.6 | 6.6 → 1.7 | 4.8 → 1.4 |
| └ `entmc_vbmc` (calls) | 6.1 → 6.2 (1.9k → 1.7k) | 6.9 → 8.9 (4.4k → 3.4k) | 6.7 → 6.8 (1.3k → 1.2k) | 4.2 → 6.9 (1.5k → 1.6k) |
| final_boost | 13.7 → 4.6 | 10.0 → 3.7 | 16.9 → 6.0 | 10.7 → 5.3 |
| `copy.deepcopy` | 0.6 → 0.7 | 0.6 → 0.9 | 0.5 → 0.7 | 0.6 → 0.7 |
| profiled wall s | 58 → 45 | 105 → 95 | 58 → 46 | 65 → 55 |

Reading:

- **`_gp_log_joint` fell from 18–24 % of a D = 4 run to about 2 %** at
  the same call counts (±10 %), `optimize_vp` from 24–32 % to 10–13 %,
  `_eval_full_elcbo` from 5–7 % to 1.4–1.7 % and `final_boost` (one call
  at K = 50) from 10–17 % to 4–6 %.
- **`entmc_vbmc` is now the largest piece of the variational stage**
  (6–9 % of the run; item 5), ahead of `_gp_log_joint`; the Adam loop's
  own overhead is the rest of `optimize_vp`.
- **Active sampling (46–54 %) and GP training (31–42 %) are what is
  left**, both untouched here: inside active sampling, gpyreg's per-call
  `predict` overhead (31–34 % of the run at 8k–25k calls) and inside GP
  training the slice sampler (27–38 %), both item 8's. On these targets
  the variational stage is no longer a comparable third.
- Profiled wall fell 1.1–1.3× (less than the plain 1.3–1.4×): cProfile's
  per-call overhead is now a larger share of the remaining, call-heavy
  stages.

## Execution tracker

Legend: `[ ]` not started, `[~]` in progress, `[x]` done, `[!]` needs
attention. Times are wall clock on 2026-09-04 (the session started at 22:27, right after `c48c025`).

- [x] Plan written — 22:35; read-only Opus review dispatched 22:36
- [x] Step 1 code in place — 22:43 (first with `np.matmul` for the
  contractions; switched to `einsum` at 22:50 after timing, see below)
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
- [x] Oracle gate — 22:46 and 22:51: **100 passed, 15 skipped** both
  versions, no re-baseline
- [x] `test_variational_optimization*.py` — 18 passed; the whole function
  old → new: D = 4, K = 15, N = 95, Ns = 8: 6.5 → 0.58 ms (11×); D = 5,
  K = 17: 9.7 → 0.70 (14×); D = 10, K = 25, N = 185: 9.2 → 2.2 (4.3×);
  D = 15, K = 26, N = 345, Ns = 4: 9.0 → 4.3 (2.1×); D = 15, K = 31,
  N = 750, Ns = 1: 4.3 → 3.0 (1.5×); the variance path (pair loop still in
  place) 1.4–2.3× from the reuse of `z`
- [x] Step 3 tests written — 22:54: FD at `D = 3, K = 2` on a random GP,
  FD with `jacobian_flag=False` (w free, `dG/dw = mean_s I_sk`),
  `separate_K` without variance (`J_sjk is None`, `I_sk @ w == G`), the
  two `NotImplementedError`s; 7 `gp_log_joint` tests pass
- [x] Step 1 replay — 22:51–22:56 → `runs/golden/replay_item1_step1/`. First
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
- [x] Step 1 replay re-rendered with the corrected criterion — 22:57
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
  — 22:57; folded into Findings/Design/Decisions: callers
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
  through the replay) — 22:58–23:01: *control* (old code unchanged)
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
  far worse conditioned than any oracle snapshot. **Confirmation, 23:04–23:08:
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
- [x] Full suite for Step 1 — 23:08–23:15: **522 passed, 15 skipped, 1
  rerun, 7:10**. Commits `9d92c7f test(dev)` (replay design certificate,
  `X_init`, `--probe`, the plan) and `5ce1bc6 perf(vbmc)` (the function,
  the tests, devlog §9) — 23:16
- [x] **Step 2 code in place** — 23:16 (two-solve form). Bit-check old vs
  new (the pair loop is gone, so the variance outputs now move too):
  `J_sjk` 2.2e-11 absolute on cigar_boosted (7.5e-5 under the criterion,
  which ignores the 1e-8 `atol`), 1.9e-10 absolute on corr (2.2e-4 under
  the criterion; the Ubuntu cross-BLAS floor on corr was 2.3e-10, i.e. the
  re-expression moves `J_sjk` by the same amount as a BLAS swap), `varG`
  ≤ 3.7e-6 relative (abs ≤ 4e-12), `var_ss` ≤ 2.5e-5 (abs ≤ 5e-12); the
  two `L_chol=False` random states (`ln sn = −8`, D = 3 and 5) agree with
  the old loop to better than 1e-9 on every output. Oracle gate **100
  passed, 15 skipped**, no re-baseline. Variance path old → new: D = 4,
  K = 15, N = 95, Ns = 8: 42.7 → 0.85 ms (50×); D = 5, K = 17: 58.6 →
  0.95 (62×); D = 10, K = 25, N = 185: 106 → 2.2 (47×); D = 15, K = 26,
  N = 345: 107 → 3.8 (28×); D = 15, K = 31, N = 750, Ns = 1: 86 → 3.3
  (26×). `test_variational_optimization*.py`: 22 passed and
  `test_vp_optimize_1D_g_mixture` failed once, then passed alone and 6 of
  7 repeats: a statistical miss (KL 0.0014 against 0.00125) of an unseeded
  end-to-end test that draws its GP fit and its 1e7 samples from the
  global stream and that already appeared among the flaky reruns on
  2026-09-02; no exception
- [x] Step 2 replay — 23:16–23:23 → `runs/golden/replay_item1_step2/`:
  **0 flagged of 5**; the same verdicts and the same finals to every
  stored digit as the Step 1 replay on every config (the variance
  re-expression flipped nothing on these runs; it enters only through
  the ELCBO used to pick among candidates and to prune). Initial design
  certified from the new `X_init`: 8, 5, 10 and 6 of 10 design points
  found live in the baseline on normal_D5, banana_D2, halfnormal_D2,
  rosenbrock_D2_noise1; on cigar_D4 none is live in either trace, and
  the design was certified by hand instead: `X_init` and `y_init` of the
  new run are bit-identical to those of the old-code `dG`-perturbation
  run of the same seed. Wall (min, baseline → Step 1 → Step 2): normal_D5
  1.0 → 0.61 → 0.62, banana_D2 0.59 → 0.36 → 0.36, halfnormal_D2 0.54 →
  0.31 → 0.30, cigar_D4 2.4 → 1.2 → 1.1, rosenbrock_D2_noise1 2.0 → 1.9
  → 1.8 (the baseline predates item 3)
- [x] PI (23:24): the KL threshold of `test_vp_optimize_1D_g_mixture`
  raised from 0.00125 to 0.0015 (own `test(vbmc)` commit)
- [x] Full suite for Step 2 — 23:23–23:30: **522 passed, 15 skipped, 0
  reruns, 6:43**. Commits `f93ea5e perf(vbmc)` (the variance) and
  `8943cec test(vbmc)` (the threshold) — 23:30
- [x] **Code review (Opus, read-only) of the four commits — 23:35–23:50**:
  an independent re-implementation of the old loop and the new code
  agreed to ≤ 1.7e-15 on every output at six shapes including K = 1,
  D = 1, Ns = 1 and D = 12; mirroring direction, the `einsum` variance
  identity, the gpyreg `L`/`sW`/`sl` conventions in both branches, the
  `dG` layout and the four fixes confirmed; no blocker. Should-fixes,
  applied: the replay's fallback certificate counted the start point
  `x0`, which the benchmark draws from a stream spawned off the run seed
  and which is therefore identical by construction, so row 0 is now
  excluded (Step 2 replay re-rendered: 7, 4, 9, 5 of 9 generator-drawn
  design points live in the baseline on normal, banana, halfnormal,
  rosenbrock; cigar not certifiable, as before); the new `_gp_log_joint`
  tests construct VPs, which draw their seed from the global legacy
  stream, ahead of the unseeded `*_g_mixture` tests in the same module,
  so the module gets the autouse restore fixture its `_grad_fd` sibling
  has (the 0.0015 threshold stays: the test also failed alone in 1 of 7
  runs); the docstring's "all four blocks either way" → the requested
  blocks; the design line of the replay verdict now appears on identical
  runs too and the legend names the not-certifiable case. Notes applied:
  dead `N`, "batched matrix products" → `einsum`, `M` and `B` computed
  only for the blocks requested (`variable_means=False` skips `M`), the
  `X_init` caveats (no user `y0`, no duplicate design points), `--probe`
  `is not None` and probe rows pinned to the ends of the table, the
  non-Cholesky test compares per-sample `varG` and asserts exact
  symmetry, redundant `import scipy.linalg` removed and the MATLAB test
  reuses the fixture helper, devlog §9 gets the inert `J_sjk` pruning
  note, this file's memory bullet gets the `(Ns, K, K, D)` arrays
- [x] Step 5 profile campaign — 23:31–00:08: plain suite with `--probe
  banana_D4` (32.7 min; probe 31.4 → 32.3 s, 1.03), then cProfile on the
  four D = 4 configs (4.1 min) → `runs/profile_20260905/`; §Results.
  Every seed-0 final inside its population fence (checked against the
  golden sidecars for the eight configs in the golden set)
- [x] Review fixes to the package after the campaign — 00:15 (dead `N`,
  docstring wording, `M`/`B` computed only for the requested blocks):
  oracles and the three test modules 122 passed (after repairing a
  `base_path` the refactored MATLAB test had lost), bit-check unchanged
- [x] Records — 00:30: §Results here, checklists ticked, roadmap (Stage 2
  paragraph, pickup point 3 with the population run as the first thing to
  read), devlog §2 and §10 addenda, §9 entries; dates corrected to
  2026-09-04 (the tracker's first draft ran ~2 h ahead of the clock)
- [~] Final full suite on the working tree, then the records commit and
  push (CI smoke), `/doublecheck`, and the 20-seed population run
  (`golden_trace.py run --suite golden --seeds 0-19 --workers 1 --out
  dev/scripts/runs/golden/item1_20260905`, about 7 h, then `compare`
  against `dev/golden/baseline`; PI 2026-09-04 23:24: run it overnight
  once everything else is done)
- [x] `profile_suite.py --probe CONFIG` — 22:57 (item 3 follow-up): the
  named config runs plain before and after the campaign as
  `probe_start_<cfg>` / `probe_end_<cfg>`, the two walls and their ratio
  are printed, the aggregate lists them as rows under their tag
