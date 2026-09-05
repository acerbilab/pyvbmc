# Stage 2 item 5: `entmc_vbmc` vectorized over components and samples

Created: 2026-09-05 15:00. Status: **IN PROGRESS** (this file is the plan
and the worklog; tracker at the end). Roadmap pickup point 3b
(`plans/modernization-roadmap.md`); rationale in
`dev/2026-09-02-modernization-discussion.md` §2, §3, §10 (Stage 2, item 5:
"Vectorize `entmc_vbmc` component loop"); the profile that made it the next
item in `plans/stage2-gpyreg-predict-and-sampler.md` §Results: after items
3, 1, 2 and 8 the Monte Carlo entropy is 9–13 % of a profiled D = 4 run
(1.7k–3.4k calls) and 228 of 953 profiled seconds on the 15-D exhaust run,
the largest piece of the variational stage and the largest PyVBMC-local
piece left. Method as in items 1 and 2 (`plans/stage2-gp-log-joint-einsum.md`):
same formulas, rounding-level differences accepted, fixed-state oracles on
every commit, the golden replay per step, the profile suite once. Working
constraint: the laptop is in use during the day, so every gate runs as one
process of a few minutes and the two long runs (profile campaign ≈ 40 min,
20-seed population ≈ 6.5 h) wait for an idle window.

## Summary

Same estimator, one pass. `entmc_vbmc` estimates the entropy of the
variational mixture by Monte Carlo with antithetic samples, `Ns` per
component, and its gradient with respect to `(mu, sigma, lambd, w)` by the
reparameterization trick. Today it loops in Python over the `K`
components, draws each component's samples in turn, evaluates the mixture
density at them with an inner loop over the `K` components (six NumPy calls
per pair), and then, when gradients are requested, evaluates the same
density *again* as an `(Ns, D, K)` broadcast for the gradient terms. The
rewrite draws every component's samples with one call (bit-identical to
the per-component draws, so the random stream is untouched), evaluates the
density once as a broadcast over a `(g, Ns, D, K)` tensor of standardized
distances with `einsum` contractions, and takes the gradients from the same
tensors, in blocks of components (and of samples for the largest calls) so
that the tensor never exceeds 2^16 doubles. Per call 7–13× faster at the
shape Adam sees (`ceil(100 K^(2/3) / K)` samples per component, all
gradients) for K = 14–50 at D ≤ 15, 3.8× at the D = 20, K = 60 corner;
the value-only shape of `_eval_full_elcbo` (4096 samples per component,
no gradients) is bound by the arithmetic itself and changes by 0.9–1.35×
(within ±10 % of the loop at D = 4, K ≤ 17; 1.2–1.35× at K ≥ 25). Gates:
the `entmc` and `neg_elcbo` oracles (GP-free, 1e-10, no
re-baseline expected: the draw order is preserved and the values move at
rounding level), the finite-difference tests, the MATLAB-pinned test, the
replay, the full suite; a profile campaign once, compared with
`profile_compare.py`.

## Scope

- **In**: `pyvbmc/entropy/entmc_vbmc.py` (whole body; signature, return
  layout, `rng` semantics and every caller unchanged); a chunk-invariance
  test in `pyvbmc/testing/entropy/test_entmc_vbmc.py`; records
  (`dev/README.md` index, roadmap, devlog §2/§3/§9/§10 dated addenda, this
  file); a correction to two records that misdescribe the entropies'
  `jacobian_flag=False` behaviour (Findings).
- **Out**: `entlb_vbmc` (already `(K, K)` broadcasting except a small loop
  over `K`; 0.2 % of a run); the `_neg_elcbo` orchestration; the softmax
  Jacobian duplicated in four places (devlog §3); the GEMM expansion of
  the density for the value-only path (Open question 1, measured, not
  adopted here); `vp.pdf` (item 3's broadcast, same density, kept as it
  is); the memory items 6 and 7 of the roadmap; the 20-seed population
  (end of stage, or tonight after item 8, roadmap pickup 3a (ii)).

## Findings the plan rests on

Measured 2026-09-05 on `3ce627d` with gpyreg `79b4986`, one BLAS thread,
from the stored oracle snapshots (`pyvbmc/testing/oracles/fixtures/`) and
random states; scratch scripts `prof_entmc.py`, `bitcheck_entmc.py`,
`entmc_variants.py` (session scratchpad, not committed; Follow-ups says how
to rebuild them).

- **Call sites and shapes.** One production caller, `_neg_elcbo`
  (`variational_optimization.py:1181`: `entmc_vbmc(vp, Ns, grad_flags,
  jacobian_flag)`, `vp.rng` as the generator), reached in three shapes:
  (a) the Adam objective (`vb_train_mc_fun`, through `minimize_adam`),
  `Ns = ceil(ns_ent(K) / K) = ceil(100 K^(-1/3))` samples per component
  (42 at K = 14, 28 at K = 50; the total `K Ns ≈ 100 K^(2/3)` is 600–1400
  samples), all four gradients (`(True, True, True, False)` during warm-up,
  since `optimize_weights` is off there); (b) `_eval_full_elcbo`, `Ns =
  ceil(ns_ent_fine(K) / K) = 4096` per component, **no gradients**
  (`compute_grad=False`, so `any(grad_flags)` is false and only the
  density loop runs); (c) `_sieve` with `ns_ent_fast`, which is 0 by
  default and falls to `entlb_vbmc`. `final_boost` runs (a) at `K = 50`
  with `ns_ent_boost = 200 K^(2/3)` (54 per component) and (b) once, and
  the noisy path's per-sample VP update uses `ns_ent_fine_active = 200 K`
  (200 per component). The `entmc` oracle calls it with a fresh
  `default_rng(seed)` at shape (a)'s count; the `neg_elcbo` oracle at (a)
  with gradients and at (b). `K = 1` never reaches it in production (the
  sieve sets `Ns = 0` at `K = 1`), but the `normal_D2_K1` oracle and
  `test_entmc_vbmc_single_gaussian` do.
- **Where the time goes** (instrumented copy of the loop, one call,
  medians): the draws are 1–9 % of a call at every shape; at shape (a) the
  `K × K` density loop is 56–77 % and the gradient block 21–39 %
  (`cigar_D4_largeK` K = 14: 2.3 ms; `corr_D5_warped` K = 17: 3.3 ms;
  `cigar_D4_boosted` K = 50: 20.6 ms; a K = 20 slice of it: 3.9 ms); at
  shape (b) the density loop is 91–97 % (18 ms at K = 14, 32 ms at K = 17,
  222 ms at K = 50). Under cProfile the whole body is one function; the
  60k `sum` calls per 20 calls at K = 50 are the loop's `d2.sum(1)`. So the
  question of pickup point 3b ("the component loop or the sample draws")
  is answered: the loop, and inside it the density, which the gradient
  block recomputes.
- **Share of the two shapes in a run** (item 8's cProfile listings,
  `runs/profile_20260905_item8/*_cprof/profile.prof`, callers of
  `_neg_elcbo`): at D = 4 the `_eval_full_elcbo` calls (44–66 per run)
  cost 0.7–1.3 s of which the entropy is about 0.6–1.1 s, i.e. **22–25 %
  of the entropy's 2.7–8.0 s**; the Adam calls the rest. On the 15-D
  exhaust run the 817 `_eval_full_elcbo` calls cost 84 s (8.9 % of the
  run), about 81 s of it the entropy at K = 20–33: **36 % of the entropy's
  228 s**; the 20k Adam-shape calls about 145 s.
- **Draw-order identity.** `Generator.standard_normal((K, half, D))` fills
  the array in C order from the same stream that `K` sequential draws of
  `(half, D)` would consume, and the two are bit-identical (checked with
  `np.array_equal` at K = 50, half = 14, D = 4; the ziggurat method has no
  cached second value, unlike the legacy `randn`). So the vectorized
  function consumes exactly the draws the loop consumed, in the same order,
  and the `entmc` oracle's "1e-10 while the draw order is unchanged" holds
  without a re-baseline. The order within a component is `[eps; -eps]` as
  today.
- **Rounding.** Elementwise pieces (`Xs = eps * lambd * sigma_j + mu_j`,
  the standardized difference `(Xs - mu_k) / (sigma_k lambd)`, `exp`,
  `log`) are the same scalar operations in the same order and
  bit-identical. What moves: the `Σ_d` of the squared distances goes
  through `einsum` (sequential over `d` in both, but `einsum`'s
  accumulation is its own), the mixture sum `Σ_k w_k norm_k` becomes a
  GEMV (`exp(-d2/2) @ (w · nf)`, with the constant `w_k nconst / sigma_k^D`
  formed as `w_k · (nconst / sigma_k^D)` instead of `(w_k nconst) /
  sigma_k^D`), the gradient contractions over `k` become `einsum`
  products with the factors grouped differently, and the sums over the
  samples are pairwise instead of sequential. Measured against the loop:
  on the eight snapshots at the oracle's seed and sample count, `H` agrees
  to 1.3e-16 relative and `dH` to ≤ 8.6e-14 per element under the oracle
  criterion (`|d| / max(|ref|, q25 |ref|)`); on random states at D 1–20,
  K 1–60, every gradient-flag combination, both `jacobian_flag` values and
  odd or tiny `Ns`, everything agrees at rounding level relative to the
  array's peak, with one structural caveat: the `mu` block of `dH` is
  ill-conditioned by antithetic cancellation whenever `Ns` is small,
  exactly at `K = 1` (the block is zero in exact arithmetic, and both
  implementations return cancellation noise of 1e-17, whose *relative*
  comparison is meaningless; the oracle's `atol = 1e-12` covers it, the
  `normal_D2_K1` snapshot's `dH` differing by 2e-16 absolute) and
  partially above it (the review measured 1.2e-12 of the peak at `D = 5`,
  `K = 2`, `Ns = 2`, where the pair nearly cancels and an ulp in the
  summand lands at 1e-12 of the block). The conditioning is the
  estimator's, identical in the loop; a `mu`-block move of that size at
  small `Ns` is not a regression.
- **Memory.** The `(g, Ns, D, K)` tensor is `K Ns D K = 100 K^(5/3) D`
  doubles at shape (a) (280k = 2.2 MB at K = 50, D = 4; 4.3M = 34 MB at
  the `D = 20`, `K = 106` corner of `k_fun_max`) and `4096 D K²` at shape
  (b) (41M = 328 MB at K = 50, D = 4; 38M at D = 15, K = 25), so the
  value-only calls *must* be blocked. Today's code already holds an
  `(Ns, D, K)` array per component at shape (a) (6.5 MB at K = 50, D = 4;
  shape (b) never builds one, since it requests no gradient). The block
  size is chosen for cache residency, not for memory: item 3 found for
  `vp.pdf` (the same density broadcast) that 2^16-element chunks were
  fastest at every size and 2^22-element chunks memory-bound and slower
  than the loop; measured here across budgets 2^14–2^22 at both shapes,
  2^16 is fastest or within noise of it at shape (a) (K = 50: 1.9 ms
  against 2.5 at 2^14 and 2.7–2.8 at ≥ 2^18) and the best budget at
  shape (b), where it is within ±10 % of the loop at D = 4, K = 14–17
  (20 → 22 ms, 35 → 36 ms in one run, 19 → 19 and 32 → 35 in another) and
  1.2–1.35× faster at K ≥ 25, while ≥ 2^18 is 1.5–1.8× *slower* than the
  loop at K = 50. So: `g = max(1, min(K, 2^16 // (Ns D K)))` components
  per block, and when one component alone exceeds the budget (shape (b) at
  `D K > 16`), blocks of `2^16 // (D K)` samples within it. Per-block
  Python overhead is about 15 NumPy calls; at shape (b), K = 50, D = 4 the
  budget gives 327-sample blocks, 13 per component, 650 blocks per call;
  at the `D = 20`, `K = 106` corner it would be 14.5k blocks per call
  (0.1–0.2 s of dispatch), still less than the loop's `K²` inner
  iterations there. Besides the block tensor, the call keeps one
  `(K, Ns, D)` array of standard-normal draws alive (all components'
  samples are drawn at once to preserve the stream's order): 69 MB at
  that corner's value-only shape, 2 MB at K = 50, D = 4; the samples
  themselves are formed per block.
- **Value-only shape is arithmetic-bound.** At shape (b) the loop's
  `(4096, D)` arrays are already large enough to amortize NumPy's per-call
  overhead, and its work (`D` passes over `M K` distance elements, where
  `M = K Ns`, plus `M K` exponentials) equals the tensor form's. Measured
  (density only, K = 14 / 17 / 50 / 26 (D = 15) / 25 (D = 10) / 60
  (D = 20)): loop 17 / 30 / 213 / 117 / 88 / 805 ms, tensor with
  `einsum` over the `(g, n, D, K)` layout 14 / 27 / 176 / 87 / 58 / 546
  ms (1.15–1.5×), the `(g, n, K, D)` layout with a contiguous reduction
  the same within noise, `np.square` in place then `.sum(-1)` slower
  (the squared tensor is written). A **centered GEMM expansion** of the
  squared distance, `|u|² − 2 u·v + |v|²` over `sigma_k²` with `u = (x −
  mu_j) / lambd`, `v = (mu_k − mu_j) / lambd` and the cross term as a
  batched matrix product, has no tensor at all and takes 6 / 9 / 66 / 32 /
  22 / 116 ms (2.7–7×). Its error is not rounding-level: the cancellation
  in the expansion bounds the absolute error of `d2` by `eps · 2D
  (sigma_j / sigma_k)²` for the terms that matter (with centering; without
  it, by `eps · |x / (sigma lambd)|²`, unbounded in the position of the
  origin). Measured against the loop: 2e-14 on the log density on every
  snapshot, 3e-12 on an adversarial state (components 30 widths apart,
  `sigma ≈ 0.03`), and, when the same expansion is used for the gradient
  contraction `Σ_k (x − mu_k) w_k norm_k / (sigma_k lambd)²` as `u A − B`,
  up to 1e-4 of the gradient's peak on that state (the tensor form: 4e-8
  of the peak there, i.e. rounding on the near-zero entries). This is the
  Gram-form situation of item 2 (`plans/stage2-gp-log-joint-einsum.md`
  §Findings, "Item 2's arithmetic"), where the re-expression was rejected
  for `J_sjk`; the decision here is the same for the gradient path and is
  left open for the value-only path (Open question 1).
- **A record error found on the way.** Devlog §9 (the `jacobian_flag`
  bullet, added 2026-09-04) and `plans/stage2-gp-log-joint-einsum.md`
  §Follow-ups say the entropies return only the `mu` block of the gradient
  at `jacobian_flag=False`. They do not: both `entlb_vbmc` and `entmc_vbmc`
  allocate the blocks by `grad_flags` and concatenate all of them
  unconditionally; `jacobian_flag` only switches the three Jacobian
  corrections, and `test_entmc_vbmc.py`'s finite-difference wrapper relies
  on exactly that (it calls with `jacobian_flag=False` and compares the
  full `D K + K + D + K` gradient). The item 1 fix made `_gp_log_joint`
  behave like the entropies, not unlike them. Corrected in both records
  with this item.
- **Existing tests.** `test_entmc_vbmc.py`: the exact single-Gaussian
  entropy and gradient at `K = 1` (1 % and a finite-difference check at
  rtol 0.03), nearly non-overlapping mixtures at `D ∈ {1, 2}`, `K ∈ {2,
  3}` against the closed form (1 %) plus finite differences, overlapping
  mixtures by finite differences (common random numbers through
  `rng=42`), the MATLAB-pinned `H`, `dH` at 1 % (`entropy-test.mat`,
  `rng=42`, not the MATLAB stream), and the `grad_flags` output shapes.
  `test_variational_optimization_grad_fd.py::test_neg_elcbo_grad_fd_mc_entropy`
  checks `dF` through `_neg_elcbo` with `Ns = 1e4` (rtol 1e-2). Every
  test sets `sigma`, `lambd`, `w` or `eta` in 1-D or list form at some
  point (`vp.sigma = np.ones(K)`, `vp.eta = [0.6, 0.4]`), so the rewrite
  reshapes rather than indexes, as the loop does with `.ravel()`.
- **Profiling baseline.** `runs/profile_20260905_item8/` (item 8's
  campaign, plain and cProfile, code `284747e` + gpyreg `79b4986`; every
  trajectory identical to `runs/profile_20260905/`). The seam removal
  (`c4313a3`) has since shifted every seed's stream, so the campaign
  after this item will run on *different trajectories* from item 8's and
  is compared per config with `profile_compare.py` on stage seconds and
  shares, with the machine-speed control on a stage this item does not
  touch (`--control gp_train`; the default `variational_fit` is the stage
  being changed). Same seed (0), one BLAS thread, an idle machine, the
  `banana_D4` speed probe.

## Design

Private copies with fixed shapes: `mu (D, K)`, `mu_t = mu.T`, `sigma
(K,)`, `lambd (D,)`, `w (K,)`, `eta (K,)` (all through `reshape`), and the
constants `sigmalambd = sigma * lambd[:, None]` `(D, K)`, `nf = nconst /
sigma**D` `(K,)`, `wnf = w * nf`, `C = wnf / sigmalambd` `(D, K)` (the
factor that turns the standardized distance and `exp(-d2/2)` into the
mu-gradient summand `(x − mu_k) w_k norm_k / (sigma_k lambd)²`).

```
Ns = ceil(Ns / 2) * 2; half = Ns // 2
eps_half = rng.standard_normal((K, half, D))              # the loop's draws, in its order
epsilon  = concatenate([eps_half, -eps_half], axis=1)     # (K, Ns, D), antithetic as today; eps_half freed
g    = max(1, min(K, BUDGET // (Ns D K)));  step = Ns if Ns D K <= BUDGET else max(1, BUDGET // (D K))
for each block of g components jj, each block of `step` samples nn:
    X_b   = epsilon[jj, nn] * lambd * sigma[jj, None, None] + mu_t[jj, None, :]   # (g, n, D), today's expression
    delta = (X_b[..., None] - mu) / sigmalambd             # (g, n, D, K), today's quotient
    d2    = einsum("jndk,jndk->jnk", delta, delta)          # Σ_d, delta² never materialized
    E     = exp(-0.5 * d2)                                  # (g, n, K)
    q     = E @ wnf                                         # mixture density at each sample, (g, n)
    sum_logq[jj] += log(q).sum(1)
    if any of mu / sigma / lambd requested:
        lsum = einsum("jndk,dk,jnk->jnd", delta, C, E)      # Σ_k (x - mu_k) w_k norm_k / (sigma_k lambd)², (g, n, D)
        r = lsum / q[..., None]
        mu_acc[jj]    += r.sum(1)                            # (g, D) into the (K, D) accumulator
        sigma_acc[jj] += (r * epsilon[jj, nn] * lambd).sum((1, 2))
        lambd_acc     += einsum("j,jnd,jnd->d", w[jj] * sigma[jj], epsilon[jj, nn], r)
    if w requested:
        w_acc += einsum("j,jnk,jn->k", w[jj], E, 1 / q)
H          = -(w * sum_logq).sum() / Ns
mu_grad    = (w[:, None] * mu_acc).T / Ns;  sigma_grad = w * sigma_acc / Ns
lambd_grad = lambd_acc / Ns;                w_grad = -(sum_logq + nf * w_acc) / Ns
```

then the three Jacobian corrections under `jacobian_flag` exactly as today
(`sigma_grad *= sigma`, `lambd_grad *= lambd`, `w_grad = J_softmax @
w_grad` with `J_softmax` from `np.outer` on the raveled `eta`, which the
loop already uses), and `dH = concatenate([mu_grad.ravel("F"), sigma_grad,
lambd_grad, w_grad])` with the unrequested blocks empty, as today. `BUDGET`
is a module-level `_MAX_TENSOR_ELEMENTS = 2**16` (Findings, memory), read
at call time so a test can shrink it.

Term by term against the loop: `q` is the loop's `ys` (and its `q_j`, which
the loop computed a second time), `lsum` its `lsum`, `r.sum(1)` its
`(lsum / q_j).sum(0)`, `sigma_acc` its `(isum / q_j).sum()` with the
division moved inside, `lambd_acc` its `(w_j sigma_j eps lsum / q_j).sum(0)`
summed over `j` as well, `w_acc` its `(w_j norm / q_j).sum(0)` accumulated
over `j` with `nf` factored out, and `sum_logq` the `log(ys).sum()` that
`H` and `w_grad[j]` share.

Gates for the single commit: `pytest pyvbmc/testing/entropy` (the
finite-difference and MATLAB tests, the new chunk-invariance test);
`pytest pyvbmc/testing/oracles` (all green, no re-baseline expected);
`make_oracle_fixtures.py --check --exact --against
runs/oracle_outputs_prechange_3ce627d` (the dump made before the change:
expected to differ in exactly `entmc` and the `H`-carrying outputs of
`neg_elcbo`, at rounding level, and nowhere else); `pytest
pyvbmc/testing/vbmc/test_variational_optimization*.py`; the replay against
the item 8 Step 8 traces (`--baseline runs/golden/replay_item8_step8`, the
current code's trajectories: expected to part at iteration 0, since the
ELBO arithmetic moves by rounding, with the initial design identical and
the finals inside the population envelope); the full suite. Commit
`perf(entropy): vectorize entmc_vbmc over components and samples`.

### Tests added

- **Chunk invariance**: with `_MAX_TENSOR_ELEMENTS` monkeypatched to
  values that force blocks of several components with a partial last
  block (as K = 50 in production: 11, 11, 11, 11, 6), one component per
  block, and sample blocks inside a component with and without a partial
  last block, `H` and `dH` equal the default's to 1e-12 for the same seed
  (the blocks only change summation order), with and without the
  Jacobian corrections and for the `w`-only and `sigma`-only gradient
  requests.
- The existing finite-difference tests, the closed-form tests and the
  MATLAB-pinned test are unchanged and are the correctness gate; the
  `entmc` oracle is the regression gate against the loop's numerics on
  eight stored states.

### Measure

When the machine is idle (evening): `OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u
dev/scripts/profile_suite.py --suite profile --mode plain --probe
banana_D4 --out dev/scripts/runs/profile_<stamp>_item5` (≈ 30 min), then
`--mode cprof --only banana_D4,cigar_D4,lumpy_D4,student_D4` (≈ 4 min) and
the exhaust cProfile if time allows (≈ 20 min). Compare with
`runs/profile_20260905_item8/` through `profile_compare.py --control
gp_train`: wall, the variational-fit share, the `entmc_vbmc`,
`_eval_full_elcbo` and `final_boost` buckets and call counts. Because the
seam removal changed every trajectory since item 8's campaign, per-config
walls carry trajectory noise (iterations and evaluations differ); the
per-call figures from cProfile and the stage shares are the clean
comparison. Record here, in the roadmap and as a dated addendum in devlog
§2/§10.

## Steps

- [x] Profile one call at K = 14, 17, 20, 50 from the snapshots (draws /
      density loop / gradient block); the caller split from item 8's
      cProfile listings
- [x] Candidate implementation in the scratchpad; bit-check on the eight
      snapshots (against the loop and against the stored references) and
      on random states; budget sweep; layout and GEMM variants measured
- [x] Plan written (this file)
- [x] Implementation in `pyvbmc/entropy/entmc_vbmc.py`; chunk-invariance
      test; entropy tests; oracles; `--check --exact --against` the dump;
      variational tests; replay; full suite; commit `5a8e181`
- [~] Records: `dev/README.md`, roadmap (Stage 2 bullet, pickup 3b →
      done, next pickup), devlog §3/§9/§10 addenda and the §9
      correction, this file (done, committed after `5a8e181`); devlog §2
      and §Results after the campaign; push; CI smoke
- [x] Read-only Opus code review of the diff and the plan (folded in,
      tracker); `/doublecheck` after the campaign's records
- [ ] Profile campaign (idle machine) and §Results
- [ ] 20-seed population (roadmap 3a (ii); tonight if the laptop is idle,
      about 6.5 h): `golden_trace.py run --suite golden --seeds 0-19
      --workers 1 --out dev/scripts/runs/golden/item8_<date>`, then
      `summary` and `compare dev/golden/baseline <out>`

## Verification

- [x] `pytest pyvbmc/testing/entropy` green (finite differences, closed
      forms, MATLAB values, shapes, chunk invariance): 11 passed
- [x] `pytest pyvbmc/testing/oracles` green with no re-baseline (`entmc`
      and `neg_elcbo` at 1e-10 on every snapshot including `normal_D2_K1`
      and `cigar_D4_boosted`): 116 passed, 15 skipped
- [x] `--check --exact --against` the pre-change dump: exactly `entmc` and
      the `H`-carrying `neg_elcbo` outputs differ, by ≤ 1.4e-15 absolute
- [x] Replay against the item 8 Step 8 traces: initial design identical
      on every config, finals inside the envelope, 0 flagged of 5 (no
      excursion, so no second seed was needed)
- [x] `pytest --reruns=5 -x` green: 541 passed, 15 skipped, 0 reruns
- [ ] Profile: `entmc_vbmc` per call and share against item 8's campaign;
      untouched stages within trajectory noise (control `gp_train`)

## Decisions

- **Tensor of exact summands, not the GEMM expansion** (Findings,
  "Value-only shape is arithmetic-bound"): the expansion's error scales
  with the squared width ratio of the components and is visibly not
  rounding-level on adversarial states, the same reason item 2 kept the
  two-solve form for `J_sjk`. One code path for both shapes. The
  value-only path therefore gains little (1.0–1.3×); Open question 1
  records the alternative with its numbers.
- **Draw order preserved** (one `(K, half, D)` draw): free, keeps the
  `entmc` oracle and the `neg_elcbo` oracle's `H` at 1e-10 without a
  re-baseline, and keeps the estimator's samples identical to the loop's
  for a given generator state.
- **Blocks of 2^16 elements**, chosen by measurement at both shapes and
  consistent with item 3's `vp.pdf` chunking; blocks of components first,
  blocks of samples only when a single component exceeds the budget. The
  block size changes only summation order (a test asserts it).
- **`(g, n, D, K)` layout with `einsum` over `d`**: measured equal to or
  faster than the `(g, n, K, D)` layout, and it mirrors the loop's
  `(Ns, D, K)` gradient arrays, so the term-by-term correspondence in
  §Design reads directly.
- **Rounding-level differences accepted, bit-identity not pursued**
  (item 3's rule): the elementwise expressions are kept as the loop wrote
  them where it costs nothing (`Xs`, the quotient `delta`), the
  contractions and sums are not contorted to reproduce the loop's order.
- **Density computed once**: the loop's separate `ys` pass exists only
  because the gradient block came later; the vectorized function has no
  second pass.
- **The mixture sum is a matrix-vector product** (`exp(-d2/2) @ (w nf)`),
  the function's only BLAS call; the loop's `+=` over components was pure
  NumPy. The estimate therefore depends on the BLAS build and thread count
  at the level of the product's rounding (1e-16 relative), which the
  GP-free oracle class (1e-10) and the replay cannot see; noted because
  the oracle tolerance paragraph records the entropies as bit-identical
  across thread counts, which was measured on the loop.
- **Samples formed per block, draws kept whole**: the review of the
  first version found three `(K, Ns, D)` arrays alive for the whole call
  (the half draws, the antithetic pairs, the samples), 175 MB at the
  value-only corner; the draws must be made at once to preserve the
  stream's order, but the samples are an elementwise expression and are
  formed inside the block loop (bit-identical), so one such array remains.
- **`vp.mu` must be `(D, K)`**, checked explicitly: the loop's `mu[:, j]`
  failed loudly on any other shape, and a `reshape` would have accepted a
  transposed `mu` silently.

## Open questions (defaults in bold)

1. Adopt the centered GEMM expansion for the value-only path
   (`_eval_full_elcbo`, 22–36 % of the entropy's time, 2.7–7× faster
   there, about 6 % of the exhaust run's wall), with its error bound
   `eps · 2D (sigma_j / sigma_k)²` documented and a test on an adversarial
   state? **Not in this item** (PI's call): it is a second formula with a
   cancellation the Stage 2 rule has so far rejected, and the gain at
   D ≤ 10 is about 1 % of a run. A 25-line separate commit if wanted;
   `vp.pdf` would deserve the same treatment for consistency.
2. Should the `(g, Ns, D, K)` tensor be avoided at shape (a) too by
   the D-loop form (`D` passes over `(g n, K)` arrays)? **No**: measured
   equal at best; the tensor form is simpler.

## Risks

- A shape slip in the block layout of `dH` (d-fastest `mu` block, then
  `sigma`, `lambd`, `w`) passes a value check and fails the
  finite-difference tests and the oracle: both run.
- `K = 1`, `D = 1`, odd `Ns`, `Ns = 1`, 1-D `sigma` / `lambd` / `w`, list
  `eta`, NumPy-integer `K`: covered by the scratch bit-check and by the
  existing tests (`K = 1` closed form, 1-D assignments, `Ns = 1e5`).
- The replay parts at iteration 0 everywhere (the ELBO arithmetic moves
  by an ulp); a config's finals can land outside the envelope by chance;
  a second seed decides, as in items 1 and 3.
- The profile campaign compares different trajectories (the seam removal
  moved every stream after item 8's campaign): per-config walls are
  noisy, the cProfile per-call figures and the shares are the evidence.
- Profiling must run alone on an idle machine (item 8: three configs
  slowed by desktop use, rerun clean; the control stage tells).

## Results

(to be written after the profile campaign)

## Follow-ups

- **Reproducing the scratch checks.** `prof_entmc.py` (the instrumented
  copy of the loop with three stopwatches and the draw-order check),
  `bitcheck_entmc.py` (old vs new on the eight snapshots under the oracle
  criterion and against the stored `ref/entmc`, random states at D 1–20,
  K 1–60 over every flag combination, the budget sweep) and
  `entmc_variants.py` (the density layouts, the centered GEMM expansion,
  the adversarial states) were session scratch scripts, not committed;
  their numbers live in §Findings and the tracker. To rebuild: `git show
  3ce627d:pyvbmc/entropy/entmc_vbmc.py` is the loop; states come from
  `pyvbmc.testing.oracles._state.build_state(load_snapshot(fixtures /
  name))["vp"]` and from `VariationalPosterior(D, K)` with random `mu`,
  log-normal `sigma`, `lambd` and a random `eta`; compare `H`, `dH` with
  `|d| / max(|ref|, q25 |ref|)` (the oracle criterion) on the snapshots
  and `|d| / max(|ref|, 1e-8 max |ref|)` on random states (whose `K = 1`
  `mu` block is exact zero plus noise); time with `time.perf_counter`
  medians, one BLAS thread.
- **`vp.pdf` and `entmc_vbmc` compute the same mixture density** with
  the same tensor broadcast and the same 2^16 chunking; if Open question
  1 is taken up, both should move together.
- **Item 6 and 7 (memory)** are the next roadmap items after this one.

## Execution tracker

Legend: `[ ]` not started, `[~]` in progress, `[x]` done, `[!]` needs
attention. Times are wall clock on 2026-09-05.

- [x] Reading list of pickup point 3b — 14:25–14:38
- [x] Profile of one call (`prof_entmc.py`) — 14:42: draws 1–9 %, density
  loop 56–97 %, gradient block 21–39 % (shape (a)) / 0 % (shape (b));
  `(K, half, D)` draw bit-identical to `K` sequential draws
- [x] Pre-change oracle dump — 14:44: `runs/oracle_outputs_prechange_3ce627d/`
  (53 arrays per snapshot)
- [x] Candidate implementation and bit-check (`bitcheck_entmc.py`) —
  14:47–14:51: snapshots `H` ≤ 1.3e-16, `dH` ≤ 8.6e-14 under the oracle
  criterion (the stored references are reproduced by the loop bit for bit,
  so new-vs-reference equals new-vs-loop); random states at rounding level
  relative to the peak, 1512 comparisons; the `K = 1` `mu` block is
  cancellation noise on both sides. Timing (ms, loop → tensor at 2^16):
  K = 14: 2.5 → 0.33; K = 17: 3.8 → 0.45; K = 50: 23.8 → 1.9; D = 15
  K = 26: 10.0 → 1.5; K = 31: 12.7 → 1.8; D = 10 K = 25: 7.5 → 1.0; D = 20
  K = 60: 46 → 12 (shape (a)); shape (b) 20 → 22, 35 → 36, 251 → 207,
  152 → 120, 209 → 157, 114 → 86, 888 → 659 (a second, loaded run had the
  loop and the tensor within ±20 % of each other at shape (b))
- [x] Caller split from item 8's cProfile (`entmc_callers.py`) — 14:53:
  `_eval_full_elcbo` 22–25 % of the entropy at D = 4, 36 % on the exhaust
  run
- [x] Variants (`entmc_variants.py`) — 14:57: density layouts equal within
  noise, `np.square` in place slower; centered GEMM 2.7–7× at shape (b)
  but 3e-12 on the log density and 1e-4 of the gradient's peak on an
  adversarial state (tensor form 1.8e-15 and 4e-8 there); decision in
  §Decisions
- [x] Plan written — 15:01
- [x] Implementation in the package and the chunk-invariance test —
  15:02 (the test's first version fetched the function instead of the
  module for the monkeypatch, since the package re-exports the function
  under the module's name; fixed with `importlib.import_module`).
  Package against the scratch version: identical on 300 comparisons;
  against the loop: rounding level (2.6e-9 of the peak on near-zero
  entries at worst); the loop and `entlb_vbmc` return all 23 gradient
  entries at `jacobian_flag=False` on a `D = 3, K = 4` VP (the record
  correction of §Findings, checked)
- [x] Gates — 15:04–15:14: `pytest pyvbmc/testing/entropy` **11 passed**;
  `pytest pyvbmc/testing/oracles` **116 passed, 15 skipped**, no
  re-baseline; `--check --exact --against` the pre-change dump
  (`runs/oracle_exact_item5_step1.log`): exactly `entmc` (`H`, `dH`) and
  `neg_elcbo` (`F`, `dF`, `H`, `F_full`, `H_full`) differ, max |d|
  ≤ 1.4e-15 absolute, the rest bit-identical (the `normal_D2_K1` `dH`
  reads 0.83 under the scaled criterion on a 2.2e-16 absolute difference:
  the `K = 1` `mu` block); the three `test_variational_optimization*`
  modules **24 passed**; pre-commit clean; replay against the item 8
  Step 8 traces (`runs/golden/replay_item5_step1/`, 2.9 min): **0
  flagged of 5**, normal_D5 parts at iteration 1, the others at 0,
  initial design identical on all five (`X_init` in both traces), finals
  inside every fence (cigar ΔLML 0.0017 vs 0.045, gsKL 4.2e-4 vs 8.1e-3;
  halfnormal MMTV 0.0213 vs 0.0251; rosenbrock gsKL 0.064 vs 0.084);
  full suite `pytest --reruns=5 -x` **541 passed, 15 skipped, 0 reruns,
  5:40** (`runs/pytest_full_item5_step1_1788609981.log`)
- [x] **Read-only Opus review of the diff and the plan** — 15:06–15:19:
  no blocker; the arithmetic re-derived term by term and checked against a
  transcription of the loop over 672 cases (D 1–5, K 1–6, Ns 2–8, both
  `jacobian_flag` values, seven flag combinations): `H` to 2.3e-16, `dH`
  to 1.8e-15 absolute; block invariance exact (0.0) at budgets from 65536
  down to 1, partial blocks included; the draw-order identity confirmed
  at three shapes; 1-D `sigma` / `lambd` / `w`, a list `eta`, NumPy
  integer `K`, float `Ns` all handled. Should-fixes, applied: the first
  version kept three `(K, Ns, D)` arrays alive for the whole call (the
  half draws, the antithetic pairs, the samples; 175 MB at the D = 20,
  K = 106 value-only corner against 2 MB in the loop) → the samples are
  formed per block and the half draws freed, one array remains
  (bit-identical outputs); the chunk-invariance test exercised neither a
  partial component block nor a partial sample block, which K = 50
  production hits (blocks of 11, 11, 11, 11, 6) → budgets 2600 and 500
  added, plus `jacobian_flag=False` and a `sigma`-only request at each
  budget; three numbers in §Summary/§Findings contradicted the tracker
  (the Adam-shape range hid the 3.8× D = 20 corner, the value-only range
  hid two ≈ 0.9× rows at D = 4, and "never loses to the loop" was false
  for them) → corrected here, in the roadmap and in devlog §10. Notes,
  applied: the antithetic-cancellation caveat is not confined to K = 1
  (1.2e-12 of the peak at D = 5, K = 2, Ns = 2; inherent to the
  estimator, the same in the loop) → §Findings reworded; the mixture sum
  is the function's first BLAS call → §Decisions; the docstring's tensor
  bound holds only when one sample's `D × K` slab fits → reworded; a
  `reshape` would accept a transposed `mu` silently where the loop
  failed loudly → explicit shape check; the pseudocode's `(K, D)`
  comment on a `(g, D)` increment. After the edits: `pytest
  pyvbmc/testing/entropy pyvbmc/testing/oracles` **127 passed, 15
  skipped**; the package agrees with the reviewed scratch version bit for
  bit on 300 comparisons, so the replay and full-suite results above
  stand; the exact check against the dump moves the same outputs;
  pre-commit clean — 15:30
- [x] **Committed** `5a8e181 perf(entropy): vectorize entmc_vbmc over
  components and samples` (the function and the test) — 15:31; the
  records (this plan, `dev/README.md`, the roadmap, devlog §3/§9/§10, the
  einsum plan's correction) in the `docs(dev)` commit after it; pushed,
  CI smoke on the push
- [ ] Profile campaign and 20-seed population: a scheduled check at
  19:33 starts them only if the laptop is idle (no desktop use) and
  otherwise re-checks every 30 min; §Results and devlog §2 after the
  campaign
