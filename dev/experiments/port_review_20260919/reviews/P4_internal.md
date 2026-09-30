# P4 — noisy importance sampling, internal-correctness track

Raw reviewer report, wave 4 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-20. One Opus agent, fresh
context, read-only on the repositories, reading the code at `f873556`; it
did not read the MATLAB source. Its brief carried a first question on
uncertainty level 1, which the report answers before its findings. Its check
scripts ran in the session scratchpad and are kept only on the machine that
ran them (`dev/scripts/runs/LOCAL.md`). The text below is the agent's final
message, unedited.

---

# P4 — noisy importance sampling (internal-correctness track)

## 1. Coverage

**Read completely**
- `pyvbmc/vbmc/active_importance_sampling.py` (all five functions, the whole file).
- `pyvbmc/acquisition_functions/acq_fcn_viqr.py`, `acq_fcn_imiqr.py`, `abstract_acq_fcn.py` (read for the interface and for everything the slice feeds: `is_log_base` / `is_log_added` / `is_log_full`, `_estimate_observation_noise`, the consumption of `X`, `ln_weights`, `f_s2`, `K_Xa_X`, `C_tmp`).
- `pyvbmc/vbmc/active_sample.py` (all of it; in particular the noise pre-computations at `:326-363`, the call site at `:414-417`, and the post-acquisition GP update at `:710-815`).
- `pyvbmc/function_logger/function_logger.py`; `pyvbmc/vbmc/gaussian_process_train.py`; the noise/uncertainty parts of `pyvbmc/vbmc/vbmc.py` (`:1045-1108`) and `pyvbmc/vbmc/options.py: update_defaults`.
- gpyreg: `slice_sample.py` (all), `noise_functions.py` (all), and in `gaussian_process.py` the whole of `update`, `predict`, `__core_computation`, `_convert_shapes`.
- Tests: `pyvbmc/testing/vbmc/test_active_importance_sampling.py`, `pyvbmc/testing/acquisition_functions/{_scenario.py,_look_ahead.py,test_acq_fcn_imiqr.py,test_acq_fcn_viqr.py}` (the two `test_complex__call__` bodies in full).
- `papers/acerbi2020variational_appendix.md` §C.1 and §C.3 (Eq. S13, Eq. S19).
- `dev/experiments/port_review_20260919/known_differences.md`: the P4, P3, P2 (repeats, rank-one update), P1b (history omits the importance samples), GP-layer (`SliceSampler`) entries and all of "Settled non-differences". No other file under `dev/`.

**Skimmed**: `pyvbmc/testing/oracles/_oracles.py` (only `prepare_importance_sampling`), `variational_posterior.py` (`sample`, `__deepcopy__`, the `rng` property), the option `.ini` defaults.

**Not reached**: MATLAB sources (out of track); `vp.pdf`'s internals (P7); gpyreg's `fit`/hyperparameter-bound machinery beyond the noise block (P5/G); `pyvbmc/testing/vbmc/compare_MATLAB/fess.m` (deliberately not opened, to stay inside the internal track — I re-derived `fess` from its definition instead).

**Checks run** (all in the scratchpad, read-only against the repos, BLAS single-threaded; `pyvbmc.__file__` and `gpyreg.__file__` printed and confirmed to be these two checkouts):
1. Replay of the ISR resampling block (`:240-247`) on a rebuilt state → the resampling probabilities are exactly uniform (all `0.025` over 40 candidates), while the intended weights have ESS 11.3/40 and 7.2/40. (F1.)
2. Proposal-density normalization in D=2: grid integral of the density `active_sample_proposal_pdf` assigns = 1.000000 (`w_vp=1`), 0.998 (`w_vp=0.5`), 0.996 (`w_vp=0`, over the box union; the residue is grid truncation); and an importance-sampling estimate of a known unit integral from samples drawn exactly as the module draws them: 0.9984 ± 0.0144 over 5 replicates of 3000 points. The composition, the mixture weights and the normalization are right.
3. `K_Xa_X` and `C_tmp` against a direct `K(Xs,Xa) − K(Xs,X)(K+Σ_obs)^{-1}K(X,Xa)`, two hyperparameter samples: max abs difference 5.6e-16; `K_Xa_X` bit-identical to a direct kernel evaluation.
4. `renormalize_weights` totals: whole-array sum 1.0, per-row sums unequal (0.448/0.552 with 2 samples; 0.370/0.347/0.283 with 3 samples in the MCMC branch). (F2.)
5. `fess` with its documented default `X=100` → `AttributeError`. (F4.) `fess(vp, gp, X)` matches a hand computation of the fractional ESS to the last digit.
6. Function-logger/noise trace at levels 1 and 2 (repeats, pooled `S`, `S**2 * n_evals`), and `gp.noise.compute` on the level-1 model. (First question, part a.)
7. Rank-one `GP.update` vs a full rebuild of the enlarged training set, at levels 0/1/2: max |Δμ|, |Δs²|, |Δalpha| ≤ 7e-15. (Part b.)
8. `predict(add_noise=True)` without `s2_star` at levels 0/1/2. (Part c, F5.)
9. The retained `mcmc_importance_sampling` hook, forced on → `ValueError: The initial point x0 needs to be a scalar or a 1D array`. (F3.)

No test suite was run, no `optimize()`, nothing written into either repository.

---

## 2. First question — is the observation noise treated consistently at level 1?

**Short answer: yes for (a) and (b) — they are correct and mutually consistent at all three levels, with two small caveats. (c) has one real discrepancy: the only place in the slice that depends on observation noise is the MCMC target, and there `add_noise=True` cannot deliver the observation noise at levels 1 and 2 (no `s2_star` is passed), so it adds only the constant floor. It biases nothing, because the weights are formed from the same density, but it is not what the call names, and it makes the two branches of the slice sample from two different densities.**

What each level records and fits (established from the sources, not from the description):

| | level 0 | level 1 | level 2 |
|---|---|---|---|
| `gp_noise_fun` (`vbmc.py:1094-1102`) | `[1,0,0]` | `[1,2,0]` | `[1,1,0]` |
| gpyreg noise model | `sn2 = e^{2h₀}` | `sn2 = e^{2h₀} + e^{h₁}·s2` | `sn2 = e^{2h₀} + s2` |
| `noise_N` | 1 | 2 | 1 |
| logger `S` (`_record`) | absent | 1 on a first eval; precision-pooled to `1/√n` after `n` repeats | user SD; precision-pooled |
| `s2` handed to `fit`/`reupdate_gp` (`_get_training_data:762`) | `None` | `S²` = `1/n` | `S²` |
| hyperprior/x0 (`_gp_hyp:343-363`) | `noise_size` on h₀ | h₀ pinned at the floor `tol_gp_noise`, magnitude carried by h₁ | h₀ at the floor |

**(a) `active_sample.py:326-351`.** All three quantities are right.
- *Hyperparameter slice*: `hyp[cov_N : cov_N + noise_N]` is the noise block — confirmed against `__core_computation` (`hyp = [cov | noise | mean]`) and against a level-1 GP (`noise_N == 2`, slice `[log√1e-5, log 3]`).
- *`s2 = S**2 * n_evals`*: at level 1 the logger's pooling makes `S² = 1/n` exactly, so `S²·n ≡ 1` for **every** training point whatever its repeat count (measured: 1.000000 after 1, 2, 3, 4 evaluations). Since level 1's user-provided variance is a placeholder of 1 scaled by `e^{h₁}`, the resulting `sn2_new = e^{2h₀} + e^{h₁}` is exactly the variance of *one* observation — which is what the look-ahead formula `s²_{Ξ∪θ*} = s² − C²/(s²(θ*) + σ²_obs(θ*))` needs, since the acquisition contemplates one new evaluation at θ*. Measured on a level-1 GP with rows evaluated 1/2/4 times: the GP's own training-row noise is `[3.00001, 1.50001, 0.75001]` (correctly reduced by the repeats) while `sn2_new` is `[3.00001, 3.00001, 3.00001]` — the single-observation value at every row. At level 0, `hasattr(function_logger, "S")` is false, `s2=None`, and `noise.compute` returns the scalar `e^{2h₀}`, the whole model. At level 2 the same expression gives the user's variance for a point evaluated once.
- *Average over hyperparameter samples*: `sn2new.mean(1)` is an arithmetic mean of variances; the paper (§C.1, Eq. S13) prescribes only the nearest-neighbour rule and the *geometric* mean of the input length scales, which `:358` (`np.exp(ln_ell.mean(1))`) implements correctly.
- Caveats (both minor, listed as F7 and F8): `sn2_new` omits the posterior's `sn2_mult`, which `predict(add_noise=True)` and `GP.update` both apply; and at level 2 with *unequal* SDs at a repeated input, `S²·n` is the harmonic mean of the per-observation variances (measured 2.286 for SDs 1, 2, 4, against an arithmetic mean of 7.0) rather than any single observation's variance.

**(b) `active_sample.py:710-815`.** Right at each level, and the update is equivalent to a rebuild.
- `s2new = function_logger.S[idx_new] ** 2` is taken *after* the evaluation. For a fresh input at level 1 that is `1`, exactly what `_get_training_data` would hand `fit` for that row; at level 2 it is the user's variance for that observation; at level 0 `S` does not exist and `s2new` is `None`, matching `gp.s2 is None`. A repeat never reaches `gp.update`: `update1` requires `n_evals[idx_new] == 1`, so a pooled observation goes through `reupdate_gp`, which rebuilds `X`, `y`, `s2` from the logger. (This is the P2 entry "The rank-one GP update is taken for a fresh observation"; the entry's description matches the code.)
- gpyreg's rank-one path calls the noise function with `X_new, y_new, s2_new` exactly as `__core_computation` calls it with `X, y, s2`, multiplies by the stored `sn2_mult`, and extends the factor in the stored `sl` parametrization. I re-derived the extension: with `Lᵀ L = (K + sn2_mult·diag(sn2))/sl`, the new column must be `L^{-T}k*/sl` and the new diagonal `√(sl·k** + sl·sn2_eff − uᵀu)/sl`, which is what `:880-918` computes; `sW = 1/√sl` for every row makes the predictive variance independent of `sl`. Measured against a full rebuild with the same hyperparameters at all three levels: max |Δμ| ≤ 4.4e-16, max |Δs²| ≤ 4.4e-16, max |Δalpha| ≤ 6.8e-15, and `gp.s2` identical.

**(c) Inside the slice.** Two of the three uses of the GP are noise-free and correct: `active_sample_proposal_pdf` and the MCMC branch both take `gp.predict(..., separate_samples=True)` without noise, so `f_s2` — the quantity the acquisitions use as `s²_Ξ(θ_a)` in the interquantile range — is the *latent* posterior variance, which is what Eq. S19 requires (the IQR is of the latent log-joint `f`, not of a future noisy observation). The third use is the MCMC target: `log_p_fun = lambda x: acq_fcn.is_log_full(x, vp=vp, gp=gp1)` reaches the fallback branch of `is_log_full`, which calls `gp.predict(np.atleast_2d(x), add_noise=True)` **without `s2_star`**. Measured: the variance that `add_noise=True` then adds is `e^{2h₀}` alone — at level 1, 1e-5 where the model's single-observation variance is 3.0; at level 2, 1e-5 where it is 0.4; at level 0 it is the full model noise. So at levels 1 and 2 the MCMC targets essentially the latent density (plus a 1e-5 floor), while the call reads as though it targeted the noisy predictive density. This does not bias the estimator: the weights are `ln_y − log_p` with `log_p` the sampler's own evaluations of that very density (`results["f_vals"]`, which I verified correspond to the recorded samples), so target-over-proposal is exact whatever the proposal is. What it does mean is (i) the intent of `add_noise=True` is not met at the two noisy levels, and (ii) the ISR pool and the MCMC chain are drawn from two different densities — the resampling step at `:237-247` scores the ISR candidates with the *latent* `is_log_added`, and the chain then targets the noise-inflated one. Reported as F5.

One adjacent observation, outside this slice and low confidence: at level 1 the entire noise magnitude is carried by `noise_provided_log_multiplier`, whose *bounds* are gpyreg's hard-coded `[log 1e-3, log 1e3]` (`noise_functions.py:139-140`) and are not widened by `_gp_hyp` (which sets only the prior, `:426-429`). A level-1 target whose true noise SD lies outside `[0.032, 31.6]` therefore cannot be fitted. Belongs to the GP-training slice.

---

## 3. Findings

### F1. The ISR resampling weights are reduced along the wrong axis, so the MCMC chain starts at a uniformly drawn candidate
- Location: `pyvbmc/vbmc/active_importance_sampling.py:243` (and its consequences at `:244-245`, `:250`); MATLAB: "not read (internal track)"
- Category: indexing/shape
- Proposed classification: suspected defect
- Confidence: high
- What the code does: `ln_weights` at `:240-242` has shape `(Ntot, 1)` — one column, because `gp1` carries a single posterior. `ln_weights_max = np.amax(ln_weights, axis=1).reshape(-1, 1)` therefore reduces over the *length-one* axis and returns each element itself, so `np.exp(ln_weights - ln_weights_max)` is a vector of ones and `weights` is uniform. The max must be taken over the candidates (globally, or `axis=0`), as the identical log-sum-exp idioms elsewhere in the file do (`:396` uses `axis=1` over the *mixture* axis, correctly; `:498` uses a global `amax`). Measured on a rebuilt state: the probabilities passed to `rng.choice` are all `0.025` across 40 candidates, while the intended weights give ESS 11.3/40 and 7.2/40 — i.e. a handful of candidates should carry most of the mass.
  The same line disables the guard below it: with a per-element max, `np.any(ln_weights_max == -np.inf)` raises `ValueError("Invalid value.")` if *any single* candidate has zero weight, where the intended test ("is the best candidate still impossible?") would raise only if all of them did. So the guard is simultaneously inert for its purpose and able to abort a run spuriously.
- Consequence if real: step 1 of the docstring's three-step scheme ("use importance sampling-resampling (ISR) to sample from the base IS density") does not happen; the slice sampler is started from a uniform draw out of the 200 mixture candidates, most of which are box-uniform points far in the tails, instead of from a point drawn by importance weight. With the shipped defaults the chain is 100 draws after a 50-draw burn-in, so a bad start is not washed out — the test that does exercise this path uses 8000 draws with 4000 burn-in and notes that even then the self-normalized estimate scatters by 2% across seeds. The affected acquisition is `AcqFcnIMIQR` (and any custom acquisition with `importance_sampling` but not `variational_importance_sampling`); the noisy default, `AcqFcnVIQR`, takes the `only_vp_flag` path and is untouched.
- Suggested reproduction: ran it — replay `:240-247` on any state with `Ns_gp ≥ 1` and print `weights`; they are `1/Ntot` exactly. A one-line assertion `assert not np.allclose(weights, weights[0])` on a state whose candidate weights are known to differ settles it.
- Test adequacy: no. `test_active_importance_sampling` asserts only output shapes; `test_draws_come_from_vp_rng_only` asserts determinism and that the global stream is untouched — both pass with any resampling distribution. `test_acq_fcn_imiqr.test_complex__call__` is a genuine statistical check, but at 8000 samples with 4000 burn-in it is insensitive to the starting point (and its 10% tolerance was sized for the chain's own scatter).

### F2. The importance weights are normalized over the hyperparameter samples jointly, which is unjustified in the MCMC branch
- Location: `pyvbmc/vbmc/active_importance_sampling.py:329` and `:497-499` (`renormalize_weights`); consumed at `acq_fcn_imiqr.py:152-168` and `acq_fcn_viqr.py:364-369`; MATLAB: "not read (internal track)"
- Category: formula
- Proposed classification: possibly intentional
- Confidence: medium (on the derivation), low (on what was intended)
- What the code does: `renormalize_weights` subtracts one global log-sum-exp from the whole `(Ns_gp, Na)` array, so the weights sum to 1 across hyperparameter samples *and* importance points together. In the non-MCMC branch that is defensible and probably right: every row scores the *same* points drawn from the *same*, normalized mixture proposal, so row `s`'s total is an estimate of `∫exp(f̄_s)` and the joint normalization weights each hyperparameter sample by its estimated evidence. In the MCMC branch the picture changes: each row has its own sample set drawn from its own *unnormalized* density `q_s ∝ exp(full_s)`, so row `s`'s total estimates `N·∫exp(f̄_s)/Z_s` with `Z_s = ∫exp(full_s)` unknown and different for every `s`. The final `logsumexp(acq)/Ns_gp` at `acq_fcn_imiqr.py:170-175` then mixes the per-sample integrals with weights containing those arbitrary `Z_s`. Self-normalizing within each row (dividing each row by its own total) would remove `Z_s` and make the rows comparable.
- Consequence if real: the IMIQR value is a non-uniformly weighted average over GP hyperparameter samples instead of the uniform one the `/Ns_gp` implies. Measured on a small D=2 state with three hyperparameter samples and 100 MCMC draws each: row totals 0.370 / 0.347 / 0.283, a 1.31 max/min ratio. A modest distortion, not an order of magnitude, and it disappears when `Ns_gp == 1` (late in a run, and in every test).
- Suggested reproduction: ran it — print `np.exp(active_is["ln_weights"]).sum(axis=1)` after a call with `active_importance_sampling_mcmc_samples > 0` and `Ns_gp > 1`. To settle whether it matters, compare the IMIQR value against a Gauss-Hermite reference with `Ns_gp = 3` under both normalizations. Note the VIQR docstring (`acq_fcn_viqr.py:106-110`) documents the joint normalization as deliberate, but does so for the simple-Monte-Carlo path, where it is a constant.
- Test adequacy: no. Both `test_complex__call__` scenarios build a GP with one hyperparameter sample, and `test_acq_fcn_imiqr.py:184-185` even asserts `np.sum(np.exp(ln_weights[0])) == 1`, which holds only because there is a single row.

### F3. The retained `mcmc_importance_sampling` hook cannot run, and its weights would be wrong if it could
- Location: `pyvbmc/vbmc/active_importance_sampling.py:86-126` (`:98` `burn_in = 0`, `:106-114` the sampler construction, `:119` `Xa[-Na:]`, `:122-126` the weights); MATLAB: "not read (internal track)"
- Category: control flow
- Proposed classification: suspected defect (in a dormant path)
- Confidence: high (that it raises), medium (that the weights are wrong)
- What the code does: three problems in the branch the known-differences sheet records as "a retained but dormant hook". (i) `x0` is `Xa`, shape `(Na, D)`; `SliceSampler.__init__` requires a 1-D `x0` and raises immediately (`ValueError: The initial point x0 needs to be a scalar or a 1D array`) — reproduced by forcing the branch with `acq_info["mcmc_importance_sampling"] = True` and `fess_thresh = 1`. The sibling branch at `:251` correctly passes a single row. (ii) `burn_in = 0` and `thin = 1`, and the thinning is then done by hand as `Xa[-Na:, :]`, which keeps the last `Na` *consecutive* draws of one chain rather than every `thin`-th — an ensemble-sampler idiom (one iteration across `Na` walkers) applied to a single coordinate-wise chain. (iii) After the MCMC the weights are set to `ln_y` alone (`:125`), with no `− log_p` correction, so samples drawn from the full IS density would be given the weights appropriate to samples drawn from the variational posterior; for VIQR, `ln_y` is identically zero, i.e. uniform weights over non-uniformly drawn points.
- Consequence if real: a user-supplied acquisition that sets the documented hook crashes at once; if (i) were repaired, (ii) and (iii) would silently bias the estimate. The shipped acquisitions never set the flag, so production runs are unaffected. Note that the sheet's P4 entry justifies retention by "a user-supplied acquisition object can still request it" — that justification does not hold as written, because the path raises on the first call.
- Suggested reproduction: ran it — set `acq.acq_info["mcmc_importance_sampling"] = True` on an `AcqFcnVIQR` and `active_importance_sampling_fess_thresh = 1.0`, then call `active_importance_sampling`.
- Test adequacy: no test reaches this branch (`fess_thresh` is never raised above the default in any test, and no test sets the flag).

### F4. `fess`'s documented default for `X` raises
- Location: `pyvbmc/vbmc/active_importance_sampling.py:470-472`; MATLAB: "not read (internal track)"
- Category: control flow
- Proposed classification: suspected defect
- Confidence: high
- What the code does: the signature is `fess(vp, gp, X=100)` and the docstring says an integer `X` is "an integer number of samples to draw from the VP". The branch does `X = vp.sample(N, orig_flag=False)`, but `VariationalPosterior.sample` returns the pair `(X, I)`, so `X` becomes a tuple and the next use of it (`gp.predict(X)` or `X.shape[0]`) raises `AttributeError: 'tuple' object has no attribute 'ndim'`. The line needs `X, _ = vp.sample(...)`.
- Consequence if real: the documented default of a public-ish helper is unusable. Nothing in the package calls `fess` without an explicit `X` (the one live call site, `:88`, passes `Xa`), so no run is affected.
- Suggested reproduction: ran it — `fess(vp, gp)` on any state raises `AttributeError`.
- Test adequacy: no. `test_fess` passes `X` explicitly in both of its calls and compares against the MATLAB fixture; the scalar branch is never entered.

### F5. `is_log_full`'s `add_noise=True` cannot add the observation noise at uncertainty levels 1 and 2, and disagrees with the density used everywhere else in the slice
- Location: `pyvbmc/acquisition_functions/acq_fcn_imiqr.py:264` and `acq_fcn_viqr.py:526`, reached from `pyvbmc/vbmc/active_importance_sampling.py:226` (and `:91`); MATLAB: "not read (internal track)"
- Category: cross-module
- Proposed classification: unsure
- Confidence: medium
- What the code does: the MCMC target calls `gp.predict(x, add_noise=True)` with no `s2_star`. `GaussianNoise.compute` then treats the user-provided variance as 0, so the added variance is the constant term `e^{2h₀}` alone. Measured: at level 1 it adds 1e-5 where the model's single-observation variance is 3.0; at level 2, 1e-5 where it is 0.4; at level 0 it adds the model's whole noise. So the flag's effect is the reverse of what the levels suggest — it is a no-op exactly where observation noise exists. Meanwhile every other density in the slice (`active_sample_proposal_pdf:367`, the resampling at `:237`, `ln_y` at `:274`, `f_s2` at `:276`) uses the latent `f_s2`. Eq. S19 of `papers/acerbi2020variational_appendix.md` defines the acquisition through the latent `s_{Ξ∪θ*}`, so the latent variance is the right quantity for the proposal's `sinh` factor; adding observation noise there is at best a deliberate widening of the proposal. Because the weights are `ln_y − log_p` with `log_p` read back from the sampler, the estimator is unbiased under either choice; only the proposal's shape and the comparability of the two branches change.
- Consequence if real: small at levels 1 and 2 (a 1e-5 floor). At level 0 — a noiseless target with a hand-picked IMIQR — the proposal is materially wider than the ISR pool's scoring density, so the ISR-selected start and the chain's stationary distribution disagree.
- Suggested reproduction: ran it — `gp.predict(Xt, add_noise=True)[1] - gp.predict(Xt)[1]` on a level-1 GP returns `e^{2h₀}`, not `e^{2h₀} + e^{h₁}`. Whether `add_noise` belongs there at all is a question for the MATLAB comparison: `test_acq_log_f` matches MATLAB to `atol=1e-3` on a GP whose `sn2` is 1, which suggests MATLAB includes the noise too — in which case the finding reduces to "the port is faithful but the semantics differ between levels".
- Test adequacy: partly. `test_acq_log_f` pins `is_log_full` against MATLAB, but only on a level-0-shaped GP (constant noise only), so it cannot see that the term vanishes at levels 1 and 2.

### F6. The known-differences sheet's P3 entry contradicts the shipped VIQR
- Location: `dev/experiments/port_review_20260919/known_differences.md:1084-1098` vs `pyvbmc/acquisition_functions/acq_fcn_viqr.py:120`; MATLAB: "not read (internal track)"
- Category: defaults
- Proposed classification: suspected defect (in the document)
- Confidence: high
- What the code does: the entry lists `iqr_reduction` among "the `loss` variants ... retained on the branch `retain/experimental-acquisitions`" and concludes "`AcqFcnVIQR` offers only the standard `loss="iqr"`". At HEAD, `LOSSES = ("iqr", "iqr_reduction")` and the whole `_log_iqr_reduction` implementation is live. The commit the entry cites, `fd9c7e8e`, says so itself: "`AcqFcnVIQR` offers the losses `iqr` and `iqr_reduction`; `var_reduction` and `sd_reduction` are gone". Only two of the four named variants were removed.
- Consequence if real: a reviewer or a user reading the sheet will believe a shipped, documented and tested code path does not exist. The path is also the one that *does* consume `active_is["ln_weights"]` in VIQR (the `iqr` loss ignores them), so it is load-bearing for slice P4's outputs.
- Suggested reproduction: `grep -n "LOSSES = " pyvbmc/acquisition_functions/acq_fcn_viqr.py`; `git show fd9c7e8e -- pyvbmc/acquisition_functions/acq_fcn_viqr.py`. Both run.
- Test adequacy: `pyvbmc/testing/acquisition_functions/test_acq_fcn_viqr_losses.py` exists and covers `iqr_reduction`, which is itself evidence against the entry.

### F7. `sn2_mult` is applied by `GP.predict`/`GP.update` but not by the acquisitions' `y_s2`
- Location: `pyvbmc/vbmc/active_sample.py:345-351`, consumed at `acq_fcn_viqr.py:300-301` and `acq_fcn_imiqr.py:81-83`; against `gaussian_process.py:2032` and `:858`; MATLAB: "not read (internal track)"
- Category: cross-module
- Proposed classification: unsure
- Confidence: low
- What the code does: when `__core_computation` has to retry the Cholesky factorization it inflates the noise by `sn2_mult` (×10 per retry) and stores it on the posterior; `predict(add_noise=True)` and the rank-one update both multiply the noise by it. `sn2_new`, and hence the `y_s2 = f_s2 + sn2` of the look-ahead denominator, does not. The GP the acquisition reasons about would then have a different effective observation noise from the GP that will actually absorb the new point.
- Consequence if real: none in ordinary runs (`sn2_mult == 1`); on a borderline-singular kernel the look-ahead variance reduction `C²/(s²+σ²)` is overstated by up to the retry factor, making the acquisition over-optimistic about what a new evaluation buys. The AGENTS note that "the Cholesky retry ladder exists because the matrices are borderline singular" suggests this is not unreachable.
- Suggested reproduction: not run. Force a retry (a near-duplicate training set with a tiny constant noise), then compare `gp.temporary_data["sn2_new"]` with `predict(add_noise=True) - predict()`.
- Test adequacy: no test constructs a GP whose factorization retries.

### F8. At level 2, repeated observations with unequal SDs give `sn2_new` the harmonic mean of their variances
- Location: `pyvbmc/vbmc/active_sample.py:336-341`; MATLAB: "not read (internal track)"
- Category: formula
- Proposed classification: possibly intentional
- Confidence: low
- What the code does: `S**2 * n_evals` inverts the precision pooling of `FunctionLogger._record`. With `n` observations of variances `v₁…vₙ` at one input, the pooled `S² = 1/Σ(1/vᵢ)`, so `S²·n = n/Σ(1/vᵢ)` — the harmonic mean. Measured for SDs (1, 2, 4): 2.286, against an arithmetic mean of 7.0 and a largest observation variance of 16. With equal SDs the expression is exact, which is the common case (a target whose noise estimate barely varies at a fixed input).
- Consequence if real: where the user's SD estimates at one input vary a lot, the acquisition assumes a new observation there would be more precise than the typical past one, so the look-ahead variance reduction at that neighbourhood is overstated. Only bites with `max_repeated_observations > 0` (default 0) or duplicate `x0`/precomputed rows.
- Suggested reproduction: ran the logger part — `FunctionLogger.add` three times at one point with SDs 1, 2, 4 and print `S**2 * n_evals`.
- Test adequacy: no. No test feeds unequal SDs at one input and then inspects `sn2_new`.

**Checked and found correct** (recorded so the next reviewer need not redo them): the composition of the proposal and its density, including the `w_vp`/`(1−w_vp)/VV/N` mixture weights, the uniform choice of training point, the box half-widths, and the normalization (three grid integrals ≈ 1 and an IS estimate of a unit integral at 0.998 ± 0.014); the deterministic-mixture use of one density for both sub-samples; the four-group smoothing of `vp_is` (`sigma² + scale²` taken from the *original* `vp.sigma`, weights renormalized to 1) and the fact that the smoothed copy shares `vp.rng` through `VariationalPosterior.__deepcopy__`; the MCMC weights `ln_y − log_p` being exactly base-over-proposal, with `f_vals` verified to be the log density at the recorded samples; the `-inf` cleanup at `:201-203` and `-inf` surviving `renormalize_weights` as weight 0 (only an all-`-inf` or NaN-containing array degenerates to NaN, which the cleanup prevents in the branch that can produce them); `fess` against the definition `(1/Σw²)/N` to the last digit; `K_Xa_X` and `C_tmp` against a direct computation for both the Cholesky and the low-noise parametrizations and for several hyperparameter samples (5.6e-16); the `tau2 = C²/(f_s2 + sn2)` look-ahead formula against Eq. S13/S19; the axis conventions of every stored array against both acquisitions (`ln_weights[s,:]`, `f_s2[:,s]`, `X[s]`, `K_Xa_X[s]`, `C_tmp[s]` — all consistent, and the ordering of the VP-then-box concatenation is the same in all three arrays); every rounded/divided count (`ceil` for `Na` and `burn_in`, true division for `w_vp`, `rng.integers` half-open); and that every draw comes from `vp.rng` in a fixed order, the slice sampler's included, with the caller's `widths` array copied by the sampler so that consecutive hyperparameter samples are not affected by each other's width adaptation.

---

## 4. Test adequacy notes

- `test_active_importance_sampling` asserts only the *shapes* of `X`, `ln_weights` and `f_s2`. It exercises both branches of the module but would pass with any values in them — it is the clearest case in this slice of a test that mirrors the implementation's structure rather than its specification. It is what stands between the repository and F1, F2 and F5, and it catches none of them.
- `test_draws_come_from_vp_rng_only` is a good test of what it tests (determinism given the generator, and that the global stream is untouched), but it pins reproducibility, not correctness: every finding above survives it unchanged.
- `test_active_sample_proposal_pdf` and `test_fess` compare against stored MATLAB arrays and are genuine specification tests for those two functions — they are why I have no finding against the proposal density or the ESS formula. Their blind spots are the argument shapes they never use: `test_fess` never passes a scalar `X` (F4), and `test_active_sample_proposal_pdf` passes the *unsmoothed* `vp` as `vp_is`, so the four-group smoothing at `:139-149` has no reference of any kind.
- `test_acq_fcn_imiqr.test_complex__call__` is the strongest test touching this slice: it recomputes the acquisition from the module's own samples and weights (exact), then compares against a Gauss-Hermite reference (rtol 0.1). But its first check is a consistency check between the acquisition and the stored state — it would pass with wrong samples as long as the weights match them — and the second is run with one hyperparameter sample and an 8000-draw chain, which is precisely the configuration in which F1 and F2 are invisible. Its docstring's own history (a reference that was 3–4% high and a mislaid hyperparameter array, which together "left a 5% tolerance to a coin toss") is a useful warning about how much a loose statistical tolerance can hide.
- The `pyvbmc/testing/oracles/` fixtures re-run `active_importance_sampling` and compare the acquisition outputs bit-for-bit. They would catch any *change* to the slice, including a repair of F1 — which is worth saying plainly: fixing F1 will move the `acq_AcqFcnIMIQR` and `active_sample_step` oracles, and that movement is the fix working, not a regression.
