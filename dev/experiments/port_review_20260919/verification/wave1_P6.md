# Wave 1 verification: slice P6 (variational optimization, the ELBO, `update_K`)

Verification of the findings of `../reviews/P6_internal.md` and
`../reviews/P6_comparison.md` by a fresh Opus agent, 2026-09-19, on
`dev-port-review` at `e024b1b` and MATLAB VBMC at `396d649`. The agent wrote
its own checks without opening the reviewers' scripts, dated each finding
from both histories, and searched `dev/`, the porting log and the commit
messages for recorded decisions. Its check scripts ran in the session
scratchpad and are not retained. The text below is its final message,
unedited.

---

## Ledger

| ids | statement | verdict (class) | fires at defaults? | how verified | record |
|---|---|---|---|---|---|
| int F1 + cmp F1 | `_vb_init` type 2 computes `mean(V)·mean(1/λ²)` instead of `mean(V/λ²)`: `V` is `(D,)`, `lambd0` is `(D,1)`, so `V / lambd0**2` broadcasts to `(D,D)` | **confirmed port discrepancy** | yes, whenever `D > 1` and `slow_opts_N = 2` | recomputed both expressions on all 8 oracle states (`chk_f1.py`); MATLAB `misc/vbinit_vbmc.m:32` read directly | none found |
| int F2 + cmp F6 | the returned posterior's `eta` is the endpoint's while `w` comes from the winning midpoint theta, so `softmax(eta) ≠ w`; MATLAB's `rescale_params` deletes the field | **confirmed port discrepancy** (latent; no numerical consequence today) | yes | 6 `optimize_vp` runs on `normal_D2_warmup`, 6/6 mismatched (`chk_eta.py`) | none found |
| int F13b + cmp F2 | `_vb_init` sets `new_vp.bounds = None`, `optimize_vp` returns a candidate, so the soft bounds never accumulate across iterations; MATLAB's `vpbounds.m` accumulates and the struct rides back through the candidate | **confirmed port discrepancy** | yes (every call); observable only when the training range shrinks | `chk_eta.py` prints `out.bounds is None` against the incoming posterior's six accumulated keys; MATLAB chain re-traced in source | none found (the test at `test_variational_optimization.py:603` pins the Python behaviour, added 2026-09-05 in `d144a83`, a perf commit) |
| int F5 + cmp F8 | `_sieve(init_N=0)` returns a bare `VariationalPosterior` and a Python `int`, which `optimize_vp:168` cannot consume | **confirmed port discrepancy** (unreachable in PyVBMC) | no | called `_sieve(..., init_N=0)`; got `TypeError: 'VariationalPosterior' object is not subscriptable` and a `ValueError` from `np.where` on a 0-d scalar (`chk_misc.py`) | none found |
| int F10 + cmp F9 | `minimize_adam` binds `x = x0` and the first update is an in-place `-=`, so the caller's array carries one Adam step; MATLAB `x = x0(:)` copies | **confirmed port discrepancy** (latent) | yes, but harmless at the only call site | `x0 = [1, -2]` → `[0.90049377, -1.90049376]` after `max_iter=5` (`chk_misc.py`) | `dev/2026-09-02-modernization-discussion.md:240-242` records it ("Autodiff pitfalls identified", item 1) — noted, not dispositioned |
| int F10 (2nd half) | the trailing-average window `x_tab[:, i-19:i+1]` wraps for short runs | **confirmed Python-only defect** (latent) | no | `chk_win.py` maps the regime exactly | none found |
| int F7 + cmp F4 | the deterministic-entropy branch raises `RuntimeError` on `not res.success`, has no evaluation cap, and reinterprets `DetEntTolOpt` as a gradient-norm tolerance; MATLAB caps at `50*(D+2)` and falls back to CMA-ES only on an exception | **confirmed port discrepancy** | only when `ns_ent_K == 0`, i.e. `K == 1` (reachable after pruning) or `entropy_switch` (non-default) | `chk_det.py`/`chk_det2.py`: scipy maps `tol`→`gtol` for BFGS, default `maxiter = 200·len(x0)`, `success=False` on exhaustion with a usable `res.x` | none found (the sheet has a "substituted library" entry for CMA-ES in P2 but not for this branch) |
| int F4 | `y_tab[i] == f(x_tab[:, i-1])` — value stored before the update, point after | **not a defect** (shared with MATLAB, deliberate-looking in both) | yes, every stochastic optimization | `max|y_tab[1:] − f(x_tab[:, :-1])| = 0.0`; `utils/fminadam.m:47,68` has the same order | the comparison report is right, the internal report's "suspected defect" is wrong |
| int F3 | `_eval_full_elcbo` ignores `K == 1` and `entropy_switch`, using the MC entropy where the exact one exists | **paper-versus-code difference shared with MATLAB** (not a port matter) | yes when `K == 1` | `misc/vpoptimize_vbmc.m:279,288` does exactly the same; `chk_ent.py` quantifies the noise | none needed |
| int F6 | `var_ss = varG_ss + np.std(varG, ddof=1)` adds an SD of variances to a variance of means | **not a defect** (exact port; `misc/gplogjoint.m:404` reads `varss = varFss + std(varF)`) | yes when `Ns > 1` | source comparison; MATLAB line unchanged since the first commit `a1680e3` (2018-05-04) | the formula has no paper origin on either side; `dev/plans/latent-bug-fixes.md` touches only its dtype/docstring |
| int F8 | `eta` excluded from `_vp_bound_loss` while `get_bounds` still ships `eta_lb`/`eta_ub` | **intentional difference** | yes | source read | `dev/experiments/port_review_20260919/known_differences.md:736-761`; PI decision 2026-09-08 treatment "A", `dev/plans/latent-bug-fixes.md` Phase 6 and Q1, `dev/results/2026-09-08-eta-bound-fix.md` |
| int F9 | `elcbo_beta` hard-coded to 0 | **intentional difference** (unported feature, inactive at MATLAB defaults) | yes; matches MATLAB's `defopts.ELCBOWeight = 0` | source read | `known_differences.md:797-809` |
| int F11 | Adam `β₂ = 0.999` and `α_max = 0.05/0.005` against the paper's `0.99` and `0.1/0.01` | **paper-versus-code difference shared with MATLAB** | yes | `utils/fminadam.m:23` (`beta2 = 0.999`), `misc/vpoptimize_vbmc.m:110-118`; `papers/acerbi2018variational_appendix.md:274` | none needed |
| int F12 | `update_K`'s `-inf` mask really excludes the two oldest entries of the recent window; `recent_iters = 6` vs the paper's 4 | **paper-versus-code difference shared with MATLAB** | yes | `private/updateK.m:16-26` is line-for-line the same, comment included; all option defaults compared | none needed |
| int F13a | `"skip_elbo_variance" in options` is always False | **not a defect** (faithful port of an equally dead MATLAB guard) | no (dead on both sides) | `misc/vpoptimize_vbmc.m:281`; `SkipELBOVariance` absent from `vbmc.m` defopts; absent from both PyVBMC `.ini` files | none found |
| cmp F3 | the sieve candidate count is `50·vp.K`; MATLAB's main loop uses the stale `K = options.Kwarmup = 2` | **confirmed port discrepancy** (PyVBMC's value is the more sensible one) | yes, once `vp.K > 2` (i.e. after warm-up) | `grep` confirms `vbmc.m:459` is the only bare-`K` assignment; `vbmc.m:584` and `misc/finalboost_vbmc.m:32` use `Knew` | none found |
| cmp F5 | `vp.delta` / `options.Bandwidth` (the GP-smoothing widening of `tau`/`nu`) is unported | **intentional difference in effect / unported feature**, inactive at MATLAB defaults | no (`defopts.Bandwidth = 0`) | `vbmc.m:313`, `misc/setupvars_vbmc.m:247`; no `bandwidth` option anywhere in `pyvbmc/` | none found — the sheet lacks this entry (the reviewer is right to flag it) |
| cmp F7 | `options.eval("adaptive_k", {"unkn": K_new})` makes a user-supplied callable uncallable; `round` semantics also differ | **confirmed port discrepancy** | no (the default is the constant `2`) | `Options.eval` calls `f(**evaluation_parameters)` (`options.py:248-251`); MATLAB `evaloption_vbmc` calls positionally | none found |
| cmp F10a | `np.argsort` is unstable where MATLAB's `sort` is stable | **confirmed port discrepancy** (measure-zero) | ties only | source read | none found |
| cmp F10b | `np.argmin` selects a `NaN`; MATLAB's `min` skips it | **confirmed port discrepancy** (needs a NaN evaluation) | no | `np.argmin([nan,1,0.5]) = 0` vs `nanargmin = 2` | none found |
| cmp sheet note 1 | the sheet's `J_sjk` entry mischaracterizes MATLAB | **reviewer correct** | n/a | `misc/vpoptimize_vbmc.m:236-237` reads `I_sk(:,idx) = []; J_sjk(:,:,idx) = [];` | `known_differences.md:931-940` |
| cmp sheet note 2 | the `VarParamsBack` entry omits `variational_sampler` and `active_variational_samples` | **reviewer correct** | n/a | grep: both appear only in `advanced_vbmc_options.ini:143,273` | `known_differences.md:821-832` |
| cmp sheet note 3 | `active_sample.py:716` discards `np.append`'s result | **reviewer correct** (confirmed Python-only defect, dormant) | n/a | source read | none found |

---

## (a) `_vb_init` type 2 — the exact expressions

**MATLAB** (`misc/vbinit_vbmc.m:31-32`), `V` and `lambda0` both `(D,1)`:

```matlab
if K > 1; V = var(mu0,[],2); else; V = var(Xstar)'; end
sigma0 = sqrt(mean(V./lambda0.^2)/Knew).*exp(0.2*randn(1,Knew));
```
so the base scale is `sqrt( mean_d(V_d / λ_d²) / K_new )`.

**Python** (`variational_optimization.py:904-910`), `V` is `(D,)` (`np.var(..., axis=1)` / `axis=0`), `lambd0 = vp.lambd.copy()` is `(D,1)` (`:889`):

```python
sigma0 = np.sqrt(np.mean(V / lambd0**2) / K_new) * np.exp(
    0.2 * rng.standard_normal((1, K_new))
)
```
`V / lambd0**2` is `(D,D)` with entry `[i,j] = V_j / λ_i²`, so `np.mean` gives
`mean_d(V_d) · mean_d(1/λ_d²)`.

**Corrected one-liner** (nothing else on the three lines changes):

```python
sigma0 = np.sqrt(np.mean(V / lambd0.ravel() ** 2) / K_new) * np.exp(
```

It draws exactly the same number and shape of randoms, so the RNG stream is untouched;
only the candidate widths move, and only for `D > 1`.

**Sign and size.** The error is an exact covariance identity:
`coded − correct = mean(V)·mean(1/λ²) − mean(V/λ²) = −Cov_d(V_d, 1/λ_d²)`.
So the coded value is *larger* whenever `V` and `1/λ²` are negatively associated across
dimensions — the normal case, because `λ_d` tracks the spread in dimension `d`. **The
comparison report's account holds; the internal report's 0.785× does not describe realistic
states.** My independent measurement on the eight oracle fixtures (`chk_f1.py`, ratio of the
coded base scale to MATLAB's, at `K_new = K`):

```
cigar_D4_boosted 2.146 | cigar_D4_largeK 3.020 | corr_D5_warped 1.043
halfnormal_D2_bounded 1.012 | normal_D2_K1 1.062 | normal_D2_singlesample 1.510
normal_D2_warmup 1.473 | rosenbrock_D2_noise1_viqr 1.027
```

which reproduces the comparison report's 1.01/1.03/1.04/1.47/1.51/2.15/3.02. Of 16 rows
(`K_new ∈ {K, K+1}`) only one fell below 1 (`normal_D2_singlesample`, `K_new = 6`, 0.938),
so the direction is data-dependent but overwhelmingly inflationary. The internal reviewer's
0.785× requires `V` positively associated with `1/λ²` — the dimensions with the largest
component spread having the *smallest* length scales — which their synthetic `lambd ∝
(0.3, 1.0, 1.6)` example produced by construction. Inert for `D == 1` (both readings coincide).

**Sibling branches are clean.** Type 1 (`:888`) uses `sigma0 = vp.sigma.copy()`, shape
`(1,K)`, no `λ` division. Type 3 (`:981-986`) computes `np.sqrt(np.mean(V) / K_new)`;
`np.mean` of the `(D,)` array is a scalar, identical to MATLAB's `mean(V)` on a `(D,1)`
column — no broadcast, no discrepancy. (Type 3's other deviation, the frozen-sigma
`vp.sigma[:, np.arange(K_new) % K]` branch, is the recorded deliberate change in
`known_differences.md:834-855`.) F1 is the only expression of this kind in `_vb_init`.

**History.** `vbinit_vbmc.m:32` unchanged since `1eb4030` (2019-04-17), i.e. long before
the port; the only post-port change to the file, `c387612` (2022-10-26), moved `add_jitter`
into the loop and PyVBMC already matches that. The Python line dates from `9267816`
(2021-08-25, "feat: added VP optimization (#26)"). So the lines never matched.

**When it fires at defaults.** Type-2 candidates exist only when `slow_opts_N ≥ 2`, i.e.
`N_slowopts = elbo_starts = 2`: iteration 0 and the iteration after warm-up ends (the two
sites that set `recompute_var_post = True`, `vbmc.py:948` and `:2080`), every warp
evaluation and the warp-undo refit (`vbmc.py:1324`, `warp_rotoscaling = True` by default),
and every iteration if `always_refit_vp` is set. Correction to the comparison report: the
warp branch reaches `slow_opts_N = 2` by passing `elbo_starts` directly, not through
`recompute_var_post`.

## (b) internal F3 — `_eval_full_elcbo` and `entropy_switch`

**Behavior shared with MATLAB; not a port discrepancy.** MATLAB's nested
`eval_fullelcbo` (`misc/vpoptimize_vbmc.m:257-303`) computes
`NSentFineK = ceil(evaloption_vbmc(options.NSentFine,K)/K)` at `:279` unconditionally —
no `K == 1` test, no `optimState.EntropySwitch` test — and calls
`negelcbo_vbmc(theta,0,vp,gp,NSentFineK,0,computevar_flag,...)` at `:288-289`;
`misc/negelcbo_vbmc.m:104-109` branches on `Ns > 0` to `entmc_vbmc`. Python is
line-for-line the same: `ns_ent_fine_K` at `:482`, passed at `:493`, and `_neg_elcbo`'s
`if Ns > 0: entmc_vbmc(...)`. With `NSentFine = @(K) 2^12*K` on both sides, `K = 1`
gives 4096 samples on both.

`entropy_switch` is read in exactly one place per side within this slice —
`misc/vpsieve_vbmc.m:30` / `variational_optimization.py:766` — where it (or `K == 1`)
zeroes `NSentK`/`ns_ent_K` and thus selects the *deterministic optimizer* branch for the
inner optimization. Neither side's full-ELCBO evaluation consults it at all, so the
"second, milder inconsistency" the internal reviewer describes is also shared.
(`optim_state["entropy_switch"]` is itself forced False for `D < det_entropy_min_d = 5`,
matching `misc/setupvars_vbmc.m:210-213`.)

The numerical claim holds, with my own numbers (`chk_ent.py`, 30 repeats, unit
`sigma`/`lambd`, `Ns = 4096`): `entlb_vbmc` returns the exact Gaussian entropy
(1.41894 / 7.09469 / 14.18939 for `D = 1/5/10`, matching the closed form), while
`entmc_vbmc` at the same budget has SD 0.0156 / 0.0376 / 0.0384 — at or above
`tol_improvement = 0.01`, and `varH` is hard-coded to 0.0 so `elbo_sd` does not report it.
I could not reproduce the reviewer's 0.0546 at `D = 10`; my measurement is 0.038.

---

## Evidence and corrections for the remaining items

**int F2 + cmp F6 (stale `eta`).** The two accounts agree and both are right; the
comparison report supplies the missing half — MATLAB has no such state because
`misc/rescale_params.m:33-37` does `rmfield(vp,'eta')` (and `:40` `rmfield(vp,'mode')`)
on both the starting candidate (`vpoptimize_vbmc.m:62`) and the winner (`:189`).
PyVBMC's `set_parameters` (`variational_posterior.py:1122-1141`) writes `self.w` and
clears `self._mode` but never touches `self.eta`; `_neg_elcbo:1179-1184` is the only
writer. Sequence confirmed in source: `_eval_full_elcbo(i_mid,…)` at `:291`, then
`_eval_full_elcbo(i_end,…)` at `:304`, then both `copy.deepcopy(vp0)` at `:308-309`,
then `vp.set_parameters(elbo_stats["theta"][idx,:])` at `:326`. My run gave a mismatch in
6 of 6 seeds (e.g. `w = [0.5178, 0.4822]` vs `softmax(eta) = [0.5235, 0.4765]`); the
internal reviewer's 4-of-8 is consistent — it depends on whether a midpoint wins.
One extra consequence neither report states: the inconsistency is also carried by
`IterationHistory`'s deep-copied VP records and by saved `.pkl` posteriors.

**int F13b + cmp F2 (bounds).** The two accounts agree on mechanism; the comparison
report is the complete one. MATLAB's chain: `vpsieve_vbmc.m:40` calls `vpbounds` on its
local `vp` *before* `vbinit_vbmc`, `vbinit_vbmc.m:39` (`vp0_vec(iOpt) = vp;`) copies the
whole struct including `bounds` into every candidate, and `vpoptimize_vbmc.m:188` returns
a candidate, so the accumulation survives into the next iteration; `setupvars_vbmc.m:98`
is the only reset. `vpbounds.m:9-15` only initializes the reversed sentinels when the field
is empty. PyVBMC's `_vb_init:1027` clears `bounds` on every candidate, so `get_bounds`
(`variational_posterior.py:295-390`) always starts from `±inf` and the `np.minimum`/
`np.maximum` accumulation is idempotent within a call. Confirmed in `chk_eta.py`: the
incoming posterior had all six bound keys and the returned one has `bounds is None`.
Identical while `gp.X` only grows; divergent after the post-warm-up trim and after a warp.
No record anywhere in `dev/` or `pyvbmc/vbmc/README.md` explains the reset; the line
dates from the original port (`9267816`), and the test that pins it
(`test_variational_optimization.py:603`) was added on 2026-09-05 in `d144a83`, a
performance commit whose message is about candidate shells, not about bounds. Small
correction to the internal report's line citation: `get_bounds` is
`variational_posterior.py:295-390`, not `:331-352`.

**int F10 + cmp F9, window wrap.** Both reports get the `x0` aliasing right (exactly one
update leaks, because `np.minimum`/`np.maximum` at `minimize_adam.py:98` rebinds `x`
from the second iteration on — confirmed: `x0` ends one Adam step away). The internal
report's account of the trailing window is wrong in two details. The regime is
`max_iter ∈ {11,…,19}` only, not `max_iter < 40`: with `i = max_iter − 1`, the start index
`i − 19` is negative for `max_iter < 20` and Python reads it as `2·max_iter − 20`, which
is `≤ 0` (hence clamped, hence harmlessly the full range) for `max_iter ≤ 10`. Measured
window widths: `max_iter = 11 → 9`, `15 → 5`, `19 → 1`, `20 → 20` (`chk_win.py`). And for
`max_iter = 15` the window is columns 10–14, not "11–14". Early stopping cannot reach it
(`min_iter = 40`). MATLAB is not merely different here: `xtab(:,iter-batchsize+1:iter)`
with `iter < 20` is a MATLAB index error, so MATLAB raises where Python silently averages
the wrong window. Latent either way — the shipped `max_iter_stochastic = 100·(2+D) ≥ 300`.

**int F7 + cmp F4 (deterministic branch).** Both accounts hold; the comparison report is
the complete one. Verified by check: `sp.optimize.minimize(..., tol=t)` for BFGS sets
`gtol`, a gradient-norm tolerance, while MATLAB's `vbtrain_options.TolFun` is a
function-value tolerance (`misc/vpoptimize_vbmc.m:76`); MATLAB's
`MaxFunEvals = 50*(vp.D+2)` (`:77`) has no Python counterpart and BFGS's own default is
`200·len(x0)`; MATLAB's `try/catch` fires only on a thrown error, never on `exitflag = 0`,
so a precision-loss or iteration-limit outcome continues in MATLAB and aborts the whole
`VBMC.optimize()` in Python. Demonstrated `success=False` with a usable `res.x`
(`chk_det2.py`, `maxiter` exhaustion: `status 1`, `x = [4.87e7, −4.87e7]`). On the easy
`normal_D2_K1` fixture the branch converges in 9 evaluations against MATLAB's cap of 200,
so the cap difference is latent on well-conditioned problems. The `RuntimeError` is indeed
built from two positional arguments and renders as
`('Cannot optimize variational parameters with', 'scipy.optimize.minimize.')`.

**int F4 vs cmp (the `x_tab`/`y_tab` offset).** The comparison report is right and the
internal report's "suspected defect" is wrong. `utils/fminadam.m` has the identical order:
`[ftab(iter),grad] = fun(x);` at `:47`, the update at `:59-60`, `xtab(:,iter) = x;` at `:63`.
The final `x = mean(xtab(:,iter-batchsize+1:iter),2)` / `f = mean(ftab(...))` at `:96-97`
are shifted against each other on both sides, and `[~,idx_mid] = min(fval_lst)` feeding
`theta_lst(idx_mid,:)` at `vpoptimize_vbmc.m:133-134` is the same pairing as
`variational_optimization.py:290-292`. My reproduction: `max|y_tab[1:] − f(x_tab[:, :-1])|
= 0.0` while `max|y_tab − f(x_tab)| = 0.577`, and the sequence of evaluation points is
`[x0] + x_tab[:, :-1]`. Classify as shared behavior, not a defect to fix in the port.

**cmp F3 (sieve size).** Verified independently: `grep` over `vbmc.m` finds exactly one
bare assignment to `K`, at `:459` (`K = options.Kwarmup;`, default `2`), and `:699` reads
`ceil(evaloption_vbmc(options.NSelbo,K))`, so MATLAB's main loop always requests 100 fast
candidates (10 incremental), while `vbmc.m:584` and `misc/finalboost_vbmc.m:32` both use
`Knew`. PyVBMC uses `50·vp.K` / `⌈5·vp.K⌉`. Location correction: `vbmc.py:1504-1506`, not
`:1505-1507`. The two sides agree at `vp.K = 2` — i.e. throughout warm-up — and diverge
once `update_K` grows `K`: at `vp.K = 10`, 500 vs 100 candidates. PyVBMC's warp branch
(`:1322`) and `final_boost` (`:2484`) both agree with their MATLAB counterparts. No record.

**int F13a (`skip_elbo_variance`).** The internal reviewer treats this as Python-only dead
state; it is a faithful port of `misc/vpoptimize_vbmc.m:281`
(`isfield(options,'SkipELBOVariance')`), and `SkipELBOVariance` is absent from `vbmc.m`'s
defopts too, so the branch is dead on both sides. The claim that it would break if
reachable is correct on both sides (`elbostats.J_sjk(idx,:,1:K,1:K) = []` in MATLAB,
assigning `None` into a float array in Python).

**int F6 (`var_ss`).** The formula has a stated origin after all — MATLAB's, exactly:
`misc/gplogjoint.m:399-406` reads `varss = varFss + std(varF);` (MATLAB's `std` on a
vector normalizes by `N−1`, matching `ddof=1`), introduced in the first commit `a1680e3`
(2018-05-04) and never touched. The internal reviewer's observation that this adds an SD
of variances to a variance of means stands as an observation about *both* codebases and
about the appendix (§A.2.2 defines no such term), but it is not a port matter. The
docstring of `optimize_vp` (`var_ss` "Estimated variance of the ELBO, due to variance of
the expected log-joint") describes `varG_ss` alone and is inaccurate for what is returned.

**int F8 / int F9 (recorded decisions).** The exclusion of `eta` from the soft-bound loss is
deliberate and fully recorded: `known_differences.md:736-761` ("The eta soft bound was
removed…"), PI decision of 2026-09-08 selecting treatment A, `dev/plans/latent-bug-fixes.md`
Phase 6 and Q1, evidence in `dev/results/2026-09-08-eta-bound-fix.md`,
`…-eta-bound-comparison.md`, `…-eta-equal-budget.md`. The reviewer's residual observation
that `get_bounds` still computes `eta_lb`/`eta_ub` (and its `tol_weight == 0` special case)
is correct and the sheet entry does not mention it. `elcbo_beta = 0` is recorded at
`known_differences.md:797-809` as an unported feature inactive at MATLAB defaults.

---

## Things neither report mentions

- **`_vb_init` type 3 consumes one fewer random draw than MATLAB when `optimize_mu` is
  False.** MATLAB draws `ord = randperm(Nstar)` unconditionally at `misc/vbinit_vbmc.m:86`,
  *before* the `if vp.optimize_mu` at `:87`; Python draws `rng.permutation(N_star)` inside
  the `if` (`variational_optimization.py:969-971`). With `variable_means = False` the random
  streams part from there. Non-default on both sides (`VariableMeans`/`variable_means`
  default to yes/True), but it contradicts the comparison report's blanket statement that
  "*what* is drawn and *in which order* is identical in every branch".
- **A second unstable-sort site.** `variational_optimization.py:899` uses
  `np.argsort(y_star, axis=None)[::-1]` where MATLAB has `sort(ystar,'descend')`
  (`vbinit_vbmc.m:26`). Reversing an unstable ascending sort orders ties in the *opposite*
  direction from a stable descending sort, so tied `y_star` values select different HPD
  points for the type-2 means. The comparison report's F10a names only the `_sieve` sort at
  `:821`; ties in `y_star` (e.g. repeated observations) are more plausible than ties in
  `nelcbo`.
- **`get_bounds`'s `tol_weight == 0` special case is a PyVBMC addition, not a port.**
  `variational_posterior.py:359-363` sets `eta_lb = -inf` when `tol_weight == 0`
  "to prevent warning to be printed when doing final boost"; `misc/vpbounds.m:28` has no
  such case (MATLAB simply gets `log(0) = -Inf`). With the eta bounds now excluded from
  the loss it is doubly dead. The internal report mentions the special case but not that
  MATLAB lacks it.
- **`get_parameters` does not invalidate the cached mode.** MATLAB's `rescale_params`
  ends with `rmfield(vp,'mode')` (`:40`); `VariationalPosterior.get_parameters`
  (`:1025-1035`) performs the same `nl` renormalization and weight normalization but leaves
  `self._mode`, with only the comment "remove mode (at least this is done in Matlab)".
  Harmless — the renormalization is a pure gauge change that leaves the density and hence
  the mode unchanged — but it is an unported line.
- **The pruning loop's `np.delete(vp_pruned.eta, idx)` drops the `(1,K)` shape** (no
  `axis=`), leaving `eta` 1-D; it is restored to `(1,K−1)` by `_neg_elcbo:1181` on the very
  next call, so nothing observes it. MATLAB's `vp_pruned.eta(idx) = []` keeps the row
  vector. Cosmetic; noted only because it interacts with the stale-`eta` item above.
