# 2026-09-04 — The final boost can destroy a converged posterior

**Status:** finding recorded, decision deferred (PI, 2026-09-04: an
independent, small algorithmic tweak; to be thought about later). No code
changed. The behaviour is inherited from MATLAB VBMC, so this is not a
porting bug.

## What was seen

In the regenerated golden population
(`plans/benchmark-suite-and-golden-traces.md` §Results (regenerated)),
`student_D4` seed 19 is the one run of 280 whose posterior is wrong (ΔLML
1.33, gsKL 53.8, MMTV 0.44; three noisy-logreg seeds also miss the papers'
usability line, but only on MMTV by 0.00–0.05 and with correct ELBO and
covariance). It is not a convergence failure. The trace and a bit-for-bit
reproduction show:

| | ELBO | ELBO sd | K | gsKL | MMTV |
|---|---|---|---|---|---|
| main loop, iteration 17 (stable, `best_iter`) | −10.340 | 0.044 | 11 | 0.058 | 0.057 |
| after `final_boost` (what VBMC returns) | −9.031 | 0.494 | 50 | 53.8 | 0.441 |
| true ln Z | −10.358 | | | | |

The boosted posterior has one component of weight 0.55 with width 1.2 in
box-scaled units, against 0.06–0.20 for the pre-boost components; it sits
at the soft upper bound on a component's scale (the data range) and its
centre is 1–4 posterior SD from the mode per coordinate (2.1, 3.9, 3.8,
1.0). Two signals mark the failure: the
boosted ELBO exceeds the true log evidence by 1.3, which an honest bound
cannot do, and the ELBO uncertainty grows tenfold. Over the 280 runs the
next largest change of ELBO through the boost is 0.21.

## Mechanism

The boost re-optimizes the variational posterior against the *same* GP as
the last iteration, and for this seed that GP is unusual. Its
negative-quadratic mean function is nearly flat: curvature scale ω over the
data extent about 12, 2, 2.4, 1.5 per dimension (medians over the 8
hyperparameter samples), and `m0` 8 nats below the observed peak. On 17 of
the other 19 seeds ω/extent is 0.05–0.2 and `m0` sits 1–4 nats below the
peak; seed 15 is a milder case of the same (ω/extent 0.6 and 1.0 in two
dimensions, `m0` 5.5 nats down) and stays usable. The log-ω spread over
seed 19's 8 samples is 2–5 nats, so some samples have essentially no
curvature. Away from the data the GP posterior mean reverts to that mean
function, and the surrogate predicts a log density about 5 nats too high
in the empty parts of the plausible box: at the giant component's centre
the GP mean is −13.6 ± 2.0 in the GP's units (the transformed-space log
joint, which carries the +9.3 log-Jacobian of the affine transform), where
the true value is −19.1 (−28.3 in original units).

Under such a surrogate a box-wide component gains about 9 nats of entropy
and loses about 8.5 nats of expected log joint (the flat level is 8–9 nats
below the peak, against roughly 15 for the truth), so the surrogate ELBO is
nearly indifferent to it, and the boost's optimization can drift into it:
it optimizes the plain ELBO (`elcbo_beta = 0` in `optimize_vp`, so the
GP's uncertainty of 2 nats there costs nothing), turns pruning off
(`tol_weight = 0`) and starts 39 new components from the sieve. The main
loop, with K = 11, Adam started from the previous solution and pruning on,
had stayed in the good basin under the same GP.

Re-running `final_boost` six times from the stored pre-boost state, with
fresh random draws each time, gave gsKL 4.8, 0.61, 1.56, 47, 0.29, 52: four
of six unusable, two of them as bad as the recorded run. For this GP fit the
boost is a coin flip, not a rare draw.

## Parity with MATLAB

`misc/finalboost_vbmc.m` in `acerbilab/vbmc` does the same: `TolWeight =
0`, entropy sample counts raised, warm-up ended, then `vpoptimize_vbmc` on
the plain ELBO, and the result is returned unconditionally. PyVBMC's
`VBMC.final_boost` is a faithful port. `optimize_vp` notes that MATLAB's
`ELCBOWeight` option (default 0) was not ported at all.

## Options

Each changes results on this one trace of the 280 in the reference
population, so the baseline stays valid up to that trace.

1. **Guard the boost** (least invasive). Accept the boosted VP only if its
   ELCBO with the existing `safe_sd = 5` (the `determine_best_vp`
   criterion) is not worse than the pre-boost one; otherwise keep the
   pre-boost VP and warn. Here −11.5 vs −10.56: the guard would have kept
   the good solution. Costs nothing on the other 279 runs, where the boost
   changes the ELBO by less than 0.21.
2. **Optimize an ELCBO with β > 0 in the boost**, i.e. port `ELCBOWeight`
   and set it for the boost only. Penalizes exactly the region where the
   failure lives, but changes every boosted posterior slightly.
3. **Bound new components' scale in the boost** relative to the existing
   VP's scales rather than the data range. Targets the symptom, not the
   cause (the flat GP mean), and touches `get_bounds`.

A flat GP mean function far from the data is also what makes the
acquisition function over-value empty regions, so the same GP fits are
worth a look when Stage 2 item 8 touches the hyperparameter sampler.

## Reproduction

Everything is in `dev/scripts/runs/golden/baseline_20260903/
student_D4_seed19.{npz,json}` (per-iteration VP parameters and transformer
state, final VP, GP hyperparameters per sample). Live reproduction, from the
repo root with `dev/scripts` on `sys.path`, one BLAS thread, about 80 s:

```python
from benchmark_targets import find_config, metrics
from pyvbmc import VBMC

prob = find_config("student_D4").make(seed=19)
args, options = prob.vbmc_args()
options.update(display="off", plot=False, print_iteration_header=False)
vbmc = VBMC(*args, options=options, seed=19)
vp, results = vbmc.optimize()                    # ELBO -9.031 +/- 0.494
hist, b = vbmc.iteration_history, results["best_iter"]
vp_pre, gp_pre = hist["vp"][b], hist["gp"][b]    # ELBO -10.340 +/- 0.044
print(metrics(prob, vp_pre, hist["elbo"][b])["gskl"])   # 0.058
for _ in range(6):                               # 4 of 6 give gskl > 1
    vp_b, elbo_b, sd_b, _ = vbmc.final_boost(vp_pre, gp_pre)
    print(elbo_b, sd_b, metrics(prob, vp_b, elbo_b)["gskl"])
```

`gp_pre.predict(vp_b.mu.T)` against `prob.log_density_vec(
vp_b.parameter_transformer.inverse(vp_b.mu.T))` shows the over-prediction
at the boosted components; `gp_pre.get_hyperparameters(as_array=True)`
columns 6 and 11–14 are `m0` and log ω (D = 4: 5 covariance, 1 noise, 9
mean hyperparameters).
