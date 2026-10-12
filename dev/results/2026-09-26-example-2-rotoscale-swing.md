# Example 2: the swing of the ELBO after a kept rotoscaling

Example 2, executed on 2026-09-26 (Rosenbrock's banana with an exponential
prior, bounded, `D = 2`, seeded, plotting on), keeps a rotoscaling in
iteration 10. Its trace reads −0.82 ± 0.87 there, −104.79 ± 10.60 in
iteration 11, then −2.44 and −2.03, and the run ends stable at −1.871
against the true −1.836. The question was whether the warp leaves the GP or
the variational posterior in a state that a later iteration has to repair.

It does not. The warp re-expresses the training data and the posterior
correctly. The swing comes from the undo check that follows the warp: the
refit posterior spreads into a region where the refit GP has no data and
extrapolates optimistically, its reported ELBO rises by about one nat, and
the check reads the rise as the improvement it requires. The next iteration
evaluates the target where the posterior spread, and two iterations later
the estimates have recovered. The rule, and everything the refit depends
on, are MATLAB VBMC's. One other run of the 990 in the golden reference
shows the pattern, and it recovers too. The rule stays unchanged for 1.5;
a possible guard is recorded in `dev/TODO.md` under "Outside 1.5 scope".

## Method

[`replay_example2.py`](../experiments/example2_rotoscale_20260926/replay_example2.py)
runs the notebook's problem with its options and seed, with `train_gp`,
`optimize_vp` and `warp_gp_and_vp` wrapped in the namespace of
`pyvbmc.vbmc.vbmc`, so that each GP fit, each variational optimization and
the warp are copied as they happen. The wrappers draw no random numbers,
and the script asserts that its trace equals the notebook's stored output.
It ran in the notebooks environment that `dev/scripts/runs/LOCAL.md`
describes, on a clean worktree of `812d66ef`, whose numerics are those that
executed the notebook, with BLAS single-threaded as the notebook executes;
the provenance of
[`summary.json`](../experiments/example2_rotoscale_20260926/summary.json)
names the commit, the gpyreg checkout and the versions, and the file holds
every number quoted here for iterations 8 to 14.

For each optimized posterior the script estimates the true ELBO,
E_q[log p(x)] − E_q[log q(x)] in the original space, from 2 × 10^5 draws
against the real log joint, with standard errors of at most 0.003 nats
near the solution, 0.5 in iteration 11, and 73 and 213 for the spread
posteriors of iteration 10. It also measures how far the posterior's mass
lies from the GP's training inputs, in units of the GP's mean length
scales, and its Gaussianized sKL from the posterior of the iteration
before, the measure of the trace's `sKL-iter[q]`.

## What happens

| Fit | Reported ELBO | True ELBO | Mass beyond 3 length scales |
|---|---|---|---|
| iteration 9 | −1.984 ± 0.017 | −1.991 | 0 |
| iteration 10, refit of the undo check | −1.064 ± 0.303 | −4,659 | 28 % |
| iteration 10, the fit of the trace | −0.822 ± 0.874 | −23,096 | 49 % |
| iteration 11 | −104.787 ± 10.601 | −30.0 | 0 |
| iteration 12 | −2.438 ± 0.055 | −2.374 | 0 |
| iteration 13 | −2.027 ± 0.009 | −2.024 | 0 |

1. **The warp is exact.** In the new space the training targets equal the
   log joint plus the log-Jacobian to 2 × 10^-14, and the transformed
   posterior's true ELBO is −2.086, against −1.991 before the warp; the
   0.1 nat is the cost of the unscented transform of the mixture.
2. **The refit GP extrapolates optimistically.** The 51 training inputs left
   after the warm-up trim span 8.1 nats of the log joint. Before the warp
   the GP's output scale is 46 to 74 across its hyperparameter samples, and
   the width ω₂ of its negative-quadratic mean along the second axis 0.1 to
   3.2. The refit lands in another mode: output scale 0.9 to 1.5, and ω₂
   from 3.7 to 175 in the whitened space, within the bound of 183 that
   gpyreg sets at e³ times the spread of the training inputs along that
   axis. Along that axis the mean barely falls: where the refit posterior
   puts its mass beyond three length scales from the data, the GP predicts
   −6.1 on average and the log joint averages −16,000.
3. **The posterior spreads into the region the GP misjudges.** The undo
   check optimizes the posterior with full restarts; spreading gains
   entropy and, against this GP, loses little expected log joint.
4. **The undo check keeps the warp.** It keeps a warp whose refit ELBO
   exceeds the previous iteration's by more than `warp_tol_improvement`
   (0.1) and whose SD stays below `warp_tol_sd_multiplier` (2) times the
   previous SD plus `warp_tol_sd_base` (1): −1.064 against −1.984, and
   0.303 against 1.034. The gain is an overestimate of the GP, not a better
   posterior; the refit posterior's sKL from that of iteration 9 is 38.
   Active sampling is skipped in a warp's iteration, so the GP is fitted
   again on the same data and the posterior spreads further, which gives
   the trace's −0.82 and its sKL of 118.
5. **The next iteration corrects it.** Active sampling follows the spread
   posterior and evaluates the target where the log joint reaches −12,474.
   Fitted to that range of outputs, the GP's output scale grows to 3,700 to
   7,100, and near the peak it is now pessimistic: the reported ELBO is
   −104.79 where the posterior's true ELBO is −30.0. Iterations 12 and 13
   add points near the peak, and the reported and true ELBOs agree again.

## Against MATLAB VBMC

Nothing in this sequence differs from MATLAB VBMC (its repository at
`396d649`):

- the undo check and its tolerances (`vbmc.m:566-620`, options at
  `vbmc.m:352-355`);
- the starting points of the refit, which include the hyperparameters of
  the GPs recorded before the warp (`misc/gptrain_vbmc.m:36-47`), as
  `train_gp` does;
- the bounds of the negative-quadratic mean, ω up to `exp(3)` times the
  spread of the training inputs (`gplite/gplite_meanfun.m:141`, `:227`),
  and the cap on its maximum (`misc/gptrain_vbmc.m:186`, option
  `gpQuadraticMeanBound`);
- noise shaping, which would discount the far evaluations of iteration 11,
  is off by default (`vbmc.m:320`), and PyVBMC does not port it.

## How often

[`scan_golden_warps.py`](../experiments/example2_rotoscale_20260926/scan_golden_warps.py)
reads the traces of the golden reference `reference_990_20260913`, whose
archives are local to the machine that `dev/scripts/runs/LOCAL.md` lists,
and writes
[`golden_scan.json`](../experiments/example2_rotoscale_20260926/golden_scan.json).
The traces record the ELBO of each iteration, not that of the undo check's
refit. The 990 runs propose 1,736 rotoscalings and keep 641.

- One kept warp is followed by a drop of more than 5 nats: `banana_D10`,
  seed 42, whose iteration 19 reads 11.93 ± 4.26 against a true log
  evidence of 0, after −0.32, and then −12.91, −2.97 and −1.87. The run
  ends 0.11 nats below the truth.
- 31 other kept warps report an ELBO more than 0.5 above the true log
  evidence in their iteration, mostly on noisy targets (`logreg_D5_noise3`
  9, `rosenbrock_D2_noise3` 7, `timing_D5_noise2.2` 6). They overshoot by
  0.51 to 1.85 nats, except `rosenbrock_D2_noise1` seed 9 (5.83, with an
  SD of 21.2), and no drop larger than 2 nats follows any of them.
- 11 kept warps report an sKL above 10 in their iteration (`cigar_D8` 7,
  and `banana_D10`, `cigar_D4`, `lumpy_D10` and `student_D8` one each),
  the `banana_D10` run among them.

## Disposition

The acceptance rule stays MATLAB's for 1.5: changing it moves the default
trajectories, and the pattern is rare and recovers within two iterations.
A candidate guard follows from what a warp is: it only reparameterizes the
space, so a refit posterior that differs greatly from the one before the
warp, in the original space, signals a gain of the surrogate rather than
of the fit. A threshold of 10 on the sKL of the undo
check's refit would have undone this warp (38). The guard is untested: the
traces hold the sKL of each iteration's final fit, not of the refit, so
they cannot say which of the 641 kept warps it would undo, and it needs
the benchmark suite before adoption. Example 2's text on the output trace
says that the estimated ELBO can first rise after a rotoscale, where the
posterior spreads into a region that the surrogate has no evaluations of,
and then drop.
