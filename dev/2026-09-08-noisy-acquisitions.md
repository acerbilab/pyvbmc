# Noisy-target acquisitions: what VIQR costs, what could replace it, and where the search goes wrong

*Written 10 September 2026 from a two-day measurement session; interim.*

The question was whether VIQR, which makes noisy runs slow, could be
replaced or joined by acquisition functions that work with noisy targets,
given that every pointwise acquisition in the package is unusable with
noise (the 2020 paper's NPRO result; the mechanism is in the first
report). The short answer after two days of measurement is that VIQR's
criterion is sound and should stay, that the cost of a noisy run has two
identifiable sources of which one can be cut without loss, and that the
*optimization* of the acquisition, not the acquisition itself, is where
the best evaluations are being lost. Interim conclusions and suggestions
follow; the evidence is in two reports,
[the acquisition experiments](results/2026-09-08-noisy-acquisition-experiments.md)
and [the search analysis](results/2026-09-09-acquisition-search-analysis.md).
All of it is on synthetic targets with emulated Gaussian noise, one seed
per saved state, ten seeds per end-to-end arm on the two-dimensional
target and six on the harder configurations, with wall times measured
while several runs shared the container's cores, so the conclusions are
interim by construction. Nothing on the branch
(`claude/pyvbmc-noisy-acq-funcs-9kwdny`) changes a default.

## Where a noisy run spends its time

A noisy run does two things a noiseless one does not. It refits the GP
hyperparameters, with slice sampling, after every single evaluation while
the solution is unstable (the 2020 paper's "frequent retrain"), and it
evaluates VIQR on a sieve of 8192 candidates per new point. Timed
operation by operation on five targets from D = 2 to 10, the sieve call is
the largest item on four of them (47 to 70 % of the run) and the GP fits
the rest; the in-loop VP optimization, the CMA-ES stage, the
importance-sample set-up and the final boost are a few percent together.
Inside a sieve call, a third to a half is the GP prediction at the
candidates, a fifth forms kernel matrices of which the block against the
training inputs the prediction already formed, and a sixth to a third is
elementwise transcendental work on 8192 × 100 arrays that could be two
sweeps instead of four. Inside a GP fit, a quarter is a space-filling
initial design on four of the five targets, which the in-loop refits
could replace with a warm start from the previous hyperparameters.
Together those savings are a fifth to a third of the wall time of these
runs, least where the fits dominate. The first two change no numerics
beyond rounding; the warm start changes which hyperparameters a refit
returns and how many draws it consumes, so it moves the golden traces
and the `gp_fit` oracle and needs their gate.

## What was tried on the acquisition itself

Every noise-capable acquisition here has the same three parts: a weight
saying where the posterior mass is, the exact look-ahead of the GP
variance after a hypothetical observation at the candidate (independent of
the value observed there), and a loss transform integrated over the
weight. The pointwise acquisitions use the candidate's own variance
reduction and its own weight, which with noise is neither global nor
robust; tweaking them is not a path. The prototypes changed the loss, the
criterion, or what one acquisition step is:

- **Cheaper losses in the VIQR code path** (`AcqFcnVIQR(loss=...)`, the
  default bit-identical to before). The integrated variance reduction is
  three times cheaper per sieve call and ties VIQR on the two-dimensional
  targets, but loses clearly at D = 5 and D = 8 with no win in twelve paired
  runs. The interquantile range's exponential weighting of uncertain
  regions is doing real work on harder targets.
- **The expected information gain** (`AcqFcnEIG`, the port of the MATLAB
  acquisition, plus a new per-component variant) is five times cheaper per
  sieve call and picks the same point as VIQR on a snapshot, yet is
  unusable end to end; the per-component variant is competitive only at
  low noise.
- **Repeated observations** (training inputs join the sieve behind
  `max_repeated_observations`; an exact repeat is pooled by the logger and
  adds no GP row) never hurt, halve the posterior error on the
  low-dimensional high-noise case from a handful of repeats, and are
  neutral on the harder configurations. The evidence error trends the other
  way, which six to ten seeds cannot settle.
- **Switching the frequent retrain off** halves the wall time at σ = 1 at
  no cost and is five times worse at σ = 3. Refitting only the VP keeps the
  median accuracy but not the usable fraction; a MAP-only in-loop GP refit
  (`ns_gp_max_active = 0`) loses. The expensive part is the GP
  hyperparameter refit, and it earns its cost only when the noise is
  large, which argues for keying it to the noise estimate the loop already
  computes rather than for removing it.

So the criterion stays VIQR, and the acquisition-side savings are the
operation-level ones above plus a noise-adaptive retrain policy.

## Where the search goes wrong

The PI's recollection was that CMA-ES did poorly on the acquisition and
that a larger sieve kept helping, which is why the sieve is 8192. On saved
mid-run states this is exactly what the numbers show, with a mechanism
behind each half.

CMA-ES never improves on the sieve's point. Its objective is the log of the
*residual* interquantile range, which varies by a few thousandths across
the whole candidate set because one observation reduces the range by a
thousandth, against an absolute tolerance of a hundredth; and its initial
step is the posterior width, so the first generations sample far from the
sieve point. It stops after two generations on every state. The sieve
alone captures three quarters of the achievable reduction at D = 5 and
three fifths at D = 8, measured on the 100-sample surface the search
sees, and grows only logarithmically with its size.

A gradient step from the sieve's point (L-BFGS-B on the log of the
*reduction*, which has the same minimizer and an order-one range, with the
gradient from one batched acquisition call) appeared to recover the whole
gap at a few percent of a sieve's cost. That result was wrong in an
instructive way, and the PI's caution about bumpy landscapes was the right
instinct. The surface VBMC optimizes is a 100-sample Monte Carlo estimate;
judged on an independent 4000-sample set, a point refined on the 100-sample
surface is worse than the sieve's point in up to three draws of five. The
refinement was climbing the sampling noise, and the sieve's own pick is
fitted to the same set: the same sieve on a 400-sample set gains up to 1.7
times the reduction under the judge. Where the GP length scales approach the
posterior width or N is large the surface also fragments into many genuine
basins, so random restarts mostly land in poor ones.

Refining on a larger importance set fixes the first problem: with 400 to
1600 samples the refined point beats the sieve's by 7 to 27 % in the median
under the independent judge, and is worse in one draw of twenty at 1600. The
measured cost of the full pipeline then favours a split: a sieve of 1024
candidates on the cheap 100-sample set, the top few re-scored on a
1600-sample set, and one L-BFGS-B run there. That pipeline's judged
reduction is 1.26 to 1.95 times today's search's, with no draw below 1.05,
at 61 to 75 % of its time. The large set's set-up is a real cost, about a
third of a sieve call, and it scales as Ns · N² · Na; evaluating the
refinement candidate-side (solve for the D + 1 points, contract with a
kernel block formed once) removes that term and is three times cheaper over
a refinement. The size of the refinement set should follow the noise of the
estimate, which is free to read off the sieve's own evaluation: on the
smooth states a 5 % standard error needs 300 to 1400 samples, while on the
bumpy ones a few importance points carry the whole reduction and no
affordable set is precise enough, which is the signal to keep the sieve's
point and skip the refinement.

## Interim conclusions

1. Keep VIQR. The cheaper losses and the information-gain criteria do not
   survive the harder targets.
2. The frequent retrain is worth its cost only at high noise; make it a
   function of the noise estimate rather than a noisy-run default.
3. The sieve is the optimizer today, and it is a weak one above D = 2
   because it resolves neither the surface nor the Monte Carlo noise of
   its own estimate. The remedy is a smaller sieve plus a gated gradient
   refinement on a larger importance set, not a larger sieve. The one
   end-to-end arm with a smaller sieve and no refinement (2048) lost at
   σ = 3, so the refinement is not optional, and the combination has to
   be tested end to end before anything changes.
4. Gradient refinement of a Monte Carlo acquisition needs an estimate more
   precise than the differences it optimizes; otherwise it fits the sample.
   That holds for any optimizer that resolves fine structure, including
   the coarse-to-fine resampling that was tried as the gradient-free
   alternative.
5. The operation-level savings in the sieve call and the GP fit are
   independent of all of the above and cost nothing in accuracy.

## Suggestions, in order

1. Implement the split search as a `search_optimizer` option: sieve of
   1024 on 100 samples, CV-gated refinement set sized between 400 and an
   N-dependent cap, top few re-scored, one L-BFGS-B run candidate-side on
   the log-reduction, the sieve's point kept unless the large-set value
   improves. Test end to end on the noisy golden configurations against
   the existing reference populations; that is the experiment that decides.
2. Take the two numerics-preserving operation-level savings (direct
   `sinh`, kernel reuse) as a separate change, and the warm-started
   in-loop refits as another, gated by the golden traces since it moves
   the fit.
3. Key the in-loop refits and the sieve size to the noise estimate at the
   high-posterior-density region, with a threshold sweep over noise levels.
4. Keep `max_repeated_observations` available; enabling it by default
   waits on more seeds, given the evidence-error trend.
5. Drop the scalar EIG; keep the per-component variant and the
   variance-reduction loss as documented options for low-noise use only
   (the `sd_reduction` and `iqr_reduction` losses were never run end to
   end), or remove them before release to avoid an untested surface.
   Whichever stay need the hand-written API page AGENTS.md asks for.
6. Look separately at the GP hyperparameter fit under heavy noise: on the
   ten-dimensional lumpy target at N = 246 the length scales collapsed in
   two dimensions and the acquisition surface went flat to one part in
   10¹³, which no search can act on.

## What is on the branch

`AcqFcnVIQR(loss=...)`, `AcqFcnEIG(components=...)`, repeated-observation
candidates behind `max_repeated_observations`, the `ns_gp_max_active`
option, the `compute_var_log_joint` hook storing the component covariance,
and their unit tests. No oracle entries for the new acquisitions, no API
pages beyond the automodule listing, no default changed. The per-run
metrics of every end-to-end arm, the operation-level timers and the
printed outputs of the search benchmarks are committed under
[experiments/noisy_acq_20260908/](experiments/noisy_acq_20260908/README.md);
the measurement scripts and the saved states they used are not.

The definition tests of VIQR and IMIQR (`test_complex__call__` in both
modules) were unseeded, and the IMIQR one failed about one run in ten:
its reference was a coarse grid 3 to 4 % high, and the acquisition's
self-normalized importance-sampling estimate scatters by 2 % across seeds
on that scenario (single seeds 4 % off), because the single slice-sampling
chain visits the low-noise half-plane, where the weights are large and
the integrand small, in correlated stretches. A 2 % offset first
attributed to the sampler was the test's own GP: its hyperparameter
array was laid out for a noise hyperparameter the noise function did not
carry, so the mean function read shifted entries and was not the target,
and IMIQR's importance-sampling target exp(f_mu) differed from the VP by
2.6 %; with the array aligned the estimate is unbiased within its scatter
(mean +0.7 % over twelve seeds). Both tests are now seeded, share one
scenario helper, check the acquisition's formula exactly against a
recomputation from its own samples, and compare with a converged
Gauss-Hermite reference at a tolerance set from the measured spread. Fixing them showed that the MCMC
step of `active_importance_sampling` (IMIQR only; VIQR samples the VP
directly) still drew from NumPy's global state, the one remaining
exception to the generator: it now receives `vp.rng`, and the
`acq_AcqFcnIMIQR` oracle was re-baselined from the stored state of the
one noisy fixture.
