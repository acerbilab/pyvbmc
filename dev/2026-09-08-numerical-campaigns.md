# Numerical campaigns: final boost, eta bounds and main-loop repairs

The September 8 campaigns addressed three problems: the final posterior
refinement could destroy a good solution, mixture-weight optimization could
return a gradient inconsistent with its objective, and several controls for
GP sampling and acquisition were incorrect or inactive. The selected repairs
are implemented; the final population benchmark remains pending.

The sections below explain each component, its defect, and the evidence behind
the chosen repair. Detailed experiment reports remain in `results/`, and the
[latent-fix plan](plans/latent-bug-fixes.md) retains the execution history.

## Where these problems sit in VBMC

The fixes concern final variational refinement, mixture-weight parameterization,
and the main-loop GP-sampling and acquisition controls. The boost comparisons
use an evidence lower confidence bound, `ELCBO = ELBO - 5 * SD`, alongside the
ELBO itself. Here SD is the GP contribution to ELBO uncertainty; it excludes
Monte Carlo error in the entropy estimate. Reported usability follows the
benchmark criterion: evidence error < 1, gsKL < 1, and MMTV < 0.2.

## Final boost: why an end-of-run refinement needs a guard

After the main loop selects its best posterior, `final_boost` performs one last
variational refit against the same fitted GP. It normally raises the mixture to
at least 50 components, increases the entropy-sampling effort, disables component
pruning, and ends warmup. It makes no new target evaluations. Historically,
PyVBMC returned this refined posterior unconditionally.

The failure that motivated the change occurred on Student D4 seed 19. The main
loop had converged to a useful posterior with gsKL 0.058 and MMTV 0.057. Final
boost returned a posterior with gsKL 53.8 and MMTV 0.441; four of six fresh
boost reruns from the same pre-boost state were unusable.

The GP for this run had an unusually flat mean function away from observed data.
In an empty part of the plausible box, it therefore predicted a log density much
higher than the true target. A very broad Gaussian component could gain enough
entropy to look attractive under that surrogate even though it put mass in the
wrong place. The boost optimized the plain ELBO, so the large GP uncertainty in
that region carried no cost, and its extra components and disabled pruning made
the bad basin easier to reach. The returned ELBO rose above the true log evidence
while its estimated SD increased about tenfold. MATLAB VBMC had the same
unconditional behavior; the PI classified guarding it as a 1.5 fix.

Early checks showed why the guard had to be treated as a policy tradeoff. It
rejected a severely inaccurate Logistic D5 candidate and recovered the good
pre-boost posterior. On selected noisy Rosenbrock cases, however, the same
score rule rejected candidates that improved posterior shape, because a lower
GP-based score does not imply worse truth-based accuracy. Those selected cases
motivated a population comparison rather than determining a threshold by
themselves.

The full experiment generated two boost candidates at each of 870 saved
endpoints: one with the historical boost weight penalty of 0.1 and one with
that penalty set to zero. All 1,740 boosts completed. Each pair began from the
same state and fresh optimization RNG state. Nine endpoints used authentic
captured GP factors; the other 861 used a common reconstructed state.
Reconstruction was sufficient for a paired comparison, but it was not an exact
replay of the historical boost. In noisy Rosenbrock states, tiny roundoff from
reconstructing transformed inputs could noticeably change the GP. Historical
golden results are therefore context, while the paired conclusions are
conditional on the shared reconstructed inputs.

The typical effect of removing the penalty was tiny, although a few raw
candidates differed substantially. The PI selected the joint policy:

- use zero small-weight penalty during final boost; and
- accept the candidate only if neither its ELBO nor its ELCBO falls by 0.1 or
  more from its pre-boost value.

Writing `dE` for candidate minus pre-boost ELBO and `dS` for the corresponding
SD change, the strict rule is

```text
dE > -0.1  and  dE - 5*dS > -0.1
```

Equality rejects. Rejection returns the preserved pre-boost posterior. The
small-weight regularization used in ordinary main-loop variational fits is
unchanged. Setting the guard option explicitly to `None`, or loading an old save
without it, preserves the historical penalized and unguarded refinement.

On the saved no-penalty candidates, the selected rule accepted 859 of 870.
Tolerance 0.1 and the more permissive 0.2 rule differed on eight endpoints:
falling back at 0.1 improved evidence error and MMTV in all eight and gsKL in
seven. Both policies produced 828 of 870 usable returned posteriors. These are
exploratory comparisons on the same benchmark used to choose the policy, and
some classifications near the thresholds changed under fresh entropy draws.
Production applies the rule to the optimizer's stored scores rather than the
campaign's independent common-GP rescoring.

The guard addresses the observed surrogate failure; it cannot certify posterior
accuracy. It can reject a beneficial candidate, and a candidate can pass its
score checks while worsening truth-based metrics. For example, Student D8
seed 12 worsened from a pre-boost gsKL of 1.859 to about 10.8–10.9 in either
penalty arm while it passed both tested guard tolerances. The pre-boost
posterior was already unusable, so the usability totals conceal this worsening.
The 859 acceptance count is not a prediction for the final integrated run
because the main-loop repairs change the endpoints. Implementation checks
passed 125 focused tests and all 11 exact stage fixtures.

Detailed boost evidence:

- [Original failure and mechanism](results/2026-09-04-final-boost-failure.md)
- [Initial guard comparison and noisy
  counterexamples](results/2026-09-07-final-boost-comparison.md)
- [Input reconstruction and its numerical limits](results/2026-09-08-boost-reconstruction.md)
- [Three-case timing pilot](results/2026-09-08-boost-penalty-pilot.md)
- [Completed 870-endpoint campaign](results/2026-09-08-boost-campaign.md)
- [Paired analysis, policy tradeoffs and final PI decision](results/2026-09-08-boost-analysis.md)

## Eta bounds: a coordinate-dependent penalty with an inconsistent gradient

The mixture weights are represented during optimization by unconstrained logits
`eta`. Softmax maps them to physical weights,

```text
w_k = exp(eta_k) / sum_j exp(eta_j).
```

Adding the same constant to every `eta_k` leaves every weight unchanged. It is
therefore legitimate, and numerically safer, to subtract the maximum eta before
evaluating the softmax. Bounds on the absolute eta values are different: they
constrain an arbitrary coordinate choice rather than the posterior itself.

The old objective had a more immediate implementation defect. It subtracted the
maximum eta in place from the optimizer's input vector, then evaluated soft eta
bounds on those shifted values while returning the unadjusted componentwise
bound gradient. Once a bound was active, that gradient was not the derivative
of the loss that had been evaluated. Mutating a vector supplied by the optimizer
was also an unsafe objective-function side effect.

The initial comparison gave names to three treatments:

- **A** removed the eta-coordinate bound loss and gradient and stopped mutating
  the caller's parameter vector.
- **B** used MATLAB's bounds on raw, unshifted eta values with a consistent
  gradient and no caller mutation.
- **C** preserved the existing PyVBMC behavior, including the mutation and the
  inconsistent active-bound gradient.

All three retained bounds on component locations and scales, pruning, and the
separate capped penalty on small *physical weights*. That weight penalty depends
on `w`, so it respects the common-shift invariance of eta. Removing eta-coordinate
bounds did not remove ordinary main-loop weight regularization.

The first local experiment appeared to favor B on two noisy Rosenbrock states,
but the optimizer often ran B for 160–400 iterations while A and C stopped
around 40. The extra eta penalty had changed stopping behavior as well as the
objective, so score differences could not identify the cause.

The follow-up therefore ran A and B for exactly 400 Adam iterations from the
same saved starts, with three optimizer RNG streams on each of two noisy states
and repeated common-draw scoring. The apparent large advantage disappeared.
At iteration 400 every absolute ELBO difference between B and A was below
`5e-4`, and ELCBO differences changed sign across states and replicates. Both
arms benefited similarly from simply optimizing longer.

The PI selected A. Production now excludes eta from the generic soft-bound loss,
does not mutate the caller's theta, and retains stable private softmax arithmetic,
location and scale bounds, the capped physical-weight penalty, pruning, and the
existing optimizer settings. Whether noisy variational fits stop too early is a
separate research question; it was explicitly deferred and is not a release
blocker.

The correction passed 64 focused tests and all 11 stage fixtures exactly. In
three paired trajectories, all results remained usable. Normal D5 seed 0 did
cross its fixed population-derived gsKL fence (`0.000490743 > 0.000429099`),
even though its pre-boost evidence error and gsKL improved. That adverse flag is
retained for the final population assessment; the change is not claimed to be
trajectory-neutral.

Detailed eta evidence:
[initial three-treatment comparison](results/2026-09-08-eta-bound-comparison.md),
[equal-budget experiment](results/2026-09-08-eta-equal-budget.md), and
[selected production fix](results/2026-09-08-eta-bound-fix.md).

## Three repaired main-loop paths

The campaign also repaired three independent paths that affect how VBMC learns
and when it spends computation.

**GP hyperparameter-history covariance.** A GP fit can retain multiple plausible
hyperparameter samples. PyVBMC uses the covariance of recent samples to choose
the proposal widths for later hyperparameter sampling. The historical estimator
had its sample orientation and counts wrong, used a scalar dot product where an
outer-product covariance was required, and applied history decay with incorrect
parentheses. The repair treats every historical sample as one row, distributes
an iteration's decaying weight across however many samples it contains, and
computes the unbiased weighted outer-product covariance. Histories with changing
sample counts are supported; incompatible dimensions are skipped, and undefined
covariance falls back to the existing default proposal widths.

**Acquisition variance regularization.** The acquisition function ranks
candidate locations for the next expensive target evaluation. Its variance
regularizer is meant to suppress candidates whose GP uncertainty is already
below `tol_gp_var`, discouraging repeated effort in well-constrained regions.
Initialization wrote `variance_regularized_acq_fcn`, while the reader looked for
the older `variance_regularized_acqfcn` spelling, so new runs silently left the
regularizer inactive. Column-vector acquisition outputs could also broadcast
against pointwise masks into a matrix. The repair gives the canonical key
precedence with a legacy fallback and normalizes supported vector shapes to one
value per candidate before regularization and bound masks.

**Termination of GP hyperparameter sampling.** Early in a run, fitting the GP
uses multiple hyperparameter samples to represent uncertainty. Once their
contribution to ELBO uncertainty is sufficiently small under the history-based
criterion, VBMC is supposed to switch to its configured stable GP fit. By
default this uses optimized hyperparameters without drawing further samples,
avoiding the continuing sampling cost. The criterion had no reliable history of
the number of distinct logged training locations and consulted the wrong state
path, so the optional transition
could not activate as intended. The repair records that count as `N`, restores
the `stop_sampling` transition, and backfills compatible older histories. Missing
counts or variances conservatively leave sampling enabled. A forced integration
case demonstrated the real loop transition; the default five-case replay did not
cross the threshold and therefore does not claim a default speedup.

All three repairs passed their focused checks and the 11 exact fixtures. The GP
covariance change alone moved Cigar D4 seed 0 beyond its fixed accuracy fences,
although the result remained inside the broader usability thresholds; a second
seed passed. After acquisition regularization was added, the cumulative seed-0
result returned inside its fences. The isolated adverse result remains part of
the evidence to assess, rather than being erased by the later cumulative result.

Activating acquisition regularization also exposed a cross-platform oracle issue.
Below the uncertainty threshold, its exponential penalty can amplify tiny GP
variance differences into much larger *relative* changes in very small
acquisition values. Some values underflow to zero. The comparison now scales its
denominator by a reference-derived local condition factor on affected nonzero
entries, while preserving the checks on underflow zeros. This changed no
solver, stored reference value, base tolerance, or gpyreg pin. A separate Python
3.10 mock-target problem was test-only. The final Ubuntu, Windows and macOS
matrix across Python 3.10, 3.11 and 3.12 passed all nine jobs.

Detailed main-loop evidence:
[repairs and trajectory checks](results/2026-09-08-main-loop-fixes.md) and
[cross-platform acquisition-oracle
conditioning](results/2026-09-08-acquisition-oracle-conditioning.md).

## Current boundary

The eta and final-boost policies are implemented, the main-loop repairs are
integrated, and the supported CI matrix is green. The optional gpyreg step-out
repair is deferred to issue #44; current PyVBMC GP training does not enable that
path, so it is not a release prerequisite and the dependency pin stays unchanged.

The integrated population assessment starts with an
[overnight first stage](plans/final-population-benchmark.md): 273 candidate
runs compared with the existing 870-run reference. Assess quality, usability
and boost acceptance/rejection tradeoffs, then decide whether further sampling
is useful. A full 870-run candidate population is an optional extension.

S-VBMC compatibility and future algorithm delivery are discussed separately in
the [ecosystem proposal](2026-09-08-ecosystem-integration.md). The completed
[Stage 4 PyTorch feasibility prototype](plans/stage4-torch-feasibility.md)
led to the 2026-09-09 PI decision to retain the modernized NumPy/SciPy solver
for 1.5 and not undertake a full Torch solver port for this release.
