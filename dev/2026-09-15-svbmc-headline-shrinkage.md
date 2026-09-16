# The S-VBMC headline on noisy stacks: what the benchmark showed and the way forward

*Written 15 September 2026. The evidence is in
[results/2026-09-15-svbmc-pool-comparison.md](results/2026-09-15-svbmc-pool-comparison.md);
the execution record is the worklog of
[plans/svbmc-benchmark-campaign.md](plans/svbmc-benchmark-campaign.md);
the tracked outputs are under
[experiments/svbmc_pool/](experiments/svbmc_pool/README.md); the
shrinkage itself is explained step by step, with a worked example, in
[2026-09-15-svbmc-shrinkage-explained.md](2026-09-15-svbmc-shrinkage-explained.md).*

## The problem

S-VBMC takes M finished VBMC posteriors of the same target, each a
mixture of Gaussian components, and stacks them into one mixture by
optimizing the weights of all the components together. It estimates the
ELBO of the stack as the weighted sum of the components' expected log
joints, each taken from its own run's GP, plus the entropy of the
mixture. On a noisy target the estimate is optimistic (biased): the weights are
chosen to maximize a sum of noisy GP estimates, so the selected components are
high partly by noise, an example of "winner's curse".

PyVBMC 1.5 ships S-VBMC as `pyvbmc.svbmc.SVBMC`, ported from the
standalone `svbmc` package (version 0.1.1). The standalone package
returns three estimates side by side, the raw value and two debiased
variants, and leaves the choice to the user. For our port we decided to return
one *headline* `elbo`: the raw value for a noiseless stack, and for a noisy
stack the component-median cap, the first of the standalone package's
variants. This note answers two questions about the port.

1. Does it match the standalone package?
2. Which single number should it report on a noisy stack?

## The benchmark

For each of eight targets, 100 independent VBMC runs were generated (50
for the two noiseless targets), each saved with its posterior and the
GP behind it. Six targets are noisy (a Gaussian noise of standard
deviation 3 on the log density, 1.3 in one case): a multisensory model
on real data in 6 dimensions at both noise levels, and Rosenbrock, a
Gaussian mixture and a ring in 2 dimensions and a Student-t product in
8 dimensions at noise 3. Two are noiseless controls: the Gaussian
mixture and the multisensory model. From each target's runs, subsets of
M = 2, 3, 4, 5, 8, 16 and 32 runs were drawn at random, 20 subsets per
M (10 at M = 16 and 32); the subsets of 2, 4, 8 and 16 runs were
stacked by both implementations, those of 3, 5 and 32 by the port
alone.
Every reported ELBO is scored by its bias against the truth for that
stack: the stack's own ELBO, computed by Monte Carlo from the true log
density. All numbers below are medians of that bias over the subsets
of one target and one M.

## Does the port match the standalone package?

Yes. On every target the two implementations reach the same weights and
the same posterior quality, **the port is 1.9 to 4 times faster**, and
every table rebuilds from the recorded results. Their capped estimates
agree within 0.25 nats.

## Which number should the port report?

**The raw value** is optimistic on the five typical noisy targets: by
0.25 to 0.77 nats at M = 2 and by 0.6 to 1.3 at M = 16 (on the
heavy-tailed Student target it is 0.11 low at M = 2 and 0.27 high at
M = 16). Most of it is inherited:
a single VBMC run, scored against its own Monte Carlo ELBO, is
already optimistic by 0.13 to 0.74 nats on the typical noisy targets
(0.74 on the multisensory target at noise 3), because its variational
optimization selected components on noisy GP estimates. Stacking adds
0.02 to 0.15 nats at M = 2 and 0.27 to 0.60 at M = 16, the optimism
of picking the best of M noisy runs: the stack's raw value tracks the
bias of its highest-reported input within 0.25 nats.

**The cap** replaces the stack's expected log joint by the median of the
expected log joints over all components, weighted or not. It brings the
headline within 0.56 nats of the truth on five of the six noisy targets
at every M. On the 8-dimensional Student target it is 0.9 to 1.8 nats
pessimistic, worse than the raw value, and the error grows with M. The
reason is plain: on a heavy-tailed posterior the median component is a
tail component. Where the cap works, it works because the components
with small weight are the ones the optimization did not select, so
they act as an unselected control group. Variants of the cap that leave
out the small-weight components lose the debiasing. Relative to its
inputs the cap over-corrects on five of the six noisy targets: the
capped headline sits up to 0.38 nats below the bias level of the runs
it was built from on four typical targets, at that level on
Rosenbrock, and 0.8 to 1.5 nats below on Student, more than the
stacking added.

**Scoring each component with the other runs' GPs**, the replacement
the [ELBO optimism note](2026-09-12-svbmc-elbo-optimism.md) had
planned as its Phase 2, is not the answer. It is unbiased on the two
noiseless targets (within 0.035 nats) and on Rosenbrock (within 0.07),
biased low by 0.2 to 0.7 nats on the other noisy targets, and as low as
the cap on Student.

**Empirical-Bayes shrinkage** is the first estimate that is acceptable
on every target. Each component's estimate is treated as a noisy
measurement, `I_k ~ N(θ_k, Σ)`, of a true value drawn from the run's
population, `θ_k ~ N(μ, τ²)`, and replaced by its posterior mean. `Σ` is
the run's estimation covariance, computed from two statistics VBMC
saves with every posterior and the class already requires of every
input: `J_sjk`, the covariance of the GP quadrature over the
components, averaged over the hyperparameter samples, plus the
covariance of `I_sk` across those samples. No GP object is needed and
the class's inputs do not change. `μ` and `τ²` are estimated by
moments within the run; the full `Σ` is used because one GP estimated
all of a run's components and their errors are correlated. A second
level treats each run's own
expected log joint the same way, shrinking it toward the mean over
runs by its run-level estimation variance. This two-level estimate
matches the cap on Rosenbrock, the Gaussian mixture and the ring
(differences inside paired bootstrap intervals over subsets at M = 3
and 5), is within 0.4 nats on Student, moves the noiseless targets by
at most 0.03 nats, and has no tuned constant. Its costs: on Student it
is 0.24 to 0.39 nats pessimistic at M ≤ 5, where the raw value is
within 0.11; on the two multisensory targets the cap is closer by 0.09
to 0.32 nats. Users stack three to five runs, which is why M = 3 and 5
are in the grid; the ordering of the estimates is the same there.

**A hybrid** does better still on this benchmark. Call the noise share
the fraction of the spread of a run's component estimates that its GP
attributes to estimation noise. In the median over subsets it separates
the targets: 0.21 to 1.1 where the cap is right, 0.05 on Student, at
most 0.085 on the noiseless targets; subset by subset the ranges
overlap. The rule "cap when the share is at least 0.2, otherwise
shrink" lowers the worst case of the best single estimate by a fifth
to two fifths, with the same worst case for any threshold from 0.08 to
0.20. Its gain is statistically resolvable on the multisensory target
at both noise levels, on Rosenbrock and the Gaussian mixture at M = 16
and on Student at M ≤ 5; on the ring at M = 5 and 8 it is resolvably
worse than the two-level shrinkage.

**The yardstick.** The PI restated the requirement after these
results (2026-09-15): S-VBMC is not asked to remove the optimism VBMC
builds into each run's ELBO, only not to add to it, so a headline is
judged by its bias relative to the mean bias of its input runs, each
run scored against its own Monte Carlo ELBO. Under that yardstick the
raw value adds +0.02 to +0.60 nats, growing with M; the cap removes
more than the stacking added on five of the six noisy targets and
lands at the inputs' level on Rosenbrock; the run-level term alone
removes a sixth to a half of the addition at M ≥ 3; and the two-level
shrinkage adds nothing within 0.23 nats at every M on every noisy
target, sitting up to 0.23 nats below the inputs' level at M ≤ 5
because its within-run term also removes part of the runs' own
optimism. Anchored variants that add back what the shrinkage removes
at each run's own weights, so that only the stacking's selection is
removed, leave −0.02 to +0.21 nats of the addition at M = 3 to 5. The
baseline is the plain mean over the inputs; the mass-weighted mean is
not neutral, since the mass already concentrates on the runs that
came out highest, and against it every estimate sits lower. The
report's section "The inputs' own bias, and what stacking adds" has
the tables and the sensitivity. Debiasing the VBMC ELBO itself is a
separate question, recorded in the TODO as outside 1.5.

Worst and mean, over the six noisy targets, of the median absolute
bias against the truth, the stack's own Monte Carlo ELBO. The two
noiseless targets are within 0.1 nats for every estimate. The six targets are not exchangeable: the worst is the
multisensory target at noise 3 for the shrinkage estimates and Student
for the cap, and the mean weights the targets equally.

| headline | worst, M = 3 / 5 / 16 | mean, M = 3 / 5 / 16 |
|---|---|---|
| raw | 0.84 / 1.04 / 1.31 | 0.39 / 0.50 / 0.74 |
| capped median (the headline today) | 0.97 / 1.26 / 1.78 | 0.29 / 0.37 / 0.45 |
| within-run shrinkage (full covariance) | 0.60 / 0.75 / 0.94 | 0.25 / 0.31 / 0.40 |
| two-level shrinkage | 0.58 / 0.67 / 0.77 | 0.24 / 0.26 / 0.28 |
| cap if noise share ≥ 0.2, else within-run shrinkage | 0.42 / 0.43 / 0.44 | 0.19 / 0.22 / 0.15 |

The residual on the multisensory targets is the runs' own: a single
multisensory run at noise 3 is optimistic by 0.74 nats, and shrinking
toward a population mean removes selection within the population, not
an error common to all of a run's components. On those targets the
GP's own uncertainty is also miscalibrated (a quarter to two thirds
of the components, weighted by mass, have |z| > 2 against the truth:
the own-GP calibration lines of
[experiments/svbmc_pool/phase2_20260915/summary.md](experiments/svbmc_pool/phase2_20260915/summary.md)),
and the shrinkage takes it at face value.

## Decision

The PI endorsed the recommendation on 2026-09-15: the two-level
shrinkage is the headline candidate for a noisy stack. The raw value
stays the headline of a noiseless stack, where shrinkage changes it by
at most 0.03 nats. Under the inputs yardstick the endorsement stands:
it is the one estimate that neither adds to the inputs' optimism nor
removes much of it, and the anchored variants leave part of the
stacking's addition, so the two-level shrinkage remains the
candidate, with the caveat that part of what it removes is the runs'
own optimism.

Against the alternatives: the cap fails silently and without bound on
heavy tails. The hybrid has the best numbers, but its threshold was
chosen on the benchmark it is scored on, in a gap between six targets
and two, and it switches between estimates that differ by up to 0.5
nats; it is the candidate to revisit with further targets. At M = 32
it keeps its numbers (a worst case of 0.45 nats, the cap chosen on 90
to 100 % of the typical noisy cells; report, section "The integrated
arm at M = 32").

The shrinkage applies to the reported value at the weights the raw
optimization chose, as the cap does today. Re-optimizing the weights
on the shrunken values was tested at M = 3 to 5 and rejected: the
optimizer then selects on the shrinkage's own errors, the headline
comes out 0.05 to 0.40 nats more optimistic than the value-only
shrinkage on four of the six noisy targets, and the posterior
improves on the multisensory and ring targets, worsens on Rosenbrock
and, in the KL gap, on the Gaussian mixture and Student (report,
section "Re-optimizing on the shrunken estimates").

The class keeps the raw value and the cap in `elbo_details` and adds
the noise share as a diagnostic. The documentation must state the two
contributions to the bias of a reported value separately. The first is
VBMC's own: what the variational optimization builds into each run's
ELBO, which S-VBMC inherits and is not asked to remove (in our
benchmarks 0.1 to 0.7 nats of optimism on the typical noisy targets,
0.2 nats of pessimism on the heavy-tailed one, within 0.03 on the
noiseless ones; a separate item). The second is S-VBMC's: what
stacking adds on top of its inputs (the raw value +0.07 to +0.36 nats
at M = 3 to 5 and +0.36 to +0.79 at M = 32 on the noisy targets,
growing with M; the two-level shrinkage within 0.23 nats at every M;
the cap nothing added but up to 0.4 nats of the inputs' own optimism
removed on the typical noisy targets and 0.8 to 1.6 on the
heavy-tailed one; on noiseless targets up to about 0.1 nats at
M = 32 and within 0.05 through M = 16). And that the raw value is not
an upper bound on the truth.

The PI's position on 2026-09-16: the two-level shrinkage is the candidate
for the headline of a noisy stack; whether a noiseless stack
also reports it or keeps the raw value is undecided (through M = 16
the two differ by at most 0.03 nats on the controls, and the M = 32
control cannot resolve them). The switch is not made on this
benchmark alone. It is confirmed at the final large-scale check before
the release of PyVBMC 1.5, run on the cluster once 1.5 is consolidated:
fresh VBMC pools on the test targets, about 100 runs each, generated by
the release code, then S-VBMC at several M, and the results of this
note must hold. That check is the release gate for 1.5 as a whole, not
for S-VBMC only. The present pools hold 100 runs per noisy target and
50 per noiseless one, so the subsets at M = 16 reuse each run 1.6 to
3.2 times and those at M = 32 3.2 to 6.4 times; a pool that is to
support M = 32 needs several hundred runs per target. User-facing
recommendations for the headline follow the confirmation.

The implementation decision, also 2026-09-16, separates availability from
headline selection: add the full-covariance two-level estimate as
`elbo_details["shrunk_two_level"]` for both noisy and noiseless stacks,
alongside `shrinkage_noise_share`, before the campaign. Evaluate it at the
existing selected weights with the same final entropy estimate. The
campaign can then compare the packaged estimate directly; promote it to
the headline afterwards only if it is confirmed to be the best estimator.
The current headline, optimization and raw-estimate uncertainty remain
unchanged. `VBMC.optimize()` returns the same posterior and results objects,
and `SVBMC(vp_list)` keeps its input requirements: the existing `I_sk`,
`J_sjk`, component weights and transformers suffice. No GP, full VBMC
object, new posterior statistic or save-format change is needed.

The [integration plan](plans/svbmc-shrinkage-estimator.md) owns the package
implementation, reporting, campaign recording and verification.
`dev/scripts/svbmc_shrink_elbo.py` remains the scientific reference.

## Open

- The run-level term is too weak on its own: it takes the GP's
  variance of a run's ELBO at fixed weights (a standard deviation of
  0.2 to 0.35 nats) as the level's error, while the runs' own
  optimism scatters by a comparable amount on top of it
  (interquartile ranges of 0.3 to 0.5 nats across the runs of a noisy
  target), which the model reads as real spread between runs and the
  stacking selects on. A run-level error term that includes that
  scatter is the untested refinement.
- Stacks of 32 runs (the port alone, 2026-09-15/16) confirm the
  reading: the run-level term removes 0.07 to 0.44 nats of the raw
  bias, 15 to 55 % of what the stacking adds, against 16 to 44 % at
  M = 16; the two-level shrinkage adds −0.15 to +0.19 nats and its
  residual on multisensory noise 3 grows to +0.86, where the cap
  stays at +0.45. The refinement above remains the untested one.
- On the noiseless multisensory control, stacking 32 runs adds 0.11
  nats of optimism to the 0.03 its inputs carry (the raw value +0.14
  against the truth, 0.07 above the best input), on every subset. No
  cap applies to a noiseless stack, and the shrinkage adds 0.14
  rather than 0.11, since the GP attributes only 0.08 to 0.09 of the
  components' spread to noise on that target. The optimism that
  stacking adds is not confined to the targets the class treats as
  noisy; through M = 16 the controls' added bias stays within 0.05.
  The ten subsets of 32 draw 320 slots from that control's 50 runs, so
  the observation is closer to one measurement than to ten: a
  direction to confirm on a larger pool, not an interval.
- The subsets of one target and one M share runs (ten subsets of 16
  draw 160 slots from 100 runs), so intervals over subsets are
  optimistic, and the benchmark is one pool of runs at one seed.
- The multisensory residual is an error common to a run's estimates;
  the GP-robustness item in the TODO (far-tail evaluations, the
  quadratic mean on heavy tails) is where its cause may show up.
