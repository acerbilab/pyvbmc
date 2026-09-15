# The S-VBMC headline on noisy stacks: what the benchmark showed and the way forward

*Written 15 September 2026 from the day's results and a discussion with
the PI. The evidence is in
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
M = 2, 3, 4, 5, 8 and 16 runs were drawn at random, 20 subsets per M
(10 at M = 16), and each subset was stacked by both implementations.
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

**The raw value** is optimistic on every noisy target: by 0.25 to 0.9
nats at M = 2 and by 0.6 to 1.3 at M = 16. Most of it is inherited:
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
inputs the cap over-corrects everywhere: the capped headline sits
0.04 to 0.33 nats below the bias level of the runs it was built from
on the typical targets and 0.8 to 1.5 nats below on Student, more
than the stacking added.

**Scoring each component with the other runs' GPs**, the replacement
that had been planned, is not the answer. It is unbiased on the two
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
shrink" lowers the worst case of the best single estimate by a fifth to
a third, with the same worst case for any threshold from 0.08 to 0.20;
its gain is statistically resolvable only on the multisensory target at
noise 3.

**The yardstick.** The PI restated the requirement after these
results (2026-09-15): S-VBMC is not asked to remove the optimism VBMC
builds into each run's ELBO, only not to add to it, so a headline is
judged by its bias relative to the mean bias of its input runs, each
run scored against its own Monte Carlo ELBO. Under that yardstick the
raw value adds +0.02 to +0.60 nats, growing with M; the cap removes
more than the stacking added, on every noisy target; the run-level
term alone removes a sixth to a half of the addition; and the
two-level shrinkage adds nothing within 0.23 nats at every M on every
noisy target, sitting 0.1 to 0.2 nats below the inputs' level at
M ≤ 5 because its within-run term also removes part of the runs' own
optimism. The report's section "The inputs' own bias" has the
tables. Debiasing the VBMC ELBO itself is a separate question,
recorded in the TODO as outside 1.5.

Worst and mean, over the six noisy targets, of the median absolute
bias against the truth (the earlier yardstick). The two noiseless
targets are within 0.09 nats for every
estimate. The six targets are not exchangeable: the worst is the
multisensory target at noise 3 for the shrinkage estimates and Student
for the cap, and the mean weights the targets equally.

| headline | worst, M = 3 / 5 / 16 | mean, M = 3 / 5 / 16 |
|---|---|---|
| raw | 0.84 / 1.04 / 1.31 | 0.39 / 0.50 / 0.74 |
| capped median (the headline today) | 0.97 / 1.26 / 1.78 | 0.29 / 0.37 / 0.45 |
| within-run shrinkage | 0.60 / 0.75 / 0.94 | 0.25 / 0.31 / 0.40 |
| two-level shrinkage | 0.58 / 0.67 / 0.77 | 0.24 / 0.26 / 0.28 |
| cap if noise share ≥ 0.2, else within-run shrinkage | 0.42 / 0.43 / 0.44 | 0.19 / 0.22 / 0.15 |

The residual on the multisensory targets is the runs' own: a single
multisensory run at noise 3 is optimistic by 0.74 nats, and shrinking
toward a population mean removes selection within the population, not
an error common to all of a run's components. On those targets the
GP's own uncertainty is also miscalibrated (a quarter to two thirds
of the components have |z| > 2 against the truth), and the shrinkage
takes it at face value.

## Decision

The PI endorsed the recommendation on 2026-09-15: the two-level
shrinkage is the headline candidate for a noisy stack. The raw value
stays the headline of a noiseless stack, where shrinkage changes it by
at most 0.03 nats. Under the inputs yardstick, set later the same
day, the endorsement stands: it is the one estimate that neither adds
to the inputs' optimism nor removes much of it. The estimator the
yardstick itself suggests, the same shrinkage with what it removes at
each run's own weights added back, so that a stack of one run
reports the run's own ELBO and only the stacking's addition is
removed, is untested and listed under Open.

Against the alternatives: the cap fails silently and without bound on
heavy tails. The hybrid has the best numbers, but its threshold was
chosen on the benchmark it is scored on, in a gap between six targets
and two, and it switches between estimates that differ by up to 0.5
nats; it is the candidate to revisit with larger stacks (M = 32 is
planned) and further targets.

The class keeps the raw value and the cap in `elbo_details` and adds
the noise share as a diagnostic. The documentation must state that the
headline can remain optimistic (in our benchmarks, by about 0.8 nats on a
high-noise real-data target and pessimistic by up to 0.4 nats on a
heavy-tailed one), and that the raw value is not an upper bound on the truth.

Whether the switch ships in 1.5, or 1.5 keeps the cap with that caveat
and switches later, is the next decision. The change is contained: a
function in `pyvbmc/svbmc/svbmc.py` that computes the shrunken expected
log joint from the stored `I_sk`, `J_sjk` and the runs' own weights at
the selected weights (as the cap is applied today, so the posterior
does not move), a new `elbo_details` key and headline method, the tests
that pin the capped headline, and the user documentation.
`dev/scripts/svbmc_shrink_elbo.py` is the reference implementation.

## Open

- The anchored variant of the shrinkage (add back, per run and
  weighted by its mass in the stack, what the shrinkage removes at
  the run's own weights) removes exactly the stacking's addition by
  construction and is untested; it reads the same statistics and
  scores on the same subsets in minutes.
- Re-optimizing the weights on the shrunken estimates is untested; only
  the reported value is shrunk today.
- Stacks of 32 runs (a later overnight or cluster run) test the
  run-level term where it matters most.
- The subsets of one target and one M share runs (ten subsets of 16
  draw 160 slots from 100 runs), so intervals over subsets are
  optimistic, and the benchmark is one pool of runs at one seed.
- The multisensory residual is an error common to a run's estimates;
  the GP-robustness item in the TODO (far-tail evaluations, the
  quadratic mean on heavy tails) is where its cause may show up.
