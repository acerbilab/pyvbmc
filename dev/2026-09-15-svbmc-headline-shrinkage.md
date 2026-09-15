# The S-VBMC headline on noisy stacks: what the pools showed and the way forward

*Written 15 September 2026 from the day's results and a discussion with
the PI. The evidence is in
[results/2026-09-15-svbmc-pool-comparison.md](results/2026-09-15-svbmc-pool-comparison.md);
the execution record is the worklog of
[plans/svbmc-benchmark-campaign.md](plans/svbmc-benchmark-campaign.md);
the tracked outputs are under
[experiments/svbmc_pool/](experiments/svbmc_pool/README.md).*

S-VBMC stacks several finished VBMC posteriors and reports the stacked
ELBO, the weighted sum of the components' expected log joints (each from
its run's GP) plus the entropy of the mixture. On a noisy target that
number is optimistic, and the
[optimism note](2026-09-12-svbmc-elbo-optimism.md) set two phases to deal
with it: the component-median cap the integrated class reports as its
headline today, then a cross-run "honest" estimate from the runs' GPs.
The run-pool campaign was built to measure both. It did, on eight
conditions with 100 filtered runs each (50 for the two noiseless
controls), against a reference no estimator sees: the stacked
posterior's own Monte Carlo ELBO. What follows is the reading; the
report has every table.

## What the pools showed

**The comparison against the original implementation passes, except on
the reported number.** Same weights, same posterior quality, 1.9 to 4
times faster, tables that rebuild from the recorded cells. Criterion 3,
on evidence accuracy, holds on seven conditions and fails on the
eight-dimensional Student target at noise 3: the component-median cap,
which both implementations apply, puts the headline 0.9 to 1.8 nats
*below* the reference, deepening with the number of stacked runs, while
the raw value is within 0.4.

**The cross-run estimator is not the answer.** It is accurate on the
noiseless controls (within 0.035 nats) and on Rosenbrock (within 0.07,
with the caveat that nine Rosenbrock runs fail the estimator's own-run
check, the platform sensitivity of that condition's GPs), under-predicts
by 0.2 to 0.7 nats on the typical noisy conditions (the pilot's finding
at scale), and lands as low as the cap on Student. The cap's failure
there needs no special mechanism: the components of a heavy-tailed
posterior genuinely differ in expected log joint, and the median
component is a tail one. For the cross-run estimator, one reading fits
the numbers without being demonstrated by them: the GPs' quadratic mean
falls off faster than a Student-t log density, so predictions away from
a run's own data are low.

**Where the optimism lives.** Capping the stack at its best input run's
own expected log joint changes the raw value by at most 0.06 nats,
because the stacked value sits below the best run's own value on most
cells. That benchmark is itself the maximum of `M` noisy run-level
values, so the observation shows only that the optimism is not
*created* by mixing runs; most of it is inside each run's own estimate,
placed there by VBMC's own variational optimization, which weights
components where its GP estimate came out high. A cross-run part exists
and grows with `M`: the raw bias rises from about 0.3 to 0.7 nats
between `M = 2` and `16` on the synthetic noisy conditions and from 0.8
to 1.3 on multisensory noise 3, and the run-level shrinkage below
removes part of that at `M ≥ 4`. The cap works, where it works, because
the low-weight components it includes are the ones that optimization
did not favour, a control group for the winner's curse of the selected
ones; the weight-aware variants (`κ`) showed that removing them removes
the debiasing. On Student the control group is worse than the selected
components for a real reason, the tails, and the cap over-corrects.

**Empirical-Bayes shrinkage is the first correction acceptable
everywhere.** Each component's estimate is shrunk toward its population
mean by the share of the population's spread that the GP itself
attributes to estimation noise, which the class already carries in
`J_sjk`; within a run with the full estimation covariance, and, at the
run level, each run's own value shrunk toward the runs' mean by its
run-level estimation variance. The two-level form matches the cap on
Rosenbrock, noisy GMM and the ring (differences within the paired
bootstrap intervals over cells at `M = 3` and `5`), stays within 0.4
nats on Student D8, moves the noiseless controls by at most 0.03 nats
(to within 0.09 of the reference), and has no tuned constant. It is not
free: on Student it is worse than the raw value at `M ≤ 5` (0.24 to
0.39 nats low against raw's 0.04 to 0.11, on 65 to 85 % of the cells),
and on the two multisensory conditions the cap stays closer by 0.09 to
0.32 nats, a difference the paired bootstrap resolves. At the `M` users
run, three to five, the run-level term contributes little at `M ≤ 3`
(the two forms differ by at most 0.08 nats) and 0.1 to 0.14 nats at
`M = 4` and `5`; the integrated arm was run at `M = 3` and `5` to make
sure.

**A diagnostic-based hybrid does better still, on this data.** The share
of the components' spread that the GP attributes to noise separates, in
the median over cells, the conditions the cap is right on (0.21 to 1.1)
from the one it over-corrects (0.05) and the noiseless controls (0.005
and 0.085); cell by cell the ranges overlap (Rosenbrock's cells span
0.02 to 1.3, the noiseless multisensory control's reach 0.24). The rule
"cap when the share is at least 0.2, within-run shrinkage otherwise"
lowers the worst case of the best single rule by a fifth to a third
(0.38 to 0.56 nats against 0.52 to 0.77), with the same worst case for
any threshold from 0.08 to 0.20 and a mean that moves by at most 0.03
over that range; its gain over the two-level shrinkage is resolvable on
multisensory noise 3 and nowhere else.

## The candidates, side by side

Worst and mean median bias against the reference over the six noisy
conditions (the two noiseless controls are within 0.09 for every rule).
The six conditions are not exchangeable, so "worst" names different
conditions for different rules (multisensory noise 3 for the shrinkage,
Student D8 for the cap) and the mean is a convenience weighting; the
report gives the per-condition tables and paired bootstrap intervals.

| headline | worst, M = 3 / 5 / 16 | mean, M = 3 / 5 / 16 |
|---|---|---|
| raw | 0.84 / 1.04 / 1.31 | 0.39 / 0.50 / 0.74 |
| capped median (today's headline) | 0.97 / 1.26 / 1.78 | 0.29 / 0.37 / 0.45 |
| within-run shrinkage, full covariance | 0.60 / 0.75 / 0.94 | 0.25 / 0.31 / 0.40 |
| two-level shrinkage | 0.58 / 0.67 / 0.77 | 0.24 / 0.26 / 0.28 |
| cap if noise share ≥ 0.2, else within-run shrinkage | 0.42 / 0.43 / 0.44 | 0.19 / 0.22 / 0.15 |

What the shrinkage leaves on the multisensory conditions is, by
construction, what it cannot see: shrinking toward a population mean
removes selection on the contrast within the population, and an error
common to all of a run's components, or to all runs, is untouched. The
Phase 2 calibration records on those conditions also show the GP's own
uncertainty to be too small (own `|z| > 2` on a quarter to two thirds
of the components), which the shrinkage takes at face value.

## Decision

The PI endorsed the recommendation (2026-09-15): the two-level
empirical-Bayes shrinkage is the headline candidate for a noisy stack,
with the raw value staying the headline of a noiseless one (the
shrinkage would change it by at most 0.03 nats there). The reasons,
against the alternatives: the cap's failure is unbounded and silent on a
heavy-tailed target; a switch by `M` between the two shrinkage forms
would exist only out of distrust for the run-level estimate at small
`M`, and the two-level form is as good or better at every `M` with one
formula; the hybrid has the best numbers but its threshold sits in a gap
between six conditions and two, chosen on the data it is scored on, and
it switches discontinuously between estimates that differ by up to
0.5 nats, so it is the candidate to revisit when the `M = 32` cells and
further targets are in, not the one to ship on eight conditions.

Alongside the headline: the raw value and the cap stay in
`elbo_details`, the noise share joins them as a diagnostic (it
separates the conditions in the median, not cell by cell), and the
documentation states that the headline can remain optimistic by up to
about 0.8 nats on a high-noise real-data target and pessimistic by up
to 0.4 on a heavy-tailed one; the raw value is not an upper bound on the
truth (the interval from the headline to the raw value contains the
reference on none of the multisensory cells).

Whether the switch ships in 1.5 or the release keeps the cap with the
documented caveat and switches after is the next decision. The change
is contained: a function in `pyvbmc/svbmc/svbmc.py` that computes the
shrunken expected log joint from `I_sk`, `J_sjk` and the runs' own
weights at the selected weights, as the cap is applied today, so the
posterior does not move; a new `elbo_details` key and headline method;
the reporting-plan tests that pin the capped headline moved; the
user-facing documentation of the headline. `dev/scripts/svbmc_shrink_elbo.py`
is the reference implementation of the estimate, its `τ²` estimated
with the noise term of correlated estimates.

## Open

- Re-optimizing the weights on the shrunken estimates, rather than
  holding them at the values selected on the raw ones, is untested; it
  would also move the posterior toward the population mean.
- The `M = 32` cells of the integrated arm (a later overnight or cluster
  run) test the run-level term where it matters most.
- The cells of one condition and `M` share runs (ten `M = 16` cells draw
  160 run slots from 100), so intervals over cells are optimistic, and
  the whole comparison is one pool at one seed.
- The residual on the multisensory conditions is a common error of the
  runs' GP estimates that no covariance-based correction can see; the
  GP-robustness item in the TODO (far-tail evaluations, the quadratic
  mean on heavy tails) is the place where its cause may show up.
