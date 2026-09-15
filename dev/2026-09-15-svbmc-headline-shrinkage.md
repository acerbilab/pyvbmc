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
the reported number.** Same weights, same posterior quality, two to four
times faster, tables that rebuild from the recorded cells. Criterion 3,
on evidence accuracy, holds on seven conditions and fails on the
eight-dimensional Student target at noise 3: the component-median cap,
which both implementations apply, puts the headline 0.9 to 1.8 nats
*below* the reference, deepening with the number of stacked runs, while
the raw value is within 0.4.

**The cross-run estimator is not the answer.** It is accurate on the
noiseless controls and on Rosenbrock, under-predicts by 0.2 to 0.7 nats
on the typical noisy conditions (the pilot's finding at scale), and lands
as low as the cap on Student. One mechanism fits both failures on that
target: the GPs' negative-quadratic mean falls off faster than a
Student-t log density, so any estimate that leans on GP predictions away
from a run's own data, in the tails, is dragged down.

**Where the optimism lives.** Capping the stack at its best input run's
own expected log joint changes the raw value by at most 0.1 nats: on most
cells the stacked value sits *below* the best run's own value. The 0.3 to
1.3 nats of optimism are inside each run's own estimate, placed there by
VBMC's own variational optimization, which weights components where its
GP estimate came out high; stacking adds little cross-run selection on
top. The cap works, where it works, because the low-weight components it
includes are the ones that optimization did not favour, a control group
for the winner's curse of the selected ones; the weight-aware variants
(`κ`) showed that removing them removes the debiasing. On Student the
control group is worse than the selected components for a real reason,
the tails, and the cap over-corrects.

**Empirical-Bayes shrinkage is the first correction acceptable
everywhere.** Each component's estimate is shrunk toward its population
mean by the share of the population's spread that the GP itself
attributes to estimation noise, which the class already carries in
`J_sjk`; within a run with the full estimation covariance, and, at the
run level, each run's own value shrunk toward the runs' mean by its
run-level estimation variance. The two-level form matches the cap on
Rosenbrock, noisy GMM and the ring, stays within 0.4 nats on Student,
leaves the noiseless controls alone, and has no tuned constant. At the
`M` users run, three to five, the run-level term is negligible and the
within-run shrinkage does the work; the integrated arm was run at
`M = 3` and `5` to make sure.

**A diagnostic-based hybrid does better still, on this data.** The share
of the components' spread that the GP attributes to noise separates the
targets the cap is right on (0.23 to 1.5) from the one it over-corrects
(0.10) and the noiseless controls (0.01 to 0.10). The rule "cap when the
share is at least 0.2, within-run shrinkage otherwise" halves the
worst case of the best single rule, with the same numbers for any
threshold from 0.15 to 0.30.

## The candidates, side by side

Worst and mean median bias against the reference over the six noisy
conditions (the two noiseless controls are within 0.09 for every rule):

| headline | worst, M = 3 / 5 / 16 | mean, M = 3 / 5 / 16 |
|---|---|---|
| raw | 0.84 / 1.04 / 1.31 | 0.39 / 0.50 / 0.74 |
| capped median (today's headline) | 0.97 / 1.26 / 1.78 | 0.29 / 0.37 / 0.45 |
| within-run shrinkage | 0.59 / 0.74 / 0.92 | 0.24 / 0.31 / 0.39 |
| two-level shrinkage | 0.57 / 0.66 / 0.75 | 0.23 / 0.26 / 0.28 |
| cap if noise share ≥ 0.2, else within-run shrinkage | 0.42 / 0.43 / 0.44 | 0.18 / 0.20 / 0.15 |

Condition by condition, the cap remains 0.1 to 0.3 nats closer than the
shrinkage on the two multisensory conditions and on Rosenbrock; that
residual is the part of the optimism that the GP's own covariance does
not account for, a mean bias of the estimates, which no method keyed on
`J` can see.

## Decision

The PI endorsed the recommendation (2026-09-15): the two-level
empirical-Bayes shrinkage is the headline candidate for a noisy stack,
with the raw value staying the headline of a noiseless one (the
shrinkage leaves it unchanged there). The reasons, against the
alternatives: the cap's failure is unbounded and silent on a heavy-tailed
target; a switch by `M` between the two shrinkage forms exists only out
of distrust for the run-level estimate at small `M`, and the two-level
form is as good or better at every `M` with one formula; the hybrid has
the best numbers but its threshold sits in a gap between six conditions
and two, chosen on the data it is scored on, and it switches
discontinuously between estimates that differ by up to 0.3 nats, so it
is the candidate to revisit when the `M = 32` cells and further targets
are in, not the one to ship on eight conditions.

Alongside the headline: the raw value and the cap stay in
`elbo_details`, the noise share joins them as the diagnostic that says
whether the cap would have been safe, and the documentation states the
headline-to-raw interval as the honest range and that the headline can
remain optimistic by up to about 0.7 nats on a high-noise real-data
target.

Whether the switch ships in 1.5 or the release keeps the cap with the
documented caveat and switches after is the next decision. The change
is contained: a function in `pyvbmc/svbmc/svbmc.py` that computes the
shrunken expected log joint from `I_sk`, `J_sjk` and the runs' own
weights at the selected weights, as the cap is applied today, so the
posterior does not move; a new `elbo_details` key and headline method;
the reporting-plan tests that pin the capped headline moved; the
user-facing documentation of the headline. `dev/scripts/svbmc_shrink_elbo.py`
is the reference implementation of the estimate.

## Open

- Re-optimizing the weights on the shrunken estimates, rather than
  holding them at the values selected on the raw ones, is untested; it
  would also move the posterior toward the population mean.
- The `M = 32` cells of the integrated arm (a later overnight or cluster
  run) test the run-level term where it matters most.
- The residual on the multisensory conditions is a mean bias of the
  runs' GP estimates; the GP-robustness item in the TODO (far-tail
  evaluations, the quadratic mean on heavy tails) is the place where its
  cause may show up.
