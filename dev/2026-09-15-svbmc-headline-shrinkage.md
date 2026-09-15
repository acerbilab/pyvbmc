# The S-VBMC headline on noisy stacks: what the pools showed and the way forward

*Written 15 September 2026 from the day's results and a discussion with
the PI. The evidence is in
[results/2026-09-15-svbmc-pool-comparison.md](results/2026-09-15-svbmc-pool-comparison.md);
the execution record is the worklog of
[plans/svbmc-benchmark-campaign.md](plans/svbmc-benchmark-campaign.md);
the tracked outputs are under
[experiments/svbmc_pool/](experiments/svbmc_pool/README.md).*

S-VBMC stacks several finished VBMC posteriors and reports the ELBO of
the stack: the weighted sum of the components' expected log joints, each
estimated by its run's GP, plus the entropy of the mixture. On a noisy
target that number is too high. The
[optimism note](2026-09-12-svbmc-elbo-optimism.md) planned two remedies.
The first is the component-median cap, which the integrated class
reports as its headline today. The second is a cross-run estimate that
scores each component with the GPs of the other runs. The run-pool
campaign measured both on eight conditions, with 100 filtered runs per
noisy condition and 50 per noiseless control. The reference is the
stacked posterior's own Monte Carlo ELBO, which no estimator sees. This
note gives the reading. The report has every table.

## What the pools showed

**The comparison against the original implementation passes, except on
the reported number.** The two implementations reach the same weights
and the same posterior quality. The integrated class is 1.9 to 4 times
faster. The tables rebuild from the recorded cells. Criterion 3, on the
accuracy of the reported ELBO, holds on seven conditions and fails on
one. On the eight-dimensional Student target at noise 3, the cap puts
the headline 0.9 to 1.8 nats below the reference, and the gap widens
with the number of stacked runs. The raw value is within 0.4 nats
there. Both implementations apply the same cap, so this is a property
of the cap, not of the port.

**The cross-run estimator is not the answer.** It is accurate on the
noiseless controls (within 0.035 nats) and on Rosenbrock (within 0.07,
although nine Rosenbrock runs fail its own-run check because their GPs
are numerically fragile). On the typical noisy conditions it is 0.2 to
0.7 nats too low, which confirms the pilot's finding. On Student it is
as low as the cap. The cap's failure on Student has a simple cause: the
components of a heavy-tailed posterior really do differ in expected log
joint, and the median component sits in a tail. The cross-run
estimator's failure there has a plausible cause that the data do not
prove: the GPs' quadratic mean falls off faster than a Student-t log
density, so a run's predictions away from its own data are too low.

**Where the optimism lives.** Most of it is inside each run's own
estimate before any stacking. VBMC's own variational optimization
weights the components where its GP estimate came out high, and that
selection makes each run's expected log joint optimistic. Stacking adds
a smaller cross-run part that grows with M: the raw bias rises by about
0.3 to 0.5 nats from M = 2 to M = 16 on every noisy condition except
Student. Capping the stack at its best input run changes nothing (at
most 0.06 nats), but that benchmark is itself the largest of M noisy
values, so it only shows that mixing runs does not create the optimism.

**Why the cap works where it works.** The cap replaces the stack's
expected log joint by the median over all components, including the
low-weight ones. Those are the components the optimization did not
favour, so they act as a control group for the winner's curse of the
selected ones. Variants of the cap that drop the low-weight components
(the κ experiment) lose the debiasing. On Student the control group is
worse than the selected components for a real reason, the tails, and
the cap over-corrects.

**Empirical-Bayes shrinkage is the first correction that is acceptable
everywhere.** Each component's estimate is shrunk toward the mean of its
run's components. The amount is the share of the spread that the GP
itself attributes to estimation noise, which the class already holds in
`J_sjk`. The full within-run covariance is used, because one GP
estimates all of a run's components and their errors are correlated. A
second level shrinks each run's own value toward the mean over runs,
using the run-level estimation variance. This two-level form matches
the cap on Rosenbrock, noisy GMM and the ring: the differences are
inside the paired bootstrap intervals at M = 3 and 5. It stays within
0.4 nats on Student. It moves the noiseless controls by at most 0.03
nats. It has no tuned constant. Its costs are also clear. On Student it
is worse than the raw value at M ≤ 5: 0.24 to 0.39 nats too low against
0.04 to 0.11 for raw, on 65 to 85 percent of the cells. On the two
multisensory conditions the cap stays closer by 0.09 to 0.32 nats, and
the bootstrap resolves that difference. At the M users run, three to
five, the run-level term matters little at M ≤ 3 (the two forms differ
by at most 0.08 nats) and adds 0.1 to 0.14 nats at M = 4 and 5. The
integrated arm was run at M = 3 and 5 to check this.

**A diagnostic-based hybrid does better still on this data.** The noise
share separates the conditions in the median over cells: 0.21 to 1.1
where the cap is right, 0.05 on Student, 0.005 and 0.085 on the
noiseless controls. Cell by cell the ranges overlap: Rosenbrock's cells
span 0.02 to 1.3, and the noiseless multisensory control's reach 0.24.
The rule "apply the cap when the share is at least 0.2, otherwise
shrink" lowers the worst case of the best single rule by a fifth to a
third (0.38 to 0.56 nats against 0.52 to 0.77). Any threshold from
0.08 to 0.20 gives the same worst case, and the mean moves by at most
0.03 over that range. Its gain over the two-level shrinkage is
resolvable on multisensory noise 3 and nowhere else.

## The candidates, side by side

Worst and mean median bias against the reference over the six noisy
conditions. The two noiseless controls are within 0.09 nats for every
rule. The six conditions are not exchangeable: "worst" is multisensory
noise 3 for the shrinkage rules and Student for the cap, and the mean
weights the conditions equally for convenience. The report gives the
per-condition tables and the paired bootstrap intervals.

| headline | worst, M = 3 / 5 / 16 | mean, M = 3 / 5 / 16 |
|---|---|---|
| raw | 0.84 / 1.04 / 1.31 | 0.39 / 0.50 / 0.74 |
| capped median (today's headline) | 0.97 / 1.26 / 1.78 | 0.29 / 0.37 / 0.45 |
| within-run shrinkage, full covariance | 0.60 / 0.75 / 0.94 | 0.25 / 0.31 / 0.40 |
| two-level shrinkage | 0.58 / 0.67 / 0.77 | 0.24 / 0.26 / 0.28 |
| cap if noise share ≥ 0.2, else within-run shrinkage | 0.42 / 0.43 / 0.44 | 0.19 / 0.22 / 0.15 |

The shrinkage cannot remove what remains on the multisensory
conditions. Shrinking toward a population mean only removes selection
within that population. An error shared by all of a run's components,
or by all runs, stays. The Phase 2 calibration records show that on
those conditions the GP's own uncertainty is also too small: a quarter
to two thirds of the components have |z| above 2. The shrinkage takes
that uncertainty at face value.

## Decision

The PI endorsed the recommendation on 2026-09-15: the two-level
empirical-Bayes shrinkage is the headline candidate for a noisy stack.
The raw value stays the headline for a noiseless stack, where the
shrinkage would change it by at most 0.03 nats.

The reasons, against the alternatives. The cap fails silently and
without bound on a heavy-tailed target. A switch by M between the two
shrinkage forms would only express distrust of the run-level estimate
at small M, and the two-level form is as good or better at every M with
one formula. The hybrid has the best numbers, but its threshold sits in
a gap between six conditions on one side and two on the other, it was
chosen on the data it is scored on, and it switches between two
estimates that differ by up to 0.5 nats. It is the candidate to revisit
when the M = 32 cells and further targets are in.

Alongside the headline: the raw value and the cap stay in
`elbo_details`. The noise share joins them as a diagnostic, with the
caveat that it separates conditions in the median and not cell by
cell. The documentation says that the headline can still be too high by
about 0.8 nats on a high-noise real-data target and too low by up to
0.4 on a heavy-tailed one. The raw value is not an upper bound on the
truth: the interval from the headline to the raw value contains the
reference on none of the multisensory cells.

Whether the switch ships in 1.5, or the release keeps the cap with the
documented caveat and switches later, is the next decision. The change
is contained. A function in `pyvbmc/svbmc/svbmc.py` computes the
shrunken expected log joint from `I_sk`, `J_sjk` and the runs' own
weights at the selected weights, as the cap is applied today, so the
posterior does not move. It needs a new `elbo_details` key and headline
method, updates to the reporting-plan tests that pin the capped
headline, and the user documentation of the headline.
`dev/scripts/svbmc_shrink_elbo.py` is the reference implementation.

## Open

- Re-optimizing the weights on the shrunken estimates is untested. The
  weights are now selected on the raw estimates and only the reported
  value is shrunk. Re-optimizing would also move the posterior toward
  the population mean.
- The M = 32 cells of the integrated arm (a later overnight or cluster
  run) test the run-level term where it matters most.
- The cells of one condition and M share runs: ten M = 16 cells draw
  160 run slots from 100. Intervals over cells are therefore
  optimistic, and the whole comparison is one pool at one seed.
- The residual on the multisensory conditions is an error shared by a
  run's estimates, which no covariance-based correction can see. The
  GP-robustness item in the TODO (far-tail evaluations, the quadratic
  mean on heavy tails) is where its cause may show up.
