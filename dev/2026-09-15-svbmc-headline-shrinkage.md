# The S-VBMC headline on noisy stacks: what the pools showed and the way forward

*Written 15 September 2026 from the day's results and a discussion with
the PI. The evidence is in
[results/2026-09-15-svbmc-pool-comparison.md](results/2026-09-15-svbmc-pool-comparison.md);
the execution record is the worklog of
[plans/svbmc-benchmark-campaign.md](plans/svbmc-benchmark-campaign.md);
the tracked outputs are under
[experiments/svbmc_pool/](experiments/svbmc_pool/README.md).*

S-VBMC stacks several finished VBMC posteriors into one mixture and
estimates its ELBO: the weighted sum of the components' expected log
joints, each taken from its run's GP, plus the entropy of the mixture.
On a noisy target this estimate is too high, because the weights are
chosen to make it high and the component estimates are noisy.

PyVBMC 1.5 ships S-VBMC as `pyvbmc.svbmc.SVBMC`, ported from the
standalone `svbmc` package (version 0.1.1, the original
implementation). The run-pool campaign answered two questions about
the port:

1. Does the new implementation match the original?
2. Which single number should the new implementation report as the
   ELBO of a noisy stack?

The campaign ran both implementations on the same stacks of the same
runs, on eight conditions with 100 filtered runs per noisy condition and
50 per noiseless control. Every reported ELBO is judged against the
stacked posterior's own Monte Carlo ELBO, computed from the true log
density, which no estimator sees.

## 1. The new implementation matches the original

Yes. On every condition the two implementations reach the same weights
and the same posterior quality, the new one is 1.9 to 4 times faster,
and the tables rebuild from the recorded cells. The report's criteria 1,
2, 4 and 5 hold everywhere. The original's capped estimate and the new
class's capped headline agree within 0.25 nats.

## 2. Which number to report

The original implementation does not choose. It returns three numbers
side by side: the raw estimate, and two debiased variants that cap the
expected log joint at the median over components or over runs. The
user picks. The new class returns one headline: the raw value on a
noiseless stack, and on a noisy stack the component-median cap, the
first of the original's debiased variants. The campaign tested that
choice, and it is wrong on one of the eight conditions.

**The raw value is too high on every noisy condition.** By 0.25 to 0.9
nats at M = 2 (M is the number of stacked runs) and 0.6 to 1.3 nats at
M = 16. Most of this is already inside each run's own estimate before
stacking: VBMC's own optimization weights the components where its GP
estimate came out high. Stacking adds a smaller part that grows with M.

**The cap fixes this on five of the six noisy conditions and breaks
one.** On the typical conditions the capped headline stays within 0.56
nats of the truth at every M. On the eight-dimensional Student target
it is 0.9 to 1.8 nats too low, further from the truth than the raw
value, and the error grows with M. The cause is simple: the cap replaces
the stack's expected log joint by the median over all components, and
on a heavy-tailed posterior the median component sits in a tail. Where
the cap works, it works because the low-weight components are the ones
the optimization did not favour, so they serve as a control group for
the winner's curse of the selected ones. Variants of the cap that drop
the low-weight components lose the debiasing.

**The cross-run estimate planned as the replacement is not the
answer.** It scores each component with the GPs of the other runs. It
is accurate on the noiseless controls (within 0.035 nats) and on
Rosenbrock (within 0.07, though nine Rosenbrock runs fail its own-run
check because their GPs are numerically fragile). On the typical noisy
conditions it is 0.2 to 0.7 nats too low, and on Student it is as low
as the cap.

**Empirical-Bayes shrinkage is the first estimate that is acceptable on
every condition.** Each component's estimate is shrunk toward the mean
of its run's components, by the share of their spread that the GP
itself attributes to estimation noise; the class already holds that
noise in `J_sjk`. The full within-run covariance is used, because one
GP estimates all of a run's components and their errors are
correlated. A second level shrinks each run's own value toward the mean
over runs. This two-level shrinkage matches the cap on Rosenbrock,
noisy GMM and the ring (the differences are inside the paired bootstrap
intervals at M = 3 and 5), stays within 0.4 nats on Student, moves the
noiseless controls by at most 0.03 nats, and has no tuned constant. Its
costs: on Student it is 0.24 to 0.39 nats too low at M ≤ 5, where the
raw value is within 0.11; on the two multisensory conditions the cap
stays closer by 0.09 to 0.32 nats. Users stack three to five runs, so
the integrated arm was also run at M = 3 and 5; the ordering of the
estimates is the same there.

**A hybrid does better still, on this data.** The noise share, the
fraction of the components' spread that the GP attributes to noise,
separates the conditions in the median over cells: 0.21 to 1.1 where
the cap is right, 0.05 on Student, at most 0.085 on the noiseless
controls. Cell by cell the ranges overlap. The rule "cap when the share
is at least 0.2, otherwise shrink" lowers the worst case of the best
single estimate by a fifth to a third. Any threshold from 0.08 to 0.20
gives the same worst case. Its gain is resolvable on multisensory noise
3 and nowhere else.

The candidates side by side, as the worst and the mean over the six
noisy conditions of the median bias against the truth. The two
noiseless controls are within 0.09 nats for every rule. The six
conditions are not exchangeable, so "worst" is multisensory noise 3 for
the shrinkage rules and Student for the cap, and the mean weights the
conditions equally for convenience. The report has the per-condition
tables and the bootstrap intervals.

| headline | worst, M = 3 / 5 / 16 | mean, M = 3 / 5 / 16 |
|---|---|---|
| raw | 0.84 / 1.04 / 1.31 | 0.39 / 0.50 / 0.74 |
| capped median (the headline today) | 0.97 / 1.26 / 1.78 | 0.29 / 0.37 / 0.45 |
| within-run shrinkage | 0.60 / 0.75 / 0.94 | 0.25 / 0.31 / 0.40 |
| two-level shrinkage | 0.58 / 0.67 / 0.77 | 0.24 / 0.26 / 0.28 |
| cap if noise share ≥ 0.2, else within-run shrinkage | 0.42 / 0.43 / 0.44 | 0.19 / 0.22 / 0.15 |

What the shrinkage leaves on the multisensory conditions, it cannot
see. Shrinking toward a population mean removes only the selection
within that population; an error shared by all of a run's components,
or by all runs, stays. On those conditions the GP's own uncertainty is
also too small (a quarter to two thirds of the components have |z|
above 2 in the Phase 2 calibration records), and the shrinkage takes it
at face value.

## Decision

The PI endorsed the recommendation on 2026-09-15: the two-level
shrinkage is the headline candidate for a noisy stack. The raw value
stays the headline of a noiseless stack, where the shrinkage would
change it by at most 0.03 nats.

Against the alternatives: the cap fails silently and without bound on a
heavy-tailed target. The hybrid has the best numbers, but its threshold
was chosen on the data it is scored on, in a gap between six conditions
and two, and it switches between estimates that differ by up to 0.5
nats; it is the candidate to revisit when the M = 32 cells and further
targets are in.

Alongside the headline, the class keeps the raw value and the cap in
`elbo_details`, and adds the noise share as a diagnostic. The
documentation must say that the headline can still be too high by
about 0.8 nats on a high-noise real-data target and too low by up to
0.4 on a heavy-tailed one, and that the raw value is not an upper bound
on the truth.

Whether the switch ships in 1.5, or 1.5 keeps the cap with that caveat
and switches later, is the next decision. The change is contained: a
function in `pyvbmc/svbmc/svbmc.py` that computes the shrunken expected
log joint from `I_sk`, `J_sjk` and the runs' own weights at the selected
weights, as the cap is applied today, so the posterior does not move; a
new `elbo_details` key and headline method; the reporting-plan tests
that pin the capped headline; the user documentation.
`dev/scripts/svbmc_shrink_elbo.py` is the reference implementation.

## Open

- Re-optimizing the weights on the shrunken estimates is untested. Today
  the weights are selected on the raw estimates and only the reported
  value is shrunk.
- The M = 32 cells of the integrated arm, a later overnight or cluster
  run, test the run-level term where it matters most.
- The cells of one condition and M share runs (ten M = 16 cells draw
  160 run slots from 100), so intervals over cells are optimistic, and
  the whole comparison is one pool at one seed.
- The residual on the multisensory conditions is an error shared by a
  run's estimates. The GP-robustness item in the TODO (far-tail
  evaluations, the quadratic mean on heavy tails) is where its cause
  may show up.
