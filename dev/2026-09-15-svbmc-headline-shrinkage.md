# The S-VBMC headline on noisy stacks: what the pools showed and the way forward

*Written 15 September 2026 from the day's results and a discussion with
the PI. The evidence is in
[results/2026-09-15-svbmc-pool-comparison.md](results/2026-09-15-svbmc-pool-comparison.md);
the execution record is the worklog of
[plans/svbmc-benchmark-campaign.md](plans/svbmc-benchmark-campaign.md);
the tracked outputs are under
[experiments/svbmc_pool/](experiments/svbmc_pool/README.md).*

S-VBMC stacks several finished VBMC posteriors and estimates the ELBO
of the stack as the weighted sum of the components' expected log joints
(from each run's GP) plus the mixture entropy. On noisy targets the
estimate is optimistic: a winner's curse on noisy GP estimates.

PyVBMC 1.5 ships S-VBMC as `pyvbmc.svbmc.SVBMC`, ported from the
standalone `svbmc` 0.1.1 (the original implementation). The run-pool
campaign settled two questions:

1. Does the port match the original?
2. Which single number should the port report as the ELBO of a noisy
   stack?

Both implementations were run on the same stacks of the same runs, on
eight conditions with 100 filtered runs per noisy condition and 50 per
noiseless control. Every estimate is scored by its bias against the
stacked posterior's own Monte Carlo ELBO, computed from the true log
density.

## 1. The port matches the original

Same weights, same posterior quality, 1.9 to 4 times faster,
reproducible tables (the report's criteria 1, 2, 4 and 5, all
conditions). The two implementations' capped estimates agree within
0.25 nats.

## 2. Which number to report

The original does not choose: it returns the raw estimate and two
debiased variants (the expected log joint capped at the median over
components or over runs) side by side. The port returns one headline,
the raw value on a noiseless stack and the component-median cap on a
noisy one. The campaign tested that choice; it fails on one of eight
conditions.

**Raw.** Optimistic on every noisy condition: bias +0.25 to +0.9 nats at
M = 2 and +0.6 to +1.3 at M = 16. Most of the bias is within-run, created
by VBMC's own variational optimization selecting components with high
GP estimates; the cross-run part added by stacking is smaller and grows
with M.

**The cap.** Within 0.56 nats on five of the six noisy conditions at
every M. On the eight-dimensional Student target it is 0.9 to 1.8 nats
pessimistic, worse than raw, and the error grows with M. The mechanism
is plain: the cap replaces the stack's expected log joint by the
unweighted median over components, and on a heavy-tailed posterior the
median component is a tail component. Where the cap works, it works
because the low-weight components are the ones the optimization did
not select, so they serve as an unselected control group; variants that
drop them (the κ experiment) lose the debiasing.

**The cross-run estimate** (each component scored by the other runs'
GPs, the optimism note's Phase 2) is unbiased on the noiseless controls
(within 0.035 nats) and on Rosenbrock (within 0.07; nine Rosenbrock
runs fail its own-run check, their GPs being numerically fragile),
biased low by 0.2 to 0.7 nats on the typical noisy conditions, and as
low as the cap on Student.

**Empirical-Bayes shrinkage** is the first estimate acceptable on every
condition. Each component's estimate is the posterior mean under
`I_k ~ N(θ_k, Σ)`, `θ_k ~ N(μ, τ²)`, with `Σ` the run's estimation
covariance from `J_sjk`, `μ` and `τ²` moment-matched within the run,
and the full `Σ` used because one GP estimates all of a run's
components. A second level shrinks each run's own expected log joint
toward the mean over runs by its run-level estimation variance. The
two-level estimate matches the cap on Rosenbrock, noisy GMM and the
ring (differences inside the paired bootstrap intervals at M = 3 and
5), is within 0.4 nats on Student, moves the noiseless controls by at
most 0.03 nats, and has no tuned constant. Its costs: on Student it is
0.24 to 0.39 nats pessimistic at M ≤ 5, where raw is within 0.11; on
the two multisensory conditions the cap is closer by 0.09 to 0.32 nats.
Users stack three to five runs, so the integrated arm was also run at
M = 3 and 5; the ordering of the estimates is the same.

**A hybrid** does better on this data. The noise share (the fraction of
the components' spread that the GP attributes to estimation noise)
separates the conditions in the median over cells: 0.21 to 1.1 where
the cap is right, 0.05 on Student, at most 0.085 on the noiseless
controls; cell by cell the ranges overlap. "Cap when the share is at
least 0.2, otherwise shrink" lowers the worst case of the best single
estimate by a fifth to a third, with the same worst case for any
threshold from 0.08 to 0.20, and its gain is resolvable only on
multisensory noise 3.

Worst and mean over the six noisy conditions of the median absolute
bias (the two controls are within 0.09 for every estimate; the
conditions are not exchangeable, so "worst" is multisensory noise 3 for
the shrinkage estimates and Student for the cap, and the mean weights
them equally):

| headline | worst, M = 3 / 5 / 16 | mean, M = 3 / 5 / 16 |
|---|---|---|
| raw | 0.84 / 1.04 / 1.31 | 0.39 / 0.50 / 0.74 |
| capped median (the headline today) | 0.97 / 1.26 / 1.78 | 0.29 / 0.37 / 0.45 |
| within-run shrinkage | 0.60 / 0.75 / 0.94 | 0.25 / 0.31 / 0.40 |
| two-level shrinkage | 0.58 / 0.67 / 0.77 | 0.24 / 0.26 / 0.28 |
| cap if noise share ≥ 0.2, else within-run shrinkage | 0.42 / 0.43 / 0.44 | 0.19 / 0.22 / 0.15 |

The residual on the multisensory conditions is invisible to shrinkage
by construction: shrinkage toward a population mean removes selection
within the population, not an error common to all of a run's
components or to all runs. On those conditions the GP's own uncertainty
is also miscalibrated (a quarter to two thirds of the components have
|z| > 2 in the Phase 2 calibration), which the shrinkage takes at face
value.

## Decision

The PI endorsed the recommendation on 2026-09-15: the two-level
shrinkage is the headline candidate for a noisy stack; the raw value
stays the headline of a noiseless stack, where shrinkage changes it by
at most 0.03 nats.

Against the alternatives: the cap fails silently and without bound on
heavy tails. The hybrid has the best numbers, but its threshold was
chosen on the data it is scored on, in a gap between six conditions and
two, and it switches between estimates that differ by up to 0.5 nats;
it is the candidate to revisit with the M = 32 cells and further
targets.

The class keeps the raw value and the cap in `elbo_details` and adds
the noise share as a diagnostic. The documentation must state that the
headline can remain optimistic by about 0.8 nats on a high-noise
real-data target and pessimistic by up to 0.4 on a heavy-tailed one,
and that the raw value is not an upper bound on the truth.

Whether the switch ships in 1.5, or 1.5 keeps the cap with that caveat
and switches later, is the next decision. The change is contained: a
function in `pyvbmc/svbmc/svbmc.py` computing the shrunken expected log
joint from `I_sk`, `J_sjk` and the runs' own weights at the selected
weights (as the cap is applied today, so the posterior does not move),
a new `elbo_details` key and headline method, the reporting-plan tests
that pin the capped headline, and the user documentation.
`dev/scripts/svbmc_shrink_elbo.py` is the reference implementation.

## Open

- Re-optimizing the weights on the shrunken estimates is untested; only
  the reported value is shrunk today.
- The M = 32 cells of the integrated arm (a later overnight or cluster
  run) test the run-level term where it matters most.
- Cells of one condition and M share runs (ten M = 16 cells draw 160
  run slots from 100), so intervals over cells are optimistic, and the
  comparison is one pool at one seed.
- The multisensory residual is an error common to a run's estimates;
  the GP-robustness item in the TODO (far-tail evaluations, the
  quadratic mean on heavy tails) is where its cause may show up.
