# Boost-only reconstruction check

The PI requested a reconstruction check before scheduling the proposed
population comparison of final boosts with and without the small-weight
penalty. The acceptance rule remains deferred; keeping both full VPs and
their scores allows later criterion comparisons. This check ran no main
optimization loops and made no target evaluations.

## Outcome

All 870 compact reference traces pass the structural checks needed to
construct a state from their stored fields. Reconstruction does **not**
preserve the original GP's numerical behavior in every case. Both penalty treatments can be
compared on the same reconstructed state with a common fresh RNG, but that
is a paired experiment conditional on that state, not an exact replay of
the historical golden boost. Nine authentic pre-boost snapshots are already
available and should be used directly when possible.

## Training-data recovery

The population checks establish, for every trace:

- Final `func_count == N`, where `N = Xn + 1` includes trimmed logger rows.
  Thus no repeated evaluations were merged into individual rows.
- Final `n_eff == len(X_orig)` for the retained live rows.
- Warmup is false at and after `best_iter`, so no later warmup trimming
  changes the membership of its live training set.

The GP training set at `best_iter` is therefore the first
`n_eff[best_iter]` live rows. Thirteen runs select an earlier iteration:
nine Cigar D15 exhaust runs and four Rosenbrock D2 noise SD 3 runs. The
snapshot for Rosenbrock seed 7 validates this path: 174 selected-iteration
training rows versus 194 rows in the final compact trace.

Reconstruction uses the stored VP arrays, transformer arrays and GP
hyperparameter samples, with the benchmark model/options. Training inputs
are transformed from stored original coordinates; transformed log densities
include the Jacobian. The noisy targets have constant observation variance,
so the GP noise array is `noise_sd**2`. GP posterior factors are recomputed
without fitting hyperparameters. GP hyperparameter bounds and priors are not
used by the boost optimization. The relevant entropy switch is false in
this benchmark population.

## Numerical fidelity

Nine complete snapshots were compared against reconstructions made without
reading their fields. VP density parameters, selected ELBO/SD, GP
hyperparameters, noise arrays and relevant optimizer state agree. The trace
represents absent transformer scaling/rotation as ones/identity; this is
mathematically equivalent. The historical `eta` cache is not stored and may
be stale; optimization obtains its entry weight parameters from
`w`, so raw cache differences do not establish a different optimizer input.

Transforming stored original coordinates back into GP coordinates introduces
differences of roughly `3e-17` to `1e-13`. Most snapshot comparisons show
small resulting prediction differences, but noisy Rosenbrock seed 7 is
ill-conditioned enough to amplify a `7.1e-15` input difference into:

- Maximum prediction mean difference: 0.644531 nats.
- Expected log joint difference at the pre-boost VP: 0.057426 nats.
- Expected-log-joint variance difference: 0.024917 (about 28%).

A further screen rebuilt all 870 GPs and recomputed the pre-boost ELBO SD.
The implementation sets entropy variance to zero, so this comparison uses
the square root of the GP expected-log-joint variance. All reconstructions
completed. Absolute SD differences exceeded `1e-4` in 40 runs, `1e-3` in
four, and `1e-2` in one. These are descriptive bins, not acceptance gates.
Agreement of one scalar does not establish full numerical equivalence.
All SD differences above `1e-6` occurred in the two noisy Rosenbrock
configurations. The other 790 runs had maximum absolute SD difference
`6.93e-7` nats; this remains a scalar diagnostic, not certification of their
full GP or boost behavior.

| Case | Stored SD | Reconstructed SD | Difference |
|---|---:|---:|---:|
| Rosenbrock D2 noise SD 3, seed 7 | 0.297865 | 0.337107 | +0.039242 |
| Rosenbrock D2 noise SD 1, seed 22 | 0.188435 | 0.184936 | -0.003499 |
| Rosenbrock D2 noise SD 3, seed 12 | 0.265457 | 0.267897 | +0.002440 |
| Rosenbrock D2 noise SD 3, seed 16 | 0.264342 | 0.265620 | +0.001279 |

## Three behavioral comparisons

For each case, authentic and reconstructed inputs received the same fresh
RNG state. Both used the zero-penalty boost objective. These compare
reconstruction fidelity, **not** penalty on versus off. Raw candidates were
examined before the 0.1 guard; agreement of the guard's fallback does not
imply agreement of rejected candidates.

| Case | Candidate ELBO, authentic / reconstructed | Candidate SD, authentic / reconstructed | gsKL, authentic / reconstructed |
|---|---:|---:|---:|
| Normal D5 seed 0 | -0.004087132 / -0.004087132 | 0.000580164 / 0.000580164 | 0.000063401 / 0.000063401 |
| Logistic D5 seed 5 | -32.132043268 / -32.132043038 | 4.771838611 / 4.771838472 | 3.668966 / 3.673815 |
| Rosenbrock D2 noise SD 3 seed 7 | -2.425115 / -2.540561 | 0.438182 / 0.320456 | 0.258332 / 0.246548 |

Normal's candidate arrays and metrics were exact. Logistic's candidate
ELBO/SD changes were small; its bounded-posterior diagnostics also contain
Monte Carlo sensitivity. Rosenbrock seed 7's candidate ELBO changed by
0.115446 nats and SD by 0.117727 nats. All three pairs made the same guard
decision (normal accepted, the other two rejected) for this particular RNG.

The nine-state check took about 2.4 seconds including interpreter startup.
The population GP/SD screen took 17.6 seconds. The three paired boost checks
took 29.9 seconds including state checks and diagnostics; the six boost
optimizations themselves took 21.5 seconds, between 1.1 and 7.5 seconds each.
These selected timings are not a reliable estimate for every configuration
in a complete two-treatment campaign.

## Consequences for the proposed experiment

There is no need to default to a 22-hour rerun of all main optimizations.
A paired experiment on reconstructed states is possible, provided both
penalty settings start from the same state and RNG. Use authentic snapshots
where available, including the largest observed reconstruction discrepancy.
Report comparisons to historical golden outcomes separately: differences
there can include reconstruction and fresh-RNG effects as well as the penalty.

For later acceptance-rule analysis, preserve both full VPs and their
transformers, raw boost scores, and restart state. If using a reconstructed
GP, retain historical pre-boost scores separately from a consistent
re-evaluation of that VP under the reconstructed GP. Comparing its new
candidate score against an old score from a materially different GP would
confound the acceptance decision. Diagnostic rescoring must use independent
RNGs so it does not change either boost's optimization stream. That rescoring
design and the population penalty experiment are follow-up work, not results
of this reconstruction check.

## Evidence and reproduction

Evidence is under `dev/scripts/runs/latent_fixes/boost_20260907/`:
`reconstruction_check.json`, `reconstruction_extra4.json`,
`reconstruction_population_sd.json`, `reconstruction_population_full.json`,
`reconstruction_final_state_check.json`, and `reconstruction_replays.json`.
The latter records actual imports and source hashes. Computation used the
original `.venv/Scripts/python.exe`, NumPy 2.5.2, SciPy 1.18.1 and clean
sibling gpyreg `a2f8ddce867f502e29717959cf0ff3529f598618` (head and working
tree verified separately with Git immediately before numerical checks), with
OMP/OpenBLAS/MKL threads each set to one. The active PyVBMC import resolved
to the experimental checkout; no numerical core source was changed during
this check. No reference files were written or rebaselined.

```powershell
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:MPLBACKEND = 'Agg'
.venv/Scripts/python.exe dev/scripts/boost_reconstruction.py --replay normal_D5_seed0 logreg_D5_seed5 rosenbrock_D2_noise3_seed7 --out dev/scripts/runs/latent_fixes/boost_20260907/reconstruction_replays_repeat.json
.venv/Scripts/python.exe dev/scripts/boost_reconstruction.py --population-numerics --out dev/scripts/runs/latent_fixes/boost_20260907/reconstruction_population_repeat.json
```

Independent Sol review checked reconstruction assumptions, script behavior,
option coverage, numerical evidence and this report. The option-coverage and
gradient-canonicalization gaps identified during verification were corrected;
the final nine-state check passed with all 27 compared options matching in
each case. No remaining implementation/reporting findings. Numerical fidelity
limitations above remain findings of the experiment, not implementation gates
that were waived. Python compilation, pinned Black 23.3 formatting and
repository whitespace checks passed. The final state report records the
final checker source hash and single-thread environment; no jobs remain.
