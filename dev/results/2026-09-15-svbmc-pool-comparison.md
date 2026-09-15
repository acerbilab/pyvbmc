# S-VBMC on the run pools: the integrated class against the original implementation

Stage D of the [run-pool campaign](../plans/svbmc-benchmark-campaign.md)
stacked the same subsets of the same finished VBMC runs with the
integrated `pyvbmc.svbmc.SVBMC` and with the original standalone `svbmc`
0.1.1, on the eight pool conditions generated on the cluster
([`experiments/svbmc_pool/pool_20260914/`](../experiments/svbmc_pool/pool_20260914/README.md)),
at `M ∈ {2, 4, 8, 16}` runs per stack with 20, 20, 20 and 10 repetitions:
560 cells, both arms on every cell, 8.1 hours on the analysis machine. Of
the plan's five acceptance criteria, four hold on every condition: the
two implementations optimize the same objective to the same weights
(criterion 1), neither is worse than the other on posterior quality
anywhere (2), the integrated class is two to four times faster (4), and
the tables rebuild from the recorded cells (5). Criterion 3, on evidence
accuracy, holds on seven conditions and fails on one: on the
eight-dimensional Student target at noise 3 the component-median cap
that both implementations apply to a noisy stack puts the reported
ELBO 0.9 to 1.8 nats *below* the stacked posterior's own Monte Carlo
ELBO, further from it than the raw estimate, and the shortfall deepens
with `M`. On the other five noisy conditions the raw stacked ELBO
overshoots by 0.25 to 1.3 nats and grows with `M`, while the capped
headline stays within 0.56 nats, which is what the cap was
written for. The Phase 2 cross-run estimator, scored on the same cells,
does not repair that condition: it is accurate on the noiseless controls
and on Rosenbrock, under-predicts by 0.2 to 0.7 nats on the typical
noisy conditions, and lands as low as the cap on Student D8; and no
weight-aware variant of the cap separates the heavy-tailed case from
the typical one. An empirical-Bayes shrinkage of the components'
expected log joints by their estimation covariance (from the `J_sjk`
and `I_sk` statistics saved with each posterior; no GP object is
needed), within each run and at the run level, is the first
correction acceptable on every
condition, and a rule that applies the cap only where the GP attributes
enough of the components' spread to noise does better still on these
eight conditions; the reading and the decision are in the
[headline note](../2026-09-15-svbmc-headline-shrinkage.md). Scored
against the bias of the input runs themselves, the yardstick the PI
set on 2026-09-15 (a stack must not add to the optimism its runs
already carry), a single VBMC run is optimistic by 0.13 to 0.74 nats
on the typical noisy conditions, the raw stack adds 0.02 to 0.15 nats
at `M = 2` and 0.27 to 0.60 at `M = 16`, the cap removes more than
the stacking added on every condition, the two-level shrinkage adds
nothing within 0.23 nats, and an anchored variant that removes only
the stacking's own selection halves the addition but still adds 0.1
to 0.2 nats at `M` = 3 to 5 (section "The inputs' own bias").
Re-optimizing the weights on the shrunken values, tried at `M` = 3
to 5, makes the headline 0.1 to 0.2 nats more optimistic than
shrinking the value alone and moves the posterior for better or
worse depending on the target, so the shrinkage stays a correction
of the reported value (section "Re-optimizing on the shrunken
estimates"). The
`M = 32` cells of the integrated arm are reported in a section added
when that run completes.

Tracked outputs:
[`experiments/svbmc_pool/stack_20260914/`](../experiments/svbmc_pool/stack_20260914/)
(`summary.json`, `summary.md`, `sources.json`; the section "Stacking
comparison, stage D" of the
[experiments README](../experiments/svbmc_pool/README.md) describes
their keys). Every stage D number below is in that `summary.md`; the
later sections name the directories they read. The comparison's
`results.json` (17.6 MB, every cell), the input of the Phase 2 scoring
and of `--summarize-only`, is in the analyses asset of the draft release
`svbmc-analyses-20260915` together with the Phase 2 and weight-aware-cap
per-cell outputs (the experiments README, "Raw outputs of the
analyses").

## What was compared, and how

The design is the plan's ("Stacking comparison", "Metrics", "Acceptance
criteria for the comparison"); in short:

- **Inputs.** For each condition, subsets of `M` runs drawn without
  replacement from the filtered pool (100 runs per noisy condition, 50
  per noiseless control; `selection.json`) by a generator keyed on the
  seed, the condition, `M` and the repetition. Both arms receive the same
  subset in the same order, rebuild its posteriors from the stored
  artifacts with one generator seed per entry, and never run at the same
  time; which arm runs first alternates from cell to cell.
- **Arms.** Integrated: `SVBMC(vps, seed=cell_seed).optimize(n_samples=20,
  lr=0.1, max_steps=500, version="all-weights", n_samples_final=100)`,
  in the controller process. Original: `svbmc.SVBMC(vps, s_max=√5,
  M_min=2/3).optimize(n_samples=20, lr=0.1, max_steps=500,
  version="all-weights")` in a long-lived worker whose `PYTHONPATH`
  carries the pinned checkout (`13a78f6`) and the CPU Torch overlay,
  seeded by `np.random.seed(cell_seed)` before construction. Both import
  gpyreg 1.2.1 from the campaign's frozen worktree; BLAS and Torch
  single-threaded.
- **Reference.** Every reported ELBO is scored by its bias against the
  stacked posterior's own Monte Carlo ELBO, `elbo_mc = e_log_joint_mc +
  entropy_ref`: the mean of the noiseless log density over 10 000 of the
  stack's 100 000 draws, plus the entropy of the stacked mixture at the
  arm's final weights estimated by one estimator for both arms (the
  integrated class's `stacked_entropy`, 200 draws per component in 8
  batches from a generator seeded by the cell). `kl_gap = ln Z − elbo_mc`
  is how far the stacked posterior is from the target.
- **Headlines.** The integrated class reports the capped value on a noisy
  stack (the component-median cap of the reporting plan) and the raw
  value on a noiseless one; the original reports its raw `estimated`
  value, with `debiased_I_median` as its own capped variant.
- **Posterior quality.** MMTV and gsKL (house convention, no `1/D`) of
  100 000 draws from the stacked posterior against the target's truth;
  exact signed-rank tests on the paired per-cell differences, one Holm
  family per metric over the 32 condition-and-`M` cell sets.

Run: `svbmc_pool_stack.py --pool dev/scripts/runs/svbmc_pool_20260914
--gpyreg-source dev/scripts/runs/svbmc_pool_20260913/gpyreg_1.2.1 --seed 0`
from `dev-next` `744864a` (clean tree), the harness defaults being the
grid above. Measured cost per cell: about 2 s at `M = 2`, 8 s at `M = 4`,
47 s at `M = 8` and 240 s at `M = 16` (integrated fit 57 s, original 162
s, reference entropy 23 s); the multisensory conditions are the slowest
at about 95 minutes each. The plan's 5-hour estimate had extrapolated
the pilot's `M = 3` timings with `M²`.

## Criterion 1: the two implementations agree

Per condition over all `M`, the median paired `max |Δw|` between the two
arms' weight vectors is 0.007 (ring) to 0.028 (Student D8), inside the
within-arm seed-to-seed spread of 0.018 to 0.048 that Phase 4 measured
on the pilot posteriors. The capped estimates of the two implementations
agree as well: the integrated headline and the original's
`debiased_I_median` differ by at most 0.25 nats in the median on every
noisy cell set, including Student D8 where both are far below the
reference.

| condition | `max \|Δw\|` (median, 95 % CI) | runtime ratio int / orig | optimize seconds at `M = 16`, int / orig |
|---|---|---|---|
| multisensory noise 3 | 0.0211 [0.0178, 0.0252] | 0.254 [0.229, 0.286] | 71 / 261 |
| multisensory noise 1.3 | 0.0205 [0.0174, 0.0233] | 0.281 [0.262, 0.303] | 62 / 251 |
| Rosenbrock noise 3 | 0.0216 [0.0201, 0.0229] | 0.402 [0.321, 0.457] | 44 / 94 |
| GMM noise 3 | 0.0218 [0.0189, 0.0237] | 0.466 [0.404, 0.508] | 47 / 65 |
| ring noise 3 | 0.0070 [0.0061, 0.0088] | 0.443 [0.421, 0.520] | 50 / 79 |
| Student D8 noise 3 | 0.0281 [0.0206, 0.0335] | 0.446 [0.383, 0.492] | 65 / 280 |
| GMM (noiseless) | 0.0128 [0.0109, 0.0165] | 0.540 [0.467, 0.597] | 19 / 21 |
| multisensory (noiseless) | 0.0132 [0.0115, 0.0162] | 0.334 [0.285, 0.361] | 40 / 177 |

## Criterion 2: posterior quality is equivalent, and improves with `M`

No Holm-corrected signed-rank test rejects in either direction on any of
the 32 cell sets for MMTV or gsKL (α = 0.05, one family per metric).
Both implementations improve on the single runs as the paper reports:
medians over cells, integrated / original, with the single-run median at
`M = 1` first.

| condition | MMTV `M = 1` | `M = 2` | `M = 4` | `M = 8` | `M = 16` | gsKL `M = 1` | `M = 2` | `M = 16` |
|---|---|---|---|---|---|---|---|---|
| multisensory noise 3 | 0.279 | 0.269 / 0.265 | 0.222 / 0.219 | 0.131 / 0.129 | 0.122 / 0.125 | 4.02 | 3.74 / 3.70 | 0.48 / 0.43 |
| multisensory noise 1.3 | 0.231 | 0.209 / 0.213 | 0.176 / 0.180 | 0.166 / 0.165 | 0.120 / 0.128 | 2.70 | 2.40 / 2.42 | 0.35 / 0.34 |
| Rosenbrock noise 3 | 0.097 | 0.075 / 0.076 | 0.079 / 0.083 | 0.056 / 0.061 | 0.076 / 0.065 | 0.15 | 0.06 / 0.06 | 0.01 / 0.01 |
| GMM noise 3 | 0.484 | 0.339 / 0.338 | 0.176 / 0.172 | 0.126 / 0.121 | 0.109 / 0.108 | 22.5 | 4.79 / 4.88 | 0.02 / 0.03 |
| ring noise 3 | 0.627 | 0.380 / 0.377 | 0.259 / 0.256 | 0.181 / 0.192 | 0.134 / 0.128 | 97.5 | 1.44 / 1.38 | 0.03 / 0.03 |
| Student D8 noise 3 | 0.143 | 0.114 / 0.118 | 0.111 / 0.111 | 0.085 / 0.086 | 0.079 / 0.081 | 0.58 | 0.31 / 0.31 | 0.10 / 0.10 |
| GMM (noiseless) | 0.390 | 0.196 / 0.194 | 0.047 / 0.073 | 0.028 / 0.027 | 0.024 / 0.022 | 14.9 | 0.91 / 0.91 | 0.00 / 0.00 |
| multisensory (noiseless) | 0.118 | 0.104 / 0.101 | 0.090 / 0.093 | 0.043 / 0.047 | 0.046 / 0.033 | 0.59 | 0.44 / 0.44 | 0.06 / 0.03 |

The KL gap of the stacked posterior to the target shrinks accordingly:
from `M = 2` to `M = 16`, 0.85 to 0.43 nats on multisensory noise 3,
0.76 to 0.13 on noisy GMM, 1.01 to 0.15 on the ring, 0.52 to 0.25 on
Student D8, 0.31 to 0.01 on the noiseless GMM.

## Criterion 3: evidence accuracy

Medians over cells of the bias of each reported ELBO against `elbo_mc`,
as capped headline (integrated) / raw (integrated) / raw `estimated`
(original). The gate of the criterion's first clause is that the
headline's median bias is no larger in magnitude than the original's raw
estimate's, at every `M`; its second clause bounds the growth of the
headline bias from the smallest to the largest `M` by 0.5 nats.

| condition | `M = 2` | `M = 4` | `M = 8` | `M = 16` | gate | growth |
|---|---|---|---|---|---|---|
| multisensory noise 3 | +0.38 / +0.77 / +0.90 | +0.56 / +1.07 / +1.18 | +0.40 / +1.16 / +1.32 | +0.44 / +1.31 / +1.45 | holds | +0.06 |
| multisensory noise 1.3 | +0.08 / +0.41 / +0.51 | +0.11 / +0.47 / +0.53 | +0.14 / +0.56 / +0.66 | +0.16 / +0.61 / +0.63 | holds | +0.08 |
| Rosenbrock noise 3 | +0.09 / +0.25 / +0.31 | +0.07 / +0.35 / +0.40 | +0.10 / +0.53 / +0.57 | +0.11 / +0.74 / +0.78 | holds | +0.02 |
| GMM noise 3 | +0.07 / +0.30 / +0.33 | −0.01 / +0.30 / +0.36 | +0.06 / +0.50 / +0.50 | +0.05 / +0.73 / +0.74 | holds | −0.02 |
| ring noise 3 | +0.21 / +0.41 / +0.43 | +0.16 / +0.40 / +0.45 | +0.16 / +0.49 / +0.51 | +0.14 / +0.76 / +0.78 | holds | −0.07 |
| Student D8 noise 3 | −0.95 / −0.11 / +0.04 | −1.17 / −0.10 / −0.03 | −1.39 / +0.22 / +0.38 | −1.78 / +0.27 / +0.41 | **fails at every `M`** | −0.84 |
| GMM (noiseless) | +0.01 / +0.01 / +0.05 | +0.00 / +0.00 / +0.05 | +0.00 / +0.00 / +0.04 | +0.01 / +0.01 / +0.01 | holds | +0.00 |
| multisensory (noiseless) | +0.03 / +0.03 / +0.11 | +0.04 / +0.04 / +0.13 | +0.07 / +0.07 / +0.11 | +0.05 / +0.05 / +0.14 | holds | +0.03 |

Three readings:

- **The raw stacked ELBO is optimistic on every noisy condition and the
  optimism grows with `M`**, as the optimism note predicted: from
  `M = 2` to `M = 16` the integrated raw bias goes from +0.25 to +0.74 on
  Rosenbrock, +0.30 to +0.73 on noisy GMM, +0.41 to +0.76 on the ring and
  +0.77 to +1.31 on multisensory noise 3. The original's raw estimate is
  up to 0.16 nats higher still, because it reports the optimizer's own
  20-draw entropy, which sits above the arm-independent reference on
  every condition (by 0.16 nats at `M = 2` on multisensory noise 3, 0.10
  on Student D8), while the integrated class's fresh 100-draw evaluation
  sits within 0.035 of it.
- **The cap does its job on five of the six noisy conditions**: the
  headline stays within 0.56 nats of the reference at every `M`,
  its growth is at most +0.08 nats, and on the noiseless controls no cap
  applies and the headline is the raw value, within 0.07 nats.
- **On Student D8 the cap over-corrects.** The raw estimates are the
  least optimistic of the campaign (within 0.4 nats at every `M`, and
  slightly pessimistic at `M ≤ 4`), yet the capped headline lands 0.95
  nats below the reference at `M = 2` and 1.78 below at `M = 16`; the
  original's own capped variant, `debiased_I_median`, lands within 0.2
  of it (−0.71 to −1.70), so this is the component-median cap itself,
  not the integrated class's implementation of it. The growth bound is
  met only because it limits an increase. The single runs of this
  condition have a large evidence error (median 0.96 nats; the ring's
  1.38 and noisy GMM's 1.04 are larger), and the cap's failure needs
  no more than the real heterogeneity of a heavy-tailed posterior's
  components, whose median is a tail one; what the honest cross-run
  estimator makes of these cells is the question for the Phase 2
  scoring.

## Criterion 4: runtime

The median ratio of the integrated arm's optimization seconds to the
original's is 0.25 (multisensory noise 3) to 0.54 (noiseless GMM) per
condition, with every 95 % bootstrap interval below 0.60 (table under
criterion 1). At `M = 16` the integrated fit takes 44 to 71 s against
the original's 65 to 280 s; the advantage is largest on the
six-dimensional and eight-dimensional targets, where the original's
per-step cost is highest.

## Criterion 5: reproducibility

`svbmc_pool_stack.py --summarize-only --from-results <results.json>`
rebuilds `summary.md` byte for byte (up to the generation timestamp) and
`summary.json` up to the timestamp and the `arms_present` field the
harness gained on 2026-09-15. The pool's manifest, selection and
verification, the baseline environment record and `sources.json` name
every source: both arms imported gpyreg 1.2.1 (`9e70e6b`) from the
frozen worktree named by `--gpyreg-source`, the integrated arm PyVBMC at
`744864a` with a clean tree, the original arm `svbmc` 0.1.1 from the
pinned checkout, Torch 2.14.0 (CPU) from the recorded overlay.

## Phase 2: the cross-run estimator on these cells

The optimism note's Phase 2 estimator (`svbmc_honest_elbo.py`; the
[pilot report](2026-09-14-svbmc-honest-elbo-pilot.md) defines every
quantity) scored every cell of stage D: for each component of a stacked
posterior, the GPs of the *other* runs that cover it estimate its
expected log joint, and the weighted sum replaces the runs' own
estimates, which the weight optimization selected on. 695 runs checked
and 560 cells scored in 2.0 hours (100 draws per component, coverage
ratio 2, median combination; tracked under
[`experiments/svbmc_pool/phase2_20260915/`](../experiments/svbmc_pool/phase2_20260915/)).
Bias of the expected log joint against `e_log_joint_mc`, medians over
cells with the estimator's bootstrap intervals in `summary.md`; the
entropy is common to the three estimates, so these are the ELBO biases
of the criterion 3 table up to the entropy's 0.03. `covered` is the
weighted fraction of components some other run covers.

| condition | M | raw | capped (class) | honest | covered |
|---|---|---|---|---|---|
| multisensory noise 3 | 2 / 4 / 8 / 16 | +0.79 / +1.07 / +1.17 / +1.32 | +0.41 / +0.56 / +0.40 / +0.42 | −0.30 / −0.48 / −0.71 / −0.68 | 0.97 / 1.00 / 1.00 / 0.99 |
| multisensory noise 1.3 | 2 / 4 / 8 / 16 | +0.40 / +0.48 / +0.56 / +0.62 | +0.12 / +0.11 / +0.15 / +0.17 | −0.15 / −0.26 / −0.24 / −0.29 | 0.98 / 0.98 / 1.00 / 1.00 |
| Rosenbrock noise 3 | 2 / 4 / 8 / 16 | +0.25 / +0.35 / +0.53 / +0.73 | +0.10 / +0.08 / +0.10 / +0.11 | −0.01 / −0.07 / −0.05 / +0.04 | 0.98 / 1.00 / 1.00 / 1.00 |
| GMM noise 3 | 2 / 4 / 8 / 16 | +0.29 / +0.31 / +0.51 / +0.71 | +0.08 / +0.01 / +0.07 / +0.04 | +0.15 / −0.21 / −0.29 / −0.29 | 0.24 / 0.70 / 0.90 / 1.00 |
| ring noise 3 | 2 / 4 / 8 / 16 | +0.40 / +0.40 / +0.49 / +0.76 | +0.20 / +0.16 / +0.16 / +0.14 | +0.22 / +0.17 / −0.12 / −0.10 | 0.16 / 0.47 / 0.81 / 1.00 |
| Student D8 noise 3 | 2 / 4 / 8 / 16 | −0.08 / −0.11 / +0.25 / +0.27 | −0.91 / −1.18 / −1.39 / −1.75 | −1.18 / −1.49 / −1.42 / −1.40 | 1.00 / 1.00 / 1.00 / 1.00 |
| GMM (noiseless) | 2 / 4 / 8 / 16 | +0.01 / +0.01 / +0.01 / +0.01 | −0.00 / −0.01 / −0.02 / −0.02 | +0.01 / +0.01 / −0.00 / +0.01 | 0.28 / 0.63 / 0.99 / 1.00 |
| multisensory (noiseless) | 2 / 4 / 8 / 16 | +0.05 / +0.04 / +0.07 / +0.05 | −0.04 / −0.01 / +0.03 / +0.02 | +0.00 / −0.02 / −0.01 / −0.03 | 0.85 / 0.96 / 0.99 / 0.99 |

Four readings:

- **Accurate where the pilot said it would be.** Within 0.01 nats on the
  noiseless GMM and 0.035 on the noiseless multisensory control at every
  `M`, and within 0.07 on Rosenbrock, where it beats the cap at `M ≥ 8`.
  On Rosenbrock nine runs are flagged by the estimator's own-run check
  (`z > 4`): the GPs rebuilt on the analysis machine disagree with the
  stored statistics, the platform sensitivity described under
  "Limitations", so that condition's honest values carry that caveat.
- **The pilot's under-prediction holds at scale.** On noisy GMM and on
  both multisensory conditions the honest estimate sits *below* the
  reference at every `M ≥ 4`, by 0.2 to 0.7 nats, the shortfall
  growing with `M` on multisensory noise 3 (at `M = 2` it is 0.15 above
  on noisy GMM and 0.15 below on multisensory noise 1.3), while the cap
  sits 0.01 to 0.56 above it; on those three conditions the cap is the
  closer estimate at every `M ≥ 4` except multisensory noise 3 at
  `M = 4`. On the ring the two are comparable, the honest estimate
  closer at `M ≥ 8`.
- **Student D8 is not repaired.** The honest estimate is as low as the
  cap, 1.2 to 1.5 nats below the reference, while the raw value is
  within 0.3.
- **Why the cross-run estimate is low on Student.** The cap's failure
  there needs no mechanism beyond the real heterogeneity of the
  components (the shrinkage section measures it: only 4 % of their
  spread is estimation noise). For the cross-run estimate one reading
  fits the numbers without being demonstrated by them: the runs' GPs
  carry a negative-quadratic mean, which falls off faster than a
  Student-t log density, so other runs' predictions at a component,
  made where they have little data, sit below the truth, while the raw
  value is built on the selected central components with their own
  data and is within 0.3. The estimator's decomposition records (the
  data each run had at each component) are where to test it.

## Weight-aware caps

The cap's failure on Student D8 suggested a cap that respects the
weights: order the components by weight, take the shortest prefix whose
cumulative mass reaches `κ` (the crossing component included, so the set
is never empty), and cap the stacked expected log joint at the median
of that prefix's `I_k`; `κ = 1` includes every component and is the cap
the class applies. `svbmc_cap_kappa.py` scored this on stage D's
recorded cells without refitting (each stack rebuilt from the artifacts
with the recorded seeds and constructed; the rebuild reproduces the raw
value and the class's cap exactly), plus the weighted median of the
`I_k`; tracked under
[`experiments/svbmc_pool/cap_kappa_20260915/`](../experiments/svbmc_pool/cap_kappa_20260915/).
Median bias against `elbo_mc` at `M = 16`, with the fraction of cells
the cap binds on in brackets:

| condition (`M = 16`) | raw | κ 0.8 | κ 0.9 | κ 0.95 | κ 0.99 | κ 1 (class) |
|---|---|---|---|---|---|---|
| multisensory noise 3 | +1.31 | +1.30 [0.40] | +1.02 [1.00] | +0.87 [1.00] | +0.68 [1.00] | +0.44 [1.00] |
| multisensory noise 1.3 | +0.61 | +0.60 [0.40] | +0.51 [0.80] | +0.41 [1.00] | +0.28 [1.00] | +0.16 [1.00] |
| Rosenbrock noise 3 | +0.74 | +0.74 [0.10] | +0.72 [0.60] | +0.55 [0.90] | +0.30 [1.00] | +0.11 [1.00] |
| GMM noise 3 | +0.73 | +0.71 [0.30] | +0.66 [0.90] | +0.58 [1.00] | +0.33 [1.00] | +0.05 [1.00] |
| ring noise 3 | +0.76 | +0.76 [0.20] | +0.70 [0.70] | +0.54 [0.90] | +0.39 [1.00] | +0.14 [1.00] |
| Student D8 noise 3 | +0.27 | +0.17 [0.70] | −0.32 [1.00] | −0.73 [1.00] | −1.16 [1.00] | −1.78 [1.00] |

For `κ ≤ 0.7` the cap binds on at most 35 % of the cells and moves the
median headline from the raw value by at most 0.08 nats; at `κ = 0.8`
it binds on up to 70 % (Student at `M = 16`) and moves it by up to
0.16; `κ` from 0.9 to 0.99 binds on most cells but leaves 0.1 to 1.0
nats of the optimism on the typical noisy conditions while still
over-correcting Student by 0.15 to 1.2; the weighted median is the raw
value everywhere; and on Student the median absolute bias is smallest
at `κ = 0.8` for `M = 8` and `16` (0.14 and 0.18) and at `κ ≤ 0.7` for
`M ≤ 4`. The reading is that the class's cap works
*because* it includes the low-weight components: they act as a control
group for the winner's curse of the selected ones, and the debiasing
is the gap between the selected and the unselected. On Student D8 the
unselected components are worse than the selected ones for a real
reason, the tails, not by selection luck, so the control group is
biased, and no `κ` separates that case from the typical one.

Two run-level caps were scored on the same cells: `E_max`, the stacked
expected log joint capped at the largest of the runs' own expected log
joints (each run's posterior weights times its `I_k`), so that a stack
cannot report more than its best input run, and `E_top`, capped at the
value of the run carrying the largest stacking mass. (A cap at the
largest *component* `I_k` can never bind, since the stacked value is a
convex combination of the `I_k`; the script confirms it on every cell.)
Neither helps: `E_max` binds on 0 to 50 % of the cells and moves the
headline by at most 0.1 nats from the raw value on every condition
(multisensory noise 3 at `M = 16`: +1.31 against raw +1.31; noisy GMM
+0.67 against +0.73; Student +0.27, never binding), `E_top` by at most
0.16. The stacked value sits *below* the best input run's own expected
log joint on most cells; that benchmark is itself the maximum of `M`
noisy run-level values, so the observation shows only that the
optimism is not created by mixing runs. Most of it is inside each
run's own value, placed there by VBMC's own variational optimization,
which weights components where its GP estimate is high; a cross-run
part exists and grows with `M` (the raw bias rises by 0.3 to 0.5 nats
from `M = 2` to `16` on every noisy condition but Student), and the
run-level shrinkage of the next section removes part of it. The
component-median cap works because the low-weight components
are the ones that optimization did not favour, the least selected-on
estimates the stack holds; the run-median cap (`capped_E_median`, +0.17
to +0.82 at `M = 16` on the typical noisy conditions, −1.13 on Student)
sits between the two.

## Empirical-Bayes shrinkage

The method, its derivation and a worked example are in the
[tutorial note](../2026-09-15-svbmc-shrinkage-explained.md); this
section records what it does on stage D's cells.

Since the optimism is a selection on noisy GP estimates, the textbook
correction is to shrink each component's estimate toward its population
mean by the share of the population's spread that is estimation noise:
with `I_k ~ N(θ_k, V_k)` and `θ_k ~ N(μ, τ²)`, report
`μ + τ² / (τ² + V_k) (I_k − μ)`, where `V_k` is the estimation variance
of the estimate, read from the `J_sjk` statistic VBMC saves with every
posterior and the class already requires (the GP itself is not
needed), and `τ²` the
excess of the estimates' spread over the noise in it, by moments. One
GP estimates a run's components jointly, so their errors are
correlated, and the noise a sample variance of `K` estimates carries is
`(tr Σ − 1ᵀ Σ 1 / K) / (K − 1)` for the estimation covariance `Σ`, not
the mean `V_k`: a shared error moves every estimate together and adds
nothing to their spread. `svbmc_shrink_elbo.py` scores this on stage
D's cells at the recorded weights (tracked under
[`experiments/svbmc_pool/shrink_20260915/`](../experiments/svbmc_pool/shrink_20260915/)),
the population being the run's own components (`within`, with the
diagonal of `Σ` in the shrinkage factor, or `within full` with all of
it, the posterior mean `μ + τ² (τ² I + Σ)⁻¹ (I − μ)`) or the whole
stack (`stack`). Median bias against `elbo_mc` at `M = 16`, with the
weight-averaged shrinkage factor in brackets (1 leaves the estimates
alone; for the full-covariance form the row sum of the shrinkage
matrix, the factor a deviation shared by every component is kept by),
and the cell's noise share, the mass-weighted mean over its runs of the
noise in the spread over the spread:

| condition (`M = 16`) | raw | class cap | within | within full | stack | noise share |
|---|---|---|---|---|---|---|
| multisensory noise 3 | +1.31 | +0.44 | +1.12 [0.68] | +0.94 [0.11] | +1.07 [0.70] | 0.33 |
| multisensory noise 1.3 | +0.61 | +0.16 | +0.56 [0.79] | +0.42 [0.27] | +0.55 [0.83] | 0.24 |
| Rosenbrock noise 3 | +0.74 | +0.11 | +0.59 [0.63] | +0.46 [0.08] | +0.49 [0.66] | 0.32 |
| GMM noise 3 | +0.73 | +0.05 | +0.53 [0.65] | +0.27 [0.14] | +0.46 [0.74] | 0.32 |
| ring noise 3 | +0.76 | +0.14 | +0.44 [0.20] | +0.32 [0.03] | +0.34 [0.29] | 1.04 |
| Student D8 noise 3 | +0.27 | −1.78 | +0.22 [0.94] | −0.00 [0.37] | +0.20 [0.96] | 0.04 |
| noiseless controls, over every `M` | within 0.07 | within 0.05 | within 0.09 [0.95 to 1.00] | within 0.09 [0.87 to 1.00] | within 0.08 [0.96 to 1.00] | 0.005 to 0.11 |

Shrinkage is safe where the caps were not: it moves the noiseless
controls by at most 0.03 nats (nothing to shrink on the noiseless GMM;
a factor of 0.87 to 0.89 on the noiseless multisensory control, whose
GPs carry a little noise), and on Student D8, where the spread of the
components is real (noise share 0.04), the full-covariance form lands
within 0.01 at `M = 16`. On the typical noisy conditions it removes a
third to two thirds of the optimism (the full-covariance form: noisy
GMM from +0.73 to +0.27, the ring from +0.76 to +0.32, multisensory
noise 3 from +1.31 to +0.94) but does not reach the cap, and what it
leaves grows with `M`. The full form shrinks harder than the diagonal
one because it treats a deviation shared by all of a run's components
as noise, which the diagonal form cannot; what neither can see is an
error common to the whole population, since shrinking toward a
population mean removes selection on the contrast within it only. The
growth with `M` is a selection among runs: the stack puts its mass on
the runs whose own values came out highest, and the best of more runs
is more optimistic. The stack-wide population does not capture it,
since it reads the runs' different levels as real signal.

The two-level version does see it: each run's level, its own weighted
expected log joint with the run-level estimation variance
`w_mᵀ Σ_m w_m`, is shrunk toward the runs' mean by the share of the runs'
spread that is estimation noise, every component of the run is shifted
by the change of its level, and the within-run shrinkage is applied on
top (`run_level` is the shift alone, `two_level` and `two_level_full`
the composition with the diagonal and the full within-run variants).
Median bias against `elbo_mc` at `M` = 2 / 4 / 8 / 16:

| condition | `two_level_full` | `two_level` | `run_level` | class cap |
|---|---|---|---|---|
| multisensory noise 3 | +0.52 / +0.72 / +0.67 / +0.77 | +0.63 / +0.86 / +0.83 / +0.97 | +0.70 / +1.00 / +1.00 / +1.15 | +0.38 / +0.56 / +0.40 / +0.44 |
| multisensory noise 1.3 | +0.21 / +0.26 / +0.36 / +0.38 | +0.33 / +0.39 / +0.47 / +0.52 | +0.38 / +0.44 / +0.51 / +0.57 | +0.08 / +0.11 / +0.14 / +0.16 |
| Rosenbrock noise 3 | +0.11 / +0.07 / +0.07 / +0.16 | +0.16 / +0.19 / +0.22 / +0.30 | +0.19 / +0.28 / +0.33 / +0.47 | +0.09 / +0.07 / +0.10 / +0.11 |
| GMM noise 3 | +0.10 / −0.01 / +0.05 / +0.13 | +0.22 / +0.16 / +0.24 / +0.37 | +0.29 / +0.28 / +0.40 / +0.56 | +0.07 / −0.01 / +0.06 / +0.05 |
| ring noise 3 | +0.15 / +0.06 / +0.09 / +0.19 | +0.21 / +0.08 / +0.16 / +0.29 | +0.40 / +0.34 / +0.40 / +0.62 | +0.21 / +0.16 / +0.16 / +0.14 |
| Student D8 noise 3 | −0.36 / −0.39 / −0.20 / −0.08 | −0.19 / −0.18 / +0.09 / +0.15 | −0.15 / −0.13 / +0.14 / +0.20 | −0.95 / −1.17 / −1.39 / −1.78 |
| GMM (noiseless) | +0.01 / +0.01 / +0.01 / +0.01 | +0.01 / +0.00 / +0.00 / +0.01 | +0.01 / +0.00 / +0.00 / +0.01 | −0.02 / −0.01 / −0.03 / −0.02 |
| multisensory (noiseless) | +0.06 / +0.06 / +0.09 / +0.07 | +0.04 / +0.05 / +0.08 / +0.06 | +0.03 / +0.04 / +0.07 / +0.05 | −0.05 / −0.02 / +0.03 / +0.02 |

`two_level_full` is the first correction in this report that is
acceptable on every condition: on Rosenbrock, noisy GMM and the ring
its distance from the reference is within 0.10 nats of the cap's at
every `M` (paired bootstrap intervals over cells of the difference of
median absolute biases include zero at `M = 3` and `5`; at `M = 16` the
cap is closer on Rosenbrock by 0.10 [0.02, 0.50] and on noisy GMM by
0.08 [0.01, 0.13]); on Student D8 it stays within 0.4 nats where the
cap is off by up to 1.8; it moves the noiseless controls by at most
0.03; and its worst case, +0.77 on multisensory noise 3 at `M = 16`,
compares with the cap's 1.78 and the raw value's 1.31. It has no tuned
constant and uses only what the class already holds (`I_sk`, `J_sjk`,
the runs' own weights). Its costs are as visible: on the two
multisensory conditions the cap stays closer by 0.09 to 0.32 nats, a
difference the paired bootstrap resolves at every `M` (multisensory
noise 3: 0.17 [0.03, 0.35] at `M = 3`, 0.32 [0.25, 0.39] at `M = 16`),
and on Student D8 it is worse than the raw value at `M ≤ 5` (0.24 to
0.39 nats low against raw's 0.04 to 0.11, on 65 to 85 % of the cells).
The run-level shift alone does little below `M = 8` and most of its
work above, as expected of a selection among runs; the within-run
shrinkage with the full covariance does most of the rest. Two things
remain untested: the weights were held at the values selected on the
unshrunken estimates (re-optimizing on the shrunken ones would also
move the posterior toward the population mean), and the run-level
moments rest on two to five values at `M ≤ 5`.

**At the `M` users run.** Stacks of three to five runs are the common
case, so the small-`M` cells decide. The integrated arm was run at
`M = 3` and `5` as well (20 repetitions on every condition, 320 cells,
15 minutes; tracked under
[`experiments/svbmc_pool/stack_M35_20260915/`](../experiments/svbmc_pool/stack_M35_20260915/),
its scorings under `shrink_M35_20260915/` and `cap_kappa_M35_20260915/`,
and [`stack_merged_20260915/`](../experiments/svbmc_pool/stack_merged_20260915/)
is the one summary of stage D and this run over every `M` through
`--summarize-only --from-results`). Median bias against `elbo_mc` at
`M` = 2 / 3 / 4 / 5, raw / class cap / `within_full` / `two_level_full`:

| condition | `M = 2` | `M = 3` | `M = 4` | `M = 5` |
|---|---|---|---|---|
| multisensory noise 3 | +0.77 / +0.38 / +0.57 / +0.52 | +0.84 / +0.42 / +0.60 / +0.58 | +1.07 / +0.56 / +0.83 / +0.72 | +1.04 / +0.43 / +0.75 / +0.67 |
| multisensory noise 1.3 | +0.41 / +0.08 / +0.20 / +0.21 | +0.44 / +0.16 / +0.28 / +0.25 | +0.47 / +0.11 / +0.29 / +0.26 | +0.55 / +0.19 / +0.37 / +0.35 |
| Rosenbrock noise 3 | +0.25 / +0.09 / +0.16 / +0.11 | +0.29 / +0.08 / +0.20 / +0.12 | +0.35 / +0.07 / +0.18 / +0.07 | +0.53 / +0.13 / +0.31 / +0.18 |
| GMM noise 3 | +0.30 / +0.07 / +0.11 / +0.10 | +0.34 / +0.02 / +0.01 / −0.05 | +0.30 / −0.01 / +0.02 / −0.01 | +0.40 / +0.01 / +0.06 / −0.00 |
| ring noise 3 | +0.41 / +0.21 / +0.16 / +0.15 | +0.37 / +0.09 / +0.14 / +0.10 | +0.40 / +0.16 / +0.15 / +0.06 | +0.42 / +0.22 / +0.19 / +0.13 |
| Student D8 noise 3 | −0.11 / −0.95 / −0.34 / −0.36 | −0.09 / −0.97 / −0.28 / −0.32 | −0.10 / −1.17 / −0.34 / −0.39 | +0.04 / −1.26 / −0.18 / −0.24 |
| noiseless controls | within 0.06 for all | within 0.08 | within 0.06 | within 0.08 |

The run-level term matters less at small `M` but is not negligible:
`two_level_full` and `within_full` differ by at most 0.05 nats at
`M = 2` and 0.08 at `M = 3`, by 0.11 at `M = 4` and 0.14 at `M = 5`
(Rosenbrock and multisensory noise 3, where the run-level shift helps),
and by 0.24 and 0.30 at `M = 8` and `16`. Its moment estimate is
unreliable at these `M`: the runs' `τ²` is zero, a full shrink to the
runs' mean, in a fifth to two thirds of the cells of the typical noisy
conditions (0 to 15 % on Student) and its factor spans 0 to 0.8 across
cells, so the two-level form at small `M` rests on a coin flip that
happened to land well. Cell by cell the corrections scatter alike (10th
to 90th percentile ranges of 0.3 to 0.8 nats for raw and the shrinkage
on the noisy conditions; the cap's reaches 1.3 on Student). Worst case
over the six noisy conditions, median bias in magnitude, at `M` = 2 / 3
/ 4 / 5: raw 0.77 / 0.84 / 1.07 / 1.04, cap 0.95 / 0.97 / 1.17 / 1.26,
`within_full` 0.57 / 0.60 / 0.83 / 0.75, `two_level_full` 0.52 / 0.58 /
0.72 / 0.67; the mean over the six: raw 0.37 to 0.50, cap 0.29 to 0.37,
`within_full` 0.25 to 0.31, `two_level_full` 0.24 to 0.26. The ordering
is the same at every `M` a user is likely to run. The cap remains
closer than `two_level_full` on the two multisensory conditions (by
0.09 to 0.32 nats) and further on Student D8 (by 0.6 to 1.7); on
Rosenbrock, noisy GMM and the ring the two are within 0.10 of each
other, inside the bootstrap intervals at `M ≤ 5`.

**A switch by the noise share.** What breaks the cap on Student is
visible in the data the class holds: the cell's noise share, the share
of the components' spread that the GP attributes to estimation noise,
is in the median over cells 0.21 to 1.1 on the five typical noisy
conditions, 0.05 on Student D8, 0.005 on the noiseless GMM and 0.085
on the noiseless multisensory control. Cell by cell the ranges overlap:
Rosenbrock's cells span 0.02 to 1.3, the noiseless multisensory
control's reach 0.24, Student's 0.10, multisensory noise 1.3's start at
0.12. The `hybrid` rule of `svbmc_shrink_elbo.py` applies the cap when
the share is at least 0.2 and the within-run shrinkage otherwise; over
every `M` it chooses the cap on all cells of multisensory noise 3 and
the ring, on 88 % of noisy GMM's, 71 % of Rosenbrock's and 61 % of
multisensory noise 1.3's, on none of Student's or the noiseless GMM's
and on 3 of the 110 cells of the noiseless multisensory control. Worst
and mean median bias over the six noisy conditions:

| rule | worst, M = 2 / 3 / 4 / 5 / 8 / 16 | mean |
|---|---|---|
| raw | 0.77 / 0.84 / 1.07 / 1.04 / 1.16 / 1.31 | 0.37 to 0.74 |
| capped median | 0.95 / 0.97 / 1.17 / 1.26 / 1.39 / 1.78 | 0.29 to 0.45 |
| `within_full` | 0.57 / 0.60 / 0.83 / 0.75 / 0.79 / 0.94 | 0.25 to 0.40 |
| `two_level_full` | 0.52 / 0.58 / 0.72 / 0.67 / 0.67 / 0.77 | 0.24 to 0.28 |
| hybrid (cap if share ≥ 0.2, else `within_full`) | 0.38 / 0.42 / 0.56 / 0.43 / 0.40 / 0.44 | 0.15 to 0.25 |

The worst case is the same for any threshold from 0.08 to 0.20 and the
mean moves by at most 0.03 over that range; above 0.25 the rule stops
capping part of multisensory noise 1.3 and the worst case rises. The
hybrid lowers the worst case of the best single rule by a fifth to a
third; its gain over `two_level_full` is resolvable by the paired
bootstrap on multisensory noise 3 (0.17 to 0.32 nats) and on
multisensory noise 1.3 at `M = 16` (0.21), and within the intervals
elsewhere. Its caveats: the threshold sits in a gap between six
conditions on one side and two on the other, chosen on the data it is
scored on; the shares overlap cell by cell; and it switches between two
estimates that differ by up to 0.5 nats on the typical conditions and
by 0.6 or more on Student.

**What the numbers support.** For a headline that must not fail
silently, the two-level shrinkage: worst case below the cap's at every
`M`, mean at or below it, within 0.4 nats on the heavy-tailed target and
0.09 on the noiseless ones, one formula with no constant. The cap is
the better estimate on the two multisensory conditions by 0.09 to 0.32
nats, which the noise share tells apart in the median on this data; the
[headline note](../2026-09-15-svbmc-headline-shrinkage.md) records the
decision taken on these tables. Two cautions on reading them: the six
noisy conditions are not exchangeable (two are one target at two noise
levels, three are two-dimensional synthetics, one is eight-dimensional
and heavy-tailed), so the worst case names different conditions for
different rules and the mean is a convenience weighting; and the cells
of one condition and `M` share runs (ten `M = 16` cells draw 160 run
slots from 100), so the bootstrap intervals over cells are optimistic.
The next section restates every estimate against the bias of the
input runs, the yardstick the PI set after these tables.

## The inputs' own bias, and what stacking adds

Every bias above is against the stack's own Monte Carlo ELBO, with
zero as the unstated target. The PI restated the requirement on
2026-09-15 (the plan's decision 11): S-VBMC is not asked to remove the
optimism VBMC builds into each run's ELBO, only not to add to it, so
a stacked estimate is judged by its bias relative to the mean bias of
the runs it was built from. `svbmc_single_run_bias.py` scores every
filtered run of the pool as a stack of one against the same reference
(the target's noiseless log density over 10 000 draws from the run's
posterior, plus the mixture's entropy by the comparison's estimator)
and joins the result to every cell of stage D and of the `M = 3` and
`5` run (700 runs in 10 minutes; tracked under
[`experiments/svbmc_pool/single_run_20260915/`](../experiments/svbmc_pool/single_run_20260915/added.md)).

Single runs: the median bias of the run's reported ELBO against its
own `elbo_mc`, with a bootstrap interval over the runs and the
quartiles. The class's raw value for the run alone at its own weights
agrees with the reported ELBO within 0.02 nats in the median on every
condition; the last column is the component-median cap at those
weights, what the class would report for the run alone.

| condition | n | reported ELBO bias, median [CI] | quartiles | cap at `M = 1` |
|---|---|---|---|---|
| multisensory noise 3 | 100 | +0.74 [+0.64, +0.80] | +0.50 to +0.94 | +0.43 |
| multisensory noise 1.3 | 100 | +0.37 [+0.32, +0.42] | +0.21 to +0.50 | +0.12 |
| Rosenbrock noise 3 | 100 | +0.13 [+0.05, +0.19] | −0.05 to +0.27 | +0.06 |
| GMM noise 3 | 100 | +0.17 [+0.09, +0.23] | −0.01 to +0.36 | +0.04 |
| ring noise 3 | 100 | +0.26 [+0.21, +0.35] | +0.07 to +0.43 | +0.11 |
| Student D8 noise 3 | 100 | −0.19 [−0.34, −0.14] | −0.50 to −0.01 | −0.64 |
| GMM (noiseless) | 50 | +0.01 [+0.00, +0.01] | −0.00 to +0.01 | −0.03 |
| multisensory (noiseless) | 50 | +0.03 [+0.01, +0.05] | −0.00 to +0.07 | −0.08 |

A single VBMC run is optimistic by 0.13 to 0.74 nats on the five
typical noisy conditions and pessimistic by 0.19 on Student D8; the
noiseless controls are within 0.03. Against the raw stacked bias at
`M = 2` (0.25 to 0.9 on the typical noisy conditions), most of the
stack's bias is inherited from its inputs.

Added bias, per condition and `M`: the median over cells of the
stack's bias minus the mean bias of its inputs, for the raw value, the
cap, the run-level shrinkage alone and the two-level full shrinkage;
positive means the stacked estimate is more optimistic than the runs
it was built from. `added.md` in the tracked directory has every
estimate at every `M`, the bootstrap interval on the raw value's added
bias, the bias of the input with the highest reported ELBO and the
fraction of cells whose raw added bias is positive.

| condition | raw, `M` = 2 / 3 / 5 / 16 | cap | run level alone | two-level full | two-level anchored (mean add-back) |
|---|---|---|---|---|---|
| multisensory noise 3 | +0.10 / +0.20 / +0.24 / +0.55 | −0.33 / −0.29 / −0.26 / −0.31 | +0.06 / +0.11 / +0.20 / +0.40 | −0.19 / −0.10 / −0.08 / +0.01 | +0.03 / +0.13 / +0.17 / +0.24 |
| multisensory noise 1.3 | +0.02 / +0.07 / +0.18 / +0.27 | −0.25 / −0.21 / −0.20 / −0.19 | +0.01 / +0.05 / +0.13 / +0.21 | −0.14 / −0.09 / −0.04 / +0.03 | +0.01 / +0.04 / +0.11 / +0.19 |
| Rosenbrock noise 3 | +0.15 / +0.21 / +0.36 / +0.60 | −0.04 / +0.00 / −0.00 / −0.01 | +0.10 / +0.10 / +0.18 / +0.34 | −0.04 / −0.01 / +0.03 / +0.01 | +0.11 / +0.14 / +0.21 / +0.13 |
| GMM noise 3 | +0.08 / +0.14 / +0.19 / +0.50 | −0.16 / −0.24 / −0.16 / −0.14 | +0.06 / +0.09 / +0.16 / +0.34 | −0.16 / −0.23 / −0.15 / −0.08 | +0.03 / +0.02 / +0.07 / +0.16 |
| ring noise 3 | +0.03 / +0.10 / +0.13 / +0.43 | −0.08 / −0.11 / −0.07 / −0.16 | +0.06 / +0.06 / +0.10 / +0.30 | −0.17 / −0.14 / −0.13 / −0.11 | −0.00 / +0.00 / +0.00 / +0.02 |
| Student D8 noise 3 | +0.02 / +0.20 / +0.24 / +0.54 | −0.76 / −0.86 / −1.00 / −1.54 | +0.02 / +0.17 / +0.19 / +0.46 | −0.22 / −0.03 / −0.07 / +0.15 | +0.02 / +0.17 / +0.15 / +0.41 |
| noiseless controls, every `M` | within 0.04 | within 0.09 | within 0.04 | within 0.05 | within 0.05 |

What the yardstick shows:

- **Stacking adds optimism, and the addition grows with `M`.** The
  raw value's added bias is +0.02 to +0.15 nats at `M = 2`, +0.07 to
  +0.21 at `M = 3`, +0.13 to +0.36 at `M = 5` and +0.27 to +0.60 at
  `M = 16` over the six noisy conditions, positive on 80 to 100 % of
  the cells from `M = 3` on (55 to 100 % at `M = 2`). The stack's raw
  bias tracks the bias of the input run with the highest reported
  ELBO within 0.25 nats at every `M`: the stacking's own optimism is
  the winner's curse among the runs (and the components) it was
  given.
- **The cap over-corrects relative to the inputs everywhere.** Its
  added bias is −0.04 to −0.33 on the typical noisy conditions and
  −0.76 to −1.54 on Student D8: the capped headline is less
  optimistic than the runs it was built from, by more than the
  stacking added. Under this yardstick the cap fails on every noisy
  condition, not only on the heavy-tailed one.
- **The run-level term alone is not enough.** The shrinkage of the
  runs' levels, the term aimed at selection among runs, removes only
  15 to 51 % of the raw addition at `M = 5` and 16 to 44 % at
  `M = 16`.
- **The two-level full shrinkage adds nothing, within 0.23 nats.** Its
  added bias lies between −0.23 and +0.15 on every noisy condition at
  every `M`, and within 0.15 at `M = 16`. It gets there by removing
  part of the inputs' own optimism as well, through its within-run
  term: at `M ≤ 5` its added bias is negative on most conditions, by
  0.04 to 0.23 nats, so the stacked headline sits a little below the
  level of its inputs. The within-run term debiases the runs, which
  the requirement does not ask for, and the two effects roughly
  cancel.
- **An estimator that removes only the stacking's selection halves
  the addition and still adds.** The anchored variants of
  `svbmc_shrink_elbo.py` shrink as the two-level estimate does and add
  back what the within-run shrinkage removes at each run's own
  weights, weighted by the runs' masses in the stack (`anchored`,
  `two_level_anchored`) or as their plain mean (`anchored_mean`,
  `two_level_anchored_mean`), so that a stack of one run reports the
  run's own ELBO by construction. With the within-run term alone the
  add-back cancels nearly everything (`anchored` adds +0.06 to +0.32
  at `M = 5` against raw's +0.13 to +0.36): the stacking's addition
  is not a reweighting within runs. With the run-level term the two
  forms differ by at most 0.06 and add +0.00 to +0.21 at `M` = 3 to
  5 and +0.02 to +0.41 at `M = 16`, removing 16 to 99 % of the raw
  addition (nearly all of it on the ring, least on Student and
  multisensory noise 3); at `M = 5` the bootstrap intervals exclude
  zero on four of the six conditions. The run-level term as built is
  too weak to remove the selection among runs by itself: it takes the
  GP's variance of a run's ELBO at fixed weights (`wᵀ Σ w`, about
  0.1) as the level's error and leaves out the run-to-run scatter of
  the runs' own optimism (interquartile ranges of 0.3 to 0.5 nats in
  the single-run table), which is what the stacking selects on. The
  two-level estimate without the add-back ends nearer zero because
  its within-run term removes part of that optimism from every run
  before the levels are compared, and the two effects offset.

## Re-optimizing on the shrunken estimates

The shrinkage above changes the reported value at the weights the raw
optimization chose. `svbmc_shrink_optimize.py` runs the optimization
on the shrunken values instead: for every cell at `M` = 3, 4 and 5
(480 cells, 26 minutes; tracked under
[`experiments/svbmc_pool/shrink_opt_20260915/`](../experiments/svbmc_pool/shrink_opt_20260915/summary.md))
it rebuilds the stack, replaces the class's corrected expected log
joints by the two-level full shrinkage, optimizes the weights with the
comparison's settings, and scores the new stack as a cell is scored,
against its own `elbo_mc`. Medians over cells: the bias of the
shrunken value at the new weights (the headline this variant would
report), of the value-only shrinkage at the raw weights and of the raw
optimization; the added bias of the new headline; the ratio of the
new stack's gsKL to the raw optimization's, with the fraction of
cells it improves; the change in the KL gap; and the mass moved
between components (half the L1 distance between the two weight
vectors).

| condition | bias, `M` = 3 / 4 / 5: re-optimized | value-only | raw | added, re-optimized | gsKL ratio (cells improved) | KL gap change | mass moved |
|---|---|---|---|---|---|---|---|
| multisensory noise 3 | +0.69 / +0.89 / +0.82 | +0.58 / +0.72 / +0.67 | +0.84 / +1.07 / +1.04 | +0.01 / +0.10 / +0.10 | 0.78 (0.85 to 1.00) | −0.04 | 0.33 |
| multisensory noise 1.3 | +0.31 / +0.30 / +0.43 | +0.25 / +0.26 / +0.35 | +0.44 / +0.47 / +0.55 | −0.04 / −0.03 / +0.02 | 0.86 (0.90 to 0.95) | −0.02 | 0.24 |
| Rosenbrock noise 3 | +0.32 / +0.36 / +0.59 | +0.12 / +0.07 / +0.18 | +0.29 / +0.35 / +0.53 | +0.17 / +0.22 / +0.29 | 5.3 (0.15 to 0.25) | +0.09 | 0.48 |
| GMM noise 3 | +0.35 / +0.37 / +0.36 | −0.05 / −0.01 / −0.00 | +0.34 / +0.30 / +0.40 | +0.15 / +0.18 / +0.09 | 0.83 (0.55 to 0.80) | +0.07 | 0.42 |
| ring noise 3 | +0.11 / +0.10 / +0.19 | +0.10 / +0.06 / +0.13 | +0.37 / +0.40 / +0.42 | −0.08 / −0.15 / −0.11 | 0.70 (0.75 to 0.80) | −0.10 | 0.28 |
| Student D8 noise 3 | −0.23 / −0.35 / −0.17 | −0.32 / −0.39 / −0.24 | −0.09 / −0.10 / +0.04 | −0.03 / −0.07 / +0.03 | 1.04 (0.15 to 0.25) | +0.03 | 0.10 |
| noiseless controls | within 0.09 | within 0.08 | within 0.07 | within 0.05 | 0.99 to 1.00 | ≤ 0.004 | ≤ 0.02 |

Re-optimizing recovers part of the optimism the value-only shrinkage
removed. On the two multisensory conditions, Rosenbrock and noisy GMM
the re-optimized headline is 0.06 to 0.4 nats more optimistic than
the value-only one (the median difference over the noisy cells is
+0.08 to +0.13 at the three `M`, and the value-only value is closer
to the truth on 85 to 95 % of those conditions' cells), and its added
bias is +0.09 to +0.29 on Rosenbrock and GMM. The optimizer selects on
the shrunken values as it selected on the raw ones, and the
shrinkage's own errors become the noise to win on: the raw value at
the re-optimized weights is 0.1 to 0.3 nats less optimistic than at
the raw weights, since the new weights select less on the raw noise,
but the shrunken value at those weights sits 0.1 to 0.25 nats above
the raw value there on Rosenbrock and GMM, because the optimizer
found the below-average components the shrinkage had raised toward
their run's mean. On the ring and Student the two headlines agree
within 0.1.

The posterior moves with the weights (a tenth to a half of the mass,
no single weight by more than 0.08), and where it lands depends on
the target: better on the multisensory conditions and the ring (gsKL
down by a fifth to a third and MMTV by a tenth to a fifth on 75 to
100 % of the cells, the KL gap smaller by 0.02 to 0.10 nats),
unchanged on Student, worse on Rosenbrock (gsKL five times and MMTV
twice the raw optimization's, the KL gap larger by 0.09) and on noisy
GMM's KL gap (+0.07 despite a smaller gsKL). The shrinkage therefore
stays a correction of the reported value at the raw weights, as the
cap is applied today; the posterior gains on the real-data target are
noted for later work on the stacking objective, with the losses
elsewhere.

## Limitations

- **The Rosenbrock GPs are numerically fragile.** The runs of that
  condition evaluated far-tail points, and the GPs that fit them have
  kernel matrices with condition numbers up to 3e17, so another
  machine's BLAS recomputes their stored statistics with relative
  deviations up to 0.29 (the campaign plan's worklog of 2026-09-14). The
  comparison is unaffected, since both arms stack the stored statistics,
  which are the cluster's; the Phase 2 estimator, which re-evaluates the
  GPs on the analysis machine, is not, and its Rosenbrock results are to
  be read with that in mind. The PI decided to keep the condition with
  this caveat; the behaviour itself is a TODO item on GP robustness.
- **`M` stops at 16 here.** The paper's protocol goes to `M = 40`; the
  integrated arm alone is to be run at `M = 32` (10 repetitions on
  every condition, on a later overnight or cluster slot), since the
  original's cost at that size is about an hour per condition for no
  equivalence information that `M ≤ 16` has not given.
- **Cost.** Stage D took 8.1 hours rather than the plan's 5, the `M = 16`
  cells dominating; the harness's per-`M` costs above are the figures
  for planning further runs.
- **One machine, one seed, shared runs.** The subsets are those of seed
  0; the paired design makes the arm-to-arm differences robust to that,
  the absolute medians less so, and the bootstrap intervals are over
  the repetitions of a cell set only. The cells of one condition and
  `M` draw from one pool of 100 runs (50 for the controls) and share
  runs, so every interval over cells is optimistic.
- **Medians over 20 cells (10 at `M = 16`).** Differences between rules
  of about 0.1 nats are at the edge of what the cells resolve; the
  shrinkage section quotes paired bootstrap intervals where a
  comparison rests on them.

## Files

- [`experiments/svbmc_pool/stack_20260914/summary.md`](../experiments/svbmc_pool/stack_20260914/summary.md):
  every table of this report with its bootstrap intervals, the entropy
  and reference columns, and the equivalence tests.
- [`experiments/svbmc_pool/stack_20260914/summary.json`](../experiments/svbmc_pool/stack_20260914/summary.json),
  [`sources.json`](../experiments/svbmc_pool/stack_20260914/sources.json).
- The later sections' directories under `experiments/svbmc_pool/`:
  `phase2_20260915/`, `cap_kappa_20260915/`, `shrink_20260915/`,
  `stack_M35_20260915/`, `shrink_M35_20260915/`,
  `cap_kappa_M35_20260915/` and `stack_merged_20260915/`, each
  described in the experiments README; the scoring directories carry
  a `sources.json` with the script's and the cells file's hashes.
- The per-cell outputs of stage D, the Phase 2 scoring and the
  weight-aware caps: the asset `svbmc_pool_20260914_analyses.tar.gz` of
  the draft release `svbmc-analyses-20260915` (the experiments README
  names its size and SHA-256), unpacking to the three raw directories
  under `dev/scripts/runs/`.
- The re-optimization on the shrunken estimates:
  [`experiments/svbmc_pool/shrink_opt_20260915/`](../experiments/svbmc_pool/shrink_opt_20260915/summary.md)
  (`cells.jsonl` with the new weights, the summaries, `sources.json`).
- The single-run baseline and the added-bias join:
  [`experiments/svbmc_pool/single_run_20260915/`](../experiments/svbmc_pool/single_run_20260915/added.md)
  (`runs.jsonl`, `cells.jsonl`, the summaries, `sources.json`).
- The pool: [`experiments/svbmc_pool/pool_20260914/`](../experiments/svbmc_pool/pool_20260914/README.md).
