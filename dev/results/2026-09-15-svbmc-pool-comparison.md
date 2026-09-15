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
headline stays within 0.05 to 0.56 nats, which is what the cap was
written for. The Phase 2 cross-run estimator, scored on the same cells,
does not repair that condition: it is accurate on the noiseless controls
and on Rosenbrock, under-predicts by 0.2 to 0.7 nats on the typical
noisy conditions, and lands as low as the cap on Student D8; and no
weight-aware variant of the cap separates the heavy-tailed case from
the typical one. The `M = 32` cells of the integrated arm are reported
in a section added when that run completes.

Tracked outputs:
[`experiments/svbmc_pool/stack_20260914/`](../experiments/svbmc_pool/stack_20260914/)
(`summary.json`, `summary.md`, `sources.json`; the section "Stacking
comparison, stage D" of the
[experiments README](../experiments/svbmc_pool/README.md) describes
their keys). Every number below is in `summary.md`. The comparison's
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
`debiased_I_median` differ by at most 0.2 nats in the median on every
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
  0.02 to 0.16 nats higher still, because it reports the optimizer's own
  20-draw entropy, which sits above the arm-independent reference on
  every condition (by 0.16 nats at `M = 2` on multisensory noise 3, 0.10
  on Student D8), while the integrated class's fresh 100-draw evaluation
  sits within 0.035 of it.
- **The cap does its job on five of the six noisy conditions**: the
  headline stays within 0.05 to 0.56 nats of the reference at every `M`,
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
  condition have the largest evidence error of the noisy conditions
  (median 0.96 nats) and a median ELBO standard deviation that the cap
  reads as a large uncertainty to remove; what the honest cross-run
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
  both multisensory conditions the honest estimate sits 0.2 to 0.7 nats
  *below* the reference, the shortfall growing with `M` on multisensory
  noise 3, while the cap sits 0.01 to 0.56 above it; on those three
  conditions the cap is the closer estimate at every `M ≥ 4`. On the
  ring the two are comparable, the honest estimate closer at `M ≥ 8`.
- **Student D8 is not repaired.** The honest estimate is as low as the
  cap, 1.2 to 1.5 nats below the reference, while the raw value is
  within 0.3.
- **One mechanism fits both failures on Student.** The runs' GPs carry a
  negative-quadratic mean, which falls off faster than a Student-t log
  density; away from a run's own data, in the tails, the GP's prediction
  sits well below the truth. The median over all components leans on the
  tail components, and the cross-run estimate leans on other runs'
  predictions where they have little data; both are dragged down. The
  raw value is built on the selected central components with their own
  data, and is within 0.3. This is a reading consistent with the
  numbers, not a demonstration; the estimator's decomposition records
  (the data each run had at each component) are where to test it.

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

For `κ ≤ 0.8` the cap binds on at most 30 % of the cells and the
headline is the raw value on every condition; `κ` from 0.9 to 0.99
binds on most cells but leaves 0.3 to 1.0 nats of the optimism on the
typical noisy conditions while still over-correcting Student by 0.3 to
1.2; the weighted median is the raw value everywhere; and on Student
`κ = 0.8` is the least wrong at `M = 8` and `16` (0.14 and 0.18) and
`κ ≤ 0.7` at `M ≤ 4`. The reading is that the class's cap works
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
0.16. That is itself a finding: the stacked value sits *below* the best
input run's own expected log joint on most cells, so the 0.3 to 1.3
nats of optimism are already inside each run's own value, placed there
by VBMC's own variational optimization, which weights components where
its GP estimate is high; stacking adds little cross-run selection on
top. The component-median cap works because the low-weight components
are the ones that optimization did not favour, the least selected-on
estimates the stack holds; the run-median cap (`capped_E_median`, +0.17
to +0.82 at `M = 16` on the typical noisy conditions, −1.13 on Student)
sits between the two.

## Empirical-Bayes shrinkage

Since the optimism is a selection on noisy GP estimates, the textbook
correction is to shrink each component's estimate toward its population
mean by the share of the population's spread that is estimation noise:
with `I_k ~ N(θ_k, V_k)` and `θ_k ~ N(μ, τ²)`, report
`μ + τ² / (τ² + V_k) (I_k − μ)`, where `V_k` is the GP's own variance of
the estimate (the class already carries it in `J_sjk`) and `τ²` the
excess variance of the estimates over the mean `V_k`, by moments.
`svbmc_shrink_elbo.py` scored it on stage D's cells at the recorded
weights (tracked under
[`experiments/svbmc_pool/shrink_20260915/`](../experiments/svbmc_pool/shrink_20260915/)),
the population being the run's own components (`within`, with the
diagonal of the estimation covariance, or `within full` with all of it,
since one GP estimates a run's components jointly) or the whole stack
(`stack`). Median bias against `elbo_mc` at `M = 16`, with the
weight-averaged shrinkage factor in brackets (1 leaves the estimates
alone), and the share of the within-run spread that is estimation noise:

| condition (`M = 16`) | raw | class cap | within | within full | stack | noise share |
|---|---|---|---|---|---|---|
| multisensory noise 3 | +1.31 | +0.44 | +1.10 [0.63] | +0.92 [0.85] | +1.07 [0.70] | 0.45 |
| multisensory noise 1.3 | +0.61 | +0.16 | +0.55 [0.77] | +0.42 [0.86] | +0.55 [0.83] | 0.26 |
| Rosenbrock noise 3 | +0.74 | +0.11 | +0.57 [0.55] | +0.45 [0.81] | +0.49 [0.65] | 0.41 |
| GMM noise 3 | +0.73 | +0.05 | +0.51 [0.60] | +0.25 [0.80] | +0.46 [0.73] | 0.37 |
| ring noise 3 | +0.76 | +0.14 | +0.41 [0.14] | +0.31 [0.33] | +0.34 [0.29] | 1.09 |
| Student D8 noise 3 | +0.27 | −1.78 | +0.22 [0.94] | −0.01 [0.96] | +0.20 [0.96] | 0.08 |
| noiseless controls | within 0.07 | within 0.03 | within 0.09 [1.00] | within 0.09 [1.00] | within 0.08 [1.00] | 0.00 to 0.06 |

Shrinkage is safe where the caps were not: it leaves the noiseless
controls untouched (nothing to shrink), and on Student D8, where the
spread of the components is real (noise share 0.08), it moves the raw
value by at most 0.3 nats and the full-covariance variant lands within
0.01 at `M = 16`. On the typical noisy conditions it removes a fifth to
a half of the optimism (the full-covariance variant the most:
multisensory noise 1.3 from +0.61 to +0.42, noisy GMM from +0.73 to
+0.25) but does not reach the cap, and what it leaves grows with `M`.
The reason is in the noise shares: within a run about a quarter to a
half of the spread of the estimates is estimation noise by the GP's own
account, so the shrinkage factor is 0.6 to 0.85 and the selected
components keep most of their excess. The growth with `M` is a
run-level selection the within-run population cannot see (the stack
puts its mass on the runs whose own values came out highest, and the
best of more runs is more optimistic), and the stack-wide population
does not capture it either, since it reads the runs' different levels
as real signal.

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
| multisensory noise 3 | +0.51 / +0.71 / +0.65 / +0.75 | +0.62 / +0.84 / +0.81 / +0.93 | +0.70 / +1.00 / +1.00 / +1.15 | +0.38 / +0.56 / +0.40 / +0.44 |
| multisensory noise 1.3 | +0.20 / +0.25 / +0.35 / +0.37 | +0.33 / +0.39 / +0.47 / +0.51 | +0.38 / +0.44 / +0.51 / +0.57 | +0.08 / +0.11 / +0.14 / +0.16 |
| Rosenbrock noise 3 | +0.14 / +0.09 / +0.12 / +0.16 | +0.15 / +0.16 / +0.20 / +0.27 | +0.19 / +0.28 / +0.33 / +0.47 | +0.09 / +0.07 / +0.10 / +0.11 |
| GMM noise 3 | +0.08 / −0.02 / +0.03 / +0.11 | +0.21 / +0.14 / +0.22 / +0.35 | +0.29 / +0.28 / +0.40 / +0.56 | +0.07 / −0.01 / +0.06 / +0.05 |
| ring noise 3 | +0.15 / +0.06 / +0.09 / +0.19 | +0.18 / +0.06 / +0.14 / +0.26 | +0.40 / +0.34 / +0.40 / +0.62 | +0.21 / +0.16 / +0.16 / +0.14 |
| Student D8 noise 3 | −0.37 / −0.39 / −0.20 / −0.08 | −0.19 / −0.19 / +0.09 / +0.14 | −0.15 / −0.13 / +0.14 / +0.20 | −0.95 / −1.17 / −1.39 / −1.78 |
| GMM (noiseless) | +0.01 / +0.01 / +0.01 / +0.01 | +0.01 / +0.00 / +0.00 / +0.01 | +0.01 / +0.00 / +0.00 / +0.01 | −0.02 / −0.01 / −0.03 / −0.02 |
| multisensory (noiseless) | +0.06 / +0.06 / +0.09 / +0.07 | +0.04 / +0.05 / +0.08 / +0.06 | +0.03 / +0.04 / +0.07 / +0.05 | −0.05 / −0.02 / +0.03 / +0.02 |

`two_level_full` is the first correction in this report that is
acceptable on every condition: it matches the cap on Rosenbrock, noisy
GMM and the ring, stays within 0.4 nats on Student D8 where the cap is
off by up to 1.8, leaves the noiseless controls alone, and its worst
case, +0.75 on multisensory noise 3 at `M = 16`, compares with the
cap's 1.78 and the raw value's 1.31. It has no tuned constant and uses
only what the class already holds (`I_sk`, `J_sjk`, the runs' own
weights). What it leaves on the two multisensory conditions, 0.2 to
0.75 nats, is the part of the optimism that the GP's own covariance
does not account for, a mean bias of the estimates rather than
variance; nothing keyed on `J` can see it. The run-level shift alone
does little at small `M` and most of its work at `M ≥ 8`, as expected
of a selection among runs; the within-run shrinkage with the full
covariance does most of the rest. Two things remain untested: the
weights were held at the values selected on the unshrunken estimates
(re-optimizing on the shrunken ones would also move the posterior
toward the population mean), and the run-level moments rest on two to
four values at `M ≤ 4`.

**At the `M` users run.** Stacks of three to five runs are the common
case, so the small-`M` cells decide. There the run-level term is
unreliable and nearly irrelevant: with two to four runs its
moment-matched `τ²` is zero in 40 to 65 % of the cells (a full shrink to
the runs' mean) and 0.4 to 0.8 otherwise, a coin flip, yet
`two_level_full` and `within_full` differ by at most 0.06 nats on every
condition at `M = 2` and `4`, so the within-run full-covariance
shrinkage does the work at small `M` without depending on that
estimate. Cell by cell the corrections scatter alike (10th to 90th
percentile ranges of 0.5 to 0.7 nats for raw, cap and shrinkage on the
noisy conditions). Worst case over the six noisy conditions, median
bias in magnitude: at `M = 2`, raw 0.77, cap 0.95, `within_full` 0.56,
`two_level_full` 0.51; at `M = 4`, raw 1.07, cap 1.17, `within_full`
0.82, `two_level_full` 0.71. The cap remains 0.1 to 0.25 nats closer on
the two multisensory conditions and 0.6 to 0.8 further on Student D8.
The grid holds no `M = 3` or `5` cells; the integrated arm alone would
score them in minutes.

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
  integrated arm alone is being run at `M = 32` (10 repetitions on every
  condition), since the original's cost at that size is about an hour
  per condition for no equivalence information that `M ≤ 16` has not
  given.
- **Cost.** Stage D took 8.1 hours rather than the plan's 5, the `M = 16`
  cells dominating; the harness's per-`M` costs above are the figures
  for planning further runs.
- **One machine, one seed.** The subsets are those of seed 0; the paired
  design makes the arm-to-arm differences robust to that, the absolute
  medians less so, and the bootstrap intervals are over the repetitions
  of a cell only.

## Files

- [`experiments/svbmc_pool/stack_20260914/summary.md`](../experiments/svbmc_pool/stack_20260914/summary.md):
  every table of this report with its bootstrap intervals, the entropy
  and reference columns, and the equivalence tests.
- [`experiments/svbmc_pool/stack_20260914/summary.json`](../experiments/svbmc_pool/stack_20260914/summary.json),
  [`sources.json`](../experiments/svbmc_pool/stack_20260914/sources.json).
- The per-cell outputs of stage D, the Phase 2 scoring and the
  weight-aware caps: the asset `svbmc_pool_20260914_analyses.tar.gz` of
  the draft release `svbmc-analyses-20260915` (the experiments README
  names its size and SHA-256), unpacking to the three raw directories
  under `dev/scripts/runs/`.
- The pool: [`experiments/svbmc_pool/pool_20260914/`](../experiments/svbmc_pool/pool_20260914/README.md).
