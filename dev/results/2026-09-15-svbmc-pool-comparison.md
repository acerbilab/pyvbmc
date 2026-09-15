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
written for. This report covers stage D; the Phase 2 scoring of its
cells by the cross-run estimator and the `M = 32` cells of the
integrated arm are reported in sections added as they complete.

Tracked outputs:
[`experiments/svbmc_pool/stack_20260914/`](../experiments/svbmc_pool/stack_20260914/)
(`summary.json`, `summary.md`, `sources.json`; the section "Stacking
comparison, stage D" of the
[experiments README](../experiments/svbmc_pool/README.md) describes
their keys). Every number below is in `summary.md`. The comparison's
`results.json` (17.6 MB, every cell) stays with the raw directory on
the analysis machine, listed in its gitignored
`dev/scripts/runs/LOCAL.md`; it is the input of the Phase 2 scoring and
of `--summarize-only`.

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
- The raw directory `dev/scripts/runs/svbmc_pool_20260914_stack/` on the
  analysis machine: `results.json`, `cells.jsonl`, the controller's log,
  `original_arm.log`.
- The pool: [`experiments/svbmc_pool/pool_20260914/`](../experiments/svbmc_pool/pool_20260914/README.md).
