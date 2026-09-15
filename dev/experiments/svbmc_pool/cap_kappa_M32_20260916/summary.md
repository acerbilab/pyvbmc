# Weight-aware caps on the stacked expected log joint

Generated 2026-09-16 01:57:20 from 80 cells of the integrated arm. Each entry is the median over the cells of a condition and `M` of the bias against the cell's `elbo_mc`; `raw` is the uncapped value, `class` the cap the integrated class applies (every component; equal to `kappa_1`), `kappa_x` the cap at the median of the top-weight components carrying mass `x`, `wmed` the weighted median of the components' expected log joints, `E_max` the cap at the largest run-level expected log joint, `E_top` at that of the run carrying the largest stacking mass, `classE` the class's run-median cap (`capped_E_median`). In brackets, the fraction of cells on which the cap binds. A cap at the largest component expected log joint would bind on 0 of 80 cells: it is the raw value.

| condition | M | raw | class | classE | κ 0.5 | κ 0.6 | κ 0.7 | κ 0.8 | κ 0.9 | κ 0.95 | κ 0.99 | κ 1 | wmed | E_max | E_top |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| multisensory_s1_D6_noise3 | 32 | +1.45 | +0.45 | +0.77 | +1.45 [0.00] | +1.45 [0.00] | +1.45 [0.00] | +1.45 [0.40] | +1.23 [1.00] | +1.08 [1.00] | +0.73 [1.00] | +0.45 [1.00] | +1.45 [0.00] | +1.45 [0.00] | +1.44 [0.10] |
| multisensory_s1_D6_noise1.3 | 32 | +0.71 | +0.19 | +0.43 | +0.71 [0.10] | +0.71 [0.00] | +0.71 [0.00] | +0.68 [0.60] | +0.59 [0.90] | +0.54 [1.00] | +0.38 [1.00] | +0.19 [1.00] | +0.71 [0.00] | +0.71 [0.00] | +0.70 [0.30] |
| rosenbrock_D2_noise3 | 32 | +0.91 | +0.06 | +0.12 | +0.91 [0.00] | +0.91 [0.00] | +0.90 [0.10] | +0.87 [0.30] | +0.80 [0.70] | +0.67 [1.00] | +0.35 [1.00] | +0.06 [1.00] | +0.90 [0.10] | +0.91 [0.20] | +0.90 [0.40] |
| gmm_D2_noise3 | 32 | +0.92 | +0.04 | +0.18 | +0.92 [0.00] | +0.92 [0.00] | +0.92 [0.10] | +0.92 [0.10] | +0.79 [0.80] | +0.63 [1.00] | +0.41 [1.00] | +0.04 [1.00] | +0.92 [0.00] | +0.92 [0.10] | +0.92 [0.20] |
| ring_D2_noise3 | 32 | +0.84 | +0.13 | +0.22 | +0.84 [0.00] | +0.84 [0.20] | +0.83 [0.20] | +0.83 [0.40] | +0.79 [0.90] | +0.71 [1.00] | +0.37 [1.00] | +0.13 [1.00] | +0.82 [0.50] | +0.83 [0.10] | +0.54 [0.80] |
| student_D8_noise3 | 32 | +0.38 | -1.85 | -1.28 | +0.38 [0.00] | +0.38 [0.00] | +0.38 [0.10] | +0.15 [0.80] | -0.20 [1.00] | -0.62 [1.00] | -1.16 [1.00] | -1.85 [1.00] | +0.38 [0.00] | +0.38 [0.00] | +0.38 [0.10] |
| gmm_D2 | 32 | +0.01 | +0.00 | +0.01 | +0.01 [0.00] | +0.01 [0.00] | +0.01 [0.00] | +0.01 [0.00] | +0.01 [0.00] | +0.01 [0.00] | +0.01 [0.00] | +0.00 [0.60] | +0.01 [0.00] | +0.01 [0.00] | +0.01 [0.50] |
| multisensory_s1_D6 | 32 | +0.14 | +0.14 | +0.14 | +0.14 [0.10] | +0.14 [0.00] | +0.14 [0.00] | +0.14 [0.00] | +0.14 [0.00] | +0.14 [0.00] | +0.14 [0.00] | +0.14 [0.30] | +0.14 [0.00] | +0.14 [0.00] | +0.14 [0.20] |

Median absolute bias over the cells of each condition and `M`, the quantity a headline should make small:

| condition | M | raw | class | classE | κ 0.5 | κ 0.6 | κ 0.7 | κ 0.8 | κ 0.9 | κ 0.95 | κ 0.99 | κ 1 | wmed | E_max | E_top |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| multisensory_s1_D6_noise3 | 32 | 1.45 | 0.45 | 0.77 | 1.45 | 1.45 | 1.45 | 1.45 | 1.23 | 1.08 | 0.73 | 0.45 | 1.45 | 1.45 | 1.44 |
| multisensory_s1_D6_noise1.3 | 32 | 0.71 | 0.19 | 0.43 | 0.71 | 0.71 | 0.71 | 0.68 | 0.59 | 0.54 | 0.38 | 0.19 | 0.71 | 0.71 | 0.70 |
| rosenbrock_D2_noise3 | 32 | 0.91 | 0.06 | 0.12 | 0.91 | 0.91 | 0.90 | 0.87 | 0.80 | 0.67 | 0.35 | 0.07 | 0.90 | 0.91 | 0.90 |
| gmm_D2_noise3 | 32 | 0.92 | 0.04 | 0.18 | 0.92 | 0.92 | 0.92 | 0.92 | 0.79 | 0.63 | 0.41 | 0.05 | 0.92 | 0.92 | 0.92 |
| ring_D2_noise3 | 32 | 0.84 | 0.13 | 0.22 | 0.84 | 0.84 | 0.83 | 0.83 | 0.79 | 0.71 | 0.37 | 0.13 | 0.82 | 0.83 | 0.54 |
| student_D8_noise3 | 32 | 0.38 | 1.85 | 1.28 | 0.38 | 0.38 | 0.38 | 0.15 | 0.20 | 0.62 | 1.16 | 1.85 | 0.38 | 0.38 | 0.38 |
| gmm_D2 | 32 | 0.01 | 0.00 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| multisensory_s1_D6 | 32 | 0.14 | 0.14 | 0.14 | 0.14 | 0.14 | 0.14 | 0.14 | 0.14 | 0.14 | 0.14 | 0.14 | 0.14 | 0.14 | 0.14 |
