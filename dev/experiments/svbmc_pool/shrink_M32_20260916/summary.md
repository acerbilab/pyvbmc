# Empirical-Bayes shrinkage of the stacked expected log joint

Generated 2026-09-16 01:56:29 from 80 cells of the integrated arm. Medians over the cells of a condition and `M` of the bias against `elbo_mc`: `raw` the uncapped value, `class` the component-median cap the class applies, then each shrinkage variant with, in brackets, the weight-averaged shrinkage factor (for the full-covariance forms the row sum of the shrinkage matrix; 1 leaves the estimates unchanged, 0 replaces them by the population mean). `noise share` is the mean estimation variance of the components over the spread of their estimates, the share of the spread that is noise, within a run, over the stack's components and over the runs' levels. `hybrid` is the class's cap when the cell's noise share is at least 0.2 and `within_full` otherwise; its bracket is the fraction of cells on which it chose the cap.

| condition | M | raw | class | within [factor] | within_full [factor] | stack [factor] | run_level [factor] | two_level [factor] | two_level_full [factor] | anchored [factor] | two_level_anchored [factor] | anchored_mean [factor] | two_level_anchored_mean [factor] | hybrid [factor] | noise share within / stack / runs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| multisensory_s1_D6_noise3 | 32 | +1.45 | +0.45 | +1.22 [0.61] | +1.06 [0.09] | +1.14 [0.68] | +1.25 [0.49] | +1.03 [0.30] | +0.86 [0.05] | +1.33 [0.09] | +1.12 [0.05] | +1.31 [0.09] | +1.11 [0.05] | +0.45 [1.00] | 0.34 / 0.38 / 0.51 |
| multisensory_s1_D6_noise1.3 | 32 | +0.71 | +0.19 | +0.64 [0.78] | +0.51 [0.25] | +0.63 [0.82] | +0.64 [0.56] | +0.57 [0.43] | +0.43 [0.14] | +0.68 [0.25] | +0.61 [0.14] | +0.67 [0.25] | +0.59 [0.14] | +0.19 [0.90] | 0.21 / 0.22 / 0.44 |
| rosenbrock_D2_noise3 | 32 | +0.91 | +0.06 | +0.75 [0.60] | +0.46 [0.07] | +0.55 [0.67] | +0.47 [0.09] | +0.31 [0.03] | +0.06 [0.01] | +0.59 [0.07] | +0.22 [0.01] | +0.60 [0.07] | +0.18 [0.01] | +0.06 [1.00] | 0.20 / 0.33 / 0.89 |
| gmm_D2_noise3 | 32 | +0.92 | +0.04 | +0.74 [0.69] | +0.46 [0.18] | +0.67 [0.75] | +0.67 [0.40] | +0.50 [0.29] | +0.21 [0.08] | +0.74 [0.18] | +0.51 [0.08] | +0.70 [0.18] | +0.46 [0.08] | +0.04 [1.00] | 0.23 / 0.29 / 0.61 |
| ring_D2_noise3 | 32 | +0.84 | +0.13 | +0.38 [0.22] | +0.29 [0.03] | +0.23 [0.19] | +0.64 [0.11] | +0.21 [0.03] | +0.10 [0.00] | +0.46 [0.03] | +0.28 [0.00] | +0.41 [0.03] | +0.23 [0.00] | +0.13 [1.00] | 0.88 / 0.84 / 0.90 |
| student_D8_noise3 | 32 | +0.38 | -1.85 | +0.32 [0.94] | +0.07 [0.36] | +0.28 [0.95] | +0.26 [0.92] | +0.21 [0.86] | -0.06 [0.34] | +0.35 [0.36] | +0.24 [0.34] | +0.32 [0.36] | +0.19 [0.34] | +0.07 [0.00] | 0.05 / 0.05 / 0.08 |
| gmm_D2 | 32 | +0.01 | +0.00 | +0.01 [1.00] | +0.01 [1.00] | +0.01 [1.00] | +0.01 [0.99] | +0.01 [0.99] | +0.01 [0.99] | +0.01 [1.00] | +0.01 [0.99] | +0.01 [1.00] | +0.01 [0.99] | +0.01 [0.00] | 0.00 / 0.01 / 0.01 |
| multisensory_s1_D6 | 32 | +0.14 | +0.14 | +0.16 [0.93] | +0.18 [0.83] | +0.16 [0.93] | +0.14 [0.45] | +0.16 [0.42] | +0.18 [0.39] | +0.16 [0.83] | +0.16 [0.39] | +0.16 [0.83] | +0.16 [0.39] | +0.18 [0.00] | 0.05 / 0.08 / 0.62 |

Median absolute bias over the cells of each condition and `M`:

| condition | M | raw | class | within | within_full | stack | run_level | two_level | two_level_full | anchored | two_level_anchored | anchored_mean | two_level_anchored_mean | hybrid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| multisensory_s1_D6_noise3 | 32 | 1.45 | 0.45 | 1.22 | 1.06 | 1.14 | 1.25 | 1.03 | 0.86 | 1.33 | 1.12 | 1.31 | 1.11 | 0.45 |
| multisensory_s1_D6_noise1.3 | 32 | 0.71 | 0.19 | 0.64 | 0.51 | 0.63 | 0.64 | 0.57 | 0.43 | 0.68 | 0.61 | 0.67 | 0.59 | 0.19 |
| rosenbrock_D2_noise3 | 32 | 0.91 | 0.06 | 0.75 | 0.46 | 0.55 | 0.47 | 0.31 | 0.18 | 0.59 | 0.28 | 0.60 | 0.26 | 0.07 |
| gmm_D2_noise3 | 32 | 0.92 | 0.04 | 0.74 | 0.46 | 0.67 | 0.67 | 0.50 | 0.21 | 0.74 | 0.51 | 0.70 | 0.46 | 0.05 |
| ring_D2_noise3 | 32 | 0.84 | 0.13 | 0.38 | 0.29 | 0.23 | 0.64 | 0.21 | 0.10 | 0.46 | 0.28 | 0.41 | 0.23 | 0.13 |
| student_D8_noise3 | 32 | 0.38 | 1.85 | 0.32 | 0.07 | 0.28 | 0.26 | 0.21 | 0.07 | 0.35 | 0.24 | 0.32 | 0.19 | 0.07 |
| gmm_D2 | 32 | 0.01 | 0.00 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| multisensory_s1_D6 | 32 | 0.14 | 0.14 | 0.16 | 0.18 | 0.16 | 0.14 | 0.16 | 0.18 | 0.16 | 0.16 | 0.16 | 0.16 | 0.18 |
