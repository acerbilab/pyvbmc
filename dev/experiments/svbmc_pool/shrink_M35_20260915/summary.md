# Empirical-Bayes shrinkage of the stacked expected log joint

Generated 2026-09-15 13:09:51 from 320 cells of the integrated arm. Medians over the cells of a condition and `M` of the bias against `elbo_mc`: `raw` the uncapped value, `class` the component-median cap the class applies, then each shrinkage variant with, in brackets, the weight-averaged shrinkage factor (for the full-covariance forms the row sum of the shrinkage matrix; 1 leaves the estimates unchanged, 0 replaces them by the population mean). `noise share` is the mean estimation variance of the components over the spread of their estimates, the share of the spread that is noise, within a run, over the stack's components and over the runs' levels. `hybrid` is the class's cap when the cell's noise share is at least 0.2 and `within_full` otherwise; its bracket is the fraction of cells on which it chose the cap.

| condition | M | raw | class | within [factor] | within_full [factor] | stack [factor] | run_level [factor] | two_level [factor] | two_level_full [factor] | hybrid [factor] | noise share within / stack / runs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| multisensory_s1_D6_noise3 | 3 | +0.84 | +0.42 | +0.75 [0.65] | +0.60 [0.09] | +0.74 [0.70] | +0.82 [0.41] | +0.73 [0.22] | +0.58 [0.02] | +0.42 [1.00] | 0.36 / 0.35 / 0.60 |
| multisensory_s1_D6_noise3 | 5 | +1.04 | +0.43 | +0.92 [0.62] | +0.75 [0.10] | +0.86 [0.66] | +0.94 [0.36] | +0.81 [0.23] | +0.67 [0.04] | +0.43 [1.00] | 0.35 / 0.38 / 0.64 |
| multisensory_s1_D6_noise1.3 | 3 | +0.44 | +0.16 | +0.42 [0.78] | +0.28 [0.26] | +0.40 [0.81] | +0.41 [0.08] | +0.39 [0.06] | +0.25 [0.02] | +0.21 [0.70] | 0.21 / 0.22 / 0.97 |
| multisensory_s1_D6_noise1.3 | 5 | +0.55 | +0.19 | +0.50 [0.80] | +0.37 [0.26] | +0.50 [0.83] | +0.52 [0.42] | +0.48 [0.33] | +0.35 [0.10] | +0.28 [0.50] | 0.19 / 0.21 / 0.58 |
| rosenbrock_D2_noise3 | 3 | +0.29 | +0.08 | +0.25 [0.69] | +0.20 [0.09] | +0.23 [0.74] | +0.25 [0.00] | +0.19 [0.00] | +0.12 [0.00] | +0.13 [0.55] | 0.17 / 0.24 / 1.38 |
| rosenbrock_D2_noise3 | 5 | +0.53 | +0.13 | +0.42 [0.65] | +0.31 [0.08] | +0.40 [0.71] | +0.37 [0.15] | +0.28 [0.08] | +0.18 [0.03] | +0.21 [0.80] | 0.20 / 0.32 / 0.82 |
| gmm_D2_noise3 | 3 | +0.34 | +0.02 | +0.22 [0.69] | +0.01 [0.17] | +0.18 [0.76] | +0.27 [0.44] | +0.19 [0.26] | -0.05 [0.04] | +0.01 [0.85] | 0.24 / 0.27 / 0.54 |
| gmm_D2_noise3 | 5 | +0.40 | +0.01 | +0.26 [0.69] | +0.06 [0.14] | +0.25 [0.76] | +0.32 [0.39] | +0.18 [0.20] | -0.00 [0.04] | +0.01 [0.85] | 0.21 / 0.28 / 0.60 |
| ring_D2_noise3 | 3 | +0.37 | +0.09 | +0.21 [0.17] | +0.14 [0.03] | +0.17 [0.24] | +0.33 [0.05] | +0.18 [0.00] | +0.10 [0.00] | +0.09 [1.00] | 0.87 / 0.79 / 1.02 |
| ring_D2_noise3 | 5 | +0.42 | +0.22 | +0.26 [0.17] | +0.19 [0.03] | +0.22 [0.21] | +0.39 [0.00] | +0.19 [0.00] | +0.13 [0.00] | +0.22 [1.00] | 0.89 / 0.81 / 1.22 |
| student_D8_noise3 | 3 | -0.09 | -0.97 | -0.12 [0.93] | -0.28 [0.33] | -0.15 [0.95] | -0.12 [0.94] | -0.16 [0.86] | -0.32 [0.29] | -0.28 [0.00] | 0.05 / 0.05 / 0.06 |
| student_D8_noise3 | 5 | +0.04 | -1.26 | +0.00 [0.93] | -0.18 [0.32] | -0.01 [0.95] | +0.01 [0.91] | -0.03 [0.85] | -0.24 [0.30] | -0.18 [0.00] | 0.05 / 0.06 / 0.09 |
| gmm_D2 | 3 | -0.01 | -0.04 | -0.01 [1.00] | -0.00 [1.00] | -0.01 [1.00] | -0.01 [0.99] | -0.01 [0.99] | -0.00 [0.99] | -0.00 [0.00] | 0.00 / 0.01 / 0.01 |
| gmm_D2 | 5 | +0.01 | -0.01 | +0.01 [1.00] | +0.02 [1.00] | +0.01 [1.00] | +0.01 [0.99] | +0.01 [0.99] | +0.02 [0.99] | +0.02 [0.00] | 0.00 / 0.01 / 0.01 |
| multisensory_s1_D6 | 3 | +0.07 | -0.02 | +0.07 [0.96] | +0.08 [0.88] | +0.07 [0.97] | +0.07 [0.34] | +0.07 [0.34] | +0.08 [0.31] | +0.08 [0.10] | 0.05 / 0.08 / 0.68 |
| multisensory_s1_D6 | 5 | +0.07 | +0.00 | +0.07 [0.96] | +0.08 [0.88] | +0.07 [0.96] | +0.06 [0.34] | +0.07 [0.33] | +0.08 [0.30] | +0.08 [0.00] | 0.05 / 0.07 / 0.69 |

Median absolute bias over the cells of each condition and `M`:

| condition | M | raw | class | within | within_full | stack | run_level | two_level | two_level_full | hybrid |
|---|---|---|---|---|---|---|---|---|---|---|
| multisensory_s1_D6_noise3 | 3 | 0.84 | 0.42 | 0.75 | 0.60 | 0.74 | 0.82 | 0.73 | 0.58 | 0.42 |
| multisensory_s1_D6_noise3 | 5 | 1.04 | 0.43 | 0.92 | 0.75 | 0.86 | 0.94 | 0.81 | 0.67 | 0.43 |
| multisensory_s1_D6_noise1.3 | 3 | 0.44 | 0.16 | 0.42 | 0.28 | 0.40 | 0.41 | 0.39 | 0.25 | 0.23 |
| multisensory_s1_D6_noise1.3 | 5 | 0.55 | 0.19 | 0.50 | 0.37 | 0.50 | 0.52 | 0.48 | 0.35 | 0.28 |
| rosenbrock_D2_noise3 | 3 | 0.29 | 0.08 | 0.25 | 0.22 | 0.23 | 0.25 | 0.19 | 0.16 | 0.15 |
| rosenbrock_D2_noise3 | 5 | 0.53 | 0.13 | 0.42 | 0.33 | 0.40 | 0.37 | 0.28 | 0.20 | 0.23 |
| gmm_D2_noise3 | 3 | 0.34 | 0.02 | 0.22 | 0.10 | 0.18 | 0.27 | 0.19 | 0.12 | 0.13 |
| gmm_D2_noise3 | 5 | 0.40 | 0.01 | 0.26 | 0.09 | 0.25 | 0.32 | 0.18 | 0.16 | 0.18 |
| ring_D2_noise3 | 3 | 0.37 | 0.09 | 0.21 | 0.16 | 0.17 | 0.33 | 0.18 | 0.13 | 0.10 |
| ring_D2_noise3 | 5 | 0.42 | 0.22 | 0.26 | 0.19 | 0.22 | 0.39 | 0.19 | 0.13 | 0.22 |
| student_D8_noise3 | 3 | 0.09 | 0.97 | 0.28 | 0.29 | 0.28 | 0.26 | 0.29 | 0.32 | 0.29 |
| student_D8_noise3 | 5 | 0.04 | 1.26 | 0.20 | 0.18 | 0.19 | 0.20 | 0.19 | 0.24 | 0.18 |
| gmm_D2 | 3 | 0.01 | 0.04 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| gmm_D2 | 5 | 0.01 | 0.01 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| multisensory_s1_D6 | 3 | 0.07 | 0.02 | 0.07 | 0.08 | 0.07 | 0.07 | 0.07 | 0.08 | 0.08 |
| multisensory_s1_D6 | 5 | 0.07 | 0.00 | 0.07 | 0.08 | 0.07 | 0.06 | 0.07 | 0.08 | 0.08 |
