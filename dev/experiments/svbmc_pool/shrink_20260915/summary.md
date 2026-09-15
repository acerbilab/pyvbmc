# Empirical-Bayes shrinkage of the stacked expected log joint

Generated 2026-09-15 09:30:54 from 560 cells of the integrated arm. Medians over the cells of a condition and `M` of the bias against `elbo_mc`: `raw` the uncapped value, `class` the component-median cap the class applies, then each shrinkage variant with, in brackets, the weight-averaged shrinkage factor (1 leaves the estimates unchanged, 0 replaces them by the population mean). `noise share` is the mean estimation variance of the components over the spread of their estimates, the share of the spread that is noise, within a run and over the stack.

| condition | M | raw | class | within [factor] | within full [factor] | stack [factor] | noise share within / stack |
|---|---|---|---|---|---|---|---|
| multisensory_s1_D6_noise3 | 2 | +0.77 | +0.38 | +0.67 [0.53] | +0.56 [0.81] | +0.62 [0.63] | 0.55 / 0.47 |
| multisensory_s1_D6_noise3 | 4 | +1.07 | +0.56 | +0.93 [0.61] | +0.82 [0.85] | +0.92 [0.68] | 0.45 / 0.38 |
| multisensory_s1_D6_noise3 | 8 | +1.16 | +0.40 | +0.98 [0.61] | +0.77 [0.85] | +0.93 [0.70] | 0.49 / 0.38 |
| multisensory_s1_D6_noise3 | 16 | +1.31 | +0.44 | +1.10 [0.63] | +0.92 [0.85] | +1.07 [0.70] | 0.45 / 0.38 |
| multisensory_s1_D6_noise1.3 | 2 | +0.41 | +0.08 | +0.35 [0.77] | +0.20 [0.87] | +0.35 [0.80] | 0.27 / 0.26 |
| multisensory_s1_D6_noise1.3 | 4 | +0.47 | +0.11 | +0.42 [0.82] | +0.28 [0.89] | +0.42 [0.85] | 0.23 / 0.20 |
| multisensory_s1_D6_noise1.3 | 8 | +0.56 | +0.14 | +0.52 [0.79] | +0.40 [0.88] | +0.50 [0.82] | 0.24 / 0.23 |
| multisensory_s1_D6_noise1.3 | 16 | +0.61 | +0.16 | +0.55 [0.77] | +0.42 [0.86] | +0.55 [0.83] | 0.26 / 0.22 |
| rosenbrock_D2_noise3 | 2 | +0.25 | +0.09 | +0.23 [0.59] | +0.20 [0.90] | +0.20 [0.72] | 0.36 / 0.32 |
| rosenbrock_D2_noise3 | 4 | +0.35 | +0.07 | +0.25 [0.55] | +0.23 [0.83] | +0.23 [0.68] | 0.4 / 0.41 |
| rosenbrock_D2_noise3 | 8 | +0.53 | +0.10 | +0.41 [0.54] | +0.30 [0.68] | +0.37 [0.67] | 0.4 / 0.38 |
| rosenbrock_D2_noise3 | 16 | +0.74 | +0.11 | +0.57 [0.55] | +0.45 [0.81] | +0.49 [0.65] | 0.41 / 0.42 |
| gmm_D2_noise3 | 2 | +0.30 | +0.07 | +0.20 [0.63] | +0.10 [0.88] | +0.23 [0.76] | 0.37 / 0.29 |
| gmm_D2_noise3 | 4 | +0.30 | -0.01 | +0.18 [0.50] | +0.01 [0.85] | +0.20 [0.69] | 0.4 / 0.36 |
| gmm_D2_noise3 | 8 | +0.50 | +0.06 | +0.34 [0.54] | +0.17 [0.86] | +0.33 [0.71] | 0.41 / 0.33 |
| gmm_D2_noise3 | 16 | +0.73 | +0.05 | +0.51 [0.60] | +0.25 [0.80] | +0.46 [0.73] | 0.37 / 0.30 |
| ring_D2_noise3 | 2 | +0.41 | +0.21 | +0.19 [0.06] | +0.16 [0.32] | +0.20 [0.02] | 1.15 / 0.98 |
| ring_D2_noise3 | 4 | +0.40 | +0.16 | +0.20 [0.09] | +0.14 [0.31] | +0.17 [0.06] | 1.09 / 0.97 |
| ring_D2_noise3 | 8 | +0.49 | +0.16 | +0.21 [0.13] | +0.16 [0.34] | +0.18 [0.17] | 1.1 / 0.85 |
| ring_D2_noise3 | 16 | +0.76 | +0.14 | +0.41 [0.14] | +0.31 [0.33] | +0.34 [0.29] | 1.09 / 0.75 |
| student_D8_noise3 | 2 | -0.11 | -0.95 | -0.15 [0.93] | -0.34 [0.96] | -0.16 [0.95] | 0.09 / 0.06 |
| student_D8_noise3 | 4 | -0.10 | -1.17 | -0.15 [0.92] | -0.35 [0.96] | -0.17 [0.95] | 0.09 / 0.06 |
| student_D8_noise3 | 8 | +0.22 | -1.39 | +0.16 [0.93] | -0.09 [0.96] | +0.12 [0.95] | 0.08 / 0.06 |
| student_D8_noise3 | 16 | +0.27 | -1.78 | +0.22 [0.94] | -0.01 [0.96] | +0.20 [0.96] | 0.08 / 0.05 |
| gmm_D2 | 2 | +0.01 | -0.02 | +0.01 [1.00] | +0.01 [1.00] | +0.01 [1.00] | 0.0 / 0.00 |
| gmm_D2 | 4 | +0.00 | -0.01 | +0.00 [1.00] | +0.01 [1.00] | +0.00 [1.00] | 0.0 / 0.01 |
| gmm_D2 | 8 | +0.00 | -0.03 | +0.00 [1.00] | +0.01 [1.00] | +0.00 [1.00] | 0.0 / 0.01 |
| gmm_D2 | 16 | +0.01 | -0.02 | +0.01 [1.00] | +0.01 [1.00] | +0.01 [1.00] | 0.0 / 0.01 |
| multisensory_s1_D6 | 2 | +0.03 | -0.05 | +0.04 [0.96] | +0.06 [0.97] | +0.04 [0.96] | 0.05 / 0.07 |
| multisensory_s1_D6 | 4 | +0.04 | -0.02 | +0.05 [0.96] | +0.06 [0.96] | +0.05 [0.97] | 0.06 / 0.08 |
| multisensory_s1_D6 | 8 | +0.07 | +0.03 | +0.09 [0.95] | +0.09 [0.96] | +0.08 [0.96] | 0.05 / 0.08 |
| multisensory_s1_D6 | 16 | +0.05 | +0.02 | +0.06 [0.95] | +0.07 [0.96] | +0.06 [0.96] | 0.06 / 0.08 |

Median absolute bias over the cells of each condition and `M`:

| condition | M | raw | class | within | within full | stack |
|---|---|---|---|---|---|---|
| multisensory_s1_D6_noise3 | 2 | 0.77 | 0.38 | 0.67 | 0.56 | 0.62 |
| multisensory_s1_D6_noise3 | 4 | 1.07 | 0.56 | 0.93 | 0.82 | 0.92 |
| multisensory_s1_D6_noise3 | 8 | 1.16 | 0.40 | 0.98 | 0.77 | 0.93 |
| multisensory_s1_D6_noise3 | 16 | 1.31 | 0.44 | 1.10 | 0.92 | 1.07 |
| multisensory_s1_D6_noise1.3 | 2 | 0.41 | 0.08 | 0.35 | 0.20 | 0.35 |
| multisensory_s1_D6_noise1.3 | 4 | 0.47 | 0.11 | 0.42 | 0.28 | 0.42 |
| multisensory_s1_D6_noise1.3 | 8 | 0.56 | 0.14 | 0.52 | 0.40 | 0.50 |
| multisensory_s1_D6_noise1.3 | 16 | 0.61 | 0.16 | 0.55 | 0.42 | 0.55 |
| rosenbrock_D2_noise3 | 2 | 0.25 | 0.09 | 0.25 | 0.20 | 0.21 |
| rosenbrock_D2_noise3 | 4 | 0.35 | 0.07 | 0.25 | 0.23 | 0.23 |
| rosenbrock_D2_noise3 | 8 | 0.53 | 0.10 | 0.41 | 0.30 | 0.37 |
| rosenbrock_D2_noise3 | 16 | 0.74 | 0.11 | 0.57 | 0.45 | 0.49 |
| gmm_D2_noise3 | 2 | 0.30 | 0.07 | 0.20 | 0.18 | 0.23 |
| gmm_D2_noise3 | 4 | 0.30 | 0.01 | 0.20 | 0.14 | 0.20 |
| gmm_D2_noise3 | 8 | 0.50 | 0.06 | 0.34 | 0.17 | 0.33 |
| gmm_D2_noise3 | 16 | 0.73 | 0.05 | 0.51 | 0.25 | 0.46 |
| ring_D2_noise3 | 2 | 0.41 | 0.21 | 0.23 | 0.23 | 0.25 |
| ring_D2_noise3 | 4 | 0.40 | 0.16 | 0.20 | 0.20 | 0.17 |
| ring_D2_noise3 | 8 | 0.49 | 0.16 | 0.21 | 0.16 | 0.18 |
| ring_D2_noise3 | 16 | 0.76 | 0.14 | 0.41 | 0.31 | 0.34 |
| student_D8_noise3 | 2 | 0.11 | 0.95 | 0.20 | 0.34 | 0.20 |
| student_D8_noise3 | 4 | 0.10 | 1.17 | 0.22 | 0.35 | 0.23 |
| student_D8_noise3 | 8 | 0.22 | 1.39 | 0.20 | 0.12 | 0.17 |
| student_D8_noise3 | 16 | 0.27 | 1.78 | 0.22 | 0.09 | 0.20 |
| gmm_D2 | 2 | 0.01 | 0.02 | 0.02 | 0.02 | 0.02 |
| gmm_D2 | 4 | 0.00 | 0.01 | 0.01 | 0.01 | 0.01 |
| gmm_D2 | 8 | 0.00 | 0.03 | 0.01 | 0.01 | 0.01 |
| gmm_D2 | 16 | 0.01 | 0.02 | 0.01 | 0.01 | 0.01 |
| multisensory_s1_D6 | 2 | 0.03 | 0.05 | 0.04 | 0.06 | 0.04 |
| multisensory_s1_D6 | 4 | 0.04 | 0.02 | 0.05 | 0.06 | 0.05 |
| multisensory_s1_D6 | 8 | 0.07 | 0.03 | 0.09 | 0.09 | 0.08 |
| multisensory_s1_D6 | 16 | 0.05 | 0.02 | 0.06 | 0.07 | 0.06 |
