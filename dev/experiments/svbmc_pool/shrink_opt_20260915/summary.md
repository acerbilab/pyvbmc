Weights re-optimized on the two-level full-covariance shrinkage of the expected log joints, scored against the new stack's own `elbo_mc`. Per condition and `M`, medians over cells: the bias of the raw optimization (from the record), of the value-only shrinkage at the raw weights, of the shrunken value at the shrunken-optimized weights (`shrunk_opt`, the headline this variant would report) and of the raw value at those weights; the added bias of `shrunk_opt` and of raw (bias minus the mean bias of the cell's input runs) with a bootstrap interval; gsKL and MMTV of both optimizations with the fraction of cells the new weights improve; the KL gap of both; and the largest weight change.

## multisensory_s1_D6_noise3_svbmc

| M | n | raw | value-only | shrunk_opt | raw_at_opt | shrunk_opt added [CI] | raw added | gsKL new / recorded (improved) | MMTV new / recorded (improved) | KL gap new / recorded | max abs dw |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 20 | +0.84 | +0.58 | +0.69 | +0.70 | +0.01 [-0.05, +0.09] | +0.20 | 2.22 / 2.83 (0.85) | 0.217 / 0.233 (0.80) | +0.71 / +0.75 | 0.041 |
| 4 | 20 | +1.07 | +0.72 | +0.89 | +0.92 | +0.10 [+0.03, +0.16] | +0.26 | 2.03 / 2.38 (1.00) | 0.209 / 0.222 (0.85) | +0.71 / +0.73 | 0.045 |
| 5 | 20 | +1.04 | +0.67 | +0.82 | +0.73 | +0.10 [+0.03, +0.19] | +0.24 | 1.14 / 1.39 (0.90) | 0.161 / 0.192 (0.85) | +0.58 / +0.62 | 0.043 |

## multisensory_s1_D6_noise1.3_svbmc

| M | n | raw | value-only | shrunk_opt | raw_at_opt | shrunk_opt added [CI] | raw added | gsKL new / recorded (improved) | MMTV new / recorded (improved) | KL gap new / recorded | max abs dw |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 20 | +0.44 | +0.25 | +0.31 | +0.35 | -0.04 [-0.08, +0.02] | +0.07 | 1.27 / 1.53 (0.95) | 0.165 / 0.182 (0.85) | +0.47 / +0.49 | 0.026 |
| 4 | 20 | +0.47 | +0.26 | +0.30 | +0.38 | -0.03 [-0.07, +0.04] | +0.14 | 1.27 / 1.69 (0.95) | 0.154 / 0.176 (0.90) | +0.51 / +0.49 | 0.034 |
| 5 | 20 | +0.55 | +0.35 | +0.43 | +0.47 | +0.02 [+0.00, +0.08] | +0.18 | 1.61 / 1.92 (0.90) | 0.179 / 0.190 (0.80) | +0.51 / +0.52 | 0.034 |

## rosenbrock_D2_noise3_svbmc

| M | n | raw | value-only | shrunk_opt | raw_at_opt | shrunk_opt added [CI] | raw added | gsKL new / recorded (improved) | MMTV new / recorded (improved) | KL gap new / recorded | max abs dw |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 20 | +0.29 | +0.12 | +0.32 | +0.19 | +0.17 [+0.08, +0.38] | +0.21 | 0.10 / 0.04 (0.25) | 0.147 / 0.075 (0.05) | +0.13 / +0.05 | 0.062 |
| 4 | 20 | +0.35 | +0.07 | +0.36 | +0.20 | +0.22 [+0.20, +0.38] | +0.28 | 0.12 / 0.03 (0.15) | 0.148 / 0.079 (0.15) | +0.12 / +0.05 | 0.069 |
| 5 | 20 | +0.53 | +0.18 | +0.59 | +0.26 | +0.29 [+0.22, +1.83] | +0.36 | 0.10 / 0.02 (0.15) | 0.153 / 0.064 (0.05) | +0.13 / +0.04 | 0.082 |

## gmm_D2_noise3_svbmc

| M | n | raw | value-only | shrunk_opt | raw_at_opt | shrunk_opt added [CI] | raw added | gsKL new / recorded (improved) | MMTV new / recorded (improved) | KL gap new / recorded | max abs dw |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 20 | +0.34 | -0.05 | +0.35 | +0.12 | +0.15 [+0.04, +0.24] | +0.14 | 0.73 / 0.96 (0.80) | 0.229 / 0.287 (0.80) | +0.60 / +0.55 | 0.062 |
| 4 | 20 | +0.30 | -0.01 | +0.37 | +0.20 | +0.18 [+0.06, +0.39] | +0.12 | 0.05 / 0.06 (0.55) | 0.161 / 0.176 (0.55) | +0.46 / +0.33 | 0.060 |
| 5 | 20 | +0.40 | -0.00 | +0.36 | +0.23 | +0.09 [+0.07, +0.32] | +0.19 | 0.03 / 0.04 (0.60) | 0.155 / 0.170 (0.50) | +0.43 / +0.30 | 0.043 |

## ring_D2_noise3_svbmc

| M | n | raw | value-only | shrunk_opt | raw_at_opt | shrunk_opt added [CI] | raw added | gsKL new / recorded (improved) | MMTV new / recorded (improved) | KL gap new / recorded | max abs dw |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 20 | +0.37 | +0.10 | +0.11 | +0.06 | -0.08 [-0.19, -0.04] | +0.10 | 0.27 / 0.40 (0.80) | 0.263 / 0.327 (0.95) | +0.68 / +0.79 | 0.020 |
| 4 | 20 | +0.40 | +0.06 | +0.10 | +0.10 | -0.15 [-0.18, -0.11] | +0.12 | 0.10 / 0.14 (0.75) | 0.217 / 0.259 (0.95) | +0.44 / +0.53 | 0.023 |
| 5 | 20 | +0.42 | +0.13 | +0.19 | +0.12 | -0.11 [-0.13, -0.09] | +0.13 | 0.14 / 0.18 (0.80) | 0.181 / 0.245 (1.00) | +0.37 / +0.48 | 0.018 |

## student_D8_noise3_svbmc

| M | n | raw | value-only | shrunk_opt | raw_at_opt | shrunk_opt added [CI] | raw added | gsKL new / recorded (improved) | MMTV new / recorded (improved) | KL gap new / recorded | max abs dw |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 20 | -0.09 | -0.32 | -0.23 | -0.04 | -0.03 [-0.06, +0.10] | +0.20 | 0.25 / 0.25 (0.25) | 0.106 / 0.104 (0.25) | +0.45 / +0.41 | 0.019 |
| 4 | 20 | -0.10 | -0.39 | -0.35 | -0.08 | -0.07 [-0.16, +0.04] | +0.20 | 0.27 / 0.25 (0.15) | 0.115 / 0.111 (0.00) | +0.47 / +0.43 | 0.026 |
| 5 | 20 | +0.04 | -0.24 | -0.17 | +0.08 | +0.03 [-0.09, +0.09] | +0.24 | 0.18 / 0.17 (0.25) | 0.106 / 0.096 (0.05) | +0.41 / +0.35 | 0.017 |

## gmm_D2_svbmc

| M | n | raw | value-only | shrunk_opt | raw_at_opt | shrunk_opt added [CI] | raw added | gsKL new / recorded (improved) | MMTV new / recorded (improved) | KL gap new / recorded | max abs dw |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 20 | -0.01 | -0.00 | +0.00 | -0.00 | -0.00 [-0.01, +0.01] | -0.01 | 0.43 / 0.43 (0.55) | 0.182 / 0.183 (0.55) | +0.29 / +0.29 | 0.001 |
| 4 | 20 | +0.00 | +0.01 | +0.02 | +0.01 | +0.01 [+0.00, +0.03] | -0.00 | 0.01 / 0.01 (0.55) | 0.048 / 0.047 (0.40) | +0.05 / +0.04 | 0.001 |
| 5 | 20 | +0.01 | +0.02 | +0.02 | +0.01 | +0.01 [+0.00, +0.03] | +0.01 | 0.00 / 0.00 (0.55) | 0.032 / 0.032 (0.40) | +0.03 / +0.02 | 0.001 |

## multisensory_s1_D6_svbmc

| M | n | raw | value-only | shrunk_opt | raw_at_opt | shrunk_opt added [CI] | raw added | gsKL new / recorded (improved) | MMTV new / recorded (improved) | KL gap new / recorded | max abs dw |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 20 | +0.07 | +0.08 | +0.07 | +0.05 | +0.03 [-0.00, +0.05] | +0.01 | 0.15 / 0.15 (0.55) | 0.077 / 0.082 (0.60) | +0.24 / +0.24 | 0.002 |
| 4 | 20 | +0.04 | +0.06 | +0.07 | +0.05 | +0.04 [+0.02, +0.05] | +0.01 | 0.29 / 0.32 (0.65) | 0.084 / 0.090 (0.65) | +0.26 / +0.25 | 0.003 |
| 5 | 20 | +0.07 | +0.08 | +0.09 | +0.06 | +0.05 [+0.03, +0.07] | +0.03 | 0.08 / 0.10 (0.70) | 0.056 / 0.061 (0.75) | +0.23 / +0.23 | 0.004 |
