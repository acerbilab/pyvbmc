Weights re-optimized on the two-level full-covariance shrinkage of the expected log joints, scored against the new stack's own `elbo_mc`. Per condition and `M`, medians over cells: the bias of the raw optimization (from the record), of the value-only shrinkage at the raw weights, of the shrunken value at the shrunken-optimized weights (`shrunk_opt`, the headline this variant would report) and of the raw value at those weights; the added bias of `shrunk_opt` and of raw (bias minus the mean bias of the cell's input runs) with a bootstrap interval; gsKL and MMTV of both optimizations with the fraction of cells the new weights improve; the KL gap of both; and the largest weight change.

## multisensory_s1_D6_noise3_svbmc

| M | n | raw | value-only | shrunk_opt | raw_at_opt | shrunk_opt added [CI] | raw added | gsKL new / recorded (improved) | MMTV new / recorded (improved) | KL gap new / recorded | max abs dw |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 20 | +0.84 | +0.58 | +0.71 | +0.78 | +0.01 [-0.04, +0.08] | +0.20 | 2.28 / 2.83 (0.85) | 0.219 / 0.233 (0.75) | +0.72 / +0.75 | 0.041 |
| 4 | 20 | +1.07 | +0.72 | +0.83 | +0.89 | +0.08 [-0.07, +0.15] | +0.26 | 2.05 / 2.38 (0.90) | 0.213 / 0.222 (0.80) | +0.68 / +0.73 | 0.043 |
| 5 | 20 | +1.04 | +0.67 | +0.90 | +0.70 | +0.13 [+0.04, +0.24] | +0.24 | 0.90 / 1.39 (0.90) | 0.157 / 0.192 (0.90) | +0.60 / +0.62 | 0.043 |

## multisensory_s1_D6_noise1.3_svbmc

| M | n | raw | value-only | shrunk_opt | raw_at_opt | shrunk_opt added [CI] | raw added | gsKL new / recorded (improved) | MMTV new / recorded (improved) | KL gap new / recorded | max abs dw |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 20 | +0.44 | +0.25 | +0.31 | +0.36 | -0.04 [-0.06, +0.02] | +0.07 | 1.17 / 1.53 (0.90) | 0.162 / 0.182 (0.85) | +0.46 / +0.49 | 0.031 |
| 4 | 20 | +0.47 | +0.26 | +0.31 | +0.37 | -0.00 [-0.05, +0.06] | +0.14 | 1.29 / 1.69 (0.95) | 0.154 / 0.176 (0.90) | +0.49 / +0.49 | 0.034 |
| 5 | 20 | +0.55 | +0.35 | +0.40 | +0.44 | +0.01 [-0.02, +0.03] | +0.18 | 1.69 / 1.92 (0.85) | 0.181 / 0.190 (0.90) | +0.50 / +0.52 | 0.036 |

## rosenbrock_D2_noise3_svbmc

| M | n | raw | value-only | shrunk_opt | raw_at_opt | shrunk_opt added [CI] | raw added | gsKL new / recorded (improved) | MMTV new / recorded (improved) | KL gap new / recorded | max abs dw |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 20 | +0.29 | +0.12 | +0.31 | +0.20 | +0.17 [+0.09, +0.37] | +0.21 | 0.11 / 0.04 (0.25) | 0.153 / 0.075 (0.05) | +0.14 / +0.05 | 0.063 |
| 4 | 20 | +0.35 | +0.07 | +0.36 | +0.19 | +0.23 [+0.21, +0.37] | +0.28 | 0.12 / 0.03 (0.15) | 0.149 / 0.079 (0.15) | +0.13 / +0.05 | 0.071 |
| 5 | 20 | +0.53 | +0.18 | +0.57 | +0.31 | +0.29 [+0.22, +1.83] | +0.36 | 0.13 / 0.02 (0.10) | 0.160 / 0.064 (0.05) | +0.13 / +0.04 | 0.086 |

## gmm_D2_noise3_svbmc

| M | n | raw | value-only | shrunk_opt | raw_at_opt | shrunk_opt added [CI] | raw added | gsKL new / recorded (improved) | MMTV new / recorded (improved) | KL gap new / recorded | max abs dw |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 20 | +0.34 | -0.05 | +0.32 | +0.12 | +0.09 [+0.02, +0.20] | +0.14 | 0.71 / 0.96 (0.80) | 0.231 / 0.287 (0.85) | +0.60 / +0.55 | 0.065 |
| 4 | 20 | +0.30 | -0.01 | +0.36 | +0.17 | +0.15 [+0.10, +0.32] | +0.12 | 0.05 / 0.06 (0.55) | 0.163 / 0.176 (0.50) | +0.45 / +0.33 | 0.051 |
| 5 | 20 | +0.40 | -0.00 | +0.31 | +0.20 | +0.07 [+0.04, +0.21] | +0.19 | 0.03 / 0.04 (0.55) | 0.154 / 0.170 (0.50) | +0.39 / +0.30 | 0.046 |

## ring_D2_noise3_svbmc

| M | n | raw | value-only | shrunk_opt | raw_at_opt | shrunk_opt added [CI] | raw added | gsKL new / recorded (improved) | MMTV new / recorded (improved) | KL gap new / recorded | max abs dw |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 20 | +0.37 | +0.10 | +0.11 | +0.08 | -0.09 [-0.17, -0.05] | +0.10 | 0.23 / 0.40 (0.80) | 0.269 / 0.327 (0.95) | +0.68 / +0.79 | 0.019 |
| 4 | 20 | +0.40 | +0.06 | +0.09 | +0.10 | -0.13 [-0.18, -0.11] | +0.12 | 0.09 / 0.14 (0.75) | 0.219 / 0.259 (0.95) | +0.43 / +0.53 | 0.025 |
| 5 | 20 | +0.42 | +0.13 | +0.20 | +0.13 | -0.10 [-0.13, -0.08] | +0.13 | 0.17 / 0.18 (0.85) | 0.182 / 0.245 (1.00) | +0.38 / +0.48 | 0.019 |

## student_D8_noise3_svbmc

| M | n | raw | value-only | shrunk_opt | raw_at_opt | shrunk_opt added [CI] | raw added | gsKL new / recorded (improved) | MMTV new / recorded (improved) | KL gap new / recorded | max abs dw |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 20 | -0.09 | -0.32 | -0.25 | -0.10 | -0.02 [-0.07, +0.04] | +0.20 | 0.24 / 0.25 (0.45) | 0.106 / 0.104 (0.15) | +0.44 / +0.41 | 0.019 |
| 4 | 20 | -0.10 | -0.39 | -0.36 | -0.12 | -0.06 [-0.15, +0.04] | +0.20 | 0.27 / 0.25 (0.20) | 0.116 / 0.111 (0.00) | +0.46 / +0.43 | 0.028 |
| 5 | 20 | +0.04 | -0.24 | -0.13 | +0.09 | -0.00 [-0.07, +0.08] | +0.24 | 0.18 / 0.17 (0.25) | 0.106 / 0.096 (0.05) | +0.40 / +0.35 | 0.016 |

## gmm_D2_svbmc

| M | n | raw | value-only | shrunk_opt | raw_at_opt | shrunk_opt added [CI] | raw added | gsKL new / recorded (improved) | MMTV new / recorded (improved) | KL gap new / recorded | max abs dw |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 20 | -0.01 | -0.00 | +0.00 | -0.00 | -0.00 [-0.01, +0.01] | -0.01 | 0.43 / 0.43 (0.30) | 0.182 / 0.183 (0.60) | +0.29 / +0.29 | 0.001 |
| 4 | 20 | +0.00 | +0.01 | +0.01 | +0.00 | +0.01 [-0.00, +0.02] | -0.00 | 0.01 / 0.01 (0.60) | 0.047 / 0.047 (0.40) | +0.04 / +0.04 | 0.001 |
| 5 | 20 | +0.01 | +0.02 | +0.02 | +0.01 | +0.02 [+0.01, +0.02] | +0.01 | 0.00 / 0.00 (0.55) | 0.032 / 0.032 (0.50) | +0.02 / +0.02 | 0.001 |

## multisensory_s1_D6_svbmc

| M | n | raw | value-only | shrunk_opt | raw_at_opt | shrunk_opt added [CI] | raw added | gsKL new / recorded (improved) | MMTV new / recorded (improved) | KL gap new / recorded | max abs dw |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 20 | +0.07 | +0.08 | +0.08 | +0.07 | +0.03 [+0.01, +0.05] | +0.01 | 0.15 / 0.15 (0.45) | 0.082 / 0.082 (0.45) | +0.27 / +0.24 | 0.003 |
| 4 | 20 | +0.04 | +0.06 | +0.08 | +0.07 | +0.04 [+0.02, +0.06] | +0.01 | 0.32 / 0.32 (0.60) | 0.089 / 0.090 (0.70) | +0.25 / +0.25 | 0.003 |
| 5 | 20 | +0.07 | +0.08 | +0.09 | +0.07 | +0.05 [+0.04, +0.07] | +0.03 | 0.08 / 0.10 (0.75) | 0.056 / 0.061 (0.80) | +0.23 / +0.23 | 0.004 |
