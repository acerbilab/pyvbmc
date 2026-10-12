# S-VBMC stacking comparison: integrated against original 0.1.1

Generated 2026-09-15 06:54:39. Every cell stacks the same subset of filtered pool runs with both implementations, `n_samples=20`, `lr=0.1`, `max_steps=500`, `version="all-weights"`, subsets drawn with seed 0. Medians over the repetitions of a cell with a 10 000-resample bootstrap 95 % interval; `d` columns are the paired difference integrated minus original, so a negative value favours the integrated class. The headline ELBO is the capped value for the integrated class on noisy stacks and the raw estimate for the original, which is what each implementation reports. gsKL is the house convention (no `1/D` factor).

Every reported ELBO is scored by its bias against `elbo_mc = e_log_joint_mc + entropy_ref`, the Monte Carlo ELBO of the posterior that arm's fit produced. `e_log_joint_mc` is the mean of the target's noiseless log density over 10 000 of the stack's own draws; `entropy_ref` is the entropy of the stacked mixture at that arm's final weights, estimated here for both arms by one estimator (the integrated class's `stacked_entropy`, 200 draws per component in 8 batches, from a generator seeded by the cell, the same draws for both arms). The reference is arm-independent by construction: an arm's own reported entropy is estimated at the weights that arm selected, from its own draws (the optimizer's 20 per component for the original, a fresh 100 for the integrated class), so a reference built on it would move with the arm and credit whichever entropy estimate came out high. `kl_gap = ln Z - elbo_mc` is how far the stacked posterior itself is from the target, in evidence units; the `err` columns further below are the error against `ln Z`, descriptive only.

## multisensory_s1_D6_noise3_svbmc

Single runs (`M = 1`, n = 100): MMTV 0.279 [0.270, 0.288], gsKL 4.02 [3.38, 4.71], evidence error 0.31 [0.26, 0.38].

All 70 cells of this condition: runtime ratio 0.254 [0.229, 0.286], max\|dw\| 0.0211 [0.0178, 0.0252].

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.379 [0.136, 0.480] | 0.768 [0.664, 0.939] | 0.904 [0.747, 1.051] | 0.463 [0.336, 0.573] | -0.575 [-0.655, -0.468] | 0.854 [0.804, 0.907] | 0.852 [0.781, 0.930] | yes |
| 4 | 20 | 0.558 [0.429, 0.794] | 1.072 [0.961, 1.159] | 1.177 [1.080, 1.276] | 0.620 [0.538, 0.823] | -0.585 [-0.687, -0.513] | 0.733 [0.660, 0.801] | 0.748 [0.676, 0.802] | yes |
| 8 | 20 | 0.399 [0.300, 0.493] | 1.163 [1.078, 1.277] | 1.319 [1.221, 1.479] | 0.510 [0.382, 0.582] | -0.965 [-1.021, -0.764] | 0.479 [0.422, 0.520] | 0.468 [0.434, 0.527] | yes |
| 16 | 10 | 0.443 [0.395, 0.513] | 1.309 [1.256, 1.426] | 1.447 [1.276, 1.577] | 0.506 [0.470, 0.586] | -0.995 [-1.116, -0.860] | 0.425 [0.338, 0.510] | 0.443 [0.360, 0.533] | yes |

Growth of the integrated headline's median bias from `M = 2` to `M = 16`: 0.064 nats (0.379 to 0.443), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 2 | 2.754 [2.671, 2.901] | 2.778 [2.674, 2.899] | 2.748 [2.675, 2.943] | 2.908 [2.768, 3.071] | -503.040 [-503.093, -502.990] | 0.024 | -503.038 [-503.116, -502.966] | 0.025 |
| 4 | 2.953 [2.873, 3.062] | 2.968 [2.884, 3.064] | 2.952 [2.846, 3.060] | 3.027 [2.927, 3.138] | -502.919 [-502.987, -502.846] | 0.021 | -502.934 [-502.988, -502.862] | 0.021 |
| 8 | 3.127 [3.061, 3.215] | 3.103 [3.054, 3.239] | 3.119 [3.031, 3.223] | 3.178 [3.118, 3.344] | -502.665 [-502.706, -502.608] | 0.020 | -502.654 [-502.713, -502.620] | 0.020 |
| 16 | 3.269 [3.130, 3.346] | 3.235 [3.136, 3.386] | 3.229 [3.131, 3.319] | 3.307 [3.209, 3.386] | -502.611 [-502.696, -502.524] | 0.020 | -502.629 [-502.719, -502.546] | 0.019 |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.269 [0.207, 0.278] | 0.265 [0.210, 0.273] | 0.0009 [-0.0033, 0.0054] | 3.74 [2.08, 4.77] | 3.70 [2.04, 4.58] | 0.026 [-0.015, 0.079] | 0.44 [0.34, 0.62] | 0.19 [0.09, 0.29] | 0.374 [0.094, 0.528] | 0.0240 [0.0188, 0.0301] | 0.3 [0.2, 0.3] | 1.2 [1.0, 1.3] | 0.225 [0.192, 0.263] |
| 4 | 20 | 0.222 [0.202, 0.247] | 0.219 [0.203, 0.247] | 0.0011 [-0.0027, 0.0021] | 2.38 [1.31, 3.06] | 2.39 [1.39, 3.05] | -0.010 [-0.081, 0.015] | 0.19 [0.11, 0.26] | 0.46 [0.32, 0.52] | -0.251 [-0.365, -0.027] | 0.0267 [0.0222, 0.0380] | 1.7 [1.5, 1.8] | 6.8 [5.6, 7.5] | 0.240 [0.214, 0.338] |
| 8 | 20 | 0.131 [0.114, 0.158] | 0.129 [0.117, 0.156] | 0.0003 [-0.0017, 0.0016] | 0.65 [0.51, 1.03] | 0.62 [0.51, 0.95] | 0.002 [-0.008, 0.016] | 0.09 [0.06, 0.13] | 0.85 [0.68, 0.90] | -0.658 [-0.813, -0.516] | 0.0164 [0.0122, 0.0224] | 13.3 [12.3, 14.5] | 45.7 [41.6, 56.7] | 0.298 [0.250, 0.333] |
| 16 | 10 | 0.122 [0.108, 0.162] | 0.125 [0.106, 0.156] | 0.0015 [-0.0031, 0.0047] | 0.48 [0.35, 0.69] | 0.43 [0.33, 0.74] | 0.002 [-0.017, 0.040] | 0.15 [0.04, 0.19] | 1.05 [0.83, 1.09] | -0.809 [-0.960, -0.703] | 0.0161 [0.0151, 0.0227] | 71.2 [59.4, 79.3] | 260.7 [255.4, 298.1] | 0.286 [0.218, 0.326] |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.0009 | 0.4304 | 1 | 0.026 | 0.1327 | 1 | no |
| 4 | 20 | 0.0011 | 0.9854 | 1 | -0.010 | 0.2611 | 1 | no |
| 8 | 20 | 0.0003 | 0.8408 | 1 | 0.002 | 0.5217 | 1 | no |
| 16 | 10 | 0.0015 | 0.5566 | 1 | 0.002 | 0.9219 | 1 | no |

## multisensory_s1_D6_noise1.3_svbmc

Single runs (`M = 1`, n = 100): MMTV 0.231 [0.221, 0.240], gsKL 2.70 [2.39, 3.09], evidence error 0.37 [0.33, 0.41].

All 70 cells of this condition: runtime ratio 0.281 [0.262, 0.303], max\|dw\| 0.0205 [0.0174, 0.0233].

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.077 [0.051, 0.191] | 0.406 [0.318, 0.462] | 0.507 [0.419, 0.605] | 0.246 [0.175, 0.301] | -0.426 [-0.455, -0.283] | 0.590 [0.557, 0.654] | 0.596 [0.546, 0.667] | yes |
| 4 | 20 | 0.110 [0.061, 0.150] | 0.468 [0.456, 0.510] | 0.534 [0.480, 0.599] | 0.179 [0.107, 0.268] | -0.445 [-0.469, -0.413] | 0.492 [0.400, 0.546] | 0.475 [0.405, 0.548] | yes |
| 8 | 20 | 0.139 [0.113, 0.185] | 0.562 [0.511, 0.633] | 0.660 [0.564, 0.764] | 0.206 [0.164, 0.253] | -0.496 [-0.583, -0.443] | 0.432 [0.364, 0.448] | 0.429 [0.391, 0.453] | yes |
| 16 | 10 | 0.156 [0.101, 0.191] | 0.606 [0.546, 0.694] | 0.632 [0.575, 0.734] | 0.173 [0.131, 0.257] | -0.485 [-0.577, -0.410] | 0.246 [0.206, 0.322] | 0.271 [0.194, 0.347] | yes |

Growth of the integrated headline's median bias from `M = 2` to `M = 16`: 0.078 nats (0.077 to 0.156), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 2 | 3.067 [2.994, 3.202] | 3.052 [2.976, 3.214] | 3.074 [2.984, 3.217] | 3.151 [3.064, 3.331] | -502.776 [-502.840, -502.743] | 0.022 | -502.781 [-502.853, -502.732] | 0.022 |
| 4 | 3.160 [3.114, 3.387] | 3.186 [3.101, 3.364] | 3.172 [3.109, 3.347] | 3.303 [3.201, 3.404] | -502.678 [-502.732, -502.586] | 0.023 | -502.661 [-502.734, -502.591] | 0.021 |
| 8 | 3.315 [3.265, 3.408] | 3.310 [3.259, 3.409] | 3.309 [3.291, 3.400] | 3.399 [3.345, 3.428] | -502.618 [-502.634, -502.550] | 0.020 | -502.615 [-502.639, -502.577] | 0.020 |
| 16 | 3.496 [3.452, 3.550] | 3.481 [3.443, 3.544] | 3.478 [3.413, 3.556] | 3.514 [3.468, 3.574] | -502.432 [-502.508, -502.392] | 0.019 | -502.457 [-502.533, -502.380] | 0.020 |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.209 [0.188, 0.232] | 0.213 [0.184, 0.233] | -0.0020 [-0.0040, 0.0005] | 2.40 [1.67, 2.86] | 2.42 [1.59, 2.91] | -0.009 [-0.055, 0.046] | 0.45 [0.34, 0.57] | 0.13 [0.07, 0.18] | 0.300 [0.233, 0.443] | 0.0188 [0.0135, 0.0242] | 0.5 [0.5, 0.6] | 1.6 [1.3, 2.1] | 0.317 [0.279, 0.369] |
| 4 | 20 | 0.176 [0.154, 0.206] | 0.180 [0.165, 0.203] | 0.0005 [-0.0028, 0.0045] | 1.69 [0.77, 2.33] | 1.69 [1.05, 2.23] | -0.015 [-0.039, 0.040] | 0.37 [0.32, 0.44] | 0.11 [0.04, 0.16] | 0.259 [0.139, 0.374] | 0.0229 [0.0202, 0.0279] | 3.0 [2.4, 3.2] | 9.8 [7.4, 12.4] | 0.279 [0.249, 0.321] |
| 8 | 20 | 0.166 [0.149, 0.177] | 0.165 [0.153, 0.175] | 0.0004 [-0.0019, 0.0014] | 1.20 [0.70, 1.45] | 1.14 [0.71, 1.40] | 0.005 [-0.010, 0.040] | 0.25 [0.18, 0.32] | 0.25 [0.15, 0.42] | 0.006 [-0.226, 0.153] | 0.0177 [0.0166, 0.0257] | 13.5 [12.2, 15.2] | 55.0 [45.0, 56.1] | 0.271 [0.234, 0.296] |
| 16 | 10 | 0.120 [0.092, 0.145] | 0.128 [0.095, 0.147] | -0.0008 [-0.0081, 0.0071] | 0.35 [0.16, 0.87] | 0.34 [0.19, 0.87] | -0.022 [-0.047, 0.034] | 0.12 [0.05, 0.17] | 0.40 [0.33, 0.43] | -0.294 [-0.365, -0.145] | 0.0190 [0.0138, 0.0274] | 62.4 [57.6, 75.6] | 251.0 [212.2, 282.3] | 0.257 [0.216, 0.309] |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 2 | 20 | -0.0020 | 0.08255 | 1 | -0.009 | 0.6742 | 1 | no |
| 4 | 20 | 0.0005 | 0.9563 | 1 | -0.015 | 0.7012 | 1 | no |
| 8 | 20 | 0.0004 | 0.8408 | 1 | 0.005 | 0.3683 | 1 | no |
| 16 | 10 | -0.0008 | 0.9219 | 1 | -0.022 | 0.7695 | 1 | no |

## rosenbrock_D2_noise3_svbmc

Single runs (`M = 1`, n = 100): MMTV 0.097 [0.076, 0.122], gsKL 0.15 [0.08, 0.22], evidence error 0.17 [0.14, 0.22].

All 70 cells of this condition: runtime ratio 0.402 [0.321, 0.457], max\|dw\| 0.0216 [0.0201, 0.0229].

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.093 [-0.089, 0.177] | 0.251 [0.154, 0.414] | 0.305 [0.201, 0.428] | 0.135 [-0.038, 0.218] | -0.300 [-0.377, -0.142] | 0.070 [0.042, 0.091] | 0.069 [0.037, 0.105] | yes |
| 4 | 20 | 0.072 [-0.008, 0.195] | 0.353 [0.304, 0.546] | 0.397 [0.345, 0.553] | 0.147 [0.065, 0.238] | -0.272 [-0.434, -0.214] | 0.050 [0.040, 0.082] | 0.063 [0.046, 0.082] | yes |
| 8 | 20 | 0.101 [0.045, 0.137] | 0.530 [0.461, 0.579] | 0.572 [0.516, 0.685] | 0.131 [0.110, 0.186] | -0.492 [-0.565, -0.384] | 0.039 [0.028, 0.046] | 0.045 [0.036, 0.056] | yes |
| 16 | 10 | 0.109 [0.031, 0.135] | 0.739 [0.660, 0.991] | 0.777 [0.689, 1.040] | 0.144 [0.091, 0.183] | -0.639 [-0.942, -0.580] | 0.050 [0.039, 0.061] | 0.043 [0.033, 0.065] | yes |

Growth of the integrated headline's median bias from `M = 2` to `M = 16`: 0.017 nats (0.093 to 0.109), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 2 | 2.475 [2.449, 2.550] | 2.476 [2.458, 2.554] | 2.477 [2.454, 2.550] | 2.523 [2.495, 2.588] | -2.330 [-2.350, -2.301] | 0.012 | -2.329 [-2.365, -2.297] | 0.012 |
| 4 | 2.528 [2.470, 2.570] | 2.527 [2.464, 2.566] | 2.534 [2.473, 2.569] | 2.573 [2.518, 2.598] | -2.310 [-2.342, -2.300] | 0.012 | -2.323 [-2.342, -2.306] | 0.012 |
| 8 | 2.556 [2.527, 2.591] | 2.562 [2.531, 2.596] | 2.554 [2.529, 2.574] | 2.602 [2.575, 2.626] | -2.299 [-2.306, -2.288] | 0.011 | -2.305 [-2.316, -2.296] | 0.011 |
| 16 | 2.538 [2.472, 2.568] | 2.542 [2.465, 2.565] | 2.544 [2.477, 2.563] | 2.588 [2.526, 2.613] | -2.309 [-2.321, -2.299] | 0.012 | -2.303 [-2.325, -2.293] | 0.012 |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.075 [0.066, 0.092] | 0.076 [0.068, 0.100] | 0.0020 [-0.0041, 0.0066] | 0.06 [0.02, 0.12] | 0.06 [0.02, 0.09] | 0.002 [-0.004, 0.011] | 0.20 [0.07, 0.27] | 0.26 [0.19, 0.39] | -0.100 [-0.199, 0.023] | 0.0227 [0.0182, 0.0331] | 0.3 [0.3, 0.4] | 1.3 [1.1, 1.6] | 0.267 [0.195, 0.289] |
| 4 | 20 | 0.079 [0.068, 0.095] | 0.083 [0.065, 0.100] | -0.0014 [-0.0035, 0.0045] | 0.03 [0.02, 0.05] | 0.03 [0.02, 0.06] | 0.001 [-0.001, 0.006] | 0.12 [0.06, 0.15] | 0.33 [0.29, 0.50] | -0.236 [-0.352, -0.176] | 0.0215 [0.0198, 0.0232] | 1.9 [1.8, 2.0] | 4.9 [4.2, 5.9] | 0.358 [0.305, 0.427] |
| 8 | 20 | 0.056 [0.047, 0.073] | 0.061 [0.050, 0.077] | -0.0018 [-0.0082, 0.0033] | 0.02 [0.01, 0.02] | 0.01 [0.01, 0.02] | -0.001 [-0.002, 0.002] | 0.07 [0.06, 0.10] | 0.55 [0.47, 0.62] | -0.458 [-0.539, -0.341] | 0.0225 [0.0146, 0.0307] | 9.7 [7.8, 10.4] | 20.2 [17.3, 21.4] | 0.477 [0.436, 0.535] |
| 16 | 10 | 0.076 [0.052, 0.102] | 0.065 [0.054, 0.091] | 0.0030 [-0.0045, 0.0128] | 0.01 [0.00, 0.04] | 0.01 [0.01, 0.03] | 0.001 [-0.001, 0.010] | 0.08 [0.05, 0.09] | 0.73 [0.66, 0.95] | -0.644 [-0.857, -0.588] | 0.0195 [0.0154, 0.0216] | 44.2 [40.6, 48.4] | 94.4 [72.8, 99.9] | 0.489 [0.470, 0.568] |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.0020 | 0.8983 | 1 | 0.002 | 0.3884 | 1 | no |
| 4 | 20 | -0.0014 | 0.9854 | 1 | 0.001 | 0.33 | 1 | no |
| 8 | 20 | -0.0018 | 0.2024 | 1 | -0.001 | 0.7012 | 1 | no |
| 16 | 10 | 0.0030 | 0.2754 | 1 | 0.001 | 0.1934 | 1 | no |

## gmm_D2_noise3_svbmc

Single runs (`M = 1`, n = 100): MMTV 0.484 [0.449, 0.549], gsKL 22.47 [19.16, 31.16], evidence error 1.04 [0.93, 1.20].

All 70 cells of this condition: runtime ratio 0.466 [0.404, 0.508], max\|dw\| 0.0218 [0.0189, 0.0237].

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.070 [-0.203, 0.248] | 0.297 [0.179, 0.414] | 0.331 [0.238, 0.460] | 0.117 [-0.116, 0.333] | -0.298 [-0.433, -0.177] | 0.759 [0.534, 0.893] | 0.755 [0.539, 0.884] | yes |
| 4 | 20 | -0.013 [-0.063, 0.077] | 0.301 [0.183, 0.413] | 0.361 [0.243, 0.437] | 0.039 [-0.035, 0.121] | -0.294 [-0.439, -0.235] | 0.331 [0.225, 0.455] | 0.293 [0.221, 0.467] | yes |
| 8 | 20 | 0.063 [0.011, 0.112] | 0.501 [0.455, 0.570] | 0.503 [0.467, 0.555] | 0.091 [0.065, 0.145] | -0.483 [-0.512, -0.381] | 0.210 [0.174, 0.229] | 0.200 [0.155, 0.218] | yes |
| 16 | 10 | 0.045 [-0.005, 0.096] | 0.726 [0.562, 0.900] | 0.736 [0.661, 0.844] | 0.098 [0.025, 0.146] | -0.697 [-0.816, -0.536] | 0.127 [0.106, 0.190] | 0.148 [0.118, 0.168] | yes |

Growth of the integrated headline's median bias from `M = 2` to `M = 16`: -0.024 nats (0.070 to 0.045), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 2 | 3.879 [3.603, 4.082] | 3.905 [3.571, 4.078] | 3.860 [3.620, 4.085] | 3.915 [3.664, 4.129] | 2.237 [2.103, 2.462] | 0.015 | 2.241 [2.111, 2.457] | 0.015 |
| 4 | 4.311 [4.236, 4.442] | 4.307 [4.239, 4.425] | 4.299 [4.220, 4.451] | 4.339 [4.259, 4.482] | 2.665 [2.541, 2.771] | 0.015 | 2.703 [2.529, 2.774] | 0.014 |
| 8 | 4.415 [4.392, 4.477] | 4.423 [4.388, 4.481] | 4.423 [4.402, 4.491] | 4.475 [4.455, 4.535] | 2.785 [2.766, 2.821] | 0.013 | 2.796 [2.777, 2.841] | 0.013 |
| 16 | 4.416 [4.402, 4.499] | 4.426 [4.402, 4.507] | 4.420 [4.385, 4.511] | 4.459 [4.430, 4.551] | 2.869 [2.779, 2.890] | 0.013 | 2.848 [2.828, 2.878] | 0.013 |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.339 [0.280, 0.382] | 0.338 [0.279, 0.379] | 0.0017 [-0.0016, 0.0048] | 4.79 [0.76, 18.11] | 4.88 [0.75, 17.86] | 0.006 [-0.003, 0.120] | 0.74 [0.51, 1.01] | 0.40 [0.25, 0.51] | 0.300 [0.166, 0.460] | 0.0261 [0.0211, 0.0291] | 0.3 [0.3, 0.3] | 1.0 [0.8, 1.1] | 0.338 [0.260, 0.423] |
| 4 | 20 | 0.176 [0.152, 0.216] | 0.172 [0.144, 0.208] | 0.0002 [-0.0026, 0.0034] | 0.06 [0.03, 0.48] | 0.06 [0.03, 0.48] | 0.001 [-0.008, 0.004] | 0.34 [0.22, 0.43] | 0.15 [0.05, 0.19] | 0.147 [0.099, 0.270] | 0.0170 [0.0135, 0.0250] | 1.4 [1.2, 1.5] | 3.4 [2.9, 4.2] | 0.397 [0.359, 0.489] |
| 8 | 20 | 0.126 [0.109, 0.151] | 0.121 [0.105, 0.139] | 0.0014 [-0.0017, 0.0051] | 0.02 [0.01, 0.05] | 0.02 [0.01, 0.04] | 0.002 [-0.001, 0.009] | 0.16 [0.10, 0.19] | 0.29 [0.27, 0.34] | -0.119 [-0.242, -0.055] | 0.0201 [0.0141, 0.0239] | 8.7 [8.4, 9.5] | 15.6 [13.4, 17.5] | 0.511 [0.475, 0.632] |
| 16 | 10 | 0.109 [0.084, 0.141] | 0.108 [0.078, 0.135] | -0.0025 [-0.0060, 0.0068] | 0.02 [0.01, 0.04] | 0.03 [0.00, 0.06] | 0.000 [-0.010, 0.003] | 0.12 [0.09, 0.18] | 0.59 [0.53, 0.68] | -0.470 [-0.525, -0.417] | 0.0220 [0.0182, 0.0365] | 47.2 [35.0, 56.4] | 65.0 [60.1, 84.6] | 0.585 [0.511, 0.869] |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.0017 | 0.2611 | 1 | 0.006 | 0.2024 | 1 | no |
| 4 | 20 | 0.0002 | 0.9854 | 1 | 0.001 | 0.6477 | 1 | no |
| 8 | 20 | 0.0014 | 0.2305 | 1 | 0.002 | 0.1769 | 1 | no |
| 16 | 10 | -0.0025 | 0.9219 | 1 | 0.000 | 0.9219 | 1 | no |

## ring_D2_noise3_svbmc

Single runs (`M = 1`, n = 100): MMTV 0.627 [0.577, 0.673], gsKL 97.49 [29.42, 224.65], evidence error 1.38 [1.24, 1.53].

All 70 cells of this condition: runtime ratio 0.443 [0.421, 0.520], max\|dw\| 0.0070 [0.0061, 0.0088].

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.209 [0.030, 0.347] | 0.414 [0.285, 0.499] | 0.427 [0.288, 0.514] | 0.249 [0.066, 0.376] | -0.211 [-0.251, -0.158] | 1.014 [0.829, 1.213] | 1.009 [0.857, 1.189] | yes |
| 4 | 20 | 0.155 [0.049, 0.243] | 0.402 [0.276, 0.479] | 0.452 [0.318, 0.522] | 0.183 [0.065, 0.286] | -0.266 [-0.338, -0.189] | 0.531 [0.485, 0.628] | 0.541 [0.482, 0.629] | yes |
| 8 | 20 | 0.157 [0.097, 0.175] | 0.491 [0.378, 0.550] | 0.514 [0.394, 0.588] | 0.174 [0.117, 0.191] | -0.356 [-0.388, -0.291] | 0.329 [0.229, 0.378] | 0.330 [0.219, 0.374] | yes |
| 16 | 10 | 0.141 [0.111, 0.199] | 0.757 [0.690, 0.819] | 0.783 [0.710, 0.850] | 0.171 [0.139, 0.209] | -0.642 [-0.680, -0.546] | 0.152 [0.136, 0.161] | 0.146 [0.140, 0.153] | yes |

Growth of the integrated headline's median bias from `M = 2` to `M = 16`: -0.067 nats (0.209 to 0.141), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 2 | 2.154 [1.881, 2.347] | 2.154 [1.875, 2.354] | 2.164 [1.893, 2.345] | 2.197 [1.943, 2.396] | 1.520 [1.320, 1.705] | 0.011 | 1.525 [1.345, 1.676] | 0.011 |
| 4 | 2.595 [2.483, 2.646] | 2.593 [2.474, 2.647] | 2.593 [2.478, 2.650] | 2.628 [2.501, 2.674] | 2.002 [1.906, 2.049] | 0.010 | 1.992 [1.905, 2.051] | 0.010 |
| 8 | 2.802 [2.757, 2.885] | 2.800 [2.762, 2.885] | 2.808 [2.746, 2.887] | 2.840 [2.779, 2.907] | 2.205 [2.155, 2.305] | 0.010 | 2.203 [2.160, 2.314] | 0.010 |
| 16 | 2.948 [2.917, 2.967] | 2.951 [2.918, 2.964] | 2.944 [2.926, 2.967] | 2.961 [2.939, 2.991] | 2.382 [2.373, 2.399] | 0.009 | 2.388 [2.382, 2.394] | 0.009 |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.380 [0.346, 0.429] | 0.377 [0.346, 0.430] | 0.0002 [-0.0026, 0.0052] | 1.44 [0.88, 5.48] | 1.38 [0.77, 5.44] | 0.006 [-0.020, 0.033] | 0.85 [0.60, 1.01] | 0.52 [0.42, 0.85] | 0.203 [0.166, 0.261] | 0.0104 [0.0085, 0.0149] | 0.3 [0.3, 0.4] | 0.9 [0.8, 1.4] | 0.327 [0.295, 0.372] |
| 4 | 20 | 0.259 [0.241, 0.289] | 0.256 [0.244, 0.298] | 0.0008 [-0.0026, 0.0026] | 0.14 [0.11, 0.16] | 0.14 [0.11, 0.15] | 0.003 [-0.001, 0.006] | 0.42 [0.29, 0.57] | 0.18 [0.10, 0.27] | 0.242 [0.114, 0.304] | 0.0074 [0.0064, 0.0091] | 1.6 [1.3, 1.9] | 3.5 [2.6, 5.0] | 0.436 [0.404, 0.498] |
| 8 | 20 | 0.181 [0.160, 0.216] | 0.192 [0.160, 0.213] | 0.0006 [-0.0033, 0.0031] | 0.07 [0.05, 0.11] | 0.07 [0.05, 0.09] | 0.001 [-0.001, 0.010] | 0.24 [0.07, 0.29] | 0.21 [0.13, 0.37] | -0.007 [-0.294, 0.189] | 0.0055 [0.0050, 0.0077] | 9.1 [7.5, 11.0] | 16.2 [14.8, 19.7] | 0.563 [0.448, 0.674] |
| 16 | 10 | 0.134 [0.118, 0.151] | 0.128 [0.121, 0.149] | 0.0004 [-0.0054, 0.0086] | 0.03 [0.01, 0.04] | 0.03 [0.01, 0.03] | 0.001 [-0.001, 0.005] | 0.05 [0.02, 0.07] | 0.63 [0.56, 0.68] | -0.547 [-0.656, -0.488] | 0.0050 [0.0043, 0.0069] | 50.4 [44.9, 58.0] | 78.6 [67.7, 89.5] | 0.621 [0.497, 0.771] |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.0002 | 0.9273 | 1 | 0.006 | 0.8408 | 1 | no |
| 4 | 20 | 0.0008 | 0.8983 | 1 | 0.003 | 0.498 | 1 | no |
| 8 | 20 | 0.0006 | 0.9854 | 1 | 0.001 | 0.1769 | 1 | no |
| 16 | 10 | 0.0004 | 0.7695 | 1 | 0.001 | 0.1934 | 1 | no |

## student_D8_noise3_svbmc

Single runs (`M = 1`, n = 100): MMTV 0.143 [0.137, 0.146], gsKL 0.58 [0.52, 0.66], evidence error 0.96 [0.84, 1.11].

All 70 cells of this condition: runtime ratio 0.446 [0.383, 0.492], max\|dw\| 0.0281 [0.0206, 0.0335].

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | -0.945 [-1.182, -0.655] | -0.109 [-0.311, -0.039] | 0.037 [-0.122, 0.158] | -0.714 [-1.067, -0.534] | -0.993 [-1.257, -0.592] | 0.522 [0.421, 0.597] | 0.543 [0.409, 0.604] | **no** |
| 4 | 20 | -1.167 [-1.456, -0.974] | -0.100 [-0.175, 0.030] | -0.027 [-0.092, 0.115] | -1.089 [-1.282, -0.840] | -1.176 [-1.438, -0.907] | 0.432 [0.412, 0.452] | 0.429 [0.379, 0.466] | **no** |
| 8 | 20 | -1.391 [-1.612, -1.244] | 0.222 [0.094, 0.360] | 0.377 [0.102, 0.458] | -1.272 [-1.597, -0.982] | -1.802 [-2.032, -1.276] | 0.304 [0.273, 0.345] | 0.288 [0.276, 0.309] | **no** |
| 16 | 10 | -1.783 [-1.897, -1.526] | 0.274 [0.224, 0.439] | 0.410 [0.348, 0.578] | -1.704 [-1.871, -1.496] | -2.156 [-2.540, -1.956] | 0.251 [0.213, 0.304] | 0.247 [0.224, 0.304] | **no** |

Growth of the integrated headline's median bias from `M = 2` to `M = 16`: -0.838 nats (-0.945 to -1.783), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 2 | 13.545 [13.047, 13.921] | 13.540 [13.014, 13.880] | 13.551 [13.030, 13.907] | 13.655 [13.112, 13.999] | -20.126 [-20.200, -20.025] | 0.037 | -20.147 [-20.207, -20.013] | 0.037 |
| 4 | 13.697 [13.453, 13.847] | 13.716 [13.434, 13.829] | 13.740 [13.430, 13.905] | 13.837 [13.519, 13.999] | -20.036 [-20.056, -20.015] | 0.035 | -20.033 [-20.069, -19.982] | 0.035 |
| 8 | 13.241 [13.130, 13.563] | 13.213 [13.101, 13.558] | 13.352 [13.193, 13.612] | 13.418 [13.309, 13.682] | -19.908 [-19.948, -19.877] | 0.033 | -19.891 [-19.912, -19.879] | 0.032 |
| 16 | 13.266 [13.114, 13.352] | 13.236 [13.095, 13.369] | 13.211 [13.071, 13.314] | 13.334 [13.116, 13.400] | -19.855 [-19.907, -19.817] | 0.031 | -19.850 [-19.908, -19.828] | 0.031 |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.114 [0.101, 0.130] | 0.118 [0.102, 0.130] | -0.0005 [-0.0014, 0.0003] | 0.31 [0.24, 0.38] | 0.31 [0.24, 0.39] | -0.006 [-0.009, -0.000] | 1.35 [1.22, 1.58] | 0.40 [0.31, 0.77] | 1.020 [0.562, 1.218] | 0.0266 [0.0185, 0.0334] | 0.5 [0.5, 0.5] | 1.3 [1.1, 1.6] | 0.423 [0.336, 0.476] |
| 4 | 20 | 0.111 [0.102, 0.114] | 0.111 [0.100, 0.113] | 0.0001 [-0.0016, 0.0018] | 0.25 [0.22, 0.27] | 0.25 [0.17, 0.27] | 0.005 [-0.001, 0.012] | 1.63 [1.39, 1.93] | 0.47 [0.38, 0.57] | 1.186 [0.889, 1.330] | 0.0378 [0.0202, 0.0441] | 2.7 [2.5, 3.1] | 5.1 [3.4, 6.4] | 0.545 [0.420, 0.762] |
| 8 | 20 | 0.085 [0.082, 0.092] | 0.086 [0.082, 0.094] | -0.0015 [-0.0026, 0.0007] | 0.12 [0.10, 0.15] | 0.13 [0.11, 0.14] | -0.002 [-0.011, 0.003] | 1.72 [1.51, 2.02] | 0.18 [0.16, 0.25] | 1.435 [1.200, 1.758] | 0.0299 [0.0195, 0.0342] | 13.7 [11.8, 15.7] | 25.8 [20.8, 29.7] | 0.560 [0.430, 0.728] |
| 16 | 10 | 0.079 [0.075, 0.085] | 0.081 [0.074, 0.087] | -0.0007 [-0.0025, 0.0005] | 0.10 [0.07, 0.11] | 0.10 [0.08, 0.14] | -0.007 [-0.022, 0.002] | 2.01 [1.79, 2.20] | 0.14 [0.12, 0.28] | 1.804 [1.520, 1.985] | 0.0199 [0.0149, 0.0281] | 65.0 [57.3, 74.0] | 279.6 [236.8, 394.9] | 0.221 [0.188, 0.264] |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 2 | 20 | -0.0005 | 0.165 | 1 | -0.006 | 0.01718 | 0.5498 | no |
| 4 | 20 | 0.0001 | 0.8124 | 1 | 0.005 | 0.4304 | 1 | no |
| 8 | 20 | -0.0015 | 0.2305 | 1 | -0.002 | 0.5217 | 1 | no |
| 16 | 10 | -0.0007 | 0.3223 | 1 | -0.007 | 0.1602 | 1 | no |

## gmm_D2_svbmc

Single runs (`M = 1`, n = 50): MMTV 0.390 [0.365, 0.392], gsKL 14.92 [11.95, 15.40], evidence error 0.70 [0.70, 0.71].

All 70 cells of this condition: runtime ratio 0.540 [0.467, 0.597], max\|dw\| 0.0128 [0.0109, 0.0165].

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.007 [-0.009, 0.027] | 0.007 [-0.009, 0.027] | 0.050 [0.037, 0.058] | 0.024 [0.008, 0.047] | -0.038 [-0.052, -0.027] | 0.314 [0.290, 0.695] | 0.307 [0.285, 0.684] | yes |
| 4 | 20 | 0.001 [-0.004, 0.014] | 0.001 [-0.004, 0.014] | 0.051 [0.035, 0.062] | 0.031 [0.010, 0.054] | -0.043 [-0.055, -0.021] | 0.042 [0.015, 0.301] | 0.046 [0.020, 0.296] | yes |
| 8 | 20 | 0.002 [-0.005, 0.008] | 0.002 [-0.005, 0.008] | 0.037 [0.032, 0.039] | 0.009 [-0.022, 0.025] | -0.034 [-0.042, -0.026] | 0.012 [0.005, 0.014] | 0.010 [0.004, 0.019] | yes |
| 16 | 10 | 0.008 [-0.003, 0.016] | 0.008 [-0.003, 0.016] | 0.015 [0.009, 0.027] | -0.004 [-0.037, 0.012] | -0.009 [-0.024, 0.002] | 0.009 [-0.000, 0.023] | 0.008 [-0.002, 0.016] | yes |

Growth of the integrated headline's median bias from `M = 2` to `M = 16`: 0.001 nats (0.007 to 0.008), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 2 | 4.313 [3.954, 4.330] | 4.296 [3.958, 4.343] | 4.312 [3.964, 4.330] | 4.349 [4.028, 4.401] | 2.682 [2.301, 2.705] | 0.015 | 2.688 [2.312, 2.711] | 0.015 |
| 4 | 4.601 [4.344, 4.627] | 4.604 [4.325, 4.624] | 4.596 [4.337, 4.628] | 4.628 [4.375, 4.656] | 2.954 [2.695, 2.981] | 0.014 | 2.950 [2.700, 2.976] | 0.013 |
| 8 | 4.635 [4.625, 4.641] | 4.629 [4.622, 4.642] | 4.629 [4.625, 4.638] | 4.656 [4.650, 4.663] | 2.984 [2.982, 2.990] | 0.013 | 2.985 [2.976, 2.992] | 0.012 |
| 16 | 4.633 [4.617, 4.634] | 4.629 [4.619, 4.633] | 4.626 [4.621, 4.630] | 4.642 [4.633, 4.652] | 2.987 [2.973, 2.996] | 0.012 | 2.988 [2.980, 2.998] | 0.012 |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.196 [0.188, 0.310] | 0.194 [0.185, 0.307] | 0.0026 [-0.0019, 0.0042] | 0.91 [0.22, 8.09] | 0.91 [0.22, 8.00] | 0.002 [-0.004, 0.021] | 0.31 [0.28, 0.68] | 0.26 [0.24, 0.62] | 0.038 [0.022, 0.055] | 0.0154 [0.0125, 0.0212] | 0.3 [0.2, 0.3] | 0.7 [0.7, 0.8] | 0.356 [0.335, 0.406] |
| 4 | 20 | 0.047 [0.031, 0.190] | 0.073 [0.029, 0.191] | -0.0017 [-0.0112, 0.0040] | 0.01 [0.00, 0.39] | 0.03 [0.00, 0.38] | 0.000 [-0.004, 0.003] | 0.03 [0.01, 0.30] | 0.07 [0.03, 0.25] | 0.009 [-0.002, 0.029] | 0.0174 [0.0131, 0.0208] | 1.2 [1.1, 1.3] | 2.5 [2.1, 2.6] | 0.533 [0.462, 0.614] |
| 8 | 20 | 0.028 [0.023, 0.036] | 0.027 [0.022, 0.036] | -0.0036 [-0.0097, 0.0098] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | -0.000 [-0.001, 0.003] | 0.01 [0.01, 0.01] | 0.02 [0.02, 0.03] | -0.010 [-0.015, -0.006] | 0.0104 [0.0084, 0.0148] | 4.9 [4.4, 5.9] | 8.6 [7.1, 10.2] | 0.597 [0.552, 0.692] |
| 16 | 10 | 0.024 [0.017, 0.032] | 0.022 [0.014, 0.034] | 0.0012 [-0.0089, 0.0122] | 0.00 [0.00, 0.01] | 0.00 [0.00, 0.01] | 0.000 [-0.003, 0.005] | 0.00 [0.00, 0.01] | 0.01 [0.01, 0.02] | -0.006 [-0.010, 0.000] | 0.0056 [0.0046, 0.0073] | 18.9 [15.5, 21.9] | 21.4 [17.5, 24.9] | 0.829 [0.762, 1.093] |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.0026 | 0.4749 | 1 | 0.002 | 0.5958 | 1 | no |
| 4 | 20 | -0.0017 | 0.5706 | 1 | 0.000 | 0.9854 | 1 | no |
| 8 | 20 | -0.0036 | 0.8983 | 1 | -0.000 | 0.5459 | 1 | no |
| 16 | 10 | 0.0012 | 0.6953 | 1 | 0.000 | 0.6953 | 1 | no |

## multisensory_s1_D6_svbmc

Single runs (`M = 1`, n = 50): MMTV 0.118 [0.112, 0.127], gsKL 0.59 [0.42, 0.71], evidence error 0.37 [0.34, 0.42].

All 70 cells of this condition: runtime ratio 0.334 [0.285, 0.361], max\|dw\| 0.0132 [0.0115, 0.0162].

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.029 [0.020, 0.061] | 0.029 [0.020, 0.061] | 0.108 [0.090, 0.133] | 0.018 [-0.024, 0.071] | -0.075 [-0.099, -0.064] | 0.322 [0.300, 0.362] | 0.321 [0.300, 0.355] | yes |
| 4 | 20 | 0.045 [0.020, 0.068] | 0.045 [0.020, 0.068] | 0.132 [0.107, 0.150] | 0.080 [0.014, 0.106] | -0.083 [-0.092, -0.051] | 0.248 [0.203, 0.280] | 0.258 [0.225, 0.297] | yes |
| 8 | 20 | 0.072 [0.054, 0.082] | 0.072 [0.054, 0.082] | 0.113 [0.096, 0.135] | 0.075 [0.046, 0.088] | -0.045 [-0.076, -0.025] | 0.185 [0.155, 0.217] | 0.175 [0.153, 0.221] | yes |
| 16 | 10 | 0.055 [0.034, 0.085] | 0.055 [0.034, 0.085] | 0.135 [0.107, 0.155] | 0.099 [0.070, 0.133] | -0.072 [-0.094, -0.057] | 0.143 [0.119, 0.167] | 0.158 [0.141, 0.186] | yes |

Growth of the integrated headline's median bias from `M = 2` to `M = 16`: 0.025 nats (0.029 to 0.055), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 2 | 3.671 [3.622, 3.696] | 3.667 [3.619, 3.689] | 3.641 [3.616, 3.682] | 3.721 [3.691, 3.744] | -502.508 [-502.548, -502.486] | 0.025 | -502.507 [-502.541, -502.486] | 0.024 |
| 4 | 3.743 [3.695, 3.778] | 3.726 [3.681, 3.769] | 3.711 [3.685, 3.758] | 3.776 [3.746, 3.821] | -502.434 [-502.466, -502.389] | 0.022 | -502.444 [-502.483, -502.411] | 0.022 |
| 8 | 3.843 [3.797, 3.879] | 3.851 [3.791, 3.876] | 3.830 [3.805, 3.854] | 3.883 [3.838, 3.902] | -502.371 [-502.402, -502.341] | 0.020 | -502.361 [-502.407, -502.339] | 0.020 |
| 16 | 3.887 [3.840, 3.920] | 3.884 [3.831, 3.921] | 3.898 [3.877, 3.921] | 3.933 [3.906, 3.949] | -502.328 [-502.353, -502.305] | 0.020 | -502.343 [-502.368, -502.327] | 0.020 |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.104 [0.090, 0.115] | 0.101 [0.089, 0.115] | 0.0015 [-0.0075, 0.0057] | 0.44 [0.29, 0.56] | 0.44 [0.30, 0.59] | -0.002 [-0.033, 0.027] | 0.28 [0.26, 0.31] | 0.21 [0.18, 0.23] | 0.075 [0.064, 0.098] | 0.0184 [0.0173, 0.0205] | 0.5 [0.4, 0.5] | 1.5 [1.2, 1.9] | 0.310 [0.239, 0.386] |
| 4 | 20 | 0.090 [0.074, 0.103] | 0.093 [0.067, 0.104] | -0.0026 [-0.0060, 0.0006] | 0.32 [0.16, 0.48] | 0.30 [0.14, 0.48] | -0.018 [-0.033, 0.001] | 0.21 [0.15, 0.24] | 0.14 [0.10, 0.18] | 0.056 [0.041, 0.079] | 0.0163 [0.0124, 0.0194] | 1.7 [1.5, 1.9] | 4.4 [3.6, 6.1] | 0.360 [0.328, 0.459] |
| 8 | 20 | 0.043 [0.031, 0.067] | 0.047 [0.035, 0.068] | -0.0035 [-0.0086, 0.0077] | 0.05 [0.03, 0.15] | 0.07 [0.04, 0.14] | -0.009 [-0.028, 0.013] | 0.11 [0.09, 0.15] | 0.06 [0.05, 0.12] | 0.047 [0.023, 0.063] | 0.0100 [0.0084, 0.0122] | 10.2 [8.8, 11.8] | 31.8 [26.7, 33.9] | 0.371 [0.307, 0.411] |
| 16 | 10 | 0.046 [0.034, 0.059] | 0.033 [0.027, 0.047] | 0.0083 [-0.0023, 0.0152] | 0.06 [0.03, 0.11] | 0.03 [0.02, 0.07] | 0.010 [-0.002, 0.039] | 0.08 [0.06, 0.10] | 0.04 [0.03, 0.05] | 0.043 [0.029, 0.067] | 0.0071 [0.0059, 0.0111] | 39.6 [35.3, 43.7] | 177.4 [150.2, 241.3] | 0.212 [0.147, 0.277] |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.0015 | 0.8695 | 1 | -0.002 | 0.7012 | 1 | no |
| 4 | 20 | -0.0026 | 0.2611 | 1 | -0.018 | 0.1536 | 1 | no |
| 8 | 20 | -0.0035 | 0.7012 | 1 | -0.009 | 0.6477 | 1 | no |
| 16 | 10 | 0.0083 | 0.1309 | 1 | 0.010 | 0.1602 | 1 | no |
