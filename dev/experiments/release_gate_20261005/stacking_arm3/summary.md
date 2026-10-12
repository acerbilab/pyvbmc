# S-VBMC stacking comparison: integrated against original 0.1.1

Generated 2026-10-05 22:23:27. Every cell stacks the same subset of filtered pool runs with both implementations, `n_samples=20`, `lr=0.1`, `max_steps=500`, `version="all-weights"`, subsets drawn with seed 0, disjoint within each `M`. Medians over the repetitions of a cell with a 10 000-resample bootstrap 95 % interval; `d` columns are the paired difference integrated minus original, so a negative value favours the integrated class. The headline ELBO is the capped value for the integrated class on noisy stacks and the raw estimate for the original, which is what each implementation reports. gsKL is the house convention (no `1/D` factor).

Every reported ELBO is scored by its bias against `elbo_mc = e_log_joint_mc + entropy_ref`, the Monte Carlo ELBO of the posterior that arm's fit produced. `e_log_joint_mc` is the mean of the target's noiseless log density over 10 000 of the stack's own draws; `entropy_ref` is the entropy of the stacked mixture at that arm's final weights, estimated here for both arms by one estimator (the integrated class's `stacked_entropy`, 200 draws per component in 8 batches, from a generator seeded by the cell, the same draws for both arms). The reference is arm-independent by construction: an arm's own reported entropy is estimated at the weights that arm selected, from its own draws (the optimizer's 20 per component for the original, a fresh 100 for the integrated class), so a reference built on it would move with the arm and credit whichever entropy estimate came out high. `kl_gap = ln Z - elbo_mc` is how far the stacked posterior itself is from the target, in evidence units; the `err` columns further below are the error against `ln Z`, descriptive only.

## multisensory_s1_D6_noise3_svbmc

Single runs (`M = 1`, n = 320): MMTV 0.268 [0.262, 0.274], gsKL 3.87 [3.58, 4.11], evidence error 0.43 [0.40, 0.48].

All 120 cells of this condition: runtime ratio 0.188 [0.170, 0.216], max\|dw\| 0.0201 [0.0171, 0.0211].

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. The shrinkage column shows its contributing and numerically unavailable cell counts; older cells that did not record shrinkage are counted separately. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias shrink int (contributing / unavailable / not recorded) | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.333 [0.273, 0.568] | 0.725 [0.592, 0.826] | 0.462 [0.339, 0.622] (20 / 0 / 0) | 0.812 [0.686, 0.958] | 0.537 [0.308, 0.590] | -0.468 [-0.587, -0.387] | 0.760 [0.696, 0.837] | 0.756 [0.704, 0.852] | yes |
| 3 | 20 | 0.334 [0.197, 0.481] | 0.720 [0.620, 0.823] | 0.406 [0.303, 0.523] (20 / 0 / 0) | - | - | - | 0.645 [0.557, 0.754] | - | - |
| 4 | 20 | 0.392 [0.285, 0.439] | 0.827 [0.733, 0.923] | 0.506 [0.467, 0.561] (20 / 0 / 0) | 0.946 [0.787, 1.026] | 0.458 [0.353, 0.528] | -0.512 [-0.656, -0.427] | 0.654 [0.510, 0.728] | 0.624 [0.544, 0.704] | yes |
| 5 | 20 | 0.397 [0.306, 0.494] | 0.926 [0.788, 1.005] | 0.528 [0.465, 0.656] (20 / 0 / 0) | - | - | - | 0.541 [0.465, 0.650] | - | - |
| 8 | 20 | 0.389 [0.291, 0.425] | 1.006 [0.955, 1.051] | 0.579 [0.486, 0.619] (20 / 0 / 0) | 1.056 [1.018, 1.102] | 0.453 [0.363, 0.516] | -0.686 [-0.822, -0.613] | 0.473 [0.388, 0.575] | 0.451 [0.395, 0.583] | yes |
| 16 | 10 | 0.354 [0.334, 0.380] | 1.020 [0.980, 1.216] | 0.525 [0.494, 0.706] (10 / 0 / 0) | 1.091 [1.003, 1.268] | 0.429 [0.363, 0.494] | -0.767 [-0.875, -0.630] | 0.390 [0.263, 0.426] | 0.358 [0.247, 0.440] | yes |
| 32 | 10 | 0.431 [0.330, 0.514] | 1.216 [1.147, 1.339] | 0.701 [0.609, 0.775] (10 / 0 / 0) | - | - | - | 0.291 [0.266, 0.342] | - | - |

Growth of the integrated headline's median bias from `M = 2` to `M = 32`: 0.099 nats (0.333 to 0.431), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 2 | 3.052 [2.833, 3.113] | 3.075 [2.839, 3.125] | 3.054 [2.812, 3.131] | 3.114 [2.873, 3.244] | -502.946 [-503.022, -502.882] | 0.023 | -502.942 [-503.038, -502.889] | 0.023 |
| 3 | 3.117 [3.008, 3.254] | 3.088 [3.018, 3.205] | - | - | -502.831 [-502.939, -502.743] | 0.024 | - | - |
| 4 | 3.197 [3.031, 3.296] | 3.178 [3.027, 3.268] | 3.171 [3.057, 3.316] | 3.249 [3.148, 3.367] | -502.840 [-502.914, -502.696] | 0.021 | -502.810 [-502.890, -502.730] | 0.021 |
| 5 | 3.266 [3.142, 3.406] | 3.280 [3.147, 3.412] | - | - | -502.727 [-502.836, -502.650] | 0.021 | - | - |
| 8 | 3.310 [3.154, 3.459] | 3.350 [3.153, 3.450] | 3.322 [3.165, 3.456] | 3.378 [3.243, 3.535] | -502.659 [-502.761, -502.574] | 0.020 | -502.637 [-502.769, -502.581] | 0.021 |
| 16 | 3.406 [3.342, 3.545] | 3.426 [3.325, 3.540] | 3.442 [3.331, 3.567] | 3.508 [3.397, 3.607] | -502.576 [-502.611, -502.449] | 0.020 | -502.543 [-502.626, -502.432] | 0.020 |
| 32 | 3.520 [3.468, 3.579] | 3.532 [3.454, 3.593] | - | - | -502.477 [-502.527, -502.455] | 0.019 | - | - |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.216 [0.181, 0.249] | 0.219 [0.175, 0.248] | 0.0009 [-0.0032, 0.0041] | 2.61 [1.20, 3.59] | 2.69 [1.17, 3.64] | -0.021 [-0.071, 0.013] | 0.45 [0.25, 0.57] | 0.25 [0.14, 0.35] | 0.325 [0.003, 0.389] | 0.0247 [0.0201, 0.0330] | 0.4 [0.3, 0.4] | 2.1 [1.9, 2.2] | 0.165 [0.143, 0.216] |
| 3 | 20 | 0.187 [0.169, 0.225] | - | - | 1.73 [1.13, 2.78] | - | - | 0.32 [0.17, 0.38] | - | - | - | 1.1 [1.0, 1.3] | - | - |
| 4 | 20 | 0.182 [0.161, 0.227] | 0.178 [0.168, 0.225] | 0.0021 [-0.0017, 0.0070] | 1.59 [1.14, 2.39] | 1.60 [1.14, 2.40] | 0.019 [-0.004, 0.080] | 0.28 [0.15, 0.35] | 0.30 [0.23, 0.37] | 0.007 [-0.125, 0.123] | 0.0206 [0.0166, 0.0259] | 2.4 [1.9, 3.0] | 11.7 [10.5, 12.9] | 0.209 [0.188, 0.239] |
| 5 | 20 | 0.159 [0.133, 0.214] | - | - | 1.23 [0.50, 2.00] | - | - | 0.20 [0.09, 0.31] | - | - | - | 3.6 [3.3, 3.9] | - | - |
| 8 | 20 | 0.136 [0.109, 0.190] | 0.134 [0.108, 0.188] | 0.0008 [-0.0018, 0.0030] | 0.56 [0.27, 1.59] | 0.57 [0.27, 1.62] | -0.003 [-0.030, 0.015] | 0.14 [0.07, 0.26] | 0.61 [0.51, 0.66] | -0.458 [-0.583, -0.284] | 0.0165 [0.0131, 0.0208] | 10.3 [9.4, 11.1] | 54.0 [44.8, 62.4] | 0.176 [0.153, 0.247] |
| 16 | 10 | 0.132 [0.076, 0.160] | 0.131 [0.074, 0.162] | 0.0003 [-0.0037, 0.0016] | 0.41 [0.13, 0.80] | 0.40 [0.13, 0.98] | 0.001 [-0.028, 0.003] | 0.06 [0.04, 0.10] | 0.78 [0.67, 0.87] | -0.706 [-0.822, -0.580] | 0.0128 [0.0112, 0.0194] | 52.9 [51.6, 59.6] | 276.7 [220.7, 354.1] | 0.186 [0.152, 0.307] |
| 32 | 10 | 0.107 [0.097, 0.131] | - | - | 0.20 [0.17, 0.27] | - | - | 0.12 [0.04, 0.18] | - | - | - | 365.2 [297.1, 402.0] | - | - |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.0009 | 0.8695 | 1 | -0.021 | 0.3488 | 1 | no |
| 3 | 0 | - | - | - | - | - | - | - |
| 4 | 20 | 0.0021 | 0.165 | 1 | 0.019 | 0.07585 | 1 | no |
| 5 | 0 | - | - | - | - | - | - | - |
| 8 | 20 | 0.0008 | 0.5958 | 1 | -0.003 | 0.4749 | 1 | no |
| 16 | 10 | 0.0003 | 0.8457 | 1 | 0.001 | 0.8457 | 1 | no |
| 32 | 0 | - | - | - | - | - | - | - |

## multisensory_s1_D6_noise1.3_svbmc

Single runs (`M = 1`, n = 320): MMTV 0.221 [0.216, 0.226], gsKL 2.59 [2.44, 2.73], evidence error 0.42 [0.38, 0.44].

All 120 cells of this condition: runtime ratio 0.204 [0.188, 0.238], max\|dw\| 0.0193 [0.0157, 0.0231].

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. The shrinkage column shows its contributing and numerically unavailable cell counts; older cells that did not record shrinkage are counted separately. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias shrink int (contributing / unavailable / not recorded) | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.134 [-0.017, 0.272] | 0.363 [0.294, 0.497] | 0.211 [0.097, 0.291] (20 / 0 / 0) | 0.443 [0.387, 0.578] | 0.277 [0.027, 0.374] | -0.380 [-0.452, -0.300] | 0.580 [0.532, 0.612] | 0.587 [0.547, 0.602] | yes |
| 3 | 20 | -0.024 [-0.089, 0.103] | 0.373 [0.327, 0.433] | 0.157 [0.107, 0.228] (20 / 0 / 0) | - | - | - | 0.421 [0.298, 0.544] | - | - |
| 4 | 20 | 0.085 [0.041, 0.173] | 0.435 [0.402, 0.511] | 0.247 [0.171, 0.290] (20 / 0 / 0) | 0.521 [0.421, 0.564] | 0.167 [0.066, 0.241] | -0.393 [-0.470, -0.328] | 0.393 [0.318, 0.448] | 0.408 [0.325, 0.466] | yes |
| 5 | 20 | 0.127 [0.016, 0.174] | 0.483 [0.421, 0.545] | 0.249 [0.218, 0.317] (20 / 0 / 0) | - | - | - | 0.349 [0.292, 0.513] | - | - |
| 8 | 20 | 0.113 [0.066, 0.175] | 0.556 [0.517, 0.593] | 0.307 [0.273, 0.371] (20 / 0 / 0) | 0.639 [0.498, 0.674] | 0.181 [0.116, 0.223] | -0.429 [-0.541, -0.374] | 0.379 [0.263, 0.474] | 0.351 [0.270, 0.472] | yes |
| 16 | 10 | 0.114 [0.078, 0.185] | 0.656 [0.586, 0.686] | 0.367 [0.330, 0.415] (10 / 0 / 0) | 0.727 [0.585, 0.803] | 0.202 [0.132, 0.236] | -0.580 [-0.710, -0.434] | 0.290 [0.226, 0.335] | 0.301 [0.221, 0.351] | yes |
| 32 | 10 | 0.118 [0.068, 0.166] | 0.738 [0.672, 0.794] | 0.446 [0.378, 0.482] (10 / 0 / 0) | - | - | - | 0.190 [0.140, 0.215] | - | - |

Growth of the integrated headline's median bias from `M = 2` to `M = 32`: -0.016 nats (0.134 to 0.118), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 2 | 3.152 [3.095, 3.204] | 3.149 [3.065, 3.202] | 3.150 [3.097, 3.223] | 3.243 [3.174, 3.291] | -502.766 [-502.798, -502.718] | 0.024 | -502.773 [-502.788, -502.733] | 0.024 |
| 3 | 3.301 [3.195, 3.435] | 3.279 [3.187, 3.429] | - | - | -502.606 [-502.730, -502.484] | 0.023 | - | - |
| 4 | 3.344 [3.253, 3.456] | 3.349 [3.244, 3.465] | 3.345 [3.253, 3.464] | 3.454 [3.322, 3.502] | -502.579 [-502.634, -502.504] | 0.020 | -502.593 [-502.652, -502.511] | 0.019 |
| 5 | 3.324 [3.228, 3.482] | 3.331 [3.217, 3.475] | - | - | -502.535 [-502.699, -502.478] | 0.020 | - | - |
| 8 | 3.336 [3.289, 3.492] | 3.333 [3.290, 3.500] | 3.367 [3.305, 3.492] | 3.424 [3.351, 3.546] | -502.565 [-502.660, -502.448] | 0.020 | -502.537 [-502.658, -502.456] | 0.020 |
| 16 | 3.476 [3.362, 3.576] | 3.481 [3.353, 3.580] | 3.476 [3.380, 3.547] | 3.540 [3.426, 3.595] | -502.476 [-502.521, -502.411] | 0.020 | -502.487 [-502.537, -502.407] | 0.019 |
| 32 | 3.588 [3.494, 3.665] | 3.590 [3.501, 3.663] | - | - | -502.376 [-502.401, -502.326] | 0.018 | - | - |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.201 [0.180, 0.212] | 0.200 [0.185, 0.210] | 0.0006 [-0.0029, 0.0022] | 2.01 [1.50, 2.49] | 1.96 [1.44, 2.59] | -0.036 [-0.062, 0.082] | 0.48 [0.24, 0.60] | 0.17 [0.11, 0.23] | 0.326 [0.169, 0.404] | 0.0232 [0.0187, 0.0285] | 0.4 [0.3, 0.5] | 1.9 [1.5, 2.1] | 0.209 [0.155, 0.283] |
| 3 | 20 | 0.154 [0.097, 0.202] | - | - | 0.95 [0.36, 1.98] | - | - | 0.40 [0.33, 0.55] | - | - | - | 1.0 [1.0, 1.1] | - | - |
| 4 | 20 | 0.144 [0.113, 0.167] | 0.152 [0.113, 0.170] | -0.0018 [-0.0036, 0.0010] | 0.75 [0.30, 1.35] | 0.80 [0.27, 1.32] | -0.010 [-0.062, -0.005] | 0.23 [0.17, 0.46] | 0.14 [0.11, 0.21] | 0.105 [-0.047, 0.368] | 0.0176 [0.0140, 0.0243] | 2.2 [1.9, 2.5] | 10.1 [8.6, 11.8] | 0.237 [0.185, 0.262] |
| 5 | 20 | 0.135 [0.109, 0.186] | - | - | 0.59 [0.25, 2.01] | - | - | 0.28 [0.21, 0.36] | - | - | - | 3.4 [3.2, 3.9] | - | - |
| 8 | 20 | 0.148 [0.110, 0.180] | 0.157 [0.118, 0.177] | 0.0009 [-0.0018, 0.0061] | 1.06 [0.37, 1.52] | 1.14 [0.34, 1.40] | 0.033 [-0.007, 0.070] | 0.31 [0.18, 0.38] | 0.24 [0.15, 0.33] | 0.122 [-0.146, 0.214] | 0.0199 [0.0134, 0.0241] | 10.1 [9.5, 11.0] | 53.0 [50.7, 58.1] | 0.196 [0.167, 0.250] |
| 16 | 10 | 0.126 [0.096, 0.147] | 0.124 [0.095, 0.149] | -0.0000 [-0.0045, 0.0032] | 0.41 [0.20, 0.77] | 0.41 [0.22, 0.83] | -0.013 [-0.054, 0.024] | 0.14 [0.06, 0.31] | 0.41 [0.35, 0.48] | -0.300 [-0.362, -0.102] | 0.0132 [0.0113, 0.0213] | 63.1 [52.8, 81.1] | 337.1 [295.3, 383.2] | 0.193 [0.159, 0.236] |
| 32 | 10 | 0.085 [0.066, 0.101] | - | - | 0.10 [0.08, 0.28] | - | - | 0.05 [0.01, 0.14] | - | - | - | 428.1 [409.8, 493.3] | - | - |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.0006 | 0.8983 | 1 | -0.036 | 0.9854 | 1 | no |
| 3 | 0 | - | - | - | - | - | - | - |
| 4 | 20 | -0.0018 | 0.4524 | 1 | -0.010 | 0.05826 | 1 | no |
| 5 | 0 | - | - | - | - | - | - | - |
| 8 | 20 | 0.0009 | 0.6742 | 1 | 0.033 | 0.3884 | 1 | no |
| 16 | 10 | -0.0000 | 0.9219 | 1 | -0.013 | 0.2324 | 1 | no |
| 32 | 0 | - | - | - | - | - | - | - |

## rosenbrock_D2_noise3_svbmc

Single runs (`M = 1`, n = 320): MMTV 0.094 [0.084, 0.113], gsKL 0.12 [0.09, 0.16], evidence error 0.19 [0.18, 0.22].

All 120 cells of this condition: runtime ratio 0.371 [0.343, 0.401], max\|dw\| 0.0235 [0.0193, 0.0272].

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. The shrinkage column shows its contributing and numerically unavailable cell counts; older cells that did not record shrinkage are counted separately. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias shrink int (contributing / unavailable / not recorded) | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.032 [-0.132, 0.209] | 0.220 [0.011, 0.429] | 0.023 [-0.192, 0.117] (20 / 0 / 0) | 0.284 [0.087, 0.468] | 0.067 [-0.067, 0.271] | -0.246 [-0.313, -0.172] | 0.072 [0.038, 0.116] | 0.080 [0.046, 0.117] | yes |
| 3 | 20 | 0.015 [-0.055, 0.106] | 0.444 [0.362, 0.527] | 0.158 [-0.002, 0.216] (20 / 0 / 0) | - | - | - | 0.040 [0.023, 0.117] | - | - |
| 4 | 20 | 0.075 [0.008, 0.206] | 0.469 [0.398, 0.523] | 0.100 [-0.066, 0.223] (20 / 0 / 0) | 0.475 [0.399, 0.510] | 0.117 [0.038, 0.243] | -0.404 [-0.498, -0.308] | 0.047 [0.034, 0.099] | 0.043 [0.032, 0.074] | yes |
| 5 | 20 | 0.056 [-0.019, 0.156] | 0.496 [0.459, 0.588] | 0.143 [0.074, 0.186] (20 / 0 / 0) | - | - | - | 0.057 [0.030, 0.102] | - | - |
| 8 | 20 | 0.108 [0.019, 0.148] | 0.665 [0.598, 0.703] | 0.179 [0.072, 0.248] (18 / 2 / 0) | 0.704 [0.630, 0.758] | 0.165 [0.057, 0.215] | -0.607 [-0.699, -0.521] | 0.062 [0.036, 0.107] | 0.074 [0.046, 0.097] | yes |
| 16 | 10 | 0.027 [-0.020, 0.146] | 0.770 [0.662, 0.911] | 0.149 [-0.061, 0.262] (8 / 2 / 0) | 0.846 [0.698, 1.008] | 0.064 [0.041, 0.234] | -0.730 [-0.994, -0.570] | 0.096 [0.057, 0.140] | 0.095 [0.061, 0.176] | yes |
| 32 | 10 | 0.112 [0.043, 0.175] | 0.958 [0.881, 1.130] | 0.193 [-0.019, 0.312] (6 / 4 / 0) | - | - | - | 0.094 [0.062, 0.127] | - | - |

Growth of the integrated headline's median bias from `M = 2` to `M = 32`: 0.080 nats (0.032 to 0.112), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 2 | 2.494 [2.445, 2.546] | 2.491 [2.449, 2.552] | 2.494 [2.451, 2.547] | 2.541 [2.493, 2.596] | -2.331 [-2.376, -2.298] | 0.012 | -2.340 [-2.377, -2.306] | 0.012 |
| 3 | 2.524 [2.472, 2.558] | 2.522 [2.456, 2.552] | - | - | -2.300 [-2.377, -2.283] | 0.011 | - | - |
| 4 | 2.505 [2.468, 2.564] | 2.520 [2.466, 2.563] | 2.533 [2.468, 2.566] | 2.586 [2.512, 2.634] | -2.306 [-2.359, -2.294] | 0.012 | -2.303 [-2.334, -2.292] | 0.012 |
| 5 | 2.534 [2.404, 2.568] | 2.543 [2.406, 2.560] | - | - | -2.317 [-2.362, -2.289] | 0.011 | - | - |
| 8 | 2.529 [2.492, 2.563] | 2.532 [2.492, 2.577] | 2.526 [2.498, 2.555] | 2.569 [2.545, 2.600] | -2.322 [-2.366, -2.296] | 0.011 | -2.334 [-2.357, -2.306] | 0.012 |
| 16 | 2.489 [2.429, 2.592] | 2.489 [2.420, 2.598] | 2.483 [2.425, 2.582] | 2.538 [2.489, 2.639] | -2.356 [-2.400, -2.317] | 0.011 | -2.355 [-2.432, -2.320] | 0.011 |
| 32 | 2.527 [2.502, 2.555] | 2.521 [2.496, 2.566] | - | - | -2.354 [-2.391, -2.322] | 0.012 | - | - |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.086 [0.070, 0.114] | 0.093 [0.076, 0.103] | -0.0019 [-0.0090, 0.0020] | 0.06 [0.03, 0.09] | 0.06 [0.03, 0.09] | -0.004 [-0.012, 0.003] | 0.17 [0.10, 0.21] | 0.24 [0.10, 0.34] | -0.091 [-0.229, 0.069] | 0.0258 [0.0166, 0.0301] | 0.3 [0.3, 0.4] | 1.4 [1.3, 1.5] | 0.219 [0.189, 0.315] |
| 3 | 20 | 0.081 [0.055, 0.126] | - | - | 0.04 [0.02, 0.07] | - | - | 0.14 [0.08, 0.23] | - | - | - | 0.9 [0.9, 1.1] | - | - |
| 4 | 20 | 0.067 [0.054, 0.091] | 0.069 [0.055, 0.095] | -0.0008 [-0.0057, 0.0049] | 0.02 [0.01, 0.03] | 0.03 [0.01, 0.03] | -0.001 [-0.004, 0.002] | 0.11 [0.07, 0.21] | 0.37 [0.34, 0.44] | -0.286 [-0.370, -0.135] | 0.0290 [0.0238, 0.0323] | 1.8 [1.7, 1.9] | 4.6 [3.6, 5.0] | 0.385 [0.328, 0.489] |
| 5 | 20 | 0.081 [0.060, 0.122] | - | - | 0.03 [0.01, 0.04] | - | - | 0.10 [0.06, 0.19] | - | - | - | 3.1 [2.8, 3.5] | - | - |
| 8 | 20 | 0.076 [0.057, 0.097] | 0.073 [0.063, 0.094] | -0.0006 [-0.0056, 0.0024] | 0.02 [0.01, 0.03] | 0.02 [0.01, 0.02] | -0.000 [-0.001, 0.001] | 0.11 [0.09, 0.18] | 0.61 [0.58, 0.67] | -0.532 [-0.579, -0.381] | 0.0193 [0.0160, 0.0234] | 8.6 [8.0, 9.7] | 21.4 [20.0, 21.9] | 0.407 [0.359, 0.448] |
| 16 | 10 | 0.100 [0.075, 0.124] | 0.092 [0.068, 0.142] | -0.0056 [-0.0233, 0.0096] | 0.03 [0.01, 0.04] | 0.03 [0.01, 0.05] | -0.001 [-0.015, 0.008] | 0.10 [0.08, 0.16] | 0.75 [0.62, 0.82] | -0.624 [-0.713, -0.496] | 0.0145 [0.0117, 0.0297] | 47.2 [39.4, 51.5] | 102.5 [90.5, 142.1] | 0.401 [0.369, 0.485] |
| 32 | 10 | 0.102 [0.072, 0.126] | - | - | 0.02 [0.01, 0.03] | - | - | 0.08 [0.02, 0.13] | - | - | - | 331.7 [308.2, 362.4] | - | - |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 2 | 20 | -0.0019 | 0.1769 | 1 | -0.004 | 0.2774 | 1 | no |
| 3 | 0 | - | - | - | - | - | - | - |
| 4 | 20 | -0.0008 | 0.9854 | 1 | -0.001 | 0.4524 | 1 | no |
| 5 | 0 | - | - | - | - | - | - | - |
| 8 | 20 | -0.0006 | 0.5459 | 1 | -0.000 | 0.7285 | 1 | no |
| 16 | 10 | -0.0056 | 0.625 | 1 | -0.001 | 0.5566 | 1 | no |
| 32 | 0 | - | - | - | - | - | - | - |

## gmm_D2_noise3_svbmc

Single runs (`M = 1`, n = 320): MMTV 0.466 [0.442, 0.504], gsKL 20.32 [18.37, 24.98], evidence error 1.05 [0.93, 1.12].

All 120 cells of this condition: runtime ratio 0.441 [0.387, 0.479], max\|dw\| 0.0190 [0.0162, 0.0219].

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. The shrinkage column shows its contributing and numerically unavailable cell counts; older cells that did not record shrinkage are counted separately. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias shrink int (contributing / unavailable / not recorded) | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.111 [-0.316, 0.278] | 0.275 [0.094, 0.415] | 0.040 [-0.193, 0.125] (20 / 0 / 0) | 0.320 [0.181, 0.486] | 0.168 [-0.274, 0.326] | -0.235 [-0.425, -0.166] | 0.755 [0.548, 0.870] | 0.755 [0.558, 0.864] | yes |
| 3 | 20 | -0.006 [-0.183, 0.209] | 0.265 [0.140, 0.411] | -0.026 [-0.207, 0.189] (20 / 0 / 0) | - | - | - | 0.460 [0.414, 0.497] | - | - |
| 4 | 20 | 0.096 [-0.039, 0.233] | 0.355 [0.208, 0.485] | 0.073 [-0.027, 0.200] (20 / 0 / 0) | 0.403 [0.323, 0.515] | 0.140 [0.047, 0.334] | -0.293 [-0.415, -0.201] | 0.450 [0.251, 0.464] | 0.448 [0.256, 0.470] | yes |
| 5 | 20 | 0.019 [-0.092, 0.147] | 0.388 [0.309, 0.487] | 0.067 [-0.081, 0.122] (20 / 0 / 0) | - | - | - | 0.269 [0.256, 0.454] | - | - |
| 8 | 20 | -0.008 [-0.119, 0.091] | 0.516 [0.467, 0.633] | 0.076 [-0.009, 0.192] (20 / 0 / 0) | 0.575 [0.500, 0.654] | 0.062 [-0.091, 0.111] | -0.623 [-0.691, -0.410] | 0.188 [0.164, 0.231] | 0.186 [0.164, 0.202] | yes |
| 16 | 10 | 0.019 [-0.144, 0.123] | 0.652 [0.538, 0.734] | 0.065 [-0.008, 0.171] (10 / 0 / 0) | 0.732 [0.608, 0.776] | 0.052 [-0.054, 0.170] | -0.675 [-0.763, -0.570] | 0.138 [0.127, 0.156] | 0.147 [0.140, 0.158] | yes |
| 32 | 10 | 0.045 [-0.004, 0.104] | 0.853 [0.783, 0.915] | 0.198 [0.129, 0.302] (10 / 0 / 0) | - | - | - | 0.138 [0.120, 0.156] | - | - |

Growth of the integrated headline's median bias from `M = 2` to `M = 32`: -0.066 nats (0.111 to 0.045), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 2 | 3.855 [3.803, 3.970] | 3.861 [3.806, 3.963] | 3.904 [3.814, 3.966] | 3.987 [3.858, 4.004] | 2.241 [2.126, 2.448] | 0.014 | 2.241 [2.132, 2.438] | 0.015 |
| 3 | 4.162 [4.044, 4.302] | 4.167 [4.053, 4.299] | - | - | 2.536 [2.499, 2.582] | 0.014 | - | - |
| 4 | 4.292 [4.163, 4.375] | 4.284 [4.160, 4.366] | 4.285 [4.130, 4.359] | 4.334 [4.186, 4.418] | 2.546 [2.531, 2.744] | 0.015 | 2.548 [2.526, 2.740] | 0.015 |
| 5 | 4.368 [4.155, 4.393] | 4.377 [4.167, 4.394] | - | - | 2.727 [2.541, 2.740] | 0.014 | - | - |
| 8 | 4.449 [4.404, 4.485] | 4.445 [4.398, 4.487] | 4.426 [4.384, 4.488] | 4.481 [4.430, 4.554] | 2.807 [2.765, 2.831] | 0.014 | 2.809 [2.794, 2.832] | 0.013 |
| 16 | 4.482 [4.416, 4.519] | 4.475 [4.429, 4.512] | 4.485 [4.412, 4.520] | 4.516 [4.447, 4.565] | 2.858 [2.839, 2.869] | 0.013 | 2.849 [2.838, 2.855] | 0.013 |
| 32 | 4.531 [4.499, 4.554] | 4.537 [4.503, 4.548] | - | - | 2.858 [2.840, 2.874] | 0.012 | - | - |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.329 [0.282, 0.392] | 0.334 [0.288, 0.390] | 0.0004 [-0.0038, 0.0055] | 2.12 [0.70, 12.38] | 1.91 [0.74, 12.20] | -0.004 [-0.044, 0.077] | 0.73 [0.48, 0.97] | 0.40 [0.30, 0.53] | 0.189 [0.123, 0.322] | 0.0215 [0.0166, 0.0272] | 0.3 [0.2, 0.3] | 0.8 [0.7, 1.2] | 0.290 [0.249, 0.397] |
| 3 | 20 | 0.215 [0.193, 0.236] | - | - | 0.29 [0.18, 0.62] | - | - | 0.41 [0.31, 0.60] | - | - | - | 0.7 [0.6, 0.8] | - | - |
| 4 | 20 | 0.193 [0.165, 0.234] | 0.191 [0.163, 0.242] | -0.0020 [-0.0073, 0.0018] | 0.18 [0.09, 0.33] | 0.18 [0.11, 0.34] | -0.002 [-0.020, 0.006] | 0.20 [0.16, 0.40] | 0.16 [0.12, 0.26] | 0.090 [-0.003, 0.189] | 0.0172 [0.0138, 0.0237] | 1.3 [1.2, 1.4] | 2.5 [2.2, 3.6] | 0.462 [0.386, 0.578] |
| 5 | 20 | 0.164 [0.143, 0.212] | - | - | 0.11 [0.05, 0.18] | - | - | 0.33 [0.27, 0.45] | - | - | - | 2.3 [2.0, 2.8] | - | - |
| 8 | 20 | 0.113 [0.094, 0.134] | 0.111 [0.100, 0.138] | -0.0000 [-0.0032, 0.0043] | 0.03 [0.01, 0.04] | 0.02 [0.01, 0.05] | -0.001 [-0.004, 0.003] | 0.24 [0.14, 0.31] | 0.39 [0.30, 0.45] | -0.149 [-0.238, -0.027] | 0.0177 [0.0141, 0.0240] | 7.6 [6.9, 8.2] | 16.3 [13.1, 18.4] | 0.462 [0.395, 0.650] |
| 16 | 10 | 0.092 [0.086, 0.117] | 0.100 [0.092, 0.111] | -0.0009 [-0.0115, 0.0040] | 0.01 [0.01, 0.03] | 0.01 [0.01, 0.03] | -0.001 [-0.007, 0.003] | 0.14 [0.06, 0.28] | 0.59 [0.46, 0.63] | -0.482 [-0.544, -0.209] | 0.0177 [0.0138, 0.0234] | 38.4 [35.0, 45.0] | 81.2 [71.8, 94.5] | 0.490 [0.393, 0.626] |
| 32 | 10 | 0.092 [0.078, 0.101] | - | - | 0.01 [0.01, 0.01] | - | - | 0.08 [0.04, 0.16] | - | - | - | 273.9 [244.0, 304.9] | - | - |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.0004 | 0.6742 | 1 | -0.004 | 0.8124 | 1 | no |
| 3 | 0 | - | - | - | - | - | - | - |
| 4 | 20 | -0.0020 | 0.2611 | 1 | -0.002 | 0.3488 | 1 | no |
| 5 | 0 | - | - | - | - | - | - | - |
| 8 | 20 | -0.0000 | 0.9854 | 1 | -0.001 | 0.5459 | 1 | no |
| 16 | 10 | -0.0009 | 0.375 | 1 | -0.001 | 0.4316 | 1 | no |
| 32 | 0 | - | - | - | - | - | - | - |

## ring_D2_noise3_svbmc

Single runs (`M = 1`, n = 320): MMTV 0.598 [0.574, 0.627], gsKL 39.65 [25.03, 64.79], evidence error 1.30 [1.19, 1.39].

All 120 cells of this condition: runtime ratio 0.486 [0.393, 0.518], max\|dw\| 0.0076 [0.0067, 0.0089].

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. The shrinkage column shows its contributing and numerically unavailable cell counts; older cells that did not record shrinkage are counted separately. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias shrink int (contributing / unavailable / not recorded) | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.214 [0.152, 0.309] | 0.345 [0.218, 0.485] | 0.211 [0.005, 0.301] (20 / 0 / 0) | 0.375 [0.261, 0.515] | 0.269 [0.190, 0.329] | -0.148 [-0.215, -0.081] | 1.070 [0.899, 1.199] | 1.098 [0.903, 1.178] | yes |
| 3 | 20 | 0.237 [0.144, 0.305] | 0.453 [0.356, 0.542] | 0.187 [0.095, 0.276] (20 / 0 / 0) | - | - | - | 0.766 [0.614, 0.902] | - | - |
| 4 | 20 | 0.197 [0.143, 0.292] | 0.389 [0.311, 0.497] | 0.165 [0.093, 0.219] (20 / 0 / 0) | 0.446 [0.334, 0.497] | 0.233 [0.155, 0.324] | -0.157 [-0.292, -0.129] | 0.537 [0.432, 0.586] | 0.532 [0.415, 0.592] | yes |
| 5 | 20 | 0.201 [0.135, 0.293] | 0.506 [0.314, 0.575] | 0.187 [0.082, 0.244] (20 / 0 / 0) | - | - | - | 0.427 [0.352, 0.552] | - | - |
| 8 | 20 | 0.219 [0.151, 0.242] | 0.590 [0.485, 0.621] | 0.167 [0.127, 0.232] (20 / 0 / 0) | 0.606 [0.553, 0.661] | 0.240 [0.183, 0.270] | -0.396 [-0.441, -0.361] | 0.213 [0.190, 0.343] | 0.226 [0.195, 0.348] | yes |
| 16 | 10 | 0.162 [0.117, 0.273] | 0.723 [0.656, 0.798] | 0.123 [0.103, 0.217] (10 / 0 / 0) | 0.734 [0.684, 0.839] | 0.196 [0.143, 0.289] | -0.575 [-0.687, -0.448] | 0.151 [0.127, 0.171] | 0.146 [0.127, 0.169] | yes |
| 32 | 10 | 0.177 [0.157, 0.224] | 0.893 [0.843, 0.980] | 0.155 [0.108, 0.220] (10 / 0 / 0) | - | - | - | 0.097 [0.074, 0.124] | - | - |

Growth of the integrated headline's median bias from `M = 2` to `M = 32`: -0.037 nats (0.214 to 0.177), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 2 | 2.079 [1.926, 2.258] | 2.076 [1.947, 2.256] | 2.089 [1.938, 2.256] | 2.113 [1.971, 2.284] | 1.464 [1.335, 1.635] | 0.011 | 1.436 [1.356, 1.631] | 0.012 |
| 3 | 2.337 [2.209, 2.539] | 2.345 [2.214, 2.554] | - | - | 1.768 [1.632, 1.920] | 0.010 | - | - |
| 4 | 2.599 [2.547, 2.727] | 2.603 [2.550, 2.724] | 2.586 [2.550, 2.736] | 2.616 [2.577, 2.767] | 1.997 [1.948, 2.102] | 0.011 | 2.001 [1.942, 2.119] | 0.010 |
| 5 | 2.739 [2.569, 2.799] | 2.735 [2.575, 2.785] | - | - | 2.107 [1.982, 2.182] | 0.011 | - | - |
| 8 | 2.882 [2.779, 2.942] | 2.876 [2.777, 2.945] | 2.885 [2.772, 2.937] | 2.908 [2.801, 2.958] | 2.321 [2.190, 2.344] | 0.009 | 2.307 [2.186, 2.339] | 0.010 |
| 16 | 2.940 [2.917, 2.969] | 2.942 [2.913, 2.966] | 2.943 [2.921, 2.972] | 2.961 [2.935, 2.985] | 2.383 [2.363, 2.407] | 0.009 | 2.387 [2.365, 2.407] | 0.009 |
| 32 | 3.001 [2.958, 3.014] | 3.001 [2.957, 3.016] | - | - | 2.437 [2.411, 2.460] | 0.009 | - | - |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.413 [0.353, 0.467] | 0.406 [0.349, 0.470] | -0.0002 [-0.0042, 0.0016] | 1.34 [0.63, 5.30] | 1.30 [0.63, 4.92] | -0.002 [-0.020, 0.053] | 0.85 [0.73, 1.07] | 0.76 [0.56, 0.91] | 0.122 [0.045, 0.190] | 0.0112 [0.0095, 0.0138] | 0.3 [0.3, 0.4] | 0.9 [0.8, 1.0] | 0.354 [0.311, 0.424] |
| 3 | 20 | 0.329 [0.292, 0.357] | - | - | 0.49 [0.19, 1.68] | - | - | 0.59 [0.44, 0.64] | - | - | - | 0.8 [0.8, 0.9] | - | - |
| 4 | 20 | 0.247 [0.225, 0.269] | 0.243 [0.221, 0.271] | 0.0002 [-0.0029, 0.0026] | 0.13 [0.05, 0.26] | 0.13 [0.04, 0.25] | 0.001 [-0.004, 0.006] | 0.30 [0.22, 0.42] | 0.19 [0.10, 0.26] | 0.133 [0.035, 0.177] | 0.0088 [0.0073, 0.0101] | 1.5 [1.4, 1.7] | 3.7 [2.7, 4.1] | 0.487 [0.352, 0.539] |
| 5 | 20 | 0.227 [0.198, 0.255] | - | - | 0.09 [0.05, 0.25] | - | - | 0.22 [0.13, 0.38] | - | - | - | 2.9 [2.2, 3.4] | - | - |
| 8 | 20 | 0.169 [0.133, 0.207] | 0.167 [0.131, 0.206] | -0.0007 [-0.0028, 0.0007] | 0.03 [0.02, 0.06] | 0.03 [0.02, 0.06] | 0.001 [-0.001, 0.004] | 0.12 [0.09, 0.16] | 0.30 [0.25, 0.40] | -0.194 [-0.318, -0.080] | 0.0063 [0.0054, 0.0069] | 9.5 [7.5, 10.6] | 18.1 [15.7, 19.9] | 0.546 [0.441, 0.592] |
| 16 | 10 | 0.134 [0.123, 0.147] | 0.130 [0.120, 0.154] | 0.0037 [-0.0006, 0.0075] | 0.02 [0.01, 0.03] | 0.02 [0.01, 0.03] | -0.000 [-0.004, 0.001] | 0.04 [0.02, 0.11] | 0.59 [0.54, 0.67] | -0.526 [-0.592, -0.430] | 0.0045 [0.0038, 0.0055] | 50.8 [46.8, 55.3] | 98.8 [90.9, 110.7] | 0.514 [0.438, 0.578] |
| 32 | 10 | 0.095 [0.084, 0.122] | - | - | 0.01 [0.00, 0.03] | - | - | 0.09 [0.04, 0.14] | - | - | - | 259.8 [220.0, 279.5] | - | - |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 2 | 20 | -0.0002 | 0.8124 | 1 | -0.002 | 0.8408 | 1 | no |
| 3 | 0 | - | - | - | - | - | - | - |
| 4 | 20 | 0.0002 | 0.9854 | 1 | 0.001 | 0.7562 | 1 | no |
| 5 | 0 | - | - | - | - | - | - | - |
| 8 | 20 | -0.0007 | 0.5459 | 1 | 0.001 | 0.3488 | 1 | no |
| 16 | 10 | 0.0037 | 0.1934 | 1 | -0.000 | 0.4316 | 1 | no |
| 32 | 0 | - | - | - | - | - | - | - |

## student_D8_noise3_svbmc

Single runs (`M = 1`, n = 320): MMTV 0.137 [0.133, 0.140], gsKL 0.58 [0.53, 0.60], evidence error 0.86 [0.81, 0.93].

All 120 cells of this condition: runtime ratio 0.414 [0.339, 0.464], max\|dw\| 0.0298 [0.0279, 0.0326].

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. The shrinkage column shows its contributing and numerically unavailable cell counts; older cells that did not record shrinkage are counted separately. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias shrink int (contributing / unavailable / not recorded) | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | -0.887 [-1.235, -0.710] | -0.135 [-0.298, 0.070] | -0.416 [-0.613, -0.234] (20 / 0 / 0) | -0.039 [-0.142, 0.189] | -0.789 [-1.047, -0.566] | -0.858 [-1.204, -0.768] | 0.546 [0.514, 0.638] | 0.586 [0.542, 0.659] | **no** |
| 3 | 20 | -0.983 [-1.340, -0.817] | -0.017 [-0.134, 0.132] | -0.353 [-0.415, -0.161] (20 / 0 / 0) | - | - | - | 0.410 [0.376, 0.440] | - | - |
| 4 | 20 | -1.148 [-1.285, -0.836] | 0.102 [0.036, 0.208] | -0.224 [-0.313, -0.162] (20 / 0 / 0) | 0.164 [0.075, 0.308] | -1.053 [-1.235, -0.763] | -1.163 [-1.585, -0.949] | 0.411 [0.346, 0.446] | 0.389 [0.347, 0.451] | **no** |
| 5 | 20 | -1.310 [-1.531, -1.182] | 0.033 [-0.038, 0.179] | -0.348 [-0.419, -0.192] (20 / 0 / 0) | - | - | - | 0.362 [0.314, 0.410] | - | - |
| 8 | 20 | -1.340 [-1.452, -1.210] | 0.129 [-0.023, 0.252] | -0.240 [-0.347, -0.140] (20 / 0 / 0) | 0.290 [0.142, 0.357] | -1.192 [-1.337, -1.031] | -1.545 [-1.838, -1.386] | 0.308 [0.257, 0.368] | 0.319 [0.284, 0.339] | **no** |
| 16 | 10 | -1.537 [-1.656, -1.341] | 0.247 [0.116, 0.379] | -0.167 [-0.238, -0.065] (10 / 0 / 0) | 0.298 [0.165, 0.381] | -1.362 [-1.528, -1.329] | -1.805 [-1.941, -1.658] | 0.217 [0.193, 0.279] | 0.222 [0.178, 0.269] | **no** |
| 32 | 10 | -1.685 [-1.890, -1.468] | 0.314 [0.256, 0.480] | -0.102 [-0.169, 0.037] (10 / 0 / 0) | - | - | - | 0.182 [0.129, 0.228] | - | - |

Growth of the integrated headline's median bias from `M = 2` to `M = 32`: -0.798 nats (-0.887 to -1.685), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 2 | 13.784 [13.486, 13.988] | 13.789 [13.467, 13.979] | 13.780 [13.637, 13.989] | 13.986 [13.721, 14.059] | -20.150 [-20.242, -20.118] | 0.037 | -20.190 [-20.263, -20.145] | 0.038 |
| 3 | 13.457 [13.258, 13.806] | 13.498 [13.284, 13.791] | - | - | -20.014 [-20.044, -19.977] | 0.036 | - | - |
| 4 | 13.554 [13.190, 13.693] | 13.571 [13.214, 13.706] | 13.549 [13.173, 13.742] | 13.659 [13.318, 13.843] | -20.015 [-20.049, -19.950] | 0.035 | -19.992 [-20.054, -19.950] | 0.034 |
| 5 | 13.455 [13.290, 13.611] | 13.462 [13.255, 13.592] | - | - | -19.965 [-20.014, -19.918] | 0.033 | - | - |
| 8 | 13.412 [13.319, 13.542] | 13.438 [13.321, 13.541] | 13.399 [13.296, 13.571] | 13.506 [13.427, 13.691] | -19.912 [-19.972, -19.861] | 0.033 | -19.922 [-19.943, -19.888] | 0.032 |
| 16 | 13.171 [13.105, 13.494] | 13.173 [13.131, 13.525] | 13.245 [13.214, 13.385] | 13.378 [13.312, 13.480] | -19.821 [-19.883, -19.796] | 0.032 | -19.825 [-19.872, -19.781] | 0.031 |
| 32 | 13.241 [13.073, 13.343] | 13.242 [13.068, 13.345] | - | - | -19.786 [-19.831, -19.733] | 0.030 | - | - |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.124 [0.114, 0.130] | 0.124 [0.116, 0.132] | 0.0001 [-0.0011, 0.0008] | 0.40 [0.35, 0.47] | 0.41 [0.35, 0.46] | -0.003 [-0.010, 0.001] | 1.42 [1.24, 1.77] | 0.56 [0.44, 0.78] | 0.850 [0.736, 1.172] | 0.0286 [0.0174, 0.0449] | 0.4 [0.3, 0.4] | 1.1 [0.8, 1.4] | 0.379 [0.296, 0.414] |
| 3 | 20 | 0.103 [0.093, 0.111] | - | - | 0.23 [0.20, 0.30] | - | - | 1.41 [1.18, 1.77] | - | - | - | 0.8 [0.8, 1.0] | - | - |
| 4 | 20 | 0.097 [0.091, 0.107] | 0.097 [0.091, 0.106] | 0.0003 [-0.0018, 0.0020] | 0.24 [0.17, 0.26] | 0.23 [0.17, 0.28] | 0.001 [-0.005, 0.012] | 1.50 [1.27, 1.75] | 0.22 [0.14, 0.30] | 1.134 [0.994, 1.518] | 0.0320 [0.0295, 0.0379] | 1.9 [1.7, 2.1] | 4.4 [3.7, 5.9] | 0.442 [0.339, 0.493] |
| 5 | 20 | 0.098 [0.088, 0.105] | - | - | 0.18 [0.14, 0.20] | - | - | 1.68 [1.55, 1.88] | - | - | - | 3.4 [3.2, 3.8] | - | - |
| 8 | 20 | 0.091 [0.084, 0.094] | 0.087 [0.083, 0.092] | 0.0005 [-0.0007, 0.0017] | 0.15 [0.13, 0.17] | 0.14 [0.12, 0.17] | 0.003 [-0.009, 0.019] | 1.69 [1.58, 1.82] | 0.09 [0.06, 0.19] | 1.468 [1.295, 1.655] | 0.0285 [0.0236, 0.0355] | 9.1 [7.7, 10.0] | 20.1 [17.8, 22.2] | 0.441 [0.332, 0.537] |
| 16 | 10 | 0.074 [0.069, 0.082] | 0.075 [0.067, 0.082] | -0.0005 [-0.0053, 0.0030] | 0.09 [0.06, 0.10] | 0.07 [0.05, 0.10] | 0.008 [0.001, 0.023] | 1.74 [1.59, 1.92] | 0.17 [0.09, 0.21] | 1.614 [1.404, 1.787] | 0.0283 [0.0231, 0.0505] | 54.2 [43.4, 65.6] | 134.5 [101.7, 161.1] | 0.493 [0.263, 0.582] |
| 32 | 10 | 0.064 [0.058, 0.072] | - | - | 0.06 [0.04, 0.08] | - | - | 1.85 [1.74, 2.05] | - | - | - | 351.8 [318.2, 403.1] | - | - |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.0001 | 0.9563 | 1 | -0.003 | 0.3118 | 1 | no |
| 3 | 0 | - | - | - | - | - | - | - |
| 4 | 20 | 0.0003 | 0.9273 | 1 | 0.001 | 0.5706 | 1 | no |
| 5 | 0 | - | - | - | - | - | - | - |
| 8 | 20 | 0.0005 | 0.3884 | 1 | 0.003 | 0.2455 | 1 | no |
| 16 | 10 | -0.0005 | 0.8457 | 1 | 0.008 | 0.1309 | 1 | no |
| 32 | 0 | - | - | - | - | - | - | - |

## gmm_D2_svbmc

Single runs (`M = 1`, n = 320): MMTV 0.389 [0.387, 0.391], gsKL 15.06 [14.26, 15.33], evidence error 0.70 [0.70, 0.71].

All 120 cells of this condition: runtime ratio 0.415 [0.389, 0.472], max\|dw\| 0.0150 [0.0116, 0.0192].

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. The shrinkage column shows its contributing and numerically unavailable cell counts; older cells that did not record shrinkage are counted separately. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias shrink int (contributing / unavailable / not recorded) | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.004 [-0.010, 0.008] | 0.004 [-0.010, 0.008] | 0.005 [-0.008, 0.013] (20 / 0 / 0) | 0.054 [0.033, 0.079] | 0.032 [0.019, 0.077] | -0.058 [-0.081, -0.034] | 0.521 [0.305, 0.710] | 0.521 [0.307, 0.695] | yes |
| 3 | 20 | 0.004 [-0.007, 0.018] | 0.004 [-0.007, 0.018] | 0.009 [-0.004, 0.024] (20 / 0 / 0) | - | - | - | 0.166 [0.018, 0.307] | - | - |
| 4 | 20 | 0.013 [0.001, 0.032] | 0.013 [0.001, 0.032] | 0.016 [0.004, 0.039] (20 / 0 / 0) | 0.057 [0.040, 0.074] | 0.037 [0.008, 0.053] | -0.036 [-0.049, -0.020] | 0.051 [0.027, 0.301] | 0.037 [0.021, 0.288] | yes |
| 5 | 20 | 0.007 [-0.000, 0.027] | 0.007 [-0.000, 0.027] | 0.014 [0.004, 0.031] (20 / 0 / 0) | - | - | - | 0.026 [0.018, 0.219] | - | - |
| 8 | 20 | 0.027 [0.013, 0.051] | 0.027 [0.013, 0.051] | 0.029 [0.020, 0.066] (20 / 0 / 0) | 0.068 [0.042, 0.087] | 0.055 [0.035, 0.085] | -0.035 [-0.055, -0.020] | 0.050 [0.021, 0.129] | 0.048 [0.019, 0.148] | yes |
| 16 | 10 | 0.010 [-0.004, 0.094] | 0.010 [-0.004, 0.094] | 0.017 [0.002, 0.111] (10 / 0 / 0) | 0.051 [0.023, 0.160] | 0.045 [0.022, 0.160] | -0.032 [-0.067, -0.022] | 0.019 [-0.002, 0.109] | 0.026 [0.000, 0.138] | yes |
| 32 | 10 | 0.030 [0.019, 0.186] | 0.030 [0.019, 0.186] | 0.038 [0.023, 0.203] (10 / 0 / 0) | - | - | - | 0.029 [0.018, 0.191] | - | - |

Growth of the integrated headline's median bias from `M = 2` to `M = 32`: 0.026 nats (0.004 to 0.030), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 2 | 4.138 [3.905, 4.349] | 4.125 [3.897, 4.333] | 4.132 [3.918, 4.345] | 4.179 [3.957, 4.386] | 2.475 [2.286, 2.691] | 0.017 | 2.475 [2.301, 2.688] | 0.017 |
| 3 | 4.590 [4.336, 4.628] | 4.590 [4.330, 4.617] | - | - | 2.830 [2.689, 2.977] | 0.014 | - | - |
| 4 | 4.600 [4.376, 4.628] | 4.604 [4.379, 4.631] | 4.610 [4.367, 4.628] | 4.658 [4.406, 4.683] | 2.945 [2.695, 2.969] | 0.014 | 2.958 [2.708, 2.975] | 0.014 |
| 5 | 4.621 [4.590, 4.648] | 4.620 [4.594, 4.632] | - | - | 2.970 [2.777, 2.978] | 0.014 | - | - |
| 8 | 4.638 [4.615, 4.652] | 4.634 [4.610, 4.645] | 4.630 [4.625, 4.648] | 4.663 [4.645, 4.681] | 2.946 [2.866, 2.974] | 0.014 | 2.948 [2.848, 2.977] | 0.014 |
| 16 | 4.642 [4.630, 4.653] | 4.635 [4.629, 4.648] | 4.643 [4.631, 4.663] | 4.665 [4.657, 4.681] | 2.976 [2.887, 2.998] | 0.012 | 2.970 [2.858, 2.995] | 0.012 |
| 32 | 4.643 [4.636, 4.663] | 4.643 [4.635, 4.662] | - | - | 2.966 [2.805, 2.978] | 0.014 | - | - |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.236 [0.191, 0.367] | 0.238 [0.192, 0.366] | -0.0010 [-0.0022, 0.0024] | 2.79 [0.45, 11.93] | 3.00 [0.43, 11.62] | 0.006 [-0.012, 0.032] | 0.51 [0.30, 0.70] | 0.43 [0.25, 0.65] | 0.043 [0.022, 0.068] | 0.0251 [0.0178, 0.0356] | 0.3 [0.2, 0.3] | 0.7 [0.7, 0.9] | 0.349 [0.309, 0.431] |
| 3 | 20 | 0.043 [0.036, 0.200] | - | - | 0.01 [0.00, 0.44] | - | - | 0.03 [0.01, 0.31] | - | - | - | 0.6 [0.6, 0.7] | - | - |
| 4 | 20 | 0.047 [0.039, 0.174] | 0.039 [0.033, 0.171] | 0.0020 [-0.0071, 0.0129] | 0.01 [0.01, 0.27] | 0.01 [0.01, 0.30] | -0.002 [-0.006, 0.003] | 0.02 [0.01, 0.29] | 0.04 [0.04, 0.25] | -0.006 [-0.025, 0.031] | 0.0199 [0.0139, 0.0259] | 1.1 [1.0, 1.1] | 2.9 [2.3, 3.3] | 0.386 [0.317, 0.434] |
| 5 | 20 | 0.042 [0.030, 0.055] | - | - | 0.01 [0.00, 0.01] | - | - | 0.02 [0.01, 0.03] | - | - | - | 1.8 [1.6, 2.1] | - | - |
| 8 | 20 | 0.042 [0.025, 0.059] | 0.040 [0.024, 0.044] | -0.0008 [-0.0035, 0.0186] | 0.01 [0.00, 0.02] | 0.00 [0.00, 0.01] | -0.000 [-0.003, 0.004] | 0.01 [0.00, 0.02] | 0.03 [0.02, 0.04] | -0.011 [-0.021, 0.006] | 0.0128 [0.0095, 0.0167] | 4.8 [4.6, 5.3] | 10.3 [7.6, 12.1] | 0.476 [0.419, 0.607] |
| 16 | 10 | 0.028 [0.020, 0.042] | 0.020 [0.014, 0.026] | 0.0042 [-0.0009, 0.0244] | 0.00 [0.00, 0.01] | 0.00 [0.00, 0.00] | 0.002 [-0.000, 0.006] | 0.01 [0.00, 0.01] | 0.02 [0.02, 0.03] | -0.017 [-0.026, -0.010] | 0.0062 [0.0049, 0.0095] | 20.6 [16.5, 22.1] | 30.5 [24.5, 39.9] | 0.688 [0.530, 0.749] |
| 32 | 10 | 0.017 [0.015, 0.034] | - | - | 0.00 [0.00, 0.00] | - | - | 0.00 [0.00, 0.01] | - | - | - | 91.4 [75.3, 105.8] | - | - |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 2 | 20 | -0.0010 | 0.8408 | 1 | 0.006 | 0.4304 | 1 | no |
| 3 | 0 | - | - | - | - | - | - | - |
| 4 | 20 | 0.0020 | 0.498 | 1 | -0.002 | 0.3884 | 1 | no |
| 5 | 0 | - | - | - | - | - | - | - |
| 8 | 20 | -0.0008 | 0.3118 | 1 | -0.000 | 0.5706 | 1 | no |
| 16 | 10 | 0.0042 | 0.06445 | 1 | 0.002 | 0.04883 | 1 | no |
| 32 | 0 | - | - | - | - | - | - | - |

## multisensory_s1_D6_svbmc

Single runs (`M = 1`, n = 320): MMTV 0.114 [0.112, 0.118], gsKL 0.52 [0.48, 0.56], evidence error 0.38 [0.37, 0.39].

All 120 cells of this condition: runtime ratio 0.259 [0.225, 0.294], max\|dw\| 0.0117 [0.0101, 0.0141].

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. The shrinkage column shows its contributing and numerically unavailable cell counts; older cells that did not record shrinkage are counted separately. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias shrink int (contributing / unavailable / not recorded) | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.037 [0.027, 0.056] | 0.037 [0.027, 0.056] | 0.048 [0.029, 0.075] (20 / 0 / 0) | 0.105 [0.076, 0.141] | 0.041 [-0.007, 0.069] | -0.065 [-0.099, -0.036] | 0.330 [0.310, 0.345] | 0.325 [0.300, 0.354] | yes |
| 3 | 20 | 0.033 [0.023, 0.049] | 0.033 [0.023, 0.049] | 0.045 [0.035, 0.062] (20 / 0 / 0) | - | - | - | 0.218 [0.192, 0.285] | - | - |
| 4 | 20 | 0.036 [0.006, 0.053] | 0.036 [0.006, 0.053] | 0.046 [0.021, 0.062] (20 / 0 / 0) | 0.087 [0.072, 0.117] | 0.046 [0.012, 0.057] | -0.070 [-0.078, -0.032] | 0.239 [0.206, 0.280] | 0.240 [0.211, 0.272] | yes |
| 5 | 20 | 0.048 [0.022, 0.053] | 0.048 [0.022, 0.053] | 0.059 [0.038, 0.075] (20 / 0 / 0) | - | - | - | 0.202 [0.187, 0.214] | - | - |
| 8 | 20 | 0.040 [0.023, 0.063] | 0.040 [0.023, 0.063] | 0.055 [0.043, 0.080] (20 / 0 / 0) | 0.094 [0.076, 0.116] | 0.072 [0.027, 0.092] | -0.052 [-0.063, -0.034] | 0.166 [0.156, 0.184] | 0.182 [0.167, 0.206] | yes |
| 16 | 10 | 0.066 [0.036, 0.096] | 0.066 [0.036, 0.096] | 0.080 [0.046, 0.117] (10 / 0 / 0) | 0.109 [0.077, 0.142] | 0.068 [0.062, 0.126] | -0.043 [-0.091, 0.003] | 0.157 [0.140, 0.177] | 0.144 [0.123, 0.170] | yes |
| 32 | 10 | 0.104 [0.071, 0.141] | 0.104 [0.066, 0.141] | 0.130 [0.084, 0.162] (10 / 0 / 0) | - | - | - | 0.143 [0.132, 0.167] | - | - |

Growth of the integrated headline's median bias from `M = 2` to `M = 32`: 0.066 nats (0.037 to 0.104), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 2 | 3.636 [3.571, 3.668] | 3.629 [3.589, 3.691] | 3.620 [3.577, 3.671] | 3.709 [3.672, 3.761] | -502.516 [-502.531, -502.496] | 0.024 | -502.511 [-502.540, -502.486] | 0.025 |
| 3 | 3.736 [3.677, 3.804] | 3.739 [3.684, 3.803] | - | - | -502.404 [-502.471, -502.378] | 0.022 | - | - |
| 4 | 3.704 [3.680, 3.732] | 3.706 [3.682, 3.747] | 3.709 [3.663, 3.736] | 3.760 [3.724, 3.808] | -502.425 [-502.466, -502.392] | 0.021 | -502.426 [-502.458, -502.397] | 0.021 |
| 5 | 3.768 [3.724, 3.818] | 3.764 [3.726, 3.813] | - | - | -502.388 [-502.400, -502.373] | 0.021 | - | - |
| 8 | 3.835 [3.790, 3.859] | 3.821 [3.795, 3.861] | 3.832 [3.774, 3.851] | 3.855 [3.817, 3.886] | -502.352 [-502.370, -502.342] | 0.020 | -502.368 [-502.392, -502.352] | 0.020 |
| 16 | 3.873 [3.846, 3.907] | 3.875 [3.844, 3.891] | 3.913 [3.878, 3.934] | 3.940 [3.896, 3.959] | -502.342 [-502.364, -502.323] | 0.020 | -502.330 [-502.356, -502.309] | 0.020 |
| 32 | 3.942 [3.875, 3.979] | 3.943 [3.882, 3.974] | - | - | -502.329 [-502.353, -502.318] | 0.020 | - | - |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.096 [0.070, 0.104] | 0.096 [0.078, 0.110] | -0.0035 [-0.0122, 0.0011] | 0.23 [0.10, 0.46] | 0.24 [0.12, 0.52] | -0.018 [-0.041, 0.013] | 0.29 [0.25, 0.31] | 0.21 [0.15, 0.24] | 0.065 [0.034, 0.099] | 0.0152 [0.0101, 0.0180] | 0.3 [0.3, 0.4] | 1.3 [1.0, 1.5] | 0.260 [0.219, 0.312] |
| 3 | 20 | 0.056 [0.047, 0.082] | - | - | 0.08 [0.05, 0.28] | - | - | 0.19 [0.17, 0.22] | - | - | - | 0.9 [0.8, 1.1] | - | - |
| 4 | 20 | 0.081 [0.064, 0.085] | 0.076 [0.064, 0.087] | -0.0004 [-0.0104, 0.0059] | 0.22 [0.10, 0.29] | 0.16 [0.11, 0.30] | 0.001 [-0.041, 0.013] | 0.22 [0.18, 0.23] | 0.17 [0.14, 0.20] | 0.055 [0.044, 0.073] | 0.0138 [0.0118, 0.0166] | 1.8 [1.6, 2.0] | 6.8 [5.6, 8.0] | 0.262 [0.221, 0.346] |
| 5 | 20 | 0.063 [0.034, 0.077] | - | - | 0.10 [0.04, 0.16] | - | - | 0.16 [0.13, 0.19] | - | - | - | 2.7 [2.1, 3.1] | - | - |
| 8 | 20 | 0.042 [0.032, 0.058] | 0.052 [0.045, 0.061] | -0.0050 [-0.0098, 0.0024] | 0.04 [0.03, 0.10] | 0.07 [0.04, 0.09] | -0.012 [-0.027, 0.004] | 0.12 [0.10, 0.14] | 0.09 [0.07, 0.10] | 0.038 [0.024, 0.044] | 0.0101 [0.0092, 0.0138] | 7.4 [6.7, 8.1] | 28.5 [19.8, 41.4] | 0.275 [0.220, 0.328] |
| 16 | 10 | 0.032 [0.027, 0.044] | 0.026 [0.024, 0.030] | 0.0044 [-0.0033, 0.0210] | 0.03 [0.02, 0.05] | 0.02 [0.02, 0.02] | 0.008 [-0.002, 0.030] | 0.09 [0.07, 0.10] | 0.04 [0.02, 0.07] | 0.042 [0.017, 0.057] | 0.0075 [0.0054, 0.0097] | 38.7 [32.8, 51.0] | 231.2 [144.4, 264.6] | 0.187 [0.124, 0.349] |
| 32 | 10 | 0.025 [0.023, 0.037] | - | - | 0.02 [0.02, 0.04] | - | - | 0.04 [0.01, 0.07] | - | - | - | 206.7 [162.1, 238.3] | - | - |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 2 | 20 | -0.0035 | 0.114 | 1 | -0.018 | 0.07585 | 1 | no |
| 3 | 0 | - | - | - | - | - | - | - |
| 4 | 20 | -0.0004 | 0.7841 | 1 | 0.001 | 0.7562 | 1 | no |
| 5 | 0 | - | - | - | - | - | - | - |
| 8 | 20 | -0.0050 | 0.33 | 1 | -0.012 | 0.33 | 1 | no |
| 16 | 10 | 0.0044 | 0.4316 | 1 | 0.008 | 0.2324 | 1 | no |
| 32 | 0 | - | - | - | - | - | - | - |
