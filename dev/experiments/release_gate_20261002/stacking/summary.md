# S-VBMC stacking comparison: integrated against original 0.1.1

Generated 2026-10-02 15:46:22. Every cell stacks the same subset of filtered pool runs with both implementations, `n_samples=20`, `lr=0.1`, `max_steps=500`, `version="all-weights"`, subsets drawn with seed 0, disjoint within each `M`. Medians over the repetitions of a cell with a 10 000-resample bootstrap 95 % interval; `d` columns are the paired difference integrated minus original, so a negative value favours the integrated class. The headline ELBO is the capped value for the integrated class on noisy stacks and the raw estimate for the original, which is what each implementation reports. gsKL is the house convention (no `1/D` factor).

Every reported ELBO is scored by its bias against `elbo_mc = e_log_joint_mc + entropy_ref`, the Monte Carlo ELBO of the posterior that arm's fit produced. `e_log_joint_mc` is the mean of the target's noiseless log density over 10 000 of the stack's own draws; `entropy_ref` is the entropy of the stacked mixture at that arm's final weights, estimated here for both arms by one estimator (the integrated class's `stacked_entropy`, 200 draws per component in 8 batches, from a generator seeded by the cell, the same draws for both arms). The reference is arm-independent by construction: an arm's own reported entropy is estimated at the weights that arm selected, from its own draws (the optimizer's 20 per component for the original, a fresh 100 for the integrated class), so a reference built on it would move with the arm and credit whichever entropy estimate came out high. `kl_gap = ln Z - elbo_mc` is how far the stacked posterior itself is from the target, in evidence units; the `err` columns further below are the error against `ln Z`, descriptive only.

## multisensory_s1_D6_noise3_svbmc

Single runs (`M = 1`, n = 320): MMTV 0.263 [0.257, 0.269], gsKL 3.65 [3.35, 3.98], evidence error 0.41 [0.37, 0.46].

All 120 cells of this condition: runtime ratio 0.193 [0.178, 0.210], max\|dw\| 0.0204 [0.0185, 0.0231].

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. The shrinkage column shows its contributing and numerically unavailable cell counts; older cells that did not record shrinkage are counted separately. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias shrink int (contributing / unavailable / not recorded) | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.382 [0.268, 0.479] | 0.654 [0.482, 0.859] | 0.386 [0.320, 0.611] (20 / 0 / 0) | 0.742 [0.600, 0.959] | 0.476 [0.275, 0.564] | -0.408 [-0.478, -0.335] | 0.777 [0.642, 0.892] | 0.769 [0.655, 0.886] | yes |
| 3 | 20 | 0.311 [0.174, 0.479] | 0.793 [0.633, 0.908] | 0.462 [0.252, 0.573] (20 / 0 / 0) | - | - | - | 0.669 [0.621, 0.748] | - | - |
| 4 | 20 | 0.202 [0.099, 0.394] | 0.726 [0.682, 0.765] | 0.373 [0.342, 0.489] (20 / 0 / 0) | 0.847 [0.757, 0.902] | 0.321 [0.195, 0.458] | -0.622 [-0.736, -0.501] | 0.559 [0.457, 0.641] | 0.565 [0.462, 0.639] | yes |
| 5 | 20 | 0.280 [0.176, 0.327] | 0.903 [0.807, 1.013] | 0.519 [0.465, 0.596] (20 / 0 / 0) | - | - | - | 0.576 [0.426, 0.655] | - | - |
| 8 | 20 | 0.377 [0.293, 0.462] | 0.975 [0.927, 1.091] | 0.594 [0.532, 0.673] (20 / 0 / 0) | 1.090 [0.968, 1.209] | 0.410 [0.350, 0.523] | -0.646 [-0.893, -0.574] | 0.513 [0.404, 0.577] | 0.485 [0.384, 0.596] | yes |
| 16 | 10 | 0.347 [0.239, 0.460] | 1.174 [1.052, 1.293] | 0.664 [0.560, 0.763] (10 / 0 / 0) | 1.248 [1.059, 1.329] | 0.406 [0.270, 0.569] | -0.864 [-1.108, -0.610] | 0.394 [0.286, 0.538] | 0.398 [0.293, 0.510] | yes |
| 32 | 10 | 0.343 [0.250, 0.452] | 1.359 [1.303, 1.399] | 0.740 [0.680, 0.798] (10 / 0 / 0) | - | - | - | 0.333 [0.259, 0.380] | - | - |

Growth of the integrated headline's median bias from `M = 2` to `M = 32`: -0.039 nats (0.382 to 0.343), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 2 | 2.972 [2.836, 3.093] | 2.959 [2.817, 3.112] | 2.955 [2.810, 3.076] | 3.054 [2.895, 3.165] | -502.963 [-503.078, -502.828] | 0.024 | -502.955 [-503.072, -502.841] | 0.023 |
| 3 | 3.114 [2.953, 3.284] | 3.147 [2.942, 3.304] | - | - | -502.855 [-502.934, -502.807] | 0.022 | - | - |
| 4 | 3.177 [3.024, 3.323] | 3.181 [3.015, 3.338] | 3.181 [3.025, 3.353] | 3.248 [3.109, 3.444] | -502.745 [-502.826, -502.643] | 0.021 | -502.751 [-502.825, -502.648] | 0.021 |
| 5 | 3.206 [3.104, 3.289] | 3.225 [3.063, 3.301] | - | - | -502.762 [-502.841, -502.612] | 0.020 | - | - |
| 8 | 3.285 [3.117, 3.498] | 3.266 [3.099, 3.509] | 3.295 [3.074, 3.491] | 3.323 [3.161, 3.543] | -502.699 [-502.762, -502.590] | 0.020 | -502.671 [-502.782, -502.570] | 0.020 |
| 16 | 3.425 [3.171, 3.554] | 3.411 [3.172, 3.539] | 3.444 [3.196, 3.563] | 3.504 [3.285, 3.619] | -502.580 [-502.724, -502.472] | 0.020 | -502.584 [-502.673, -502.486] | 0.021 |
| 32 | 3.410 [3.308, 3.593] | 3.414 [3.308, 3.588] | - | - | -502.519 [-502.566, -502.445] | 0.018 | - | - |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.223 [0.203, 0.256] | 0.223 [0.190, 0.257] | 0.0003 [-0.0021, 0.0031] | 2.72 [2.10, 3.96] | 2.74 [1.85, 4.09] | 0.018 [-0.008, 0.052] | 0.42 [0.24, 0.58] | 0.17 [0.07, 0.26] | 0.343 [0.159, 0.388] | 0.0280 [0.0220, 0.0327] | 0.4 [0.4, 0.5] | 2.3 [1.8, 2.8] | 0.168 [0.156, 0.195] |
| 3 | 20 | 0.207 [0.170, 0.222] | - | - | 2.13 [1.09, 2.68] | - | - | 0.26 [0.19, 0.54] | - | - | - | 1.0 [0.9, 1.1] | - | - |
| 4 | 20 | 0.176 [0.145, 0.221] | 0.178 [0.143, 0.227] | 0.0018 [-0.0007, 0.0045] | 1.67 [0.54, 2.35] | 1.70 [0.53, 2.36] | 0.021 [0.008, 0.035] | 0.27 [0.17, 0.50] | 0.32 [0.19, 0.42] | 0.005 [-0.307, 0.181] | 0.0202 [0.0154, 0.0287] | 2.0 [1.9, 2.3] | 10.3 [9.5, 12.7] | 0.183 [0.173, 0.217] |
| 5 | 20 | 0.181 [0.144, 0.218] | - | - | 1.29 [0.49, 2.30] | - | - | 0.32 [0.20, 0.42] | - | - | - | 3.4 [3.1, 3.7] | - | - |
| 8 | 20 | 0.168 [0.127, 0.205] | 0.167 [0.132, 0.200] | 0.0028 [-0.0002, 0.0059] | 0.89 [0.44, 1.93] | 0.80 [0.43, 1.88] | 0.031 [0.005, 0.072] | 0.15 [0.09, 0.22] | 0.65 [0.59, 0.67] | -0.476 [-0.550, -0.350] | 0.0199 [0.0155, 0.0220] | 10.8 [9.1, 12.6] | 49.6 [47.4, 52.3] | 0.237 [0.182, 0.273] |
| 16 | 10 | 0.125 [0.099, 0.173] | 0.123 [0.102, 0.180] | 0.0009 [-0.0041, 0.0034] | 0.44 [0.14, 1.47] | 0.41 [0.15, 1.55] | -0.008 [-0.044, 0.046] | 0.13 [0.04, 0.24] | 0.83 [0.73, 0.95] | -0.668 [-0.838, -0.550] | 0.0129 [0.0108, 0.0167] | 61.3 [54.1, 67.0] | 299.7 [227.3, 332.0] | 0.215 [0.182, 0.243] |
| 32 | 10 | 0.117 [0.089, 0.149] | - | - | 0.26 [0.17, 0.33] | - | - | 0.12 [0.06, 0.25] | - | - | - | 501.7 [367.1, 537.6] | - | - |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.0003 | 0.5958 | 1 | 0.018 | 0.3488 | 1 | no |
| 3 | 0 | - | - | - | - | - | - | - |
| 4 | 20 | 0.0018 | 0.1893 | 1 | 0.021 | 0.09731 | 1 | no |
| 5 | 0 | - | - | - | - | - | - | - |
| 8 | 20 | 0.0028 | 0.05826 | 1 | 0.031 | 0.01208 | 0.3865 | no |
| 16 | 10 | 0.0009 | 0.8457 | 1 | -0.008 | 0.7695 | 1 | no |
| 32 | 0 | - | - | - | - | - | - | - |

## multisensory_s1_D6_noise1.3_svbmc

Single runs (`M = 1`, n = 320): MMTV 0.225 [0.219, 0.229], gsKL 2.66 [2.43, 2.77], evidence error 0.42 [0.38, 0.46].

All 120 cells of this condition: runtime ratio 0.205 [0.171, 0.229], max\|dw\| 0.0184 [0.0164, 0.0216].

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. The shrinkage column shows its contributing and numerically unavailable cell counts; older cells that did not record shrinkage are counted separately. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias shrink int (contributing / unavailable / not recorded) | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.115 [0.060, 0.161] | 0.360 [0.289, 0.457] | 0.205 [0.152, 0.309] (20 / 0 / 0) | 0.459 [0.392, 0.582] | 0.216 [0.142, 0.278] | -0.380 [-0.464, -0.235] | 0.527 [0.400, 0.569] | 0.529 [0.394, 0.567] | yes |
| 3 | 20 | 0.113 [0.058, 0.202] | 0.370 [0.279, 0.421] | 0.157 [0.110, 0.270] (20 / 0 / 0) | - | - | - | 0.445 [0.363, 0.506] | - | - |
| 4 | 20 | 0.112 [0.035, 0.175] | 0.360 [0.329, 0.432] | 0.175 [0.127, 0.251] (20 / 0 / 0) | 0.426 [0.394, 0.580] | 0.200 [0.144, 0.237] | -0.368 [-0.461, -0.279] | 0.299 [0.254, 0.428] | 0.285 [0.257, 0.442] | yes |
| 5 | 20 | 0.106 [-0.003, 0.172] | 0.449 [0.427, 0.476] | 0.226 [0.198, 0.263] (20 / 0 / 0) | - | - | - | 0.296 [0.237, 0.488] | - | - |
| 8 | 20 | 0.131 [0.094, 0.161] | 0.519 [0.476, 0.569] | 0.271 [0.237, 0.325] (20 / 0 / 0) | 0.592 [0.547, 0.674] | 0.195 [0.152, 0.206] | -0.468 [-0.537, -0.428] | 0.269 [0.229, 0.344] | 0.262 [0.239, 0.318] | yes |
| 16 | 10 | 0.178 [0.108, 0.242] | 0.682 [0.587, 0.746] | 0.382 [0.353, 0.403] (10 / 0 / 0) | 0.674 [0.568, 0.773] | 0.215 [0.100, 0.255] | -0.454 [-0.708, -0.365] | 0.255 [0.212, 0.280] | 0.230 [0.209, 0.261] | yes |
| 32 | 10 | 0.133 [0.059, 0.201] | 0.691 [0.650, 0.803] | 0.406 [0.382, 0.424] (10 / 0 / 0) | - | - | - | 0.162 [0.137, 0.202] | - | - |

Growth of the integrated headline's median bias from `M = 2` to `M = 32`: 0.018 nats (0.115 to 0.133), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 2 | 3.213 [3.073, 3.385] | 3.236 [3.058, 3.412] | 3.237 [3.054, 3.369] | 3.334 [3.147, 3.450] | -502.712 [-502.755, -502.586] | 0.023 | -502.715 [-502.753, -502.580] | 0.022 |
| 3 | 3.303 [3.231, 3.362] | 3.297 [3.224, 3.357] | - | - | -502.631 [-502.692, -502.549] | 0.021 | - | - |
| 4 | 3.440 [3.308, 3.526] | 3.445 [3.287, 3.508] | 3.457 [3.316, 3.490] | 3.509 [3.381, 3.574] | -502.484 [-502.614, -502.440] | 0.020 | -502.471 [-502.628, -502.443] | 0.020 |
| 5 | 3.425 [3.244, 3.537] | 3.442 [3.246, 3.542] | - | - | -502.482 [-502.674, -502.423] | 0.020 | - | - |
| 8 | 3.457 [3.411, 3.523] | 3.451 [3.409, 3.517] | 3.490 [3.427, 3.529] | 3.543 [3.462, 3.575] | -502.455 [-502.530, -502.414] | 0.019 | -502.447 [-502.504, -502.425] | 0.020 |
| 16 | 3.564 [3.500, 3.596] | 3.575 [3.501, 3.600] | 3.552 [3.486, 3.601] | 3.606 [3.536, 3.634] | -502.441 [-502.466, -502.398] | 0.019 | -502.415 [-502.447, -502.393] | 0.019 |
| 32 | 3.616 [3.537, 3.637] | 3.624 [3.541, 3.644] | - | - | -502.347 [-502.388, -502.323] | 0.018 | - | - |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.180 [0.133, 0.196] | 0.183 [0.130, 0.199] | 0.0008 [-0.0018, 0.0032] | 1.48 [0.52, 1.86] | 1.50 [0.52, 1.90] | 0.018 [-0.012, 0.066] | 0.39 [0.28, 0.41] | 0.14 [0.07, 0.18] | 0.190 [0.125, 0.365] | 0.0216 [0.0160, 0.0298] | 0.4 [0.3, 0.4] | 1.9 [1.5, 2.4] | 0.178 [0.153, 0.263] |
| 3 | 20 | 0.172 [0.110, 0.191] | - | - | 1.27 [0.40, 1.68] | - | - | 0.25 [0.18, 0.41] | - | - | - | 1.0 [0.9, 1.1] | - | - |
| 4 | 20 | 0.116 [0.092, 0.170] | 0.101 [0.086, 0.164] | 0.0045 [-0.0017, 0.0110] | 0.25 [0.15, 1.51] | 0.26 [0.14, 1.46] | 0.017 [-0.007, 0.060] | 0.24 [0.10, 0.41] | 0.09 [0.05, 0.18] | 0.093 [0.013, 0.327] | 0.0191 [0.0162, 0.0226] | 2.1 [1.8, 2.4] | 11.7 [9.6, 13.9] | 0.174 [0.139, 0.225] |
| 5 | 20 | 0.110 [0.083, 0.198] | - | - | 0.48 [0.15, 2.09] | - | - | 0.21 [0.12, 0.40] | - | - | - | 3.9 [3.0, 4.2] | - | - |
| 8 | 20 | 0.109 [0.097, 0.125] | 0.107 [0.094, 0.131] | 0.0034 [-0.0036, 0.0084] | 0.33 [0.18, 0.66] | 0.27 [0.16, 0.62] | 0.008 [-0.019, 0.042] | 0.14 [0.11, 0.26] | 0.32 [0.22, 0.38] | -0.187 [-0.260, -0.011] | 0.0200 [0.0167, 0.0256] | 10.3 [8.9, 12.1] | 51.2 [46.5, 58.5] | 0.209 [0.178, 0.238] |
| 16 | 10 | 0.090 [0.075, 0.111] | 0.096 [0.072, 0.122] | -0.0035 [-0.0072, 0.0001] | 0.18 [0.14, 0.29] | 0.20 [0.14, 0.35] | -0.021 [-0.050, 0.000] | 0.06 [0.02, 0.16] | 0.46 [0.36, 0.53] | -0.370 [-0.452, -0.210] | 0.0146 [0.0114, 0.0171] | 67.7 [58.3, 72.8] | 299.2 [264.3, 322.6] | 0.228 [0.171, 0.281] |
| 32 | 10 | 0.079 [0.067, 0.090] | - | - | 0.12 [0.08, 0.14] | - | - | 0.05 [0.03, 0.11] | - | - | - | 510.4 [440.6, 535.2] | - | - |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.0008 | 0.4304 | 1 | 0.018 | 0.2455 | 1 | no |
| 3 | 0 | - | - | - | - | - | - | - |
| 4 | 20 | 0.0045 | 0.1429 | 1 | 0.017 | 0.1893 | 1 | no |
| 5 | 0 | - | - | - | - | - | - | - |
| 8 | 20 | 0.0034 | 0.2943 | 1 | 0.008 | 0.3118 | 1 | no |
| 16 | 10 | -0.0035 | 0.1309 | 1 | -0.021 | 0.08398 | 1 | no |
| 32 | 0 | - | - | - | - | - | - | - |

## rosenbrock_D2_noise3_svbmc

Single runs (`M = 1`, n = 320): MMTV 0.114 [0.100, 0.128], gsKL 0.19 [0.15, 0.25], evidence error 0.20 [0.18, 0.22].

All 120 cells of this condition: runtime ratio 0.363 [0.308, 0.390], max\|dw\| 0.0168 [0.0161, 0.0205].

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. The shrinkage column shows its contributing and numerically unavailable cell counts; older cells that did not record shrinkage are counted separately. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias shrink int (contributing / unavailable / not recorded) | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.062 [-0.042, 0.168] | 0.273 [0.198, 0.500] | 0.071 [-0.044, 0.272] (19 / 1 / 0) | 0.337 [0.220, 0.563] | 0.095 [-0.024, 0.235] | -0.233 [-0.340, -0.121] | 0.106 [0.064, 0.194] | 0.107 [0.065, 0.203] | yes |
| 3 | 20 | 0.129 [0.014, 0.199] | 0.436 [0.355, 0.618] | 0.162 [0.009, 0.314] (20 / 0 / 0) | - | - | - | 0.094 [0.060, 0.141] | - | - |
| 4 | 20 | 0.091 [0.002, 0.258] | 0.507 [0.287, 0.571] | 0.171 [-0.052, 0.249] (20 / 0 / 0) | 0.439 [0.356, 0.660] | 0.137 [0.033, 0.289] | -0.354 [-0.523, -0.263] | 0.093 [0.038, 0.108] | 0.082 [0.045, 0.140] | yes |
| 5 | 20 | 0.098 [-0.066, 0.189] | 0.654 [0.515, 0.777] | 0.152 [0.053, 0.356] (20 / 0 / 0) | - | - | - | 0.099 [0.073, 0.182] | - | - |
| 8 | 20 | 0.131 [0.049, 0.188] | 0.661 [0.562, 0.881] | 0.172 [0.078, 0.260] (19 / 1 / 0) | 0.713 [0.608, 0.921] | 0.165 [0.103, 0.230] | -0.623 [-0.831, -0.484] | 0.084 [0.049, 0.103] | 0.082 [0.044, 0.108] | yes |
| 16 | 10 | 0.043 [-0.023, 0.104] | 0.798 [0.537, 0.943] | 0.189 [0.027, 0.338] (9 / 1 / 0) | 0.815 [0.666, 0.989] | 0.115 [0.092, 0.140] | -0.718 [-0.957, -0.563] | 0.088 [0.033, 0.155] | 0.095 [0.047, 0.154] | yes |
| 32 | 10 | 0.073 [-0.018, 0.196] | 0.996 [0.917, 1.210] | 0.253 [0.201, 0.400] (9 / 1 / 0) | - | - | - | 0.075 [0.060, 0.201] | - | - |

Growth of the integrated headline's median bias from `M = 2` to `M = 32`: 0.010 nats (0.062 to 0.073), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 2 | 2.449 [2.368, 2.505] | 2.435 [2.376, 2.502] | 2.441 [2.385, 2.514] | 2.507 [2.416, 2.545] | -2.366 [-2.454, -2.324] | 0.013 | -2.366 [-2.462, -2.325] | 0.013 |
| 3 | 2.459 [2.390, 2.491] | 2.454 [2.393, 2.499] | - | - | -2.354 [-2.401, -2.320] | 0.012 | - | - |
| 4 | 2.509 [2.458, 2.553] | 2.502 [2.443, 2.541] | 2.510 [2.456, 2.538] | 2.552 [2.500, 2.575] | -2.353 [-2.368, -2.298] | 0.012 | -2.342 [-2.400, -2.305] | 0.011 |
| 5 | 2.424 [2.381, 2.548] | 2.417 [2.380, 2.550] | - | - | -2.359 [-2.441, -2.333] | 0.011 | - | - |
| 8 | 2.542 [2.460, 2.553] | 2.529 [2.465, 2.551] | 2.542 [2.434, 2.551] | 2.576 [2.489, 2.607] | -2.344 [-2.362, -2.309] | 0.012 | -2.342 [-2.368, -2.304] | 0.012 |
| 16 | 2.537 [2.437, 2.557] | 2.531 [2.425, 2.552] | 2.530 [2.437, 2.554] | 2.566 [2.472, 2.592] | -2.348 [-2.415, -2.292] | 0.011 | -2.354 [-2.414, -2.307] | 0.011 |
| 32 | 2.492 [2.427, 2.518] | 2.483 [2.423, 2.524] | - | - | -2.335 [-2.461, -2.320] | 0.011 | - | - |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.112 [0.080, 0.151] | 0.103 [0.072, 0.153] | 0.0012 [-0.0045, 0.0070] | 0.09 [0.03, 0.17] | 0.10 [0.03, 0.16] | -0.002 [-0.013, 0.004] | 0.18 [0.10, 0.28] | 0.29 [0.19, 0.45] | -0.117 [-0.251, 0.020] | 0.0202 [0.0162, 0.0266] | 0.4 [0.3, 0.4] | 1.4 [1.3, 1.6] | 0.259 [0.210, 0.296] |
| 3 | 20 | 0.109 [0.094, 0.139] | - | - | 0.08 [0.05, 0.11] | - | - | 0.08 [0.04, 0.19] | - | - | - | 1.0 [0.9, 1.1] | - | - |
| 4 | 20 | 0.105 [0.064, 0.125] | 0.106 [0.062, 0.138] | -0.0014 [-0.0050, 0.0034] | 0.05 [0.02, 0.08] | 0.06 [0.02, 0.09] | -0.005 [-0.012, 0.003] | 0.19 [0.09, 0.24] | 0.39 [0.25, 0.52] | -0.161 [-0.345, -0.141] | 0.0198 [0.0164, 0.0293] | 1.8 [1.6, 2.1] | 5.7 [4.6, 6.4] | 0.308 [0.272, 0.374] |
| 5 | 20 | 0.114 [0.091, 0.129] | - | - | 0.07 [0.05, 0.11] | - | - | 0.17 [0.12, 0.25] | - | - | - | 3.4 [3.2, 4.0] | - | - |
| 8 | 20 | 0.090 [0.068, 0.097] | 0.092 [0.074, 0.096] | 0.0001 [-0.0041, 0.0028] | 0.03 [0.02, 0.05] | 0.04 [0.02, 0.05] | -0.002 [-0.006, 0.001] | 0.10 [0.06, 0.15] | 0.64 [0.52, 0.83] | -0.510 [-0.784, -0.362] | 0.0141 [0.0116, 0.0176] | 9.7 [7.9, 11.0] | 22.0 [20.6, 25.0] | 0.400 [0.386, 0.444] |
| 16 | 10 | 0.095 [0.056, 0.148] | 0.089 [0.068, 0.152] | -0.0033 [-0.0085, 0.0033] | 0.03 [0.01, 0.05] | 0.03 [0.01, 0.06] | -0.002 [-0.005, 0.002] | 0.04 [0.02, 0.19] | 0.68 [0.60, 0.87] | -0.623 [-0.825, -0.489] | 0.0142 [0.0089, 0.0327] | 46.6 [39.6, 55.5] | 116.1 [92.3, 134.0] | 0.443 [0.341, 0.544] |
| 32 | 10 | 0.093 [0.069, 0.175] | - | - | 0.03 [0.01, 0.12] | - | - | 0.10 [0.06, 0.14] | - | - | - | 301.4 [291.6, 342.6] | - | - |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.0012 | 0.3884 | 1 | -0.002 | 0.4524 | 1 | no |
| 3 | 0 | - | - | - | - | - | - | - |
| 4 | 20 | -0.0014 | 0.4304 | 1 | -0.005 | 0.06958 | 1 | no |
| 5 | 0 | - | - | - | - | - | - | - |
| 8 | 20 | 0.0001 | 0.6215 | 1 | -0.002 | 0.114 | 1 | no |
| 16 | 10 | -0.0033 | 0.3223 | 1 | -0.002 | 0.3223 | 1 | no |
| 32 | 0 | - | - | - | - | - | - | - |

## gmm_D2_noise3_svbmc

Single runs (`M = 1`, n = 320): MMTV 0.471 [0.449, 0.501], gsKL 19.67 [18.61, 21.74], evidence error 1.02 [0.95, 1.09].

All 120 cells of this condition: runtime ratio 0.434 [0.378, 0.503], max\|dw\| 0.0184 [0.0162, 0.0206].

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. The shrinkage column shows its contributing and numerically unavailable cell counts; older cells that did not record shrinkage are counted separately. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias shrink int (contributing / unavailable / not recorded) | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.061 [-0.133, 0.228] | 0.227 [0.035, 0.351] | 0.010 [-0.073, 0.041] (20 / 0 / 0) | 0.306 [0.095, 0.393] | 0.083 [-0.053, 0.277] | -0.156 [-0.259, -0.073] | 0.563 [0.522, 0.843] | 0.561 [0.511, 0.830] | yes |
| 3 | 20 | 0.050 [-0.155, 0.214] | 0.364 [0.285, 0.570] | 0.085 [-0.077, 0.217] (20 / 0 / 0) | - | - | - | 0.472 [0.404, 0.569] | - | - |
| 4 | 20 | -0.082 [-0.180, 0.239] | 0.458 [0.284, 0.557] | 0.052 [-0.111, 0.204] (20 / 0 / 0) | 0.501 [0.314, 0.591] | -0.022 [-0.132, 0.293] | -0.429 [-0.607, -0.320] | 0.329 [0.252, 0.397] | 0.314 [0.266, 0.409] | yes |
| 5 | 20 | 0.031 [-0.077, 0.106] | 0.420 [0.302, 0.519] | 0.066 [-0.058, 0.169] (20 / 0 / 0) | - | - | - | 0.233 [0.213, 0.287] | - | - |
| 8 | 20 | 0.023 [-0.056, 0.100] | 0.541 [0.418, 0.634] | 0.124 [0.027, 0.247] (20 / 0 / 0) | 0.576 [0.463, 0.742] | 0.068 [-0.017, 0.124] | -0.616 [-0.709, -0.478] | 0.217 [0.203, 0.226] | 0.215 [0.200, 0.230] | yes |
| 16 | 10 | 0.023 [-0.037, 0.159] | 0.800 [0.657, 0.902] | 0.226 [0.033, 0.274] (10 / 0 / 0) | 0.827 [0.689, 0.908] | 0.077 [0.019, 0.191] | -0.677 [-0.899, -0.607] | 0.191 [0.152, 0.230] | 0.176 [0.147, 0.207] | yes |
| 32 | 10 | 0.058 [-0.016, 0.128] | 0.925 [0.834, 0.989] | 0.256 [0.132, 0.302] (10 / 0 / 0) | - | - | - | 0.139 [0.120, 0.156] | - | - |

Growth of the integrated headline's median bias from `M = 2` to `M = 32`: -0.002 nats (0.061 to 0.058), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 2 | 3.995 [3.811, 4.080] | 3.994 [3.803, 4.093] | 3.999 [3.807, 4.102] | 4.075 [3.859, 4.141] | 2.433 [2.153, 2.473] | 0.016 | 2.435 [2.166, 2.484] | 0.016 |
| 3 | 4.103 [4.028, 4.234] | 4.133 [4.034, 4.213] | - | - | 2.524 [2.427, 2.592] | 0.013 | - | - |
| 4 | 4.241 [4.149, 4.337] | 4.238 [4.147, 4.334] | 4.248 [4.159, 4.315] | 4.287 [4.221, 4.367] | 2.667 [2.599, 2.744] | 0.014 | 2.681 [2.587, 2.729] | 0.014 |
| 5 | 4.333 [4.274, 4.371] | 4.332 [4.275, 4.367] | - | - | 2.762 [2.709, 2.783] | 0.013 | - | - |
| 8 | 4.394 [4.343, 4.427] | 4.371 [4.347, 4.428] | 4.388 [4.341, 4.409] | 4.442 [4.416, 4.464] | 2.778 [2.769, 2.793] | 0.013 | 2.780 [2.766, 2.796] | 0.014 |
| 16 | 4.390 [4.368, 4.484] | 4.396 [4.373, 4.472] | 4.403 [4.360, 4.488] | 4.484 [4.392, 4.565] | 2.804 [2.766, 2.844] | 0.013 | 2.819 [2.788, 2.848] | 0.013 |
| 32 | 4.470 [4.449, 4.508] | 4.480 [4.438, 4.506] | - | - | 2.857 [2.840, 2.875] | 0.013 | - | - |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.264 [0.245, 0.304] | 0.260 [0.239, 0.302] | -0.0005 [-0.0049, 0.0073] | 0.83 [0.41, 6.20] | 0.75 [0.38, 6.12] | 0.018 [-0.007, 0.053] | 0.64 [0.44, 0.78] | 0.39 [0.27, 0.48] | 0.142 [0.069, 0.268] | 0.0192 [0.0154, 0.0209] | 0.3 [0.2, 0.3] | 0.9 [0.7, 1.2] | 0.301 [0.246, 0.378] |
| 3 | 20 | 0.249 [0.222, 0.285] | - | - | 0.28 [0.18, 0.54] | - | - | 0.42 [0.32, 0.63] | - | - | - | 0.7 [0.6, 0.8] | - | - |
| 4 | 20 | 0.183 [0.165, 0.232] | 0.189 [0.162, 0.234] | 0.0010 [-0.0037, 0.0050] | 0.13 [0.07, 0.28] | 0.13 [0.08, 0.31] | -0.004 [-0.014, 0.002] | 0.44 [0.21, 0.58] | 0.24 [0.17, 0.39] | 0.144 [-0.019, 0.302] | 0.0173 [0.0156, 0.0235] | 1.5 [1.3, 1.7] | 3.2 [2.7, 3.6] | 0.419 [0.361, 0.568] |
| 5 | 20 | 0.140 [0.115, 0.161] | - | - | 0.04 [0.03, 0.06] | - | - | 0.23 [0.19, 0.32] | - | - | - | 2.7 [2.2, 3.0] | - | - |
| 8 | 20 | 0.140 [0.122, 0.169] | 0.140 [0.116, 0.177] | -0.0013 [-0.0033, 0.0032] | 0.04 [0.03, 0.06] | 0.03 [0.02, 0.06] | -0.000 [-0.004, 0.007] | 0.20 [0.15, 0.26] | 0.38 [0.25, 0.53] | -0.245 [-0.330, -0.025] | 0.0229 [0.0159, 0.0266] | 8.0 [6.9, 9.4] | 17.2 [13.4, 19.7] | 0.480 [0.416, 0.581] |
| 16 | 10 | 0.119 [0.102, 0.134] | 0.120 [0.101, 0.136] | -0.0001 [-0.0029, 0.0032] | 0.02 [0.01, 0.03] | 0.02 [0.01, 0.03] | -0.000 [-0.004, 0.002] | 0.14 [0.09, 0.21] | 0.63 [0.51, 0.71] | -0.528 [-0.563, -0.382] | 0.0153 [0.0109, 0.0203] | 44.0 [39.8, 45.1] | 78.1 [63.4, 92.4] | 0.552 [0.479, 0.674] |
| 32 | 10 | 0.097 [0.086, 0.111] | - | - | 0.01 [0.01, 0.03] | - | - | 0.10 [0.03, 0.14] | - | - | - | 240.5 [232.9, 254.7] | - | - |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 2 | 20 | -0.0005 | 0.7012 | 1 | 0.018 | 0.1536 | 1 | no |
| 3 | 0 | - | - | - | - | - | - | - |
| 4 | 20 | 0.0010 | 0.8983 | 1 | -0.004 | 0.2611 | 1 | no |
| 5 | 0 | - | - | - | - | - | - | - |
| 8 | 20 | -0.0013 | 0.7841 | 1 | -0.000 | 0.7841 | 1 | no |
| 16 | 10 | -0.0001 | 0.9219 | 1 | -0.000 | 0.5566 | 1 | no |
| 32 | 0 | - | - | - | - | - | - | - |

## ring_D2_noise3_svbmc

Single runs (`M = 1`, n = 320): MMTV 0.613 [0.591, 0.650], gsKL 68.21 [37.06, 129.86], evidence error 1.38 [1.27, 1.49].

All 120 cells of this condition: runtime ratio 0.460 [0.420, 0.502], max\|dw\| 0.0070 [0.0063, 0.0091].

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. The shrinkage column shows its contributing and numerically unavailable cell counts; older cells that did not record shrinkage are counted separately. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias shrink int (contributing / unavailable / not recorded) | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.279 [0.179, 0.387] | 0.399 [0.291, 0.531] | 0.273 [0.132, 0.331] (20 / 0 / 0) | 0.439 [0.335, 0.575] | 0.335 [0.234, 0.425] | -0.205 [-0.280, -0.069] | 1.018 [0.845, 1.150] | 1.019 [0.843, 1.148] | yes |
| 3 | 20 | 0.239 [0.180, 0.307] | 0.374 [0.325, 0.481] | 0.174 [0.100, 0.222] (20 / 0 / 0) | - | - | - | 0.712 [0.609, 0.941] | - | - |
| 4 | 20 | 0.193 [0.056, 0.291] | 0.334 [0.238, 0.475] | 0.180 [0.018, 0.232] (20 / 0 / 0) | 0.402 [0.267, 0.532] | 0.264 [0.081, 0.323] | -0.243 [-0.278, -0.188] | 0.629 [0.535, 0.723] | 0.628 [0.537, 0.736] | yes |
| 5 | 20 | 0.202 [0.137, 0.282] | 0.398 [0.343, 0.508] | 0.135 [0.083, 0.242] (20 / 0 / 0) | - | - | - | 0.417 [0.295, 0.509] | - | - |
| 8 | 20 | 0.232 [0.187, 0.270] | 0.565 [0.488, 0.693] | 0.183 [0.150, 0.239] (20 / 0 / 0) | 0.591 [0.562, 0.733] | 0.248 [0.210, 0.304] | -0.395 [-0.473, -0.344] | 0.222 [0.193, 0.356] | 0.234 [0.199, 0.367] | yes |
| 16 | 10 | 0.224 [0.171, 0.250] | 0.694 [0.615, 0.748] | 0.170 [0.131, 0.224] (10 / 0 / 0) | 0.721 [0.659, 0.779] | 0.254 [0.192, 0.276] | -0.541 [-0.584, -0.414] | 0.129 [0.110, 0.188] | 0.129 [0.114, 0.188] | yes |
| 32 | 10 | 0.190 [0.158, 0.241] | 0.904 [0.815, 0.927] | 0.172 [0.100, 0.225] (10 / 0 / 0) | - | - | - | 0.091 [0.080, 0.106] | - | - |

Growth of the integrated headline's median bias from `M = 2` to `M = 32`: -0.089 nats (0.279 to 0.190), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 2 | 2.086 [1.988, 2.320] | 2.091 [1.977, 2.316] | 2.085 [1.987, 2.319] | 2.121 [2.008, 2.365] | 1.516 [1.384, 1.688] | 0.011 | 1.515 [1.385, 1.691] | 0.011 |
| 3 | 2.414 [2.146, 2.556] | 2.412 [2.142, 2.568] | - | - | 1.822 [1.593, 1.925] | 0.010 | - | - |
| 4 | 2.508 [2.403, 2.605] | 2.512 [2.395, 2.600] | 2.499 [2.407, 2.606] | 2.544 [2.427, 2.650] | 1.904 [1.810, 1.999] | 0.010 | 1.906 [1.798, 1.997] | 0.010 |
| 5 | 2.747 [2.634, 2.844] | 2.745 [2.633, 2.842] | - | - | 2.117 [2.030, 2.238] | 0.010 | - | - |
| 8 | 2.909 [2.746, 2.942] | 2.912 [2.752, 2.936] | 2.904 [2.745, 2.944] | 2.930 [2.771, 2.963] | 2.311 [2.178, 2.340] | 0.010 | 2.299 [2.166, 2.334] | 0.010 |
| 16 | 2.976 [2.931, 3.001] | 2.967 [2.942, 2.998] | 2.976 [2.934, 3.003] | 2.986 [2.974, 3.023] | 2.405 [2.346, 2.423] | 0.009 | 2.405 [2.345, 2.419] | 0.009 |
| 32 | 3.013 [3.008, 3.020] | 3.013 [3.007, 3.018] | - | - | 2.443 [2.428, 2.454] | 0.009 | - | - |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.404 [0.357, 0.450] | 0.409 [0.353, 0.449] | -0.0012 [-0.0024, 0.0017] | 1.33 [0.48, 4.50] | 1.34 [0.48, 4.59] | 0.003 [-0.028, 0.021] | 0.71 [0.58, 0.95] | 0.57 [0.43, 0.72] | 0.183 [0.060, 0.273] | 0.0109 [0.0080, 0.0131] | 0.3 [0.3, 0.4] | 0.9 [0.8, 1.1] | 0.373 [0.314, 0.433] |
| 3 | 20 | 0.305 [0.259, 0.394] | - | - | 0.54 [0.19, 2.12] | - | - | 0.47 [0.31, 0.75] | - | - | - | 0.9 [0.9, 1.0] | - | - |
| 4 | 20 | 0.275 [0.233, 0.305] | 0.277 [0.236, 0.313] | -0.0023 [-0.0028, 0.0024] | 0.22 [0.11, 0.26] | 0.22 [0.11, 0.27] | -0.001 [-0.019, 0.003] | 0.47 [0.32, 0.52] | 0.22 [0.14, 0.34] | 0.173 [0.140, 0.261] | 0.0076 [0.0066, 0.0100] | 1.7 [1.6, 1.9] | 3.6 [3.0, 4.3] | 0.454 [0.380, 0.554] |
| 5 | 20 | 0.208 [0.177, 0.251] | - | - | 0.12 [0.04, 0.17] | - | - | 0.17 [0.10, 0.28] | - | - | - | 3.5 [2.9, 3.7] | - | - |
| 8 | 20 | 0.159 [0.144, 0.221] | 0.161 [0.151, 0.224] | -0.0015 [-0.0025, 0.0004] | 0.05 [0.03, 0.07] | 0.06 [0.04, 0.07] | -0.002 [-0.004, 0.002] | 0.10 [0.05, 0.14] | 0.29 [0.22, 0.38] | -0.193 [-0.288, -0.059] | 0.0064 [0.0054, 0.0086] | 9.6 [8.8, 11.9] | 21.3 [17.9, 24.4] | 0.484 [0.440, 0.539] |
| 16 | 10 | 0.118 [0.108, 0.134] | 0.117 [0.101, 0.137] | 0.0007 [-0.0040, 0.0031] | 0.01 [0.01, 0.03] | 0.01 [0.01, 0.03] | -0.000 [-0.002, 0.001] | 0.09 [0.01, 0.15] | 0.59 [0.47, 0.67] | -0.483 [-0.555, -0.405] | 0.0051 [0.0036, 0.0059] | 47.8 [41.7, 54.7] | 91.3 [81.1, 105.7] | 0.514 [0.468, 0.585] |
| 32 | 10 | 0.092 [0.088, 0.106] | - | - | 0.01 [0.00, 0.02] | - | - | 0.11 [0.04, 0.15] | - | - | - | 274.9 [247.9, 302.3] | - | - |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 2 | 20 | -0.0012 | 0.7012 | 1 | 0.003 | 0.9273 | 1 | no |
| 3 | 0 | - | - | - | - | - | - | - |
| 4 | 20 | -0.0023 | 0.3884 | 1 | -0.001 | 0.2943 | 1 | no |
| 5 | 0 | - | - | - | - | - | - | - |
| 8 | 20 | -0.0015 | 0.2774 | 1 | -0.002 | 0.1231 | 1 | no |
| 16 | 10 | 0.0007 | 0.8457 | 1 | -0.000 | 0.5566 | 1 | no |
| 32 | 0 | - | - | - | - | - | - | - |

## student_D8_noise3_svbmc

Single runs (`M = 1`, n = 320): MMTV 0.138 [0.134, 0.141], gsKL 0.56 [0.52, 0.60], evidence error 0.94 [0.86, 0.99].

All 120 cells of this condition: runtime ratio 0.433 [0.357, 0.493], max\|dw\| 0.0269 [0.0254, 0.0302].

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. The shrinkage column shows its contributing and numerically unavailable cell counts; older cells that did not record shrinkage are counted separately. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias shrink int (contributing / unavailable / not recorded) | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | -0.709 [-1.078, -0.575] | -0.105 [-0.296, 0.012] | -0.364 [-0.582, -0.243] (20 / 0 / 0) | -0.013 [-0.142, 0.129] | -0.554 [-0.949, -0.470] | -0.841 [-1.092, -0.712] | 0.574 [0.490, 0.644] | 0.594 [0.483, 0.643] | **no** |
| 3 | 20 | -1.215 [-1.466, -0.787] | -0.102 [-0.239, 0.099] | -0.391 [-0.587, -0.212] (20 / 0 / 0) | - | - | - | 0.415 [0.374, 0.476] | - | - |
| 4 | 20 | -1.143 [-1.210, -0.904] | 0.038 [-0.033, 0.114] | -0.292 [-0.338, -0.201] (20 / 0 / 0) | 0.158 [0.021, 0.279] | -0.932 [-1.036, -0.704] | -1.224 [-1.317, -0.956] | 0.402 [0.359, 0.450] | 0.401 [0.367, 0.474] | **no** |
| 5 | 20 | -1.509 [-1.707, -0.927] | 0.049 [-0.050, 0.098] | -0.335 [-0.370, -0.277] (20 / 0 / 0) | - | - | - | 0.371 [0.327, 0.394] | - | - |
| 8 | 20 | -1.385 [-1.458, -1.107] | 0.128 [-0.054, 0.151] | -0.214 [-0.362, -0.172] (20 / 0 / 0) | 0.162 [0.091, 0.216] | -1.202 [-1.326, -1.055] | -1.539 [-1.721, -1.240] | 0.279 [0.261, 0.339] | 0.307 [0.253, 0.363] | **no** |
| 16 | 10 | -1.286 [-1.721, -1.168] | 0.151 [0.021, 0.232] | -0.243 [-0.344, -0.185] (10 / 0 / 0) | 0.283 [0.223, 0.354] | -1.276 [-1.587, -1.165] | -1.678 [-2.030, -1.391] | 0.222 [0.191, 0.265] | 0.229 [0.186, 0.267] | **no** |
| 32 | 10 | -1.637 [-1.744, -1.544] | 0.268 [0.193, 0.426] | -0.179 [-0.230, -0.018] (10 / 0 / 0) | - | - | - | 0.184 [0.170, 0.215] | - | - |

Growth of the integrated headline's median bias from `M = 2` to `M = 32`: -0.929 nats (-0.709 to -1.637), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 2 | 13.688 [13.523, 13.896] | 13.699 [13.535, 13.890] | 13.724 [13.612, 13.946] | 13.872 [13.711, 14.056] | -20.177 [-20.247, -20.094] | 0.038 | -20.197 [-20.247, -20.086] | 0.038 |
| 3 | 13.658 [13.408, 13.773] | 13.614 [13.380, 13.780] | - | - | -20.018 [-20.079, -19.977] | 0.035 | - | - |
| 4 | 13.608 [13.406, 13.755] | 13.651 [13.412, 13.789] | 13.660 [13.464, 13.741] | 13.808 [13.599, 13.888] | -20.005 [-20.053, -19.962] | 0.035 | -20.004 [-20.077, -19.970] | 0.034 |
| 5 | 13.664 [13.186, 13.751] | 13.639 [13.222, 13.747] | - | - | -19.975 [-19.997, -19.930] | 0.033 | - | - |
| 8 | 13.472 [13.279, 13.624] | 13.501 [13.251, 13.631] | 13.440 [13.351, 13.670] | 13.580 [13.413, 13.760] | -19.883 [-19.942, -19.864] | 0.032 | -19.911 [-19.966, -19.857] | 0.032 |
| 16 | 13.413 [13.266, 13.775] | 13.413 [13.250, 13.763] | 13.379 [13.238, 13.563] | 13.495 [13.300, 13.632] | -19.826 [-19.869, -19.795] | 0.031 | -19.832 [-19.871, -19.790] | 0.032 |
| 32 | 13.415 [13.164, 13.450] | 13.417 [13.174, 13.446] | - | - | -19.788 [-19.822, -19.774] | 0.031 | - | - |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.120 [0.116, 0.128] | 0.120 [0.117, 0.128] | 0.0002 [-0.0004, 0.0012] | 0.40 [0.32, 0.46] | 0.40 [0.31, 0.46] | 0.005 [-0.004, 0.010] | 1.28 [1.18, 1.70] | 0.50 [0.41, 0.68] | 0.844 [0.740, 1.028] | 0.0273 [0.0211, 0.0487] | 0.5 [0.4, 0.6] | 1.2 [0.8, 1.3] | 0.409 [0.356, 0.548] |
| 3 | 20 | 0.111 [0.098, 0.119] | - | - | 0.25 [0.22, 0.29] | - | - | 1.62 [1.23, 1.86] | - | - | - | 1.0 [0.9, 1.1] | - | - |
| 4 | 20 | 0.101 [0.097, 0.108] | 0.102 [0.098, 0.110] | -0.0007 [-0.0022, 0.0000] | 0.19 [0.17, 0.26] | 0.19 [0.18, 0.26] | -0.005 [-0.011, 0.006] | 1.54 [1.27, 1.63] | 0.25 [0.16, 0.35] | 1.201 [0.949, 1.307] | 0.0285 [0.0255, 0.0375] | 2.6 [2.2, 3.0] | 5.0 [3.9, 6.9] | 0.475 [0.356, 0.628] |
| 5 | 20 | 0.097 [0.088, 0.104] | - | - | 0.18 [0.15, 0.21] | - | - | 1.84 [1.34, 2.11] | - | - | - | 3.8 [3.2, 4.3] | - | - |
| 8 | 20 | 0.090 [0.087, 0.093] | 0.090 [0.086, 0.094] | -0.0005 [-0.0033, 0.0022] | 0.12 [0.11, 0.17] | 0.13 [0.10, 0.18] | -0.001 [-0.005, 0.005] | 1.65 [1.43, 1.82] | 0.18 [0.09, 0.25] | 1.547 [1.105, 1.629] | 0.0255 [0.0198, 0.0357] | 10.0 [7.3, 12.1] | 24.2 [17.3, 26.7] | 0.507 [0.312, 0.567] |
| 16 | 10 | 0.079 [0.069, 0.091] | 0.080 [0.067, 0.088] | -0.0010 [-0.0017, 0.0032] | 0.09 [0.06, 0.11] | 0.09 [0.06, 0.11] | -0.002 [-0.011, 0.004] | 1.52 [1.39, 1.95] | 0.10 [0.04, 0.16] | 1.394 [1.269, 1.856] | 0.0273 [0.0172, 0.0326] | 59.7 [45.4, 69.8] | 241.8 [149.8, 288.0] | 0.239 [0.188, 0.445] |
| 32 | 10 | 0.073 [0.065, 0.080] | - | - | 0.07 [0.05, 0.09] | - | - | 1.87 [1.72, 1.96] | - | - | - | 382.8 [345.7, 406.5] | - | - |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.0002 | 0.6477 | 1 | 0.005 | 0.5459 | 1 | no |
| 3 | 0 | - | - | - | - | - | - | - |
| 4 | 20 | -0.0007 | 0.09731 | 1 | -0.005 | 0.4091 | 1 | no |
| 5 | 0 | - | - | - | - | - | - | - |
| 8 | 20 | -0.0005 | 0.6742 | 1 | -0.001 | 0.9273 | 1 | no |
| 16 | 10 | -0.0010 | 0.5566 | 1 | -0.002 | 0.375 | 1 | no |
| 32 | 0 | - | - | - | - | - | - | - |

## gmm_D2_svbmc

Single runs (`M = 1`, n = 320): MMTV 0.389 [0.387, 0.391], gsKL 15.08 [14.23, 15.36], evidence error 0.70 [0.70, 0.71].

All 120 cells of this condition: runtime ratio 0.526 [0.448, 0.560], max\|dw\| 0.0130 [0.0103, 0.0160].

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. The shrinkage column shows its contributing and numerically unavailable cell counts; older cells that did not record shrinkage are counted separately. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias shrink int (contributing / unavailable / not recorded) | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.008 [0.000, 0.019] | 0.008 [0.000, 0.019] | 0.013 [0.004, 0.027] (20 / 0 / 0) | 0.061 [0.042, 0.071] | 0.050 [0.027, 0.066] | -0.048 [-0.076, -0.027] | 0.352 [0.309, 0.709] | 0.358 [0.301, 0.717] | yes |
| 3 | 20 | 0.004 [-0.006, 0.015] | 0.004 [-0.006, 0.015] | 0.012 [-0.003, 0.022] (20 / 0 / 0) | - | - | - | 0.296 [0.039, 0.307] | - | - |
| 4 | 20 | 0.009 [0.001, 0.015] | 0.009 [0.001, 0.015] | 0.016 [0.006, 0.021] (20 / 0 / 0) | 0.054 [0.043, 0.070] | 0.039 [0.002, 0.055] | -0.037 [-0.049, -0.028] | 0.032 [0.026, 0.299] | 0.038 [0.019, 0.306] | yes |
| 5 | 20 | 0.019 [0.006, 0.022] | 0.019 [0.006, 0.022] | 0.024 [0.010, 0.029] (20 / 0 / 0) | - | - | - | 0.032 [0.023, 0.190] | - | - |
| 8 | 20 | 0.020 [0.016, 0.029] | 0.020 [0.016, 0.029] | 0.027 [0.022, 0.036] (20 / 0 / 0) | 0.065 [0.043, 0.077] | 0.054 [0.037, 0.076] | -0.036 [-0.050, -0.020] | 0.034 [0.020, 0.101] | 0.041 [0.025, 0.128] | yes |
| 16 | 10 | 0.019 [-0.000, 0.106] | 0.019 [-0.000, 0.106] | 0.025 [0.004, 0.126] (10 / 0 / 0) | 0.038 [0.013, 0.104] | 0.025 [0.002, 0.106] | -0.011 [-0.021, 0.001] | 0.024 [0.002, 0.100] | 0.019 [0.003, 0.092] | yes |
| 32 | 10 | 0.037 [0.001, 0.155] | 0.037 [0.001, 0.155] | 0.046 [0.006, 0.178] (10 / 0 / 0) | - | - | - | 0.031 [0.007, 0.145] | - | - |

Growth of the integrated headline's median bias from `M = 2` to `M = 32`: 0.029 nats (0.008 to 0.037), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 2 | 4.303 [3.910, 4.342] | 4.305 [3.907, 4.360] | 4.299 [3.908, 4.346] | 4.321 [3.966, 4.384] | 2.644 [2.287, 2.687] | 0.015 | 2.637 [2.279, 2.694] | 0.015 |
| 3 | 4.343 [4.323, 4.591] | 4.346 [4.319, 4.589] | - | - | 2.700 [2.688, 2.957] | 0.014 | - | - |
| 4 | 4.615 [4.385, 4.631] | 4.613 [4.382, 4.624] | 4.626 [4.373, 4.634] | 4.656 [4.428, 4.666] | 2.964 [2.697, 2.970] | 0.014 | 2.957 [2.690, 2.977] | 0.014 |
| 5 | 4.626 [4.603, 4.640] | 4.624 [4.602, 4.636] | - | - | 2.964 [2.806, 2.973] | 0.014 | - | - |
| 8 | 4.630 [4.612, 4.647] | 4.633 [4.618, 4.647] | 4.632 [4.622, 4.647] | 4.678 [4.655, 4.687] | 2.962 [2.895, 2.976] | 0.013 | 2.955 [2.868, 2.971] | 0.013 |
| 16 | 4.644 [4.631, 4.692] | 4.644 [4.630, 4.699] | 4.648 [4.632, 4.677] | 4.665 [4.646, 4.694] | 2.972 [2.896, 2.994] | 0.013 | 2.977 [2.904, 2.993] | 0.013 |
| 32 | 4.647 [4.634, 4.672] | 4.643 [4.632, 4.675] | - | - | 2.965 [2.872, 2.989] | 0.017 | - | - |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.207 [0.190, 0.369] | 0.194 [0.185, 0.372] | 0.0002 [-0.0048, 0.0036] | 0.92 [0.43, 12.05] | 0.93 [0.43, 11.88] | -0.001 [-0.005, 0.023] | 0.30 [0.29, 0.72] | 0.27 [0.25, 0.65] | 0.036 [0.026, 0.057] | 0.0149 [0.0101, 0.0223] | 0.3 [0.2, 0.3] | 0.7 [0.6, 0.9] | 0.358 [0.288, 0.476] |
| 3 | 20 | 0.189 [0.045, 0.196] | - | - | 0.23 [0.01, 0.69] | - | - | 0.28 [0.03, 0.30] | - | - | - | 0.7 [0.6, 0.8] | - | - |
| 4 | 20 | 0.043 [0.036, 0.182] | 0.049 [0.029, 0.182] | -0.0024 [-0.0054, 0.0093] | 0.01 [0.00, 0.42] | 0.01 [0.00, 0.43] | -0.000 [-0.004, 0.010] | 0.02 [0.02, 0.29] | 0.04 [0.03, 0.23] | 0.002 [-0.017, 0.020] | 0.0168 [0.0127, 0.0229] | 1.4 [1.2, 1.6] | 2.9 [2.6, 3.2] | 0.526 [0.424, 0.653] |
| 5 | 20 | 0.040 [0.028, 0.051] | - | - | 0.01 [0.00, 0.01] | - | - | 0.01 [0.01, 0.03] | - | - | - | 2.6 [2.3, 2.8] | - | - |
| 8 | 20 | 0.039 [0.031, 0.054] | 0.039 [0.033, 0.045] | -0.0033 [-0.0069, 0.0080] | 0.01 [0.00, 0.01] | 0.01 [0.00, 0.01] | 0.001 [-0.002, 0.003] | 0.01 [0.01, 0.02] | 0.04 [0.02, 0.05] | -0.013 [-0.026, 0.002] | 0.0128 [0.0099, 0.0154] | 5.5 [4.7, 6.4] | 10.8 [8.6, 11.5] | 0.541 [0.454, 0.646] |
| 16 | 10 | 0.032 [0.028, 0.036] | 0.023 [0.017, 0.032] | 0.0065 [0.0023, 0.0101] | 0.00 [0.00, 0.01] | 0.00 [0.00, 0.01] | 0.000 [-0.000, 0.002] | 0.01 [0.00, 0.01] | 0.01 [0.01, 0.02] | -0.007 [-0.014, -0.000] | 0.0065 [0.0051, 0.0080] | 25.8 [21.2, 30.8] | 41.2 [31.1, 44.2] | 0.673 [0.618, 0.731] |
| 32 | 10 | 0.025 [0.016, 0.038] | - | - | 0.00 [0.00, 0.00] | - | - | 0.00 [0.00, 0.01] | - | - | - | 71.8 [65.2, 102.6] | - | - |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.0002 | 0.8695 | 1 | -0.001 | 0.5958 | 1 | no |
| 3 | 0 | - | - | - | - | - | - | - |
| 4 | 20 | -0.0024 | 0.9854 | 1 | -0.000 | 0.7285 | 1 | no |
| 5 | 0 | - | - | - | - | - | - | - |
| 8 | 20 | -0.0033 | 0.9854 | 1 | 0.001 | 0.8983 | 1 | no |
| 16 | 10 | 0.0065 | 0.009766 | 0.3125 | 0.000 | 0.1934 | 1 | no |
| 32 | 0 | - | - | - | - | - | - | - |

## multisensory_s1_D6_svbmc

Single runs (`M = 1`, n = 320): MMTV 0.114 [0.111, 0.117], gsKL 0.49 [0.45, 0.53], evidence error 0.39 [0.38, 0.40].

All 120 cells of this condition: runtime ratio 0.235 [0.211, 0.265], max\|dw\| 0.0149 [0.0135, 0.0162].

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. The shrinkage column shows its contributing and numerically unavailable cell counts; older cells that did not record shrinkage are counted separately. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias shrink int (contributing / unavailable / not recorded) | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.005 [-0.007, 0.060] | 0.005 [-0.008, 0.060] | 0.014 [-0.000, 0.087] (20 / 0 / 0) | 0.099 [0.077, 0.118] | 0.044 [0.001, 0.079] | -0.091 [-0.124, -0.039] | 0.317 [0.275, 0.356] | 0.329 [0.293, 0.360] | yes |
| 3 | 20 | 0.028 [0.008, 0.068] | 0.028 [0.008, 0.068] | 0.041 [0.016, 0.084] (20 / 0 / 0) | - | - | - | 0.250 [0.208, 0.285] | - | - |
| 4 | 20 | 0.029 [-0.007, 0.056] | 0.029 [-0.007, 0.056] | 0.031 [0.003, 0.073] (20 / 0 / 0) | 0.095 [0.074, 0.117] | 0.052 [0.016, 0.075] | -0.075 [-0.091, -0.053] | 0.233 [0.192, 0.249] | 0.222 [0.189, 0.255] | yes |
| 5 | 20 | 0.055 [0.028, 0.076] | 0.055 [0.028, 0.076] | 0.073 [0.043, 0.084] (20 / 0 / 0) | - | - | - | 0.205 [0.178, 0.230] | - | - |
| 8 | 20 | 0.063 [0.027, 0.080] | 0.063 [0.027, 0.080] | 0.078 [0.042, 0.108] (20 / 0 / 0) | 0.117 [0.106, 0.131] | 0.107 [0.057, 0.113] | -0.053 [-0.077, -0.020] | 0.172 [0.137, 0.195] | 0.179 [0.158, 0.198] | yes |
| 16 | 10 | 0.063 [0.031, 0.092] | 0.063 [0.031, 0.092] | 0.082 [0.052, 0.116] (10 / 0 / 0) | 0.109 [0.074, 0.155] | 0.095 [0.050, 0.155] | -0.038 [-0.070, -0.004] | 0.150 [0.139, 0.171] | 0.148 [0.134, 0.178] | yes |
| 32 | 10 | 0.099 [0.075, 0.149] | 0.099 [0.075, 0.149] | 0.123 [0.092, 0.187] (10 / 0 / 0) | - | - | - | 0.133 [0.121, 0.169] | - | - |

Growth of the integrated headline's median bias from `M = 2` to `M = 32`: 0.094 nats (0.005 to 0.099), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 2 | 3.643 [3.619, 3.659] | 3.635 [3.611, 3.664] | 3.639 [3.607, 3.659] | 3.717 [3.649, 3.766] | -502.503 [-502.542, -502.460] | 0.024 | -502.515 [-502.546, -502.478] | 0.023 |
| 3 | 3.696 [3.670, 3.722] | 3.686 [3.672, 3.745] | - | - | -502.436 [-502.471, -502.394] | 0.022 | - | - |
| 4 | 3.723 [3.686, 3.783] | 3.724 [3.674, 3.792] | 3.746 [3.694, 3.786] | 3.792 [3.759, 3.835] | -502.419 [-502.435, -502.378] | 0.022 | -502.408 [-502.441, -502.375] | 0.022 |
| 5 | 3.793 [3.746, 3.828] | 3.797 [3.762, 3.826] | - | - | -502.391 [-502.416, -502.364] | 0.021 | - | - |
| 8 | 3.860 [3.825, 3.884] | 3.850 [3.826, 3.882] | 3.852 [3.815, 3.874] | 3.895 [3.868, 3.920] | -502.358 [-502.381, -502.323] | 0.020 | -502.365 [-502.383, -502.344] | 0.020 |
| 16 | 3.886 [3.837, 3.949] | 3.887 [3.832, 3.942] | 3.919 [3.864, 3.954] | 3.953 [3.887, 3.982] | -502.336 [-502.358, -502.325] | 0.020 | -502.334 [-502.364, -502.320] | 0.020 |
| 32 | 3.967 [3.906, 4.001] | 3.963 [3.899, 4.001] | - | - | -502.319 [-502.355, -502.307] | 0.020 | - | - |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.085 [0.074, 0.098] | 0.088 [0.075, 0.098] | 0.0000 [-0.0053, 0.0030] | 0.17 [0.13, 0.36] | 0.22 [0.10, 0.36] | 0.002 [-0.025, 0.011] | 0.28 [0.26, 0.33] | 0.22 [0.16, 0.26] | 0.075 [0.049, 0.098] | 0.0204 [0.0173, 0.0268] | 0.4 [0.4, 0.5] | 1.8 [1.6, 2.1] | 0.235 [0.192, 0.316] |
| 3 | 20 | 0.075 [0.062, 0.080] | - | - | 0.15 [0.07, 0.25] | - | - | 0.23 [0.19, 0.25] | - | - | - | 0.9 [0.8, 0.9] | - | - |
| 4 | 20 | 0.079 [0.037, 0.087] | 0.059 [0.048, 0.075] | 0.0041 [-0.0070, 0.0139] | 0.20 [0.04, 0.25] | 0.11 [0.04, 0.22] | 0.006 [-0.016, 0.064] | 0.20 [0.15, 0.24] | 0.12 [0.09, 0.16] | 0.074 [0.059, 0.090] | 0.0147 [0.0130, 0.0165] | 1.9 [1.6, 2.1] | 8.0 [7.6, 9.6] | 0.227 [0.157, 0.315] |
| 5 | 20 | 0.054 [0.039, 0.069] | - | - | 0.07 [0.03, 0.17] | - | - | 0.15 [0.11, 0.18] | - | - | - | 3.5 [2.8, 4.2] | - | - |
| 8 | 20 | 0.035 [0.030, 0.048] | 0.036 [0.029, 0.049] | 0.0009 [-0.0035, 0.0077] | 0.04 [0.03, 0.06] | 0.03 [0.03, 0.05] | 0.000 [-0.003, 0.016] | 0.11 [0.10, 0.12] | 0.06 [0.04, 0.08] | 0.038 [0.025, 0.060] | 0.0129 [0.0107, 0.0164] | 8.1 [6.7, 9.2] | 37.1 [30.2, 41.6] | 0.220 [0.193, 0.287] |
| 16 | 10 | 0.032 [0.023, 0.047] | 0.027 [0.025, 0.044] | 0.0002 [-0.0036, 0.0094] | 0.03 [0.02, 0.05] | 0.03 [0.02, 0.04] | 0.000 [-0.004, 0.012] | 0.07 [0.05, 0.13] | 0.04 [0.01, 0.08] | 0.042 [-0.004, 0.067] | 0.0084 [0.0067, 0.0136] | 37.5 [28.8, 56.8] | 192.7 [112.3, 321.7] | 0.251 [0.103, 0.316] |
| 32 | 10 | 0.026 [0.024, 0.031] | - | - | 0.03 [0.02, 0.03] | - | - | 0.03 [0.02, 0.05] | - | - | - | 220.9 [163.1, 267.6] | - | - |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 0.0000 | 0.6477 | 1 | 0.002 | 0.6477 | 1 | no |
| 3 | 0 | - | - | - | - | - | - | - |
| 4 | 20 | 0.0041 | 0.3683 | 1 | 0.006 | 0.3118 | 1 | no |
| 5 | 0 | - | - | - | - | - | - | - |
| 8 | 20 | 0.0009 | 0.3683 | 1 | 0.000 | 0.3884 | 1 | no |
| 16 | 10 | 0.0002 | 0.6953 | 1 | 0.000 | 0.7695 | 1 | no |
| 32 | 0 | - | - | - | - | - | - | - |
