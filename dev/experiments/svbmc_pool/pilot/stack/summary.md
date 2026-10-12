# S-VBMC stacking comparison: integrated against original 0.1.1

Generated 2026-09-14 10:29:54. Every cell stacks the same subset of filtered pool runs with both implementations, `n_samples=20`, `lr=0.1`, `max_steps=500`, `version="all-weights"`, subsets drawn with seed 0. Medians over the repetitions of a cell with a 10 000-resample bootstrap 95 % interval; `d` columns are the paired difference integrated minus original, so a negative value favours the integrated class. The headline ELBO is the capped value for the integrated class on noisy stacks and the raw estimate for the original, which is what each implementation reports. gsKL is the house convention (no `1/D` factor).

Every reported ELBO is scored by its bias against `elbo_mc = e_log_joint_mc + entropy_ref`, the Monte Carlo ELBO of the posterior that arm's fit produced. `e_log_joint_mc` is the mean of the target's noiseless log density over 10 000 of the stack's own draws; `entropy_ref` is the entropy of the stacked mixture at that arm's final weights, estimated here for both arms by one estimator (the integrated class's `stacked_entropy`, 200 draws per component in 8 batches, from a generator seeded by the cell, the same draws for both arms). The reference is arm-independent by construction: an arm's own reported entropy is estimated at the weights that arm selected, from its own draws (the optimizer's 20 per component for the original, a fresh 100 for the integrated class), so a reference built on it would move with the arm and credit whichever entropy estimate came out high. `kl_gap = ln Z - elbo_mc` is how far the stacked posterior itself is from the target, in evidence units; the `err` columns further below are the error against `ln Z`, descriptive only.

## rosenbrock_D2_noise3_svbmc

Single runs (`M = 1`, n = 3): MMTV 0.083 [0.030, 0.100], gsKL 0.04 [0.03, 0.14], evidence error 0.03 [0.02, 0.27].

All 5 cells of this condition: runtime ratio 0.293 [0.253, 0.333], max\|dw\| 0.0240 [0.0148, 0.0277].

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 5 | 0.031 [0.025, 0.046] | 0.127 [0.099, 0.154] | 0.196 [0.161, 0.218] | 0.078 [0.041, 0.096] | -0.150 [-0.187, -0.129] | 0.026 [0.021, 0.054] | 0.039 [0.014, 0.060] | yes |

Growth of the integrated headline's median bias from `M = 3` to `M = 3`: 0.000 nats (0.031 to 0.031), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 3 | 2.537 [2.522, 2.549] | 2.547 [2.535, 2.551] | 2.523 [2.516, 2.542] | 2.574 [2.561, 2.591] | -2.286 [-2.314, -2.280] | 0.012 | -2.299 [-2.320, -2.274] | 0.011 |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 5 | 0.048 [0.041, 0.051] | 0.041 [0.036, 0.047] | 0.0055 [-0.0006, 0.0106] | 0.01 [0.00, 0.01] | 0.01 [0.00, 0.01] | -0.002 [-0.004, 0.003] | 0.01 [0.00, 0.01] | 0.15 [0.13, 0.17] | -0.143 [-0.169, -0.118] | 0.0240 [0.0148, 0.0277] | 0.7 [0.6, 0.8] | 2.3 [2.0, 2.8] | 0.293 [0.253, 0.333] |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 3 | 5 | 0.0055 | 0.125 | 0.5 | -0.002 | 0.4375 | 1 | no |

## gmm_D2_svbmc

Single runs (`M = 1`, n = 3): MMTV 0.363 [0.261, 0.570], gsKL 10.96 [5.33, 36.18], evidence error 0.71 [0.69, 1.39].

All 5 cells of this condition: runtime ratio 0.483 [0.300, 0.706], max\|dw\| 0.0094 [0.0022, 0.0391].

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 5 | 0.019 [0.012, 0.032] | 0.019 [0.012, 0.032] | 0.062 [0.044, 0.079] | 0.008 [-0.006, 0.039] | -0.045 [-0.067, -0.024] | 0.326 [0.305, 0.337] | 0.305 [0.300, 0.322] | yes |

Growth of the integrated headline's median bias from `M = 3` to `M = 3`: 0.000 nats (0.019 to 0.019), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 3 | 4.318 [4.307, 4.330] | 4.315 [4.298, 4.347] | 4.309 [4.300, 4.325] | 4.367 [4.355, 4.381] | 2.669 [2.659, 2.691] | 0.016 | 2.691 [2.673, 2.695] | 0.017 |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 5 | 0.205 [0.188, 0.219] | 0.200 [0.186, 0.227] | -0.0005 [-0.0290, 0.0226] | 0.91 [0.90, 0.93] | 0.92 [0.90, 0.97] | -0.007 [-0.057, 0.008] | 0.30 [0.27, 0.32] | 0.24 [0.24, 0.26] | 0.052 [0.033, 0.074] | 0.0094 [0.0022, 0.0391] | 0.5 [0.4, 0.6] | 1.0 [0.8, 1.6] | 0.483 [0.300, 0.706] |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 3 | 5 | -0.0005 | 1 | 1 | -0.007 | 0.4375 | 1 | no |

## gmm_D2_noise3_svbmc

Single runs (`M = 1`, n = 3): MMTV 0.317 [0.230, 0.429], gsKL 1.17 [0.39, 4.71], evidence error 0.46 [0.21, 0.94].

All 5 cells of this condition: runtime ratio 0.256 [0.216, 0.405], max\|dw\| 0.0303 [0.0162, 0.0379].

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 5 | -0.295 [-0.327, -0.272] | 0.688 [0.680, 0.703] | 0.785 [0.731, 0.809] | -0.237 [-0.269, -0.216] | -1.075 [-1.112, -1.022] | 0.561 [0.549, 0.581] | 0.562 [0.528, 0.586] | yes |

Growth of the integrated headline's median bias from `M = 3` to `M = 3`: 0.000 nats (-0.295 to -0.295), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 3 | 4.046 [4.037, 4.071] | 4.038 [4.035, 4.081] | 4.050 [4.016, 4.081] | 4.130 [4.084, 4.148] | 2.435 [2.415, 2.447] | 0.014 | 2.434 [2.410, 2.468] | 0.015 |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 5 | 0.305 [0.300, 0.314] | 0.304 [0.292, 0.315] | 0.0086 [-0.0141, 0.0211] | 0.42 [0.36, 0.46] | 0.38 [0.34, 0.43] | 0.041 [-0.058, 0.088] | 0.88 [0.83, 0.88] | 0.21 [0.18, 0.26] | 0.672 [0.569, 0.676] | 0.0303 [0.0162, 0.0379] | 0.5 [0.5, 0.7] | 2.2 [1.6, 2.3] | 0.256 [0.216, 0.405] |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 3 | 5 | 0.0086 | 0.8125 | 1 | 0.041 | 0.4375 | 1 | no |

## ring_D2_noise3_svbmc

Single runs (`M = 1`, n = 3): MMTV 0.651 [0.649, 0.826], gsKL 154.21 [6.24, 1153.83], evidence error 1.91 [1.69, 2.28].

All 5 cells of this condition: runtime ratio 0.303 [0.216, 0.352], max\|dw\| 0.0215 [0.0170, 0.0439].

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 5 | 0.003 [-0.016, 0.023] | 0.118 [0.091, 0.134] | 0.170 [0.133, 0.241] | 0.051 [0.011, 0.070] | -0.158 [-0.218, -0.130] | 1.191 [1.183, 1.196] | 1.213 [1.173, 1.252] | yes |

Growth of the integrated headline's median bias from `M = 3` to `M = 3`: 0.000 nats (0.003 to 0.003), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 3 | 1.975 [1.964, 1.977] | 1.970 [1.957, 1.983] | 1.974 [1.934, 1.988] | 1.994 [1.974, 2.000] | 1.343 [1.337, 1.351] | 0.011 | 1.321 [1.281, 1.361] | 0.011 |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 5 | 0.467 [0.465, 0.469] | 0.464 [0.463, 0.471] | 0.0027 [-0.0040, 0.0057] | 0.97 [0.91, 0.98] | 0.92 [0.91, 1.01] | 0.049 [-0.096, 0.073] | 1.19 [1.17, 1.20] | 1.04 [1.01, 1.04] | 0.148 [0.132, 0.163] | 0.0215 [0.0170, 0.0439] | 0.6 [0.5, 0.7] | 2.1 [1.8, 2.8] | 0.303 [0.216, 0.352] |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 3 | 5 | 0.0027 | 0.625 | 1 | 0.049 | 0.625 | 1 | no |

## multisensory_s1_D6_noise3_svbmc

Single runs (`M = 1`, n = 3): MMTV 0.343 [0.305, 0.382], gsKL 5.85 [5.78, 7.47], evidence error 0.72 [0.44, 1.33].

All 5 cells of this condition: runtime ratio 0.209 [0.102, 0.265], max\|dw\| 0.0259 [0.0193, 0.0316].

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 5 | 0.265 [0.246, 0.294] | 0.549 [0.527, 0.576] | 0.625 [0.624, 0.759] | 0.408 [0.350, 0.468] | -0.374 [-0.513, -0.330] | 0.668 [0.621, 0.687] | 0.646 [0.619, 0.691] | yes |

Growth of the integrated headline's median bias from `M = 3` to `M = 3`: 0.000 nats (0.265 to 0.265), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 3 | 3.237 [3.204, 3.256] | 3.222 [3.202, 3.280] | 3.304 [3.267, 3.322] | 3.368 [3.363, 3.392] | -502.854 [-502.873, -502.807] | 0.022 | -502.832 [-502.877, -502.805] | 0.022 |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 5 | 0.201 [0.182, 0.218] | 0.172 [0.166, 0.183] | 0.0344 [0.0105, 0.0373] | 0.34 [0.31, 0.40] | 0.29 [0.27, 0.32] | 0.063 [0.019, 0.084] | 0.40 [0.35, 0.42] | 0.02 [0.00, 0.07] | 0.370 [0.335, 0.410] | 0.0259 [0.0193, 0.0316] | 0.6 [0.5, 0.6] | 2.7 [2.2, 5.6] | 0.209 [0.102, 0.265] |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 3 | 5 | 0.0344 | 0.0625 | 0.3125 | 0.063 | 0.0625 | 0.3125 | no |
