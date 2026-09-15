# S-VBMC stacking comparison: integrated against original 0.1.1

Generated 2026-09-15 10:52:41. Every cell stacks the same subset of filtered pool runs with both implementations, `n_samples=20`, `lr=0.1`, `max_steps=500`, `version="all-weights"`, subsets drawn with seed 0. Medians over the repetitions of a cell with a 10 000-resample bootstrap 95 % interval; `d` columns are the paired difference integrated minus original, so a negative value favours the integrated class. The headline ELBO is the capped value for the integrated class on noisy stacks and the raw estimate for the original, which is what each implementation reports. gsKL is the house convention (no `1/D` factor).

Every reported ELBO is scored by its bias against `elbo_mc = e_log_joint_mc + entropy_ref`, the Monte Carlo ELBO of the posterior that arm's fit produced. `e_log_joint_mc` is the mean of the target's noiseless log density over 10 000 of the stack's own draws; `entropy_ref` is the entropy of the stacked mixture at that arm's final weights, estimated here for both arms by one estimator (the integrated class's `stacked_entropy`, 200 draws per component in 8 batches, from a generator seeded by the cell, the same draws for both arms). The reference is arm-independent by construction: an arm's own reported entropy is estimated at the weights that arm selected, from its own draws (the optimizer's 20 per component for the original, a fresh 100 for the integrated class), so a reference built on it would move with the arm and credit whichever entropy estimate came out high. `kl_gap = ln Z - elbo_mc` is how far the stacked posterior itself is from the target, in evidence units; the `err` columns further below are the error against `ln Z`, descriptive only.

## multisensory_s1_D6_noise3_svbmc

Single runs (`M = 1`, n = 100): MMTV 0.279 [0.270, 0.288], gsKL 4.02 [3.40, 4.71], evidence error 0.31 [0.26, 0.38].

All 40 cells of this condition: runtime ratio -, max\|dw\| -.

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 20 | 0.419 [0.241, 0.550] | 0.841 [0.775, 1.070] | - | - | - | 0.745 [0.689, 0.817] | - | - |
| 5 | 20 | 0.432 [0.329, 0.495] | 1.042 [0.887, 1.126] | - | - | - | 0.615 [0.546, 0.702] | - | - |

Growth of the integrated headline's median bias from `M = 3` to `M = 5`: 0.014 nats (0.419 to 0.432), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 3 | 2.938 [2.832, 3.019] | 2.917 [2.812, 2.997] | - | - | -502.931 [-503.003, -502.874] | 0.023 | - | - |
| 5 | 3.002 [2.956, 3.188] | 3.010 [2.921, 3.183] | - | - | -502.801 [-502.887, -502.732] | 0.022 | - | - |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 20 | 0.233 [0.220, 0.241] | - | - | 2.83 [2.12, 3.13] | - | - | 0.25 [0.21, 0.48] | - | - | - | 0.8 [0.7, 0.9] | - | - |
| 5 | 20 | 0.192 [0.153, 0.212] | - | - | 1.39 [0.64, 2.16] | - | - | 0.25 [0.10, 0.36] | - | - | - | 2.8 [2.5, 3.2] | - | - |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 3 | 0 | - | - | - | - | - | - | - |
| 5 | 0 | - | - | - | - | - | - | - |

## multisensory_s1_D6_noise1.3_svbmc

Single runs (`M = 1`, n = 100): MMTV 0.231 [0.221, 0.240], gsKL 2.70 [2.39, 3.14], evidence error 0.37 [0.33, 0.41].

All 40 cells of this condition: runtime ratio -, max\|dw\| -.

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 20 | 0.159 [0.061, 0.206] | 0.440 [0.367, 0.478] | - | - | - | 0.489 [0.445, 0.528] | - | - |
| 5 | 20 | 0.188 [0.122, 0.222] | 0.549 [0.513, 0.624] | - | - | - | 0.522 [0.465, 0.547] | - | - |

Growth of the integrated headline's median bias from `M = 3` to `M = 5`: 0.028 nats (0.159 to 0.188), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 3 | 3.274 [3.154, 3.384] | 3.241 [3.170, 3.374] | - | - | -502.674 [-502.713, -502.636] | 0.022 | - | - |
| 5 | 3.188 [3.095, 3.269] | 3.190 [3.104, 3.262] | - | - | -502.707 [-502.733, -502.651] | 0.021 | - | - |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 20 | 0.182 [0.167, 0.194] | - | - | 1.53 [1.14, 2.04] | - | - | 0.34 [0.29, 0.40] | - | - | - | 0.8 [0.8, 0.9] | - | - |
| 5 | 20 | 0.190 [0.173, 0.211] | - | - | 1.92 [1.61, 2.37] | - | - | 0.32 [0.28, 0.36] | - | - | - | 2.6 [2.4, 3.0] | - | - |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 3 | 0 | - | - | - | - | - | - | - |
| 5 | 0 | - | - | - | - | - | - | - |

## rosenbrock_D2_noise3_svbmc

Single runs (`M = 1`, n = 100): MMTV 0.097 [0.076, 0.122], gsKL 0.15 [0.08, 0.22], evidence error 0.17 [0.14, 0.22].

All 40 cells of this condition: runtime ratio -, max\|dw\| -.

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 20 | 0.083 [0.012, 0.145] | 0.289 [0.233, 0.408] | - | - | - | 0.051 [0.029, 0.066] | - | - |
| 5 | 20 | 0.133 [0.082, 0.267] | 0.531 [0.488, 0.561] | - | - | - | 0.039 [0.030, 0.046] | - | - |

Growth of the integrated headline's median bias from `M = 3` to `M = 5`: 0.050 nats (0.083 to 0.133), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 3 | 2.544 [2.496, 2.579] | 2.540 [2.491, 2.588] | - | - | -2.310 [-2.326, -2.289] | 0.012 | - | - |
| 5 | 2.544 [2.504, 2.557] | 2.531 [2.502, 2.565] | - | - | -2.299 [-2.306, -2.290] | 0.012 | - | - |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 20 | 0.075 [0.056, 0.094] | - | - | 0.04 [0.02, 0.06] | - | - | 0.08 [0.05, 0.12] | - | - | - | 0.5 [0.5, 0.7] | - | - |
| 5 | 20 | 0.064 [0.049, 0.079] | - | - | 0.02 [0.01, 0.03] | - | - | 0.10 [0.04, 0.24] | - | - | - | 2.0 [1.8, 2.1] | - | - |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 3 | 0 | - | - | - | - | - | - | - |
| 5 | 0 | - | - | - | - | - | - | - |

## gmm_D2_noise3_svbmc

Single runs (`M = 1`, n = 100): MMTV 0.484 [0.449, 0.549], gsKL 22.47 [19.16, 31.16], evidence error 1.04 [0.93, 1.21].

All 40 cells of this condition: runtime ratio -, max\|dw\| -.

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 20 | 0.018 [-0.099, 0.116] | 0.338 [0.250, 0.400] | - | - | - | 0.546 [0.465, 0.705] | - | - |
| 5 | 20 | 0.015 [-0.136, 0.192] | 0.396 [0.263, 0.476] | - | - | - | 0.299 [0.242, 0.349] | - | - |

Growth of the integrated headline's median bias from `M = 3` to `M = 5`: -0.004 nats (0.018 to 0.015), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 3 | 4.064 [3.765, 4.218] | 4.061 [3.758, 4.205] | - | - | 2.450 [2.291, 2.530] | 0.015 | - | - |
| 5 | 4.341 [4.207, 4.465] | 4.344 [4.224, 4.465] | - | - | 2.697 [2.647, 2.754] | 0.014 | - | - |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 20 | 0.287 [0.243, 0.345] | - | - | 0.96 [0.34, 1.41] | - | - | 0.58 [0.46, 0.69] | - | - | - | 0.5 [0.4, 0.6] | - | - |
| 5 | 20 | 0.170 [0.138, 0.193] | - | - | 0.04 [0.02, 0.11] | - | - | 0.23 [0.13, 0.39] | - | - | - | 1.7 [1.4, 1.7] | - | - |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 3 | 0 | - | - | - | - | - | - | - |
| 5 | 0 | - | - | - | - | - | - | - |

## ring_D2_noise3_svbmc

Single runs (`M = 1`, n = 100): MMTV 0.627 [0.577, 0.673], gsKL 97.49 [29.42, 224.65], evidence error 1.38 [1.24, 1.53].

All 40 cells of this condition: runtime ratio -, max\|dw\| -.

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 20 | 0.091 [0.049, 0.227] | 0.368 [0.251, 0.418] | - | - | - | 0.787 [0.626, 0.872] | - | - |
| 5 | 20 | 0.224 [0.170, 0.310] | 0.423 [0.367, 0.500] | - | - | - | 0.483 [0.315, 0.589] | - | - |

Growth of the integrated headline's median bias from `M = 3` to `M = 5`: 0.133 nats (0.091 to 0.224), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 3 | 2.356 [2.274, 2.506] | 2.361 [2.269, 2.507] | - | - | 1.746 [1.661, 1.908] | 0.011 | - | - |
| 5 | 2.633 [2.520, 2.819] | 2.631 [2.521, 2.812] | - | - | 2.051 [1.945, 2.218] | 0.010 | - | - |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 20 | 0.327 [0.289, 0.369] | - | - | 0.40 [0.21, 0.85] | - | - | 0.67 [0.46, 0.74] | - | - | - | 0.6 [0.5, 0.8] | - | - |
| 5 | 20 | 0.245 [0.191, 0.283] | - | - | 0.18 [0.03, 0.37] | - | - | 0.27 [0.15, 0.37] | - | - | - | 2.0 [1.7, 2.3] | - | - |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 3 | 0 | - | - | - | - | - | - | - |
| 5 | 0 | - | - | - | - | - | - | - |

## student_D8_noise3_svbmc

Single runs (`M = 1`, n = 100): MMTV 0.143 [0.137, 0.146], gsKL 0.58 [0.52, 0.66], evidence error 0.96 [0.84, 1.11].

All 40 cells of this condition: runtime ratio -, max\|dw\| -.

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 20 | -0.973 [-1.559, -0.742] | -0.085 [-0.209, 0.165] | - | - | - | 0.414 [0.347, 0.488] | - | - |
| 5 | 20 | -1.257 [-1.577, -0.858] | 0.041 [-0.045, 0.217] | - | - | - | 0.350 [0.303, 0.408] | - | - |

Growth of the integrated headline's median bias from `M = 3` to `M = 5`: -0.283 nats (-0.973 to -1.257), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 3 | 13.320 [13.162, 13.644] | 13.312 [13.170, 13.689] | - | - | -20.017 [-20.092, -19.950] | 0.037 | - | - |
| 5 | 13.529 [13.328, 13.648] | 13.551 [13.279, 13.683] | - | - | -19.953 [-20.011, -19.906] | 0.034 | - | - |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 20 | 0.104 [0.094, 0.113] | - | - | 0.25 [0.18, 0.29] | - | - | 1.44 [1.26, 1.93] | - | - | - | 0.7 [0.7, 0.8] | - | - |
| 5 | 20 | 0.096 [0.089, 0.103] | - | - | 0.17 [0.15, 0.21] | - | - | 1.65 [1.22, 2.00] | - | - | - | 2.3 [2.1, 2.6] | - | - |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 3 | 0 | - | - | - | - | - | - | - |
| 5 | 0 | - | - | - | - | - | - | - |

## gmm_D2_svbmc

Single runs (`M = 1`, n = 50): MMTV 0.390 [0.365, 0.392], gsKL 14.92 [11.95, 15.40], evidence error 0.70 [0.70, 0.71].

All 40 cells of this condition: runtime ratio -, max\|dw\| -.

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 20 | -0.008 [-0.013, 0.006] | -0.008 [-0.013, 0.006] | - | - | - | 0.286 [0.024, 0.300] | - | - |
| 5 | 20 | 0.010 [0.001, 0.024] | 0.010 [0.001, 0.024] | - | - | - | 0.024 [0.015, 0.035] | - | - |

Growth of the integrated headline's median bias from `M = 3` to `M = 5`: 0.018 nats (-0.008 to 0.010), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 3 | 4.345 [4.339, 4.598] | 4.341 [4.331, 4.583] | - | - | 2.710 [2.695, 2.971] | 0.014 | - | - |
| 5 | 4.615 [4.602, 4.627] | 4.620 [4.612, 4.633] | - | - | 2.972 [2.960, 2.981] | 0.013 | - | - |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 20 | 0.183 [0.070, 0.196] | - | - | 0.43 [0.02, 0.68] | - | - | 0.29 [0.05, 0.30] | - | - | - | 0.3 [0.3, 0.4] | - | - |
| 5 | 20 | 0.032 [0.025, 0.051] | - | - | 0.00 [0.00, 0.01] | - | - | 0.01 [0.00, 0.02] | - | - | - | 1.0 [0.9, 1.1] | - | - |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 3 | 0 | - | - | - | - | - | - | - |
| 5 | 0 | - | - | - | - | - | - | - |

## multisensory_s1_D6_svbmc

Single runs (`M = 1`, n = 50): MMTV 0.118 [0.112, 0.126], gsKL 0.59 [0.42, 0.71], evidence error 0.37 [0.34, 0.42].

All 40 cells of this condition: runtime ratio -, max\|dw\| -.

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 20 | 0.066 [0.027, 0.079] | 0.066 [0.027, 0.079] | - | - | - | 0.245 [0.229, 0.284] | - | - |
| 5 | 20 | 0.069 [0.036, 0.083] | 0.069 [0.036, 0.083] | - | - | - | 0.231 [0.193, 0.237] | - | - |

Growth of the integrated headline's median bias from `M = 3` to `M = 5`: 0.003 nats (0.066 to 0.069), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 3 | 3.720 [3.692, 3.755] | 3.714 [3.697, 3.760] | - | - | -502.431 [-502.470, -502.414] | 0.024 | - | - |
| 5 | 3.817 [3.753, 3.833] | 3.814 [3.764, 3.838] | - | - | -502.417 [-502.423, -502.378] | 0.021 | - | - |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 20 | 0.082 [0.067, 0.090] | - | - | 0.15 [0.12, 0.28] | - | - | 0.20 [0.18, 0.24] | - | - | - | 0.5 [0.5, 0.6] | - | - |
| 5 | 20 | 0.061 [0.052, 0.068] | - | - | 0.10 [0.06, 0.16] | - | - | 0.15 [0.11, 0.20] | - | - | - | 1.7 [1.6, 1.9] | - | - |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 3 | 0 | - | - | - | - | - | - | - |
| 5 | 0 | - | - | - | - | - | - | - |
