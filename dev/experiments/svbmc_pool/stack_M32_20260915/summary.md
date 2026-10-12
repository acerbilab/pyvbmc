# S-VBMC stacking comparison: integrated against original 0.1.1

Generated 2026-09-16 01:53:36. Every cell stacks the same subset of filtered pool runs with both implementations, `n_samples=20`, `lr=0.1`, `max_steps=500`, `version="all-weights"`, subsets drawn with seed 0. Medians over the repetitions of a cell with a 10 000-resample bootstrap 95 % interval; `d` columns are the paired difference integrated minus original, so a negative value favours the integrated class. The headline ELBO is the capped value for the integrated class on noisy stacks and the raw estimate for the original, which is what each implementation reports. gsKL is the house convention (no `1/D` factor).

Every reported ELBO is scored by its bias against `elbo_mc = e_log_joint_mc + entropy_ref`, the Monte Carlo ELBO of the posterior that arm's fit produced. `e_log_joint_mc` is the mean of the target's noiseless log density over 10 000 of the stack's own draws; `entropy_ref` is the entropy of the stacked mixture at that arm's final weights, estimated here for both arms by one estimator (the integrated class's `stacked_entropy`, 200 draws per component in 8 batches, from a generator seeded by the cell, the same draws for both arms). The reference is arm-independent by construction: an arm's own reported entropy is estimated at the weights that arm selected, from its own draws (the optimizer's 20 per component for the original, a fresh 100 for the integrated class), so a reference built on it would move with the arm and credit whichever entropy estimate came out high. `kl_gap = ln Z - elbo_mc` is how far the stacked posterior itself is from the target, in evidence units; the `err` columns further below are the error against `ln Z`, descriptive only.

## multisensory_s1_D6_noise3_svbmc

Single runs (`M = 1`, n = 100): MMTV 0.279 [0.270, 0.288], gsKL 4.02 [3.40, 4.71], evidence error 0.31 [0.26, 0.38].

All 10 cells of this condition: runtime ratio -, max\|dw\| -.

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|
| 32 | 10 | 0.449 [0.402, 0.512] | 1.454 [1.352, 1.574] | - | - | - | 0.377 [0.328, 0.405] | - | - |

Growth of the integrated headline's median bias from `M = 32` to `M = 32`: 0.000 nats (0.449 to 0.449), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 32 | 3.333 [3.233, 3.481] | 3.329 [3.212, 3.474] | - | - | -502.563 [-502.591, -502.513] | 0.018 | - | - |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 10 | 0.141 [0.107, 0.155] | - | - | 0.49 [0.27, 0.78] | - | - | 0.07 [0.04, 0.21] | - | - | - | 174.0 [163.4, 195.2] | - | - |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 32 | 0 | - | - | - | - | - | - | - |

## multisensory_s1_D6_noise1.3_svbmc

Single runs (`M = 1`, n = 100): MMTV 0.231 [0.221, 0.240], gsKL 2.70 [2.39, 3.09], evidence error 0.37 [0.33, 0.41].

All 10 cells of this condition: runtime ratio -, max\|dw\| -.

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|
| 32 | 10 | 0.190 [0.175, 0.230] | 0.707 [0.664, 0.801] | - | - | - | 0.185 [0.164, 0.295] | - | - |

Growth of the integrated headline's median bias from `M = 32` to `M = 32`: 0.000 nats (0.190 to 0.190), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 32 | 3.621 [3.457, 3.644] | 3.611 [3.480, 3.641] | - | - | -502.371 [-502.481, -502.350] | 0.019 | - | - |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 10 | 0.098 [0.084, 0.131] | - | - | 0.23 [0.13, 0.60] | - | - | 0.03 [0.01, 0.11] | - | - | - | 187.4 [167.2, 212.1] | - | - |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 32 | 0 | - | - | - | - | - | - | - |

## rosenbrock_D2_noise3_svbmc

Single runs (`M = 1`, n = 100): MMTV 0.097 [0.076, 0.123], gsKL 0.15 [0.08, 0.22], evidence error 0.17 [0.14, 0.22].

All 10 cells of this condition: runtime ratio -, max\|dw\| -.

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|
| 32 | 10 | 0.056 [-0.007, 0.105] | 0.913 [0.760, 1.054] | - | - | - | 0.048 [0.030, 0.103] | - | - |

Growth of the integrated headline's median bias from `M = 32` to `M = 32`: 0.000 nats (0.056 to 0.056), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 32 | 2.500 [2.385, 2.593] | 2.509 [2.384, 2.590] | - | - | -2.308 [-2.363, -2.290] | 0.011 | - | - |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 10 | 0.064 [0.048, 0.093] | - | - | 0.02 [0.01, 0.03] | - | - | 0.10 [0.02, 0.13] | - | - | - | 138.3 [128.2, 153.6] | - | - |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 32 | 0 | - | - | - | - | - | - | - |

## gmm_D2_noise3_svbmc

Single runs (`M = 1`, n = 100): MMTV 0.484 [0.449, 0.549], gsKL 22.47 [19.16, 31.16], evidence error 1.04 [0.93, 1.20].

All 10 cells of this condition: runtime ratio -, max\|dw\| -.

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|
| 32 | 10 | 0.038 [-0.022, 0.091] | 0.918 [0.836, 0.948] | - | - | - | 0.155 [0.142, 0.168] | - | - |

Growth of the integrated headline's median bias from `M = 32` to `M = 32`: 0.000 nats (0.038 to 0.038), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 32 | 4.506 [4.466, 4.525] | 4.500 [4.464, 4.528] | - | - | 2.840 [2.828, 2.856] | 0.014 | - | - |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 10 | 0.109 [0.100, 0.122] | - | - | 0.02 [0.01, 0.03] | - | - | 0.13 [0.08, 0.17] | - | - | - | 140.7 [123.3, 156.3] | - | - |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 32 | 0 | - | - | - | - | - | - | - |

## ring_D2_noise3_svbmc

Single runs (`M = 1`, n = 100): MMTV 0.627 [0.578, 0.673], gsKL 97.49 [29.42, 224.65], evidence error 1.38 [1.24, 1.53].

All 10 cells of this condition: runtime ratio -, max\|dw\| -.

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|
| 32 | 10 | 0.129 [0.090, 0.144] | 0.842 [0.756, 0.872] | - | - | - | 0.098 [0.080, 0.113] | - | - |

Growth of the integrated headline's median bias from `M = 32` to `M = 32`: 0.000 nats (0.129 to 0.129), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 32 | 3.014 [2.991, 3.021] | 3.014 [2.992, 3.021] | - | - | 2.436 [2.421, 2.454] | 0.009 | - | - |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 10 | 0.096 [0.086, 0.107] | - | - | 0.01 [0.00, 0.01] | - | - | 0.03 [0.02, 0.06] | - | - | - | 139.7 [126.9, 159.5] | - | - |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 32 | 0 | - | - | - | - | - | - | - |

## student_D8_noise3_svbmc

Single runs (`M = 1`, n = 100): MMTV 0.143 [0.137, 0.146], gsKL 0.58 [0.52, 0.66], evidence error 0.96 [0.84, 1.11].

All 10 cells of this condition: runtime ratio -, max\|dw\| -.

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|
| 32 | 10 | -1.847 [-1.979, -1.609] | 0.384 [0.303, 0.446] | - | - | - | 0.189 [0.169, 0.211] | - | - |

Growth of the integrated headline's median bias from `M = 32` to `M = 32`: 0.000 nats (-1.847 to -1.847), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 32 | 13.191 [13.075, 13.403] | 13.207 [13.098, 13.409] | - | - | -19.793 [-19.814, -19.773] | 0.031 | - | - |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 10 | 0.070 [0.064, 0.076] | - | - | 0.06 [0.04, 0.08] | - | - | 2.03 [1.80, 2.13] | - | - | - | 170.4 [154.0, 186.7] | - | - |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 32 | 0 | - | - | - | - | - | - | - |

## gmm_D2_svbmc

Single runs (`M = 1`, n = 50): MMTV 0.390 [0.365, 0.392], gsKL 14.92 [11.93, 15.40], evidence error 0.70 [0.70, 0.71].

All 10 cells of this condition: runtime ratio -, max\|dw\| -.

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|
| 32 | 10 | 0.010 [0.008, 0.019] | 0.010 [0.008, 0.019] | - | - | - | 0.015 [0.008, 0.019] | - | - |

Growth of the integrated headline's median bias from `M = 32` to `M = 32`: 0.000 nats (0.010 to 0.010), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 32 | 4.631 [4.626, 4.638] | 4.632 [4.628, 4.639] | - | - | 2.980 [2.977, 2.987] | 0.012 | - | - |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 10 | 0.021 [0.014, 0.028] | - | - | 0.00 [0.00, 0.00] | - | - | 0.00 [0.00, 0.00] | - | - | - | 48.6 [45.2, 51.0] | - | - |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 32 | 0 | - | - | - | - | - | - | - |

## multisensory_s1_D6_svbmc

Single runs (`M = 1`, n = 50): MMTV 0.118 [0.112, 0.127], gsKL 0.59 [0.42, 0.70], evidence error 0.37 [0.34, 0.42].

All 10 cells of this condition: runtime ratio -, max\|dw\| -.

Bias of every reported ELBO against `elbo_mc` (criterion 3), and the KL gap of the stacked posterior. `d(head-est)` is the paired difference of the two arms' headline biases; the last column is the criterion's gate, `|median bias headline| <= |median bias estimated|`.

| M | cells | bias headline int | bias raw int | bias estimated orig | bias debiased_I orig | d(head-est) | KL gap int | KL gap orig | not worse |
|---|---|---|---|---|---|---|---|---|---|
| 32 | 10 | 0.144 [0.131, 0.162] | 0.144 [0.133, 0.162] | - | - | - | 0.153 [0.132, 0.169] | - | - |

Growth of the integrated headline's median bias from `M = 32` to `M = 32`: 0.000 nats (0.144 to 0.144), below the 0.5-nat bound: yes. The headline is the capped value on a noisy stack and the raw value where no cap applies.

The reference and the arms' own entropies. `H ref` is the arm-independent estimate at that arm's weights, `H` what the arm reports; `sd` columns are medians of the per-cell Monte Carlo standard deviation of `elbo_mc`.

| M | H ref int | H int | H ref orig | H orig | elbo_mc int | sd | elbo_mc orig | sd |
|---|---|---|---|---|---|---|---|---|
| 32 | 3.998 [3.976, 4.018] | 4.002 [3.980, 4.034] | - | - | -502.339 [-502.355, -502.322] | 0.020 | - | - |

Posterior quality, weights and runtime. The `err` columns are the error of each arm's headline against `ln Z`, descriptive only: they mix the estimator's bias with the stack's KL gap, so a small value can arise by cancellation and cannot rank estimators.

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max\|dw\| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 10 | 0.027 [0.025, 0.030] | - | - | 0.03 [0.02, 0.03] | - | - | 0.01 [0.01, 0.02] | - | - | - | 156.1 [129.5, 176.2] | - | - |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 32 | 0 | - | - | - | - | - | - | - |
