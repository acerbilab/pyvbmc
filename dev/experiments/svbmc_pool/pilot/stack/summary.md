# S-VBMC stacking comparison: integrated against original 0.1.1

Generated 2026-09-14 04:15:09. Every cell stacks the same subset of filtered pool runs with both implementations, `n_samples=20`, `lr=0.1`, `max_steps=500`, `version="all-weights"`, subsets drawn with seed 0. Medians over the repetitions of a cell with a 10 000-resample bootstrap 95 % interval; `d` columns are the paired difference integrated minus original, so a negative value favours the integrated class. The headline ELBO is the capped value for the integrated class on noisy stacks and the raw estimate for the original, which is what each implementation reports. The two arms' entropies are different estimators — the integrated class re-evaluates with 100 fresh draws, the original reports the optimizer's own 20-draw estimate — and that difference enters `elbo_mc`. gsKL is the house convention (no `1/D` factor).

## rosenbrock_D2_noise3_svbmc

Single runs (`M = 1`, n = 3): MMTV 0.083 [0.030, 0.100], gsKL 0.04 [0.03, 0.14], evidence error 0.03 [0.02, 0.27].

All 5 cells of this condition: runtime ratio 0.307 [0.247, 0.360], max|dw| 0.0240 [0.0148, 0.0277].

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max|dw| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 5 | 0.048 [0.041, 0.051] | 0.041 [0.036, 0.047] | 0.0055 [-0.0006, 0.0106] | 0.01 [0.00, 0.01] | 0.01 [0.00, 0.01] | -0.002 [-0.004, 0.003] | 0.01 [0.00, 0.01] | 0.15 [0.13, 0.17] | -0.143 [-0.169, -0.118] | 0.0240 [0.0148, 0.0277] | 0.7 [0.6, 0.9] | 2.5 [1.8, 2.9] | 0.307 [0.247, 0.360] |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 3 | 5 | 0.0055 | 0.125 | 0.5 | -0.002 | 0.4375 | 1 | no |

## gmm_D2_svbmc

Single runs (`M = 1`, n = 3): MMTV 0.363 [0.261, 0.570], gsKL 10.96 [5.33, 36.18], evidence error 0.71 [0.69, 1.39].

All 5 cells of this condition: runtime ratio 0.514 [0.300, 0.591], max|dw| 0.0094 [0.0022, 0.0391].

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max|dw| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 5 | 0.205 [0.188, 0.219] | 0.200 [0.186, 0.227] | -0.0005 [-0.0290, 0.0226] | 0.91 [0.90, 0.93] | 0.92 [0.90, 0.97] | -0.007 [-0.057, 0.008] | 0.30 [0.27, 0.32] | 0.24 [0.24, 0.26] | 0.052 [0.033, 0.074] | 0.0094 [0.0022, 0.0391] | 0.5 [0.5, 0.6] | 1.1 [0.9, 1.7] | 0.514 [0.300, 0.591] |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 3 | 5 | -0.0005 | 1 | 1 | -0.007 | 0.4375 | 1 | no |

## gmm_D2_noise3_svbmc

Single runs (`M = 1`, n = 3): MMTV 0.317 [0.230, 0.429], gsKL 1.17 [0.39, 4.71], evidence error 0.46 [0.21, 0.94].

All 5 cells of this condition: runtime ratio 0.287 [0.229, 0.407], max|dw| 0.0303 [0.0162, 0.0379].

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max|dw| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 5 | 0.305 [0.300, 0.314] | 0.304 [0.292, 0.315] | 0.0086 [-0.0141, 0.0211] | 0.42 [0.36, 0.46] | 0.38 [0.34, 0.43] | 0.041 [-0.058, 0.088] | 0.88 [0.83, 0.88] | 0.21 [0.18, 0.26] | 0.672 [0.569, 0.676] | 0.0303 [0.0162, 0.0379] | 0.6 [0.6, 0.7] | 2.3 [1.6, 2.5] | 0.287 [0.229, 0.407] |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 3 | 5 | 0.0086 | 0.8125 | 1 | 0.041 | 0.4375 | 1 | no |

## ring_D2_noise3_svbmc

Single runs (`M = 1`, n = 3): MMTV 0.651 [0.649, 0.826], gsKL 154.21 [6.24, 1153.83], evidence error 1.91 [1.69, 2.28].

All 5 cells of this condition: runtime ratio 0.253 [0.202, 0.390], max|dw| 0.0215 [0.0170, 0.0439].

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max|dw| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 5 | 0.467 [0.465, 0.469] | 0.464 [0.463, 0.471] | 0.0027 [-0.0040, 0.0057] | 0.97 [0.91, 0.98] | 0.92 [0.91, 1.01] | 0.049 [-0.096, 0.073] | 1.19 [1.17, 1.20] | 1.04 [1.01, 1.04] | 0.148 [0.132, 0.163] | 0.0215 [0.0170, 0.0439] | 0.6 [0.5, 0.7] | 2.0 [1.8, 3.0] | 0.253 [0.202, 0.390] |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 3 | 5 | 0.0027 | 0.625 | 1 | 0.049 | 0.625 | 1 | no |

## multisensory_s1_D6_noise3_svbmc

Single runs (`M = 1`, n = 3): MMTV 0.343 [0.305, 0.382], gsKL 5.85 [5.78, 7.47], evidence error 0.72 [0.44, 1.33].

All 5 cells of this condition: runtime ratio 0.202 [0.123, 0.282], max|dw| 0.0259 [0.0193, 0.0316].

| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | gsKL orig | dgsKL | err int | err orig | derr | max|dw| | opt s int | opt s orig | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 5 | 0.201 [0.182, 0.218] | 0.172 [0.166, 0.183] | 0.0344 [0.0105, 0.0373] | 0.34 [0.31, 0.40] | 0.29 [0.27, 0.32] | 0.063 [0.019, 0.084] | 0.40 [0.35, 0.42] | 0.02 [0.00, 0.07] | 0.370 [0.335, 0.410] | 0.0259 [0.0193, 0.0316] | 0.6 [0.5, 0.7] | 2.8 [2.3, 5.4] | 0.202 [0.123, 0.282] |

Paired equivalence tests (exact signed-rank on the per-cell differences, one Holm family per metric over every condition and `M` at alpha 0.05); a flagged cell is one the family rejects with the integrated class the worse of the two.

| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median | gsKL p | gsKL Holm | flagged |
|---|---|---|---|---|---|---|---|---|
| 3 | 5 | 0.0344 | 0.0625 | 0.3125 | 0.063 | 0.0625 | 0.3125 | no |
