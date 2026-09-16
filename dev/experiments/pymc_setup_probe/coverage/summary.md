# Filtering and initial coverage: focused comparison

Logger seeding and single-point initialization in every arm. Five seeds per model; setup charged in full. Lower distances are better.
Medians summarize returned fits only; numerical failures are retained and listed separately. Every arm has the same budget cap.
Filter: `log_density`. The density rule retains points within 10D nats of the best, with a D+1 minimum. Initial uniform counts are nine (full) and zero (shortened) on both models.

| Model | Arm | Completed | Median marginal distance [range] | Median absolute ELBO error | Reached stability | Median seconds |
|---|---|---:|---:|---:|---:|---:|
| schools_noncentered | logger | 5/5 | 0.1302 [0.1213, 0.2367] | 0.5765 | 0/5 | 165.2 |
| schools_noncentered | filtered_full | 3/5 | 0.1385 [0.09856, 0.1833] | 0.7013 | 0/5 | 77 |
| schools_noncentered | filtered_short | 5/5 | 0.1919 [0.1449, 0.2689] | 0.7722 | 0/5 | 79.87 |
| ill_conditioned | logger | 5/5 | 0.3758 [0.1711, 0.3908] | 2.945 | 0/5 | 29.08 |
| ill_conditioned | filtered_full | 5/5 | 0.3972 [0.377, 0.4205] | 3.105 | 0/5 | 32.51 |
| ill_conditioned | filtered_short | 5/5 | 0.4129 [0.396, 0.5462] | 3.558 | 0/5 | 28.51 |

Negative paired deltas favor filtering or full coverage, respectively.

| Model | Seed | Filtering delta | Full-coverage delta |
|---|---:|---:|---:|
| schools_noncentered | 0 | 0.00842002 | -0.0534214 |
| schools_noncentered | 1 | 0.0530378 | 0.0383948 |
| schools_noncentered | 4 | -0.0374821 | -0.128024 |
| ill_conditioned | 0 | 0.00119584 | -0.0359217 |
| ill_conditioned | 1 | 0.249431 | 0.0245292 |
| ill_conditioned | 2 | -0.0100501 | -0.165413 |
| ill_conditioned | 3 | 0.0234123 | -0.0137962 |
| ill_conditioned | 4 | 0.0161846 | -0.0428673 |

## Individual cases

| Model | Seed | Arm | ELBO error | Marginal distance | Gaussian sym. KL | First stable iteration |
|---|---:|---|---:|---:|---:|---:|
| ill_conditioned | 0 | logger | -2.94516 | 0.375764 | 3.05313 | None |
| ill_conditioned | 0 | filtered_full | -2.88968 | 0.37696 | 3.01533 | None |
| ill_conditioned | 0 | filtered_short | -3.55779 | 0.412882 | 3.50726 | None |
| ill_conditioned | 1 | logger | -0.893542 | 0.171077 | 0.756383 | None |
| ill_conditioned | 1 | filtered_full | -3.51413 | 0.420508 | 4.49657 | None |
| ill_conditioned | 1 | filtered_short | -3.12091 | 0.395979 | 3.43102 | None |
| ill_conditioned | 2 | logger | -3.14144 | 0.390791 | 3.43177 | None |
| ill_conditioned | 2 | filtered_full | -2.94708 | 0.38074 | 3.10676 | None |
| ill_conditioned | 2 | filtered_short | 3.9985 | 0.546153 | 313.579 | None |
| ill_conditioned | 3 | logger | -2.89935 | 0.373805 | 3.01337 | None |
| ill_conditioned | 3 | filtered_full | -3.10483 | 0.397217 | 3.42434 | None |
| ill_conditioned | 3 | filtered_short | -3.2193 | 0.411013 | 3.79243 | None |
| ill_conditioned | 4 | logger | -3.00954 | 0.382492 | 3.24038 | None |
| ill_conditioned | 4 | filtered_full | -3.11638 | 0.398676 | 3.54852 | None |
| ill_conditioned | 4 | filtered_short | 3.90908 | 0.441544 | 98.2321 | None |
| schools_noncentered | 0 | logger | -0.576479 | 0.130031 | 1.09359 | None |
| schools_noncentered | 0 | filtered_full | -0.714735 | 0.138451 | 0.591002 | None |
| schools_noncentered | 0 | filtered_short | -0.694922 | 0.191872 | 5.64498 | None |
| schools_noncentered | 1 | logger | -0.59167 | 0.130233 | 0.635815 | None |
| schools_noncentered | 1 | filtered_full | -0.701286 | 0.183271 | 0.547323 | None |
| schools_noncentered | 1 | filtered_short | 0.772168 | 0.144876 | 0.582418 | None |
| schools_noncentered | 2 | logger | -0.873463 | 0.236683 | 4.19061 | None |
| schools_noncentered | 2 | filtered_full | FAILED | — | — | — |
| schools_noncentered | 2 | filtered_short | -0.644513 | 0.165161 | 0.641438 | None |
| schools_noncentered | 3 | logger | -0.42206 | 0.12126 | 0.563006 | None |
| schools_noncentered | 3 | filtered_full | FAILED | — | — | — |
| schools_noncentered | 3 | filtered_short | -1.18753 | 0.268855 | 1.72078 | None |
| schools_noncentered | 4 | logger | -0.183756 | 0.136045 | 0.499519 | None |
| schools_noncentered | 4 | filtered_full | -0.426793 | 0.0985632 | 0.288516 | None |
| schools_noncentered | 4 | filtered_short | 1.15398 | 0.226587 | 1.00765 | None |

## Numerical failures

- schools_noncentered, seed 2, filtered_full: LinAlgError: Singular matrix for L Cholesky decomposition; spent 123/150 units.
- schools_noncentered, seed 3, filtered_full: LinAlgError: Singular matrix for L Cholesky decomposition; spent 143/150 units.
