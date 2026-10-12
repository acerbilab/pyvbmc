# Golden replay 2026-09-06 20:04

Code `7314a6a`; baseline `item7_20260906`; threads 1; 2.8 min.

| config | seed | verdict | identical iterations / iters ref → new | live points identical / ref → new | ΔLML ref → new (fence) | gsKL ref → new (fence) | MMTV ref → new (fence) | evals ref → new [pop] | wall min ref → new |
|---|---|---|---|---|---|---|---|---|---|
| normal_D5 | 0 | identical; initial design identical (X_init in both traces) | 13 / 13 → 13 | 69 / 69 → 69 | 0.000307 → 0.000307 (0.0305) | 0.000105 → 0.000105 (0.000405) | 0.00783 → 0.00783 (0.0125) | 70 → 70 [70, 80] | 0.35 → 0.34 |
| banana_D2 | 0 | identical; initial design identical (X_init in both traces) | 15 / 15 → 15 | 74 / 74 → 74 | 0.043 → 0.043 (0.135) | 0.15 → 0.15 (0.457) | 0.0392 → 0.0392 (0.0938) | 80 → 80 [70, 115] | 0.23 → 0.22 |
| halfnormal_D2 | 0 | identical; initial design identical (X_init in both traces) | 12 / 12 → 12 | 65 / 65 → 65 | 0.009 → 0.009 (0.025) | 0.000318 → 0.000318 (0.000852) | 0.0213 → 0.0213 (0.0308) | 65 → 65 [65, 75] | 0.17 → 0.18 |
| cigar_D4 | 0 | identical; initial design identical (X_init in both traces) | 26 / 26 → 26 | 114 / 114 → 114 | 0.00174 → 0.00174 (0.0422) | 0.000422 → 0.000422 (0.00827) | 0.0088 → 0.0088 (0.0498) | 130 → 130 [120, 155] | 0.78 → 0.79 |
| rosenbrock_D2_noise1 | 0 | identical; initial design identical (X_init in both traces) | 22 / 22 → 22 | 108 / 108 → 108 | 0.0267 → 0.0267 (0.476) | 0.0635 → 0.0635 (0.136) | 0.0455 → 0.0455 (0.1) | 115 → 115 [105, 150] | 1.2 → 1.3 |

0 flagged of 5. `identical` = same live points and ELBO path. `parted at iteration i` (0-based) = iterations 0..i−1 are bit-identical and iteration i is the first to differ (expected for an arithmetic-preserving change once a CMA-ES ranking flips); the live-point count is a lower bound on the evaluation horizon because warm-up trimming removes rows. Flags: a run failed, the initial design differs (exact where both traces store it, else a generator-drawn design point of the new run found live in the reference; where none is live, e.g. cigar, the trace cannot certify the design and no flag is raised), a final is not finite, or a parted run's ΔLML/gsKL/MMTV exceeds the population's Q3 + 3 IQR fence.
