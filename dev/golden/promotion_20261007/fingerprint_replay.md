# Golden replay 2026-10-04 17:13

Code `dce4e18c`; baseline `no-traces`; threads 1; historical default budgets; 43.1 min.

| config | seed | verdict | identical iterations / iters ref → new | live points identical / ref → new | ΔLML ref → new (fence) | gsKL ref → new (fence) | MMTV ref → new (fence) | evals ref → new [pop] | wall min ref → new |
|---|---|---|---|---|---|---|---|---|---|
| normal_D5 | 0 | finals only | - | - | 0.000136 → 0.00424 (0.025) | 0.000314 → 4.82e-05 (0.000799) | 0.00764 → 0.00705 (0.0129) | 75 → 75 [65, 80] | 0.69 → 0.31 |
| corr_D5 | 0 | finals only; OUTSIDE envelope: elbo_err, gskl, mmtv | - | - | 0.00307 → 0.0483 (0.0326) | 0.00091 → 0.0123 (0.00502) | 0.00765 → 0.0178 (0.0128) | 100 → 105 [90, 135] | 0.93 → 0.4 |
| halfnormal_D2 | 0 | finals only | - | - | 0.00907 → 0.00907 (0.0173) | 0.000137 → 0.000119 (0.00107) | 0.0186 → 0.0201 (0.0303) | 65 → 65 [65, 75] | 0.37 → 0.14 |
| rosenbrock_D2 | 0 | finals only | - | - | 0.021 → 0.0244 (0.0667) | 0.0172 → 0.0149 (0.0892) | 0.0204 → 0.0208 (0.0636) | 75 → 75 [70, 120] | 0.46 → 0.17 |
| banana_D2 | 0 | finals only | - | - | 0.0456 → 0.0361 (0.117) | 0.169 → 0.121 (0.391) | 0.0461 → 0.0335 (0.0856) | 80 → 80 [75, 115] | 0.47 → 0.18 |
| banana_D6 | 0 | finals only | - | - | 0.0885 → 0.0695 (0.196) | 0.272 → 0.247 (0.704) | 0.0258 → 0.0256 (0.0508) | 90 → 90 [85, 155] | 0.76 → 0.35 |
| banana_D10 | 0 | finals only | - | - | 0.194 → 0.0647 (0.208) | 0.751 → 0.228 (0.818) | 0.0323 → 0.016 (0.0367) | 110 → 120 [110, 165] | 1.3 → 0.79 |
| cigar_D4 | 0 | finals only | - | - | 0.0059 → 0.00401 (0.0242) | 0.000126 → 0.000447 (0.00641) | 0.00765 → 0.00987 (0.042) | 130 → 125 [115, 155] | 1.3 → 0.81 |
| lumpy_D4 | 0 | finals only | - | - | 0.0685 → 0.0775 (0.186) | 0.024 → 0.0222 (0.0463) | 0.0285 → 0.0265 (0.0433) | 90 → 90 [70, 110] | 0.75 → 0.28 |
| lumpy_D10 | 0 | finals only | - | - | 0.462 → 0.817 (1.21) | 0.333 → 0.589 (1.11) | 0.0708 → 0.101 (0.17) | 325 → 310 [125, 350] | 8 → 2.8 |
| student_D4 | 0 | finals only | - | - | 0.00922 → 0.000404 (0.103) | 0.0158 → 0.0115 (0.161) | 0.0296 → 0.0333 (0.0757) | 135 → 130 [80, 160] | 1.2 → 0.48 |
| logreg_D5 | 0 | finals only | - | - | 0.0911 → 0.0855 (0.254) | 0.015 → 0.0197 (0.087) | 0.0324 → 0.0388 (0.0779) | 130 → 105 [110, 185] | 1.3 → 0.45 |
| cigar_D8 | 0 | finals only | - | - | 0.0137 → 0.00266 (0.0511) | 0.000764 → 0.00583 (0.0139) | 0.00503 → 0.0259 (0.0351) | 160 → 160 [140, 195] | 1.9 → 0.9 |
| student_D8 | 0 | finals only | - | - | 0.0868 → 0.00278 (0.475) | 0.298 → 0.129 (0.922) | 0.0808 → 0.0592 (0.115) | 290 → 160 [110, 335] | 5.9 → 1.2 |
| multisensory_s1_D6 | 0 | finals only | - | - | 0.485 → 0.417 (0.825) | 1.11 → 0.548 (2.04) | 0.145 → 0.114 (0.239) | 175 → 175 [130, 225] | 1.9 → 1.1 |
| cigar_D15_exhaust | 0 | finals only | - | - | 0.0115 → 0.00317 (0.0656) | 0.00183 → 0.00214 (0.00877) | 0.00948 → 0.00654 (0.0134) | 750 → 750 [750, 750] | 22 → 9.1 |
| rosenbrock_D2_noise1_production | 0 | finals only | - | - | 0.287 → 0.0712 (0.429) | 0.301 → 0.084 (0.163) | 0.17 → 0.0533 (0.124) | 300 → 120 [105, 300] | 6.5 → 0.92 |
| rosenbrock_D2_noise3_production | 0 | finals only | - | - | 0.204 → 0.133 (1.22) | 0.0485 → 0.0775 (2.82) | 0.0688 → 0.0898 (0.556) | 160 → 170 [130, 245] | 2.6 → 1.4 |
| logreg_D5_noise3_production | 0 | finals only | - | - | 0.296 → 0.0751 (1.48) | 0.389 → 0.244 (1.01) | 0.179 → 0.0929 (0.314) | 240 → 260 [190, 320] | 5.4 → 2.7 |
| student_D8_noise3_production | 0 | finals only | - | - | 1.12 → 0.914 (3.37) | 0.645 → 0.444 (1.59) | 0.154 → 0.14 (0.257) | 340 → 315 [235, 400] | 9.4 → 4 |
| lumpy_D10_noise3_production | 0 | finals only | - | - | 0.54 → 0.994 (1.78) | 0.785 → 1.02 (1.28) | 0.116 → 0.132 (0.179) | 395 → 430 [230, 525] | 11 → 8 |
| timing_D5_noise2.2_production | 0 | finals only | - | - | 0.0916 → 0.0718 (1.03) | 0.275 → 0.442 (1.43) | 0.102 → 0.15 (0.284) | 210 → 195 [170, 455] | 4.4 → 2.1 |
| multisensory_s1_D6_noise1.3_production | 0 | finals only | - | - | 0.278 → 0.386 (1.47) | 4.65 → 4.27 (8.56) | 0.231 → 0.254 (0.375) | 170 → 205 [165, 280] | 3.3 → 2.3 |
| multisensory_s2_D6_noise1.3_production | 0 | finals only | - | - | 0.366 → 0.555 (1.41) | 0.533 → 1.71 (2.23) | 0.133 → 0.191 (0.294) | 205 → 185 [170, 255] | 4.9 → 2.1 |

1 flagged of 24. `identical stored loop and final` = exact shape and value equality for every stored non-timer NPZ array and all 11 semantic final sidecar fields. `same loop, changed final` separates a returned posterior/result change from the main loop. A historical trace without the returned posterior's transformer is explicitly not certifiable for that state. `parted at iteration i` (0-based) = iterations 0..i−1 are bit-identical and iteration i is the first ELBO difference; the JSON report lists every differing loop and final array. Toleranced horizons are diagnostics only. The live-point count is a lower bound on the evaluation horizon because warm-up trimming removes rows. Flags: a run failed, the initial design differs (exact where both traces store it, else a generator-drawn design point of the new run found live in the reference; where none is live, e.g. cigar, the trace cannot certify the design and no flag is raised), a sidecar count contradicts its NPZ, a final is not finite, or a changed loop/final's ΔLML/gsKL/MMTV exceeds the population's Q3 + 3 IQR fence.
