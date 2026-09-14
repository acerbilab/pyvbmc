# Honest (cross-run) expected log joint of stacked posteriors

Generated 2026-09-14 14:02:27 from the cells of `results.json` (arm `integrated`), 100 draws per component, seed 0. Every estimate of the stack's expected log joint is scored by its bias against the cell's `e_log_joint_mc` (the mean of the target's noiseless log density over the arm's own draws of the stacked posterior); adding the cell's `entropy_ref` to every estimate gives the ELBOs and leaves every bias unchanged. `raw` is `sum w I_corr`, `capped_I` and `capped_E` the class's component-median and run-median caps of it; an honest value replaces every component's expected log joint by the covering other runs' GP estimates under a coverage rule (`ratioR`: a run's mean predictive SD on the component at most `R` times the component's own run's or below 0.1 nats, and at most 2.236 nats; `cap_only`: the cap alone; `none`: every other run) combined by the `median`, `precision` weighting or the `mean`, the component's own value when no other run covers it. The headline is `ratio2` with the `median`. `truth_strat` is the bias of the stratified Monte Carlo truth (`sum w E_true`) against `e_log_joint_mc`, a consistency check between the two Monte Carlo routes, and `pooled` the headline rule with the component's own run included (not honest). Medians over the cells of a condition and `M` with a 10,000-resample bootstrap 95 % interval. `covered` is the stacked weight on components at least one other run covers. `mc_se` is the Monte Carlo standard error of the honest value from the draws (the covering runs average the same draws), `gp_sd` its GP-side standard deviation under independence across components and under full correlation within each evaluating run. `within 2 SE` is the fraction of cells whose headline bias is within twice the combined Monte Carlo standard error of the estimate and of the reference.

## Headline across conditions

| Condition | M | cells | raw | capped_I | honest (ratio2, median) | honest (ratio2, precision) | covered | mc_se | within 2 SE | median abs bias raw / capped_I / honest | KL gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| gmm_D2_noise3_svbmc | 3 | 5 | 0.688 [0.674, 0.702] | -0.296 [-0.327, -0.281] | -0.822 [-0.873, -0.797] | -0.801 [-0.856, -0.775] | 1.000 | 0.015 | 0.00 | 0.688 / 0.296 / 0.822 | 0.56 [0.55, 0.58] |
| gmm_D2_svbmc | 3 | 5 | 0.026 [0.015, 0.029] | -0.023 [-0.032, -0.020] | 0.026 [0.008, 0.035] | 0.025 [0.007, 0.035] | 0.296 | 0.006 | 0.40 | 0.026 / 0.023 / 0.026 | 0.33 [0.30, 0.34] |
| multisensory_s1_D6_noise3_svbmc | 3 | 5 | 0.558 [0.526, 0.581] | 0.267 [0.252, 0.299] | 0.105 [0.062, 0.182] | 0.106 [0.063, 0.183] | 0.779 | 0.023 | 0.00 | 0.558 / 0.267 / 0.105 | 0.67 [0.62, 0.69] |
| ring_D2_noise3_svbmc | 3 | 5 | 0.109 [0.103, 0.134] | 0.003 [0.002, 0.012] | 0.048 [0.028, 0.079] | 0.048 [0.028, 0.079] | 0.304 | 0.007 | 0.00 | 0.109 / 0.003 / 0.048 | 1.19 [1.18, 1.20] |
| rosenbrock_D2_noise3_svbmc | 3 | 5 | 0.123 [0.099, 0.140] | 0.027 [0.018, 0.033] | -0.227 [-0.238, -0.209] | -0.232 [-0.244, -0.215] | 1.000 | 0.011 | 0.00 | 0.123 / 0.027 / 0.227 | 0.03 [0.02, 0.05] |

## gmm_D2_noise3_svbmc (noisy)

### M = 3 (5 cells)

| Estimator | bias | bias with the arm's entropy | covered | mc_se | gp_sd (indep / corr) |
|---|---:|---:|---:|---:|---:|
| raw | 0.688 [0.674, 0.702] | 0.688 [0.680, 0.703] | - | - | - |
| capped_I | -0.296 [-0.327, -0.281] | -0.295 [-0.327, -0.272] | - | - | - |
| capped_E | -0.002 [-0.034, 0.012] | -0.001 [-0.033, 0.022] | - | - | - |
| honest ratio1.5 median | -0.776 [-0.829, -0.750] | -0.772 [-0.829, -0.758] | 0.999 | 0.015 | 0.114 / 0.460 |
| honest ratio1.5 precision | -0.758 [-0.816, -0.733] | -0.754 [-0.815, -0.741] | 0.999 | 0.015 | 0.114 / 0.466 |
| honest ratio1.5 mean | -0.776 [-0.829, -0.750] | -0.772 [-0.829, -0.758] | 0.999 | 0.015 | 0.114 / 0.460 |
| honest ratio2 median | -0.822 [-0.873, -0.797] | -0.818 [-0.873, -0.805] | 1.000 | 0.015 | 0.111 / 0.463 |
| honest ratio2 precision | -0.801 [-0.856, -0.775] | -0.797 [-0.856, -0.783] | 1.000 | 0.015 | 0.111 / 0.467 |
| honest ratio2 mean | -0.822 [-0.873, -0.797] | -0.818 [-0.873, -0.805] | 1.000 | 0.015 | 0.111 / 0.463 |
| honest ratio3 median | -0.822 [-0.873, -0.797] | -0.818 [-0.873, -0.805] | 1.000 | 0.015 | 0.111 / 0.463 |
| honest ratio3 precision | -0.801 [-0.856, -0.775] | -0.797 [-0.856, -0.783] | 1.000 | 0.015 | 0.111 / 0.467 |
| honest ratio3 mean | -0.822 [-0.873, -0.797] | -0.818 [-0.873, -0.805] | 1.000 | 0.015 | 0.111 / 0.463 |
| honest ratio5 median | -0.822 [-0.873, -0.797] | -0.818 [-0.873, -0.805] | 1.000 | 0.015 | 0.111 / 0.463 |
| honest ratio5 precision | -0.801 [-0.856, -0.775] | -0.797 [-0.856, -0.783] | 1.000 | 0.015 | 0.111 / 0.467 |
| honest ratio5 mean | -0.822 [-0.873, -0.797] | -0.818 [-0.873, -0.805] | 1.000 | 0.015 | 0.111 / 0.463 |
| honest cap_only median | -0.822 [-0.873, -0.797] | -0.818 [-0.873, -0.805] | 1.000 | 0.015 | 0.111 / 0.463 |
| honest cap_only precision | -0.801 [-0.856, -0.775] | -0.797 [-0.856, -0.783] | 1.000 | 0.015 | 0.111 / 0.467 |
| honest cap_only mean | -0.822 [-0.873, -0.797] | -0.818 [-0.873, -0.805] | 1.000 | 0.015 | 0.111 / 0.463 |
| honest none median | -7.822 [-8.057, -7.621] | -7.826 [-8.065, -7.621] | 1.000 | 0.021 | 0.591 / 2.249 |
| honest none precision | -0.895 [-0.948, -0.871] | -0.891 [-0.948, -0.879] | 1.000 | 0.015 | 0.111 / 0.473 |
| honest none mean | -7.822 [-8.057, -7.621] | -7.826 [-8.065, -7.621] | 1.000 | 0.021 | 0.591 / 2.249 |
| pooled (ratio2, own included) | -0.205 [-0.246, -0.162] | - | - | - | - |
| truth_strat | 0.005 [-0.032, 0.033] | - | - | - | - |

Reference `e_log_joint_mc_sd` median 0.010; the headline bias is within 2 SE in 0% of the cells; median absolute bias raw 0.688, capped_I 0.296, capped_E 0.010, honest 0.822 (the honest estimate is **worse** than raw and **worse** than capped_I). Calibration of the self-reported SDs (weighted over components): own z median 1.79 [1.66, 1.85], abs z median 1.80 [1.67, 1.85], abs z > 2 fraction 0.44 [0.42, 0.45]; cross z median -0.94 [-1.09, -0.90], abs z median 1.06 [0.99, 1.15], abs z > 2 fraction 0.16 [0.09, 0.18]. Effective training points on a component (weighted median): own run 12.1 [12.0, 12.2], covering other runs 8.0 [7.6, 8.1]. Checks: own-run max abs z 3.45 (weighted offset abs z 2.25), Jacobian max abs z 0.00 (offset abs z 0.00), raw consistency 2.2e-16, flagged cells 0.

Heaviest component of every run, repetition 0:

| Run, heaviest component (weight) | truth | own I_corr (error; BQ sd; points within 2 SD, their data offset; mean fn) | other runs: estimate (error; pred sd; covered; points within 2 SD, their data offset; mean fn) |
|---|---:|---|---|
| gmm_D2_noise3_svbmc_seed1000, component 6 (0.068) | -1.322 ± 0.069 | -1.392 (-0.070; 0.461; 26, -0.013; -9.5) | gmm_D2_noise3_svbmc_seed1001: -2.573 (-1.251; 0.798; yes; 11, -1.018; -10.0); gmm_D2_noise3_svbmc_seed1002: -2.482 (-1.160; 0.747; yes; 8, -2.338; -26.2) |
| gmm_D2_noise3_svbmc_seed1001, component 18 (0.077) | -0.931 ± 0.036 | -0.025 (+0.905; 0.429; 23, 0.863; -25.7) | gmm_D2_noise3_svbmc_seed1000: -30.738 (-29.808; 8.356; no; 0, -; -31.2); gmm_D2_noise3_svbmc_seed1002: -1.935 (-1.004; 0.603; yes; 11, 0.634; -58.7) |
| gmm_D2_noise3_svbmc_seed1002, component 17 (0.057) | -1.416 ± 0.068 | -0.816 (+0.600; 0.432; 20, 0.778; -49.6) | gmm_D2_noise3_svbmc_seed1000: -1.579 (-0.163; 0.734; yes; 22, -0.508; -9.9); gmm_D2_noise3_svbmc_seed1001: -1.781 (-0.365; 0.753; yes; 20, -1.155; -19.1) |

| Run | own error | se | own z median | own abs z median | BQ sd median | pred sd median | own-run max abs z (offset z) | Jacobian max abs z (offset z) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gmm_D2_noise3_svbmc_seed1000 | -0.112 | 0.018 | -0.40 | 0.42 | 0.544 | 0.779 | 2.77 (+0.44) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1001 | 0.675 | 0.018 | 2.40 | 2.40 | 0.518 | 0.685 | 2.50 (-2.91) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1002 | 0.307 | 0.020 | 1.27 | 1.28 | 0.493 | 0.678 | 2.72 (+1.04) | 0.00 (+0.00) |

## gmm_D2_svbmc (noiseless)

### M = 3 (5 cells)

| Estimator | bias | bias with the arm's entropy | covered | mc_se | gp_sd (indep / corr) |
|---|---:|---:|---:|---:|---:|
| raw | 0.026 [0.015, 0.029] | 0.019 [0.012, 0.032] | - | - | - |
| capped_I | -0.023 [-0.032, -0.020] | -0.028 [-0.039, -0.013] | - | - | - |
| capped_E | 0.026 [0.015, 0.029] | 0.019 [0.012, 0.032] | - | - | - |
| honest ratio1.5 median | 0.024 [0.006, 0.034] | 0.021 [0.005, 0.026] | 0.292 | 0.006 | 0.005 / 0.013 |
| honest ratio1.5 precision | 0.024 [0.006, 0.034] | 0.021 [0.005, 0.026] | 0.292 | 0.006 | 0.005 / 0.012 |
| honest ratio1.5 mean | 0.024 [0.006, 0.034] | 0.021 [0.005, 0.026] | 0.292 | 0.006 | 0.005 / 0.013 |
| honest ratio2 median | 0.026 [0.008, 0.035] | 0.023 [0.007, 0.027] | 0.296 | 0.006 | 0.005 / 0.015 |
| honest ratio2 precision | 0.025 [0.007, 0.035] | 0.022 [0.007, 0.027] | 0.296 | 0.006 | 0.005 / 0.012 |
| honest ratio2 mean | 0.026 [0.008, 0.035] | 0.023 [0.007, 0.027] | 0.296 | 0.006 | 0.005 / 0.015 |
| honest ratio3 median | 0.025 [0.008, 0.032] | 0.022 [0.005, 0.024] | 0.302 | 0.006 | 0.007 / 0.018 |
| honest ratio3 precision | 0.024 [0.007, 0.032] | 0.021 [0.005, 0.024] | 0.302 | 0.006 | 0.007 / 0.016 |
| honest ratio3 mean | 0.025 [0.008, 0.032] | 0.022 [0.005, 0.024] | 0.302 | 0.006 | 0.007 / 0.018 |
| honest ratio5 median | 0.017 [-0.002, 0.023] | 0.012 [-0.009, 0.017] | 0.332 | 0.006 | 0.017 / 0.043 |
| honest ratio5 precision | 0.016 [-0.003, 0.022] | 0.012 [-0.009, 0.017] | 0.332 | 0.006 | 0.016 / 0.042 |
| honest ratio5 mean | 0.017 [-0.002, 0.023] | 0.012 [-0.009, 0.017] | 0.332 | 0.006 | 0.017 / 0.043 |
| honest cap_only median | -0.049 [-0.068, -0.038] | -0.051 [-0.074, -0.039] | 0.653 | 0.009 | 0.043 / 0.221 |
| honest cap_only precision | -0.048 [-0.067, -0.037] | -0.050 [-0.073, -0.038] | 0.653 | 0.009 | 0.043 / 0.222 |
| honest cap_only mean | -0.049 [-0.068, -0.038] | -0.051 [-0.074, -0.039] | 0.653 | 0.009 | 0.043 / 0.221 |
| honest none median | -2.516 [-2.726, -2.221] | -2.523 [-2.729, -2.229] | 1.000 | 0.011 | 0.391 / 1.511 |
| honest none precision | -2.094 [-2.302, -1.846] | -2.101 [-2.306, -1.854] | 1.000 | 0.011 | 0.372 / 1.052 |
| honest none mean | -2.516 [-2.726, -2.221] | -2.523 [-2.729, -2.229] | 1.000 | 0.011 | 0.391 / 1.511 |
| pooled (ratio2, own included) | 0.024 [0.019, 0.037] | - | - | - | - |
| truth_strat | 0.011 [0.007, 0.021] | - | - | - | - |

Reference `e_log_joint_mc_sd` median 0.011; the headline bias is within 2 SE in 40% of the cells; median absolute bias raw 0.026, capped_I 0.023, capped_E 0.026, honest 0.026 (the honest estimate is **worse** than raw and **worse** than capped_I). Calibration of the self-reported SDs (weighted over components): own z median 0.02 [-0.24, 0.22], abs z median 0.81 [0.68, 0.87], abs z > 2 fraction 0.07 [0.03, 0.15]; cross z median 0.01 [-0.00, 0.06], abs z median 0.18 [0.13, 0.20], abs z > 2 fraction 0.00 [0.00, 0.00]. Effective training points on a component (weighted median): own run 5.4 [5.4, 5.5], covering other runs 6.6 [6.5, 6.7]. Checks: own-run max abs z 4.32 (weighted offset abs z 1.23), Jacobian max abs z 0.00 (offset abs z 0.00), raw consistency 2.2e-16, flagged cells 0.

Heaviest component of every run, repetition 0:

| Run, heaviest component (weight) | truth | own I_corr (error; BQ sd; points within 2 SD, their data offset; mean fn) | other runs: estimate (error; pred sd; covered; points within 2 SD, their data offset; mean fn) |
|---|---:|---|---|
| gmm_D2_svbmc_seed1000, component 12 (0.092) | -1.543 ± 0.130 | -1.674 (-0.131; 0.004; 17, 0.000; -6.6) | gmm_D2_svbmc_seed1001: -8.135 (-6.592; 4.590; no; 0, -; -8.6); gmm_D2_svbmc_seed1002: -8.012 (-6.468; 4.028; no; 0, -; -7.9) |
| gmm_D2_svbmc_seed1001, component 2 (0.007) | -0.906 ± 0.036 | -0.932 (-0.026; 0.003; 9, 0.000; -8.7) | gmm_D2_svbmc_seed1000: -0.907 (-0.001; 0.008; yes; 7, -0.000; -6.6); gmm_D2_svbmc_seed1002: -0.905 (+0.001; 0.012; yes; 7, 0.000; -7.7) |
| gmm_D2_svbmc_seed1002, component 0 (0.048) | -1.759 ± 0.103 | -1.736 (+0.023; 0.009; 16, -0.000; -7.7) | gmm_D2_svbmc_seed1000: -1.746 (+0.014; 0.044; yes; 15, -0.000; -6.6); gmm_D2_svbmc_seed1001: -1.758 (+0.002; 0.018; yes; 26, 0.000; -8.7) |

| Run | own error | se | own z median | own abs z median | BQ sd median | pred sd median | own-run max abs z (offset z) | Jacobian max abs z (offset z) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gmm_D2_svbmc_seed1000 | 0.018 | 0.020 | -0.04 | 0.69 | 0.004 | 0.032 | 2.27 (+0.39) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1001 | 0.001 | 0.014 | 0.07 | 0.69 | 0.004 | 0.009 | 3.18 (-0.15) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1002 | 0.058 | 0.036 | 0.69 | 1.04 | 0.007 | 0.059 | 2.97 (-1.46) | 0.00 (+0.00) |

## multisensory_s1_D6_noise3_svbmc (noisy)

### M = 3 (5 cells)

| Estimator | bias | bias with the arm's entropy | covered | mc_se | gp_sd (indep / corr) |
|---|---:|---:|---:|---:|---:|
| raw | 0.558 [0.526, 0.581] | 0.549 [0.527, 0.576] | - | - | - |
| capped_I | 0.267 [0.252, 0.299] | 0.265 [0.246, 0.294] | - | - | - |
| capped_E | 0.458 [0.444, 0.490] | 0.456 [0.437, 0.485] | - | - | - |
| honest ratio1.5 median | 0.172 [0.069, 0.201] | 0.152 [0.093, 0.199] | 0.777 | 0.023 | 0.114 / 0.493 |
| honest ratio1.5 precision | 0.172 [0.069, 0.201] | 0.152 [0.093, 0.199] | 0.777 | 0.023 | 0.114 / 0.493 |
| honest ratio1.5 mean | 0.172 [0.069, 0.201] | 0.152 [0.093, 0.199] | 0.777 | 0.023 | 0.114 / 0.493 |
| honest ratio2 median | 0.105 [0.062, 0.182] | 0.100 [0.084, 0.180] | 0.779 | 0.023 | 0.115 / 0.500 |
| honest ratio2 precision | 0.106 [0.063, 0.183] | 0.100 [0.084, 0.181] | 0.779 | 0.023 | 0.115 / 0.500 |
| honest ratio2 mean | 0.105 [0.062, 0.182] | 0.100 [0.084, 0.180] | 0.779 | 0.023 | 0.115 / 0.500 |
| honest ratio3 median | 0.092 [-0.001, 0.143] | 0.072 [0.023, 0.141] | 0.791 | 0.023 | 0.115 / 0.506 |
| honest ratio3 precision | 0.092 [-0.001, 0.143] | 0.072 [0.023, 0.141] | 0.791 | 0.023 | 0.115 / 0.506 |
| honest ratio3 mean | 0.092 [-0.001, 0.143] | 0.072 [0.023, 0.141] | 0.791 | 0.023 | 0.115 / 0.506 |
| honest ratio5 median | 0.092 [-0.001, 0.143] | 0.072 [0.023, 0.141] | 0.791 | 0.023 | 0.115 / 0.506 |
| honest ratio5 precision | 0.092 [-0.001, 0.143] | 0.072 [0.023, 0.141] | 0.791 | 0.023 | 0.115 / 0.506 |
| honest ratio5 mean | 0.092 [-0.001, 0.143] | 0.072 [0.023, 0.141] | 0.791 | 0.023 | 0.115 / 0.506 |
| honest cap_only median | 0.092 [-0.001, 0.143] | 0.072 [0.023, 0.141] | 0.791 | 0.023 | 0.115 / 0.506 |
| honest cap_only precision | 0.092 [-0.001, 0.143] | 0.072 [0.023, 0.141] | 0.791 | 0.023 | 0.115 / 0.506 |
| honest cap_only mean | 0.092 [-0.001, 0.143] | 0.072 [0.023, 0.141] | 0.791 | 0.023 | 0.115 / 0.506 |
| honest none median | -18.862 [-19.473, -18.184] | -18.882 [-19.478, -18.186] | 1.000 | 0.056 | 0.802 / 4.450 |
| honest none precision | -6.570 [-7.617, -5.511] | -6.590 [-7.623, -5.513] | 1.000 | 0.040 | 0.401 / 2.067 |
| honest none mean | -18.862 [-19.473, -18.184] | -18.882 [-19.478, -18.186] | 1.000 | 0.056 | 0.802 / 4.450 |
| pooled (ratio2, own included) | 0.351 [0.278, 0.391] | - | - | - | - |
| truth_strat | 0.010 [-0.072, 0.033] | - | - | - | - |

Reference `e_log_joint_mc_sd` median 0.018; the headline bias is within 2 SE in 0% of the cells; median absolute bias raw 0.558, capped_I 0.267, capped_E 0.458, honest 0.105 (the honest estimate is not worse than raw and not worse than capped_I). Calibration of the self-reported SDs (weighted over components): own z median 1.08 [0.97, 1.15], abs z median 1.11 [1.07, 1.21], abs z > 2 fraction 0.18 [0.13, 0.19]; cross z median 0.29 [0.26, 0.35], abs z median 0.61 [0.60, 0.76], abs z > 2 fraction 0.02 [0.02, 0.03]. Effective training points on a component (weighted median): own run 0.5 [0.5, 0.5], covering other runs 0.5 [0.5, 0.5]. Checks: own-run max abs z 3.82 (weighted offset abs z 1.76), Jacobian max abs z 4.11 (offset abs z 1.50), raw consistency 0.0e+00, flagged cells 0.

Heaviest component of every run, repetition 0:

| Run, heaviest component (weight) | truth | own I_corr (error; BQ sd; points within 2 SD, their data offset; mean fn) | other runs: estimate (error; pred sd; covered; points within 2 SD, their data offset; mean fn) |
|---|---:|---|---|
| multisensory_s1_D6_noise3_svbmc_seed1000, component 1 (0.055) | -505.997 ± 0.141 | -504.857 (+1.140; 0.501; 1, -1.745; -576.4) | multisensory_s1_D6_noise3_svbmc_seed1001: -545.605 (-39.608; 12.397; no; 0, -; -545.5); multisensory_s1_D6_noise3_svbmc_seed1002: -506.032 (-0.035; 0.958; yes; 2, -1.524; -569.9) |
| multisensory_s1_D6_noise3_svbmc_seed1001, component 2 (0.040) | -505.477 ± 0.161 | -505.006 (+0.471; 0.474; 1, 1.558; -541.5) | multisensory_s1_D6_noise3_svbmc_seed1000: -524.199 (-18.722; 7.323; no; 0, -; -584.0); multisensory_s1_D6_noise3_svbmc_seed1002: -538.498 (-33.021; 9.288; no; 0, -; -614.2) |
| multisensory_s1_D6_noise3_svbmc_seed1002, component 4 (0.046) | -505.300 ± 0.130 | -505.023 (+0.277; 0.438; 2, 1.202; -587.6) | multisensory_s1_D6_noise3_svbmc_seed1000: -505.573 (-0.273; 0.760; yes; 1, -3.997; -582.3); multisensory_s1_D6_noise3_svbmc_seed1001: -528.380 (-23.080; 7.955; no; 0, -; -543.1) |

| Run | own error | se | own z median | own abs z median | BQ sd median | pred sd median | own-run max abs z (offset z) | Jacobian max abs z (offset z) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| multisensory_s1_D6_noise3_svbmc_seed1000 | 0.887 | 0.043 | 1.96 | 1.96 | 0.537 | 0.799 | 2.48 (-0.26) | 2.50 (-1.00) |
| multisensory_s1_D6_noise3_svbmc_seed1001 | 0.326 | 0.041 | 0.83 | 0.87 | 0.676 | 1.019 | 2.41 (-1.90) | 2.66 (-0.52) |
| multisensory_s1_D6_noise3_svbmc_seed1002 | 0.366 | 0.033 | 0.61 | 0.77 | 0.563 | 0.853 | 3.05 (-0.17) | 2.56 (-0.14) |

## ring_D2_noise3_svbmc (noisy)

### M = 3 (5 cells)

| Estimator | bias | bias with the arm's entropy | covered | mc_se | gp_sd (indep / corr) |
|---|---:|---:|---:|---:|---:|
| raw | 0.109 [0.103, 0.134] | 0.118 [0.091, 0.134] | - | - | - |
| capped_I | 0.003 [0.002, 0.012] | 0.003 [-0.016, 0.023] | - | - | - |
| capped_E | 0.105 [0.103, 0.115] | 0.106 [0.087, 0.126] | - | - | - |
| honest ratio1.5 median | 0.067 [0.049, 0.086] | 0.070 [0.045, 0.086] | 0.235 | 0.007 | 0.062 / 0.296 |
| honest ratio1.5 precision | 0.067 [0.049, 0.086] | 0.070 [0.045, 0.086] | 0.235 | 0.007 | 0.062 / 0.296 |
| honest ratio1.5 mean | 0.067 [0.049, 0.086] | 0.070 [0.045, 0.086] | 0.235 | 0.007 | 0.062 / 0.296 |
| honest ratio2 median | 0.048 [0.028, 0.079] | 0.056 [0.027, 0.070] | 0.304 | 0.007 | 0.066 / 0.298 |
| honest ratio2 precision | 0.048 [0.028, 0.079] | 0.056 [0.027, 0.070] | 0.304 | 0.007 | 0.066 / 0.298 |
| honest ratio2 mean | 0.048 [0.028, 0.079] | 0.056 [0.027, 0.070] | 0.304 | 0.007 | 0.066 / 0.298 |
| honest ratio3 median | -0.032 [-0.066, -0.031] | -0.042 [-0.049, -0.036] | 0.436 | 0.011 | 0.102 / 0.387 |
| honest ratio3 precision | -0.032 [-0.066, -0.031] | -0.042 [-0.049, -0.036] | 0.436 | 0.011 | 0.102 / 0.387 |
| honest ratio3 mean | -0.032 [-0.066, -0.031] | -0.042 [-0.049, -0.036] | 0.436 | 0.011 | 0.102 / 0.387 |
| honest ratio5 median | -0.030 [-0.063, -0.025] | -0.039 [-0.045, -0.030] | 0.508 | 0.011 | 0.116 / 0.496 |
| honest ratio5 precision | -0.030 [-0.063, -0.025] | -0.039 [-0.045, -0.030] | 0.508 | 0.011 | 0.116 / 0.496 |
| honest ratio5 mean | -0.030 [-0.063, -0.025] | -0.039 [-0.045, -0.030] | 0.508 | 0.011 | 0.116 / 0.496 |
| honest cap_only median | -0.030 [-0.063, -0.025] | -0.039 [-0.045, -0.030] | 0.508 | 0.011 | 0.116 / 0.496 |
| honest cap_only precision | -0.030 [-0.063, -0.025] | -0.039 [-0.045, -0.030] | 0.508 | 0.011 | 0.116 / 0.496 |
| honest cap_only mean | -0.030 [-0.063, -0.025] | -0.039 [-0.045, -0.030] | 0.508 | 0.011 | 0.116 / 0.496 |
| honest none median | -1096.259 [-1120.786, -1088.647] | -1096.278 [-1120.795, -1088.633] | 1.000 | 0.627 | 106.995 / 626.409 |
| honest none precision | -13.108 [-13.271, -12.452] | -13.126 [-13.279, -12.438] | 1.000 | 0.065 | 3.542 / 13.997 |
| honest none mean | -1096.259 [-1120.786, -1088.647] | -1096.278 [-1120.795, -1088.633] | 1.000 | 0.627 | 106.995 / 626.409 |
| pooled (ratio2, own included) | 0.081 [0.058, 0.102] | - | - | - | - |
| truth_strat | 0.007 [-0.019, 0.017] | - | - | - | - |

Reference `e_log_joint_mc_sd` median 0.010; the headline bias is within 2 SE in 0% of the cells; median absolute bias raw 0.109, capped_I 0.003, capped_E 0.105, honest 0.048 (the honest estimate is not worse than raw and **worse** than capped_I). Calibration of the self-reported SDs (weighted over components): own z median 0.21 [0.20, 0.25], abs z median 0.63 [0.52, 0.65], abs z > 2 fraction 0.00 [0.00, 0.04]; cross z median 0.03 [-0.06, 0.11], abs z median 0.37 [0.34, 0.38], abs z > 2 fraction 0.00 [0.00, 0.00]. Effective training points on a component (weighted median): own run 4.2 [4.1, 4.4], covering other runs 4.0 [4.0, 4.2]. Checks: own-run max abs z 5.84 (weighted offset abs z 1.27), Jacobian max abs z 0.00 (offset abs z 0.00), raw consistency 1.1e-16, flagged cells 0.

Heaviest component of every run, repetition 0:

| Run, heaviest component (weight) | truth | own I_corr (error; BQ sd; points within 2 SD, their data offset; mean fn) | other runs: estimate (error; pred sd; covered; points within 2 SD, their data offset; mean fn) |
|---|---:|---|---|
| ring_D2_noise3_svbmc_seed1000, component 21 (0.036) | -0.398 ± 0.055 | 0.158 (+0.556; 0.545; 6, -0.925; -260.5) | ring_D2_noise3_svbmc_seed1001: -1102.759 (-1102.360; 1476.954; no; 0, -; -1086.3); ring_D2_noise3_svbmc_seed1002: -0.566 (-0.167; 0.421; yes; 10, 0.772; -4503.1) |
| ring_D2_noise3_svbmc_seed1001, component 1 (0.027) | -0.409 ± 0.052 | -0.433 (-0.024; 0.506; 7, -0.288; -966.1) | ring_D2_noise3_svbmc_seed1000: -83.261 (-82.852; 89.802; no; 0, -; -397.6); ring_D2_noise3_svbmc_seed1002: -3544.461 (-3544.053; 1863.094; no; 0, -; -4278.1) |
| ring_D2_noise3_svbmc_seed1002, component 10 (0.053) | -0.869 ± 0.133 | -0.153 (+0.716; 0.505; 4, 0.835; -4478.7) | ring_D2_noise3_svbmc_seed1000: -3.176 (-2.307; 1.302; no; 0, -; -289.3); ring_D2_noise3_svbmc_seed1001: -478.810 (-477.941; 1170.331; no; 0, -; -930.4) |

| Run | own error | se | own z median | own abs z median | BQ sd median | pred sd median | own-run max abs z (offset z) | Jacobian max abs z (offset z) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ring_D2_noise3_svbmc_seed1000 | 0.108 | 0.026 | 0.32 | 0.48 | 0.533 | 0.661 | 2.32 (+0.82) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1001 | 0.025 | 0.015 | -0.04 | 0.54 | 0.423 | 0.491 | 3.97 (+0.97) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1002 | 0.145 | 0.020 | 0.05 | 0.68 | 0.434 | 0.448 | 2.57 (-0.31) | 0.00 (+0.00) |

## rosenbrock_D2_noise3_svbmc (noisy)

### M = 3 (5 cells)

| Estimator | bias | bias with the arm's entropy | covered | mc_se | gp_sd (indep / corr) |
|---|---:|---:|---:|---:|---:|
| raw | 0.123 [0.099, 0.140] | 0.127 [0.099, 0.154] | - | - | - |
| capped_I | 0.027 [0.018, 0.033] | 0.031 [0.025, 0.046] | - | - | - |
| capped_E | 0.036 [0.027, 0.041] | 0.040 [0.034, 0.055] | - | - | - |
| honest ratio1.5 median | -0.225 [-0.237, -0.207] | -0.220 [-0.224, -0.191] | 1.000 | 0.011 | 0.039 / 0.218 |
| honest ratio1.5 precision | -0.231 [-0.243, -0.214] | -0.226 [-0.230, -0.197] | 1.000 | 0.011 | 0.036 / 0.224 |
| honest ratio1.5 mean | -0.225 [-0.237, -0.207] | -0.220 [-0.224, -0.191] | 1.000 | 0.011 | 0.039 / 0.218 |
| honest ratio2 median | -0.227 [-0.238, -0.209] | -0.221 [-0.227, -0.193] | 1.000 | 0.011 | 0.039 / 0.219 |
| honest ratio2 precision | -0.232 [-0.244, -0.215] | -0.227 [-0.232, -0.199] | 1.000 | 0.011 | 0.036 / 0.224 |
| honest ratio2 mean | -0.227 [-0.238, -0.209] | -0.221 [-0.227, -0.193] | 1.000 | 0.011 | 0.039 / 0.219 |
| honest ratio3 median | -0.227 [-0.238, -0.209] | -0.221 [-0.227, -0.193] | 1.000 | 0.011 | 0.039 / 0.221 |
| honest ratio3 precision | -0.232 [-0.244, -0.215] | -0.227 [-0.232, -0.199] | 1.000 | 0.011 | 0.036 / 0.224 |
| honest ratio3 mean | -0.227 [-0.238, -0.209] | -0.221 [-0.227, -0.193] | 1.000 | 0.011 | 0.039 / 0.221 |
| honest ratio5 median | -0.227 [-0.237, -0.208] | -0.220 [-0.226, -0.192] | 1.000 | 0.011 | 0.039 / 0.223 |
| honest ratio5 precision | -0.232 [-0.244, -0.215] | -0.227 [-0.232, -0.198] | 1.000 | 0.011 | 0.036 / 0.224 |
| honest ratio5 mean | -0.227 [-0.237, -0.208] | -0.220 [-0.226, -0.192] | 1.000 | 0.011 | 0.039 / 0.223 |
| honest cap_only median | -0.227 [-0.237, -0.208] | -0.220 [-0.226, -0.192] | 1.000 | 0.011 | 0.039 / 0.223 |
| honest cap_only precision | -0.232 [-0.244, -0.215] | -0.227 [-0.232, -0.198] | 1.000 | 0.011 | 0.036 / 0.224 |
| honest cap_only mean | -0.227 [-0.237, -0.208] | -0.220 [-0.226, -0.192] | 1.000 | 0.011 | 0.039 / 0.223 |
| honest none median | -0.227 [-0.237, -0.208] | -0.220 [-0.226, -0.192] | 1.000 | 0.011 | 0.039 / 0.223 |
| honest none precision | -0.232 [-0.244, -0.215] | -0.227 [-0.232, -0.198] | 1.000 | 0.011 | 0.036 / 0.224 |
| honest none mean | -0.227 [-0.237, -0.208] | -0.220 [-0.226, -0.192] | 1.000 | 0.011 | 0.039 / 0.223 |
| pooled (ratio2, own included) | -0.193 [-0.201, -0.178] | - | - | - | - |
| truth_strat | -0.023 [-0.028, -0.004] | - | - | - | - |

Reference `e_log_joint_mc_sd` median 0.009; the headline bias is within 2 SE in 0% of the cells; median absolute bias raw 0.123, capped_I 0.027, capped_E 0.036, honest 0.227 (the honest estimate is **worse** than raw and **worse** than capped_I). Calibration of the self-reported SDs (weighted over components): own z median 0.49 [0.35, 0.51], abs z median 0.53 [0.43, 0.55], abs z > 2 fraction 0.00 [0.00, 0.00]; cross z median -0.58 [-0.60, -0.57], abs z median 0.60 [0.58, 0.61], abs z > 2 fraction 0.00 [0.00, 0.00]. Effective training points on a component (weighted median): own run 3.8 [3.6, 4.1], covering other runs 3.5 [3.5, 3.6]. Checks: own-run max abs z 4.02 (weighted offset abs z 2.64), Jacobian max abs z 0.00 (offset abs z 0.00), raw consistency 0.0e+00, flagged cells 0.

Heaviest component of every run, repetition 0:

| Run, heaviest component (weight) | truth | own I_corr (error; BQ sd; points within 2 SD, their data offset; mean fn) | other runs: estimate (error; pred sd; covered; points within 2 SD, their data offset; mean fn) |
|---|---:|---|---|
| rosenbrock_D2_noise3_svbmc_seed1000, component 6 (0.034) | -4.956 ± 0.068 | -4.774 (+0.182; 0.441; 4, -0.427; -119126.4) | rosenbrock_D2_noise3_svbmc_seed1001: -5.428 (-0.472; 0.450; yes; 2, -0.697; -145309.3); rosenbrock_D2_noise3_svbmc_seed1002: -5.333 (-0.377; 0.345; yes; 2, -2.616; -87763.5) |
| rosenbrock_D2_noise3_svbmc_seed1001, component 40 (0.056) | -4.732 ± 0.102 | -4.431 (+0.301; 0.303; 20, 0.832; -108780.6) | rosenbrock_D2_noise3_svbmc_seed1000: -4.933 (-0.200; 0.446; yes; 10, -1.224; -110461.7); rosenbrock_D2_noise3_svbmc_seed1002: -4.951 (-0.219; 0.288; yes; 14, -0.994; -113619.6) |
| rosenbrock_D2_noise3_svbmc_seed1002, component 37 (0.006) | -4.519 ± 0.064 | -5.052 (-0.534; 0.303; 25, -0.655; -98101.0) | rosenbrock_D2_noise3_svbmc_seed1000: -4.576 (-0.058; 0.450; yes; 9, -1.286; -117125.2); rosenbrock_D2_noise3_svbmc_seed1001: -4.586 (-0.067; 0.351; yes; 10, -0.418; -126672.5) |

| Run | own error | se | own z median | own abs z median | BQ sd median | pred sd median | own-run max abs z (offset z) | Jacobian max abs z (offset z) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rosenbrock_D2_noise3_svbmc_seed1000 | 0.036 | 0.013 | 0.02 | 0.27 | 0.448 | 0.466 | 3.01 (+0.77) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1001 | 0.012 | 0.012 | 0.19 | 0.44 | 0.318 | 0.328 | 2.76 (+1.47) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1002 | -0.255 | 0.014 | -0.92 | 0.91 | 0.304 | 0.300 | 3.93 (+1.07) | 0.00 (+0.00) |
