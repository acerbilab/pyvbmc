# Honest (cross-run) expected log joint of stacked posteriors

Generated 2026-09-15 08:58:01 from the cells of `results.json` (arm `integrated`), 100 draws per component, seed 0. Every estimate of the stack's expected log joint is scored by its bias against the cell's `e_log_joint_mc` (the mean of the target's noiseless log density over the arm's own draws of the stacked posterior); adding the cell's `entropy_ref` to every estimate gives the ELBOs and leaves every bias unchanged. `raw` is `sum w I_corr`, `capped_I` and `capped_E` the class's component-median and run-median caps of it; an honest value replaces every component's expected log joint by the covering other runs' GP estimates under a coverage rule (`ratioR`: a run's mean predictive SD on the component at most `R` times the component's own run's or below 0.1 nats, and at most 2.236 nats; `cap_only`: the cap alone; `none`: every other run) combined by the `median`, `precision` weighting or the `mean`, the component's own value when no other run covers it. The headline is `ratio2` with the `median`. `truth_strat` is the bias of the stratified Monte Carlo truth (`sum w E_true`) against `e_log_joint_mc`, a consistency check between the two Monte Carlo routes, and `pooled` the headline rule with the component's own run included (not honest). Medians over the cells of a condition and `M` with a 10,000-resample bootstrap 95 % interval. `covered` is the stacked weight on components at least one other run covers. `mc_se` is the Monte Carlo standard error of the honest value from the draws (the covering runs average the same draws), `gp_sd` its GP-side standard deviation under independence across components and under full correlation within each evaluating run. `within 2 SE` is the fraction of cells whose headline bias is within twice the combined Monte Carlo standard error of the estimate and of the reference.

## Headline across conditions

| Condition | M | cells | raw | capped_I | honest (ratio2, median) | honest (ratio2, precision) | covered | mc_se | within 2 SE | median abs bias raw / capped_I / honest | KL gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| gmm_D2_noise3_svbmc | 2 | 20 | 0.285 [0.189, 0.411] | 0.083 [-0.204, 0.238] | 0.147 [-0.019, 0.289] | 0.147 [-0.019, 0.289] | 0.238 | 0.005 | 0.05 | 0.285 / 0.265 / 0.289 | 0.76 [0.53, 0.89] |
| gmm_D2_noise3_svbmc | 4 | 20 | 0.308 [0.194, 0.419] | 0.005 [-0.057, 0.086] | -0.211 [-0.268, -0.104] | -0.199 [-0.268, -0.085] | 0.703 | 0.011 | 0.00 | 0.308 / 0.086 / 0.240 | 0.33 [0.22, 0.45] |
| gmm_D2_noise3_svbmc | 8 | 20 | 0.506 [0.454, 0.558] | 0.074 [0.009, 0.111] | -0.292 [-0.368, -0.253] | -0.251 [-0.322, -0.210] | 0.901 | 0.011 | 0.00 | 0.506 / 0.082 / 0.292 | 0.21 [0.17, 0.23] |
| gmm_D2_noise3_svbmc | 16 | 10 | 0.713 [0.572, 0.914] | 0.035 [-0.005, 0.099] | -0.293 [-0.324, -0.185] | -0.246 [-0.287, -0.191] | 1.000 | 0.012 | 0.00 | 0.713 / 0.035 / 0.293 | 0.13 [0.11, 0.22] |
| gmm_D2_svbmc | 2 | 20 | 0.009 [0.008, 0.015] | -0.002 [-0.030, 0.009] | 0.009 [0.005, 0.013] | 0.009 [0.005, 0.013] | 0.282 | 0.005 | 0.85 | 0.009 / 0.017 / 0.009 | 0.31 [0.29, 0.69] |
| gmm_D2_svbmc | 4 | 20 | 0.011 [0.004, 0.019] | -0.007 [-0.041, 0.008] | 0.005 [-0.001, 0.014] | 0.005 [-0.001, 0.014] | 0.631 | 0.008 | 0.90 | 0.013 / 0.023 / 0.007 | 0.04 [0.02, 0.30] |
| gmm_D2_svbmc | 8 | 20 | 0.006 [-0.001, 0.012] | -0.017 [-0.051, 0.002] | -0.001 [-0.007, 0.005] | -0.001 [-0.007, 0.005] | 0.994 | 0.008 | 0.95 | 0.007 / 0.028 / 0.008 | 0.01 [0.01, 0.01] |
| gmm_D2_svbmc | 16 | 10 | 0.009 [-0.003, 0.015] | -0.020 [-0.055, 0.002] | 0.007 [-0.009, 0.013] | 0.006 [-0.010, 0.012] | 0.999 | 0.006 | 1.00 | 0.009 / 0.021 / 0.009 | 0.01 [-0.00, 0.02] |
| multisensory_s1_D6_noise1.3_svbmc | 2 | 20 | 0.403 [0.334, 0.482] | 0.115 [0.039, 0.202] | -0.147 [-0.202, -0.061] | -0.147 [-0.202, -0.061] | 0.978 | 0.026 | 0.15 | 0.403 / 0.150 / 0.194 | 0.59 [0.56, 0.65] |
| multisensory_s1_D6_noise1.3_svbmc | 4 | 20 | 0.483 [0.451, 0.508] | 0.113 [0.061, 0.157] | -0.264 [-0.353, -0.211] | -0.238 [-0.304, -0.187] | 0.984 | 0.024 | 0.15 | 0.483 / 0.113 / 0.264 | 0.49 [0.40, 0.55] |
| multisensory_s1_D6_noise1.3_svbmc | 8 | 20 | 0.555 [0.521, 0.637] | 0.151 [0.117, 0.195] | -0.238 [-0.335, -0.186] | -0.182 [-0.252, -0.142] | 0.997 | 0.018 | 0.05 | 0.555 / 0.151 / 0.238 | 0.43 [0.36, 0.45] |
| multisensory_s1_D6_noise1.3_svbmc | 16 | 10 | 0.617 [0.561, 0.685] | 0.165 [0.110, 0.191] | -0.294 [-0.351, -0.253] | -0.259 [-0.291, -0.225] | 1.000 | 0.015 | 0.00 | 0.617 / 0.165 / 0.294 | 0.25 [0.21, 0.32] |
| multisensory_s1_D6_noise3_svbmc | 2 | 20 | 0.785 [0.626, 0.918] | 0.406 [0.147, 0.492] | -0.299 [-0.594, -0.025] | -0.299 [-0.594, -0.025] | 0.974 | 0.031 | 0.25 | 0.785 / 0.406 / 0.383 | 0.85 [0.80, 0.91] |
| multisensory_s1_D6_noise3_svbmc | 4 | 20 | 1.071 [0.951, 1.150] | 0.555 [0.413, 0.767] | -0.479 [-0.632, -0.161] | -0.349 [-0.549, -0.187] | 0.999 | 0.029 | 0.10 | 1.071 / 0.555 / 0.479 | 0.73 [0.66, 0.80] |
| multisensory_s1_D6_noise3_svbmc | 8 | 20 | 1.173 [1.070, 1.304] | 0.401 [0.316, 0.513] | -0.713 [-0.796, -0.523] | -0.609 [-0.669, -0.578] | 0.999 | 0.024 | 0.00 | 1.173 / 0.401 / 0.713 | 0.48 [0.42, 0.52] |
| multisensory_s1_D6_noise3_svbmc | 16 | 10 | 1.315 [1.243, 1.410] | 0.416 [0.392, 0.496] | -0.678 [-0.791, -0.467] | -0.603 [-0.703, -0.475] | 0.991 | 0.021 | 0.00 | 1.315 / 0.416 / 0.678 | 0.43 [0.34, 0.51] |
| multisensory_s1_D6_svbmc | 2 | 20 | 0.045 [0.029, 0.061] | -0.042 [-0.079, 0.010] | 0.003 [-0.033, 0.013] | 0.003 [-0.033, 0.013] | 0.854 | 0.024 | 0.75 | 0.045 / 0.053 / 0.031 | 0.32 [0.30, 0.36] |
| multisensory_s1_D6_svbmc | 4 | 20 | 0.042 [0.031, 0.069] | -0.009 [-0.027, 0.010] | -0.015 [-0.029, 0.005] | -0.000 [-0.025, 0.013] | 0.962 | 0.019 | 0.75 | 0.042 / 0.026 / 0.024 | 0.25 [0.20, 0.28] |
| multisensory_s1_D6_svbmc | 8 | 20 | 0.072 [0.056, 0.079] | 0.032 [0.006, 0.053] | -0.012 [-0.024, -0.002] | -0.018 [-0.028, -0.005] | 0.988 | 0.013 | 0.95 | 0.072 / 0.043 / 0.021 | 0.18 [0.15, 0.22] |
| multisensory_s1_D6_svbmc | 16 | 10 | 0.053 [0.037, 0.079] | 0.015 [-0.009, 0.047] | -0.034 [-0.039, -0.030] | -0.035 [-0.049, -0.029] | 0.994 | 0.009 | 0.90 | 0.053 / 0.024 / 0.036 | 0.14 [0.12, 0.17] |
| ring_D2_noise3_svbmc | 2 | 20 | 0.403 [0.283, 0.485] | 0.204 [0.041, 0.346] | 0.223 [0.052, 0.328] | 0.223 [0.052, 0.328] | 0.157 | 0.005 | 0.00 | 0.403 / 0.217 / 0.277 | 1.01 [0.83, 1.21] |
| ring_D2_noise3_svbmc | 4 | 20 | 0.401 [0.267, 0.488] | 0.160 [0.051, 0.246] | 0.173 [0.033, 0.218] | 0.173 [0.024, 0.240] | 0.465 | 0.006 | 0.10 | 0.401 / 0.160 / 0.193 | 0.53 [0.48, 0.63] |
| ring_D2_noise3_svbmc | 8 | 20 | 0.489 [0.379, 0.541] | 0.155 [0.086, 0.183] | -0.116 [-0.200, -0.056] | -0.071 [-0.171, -0.024] | 0.813 | 0.007 | 0.05 | 0.489 / 0.155 / 0.116 | 0.33 [0.23, 0.38] |
| ring_D2_noise3_svbmc | 16 | 10 | 0.756 [0.686, 0.815] | 0.140 [0.118, 0.196] | -0.098 [-0.143, -0.060] | -0.052 [-0.100, -0.005] | 0.997 | 0.007 | 0.10 | 0.756 / 0.140 / 0.098 | 0.15 [0.13, 0.16] |
| rosenbrock_D2_noise3_svbmc | 2 | 20 | 0.251 [0.148, 0.426] | 0.104 [-0.082, 0.179] | -0.014 [-0.253, 0.087] | -0.014 [-0.253, 0.087] | 0.977 | 0.013 | 0.00 | 0.251 / 0.206 / 0.207 | 0.07 [0.04, 0.09] |
| rosenbrock_D2_noise3_svbmc | 4 | 20 | 0.351 [0.301, 0.535] | 0.079 [-0.006, 0.201] | -0.069 [-0.183, -0.000] | -0.045 [-0.123, 0.000] | 1.000 | 0.010 | 0.05 | 0.351 / 0.110 / 0.170 | 0.05 [0.04, 0.08] |
| rosenbrock_D2_noise3_svbmc | 8 | 20 | 0.531 [0.466, 0.584] | 0.095 [0.045, 0.131] | -0.048 [-0.075, 0.073] | -0.010 [-0.077, 0.054] | 1.000 | 0.010 | 0.05 | 0.531 / 0.095 / 0.076 | 0.04 [0.03, 0.05] |
| rosenbrock_D2_noise3_svbmc | 16 | 10 | 0.729 [0.673, 0.993] | 0.111 [0.036, 0.134] | 0.044 [0.007, 0.072] | 0.045 [0.008, 0.062] | 1.000 | 0.011 | 0.40 | 0.729 / 0.111 / 0.044 | 0.05 [0.04, 0.06] |
| student_D8_noise3_svbmc | 2 | 20 | -0.083 [-0.246, -0.002] | -0.911 [-1.144, -0.602] | -1.175 [-1.709, -0.983] | -1.175 [-1.709, -1.010] | 1.000 | 0.038 | 0.00 | 0.151 / 0.911 / 1.175 | 0.52 [0.42, 0.60] |
| student_D8_noise3_svbmc | 4 | 20 | -0.105 [-0.159, 0.050] | -1.178 [-1.456, -0.997] | -1.485 [-1.792, -1.232] | -1.573 [-1.834, -1.365] | 1.000 | 0.034 | 0.00 | 0.159 / 1.178 / 1.485 | 0.43 [0.41, 0.45] |
| student_D8_noise3_svbmc | 8 | 20 | 0.250 [0.075, 0.383] | -1.386 [-1.623, -1.244] | -1.418 [-1.515, -1.325] | -1.457 [-1.584, -1.337] | 1.000 | 0.030 | 0.00 | 0.296 / 1.386 / 1.418 | 0.30 [0.27, 0.34] |
| student_D8_noise3_svbmc | 16 | 10 | 0.273 [0.239, 0.460] | -1.753 [-1.869, -1.493] | -1.397 [-1.452, -1.186] | -1.485 [-1.550, -1.386] | 1.000 | 0.025 | 0.00 | 0.273 / 1.753 / 1.397 | 0.25 [0.21, 0.30] |

## gmm_D2_noise3_svbmc (noisy)

### M = 2 (20 cells)

| Estimator | bias | bias with the arm's entropy | covered | mc_se | gp_sd (indep / corr) |
|---|---:|---:|---:|---:|---:|
| raw | 0.285 [0.189, 0.411] | 0.297 [0.179, 0.414] | - | - | - |
| capped_I | 0.083 [-0.204, 0.238] | 0.070 [-0.203, 0.248] | - | - | - |
| capped_E | 0.178 [0.051, 0.318] | 0.157 [0.039, 0.307] | - | - | - |
| honest ratio1.5 median | 0.153 [0.005, 0.289] | 0.124 [-0.002, 0.294] | 0.238 | 0.005 | 0.094 / 0.388 |
| honest ratio1.5 precision | 0.153 [0.005, 0.289] | 0.124 [-0.002, 0.294] | 0.238 | 0.005 | 0.094 / 0.388 |
| honest ratio1.5 mean | 0.153 [0.005, 0.289] | 0.124 [-0.002, 0.294] | 0.238 | 0.005 | 0.094 / 0.388 |
| honest ratio2 median | 0.147 [-0.019, 0.289] | 0.118 [-0.009, 0.294] | 0.238 | 0.005 | 0.096 / 0.389 |
| honest ratio2 precision | 0.147 [-0.019, 0.289] | 0.118 [-0.009, 0.294] | 0.238 | 0.005 | 0.096 / 0.389 |
| honest ratio2 mean | 0.147 [-0.019, 0.289] | 0.118 [-0.009, 0.294] | 0.238 | 0.005 | 0.096 / 0.389 |
| honest ratio3 median | 0.147 [-0.019, 0.254] | 0.118 [-0.009, 0.262] | 0.238 | 0.005 | 0.096 / 0.395 |
| honest ratio3 precision | 0.147 [-0.019, 0.254] | 0.118 [-0.009, 0.262] | 0.238 | 0.005 | 0.096 / 0.395 |
| honest ratio3 mean | 0.147 [-0.019, 0.254] | 0.118 [-0.009, 0.262] | 0.238 | 0.005 | 0.096 / 0.395 |
| honest ratio5 median | 0.069 [-0.134, 0.192] | 0.051 [-0.143, 0.197] | 0.316 | 0.006 | 0.098 / 0.397 |
| honest ratio5 precision | 0.069 [-0.134, 0.192] | 0.051 [-0.143, 0.197] | 0.316 | 0.006 | 0.098 / 0.397 |
| honest ratio5 mean | 0.069 [-0.134, 0.192] | 0.051 [-0.143, 0.197] | 0.316 | 0.006 | 0.098 / 0.397 |
| honest cap_only median | 0.069 [-0.134, 0.192] | 0.051 [-0.143, 0.197] | 0.316 | 0.006 | 0.098 / 0.397 |
| honest cap_only precision | 0.069 [-0.134, 0.192] | 0.051 [-0.143, 0.197] | 0.316 | 0.006 | 0.098 / 0.397 |
| honest cap_only mean | 0.069 [-0.134, 0.192] | 0.051 [-0.143, 0.197] | 0.316 | 0.006 | 0.098 / 0.397 |
| honest none median | -5.326 [-6.625, -1.193] | -5.337 [-6.629, -1.195] | 1.000 | 0.014 | 0.795 / 2.574 |
| honest none precision | -5.326 [-6.625, -1.193] | -5.337 [-6.629, -1.195] | 1.000 | 0.014 | 0.795 / 2.574 |
| honest none mean | -5.326 [-6.625, -1.193] | -5.337 [-6.629, -1.195] | 1.000 | 0.014 | 0.795 / 2.574 |
| pooled (ratio2, own included) | 0.225 [0.100, 0.376] | - | - | - | - |
| truth_strat | -0.004 [-0.011, 0.011] | - | - | - | - |

Reference `e_log_joint_mc_sd` median 0.010; the headline bias is within 2 SE in 5% of the cells; median absolute bias raw 0.285, capped_I 0.265, capped_E 0.193, honest 0.289 (against raw the honest estimate is a tie, against capped_I **worse**; tie band 0.010, one reference standard error). Calibration of the self-reported SDs (weighted over components): own z median 0.69 [0.45, 1.17], abs z median 0.92 [0.73, 1.20], abs z > 2 fraction 0.04 [0.01, 0.09]; cross z median -0.47 [-0.69, -0.20], abs z median 0.61 [0.48, 0.77], abs z > 2 fraction 0.00 [0.00, 0.02]. Effective training points on a component (weighted median): own run 12.2 [11.9, 13.8], covering other runs 7.8 [5.4, 9.8]. Checks: own-run max abs z 4.38 (weighted offset abs z 2.62), Jacobian max abs z 0.00 (offset abs z 0.00), raw consistency 6.7e-16, flagged cells 0.

Heaviest component of every run, repetition 0:

| Run, heaviest component (weight) | truth | own I_corr (error; BQ sd; points within 2 SD, their data offset; mean fn) | other runs: estimate (error; pred sd; covered; points within 2 SD, their data offset; mean fn) |
|---|---:|---|---|
| gmm_D2_noise3_svbmc_seed1022, component 46 (0.055) | -1.522 ± 0.092 | -1.185 (+0.337; 0.448; 23, -0.045; -37.5) | gmm_D2_noise3_svbmc_seed1064: -1.589 (-0.067; 0.716; yes; 24, 0.382; -7.6) |
| gmm_D2_noise3_svbmc_seed1064, component 11 (0.039) | -1.551 ± 0.065 | -0.952 (+0.598; 0.457; 29, 1.222; -8.4) | gmm_D2_noise3_svbmc_seed1022: -1.818 (-0.267; 0.543; yes; 12, 0.353; -30.6) |

### M = 4 (20 cells)

| Estimator | bias | bias with the arm's entropy | covered | mc_se | gp_sd (indep / corr) |
|---|---:|---:|---:|---:|---:|
| raw | 0.308 [0.194, 0.419] | 0.301 [0.183, 0.413] | - | - | - |
| capped_I | 0.005 [-0.057, 0.086] | -0.013 [-0.063, 0.077] | - | - | - |
| capped_E | 0.180 [0.109, 0.229] | 0.171 [0.116, 0.218] | - | - | - |
| honest ratio1.5 median | -0.085 [-0.208, -0.018] | -0.090 [-0.218, -0.032] | 0.599 | 0.010 | 0.092 / 0.381 |
| honest ratio1.5 precision | -0.085 [-0.207, -0.011] | -0.089 [-0.218, -0.026] | 0.599 | 0.010 | 0.087 / 0.377 |
| honest ratio1.5 mean | -0.085 [-0.208, -0.018] | -0.090 [-0.218, -0.032] | 0.599 | 0.010 | 0.087 / 0.381 |
| honest ratio2 median | -0.211 [-0.268, -0.104] | -0.213 [-0.278, -0.121] | 0.703 | 0.011 | 0.099 / 0.408 |
| honest ratio2 precision | -0.199 [-0.268, -0.085] | -0.201 [-0.278, -0.100] | 0.703 | 0.011 | 0.096 / 0.404 |
| honest ratio2 mean | -0.211 [-0.268, -0.125] | -0.213 [-0.278, -0.142] | 0.703 | 0.011 | 0.096 / 0.408 |
| honest ratio3 median | -0.248 [-0.331, -0.099] | -0.250 [-0.331, -0.116] | 0.705 | 0.011 | 0.099 / 0.415 |
| honest ratio3 precision | -0.227 [-0.309, -0.085] | -0.224 [-0.306, -0.100] | 0.705 | 0.011 | 0.096 / 0.404 |
| honest ratio3 mean | -0.248 [-0.331, -0.123] | -0.250 [-0.331, -0.140] | 0.705 | 0.011 | 0.096 / 0.415 |
| honest ratio5 median | -0.237 [-0.331, -0.099] | -0.247 [-0.331, -0.116] | 0.705 | 0.011 | 0.100 / 0.415 |
| honest ratio5 precision | -0.225 [-0.309, -0.085] | -0.220 [-0.306, -0.100] | 0.705 | 0.011 | 0.096 / 0.404 |
| honest ratio5 mean | -0.237 [-0.331, -0.123] | -0.248 [-0.331, -0.140] | 0.705 | 0.011 | 0.096 / 0.415 |
| honest cap_only median | -0.237 [-0.331, -0.099] | -0.247 [-0.331, -0.116] | 0.705 | 0.011 | 0.100 / 0.415 |
| honest cap_only precision | -0.225 [-0.309, -0.085] | -0.220 [-0.306, -0.100] | 0.705 | 0.011 | 0.096 / 0.404 |
| honest cap_only mean | -0.237 [-0.331, -0.123] | -0.248 [-0.331, -0.140] | 0.705 | 0.011 | 0.096 / 0.415 |
| honest none median | -4.637 [-5.729, -3.270] | -4.655 [-5.742, -3.271] | 1.000 | 0.012 | 0.596 / 2.493 |
| honest none precision | -2.727 [-3.397, -1.828] | -2.730 [-3.413, -1.818] | 1.000 | 0.012 | 0.247 / 1.257 |
| honest none mean | -4.983 [-6.660, -4.045] | -4.995 [-6.664, -4.062] | 1.000 | 0.011 | 0.453 / 2.459 |
| pooled (ratio2, own included) | 0.020 [-0.047, 0.086] | - | - | - | - |
| truth_strat | -0.002 [-0.005, 0.009] | - | - | - | - |

Reference `e_log_joint_mc_sd` median 0.011; the headline bias is within 2 SE in 0% of the cells; median absolute bias raw 0.308, capped_I 0.086, capped_E 0.185, honest 0.240 (against raw the honest estimate is better, against capped_I **worse**; tie band 0.011, one reference standard error). Calibration of the self-reported SDs (weighted over components): own z median 0.72 [0.43, 0.94], abs z median 0.89 [0.66, 0.96], abs z > 2 fraction 0.05 [0.02, 0.14]; cross z median -0.54 [-0.65, -0.38], abs z median 0.61 [0.53, 0.70], abs z > 2 fraction 0.01 [0.00, 0.02]. Effective training points on a component (weighted median): own run 11.9 [10.5, 12.8], covering other runs 8.4 [7.6, 10.9]. Checks: own-run max abs z 5.12 (weighted offset abs z 1.96), Jacobian max abs z 0.00 (offset abs z 0.00), raw consistency 6.7e-16, flagged cells 0.

Heaviest component of every run, repetition 0:

| Run, heaviest component (weight) | truth | own I_corr (error; BQ sd; points within 2 SD, their data offset; mean fn) | other runs: estimate (error; pred sd; covered; points within 2 SD, their data offset; mean fn) |
|---|---:|---|---|
| gmm_D2_noise3_svbmc_seed1026, component 24 (0.029) | -1.316 ± 0.045 | -0.567 (+0.749; 0.501; 16, 0.561; -14.5) | gmm_D2_noise3_svbmc_seed1047: -4.729 (-3.413; 3.828; no; 1, -1.604; -10.1); gmm_D2_noise3_svbmc_seed1079: -32.967 (-31.651; 13.849; no; 0, -; -31.7); gmm_D2_noise3_svbmc_seed1100: -7.731 (-6.415; 4.041; no; 0, -; -8.5) |
| gmm_D2_noise3_svbmc_seed1047, component 1 (0.070) | -1.967 ± 0.109 | -1.155 (+0.812; 0.587; 16, 1.108; -8.7) | gmm_D2_noise3_svbmc_seed1026: -2.613 (-0.646; 0.880; yes; 13, 0.416; -13.6); gmm_D2_noise3_svbmc_seed1079: -10.693 (-8.726; 3.329; no; 0, -; -11.3); gmm_D2_noise3_svbmc_seed1100: -10.436 (-8.470; 4.056; no; 0, -; -8.8) |
| gmm_D2_noise3_svbmc_seed1079, component 2 (0.057) | -1.709 ± 0.104 | -1.342 (+0.367; 0.393; 38, -0.114; -9.8) | gmm_D2_noise3_svbmc_seed1026: -7.347 (-5.638; 7.051; no; 0, -; -11.2); gmm_D2_noise3_svbmc_seed1047: -2.628 (-0.920; 1.193; no; 19, -0.705; -9.1); gmm_D2_noise3_svbmc_seed1100: -2.217 (-0.509; 0.833; yes; 62, -0.323; -8.7) |
| gmm_D2_noise3_svbmc_seed1100, component 26 (0.006) | -1.450 ± 0.057 | -1.156 (+0.294; 0.444; 31, -0.043; -8.6) | gmm_D2_noise3_svbmc_seed1026: -7.451 (-6.001; 6.343; no; 0, -; -11.3); gmm_D2_noise3_svbmc_seed1047: -2.187 (-0.737; 1.210; yes; 5, -0.394; -9.3); gmm_D2_noise3_svbmc_seed1079: -1.378 (+0.072; 0.557; yes; 13, -0.460; -9.1) |

### M = 8 (20 cells)

| Estimator | bias | bias with the arm's entropy | covered | mc_se | gp_sd (indep / corr) |
|---|---:|---:|---:|---:|---:|
| raw | 0.506 [0.454, 0.558] | 0.501 [0.455, 0.570] | - | - | - |
| capped_I | 0.074 [0.009, 0.111] | 0.063 [0.011, 0.112] | - | - | - |
| capped_E | 0.169 [0.141, 0.273] | 0.170 [0.147, 0.290] | - | - | - |
| honest ratio1.5 median | -0.249 [-0.318, -0.190] | -0.243 [-0.320, -0.183] | 0.889 | 0.011 | 0.081 / 0.309 |
| honest ratio1.5 precision | -0.223 [-0.287, -0.186] | -0.227 [-0.294, -0.176] | 0.889 | 0.010 | 0.064 / 0.282 |
| honest ratio1.5 mean | -0.264 [-0.321, -0.217] | -0.257 [-0.326, -0.207] | 0.889 | 0.010 | 0.066 / 0.289 |
| honest ratio2 median | -0.292 [-0.368, -0.253] | -0.291 [-0.378, -0.247] | 0.901 | 0.011 | 0.086 / 0.310 |
| honest ratio2 precision | -0.251 [-0.322, -0.210] | -0.242 [-0.334, -0.208] | 0.901 | 0.010 | 0.063 / 0.280 |
| honest ratio2 mean | -0.317 [-0.360, -0.243] | -0.303 [-0.372, -0.250] | 0.901 | 0.010 | 0.066 / 0.300 |
| honest ratio3 median | -0.330 [-0.409, -0.278] | -0.346 [-0.407, -0.271] | 0.901 | 0.010 | 0.084 / 0.312 |
| honest ratio3 precision | -0.302 [-0.349, -0.229] | -0.289 [-0.359, -0.227] | 0.901 | 0.010 | 0.063 / 0.294 |
| honest ratio3 mean | -0.358 [-0.501, -0.266] | -0.365 [-0.490, -0.277] | 0.901 | 0.010 | 0.066 / 0.318 |
| honest ratio5 median | -0.349 [-0.452, -0.285] | -0.355 [-0.447, -0.282] | 0.951 | 0.010 | 0.085 / 0.321 |
| honest ratio5 precision | -0.308 [-0.415, -0.233] | -0.310 [-0.406, -0.227] | 0.951 | 0.010 | 0.064 / 0.296 |
| honest ratio5 mean | -0.354 [-0.674, -0.289] | -0.369 [-0.667, -0.283] | 0.951 | 0.010 | 0.067 / 0.333 |
| honest cap_only median | -0.349 [-0.452, -0.285] | -0.355 [-0.447, -0.282] | 0.951 | 0.010 | 0.085 / 0.321 |
| honest cap_only precision | -0.308 [-0.415, -0.234] | -0.310 [-0.406, -0.227] | 0.951 | 0.010 | 0.064 / 0.296 |
| honest cap_only mean | -0.354 [-0.674, -0.289] | -0.369 [-0.667, -0.282] | 0.951 | 0.010 | 0.067 / 0.333 |
| honest none median | -3.527 [-4.122, -3.299] | -3.518 [-4.125, -3.287] | 1.000 | 0.012 | 0.540 / 1.874 |
| honest none precision | -1.021 [-1.636, -0.705] | -1.011 [-1.640, -0.707] | 1.000 | 0.010 | 0.073 / 0.433 |
| honest none mean | -4.956 [-5.158, -4.698] | -4.961 [-5.161, -4.705] | 1.000 | 0.007 | 0.255 / 1.529 |
| pooled (ratio2, own included) | -0.054 [-0.137, -0.035] | - | - | - | - |
| truth_strat | 0.002 [-0.008, 0.008] | - | - | - | - |

Reference `e_log_joint_mc_sd` median 0.011; the headline bias is within 2 SE in 0% of the cells; median absolute bias raw 0.506, capped_I 0.082, capped_E 0.169, honest 0.292 (against raw the honest estimate is better, against capped_I **worse**; tie band 0.011, one reference standard error). Calibration of the self-reported SDs (weighted over components): own z median 1.08 [0.95, 1.17], abs z median 1.09 [0.96, 1.20], abs z > 2 fraction 0.10 [0.06, 0.16]; cross z median -0.42 [-0.52, -0.35], abs z median 0.60 [0.51, 0.63], abs z > 2 fraction 0.01 [0.01, 0.04]. Effective training points on a component (weighted median): own run 13.0 [12.3, 13.9], covering other runs 9.2 [7.7, 10.1]. Checks: own-run max abs z 4.83 (weighted offset abs z 3.35), Jacobian max abs z 0.00 (offset abs z 0.00), raw consistency 4.4e-16, flagged cells 0.

### M = 16 (10 cells)

| Estimator | bias | bias with the arm's entropy | covered | mc_se | gp_sd (indep / corr) |
|---|---:|---:|---:|---:|---:|
| raw | 0.713 [0.572, 0.914] | 0.726 [0.562, 0.900] | - | - | - |
| capped_I | 0.035 [-0.005, 0.099] | 0.045 [-0.005, 0.096] | - | - | - |
| capped_E | 0.203 [0.172, 0.251] | 0.204 [0.162, 0.269] | - | - | - |
| honest ratio1.5 median | -0.195 [-0.280, -0.114] | -0.192 [-0.275, -0.109] | 0.999 | 0.012 | 0.081 / 0.209 |
| honest ratio1.5 precision | -0.190 [-0.226, -0.107] | -0.194 [-0.228, -0.098] | 0.999 | 0.011 | 0.047 / 0.193 |
| honest ratio1.5 mean | -0.225 [-0.319, -0.162] | -0.224 [-0.301, -0.156] | 0.999 | 0.010 | 0.052 / 0.202 |
| honest ratio2 median | -0.293 [-0.324, -0.185] | -0.276 [-0.321, -0.190] | 1.000 | 0.012 | 0.086 / 0.226 |
| honest ratio2 precision | -0.246 [-0.287, -0.191] | -0.257 [-0.275, -0.190] | 1.000 | 0.011 | 0.043 / 0.195 |
| honest ratio2 mean | -0.326 [-0.432, -0.275] | -0.331 [-0.436, -0.267] | 1.000 | 0.010 | 0.046 / 0.215 |
| honest ratio3 median | -0.306 [-0.359, -0.203] | -0.289 [-0.357, -0.208] | 1.000 | 0.012 | 0.086 / 0.232 |
| honest ratio3 precision | -0.263 [-0.304, -0.212] | -0.263 [-0.300, -0.210] | 1.000 | 0.011 | 0.043 / 0.196 |
| honest ratio3 mean | -0.416 [-0.462, -0.356] | -0.414 [-0.461, -0.354] | 1.000 | 0.010 | 0.048 / 0.231 |
| honest ratio5 median | -0.304 [-0.360, -0.245] | -0.294 [-0.361, -0.236] | 1.000 | 0.012 | 0.086 / 0.239 |
| honest ratio5 precision | -0.267 [-0.305, -0.229] | -0.263 [-0.301, -0.228] | 1.000 | 0.011 | 0.043 / 0.196 |
| honest ratio5 mean | -0.448 [-0.507, -0.430] | -0.448 [-0.516, -0.415] | 1.000 | 0.010 | 0.049 / 0.237 |
| honest cap_only median | -0.304 [-0.360, -0.246] | -0.295 [-0.361, -0.240] | 1.000 | 0.012 | 0.085 / 0.239 |
| honest cap_only precision | -0.267 [-0.305, -0.230] | -0.263 [-0.301, -0.230] | 1.000 | 0.011 | 0.043 / 0.196 |
| honest cap_only mean | -0.457 [-0.516, -0.430] | -0.458 [-0.519, -0.415] | 1.000 | 0.010 | 0.049 / 0.237 |
| honest none median | -3.693 [-4.418, -3.024] | -3.687 [-4.409, -3.021] | 1.000 | 0.013 | 0.491 / 1.543 |
| honest none precision | -0.543 [-0.632, -0.488] | -0.538 [-0.633, -0.488] | 1.000 | 0.010 | 0.042 / 0.238 |
| honest none mean | -4.617 [-5.305, -4.363] | -4.627 [-5.297, -4.359] | 1.000 | 0.006 | 0.163 / 0.993 |
| pooled (ratio2, own included) | -0.148 [-0.187, -0.074] | - | - | - | - |
| truth_strat | -0.009 [-0.014, 0.000] | - | - | - | - |

Reference `e_log_joint_mc_sd` median 0.011; the headline bias is within 2 SE in 0% of the cells; median absolute bias raw 0.713, capped_I 0.035, capped_E 0.203, honest 0.293 (against raw the honest estimate is better, against capped_I **worse**; tie band 0.011, one reference standard error). Calibration of the self-reported SDs (weighted over components): own z median 1.34 [1.17, 1.92], abs z median 1.34 [1.18, 1.92], abs z > 2 fraction 0.21 [0.09, 0.37]; cross z median -0.31 [-0.42, -0.28], abs z median 0.52 [0.51, 0.61], abs z > 2 fraction 0.03 [0.02, 0.04]. Effective training points on a component (weighted median): own run 13.4 [11.6, 14.8], covering other runs 8.5 [7.2, 9.7]. Checks: own-run max abs z 5.00 (weighted offset abs z 1.47), Jacobian max abs z 0.00 (offset abs z 0.00), raw consistency 2.2e-16, flagged cells 0.

| Run | own error | se | own z median | own abs z median | BQ sd median | pred sd median | own-run max abs z (offset z) | Jacobian max abs z (offset z) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gmm_D2_noise3_svbmc_seed1000 | 0.403 | 0.019 | 1.04 | 1.06 | 0.435 | 0.711 | 2.74 (-2.10) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1001 | -0.103 | 0.031 | -0.13 | 0.36 | 0.531 | 0.796 | 2.03 (-1.18) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1002 | -0.060 | 0.020 | 0.05 | 0.46 | 0.575 | 0.791 | 2.15 (+0.37) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1003 | -0.052 | 0.026 | -0.07 | 0.59 | 0.498 | 0.702 | 2.84 (+0.31) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1004 | -0.206 | 0.015 | -0.34 | 0.44 | 0.422 | 0.563 | 2.91 (-0.71) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1005 | -0.223 | 0.014 | -0.47 | 0.49 | 0.565 | 0.738 | 2.28 (+0.79) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1006 | 0.044 | 0.014 | 0.35 | 0.83 | 0.496 | 0.680 | 3.36 (+0.38) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1007 | 0.270 | 0.012 | 0.60 | 0.60 | 0.466 | 0.687 | 3.01 (-0.06) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1008 | -0.331 | 0.024 | -0.88 | 0.89 | 0.414 | 0.593 | 3.55 (-0.08) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1009 | 0.116 | 0.011 | 0.35 | 0.45 | 0.507 | 0.774 | 2.54 (+0.97) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1010 | 0.375 | 0.019 | 0.68 | 0.68 | 0.588 | 0.857 | 3.17 (+0.04) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1011 | -0.024 | 0.020 | -0.00 | 0.20 | 0.357 | 0.478 | 2.21 (-0.35) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1012 | 0.492 | 0.014 | 1.10 | 1.11 | 0.424 | 0.515 | 3.31 (+1.31) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1013 | 0.336 | 0.021 | 0.55 | 0.58 | 0.466 | 0.640 | 2.85 (-1.09) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1015 | 0.030 | 0.016 | 0.02 | 0.37 | 0.416 | 0.525 | 3.95 (+2.18) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1016 | -0.164 | 0.016 | -0.50 | 0.54 | 0.406 | 0.580 | 2.67 (+1.01) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1017 | 0.408 | 0.026 | 1.07 | 1.07 | 0.481 | 0.615 | 2.51 (+3.24) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1018 | -0.019 | 0.021 | -0.15 | 0.36 | 0.505 | 0.742 | 2.37 (-0.10) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1019 | 0.284 | 0.021 | 0.72 | 0.79 | 0.457 | 0.561 | 2.00 (+0.28) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1020 | 0.083 | 0.020 | 0.19 | 0.52 | 0.562 | 0.770 | 3.98 (-0.28) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1022 | 0.233 | 0.019 | 0.41 | 0.49 | 0.456 | 0.547 | 2.31 (-0.57) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1024 | -0.003 | 0.017 | 0.16 | 0.70 | 0.428 | 0.580 | 2.69 (-0.53) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1025 | 0.713 | 0.025 | 1.58 | 1.58 | 0.418 | 0.552 | 2.09 (-0.01) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1026 | 0.409 | 0.014 | 0.96 | 1.03 | 0.598 | 0.767 | 3.90 (-0.33) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1027 | 0.018 | 0.015 | 0.10 | 0.48 | 0.488 | 0.614 | 2.67 (-0.66) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1028 | 0.410 | 0.016 | 1.10 | 1.10 | 0.383 | 0.455 | 2.95 (-0.34) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1029 | 0.799 | 0.015 | 1.16 | 1.16 | 0.600 | 0.821 | 3.20 (+0.61) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1030 | 0.408 | 0.013 | 0.97 | 0.97 | 0.476 | 0.734 | 2.75 (-0.59) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1031 | -0.047 | 0.016 | -0.39 | 1.05 | 0.458 | 0.609 | 3.54 (+1.13) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1032 | 0.026 | 0.024 | 0.16 | 0.64 | 0.465 | 0.592 | 3.25 (+1.57) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1033 | 0.264 | 0.017 | 0.74 | 0.74 | 0.433 | 0.600 | 2.59 (+0.89) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1034 | 0.432 | 0.026 | 0.75 | 0.80 | 0.444 | 0.593 | 3.11 (+2.81) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1035 | -0.380 | 0.021 | -0.76 | 0.73 | 0.565 | 0.741 | 2.24 (+1.04) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1036 | 0.250 | 0.028 | 0.45 | 1.02 | 0.437 | 0.596 | 2.74 (-1.46) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1037 | 0.096 | 0.020 | 0.16 | 0.36 | 0.542 | 0.823 | 2.81 (+2.55) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1038 | -0.005 | 0.026 | -0.01 | 0.43 | 0.598 | 0.836 | 3.14 (+0.44) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1039 | 0.263 | 0.042 | 0.76 | 0.81 | 0.514 | 0.670 | 3.13 (-0.95) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1040 | -0.170 | 0.042 | 0.02 | 0.41 | 0.520 | 0.619 | 3.26 (+0.34) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1041 | 0.628 | 0.018 | 1.43 | 1.43 | 0.610 | 0.868 | 2.57 (-1.65) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1042 | 0.205 | 0.021 | 0.49 | 0.74 | 0.554 | 0.776 | 2.02 (+0.28) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1043 | -0.009 | 0.024 | -0.11 | 0.43 | 0.534 | 0.741 | 2.67 (+0.48) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1044 | 0.130 | 0.019 | 0.52 | 0.77 | 0.426 | 0.538 | 3.05 (-0.34) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1045 | -0.364 | 0.017 | -1.07 | 0.96 | 0.458 | 0.637 | 2.46 (+0.19) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1046 | 0.201 | 0.020 | 0.46 | 0.48 | 0.583 | 0.936 | 3.29 (+1.27) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1047 | 0.173 | 0.027 | 0.03 | 0.59 | 0.731 | 1.127 | 2.37 (-0.80) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1048 | 1.028 | 0.014 | 2.55 | 2.55 | 0.424 | 0.576 | 2.36 (-1.60) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1049 | 0.539 | 0.017 | 1.42 | 1.46 | 0.543 | 0.709 | 2.93 (+1.66) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1050 | -0.118 | 0.020 | 0.23 | 0.32 | 0.625 | 0.858 | 3.16 (-0.73) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1051 | 0.416 | 0.013 | 1.15 | 1.15 | 0.517 | 0.693 | 2.63 (-0.69) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1052 | 0.143 | 0.018 | 0.68 | 0.79 | 0.445 | 0.703 | 2.19 (-1.46) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1053 | 0.563 | 0.017 | 1.94 | 1.94 | 0.361 | 0.459 | 2.46 (+0.14) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1054 | 0.220 | 0.019 | 0.65 | 0.68 | 0.353 | 0.564 | 2.09 (-1.24) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1055 | 0.964 | 0.018 | 1.79 | 1.79 | 0.602 | 0.913 | 2.60 (-0.14) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1056 | -0.293 | 0.017 | -0.74 | 0.73 | 0.460 | 0.611 | 2.14 (-0.43) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1058 | 0.193 | 0.016 | 0.79 | 1.02 | 0.432 | 0.577 | 2.55 (+0.01) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1059 | 0.063 | 0.022 | -0.42 | 0.66 | 0.450 | 0.623 | 3.27 (-0.45) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1060 | -0.156 | 0.014 | -0.12 | 0.84 | 0.560 | 0.762 | 2.71 (+0.08) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1061 | 0.228 | 0.020 | 0.35 | 0.62 | 0.540 | 0.931 | 3.56 (-1.32) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1062 | 0.352 | 0.016 | 0.44 | 0.48 | 0.597 | 0.968 | 3.21 (-1.43) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1063 | 0.212 | 0.015 | 0.62 | 0.65 | 0.413 | 0.565 | 3.50 (+2.51) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1064 | 0.067 | 0.022 | -0.16 | 0.36 | 0.482 | 0.667 | 2.63 (+0.67) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1065 | -0.273 | 0.025 | -0.88 | 0.95 | 0.450 | 0.620 | 2.76 (+0.18) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1066 | -0.004 | 0.018 | -0.19 | 0.54 | 0.473 | 0.660 | 2.75 (+0.75) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1067 | -0.003 | 0.013 | -0.03 | 0.37 | 0.540 | 0.669 | 2.33 (-0.02) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1068 | 0.135 | 0.020 | -0.23 | 0.68 | 0.417 | 0.515 | 3.02 (-0.75) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1069 | -0.243 | 0.015 | -0.64 | 0.69 | 0.431 | 0.533 | 2.31 (-0.95) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1070 | 0.324 | 0.020 | 0.68 | 0.69 | 0.522 | 0.702 | 2.37 (+2.69) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1071 | 0.435 | 0.022 | 1.29 | 1.29 | 0.542 | 0.878 | 2.61 (-1.50) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1072 | 0.389 | 0.022 | 0.73 | 0.77 | 0.533 | 0.746 | 3.11 (+0.70) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1073 | 0.348 | 0.021 | 0.98 | 1.11 | 0.534 | 0.772 | 2.80 (-0.36) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1074 | 0.478 | 0.018 | 1.25 | 1.25 | 0.573 | 0.909 | 3.53 (+1.05) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1076 | -0.094 | 0.018 | -0.23 | 0.41 | 0.748 | 0.958 | 2.38 (+0.65) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1078 | 0.260 | 0.017 | 0.81 | 0.88 | 0.545 | 0.773 | 2.59 (+0.29) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1079 | -0.095 | 0.026 | -0.26 | 0.66 | 0.521 | 0.696 | 2.91 (+0.72) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1080 | 0.459 | 0.019 | 0.57 | 0.57 | 0.521 | 0.703 | 2.04 (-0.00) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1081 | 0.006 | 0.028 | -0.21 | 0.68 | 0.922 | 1.288 | 4.50 (-1.45) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1082 | 0.082 | 0.019 | 0.09 | 0.16 | 0.529 | 0.825 | 2.57 (+0.39) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1083 | 0.368 | 0.022 | 1.24 | 1.25 | 0.438 | 0.565 | 3.45 (-0.08) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1084 | 0.344 | 0.020 | 0.62 | 0.62 | 0.496 | 0.648 | 2.84 (+1.75) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1085 | 0.090 | 0.023 | 0.06 | 0.20 | 0.443 | 0.516 | 3.11 (-0.48) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1086 | 0.143 | 0.015 | 0.42 | 0.54 | 0.441 | 0.638 | 2.06 (+0.51) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1087 | 0.088 | 0.019 | 0.14 | 0.38 | 0.474 | 0.712 | 3.24 (-0.30) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1088 | -0.334 | 0.020 | -0.64 | 0.64 | 0.669 | 0.933 | 3.96 (+0.16) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1089 | 0.671 | 0.020 | 1.56 | 1.56 | 0.441 | 0.746 | 2.37 (-1.64) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1090 | 0.019 | 0.026 | 0.45 | 0.68 | 0.621 | 0.827 | 2.01 (+1.41) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1091 | 0.235 | 0.011 | 0.35 | 0.96 | 0.434 | 0.538 | 2.92 (+0.54) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1092 | -0.326 | 0.022 | -0.47 | 0.47 | 0.565 | 0.799 | 2.70 (-1.20) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1093 | -0.045 | 0.021 | 0.01 | 0.31 | 0.501 | 0.678 | 2.45 (+0.51) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1094 | 0.198 | 0.020 | 0.79 | 1.02 | 0.526 | 0.742 | 2.20 (+1.29) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1095 | 0.014 | 0.021 | 0.34 | 0.37 | 0.388 | 0.653 | 1.72 (-0.46) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1096 | 0.956 | 0.014 | 2.92 | 2.92 | 0.460 | 0.645 | 2.44 (-0.38) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1097 | 0.047 | 0.021 | 0.05 | 0.35 | 0.598 | 0.818 | 2.24 (-1.47) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1098 | 0.485 | 0.013 | 1.30 | 1.30 | 0.463 | 0.685 | 2.07 (-2.27) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1099 | 0.268 | 0.015 | 0.82 | 0.84 | 0.426 | 0.544 | 2.06 (-0.47) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1100 | -0.050 | 0.017 | 0.15 | 0.65 | 0.458 | 0.723 | 2.31 (+0.36) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1101 | 0.413 | 0.011 | 1.23 | 1.23 | 0.449 | 0.638 | 2.54 (-0.09) | 0.00 (+0.00) |
| gmm_D2_noise3_svbmc_seed1102 | 0.540 | 0.026 | 1.93 | 1.98 | 0.576 | 0.728 | 2.00 (+0.78) | 0.00 (+0.00) |

## gmm_D2_svbmc (noiseless)

### M = 2 (20 cells)

| Estimator | bias | bias with the arm's entropy | covered | mc_se | gp_sd (indep / corr) |
|---|---:|---:|---:|---:|---:|
| raw | 0.009 [0.008, 0.015] | 0.007 [-0.009, 0.027] | - | - | - |
| capped_I | -0.002 [-0.030, 0.009] | -0.018 [-0.039, 0.007] | - | - | - |
| capped_E | 0.008 [-0.002, 0.009] | 0.004 [-0.014, 0.021] | - | - | - |
| honest ratio1.5 median | 0.009 [0.006, 0.014] | 0.000 [-0.010, 0.026] | 0.271 | 0.004 | 0.003 / 0.012 |
| honest ratio1.5 precision | 0.009 [0.006, 0.014] | 0.000 [-0.010, 0.026] | 0.271 | 0.004 | 0.003 / 0.012 |
| honest ratio1.5 mean | 0.009 [0.006, 0.014] | 0.000 [-0.010, 0.026] | 0.271 | 0.004 | 0.003 / 0.012 |
| honest ratio2 median | 0.009 [0.005, 0.013] | 0.001 [-0.010, 0.026] | 0.282 | 0.005 | 0.003 / 0.012 |
| honest ratio2 precision | 0.009 [0.005, 0.013] | 0.001 [-0.010, 0.026] | 0.282 | 0.005 | 0.003 / 0.012 |
| honest ratio2 mean | 0.009 [0.005, 0.013] | 0.001 [-0.010, 0.026] | 0.282 | 0.005 | 0.003 / 0.012 |
| honest ratio3 median | 0.008 [0.005, 0.013] | 0.001 [-0.011, 0.026] | 0.282 | 0.005 | 0.003 / 0.013 |
| honest ratio3 precision | 0.008 [0.005, 0.013] | 0.001 [-0.011, 0.026] | 0.282 | 0.005 | 0.003 / 0.013 |
| honest ratio3 mean | 0.008 [0.005, 0.013] | 0.001 [-0.011, 0.026] | 0.282 | 0.005 | 0.003 / 0.013 |
| honest ratio5 median | 0.008 [0.003, 0.013] | 0.001 [-0.011, 0.026] | 0.287 | 0.005 | 0.005 / 0.015 |
| honest ratio5 precision | 0.008 [0.003, 0.013] | 0.001 [-0.011, 0.026] | 0.287 | 0.005 | 0.005 / 0.015 |
| honest ratio5 mean | 0.008 [0.003, 0.013] | 0.001 [-0.011, 0.026] | 0.287 | 0.005 | 0.005 / 0.015 |
| honest cap_only median | -0.001 [-0.016, 0.009] | -0.008 [-0.031, 0.010] | 0.345 | 0.006 | 0.013 / 0.024 |
| honest cap_only precision | -0.001 [-0.016, 0.009] | -0.008 [-0.031, 0.010] | 0.345 | 0.006 | 0.013 / 0.024 |
| honest cap_only mean | -0.001 [-0.016, 0.009] | -0.008 [-0.031, 0.010] | 0.345 | 0.006 | 0.013 / 0.024 |
| honest none median | -3.808 [-4.818, -2.861] | -3.805 [-4.831, -2.843] | 1.000 | 0.016 | 0.533 / 1.880 |
| honest none precision | -3.808 [-4.818, -2.861] | -3.805 [-4.831, -2.843] | 1.000 | 0.016 | 0.533 / 1.880 |
| honest none mean | -3.808 [-4.818, -2.861] | -3.805 [-4.831, -2.843] | 1.000 | 0.016 | 0.533 / 1.880 |
| pooled (ratio2, own included) | 0.003 [-0.007, 0.022] | - | - | - | - |
| truth_strat | -0.005 [-0.009, 0.019] | - | - | - | - |

Reference `e_log_joint_mc_sd` median 0.011; the headline bias is within 2 SE in 85% of the cells; median absolute bias raw 0.009, capped_I 0.017, capped_E 0.009, honest 0.009 (against raw the honest estimate is a tie, against capped_I a tie; tie band 0.011, one reference standard error). Calibration of the self-reported SDs (weighted over components): own z median 0.07 [-0.09, 0.17], abs z median 0.69 [0.64, 0.74], abs z > 2 fraction 0.05 [0.03, 0.06]; cross z median 0.00 [-0.02, 0.01], abs z median 0.14 [0.12, 0.16], abs z > 2 fraction 0.00 [0.00, 0.00]. Effective training points on a component (weighted median): own run 6.2 [5.8, 6.9], covering other runs 6.2 [5.4, 7.5]. Checks: own-run max abs z 3.88 (weighted offset abs z 2.11), Jacobian max abs z 0.00 (offset abs z 0.00), raw consistency 4.4e-16, flagged cells 0.

Heaviest component of every run, repetition 0:

| Run, heaviest component (weight) | truth | own I_corr (error; BQ sd; points within 2 SD, their data offset; mean fn) | other runs: estimate (error; pred sd; covered; points within 2 SD, their data offset; mean fn) |
|---|---:|---|---|
| gmm_D2_svbmc_seed1027, component 0 (0.101) | -1.223 ± 0.074 | -1.169 (+0.054; 0.002; 11, -0.000; -8.2) | gmm_D2_svbmc_seed1044: -3.183 (-1.960; 1.923; no; 1, 0.000; -7.9) |
| gmm_D2_svbmc_seed1044, component 4 (0.033) | -2.348 ± 0.129 | -2.331 (+0.016; 0.003; 14, 0.000; -7.7) | gmm_D2_svbmc_seed1027: -2.346 (+0.002; 0.204; no; 8, 0.000; -8.2) |

### M = 4 (20 cells)

| Estimator | bias | bias with the arm's entropy | covered | mc_se | gp_sd (indep / corr) |
|---|---:|---:|---:|---:|---:|
| raw | 0.011 [0.004, 0.019] | 0.001 [-0.004, 0.014] | - | - | - |
| capped_I | -0.007 [-0.041, 0.008] | -0.009 [-0.040, 0.001] | - | - | - |
| capped_E | 0.004 [-0.005, 0.015] | -0.003 [-0.012, 0.008] | - | - | - |
| honest ratio1.5 median | 0.005 [-0.000, 0.014] | -0.005 [-0.011, 0.018] | 0.622 | 0.008 | 0.005 / 0.018 |
| honest ratio1.5 precision | 0.005 [-0.000, 0.014] | -0.006 [-0.011, 0.018] | 0.622 | 0.008 | 0.005 / 0.016 |
| honest ratio1.5 mean | 0.005 [-0.000, 0.014] | -0.006 [-0.011, 0.018] | 0.622 | 0.008 | 0.005 / 0.018 |
| honest ratio2 median | 0.005 [-0.001, 0.014] | -0.007 [-0.012, 0.018] | 0.631 | 0.008 | 0.005 / 0.020 |
| honest ratio2 precision | 0.005 [-0.001, 0.014] | -0.006 [-0.012, 0.018] | 0.631 | 0.008 | 0.005 / 0.017 |
| honest ratio2 mean | 0.005 [-0.001, 0.014] | -0.007 [-0.012, 0.017] | 0.631 | 0.008 | 0.005 / 0.020 |
| honest ratio3 median | 0.004 [-0.001, 0.013] | -0.007 [-0.012, 0.017] | 0.641 | 0.008 | 0.006 / 0.021 |
| honest ratio3 precision | 0.004 [-0.001, 0.013] | -0.007 [-0.013, 0.017] | 0.641 | 0.008 | 0.005 / 0.018 |
| honest ratio3 mean | 0.004 [-0.001, 0.013] | -0.007 [-0.012, 0.017] | 0.641 | 0.008 | 0.006 / 0.021 |
| honest ratio5 median | 0.003 [-0.002, 0.013] | -0.008 [-0.014, 0.017] | 0.645 | 0.008 | 0.006 / 0.024 |
| honest ratio5 precision | 0.002 [-0.002, 0.013] | -0.008 [-0.015, 0.017] | 0.645 | 0.008 | 0.006 / 0.020 |
| honest ratio5 mean | 0.003 [-0.002, 0.013] | -0.008 [-0.014, 0.017] | 0.645 | 0.008 | 0.006 / 0.024 |
| honest cap_only median | -0.107 [-0.143, -0.035] | -0.107 [-0.156, -0.034] | 0.788 | 0.010 | 0.051 / 0.126 |
| honest cap_only precision | -0.034 [-0.076, 0.002] | -0.033 [-0.092, -0.007] | 0.788 | 0.010 | 0.029 / 0.058 |
| honest cap_only mean | -0.114 [-0.176, -0.035] | -0.123 [-0.163, -0.035] | 0.788 | 0.010 | 0.053 / 0.131 |
| honest none median | -2.959 [-3.744, -2.499] | -2.946 [-3.746, -2.517] | 1.000 | 0.012 | 0.397 / 1.436 |
| honest none precision | -1.308 [-2.183, -0.624] | -1.325 [-2.202, -0.615] | 1.000 | 0.011 | 0.170 / 0.555 |
| honest none mean | -3.536 [-4.985, -3.178] | -3.547 [-4.976, -3.201] | 1.000 | 0.010 | 0.254 / 1.303 |
| pooled (ratio2, own included) | 0.007 [-0.001, 0.010] | - | - | - | - |
| truth_strat | -0.002 [-0.007, 0.005] | - | - | - | - |

Reference `e_log_joint_mc_sd` median 0.012; the headline bias is within 2 SE in 90% of the cells; median absolute bias raw 0.013, capped_I 0.023, capped_E 0.016, honest 0.007 (against raw the honest estimate is a tie, against capped_I better; tie band 0.012, one reference standard error). Calibration of the self-reported SDs (weighted over components): own z median 0.05 [-0.13, 0.11], abs z median 0.64 [0.61, 0.69], abs z > 2 fraction 0.06 [0.04, 0.09]; cross z median -0.01 [-0.03, 0.01], abs z median 0.13 [0.13, 0.14], abs z > 2 fraction 0.00 [0.00, 0.00]. Effective training points on a component (weighted median): own run 6.4 [5.9, 6.7], covering other runs 6.1 [5.6, 6.3]. Checks: own-run max abs z 5.48 (weighted offset abs z 2.41), Jacobian max abs z 0.00 (offset abs z 0.00), raw consistency 6.7e-16, flagged cells 0.

Heaviest component of every run, repetition 0:

| Run, heaviest component (weight) | truth | own I_corr (error; BQ sd; points within 2 SD, their data offset; mean fn) | other runs: estimate (error; pred sd; covered; points within 2 SD, their data offset; mean fn) |
|---|---:|---|---|
| gmm_D2_svbmc_seed1003, component 0 (0.019) | -1.259 ± 0.069 | -1.203 (+0.056; 0.002; 10, -0.000; -7.3) | gmm_D2_svbmc_seed1009: -6.259 (-4.999; 3.486; no; 0, -; -8.2); gmm_D2_svbmc_seed1010: -1.263 (-0.004; 0.017; yes; 26, -0.000; -7.8); gmm_D2_svbmc_seed1020: -10.550 (-9.291; 4.527; no; 0, -; -10.5) |
| gmm_D2_svbmc_seed1009, component 5 (0.011) | -0.977 ± 0.057 | -0.993 (-0.016; 0.002; 15, -0.000; -8.3) | gmm_D2_svbmc_seed1003: -0.979 (-0.002; 0.025; yes; 8, -0.000; -7.5); gmm_D2_svbmc_seed1010: -1.828 (-0.851; 1.558; no; 1, 0.000; -7.7); gmm_D2_svbmc_seed1020: -9.277 (-8.300; 4.270; no; 0, -; -10.3) |
| gmm_D2_svbmc_seed1010, component 0 (0.015) | -1.136 ± 0.059 | -1.136 (+0.001; 0.006; 22, 0.000; -7.8) | gmm_D2_svbmc_seed1003: -1.138 (-0.002; 0.015; yes; 12, -0.000; -7.3); gmm_D2_svbmc_seed1009: -6.202 (-5.066; 3.507; no; 0, -; -8.2); gmm_D2_svbmc_seed1020: -10.516 (-9.379; 4.522; no; 0, -; -10.5) |
| gmm_D2_svbmc_seed1020, component 1 (0.052) | -1.294 ± 0.074 | -1.275 (+0.019; 0.002; 10, 0.000; -9.7) | gmm_D2_svbmc_seed1003: -7.301 (-6.008; 3.759; no; 0, -; -7.3); gmm_D2_svbmc_seed1009: -9.153 (-7.860; 3.960; no; 0, -; -9.2); gmm_D2_svbmc_seed1010: -1.869 (-0.575; 0.967; no; 1, -0.000; -7.4) |

### M = 8 (20 cells)

| Estimator | bias | bias with the arm's entropy | covered | mc_se | gp_sd (indep / corr) |
|---|---:|---:|---:|---:|---:|
| raw | 0.006 [-0.001, 0.012] | 0.002 [-0.005, 0.008] | - | - | - |
| capped_I | -0.017 [-0.051, 0.002] | -0.027 [-0.065, 0.001] | - | - | - |
| capped_E | -0.001 [-0.008, 0.009] | -0.005 [-0.009, 0.003] | - | - | - |
| honest ratio1.5 median | -0.002 [-0.006, 0.005] | -0.005 [-0.011, 0.000] | 0.981 | 0.008 | 0.005 / 0.018 |
| honest ratio1.5 precision | -0.002 [-0.006, 0.006] | -0.005 [-0.011, 0.001] | 0.981 | 0.008 | 0.004 / 0.016 |
| honest ratio1.5 mean | -0.002 [-0.006, 0.005] | -0.005 [-0.010, 0.000] | 0.981 | 0.008 | 0.004 / 0.019 |
| honest ratio2 median | -0.001 [-0.007, 0.005] | -0.005 [-0.012, -0.000] | 0.994 | 0.008 | 0.005 / 0.020 |
| honest ratio2 precision | -0.001 [-0.007, 0.005] | -0.005 [-0.012, 0.000] | 0.994 | 0.008 | 0.004 / 0.017 |
| honest ratio2 mean | -0.002 [-0.008, 0.005] | -0.005 [-0.012, -0.000] | 0.994 | 0.008 | 0.004 / 0.020 |
| honest ratio3 median | -0.001 [-0.007, 0.004] | -0.005 [-0.012, -0.000] | 0.998 | 0.008 | 0.005 / 0.021 |
| honest ratio3 precision | -0.001 [-0.007, 0.005] | -0.005 [-0.012, 0.000] | 0.998 | 0.008 | 0.004 / 0.017 |
| honest ratio3 mean | -0.002 [-0.007, 0.004] | -0.005 [-0.012, -0.000] | 0.998 | 0.008 | 0.004 / 0.022 |
| honest ratio5 median | -0.002 [-0.007, 0.004] | -0.006 [-0.013, -0.000] | 1.000 | 0.008 | 0.006 / 0.022 |
| honest ratio5 precision | -0.001 [-0.007, 0.004] | -0.005 [-0.012, 0.000] | 1.000 | 0.008 | 0.004 / 0.017 |
| honest ratio5 mean | -0.003 [-0.007, 0.004] | -0.008 [-0.013, -0.000] | 1.000 | 0.008 | 0.004 / 0.023 |
| honest cap_only median | -0.051 [-0.069, -0.028] | -0.055 [-0.083, -0.037] | 1.000 | 0.008 | 0.032 / 0.107 |
| honest cap_only precision | -0.006 [-0.012, 0.001] | -0.009 [-0.026, -0.004] | 1.000 | 0.008 | 0.004 / 0.019 |
| honest cap_only mean | -0.117 [-0.196, -0.099] | -0.122 [-0.195, -0.112] | 1.000 | 0.009 | 0.035 / 0.149 |
| honest none median | -2.571 [-3.009, -2.262] | -2.573 [-3.010, -2.281] | 1.000 | 0.011 | 0.259 / 1.330 |
| honest none precision | -0.010 [-0.021, -0.006] | -0.019 [-0.036, -0.009] | 1.000 | 0.008 | 0.004 / 0.021 |
| honest none mean | -3.890 [-4.711, -3.611] | -3.893 [-4.716, -3.615] | 1.000 | 0.006 | 0.114 / 0.937 |
| pooled (ratio2, own included) | 0.000 [-0.005, 0.006] | - | - | - | - |
| truth_strat | -0.004 [-0.010, 0.005] | - | - | - | - |

Reference `e_log_joint_mc_sd` median 0.011; the headline bias is within 2 SE in 95% of the cells; median absolute bias raw 0.007, capped_I 0.028, capped_E 0.009, honest 0.008 (against raw the honest estimate is a tie, against capped_I better; tie band 0.011, one reference standard error). Calibration of the self-reported SDs (weighted over components): own z median -0.03 [-0.07, 0.09], abs z median 0.73 [0.67, 0.76], abs z > 2 fraction 0.05 [0.05, 0.06]; cross z median -0.01 [-0.02, 0.00], abs z median 0.13 [0.12, 0.14], abs z > 2 fraction 0.00 [0.00, 0.00]. Effective training points on a component (weighted median): own run 6.1 [5.9, 6.3], covering other runs 6.1 [5.9, 6.2]. Checks: own-run max abs z 6.41 (weighted offset abs z 2.51), Jacobian max abs z 0.00 (offset abs z 0.00), raw consistency 2.2e-16, flagged cells 1.

### M = 16 (10 cells)

| Estimator | bias | bias with the arm's entropy | covered | mc_se | gp_sd (indep / corr) |
|---|---:|---:|---:|---:|---:|
| raw | 0.009 [-0.003, 0.015] | 0.008 [-0.003, 0.016] | - | - | - |
| capped_I | -0.020 [-0.055, 0.002] | -0.020 [-0.056, 0.004] | - | - | - |
| capped_E | 0.002 [-0.005, 0.010] | 0.001 [-0.006, 0.011] | - | - | - |
| honest ratio1.5 median | 0.007 [-0.009, 0.013] | 0.004 [-0.007, 0.016] | 0.997 | 0.006 | 0.003 / 0.013 |
| honest ratio1.5 precision | 0.006 [-0.009, 0.012] | 0.003 [-0.007, 0.016] | 0.997 | 0.006 | 0.001 / 0.010 |
| honest ratio1.5 mean | 0.006 [-0.010, 0.012] | 0.003 [-0.008, 0.015] | 0.997 | 0.006 | 0.002 / 0.013 |
| honest ratio2 median | 0.007 [-0.009, 0.013] | 0.004 [-0.007, 0.016] | 0.999 | 0.006 | 0.003 / 0.013 |
| honest ratio2 precision | 0.006 [-0.010, 0.012] | 0.004 [-0.007, 0.016] | 0.999 | 0.006 | 0.001 / 0.011 |
| honest ratio2 mean | 0.006 [-0.010, 0.012] | 0.003 [-0.008, 0.015] | 0.999 | 0.006 | 0.002 / 0.014 |
| honest ratio3 median | 0.007 [-0.009, 0.013] | 0.004 [-0.007, 0.016] | 1.000 | 0.006 | 0.003 / 0.014 |
| honest ratio3 precision | 0.006 [-0.010, 0.012] | 0.004 [-0.007, 0.016] | 1.000 | 0.006 | 0.001 / 0.011 |
| honest ratio3 mean | 0.006 [-0.010, 0.012] | 0.003 [-0.008, 0.015] | 1.000 | 0.006 | 0.002 / 0.015 |
| honest ratio5 median | 0.007 [-0.009, 0.013] | 0.004 [-0.007, 0.016] | 1.000 | 0.006 | 0.003 / 0.014 |
| honest ratio5 precision | 0.006 [-0.010, 0.012] | 0.003 [-0.007, 0.016] | 1.000 | 0.006 | 0.001 / 0.011 |
| honest ratio5 mean | 0.005 [-0.012, 0.011] | 0.003 [-0.010, 0.015] | 1.000 | 0.006 | 0.002 / 0.016 |
| honest cap_only median | 0.004 [-0.012, 0.008] | 0.002 [-0.011, 0.011] | 1.000 | 0.006 | 0.005 / 0.017 |
| honest cap_only precision | 0.006 [-0.010, 0.012] | 0.003 [-0.008, 0.016] | 1.000 | 0.006 | 0.001 / 0.011 |
| honest cap_only mean | -0.098 [-0.123, -0.088] | -0.103 [-0.125, -0.085] | 1.000 | 0.006 | 0.014 / 0.099 |
| honest none median | -1.986 [-2.735, -1.160] | -1.988 [-2.650, -1.159] | 1.000 | 0.007 | 0.159 / 0.926 |
| honest none precision | 0.003 [-0.013, 0.009] | -0.000 [-0.013, 0.013] | 1.000 | 0.006 | 0.001 / 0.011 |
| honest none mean | -3.743 [-4.611, -3.001] | -3.744 [-4.618, -3.002] | 1.000 | 0.004 | 0.049 / 0.584 |
| pooled (ratio2, own included) | 0.008 [-0.008, 0.013] | - | - | - | - |
| truth_strat | 0.003 [-0.014, 0.009] | - | - | - | - |

Reference `e_log_joint_mc_sd` median 0.011; the headline bias is within 2 SE in 100% of the cells; median absolute bias raw 0.009, capped_I 0.021, capped_E 0.007, honest 0.009 (against raw the honest estimate is a tie, against capped_I better; tie band 0.011, one reference standard error). Calibration of the self-reported SDs (weighted over components): own z median 0.00 [-0.06, 0.04], abs z median 0.68 [0.64, 0.74], abs z > 2 fraction 0.05 [0.04, 0.06]; cross z median -0.01 [-0.02, 0.00], abs z median 0.13 [0.12, 0.14], abs z > 2 fraction 0.00 [0.00, 0.00]. Effective training points on a component (weighted median): own run 6.0 [5.8, 6.1], covering other runs 5.9 [5.7, 6.1]. Checks: own-run max abs z 4.84 (weighted offset abs z 0.86), Jacobian max abs z 0.00 (offset abs z 0.00), raw consistency 2.2e-16, flagged cells 0.

| Run | own error | se | own z median | own abs z median | BQ sd median | pred sd median | own-run max abs z (offset z) | Jacobian max abs z (offset z) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gmm_D2_svbmc_seed1000 | 0.029 | 0.023 | 0.49 | 0.86 | 0.004 | 0.025 | 2.45 (-1.14) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1001 | -0.004 | 0.015 | 0.10 | 0.60 | 0.005 | 0.044 | 3.08 (+0.19) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1002 | 0.045 | 0.022 | 0.57 | 1.02 | 0.005 | 0.037 | 2.97 (-1.62) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1003 | 0.008 | 0.019 | -0.07 | 0.58 | 0.007 | 0.050 | 2.62 (-0.46) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1004 | 0.004 | 0.019 | -0.06 | 0.50 | 0.003 | 0.020 | 3.15 (-0.16) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1005 | 0.013 | 0.018 | -0.05 | 0.46 | 0.005 | 0.033 | 3.61 (-0.12) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1006 | 0.024 | 0.017 | 0.01 | 0.47 | 0.003 | 0.021 | 3.93 (-0.23) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1007 | -0.013 | 0.015 | -0.50 | 0.80 | 0.003 | 0.017 | 2.39 (+0.98) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1008 | 0.001 | 0.017 | -0.14 | 0.54 | 0.004 | 0.023 | 2.79 (+0.09) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1009 | 0.002 | 0.016 | -0.04 | 0.60 | 0.003 | 0.011 | 2.23 (-0.15) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1010 | -0.008 | 0.017 | -0.35 | 0.69 | 0.009 | 0.031 | 2.35 (+0.67) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1011 | -0.003 | 0.016 | -0.18 | 0.71 | 0.001 | 0.004 | 2.55 (+0.35) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1012 | -0.003 | 0.015 | -0.31 | 0.52 | 0.005 | 0.041 | 2.10 (+0.76) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1013 | 0.002 | 0.016 | -0.14 | 0.65 | 0.004 | 0.012 | 2.01 (-0.02) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1014 | 0.015 | 0.025 | 0.05 | 0.66 | 0.002 | 0.006 | 3.57 (-0.62) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1015 | -0.007 | 0.023 | -0.08 | 0.48 | 0.004 | 0.046 | 2.92 (+0.31) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1016 | -0.019 | 0.018 | -0.59 | 0.65 | 0.002 | 0.012 | 2.75 (+1.27) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1017 | 0.030 | 0.018 | 0.39 | 0.70 | 0.002 | 0.018 | 3.60 (-1.38) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1018 | 0.043 | 0.025 | 0.64 | 0.97 | 0.013 | 0.099 | 3.05 (-1.26) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1019 | 0.024 | 0.017 | -0.17 | 0.71 | 0.002 | 0.025 | 2.25 (+0.19) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1020 | -0.011 | 0.018 | -0.13 | 0.64 | 0.003 | 0.022 | 3.68 (+0.41) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1021 | 0.007 | 0.020 | 0.28 | 0.64 | 0.001 | 0.004 | 1.79 (-0.33) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1022 | -0.010 | 0.017 | -0.38 | 0.83 | 0.003 | 0.026 | 2.84 (+1.29) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1023 | 0.024 | 0.021 | -0.21 | 0.65 | 0.014 | 0.086 | 2.84 (+0.15) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1024 | 0.029 | 0.024 | 0.43 | 0.83 | 0.007 | 0.044 | 2.69 (-0.86) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1025 | -0.034 | 0.021 | -0.61 | 0.64 | 0.003 | 0.030 | 3.69 (+2.17) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1026 | -0.019 | 0.035 | -0.67 | 0.72 | 0.019 | 0.095 | 2.19 (+0.27) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1027 | -0.013 | 0.020 | -0.24 | 0.68 | 0.003 | 0.028 | 2.39 (+0.90) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1028 | -0.006 | 0.023 | 0.42 | 0.70 | 0.003 | 0.014 | 4.11 (+0.66) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1029 | 0.023 | 0.017 | -0.04 | 0.42 | 0.005 | 0.037 | 2.25 (-0.75) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1030 | 0.010 | 0.018 | 0.23 | 0.63 | 0.002 | 0.013 | 2.28 (-0.31) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1031 | 0.032 | 0.021 | 0.33 | 0.60 | 0.005 | 0.037 | 2.92 (-1.03) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1032 | 0.017 | 0.023 | -0.01 | 0.48 | 0.005 | 0.053 | 2.64 (-0.38) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1033 | -0.004 | 0.018 | 0.18 | 0.67 | 0.002 | 0.020 | 2.42 (+0.35) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1034 | -0.006 | 0.022 | -0.26 | 0.66 | 0.007 | 0.041 | 2.16 (+0.45) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1035 | 0.017 | 0.015 | -0.07 | 0.96 | 0.008 | 0.021 | 2.83 (-0.95) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1036 | 0.019 | 0.024 | 0.25 | 0.38 | 0.005 | 0.041 | 3.12 (-0.71) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1037 | -0.019 | 0.018 | -0.39 | 0.72 | 0.002 | 0.007 | 1.80 (+1.02) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1038 | 0.014 | 0.026 | 0.04 | 0.62 | 0.004 | 0.033 | 1.82 (-0.05) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1039 | 0.010 | 0.021 | 0.10 | 0.43 | 0.001 | 0.011 | 2.03 (-0.50) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1040 | 0.036 | 0.026 | 0.54 | 1.00 | 0.006 | 0.055 | 3.27 (-1.38) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1041 | 0.029 | 0.035 | 0.36 | 0.79 | 0.005 | 0.028 | 3.69 (-0.92) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1042 | 0.050 | 0.021 | 0.30 | 0.82 | 0.003 | 0.032 | 1.82 (-1.64) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1043 | -0.006 | 0.014 | -0.42 | 0.79 | 0.005 | 0.017 | 1.86 (+1.06) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1044 | -0.009 | 0.018 | -0.19 | 0.42 | 0.001 | 0.006 | 2.63 (+0.52) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1045 | -0.002 | 0.023 | 0.12 | 0.68 | 0.004 | 0.034 | 2.74 (+0.77) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1046 | 0.022 | 0.024 | 0.26 | 0.54 | 0.007 | 0.060 | 3.30 (-0.68) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1047 | 0.006 | 0.015 | -0.27 | 1.16 | 0.004 | 0.025 | 2.77 (+0.25) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1048 | -0.007 | 0.021 | -0.21 | 0.37 | 0.002 | 0.013 | 3.17 (+0.85) | 0.00 (+0.00) |
| gmm_D2_svbmc_seed1049 | 0.021 | 0.015 | -0.08 | 0.55 | 0.003 | 0.031 | 2.05 (-0.93) | 0.00 (+0.00) |

## multisensory_s1_D6_noise1.3_svbmc (noisy)

### M = 2 (20 cells)

| Estimator | bias | bias with the arm's entropy | covered | mc_se | gp_sd (indep / corr) |
|---|---:|---:|---:|---:|---:|
| raw | 0.403 [0.334, 0.482] | 0.406 [0.318, 0.462] | - | - | - |
| capped_I | 0.115 [0.039, 0.202] | 0.077 [0.051, 0.191] | - | - | - |
| capped_E | 0.362 [0.303, 0.470] | 0.363 [0.291, 0.440] | - | - | - |
| honest ratio1.5 median | 0.005 [-0.094, 0.087] | -0.030 [-0.101, 0.078] | 0.923 | 0.024 | 0.092 / 0.446 |
| honest ratio1.5 precision | 0.005 [-0.094, 0.087] | -0.030 [-0.101, 0.078] | 0.923 | 0.024 | 0.092 / 0.446 |
| honest ratio1.5 mean | 0.005 [-0.094, 0.087] | -0.030 [-0.101, 0.078] | 0.923 | 0.024 | 0.092 / 0.446 |
| honest ratio2 median | -0.147 [-0.202, -0.061] | -0.150 [-0.212, -0.072] | 0.978 | 0.026 | 0.098 / 0.487 |
| honest ratio2 precision | -0.147 [-0.202, -0.061] | -0.150 [-0.212, -0.072] | 0.978 | 0.026 | 0.098 / 0.487 |
| honest ratio2 mean | -0.147 [-0.202, -0.061] | -0.150 [-0.212, -0.072] | 0.978 | 0.026 | 0.098 / 0.487 |
| honest ratio3 median | -0.207 [-0.394, -0.089] | -0.222 [-0.404, -0.119] | 0.994 | 0.029 | 0.111 / 0.514 |
| honest ratio3 precision | -0.207 [-0.394, -0.089] | -0.222 [-0.404, -0.119] | 0.994 | 0.029 | 0.111 / 0.514 |
| honest ratio3 mean | -0.207 [-0.394, -0.089] | -0.222 [-0.404, -0.119] | 0.994 | 0.029 | 0.111 / 0.514 |
| honest ratio5 median | -0.220 [-0.394, -0.089] | -0.240 [-0.404, -0.119] | 0.994 | 0.029 | 0.111 / 0.517 |
| honest ratio5 precision | -0.220 [-0.394, -0.089] | -0.240 [-0.404, -0.119] | 0.994 | 0.029 | 0.111 / 0.517 |
| honest ratio5 mean | -0.220 [-0.394, -0.089] | -0.240 [-0.404, -0.119] | 0.994 | 0.029 | 0.111 / 0.517 |
| honest cap_only median | -0.220 [-0.394, -0.089] | -0.240 [-0.404, -0.119] | 0.994 | 0.029 | 0.111 / 0.517 |
| honest cap_only precision | -0.220 [-0.394, -0.089] | -0.240 [-0.404, -0.119] | 0.994 | 0.029 | 0.111 / 0.517 |
| honest cap_only mean | -0.220 [-0.394, -0.089] | -0.240 [-0.404, -0.119] | 0.994 | 0.029 | 0.111 / 0.517 |
| honest none median | -0.278 [-0.549, -0.107] | -0.259 [-0.541, -0.151] | 1.000 | 0.029 | 0.118 / 0.528 |
| honest none precision | -0.278 [-0.549, -0.107] | -0.259 [-0.541, -0.151] | 1.000 | 0.029 | 0.118 / 0.528 |
| honest none mean | -0.278 [-0.549, -0.107] | -0.259 [-0.541, -0.151] | 1.000 | 0.029 | 0.118 / 0.528 |
| pooled (ratio2, own included) | 0.111 [0.049, 0.201] | - | - | - | - |
| truth_strat | 0.009 [-0.007, 0.018] | - | - | - | - |

Reference `e_log_joint_mc_sd` median 0.016; the headline bias is within 2 SE in 15% of the cells; median absolute bias raw 0.403, capped_I 0.150, capped_E 0.362, honest 0.194 (against raw the honest estimate is better, against capped_I **worse**; tie band 0.016, one reference standard error). Calibration of the self-reported SDs (weighted over components): own z median 1.15 [0.87, 1.46], abs z median 1.18 [0.93, 1.49], abs z > 2 fraction 0.24 [0.13, 0.29]; cross z median -0.02 [-0.11, 0.04], abs z median 0.51 [0.43, 0.65], abs z > 2 fraction 0.02 [0.02, 0.04]. Effective training points on a component (weighted median): own run 0.4 [0.3, 0.4], covering other runs 0.3 [0.2, 0.3]. Checks: own-run max abs z 3.59 (weighted offset abs z 1.54), Jacobian max abs z 3.85 (offset abs z 2.18), raw consistency 2.3e-13, flagged cells 0.

Heaviest component of every run, repetition 0:

| Run, heaviest component (weight) | truth | own I_corr (error; BQ sd; points within 2 SD, their data offset; mean fn) | other runs: estimate (error; pred sd; covered; points within 2 SD, their data offset; mean fn) |
|---|---:|---|---|
| multisensory_s1_D6_noise1.3_svbmc_seed1005, component 0 (0.052) | -505.657 ± 0.166 | -505.262 (+0.395; 0.343; 1, 0.172; -555.9) | multisensory_s1_D6_noise1.3_svbmc_seed1016: -507.076 (-1.419; 0.852; yes; 0, -; -574.5) |
| multisensory_s1_D6_noise1.3_svbmc_seed1016, component 2 (0.060) | -505.572 ± 0.123 | -504.750 (+0.822; 0.272; 4, -0.831; -573.2) | multisensory_s1_D6_noise1.3_svbmc_seed1005: -506.416 (-0.844; 0.688; yes; 0, -; -550.6) |

### M = 4 (20 cells)

| Estimator | bias | bias with the arm's entropy | covered | mc_se | gp_sd (indep / corr) |
|---|---:|---:|---:|---:|---:|
| raw | 0.483 [0.451, 0.508] | 0.468 [0.456, 0.510] | - | - | - |
| capped_I | 0.113 [0.061, 0.157] | 0.110 [0.061, 0.150] | - | - | - |
| capped_E | 0.347 [0.292, 0.396] | 0.332 [0.270, 0.399] | - | - | - |
| honest ratio1.5 median | -0.163 [-0.237, -0.079] | -0.166 [-0.249, -0.061] | 0.944 | 0.022 | 0.081 / 0.361 |
| honest ratio1.5 precision | -0.142 [-0.207, -0.053] | -0.151 [-0.208, -0.056] | 0.944 | 0.022 | 0.059 / 0.347 |
| honest ratio1.5 mean | -0.171 [-0.264, -0.094] | -0.172 [-0.263, -0.093] | 0.944 | 0.022 | 0.059 / 0.347 |
| honest ratio2 median | -0.264 [-0.353, -0.211] | -0.278 [-0.353, -0.210] | 0.984 | 0.024 | 0.093 / 0.390 |
| honest ratio2 precision | -0.238 [-0.304, -0.187] | -0.247 [-0.318, -0.181] | 0.984 | 0.022 | 0.058 / 0.349 |
| honest ratio2 mean | -0.291 [-0.386, -0.248] | -0.314 [-0.387, -0.252] | 0.984 | 0.022 | 0.060 / 0.361 |
| honest ratio3 median | -0.323 [-0.498, -0.277] | -0.337 [-0.493, -0.291] | 0.996 | 0.024 | 0.096 / 0.422 |
| honest ratio3 precision | -0.289 [-0.410, -0.233] | -0.309 [-0.423, -0.237] | 0.996 | 0.022 | 0.058 / 0.359 |
| honest ratio3 mean | -0.418 [-0.548, -0.304] | -0.423 [-0.545, -0.327] | 0.996 | 0.022 | 0.061 / 0.379 |
| honest ratio5 median | -0.330 [-0.605, -0.286] | -0.337 [-0.614, -0.298] | 0.996 | 0.024 | 0.099 / 0.439 |
| honest ratio5 precision | -0.289 [-0.503, -0.236] | -0.309 [-0.500, -0.238] | 0.996 | 0.022 | 0.058 / 0.360 |
| honest ratio5 mean | -0.420 [-0.705, -0.304] | -0.425 [-0.702, -0.327] | 0.996 | 0.022 | 0.062 / 0.385 |
| honest cap_only median | -0.330 [-0.605, -0.286] | -0.337 [-0.614, -0.298] | 0.996 | 0.024 | 0.099 / 0.439 |
| honest cap_only precision | -0.289 [-0.503, -0.236] | -0.309 [-0.500, -0.238] | 0.996 | 0.022 | 0.058 / 0.360 |
| honest cap_only mean | -0.420 [-0.705, -0.304] | -0.425 [-0.702, -0.327] | 0.996 | 0.022 | 0.062 / 0.385 |
| honest none median | -0.380 [-1.159, -0.298] | -0.392 [-1.175, -0.310] | 1.000 | 0.024 | 0.116 / 0.477 |
| honest none precision | -0.354 [-0.941, -0.237] | -0.374 [-0.947, -0.239] | 1.000 | 0.022 | 0.062 / 0.406 |
| honest none mean | -0.530 [-1.547, -0.303] | -0.554 [-1.565, -0.326] | 1.000 | 0.023 | 0.068 / 0.432 |
| pooled (ratio2, own included) | -0.072 [-0.122, -0.015] | - | - | - | - |
| truth_strat | -0.000 [-0.014, 0.024] | - | - | - | - |

Reference `e_log_joint_mc_sd` median 0.017; the headline bias is within 2 SE in 15% of the cells; median absolute bias raw 0.483, capped_I 0.113, capped_E 0.347, honest 0.264 (against raw the honest estimate is better, against capped_I **worse**; tie band 0.017, one reference standard error). Calibration of the self-reported SDs (weighted over components): own z median 1.48 [1.31, 1.61], abs z median 1.50 [1.35, 1.62], abs z > 2 fraction 0.28 [0.20, 0.32]; cross z median -0.20 [-0.33, -0.13], abs z median 0.52 [0.49, 0.59], abs z > 2 fraction 0.03 [0.02, 0.05]. Effective training points on a component (weighted median): own run 0.4 [0.3, 0.4], covering other runs 0.3 [0.2, 0.3]. Checks: own-run max abs z 4.28 (weighted offset abs z 2.00), Jacobian max abs z 3.44 (offset abs z 2.21), raw consistency 5.7e-14, flagged cells 0.

Heaviest component of every run, repetition 0:

| Run, heaviest component (weight) | truth | own I_corr (error; BQ sd; points within 2 SD, their data offset; mean fn) | other runs: estimate (error; pred sd; covered; points within 2 SD, their data offset; mean fn) |
|---|---:|---|---|
| multisensory_s1_D6_noise1.3_svbmc_seed1045, component 1 (0.059) | -505.918 ± 0.145 | -505.508 (+0.410; 0.329; 1, 0.769; -539.7) | multisensory_s1_D6_noise1.3_svbmc_seed1067: -506.593 (-0.675; 0.568; yes; 0, -; -549.1); multisensory_s1_D6_noise1.3_svbmc_seed1077: -505.824 (+0.093; 0.665; yes; 2, 0.965; -565.8); multisensory_s1_D6_noise1.3_svbmc_seed1099: -506.372 (-0.455; 0.647; yes; 0, -; -614.0) |
| multisensory_s1_D6_noise1.3_svbmc_seed1067, component 3 (0.010) | -505.052 ± 0.102 | -504.893 (+0.159; 0.270; 0, -; -550.7) | multisensory_s1_D6_noise1.3_svbmc_seed1045: -505.011 (+0.041; 0.576; yes; 0, -; -538.6); multisensory_s1_D6_noise1.3_svbmc_seed1077: -504.836 (+0.215; 0.580; yes; 3, 0.356; -565.6); multisensory_s1_D6_noise1.3_svbmc_seed1099: -504.803 (+0.249; 0.462; yes; 1, -1.281; -613.0) |
| multisensory_s1_D6_noise1.3_svbmc_seed1077, component 5 (0.116) | -505.519 ± 0.148 | -504.885 (+0.635; 0.285; 1, 1.648; -566.2) | multisensory_s1_D6_noise1.3_svbmc_seed1045: -506.111 (-0.592; 0.658; yes; 1, -0.654; -539.3); multisensory_s1_D6_noise1.3_svbmc_seed1067: -507.046 (-1.526; 0.669; yes; 0, -; -554.0); multisensory_s1_D6_noise1.3_svbmc_seed1099: -505.971 (-0.452; 0.806; yes; 0, -; -614.2) |
| multisensory_s1_D6_noise1.3_svbmc_seed1099, component 1 (0.050) | -505.787 ± 0.167 | -504.919 (+0.868; 0.345; 2, 0.382; -610.4) | multisensory_s1_D6_noise1.3_svbmc_seed1045: -510.050 (-4.263; 2.755; no; 0, -; -538.9); multisensory_s1_D6_noise1.3_svbmc_seed1067: -506.042 (-0.254; 0.887; yes; 2, -0.116; -537.5); multisensory_s1_D6_noise1.3_svbmc_seed1077: -510.942 (-5.154; 5.519; no; 0, -; -563.3) |

### M = 8 (20 cells)

| Estimator | bias | bias with the arm's entropy | covered | mc_se | gp_sd (indep / corr) |
|---|---:|---:|---:|---:|---:|
| raw | 0.555 [0.521, 0.637] | 0.562 [0.511, 0.633] | - | - | - |
| capped_I | 0.151 [0.117, 0.195] | 0.139 [0.113, 0.185] | - | - | - |
| capped_E | 0.406 [0.326, 0.445] | 0.390 [0.310, 0.439] | - | - | - |
| honest ratio1.5 median | -0.129 [-0.166, -0.077] | -0.133 [-0.179, -0.092] | 0.990 | 0.018 | 0.065 / 0.261 |
| honest ratio1.5 precision | -0.106 [-0.144, -0.052] | -0.115 [-0.155, -0.050] | 0.990 | 0.018 | 0.033 / 0.241 |
| honest ratio1.5 mean | -0.155 [-0.195, -0.097] | -0.170 [-0.205, -0.104] | 0.990 | 0.018 | 0.034 / 0.247 |
| honest ratio2 median | -0.238 [-0.335, -0.186] | -0.242 [-0.337, -0.180] | 0.997 | 0.018 | 0.070 / 0.270 |
| honest ratio2 precision | -0.182 [-0.252, -0.142] | -0.189 [-0.260, -0.150] | 0.997 | 0.018 | 0.032 / 0.252 |
| honest ratio2 mean | -0.265 [-0.377, -0.236] | -0.278 [-0.383, -0.235] | 0.997 | 0.018 | 0.034 / 0.262 |
| honest ratio3 median | -0.360 [-0.416, -0.276] | -0.356 [-0.413, -0.291] | 0.998 | 0.019 | 0.074 / 0.280 |
| honest ratio3 precision | -0.245 [-0.377, -0.203] | -0.258 [-0.384, -0.210] | 0.998 | 0.018 | 0.033 / 0.257 |
| honest ratio3 mean | -0.439 [-0.521, -0.344] | -0.442 [-0.512, -0.355] | 0.998 | 0.018 | 0.035 / 0.278 |
| honest ratio5 median | -0.433 [-0.507, -0.284] | -0.426 [-0.519, -0.299] | 0.998 | 0.019 | 0.079 / 0.286 |
| honest ratio5 precision | -0.252 [-0.415, -0.226] | -0.265 [-0.404, -0.232] | 0.998 | 0.018 | 0.033 / 0.259 |
| honest ratio5 mean | -0.533 [-0.615, -0.353] | -0.527 [-0.614, -0.364] | 0.998 | 0.019 | 0.036 / 0.280 |
| honest cap_only median | -0.433 [-0.507, -0.284] | -0.426 [-0.519, -0.299] | 0.998 | 0.019 | 0.079 / 0.286 |
| honest cap_only precision | -0.252 [-0.415, -0.226] | -0.265 [-0.404, -0.232] | 0.998 | 0.018 | 0.033 / 0.259 |
| honest cap_only mean | -0.533 [-0.615, -0.353] | -0.527 [-0.614, -0.364] | 0.998 | 0.019 | 0.036 / 0.280 |
| honest none median | -0.719 [-1.277, -0.387] | -0.706 [-1.276, -0.400] | 1.000 | 0.020 | 0.099 / 0.371 |
| honest none precision | -0.418 [-0.488, -0.298] | -0.428 [-0.479, -0.296] | 1.000 | 0.018 | 0.032 / 0.272 |
| honest none mean | -0.723 [-1.409, -0.522] | -0.748 [-1.405, -0.530] | 1.000 | 0.019 | 0.039 / 0.328 |
| pooled (ratio2, own included) | -0.097 [-0.172, -0.063] | - | - | - | - |
| truth_strat | 0.000 [-0.013, 0.008] | - | - | - | - |

Reference `e_log_joint_mc_sd` median 0.017; the headline bias is within 2 SE in 5% of the cells; median absolute bias raw 0.555, capped_I 0.151, capped_E 0.406, honest 0.238 (against raw the honest estimate is better, against capped_I **worse**; tie band 0.017, one reference standard error). Calibration of the self-reported SDs (weighted over components): own z median 1.59 [1.50, 1.78], abs z median 1.59 [1.51, 1.78], abs z > 2 fraction 0.37 [0.32, 0.41]; cross z median -0.10 [-0.20, -0.06], abs z median 0.50 [0.49, 0.54], abs z > 2 fraction 0.05 [0.03, 0.05]. Effective training points on a component (weighted median): own run 0.3 [0.3, 0.3], covering other runs 0.2 [0.2, 0.3]. Checks: own-run max abs z 5.02 (weighted offset abs z 1.51), Jacobian max abs z 3.81 (offset abs z 3.55), raw consistency 0.0e+00, flagged cells 0.

### M = 16 (10 cells)

| Estimator | bias | bias with the arm's entropy | covered | mc_se | gp_sd (indep / corr) |
|---|---:|---:|---:|---:|---:|
| raw | 0.617 [0.561, 0.685] | 0.606 [0.546, 0.694] | - | - | - |
| capped_I | 0.165 [0.110, 0.191] | 0.156 [0.101, 0.191] | - | - | - |
| capped_E | 0.392 [0.364, 0.476] | 0.374 [0.353, 0.474] | - | - | - |
| honest ratio1.5 median | -0.182 [-0.195, -0.140] | -0.184 [-0.219, -0.134] | 0.998 | 0.015 | 0.051 / 0.192 |
| honest ratio1.5 precision | -0.170 [-0.189, -0.121] | -0.182 [-0.200, -0.104] | 0.998 | 0.014 | 0.022 / 0.178 |
| honest ratio1.5 mean | -0.229 [-0.255, -0.180] | -0.243 [-0.269, -0.164] | 0.998 | 0.014 | 0.023 / 0.181 |
| honest ratio2 median | -0.294 [-0.351, -0.253] | -0.295 [-0.370, -0.246] | 1.000 | 0.015 | 0.057 / 0.204 |
| honest ratio2 precision | -0.259 [-0.291, -0.225] | -0.271 [-0.305, -0.204] | 1.000 | 0.014 | 0.021 / 0.182 |
| honest ratio2 mean | -0.377 [-0.432, -0.351] | -0.395 [-0.444, -0.336] | 1.000 | 0.015 | 0.022 / 0.193 |
| honest ratio3 median | -0.434 [-0.525, -0.364] | -0.434 [-0.534, -0.365] | 1.000 | 0.016 | 0.059 / 0.213 |
| honest ratio3 precision | -0.314 [-0.368, -0.296] | -0.328 [-0.377, -0.278] | 1.000 | 0.015 | 0.021 / 0.187 |
| honest ratio3 mean | -0.537 [-0.633, -0.505] | -0.537 [-0.647, -0.506] | 1.000 | 0.015 | 0.023 / 0.208 |
| honest ratio5 median | -0.538 [-0.629, -0.456] | -0.551 [-0.638, -0.458] | 1.000 | 0.016 | 0.064 / 0.220 |
| honest ratio5 precision | -0.340 [-0.390, -0.316] | -0.349 [-0.407, -0.308] | 1.000 | 0.015 | 0.020 / 0.188 |
| honest ratio5 mean | -0.625 [-0.709, -0.583] | -0.637 [-0.741, -0.598] | 1.000 | 0.015 | 0.024 / 0.214 |
| honest cap_only median | -0.538 [-0.629, -0.456] | -0.551 [-0.638, -0.458] | 1.000 | 0.016 | 0.064 / 0.220 |
| honest cap_only precision | -0.340 [-0.390, -0.316] | -0.349 [-0.401, -0.308] | 1.000 | 0.015 | 0.020 / 0.188 |
| honest cap_only mean | -0.625 [-0.730, -0.583] | -0.637 [-0.741, -0.598] | 1.000 | 0.015 | 0.024 / 0.214 |
| honest none median | -2.368 [-3.155, -0.934] | -2.375 [-3.377, -0.967] | 1.000 | 0.020 | 0.208 / 0.535 |
| honest none precision | -0.497 [-0.623, -0.408] | -0.487 [-0.640, -0.425] | 1.000 | 0.015 | 0.019 / 0.200 |
| honest none mean | -2.906 [-3.849, -1.120] | -2.918 [-3.944, -1.153] | 1.000 | 0.019 | 0.061 / 0.422 |
| pooled (ratio2, own included) | -0.214 [-0.256, -0.187] | - | - | - | - |
| truth_strat | -0.001 [-0.013, 0.006] | - | - | - | - |

Reference `e_log_joint_mc_sd` median 0.017; the headline bias is within 2 SE in 0% of the cells; median absolute bias raw 0.617, capped_I 0.165, capped_E 0.392, honest 0.294 (against raw the honest estimate is better, against capped_I **worse**; tie band 0.017, one reference standard error). Calibration of the self-reported SDs (weighted over components): own z median 1.79 [1.64, 1.92], abs z median 1.80 [1.65, 1.92], abs z > 2 fraction 0.44 [0.36, 0.47]; cross z median -0.16 [-0.24, -0.12], abs z median 0.54 [0.51, 0.56], abs z > 2 fraction 0.04 [0.03, 0.05]. Effective training points on a component (weighted median): own run 0.3 [0.3, 0.3], covering other runs 0.2 [0.2, 0.3]. Checks: own-run max abs z 4.60 (weighted offset abs z 3.18), Jacobian max abs z 3.96 (offset abs z 2.10), raw consistency 0.0e+00, flagged cells 0.

| Run | own error | se | own z median | own abs z median | BQ sd median | pred sd median | own-run max abs z (offset z) | Jacobian max abs z (offset z) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| multisensory_s1_D6_noise1.3_svbmc_seed1000 | 0.504 | 0.027 | 1.19 | 1.19 | 0.413 | 0.751 | 2.04 (+1.36) | 3.33 (-0.56) |
| multisensory_s1_D6_noise1.3_svbmc_seed1001 | 0.069 | 0.032 | 0.28 | 0.99 | 0.389 | 0.633 | 2.94 (-0.29) | 1.95 (-2.59) |
| multisensory_s1_D6_noise1.3_svbmc_seed1002 | 0.469 | 0.027 | 1.44 | 1.55 | 0.374 | 0.577 | 2.10 (-1.57) | 2.29 (+0.01) |
| multisensory_s1_D6_noise1.3_svbmc_seed1003 | 0.126 | 0.028 | 0.48 | 0.67 | 0.359 | 0.595 | 1.83 (+0.42) | 3.16 (+0.16) |
| multisensory_s1_D6_noise1.3_svbmc_seed1004 | 0.212 | 0.026 | 0.39 | 0.62 | 0.416 | 0.875 | 3.02 (+0.58) | 2.62 (-1.40) |
| multisensory_s1_D6_noise1.3_svbmc_seed1005 | 0.327 | 0.030 | 1.14 | 1.23 | 0.390 | 0.687 | 2.87 (-0.02) | 2.47 (+0.59) |
| multisensory_s1_D6_noise1.3_svbmc_seed1006 | 0.412 | 0.029 | 1.38 | 1.38 | 0.391 | 0.686 | 2.24 (-0.70) | 2.55 (+1.88) |
| multisensory_s1_D6_noise1.3_svbmc_seed1007 | 0.153 | 0.029 | 0.28 | 0.54 | 0.434 | 0.723 | 2.45 (-0.11) | 2.45 (+1.43) |
| multisensory_s1_D6_noise1.3_svbmc_seed1008 | 0.335 | 0.032 | 0.92 | 0.93 | 0.381 | 0.613 | 2.16 (-0.18) | 2.84 (-1.52) |
| multisensory_s1_D6_noise1.3_svbmc_seed1009 | 0.131 | 0.025 | 0.50 | 0.55 | 0.327 | 0.663 | 2.61 (+0.24) | 2.55 (+1.76) |
| multisensory_s1_D6_noise1.3_svbmc_seed1010 | 0.101 | 0.038 | 0.26 | 0.64 | 0.386 | 0.629 | 2.90 (+2.37) | 2.91 (+0.38) |
| multisensory_s1_D6_noise1.3_svbmc_seed1011 | -0.009 | 0.036 | 0.06 | 0.69 | 0.392 | 0.739 | 2.43 (+1.45) | 3.38 (-0.52) |
| multisensory_s1_D6_noise1.3_svbmc_seed1012 | 0.460 | 0.033 | 1.62 | 1.64 | 0.353 | 0.674 | 3.21 (+0.17) | 2.71 (-0.15) |
| multisensory_s1_D6_noise1.3_svbmc_seed1013 | 0.216 | 0.031 | 0.48 | 0.65 | 0.411 | 0.724 | 2.40 (+0.46) | 2.32 (+1.37) |
| multisensory_s1_D6_noise1.3_svbmc_seed1014 | 0.164 | 0.033 | 0.42 | 0.61 | 0.364 | 0.614 | 2.73 (-1.27) | 3.06 (+1.05) |
| multisensory_s1_D6_noise1.3_svbmc_seed1015 | 0.379 | 0.030 | 1.35 | 1.42 | 0.387 | 0.644 | 2.31 (-0.22) | 2.13 (-0.58) |
| multisensory_s1_D6_noise1.3_svbmc_seed1016 | 0.394 | 0.030 | 1.13 | 1.14 | 0.332 | 0.605 | 2.77 (-0.88) | 2.29 (+0.20) |
| multisensory_s1_D6_noise1.3_svbmc_seed1017 | 0.556 | 0.036 | 1.74 | 1.84 | 0.387 | 0.669 | 2.87 (-0.03) | 2.77 (+0.04) |
| multisensory_s1_D6_noise1.3_svbmc_seed1018 | 0.180 | 0.032 | 0.40 | 0.49 | 0.423 | 0.742 | 2.25 (+1.38) | 2.75 (-0.98) |
| multisensory_s1_D6_noise1.3_svbmc_seed1019 | 0.158 | 0.029 | 0.44 | 0.53 | 0.356 | 0.667 | 2.33 (-0.14) | 2.08 (-0.27) |
| multisensory_s1_D6_noise1.3_svbmc_seed1020 | 0.221 | 0.032 | 0.61 | 0.75 | 0.383 | 0.652 | 3.31 (-1.56) | 2.80 (-0.07) |
| multisensory_s1_D6_noise1.3_svbmc_seed1021 | 0.744 | 0.028 | 2.24 | 2.24 | 0.358 | 0.667 | 2.87 (-1.32) | 2.35 (+0.40) |
| multisensory_s1_D6_noise1.3_svbmc_seed1022 | 0.147 | 0.042 | 0.49 | 1.34 | 0.408 | 0.673 | 2.56 (-0.50) | 1.80 (-0.41) |
| multisensory_s1_D6_noise1.3_svbmc_seed1023 | 0.497 | 0.038 | 1.29 | 1.31 | 0.333 | 0.571 | 2.35 (-1.09) | 2.90 (+1.17) |
| multisensory_s1_D6_noise1.3_svbmc_seed1024 | 0.374 | 0.028 | 1.06 | 1.07 | 0.345 | 0.685 | 4.60 (+1.16) | 2.65 (-0.99) |
| multisensory_s1_D6_noise1.3_svbmc_seed1025 | 0.536 | 0.039 | 1.89 | 1.89 | 0.360 | 0.609 | 3.78 (-0.49) | 2.79 (+0.82) |
| multisensory_s1_D6_noise1.3_svbmc_seed1026 | 0.515 | 0.031 | 1.45 | 1.45 | 0.357 | 0.595 | 2.15 (-0.31) | 2.34 (-1.13) |
| multisensory_s1_D6_noise1.3_svbmc_seed1027 | 0.245 | 0.033 | 0.82 | 1.08 | 0.318 | 0.569 | 2.44 (-0.46) | 2.21 (+0.18) |
| multisensory_s1_D6_noise1.3_svbmc_seed1028 | 0.294 | 0.034 | 0.99 | 1.08 | 0.351 | 0.572 | 1.95 (-0.20) | 2.66 (+0.34) |
| multisensory_s1_D6_noise1.3_svbmc_seed1029 | 0.248 | 0.032 | 0.91 | 0.91 | 0.380 | 0.625 | 2.35 (+0.58) | 2.14 (+0.36) |
| multisensory_s1_D6_noise1.3_svbmc_seed1030 | 0.477 | 0.029 | 1.45 | 1.45 | 0.366 | 0.629 | 2.44 (-0.02) | 2.24 (+1.27) |
| multisensory_s1_D6_noise1.3_svbmc_seed1031 | -0.114 | 0.031 | -0.40 | 0.61 | 0.364 | 0.653 | 2.40 (+1.03) | 3.15 (+0.07) |
| multisensory_s1_D6_noise1.3_svbmc_seed1032 | 0.366 | 0.033 | 0.83 | 0.84 | 0.410 | 0.653 | 2.94 (+1.68) | 3.64 (-1.31) |
| multisensory_s1_D6_noise1.3_svbmc_seed1033 | 1.024 | 0.026 | 3.45 | 3.45 | 0.336 | 0.592 | 2.55 (+1.37) | 2.64 (+1.03) |
| multisensory_s1_D6_noise1.3_svbmc_seed1034 | 0.618 | 0.027 | 1.52 | 1.53 | 0.404 | 0.724 | 2.81 (-0.67) | 2.10 (+0.06) |
| multisensory_s1_D6_noise1.3_svbmc_seed1035 | 0.234 | 0.029 | 0.56 | 0.75 | 0.458 | 0.818 | 2.33 (-0.51) | 1.96 (-0.15) |
| multisensory_s1_D6_noise1.3_svbmc_seed1036 | 0.076 | 0.030 | 0.21 | 0.86 | 0.321 | 0.563 | 2.68 (+1.03) | 2.16 (+0.50) |
| multisensory_s1_D6_noise1.3_svbmc_seed1037 | 0.525 | 0.023 | 1.62 | 1.64 | 0.352 | 0.648 | 1.97 (-1.57) | 2.42 (+1.18) |
| multisensory_s1_D6_noise1.3_svbmc_seed1038 | 0.131 | 0.027 | 0.26 | 0.66 | 0.384 | 0.686 | 2.45 (-1.94) | 2.31 (+1.68) |
| multisensory_s1_D6_noise1.3_svbmc_seed1039 | 0.389 | 0.044 | 1.37 | 1.37 | 0.338 | 0.582 | 2.11 (-1.88) | 3.15 (-0.39) |
| multisensory_s1_D6_noise1.3_svbmc_seed1040 | 0.127 | 0.043 | 0.32 | 0.54 | 0.405 | 0.671 | 3.37 (+0.02) | 2.55 (+0.85) |
| multisensory_s1_D6_noise1.3_svbmc_seed1041 | 0.573 | 0.035 | 1.81 | 1.81 | 0.332 | 0.542 | 2.61 (-1.17) | 1.86 (+1.00) |
| multisensory_s1_D6_noise1.3_svbmc_seed1042 | 0.314 | 0.031 | 1.10 | 1.13 | 0.314 | 0.550 | 3.14 (-1.83) | 2.56 (-0.65) |
| multisensory_s1_D6_noise1.3_svbmc_seed1043 | -0.019 | 0.032 | 0.27 | 0.63 | 0.312 | 0.525 | 3.04 (+0.01) | 2.85 (-0.37) |
| multisensory_s1_D6_noise1.3_svbmc_seed1044 | 0.312 | 0.038 | 1.14 | 1.21 | 0.353 | 0.591 | 2.88 (+0.05) | 2.69 (-1.48) |
| multisensory_s1_D6_noise1.3_svbmc_seed1045 | 0.165 | 0.034 | 0.50 | 0.82 | 0.414 | 0.764 | 2.12 (-1.25) | 2.99 (+1.34) |
| multisensory_s1_D6_noise1.3_svbmc_seed1046 | 0.357 | 0.035 | 0.80 | 0.82 | 0.416 | 0.672 | 3.22 (+0.03) | 2.32 (+0.35) |
| multisensory_s1_D6_noise1.3_svbmc_seed1047 | 0.529 | 0.028 | 1.18 | 1.18 | 0.419 | 0.719 | 2.41 (-0.30) | 2.83 (+0.01) |
| multisensory_s1_D6_noise1.3_svbmc_seed1048 | 0.519 | 0.042 | 2.00 | 2.00 | 0.341 | 0.584 | 2.77 (-0.82) | 2.66 (-1.29) |
| multisensory_s1_D6_noise1.3_svbmc_seed1050 | 0.385 | 0.031 | 1.18 | 1.19 | 0.303 | 0.504 | 2.69 (+0.60) | 2.58 (-0.91) |
| multisensory_s1_D6_noise1.3_svbmc_seed1051 | 0.564 | 0.033 | 2.03 | 2.03 | 0.387 | 0.660 | 2.20 (-0.67) | 2.42 (+1.11) |
| multisensory_s1_D6_noise1.3_svbmc_seed1052 | 0.304 | 0.031 | 0.96 | 1.04 | 0.454 | 0.750 | 3.05 (+1.30) | 2.45 (+0.14) |
| multisensory_s1_D6_noise1.3_svbmc_seed1053 | 0.754 | 0.035 | 2.43 | 2.43 | 0.405 | 0.754 | 3.57 (-1.99) | 2.38 (+1.59) |
| multisensory_s1_D6_noise1.3_svbmc_seed1054 | 0.520 | 0.028 | 1.78 | 1.78 | 0.314 | 0.525 | 2.42 (-1.42) | 2.11 (+1.11) |
| multisensory_s1_D6_noise1.3_svbmc_seed1055 | 0.537 | 0.031 | 1.70 | 1.70 | 0.342 | 0.583 | 2.56 (-1.87) | 2.87 (+1.36) |
| multisensory_s1_D6_noise1.3_svbmc_seed1056 | 0.157 | 0.039 | 0.81 | 0.82 | 0.379 | 0.691 | 3.15 (+0.79) | 1.97 (-2.15) |
| multisensory_s1_D6_noise1.3_svbmc_seed1057 | 0.249 | 0.034 | 0.50 | 0.62 | 0.372 | 0.660 | 3.62 (-1.50) | 3.50 (+0.53) |
| multisensory_s1_D6_noise1.3_svbmc_seed1058 | 0.379 | 0.037 | 0.72 | 0.84 | 0.406 | 0.746 | 2.37 (-1.16) | 2.08 (+0.63) |
| multisensory_s1_D6_noise1.3_svbmc_seed1059 | 0.682 | 0.033 | 2.21 | 2.21 | 0.333 | 0.543 | 2.71 (+0.69) | 2.40 (-0.51) |
| multisensory_s1_D6_noise1.3_svbmc_seed1060 | 0.178 | 0.030 | 0.53 | 0.54 | 0.373 | 0.619 | 3.59 (+1.95) | 2.38 (+0.93) |
| multisensory_s1_D6_noise1.3_svbmc_seed1061 | 0.361 | 0.036 | 1.15 | 1.16 | 0.409 | 0.652 | 2.28 (+1.07) | 2.46 (+0.74) |
| multisensory_s1_D6_noise1.3_svbmc_seed1062 | 0.457 | 0.030 | 1.39 | 1.45 | 0.404 | 0.663 | 2.80 (-0.58) | 2.63 (+1.23) |
| multisensory_s1_D6_noise1.3_svbmc_seed1063 | 0.470 | 0.028 | 1.65 | 1.65 | 0.385 | 0.717 | 2.59 (+0.66) | 2.58 (-0.72) |
| multisensory_s1_D6_noise1.3_svbmc_seed1064 | 0.458 | 0.036 | 1.07 | 1.42 | 0.337 | 0.642 | 2.51 (-0.56) | 2.41 (+0.15) |
| multisensory_s1_D6_noise1.3_svbmc_seed1065 | 0.325 | 0.033 | 0.83 | 0.83 | 0.409 | 0.750 | 3.18 (-0.90) | 2.59 (+0.66) |
| multisensory_s1_D6_noise1.3_svbmc_seed1066 | 0.247 | 0.034 | 0.66 | 0.80 | 0.402 | 0.622 | 2.19 (-0.10) | 1.60 (-0.75) |
| multisensory_s1_D6_noise1.3_svbmc_seed1067 | -0.048 | 0.032 | -0.13 | 0.49 | 0.347 | 0.568 | 2.41 (-0.75) | 2.69 (-0.04) |
| multisensory_s1_D6_noise1.3_svbmc_seed1068 | 0.515 | 0.026 | 1.53 | 1.57 | 0.336 | 0.595 | 3.44 (-0.43) | 2.58 (+1.00) |
| multisensory_s1_D6_noise1.3_svbmc_seed1069 | 0.388 | 0.031 | 1.07 | 1.07 | 0.353 | 0.585 | 2.23 (-1.13) | 2.73 (+2.51) |
| multisensory_s1_D6_noise1.3_svbmc_seed1070 | 0.614 | 0.032 | 2.11 | 2.11 | 0.367 | 0.600 | 2.83 (+0.22) | 2.28 (+0.63) |
| multisensory_s1_D6_noise1.3_svbmc_seed1071 | 0.546 | 0.027 | 1.37 | 1.37 | 0.412 | 0.720 | 2.09 (+0.93) | 2.51 (-1.13) |
| multisensory_s1_D6_noise1.3_svbmc_seed1073 | 0.508 | 0.028 | 1.38 | 1.38 | 0.399 | 0.849 | 2.85 (+0.61) | 2.65 (-0.41) |
| multisensory_s1_D6_noise1.3_svbmc_seed1074 | 0.416 | 0.030 | 1.54 | 1.54 | 0.369 | 0.661 | 2.26 (+1.18) | 2.21 (-1.10) |
| multisensory_s1_D6_noise1.3_svbmc_seed1075 | 0.362 | 0.028 | 1.26 | 1.38 | 0.334 | 0.562 | 2.17 (-1.18) | 3.95 (-0.30) |
| multisensory_s1_D6_noise1.3_svbmc_seed1076 | 0.312 | 0.035 | 1.15 | 1.17 | 0.401 | 0.764 | 2.13 (-0.78) | 2.18 (-0.10) |
| multisensory_s1_D6_noise1.3_svbmc_seed1077 | 0.462 | 0.038 | 0.95 | 0.96 | 0.400 | 0.713 | 2.40 (-2.08) | 2.23 (+1.01) |
| multisensory_s1_D6_noise1.3_svbmc_seed1078 | 0.346 | 0.035 | 1.08 | 1.10 | 0.370 | 0.639 | 2.35 (-0.03) | 2.31 (-0.29) |
| multisensory_s1_D6_noise1.3_svbmc_seed1079 | 0.606 | 0.026 | 2.06 | 2.06 | 0.369 | 0.665 | 2.55 (-0.55) | 2.37 (+0.15) |
| multisensory_s1_D6_noise1.3_svbmc_seed1080 | 0.807 | 0.032 | 2.31 | 2.31 | 0.330 | 0.564 | 2.25 (-0.29) | 2.13 (+0.21) |
| multisensory_s1_D6_noise1.3_svbmc_seed1081 | 0.007 | 0.027 | -0.15 | 0.71 | 0.406 | 0.751 | 3.33 (+1.96) | 3.09 (-2.07) |
| multisensory_s1_D6_noise1.3_svbmc_seed1082 | 0.310 | 0.030 | 0.68 | 0.68 | 0.372 | 0.608 | 2.27 (-0.70) | 2.31 (-0.12) |
| multisensory_s1_D6_noise1.3_svbmc_seed1083 | 0.276 | 0.032 | 1.03 | 1.05 | 0.343 | 0.581 | 1.79 (-1.30) | 2.39 (-1.30) |
| multisensory_s1_D6_noise1.3_svbmc_seed1084 | 0.437 | 0.030 | 1.41 | 1.43 | 0.362 | 0.646 | 2.12 (-1.43) | 2.52 (+1.85) |
| multisensory_s1_D6_noise1.3_svbmc_seed1085 | 0.425 | 0.030 | 1.31 | 1.31 | 0.374 | 0.655 | 2.36 (+0.48) | 2.94 (+1.73) |
| multisensory_s1_D6_noise1.3_svbmc_seed1086 | 0.262 | 0.027 | 0.68 | 0.79 | 0.337 | 0.611 | 2.93 (+1.61) | 2.83 (-0.56) |
| multisensory_s1_D6_noise1.3_svbmc_seed1087 | 0.497 | 0.025 | 1.52 | 1.54 | 0.390 | 0.702 | 3.21 (+1.09) | 3.00 (-0.08) |
| multisensory_s1_D6_noise1.3_svbmc_seed1088 | -0.027 | 0.032 | 0.02 | 0.61 | 0.359 | 0.656 | 2.76 (+0.42) | 2.81 (+0.67) |
| multisensory_s1_D6_noise1.3_svbmc_seed1089 | 0.329 | 0.027 | 1.07 | 1.20 | 0.337 | 0.535 | 2.83 (+1.75) | 2.82 (-1.72) |
| multisensory_s1_D6_noise1.3_svbmc_seed1090 | 0.376 | 0.037 | 1.12 | 1.17 | 0.347 | 0.597 | 2.53 (-0.45) | 2.79 (+0.28) |
| multisensory_s1_D6_noise1.3_svbmc_seed1091 | -0.026 | 0.031 | 0.13 | 0.36 | 0.413 | 0.699 | 2.51 (-0.73) | 3.06 (-0.72) |
| multisensory_s1_D6_noise1.3_svbmc_seed1092 | 0.085 | 0.030 | 0.21 | 0.43 | 0.362 | 0.665 | 2.84 (-0.35) | 2.41 (-0.54) |
| multisensory_s1_D6_noise1.3_svbmc_seed1093 | 0.098 | 0.035 | 0.28 | 0.57 | 0.352 | 0.648 | 2.24 (-1.64) | 2.25 (+1.58) |
| multisensory_s1_D6_noise1.3_svbmc_seed1094 | 0.477 | 0.029 | 1.41 | 1.43 | 0.354 | 0.623 | 1.94 (-0.47) | 2.14 (-0.71) |
| multisensory_s1_D6_noise1.3_svbmc_seed1095 | 0.497 | 0.039 | 1.56 | 1.56 | 0.392 | 0.714 | 1.78 (+0.04) | 1.90 (+0.30) |
| multisensory_s1_D6_noise1.3_svbmc_seed1096 | 0.497 | 0.033 | 1.52 | 1.52 | 0.293 | 0.521 | 2.89 (-0.94) | 1.91 (-0.62) |
| multisensory_s1_D6_noise1.3_svbmc_seed1097 | 0.511 | 0.032 | 1.19 | 1.23 | 0.396 | 0.663 | 3.30 (-1.04) | 2.74 (+1.31) |
| multisensory_s1_D6_noise1.3_svbmc_seed1098 | 0.450 | 0.032 | 1.15 | 1.18 | 0.386 | 0.657 | 2.31 (-0.33) | 3.02 (+1.02) |
| multisensory_s1_D6_noise1.3_svbmc_seed1099 | 0.229 | 0.029 | 0.51 | 0.66 | 0.353 | 0.608 | 2.42 (-0.44) | 2.04 (+0.85) |
| multisensory_s1_D6_noise1.3_svbmc_seed1100 | 0.535 | 0.034 | 1.42 | 1.42 | 0.404 | 0.693 | 2.41 (-0.83) | 3.00 (+1.97) |
| multisensory_s1_D6_noise1.3_svbmc_seed1101 | 0.289 | 0.028 | 0.91 | 0.93 | 0.357 | 0.572 | 2.31 (+1.41) | 2.22 (-1.14) |

## multisensory_s1_D6_noise3_svbmc (noisy)

### M = 2 (20 cells)

| Estimator | bias | bias with the arm's entropy | covered | mc_se | gp_sd (indep / corr) |
|---|---:|---:|---:|---:|---:|
| raw | 0.785 [0.626, 0.918] | 0.768 [0.664, 0.939] | - | - | - |
| capped_I | 0.406 [0.147, 0.492] | 0.379 [0.136, 0.480] | - | - | - |
| capped_E | 0.643 [0.565, 0.739] | 0.657 [0.533, 0.737] | - | - | - |
| honest ratio1.5 median | 0.010 [-0.158, 0.193] | 0.021 [-0.196, 0.164] | 0.850 | 0.027 | 0.135 / 0.635 |
| honest ratio1.5 precision | 0.010 [-0.158, 0.193] | 0.021 [-0.196, 0.164] | 0.850 | 0.027 | 0.135 / 0.635 |
| honest ratio1.5 mean | 0.010 [-0.158, 0.193] | 0.021 [-0.196, 0.164] | 0.850 | 0.027 | 0.135 / 0.635 |
| honest ratio2 median | -0.299 [-0.594, -0.025] | -0.299 [-0.575, -0.035] | 0.974 | 0.031 | 0.150 / 0.684 |
| honest ratio2 precision | -0.299 [-0.594, -0.025] | -0.299 [-0.575, -0.035] | 0.974 | 0.031 | 0.150 / 0.684 |
| honest ratio2 mean | -0.299 [-0.594, -0.025] | -0.299 [-0.575, -0.035] | 0.974 | 0.031 | 0.150 / 0.684 |
| honest ratio3 median | -0.456 [-0.942, -0.052] | -0.422 [-0.969, -0.053] | 0.980 | 0.031 | 0.161 / 0.733 |
| honest ratio3 precision | -0.456 [-0.942, -0.052] | -0.422 [-0.969, -0.053] | 0.980 | 0.031 | 0.161 / 0.733 |
| honest ratio3 mean | -0.456 [-0.942, -0.052] | -0.422 [-0.969, -0.053] | 0.980 | 0.031 | 0.161 / 0.733 |
| honest ratio5 median | -0.456 [-0.966, -0.052] | -0.422 [-0.976, -0.053] | 0.980 | 0.031 | 0.162 / 0.733 |
| honest ratio5 precision | -0.456 [-0.966, -0.052] | -0.422 [-0.976, -0.053] | 0.980 | 0.031 | 0.162 / 0.733 |
| honest ratio5 mean | -0.456 [-0.966, -0.052] | -0.422 [-0.976, -0.053] | 0.980 | 0.031 | 0.162 / 0.733 |
| honest cap_only median | -0.456 [-0.966, -0.052] | -0.422 [-0.976, -0.053] | 0.980 | 0.031 | 0.162 / 0.733 |
| honest cap_only precision | -0.456 [-0.966, -0.052] | -0.422 [-0.976, -0.053] | 0.980 | 0.031 | 0.162 / 0.733 |
| honest cap_only mean | -0.456 [-0.966, -0.052] | -0.422 [-0.976, -0.053] | 0.980 | 0.031 | 0.162 / 0.733 |
| honest none median | -0.916 [-1.956, -0.107] | -0.901 [-1.976, -0.142] | 1.000 | 0.035 | 0.181 / 0.784 |
| honest none precision | -0.916 [-1.956, -0.107] | -0.901 [-1.976, -0.142] | 1.000 | 0.035 | 0.181 / 0.784 |
| honest none mean | -0.916 [-1.956, -0.107] | -0.901 [-1.976, -0.142] | 1.000 | 0.035 | 0.181 / 0.784 |
| pooled (ratio2, own included) | 0.208 [0.074, 0.334] | - | - | - | - |
| truth_strat | -0.004 [-0.020, 0.013] | - | - | - | - |

Reference `e_log_joint_mc_sd` median 0.016; the headline bias is within 2 SE in 25% of the cells; median absolute bias raw 0.785, capped_I 0.406, capped_E 0.643, honest 0.383 (against raw the honest estimate is better, against capped_I better; tie band 0.016, one reference standard error). Calibration of the self-reported SDs (weighted over components): own z median 1.46 [1.10, 1.71], abs z median 1.47 [1.13, 1.71], abs z > 2 fraction 0.27 [0.15, 0.39]; cross z median -0.52 [-0.99, -0.03], abs z median 0.65 [0.49, 0.97], abs z > 2 fraction 0.05 [0.00, 0.13]. Effective training points on a component (weighted median): own run 0.6 [0.5, 0.6], covering other runs 0.4 [0.2, 0.5]. Checks: own-run max abs z 3.98 (weighted offset abs z 3.05), Jacobian max abs z 3.56 (offset abs z 2.60), raw consistency 1.7e-13, flagged cells 0.

Heaviest component of every run, repetition 0:

| Run, heaviest component (weight) | truth | own I_corr (error; BQ sd; points within 2 SD, their data offset; mean fn) | other runs: estimate (error; pred sd; covered; points within 2 SD, their data offset; mean fn) |
|---|---:|---|---|
| multisensory_s1_D6_noise3_svbmc_seed1052, component 2 (0.088) | -505.401 ± 0.155 | -504.605 (+0.795; 0.453; 1, -0.877; -558.5) | multisensory_s1_D6_noise3_svbmc_seed1093: -506.835 (-1.434; 1.302; yes; 0, -; -544.1) |
| multisensory_s1_D6_noise3_svbmc_seed1093, component 6 (0.021) | -506.335 ± 0.166 | -505.722 (+0.613; 0.799; 0, -; -542.6) | multisensory_s1_D6_noise3_svbmc_seed1052: -507.505 (-1.170; 1.219; yes; 1, -2.618; -558.5) |

### M = 4 (20 cells)

| Estimator | bias | bias with the arm's entropy | covered | mc_se | gp_sd (indep / corr) |
|---|---:|---:|---:|---:|---:|
| raw | 1.071 [0.951, 1.150] | 1.072 [0.961, 1.159] | - | - | - |
| capped_I | 0.555 [0.413, 0.767] | 0.558 [0.429, 0.794] | - | - | - |
| capped_E | 0.845 [0.797, 1.003] | 0.849 [0.810, 1.009] | - | - | - |
| honest ratio1.5 median | -0.250 [-0.417, -0.103] | -0.241 [-0.393, -0.098] | 0.990 | 0.028 | 0.126 / 0.510 |
| honest ratio1.5 precision | -0.165 [-0.354, -0.087] | -0.156 [-0.339, -0.086] | 0.990 | 0.027 | 0.087 / 0.492 |
| honest ratio1.5 mean | -0.272 [-0.442, -0.134] | -0.263 [-0.439, -0.133] | 0.990 | 0.027 | 0.088 / 0.488 |
| honest ratio2 median | -0.479 [-0.632, -0.161] | -0.474 [-0.622, -0.170] | 0.999 | 0.029 | 0.133 / 0.539 |
| honest ratio2 precision | -0.349 [-0.549, -0.187] | -0.334 [-0.546, -0.176] | 0.999 | 0.027 | 0.083 / 0.496 |
| honest ratio2 mean | -0.515 [-0.696, -0.335] | -0.480 [-0.693, -0.317] | 0.999 | 0.027 | 0.087 / 0.505 |
| honest ratio3 median | -0.493 [-0.757, -0.207] | -0.498 [-0.744, -0.207] | 1.000 | 0.029 | 0.138 / 0.573 |
| honest ratio3 precision | -0.407 [-0.560, -0.209] | -0.372 [-0.561, -0.209] | 1.000 | 0.027 | 0.083 / 0.501 |
| honest ratio3 mean | -0.630 [-0.862, -0.396] | -0.601 [-0.845, -0.381] | 1.000 | 0.027 | 0.088 / 0.525 |
| honest ratio5 median | -0.493 [-0.756, -0.205] | -0.498 [-0.743, -0.208] | 1.000 | 0.029 | 0.138 / 0.575 |
| honest ratio5 precision | -0.407 [-0.560, -0.209] | -0.378 [-0.561, -0.210] | 1.000 | 0.027 | 0.083 / 0.501 |
| honest ratio5 mean | -0.641 [-0.865, -0.396] | -0.612 [-0.848, -0.380] | 1.000 | 0.027 | 0.088 / 0.527 |
| honest cap_only median | -0.493 [-0.756, -0.205] | -0.498 [-0.743, -0.208] | 1.000 | 0.029 | 0.138 / 0.575 |
| honest cap_only precision | -0.407 [-0.560, -0.209] | -0.378 [-0.561, -0.210] | 1.000 | 0.027 | 0.083 / 0.501 |
| honest cap_only mean | -0.641 [-0.865, -0.396] | -0.612 [-0.848, -0.380] | 1.000 | 0.027 | 0.088 / 0.527 |
| honest none median | -0.749 [-1.408, -0.496] | -0.732 [-1.401, -0.494] | 1.000 | 0.029 | 0.152 / 0.625 |
| honest none precision | -0.546 [-1.281, -0.303] | -0.535 [-1.272, -0.294] | 1.000 | 0.027 | 0.084 / 0.517 |
| honest none mean | -0.900 [-1.720, -0.612] | -0.872 [-1.714, -0.603] | 1.000 | 0.028 | 0.093 / 0.564 |
| pooled (ratio2, own included) | -0.018 [-0.227, 0.095] | - | - | - | - |
| truth_strat | 0.000 [-0.010, 0.020] | - | - | - | - |

Reference `e_log_joint_mc_sd` median 0.016; the headline bias is within 2 SE in 10% of the cells; median absolute bias raw 1.071, capped_I 0.555, capped_E 0.845, honest 0.479 (against raw the honest estimate is better, against capped_I better; tie band 0.016, one reference standard error). Calibration of the self-reported SDs (weighted over components): own z median 2.01 [1.81, 2.38], abs z median 2.01 [1.81, 2.38], abs z > 2 fraction 0.52 [0.41, 0.63]; cross z median -0.22 [-0.48, -0.08], abs z median 0.68 [0.60, 0.72], abs z > 2 fraction 0.08 [0.07, 0.11]. Effective training points on a component (weighted median): own run 0.6 [0.5, 0.7], covering other runs 0.3 [0.3, 0.4]. Checks: own-run max abs z 4.16 (weighted offset abs z 3.61), Jacobian max abs z 3.87 (offset abs z 2.51), raw consistency 1.1e-13, flagged cells 0.

Heaviest component of every run, repetition 0:

| Run, heaviest component (weight) | truth | own I_corr (error; BQ sd; points within 2 SD, their data offset; mean fn) | other runs: estimate (error; pred sd; covered; points within 2 SD, their data offset; mean fn) |
|---|---:|---|---|
| multisensory_s1_D6_noise3_svbmc_seed1042, component 1 (0.042) | -505.390 ± 0.147 | -504.103 (+1.287; 0.440; 2, 1.423; -564.5) | multisensory_s1_D6_noise3_svbmc_seed1066: -505.871 (-0.481; 0.938; yes; 1, 0.314; -539.4); multisensory_s1_D6_noise3_svbmc_seed1079: -505.149 (+0.241; 0.805; yes; 2, 2.311; -540.5); multisensory_s1_D6_noise3_svbmc_seed1085: -504.633 (+0.757; 0.733; yes; 2, 0.642; -558.2) |
| multisensory_s1_D6_noise3_svbmc_seed1066, component 1 (0.018) | -505.185 ± 0.121 | -504.712 (+0.473; 0.523; 2, -1.889; -530.5) | multisensory_s1_D6_noise3_svbmc_seed1042: -505.218 (-0.033; 1.010; yes; 1, -1.707; -562.6); multisensory_s1_D6_noise3_svbmc_seed1079: -505.479 (-0.295; 0.937; yes; 2, 2.895; -534.0); multisensory_s1_D6_noise3_svbmc_seed1085: -505.388 (-0.203; 0.944; yes; 0, -; -544.9) |
| multisensory_s1_D6_noise3_svbmc_seed1079, component 23 (0.014) | -506.115 ± 0.164 | -504.910 (+1.205; 0.446; 3, -1.889; -538.3) | multisensory_s1_D6_noise3_svbmc_seed1042: -505.479 (+0.636; 0.905; yes; 0, -; -564.1); multisensory_s1_D6_noise3_svbmc_seed1066: -506.602 (-0.487; 0.997; yes; 1, -2.632; -536.8); multisensory_s1_D6_noise3_svbmc_seed1085: -505.594 (+0.521; 0.831; yes; 1, 0.603; -553.2) |
| multisensory_s1_D6_noise3_svbmc_seed1085, component 1 (0.042) | -505.883 ± 0.142 | -504.468 (+1.415; 0.467; 2, 1.027; -565.3) | multisensory_s1_D6_noise3_svbmc_seed1042: -504.869 (+1.014; 0.783; yes; 0, -; -565.1); multisensory_s1_D6_noise3_svbmc_seed1066: -506.523 (-0.640; 1.053; yes; 1, 1.272; -545.1); multisensory_s1_D6_noise3_svbmc_seed1079: -506.303 (-0.420; 0.856; yes; 0, -; -543.6) |

### M = 8 (20 cells)

| Estimator | bias | bias with the arm's entropy | covered | mc_se | gp_sd (indep / corr) |
|---|---:|---:|---:|---:|---:|
| raw | 1.173 [1.070, 1.304] | 1.163 [1.078, 1.277] | - | - | - |
| capped_I | 0.401 [0.316, 0.513] | 0.399 [0.300, 0.493] | - | - | - |
| capped_E | 0.682 [0.581, 0.800] | 0.676 [0.576, 0.822] | - | - | - |
| honest ratio1.5 median | -0.333 [-0.422, -0.108] | -0.359 [-0.412, -0.113] | 0.983 | 0.022 | 0.104 / 0.378 |
| honest ratio1.5 precision | -0.339 [-0.406, -0.081] | -0.330 [-0.403, -0.081] | 0.983 | 0.022 | 0.062 / 0.353 |
| honest ratio1.5 mean | -0.410 [-0.528, -0.145] | -0.423 [-0.501, -0.143] | 0.983 | 0.022 | 0.063 / 0.358 |
| honest ratio2 median | -0.713 [-0.796, -0.523] | -0.704 [-0.814, -0.531] | 0.999 | 0.024 | 0.113 / 0.420 |
| honest ratio2 precision | -0.609 [-0.669, -0.578] | -0.620 [-0.674, -0.538] | 0.999 | 0.022 | 0.064 / 0.383 |
| honest ratio2 mean | -0.836 [-0.884, -0.777] | -0.827 [-0.908, -0.746] | 0.999 | 0.023 | 0.067 / 0.400 |
| honest ratio3 median | -1.023 [-1.306, -0.831] | -1.035 [-1.316, -0.831] | 0.999 | 0.029 | 0.124 / 0.459 |
| honest ratio3 precision | -0.755 [-0.946, -0.622] | -0.754 [-0.971, -0.638] | 0.999 | 0.023 | 0.064 / 0.390 |
| honest ratio3 mean | -1.202 [-1.358, -1.061] | -1.182 [-1.369, -1.058] | 0.999 | 0.024 | 0.068 / 0.421 |
| honest ratio5 median | -1.031 [-1.349, -0.837] | -1.045 [-1.371, -0.837] | 0.999 | 0.029 | 0.124 / 0.452 |
| honest ratio5 precision | -0.758 [-0.949, -0.625] | -0.758 [-0.972, -0.641] | 0.999 | 0.023 | 0.063 / 0.390 |
| honest ratio5 mean | -1.209 [-1.413, -1.072] | -1.189 [-1.419, -1.061] | 0.999 | 0.024 | 0.068 / 0.424 |
| honest cap_only median | -1.031 [-1.349, -0.837] | -1.045 [-1.371, -0.837] | 0.999 | 0.029 | 0.124 / 0.452 |
| honest cap_only precision | -0.758 [-0.949, -0.625] | -0.758 [-0.972, -0.641] | 0.999 | 0.023 | 0.063 / 0.390 |
| honest cap_only mean | -1.209 [-1.413, -1.072] | -1.189 [-1.419, -1.061] | 0.999 | 0.024 | 0.068 / 0.424 |
| honest none median | -3.171 [-4.424, -1.612] | -3.185 [-4.461, -1.599] | 1.000 | 0.037 | 0.208 / 0.789 |
| honest none precision | -1.257 [-2.169, -0.891] | -1.254 [-2.187, -0.868] | 1.000 | 0.024 | 0.065 / 0.439 |
| honest none mean | -3.703 [-5.942, -2.403] | -3.732 [-5.927, -2.379] | 1.000 | 0.037 | 0.119 / 0.771 |
| pooled (ratio2, own included) | -0.349 [-0.410, -0.256] | - | - | - | - |
| truth_strat | -0.008 [-0.015, 0.009] | - | - | - | - |

Reference `e_log_joint_mc_sd` median 0.016; the headline bias is within 2 SE in 0% of the cells; median absolute bias raw 1.173, capped_I 0.401, capped_E 0.682, honest 0.713 (against raw the honest estimate is better, against capped_I **worse**; tie band 0.016, one reference standard error). Calibration of the self-reported SDs (weighted over components): own z median 2.20 [1.96, 2.51], abs z median 2.20 [1.96, 2.51], abs z > 2 fraction 0.58 [0.49, 0.67]; cross z median -0.42 [-0.50, -0.32], abs z median 0.70 [0.62, 0.75], abs z > 2 fraction 0.09 [0.08, 0.11]. Effective training points on a component (weighted median): own run 0.6 [0.5, 0.7], covering other runs 0.3 [0.2, 0.4]. Checks: own-run max abs z 4.94 (weighted offset abs z 2.31), Jacobian max abs z 4.13 (offset abs z 2.20), raw consistency 1.1e-13, flagged cells 0.

### M = 16 (10 cells)

| Estimator | bias | bias with the arm's entropy | covered | mc_se | gp_sd (indep / corr) |
|---|---:|---:|---:|---:|---:|
| raw | 1.315 [1.243, 1.410] | 1.309 [1.256, 1.426] | - | - | - |
| capped_I | 0.416 [0.392, 0.496] | 0.443 [0.395, 0.513] | - | - | - |
| capped_E | 0.804 [0.768, 0.823] | 0.816 [0.765, 0.826] | - | - | - |
| honest ratio1.5 median | -0.310 [-0.403, -0.200] | -0.301 [-0.432, -0.150] | 0.983 | 0.020 | 0.093 / 0.287 |
| honest ratio1.5 precision | -0.353 [-0.415, -0.195] | -0.343 [-0.430, -0.189] | 0.983 | 0.019 | 0.042 / 0.256 |
| honest ratio1.5 mean | -0.442 [-0.523, -0.286] | -0.431 [-0.518, -0.252] | 0.983 | 0.019 | 0.045 / 0.259 |
| honest ratio2 median | -0.678 [-0.791, -0.467] | -0.672 [-0.795, -0.471] | 0.991 | 0.021 | 0.105 / 0.330 |
| honest ratio2 precision | -0.603 [-0.703, -0.475] | -0.585 [-0.741, -0.427] | 0.991 | 0.019 | 0.039 / 0.264 |
| honest ratio2 mean | -0.842 [-0.912, -0.708] | -0.825 [-0.943, -0.674] | 0.991 | 0.019 | 0.041 / 0.285 |
| honest ratio3 median | -0.843 [-1.010, -0.731] | -0.851 [-1.028, -0.691] | 0.999 | 0.023 | 0.121 / 0.347 |
| honest ratio3 precision | -0.738 [-0.848, -0.643] | -0.720 [-0.880, -0.609] | 0.999 | 0.020 | 0.038 / 0.274 |
| honest ratio3 mean | -1.172 [-1.281, -0.960] | -1.179 [-1.288, -0.919] | 0.999 | 0.022 | 0.041 / 0.303 |
| honest ratio5 median | -0.861 [-1.023, -0.734] | -0.869 [-1.008, -0.693] | 0.999 | 0.023 | 0.123 / 0.351 |
| honest ratio5 precision | -0.740 [-0.859, -0.650] | -0.722 [-0.891, -0.616] | 0.999 | 0.020 | 0.038 / 0.275 |
| honest ratio5 mean | -1.188 [-1.288, -0.973] | -1.195 [-1.295, -0.920] | 0.999 | 0.022 | 0.041 / 0.304 |
| honest cap_only median | -0.861 [-1.023, -0.734] | -0.869 [-1.034, -0.693] | 0.999 | 0.023 | 0.123 / 0.351 |
| honest cap_only precision | -0.740 [-0.859, -0.650] | -0.722 [-0.891, -0.616] | 0.999 | 0.020 | 0.038 / 0.275 |
| honest cap_only mean | -1.188 [-1.282, -0.973] | -1.195 [-1.273, -0.920] | 0.999 | 0.022 | 0.041 / 0.304 |
| honest none median | -2.607 [-3.147, -1.871] | -2.606 [-3.017, -1.870] | 1.000 | 0.027 | 0.177 / 0.521 |
| honest none precision | -1.373 [-1.516, -1.219] | -1.395 [-1.516, -1.193] | 1.000 | 0.022 | 0.036 / 0.302 |
| honest none mean | -3.958 [-5.307, -2.652] | -3.966 [-5.291, -2.668] | 1.000 | 0.026 | 0.065 / 0.530 |
| pooled (ratio2, own included) | -0.511 [-0.595, -0.309] | - | - | - | - |
| truth_strat | -0.014 [-0.026, 0.009] | - | - | - | - |

Reference `e_log_joint_mc_sd` median 0.016; the headline bias is within 2 SE in 0% of the cells; median absolute bias raw 1.315, capped_I 0.416, capped_E 0.804, honest 0.678 (against raw the honest estimate is better, against capped_I **worse**; tie band 0.016, one reference standard error). Calibration of the self-reported SDs (weighted over components): own z median 2.65 [2.40, 2.79], abs z median 2.65 [2.40, 2.80], abs z > 2 fraction 0.66 [0.61, 0.76]; cross z median -0.35 [-0.41, -0.28], abs z median 0.66 [0.64, 0.68], abs z > 2 fraction 0.10 [0.08, 0.11]. Effective training points on a component (weighted median): own run 0.6 [0.5, 0.6], covering other runs 0.3 [0.2, 0.4]. Checks: own-run max abs z 5.06 (weighted offset abs z 1.58), Jacobian max abs z 4.19 (offset abs z 1.33), raw consistency 1.1e-13, flagged cells 0.

| Run | own error | se | own z median | own abs z median | BQ sd median | pred sd median | own-run max abs z (offset z) | Jacobian max abs z (offset z) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| multisensory_s1_D6_noise3_svbmc_seed1000 | 1.683 | 0.023 | 2.65 | 2.65 | 0.652 | 1.071 | 2.07 (-2.48) | 2.16 (+1.24) |
| multisensory_s1_D6_noise3_svbmc_seed1001 | 0.842 | 0.031 | 1.95 | 1.95 | 0.471 | 0.766 | 2.48 (-1.39) | 3.09 (+0.56) |
| multisensory_s1_D6_noise3_svbmc_seed1002 | 0.275 | 0.029 | 0.66 | 0.69 | 0.609 | 0.925 | 4.13 (-0.53) | 2.23 (+1.13) |
| multisensory_s1_D6_noise3_svbmc_seed1003 | 0.956 | 0.031 | 2.18 | 2.18 | 0.498 | 0.765 | 2.70 (+0.13) | 2.23 (-0.48) |
| multisensory_s1_D6_noise3_svbmc_seed1004 | 0.275 | 0.041 | 0.57 | 0.73 | 0.660 | 1.038 | 3.76 (+0.26) | 2.15 (-0.61) |
| multisensory_s1_D6_noise3_svbmc_seed1005 | 0.372 | 0.026 | 0.83 | 1.30 | 0.554 | 0.782 | 2.97 (+0.71) | 2.01 (-1.71) |
| multisensory_s1_D6_noise3_svbmc_seed1006 | 0.623 | 0.033 | 1.31 | 1.45 | 0.471 | 0.717 | 2.35 (-0.56) | 2.66 (-0.20) |
| multisensory_s1_D6_noise3_svbmc_seed1007 | 0.408 | 0.026 | 0.75 | 0.76 | 0.526 | 0.833 | 2.44 (+1.43) | 2.29 (+0.95) |
| multisensory_s1_D6_noise3_svbmc_seed1008 | -0.161 | 0.043 | -0.33 | 0.34 | 0.583 | 0.887 | 3.20 (-0.67) | 2.40 (+0.68) |
| multisensory_s1_D6_noise3_svbmc_seed1009 | 0.829 | 0.031 | 1.94 | 1.94 | 0.574 | 0.860 | 2.67 (+0.59) | 2.85 (-0.03) |
| multisensory_s1_D6_noise3_svbmc_seed1010 | 0.479 | 0.034 | 0.87 | 0.93 | 0.550 | 0.860 | 3.29 (-0.49) | 1.81 (+0.18) |
| multisensory_s1_D6_noise3_svbmc_seed1011 | 0.573 | 0.033 | 1.17 | 1.17 | 0.575 | 0.885 | 3.86 (-0.54) | 2.74 (-0.09) |
| multisensory_s1_D6_noise3_svbmc_seed1012 | 0.416 | 0.037 | 0.85 | 0.85 | 0.506 | 0.807 | 2.34 (+2.71) | 3.40 (-1.17) |
| multisensory_s1_D6_noise3_svbmc_seed1013 | 0.658 | 0.025 | 1.57 | 1.57 | 0.559 | 0.877 | 2.52 (+0.72) | 2.17 (+0.92) |
| multisensory_s1_D6_noise3_svbmc_seed1014 | 1.154 | 0.034 | 2.83 | 2.83 | 0.443 | 0.681 | 2.50 (+0.36) | 2.31 (+0.64) |
| multisensory_s1_D6_noise3_svbmc_seed1015 | 0.804 | 0.026 | 1.88 | 1.88 | 0.493 | 0.742 | 3.27 (+0.22) | 2.46 (-0.49) |
| multisensory_s1_D6_noise3_svbmc_seed1016 | 0.668 | 0.040 | 1.72 | 1.72 | 0.517 | 0.796 | 2.63 (-1.00) | 2.38 (-0.34) |
| multisensory_s1_D6_noise3_svbmc_seed1017 | 0.804 | 0.030 | 1.48 | 1.48 | 0.684 | 1.020 | 2.32 (+0.53) | 2.16 (+0.60) |
| multisensory_s1_D6_noise3_svbmc_seed1018 | 0.684 | 0.044 | 1.45 | 1.45 | 0.770 | 1.229 | 3.31 (-0.11) | 2.01 (+0.49) |
| multisensory_s1_D6_noise3_svbmc_seed1019 | 0.640 | 0.023 | 1.08 | 1.08 | 0.646 | 1.063 | 2.66 (+1.58) | 2.94 (-0.88) |
| multisensory_s1_D6_noise3_svbmc_seed1020 | 0.566 | 0.033 | 1.19 | 1.19 | 0.590 | 0.861 | 2.62 (+0.03) | 3.17 (+0.60) |
| multisensory_s1_D6_noise3_svbmc_seed1021 | 0.911 | 0.038 | 1.32 | 1.32 | 0.640 | 0.990 | 2.40 (-1.43) | 2.94 (+1.42) |
| multisensory_s1_D6_noise3_svbmc_seed1022 | 0.414 | 0.027 | 0.70 | 0.70 | 0.594 | 0.997 | 2.38 (+0.53) | 2.11 (+0.38) |
| multisensory_s1_D6_noise3_svbmc_seed1023 | 1.123 | 0.029 | 1.95 | 1.95 | 0.628 | 0.996 | 2.88 (-0.37) | 2.00 (-0.03) |
| multisensory_s1_D6_noise3_svbmc_seed1024 | 0.535 | 0.032 | 0.97 | 0.97 | 0.623 | 0.915 | 2.94 (-1.76) | 2.40 (+0.75) |
| multisensory_s1_D6_noise3_svbmc_seed1025 | 1.171 | 0.039 | 2.78 | 2.78 | 0.505 | 0.776 | 2.31 (-0.44) | 2.74 (+0.40) |
| multisensory_s1_D6_noise3_svbmc_seed1026 | 1.046 | 0.025 | 1.67 | 1.67 | 0.679 | 0.994 | 1.98 (+0.06) | 2.30 (+1.34) |
| multisensory_s1_D6_noise3_svbmc_seed1027 | 1.043 | 0.029 | 2.06 | 2.06 | 0.542 | 0.839 | 2.85 (-0.65) | 2.53 (+0.18) |
| multisensory_s1_D6_noise3_svbmc_seed1028 | 1.335 | 0.036 | 1.99 | 1.99 | 0.650 | 0.987 | 2.93 (-1.31) | 2.31 (+1.16) |
| multisensory_s1_D6_noise3_svbmc_seed1029 | 1.223 | 0.037 | 1.37 | 1.37 | 0.785 | 1.149 | 3.46 (-1.04) | 2.29 (-1.06) |
| multisensory_s1_D6_noise3_svbmc_seed1030 | 0.433 | 0.031 | 0.73 | 0.82 | 0.556 | 0.867 | 2.30 (-0.17) | 2.67 (-0.87) |
| multisensory_s1_D6_noise3_svbmc_seed1031 | 0.689 | 0.028 | 1.31 | 1.31 | 0.626 | 1.020 | 2.50 (+0.21) | 2.63 (+0.48) |
| multisensory_s1_D6_noise3_svbmc_seed1032 | 1.135 | 0.029 | 1.88 | 1.88 | 0.699 | 1.147 | 3.03 (+2.08) | 2.39 (+0.54) |
| multisensory_s1_D6_noise3_svbmc_seed1033 | 0.786 | 0.027 | 1.51 | 1.51 | 0.553 | 0.842 | 2.57 (-0.55) | 2.72 (-0.78) |
| multisensory_s1_D6_noise3_svbmc_seed1034 | 1.083 | 0.028 | 2.27 | 2.27 | 0.547 | 0.930 | 2.59 (+0.04) | 2.64 (-1.40) |
| multisensory_s1_D6_noise3_svbmc_seed1035 | 0.488 | 0.040 | 1.06 | 1.07 | 0.564 | 0.821 | 2.04 (-0.49) | 2.15 (-0.35) |
| multisensory_s1_D6_noise3_svbmc_seed1036 | 0.939 | 0.029 | 1.88 | 1.88 | 0.546 | 0.857 | 3.44 (+0.74) | 2.16 (+0.55) |
| multisensory_s1_D6_noise3_svbmc_seed1037 | 1.108 | 0.028 | 1.43 | 1.43 | 0.783 | 1.203 | 2.24 (+0.95) | 1.98 (-2.40) |
| multisensory_s1_D6_noise3_svbmc_seed1038 | 0.243 | 0.029 | 0.38 | 0.83 | 0.662 | 0.952 | 1.95 (+1.80) | 2.29 (+1.21) |
| multisensory_s1_D6_noise3_svbmc_seed1039 | 0.803 | 0.030 | 1.97 | 1.97 | 0.504 | 0.710 | 2.24 (-1.56) | 2.67 (+0.45) |
| multisensory_s1_D6_noise3_svbmc_seed1040 | 0.423 | 0.030 | 0.42 | 0.43 | 0.615 | 0.908 | 2.62 (+1.85) | 3.17 (+0.42) |
| multisensory_s1_D6_noise3_svbmc_seed1041 | 1.022 | 0.032 | 2.02 | 2.02 | 0.592 | 0.898 | 1.75 (-1.14) | 2.53 (-1.44) |
| multisensory_s1_D6_noise3_svbmc_seed1042 | 1.070 | 0.034 | 2.15 | 2.15 | 0.590 | 0.883 | 2.66 (+0.55) | 2.60 (-0.63) |
| multisensory_s1_D6_noise3_svbmc_seed1043 | 0.557 | 0.030 | 1.11 | 1.11 | 0.524 | 0.737 | 2.46 (+0.79) | 2.52 (-0.84) |
| multisensory_s1_D6_noise3_svbmc_seed1044 | 0.140 | 0.023 | 0.27 | 0.46 | 0.677 | 1.043 | 2.58 (+0.35) | 3.78 (-0.05) |
| multisensory_s1_D6_noise3_svbmc_seed1045 | 0.825 | 0.024 | 1.62 | 1.62 | 0.510 | 0.769 | 3.98 (+1.19) | 2.55 (-1.57) |
| multisensory_s1_D6_noise3_svbmc_seed1046 | 0.480 | 0.036 | 0.90 | 0.91 | 0.572 | 0.877 | 2.30 (-0.42) | 2.42 (+1.53) |
| multisensory_s1_D6_noise3_svbmc_seed1047 | 0.415 | 0.044 | 0.62 | 0.70 | 0.587 | 0.972 | 2.87 (-0.75) | 2.57 (+1.33) |
| multisensory_s1_D6_noise3_svbmc_seed1048 | 1.480 | 0.033 | 3.42 | 3.42 | 0.480 | 0.750 | 2.19 (-2.01) | 2.56 (+0.09) |
| multisensory_s1_D6_noise3_svbmc_seed1049 | 1.182 | 0.041 | 2.39 | 2.39 | 0.538 | 0.826 | 3.69 (-0.83) | 3.55 (-0.04) |
| multisensory_s1_D6_noise3_svbmc_seed1050 | 0.669 | 0.027 | 1.17 | 1.17 | 0.608 | 0.959 | 1.75 (+0.52) | 2.33 (-0.09) |
| multisensory_s1_D6_noise3_svbmc_seed1051 | 0.656 | 0.029 | 1.38 | 1.38 | 0.526 | 0.824 | 3.58 (+0.99) | 3.56 (-0.52) |
| multisensory_s1_D6_noise3_svbmc_seed1052 | 0.543 | 0.029 | 1.13 | 1.13 | 0.544 | 0.812 | 2.24 (+2.12) | 2.69 (-0.54) |
| multisensory_s1_D6_noise3_svbmc_seed1053 | 1.054 | 0.025 | 1.80 | 1.80 | 0.574 | 0.950 | 2.71 (+0.16) | 2.76 (+0.07) |
| multisensory_s1_D6_noise3_svbmc_seed1054 | 0.831 | 0.025 | 1.49 | 1.49 | 0.567 | 0.851 | 2.60 (+1.10) | 2.21 (+0.17) |
| multisensory_s1_D6_noise3_svbmc_seed1055 | 1.163 | 0.030 | 2.24 | 2.24 | 0.595 | 0.899 | 3.42 (+0.65) | 2.06 (-0.14) |
| multisensory_s1_D6_noise3_svbmc_seed1056 | 0.415 | 0.031 | 0.82 | 0.82 | 0.546 | 0.826 | 1.90 (-0.41) | 2.05 (+0.14) |
| multisensory_s1_D6_noise3_svbmc_seed1057 | 0.778 | 0.030 | 1.71 | 1.71 | 0.533 | 0.796 | 2.29 (+0.52) | 3.53 (+1.11) |
| multisensory_s1_D6_noise3_svbmc_seed1058 | 0.343 | 0.031 | 0.55 | 0.56 | 0.685 | 1.013 | 2.41 (+0.79) | 2.17 (-0.39) |
| multisensory_s1_D6_noise3_svbmc_seed1059 | 0.489 | 0.026 | 0.81 | 0.83 | 0.591 | 0.917 | 3.90 (+1.35) | 2.62 (-2.44) |
| multisensory_s1_D6_noise3_svbmc_seed1060 | 0.328 | 0.023 | 0.65 | 0.72 | 0.538 | 0.910 | 2.95 (-0.42) | 2.76 (+0.54) |
| multisensory_s1_D6_noise3_svbmc_seed1061 | 0.928 | 0.035 | 2.06 | 2.06 | 0.460 | 0.710 | 2.68 (+0.24) | 2.32 (-0.57) |
| multisensory_s1_D6_noise3_svbmc_seed1062 | 0.687 | 0.032 | 1.23 | 1.23 | 0.709 | 1.124 | 4.56 (-0.06) | 2.33 (+1.23) |
| multisensory_s1_D6_noise3_svbmc_seed1063 | 0.047 | 0.027 | 0.17 | 0.28 | 0.491 | 0.747 | 2.58 (-0.15) | 2.50 (+0.09) |
| multisensory_s1_D6_noise3_svbmc_seed1064 | 0.809 | 0.027 | 1.46 | 1.46 | 0.621 | 1.030 | 2.15 (+0.63) | 2.53 (+0.32) |
| multisensory_s1_D6_noise3_svbmc_seed1065 | 0.858 | 0.032 | 1.34 | 1.35 | 0.829 | 1.244 | 2.32 (+0.30) | 2.44 (-0.45) |
| multisensory_s1_D6_noise3_svbmc_seed1066 | 0.788 | 0.030 | 1.03 | 1.03 | 0.647 | 0.983 | 2.11 (-0.42) | 2.64 (+0.65) |
| multisensory_s1_D6_noise3_svbmc_seed1067 | 0.936 | 0.031 | 2.14 | 2.14 | 0.505 | 0.811 | 2.44 (-0.60) | 2.55 (-0.01) |
| multisensory_s1_D6_noise3_svbmc_seed1068 | 0.674 | 0.035 | 1.63 | 1.63 | 0.523 | 0.763 | 2.00 (-1.04) | 2.26 (-0.43) |
| multisensory_s1_D6_noise3_svbmc_seed1069 | 1.440 | 0.031 | 3.45 | 3.45 | 0.513 | 0.831 | 2.26 (-2.22) | 2.07 (+1.65) |
| multisensory_s1_D6_noise3_svbmc_seed1070 | 0.844 | 0.025 | 1.68 | 1.68 | 0.573 | 0.874 | 2.37 (+0.38) | 2.14 (-0.26) |
| multisensory_s1_D6_noise3_svbmc_seed1071 | 0.888 | 0.029 | 2.04 | 2.04 | 0.515 | 0.719 | 2.26 (-0.48) | 2.65 (+0.87) |
| multisensory_s1_D6_noise3_svbmc_seed1072 | 0.751 | 0.032 | 1.56 | 1.56 | 0.562 | 0.825 | 3.97 (-0.39) | 1.82 (-1.93) |
| multisensory_s1_D6_noise3_svbmc_seed1073 | 1.602 | 0.026 | 3.41 | 3.41 | 0.654 | 0.997 | 2.41 (+0.42) | 2.83 (-0.13) |
| multisensory_s1_D6_noise3_svbmc_seed1074 | 0.703 | 0.029 | 1.33 | 1.33 | 0.556 | 0.830 | 2.62 (+0.81) | 4.26 (-0.86) |
| multisensory_s1_D6_noise3_svbmc_seed1075 | 0.764 | 0.027 | 1.78 | 1.78 | 0.472 | 0.709 | 2.00 (+0.05) | 2.60 (+0.12) |
| multisensory_s1_D6_noise3_svbmc_seed1076 | 1.242 | 0.051 | 2.48 | 2.48 | 0.596 | 0.855 | 2.99 (+0.68) | 3.08 (-0.32) |
| multisensory_s1_D6_noise3_svbmc_seed1078 | 0.647 | 0.032 | 0.82 | 0.82 | 0.727 | 1.157 | 2.42 (+0.44) | 3.15 (+0.98) |
| multisensory_s1_D6_noise3_svbmc_seed1079 | 0.883 | 0.025 | 1.83 | 1.83 | 0.520 | 0.793 | 2.47 (+1.80) | 2.72 (+0.84) |
| multisensory_s1_D6_noise3_svbmc_seed1080 | 0.840 | 0.025 | 1.75 | 1.75 | 0.562 | 0.847 | 2.79 (+0.54) | 3.20 (+1.14) |
| multisensory_s1_D6_noise3_svbmc_seed1081 | 0.399 | 0.038 | 0.87 | 0.91 | 0.553 | 0.777 | 2.69 (+0.33) | 1.86 (+0.15) |
| multisensory_s1_D6_noise3_svbmc_seed1082 | 0.488 | 0.032 | 1.00 | 1.10 | 0.566 | 0.814 | 2.84 (+0.10) | 2.10 (-0.03) |
| multisensory_s1_D6_noise3_svbmc_seed1083 | -0.056 | 0.037 | -0.19 | 0.46 | 0.690 | 1.083 | 2.69 (-0.86) | 2.86 (+2.11) |
| multisensory_s1_D6_noise3_svbmc_seed1084 | 0.963 | 0.033 | 2.22 | 2.22 | 0.530 | 0.784 | 2.26 (+0.43) | 3.98 (-0.46) |
| multisensory_s1_D6_noise3_svbmc_seed1085 | 0.913 | 0.032 | 1.80 | 1.80 | 0.572 | 0.820 | 3.30 (+0.10) | 3.01 (+2.09) |
| multisensory_s1_D6_noise3_svbmc_seed1086 | 0.984 | 0.032 | 2.03 | 2.03 | 0.512 | 0.833 | 2.85 (+0.29) | 2.51 (+1.46) |
| multisensory_s1_D6_noise3_svbmc_seed1088 | 0.445 | 0.027 | 0.84 | 0.90 | 0.626 | 0.963 | 2.53 (+0.60) | 2.32 (+1.02) |
| multisensory_s1_D6_noise3_svbmc_seed1089 | 0.837 | 0.029 | 1.53 | 1.53 | 0.559 | 0.867 | 2.33 (-0.08) | 2.15 (-0.25) |
| multisensory_s1_D6_noise3_svbmc_seed1090 | 0.636 | 0.025 | 1.43 | 1.43 | 0.584 | 0.892 | 2.55 (-0.58) | 2.57 (-0.07) |
| multisensory_s1_D6_noise3_svbmc_seed1091 | 0.749 | 0.031 | 1.77 | 1.78 | 0.491 | 0.778 | 3.00 (-1.19) | 2.68 (+0.15) |
| multisensory_s1_D6_noise3_svbmc_seed1092 | 0.094 | 0.035 | 0.39 | 0.71 | 0.557 | 0.854 | 2.21 (-1.59) | 2.21 (+0.34) |
| multisensory_s1_D6_noise3_svbmc_seed1093 | -0.039 | 0.027 | -0.19 | 0.43 | 0.746 | 1.233 | 2.83 (-0.41) | 2.23 (+1.21) |
| multisensory_s1_D6_noise3_svbmc_seed1094 | 1.348 | 0.030 | 2.40 | 2.40 | 0.619 | 0.973 | 2.71 (-1.40) | 1.82 (+0.15) |
| multisensory_s1_D6_noise3_svbmc_seed1095 | 0.668 | 0.031 | 1.76 | 1.76 | 0.452 | 0.692 | 2.96 (+0.31) | 2.54 (-0.11) |
| multisensory_s1_D6_noise3_svbmc_seed1096 | 0.846 | 0.024 | 2.01 | 2.01 | 0.512 | 0.790 | 3.06 (+0.56) | 2.30 (-0.52) |
| multisensory_s1_D6_noise3_svbmc_seed1097 | 0.626 | 0.031 | 1.18 | 1.26 | 0.545 | 0.839 | 3.11 (+2.26) | 2.42 (+0.71) |
| multisensory_s1_D6_noise3_svbmc_seed1098 | 0.056 | 0.033 | 0.25 | 0.82 | 0.612 | 1.009 | 2.27 (-0.58) | 1.86 (+0.71) |
| multisensory_s1_D6_noise3_svbmc_seed1099 | 0.428 | 0.045 | 1.10 | 1.10 | 0.658 | 1.020 | 2.60 (-0.26) | 3.67 (+1.96) |

## multisensory_s1_D6_svbmc (noiseless)

### M = 2 (20 cells)

| Estimator | bias | bias with the arm's entropy | covered | mc_se | gp_sd (indep / corr) |
|---|---:|---:|---:|---:|---:|
| raw | 0.045 [0.029, 0.061] | 0.029 [0.020, 0.061] | - | - | - |
| capped_I | -0.042 [-0.079, 0.010] | -0.054 [-0.100, -0.002] | - | - | - |
| capped_E | 0.045 [0.029, 0.058] | 0.029 [0.017, 0.061] | - | - | - |
| honest ratio1.5 median | 0.021 [0.009, 0.036] | 0.023 [-0.012, 0.043] | 0.626 | 0.020 | 0.037 / 0.156 |
| honest ratio1.5 precision | 0.021 [0.009, 0.036] | 0.023 [-0.012, 0.043] | 0.626 | 0.020 | 0.037 / 0.156 |
| honest ratio1.5 mean | 0.021 [0.009, 0.036] | 0.023 [-0.012, 0.043] | 0.626 | 0.020 | 0.037 / 0.156 |
| honest ratio2 median | 0.003 [-0.033, 0.013] | -0.009 [-0.038, 0.015] | 0.854 | 0.024 | 0.052 / 0.220 |
| honest ratio2 precision | 0.003 [-0.033, 0.013] | -0.009 [-0.038, 0.015] | 0.854 | 0.024 | 0.052 / 0.220 |
| honest ratio2 mean | 0.003 [-0.033, 0.013] | -0.009 [-0.038, 0.015] | 0.854 | 0.024 | 0.052 / 0.220 |
| honest ratio3 median | -0.035 [-0.049, 0.006] | -0.027 [-0.062, -0.005] | 0.932 | 0.025 | 0.058 / 0.230 |
| honest ratio3 precision | -0.035 [-0.049, 0.006] | -0.027 [-0.063, -0.005] | 0.932 | 0.025 | 0.058 / 0.230 |
| honest ratio3 mean | -0.035 [-0.049, 0.006] | -0.027 [-0.062, -0.005] | 0.932 | 0.025 | 0.058 / 0.230 |
| honest ratio5 median | -0.050 [-0.106, 0.005] | -0.060 [-0.079, -0.019] | 0.960 | 0.025 | 0.064 / 0.256 |
| honest ratio5 precision | -0.050 [-0.106, 0.005] | -0.060 [-0.079, -0.019] | 0.960 | 0.025 | 0.064 / 0.256 |
| honest ratio5 mean | -0.050 [-0.106, 0.005] | -0.060 [-0.079, -0.019] | 0.960 | 0.025 | 0.064 / 0.256 |
| honest cap_only median | -0.084 [-0.138, -0.008] | -0.088 [-0.119, -0.035] | 0.965 | 0.025 | 0.069 / 0.281 |
| honest cap_only precision | -0.084 [-0.138, -0.008] | -0.088 [-0.119, -0.035] | 0.965 | 0.025 | 0.069 / 0.281 |
| honest cap_only mean | -0.084 [-0.138, -0.008] | -0.088 [-0.119, -0.035] | 0.965 | 0.025 | 0.069 / 0.281 |
| honest none median | -0.196 [-0.385, -0.058] | -0.191 [-0.424, -0.071] | 1.000 | 0.028 | 0.102 / 0.388 |
| honest none precision | -0.196 [-0.385, -0.058] | -0.191 [-0.424, -0.071] | 1.000 | 0.028 | 0.102 / 0.388 |
| honest none mean | -0.196 [-0.385, -0.058] | -0.191 [-0.424, -0.071] | 1.000 | 0.028 | 0.102 / 0.388 |
| pooled (ratio2, own included) | 0.017 [-0.010, 0.035] | - | - | - | - |
| truth_strat | -0.001 [-0.010, 0.007] | - | - | - | - |

Reference `e_log_joint_mc_sd` median 0.019; the headline bias is within 2 SE in 75% of the cells; median absolute bias raw 0.045, capped_I 0.053, capped_E 0.045, honest 0.031 (against raw the honest estimate is a tie, against capped_I better; tie band 0.019, one reference standard error). Calibration of the self-reported SDs (weighted over components): own z median 0.10 [-0.05, 0.17], abs z median 0.66 [0.55, 0.76], abs z > 2 fraction 0.03 [0.01, 0.04]; cross z median -0.07 [-0.16, -0.05], abs z median 0.20 [0.17, 0.23], abs z > 2 fraction 0.00 [0.00, 0.00]. Effective training points on a component (weighted median): own run 2.3 [2.0, 2.4], covering other runs 2.3 [2.1, 2.6]. Checks: own-run max abs z 4.05 (weighted offset abs z 2.41), Jacobian max abs z 3.59 (offset abs z 2.17), raw consistency 1.1e-13, flagged cells 0.

Heaviest component of every run, repetition 0:

| Run, heaviest component (weight) | truth | own I_corr (error; BQ sd; points within 2 SD, their data offset; mean fn) | other runs: estimate (error; pred sd; covered; points within 2 SD, their data offset; mean fn) |
|---|---:|---|---|
| multisensory_s1_D6_svbmc_seed1010, component 19 (0.040) | -506.304 ± 0.180 | -506.361 (-0.057; 0.075; 10, -0.000; -551.9) | multisensory_s1_D6_svbmc_seed1028: -506.476 (-0.172; 0.256; yes; 10, -0.000; -576.0) |
| multisensory_s1_D6_svbmc_seed1028, component 1 (0.052) | -506.399 ± 0.193 | -506.294 (+0.105; 0.042; 20, 0.000; -581.9) | multisensory_s1_D6_svbmc_seed1010: -506.391 (+0.009; 0.176; yes; 13, -0.000; -555.9) |

### M = 4 (20 cells)

| Estimator | bias | bias with the arm's entropy | covered | mc_se | gp_sd (indep / corr) |
|---|---:|---:|---:|---:|---:|
| raw | 0.042 [0.031, 0.069] | 0.045 [0.020, 0.068] | - | - | - |
| capped_I | -0.009 [-0.027, 0.010] | -0.019 [-0.028, 0.022] | - | - | - |
| capped_E | 0.042 [0.030, 0.069] | 0.042 [0.020, 0.068] | - | - | - |
| honest ratio1.5 median | 0.002 [-0.026, 0.021] | -0.007 [-0.031, 0.021] | 0.914 | 0.017 | 0.033 / 0.148 |
| honest ratio1.5 precision | 0.003 [-0.019, 0.022] | -0.007 [-0.030, 0.025] | 0.914 | 0.017 | 0.029 / 0.149 |
| honest ratio1.5 mean | 0.003 [-0.025, 0.017] | -0.011 [-0.030, 0.020] | 0.914 | 0.017 | 0.030 / 0.149 |
| honest ratio2 median | -0.015 [-0.029, 0.005] | -0.012 [-0.039, 0.008] | 0.962 | 0.019 | 0.039 / 0.170 |
| honest ratio2 precision | -0.000 [-0.025, 0.013] | -0.009 [-0.036, 0.010] | 0.962 | 0.019 | 0.031 / 0.160 |
| honest ratio2 mean | -0.014 [-0.030, 0.005] | -0.014 [-0.037, 0.006] | 0.962 | 0.019 | 0.033 / 0.167 |
| honest ratio3 median | -0.028 [-0.053, -0.002] | -0.024 [-0.049, 0.001] | 0.975 | 0.019 | 0.042 / 0.183 |
| honest ratio3 precision | -0.013 [-0.031, 0.003] | -0.018 [-0.041, 0.004] | 0.975 | 0.019 | 0.032 / 0.168 |
| honest ratio3 mean | -0.033 [-0.055, -0.008] | -0.035 [-0.049, -0.002] | 0.975 | 0.019 | 0.036 / 0.183 |
| honest ratio5 median | -0.031 [-0.064, -0.004] | -0.032 [-0.055, -0.005] | 0.982 | 0.019 | 0.043 / 0.190 |
| honest ratio5 precision | -0.019 [-0.037, 0.002] | -0.024 [-0.048, 0.001] | 0.982 | 0.019 | 0.032 / 0.170 |
| honest ratio5 mean | -0.044 [-0.064, -0.017] | -0.045 [-0.059, -0.021] | 0.982 | 0.019 | 0.036 / 0.189 |
| honest cap_only median | -0.057 [-0.077, -0.020] | -0.046 [-0.067, -0.023] | 0.995 | 0.019 | 0.047 / 0.204 |
| honest cap_only precision | -0.027 [-0.050, -0.019] | -0.030 [-0.054, -0.016] | 0.995 | 0.019 | 0.033 / 0.179 |
| honest cap_only mean | -0.063 [-0.094, -0.039] | -0.061 [-0.085, -0.049] | 0.995 | 0.019 | 0.038 / 0.207 |
| honest none median | -0.108 [-0.278, -0.035] | -0.115 [-0.261, -0.047] | 1.000 | 0.020 | 0.076 / 0.242 |
| honest none precision | -0.065 [-0.164, -0.029] | -0.064 [-0.160, -0.038] | 1.000 | 0.019 | 0.039 / 0.195 |
| honest none mean | -0.183 [-0.572, -0.082] | -0.191 [-0.533, -0.084] | 1.000 | 0.020 | 0.054 / 0.268 |
| pooled (ratio2, own included) | 0.017 [-0.008, 0.032] | - | - | - | - |
| truth_strat | -0.004 [-0.009, 0.006] | - | - | - | - |

Reference `e_log_joint_mc_sd` median 0.019; the headline bias is within 2 SE in 75% of the cells; median absolute bias raw 0.042, capped_I 0.026, capped_E 0.042, honest 0.024 (against raw the honest estimate is a tie, against capped_I a tie; tie band 0.019, one reference standard error). Calibration of the self-reported SDs (weighted over components): own z median 0.20 [0.11, 0.24], abs z median 0.73 [0.69, 0.79], abs z > 2 fraction 0.06 [0.03, 0.07]; cross z median -0.02 [-0.04, 0.02], abs z median 0.18 [0.16, 0.19], abs z > 2 fraction 0.00 [0.00, 0.00]. Effective training points on a component (weighted median): own run 2.4 [2.3, 2.6], covering other runs 2.6 [2.4, 2.7]. Checks: own-run max abs z 4.74 (weighted offset abs z 1.93), Jacobian max abs z 3.55 (offset abs z 2.79), raw consistency 1.1e-13, flagged cells 0.

Heaviest component of every run, repetition 0:

| Run, heaviest component (weight) | truth | own I_corr (error; BQ sd; points within 2 SD, their data offset; mean fn) | other runs: estimate (error; pred sd; covered; points within 2 SD, their data offset; mean fn) |
|---|---:|---|---|
| multisensory_s1_D6_svbmc_seed1013, component 3 (0.041) | -505.832 ± 0.179 | -505.886 (-0.054; 0.033; 23, 0.000; -552.8) | multisensory_s1_D6_svbmc_seed1022: -505.713 (+0.118; 0.304; yes; 13, -0.000; -534.9); multisensory_s1_D6_svbmc_seed1036: -505.815 (+0.017; 0.179; yes; 13, 0.000; -543.9); multisensory_s1_D6_svbmc_seed1047: -506.308 (-0.476; 0.614; no; 18, -0.000; -629.1) |
| multisensory_s1_D6_svbmc_seed1022, component 1 (0.033) | -506.316 ± 0.203 | -506.223 (+0.093; 0.033; 17, -0.000; -538.0) | multisensory_s1_D6_svbmc_seed1013: -506.295 (+0.021; 0.255; yes; 16, 0.000; -556.6); multisensory_s1_D6_svbmc_seed1036: -506.241 (+0.075; 0.179; yes; 15, -0.000; -545.1); multisensory_s1_D6_svbmc_seed1047: -506.382 (-0.066; 0.278; yes; 19, -0.000; -629.3) |
| multisensory_s1_D6_svbmc_seed1036, component 2 (0.037) | -505.840 ± 0.177 | -505.791 (+0.049; 0.019; 20, -0.000; -545.1) | multisensory_s1_D6_svbmc_seed1013: -505.826 (+0.015; 0.286; yes; 22, 0.000; -555.5); multisensory_s1_D6_svbmc_seed1022: -505.802 (+0.038; 0.307; yes; 22, -0.000; -536.8); multisensory_s1_D6_svbmc_seed1047: -506.240 (-0.399; 0.603; no; 25, -0.000; -630.3) |
| multisensory_s1_D6_svbmc_seed1047, component 1 (0.017) | -505.630 ± 0.165 | -505.737 (-0.107; 0.021; 6, 0.000; -631.1) | multisensory_s1_D6_svbmc_seed1013: -505.592 (+0.038; 0.098; yes; 18, 0.000; -557.6); multisensory_s1_D6_svbmc_seed1022: -505.604 (+0.026; 0.198; yes; 12, 0.000; -542.1); multisensory_s1_D6_svbmc_seed1036: -505.564 (+0.066; 0.085; yes; 10, 0.000; -544.7) |

### M = 8 (20 cells)

| Estimator | bias | bias with the arm's entropy | covered | mc_se | gp_sd (indep / corr) |
|---|---:|---:|---:|---:|---:|
| raw | 0.072 [0.056, 0.079] | 0.072 [0.054, 0.082] | - | - | - |
| capped_I | 0.032 [0.006, 0.053] | 0.029 [-0.004, 0.049] | - | - | - |
| capped_E | 0.072 [0.056, 0.079] | 0.072 [0.054, 0.082] | - | - | - |
| honest ratio1.5 median | -0.009 [-0.019, 0.008] | -0.003 [-0.024, 0.010] | 0.970 | 0.013 | 0.027 / 0.123 |
| honest ratio1.5 precision | -0.010 [-0.025, 0.001] | -0.003 [-0.026, 0.003] | 0.970 | 0.013 | 0.019 / 0.118 |
| honest ratio1.5 mean | -0.015 [-0.029, -0.001] | -0.010 [-0.032, 0.000] | 0.970 | 0.013 | 0.020 / 0.120 |
| honest ratio2 median | -0.012 [-0.024, -0.002] | -0.009 [-0.027, -0.001] | 0.988 | 0.013 | 0.033 / 0.132 |
| honest ratio2 precision | -0.018 [-0.028, -0.005] | -0.011 [-0.032, -0.004] | 0.988 | 0.013 | 0.018 / 0.122 |
| honest ratio2 mean | -0.025 [-0.041, -0.015] | -0.027 [-0.037, -0.012] | 0.988 | 0.013 | 0.020 / 0.130 |
| honest ratio3 median | -0.022 [-0.038, -0.013] | -0.024 [-0.038, -0.011] | 0.994 | 0.013 | 0.035 / 0.143 |
| honest ratio3 precision | -0.026 [-0.038, -0.012] | -0.019 [-0.040, -0.011] | 0.994 | 0.013 | 0.021 / 0.125 |
| honest ratio3 mean | -0.044 [-0.058, -0.031] | -0.042 [-0.060, -0.031] | 0.994 | 0.013 | 0.023 / 0.141 |
| honest ratio5 median | -0.037 [-0.049, -0.023] | -0.036 [-0.049, -0.017] | 0.998 | 0.013 | 0.036 / 0.148 |
| honest ratio5 precision | -0.033 [-0.043, -0.019] | -0.028 [-0.045, -0.017] | 0.998 | 0.013 | 0.021 / 0.127 |
| honest ratio5 mean | -0.067 [-0.077, -0.053] | -0.061 [-0.082, -0.053] | 0.998 | 0.013 | 0.023 / 0.152 |
| honest cap_only median | -0.071 [-0.088, -0.038] | -0.068 [-0.088, -0.030] | 0.998 | 0.013 | 0.038 / 0.167 |
| honest cap_only precision | -0.041 [-0.045, -0.022] | -0.031 [-0.049, -0.018] | 0.998 | 0.013 | 0.021 / 0.129 |
| honest cap_only mean | -0.107 [-0.126, -0.097] | -0.107 [-0.132, -0.093] | 0.998 | 0.013 | 0.026 / 0.174 |
| honest none median | -0.421 [-0.810, -0.161] | -0.414 [-0.809, -0.158] | 1.000 | 0.013 | 0.116 / 0.335 |
| honest none precision | -0.080 [-0.112, -0.049] | -0.076 [-0.106, -0.051] | 1.000 | 0.013 | 0.021 / 0.148 |
| honest none mean | -0.662 [-1.009, -0.482] | -0.652 [-1.006, -0.468] | 1.000 | 0.014 | 0.066 / 0.387 |
| pooled (ratio2, own included) | 0.001 [-0.008, 0.018] | - | - | - | - |
| truth_strat | 0.005 [-0.013, 0.017] | - | - | - | - |

Reference `e_log_joint_mc_sd` median 0.019; the headline bias is within 2 SE in 95% of the cells; median absolute bias raw 0.072, capped_I 0.043, capped_E 0.072, honest 0.021 (against raw the honest estimate is better, against capped_I better; tie band 0.019, one reference standard error). Calibration of the self-reported SDs (weighted over components): own z median 0.14 [0.09, 0.25], abs z median 0.69 [0.67, 0.73], abs z > 2 fraction 0.05 [0.04, 0.05]; cross z median -0.04 [-0.06, -0.03], abs z median 0.19 [0.18, 0.20], abs z > 2 fraction 0.00 [0.00, 0.01]. Effective training points on a component (weighted median): own run 2.0 [1.8, 2.1], covering other runs 2.1 [2.0, 2.2]. Checks: own-run max abs z 4.92 (weighted offset abs z 2.17), Jacobian max abs z 4.82 (offset abs z 1.78), raw consistency 5.7e-14, flagged cells 0.

### M = 16 (10 cells)

| Estimator | bias | bias with the arm's entropy | covered | mc_se | gp_sd (indep / corr) |
|---|---:|---:|---:|---:|---:|
| raw | 0.053 [0.037, 0.079] | 0.055 [0.034, 0.085] | - | - | - |
| capped_I | 0.015 [-0.009, 0.047] | 0.017 [-0.018, 0.052] | - | - | - |
| capped_E | 0.053 [0.037, 0.079] | 0.055 [0.034, 0.085] | - | - | - |
| honest ratio1.5 median | -0.025 [-0.033, -0.020] | -0.026 [-0.035, -0.019] | 0.988 | 0.009 | 0.024 / 0.093 |
| honest ratio1.5 precision | -0.027 [-0.044, -0.023] | -0.032 [-0.042, -0.021] | 0.988 | 0.009 | 0.010 / 0.087 |
| honest ratio1.5 mean | -0.035 [-0.049, -0.030] | -0.037 [-0.046, -0.027] | 0.988 | 0.009 | 0.011 / 0.090 |
| honest ratio2 median | -0.034 [-0.039, -0.030] | -0.032 [-0.046, -0.026] | 0.994 | 0.009 | 0.027 / 0.097 |
| honest ratio2 precision | -0.035 [-0.049, -0.029] | -0.038 [-0.048, -0.026] | 0.994 | 0.009 | 0.010 / 0.087 |
| honest ratio2 mean | -0.050 [-0.059, -0.044] | -0.049 [-0.060, -0.041] | 0.994 | 0.009 | 0.011 / 0.095 |
| honest ratio3 median | -0.042 [-0.046, -0.039] | -0.042 [-0.053, -0.035] | 0.999 | 0.009 | 0.027 / 0.105 |
| honest ratio3 precision | -0.039 [-0.057, -0.034] | -0.045 [-0.052, -0.033] | 0.999 | 0.009 | 0.010 / 0.089 |
| honest ratio3 mean | -0.065 [-0.076, -0.061] | -0.067 [-0.076, -0.060] | 0.999 | 0.009 | 0.011 / 0.103 |
| honest ratio5 median | -0.049 [-0.054, -0.045] | -0.048 [-0.061, -0.042] | 0.999 | 0.009 | 0.028 / 0.107 |
| honest ratio5 precision | -0.042 [-0.059, -0.036] | -0.048 [-0.055, -0.036] | 0.999 | 0.009 | 0.010 / 0.089 |
| honest ratio5 mean | -0.086 [-0.099, -0.079] | -0.085 [-0.103, -0.077] | 0.999 | 0.009 | 0.011 / 0.109 |
| honest cap_only median | -0.063 [-0.083, -0.054] | -0.067 [-0.080, -0.055] | 0.999 | 0.009 | 0.030 / 0.112 |
| honest cap_only precision | -0.044 [-0.062, -0.037] | -0.049 [-0.057, -0.038] | 0.999 | 0.009 | 0.010 / 0.090 |
| honest cap_only mean | -0.138 [-0.149, -0.108] | -0.133 [-0.150, -0.110] | 0.999 | 0.009 | 0.012 / 0.120 |
| honest none median | -0.575 [-0.629, -0.315] | -0.575 [-0.629, -0.318] | 1.000 | 0.010 | 0.088 / 0.289 |
| honest none precision | -0.080 [-0.110, -0.050] | -0.082 [-0.103, -0.055] | 1.000 | 0.009 | 0.009 / 0.094 |
| honest none mean | -0.699 [-1.026, -0.511] | -0.693 [-1.037, -0.510] | 1.000 | 0.010 | 0.030 / 0.285 |
| pooled (ratio2, own included) | -0.026 [-0.029, -0.020] | - | - | - | - |
| truth_strat | -0.022 [-0.035, -0.009] | - | - | - | - |

Reference `e_log_joint_mc_sd` median 0.019; the headline bias is within 2 SE in 90% of the cells; median absolute bias raw 0.053, capped_I 0.024, capped_E 0.053, honest 0.036 (against raw the honest estimate is a tie, against capped_I a tie; tie band 0.019, one reference standard error). Calibration of the self-reported SDs (weighted over components): own z median 0.23 [0.11, 0.28], abs z median 0.73 [0.71, 0.77], abs z > 2 fraction 0.06 [0.05, 0.07]; cross z median -0.05 [-0.06, -0.03], abs z median 0.19 [0.18, 0.20], abs z > 2 fraction 0.00 [0.00, 0.01]. Effective training points on a component (weighted median): own run 2.0 [1.9, 2.1], covering other runs 2.1 [2.0, 2.2]. Checks: own-run max abs z 4.22 (weighted offset abs z 2.49), Jacobian max abs z 4.16 (offset abs z 1.54), raw consistency 1.7e-13, flagged cells 0.

| Run | own error | se | own z median | own abs z median | BQ sd median | pred sd median | own-run max abs z (offset z) | Jacobian max abs z (offset z) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| multisensory_s1_D6_svbmc_seed1000 | 0.065 | 0.029 | 0.31 | 0.72 | 0.060 | 0.196 | 2.46 (-1.05) | 2.60 (+0.53) |
| multisensory_s1_D6_svbmc_seed1001 | -0.009 | 0.030 | 0.05 | 0.46 | 0.050 | 0.177 | 3.13 (+0.07) | 2.62 (+0.50) |
| multisensory_s1_D6_svbmc_seed1002 | -0.084 | 0.051 | -0.27 | 0.72 | 0.057 | 0.186 | 2.35 (+1.09) | 1.70 (+0.41) |
| multisensory_s1_D6_svbmc_seed1003 | 0.052 | 0.035 | 0.05 | 0.50 | 0.061 | 0.220 | 2.19 (+0.69) | 3.00 (-0.93) |
| multisensory_s1_D6_svbmc_seed1004 | 0.044 | 0.082 | -0.11 | 0.25 | 0.106 | 0.486 | 2.20 (+0.68) | 3.09 (-1.11) |
| multisensory_s1_D6_svbmc_seed1005 | -0.025 | 0.035 | -0.40 | 1.02 | 0.102 | 0.323 | 4.44 (+0.54) | 2.97 (-0.86) |
| multisensory_s1_D6_svbmc_seed1006 | 0.183 | 0.052 | 1.03 | 1.15 | 0.084 | 0.266 | 2.71 (-1.19) | 2.44 (-0.27) |
| multisensory_s1_D6_svbmc_seed1007 | 0.039 | 0.035 | 0.17 | 0.86 | 0.085 | 0.279 | 2.55 (-0.20) | 2.38 (+0.18) |
| multisensory_s1_D6_svbmc_seed1008 | -0.038 | 0.035 | -0.29 | 0.69 | 0.050 | 0.189 | 2.47 (+0.65) | 1.66 (-1.54) |
| multisensory_s1_D6_svbmc_seed1009 | 0.126 | 0.045 | 0.14 | 0.76 | 0.086 | 0.245 | 2.39 (+0.13) | 2.70 (-0.14) |
| multisensory_s1_D6_svbmc_seed1010 | -0.038 | 0.035 | -0.29 | 0.45 | 0.072 | 0.239 | 3.24 (+0.45) | 2.89 (-0.71) |
| multisensory_s1_D6_svbmc_seed1011 | 0.010 | 0.035 | -0.40 | 0.79 | 0.066 | 0.277 | 2.81 (+1.56) | 3.06 (+0.60) |
| multisensory_s1_D6_svbmc_seed1012 | 0.012 | 0.045 | -0.21 | 0.77 | 0.049 | 0.178 | 2.34 (-0.07) | 2.35 (-0.14) |
| multisensory_s1_D6_svbmc_seed1013 | 0.003 | 0.048 | 0.17 | 1.05 | 0.058 | 0.190 | 3.41 (+0.97) | 2.68 (-1.02) |
| multisensory_s1_D6_svbmc_seed1014 | 0.061 | 0.035 | 0.20 | 0.39 | 0.143 | 0.471 | 2.22 (+0.13) | 1.85 (-0.54) |
| multisensory_s1_D6_svbmc_seed1015 | 0.082 | 0.030 | 0.25 | 0.70 | 0.048 | 0.167 | 2.26 (-1.20) | 2.20 (+1.14) |
| multisensory_s1_D6_svbmc_seed1016 | 0.059 | 0.037 | 0.23 | 0.73 | 0.062 | 0.241 | 2.67 (-0.71) | 2.97 (+2.53) |
| multisensory_s1_D6_svbmc_seed1017 | 0.080 | 0.030 | 0.31 | 0.46 | 0.087 | 0.324 | 2.88 (-1.46) | 2.50 (+1.76) |
| multisensory_s1_D6_svbmc_seed1018 | -0.066 | 0.032 | -0.29 | 0.67 | 0.063 | 0.229 | 2.24 (+1.80) | 2.13 (-1.16) |
| multisensory_s1_D6_svbmc_seed1019 | 0.050 | 0.046 | 0.22 | 0.81 | 0.064 | 0.283 | 2.96 (-1.50) | 2.12 (-0.74) |
| multisensory_s1_D6_svbmc_seed1020 | -0.014 | 0.034 | 0.17 | 0.64 | 0.066 | 0.182 | 3.27 (+0.16) | 2.39 (-1.16) |
| multisensory_s1_D6_svbmc_seed1021 | -0.036 | 0.030 | -0.25 | 0.70 | 0.060 | 0.235 | 3.08 (+1.33) | 2.55 (-1.66) |
| multisensory_s1_D6_svbmc_seed1022 | 0.085 | 0.039 | 0.32 | 0.48 | 0.074 | 0.271 | 2.03 (+0.11) | 2.31 (+0.16) |
| multisensory_s1_D6_svbmc_seed1023 | 0.064 | 0.035 | 0.26 | 0.63 | 0.074 | 0.226 | 3.15 (+0.21) | 2.59 (+0.27) |
| multisensory_s1_D6_svbmc_seed1024 | 0.066 | 0.034 | 0.47 | 0.56 | 0.065 | 0.237 | 3.04 (-0.86) | 1.71 (+0.15) |
| multisensory_s1_D6_svbmc_seed1025 | 0.133 | 0.035 | 0.35 | 0.74 | 0.053 | 0.184 | 2.63 (-0.37) | 2.18 (+0.06) |
| multisensory_s1_D6_svbmc_seed1026 | 0.115 | 0.045 | 0.84 | 0.94 | 0.028 | 0.118 | 2.40 (-1.50) | 2.77 (-0.68) |
| multisensory_s1_D6_svbmc_seed1027 | 0.093 | 0.037 | 0.41 | 0.60 | 0.134 | 0.437 | 2.04 (-0.39) | 2.39 (+1.55) |
| multisensory_s1_D6_svbmc_seed1028 | 0.084 | 0.033 | 0.20 | 0.44 | 0.107 | 0.364 | 2.52 (-0.14) | 2.27 (+0.22) |
| multisensory_s1_D6_svbmc_seed1029 | -0.048 | 0.032 | -0.38 | 0.81 | 0.078 | 0.258 | 2.44 (+1.47) | 2.74 (+0.89) |
| multisensory_s1_D6_svbmc_seed1030 | 0.052 | 0.032 | 0.00 | 0.49 | 0.041 | 0.128 | 2.66 (+0.05) | 2.61 (-1.09) |
| multisensory_s1_D6_svbmc_seed1031 | 0.141 | 0.038 | 0.76 | 0.82 | 0.059 | 0.206 | 2.75 (-1.78) | 2.54 (+0.45) |
| multisensory_s1_D6_svbmc_seed1032 | 0.128 | 0.033 | 0.34 | 0.63 | 0.086 | 0.337 | 2.78 (-1.93) | 2.44 (+0.08) |
| multisensory_s1_D6_svbmc_seed1033 | 0.003 | 0.037 | -0.29 | 0.83 | 0.055 | 0.167 | 2.07 (-0.48) | 2.34 (-0.62) |
| multisensory_s1_D6_svbmc_seed1034 | 0.003 | 0.032 | -0.02 | 0.60 | 0.054 | 0.186 | 2.34 (+0.15) | 2.67 (-0.38) |
| multisensory_s1_D6_svbmc_seed1035 | 0.035 | 0.039 | -0.00 | 0.76 | 0.037 | 0.141 | 2.63 (+0.31) | 2.07 (+0.87) |
| multisensory_s1_D6_svbmc_seed1036 | 0.019 | 0.037 | 0.04 | 0.69 | 0.055 | 0.221 | 2.37 (+0.59) | 2.07 (-0.03) |
| multisensory_s1_D6_svbmc_seed1037 | 0.132 | 0.038 | 0.63 | 0.76 | 0.087 | 0.259 | 2.28 (-1.53) | 3.09 (-0.44) |
| multisensory_s1_D6_svbmc_seed1038 | 0.074 | 0.040 | 0.30 | 0.39 | 0.083 | 0.313 | 3.41 (-1.02) | 2.49 (+0.61) |
| multisensory_s1_D6_svbmc_seed1039 | 0.081 | 0.038 | 0.49 | 0.73 | 0.057 | 0.195 | 2.60 (-1.19) | 2.36 (-0.13) |
| multisensory_s1_D6_svbmc_seed1040 | 0.012 | 0.035 | -0.09 | 0.63 | 0.083 | 0.246 | 2.41 (-0.06) | 3.02 (+1.35) |
| multisensory_s1_D6_svbmc_seed1041 | 0.019 | 0.034 | 0.14 | 0.77 | 0.070 | 0.269 | 3.10 (-1.47) | 3.06 (+1.50) |
| multisensory_s1_D6_svbmc_seed1042 | 0.003 | 0.034 | -0.29 | 0.72 | 0.042 | 0.163 | 2.14 (+0.64) | 2.47 (-0.01) |
| multisensory_s1_D6_svbmc_seed1043 | 0.131 | 0.039 | 0.47 | 0.94 | 0.062 | 0.191 | 2.50 (-1.60) | 2.36 (+1.25) |
| multisensory_s1_D6_svbmc_seed1044 | 0.035 | 0.033 | 0.02 | 0.68 | 0.061 | 0.195 | 2.82 (-0.52) | 2.11 (+0.63) |
| multisensory_s1_D6_svbmc_seed1045 | 0.035 | 0.048 | 0.15 | 0.42 | 0.128 | 0.387 | 2.50 (-0.73) | 2.03 (+0.28) |
| multisensory_s1_D6_svbmc_seed1046 | 0.004 | 0.032 | -0.05 | 0.82 | 0.042 | 0.161 | 2.58 (+1.02) | 2.42 (+0.80) |
| multisensory_s1_D6_svbmc_seed1047 | 0.005 | 0.032 | -0.02 | 0.55 | 0.048 | 0.186 | 2.49 (-0.19) | 2.13 (-0.43) |
| multisensory_s1_D6_svbmc_seed1048 | 0.006 | 0.030 | 0.23 | 0.55 | 0.043 | 0.160 | 2.91 (+0.51) | 2.14 (-1.10) |
| multisensory_s1_D6_svbmc_seed1049 | 0.028 | 0.034 | 0.06 | 0.57 | 0.046 | 0.172 | 3.27 (-0.72) | 2.75 (-0.04) |

## ring_D2_noise3_svbmc (noisy)

### M = 2 (20 cells)

| Estimator | bias | bias with the arm's entropy | covered | mc_se | gp_sd (indep / corr) |
|---|---:|---:|---:|---:|---:|
| raw | 0.403 [0.283, 0.485] | 0.414 [0.285, 0.499] | - | - | - |
| capped_I | 0.204 [0.041, 0.346] | 0.209 [0.030, 0.347] | - | - | - |
| capped_E | 0.325 [0.188, 0.457] | 0.321 [0.191, 0.456] | - | - | - |
| honest ratio1.5 median | 0.283 [0.171, 0.435] | 0.286 [0.163, 0.430] | 0.085 | 0.003 | 0.076 / 0.449 |
| honest ratio1.5 precision | 0.283 [0.171, 0.435] | 0.286 [0.163, 0.430] | 0.085 | 0.003 | 0.076 / 0.449 |
| honest ratio1.5 mean | 0.283 [0.171, 0.435] | 0.286 [0.163, 0.430] | 0.085 | 0.003 | 0.076 / 0.449 |
| honest ratio2 median | 0.223 [0.052, 0.328] | 0.231 [0.050, 0.331] | 0.157 | 0.005 | 0.083 / 0.466 |
| honest ratio2 precision | 0.223 [0.052, 0.328] | 0.231 [0.050, 0.331] | 0.157 | 0.005 | 0.083 / 0.466 |
| honest ratio2 mean | 0.223 [0.052, 0.328] | 0.231 [0.050, 0.331] | 0.157 | 0.005 | 0.083 / 0.466 |
| honest ratio3 median | 0.210 [-0.074, 0.317] | 0.210 [-0.064, 0.321] | 0.234 | 0.006 | 0.088 / 0.488 |
| honest ratio3 precision | 0.210 [-0.074, 0.317] | 0.210 [-0.064, 0.321] | 0.234 | 0.006 | 0.088 / 0.488 |
| honest ratio3 mean | 0.210 [-0.074, 0.317] | 0.210 [-0.074, 0.321] | 0.234 | 0.006 | 0.088 / 0.488 |
| honest ratio5 median | 0.183 [-0.111, 0.317] | 0.194 [-0.104, 0.321] | 0.291 | 0.006 | 0.092 / 0.512 |
| honest ratio5 precision | 0.183 [-0.111, 0.317] | 0.194 [-0.104, 0.321] | 0.291 | 0.006 | 0.092 / 0.512 |
| honest ratio5 mean | 0.183 [-0.111, 0.317] | 0.194 [-0.104, 0.321] | 0.291 | 0.006 | 0.092 / 0.512 |
| honest cap_only median | 0.183 [-0.111, 0.317] | 0.194 [-0.104, 0.321] | 0.291 | 0.006 | 0.092 / 0.512 |
| honest cap_only precision | 0.183 [-0.111, 0.317] | 0.194 [-0.104, 0.321] | 0.291 | 0.006 | 0.092 / 0.512 |
| honest cap_only mean | 0.183 [-0.111, 0.317] | 0.194 [-0.104, 0.321] | 0.291 | 0.006 | 0.092 / 0.512 |
| honest none median | -181.603 [-485.082, -22.657] | -181.581 [-485.081, -22.670] | 1.000 | 0.223 | 44.554 / 180.701 |
| honest none precision | -181.603 [-485.082, -22.657] | -181.581 [-485.081, -22.670] | 1.000 | 0.223 | 44.554 / 180.701 |
| honest none mean | -181.603 [-485.082, -22.657] | -181.581 [-485.081, -22.670] | 1.000 | 0.223 | 44.554 / 180.701 |
| pooled (ratio2, own included) | 0.297 [0.207, 0.400] | - | - | - | - |
| truth_strat | 0.001 [-0.009, 0.007] | - | - | - | - |

Reference `e_log_joint_mc_sd` median 0.009; the headline bias is within 2 SE in 0% of the cells; median absolute bias raw 0.403, capped_I 0.217, capped_E 0.325, honest 0.277 (against raw the honest estimate is better, against capped_I **worse**; tie band 0.009, one reference standard error). Calibration of the self-reported SDs (weighted over components): own z median 0.72 [0.44, 0.95], abs z median 0.82 [0.68, 1.05], abs z > 2 fraction 0.07 [0.00, 0.13]; cross z median -0.57 [-0.85, 0.06], abs z median 0.78 [0.45, 0.93], abs z > 2 fraction 0.00 [0.00, 0.01]. Effective training points on a component (weighted median): own run 1.3 [1.2, 2.0], covering other runs 0.8 [0.5, 1.6]. Checks: own-run max abs z 6.11 (weighted offset abs z 1.84), Jacobian max abs z 0.00 (offset abs z 0.00), raw consistency 2.2e-16, flagged cells 1.

Heaviest component of every run, repetition 0:

| Run, heaviest component (weight) | truth | own I_corr (error; BQ sd; points within 2 SD, their data offset; mean fn) | other runs: estimate (error; pred sd; covered; points within 2 SD, their data offset; mean fn) |
|---|---:|---|---|
| ring_D2_noise3_svbmc_seed1025, component 25 (0.032) | -0.703 ± 0.105 | 0.496 (+1.199; 0.807; 0, -; -1074.9) | ring_D2_noise3_svbmc_seed1052: -3.540 (-2.838; 2.016; no; 0, -; -1019.0) |
| ring_D2_noise3_svbmc_seed1052, component 2 (0.018) | -0.375 ± 0.049 | 0.021 (+0.396; 0.374; 13, 0.439; -1040.2) | ring_D2_noise3_svbmc_seed1025: -1.369 (-0.995; 0.737; yes; 1, 1.297; -899.1) |

### M = 4 (20 cells)

| Estimator | bias | bias with the arm's entropy | covered | mc_se | gp_sd (indep / corr) |
|---|---:|---:|---:|---:|---:|
| raw | 0.401 [0.267, 0.488] | 0.402 [0.276, 0.479] | - | - | - |
| capped_I | 0.160 [0.051, 0.246] | 0.155 [0.049, 0.243] | - | - | - |
| capped_E | 0.258 [0.139, 0.319] | 0.264 [0.129, 0.315] | - | - | - |
| honest ratio1.5 median | 0.236 [0.128, 0.361] | 0.236 [0.130, 0.354] | 0.293 | 0.004 | 0.064 / 0.331 |
| honest ratio1.5 precision | 0.236 [0.115, 0.361] | 0.236 [0.122, 0.355] | 0.293 | 0.004 | 0.064 / 0.331 |
| honest ratio1.5 mean | 0.236 [0.128, 0.361] | 0.236 [0.130, 0.354] | 0.293 | 0.004 | 0.064 / 0.331 |
| honest ratio2 median | 0.173 [0.033, 0.218] | 0.165 [0.031, 0.213] | 0.465 | 0.006 | 0.068 / 0.357 |
| honest ratio2 precision | 0.173 [0.024, 0.240] | 0.165 [0.023, 0.226] | 0.465 | 0.006 | 0.068 / 0.355 |
| honest ratio2 mean | 0.173 [0.033, 0.218] | 0.165 [0.031, 0.213] | 0.465 | 0.006 | 0.068 / 0.357 |
| honest ratio3 median | 0.063 [-0.125, 0.159] | 0.055 [-0.126, 0.153] | 0.522 | 0.006 | 0.080 / 0.410 |
| honest ratio3 precision | 0.063 [-0.132, 0.174] | 0.055 [-0.133, 0.172] | 0.522 | 0.006 | 0.078 / 0.407 |
| honest ratio3 mean | 0.063 [-0.125, 0.159] | 0.055 [-0.126, 0.153] | 0.522 | 0.006 | 0.080 / 0.410 |
| honest ratio5 median | -0.003 [-0.229, 0.118] | -0.008 [-0.233, 0.114] | 0.562 | 0.007 | 0.090 / 0.444 |
| honest ratio5 precision | 0.008 [-0.230, 0.148] | 0.003 [-0.234, 0.141] | 0.562 | 0.007 | 0.088 / 0.431 |
| honest ratio5 mean | -0.003 [-0.229, 0.118] | -0.008 [-0.233, 0.114] | 0.562 | 0.007 | 0.090 / 0.444 |
| honest cap_only median | -0.003 [-0.236, 0.150] | -0.008 [-0.240, 0.144] | 0.565 | 0.007 | 0.091 / 0.443 |
| honest cap_only precision | 0.008 [-0.233, 0.173] | 0.003 [-0.238, 0.162] | 0.565 | 0.007 | 0.091 / 0.431 |
| honest cap_only mean | -0.003 [-0.236, 0.150] | -0.008 [-0.240, 0.144] | 0.565 | 0.007 | 0.091 / 0.443 |
| honest none median | -71.778 [-271.349, -12.299] | -71.782 [-271.341, -12.304] | 1.000 | 0.211 | 44.613 / 133.296 |
| honest none precision | -1.270 [-5.672, -0.448] | -1.272 [-5.676, -0.450] | 1.000 | 0.026 | 1.727 / 6.864 |
| honest none mean | -347.400 [-529.122, -144.333] | -347.392 [-529.124, -144.337] | 1.000 | 0.190 | 36.738 / 216.193 |
| pooled (ratio2, own included) | 0.238 [0.169, 0.352] | - | - | - | - |
| truth_strat | -0.007 [-0.012, 0.003] | - | - | - | - |

Reference `e_log_joint_mc_sd` median 0.009; the headline bias is within 2 SE in 10% of the cells; median absolute bias raw 0.401, capped_I 0.160, capped_E 0.258, honest 0.193 (against raw the honest estimate is better, against capped_I **worse**; tie band 0.009, one reference standard error). Calibration of the self-reported SDs (weighted over components): own z median 0.77 [0.57, 0.94], abs z median 0.83 [0.71, 1.04], abs z > 2 fraction 0.08 [0.01, 0.12]; cross z median -0.17 [-0.33, 0.01], abs z median 0.52 [0.42, 0.63], abs z > 2 fraction 0.00 [0.00, 0.02]. Effective training points on a component (weighted median): own run 1.5 [1.4, 2.0], covering other runs 1.2 [0.8, 1.3]. Checks: own-run max abs z 7.02 (weighted offset abs z 2.16), Jacobian max abs z 0.00 (offset abs z 0.00), raw consistency 2.8e-16, flagged cells 1.

Heaviest component of every run, repetition 0:

| Run, heaviest component (weight) | truth | own I_corr (error; BQ sd; points within 2 SD, their data offset; mean fn) | other runs: estimate (error; pred sd; covered; points within 2 SD, their data offset; mean fn) |
|---|---:|---|---|
| ring_D2_noise3_svbmc_seed1020, component 23 (0.028) | -0.848 ± 0.126 | 0.676 (+1.524; 0.603; 5, 2.007; -1436.8) | ring_D2_noise3_svbmc_seed1066: -0.559 (+0.289; 0.440; yes; 1, 1.347; -5903.9); ring_D2_noise3_svbmc_seed1105: 154.144 (+154.993; 96.663; no; 0, -; -1282.9); ring_D2_noise3_svbmc_seed1129: 0.189 (+1.038; 2.627; no; 1, -0.032; -5639.4) |
| ring_D2_noise3_svbmc_seed1066, component 8 (0.023) | -0.769 ± 0.118 | -0.245 (+0.525; 0.429; 12, 0.218; -5868.0) | ring_D2_noise3_svbmc_seed1020: -1.021 (-0.252; 0.945; no; 3, -0.621; -1413.7); ring_D2_noise3_svbmc_seed1105: 46.695 (+47.465; 38.716; no; 0, -; -1396.6); ring_D2_noise3_svbmc_seed1129: -2.052 (-1.283; 1.602; no; 1, -0.146; -5666.9) |
| ring_D2_noise3_svbmc_seed1105, component 32 (0.019) | -0.550 ± 0.066 | 0.150 (+0.701; 0.390; 12, 1.421; -1681.3) | ring_D2_noise3_svbmc_seed1020: -1058.769 (-1058.219; 454.350; no; 0, -; -1457.8); ring_D2_noise3_svbmc_seed1066: -20.290 (-19.740; 21.257; no; 0, -; -5569.4); ring_D2_noise3_svbmc_seed1129: 0.025 (+0.575; 0.655; yes; 1, 1.298; -5630.1) |
| ring_D2_noise3_svbmc_seed1129, component 13 (0.011) | -0.384 ± 0.063 | -0.319 (+0.065; 0.441; 10, 1.296; -5679.9) | ring_D2_noise3_svbmc_seed1020: -235.832 (-235.448; 88.454; no; 0, -; -1429.0); ring_D2_noise3_svbmc_seed1066: -1.097 (-0.714; 1.086; no; 1, 1.675; -5668.9); ring_D2_noise3_svbmc_seed1105: -0.782 (-0.398; 0.627; yes; 5, -0.174; -1619.9) |

### M = 8 (20 cells)

| Estimator | bias | bias with the arm's entropy | covered | mc_se | gp_sd (indep / corr) |
|---|---:|---:|---:|---:|---:|
| raw | 0.489 [0.379, 0.541] | 0.491 [0.378, 0.550] | - | - | - |
| capped_I | 0.155 [0.086, 0.183] | 0.157 [0.097, 0.175] | - | - | - |
| capped_E | 0.209 [0.157, 0.239] | 0.205 [0.159, 0.239] | - | - | - |
| honest ratio1.5 median | 0.020 [-0.034, 0.105] | 0.018 [-0.028, 0.101] | 0.715 | 0.007 | 0.047 / 0.225 |
| honest ratio1.5 precision | 0.036 [-0.028, 0.111] | 0.031 [-0.021, 0.108] | 0.715 | 0.007 | 0.045 / 0.223 |
| honest ratio1.5 mean | 0.016 [-0.049, 0.108] | 0.010 [-0.043, 0.103] | 0.715 | 0.007 | 0.045 / 0.227 |
| honest ratio2 median | -0.116 [-0.200, -0.056] | -0.108 [-0.198, -0.051] | 0.813 | 0.007 | 0.050 / 0.257 |
| honest ratio2 precision | -0.071 [-0.171, -0.024] | -0.060 [-0.160, -0.034] | 0.813 | 0.007 | 0.047 / 0.244 |
| honest ratio2 mean | -0.126 [-0.215, -0.057] | -0.118 [-0.204, -0.059] | 0.813 | 0.007 | 0.048 / 0.255 |
| honest ratio3 median | -0.261 [-0.367, -0.203] | -0.261 [-0.372, -0.212] | 0.849 | 0.007 | 0.059 / 0.302 |
| honest ratio3 precision | -0.206 [-0.286, -0.155] | -0.206 [-0.290, -0.154] | 0.849 | 0.007 | 0.051 / 0.276 |
| honest ratio3 mean | -0.293 [-0.357, -0.216] | -0.283 [-0.362, -0.229] | 0.849 | 0.007 | 0.055 / 0.302 |
| honest ratio5 median | -0.340 [-0.408, -0.287] | -0.341 [-0.408, -0.296] | 0.883 | 0.008 | 0.070 / 0.340 |
| honest ratio5 precision | -0.254 [-0.349, -0.192] | -0.255 [-0.334, -0.197] | 0.883 | 0.008 | 0.056 / 0.304 |
| honest ratio5 mean | -0.398 [-0.456, -0.307] | -0.396 [-0.454, -0.318] | 0.883 | 0.008 | 0.065 / 0.347 |
| honest cap_only median | -0.338 [-0.408, -0.289] | -0.338 [-0.408, -0.297] | 0.883 | 0.008 | 0.070 / 0.342 |
| honest cap_only precision | -0.252 [-0.349, -0.192] | -0.254 [-0.334, -0.197] | 0.883 | 0.008 | 0.056 / 0.304 |
| honest cap_only mean | -0.395 [-0.453, -0.307] | -0.393 [-0.448, -0.318] | 0.883 | 0.008 | 0.065 / 0.348 |
| honest none median | -41.681 [-69.206, -15.068] | -41.683 [-69.194, -15.068] | 1.000 | 0.086 | 10.742 / 39.151 |
| honest none precision | -0.441 [-1.018, -0.249] | -0.431 [-1.012, -0.248] | 1.000 | 0.013 | 0.169 / 0.724 |
| honest none mean | -428.639 [-527.576, -375.614] | -428.634 [-527.573, -375.609] | 1.000 | 0.133 | 27.025 / 165.441 |
| pooled (ratio2, own included) | 0.096 [0.057, 0.183] | - | - | - | - |
| truth_strat | -0.002 [-0.004, 0.008] | - | - | - | - |

Reference `e_log_joint_mc_sd` median 0.009; the headline bias is within 2 SE in 5% of the cells; median absolute bias raw 0.489, capped_I 0.155, capped_E 0.209, honest 0.116 (against raw the honest estimate is better, against capped_I better; tie band 0.009, one reference standard error). Calibration of the self-reported SDs (weighted over components): own z median 0.97 [0.84, 1.19], abs z median 1.01 [0.91, 1.21], abs z > 2 fraction 0.11 [0.04, 0.16]; cross z median -0.29 [-0.35, -0.17], abs z median 0.61 [0.55, 0.63], abs z > 2 fraction 0.02 [0.01, 0.04]. Effective training points on a component (weighted median): own run 1.8 [1.5, 2.0], covering other runs 1.2 [1.0, 1.4]. Checks: own-run max abs z 5.72 (weighted offset abs z 2.29), Jacobian max abs z 0.00 (offset abs z 0.00), raw consistency 2.2e-16, flagged cells 0.

### M = 16 (10 cells)

| Estimator | bias | bias with the arm's entropy | covered | mc_se | gp_sd (indep / corr) |
|---|---:|---:|---:|---:|---:|
| raw | 0.756 [0.686, 0.815] | 0.757 [0.690, 0.819] | - | - | - |
| capped_I | 0.140 [0.118, 0.196] | 0.141 [0.111, 0.199] | - | - | - |
| capped_E | 0.265 [0.223, 0.329] | 0.259 [0.222, 0.333] | - | - | - |
| honest ratio1.5 median | 0.013 [-0.030, 0.030] | 0.017 [-0.030, 0.035] | 0.955 | 0.006 | 0.041 / 0.178 |
| honest ratio1.5 precision | 0.025 [-0.010, 0.063] | 0.030 [-0.007, 0.060] | 0.955 | 0.006 | 0.032 / 0.166 |
| honest ratio1.5 mean | 0.003 [-0.037, 0.036] | 0.008 [-0.046, 0.032] | 0.955 | 0.006 | 0.032 / 0.173 |
| honest ratio2 median | -0.098 [-0.143, -0.060] | -0.098 [-0.146, -0.056] | 0.997 | 0.007 | 0.045 / 0.193 |
| honest ratio2 precision | -0.052 [-0.100, -0.005] | -0.043 [-0.096, -0.008] | 0.997 | 0.006 | 0.030 / 0.182 |
| honest ratio2 mean | -0.106 [-0.183, -0.088] | -0.106 [-0.182, -0.083] | 0.997 | 0.006 | 0.032 / 0.197 |
| honest ratio3 median | -0.211 [-0.238, -0.152] | -0.213 [-0.247, -0.144] | 1.000 | 0.007 | 0.053 / 0.214 |
| honest ratio3 precision | -0.103 [-0.157, -0.068] | -0.102 [-0.153, -0.063] | 1.000 | 0.006 | 0.030 / 0.191 |
| honest ratio3 mean | -0.262 [-0.322, -0.224] | -0.266 [-0.318, -0.220] | 1.000 | 0.007 | 0.036 / 0.222 |
| honest ratio5 median | -0.259 [-0.303, -0.218] | -0.264 [-0.302, -0.212] | 1.000 | 0.007 | 0.056 / 0.233 |
| honest ratio5 precision | -0.123 [-0.199, -0.097] | -0.129 [-0.198, -0.094] | 1.000 | 0.006 | 0.030 / 0.199 |
| honest ratio5 mean | -0.368 [-0.412, -0.293] | -0.373 [-0.412, -0.293] | 1.000 | 0.007 | 0.038 / 0.251 |
| honest cap_only median | -0.259 [-0.304, -0.228] | -0.264 [-0.303, -0.214] | 1.000 | 0.007 | 0.056 / 0.233 |
| honest cap_only precision | -0.123 [-0.199, -0.098] | -0.129 [-0.198, -0.095] | 1.000 | 0.006 | 0.030 / 0.199 |
| honest cap_only mean | -0.368 [-0.412, -0.300] | -0.373 [-0.412, -0.298] | 1.000 | 0.007 | 0.039 / 0.252 |
| honest none median | -12.246 [-30.857, -8.876] | -12.236 [-30.857, -8.871] | 1.000 | 0.079 | 7.339 / 18.989 |
| honest none precision | -0.198 [-0.304, -0.121] | -0.198 [-0.300, -0.119] | 1.000 | 0.007 | 0.029 / 0.211 |
| honest none mean | -482.992 [-603.135, -359.662] | -482.982 [-603.131, -359.672] | 1.000 | 0.093 | 13.983 / 114.220 |
| pooled (ratio2, own included) | 0.105 [0.074, 0.124] | - | - | - | - |
| truth_strat | -0.006 [-0.013, 0.011] | - | - | - | - |

Reference `e_log_joint_mc_sd` median 0.008; the headline bias is within 2 SE in 10% of the cells; median absolute bias raw 0.756, capped_I 0.140, capped_E 0.265, honest 0.098 (against raw the honest estimate is better, against capped_I better; tie band 0.008, one reference standard error). Calibration of the self-reported SDs (weighted over components): own z median 1.48 [1.32, 1.70], abs z median 1.48 [1.33, 1.70], abs z > 2 fraction 0.26 [0.23, 0.36]; cross z median -0.14 [-0.17, -0.06], abs z median 0.59 [0.56, 0.64], abs z > 2 fraction 0.02 [0.02, 0.03]. Effective training points on a component (weighted median): own run 1.7 [1.6, 1.8], covering other runs 1.2 [1.1, 1.3]. Checks: own-run max abs z 6.54 (weighted offset abs z 1.87), Jacobian max abs z 0.00 (offset abs z 0.00), raw consistency 2.2e-16, flagged cells 1.

| Run | own error | se | own z median | own abs z median | BQ sd median | pred sd median | own-run max abs z (offset z) | Jacobian max abs z (offset z) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ring_D2_noise3_svbmc_seed1002 | -0.006 | 0.013 | 0.11 | 0.43 | 0.514 | 0.526 | 3.51 (-0.47) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1003 | 0.202 | 0.017 | 0.64 | 0.77 | 0.457 | 0.480 | 1.91 (+0.26) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1004 | 0.434 | 0.015 | 1.30 | 1.30 | 0.409 | 0.421 | 3.32 (-0.10) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1005 | -0.246 | 0.015 | -0.48 | 0.52 | 0.408 | 0.432 | 2.99 (+0.54) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1007 | 0.019 | 0.016 | -0.06 | 0.63 | 0.657 | 0.675 | 3.63 (+0.56) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1009 | 0.048 | 0.015 | 0.40 | 0.48 | 0.427 | 0.455 | 1.67 (-0.30) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1010 | 0.584 | 0.014 | 1.77 | 1.77 | 0.437 | 0.518 | 2.43 (+0.25) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1012 | 0.677 | 0.016 | 2.44 | 2.44 | 0.356 | 0.376 | 4.00 (-0.20) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1013 | 0.054 | 0.014 | 0.35 | 0.66 | 0.377 | 0.403 | 2.56 (+0.22) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1014 | 0.276 | 0.020 | 0.37 | 0.47 | 0.483 | 0.499 | 2.33 (+0.24) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1015 | 0.330 | 0.016 | 0.95 | 0.96 | 0.737 | 0.760 | 2.84 (-0.19) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1017 | 0.368 | 0.018 | 1.08 | 1.08 | 0.419 | 0.434 | 2.53 (-0.26) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1018 | 0.558 | 0.016 | 1.73 | 1.81 | 0.403 | 0.416 | 2.32 (-0.10) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1019 | -0.046 | 0.013 | -0.10 | 0.19 | 0.460 | 0.473 | 3.74 (-0.30) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1020 | 0.491 | 0.018 | 0.68 | 0.81 | 0.687 | 0.703 | 2.72 (-0.55) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1021 | 0.314 | 0.014 | 0.69 | 0.92 | 0.540 | 0.558 | 4.31 (-0.19) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1023 | 0.748 | 0.018 | 1.73 | 1.73 | 0.459 | 0.470 | 2.51 (+1.76) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1025 | 0.692 | 0.016 | 1.06 | 1.06 | 0.626 | 0.650 | 3.83 (+0.87) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1029 | 0.284 | 0.016 | 0.84 | 0.93 | 0.482 | 0.507 | 3.55 (-0.93) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1030 | 0.094 | 0.017 | 0.12 | 0.54 | 0.608 | 0.625 | 2.23 (-0.66) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1031 | -0.083 | 0.015 | -0.18 | 0.34 | 0.502 | 0.530 | 3.31 (-1.08) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1034 | -0.053 | 0.017 | -0.17 | 0.37 | 0.445 | 0.450 | 2.23 (+0.03) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1037 | 0.429 | 0.013 | 0.93 | 0.99 | 0.540 | 0.555 | 2.97 (+1.04) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1039 | 0.131 | 0.016 | 0.46 | 0.91 | 0.440 | 0.461 | 2.64 (-0.19) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1043 | 0.137 | 0.015 | 0.25 | 0.59 | 0.630 | 0.645 | 2.16 (+0.04) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1045 | -0.365 | 0.015 | -0.71 | 0.69 | 0.417 | 0.436 | 3.72 (-0.69) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1047 | 0.086 | 0.015 | 0.13 | 0.31 | 0.716 | 0.743 | 2.79 (+1.54) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1048 | 0.950 | 0.013 | 2.14 | 2.14 | 0.458 | 0.482 | 3.75 (+1.52) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1049 | 0.611 | 0.015 | 0.91 | 0.91 | 0.495 | 0.512 | 3.82 (+1.40) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1050 | 0.005 | 0.013 | 0.47 | 0.78 | 0.448 | 0.473 | 3.17 (+1.87) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1051 | 0.573 | 0.016 | 1.14 | 1.22 | 0.585 | 0.603 | 3.51 (+0.69) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1052 | 0.404 | 0.016 | 1.02 | 1.31 | 0.408 | 0.423 | 2.85 (-0.88) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1053 | 0.537 | 0.023 | 0.77 | 0.77 | 0.684 | 0.707 | 3.38 (+2.06) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1054 | 0.572 | 0.018 | 1.10 | 1.10 | 0.468 | 0.491 | 2.67 (-0.07) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1055 | 1.035 | 0.012 | 1.87 | 1.87 | 0.516 | 0.530 | 2.89 (+0.66) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1056 | 0.222 | 0.014 | 0.28 | 0.57 | 0.574 | 0.589 | 3.39 (+1.70) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1057 | 0.201 | 0.015 | 0.42 | 0.45 | 0.498 | 0.510 | 3.48 (-0.66) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1058 | 0.135 | 0.017 | 0.12 | 0.59 | 0.557 | 0.573 | 3.74 (-0.08) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1059 | 0.320 | 0.014 | 0.77 | 0.98 | 0.580 | 0.603 | 2.82 (+0.49) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1060 | -0.150 | 0.018 | -0.37 | 0.57 | 0.526 | 0.548 | 3.53 (-0.72) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1062 | 0.063 | 0.014 | 0.10 | 0.21 | 0.453 | 0.476 | 2.42 (+0.14) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1063 | 0.156 | 0.015 | 0.31 | 0.50 | 0.558 | 0.574 | 2.93 (+0.04) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1065 | 0.792 | 0.010 | 1.95 | 1.95 | 0.606 | 0.730 | 3.42 (+0.56) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1066 | 0.081 | 0.028 | 0.24 | 0.70 | 0.519 | 0.537 | 1.01 (-0.62) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1067 | -0.177 | 0.013 | -0.20 | 0.34 | 0.478 | 0.495 | 3.22 (-1.01) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1068 | 0.048 | 0.014 | 0.42 | 0.57 | 0.401 | 0.430 | 3.83 (+2.95) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1069 | 0.616 | 0.013 | 1.29 | 1.37 | 0.525 | 0.536 | 2.88 (+0.84) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1071 | 0.638 | 0.015 | 1.11 | 1.11 | 0.554 | 0.568 | 3.41 (-0.82) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1072 | 0.892 | 0.015 | 2.13 | 2.13 | 0.727 | 0.772 | 5.01 (-0.12) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1073 | 0.422 | 0.017 | 0.96 | 0.96 | 0.433 | 0.465 | 2.39 (+0.07) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1074 | 0.447 | 0.022 | 0.84 | 0.84 | 0.532 | 0.549 | 2.34 (-1.64) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1075 | 0.397 | 0.015 | 0.72 | 1.06 | 0.577 | 0.592 | 1.97 (-0.30) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1076 | 0.115 | 0.014 | 0.56 | 0.75 | 0.376 | 0.406 | 3.77 (-0.84) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1077 | 0.371 | 0.014 | 0.85 | 1.09 | 0.698 | 0.885 | 2.89 (-0.05) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1078 | 0.269 | 0.018 | 1.04 | 1.14 | 0.372 | 0.403 | 2.11 (-0.23) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1080 | 0.167 | 0.015 | 0.49 | 0.69 | 0.549 | 0.561 | 2.93 (-0.48) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1083 | 0.039 | 0.014 | 0.30 | 0.74 | 0.579 | 0.590 | 3.36 (+1.39) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1084 | 0.366 | 0.016 | 0.67 | 0.83 | 0.665 | 0.693 | 2.26 (-0.28) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1085 | 0.112 | 0.018 | 0.44 | 0.53 | 0.408 | 0.436 | 2.44 (-0.26) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1087 | 0.061 | 0.020 | 0.15 | 0.47 | 0.512 | 0.596 | 2.26 (+0.07) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1090 | -0.034 | 0.024 | -0.01 | 0.51 | 0.786 | 0.940 | 2.28 (-0.24) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1091 | 0.066 | 0.014 | 0.02 | 0.54 | 0.493 | 0.508 | 3.94 (+0.14) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1092 | 0.042 | 0.015 | 0.06 | 0.67 | 0.587 | 0.645 | 2.04 (-0.07) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1093 | 0.028 | 0.014 | -0.02 | 0.36 | 0.499 | 0.508 | 2.53 (-0.56) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1094 | 0.333 | 0.015 | 0.93 | 1.03 | 0.614 | 0.633 | 2.48 (+0.59) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1096 | 0.360 | 0.014 | 0.78 | 0.79 | 0.607 | 0.627 | 3.38 (+0.19) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1097 | 0.257 | 0.013 | 0.61 | 0.67 | 0.503 | 0.524 | 2.65 (+0.33) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1098 | 0.427 | 0.026 | 1.26 | 1.26 | 0.449 | 0.465 | 1.52 (+0.36) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1099 | 0.333 | 0.015 | 0.66 | 0.78 | 0.632 | 0.651 | 2.59 (+0.14) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1100 | 0.283 | 0.014 | 0.34 | 0.72 | 0.562 | 0.576 | 2.21 (-1.03) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1101 | 0.362 | 0.013 | 0.80 | 0.80 | 0.462 | 0.476 | 2.69 (+0.58) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1104 | 0.466 | 0.016 | 1.10 | 1.10 | 0.418 | 0.443 | 2.18 (-1.36) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1105 | 0.186 | 0.015 | 0.34 | 0.65 | 0.562 | 0.578 | 2.51 (-2.38) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1106 | 0.217 | 0.015 | 0.51 | 0.89 | 0.503 | 0.517 | 2.78 (-0.72) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1107 | 0.299 | 0.015 | 0.36 | 0.64 | 0.641 | 0.658 | 3.94 (+1.01) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1109 | 0.548 | 0.015 | 0.87 | 0.88 | 0.655 | 0.677 | 2.78 (-1.28) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1110 | -0.123 | 0.016 | 0.14 | 0.43 | 0.385 | 0.399 | 2.12 (-0.59) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1111 | 0.281 | 0.015 | 0.62 | 0.81 | 0.584 | 0.597 | 2.68 (-0.49) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1113 | 0.365 | 0.015 | 0.87 | 0.90 | 0.463 | 0.490 | 2.01 (-0.63) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1115 | 0.516 | 0.011 | 1.07 | 1.07 | 0.531 | 0.558 | 3.88 (+0.93) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1116 | 0.284 | 0.022 | 0.57 | 0.70 | 0.479 | 0.503 | 2.42 (-0.51) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1117 | 0.060 | 0.016 | 0.14 | 0.61 | 0.420 | 0.438 | 3.44 (+0.03) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1118 | 0.221 | 0.015 | 0.35 | 0.52 | 0.593 | 0.609 | 2.01 (+0.29) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1119 | 0.522 | 0.015 | 0.87 | 0.88 | 0.591 | 0.607 | 2.40 (-1.72) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1120 | 0.386 | 0.014 | 0.68 | 0.69 | 0.487 | 0.504 | 2.05 (-0.29) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1121 | 0.020 | 0.014 | 0.41 | 0.91 | 0.388 | 0.420 | 5.07 (+1.89) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1122 | -0.088 | 0.023 | -0.17 | 0.63 | 0.613 | 0.635 | 2.50 (-0.56) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1123 | 0.499 | 0.014 | 0.80 | 0.80 | 0.639 | 0.733 | 2.73 (-1.98) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1124 | 0.904 | 0.014 | 2.06 | 2.06 | 0.663 | 0.736 | 3.21 (+1.53) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1126 | 0.229 | 0.017 | 0.49 | 0.57 | 0.534 | 0.553 | 1.95 (+0.06) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1128 | -0.093 | 0.016 | 0.04 | 1.12 | 0.588 | 0.600 | 2.80 (-0.05) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1129 | 0.385 | 0.013 | 1.08 | 1.15 | 0.459 | 0.474 | 2.84 (+0.37) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1130 | -0.109 | 0.016 | 0.08 | 0.40 | 0.638 | 0.661 | 6.36 (+0.70) | 0.00 (+0.00) (flagged) |
| ring_D2_noise3_svbmc_seed1131 | 0.171 | 0.013 | 0.43 | 0.45 | 0.501 | 0.528 | 3.51 (+0.56) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1132 | 0.486 | 0.017 | 1.01 | 1.02 | 0.571 | 0.588 | 3.13 (+0.37) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1133 | 0.206 | 0.012 | 0.55 | 0.78 | 0.546 | 0.561 | 2.45 (+0.71) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1134 | -0.139 | 0.012 | -0.11 | 0.38 | 0.467 | 0.480 | 3.81 (+1.30) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1135 | 0.224 | 0.014 | 0.57 | 1.02 | 0.563 | 0.574 | 3.52 (+1.74) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1140 | 0.395 | 0.015 | 0.71 | 0.71 | 0.609 | 0.626 | 2.97 (-1.17) | 0.00 (+0.00) |
| ring_D2_noise3_svbmc_seed1141 | 0.573 | 0.014 | 1.16 | 1.16 | 0.570 | 0.583 | 2.80 (-0.00) | 0.00 (+0.00) |

## rosenbrock_D2_noise3_svbmc (noisy)

### M = 2 (20 cells)

| Estimator | bias | bias with the arm's entropy | covered | mc_se | gp_sd (indep / corr) |
|---|---:|---:|---:|---:|---:|
| raw | 0.251 [0.148, 0.426] | 0.251 [0.154, 0.414] | - | - | - |
| capped_I | 0.104 [-0.082, 0.179] | 0.093 [-0.089, 0.177] | - | - | - |
| capped_E | 0.151 [0.015, 0.224] | 0.147 [0.009, 0.221] | - | - | - |
| honest ratio1.5 median | -0.012 [-0.224, 0.087] | -0.015 [-0.224, 0.085] | 0.923 | 0.013 | 0.056 / 0.302 |
| honest ratio1.5 precision | -0.012 [-0.224, 0.087] | -0.015 [-0.224, 0.085] | 0.923 | 0.013 | 0.056 / 0.302 |
| honest ratio1.5 mean | -0.012 [-0.224, 0.087] | -0.015 [-0.224, 0.085] | 0.923 | 0.013 | 0.056 / 0.302 |
| honest ratio2 median | -0.014 [-0.253, 0.087] | -0.017 [-0.249, 0.085] | 0.977 | 0.013 | 0.060 / 0.291 |
| honest ratio2 precision | -0.014 [-0.253, 0.087] | -0.017 [-0.249, 0.094] | 0.977 | 0.013 | 0.060 / 0.291 |
| honest ratio2 mean | -0.014 [-0.253, 0.087] | -0.017 [-0.249, 0.085] | 0.977 | 0.013 | 0.060 / 0.291 |
| honest ratio3 median | -0.050 [-0.253, 0.080] | -0.053 [-0.249, 0.074] | 0.994 | 0.013 | 0.067 / 0.290 |
| honest ratio3 precision | -0.050 [-0.253, 0.080] | -0.053 [-0.249, 0.074] | 0.994 | 0.013 | 0.067 / 0.290 |
| honest ratio3 mean | -0.050 [-0.253, 0.080] | -0.053 [-0.249, 0.076] | 0.994 | 0.013 | 0.067 / 0.290 |
| honest ratio5 median | -0.163 [-0.314, 0.076] | -0.177 [-0.300, 0.074] | 1.000 | 0.013 | 0.070 / 0.292 |
| honest ratio5 precision | -0.163 [-0.314, 0.076] | -0.177 [-0.300, 0.074] | 1.000 | 0.013 | 0.070 / 0.292 |
| honest ratio5 mean | -0.163 [-0.314, 0.076] | -0.177 [-0.300, 0.074] | 1.000 | 0.013 | 0.070 / 0.292 |
| honest cap_only median | -0.163 [-0.303, 0.076] | -0.177 [-0.297, 0.074] | 1.000 | 0.013 | 0.070 / 0.297 |
| honest cap_only precision | -0.163 [-0.303, 0.076] | -0.177 [-0.297, 0.074] | 1.000 | 0.013 | 0.070 / 0.297 |
| honest cap_only mean | -0.163 [-0.303, 0.076] | -0.177 [-0.297, 0.074] | 1.000 | 0.013 | 0.070 / 0.297 |
| honest none median | -0.103 [-0.262, 0.066] | -0.122 [-0.261, 0.062] | 1.000 | 0.013 | 0.070 / 0.303 |
| honest none precision | -0.103 [-0.262, 0.066] | -0.122 [-0.261, 0.062] | 1.000 | 0.013 | 0.070 / 0.303 |
| honest none mean | -0.103 [-0.262, 0.066] | -0.122 [-0.261, 0.062] | 1.000 | 0.013 | 0.070 / 0.303 |
| pooled (ratio2, own included) | 0.091 [-0.058, 0.219] | - | - | - | - |
| truth_strat | -0.006 [-0.008, 0.010] | - | - | - | - |

Reference `e_log_joint_mc_sd` median 0.009; the headline bias is within 2 SE in 0% of the cells; median absolute bias raw 0.251, capped_I 0.206, capped_E 0.175, honest 0.207 (against raw the honest estimate is better, against capped_I a tie; tie band 0.009, one reference standard error). Calibration of the self-reported SDs (weighted over components): own z median 0.72 [0.31, 1.05], abs z median 0.98 [0.72, 1.29], abs z > 2 fraction 0.01 [0.00, 0.17]; cross z median -0.40 [-0.87, 0.10], abs z median 0.70 [0.55, 0.93], abs z > 2 fraction 0.00 [0.00, 0.00]. Effective training points on a component (weighted median): own run 4.0 [3.5, 5.1], covering other runs 3.6 [3.2, 4.5]. Checks: own-run max abs z 114.85 (weighted offset abs z 18.22), Jacobian max abs z 0.00 (offset abs z 0.00), raw consistency 2.7e-15, flagged cells 5.

Heaviest component of every run, repetition 0:

| Run, heaviest component (weight) | truth | own I_corr (error; BQ sd; points within 2 SD, their data offset; mean fn) | other runs: estimate (error; pred sd; covered; points within 2 SD, their data offset; mean fn) |
|---|---:|---|---|
| rosenbrock_D2_noise3_svbmc_seed1017, component 1 (0.086) | -4.636 ± 0.087 | -4.377 (+0.259; 0.314; 9, -0.827; -53672.9) | rosenbrock_D2_noise3_svbmc_seed1082: -4.650 (-0.014; 0.292; yes; 6, 1.195; -330075.7) |
| rosenbrock_D2_noise3_svbmc_seed1082, component 2 (0.037) | -4.655 ± 0.067 | -4.579 (+0.075; 0.299; 13, -0.151; -361208.5) | rosenbrock_D2_noise3_svbmc_seed1017: -4.384 (+0.271; 0.345; yes; 4, -1.519; -60079.8) |

### M = 4 (20 cells)

| Estimator | bias | bias with the arm's entropy | covered | mc_se | gp_sd (indep / corr) |
|---|---:|---:|---:|---:|---:|
| raw | 0.351 [0.301, 0.535] | 0.353 [0.304, 0.546] | - | - | - |
| capped_I | 0.079 [-0.006, 0.201] | 0.072 [-0.008, 0.195] | - | - | - |
| capped_E | 0.155 [0.069, 0.253] | 0.155 [0.071, 0.254] | - | - | - |
| honest ratio1.5 median | -0.053 [-0.144, -0.000] | -0.061 [-0.135, -0.005] | 0.997 | 0.010 | 0.058 / 0.237 |
| honest ratio1.5 precision | -0.033 [-0.110, 0.011] | -0.036 [-0.108, 0.007] | 0.997 | 0.010 | 0.039 / 0.217 |
| honest ratio1.5 mean | -0.062 [-0.131, 0.000] | -0.069 [-0.139, -0.007] | 0.997 | 0.010 | 0.041 / 0.216 |
| honest ratio2 median | -0.069 [-0.183, -0.000] | -0.076 [-0.183, -0.004] | 1.000 | 0.010 | 0.061 / 0.240 |
| honest ratio2 precision | -0.045 [-0.123, 0.000] | -0.054 [-0.125, -0.001] | 1.000 | 0.010 | 0.039 / 0.214 |
| honest ratio2 mean | -0.062 [-0.170, -0.017] | -0.075 [-0.172, -0.019] | 1.000 | 0.010 | 0.042 / 0.219 |
| honest ratio3 median | -0.073 [-0.188, 0.002] | -0.079 [-0.189, -0.003] | 1.000 | 0.011 | 0.069 / 0.254 |
| honest ratio3 precision | -0.045 [-0.144, 0.000] | -0.054 [-0.144, -0.001] | 1.000 | 0.010 | 0.039 / 0.215 |
| honest ratio3 mean | -0.062 [-0.180, -0.028] | -0.074 [-0.180, -0.025] | 1.000 | 0.010 | 0.045 / 0.236 |
| honest ratio5 median | -0.065 [-0.223, -0.002] | -0.072 [-0.220, -0.003] | 1.000 | 0.011 | 0.070 / 0.262 |
| honest ratio5 precision | -0.045 [-0.147, 0.002] | -0.054 [-0.148, 0.002] | 1.000 | 0.010 | 0.038 / 0.217 |
| honest ratio5 mean | -0.070 [-0.219, -0.035] | -0.074 [-0.220, -0.032] | 1.000 | 0.010 | 0.047 / 0.247 |
| honest cap_only median | -0.065 [-0.230, -0.002] | -0.072 [-0.223, -0.003] | 1.000 | 0.011 | 0.070 / 0.264 |
| honest cap_only precision | -0.045 [-0.148, 0.002] | -0.054 [-0.151, 0.002] | 1.000 | 0.010 | 0.038 / 0.217 |
| honest cap_only mean | -0.075 [-0.231, -0.035] | -0.074 [-0.233, -0.030] | 1.000 | 0.010 | 0.048 / 0.250 |
| honest none median | -0.166 [-0.322, -0.044] | -0.168 [-0.309, -0.048] | 1.000 | 0.011 | 0.077 / 0.310 |
| honest none precision | -0.045 [-0.149, 0.002] | -0.054 [-0.154, 0.002] | 1.000 | 0.010 | 0.038 / 0.218 |
| honest none mean | -0.228 [-0.501, -0.054] | -0.220 [-0.488, -0.060] | 1.000 | 0.011 | 0.070 / 0.428 |
| pooled (ratio2, own included) | 0.049 [-0.033, 0.091] | - | - | - | - |
| truth_strat | 0.000 [-0.006, 0.009] | - | - | - | - |

Reference `e_log_joint_mc_sd` median 0.009; the headline bias is within 2 SE in 5% of the cells; median absolute bias raw 0.351, capped_I 0.110, capped_E 0.155, honest 0.170 (against raw the honest estimate is better, against capped_I **worse**; tie band 0.009, one reference standard error). Calibration of the self-reported SDs (weighted over components): own z median 0.82 [0.55, 1.32], abs z median 0.87 [0.61, 1.32], abs z > 2 fraction 0.06 [0.01, 0.16]; cross z median -0.14 [-0.38, -0.00], abs z median 0.60 [0.50, 0.68], abs z > 2 fraction 0.01 [0.00, 0.02]. Effective training points on a component (weighted median): own run 4.2 [3.9, 5.0], covering other runs 4.1 [3.3, 4.3]. Checks: own-run max abs z 149.22 (weighted offset abs z 16.91), Jacobian max abs z 0.00 (offset abs z 0.00), raw consistency 8.9e-16, flagged cells 4.

Heaviest component of every run, repetition 0:

| Run, heaviest component (weight) | truth | own I_corr (error; BQ sd; points within 2 SD, their data offset; mean fn) | other runs: estimate (error; pred sd; covered; points within 2 SD, their data offset; mean fn) |
|---|---:|---|---|
| rosenbrock_D2_noise3_svbmc_seed1008, component 5 (0.021) | -4.344 ± 0.030 | -4.241 (+0.103; 0.473; 28, 0.445; -16.8) | rosenbrock_D2_noise3_svbmc_seed1038: -4.521 (-0.177; 0.350; yes; 3, 1.352; -22786.4); rosenbrock_D2_noise3_svbmc_seed1040: -4.489 (-0.145; 0.282; yes; 3, 0.595; -41794.5); rosenbrock_D2_noise3_svbmc_seed1056: -4.340 (+0.003; 0.309; yes; 3, 0.284; -118643.8) |
| rosenbrock_D2_noise3_svbmc_seed1038, component 4 (0.015) | -4.626 ± 0.061 | -4.879 (-0.253; 0.346; 7, -0.239; -22843.3) | rosenbrock_D2_noise3_svbmc_seed1008: -4.582 (+0.044; 0.742; no; 26, 0.182; -16.6); rosenbrock_D2_noise3_svbmc_seed1040: -4.769 (-0.143; 0.285; yes; 2, 2.853; -40963.6); rosenbrock_D2_noise3_svbmc_seed1056: -4.632 (-0.006; 0.311; yes; 4, -1.203; -120391.6) |
| rosenbrock_D2_noise3_svbmc_seed1040, component 2 (0.025) | -4.579 ± 0.067 | -4.534 (+0.045; 0.125; 29, 0.743; -61159.2) | rosenbrock_D2_noise3_svbmc_seed1008: -5.683 (-1.104; 1.090; no; 19, -1.037; -17.9); rosenbrock_D2_noise3_svbmc_seed1038: -5.059 (-0.481; 0.337; yes; 7, 0.176; -23156.6); rosenbrock_D2_noise3_svbmc_seed1056: -4.494 (+0.084; 0.276; yes; 14, -0.644; -91711.3) |
| rosenbrock_D2_noise3_svbmc_seed1056, component 3 (0.050) | -4.504 ± 0.056 | -4.481 (+0.023; 0.282; 15, 0.286; -90013.2) | rosenbrock_D2_noise3_svbmc_seed1008: -6.049 (-1.545; 1.073; no; 13, -0.692; -18.0); rosenbrock_D2_noise3_svbmc_seed1038: -4.994 (-0.491; 0.343; yes; 4, 0.363; -23898.5); rosenbrock_D2_noise3_svbmc_seed1040: -4.600 (-0.097; 0.270; yes; 20, 0.711; -65893.1) |

### M = 8 (20 cells)

| Estimator | bias | bias with the arm's entropy | covered | mc_se | gp_sd (indep / corr) |
|---|---:|---:|---:|---:|---:|
| raw | 0.531 [0.466, 0.584] | 0.530 [0.461, 0.579] | - | - | - |
| capped_I | 0.095 [0.045, 0.131] | 0.101 [0.045, 0.137] | - | - | - |
| capped_E | 0.146 [0.125, 0.216] | 0.155 [0.136, 0.226] | - | - | - |
| honest ratio1.5 median | -0.029 [-0.078, 0.076] | -0.021 [-0.072, 0.068] | 1.000 | 0.010 | 0.046 / 0.173 |
| honest ratio1.5 precision | 0.003 [-0.071, 0.055] | 0.006 [-0.061, 0.036] | 1.000 | 0.011 | 0.023 / 0.153 |
| honest ratio1.5 mean | -0.015 [-0.105, 0.052] | -0.014 [-0.096, 0.057] | 1.000 | 0.010 | 0.025 / 0.154 |
| honest ratio2 median | -0.048 [-0.075, 0.073] | -0.037 [-0.071, 0.066] | 1.000 | 0.010 | 0.050 / 0.177 |
| honest ratio2 precision | -0.010 [-0.077, 0.054] | -0.002 [-0.068, 0.038] | 1.000 | 0.010 | 0.023 / 0.150 |
| honest ratio2 mean | -0.036 [-0.128, 0.043] | -0.037 [-0.124, 0.052] | 1.000 | 0.010 | 0.025 / 0.155 |
| honest ratio3 median | -0.055 [-0.112, 0.027] | -0.061 [-0.101, 0.025] | 1.000 | 0.011 | 0.055 / 0.185 |
| honest ratio3 precision | -0.019 [-0.078, 0.046] | -0.009 [-0.073, 0.029] | 1.000 | 0.011 | 0.023 / 0.149 |
| honest ratio3 mean | -0.078 [-0.159, 0.006] | -0.075 [-0.150, -0.004] | 1.000 | 0.011 | 0.028 / 0.165 |
| honest ratio5 median | -0.060 [-0.137, 0.006] | -0.071 [-0.133, 0.006] | 1.000 | 0.011 | 0.055 / 0.190 |
| honest ratio5 precision | -0.026 [-0.081, 0.040] | -0.014 [-0.077, 0.025] | 1.000 | 0.011 | 0.023 / 0.149 |
| honest ratio5 mean | -0.131 [-0.214, -0.035] | -0.128 [-0.208, -0.035] | 1.000 | 0.011 | 0.030 / 0.183 |
| honest cap_only median | -0.070 [-0.141, 0.006] | -0.075 [-0.137, 0.007] | 1.000 | 0.011 | 0.059 / 0.190 |
| honest cap_only precision | -0.029 [-0.082, 0.038] | -0.017 [-0.079, 0.024] | 1.000 | 0.011 | 0.023 / 0.149 |
| honest cap_only mean | -0.157 [-0.224, -0.044] | -0.153 [-0.218, -0.042] | 1.000 | 0.011 | 0.031 / 0.187 |
| honest none median | -0.101 [-0.202, 0.011] | -0.098 [-0.202, 0.010] | 1.000 | 0.011 | 0.063 / 0.208 |
| honest none precision | -0.033 [-0.084, 0.036] | -0.021 [-0.080, 0.023] | 1.000 | 0.011 | 0.023 / 0.150 |
| honest none mean | -0.296 [-0.482, -0.068] | -0.298 [-0.478, -0.058] | 1.000 | 0.011 | 0.084 / 0.443 |
| pooled (ratio2, own included) | 0.029 [-0.016, 0.121] | - | - | - | - |
| truth_strat | 0.001 [-0.015, 0.015] | - | - | - | - |

Reference `e_log_joint_mc_sd` median 0.009; the headline bias is within 2 SE in 5% of the cells; median absolute bias raw 0.531, capped_I 0.095, capped_E 0.146, honest 0.076 (against raw the honest estimate is better, against capped_I better; tie band 0.009, one reference standard error). Calibration of the self-reported SDs (weighted over components): own z median 1.36 [1.22, 1.52], abs z median 1.36 [1.23, 1.53], abs z > 2 fraction 0.15 [0.10, 0.18]; cross z median -0.13 [-0.25, 0.16], abs z median 0.56 [0.49, 0.66], abs z > 2 fraction 0.00 [0.00, 0.01]. Effective training points on a component (weighted median): own run 4.8 [4.4, 5.2], covering other runs 3.7 [3.4, 4.1]. Checks: own-run max abs z 151.53 (weighted offset abs z 16.30), Jacobian max abs z 0.00 (offset abs z 0.00), raw consistency 8.9e-16, flagged cells 8.

### M = 16 (10 cells)

| Estimator | bias | bias with the arm's entropy | covered | mc_se | gp_sd (indep / corr) |
|---|---:|---:|---:|---:|---:|
| raw | 0.729 [0.673, 0.993] | 0.739 [0.660, 0.991] | - | - | - |
| capped_I | 0.111 [0.036, 0.134] | 0.109 [0.048, 0.137] | - | - | - |
| capped_E | 0.174 [0.125, 0.234] | 0.173 [0.121, 0.231] | - | - | - |
| honest ratio1.5 median | 0.052 [0.003, 0.085] | 0.039 [0.004, 0.088] | 1.000 | 0.011 | 0.047 / 0.143 |
| honest ratio1.5 precision | 0.048 [0.008, 0.072] | 0.044 [0.007, 0.075] | 1.000 | 0.011 | 0.015 / 0.102 |
| honest ratio1.5 mean | 0.017 [-0.002, 0.042] | 0.015 [-0.005, 0.045] | 1.000 | 0.011 | 0.017 / 0.104 |
| honest ratio2 median | 0.044 [0.007, 0.072] | 0.031 [0.008, 0.071] | 1.000 | 0.011 | 0.052 / 0.138 |
| honest ratio2 precision | 0.045 [0.008, 0.062] | 0.038 [0.002, 0.066] | 1.000 | 0.011 | 0.015 / 0.102 |
| honest ratio2 mean | 0.009 [-0.010, 0.023] | 0.007 [-0.011, 0.024] | 1.000 | 0.011 | 0.017 / 0.106 |
| honest ratio3 median | 0.006 [-0.013, 0.041] | -0.002 [-0.019, 0.035] | 1.000 | 0.011 | 0.057 / 0.138 |
| honest ratio3 precision | 0.032 [-0.003, 0.056] | 0.025 [-0.006, 0.057] | 1.000 | 0.011 | 0.015 / 0.101 |
| honest ratio3 mean | -0.046 [-0.059, -0.015] | -0.044 [-0.071, -0.016] | 1.000 | 0.011 | 0.018 / 0.116 |
| honest ratio5 median | 0.001 [-0.023, 0.035] | -0.008 [-0.027, 0.037] | 1.000 | 0.011 | 0.057 / 0.137 |
| honest ratio5 precision | 0.029 [0.004, 0.054] | 0.022 [-0.006, 0.053] | 1.000 | 0.011 | 0.015 / 0.101 |
| honest ratio5 mean | -0.058 [-0.099, -0.035] | -0.057 [-0.104, -0.026] | 1.000 | 0.011 | 0.019 / 0.121 |
| honest cap_only median | -0.000 [-0.024, 0.033] | -0.009 [-0.030, 0.028] | 1.000 | 0.011 | 0.057 / 0.137 |
| honest cap_only precision | 0.029 [0.003, 0.054] | 0.021 [-0.010, 0.056] | 1.000 | 0.011 | 0.015 / 0.101 |
| honest cap_only mean | -0.060 [-0.106, -0.044] | -0.058 [-0.112, -0.039] | 1.000 | 0.011 | 0.020 / 0.123 |
| honest none median | -0.018 [-0.042, 0.015] | -0.024 [-0.050, 0.019] | 1.000 | 0.011 | 0.059 / 0.147 |
| honest none precision | 0.028 [0.002, 0.053] | 0.019 [-0.012, 0.055] | 1.000 | 0.011 | 0.015 / 0.102 |
| honest none mean | -0.247 [-0.355, -0.067] | -0.248 [-0.358, -0.068] | 1.000 | 0.011 | 0.050 / 0.265 |
| pooled (ratio2, own included) | 0.075 [0.043, 0.101] | - | - | - | - |
| truth_strat | 0.001 [-0.002, 0.004] | - | - | - | - |

Reference `e_log_joint_mc_sd` median 0.009; the headline bias is within 2 SE in 40% of the cells; median absolute bias raw 0.729, capped_I 0.111, capped_E 0.174, honest 0.044 (against raw the honest estimate is better, against capped_I better; tie band 0.009, one reference standard error). Calibration of the self-reported SDs (weighted over components): own z median 1.96 [1.78, 2.42], abs z median 1.96 [1.78, 2.42], abs z > 2 fraction 0.49 [0.35, 0.62]; cross z median 0.06 [0.00, 0.17], abs z median 0.61 [0.53, 0.72], abs z > 2 fraction 0.01 [0.01, 0.02]. Effective training points on a component (weighted median): own run 4.8 [4.2, 5.3], covering other runs 3.9 [3.1, 4.2]. Checks: own-run max abs z 127.92 (weighted offset abs z 18.10), Jacobian max abs z 0.00 (offset abs z 0.00), raw consistency 8.9e-16, flagged cells 8.

| Run | own error | se | own z median | own abs z median | BQ sd median | pred sd median | own-run max abs z (offset z) | Jacobian max abs z (offset z) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rosenbrock_D2_noise3_svbmc_seed1000 | 0.087 | 0.017 | 0.52 | 0.59 | 0.338 | 0.304 | 3.53 (+1.22) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1001 | -0.115 | 0.014 | -0.20 | 0.71 | 0.555 | 0.717 | 2.78 (+1.15) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1002 | -0.011 | 0.017 | -0.10 | 0.46 | 0.424 | 0.459 | 2.41 (+0.49) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1003 | 0.022 | 0.015 | 0.08 | 0.15 | 0.292 | 0.294 | 2.93 (-0.98) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1004 | 0.036 | 0.011 | 0.25 | 0.44 | 0.602 | 0.762 | 3.99 (-1.36) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1005 | -0.056 | 0.012 | -0.11 | 0.32 | 0.333 | 0.342 | 1.95 (+1.19) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1006 | -0.396 | 0.015 | -1.06 | 1.02 | 0.404 | 0.408 | 3.12 (+1.13) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1007 | -0.278 | 0.019 | -0.96 | 0.95 | 0.310 | 0.331 | 3.61 (-1.66) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1008 | -0.053 | 0.016 | 0.08 | 0.33 | 0.648 | 0.802 | 2.39 (-0.50) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1009 | -0.201 | 0.013 | -0.72 | 0.72 | 0.321 | 0.319 | 2.49 (+0.81) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1010 | 0.321 | 0.014 | 0.84 | 0.84 | 0.317 | 0.315 | 2.25 (-0.06) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1011 | 0.514 | 0.012 | 1.13 | 1.13 | 0.554 | 0.723 | 3.72 (+2.09) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1012 | 0.506 | 0.014 | 1.26 | 1.26 | 0.365 | 0.372 | 2.62 (-0.88) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1013 | 0.034 | 0.014 | 0.33 | 0.50 | 0.462 | 0.567 | 3.52 (+1.37) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1014 | 0.168 | 0.015 | 0.41 | 0.70 | 0.336 | 0.342 | 3.49 (+1.12) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1015 | -0.080 | 0.016 | -0.12 | 0.42 | 0.465 | 0.476 | 3.09 (-0.84) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1016 | 0.250 | 0.020 | 4.72 | 7.39 | 0.000 | 0.340 | 125.36 (-6.13) | 0.00 (+0.00) (flagged) |
| rosenbrock_D2_noise3_svbmc_seed1017 | 0.242 | 0.018 | 0.45 | 0.45 | 0.348 | 0.356 | 3.52 (-1.54) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1018 | 0.019 | 0.016 | -0.04 | 0.47 | 0.353 | 0.357 | 2.18 (+0.44) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1019 | -0.049 | 0.014 | 0.26 | 0.65 | 0.593 | 0.709 | 3.47 (+2.29) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1020 | -0.119 | 0.015 | -0.03 | 0.95 | 0.313 | 0.320 | 2.16 (-1.30) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1021 | 0.116 | 0.015 | 0.15 | 0.37 | 0.538 | 0.587 | 4.25 (+0.95) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1022 | 0.070 | 0.013 | 0.19 | 0.47 | 0.300 | 0.310 | 3.77 (-0.68) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1023 | 0.151 | 0.015 | 0.52 | 0.52 | 0.331 | 0.335 | 2.95 (+0.13) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1024 | 0.041 | 0.016 | -0.03 | 0.67 | 0.324 | 0.327 | 2.32 (-1.49) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1025 | 0.446 | 0.013 | 1.20 | 1.20 | 0.317 | 0.318 | 3.36 (+0.11) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1026 | 0.174 | 0.017 | 0.44 | 0.53 | 0.328 | 0.332 | 3.76 (+0.45) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1027 | -0.255 | 0.036 | -0.80 | 0.74 | 0.355 | 0.400 | 3.94 (-1.19) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1028 | 0.374 | 0.017 | 1.25 | 1.25 | 0.308 | 0.312 | 2.16 (-2.67) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1029 | 0.381 | 0.022 | 0.66 | 0.66 | 0.480 | 0.581 | 2.45 (+0.52) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1030 | 0.179 | 0.016 | 0.44 | 0.44 | 0.360 | 0.363 | 3.18 (+0.18) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1031 | 0.030 | 0.015 | -0.05 | 0.99 | 0.404 | 0.418 | 2.83 (+0.90) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1032 | 0.271 | 0.014 | 0.92 | 0.92 | 0.300 | 0.321 | 2.11 (-0.05) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1034 | -0.010 | 0.014 | -0.15 | 0.31 | 0.358 | 0.365 | 3.08 (+1.75) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1035 | -0.152 | 0.013 | -0.40 | 0.38 | 0.353 | 0.373 | 4.99 (+1.65) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1036 | 0.226 | 0.014 | 0.70 | 0.91 | 0.424 | 0.523 | 3.20 (-0.26) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1037 | -0.035 | 0.019 | -0.48 | 0.98 | 0.445 | 0.455 | 3.03 (+1.79) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1038 | -0.275 | 0.016 | -0.94 | 0.90 | 0.375 | 0.379 | 1.65 (-0.81) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1039 | -0.024 | 0.015 | -0.54 | 1.04 | 0.335 | 0.341 | 2.35 (+0.13) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1040 | -0.075 | 0.017 | -0.24 | 0.33 | 0.370 | 0.278 | 7.51 (-3.69) | 0.00 (+0.00) (flagged) |
| rosenbrock_D2_noise3_svbmc_seed1041 | 0.692 | 0.015 | 2.12 | 2.12 | 0.347 | 0.356 | 2.96 (-1.30) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1042 | -0.197 | 0.023 | -2.05 | 3.87 | 0.000 | 0.382 | 8.77 (+0.13) | 0.00 (+0.00) (flagged) |
| rosenbrock_D2_noise3_svbmc_seed1043 | 0.423 | 0.018 | 0.57 | 0.58 | 0.583 | 0.718 | 3.89 (+0.76) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1044 | 0.016 | 0.014 | -0.25 | 0.72 | 0.382 | 0.389 | 4.28 (+1.24) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1045 | -0.487 | 0.013 | -1.62 | 1.59 | 0.305 | 0.310 | 4.37 (+0.21) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1046 | 0.281 | 0.013 | 0.86 | 0.86 | 0.275 | 0.280 | 2.58 (-0.65) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1047 | 0.510 | 0.026 | 1.73 | 1.73 | 0.331 | 0.335 | 2.84 (-1.07) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1048 | 0.643 | 0.014 | 1.76 | 1.76 | 0.348 | 0.354 | 2.40 (-1.30) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1049 | 1.041 | 0.008 | 1.55 | 1.55 | 0.658 | 0.804 | 2.79 (-1.13) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1050 | 0.172 | 0.018 | 0.48 | 1.10 | 0.415 | 0.467 | 2.30 (-1.67) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1051 | 0.242 | 0.017 | 0.69 | 0.69 | 0.365 | 0.369 | 2.72 (-0.22) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1052 | 0.414 | 0.022 | 0.97 | 0.98 | 0.596 | 0.683 | 3.15 (+0.32) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1053 | 0.138 | 0.012 | 0.68 | 0.82 | 0.410 | 0.429 | 2.17 (-1.07) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1054 | 0.287 | 0.013 | 0.54 | 0.54 | 0.358 | 0.365 | 2.90 (-0.59) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1055 | 1.036 | 0.015 | 3.16 | 3.16 | 0.365 | 0.373 | 3.46 (-0.43) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1056 | 0.052 | 0.019 | 0.13 | 0.24 | 0.288 | 0.303 | 2.90 (-0.26) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1057 | 0.146 | 0.016 | 0.31 | 0.34 | 0.342 | 0.287 | 4.73 (-3.46) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1058 | -0.092 | 0.012 | -0.26 | 0.26 | 0.354 | 0.362 | 3.01 (+1.66) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1059 | 0.131 | 0.016 | 0.52 | 0.61 | 0.375 | 0.390 | 2.39 (-0.38) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1060 | -0.342 | 0.016 | -0.55 | 0.51 | 0.712 | 0.824 | 3.34 (-0.51) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1061 | 0.248 | 0.016 | 0.80 | 0.83 | 0.578 | 0.657 | 3.32 (-0.70) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1062 | 0.192 | 0.016 | 0.59 | 0.87 | 0.426 | 0.457 | 3.39 (-0.13) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1063 | 0.063 | 0.016 | 0.44 | 0.52 | 0.373 | 0.379 | 4.73 (+0.27) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1064 | 0.499 | 0.017 | 1.44 | 1.44 | 0.378 | 0.399 | 1.99 (-0.02) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1066 | 0.228 | 0.015 | 0.64 | 0.64 | 0.359 | 0.320 | 8.98 (+1.81) | 0.00 (+0.00) (flagged) |
| rosenbrock_D2_noise3_svbmc_seed1067 | -0.462 | 0.012 | -1.29 | 1.28 | 0.361 | 0.389 | 5.13 (+1.01) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1068 | 0.059 | 0.013 | 0.09 | 0.25 | 0.285 | 0.303 | 2.85 (+0.80) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1069 | 0.215 | 0.012 | 0.41 | 0.46 | 0.692 | 0.808 | 2.45 (-1.20) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1070 | -0.179 | 0.019 | -0.96 | 1.01 | 0.317 | 0.323 | 2.40 (-1.03) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1071 | 0.193 | 0.014 | 0.43 | 0.43 | 0.401 | 0.386 | 3.48 (+0.58) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1072 | 0.304 | 0.015 | 0.84 | 0.91 | 0.385 | 0.381 | 6.19 (+0.98) | 0.00 (+0.00) (flagged) |
| rosenbrock_D2_noise3_svbmc_seed1073 | 0.669 | 0.013 | 1.08 | 1.08 | 0.569 | 0.737 | 2.35 (-0.38) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1074 | 0.469 | 0.021 | 1.71 | 1.71 | 0.338 | 0.341 | 3.42 (-0.04) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1075 | -0.070 | 0.012 | -0.47 | 0.55 | 0.322 | 0.299 | 3.23 (+1.04) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1076 | 0.253 | 0.017 | 0.51 | 0.80 | 0.340 | 0.350 | 2.71 (-0.95) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1077 | 0.283 | 0.016 | 0.70 | 0.70 | 0.406 | 0.413 | 5.93 (+0.46) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1078 | 0.147 | 0.014 | 0.31 | 0.38 | 0.317 | 0.354 | 2.70 (-1.22) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1079 | 0.098 | 0.013 | 0.07 | 0.70 | 0.372 | 0.275 | 8.58 (-0.94) | 0.00 (+0.00) (flagged) |
| rosenbrock_D2_noise3_svbmc_seed1080 | 0.146 | 0.017 | 0.26 | 0.49 | 0.316 | 0.325 | 4.43 (+2.30) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1081 | -0.044 | 0.013 | 0.03 | 0.36 | 0.546 | 0.759 | 2.94 (+0.16) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1082 | 0.033 | 0.017 | -0.03 | 0.31 | 0.329 | 0.325 | 3.38 (+0.52) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1083 | 0.334 | 0.016 | 0.66 | 1.06 | 0.353 | 0.358 | 2.72 (-1.07) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1084 | 0.233 | 0.013 | 0.75 | 0.76 | 0.325 | 0.344 | 3.49 (+0.84) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1085 | 0.182 | 0.032 | 0.65 | 0.76 | 0.414 | 0.449 | 2.18 (+1.44) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1086 | 0.295 | 0.020 | 0.96 | 0.96 | 0.374 | 0.399 | 3.69 (-0.24) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1087 | 0.105 | 0.017 | 0.42 | 0.47 | 0.366 | 0.384 | 2.26 (-1.45) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1088 | -0.040 | 0.017 | -0.70 | 0.86 | 0.249 | 0.406 | 8.11 (-1.52) | 0.00 (+0.00) (flagged) |
| rosenbrock_D2_noise3_svbmc_seed1089 | 0.381 | 0.014 | 1.37 | 1.46 | 0.379 | 0.418 | 3.15 (-0.09) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1090 | -0.066 | 0.017 | -0.46 | 1.01 | 0.210 | 0.310 | 6.03 (+2.11) | 0.00 (+0.00) (flagged) |
| rosenbrock_D2_noise3_svbmc_seed1091 | 0.093 | 0.014 | 1.71 | 3.19 | 0.000 | 0.328 | 19.97 (-0.50) | 0.00 (+0.00) (flagged) |
| rosenbrock_D2_noise3_svbmc_seed1092 | -0.195 | 0.013 | -0.35 | 0.45 | 0.331 | 0.330 | 2.73 (+1.10) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1093 | -0.162 | 0.010 | -0.16 | 0.48 | 0.670 | 0.910 | 3.10 (+0.09) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1094 | -0.128 | 0.020 | -0.67 | 0.63 | 0.341 | 0.356 | 3.72 (-0.50) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1095 | -0.212 | 0.017 | -0.43 | 0.41 | 0.402 | 0.431 | 3.74 (+0.59) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1096 | 0.420 | 0.016 | 0.86 | 0.86 | 0.467 | 0.547 | 3.49 (+0.69) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1097 | 0.419 | 0.015 | 0.96 | 1.01 | 0.556 | 0.695 | 2.34 (-1.25) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1099 | 0.292 | 0.015 | 0.79 | 0.79 | 0.361 | 0.369 | 3.01 (+0.89) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1100 | 0.017 | 0.015 | 0.17 | 0.40 | 0.372 | 0.392 | 2.43 (+0.28) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1101 | 0.294 | 0.016 | 1.11 | 1.11 | 0.322 | 0.328 | 3.43 (-0.02) | 0.00 (+0.00) |
| rosenbrock_D2_noise3_svbmc_seed1102 | 0.197 | 0.016 | -0.10 | 0.80 | 0.502 | 0.583 | 3.25 (-0.68) | 0.00 (+0.00) |

## student_D8_noise3_svbmc (noisy)

### M = 2 (20 cells)

| Estimator | bias | bias with the arm's entropy | covered | mc_se | gp_sd (indep / corr) |
|---|---:|---:|---:|---:|---:|
| raw | -0.083 [-0.246, -0.002] | -0.109 [-0.311, -0.039] | - | - | - |
| capped_I | -0.911 [-1.144, -0.602] | -0.945 [-1.182, -0.655] | - | - | - |
| capped_E | -0.382 [-0.557, -0.235] | -0.388 [-0.570, -0.265] | - | - | - |
| honest ratio1.5 median | -1.175 [-1.678, -1.003] | -1.200 [-1.697, -1.022] | 0.999 | 0.038 | 0.150 / 0.671 |
| honest ratio1.5 precision | -1.175 [-1.678, -1.003] | -1.200 [-1.697, -1.022] | 0.999 | 0.038 | 0.150 / 0.671 |
| honest ratio1.5 mean | -1.175 [-1.678, -1.003] | -1.200 [-1.697, -1.022] | 0.999 | 0.038 | 0.150 / 0.671 |
| honest ratio2 median | -1.175 [-1.709, -0.983] | -1.204 [-1.733, -1.028] | 1.000 | 0.038 | 0.150 / 0.671 |
| honest ratio2 precision | -1.175 [-1.709, -1.010] | -1.204 [-1.733, -1.028] | 1.000 | 0.038 | 0.150 / 0.671 |
| honest ratio2 mean | -1.175 [-1.709, -1.010] | -1.204 [-1.733, -1.028] | 1.000 | 0.038 | 0.150 / 0.671 |
| honest ratio3 median | -1.175 [-1.709, -1.010] | -1.204 [-1.733, -1.028] | 1.000 | 0.038 | 0.150 / 0.671 |
| honest ratio3 precision | -1.175 [-1.709, -1.010] | -1.204 [-1.733, -1.028] | 1.000 | 0.038 | 0.150 / 0.671 |
| honest ratio3 mean | -1.175 [-1.709, -1.010] | -1.204 [-1.733, -1.028] | 1.000 | 0.038 | 0.150 / 0.671 |
| honest ratio5 median | -1.175 [-1.709, -0.983] | -1.204 [-1.733, -1.028] | 1.000 | 0.038 | 0.150 / 0.671 |
| honest ratio5 precision | -1.175 [-1.709, -1.010] | -1.204 [-1.733, -1.028] | 1.000 | 0.038 | 0.150 / 0.671 |
| honest ratio5 mean | -1.175 [-1.709, -1.010] | -1.204 [-1.733, -1.028] | 1.000 | 0.038 | 0.150 / 0.671 |
| honest cap_only median | -1.175 [-1.709, -1.010] | -1.204 [-1.733, -1.028] | 1.000 | 0.038 | 0.150 / 0.671 |
| honest cap_only precision | -1.175 [-1.709, -1.010] | -1.204 [-1.733, -1.028] | 1.000 | 0.038 | 0.150 / 0.671 |
| honest cap_only mean | -1.175 [-1.709, -1.010] | -1.204 [-1.733, -1.028] | 1.000 | 0.038 | 0.150 / 0.671 |
| honest none median | -1.175 [-1.709, -1.010] | -1.204 [-1.733, -1.028] | 1.000 | 0.038 | 0.150 / 0.671 |
| honest none precision | -1.175 [-1.709, -1.010] | -1.204 [-1.733, -1.028] | 1.000 | 0.038 | 0.150 / 0.671 |
| honest none mean | -1.175 [-1.709, -1.010] | -1.204 [-1.733, -1.028] | 1.000 | 0.038 | 0.150 / 0.671 |
| pooled (ratio2, own included) | -0.713 [-0.986, -0.533] | - | - | - | - |
| truth_strat | 0.004 [-0.030, 0.039] | - | - | - | - |

Reference `e_log_joint_mc_sd` median 0.029; the headline bias is within 2 SE in 0% of the cells; median absolute bias raw 0.151, capped_I 0.911, capped_E 0.382, honest 1.175 (against raw the honest estimate is **worse**, against capped_I **worse**; tie band 0.029, one reference standard error). Calibration of the self-reported SDs (weighted over components): own z median -0.18 [-0.43, 0.09], abs z median 0.72 [0.55, 0.92], abs z > 2 fraction 0.07 [0.03, 0.09]; cross z median -1.62 [-1.97, -1.25], abs z median 1.56 [1.24, 1.93], abs z > 2 fraction 0.22 [0.08, 0.49]. Effective training points on a component (weighted median): own run 1.0 [0.8, 1.1], covering other runs 0.5 [0.4, 0.6]. Checks: own-run max abs z 3.54 (weighted offset abs z 3.01), Jacobian max abs z 0.00 (offset abs z 0.00), raw consistency 1.4e-14, flagged cells 0.

Heaviest component of every run, repetition 0:

| Run, heaviest component (weight) | truth | own I_corr (error; BQ sd; points within 2 SD, their data offset; mean fn) | other runs: estimate (error; pred sd; covered; points within 2 SD, their data offset; mean fn) |
|---|---:|---|---|
| student_D8_noise3_svbmc_seed1056, component 0 (0.050) | -34.361 ± 0.309 | -34.577 (-0.216; 0.350; 3, 0.029; -50.8) | student_D8_noise3_svbmc_seed1078: -35.664 (-1.303; 0.920; yes; 1, 0.075; -49.6) |
| student_D8_noise3_svbmc_seed1078, component 3 (0.058) | -33.132 ± 0.238 | -33.892 (-0.760; 0.367; 0, -; -49.7) | student_D8_noise3_svbmc_seed1056: -34.096 (-0.964; 0.766; yes; 2, 2.013; -50.9) |

### M = 4 (20 cells)

| Estimator | bias | bias with the arm's entropy | covered | mc_se | gp_sd (indep / corr) |
|---|---:|---:|---:|---:|---:|
| raw | -0.105 [-0.159, 0.050] | -0.100 [-0.175, 0.030] | - | - | - |
| capped_I | -1.178 [-1.456, -0.997] | -1.167 [-1.456, -0.974] | - | - | - |
| capped_E | -0.612 [-0.695, -0.439] | -0.614 [-0.690, -0.415] | - | - | - |
| honest ratio1.5 median | -1.466 [-1.722, -1.241] | -1.439 [-1.740, -1.236] | 1.000 | 0.033 | 0.125 / 0.529 |
| honest ratio1.5 precision | -1.530 [-1.791, -1.358] | -1.513 [-1.801, -1.347] | 1.000 | 0.032 | 0.080 / 0.458 |
| honest ratio1.5 mean | -1.450 [-1.742, -1.342] | -1.435 [-1.761, -1.307] | 1.000 | 0.033 | 0.083 / 0.451 |
| honest ratio2 median | -1.485 [-1.792, -1.232] | -1.465 [-1.797, -1.228] | 1.000 | 0.034 | 0.131 / 0.541 |
| honest ratio2 precision | -1.573 [-1.834, -1.365] | -1.555 [-1.833, -1.343] | 1.000 | 0.033 | 0.077 / 0.455 |
| honest ratio2 mean | -1.540 [-1.830, -1.333] | -1.531 [-1.818, -1.309] | 1.000 | 0.034 | 0.082 / 0.455 |
| honest ratio3 median | -1.488 [-1.798, -1.232] | -1.469 [-1.803, -1.228] | 1.000 | 0.034 | 0.131 / 0.540 |
| honest ratio3 precision | -1.576 [-1.836, -1.365] | -1.558 [-1.835, -1.344] | 1.000 | 0.033 | 0.077 / 0.455 |
| honest ratio3 mean | -1.548 [-1.833, -1.335] | -1.539 [-1.819, -1.309] | 1.000 | 0.034 | 0.082 / 0.460 |
| honest ratio5 median | -1.488 [-1.798, -1.232] | -1.469 [-1.803, -1.228] | 1.000 | 0.034 | 0.131 / 0.540 |
| honest ratio5 precision | -1.576 [-1.836, -1.365] | -1.558 [-1.835, -1.344] | 1.000 | 0.033 | 0.077 / 0.455 |
| honest ratio5 mean | -1.548 [-1.833, -1.335] | -1.539 [-1.819, -1.309] | 1.000 | 0.034 | 0.082 / 0.460 |
| honest cap_only median | -1.488 [-1.798, -1.232] | -1.469 [-1.803, -1.228] | 1.000 | 0.034 | 0.131 / 0.540 |
| honest cap_only precision | -1.576 [-1.836, -1.365] | -1.558 [-1.835, -1.344] | 1.000 | 0.033 | 0.077 / 0.455 |
| honest cap_only mean | -1.548 [-1.833, -1.335] | -1.539 [-1.819, -1.309] | 1.000 | 0.034 | 0.082 / 0.460 |
| honest none median | -1.488 [-1.798, -1.232] | -1.469 [-1.803, -1.228] | 1.000 | 0.034 | 0.131 / 0.540 |
| honest none precision | -1.576 [-1.836, -1.365] | -1.558 [-1.835, -1.345] | 1.000 | 0.033 | 0.077 / 0.455 |
| honest none mean | -1.548 [-1.833, -1.335] | -1.539 [-1.819, -1.309] | 1.000 | 0.034 | 0.082 / 0.460 |
| pooled (ratio2, own included) | -1.254 [-1.430, -1.035] | - | - | - | - |
| truth_strat | -0.024 [-0.044, 0.003] | - | - | - | - |

Reference `e_log_joint_mc_sd` median 0.029; the headline bias is within 2 SE in 0% of the cells; median absolute bias raw 0.159, capped_I 1.178, capped_E 0.612, honest 1.485 (against raw the honest estimate is **worse**, against capped_I **worse**; tie band 0.029, one reference standard error). Calibration of the self-reported SDs (weighted over components): own z median -0.15 [-0.42, 0.26], abs z median 0.68 [0.64, 0.83], abs z > 2 fraction 0.07 [0.05, 0.10]; cross z median -1.67 [-1.81, -1.46], abs z median 1.66 [1.46, 1.81], abs z > 2 fraction 0.36 [0.26, 0.42]. Effective training points on a component (weighted median): own run 0.8 [0.7, 0.8], covering other runs 0.4 [0.3, 0.5]. Checks: own-run max abs z 3.76 (weighted offset abs z 1.68), Jacobian max abs z 0.00 (offset abs z 0.00), raw consistency 1.4e-14, flagged cells 0.

Heaviest component of every run, repetition 0:

| Run, heaviest component (weight) | truth | own I_corr (error; BQ sd; points within 2 SD, their data offset; mean fn) | other runs: estimate (error; pred sd; covered; points within 2 SD, their data offset; mean fn) |
|---|---:|---|---|
| student_D8_noise3_svbmc_seed1007, component 0 (0.007) | -34.716 ± 0.262 | -34.487 (+0.229; 0.328; 3, 1.888; -47.3) | student_D8_noise3_svbmc_seed1036: -35.331 (-0.615; 0.851; yes; 3, -1.491; -50.1); student_D8_noise3_svbmc_seed1048: -34.611 (+0.105; 1.216; yes; 1, 1.246; -46.6); student_D8_noise3_svbmc_seed1085: -35.084 (-0.368; 1.199; yes; 4, 0.146; -43.3) |
| student_D8_noise3_svbmc_seed1036, component 0 (0.006) | -34.475 ± 0.292 | -34.058 (+0.417; 0.334; 1, -3.344; -50.2) | student_D8_noise3_svbmc_seed1007: -34.915 (-0.440; 0.796; yes; 1, 1.696; -47.0); student_D8_noise3_svbmc_seed1048: -34.753 (-0.278; 1.181; yes; 0, -; -46.6); student_D8_noise3_svbmc_seed1085: -35.115 (-0.640; 1.226; yes; 5, 1.430; -43.1) |
| student_D8_noise3_svbmc_seed1048, component 1 (0.181) | -31.835 ± 0.203 | -31.422 (+0.413; 0.414; 0, -; -46.5) | student_D8_noise3_svbmc_seed1007: -33.630 (-1.795; 0.705; yes; 1, 1.696; -46.8); student_D8_noise3_svbmc_seed1036: -33.691 (-1.856; 0.716; yes; 1, -1.762; -49.3); student_D8_noise3_svbmc_seed1085: -33.086 (-1.251; 1.021; yes; 3, 1.036; -42.4) |
| student_D8_noise3_svbmc_seed1085, component 4 (0.076) | -32.231 ± 0.198 | -31.699 (+0.532; 0.403; 2, 2.022; -42.9) | student_D8_noise3_svbmc_seed1007: -33.720 (-1.489; 0.700; yes; 0, -; -46.8); student_D8_noise3_svbmc_seed1036: -33.841 (-1.611; 0.739; yes; 0, -; -49.1); student_D8_noise3_svbmc_seed1048: -32.583 (-0.352; 0.992; yes; 0, -; -46.4) |

### M = 8 (20 cells)

| Estimator | bias | bias with the arm's entropy | covered | mc_se | gp_sd (indep / corr) |
|---|---:|---:|---:|---:|---:|
| raw | 0.250 [0.075, 0.383] | 0.222 [0.094, 0.360] | - | - | - |
| capped_I | -1.386 [-1.623, -1.244] | -1.391 [-1.612, -1.244] | - | - | - |
| capped_E | -0.912 [-1.093, -0.633] | -0.919 [-1.089, -0.641] | - | - | - |
| honest ratio1.5 median | -1.412 [-1.516, -1.285] | -1.393 [-1.603, -1.278] | 1.000 | 0.030 | 0.115 / 0.399 |
| honest ratio1.5 precision | -1.455 [-1.570, -1.325] | -1.474 [-1.573, -1.316] | 1.000 | 0.026 | 0.044 / 0.309 |
| honest ratio1.5 mean | -1.406 [-1.497, -1.308] | -1.418 [-1.543, -1.306] | 1.000 | 0.027 | 0.045 / 0.311 |
| honest ratio2 median | -1.418 [-1.515, -1.325] | -1.398 [-1.602, -1.314] | 1.000 | 0.030 | 0.122 / 0.402 |
| honest ratio2 precision | -1.457 [-1.584, -1.337] | -1.479 [-1.585, -1.341] | 1.000 | 0.026 | 0.043 / 0.309 |
| honest ratio2 mean | -1.418 [-1.532, -1.338] | -1.431 [-1.579, -1.331] | 1.000 | 0.027 | 0.044 / 0.315 |
| honest ratio3 median | -1.418 [-1.516, -1.326] | -1.398 [-1.603, -1.314] | 1.000 | 0.030 | 0.122 / 0.402 |
| honest ratio3 precision | -1.458 [-1.585, -1.337] | -1.480 [-1.586, -1.342] | 1.000 | 0.026 | 0.043 / 0.309 |
| honest ratio3 mean | -1.420 [-1.534, -1.339] | -1.433 [-1.581, -1.332] | 1.000 | 0.027 | 0.044 / 0.315 |
| honest ratio5 median | -1.418 [-1.516, -1.326] | -1.398 [-1.603, -1.314] | 1.000 | 0.030 | 0.122 / 0.402 |
| honest ratio5 precision | -1.458 [-1.585, -1.337] | -1.480 [-1.586, -1.342] | 1.000 | 0.026 | 0.043 / 0.309 |
| honest ratio5 mean | -1.420 [-1.534, -1.339] | -1.433 [-1.581, -1.332] | 1.000 | 0.027 | 0.044 / 0.315 |
| honest cap_only median | -1.418 [-1.516, -1.326] | -1.398 [-1.603, -1.314] | 1.000 | 0.030 | 0.122 / 0.402 |
| honest cap_only precision | -1.458 [-1.585, -1.337] | -1.480 [-1.586, -1.342] | 1.000 | 0.026 | 0.043 / 0.309 |
| honest cap_only mean | -1.420 [-1.534, -1.339] | -1.433 [-1.581, -1.332] | 1.000 | 0.027 | 0.044 / 0.315 |
| honest none median | -1.424 [-1.517, -1.326] | -1.404 [-1.604, -1.314] | 1.000 | 0.030 | 0.122 / 0.403 |
| honest none precision | -1.459 [-1.585, -1.337] | -1.480 [-1.586, -1.342] | 1.000 | 0.026 | 0.043 / 0.309 |
| honest none mean | -1.428 [-1.534, -1.339] | -1.436 [-1.584, -1.332] | 1.000 | 0.027 | 0.044 / 0.316 |
| pooled (ratio2, own included) | -1.318 [-1.389, -1.191] | - | - | - | - |
| truth_strat | 0.005 [-0.021, 0.023] | - | - | - | - |

Reference `e_log_joint_mc_sd` median 0.028; the headline bias is within 2 SE in 0% of the cells; median absolute bias raw 0.296, capped_I 1.386, capped_E 0.912, honest 1.418 (against raw the honest estimate is **worse**, against capped_I **worse**; tie band 0.028, one reference standard error). Calibration of the self-reported SDs (weighted over components): own z median 0.56 [0.20, 0.85], abs z median 0.86 [0.71, 0.92], abs z > 2 fraction 0.05 [0.04, 0.12]; cross z median -1.60 [-1.74, -1.43], abs z median 1.60 [1.43, 1.74], abs z > 2 fraction 0.33 [0.25, 0.42]. Effective training points on a component (weighted median): own run 0.9 [0.8, 0.9], covering other runs 0.4 [0.4, 0.5]. Checks: own-run max abs z 5.07 (weighted offset abs z 1.99), Jacobian max abs z 0.00 (offset abs z 0.00), raw consistency 1.4e-14, flagged cells 0.

### M = 16 (10 cells)

| Estimator | bias | bias with the arm's entropy | covered | mc_se | gp_sd (indep / corr) |
|---|---:|---:|---:|---:|---:|
| raw | 0.273 [0.239, 0.460] | 0.274 [0.224, 0.439] | - | - | - |
| capped_I | -1.753 [-1.869, -1.493] | -1.783 [-1.897, -1.526] | - | - | - |
| capped_E | -1.104 [-1.387, -0.947] | -1.129 [-1.414, -0.951] | - | - | - |
| honest ratio1.5 median | -1.379 [-1.447, -1.179] | -1.385 [-1.475, -1.186] | 0.998 | 0.025 | 0.103 / 0.275 |
| honest ratio1.5 precision | -1.470 [-1.547, -1.378] | -1.483 [-1.580, -1.376] | 0.998 | 0.021 | 0.026 / 0.210 |
| honest ratio1.5 mean | -1.411 [-1.492, -1.306] | -1.423 [-1.523, -1.298] | 0.998 | 0.022 | 0.027 / 0.214 |
| honest ratio2 median | -1.397 [-1.452, -1.186] | -1.404 [-1.479, -1.193] | 1.000 | 0.025 | 0.103 / 0.276 |
| honest ratio2 precision | -1.485 [-1.550, -1.386] | -1.498 [-1.583, -1.377] | 1.000 | 0.021 | 0.026 / 0.210 |
| honest ratio2 mean | -1.440 [-1.504, -1.318] | -1.452 [-1.535, -1.302] | 1.000 | 0.022 | 0.027 / 0.218 |
| honest ratio3 median | -1.398 [-1.453, -1.186] | -1.405 [-1.480, -1.193] | 1.000 | 0.025 | 0.103 / 0.276 |
| honest ratio3 precision | -1.487 [-1.550, -1.386] | -1.499 [-1.583, -1.377] | 1.000 | 0.021 | 0.026 / 0.210 |
| honest ratio3 mean | -1.443 [-1.509, -1.317] | -1.455 [-1.536, -1.303] | 1.000 | 0.022 | 0.027 / 0.218 |
| honest ratio5 median | -1.398 [-1.453, -1.186] | -1.405 [-1.480, -1.193] | 1.000 | 0.025 | 0.103 / 0.276 |
| honest ratio5 precision | -1.487 [-1.550, -1.386] | -1.499 [-1.583, -1.377] | 1.000 | 0.021 | 0.026 / 0.210 |
| honest ratio5 mean | -1.443 [-1.509, -1.317] | -1.455 [-1.536, -1.303] | 1.000 | 0.022 | 0.027 / 0.218 |
| honest cap_only median | -1.398 [-1.453, -1.186] | -1.405 [-1.480, -1.193] | 1.000 | 0.025 | 0.103 / 0.276 |
| honest cap_only precision | -1.487 [-1.550, -1.386] | -1.499 [-1.583, -1.377] | 1.000 | 0.021 | 0.026 / 0.210 |
| honest cap_only mean | -1.443 [-1.509, -1.317] | -1.455 [-1.536, -1.303] | 1.000 | 0.022 | 0.027 / 0.218 |
| honest none median | -1.398 [-1.453, -1.186] | -1.405 [-1.480, -1.193] | 1.000 | 0.025 | 0.103 / 0.276 |
| honest none precision | -1.487 [-1.550, -1.386] | -1.499 [-1.583, -1.377] | 1.000 | 0.021 | 0.026 / 0.210 |
| honest none mean | -1.443 [-1.510, -1.317] | -1.455 [-1.537, -1.303] | 1.000 | 0.022 | 0.027 / 0.218 |
| pooled (ratio2, own included) | -1.337 [-1.396, -1.132] | - | - | - | - |
| truth_strat | 0.017 [-0.010, 0.042] | - | - | - | - |

Reference `e_log_joint_mc_sd` median 0.028; the headline bias is within 2 SE in 0% of the cells; median absolute bias raw 0.273, capped_I 1.753, capped_E 1.104, honest 1.397 (against raw the honest estimate is **worse**, against capped_I better; tie band 0.028, one reference standard error). Calibration of the self-reported SDs (weighted over components): own z median 0.69 [0.58, 1.09], abs z median 0.96 [0.83, 1.16], abs z > 2 fraction 0.09 [0.05, 0.12]; cross z median -1.58 [-1.76, -1.44], abs z median 1.58 [1.44, 1.76], abs z > 2 fraction 0.34 [0.30, 0.40]. Effective training points on a component (weighted median): own run 0.9 [0.7, 1.0], covering other runs 0.4 [0.3, 0.4]. Checks: own-run max abs z 3.98 (weighted offset abs z 1.43), Jacobian max abs z 0.00 (offset abs z 0.00), raw consistency 0.0e+00, flagged cells 0.

| Run | own error | se | own z median | own abs z median | BQ sd median | pred sd median | own-run max abs z (offset z) | Jacobian max abs z (offset z) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| student_D8_noise3_svbmc_seed1000 | -0.678 | 0.075 | -1.29 | 1.27 | 0.328 | 0.616 | 2.14 (-0.35) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1001 | -0.769 | 0.095 | -1.78 | 1.76 | 0.513 | 0.930 | 2.36 (+1.45) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1002 | -0.683 | 0.061 | -1.20 | 1.18 | 0.323 | 0.602 | 2.62 (-0.19) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1003 | 0.305 | 0.073 | 0.94 | 1.01 | 0.378 | 0.809 | 2.79 (-0.46) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1004 | 0.031 | 0.059 | 0.06 | 0.52 | 0.406 | 0.777 | 3.06 (-1.42) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1005 | -0.221 | 0.055 | -0.41 | 0.76 | 0.414 | 0.812 | 3.12 (+0.76) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1006 | -1.048 | 0.062 | -2.10 | 2.03 | 0.397 | 0.835 | 2.45 (+0.92) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1007 | -0.166 | 0.068 | -0.67 | 1.01 | 0.401 | 0.773 | 2.11 (-1.22) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1008 | -0.635 | 0.056 | -1.37 | 1.36 | 0.409 | 0.858 | 2.50 (-0.92) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1009 | 0.211 | 0.055 | 0.57 | 0.63 | 0.409 | 0.859 | 2.67 (+0.01) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1010 | -0.112 | 0.049 | -0.17 | 0.58 | 0.403 | 0.847 | 2.85 (+0.42) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1011 | 0.132 | 0.064 | 0.27 | 0.35 | 0.419 | 1.026 | 2.76 (-0.99) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1012 | -0.634 | 0.073 | -1.36 | 1.31 | 0.405 | 0.883 | 2.53 (+0.08) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1013 | 0.446 | 0.051 | 1.15 | 1.15 | 0.389 | 0.854 | 3.21 (-0.87) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1014 | -0.607 | 0.061 | -1.42 | 1.34 | 0.408 | 0.784 | 2.78 (+0.74) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1015 | 0.002 | 0.047 | -0.04 | 0.76 | 0.412 | 0.821 | 2.52 (+1.50) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1016 | -0.223 | 0.089 | -0.80 | 0.77 | 0.476 | 0.968 | 1.95 (-0.06) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1017 | 0.037 | 0.079 | 0.16 | 0.39 | 0.389 | 0.730 | 2.71 (-0.76) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1018 | 0.239 | 0.072 | 0.55 | 0.55 | 0.425 | 0.855 | 2.39 (-0.75) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1019 | -0.467 | 0.064 | -1.29 | 1.26 | 0.431 | 0.829 | 2.37 (+0.44) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1020 | -0.462 | 0.064 | -1.20 | 1.17 | 0.425 | 0.843 | 2.64 (+2.36) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1021 | -0.544 | 0.063 | -1.21 | 1.14 | 0.372 | 0.711 | 1.74 (-0.03) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1022 | 0.040 | 0.041 | -0.02 | 0.43 | 0.432 | 0.870 | 2.06 (+0.14) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1023 | 0.074 | 0.060 | 0.23 | 0.64 | 0.403 | 0.796 | 2.46 (-2.51) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1024 | -0.140 | 0.069 | -0.13 | 0.32 | 0.501 | 0.921 | 2.99 (+0.03) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1025 | -0.345 | 0.053 | -0.82 | 0.86 | 0.443 | 0.845 | 3.22 (+1.15) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1026 | -0.593 | 0.082 | -1.27 | 1.25 | 0.408 | 0.930 | 2.38 (-0.61) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1027 | -0.175 | 0.071 | -0.70 | 0.70 | 0.403 | 0.817 | 2.05 (-0.10) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1028 | -0.041 | 0.063 | -0.13 | 0.66 | 0.391 | 0.875 | 3.12 (+0.80) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1029 | -0.148 | 0.081 | -0.72 | 1.26 | 0.430 | 0.876 | 2.43 (-1.45) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1030 | -0.175 | 0.045 | -0.42 | 0.58 | 0.411 | 0.890 | 2.39 (-2.03) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1031 | -0.112 | 0.054 | -0.13 | 0.48 | 0.446 | 0.977 | 4.09 (+1.38) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1032 | -0.079 | 0.057 | -0.13 | 0.51 | 0.458 | 0.862 | 2.91 (+0.23) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1033 | 0.301 | 0.054 | 0.33 | 0.44 | 0.426 | 0.959 | 2.41 (+0.67) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1034 | 0.012 | 0.057 | 0.03 | 0.47 | 0.417 | 0.827 | 1.94 (-1.07) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1035 | -0.524 | 0.063 | -1.36 | 1.35 | 0.388 | 0.683 | 2.22 (+2.27) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1036 | -0.148 | 0.070 | -0.20 | 0.54 | 0.389 | 0.824 | 2.77 (-0.37) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1037 | -0.320 | 0.071 | -0.74 | 0.74 | 0.510 | 0.947 | 3.45 (-0.45) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1038 | 0.388 | 0.062 | 1.00 | 1.03 | 0.422 | 1.071 | 2.45 (-1.78) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1039 | -0.098 | 0.075 | -0.37 | 0.37 | 0.394 | 0.825 | 2.18 (-1.00) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1040 | -0.397 | 0.066 | -1.04 | 1.27 | 0.373 | 0.705 | 3.26 (-0.55) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1041 | -0.119 | 0.045 | -0.60 | 0.57 | 0.387 | 0.823 | 2.94 (+1.72) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1042 | -0.524 | 0.063 | -1.07 | 1.03 | 0.410 | 0.853 | 1.95 (-0.53) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1043 | -0.602 | 0.062 | -1.23 | 1.20 | 0.399 | 0.799 | 5.27 (+1.43) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1044 | -1.027 | 0.072 | -2.14 | 2.08 | 0.460 | 0.905 | 2.22 (-0.40) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1045 | -0.033 | 0.062 | -0.13 | 0.31 | 0.453 | 0.885 | 2.18 (-0.04) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1046 | 0.211 | 0.059 | 0.39 | 0.63 | 0.401 | 0.808 | 2.79 (-1.33) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1047 | -0.641 | 0.072 | -1.76 | 1.61 | 0.344 | 0.624 | 2.06 (+0.77) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1048 | 0.462 | 0.061 | 1.04 | 1.04 | 0.536 | 1.085 | 2.38 (+0.71) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1049 | -0.659 | 0.062 | -1.50 | 1.50 | 0.432 | 0.952 | 2.77 (-0.49) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1050 | -0.173 | 0.047 | -0.48 | 0.72 | 0.397 | 0.824 | 2.42 (-1.07) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1051 | -0.029 | 0.047 | -0.04 | 0.57 | 0.428 | 0.833 | 2.36 (-1.44) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1052 | -0.764 | 0.060 | -1.49 | 1.46 | 0.435 | 0.849 | 2.74 (-0.11) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1053 | 0.128 | 0.066 | 0.19 | 0.75 | 0.455 | 0.873 | 2.57 (+1.46) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1054 | -0.200 | 0.116 | -0.16 | 0.16 | 0.413 | 0.841 | 3.22 (-1.36) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1055 | -0.129 | 0.078 | -0.57 | 0.65 | 0.413 | 0.865 | 4.15 (+1.19) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1056 | -0.514 | 0.052 | -1.35 | 1.26 | 0.401 | 0.794 | 2.03 (+0.55) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1057 | -0.279 | 0.049 | -0.69 | 1.01 | 0.405 | 0.865 | 2.18 (+0.88) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1058 | -1.025 | 0.059 | -2.54 | 2.42 | 0.400 | 0.810 | 2.99 (+2.59) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1059 | -0.256 | 0.046 | -0.54 | 0.68 | 0.417 | 0.941 | 3.03 (-0.83) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1060 | -0.430 | 0.071 | -1.35 | 1.30 | 0.411 | 0.753 | 2.38 (-1.01) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1061 | -0.331 | 0.074 | -0.70 | 0.79 | 0.399 | 0.861 | 2.74 (-1.04) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1062 | -0.417 | 0.059 | -0.95 | 0.92 | 0.398 | 0.778 | 1.97 (-0.30) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1063 | -0.247 | 0.056 | -0.64 | 1.12 | 0.422 | 0.845 | 2.69 (+0.16) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1064 | -0.035 | 0.063 | -0.46 | 0.82 | 0.431 | 0.902 | 2.42 (-1.21) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1065 | -0.363 | 0.059 | -0.99 | 0.96 | 0.411 | 0.813 | 4.33 (+1.71) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1066 | -0.220 | 0.055 | -0.62 | 0.83 | 0.431 | 0.873 | 1.95 (+1.64) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1067 | 0.571 | 0.056 | 1.29 | 1.29 | 0.415 | 0.849 | 2.85 (-0.09) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1068 | -0.899 | 0.074 | -2.25 | 2.19 | 0.358 | 0.715 | 2.91 (-0.51) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1069 | -0.135 | 0.050 | -0.24 | 0.42 | 0.391 | 0.840 | 2.11 (-0.69) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1070 | 0.101 | 0.048 | 0.30 | 0.86 | 0.447 | 0.936 | 2.31 (-0.34) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1071 | -0.522 | 0.057 | -1.14 | 1.06 | 0.418 | 0.850 | 2.75 (-0.10) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1072 | -0.030 | 0.057 | -0.23 | 0.74 | 0.362 | 0.750 | 2.77 (+1.41) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1073 | -0.206 | 0.068 | -0.63 | 0.76 | 0.391 | 0.745 | 3.09 (+0.39) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1074 | 0.646 | 0.051 | 1.44 | 1.44 | 0.409 | 0.848 | 2.12 (-0.41) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1075 | -0.170 | 0.061 | -0.31 | 0.70 | 0.388 | 0.808 | 3.03 (+0.28) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1076 | -0.296 | 0.055 | -0.78 | 0.91 | 0.446 | 1.038 | 3.48 (+1.04) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1077 | -0.692 | 0.058 | -1.68 | 1.58 | 0.338 | 0.648 | 3.42 (+0.38) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1078 | -0.579 | 0.059 | -1.11 | 1.07 | 0.416 | 0.887 | 2.44 (-0.86) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1079 | -0.532 | 0.054 | -1.04 | 0.99 | 0.461 | 1.066 | 2.41 (+3.08) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1080 | 0.057 | 0.056 | 0.25 | 0.75 | 0.430 | 0.869 | 2.90 (+0.11) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1081 | -0.501 | 0.070 | -1.17 | 1.19 | 0.491 | 0.945 | 2.08 (+0.79) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1082 | -0.047 | 0.047 | -0.35 | 0.45 | 0.442 | 0.998 | 2.69 (+1.15) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1083 | -0.178 | 0.053 | -0.43 | 0.48 | 0.458 | 0.928 | 2.26 (+1.16) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1084 | 0.023 | 0.051 | -0.11 | 0.43 | 0.386 | 0.728 | 3.26 (+0.37) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1085 | 0.418 | 0.049 | 0.64 | 0.64 | 0.432 | 0.940 | 2.70 (+0.97) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1086 | -0.041 | 0.041 | -0.28 | 0.43 | 0.389 | 0.865 | 2.80 (+0.39) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1087 | -0.537 | 0.073 | -0.84 | 1.18 | 0.429 | 0.851 | 2.77 (-1.74) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1088 | -0.312 | 0.056 | -0.86 | 0.82 | 0.459 | 0.974 | 2.32 (-0.25) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1089 | 0.004 | 0.089 | -0.23 | 0.35 | 0.478 | 0.947 | 2.95 (+0.08) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1090 | -0.437 | 0.047 | -1.16 | 1.10 | 0.451 | 0.931 | 1.98 (+1.38) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1091 | -0.633 | 0.067 | -1.25 | 1.24 | 0.411 | 0.768 | 2.74 (+0.40) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1092 | -0.665 | 0.062 | -1.38 | 1.33 | 0.469 | 0.915 | 2.44 (+0.60) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1093 | 0.110 | 0.056 | -0.11 | 0.65 | 0.571 | 1.054 | 2.74 (+1.45) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1094 | 0.432 | 0.045 | 0.95 | 0.97 | 0.399 | 0.972 | 2.67 (+0.64) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1095 | -0.712 | 0.054 | -1.48 | 1.46 | 0.360 | 0.709 | 1.84 (-1.74) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1096 | -0.943 | 0.058 | -2.02 | 1.99 | 0.398 | 0.723 | 2.09 (+0.84) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1097 | 0.282 | 0.052 | 0.61 | 0.67 | 0.392 | 0.774 | 2.20 (-0.67) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1098 | -0.420 | 0.051 | -1.10 | 1.09 | 0.346 | 0.733 | 2.99 (-1.36) | 0.00 (+0.00) |
| student_D8_noise3_svbmc_seed1099 | -0.216 | 0.060 | -0.41 | 0.44 | 0.407 | 0.963 | 1.96 (-0.70) | 0.00 (+0.00) |
