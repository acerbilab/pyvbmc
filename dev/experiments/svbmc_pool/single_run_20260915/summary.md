Single VBMC runs scored against their own Monte Carlo ELBO (`elbo_mc`, the target's noiseless log density averaged over draws from the run's posterior plus the mixture's entropy by the comparison's estimator). `bias_vbmc` is the ELBO the run reports minus `elbo_mc`; `bias_raw` the integrated class's raw value for the run alone at its own weights; `bias_cap` the component-median cap at those weights. Medians over the filtered runs with a bootstrap 95 % interval, then the mean and the quartiles.

| condition | n | bias_vbmc, median [CI] | mean [q25, q75] | bias_raw, median | bias_cap, median | KL gap, median | elbo_sd, mean |
|---|---|---|---|---|---|---|---|
| multisensory_s1_D6_noise3_svbmc | 100 | +0.74 [+0.64, +0.80] | +0.73 [+0.50, +0.94] | +0.74 [+0.65, +0.81] | +0.43 [+0.37, +0.53] | +1.05 [+1.02, +1.09] | 0.35 |
| multisensory_s1_D6_noise1.3_svbmc | 100 | +0.37 [+0.32, +0.42] | +0.35 [+0.21, +0.50] | +0.35 [+0.31, +0.42] | +0.12 [+0.05, +0.20] | +0.75 [+0.71, +0.78] | 0.19 |
| rosenbrock_D2_noise3_svbmc | 100 | +0.13 [+0.05, +0.19] | +0.13 [-0.05, +0.27] | +0.13 [+0.05, +0.19] | +0.06 [-0.01, +0.14] | +0.08 [+0.06, +0.12] | 0.29 |
| gmm_D2_noise3_svbmc | 100 | +0.17 [+0.09, +0.23] | +0.18 [-0.01, +0.36] | +0.18 [+0.11, +0.22] | +0.04 [-0.02, +0.12] | +1.15 [+1.04, +1.48] | 0.31 |
| ring_D2_noise3_svbmc | 100 | +0.26 [+0.21, +0.35] | +0.28 [+0.07, +0.43] | +0.26 [+0.20, +0.34] | +0.11 [+0.07, +0.18] | +1.80 [+1.53, +1.85] | 0.26 |
| student_D8_noise3_svbmc | 100 | -0.19 [-0.34, -0.14] | -0.23 [-0.50, -0.01] | -0.21 [-0.32, -0.15] | -0.64 [-0.75, -0.53] | +0.75 [+0.69, +0.81] | 0.31 |
| gmm_D2_svbmc | 50 | +0.01 [+0.00, +0.01] | +0.00 [-0.00, +0.01] | +0.00 [-0.00, +0.01] | -0.03 [-0.06, -0.01] | +0.71 [+0.70, +0.73] | 0.00 |
| multisensory_s1_D6_svbmc | 50 | +0.03 [+0.01, +0.05] | +0.04 [-0.00, +0.07] | +0.04 [+0.01, +0.06] | -0.08 [-0.11, -0.02] | +0.42 [+0.37, +0.44] | 0.04 |
