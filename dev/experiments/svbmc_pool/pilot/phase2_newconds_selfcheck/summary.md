# Own-run checks of pool artifacts (`--self-check`)

Generated 2026-09-14 14:18:33 from the artifacts of the pool alone, 100 draws per component, seed 0. For every run the own-run Monte Carlo estimate of each component's expected log joint is compared with the stored, Jacobian-corrected `I_corr` and the Monte Carlo log-Jacobian with the deterministic one (`abs z` per component and the weighted offset in standard errors; a run is flagged beyond 6 for a component or 4 for the offset); `own error` is the run's own expected log joint against the target's true one over its draws, weighted by the run's own weights; `BQ sd` and `pred sd` are the medians over components of the run's quadrature SD and of its GP's mean predictive SD on the component.

## multisensory_s1_D6_svbmc (noiseless)
| Run | own error | se | own z median | own abs z median | BQ sd median | pred sd median | own-run max abs z (offset z) | Jacobian max abs z (offset z) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| multisensory_s1_D6_svbmc_seed1000 | -0.017 | 0.031 | -0.37 | 0.99 | 0.066 | 0.198 | 2.96 (+0.34) | 3.26 (-0.23) |

## student_D8_noise3_svbmc (noisy)
| Run | own error | se | own z median | own abs z median | BQ sd median | pred sd median | own-run max abs z (offset z) | Jacobian max abs z (offset z) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| student_D8_noise3_svbmc_seed1000 | 0.316 | 0.057 | 0.44 | 0.45 | 0.442 | 0.841 | 2.65 (-1.41) | 0.00 (+0.00) |
