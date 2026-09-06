# Null check (even vs odd seeds) on item7_20260906

| config | metric | n ref | n new | KS | p | Holm | median shift |
|---|---|---|---|---|---|---|---|
| banana_D10 | elbo_err | 10 | 10 | 0.300 | 0.787 | ok | -0.00051 |
| banana_D10 | gskl | 10 | 10 | 0.300 | 0.787 | ok | +0.0399 |
| banana_D10 | mmtv | 10 | 10 | 0.200 | 0.994 | ok | +0.000292 |
| banana_D10 | func_count | 10 | 10 | 0.200 | 0.994 | ok | -2.5 |
| banana_D2 | elbo_err | 10 | 10 | 0.400 | 0.418 | ok | +0.0124 |
| banana_D2 | gskl | 10 | 10 | 0.400 | 0.418 | ok | +0.0254 |
| banana_D2 | mmtv | 10 | 10 | 0.400 | 0.418 | ok | +0.00935 |
| banana_D2 | func_count | 10 | 10 | 0.300 | 0.787 | ok | -12.5 |
| banana_D6 | elbo_err | 10 | 10 | 0.200 | 0.994 | ok | +0.00299 |
| banana_D6 | gskl | 10 | 10 | 0.400 | 0.418 | ok | +0.026 |
| banana_D6 | mmtv | 10 | 10 | 0.400 | 0.418 | ok | +0.00293 |
| banana_D6 | func_count | 10 | 10 | 0.300 | 0.787 | ok | -2.5 |
| cigar_D4 | elbo_err | 10 | 10 | 0.300 | 0.787 | ok | +0.00169 |
| cigar_D4 | gskl | 10 | 10 | 0.300 | 0.787 | ok | +0.00121 |
| cigar_D4 | mmtv | 10 | 10 | 0.200 | 0.994 | ok | -0.0015 |
| cigar_D4 | func_count | 10 | 10 | 0.500 | 0.168 | ok | +10 |
| corr_D5 | elbo_err | 10 | 10 | 0.200 | 0.994 | ok | -0.0026 |
| corr_D5 | gskl | 10 | 10 | 0.300 | 0.787 | ok | -0.000519 |
| corr_D5 | mmtv | 10 | 10 | 0.300 | 0.787 | ok | -0.00107 |
| corr_D5 | func_count | 10 | 10 | 0.300 | 0.787 | ok | +0 |
| halfnormal_D2 | elbo_err | 10 | 10 | 0.300 | 0.787 | ok | +0.00158 |
| halfnormal_D2 | gskl | 10 | 10 | 0.500 | 0.168 | ok | -0.000152 |
| halfnormal_D2 | mmtv | 10 | 10 | 0.400 | 0.418 | ok | -0.00167 |
| halfnormal_D2 | func_count | 10 | 10 | 0.200 | 0.994 | ok | +0 |
| logreg_D5 | elbo_err | 10 | 10 | 0.200 | 0.994 | ok | -0.0148 |
| logreg_D5 | gskl | 10 | 10 | 0.200 | 0.994 | ok | +0.000172 |
| logreg_D5 | mmtv | 10 | 10 | 0.300 | 0.787 | ok | -0.000359 |
| logreg_D5 | func_count | 10 | 10 | 0.100 | 1 | ok | +0 |
| logreg_D5_noise3 | elbo_err | 10 | 10 | 0.300 | 0.787 | ok | -0.0481 |
| logreg_D5_noise3 | gskl | 10 | 10 | 0.300 | 0.787 | ok | -0.0586 |
| logreg_D5_noise3 | mmtv | 10 | 10 | 0.300 | 0.787 | ok | -0.00994 |
| logreg_D5_noise3 | func_count | 10 | 10 | 0.200 | 0.994 | ok | -2.5 |
| lumpy_D10 | elbo_err | 10 | 10 | 0.500 | 0.168 | ok | -0.0798 |
| lumpy_D10 | gskl | 10 | 10 | 0.400 | 0.418 | ok | -0.0842 |
| lumpy_D10 | mmtv | 10 | 10 | 0.400 | 0.418 | ok | -0.00492 |
| lumpy_D10 | func_count | 10 | 10 | 0.300 | 0.787 | ok | +25 |
| lumpy_D4 | elbo_err | 10 | 10 | 0.300 | 0.787 | ok | +0.017 |
| lumpy_D4 | gskl | 10 | 10 | 0.400 | 0.418 | ok | +0.00186 |
| lumpy_D4 | mmtv | 10 | 10 | 0.300 | 0.787 | ok | +0.00144 |
| lumpy_D4 | func_count | 10 | 10 | 0.500 | 0.168 | ok | +12.5 |
| normal_D5 | elbo_err | 10 | 10 | 0.300 | 0.787 | ok | +0.00042 |
| normal_D5 | gskl | 10 | 10 | 0.400 | 0.418 | ok | +1.89e-05 |
| normal_D5 | mmtv | 10 | 10 | 0.300 | 0.787 | ok | +9.17e-05 |
| normal_D5 | func_count | 10 | 10 | 0.100 | 1 | ok | +0 |
| rosenbrock_D2 | elbo_err | 10 | 10 | 0.300 | 0.787 | ok | -0.00485 |
| rosenbrock_D2 | gskl | 10 | 10 | 0.300 | 0.787 | ok | -0.00843 |
| rosenbrock_D2 | mmtv | 10 | 10 | 0.300 | 0.787 | ok | -0.00406 |
| rosenbrock_D2 | func_count | 10 | 10 | 0.300 | 0.787 | ok | -2.5 |
| rosenbrock_D2_noise1 | elbo_err | 10 | 10 | 0.400 | 0.418 | ok | +0.0654 |
| rosenbrock_D2_noise1 | gskl | 10 | 10 | 0.200 | 0.994 | ok | +0.00234 |
| rosenbrock_D2_noise1 | mmtv | 10 | 10 | 0.300 | 0.787 | ok | -0.00553 |
| rosenbrock_D2_noise1 | func_count | 10 | 10 | 0.200 | 0.994 | ok | -5 |
| student_D4 | elbo_err | 10 | 10 | 0.400 | 0.418 | ok | -0.0126 |
| student_D4 | gskl | 10 | 10 | 0.500 | 0.168 | ok | -0.0302 |
| student_D4 | mmtv | 10 | 10 | 0.400 | 0.418 | ok | -0.00857 |
| student_D4 | func_count | 10 | 10 | 0.300 | 0.787 | ok | +12.5 |

| config | median func_count ratio (new/ref), descriptive |
|---|---|
| banana_D10 | 0.980 |
| banana_D2 | 0.868 |
| banana_D6 | 0.977 |
| cigar_D4 | 1.077 |
| corr_D5 | 1.000 |
| halfnormal_D2 | 1.000 |
| logreg_D5 | 1.000 |
| logreg_D5_noise3 | 0.990 |
| lumpy_D10 | 1.086 |
| lumpy_D4 | 1.147 |
| normal_D5 | 1.000 |
| rosenbrock_D2 | 0.970 |
| rosenbrock_D2_noise1 | 0.962 |
| student_D4 | 1.119 |

**no config flagged (56 KS tests, Holm alpha 0.05)**
