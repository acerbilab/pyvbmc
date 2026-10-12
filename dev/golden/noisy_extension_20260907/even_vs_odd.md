# Null check (even vs odd seeds) on reference_870_20260907

| config | metric | n ref | n new | KS | p | Holm | median shift |
|---|---|---|---|---|---|---|---|
| banana_D10 | elbo_err | 25 | 25 | 0.200 | 0.71 | ok | +0.0243 |
| banana_D10 | gskl | 25 | 25 | 0.200 | 0.71 | ok | +0.0584 |
| banana_D10 | mmtv | 25 | 25 | 0.160 | 0.915 | ok | +0.000174 |
| banana_D10 | func_count | 25 | 25 | 0.160 | 0.915 | ok | +0 |
| banana_D2 | elbo_err | 25 | 25 | 0.280 | 0.285 | ok | +0.00772 |
| banana_D2 | gskl | 25 | 25 | 0.280 | 0.285 | ok | +0.043 |
| banana_D2 | mmtv | 25 | 25 | 0.320 | 0.156 | ok | +0.00855 |
| banana_D2 | func_count | 25 | 25 | 0.120 | 0.996 | ok | +0 |
| banana_D6 | elbo_err | 25 | 25 | 0.160 | 0.915 | ok | -0.00151 |
| banana_D6 | gskl | 25 | 25 | 0.200 | 0.71 | ok | -0.00962 |
| banana_D6 | mmtv | 25 | 25 | 0.200 | 0.71 | ok | +0.000128 |
| banana_D6 | func_count | 25 | 25 | 0.120 | 0.996 | ok | +0 |
| cigar_D15_exhaust | elbo_err | 5 | 5 | 0.400 | 0.873 | ok | -0.00123 |
| cigar_D15_exhaust | gskl | 5 | 5 | 0.600 | 0.357 | ok | -0.0014 |
| cigar_D15_exhaust | mmtv | 5 | 5 | 0.800 | 0.0794 | ok | -0.00416 |
| cigar_D15_exhaust | func_count | 5 | 5 | 0.000 | 1 | ok | +0 |
| cigar_D4 | elbo_err | 25 | 25 | 0.200 | 0.71 | ok | -0.00203 |
| cigar_D4 | gskl | 25 | 25 | 0.200 | 0.71 | ok | +3.62e-05 |
| cigar_D4 | mmtv | 25 | 25 | 0.200 | 0.71 | ok | -0.00118 |
| cigar_D4 | func_count | 25 | 25 | 0.120 | 0.996 | ok | +0 |
| cigar_D8 | elbo_err | 25 | 25 | 0.200 | 0.71 | ok | +0.00292 |
| cigar_D8 | gskl | 25 | 25 | 0.200 | 0.71 | ok | +0.000273 |
| cigar_D8 | mmtv | 25 | 25 | 0.280 | 0.285 | ok | +0.0015 |
| cigar_D8 | func_count | 25 | 25 | 0.160 | 0.915 | ok | -5 |
| corr_D5 | elbo_err | 25 | 25 | 0.160 | 0.915 | ok | -0.00206 |
| corr_D5 | gskl | 25 | 25 | 0.280 | 0.285 | ok | -0.000576 |
| corr_D5 | mmtv | 25 | 25 | 0.240 | 0.475 | ok | -0.000784 |
| corr_D5 | func_count | 25 | 25 | 0.120 | 0.996 | ok | +0 |
| halfnormal_D2 | elbo_err | 25 | 25 | 0.120 | 0.996 | ok | +0.00131 |
| halfnormal_D2 | gskl | 25 | 25 | 0.160 | 0.915 | ok | -1.78e-05 |
| halfnormal_D2 | mmtv | 25 | 25 | 0.280 | 0.285 | ok | -0.00106 |
| halfnormal_D2 | func_count | 25 | 25 | 0.160 | 0.915 | ok | +0 |
| logreg_D5 | elbo_err | 25 | 25 | 0.280 | 0.285 | ok | +0.00532 |
| logreg_D5 | gskl | 25 | 25 | 0.160 | 0.915 | ok | -0.00015 |
| logreg_D5 | mmtv | 25 | 25 | 0.200 | 0.71 | ok | -4.72e-05 |
| logreg_D5 | func_count | 25 | 25 | 0.200 | 0.71 | ok | -5 |
| logreg_D5_noise3 | elbo_err | 25 | 25 | 0.240 | 0.475 | ok | -0.0533 |
| logreg_D5_noise3 | gskl | 25 | 25 | 0.240 | 0.475 | ok | +0.0199 |
| logreg_D5_noise3 | mmtv | 25 | 25 | 0.240 | 0.475 | ok | -0.0097 |
| logreg_D5_noise3 | func_count | 25 | 25 | 0.120 | 0.996 | ok | -5 |
| lumpy_D10 | elbo_err | 25 | 25 | 0.240 | 0.475 | ok | -0.0496 |
| lumpy_D10 | gskl | 25 | 25 | 0.320 | 0.156 | ok | -0.0948 |
| lumpy_D10 | mmtv | 25 | 25 | 0.280 | 0.285 | ok | -0.0031 |
| lumpy_D10 | func_count | 25 | 25 | 0.080 | 1 | ok | -25 |
| lumpy_D4 | elbo_err | 25 | 25 | 0.160 | 0.915 | ok | -0.0101 |
| lumpy_D4 | gskl | 25 | 25 | 0.280 | 0.285 | ok | -0.0028 |
| lumpy_D4 | mmtv | 25 | 25 | 0.200 | 0.71 | ok | -0.000702 |
| lumpy_D4 | func_count | 25 | 25 | 0.400 | 0.0356 | ok | +5 |
| normal_D5 | elbo_err | 25 | 25 | 0.240 | 0.475 | ok | -0.000111 |
| normal_D5 | gskl | 25 | 25 | 0.120 | 0.996 | ok | -9.5e-06 |
| normal_D5 | mmtv | 25 | 25 | 0.280 | 0.285 | ok | +0.000287 |
| normal_D5 | func_count | 25 | 25 | 0.040 | 1 | ok | +0 |
| rosenbrock_D2 | elbo_err | 25 | 25 | 0.200 | 0.71 | ok | -0.000739 |
| rosenbrock_D2 | gskl | 25 | 25 | 0.200 | 0.71 | ok | +0.00416 |
| rosenbrock_D2 | mmtv | 25 | 25 | 0.200 | 0.71 | ok | +0.00357 |
| rosenbrock_D2 | func_count | 25 | 25 | 0.200 | 0.71 | ok | +0 |
| rosenbrock_D2_noise1 | elbo_err | 25 | 25 | 0.280 | 0.285 | ok | +0.0311 |
| rosenbrock_D2_noise1 | gskl | 25 | 25 | 0.080 | 1 | ok | +0.000876 |
| rosenbrock_D2_noise1 | mmtv | 25 | 25 | 0.200 | 0.71 | ok | -0.000836 |
| rosenbrock_D2_noise1 | func_count | 25 | 25 | 0.160 | 0.915 | ok | +0 |
| rosenbrock_D2_noise3 | elbo_err | 15 | 15 | 0.267 | 0.678 | ok | -0.0133 |
| rosenbrock_D2_noise3 | gskl | 15 | 15 | 0.267 | 0.678 | ok | +0.0375 |
| rosenbrock_D2_noise3 | mmtv | 15 | 15 | 0.200 | 0.938 | ok | -0.00259 |
| rosenbrock_D2_noise3 | func_count | 15 | 15 | 0.400 | 0.184 | ok | +15 |
| student_D4 | elbo_err | 25 | 25 | 0.280 | 0.285 | ok | -0.00549 |
| student_D4 | gskl | 25 | 25 | 0.320 | 0.156 | ok | -0.00963 |
| student_D4 | mmtv | 25 | 25 | 0.320 | 0.156 | ok | -0.00547 |
| student_D4 | func_count | 25 | 25 | 0.200 | 0.71 | ok | +5 |
| student_D8 | elbo_err | 25 | 25 | 0.120 | 0.996 | ok | -0.00465 |
| student_D8 | gskl | 25 | 25 | 0.120 | 0.996 | ok | -0.0184 |
| student_D8 | mmtv | 25 | 25 | 0.200 | 0.71 | ok | -0.00151 |
| student_D8 | func_count | 25 | 25 | 0.200 | 0.71 | ok | -25 |
| student_D8_noise3 | elbo_err | 15 | 15 | 0.200 | 0.938 | ok | +0.171 |
| student_D8_noise3 | gskl | 15 | 15 | 0.133 | 1 | ok | +0.0713 |
| student_D8_noise3 | mmtv | 15 | 15 | 0.200 | 0.938 | ok | -0.00669 |
| student_D8_noise3 | func_count | 15 | 15 | 0.400 | 0.184 | ok | +20 |

| config | median func_count ratio (new/ref), descriptive |
|---|---|
| banana_D10 | 1.000 |
| banana_D2 | 1.000 |
| banana_D6 | 1.000 |
| cigar_D15_exhaust | 1.000 |
| cigar_D4 | 1.000 |
| cigar_D8 | 0.971 |
| corr_D5 | 1.000 |
| halfnormal_D2 | 1.000 |
| logreg_D5 | 0.964 |
| logreg_D5_noise3 | 0.980 |
| lumpy_D10 | 0.918 |
| lumpy_D4 | 1.056 |
| normal_D5 | 1.000 |
| rosenbrock_D2 | 1.000 |
| rosenbrock_D2_noise1 | 1.000 |
| rosenbrock_D2_noise3 | 1.086 |
| student_D4 | 1.050 |
| student_D8 | 0.878 |
| student_D8_noise3 | 1.062 |

**no config flagged (76 KS tests, Holm alpha 0.05)**
