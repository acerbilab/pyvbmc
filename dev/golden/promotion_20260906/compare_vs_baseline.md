# Compare baseline (ref) vs item7_20260906 (new)

| config | metric | n ref | n new | KS | p | Holm | median shift |
|---|---|---|---|---|---|---|---|
| banana_D10 | elbo_err | 20 | 20 | 0.200 | 0.832 | ok | +0.00272 |
| banana_D10 | gskl | 20 | 20 | 0.200 | 0.832 | ok | +0.00169 |
| banana_D10 | mmtv | 20 | 20 | 0.150 | 0.983 | ok | -0.000387 |
| banana_D10 | func_count | 20 | 20 | 0.150 | 0.983 | ok | +0 |
| banana_D2 | elbo_err | 20 | 20 | 0.150 | 0.983 | ok | +0.00193 |
| banana_D2 | gskl | 20 | 20 | 0.200 | 0.832 | ok | -0.0216 |
| banana_D2 | mmtv | 20 | 20 | 0.200 | 0.832 | ok | -0.000821 |
| banana_D2 | func_count | 20 | 20 | 0.300 | 0.336 | ok | +7.5 |
| banana_D6 | elbo_err | 20 | 20 | 0.150 | 0.983 | ok | -0.00886 |
| banana_D6 | gskl | 20 | 20 | 0.200 | 0.832 | ok | -0.0296 |
| banana_D6 | mmtv | 20 | 20 | 0.200 | 0.832 | ok | -0.00131 |
| banana_D6 | func_count | 20 | 20 | 0.100 | 1 | ok | +0 |
| cigar_D4 | elbo_err | 20 | 20 | 0.200 | 0.832 | ok | +0.00195 |
| cigar_D4 | gskl | 20 | 20 | 0.200 | 0.832 | ok | +0.000436 |
| cigar_D4 | mmtv | 20 | 20 | 0.150 | 0.983 | ok | +0.000155 |
| cigar_D4 | func_count | 20 | 20 | 0.100 | 1 | ok | +0 |
| corr_D5 | elbo_err | 20 | 20 | 0.200 | 0.832 | ok | -0.00136 |
| corr_D5 | gskl | 20 | 20 | 0.250 | 0.571 | ok | +0.000236 |
| corr_D5 | mmtv | 20 | 20 | 0.200 | 0.832 | ok | +0.000465 |
| corr_D5 | func_count | 20 | 20 | 0.050 | 1 | ok | +0 |
| halfnormal_D2 | elbo_err | 20 | 20 | 0.150 | 0.983 | ok | +0.000661 |
| halfnormal_D2 | gskl | 20 | 20 | 0.300 | 0.336 | ok | +9.18e-05 |
| halfnormal_D2 | mmtv | 20 | 20 | 0.250 | 0.571 | ok | +0.000926 |
| halfnormal_D2 | func_count | 20 | 20 | 0.050 | 1 | ok | +0 |
| logreg_D5 | elbo_err | 20 | 20 | 0.200 | 0.832 | ok | +0.00351 |
| logreg_D5 | gskl | 20 | 20 | 0.150 | 0.983 | ok | -0.00122 |
| logreg_D5 | mmtv | 20 | 20 | 0.200 | 0.832 | ok | -0.00279 |
| logreg_D5 | func_count | 20 | 20 | 0.150 | 0.983 | ok | +2.5 |
| logreg_D5_noise3 | elbo_err | 20 | 20 | 0.250 | 0.571 | ok | +0.0483 |
| logreg_D5_noise3 | gskl | 20 | 20 | 0.150 | 0.983 | ok | +0.00491 |
| logreg_D5_noise3 | mmtv | 20 | 20 | 0.350 | 0.175 | ok | +0.00206 |
| logreg_D5_noise3 | func_count | 20 | 20 | 0.150 | 0.983 | ok | -2.5 |
| lumpy_D10 | elbo_err | 20 | 20 | 0.250 | 0.571 | ok | -0.036 |
| lumpy_D10 | gskl | 20 | 20 | 0.250 | 0.571 | ok | -0.00699 |
| lumpy_D10 | mmtv | 20 | 20 | 0.200 | 0.832 | ok | -0.00211 |
| lumpy_D10 | func_count | 20 | 20 | 0.500 | 0.0123 | ok | +87.5 |
| lumpy_D4 | elbo_err | 20 | 20 | 0.200 | 0.832 | ok | +0.0068 |
| lumpy_D4 | gskl | 20 | 20 | 0.250 | 0.571 | ok | +0.00417 |
| lumpy_D4 | mmtv | 20 | 20 | 0.250 | 0.571 | ok | -0.00026 |
| lumpy_D4 | func_count | 20 | 20 | 0.200 | 0.832 | ok | -5 |
| normal_D5 | elbo_err | 20 | 20 | 0.300 | 0.336 | ok | +0.00131 |
| normal_D5 | gskl | 20 | 20 | 0.150 | 0.983 | ok | +1.12e-05 |
| normal_D5 | mmtv | 20 | 20 | 0.250 | 0.571 | ok | +3.05e-05 |
| normal_D5 | func_count | 20 | 20 | 0.150 | 0.983 | ok | +0 |
| rosenbrock_D2 | elbo_err | 20 | 20 | 0.150 | 0.983 | ok | +0.00131 |
| rosenbrock_D2 | gskl | 20 | 20 | 0.100 | 1 | ok | -0.00258 |
| rosenbrock_D2 | mmtv | 20 | 20 | 0.150 | 0.983 | ok | +0.000435 |
| rosenbrock_D2 | func_count | 20 | 20 | 0.100 | 1 | ok | +0 |
| rosenbrock_D2_noise1 | elbo_err | 20 | 20 | 0.300 | 0.336 | ok | +0.00719 |
| rosenbrock_D2_noise1 | gskl | 20 | 20 | 0.250 | 0.571 | ok | +0.00764 |
| rosenbrock_D2_noise1 | mmtv | 20 | 20 | 0.300 | 0.336 | ok | +0.00609 |
| rosenbrock_D2_noise1 | func_count | 20 | 20 | 0.400 | 0.0811 | ok | -12.5 |
| student_D4 | elbo_err | 20 | 20 | 0.150 | 0.983 | ok | +0.00133 |
| student_D4 | gskl | 20 | 20 | 0.150 | 0.983 | ok | -0.000273 |
| student_D4 | mmtv | 20 | 20 | 0.250 | 0.571 | ok | -0.00346 |
| student_D4 | func_count | 20 | 20 | 0.250 | 0.571 | ok | +7.5 |

| config | median func_count ratio (new/ref), descriptive |
|---|---|
| banana_D10 | 1.000 |
| banana_D2 | 1.094 |
| banana_D6 | 1.000 |
| cigar_D4 | 1.000 |
| corr_D5 | 1.000 |
| halfnormal_D2 | 1.000 |
| logreg_D5 | 1.019 |
| logreg_D5_noise3 | 0.990 |
| lumpy_D10 | 1.398 |
| lumpy_D4 | 0.947 |
| normal_D5 | 1.000 |
| rosenbrock_D2 | 1.000 |
| rosenbrock_D2_noise1 | 0.911 |
| student_D4 | 1.073 |

**no config flagged (56 KS tests, Holm alpha 0.05)**
