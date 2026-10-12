# Compare item1_20260905 (ref) vs item7_20260906 (new)

| config | metric | n ref | n new | KS | p | Holm | median shift |
|---|---|---|---|---|---|---|---|
| banana_D10 | elbo_err | 20 | 20 | 0.300 | 0.336 | ok | +0.00739 |
| banana_D10 | gskl | 20 | 20 | 0.250 | 0.571 | ok | -0.0195 |
| banana_D10 | mmtv | 20 | 20 | 0.200 | 0.832 | ok | -0.000637 |
| banana_D10 | func_count | 20 | 20 | 0.200 | 0.832 | ok | +2.5 |
| banana_D2 | elbo_err | 20 | 20 | 0.300 | 0.336 | ok | -0.0128 |
| banana_D2 | gskl | 20 | 20 | 0.200 | 0.832 | ok | -0.0419 |
| banana_D2 | mmtv | 20 | 20 | 0.250 | 0.571 | ok | -0.00497 |
| banana_D2 | func_count | 20 | 20 | 0.350 | 0.175 | ok | +5 |
| banana_D6 | elbo_err | 20 | 20 | 0.150 | 0.983 | ok | -0.00077 |
| banana_D6 | gskl | 20 | 20 | 0.150 | 0.983 | ok | -0.0219 |
| banana_D6 | mmtv | 20 | 20 | 0.150 | 0.983 | ok | +1.2e-06 |
| banana_D6 | func_count | 20 | 20 | 0.150 | 0.983 | ok | +0 |
| cigar_D4 | elbo_err | 20 | 20 | 0.250 | 0.571 | ok | -0.00283 |
| cigar_D4 | gskl | 20 | 20 | 0.250 | 0.571 | ok | +0.000115 |
| cigar_D4 | mmtv | 20 | 20 | 0.250 | 0.571 | ok | -0.00357 |
| cigar_D4 | func_count | 20 | 20 | 0.150 | 0.983 | ok | +0 |
| corr_D5 | elbo_err | 20 | 20 | 0.200 | 0.832 | ok | -0.00086 |
| corr_D5 | gskl | 20 | 20 | 0.350 | 0.175 | ok | +0.000321 |
| corr_D5 | mmtv | 20 | 20 | 0.150 | 0.983 | ok | +0.000346 |
| corr_D5 | func_count | 20 | 20 | 0.200 | 0.832 | ok | +0 |
| halfnormal_D2 | elbo_err | 20 | 20 | 0.350 | 0.175 | ok | +0.00246 |
| halfnormal_D2 | gskl | 20 | 20 | 0.400 | 0.0811 | ok | +0.000103 |
| halfnormal_D2 | mmtv | 20 | 20 | 0.200 | 0.832 | ok | +0.000767 |
| halfnormal_D2 | func_count | 20 | 20 | 0.050 | 1 | ok | +0 |
| logreg_D5 | elbo_err | 20 | 20 | 0.200 | 0.832 | ok | +0.0068 |
| logreg_D5 | gskl | 20 | 20 | 0.150 | 0.983 | ok | +0.00217 |
| logreg_D5 | mmtv | 20 | 20 | 0.200 | 0.832 | ok | +0.000463 |
| logreg_D5 | func_count | 20 | 20 | 0.250 | 0.571 | ok | +5 |
| logreg_D5_noise3 | elbo_err | 20 | 20 | 0.200 | 0.832 | ok | +0.0371 |
| logreg_D5_noise3 | gskl | 20 | 20 | 0.250 | 0.571 | ok | +0.0462 |
| logreg_D5_noise3 | mmtv | 20 | 20 | 0.200 | 0.832 | ok | +0.0122 |
| logreg_D5_noise3 | func_count | 20 | 20 | 0.150 | 0.983 | ok | +2.5 |
| lumpy_D10 | elbo_err | 20 | 20 | 0.300 | 0.336 | ok | -0.0237 |
| lumpy_D10 | gskl | 20 | 20 | 0.250 | 0.571 | ok | -0.0416 |
| lumpy_D10 | mmtv | 20 | 20 | 0.300 | 0.336 | ok | -0.00219 |
| lumpy_D10 | func_count | 20 | 20 | 0.450 | 0.0335 | ok | +65 |
| lumpy_D4 | elbo_err | 20 | 20 | 0.150 | 0.983 | ok | +0.00318 |
| lumpy_D4 | gskl | 20 | 20 | 0.200 | 0.832 | ok | +0.00368 |
| lumpy_D4 | mmtv | 20 | 20 | 0.150 | 0.983 | ok | -0.00153 |
| lumpy_D4 | func_count | 20 | 20 | 0.150 | 0.983 | ok | -5 |
| normal_D5 | elbo_err | 20 | 20 | 0.200 | 0.832 | ok | +0.00139 |
| normal_D5 | gskl | 20 | 20 | 0.250 | 0.571 | ok | +2.34e-05 |
| normal_D5 | mmtv | 20 | 20 | 0.200 | 0.832 | ok | +0.000279 |
| normal_D5 | func_count | 20 | 20 | 0.100 | 1 | ok | +0 |
| rosenbrock_D2 | elbo_err | 20 | 20 | 0.350 | 0.175 | ok | -0.00283 |
| rosenbrock_D2 | gskl | 20 | 20 | 0.200 | 0.832 | ok | -0.00465 |
| rosenbrock_D2 | mmtv | 20 | 20 | 0.150 | 0.983 | ok | -0.000649 |
| rosenbrock_D2 | func_count | 20 | 20 | 0.150 | 0.983 | ok | +0 |
| rosenbrock_D2_noise1 | elbo_err | 20 | 20 | 0.200 | 0.832 | ok | -0.0189 |
| rosenbrock_D2_noise1 | gskl | 20 | 20 | 0.250 | 0.571 | ok | +0.0106 |
| rosenbrock_D2_noise1 | mmtv | 20 | 20 | 0.400 | 0.0811 | ok | +0.00814 |
| rosenbrock_D2_noise1 | func_count | 20 | 20 | 0.450 | 0.0335 | ok | -10 |
| student_D4 | elbo_err | 20 | 20 | 0.200 | 0.832 | ok | +0.000473 |
| student_D4 | gskl | 20 | 20 | 0.150 | 0.983 | ok | +0.000745 |
| student_D4 | mmtv | 20 | 20 | 0.250 | 0.571 | ok | -0.00393 |
| student_D4 | func_count | 20 | 20 | 0.250 | 0.571 | ok | +7.5 |

| config | median func_count ratio (new/ref), descriptive |
|---|---|
| banana_D10 | 1.020 |
| banana_D2 | 1.061 |
| banana_D6 | 1.000 |
| cigar_D4 | 1.000 |
| corr_D5 | 1.000 |
| halfnormal_D2 | 1.000 |
| logreg_D5 | 1.038 |
| logreg_D5_noise3 | 1.011 |
| lumpy_D10 | 1.268 |
| lumpy_D4 | 0.947 |
| normal_D5 | 1.000 |
| rosenbrock_D2 | 1.000 |
| rosenbrock_D2_noise1 | 0.927 |
| student_D4 | 1.073 |

**no config flagged (56 KS tests, Holm alpha 0.05)**
