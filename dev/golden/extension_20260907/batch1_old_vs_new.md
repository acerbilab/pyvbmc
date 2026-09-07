# Compare seeds_0_19 (ref) vs seeds_20_49 (new)

| config | metric | n ref | n new | KS | p | Holm | median shift |
|---|---|---|---|---|---|---|---|
| banana_D10 | elbo_err | 20 | 30 | 0.233 | 0.486 | ok | -0.00469 |
| banana_D10 | gskl | 20 | 30 | 0.217 | 0.579 | ok | +0.0445 |
| banana_D10 | mmtv | 20 | 30 | 0.183 | 0.772 | ok | +0.000866 |
| banana_D10 | func_count | 20 | 30 | 0.250 | 0.399 | ok | +0 |
| banana_D2 | elbo_err | 20 | 30 | 0.150 | 0.928 | ok | -0.00265 |
| banana_D2 | gskl | 20 | 30 | 0.167 | 0.86 | ok | +0.0197 |
| banana_D2 | mmtv | 20 | 30 | 0.150 | 0.928 | ok | -0.000293 |
| banana_D2 | func_count | 20 | 30 | 0.200 | 0.679 | ok | +2.5 |
| banana_D6 | elbo_err | 20 | 30 | 0.200 | 0.679 | ok | -0.00278 |
| banana_D6 | gskl | 20 | 30 | 0.183 | 0.772 | ok | +0.0204 |
| banana_D6 | mmtv | 20 | 30 | 0.167 | 0.86 | ok | +0.000917 |
| banana_D6 | func_count | 20 | 30 | 0.167 | 0.86 | ok | +0 |
| cigar_D4 | elbo_err | 20 | 30 | 0.217 | 0.579 | ok | -0.00187 |
| cigar_D4 | gskl | 20 | 30 | 0.300 | 0.202 | ok | -0.000501 |
| cigar_D4 | mmtv | 20 | 30 | 0.300 | 0.202 | ok | -0.00138 |
| cigar_D4 | func_count | 20 | 30 | 0.100 | 0.999 | ok | +0 |
| corr_D5 | elbo_err | 20 | 30 | 0.250 | 0.399 | ok | -0.000611 |
| corr_D5 | gskl | 20 | 30 | 0.317 | 0.156 | ok | -0.000553 |
| corr_D5 | mmtv | 20 | 30 | 0.317 | 0.156 | ok | -0.000985 |
| corr_D5 | func_count | 20 | 30 | 0.167 | 0.86 | ok | +0 |
| halfnormal_D2 | elbo_err | 20 | 30 | 0.383 | 0.048 | ok | -0.00248 |
| halfnormal_D2 | gskl | 20 | 30 | 0.383 | 0.048 | ok | -6.33e-05 |
| halfnormal_D2 | mmtv | 20 | 30 | 0.217 | 0.579 | ok | +0.000105 |
| halfnormal_D2 | func_count | 20 | 30 | 0.167 | 0.86 | ok | +0 |
| logreg_D5 | elbo_err | 20 | 30 | 0.333 | 0.119 | ok | -0.0233 |
| logreg_D5 | gskl | 20 | 30 | 0.350 | 0.0893 | ok | -0.00496 |
| logreg_D5 | mmtv | 20 | 30 | 0.317 | 0.156 | ok | -0.0049 |
| logreg_D5 | func_count | 20 | 30 | 0.167 | 0.86 | ok | +0 |
| logreg_D5_noise3 | elbo_err | 20 | 30 | 0.250 | 0.399 | ok | -0.0527 |
| logreg_D5_noise3 | gskl | 20 | 30 | 0.167 | 0.86 | ok | +0.00314 |
| logreg_D5_noise3 | mmtv | 20 | 30 | 0.300 | 0.202 | ok | -0.0123 |
| logreg_D5_noise3 | func_count | 20 | 30 | 0.217 | 0.579 | ok | +7.5 |
| lumpy_D10 | elbo_err | 20 | 30 | 0.250 | 0.399 | ok | -0.0316 |
| lumpy_D10 | gskl | 20 | 30 | 0.150 | 0.928 | ok | +0.0192 |
| lumpy_D10 | mmtv | 20 | 30 | 0.217 | 0.579 | ok | -0.00346 |
| lumpy_D10 | func_count | 20 | 30 | 0.167 | 0.86 | ok | -30 |
| lumpy_D4 | elbo_err | 20 | 30 | 0.150 | 0.928 | ok | -0.00861 |
| lumpy_D4 | gskl | 20 | 30 | 0.217 | 0.579 | ok | -0.00378 |
| lumpy_D4 | mmtv | 20 | 30 | 0.167 | 0.86 | ok | -0.000438 |
| lumpy_D4 | func_count | 20 | 30 | 0.117 | 0.993 | ok | +0 |
| normal_D5 | elbo_err | 20 | 30 | 0.300 | 0.202 | ok | -0.0017 |
| normal_D5 | gskl | 20 | 30 | 0.200 | 0.679 | ok | -1.7e-05 |
| normal_D5 | mmtv | 20 | 30 | 0.183 | 0.772 | ok | -0.000174 |
| normal_D5 | func_count | 20 | 30 | 0.250 | 0.399 | ok | +2.5 |
| rosenbrock_D2 | elbo_err | 20 | 30 | 0.300 | 0.202 | ok | +0.00258 |
| rosenbrock_D2 | gskl | 20 | 30 | 0.217 | 0.579 | ok | +0.00448 |
| rosenbrock_D2 | mmtv | 20 | 30 | 0.133 | 0.972 | ok | +0.00148 |
| rosenbrock_D2 | func_count | 20 | 30 | 0.283 | 0.257 | ok | +0 |
| rosenbrock_D2_noise1 | elbo_err | 20 | 30 | 0.200 | 0.679 | ok | +0.00476 |
| rosenbrock_D2_noise1 | gskl | 20 | 30 | 0.200 | 0.679 | ok | -0.000362 |
| rosenbrock_D2_noise1 | mmtv | 20 | 30 | 0.133 | 0.972 | ok | -0.000724 |
| rosenbrock_D2_noise1 | func_count | 20 | 30 | 0.167 | 0.86 | ok | +5 |
| student_D4 | elbo_err | 20 | 30 | 0.167 | 0.86 | ok | +0.000684 |
| student_D4 | gskl | 20 | 30 | 0.217 | 0.579 | ok | -0.00376 |
| student_D4 | mmtv | 20 | 30 | 0.217 | 0.579 | ok | +0.00365 |
| student_D4 | func_count | 20 | 30 | 0.250 | 0.399 | ok | -7.5 |

| config | median func_count ratio (new/ref), descriptive |
|---|---|
| banana_D10 | 1.000 |
| banana_D2 | 1.029 |
| banana_D6 | 1.000 |
| cigar_D4 | 1.000 |
| corr_D5 | 1.000 |
| halfnormal_D2 | 1.000 |
| logreg_D5 | 1.000 |
| logreg_D5_noise3 | 1.031 |
| lumpy_D10 | 0.902 |
| lumpy_D4 | 1.000 |
| normal_D5 | 1.036 |
| rosenbrock_D2 | 1.000 |
| rosenbrock_D2_noise1 | 1.039 |
| student_D4 | 0.932 |

**no config flagged (56 KS tests, Holm alpha 0.05)**
