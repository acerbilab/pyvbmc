# Compare population_before (reference) vs population_after (candidate), rescored metrics

| config | metric | n ref | n new | KS | p | Holm | median shift |
|---|---|---|---|---|---|---|---|
| banana_D10 | elbo_err | 100 | 100 | 0.220 | 0.0156 | ok | -0.0151 |
| banana_D10 | gskl | 100 | 100 | 0.240 | 0.00613 | ok | -0.0649 |
| banana_D10 | mmtv | 100 | 100 | 0.210 | 0.0241 | ok | -0.0017 |
| banana_D10 | func_count | 100 | 100 | 0.090 | 0.815 | ok | -2.5 |
| banana_D2 | elbo_err | 100 | 100 | 0.070 | 0.968 | ok | -5.86e-05 |
| banana_D2 | gskl | 100 | 100 | 0.130 | 0.368 | ok | +0.00452 |
| banana_D2 | mmtv | 100 | 100 | 0.100 | 0.702 | ok | -0.000959 |
| banana_D2 | func_count | 100 | 100 | 0.140 | 0.282 | ok | -5 |
| banana_D6 | elbo_err | 100 | 100 | 0.140 | 0.282 | ok | -0.00559 |
| banana_D6 | gskl | 100 | 100 | 0.110 | 0.583 | ok | -0.0141 |
| banana_D6 | mmtv | 100 | 100 | 0.090 | 0.815 | ok | -0.00137 |
| banana_D6 | func_count | 100 | 100 | 0.100 | 0.702 | ok | +0 |
| cigar_D15_exhaust | elbo_err | 100 | 100 | 0.080 | 0.908 | ok | -0.00079 |
| cigar_D15_exhaust | gskl | 100 | 100 | 0.080 | 0.908 | ok | -9.99e-05 |
| cigar_D15_exhaust | mmtv | 100 | 100 | 0.090 | 0.815 | ok | -2.45e-05 |
| cigar_D15_exhaust | func_count | 100 | 100 | 0.000 | 1 | ok | +0 |
| cigar_D4 | elbo_err | 100 | 100 | 0.150 | 0.211 | ok | -0.000521 |
| cigar_D4 | gskl | 100 | 100 | 0.060 | 0.994 | ok | +2.86e-05 |
| cigar_D4 | mmtv | 100 | 100 | 0.070 | 0.968 | ok | +0.000545 |
| cigar_D4 | func_count | 100 | 100 | 0.350 | 7.85e-06 | REJECT | -10 |
| cigar_D8 | elbo_err | 100 | 100 | 0.090 | 0.815 | ok | +0.00112 |
| cigar_D8 | gskl | 100 | 100 | 0.140 | 0.282 | ok | +0.000274 |
| cigar_D8 | mmtv | 100 | 100 | 0.110 | 0.583 | ok | +0.000444 |
| cigar_D8 | func_count | 100 | 100 | 0.180 | 0.0782 | ok | -5 |
| corr_D5 | elbo_err | 100 | 100 | 0.190 | 0.0539 | ok | -0.00222 |
| corr_D5 | gskl | 100 | 100 | 0.250 | 0.00373 | ok | -0.0003 |
| corr_D5 | mmtv | 100 | 100 | 0.160 | 0.155 | ok | -0.000391 |
| corr_D5 | func_count | 100 | 100 | 0.140 | 0.282 | ok | +0 |
| halfnormal_D2 | elbo_err | 100 | 100 | 0.100 | 0.702 | ok | +0.000522 |
| halfnormal_D2 | gskl | 100 | 100 | 0.100 | 0.702 | ok | -7.37e-06 |
| halfnormal_D2 | mmtv | 100 | 100 | 0.120 | 0.47 | ok | +0.000203 |
| halfnormal_D2 | func_count | 100 | 100 | 0.030 | 1 | ok | +0 |
| logreg_D5 | elbo_err | 100 | 100 | 0.100 | 0.702 | ok | +0.00561 |
| logreg_D5 | gskl | 100 | 100 | 0.120 | 0.47 | ok | -0.000166 |
| logreg_D5 | mmtv | 100 | 100 | 0.090 | 0.815 | ok | +0.000438 |
| logreg_D5 | func_count | 100 | 100 | 0.050 | 1 | ok | +0 |
| logreg_D5_noise3_production | elbo_err | 100 | 100 | 0.080 | 0.908 | ok | +0.0319 |
| logreg_D5_noise3_production | gskl | 100 | 100 | 0.210 | 0.0241 | ok | -0.0562 |
| logreg_D5_noise3_production | mmtv | 100 | 100 | 0.120 | 0.47 | ok | -0.00882 |
| logreg_D5_noise3_production | func_count | 100 | 100 | 0.110 | 0.583 | ok | +7.5 |
| lumpy_D10 | elbo_err | 100 | 100 | 0.130 | 0.368 | ok | +0.0236 |
| lumpy_D10 | gskl | 100 | 100 | 0.100 | 0.702 | ok | -0.00895 |
| lumpy_D10 | mmtv | 100 | 100 | 0.120 | 0.47 | ok | +0.00142 |
| lumpy_D10 | func_count | 100 | 100 | 0.260 | 0.00222 | ok | -40 |
| lumpy_D10_noise3_production | elbo_err | 100 | 100 | 0.160 | 0.155 | ok | +0.0619 |
| lumpy_D10_noise3_production | gskl | 100 | 100 | 0.170 | 0.111 | ok | +0.0475 |
| lumpy_D10_noise3_production | mmtv | 100 | 100 | 0.120 | 0.47 | ok | +0.000722 |
| lumpy_D10_noise3_production | func_count | 100 | 100 | 0.230 | 0.00988 | ok | +5 |
| lumpy_D4 | elbo_err | 100 | 100 | 0.090 | 0.815 | ok | -0.00101 |
| lumpy_D4 | gskl | 100 | 100 | 0.140 | 0.282 | ok | -0.000214 |
| lumpy_D4 | mmtv | 100 | 100 | 0.080 | 0.908 | ok | -4.31e-05 |
| lumpy_D4 | func_count | 100 | 100 | 0.080 | 0.908 | ok | +0 |
| multisensory_s1_D6 | elbo_err | 100 | 100 | 0.080 | 0.908 | ok | +0.00272 |
| multisensory_s1_D6 | gskl | 100 | 100 | 0.090 | 0.815 | ok | -0.0274 |
| multisensory_s1_D6 | mmtv | 100 | 100 | 0.120 | 0.47 | ok | -0.00346 |
| multisensory_s1_D6 | func_count | 100 | 100 | 0.140 | 0.282 | ok | -2.5 |
| multisensory_s1_D6_noise1.3_production | elbo_err | 100 | 100 | 0.080 | 0.908 | ok | -0.0258 |
| multisensory_s1_D6_noise1.3_production | gskl | 100 | 100 | 0.170 | 0.111 | ok | -0.335 |
| multisensory_s1_D6_noise1.3_production | mmtv | 100 | 100 | 0.180 | 0.0782 | ok | -0.00712 |
| multisensory_s1_D6_noise1.3_production | func_count | 100 | 100 | 0.090 | 0.815 | ok | +0 |
| multisensory_s2_D6_noise1.3_production | elbo_err | 100 | 100 | 0.140 | 0.282 | ok | +0.0609 |
| multisensory_s2_D6_noise1.3_production | gskl | 100 | 100 | 0.120 | 0.47 | ok | +0.0509 |
| multisensory_s2_D6_noise1.3_production | mmtv | 100 | 100 | 0.070 | 0.968 | ok | -0.00206 |
| multisensory_s2_D6_noise1.3_production | func_count | 100 | 100 | 0.110 | 0.583 | ok | -5 |
| normal_D5 | elbo_err | 100 | 100 | 0.090 | 0.815 | ok | +0.000291 |
| normal_D5 | gskl | 100 | 100 | 0.100 | 0.702 | ok | +1.98e-06 |
| normal_D5 | mmtv | 100 | 100 | 0.190 | 0.0539 | ok | +0.000289 |
| normal_D5 | func_count | 100 | 100 | 0.100 | 0.702 | ok | +0 |
| rosenbrock_D2 | elbo_err | 100 | 100 | 0.100 | 0.702 | ok | +0.00208 |
| rosenbrock_D2 | gskl | 100 | 100 | 0.100 | 0.702 | ok | +0.00185 |
| rosenbrock_D2 | mmtv | 100 | 100 | 0.080 | 0.908 | ok | +0.000514 |
| rosenbrock_D2 | func_count | 100 | 100 | 0.080 | 0.908 | ok | +0 |
| rosenbrock_D2_noise1_production | elbo_err | 100 | 100 | 0.100 | 0.702 | ok | +0.00952 |
| rosenbrock_D2_noise1_production | gskl | 100 | 100 | 0.090 | 0.815 | ok | +0.00115 |
| rosenbrock_D2_noise1_production | mmtv | 100 | 100 | 0.130 | 0.368 | ok | -0.000161 |
| rosenbrock_D2_noise1_production | func_count | 100 | 100 | 0.190 | 0.0539 | ok | -10 |
| rosenbrock_D2_noise3_production | elbo_err | 100 | 100 | 0.120 | 0.47 | ok | +0.0246 |
| rosenbrock_D2_noise3_production | gskl | 100 | 100 | 0.170 | 0.111 | ok | +0.0581 |
| rosenbrock_D2_noise3_production | mmtv | 100 | 100 | 0.230 | 0.00988 | ok | +0.034 |
| rosenbrock_D2_noise3_production | func_count | 100 | 100 | 0.350 | 7.85e-06 | REJECT | -15 |
| student_D4 | elbo_err | 100 | 100 | 0.110 | 0.583 | ok | +0.00256 |
| student_D4 | gskl | 100 | 100 | 0.080 | 0.908 | ok | +0.00081 |
| student_D4 | mmtv | 100 | 100 | 0.110 | 0.583 | ok | -0.0014 |
| student_D4 | func_count | 100 | 100 | 0.060 | 0.994 | ok | +0 |
| student_D8 | elbo_err | 100 | 100 | 0.090 | 0.815 | ok | -0.00707 |
| student_D8 | gskl | 100 | 100 | 0.140 | 0.282 | ok | +0.0181 |
| student_D8 | mmtv | 100 | 100 | 0.210 | 0.0241 | ok | +0.0046 |
| student_D8 | func_count | 100 | 100 | 0.080 | 0.908 | ok | -5 |
| student_D8_noise3_production | elbo_err | 100 | 100 | 0.160 | 0.155 | ok | -0.198 |
| student_D8_noise3_production | gskl | 100 | 100 | 0.120 | 0.47 | ok | -0.05 |
| student_D8_noise3_production | mmtv | 100 | 100 | 0.220 | 0.0156 | ok | -0.0135 |
| student_D8_noise3_production | func_count | 100 | 100 | 0.110 | 0.583 | ok | +2.5 |
| timing_D5_noise2.2_production | elbo_err | 100 | 100 | 0.100 | 0.702 | ok | -0.0346 |
| timing_D5_noise2.2_production | gskl | 100 | 100 | 0.100 | 0.702 | ok | +0.0246 |
| timing_D5_noise2.2_production | mmtv | 100 | 100 | 0.100 | 0.702 | ok | -0.00173 |
| timing_D5_noise2.2_production | func_count | 100 | 100 | 0.120 | 0.47 | ok | -5 |

| config | median func_count ratio (new/ref), descriptive |
|---|---|
| banana_D10 | 0.980 |
| banana_D2 | 0.944 |
| banana_D6 | 1.000 |
| cigar_D15_exhaust | 1.000 |
| cigar_D4 | 0.926 |
| cigar_D8 | 0.970 |
| corr_D5 | 1.000 |
| halfnormal_D2 | 1.000 |
| logreg_D5 | 1.000 |
| logreg_D5_noise3_production | 1.031 |
| lumpy_D10 | 0.860 |
| lumpy_D10_noise3_production | 1.014 |
| lumpy_D4 | 1.000 |
| multisensory_s1_D6 | 0.985 |
| multisensory_s1_D6_noise1.3_production | 1.000 |
| multisensory_s2_D6_noise1.3_production | 0.976 |
| normal_D5 | 1.000 |
| rosenbrock_D2 | 1.000 |
| rosenbrock_D2_noise1_production | 0.926 |
| rosenbrock_D2_noise3_production | 0.919 |
| student_D4 | 1.000 |
| student_D8 | 0.976 |
| student_D8_noise3_production | 1.007 |
| timing_D5_noise2.2_production | 0.978 |

**2 config(s) flagged: ['cigar_D4', 'rosenbrock_D2_noise3_production']**
