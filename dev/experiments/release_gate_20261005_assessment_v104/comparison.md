# Compare population_v104 (reference) vs population_after (candidate), rescored metrics

| config | metric | n ref | n new | KS | p | Holm | median shift |
|---|---|---|---|---|---|---|---|
| banana_D10 | elbo_err | 100 | 100 | 0.200 | 0.0364 | ok | -0.0135 |
| banana_D10 | gskl | 100 | 100 | 0.180 | 0.0782 | ok | -0.0505 |
| banana_D10 | mmtv | 100 | 100 | 0.160 | 0.155 | ok | -0.00134 |
| banana_D10 | func_count | 100 | 100 | 0.160 | 0.155 | ok | -2.5 |
| banana_D2 | elbo_err | 100 | 100 | 0.330 | 3.21e-05 | REJECT | -0.0118 |
| banana_D2 | gskl | 100 | 100 | 0.410 | 6.62e-08 | REJECT | -0.0619 |
| banana_D2 | mmtv | 100 | 100 | 0.370 | 1.75e-06 | REJECT | -0.0122 |
| banana_D2 | func_count | 100 | 100 | 0.070 | 0.968 | ok | +0 |
| banana_D6 | elbo_err | 100 | 100 | 0.090 | 0.815 | ok | -0.00229 |
| banana_D6 | gskl | 100 | 100 | 0.090 | 0.815 | ok | -0.00695 |
| banana_D6 | mmtv | 100 | 100 | 0.100 | 0.702 | ok | -0.00124 |
| banana_D6 | func_count | 100 | 100 | 0.070 | 0.968 | ok | +0 |
| cigar_D15_exhaust | elbo_err | 98 | 100 | 0.089 | 0.778 | ok | -0.00111 |
| cigar_D15_exhaust | gskl | 98 | 100 | 0.106 | 0.584 | ok | -0.000197 |
| cigar_D15_exhaust | mmtv | 98 | 100 | 0.082 | 0.858 | ok | -0.000159 |
| cigar_D15_exhaust | func_count | 98 | 100 | 0.000 | 1 | ok | +0 |
| cigar_D4 | elbo_err | 100 | 100 | 0.080 | 0.908 | ok | -6.44e-05 |
| cigar_D4 | gskl | 100 | 100 | 0.140 | 0.282 | ok | +0.000239 |
| cigar_D4 | mmtv | 100 | 100 | 0.130 | 0.368 | ok | +0.000884 |
| cigar_D4 | func_count | 100 | 100 | 0.400 | 1.56e-07 | REJECT | -10 |
| cigar_D8 | elbo_err | 100 | 100 | 0.120 | 0.47 | ok | +0.00154 |
| cigar_D8 | gskl | 100 | 100 | 0.090 | 0.815 | ok | -0.000403 |
| cigar_D8 | mmtv | 100 | 100 | 0.080 | 0.908 | ok | -0.000428 |
| cigar_D8 | func_count | 100 | 100 | 0.400 | 1.56e-07 | REJECT | -10 |
| corr_D5 | elbo_err | 100 | 100 | 0.200 | 0.0364 | ok | -0.00223 |
| corr_D5 | gskl | 100 | 100 | 0.270 | 0.00129 | ok | -0.000512 |
| corr_D5 | mmtv | 100 | 100 | 0.180 | 0.0782 | ok | -0.000305 |
| corr_D5 | func_count | 100 | 100 | 0.210 | 0.0241 | ok | +0 |
| halfnormal_D2 | elbo_err | 100 | 100 | 0.120 | 0.47 | ok | +0.000459 |
| halfnormal_D2 | gskl | 100 | 100 | 0.250 | 0.00373 | ok | +5.4e-05 |
| halfnormal_D2 | mmtv | 100 | 100 | 0.120 | 0.47 | ok | +0.000525 |
| halfnormal_D2 | func_count | 100 | 100 | 0.110 | 0.583 | ok | +0 |
| logreg_D5 | elbo_err | 100 | 100 | 0.180 | 0.0782 | ok | -0.00497 |
| logreg_D5 | gskl | 100 | 100 | 0.180 | 0.0782 | ok | -0.0042 |
| logreg_D5 | mmtv | 100 | 100 | 0.110 | 0.583 | ok | -0.000844 |
| logreg_D5 | func_count | 100 | 100 | 0.080 | 0.908 | ok | -5 |
| logreg_D5_noise3_production | elbo_err | 100 | 100 | 0.070 | 0.968 | ok | -0.00046 |
| logreg_D5_noise3_production | gskl | 100 | 100 | 0.130 | 0.368 | ok | -0.0264 |
| logreg_D5_noise3_production | mmtv | 100 | 100 | 0.130 | 0.368 | ok | -0.00715 |
| logreg_D5_noise3_production | func_count | 100 | 100 | 0.110 | 0.583 | ok | +7.5 |
| lumpy_D10 | elbo_err | 100 | 100 | 0.150 | 0.211 | ok | +0.00364 |
| lumpy_D10 | gskl | 100 | 100 | 0.150 | 0.211 | ok | -0.0361 |
| lumpy_D10 | mmtv | 100 | 100 | 0.130 | 0.368 | ok | -0.00453 |
| lumpy_D10 | func_count | 100 | 100 | 0.180 | 0.0782 | ok | -27.5 |
| lumpy_D10_noise3_production | elbo_err | 100 | 100 | 0.150 | 0.211 | ok | +0.00921 |
| lumpy_D10_noise3_production | gskl | 100 | 100 | 0.230 | 0.00988 | ok | +0.061 |
| lumpy_D10_noise3_production | mmtv | 100 | 100 | 0.180 | 0.0782 | ok | +0.00466 |
| lumpy_D10_noise3_production | func_count | 100 | 100 | 0.300 | 0.000225 | REJECT | +10 |
| lumpy_D4 | elbo_err | 100 | 100 | 0.090 | 0.815 | ok | +0.00215 |
| lumpy_D4 | gskl | 100 | 100 | 0.100 | 0.702 | ok | +0.00066 |
| lumpy_D4 | mmtv | 100 | 100 | 0.120 | 0.47 | ok | -0.000304 |
| lumpy_D4 | func_count | 100 | 100 | 0.060 | 0.994 | ok | +0 |
| multisensory_s1_D6 | elbo_err | 100 | 100 | 0.120 | 0.47 | ok | +0.00197 |
| multisensory_s1_D6 | gskl | 100 | 100 | 0.080 | 0.908 | ok | -0.0195 |
| multisensory_s1_D6 | mmtv | 100 | 100 | 0.110 | 0.583 | ok | -0.00415 |
| multisensory_s1_D6 | func_count | 100 | 100 | 0.060 | 0.994 | ok | +2.5 |
| multisensory_s1_D6_noise1.3_production | elbo_err | 100 | 100 | 0.110 | 0.583 | ok | -0.00304 |
| multisensory_s1_D6_noise1.3_production | gskl | 100 | 100 | 0.080 | 0.908 | ok | -0.16 |
| multisensory_s1_D6_noise1.3_production | mmtv | 100 | 100 | 0.130 | 0.368 | ok | -0.0055 |
| multisensory_s1_D6_noise1.3_production | func_count | 100 | 100 | 0.130 | 0.368 | ok | -2.5 |
| multisensory_s2_D6_noise1.3_production | elbo_err | 100 | 100 | 0.170 | 0.111 | ok | +0.0703 |
| multisensory_s2_D6_noise1.3_production | gskl | 100 | 100 | 0.150 | 0.211 | ok | +0.0898 |
| multisensory_s2_D6_noise1.3_production | mmtv | 100 | 100 | 0.080 | 0.908 | ok | +0.00281 |
| multisensory_s2_D6_noise1.3_production | func_count | 100 | 100 | 0.190 | 0.0539 | ok | -10 |
| normal_D5 | elbo_err | 100 | 100 | 0.110 | 0.583 | ok | -0.000393 |
| normal_D5 | gskl | 100 | 100 | 0.110 | 0.583 | ok | -1.54e-05 |
| normal_D5 | mmtv | 100 | 100 | 0.100 | 0.702 | ok | +0.000138 |
| normal_D5 | func_count | 100 | 100 | 0.040 | 1 | ok | +0 |
| rosenbrock_D2 | elbo_err | 100 | 100 | 0.360 | 3.75e-06 | REJECT | -0.00978 |
| rosenbrock_D2 | gskl | 100 | 100 | 0.330 | 3.21e-05 | REJECT | -0.0116 |
| rosenbrock_D2 | mmtv | 100 | 100 | 0.320 | 6.28e-05 | REJECT | -0.00669 |
| rosenbrock_D2 | func_count | 100 | 100 | 0.060 | 0.994 | ok | +0 |
| rosenbrock_D2_noise1_production | elbo_err | 100 | 100 | 0.130 | 0.368 | ok | -0.00918 |
| rosenbrock_D2_noise1_production | gskl | 100 | 100 | 0.080 | 0.908 | ok | -0.000725 |
| rosenbrock_D2_noise1_production | mmtv | 100 | 100 | 0.090 | 0.815 | ok | +0.00234 |
| rosenbrock_D2_noise1_production | func_count | 100 | 100 | 0.220 | 0.0156 | ok | -10 |
| rosenbrock_D2_noise3_production | elbo_err | 100 | 100 | 0.070 | 0.968 | ok | -0.00127 |
| rosenbrock_D2_noise3_production | gskl | 100 | 100 | 0.220 | 0.0156 | ok | +0.075 |
| rosenbrock_D2_noise3_production | mmtv | 100 | 100 | 0.240 | 0.00613 | ok | +0.0421 |
| rosenbrock_D2_noise3_production | func_count | 100 | 100 | 0.270 | 0.00129 | ok | -15 |
| student_D4 | elbo_err | 100 | 100 | 0.090 | 0.815 | ok | -0.000436 |
| student_D4 | gskl | 100 | 100 | 0.110 | 0.583 | ok | +0.00118 |
| student_D4 | mmtv | 100 | 100 | 0.150 | 0.211 | ok | -0.00112 |
| student_D4 | func_count | 100 | 100 | 0.100 | 0.702 | ok | +0 |
| student_D8 | elbo_err | 100 | 100 | 0.120 | 0.47 | ok | +0.000661 |
| student_D8 | gskl | 100 | 100 | 0.200 | 0.0364 | ok | +0.0426 |
| student_D8 | mmtv | 100 | 100 | 0.230 | 0.00988 | ok | +0.00488 |
| student_D8 | func_count | 100 | 100 | 0.120 | 0.47 | ok | +0 |
| student_D8_noise3_production | elbo_err | 100 | 100 | 0.220 | 0.0156 | ok | -0.276 |
| student_D8_noise3_production | gskl | 100 | 100 | 0.180 | 0.0782 | ok | -0.0932 |
| student_D8_noise3_production | mmtv | 100 | 100 | 0.210 | 0.0241 | ok | -0.0126 |
| student_D8_noise3_production | func_count | 100 | 100 | 0.100 | 0.702 | ok | +5 |
| timing_D5_noise2.2_production | elbo_err | 100 | 100 | 0.130 | 0.368 | ok | -0.0411 |
| timing_D5_noise2.2_production | gskl | 100 | 100 | 0.110 | 0.583 | ok | -0.0497 |
| timing_D5_noise2.2_production | mmtv | 100 | 100 | 0.160 | 0.155 | ok | -0.0103 |
| timing_D5_noise2.2_production | func_count | 100 | 100 | 0.120 | 0.47 | ok | -5 |

| config | median func_count ratio (new/ref), descriptive |
|---|---|
| banana_D10 | 0.980 |
| banana_D2 | 1.000 |
| banana_D6 | 1.000 |
| cigar_D15_exhaust | 1.000 |
| cigar_D4 | 0.926 |
| cigar_D8 | 0.941 |
| corr_D5 | 1.000 |
| halfnormal_D2 | 1.000 |
| logreg_D5 | 0.963 |
| logreg_D5_noise3_production | 1.031 |
| lumpy_D10 | 0.899 |
| lumpy_D10_noise3_production | 1.028 |
| lumpy_D4 | 1.000 |
| multisensory_s1_D6 | 1.015 |
| multisensory_s1_D6_noise1.3_production | 0.988 |
| multisensory_s2_D6_noise1.3_production | 0.953 |
| normal_D5 | 1.000 |
| rosenbrock_D2 | 1.000 |
| rosenbrock_D2_noise1_production | 0.926 |
| rosenbrock_D2_noise3_production | 0.919 |
| student_D4 | 1.000 |
| student_D8 | 1.000 |
| student_D8_noise3_production | 1.015 |
| timing_D5_noise2.2_production | 0.978 |

**5 config(s) flagged: ['banana_D2', 'cigar_D4', 'cigar_D8', 'lumpy_D10_noise3_production', 'rosenbrock_D2']**
