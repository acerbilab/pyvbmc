# S-VBMC pool: pools_arm3

Generated 2026-10-05 21:44:29. The campaign was prepared on login at `ad63dd5a21afb21343cf25b278b86f184eb42a17` (gpyreg `682585f72d89`), the code every case was generated with; each case records the host that ran it. Filters: stable and `sqrt(max J_sjk) < sqrt(5)`; the counts are of the runs the directory holds, against the allocation's seed caps and filtered targets, and the metric quartiles are over the filtered runs, the wall times over every completed run. `success` counts the completed runs whose `success_flag` is true and `converged` those whose `convergence_status` is `probable`.

| condition | seeds | filtered | pass rate | usable | success | converged | wall min (med [IQR]) | evals | K | elbo_err | gskl | mmtv |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| multisensory_s1_D6_noise3_svbmc | 350/350 | 348/320 | 0.99 | 0.02 | 350/350 | 350/350 | 7.1 [6.7, 7.5] | 280 [256, 300] | 50 [50, 50] | 0.42 [0.21, 0.67] | 3.92 [2.66, 5.40] | 0.268 [0.240, 0.300] |
| multisensory_s1_D6_noise1.3_svbmc | 350/350 | 349/320 | 1.00 | 0.11 | 350/350 | 350/350 | 5.0 [4.6, 5.5] | 205 [195, 220] | 50 [50, 50] | 0.42 [0.28, 0.56] | 2.61 [1.83, 3.48] | 0.221 [0.199, 0.245] |
| rosenbrock_D2_noise3_svbmc | 350/350 | 349/320 | 1.00 | 0.77 | 349/350 | 349/350 | 3.4 [3.0, 3.9] | 180 [165, 200] | 50 [50, 50] | 0.21 [0.09, 0.33] | 0.13 [0.04, 0.51] | 0.096 [0.051, 0.173] |
| gmm_D2_noise3_svbmc | 350/350 | 343/320 | 0.98 | 0.01 | 350/350 | 350/350 | 2.8 [2.4, 3.1] | 160 [150, 175] | 50 [50, 50] | 1.07 [0.68, 1.35] | 20.43 [13.22, 37.54] | 0.468 [0.395, 0.577] |
| ring_D2_noise3_svbmc | 480/480 | 368/320 | 0.77 | 0.00 | 377/480 | 377/480 | 4.8 [3.8, 5.6] | 240 [195, 295] | 50 [50, 50] | 1.30 [0.93, 1.72] | 41.32 [7.51, 350.72] | 0.598 [0.504, 0.708] |
| student_D8_noise3_svbmc | 350/350 | 350/320 | 1.00 | 0.60 | 350/350 | 350/350 | 9.6 [9.2, 10.1] | 335 [315, 350] | 50 [50, 50] | 0.86 [0.59, 1.24] | 0.58 [0.44, 0.74] | 0.137 [0.123, 0.149] |
| gmm_D2_svbmc | 350/350 | 344/320 | 0.98 | 0.12 | 350/350 | 350/350 | 0.5 [0.5, 0.7] | 85 [75, 95] | 50 [50, 50] | 0.70 [0.69, 1.38] | 15.00 [5.44, 28.09] | 0.389 [0.264, 0.550] |
| multisensory_s1_D6_svbmc | 350/350 | 344/320 | 0.98 | 0.92 | 350/350 | 350/350 | 2.1 [1.8, 2.3] | 170 [156, 180] | 50 [50, 50] | 0.38 [0.33, 0.46] | 0.51 [0.31, 0.72] | 0.114 [0.099, 0.130] |
