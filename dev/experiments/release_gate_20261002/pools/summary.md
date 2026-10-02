# S-VBMC pool: pools

Generated 2026-10-02 14:45:30. The campaign was prepared on login at `ff3ed01463b3c2a031dd2225c9f487dc369d2397` (gpyreg `682585f72d89`), the code every case was generated with; each case records the host that ran it. Filters: stable and `sqrt(max J_sjk) < sqrt(5)`; the counts are of the runs the directory holds, against the allocation's seed caps and filtered targets, and the metric quartiles are over the filtered runs, the wall times over every completed run. `success` counts the completed runs whose `success_flag` is true and `converged` those whose `convergence_status` is `probable`.

| condition | seeds | filtered | pass rate | usable | success | converged | wall min (med [IQR]) | evals | K | elbo_err | gskl | mmtv |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| multisensory_s1_D6_noise3_svbmc | 350/350 | 350/320 | 1.00 | 0.04 | 350/350 | 350/350 | 7.1 [6.6, 7.5] | 280 [260, 305] | 50 [50, 50] | 0.41 [0.20, 0.64] | 3.69 [2.42, 5.24] | 0.264 [0.234, 0.299] |
| multisensory_s1_D6_noise1.3_svbmc | 350/350 | 349/320 | 1.00 | 0.09 | 350/350 | 350/350 | 4.8 [4.4, 5.3] | 200 [190, 210] | 50 [50, 50] | 0.42 [0.30, 0.57] | 2.69 [1.86, 3.53] | 0.226 [0.203, 0.246] |
| rosenbrock_D2_noise3_svbmc | 350/350 | 349/320 | 1.00 | 0.74 | 350/350 | 350/350 | 3.4 [3.0, 3.9] | 170 [160, 190] | 50 [50, 50] | 0.21 [0.10, 0.34] | 0.19 [0.05, 0.65] | 0.116 [0.062, 0.184] |
| gmm_D2_noise3_svbmc | 350/350 | 342/320 | 0.98 | 0.00 | 350/350 | 350/350 | 2.9 [2.6, 3.3] | 155 [145, 170] | 50 [50, 50] | 1.03 [0.66, 1.33] | 19.76 [13.25, 37.50] | 0.474 [0.399, 0.574] |
| ring_D2_noise3_svbmc | 480/480 | 381/320 | 0.79 | 0.00 | 395/480 | 395/480 | 4.9 [3.3, 5.7] | 230 [175, 285] | 50 [50, 50] | 1.39 [0.93, 1.86] | 74.09 [7.61, 458.16] | 0.612 [0.495, 0.720] |
| student_D8_noise3_svbmc | 350/350 | 349/320 | 1.00 | 0.55 | 350/350 | 350/350 | 9.7 [9.2, 10.1] | 335 [320, 350] | 50 [50, 50] | 0.92 [0.62, 1.26] | 0.56 [0.43, 0.79] | 0.136 [0.123, 0.154] |
| gmm_D2_svbmc | 350/350 | 342/320 | 0.98 | 0.12 | 350/350 | 350/350 | 0.5 [0.5, 0.7] | 85 [75, 90] | 50 [50, 50] | 0.70 [0.69, 1.38] | 15.00 [10.30, 28.01] | 0.389 [0.360, 0.550] |
| multisensory_s1_D6_svbmc | 350/350 | 341/320 | 0.97 | 0.91 | 350/350 | 350/350 | 2.0 [1.7, 2.2] | 165 [155, 175] | 50 [50, 50] | 0.39 [0.33, 0.48] | 0.49 [0.31, 0.71] | 0.114 [0.099, 0.131] |
