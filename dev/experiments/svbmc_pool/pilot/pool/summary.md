# S-VBMC pool: pool

Generated 2026-09-14 04:12:09 from `e2aaef507f0eedd38616a33563c5d95287d156b0` (gpyreg `39536b000a8c`) on ASUS-LUIGI. Filters: stable and `sqrt(max J_sjk) < sqrt(5)`; the counts are of the runs the directory holds, against the allocation's seed caps and filtered targets, and the metric quartiles are over the filtered runs, the wall times over every completed run.

The last sweep ran with `--pilot-seeds 3`, which stops every condition at that many seeds whatever its filtered target.

| condition | seeds | filtered | pass rate | usable | wall min (med [IQR]) | evals | K | elbo_err | gskl | mmtv |
|---|---|---|---|---|---|---|---|---|---|---|
| rosenbrock_D2_noise3_svbmc | 3/90 | 3/60 | 1.00 | 1.00 | 1.6 [1.5, 1.7] | 180 [175, 192] | 50 [50, 50] | 0.03 [0.03, 0.15] | 0.04 [0.03, 0.09] | 0.083 [0.057, 0.092] |
| gmm_D2_svbmc | 3/45 | 3/30 | 1.00 | 0.00 | 0.2 [0.2, 0.3] | 75 [75, 82] | 50 [50, 50] | 0.71 [0.70, 1.05] | 10.96 [8.14, 23.57] | 0.363 [0.312, 0.467] |
| gmm_D2_noise3_svbmc | 3/90 | 3/60 | 1.00 | 0.00 | 1.5 [1.5, 1.8] | 175 [168, 188] | 50 [50, 50] | 0.46 [0.33, 0.70] | 1.17 [0.78, 2.94] | 0.317 [0.274, 0.373] |
| ring_D2_noise3_svbmc | 3/80 | 3/40 | 1.00 | 0.00 | 1.9 [1.6, 2.0] | 170 [162, 190] | 50 [50, 50] | 1.91 [1.80, 2.09] | 154.21 [80.23, 654.02] | 0.651 [0.650, 0.738] |
| multisensory_s1_D6_noise3_svbmc | 3/90 | 3/60 | 1.00 | 0.00 | 3.3 [3.2, 3.4] | 265 [260, 268] | 50 [50, 50] | 0.72 [0.58, 1.02] | 5.85 [5.81, 6.66] | 0.343 [0.324, 0.363] |
