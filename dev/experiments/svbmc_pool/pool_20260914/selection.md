# S-VBMC filtered pool: /home/fsilvest/francesco_projects/pyvbmc/dev/scripts/runs/svbmc_pool_20260914

Written 2026-09-14 18:20:05 from the completion records of the campaign. Per condition the runs are walked in seed order and the ones that pass the filters (stable and `sqrt(max J_sjk) < sqrt(5)`) are taken until the filtered target is met, so the pool is the lowest-seed runs that pass, whatever order the cases were generated in. The stacking comparison reads this file.

The walk stops as soon as the target is met, so the seeds it scanned are a prefix of the condition and the seeds beyond them were never looked at; a seed whose case failed is scanned like any other and leaves no run. `pass rate over the scanned seeds` is the selected runs over that prefix, failures included, and is therefore not the condition's pass rate, which `summarize` reports over every completed case.

| condition | selected | target | shortfall | seeds scanned | pass rate over the scanned seeds | last seed |
|---|---|---|---|---|---|---|
| multisensory_s1_D6_noise3_svbmc | 100 | 100 | - | 100 | 1.00 | 1099 |
| multisensory_s1_D6_noise1.3_svbmc | 100 | 100 | - | 102 | 0.98 | 1101 |
| rosenbrock_D2_noise3_svbmc | 100 | 100 | - | 103 | 0.97 | 1102 |
| gmm_D2_noise3_svbmc | 100 | 100 | - | 103 | 0.97 | 1102 |
| ring_D2_noise3_svbmc | 100 | 100 | - | 142 | 0.70 | 1141 |
| student_D8_noise3_svbmc | 100 | 100 | - | 100 | 1.00 | 1099 |
| gmm_D2_svbmc | 50 | 50 | - | 50 | 1.00 | 1049 |
| multisensory_s1_D6_svbmc | 50 | 50 | - | 50 | 1.00 | 1049 |
