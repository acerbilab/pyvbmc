# S-VBMC filtered pool: $CAMPAIGN_PARENT/pools_arm3

Written 2026-10-05 21:40:58. Per condition every seed of its range is walked in order from its first seed, and the runs that pass the filters (stable and `sqrt(max J_sjk) < sqrt(5)`) are taken until the filtered target is met, so the pool is the lowest-seed runs that pass, whatever order the cases ran in. A seed whose case failed (an error file) is walked like any other and leaves no run; no selection is written while a seed below a condition's last selected run is unfinished (neither a completion record nor an error file). The stacking comparison reads `selection.json`, and stacks the pool only when that file was made after a `verify` that passed and agrees with its report.

The walk stops as soon as the target is met, and the seeds beyond were never looked at. `seeds scanned` counts the seeds walked that hold a completion record or an error file, and `pass rate over the scanned seeds` is the selected runs over them, failures included; it is therefore not the condition's pass rate, which `summarize` reports over every completed case.

Made after the verification report of SHA-256 `902ebc01d0116bd2cfcdc351ecf08cdca2c8e04cf1957ab7f26c930f3e9ff473`.

| condition | selected | target | shortfall | seeds scanned | pass rate over the scanned seeds | last seed |
|---|---|---|---|---|---|---|
| multisensory_s1_D6_noise3_svbmc | 320 | 320 | - | 322 | 0.99 | 1321 |
| multisensory_s1_D6_noise1.3_svbmc | 320 | 320 | - | 321 | 1.00 | 1320 |
| rosenbrock_D2_noise3_svbmc | 320 | 320 | - | 321 | 1.00 | 1320 |
| gmm_D2_noise3_svbmc | 320 | 320 | - | 327 | 0.98 | 1326 |
| ring_D2_noise3_svbmc | 320 | 320 | - | 420 | 0.76 | 1419 |
| student_D8_noise3_svbmc | 320 | 320 | - | 320 | 1.00 | 1319 |
| gmm_D2_svbmc | 320 | 320 | - | 324 | 0.99 | 1323 |
| multisensory_s1_D6_svbmc | 320 | 320 | - | 326 | 0.98 | 1325 |
