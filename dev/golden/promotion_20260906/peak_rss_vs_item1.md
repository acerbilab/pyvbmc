# Peak RSS (MB) per run, item1_20260905 (ref) vs item7_20260906 (new)

The runner is one process, so each run's peak is the process high-water mark up to that run; configs are listed in run order, so the first row is that config alone and later rows show what the earlier configs added.

| config (run order) | median ref | median new | max ref | max new |
|---|---|---|---|---|
| lumpy_D10 | 419 | 402 | 438 | 402 |
| logreg_D5_noise3 | 474 | 402 | 474 | 402 |
| banana_D10 | 474 | 402 | 474 | 402 |
| cigar_D4 | 474 | 402 | 474 | 402 |
| rosenbrock_D2_noise1 | 474 | 402 | 474 | 402 |
| logreg_D5 | 474 | 402 | 474 | 402 |
| banana_D6 | 474 | 402 | 474 | 402 |
| corr_D5 | 474 | 402 | 474 | 402 |
| student_D4 | 474 | 402 | 474 | 402 |
| normal_D5 | 474 | 402 | 474 | 402 |
| lumpy_D4 | 474 | 402 | 474 | 402 |
| rosenbrock_D2 | 474 | 402 | 474 | 402 |
| banana_D2 | 474 | 402 | 474 | 402 |
| halfnormal_D2 | 474 | 402 | 474 | 402 |

Summed run wall: ref 6.41 h, new 4.71 h (ratio 0.74). Process high-water mark: ref 474 MB, new 402 MB.
