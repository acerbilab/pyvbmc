# Matched MCMC budget: exporter

**Headline:** emcee (`W=32`), MMTV: median N* = 27,828 evaluations, 90% interval 23,098 to 31,764 (100 replicates against 100 PyVBMC runs, whose median budget is 85 evaluations). The smallest MMTV N* over slice sampling, emcee and zeus; the interval takes that minimum inside each resample. MCMC gives no estimate of the model evidence, which PyVBMC returns from the same evaluations.

## The reference (PyVBMC)

100 runs (seeds 0 to 99), 0 raised.

| quantity | 25% | median | 75% |
|---|---|---|---|
| evaluations | 80 | 85 | 90 |
| gsKL | 0.121 | 0.171 | 0.236 |
| MMTV | 0.0352 | 0.044 | 0.0558 |
| evidence error `elbo_err` | 0.0349 | 0.0454 | 0.0626 |

## The pilot

Seeds 1000 to 1009; pilot N* against the reference medians, burn-in N/2. The setting chosen for a metric is the one with the smallest pilot N* (ties broken by the mean log median error over the budgets).

**slice sampling**

| setting | runs | gsKL N* | gsKL median at N_max | MMTV N* | MMTV median at N_max | chosen for |
|---|---|---|---|---|---|---|
| `c=0.1,step_out=0,adaptive=0` | 10 | 56,050 | 0.0508 | 247,579 | 0.0397 |  |
| `c=0.1,step_out=0,adaptive=1` | 10 | 19,677 | 0.037 | 66,415 | 0.0244 |  |
| `c=0.1,step_out=1,adaptive=0` | 10 | 24,074 | 0.0495 | 112,468 | 0.0328 |  |
| `c=0.1,step_out=1,adaptive=1` | 10 | 16,296 | 0.0153 | 96,474 | 0.0252 |  |
| `c=0.3,step_out=0,adaptive=0` | 10 | 10,058 | 0.0258 | 83,594 | 0.0251 |  |
| `c=0.3,step_out=0,adaptive=1` | 10 | 9,043 | 0.0115 | 33,306 | 0.0244 | MMTV |
| `c=0.3,step_out=1,adaptive=0` | 10 | 17,210 | 0.00964 | 75,884 | 0.0268 |  |
| `c=0.3,step_out=1,adaptive=1` | 10 | 28,620 | 0.0181 | 85,102 | 0.0245 |  |
| `c=1,step_out=0,adaptive=0` | 10 | 10,873 | 0.012 | 42,040 | 0.0184 |  |
| `c=1,step_out=0,adaptive=1` | 10 | 6,006 | 0.0149 | 48,340 | 0.0185 | gsKL |
| `c=1,step_out=1,adaptive=0` | 10 | 11,865 | 0.015 | 84,698 | 0.023 |  |
| `c=1,step_out=1,adaptive=1` | 10 | 30,390 | 0.0197 | 71,777 | 0.0246 |  |

**emcee**

| setting | runs | gsKL N* | gsKL median at N_max | MMTV N* | MMTV median at N_max | chosen for |
|---|---|---|---|---|---|---|
| `W=4` | 10 | 9,731 | 0.0086 | 44,354 | 0.0211 |  |
| `W=8` | 10 | 1,619 | 0.00419 | 43,983 | 0.0166 |  |
| `W=16` | 10 | 639 | 0.004 | 34,660 | 0.0166 | gsKL |
| `W=32` | 10 | 1,403 | 0.00189 | 33,640 | 0.0176 | MMTV |

**zeus**

| setting | runs | gsKL N* | gsKL median at N_max | MMTV N* | MMTV median at N_max | chosen for |
|---|---|---|---|---|---|---|
| `W=4` | 10 | 7,744 | 0.0192 | 105,007 | 0.0257 |  |
| `W=8` | 10 | 10,678 | 0.0186 | 51,228 | 0.023 |  |
| `W=16` | 10 | 1,368 | 0.0107 | 46,534 | 0.0182 | gsKL |
| `W=32` | 10 | 1,952 | 0.00612 | 42,242 | 0.0198 | MMTV |

**random-walk Metropolis**

| setting | runs | gsKL N* | gsKL median at N_max | MMTV N* | MMTV median at N_max | chosen for |
|---|---|---|---|---|---|---|
| `c=0.05` | 10 | 31,235 | 0.0203 | 88,801 | 0.0286 |  |
| `c=0.1` | 10 | 8,218 | 0.00444 | 33,824 | 0.0183 |  |
| `c=0.2` | 10 | 6,300 | 0.00222 | 26,140 | 0.0152 | MMTV |
| `c=0.4` | 10 | 996 | 0.00358 | 33,283 | 0.0193 | gsKL |

## The matched budget

N* is the smallest budget at which the median error over the replicates falls to PyVBMC's median error (gsKL 0.171, MMTV 0.044), with burn-in N/2; interval: 5th to 95th percentile over 1000 bootstrap resamples (unpaired).

| sampler | setting | metric | replicates | raised | N* | 90% interval | median error at largest budget | largest budget |
|---|---|---|---|---|---|---|---|---|
| slice sampling | `c=1,step_out=0,adaptive=1` | gsKL | 100 | 0 | 14,916 | 10,885 to 18,024 | 0.0102 | 300,000 |
| slice sampling | `c=0.3,step_out=0,adaptive=1` | MMTV | 100 | 0 | 56,902 | 47,246 to 65,988 | 0.0216 | 300,000 |
| emcee | `W=16` | gsKL | 100 | 0 | 1,530 | 1,065 to 3,114 | 0.00328 | 300,000 |
| emcee | `W=32` | MMTV | 100 | 0 | 27,828 | 23,098 to 31,764 | 0.0156 | 300,000 |
| zeus | `W=16` | gsKL | 100 | 0 | 3,498 | 1,530 to 4,642 | 0.0104 | 300,000 |
| zeus | `W=32` | MMTV | 100 | 0 | 51,317 | 39,088 to 61,217 | 0.0211 | 300,000 |
| random-walk Metropolis | `c=0.4` | gsKL | 100 | 0 | 1,896 | 1,735 to 3,504 | 0.00285 | 300,000 |
| random-walk Metropolis | `c=0.2` | MMTV | 100 | 0 | 24,134 | 19,365 to 30,327 | 0.0162 | 300,000 |

## After the runs

- emcee, gsKL: median 0.00328 at 300,000 is below the reference 0.171.
- emcee, MMTV: median 0.0156 at 300,000 is below the reference 0.044.
- random-walk Metropolis, gsKL: median 0.00285 at 300,000 is below the reference 0.171.
- random-walk Metropolis, MMTV: median 0.0162 at 300,000 is below the reference 0.044.
- slice sampling, gsKL: median 0.0102 at 300,000 is below the reference 0.171.
- slice sampling, MMTV: median 0.0216 at 300,000 is below the reference 0.044.
- zeus, gsKL: median 0.0104 at 300,000 is below the reference 0.171.
- zeus, MMTV: median 0.0211 at 300,000 is below the reference 0.044.

## N* with other burn-in fractions

| sampler | metric | burn-in 0.5 N | burn-in 0.1 N | burn-in 0.25 N |
|---|---|---|---|---|
| emcee | gsKL | 1,530 | 1,562 | 1,341 |
| emcee | MMTV | 27,828 | 17,641 | 20,395 |
| random-walk Metropolis | gsKL | 1,896 | 1,295 | 1,540 |
| random-walk Metropolis | MMTV | 24,134 | 14,241 | 17,557 |
| slice sampling | gsKL | 14,916 | 9,916 | 11,264 |
| slice sampling | MMTV | 56,902 | 33,938 | 39,707 |
| zeus | gsKL | 3,498 | 2,809 | 3,062 |
| zeus | MMTV | 51,317 | 29,254 | 32,216 |

## Median error curves (burn-in N/2)

| N | slice sampling gsKL | emcee gsKL | zeus gsKL | random-walk Metropolis gsKL | slice sampling MMTV | emcee MMTV | zeus MMTV | random-walk Metropolis MMTV | exact draws gsKL | exact draws MMTV |
|---|---|---|---|---|---|---|---|---|---|---|
| 100 | inf | 0.401 | inf | 2.79 | inf | 0.356 | inf | 0.386 | 0.0911 | 0.121 |
| 132 | inf | 0.413 | 0.417 | 2 | 0.855 | 0.347 | inf | 0.31 | 0.0769 | 0.106 |
| 174 | 10.3 | 0.388 | 0.426 | 1.48 | 0.589 | 0.339 | inf | 0.282 | 0.0584 | 0.0909 |
| 229 | 3.91 | 0.414 | 0.488 | 1.01 | 0.41 | 0.337 | 0.356 | 0.278 | 0.0416 | 0.0817 |
| 302 | 2.64 | 0.406 | 0.444 | 0.767 | 0.365 | 0.335 | 0.354 | 0.241 | 0.0283 | 0.0709 |
| 398 | 1.8 | 0.374 | 0.366 | 0.612 | 0.309 | 0.315 | 0.318 | 0.22 | 0.0204 | 0.0638 |
| 524 | 1.47 | 0.338 | 0.363 | 0.447 | 0.244 | 0.294 | 0.323 | 0.196 | 0.0154 | 0.0573 |
| 691 | 0.976 | 0.274 | 0.294 | 0.422 | 0.242 | 0.263 | 0.28 | 0.176 | 0.0112 | 0.0521 |
| 910 | 1.34 | 0.244 | 0.261 | 0.371 | 0.209 | 0.23 | 0.254 | 0.161 | 0.00978 | 0.0464 |
| 1,200 | 1 | 0.183 | 0.243 | 0.303 | 0.176 | 0.203 | 0.217 | 0.141 | 0.00713 | 0.0426 |
| 1,581 | 0.722 | 0.169 | 0.216 | 0.227 | 0.165 | 0.166 | 0.191 | 0.128 | 0.00573 | 0.0365 |
| 2,084 | 0.666 | 0.167 | 0.201 | 0.148 | 0.169 | 0.136 | 0.162 | 0.112 | 0.00443 | 0.0334 |
| 2,747 | 0.654 | 0.144 | 0.204 | 0.184 | 0.154 | 0.117 | 0.138 | 0.103 | 0.00315 | 0.03 |
| 3,620 | 0.512 | 0.144 | 0.167 | 0.149 | 0.131 | 0.101 | 0.125 | 0.0903 | 0.00222 | 0.0277 |
| 4,771 | 0.352 | 0.149 | 0.132 | 0.0867 | 0.12 | 0.09 | 0.104 | 0.0852 | 0.00185 | 0.0245 |
| 6,288 | 0.312 | 0.114 | 0.14 | 0.0659 | 0.113 | 0.0777 | 0.101 | 0.0777 | 0.0012 | 0.023 |
| 8,287 | 0.262 | 0.105 | 0.169 | 0.0582 | 0.0928 | 0.0765 | 0.0899 | 0.0693 | 0.00106 | 0.0206 |
| 10,922 | 0.211 | 0.0825 | 0.151 | 0.0562 | 0.0877 | 0.0664 | 0.0818 | 0.0623 | 0.000788 | 0.0183 |
| 14,395 | 0.179 | 0.0692 | 0.11 | 0.0447 | 0.0744 | 0.0595 | 0.0704 | 0.0532 | 0.00053 | 0.0167 |
| 18,972 | 0.122 | 0.0434 | 0.108 | 0.0294 | 0.066 | 0.0546 | 0.0637 | 0.0492 | 0.000415 | 0.0153 |
| 25,004 | 0.102 | 0.0342 | 0.0922 | 0.0271 | 0.0605 | 0.0465 | 0.0596 | 0.0432 | 0.000346 | 0.0137 |
| 32,955 | 0.0828 | 0.0307 | 0.0574 | 0.0209 | 0.0573 | 0.0402 | 0.0521 | 0.0396 | 0.000274 | 0.0123 |
| 43,433 | 0.0677 | 0.0271 | 0.0478 | 0.016 | 0.0505 | 0.0349 | 0.0466 | 0.0351 | 0.000207 | 0.0112 |
| 57,242 | 0.0475 | 0.0223 | 0.0323 | 0.0144 | 0.0438 | 0.0323 | 0.0423 | 0.0318 | 0.000161 | 0.0105 |
| 75,443 | 0.0369 | 0.0158 | 0.0293 | 0.0105 | 0.0383 | 0.0287 | 0.0363 | 0.0271 | 0.000126 | 0.00969 |
| 99,430 | 0.0408 | 0.012 | 0.0204 | 0.0085 | 0.0353 | 0.0258 | 0.0323 | 0.0251 | 9.57e-05 | 0.00895 |
| 131,045 | 0.029 | 0.00739 | 0.0171 | 0.00618 | 0.0295 | 0.0228 | 0.0285 | 0.0222 | 8.09e-05 | 0.00854 |
| 172,711 | 0.0269 | 0.00571 | 0.0152 | 0.00454 | 0.0274 | 0.0212 | 0.0253 | 0.0205 | 5.75e-05 | 0.00789 |
| 227,625 | 0.0155 | 0.00403 | 0.0117 | 0.00454 | 0.0259 | 0.018 | 0.023 | 0.018 | 4.85e-05 | 0.00762 |
| 300,000 | 0.0102 | 0.00328 | 0.0104 | 0.00285 | 0.0216 | 0.0156 | 0.0211 | 0.0162 | 3.55e-05 | 0.0073 |

- Exact draws reach the reference gsKL at N = < 100 (< 100 draws), a bound on any sampler under this burn-in rule.
- Exact draws reach the reference MMTV at N = 1,084 (542 draws), a bound on any sampler under this burn-in rule.

![gsKL](figures/curves_gskl.png)

![MMTV](figures/curves_mmtv.png)

## Provenance

| directory | commit | wall | seeds run | argv |
|---|---|---|---|---|
| mcmc/emcee | v1.0.4-1243-g0d3cc545 | 6.9 min | 100 | `mcmc --config exporter --sampler emcee --setting W=16 --seeds 0-99 --out dev/scripts/runs/matched_mcmc_budget_20260930/mcmc --jobs 4` |
| mcmc/emcee | v1.0.4-1243-g0d3cc545 | 6.1 min | 100 | `mcmc --config exporter --sampler emcee --setting W=32 --seeds 0-99 --out dev/scripts/runs/matched_mcmc_budget_20260930/mcmc --jobs 4` |
| mcmc/exact | v1.0.4-1243-g0d3cc545 | 3.6 min | 100 | `mcmc --config exporter --sampler exact --seeds 0-99 --out dev/scripts/runs/matched_mcmc_budget_20260930/mcmc --jobs 4` |
| mcmc/rwm | v1.0.4-1243-g0d3cc545 | 5.8 min | 100 | `mcmc --config exporter --sampler rwm --setting c=0.4 --seeds 0-99 --out dev/scripts/runs/matched_mcmc_budget_20260930/mcmc --jobs 4` |
| mcmc/rwm | v1.0.4-1243-g0d3cc545 | 5.6 min | 100 | `mcmc --config exporter --sampler rwm --setting c=0.2 --seeds 0-99 --out dev/scripts/runs/matched_mcmc_budget_20260930/mcmc --jobs 4` |
| mcmc/slice | v1.0.4-1243-g0d3cc545 | 13.1 min | 100 | `mcmc --config exporter --sampler slice --setting c=1,step_out=0,adaptive=1 --seeds 0-99 --out dev/scripts/runs/matched_mcmc_budget_20260930/mcmc --jobs 4` |
| mcmc/slice | v1.0.4-1243-g0d3cc545 | 12.0 min | 100 | `mcmc --config exporter --sampler slice --setting c=0.3,step_out=0,adaptive=1 --seeds 0-99 --out dev/scripts/runs/matched_mcmc_budget_20260930/mcmc --jobs 4` |
| mcmc/zeus | v1.0.4-1243-g0d3cc545 | 7.0 min | 100 | `mcmc --config exporter --sampler zeus --setting W=16 --seeds 0-99 --out dev/scripts/runs/matched_mcmc_budget_20260930/mcmc --jobs 4` |
| mcmc/zeus | v1.0.4-1243-g0d3cc545 | 6.0 min | 100 | `mcmc --config exporter --sampler zeus --setting W=32 --seeds 0-99 --out dev/scripts/runs/matched_mcmc_budget_20260930/mcmc --jobs 4` |
| pilot/emcee | v1.0.4-1241-gbad806ec | 3.3 min | 40 | `pilot --config exporter --sampler emcee --out dev/scripts/runs/matched_mcmc_budget_20260930/pilot --jobs 4` |
| pilot/rwm | v1.0.4-1241-gbad806ec | 2.3 min | 40 | `pilot --config exporter --sampler rwm --out dev/scripts/runs/matched_mcmc_budget_20260930/pilot --jobs 4` |
| pilot/slice | v1.0.4-1241-gbad806ec | 12.9 min | 120 | `pilot --config exporter --sampler slice --out dev/scripts/runs/matched_mcmc_budget_20260930/pilot --jobs 4` |
| pilot/zeus | v1.0.4-1241-gbad806ec | 3.0 min | 40 | `pilot --config exporter --sampler zeus --out dev/scripts/runs/matched_mcmc_budget_20260930/pilot --jobs 4` |
| pyvbmc | v1.0.4-1241-gbad806ec | 7.9 min | 100 | `pyvbmc --config exporter --seeds 0-99 --out dev/scripts/runs/matched_mcmc_budget_20260930/pyvbmc --jobs 4` |

Python 3.12.3, Linux-6.18.44-fc-v50-x86_64-with-glibc2.39, 4 CPUs; pyvbmc 1.0.5.dev1239+g695385eb5, gpyreg 1.4.0, numpy 2.5.3, scipy 1.18.1, emcee 3.1.6, zeus-mcmc 2.5.4; threads {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}.
