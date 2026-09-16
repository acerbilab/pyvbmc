# PyMC setup reuse: measured results

Medians over completed cases; failures are counted separately.

| Model | Arm | Completed | Reused rows | Absolute ELBO error | Gaussian sym. KL | Mean marginal W1 / SD | Reached stability | Seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| vector | discard | 5/5 | 0 | 0.003653 | 0.0004659 | 0.01134 | 5/5 | 45.17 |
| vector | f_vals | 5/5 | 7 | 0.003332 | 0.000576 | 0.0117 | 5/5 | 44.82 |
| vector | logger | 5/5 | 11 | 0.003498 | 0.0006952 | 0.01434 | 4/5 | 51.84 |
| schools_noncentered | discard | 5/5 | 0 | 0.9448 | 1.642 | 0.2319 | 0/5 | 160.6 |
| schools_noncentered | f_vals | 5/5 | 13 | 0.7565 | 3.961 | 0.439 | 0/5 | 171.7 |
| schools_noncentered | logger | 5/5 | 14 | 0.5765 | 0.6358 | 0.1302 | 0/5 | 165.2 |
| ill_conditioned | discard | 5/5 | 0 | 3.52 | 4.602 | 0.4323 | 0/5 | 21 |
| ill_conditioned | f_vals | 5/5 | 15 | 3.424 | 4.316 | 0.4224 | 0/5 | 24.83 |
| ill_conditioned | logger | 5/5 | 27 | 2.945 | 3.053 | 0.3758 | 0/5 | 29.08 |

## Individual cases

Completed fits may still be unconverged; inspect the stability column and the full error range.

| Model | Seed | Arm | ELBO error | Gaussian sym. KL | Mean marginal W1 / SD | First stable iteration | Seconds |
|---|---:|---|---:|---:|---:|---:|---:|
| vector | 0 | discard | 0.0006605 | 0.0004759 | 0.01001 | 13 | 43.72 |
| vector | 0 | f_vals | -0.003332 | 0.000576 | 0.01102 | 14 | 44.82 |
| vector | 0 | logger | -0.003498 | 0.0003314 | 0.01036 | 12 | 34.43 |
| vector | 1 | discard | 0.002002 | 0.0007376 | 0.01942 | 12 | 39.7 |
| vector | 1 | f_vals | -0.01005 | 0.0009991 | 0.01216 | 14 | 36.31 |
| vector | 1 | logger | 0.001555 | 0.0006952 | 0.009582 | 12 | 57.1 |
| vector | 2 | discard | 0.005261 | 0.0002341 | 0.01206 | 13 | 45.17 |
| vector | 2 | f_vals | 0.001692 | 0.0008352 | 0.01498 | 14 | 47.64 |
| vector | 2 | logger | -0.005557 | 0.0007795 | 0.01481 | 16 | 51.84 |
| vector | 3 | discard | -0.003653 | 0.0004542 | 0.01074 | 13 | 50.5 |
| vector | 3 | f_vals | -0.004764 | 0.0003047 | 0.0117 | 13 | 42.32 |
| vector | 3 | logger | -0.002804 | 0.0006777 | 0.015 | 17 | 43.68 |
| vector | 4 | discard | -0.008543 | 0.0004659 | 0.01134 | 14 | 53.62 |
| vector | 4 | f_vals | -0.001532 | 0.0004433 | 0.0114 | 13 | 56.95 |
| vector | 4 | logger | 0.004348 | 0.001161 | 0.01434 | — | 57.18 |
| schools_noncentered | 0 | discard | -0.8158 | 1.07 | 0.1423 | — | 176 |
| schools_noncentered | 0 | f_vals | -0.2529 | 0.2473 | 0.1261 | — | 171.7 |
| schools_noncentered | 0 | logger | -0.5765 | 1.094 | 0.13 | — | 172.2 |
| schools_noncentered | 1 | discard | -0.9448 | 1.642 | 0.2677 | — | 160.6 |
| schools_noncentered | 1 | f_vals | -0.3168 | 0.4314 | 0.1457 | — | 186 |
| schools_noncentered | 1 | logger | -0.5917 | 0.6358 | 0.1302 | — | 165.2 |
| schools_noncentered | 2 | discard | -1.364 | 1.628 | 0.2319 | — | 158.2 |
| schools_noncentered | 2 | f_vals | 5.155 | 26.24 | 1.661 | — | 252.9 |
| schools_noncentered | 2 | logger | -0.8735 | 4.191 | 0.2367 | — | 178.2 |
| schools_noncentered | 3 | discard | 5.003e+32 | 2.016e+05 | 40.73 | — | 503.6 |
| schools_noncentered | 3 | f_vals | 1.351 | 4.484 | 0.5266 | — | 86.57 |
| schools_noncentered | 3 | logger | -0.4221 | 0.563 | 0.1213 | — | 69 |
| schools_noncentered | 4 | discard | -0.5808 | 2.317 | 0.2307 | — | 85.18 |
| schools_noncentered | 4 | f_vals | 0.7565 | 3.961 | 0.439 | — | 87.46 |
| schools_noncentered | 4 | logger | -0.1838 | 0.4995 | 0.136 | — | 92.4 |
| ill_conditioned | 0 | discard | -3.516 | 4.602 | 0.4323 | — | 23.69 |
| ill_conditioned | 0 | f_vals | -3.099 | 3.453 | 0.4036 | — | 24.69 |
| ill_conditioned | 0 | logger | -2.945 | 3.053 | 0.3758 | — | 27.69 |
| ill_conditioned | 1 | discard | -3.52 | 4.58 | 0.43 | — | 20.03 |
| ill_conditioned | 1 | f_vals | -3.424 | 4.316 | 0.4224 | — | 24.83 |
| ill_conditioned | 1 | logger | -0.8935 | 0.7564 | 0.1711 | — | 40.27 |
| ill_conditioned | 2 | discard | 4.81 | 414.1 | 1.224 | — | 21 |
| ill_conditioned | 2 | f_vals | -3.091 | 3.808 | 0.4123 | — | 25.52 |
| ill_conditioned | 2 | logger | -3.141 | 3.432 | 0.3908 | — | 23.12 |
| ill_conditioned | 3 | discard | 4.227 | 282.4 | 1.182 | — | 21.15 |
| ill_conditioned | 3 | f_vals | -3.567 | 4.735 | 0.4287 | — | 24.52 |
| ill_conditioned | 3 | logger | -2.899 | 3.013 | 0.3738 | — | 40.39 |
| ill_conditioned | 4 | discard | -3.409 | 3.967 | 0.4067 | — | 20.13 |
| ill_conditioned | 4 | f_vals | -3.469 | 4.592 | 0.4337 | — | 27.51 |
| ill_conditioned | 4 | logger | -3.01 | 3.24 | 0.3825 | — | 29.08 |
