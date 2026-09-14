# PyMC adapter feasibility check

Generated 2026-09-14 13:46:10; PyMC 6.3.2, PyTensor 3.3.1 (C++ compiler '', mode Mode, floatX float64), ArviZ 1.3.0, Python 3.12.6. Starting point and plausible box: `prior` (prior quantiles from the model's initial point, or the mode and its Laplace box).

| Model | D | VBMC coordinates and bounds | density (offset; max |Δ − offset|) | Jacobian increment | ELBO − ln Z (± sd) | evaluations | seconds | return path |
|---|---:|---|---:|---|---:|---:|---:|---|
| scalar | 1 | mu [-inf, inf] | +2.2e-07; 3.0e-14 ok | n/a (no transform kept) | -0.001 (± 0.000) | 65 | 3 | mu ['chain', 'draw'] [1, 2000]; predictive [1, 2000, 20] |
| positive | 1 | sigma_log__ (LogTransform kept, unbounded) | -1.9e-09; 8.5e-14 ok | 1.6e-14 ok | -0.001 (± 0.001) | 65 | 6 | sigma ['chain', 'draw'] [1, 2000]; predictive [1, 2000, 15] |
| vector | 3 | beta [-inf, inf]; sigma_log__ (LogTransform kept, unbounded) | -6.2e-08; 1.7e-11 ok | 8.7e-13 ok | not run | - | - | not run |

## Rejections

- `discrete`: rejected: k is int64: VBMC needs continuous parameters
- `simplex`: rejected: w: the support of a variable with a SimplexTransform is not a box of independent bounds, which VBMC cannot represent

## Errors

- vector: `LinAlgError: Singular matrix for L Cholesky decomposition`
