# PyMC adapter feasibility check

Generated 2026-09-14 13:45:38; PyMC 6.3.2, PyTensor 3.3.1 (C++ compiler '', mode Mode, floatX float64), ArviZ 1.3.0, Python 3.12.6. Starting point and plausible box: `laplace` (prior quantiles from the model's initial point, or the mode and its Laplace box).

| Model | D | VBMC coordinates and bounds | density (offset; max |Δ − offset|) | Jacobian increment | ELBO − ln Z (± sd) | evaluations | seconds | return path |
|---|---:|---|---:|---|---:|---:|---:|---|
| scalar | 1 | mu [-inf, inf] | +2.2e-07; 3.0e-14 ok | n/a (no transform kept) | +0.000 (± 0.000) | 65 | 4 | mu ['chain', 'draw'] [1, 2000]; predictive [1, 2000, 20] |
| positive | 1 | sigma_log__ (LogTransform kept, unbounded) | -1.9e-09; 2.7e-14 ok | 1.5e-14 ok | -0.002 (± 0.001) | 65 | 3 | sigma ['chain', 'draw'] [1, 2000]; predictive [1, 2000, 15] |
| vector | 3 | beta [-inf, inf]; sigma_log__ (LogTransform kept, unbounded) | -6.2e-08; 4.2e-12 ok | 3.9e-12 ok | -0.007 (± 0.001) | 65 | 14 | beta ['chain', 'draw', 'coef'] [1, 2000, 2]; sigma ['chain', 'draw'] [1, 2000]; mu [1, 2000, 30] (max |Δ| from X beta 0.0e+00); predictive [1, 2000, 30] |

## Rejections

- `discrete`: rejected: k is int64: VBMC needs continuous parameters
- `simplex`: rejected: w: the support of a variable with a SimplexTransform is not a box of independent bounds, which VBMC cannot represent
