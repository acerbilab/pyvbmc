# PyMC adapter feasibility check

Generated 2026-09-14 13:37:54; PyMC 6.3.2, PyTensor 3.3.1 (C++ compiler '', mode Mode, floatX float64), ArviZ 1.3.0, Python 3.12.6.

| Model | D | VBMC coordinates and bounds | density (offset; max |Δ − offset|) | Jacobian increment | ELBO − ln Z (± sd) | evaluations | seconds | return path |
|---|---:|---|---:|---|---:|---:|---:|---|
| scalar | 1 | mu [-inf, inf] | +2.2e-07; 3.0e-14 ok | n/a (no transform kept) | not run | - | - | not run |
| positive | 1 | sigma_log__ (LogTransform kept, unbounded) | -1.9e-09; 2.7e-14 ok | 1.5e-14 ok | not run | - | - | not run |
| vector | 3 | beta [-inf, inf]; sigma_log__ (LogTransform kept, unbounded) | -6.2e-08; 4.2e-12 ok | 3.9e-12 ok | not run | - | - | not run |

## Rejections

- `discrete`: rejected: k is int64: VBMC needs continuous parameters
- `simplex`: rejected: w: the support of a variable with a SimplexTransform is not a box of independent bounds, which VBMC cannot represent
