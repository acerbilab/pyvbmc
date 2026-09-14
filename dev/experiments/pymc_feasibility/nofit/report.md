# PyMC adapter feasibility check

Generated 2026-09-14 14:31:07; PyMC 6.3.2, PyTensor 3.3.1 (C++ compiler '', mode Mode, floatX float64), ArviZ 1.3.0, Python 3.12.6; PyVBMC commit 0418351, gpyreg commit 9e70e6b, script 03b278f7faee. Starting point and plausible box: `laplace` (the mode from `find_MAP` and its Laplace box).

| Model | D | VBMC coordinates and bounds | density: offset; max abs deviation | Jacobian increment | ms per call | ELBO − ln Z (± sd) | evaluations | setup + fit seconds | return path |
|---|---:|---|---:|---|---:|---:|---:|---:|---|
| scalar | 1 | mu [-inf, inf] | +2.2e-07; 3.0e-14 ok | n/a (no transform kept) | 0.06 | not run | - | - | not run |
| positive | 1 | sigma_log__ (LogTransform kept, unbounded) | -1.9e-09; 2.7e-14 ok | 1.5e-14 ok | 0.01 | not run | - | - | not run |
| vector | 3 | beta [-inf, inf] [-inf, inf]; sigma_log__ (LogTransform kept, unbounded) | -6.2e-08; 4.2e-12 ok | 3.9e-12 ok | 0.01 | not run | - | - | not run |
| bounded | 3 | p [0, 1]; u [-1, 2] [0, 1] | +2.2e-08; 6.9e-15 ok | n/a (no transform kept) | 0.07 | not run | - | - | not run |

## Rejections

- `discrete`: rejected: k is int64: VBMC needs continuous parameters
- `simplex`: rejected: w: unsupported transform SimplexTransform
- `suppressed`: rejected: s: its LogTransform was suppressed at construction, so its support is not the unbounded one an untransformed variable would be handed over with
- `random_bounds`: rejected: h: an interval limit depends on another random variable, so it has no fixed value
