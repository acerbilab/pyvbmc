# PyMC adapter feasibility check

Generated 2026-09-14 14:53:31; PyMC 6.3.2, PyTensor 3.3.1 (C++ compiler '', mode Mode, floatX float64), ArviZ 1.3.0, Python 3.12.6; PyVBMC commit 59e9a52, gpyreg commit 9e70e6b, script f2ade815a503. Starting point and plausible box: `laplace` (the mode from `find_MAP` and its Laplace box).

| Model | D | VBMC coordinates and bounds | density: offset; max abs deviation | Jacobian increment | ms per call | ELBO − ln Z (± sd) | evaluations | setup + fit seconds | return path |
|---|---:|---|---:|---|---:|---:|---:|---:|---|
| scalar | 1 | mu [-inf, inf] | +2.2e-07; 3.6e-14 ok | n/a (no transform kept) | 0.09 | +0.0003 (± 0.0000) | 65 | 0.2 + 5.2 | mu ['chain', 'draw'] [1, 2000]; predictive [1, 2000, 20] |
| positive | 1 | sigma_log__ (LogTransform kept, unbounded) | -1.9e-09; 8.1e-14 ok | 1.3e-14 ok | 0.02 | -0.0008 (± 0.0008) | 65 | 1.0 + 5.1 | sigma ['chain', 'draw'] [1, 2000]; predictive [1, 2000, 15] |
| vector | 3 | beta [-inf, inf] [-inf, inf]; sigma_log__ (LogTransform kept, unbounded) | -6.2e-08; 2.5e-11 ok | 3.3e-12 ok | 0.02 | -0.0037 (± 0.0009) | 65 | 2.4 + 22.9 | beta ['chain', 'draw', 'coef'] [1, 2000, 2]; sigma ['chain', 'draw'] [1, 2000]; mu [1, 2000, 30] (max abs difference from X beta 0.0e+00); predictive [1, 2000, 30] |
| bounded | 3 | p [0, 1]; u [-1, 2] [0, 1] | +2.2e-08; 1.3e-14 ok | n/a (no transform kept) | 0.16 | -0.0073 (± 0.0029) | 75 | 3.7 + 35.6 | p ['chain', 'draw'] [1, 2000]; u ['chain', 'draw', 'u_dim_0'] [1, 2000, 2]; predictive [1, 2000, 12] |
| one_sided | 2 | t_interval__ (Interval kept, unbounded); v_interval__ (Interval kept, unbounded) | -8.0e-09; 8.2e-15 ok | 6.2e-15 ok | 0.02 | -0.0048 (± 0.0008) | 75 | 2.8 + 19.4 | t ['chain', 'draw'] [1, 2000]; v ['chain', 'draw'] [1, 2000]; predictive [1, 2000, 8] |

## Rejections

- `discrete`: rejected: k is int64: VBMC needs continuous parameters
- `simplex`: rejected: w: unsupported transform SimplexTransform
- `suppressed`: rejected: s: its LogTransform was suppressed at construction, so its support is not the unbounded one an untransformed variable would be handed over with
- `random_bounds`: rejected: h: an interval limit depends on another random variable, so it has no fixed value
