# PyMC adapter feasibility check

Generated 2026-09-14 14:55:31; PyMC 6.3.2, PyTensor 3.3.1 (C++ compiler '', mode Mode, floatX float64), ArviZ 1.3.0, Python 3.12.6; PyVBMC commit 59e9a52, gpyreg commit 9e70e6b, script f2ade815a503. Starting point and plausible box: `prior` (the model's initial point and prior quantiles).

| Model | D | VBMC coordinates and bounds | density: offset; max abs deviation | Jacobian increment | ms per call | ELBO − ln Z (± sd) | evaluations | setup + fit seconds | return path |
|---|---:|---|---:|---|---:|---:|---:|---:|---|
| scalar | 1 | mu [-inf, inf] | +2.2e-07; 3.6e-14 ok | n/a (no transform kept) | 0.13 | +0.0017 (± 0.0001) | 65 | 0.1 + 4.7 | mu ['chain', 'draw'] [1, 2000]; predictive [1, 2000, 20] |
| positive | 1 | sigma_log__ (LogTransform kept, unbounded) | -1.9e-09; 5.7e-14 ok | 1.4e-14 ok | 0.03 | -0.0015 (± 0.0008) | 65 | 0.1 + 4.4 | sigma ['chain', 'draw'] [1, 2000]; predictive [1, 2000, 15] |
| vector | 3 | beta [-inf, inf] [-inf, inf]; sigma_log__ (LogTransform kept, unbounded) | -6.2e-08; 3.9e-11 ok | 8.6e-13 ok | 0.03 | not run | - | - | not run |
| bounded | 3 | p [0, 1]; u [-1, 2] [0, 1] | +2.2e-08; 1.5e-14 ok | n/a (no transform kept) | 0.09 | -0.0035 (± 0.0031) | 80 | 0.2 + 30.3 | p ['chain', 'draw'] [1, 2000]; u ['chain', 'draw', 'u_dim_0'] [1, 2000, 2]; predictive [1, 2000, 12] |
| one_sided | 2 | t_interval__ (Interval kept, unbounded); v_interval__ (Interval kept, unbounded) | -8.0e-09; 8.2e-15 ok | 6.2e-15 ok | 0.01 | -0.0043 (± 0.0007) | 65 | 4.3 + 17.3 | t ['chain', 'draw'] [1, 2000]; v ['chain', 'draw'] [1, 2000]; predictive [1, 2000, 8] |

## Rejections

- `discrete`: rejected: k is int64: VBMC needs continuous parameters
- `simplex`: rejected: w: unsupported transform SimplexTransform
- `suppressed`: rejected: s: its LogTransform was suppressed at construction, so its support is not the unbounded one an untransformed variable would be handed over with
- `random_bounds`: rejected: h: an interval limit depends on another random variable, so it has no fixed value

## Errors

- vector: `LinAlgError: Singular matrix for L Cholesky decomposition`
  - box: plb [-8.213450998360939, -8.120748330780634, -2.038767026688617], pub [8.443999714852756, 8.112010999543598, 1.3664816011298409], x0 [0.0, 0.0, 0.6931471805599453]; log joint at plb, pub, centre, x0: [-93826.45689869346, -201.04937113089022, -89.6607696514716, -63.29352234618897]
