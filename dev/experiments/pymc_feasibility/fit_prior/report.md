# PyMC adapter feasibility check

Generated 2026-09-14 14:31:57; PyMC 6.3.2, PyTensor 3.3.1 (C++ compiler '', mode Mode, floatX float64), ArviZ 1.3.0, Python 3.12.6; PyVBMC commit 0418351, gpyreg commit 9e70e6b, script 03b278f7faee. Starting point and plausible box: `prior` (the model's initial point and prior quantiles).

| Model | D | VBMC coordinates and bounds | density: offset; max abs deviation | Jacobian increment | ms per call | ELBO − ln Z (± sd) | evaluations | setup + fit seconds | return path |
|---|---:|---|---:|---|---:|---:|---:|---:|---|
| scalar | 1 | mu [-inf, inf] | +2.2e-07; 3.0e-14 ok | n/a (no transform kept) | 0.06 | -0.0005 (± 0.0001) | 65 | 0.1 + 3.2 | mu ['chain', 'draw'] [1, 2000]; predictive [1, 2000, 20] |
| positive | 1 | sigma_log__ (LogTransform kept, unbounded) | -1.9e-09; 8.5e-14 ok | 1.6e-14 ok | 0.01 | -0.0009 (± 0.0008) | 65 | 0.1 + 5.4 | sigma ['chain', 'draw'] [1, 2000]; predictive [1, 2000, 15] |
| vector | 3 | beta [-inf, inf] [-inf, inf]; sigma_log__ (LogTransform kept, unbounded) | -6.2e-08; 1.7e-11 ok | 8.7e-13 ok | 0.01 | not run | - | - | not run |
| bounded | 3 | p [0, 1]; u [-1, 2] [0, 1] | +2.2e-08; 1.4e-14 ok | n/a (no transform kept) | 0.10 | -0.0195 (± 0.0027) | 75 | 0.1 + 16.3 | p ['chain', 'draw'] [1, 2000]; u ['chain', 'draw', 'u_dim_0'] [1, 2000, 2]; predictive [1, 2000, 12] |

## Rejections

- `discrete`: rejected: k is int64: VBMC needs continuous parameters
- `simplex`: rejected: w: unsupported transform SimplexTransform
- `suppressed`: rejected: s: its LogTransform was suppressed at construction, so its support is not the unbounded one an untransformed variable would be handed over with
- `random_bounds`: rejected: h: an interval limit depends on another random variable, so it has no fixed value

## Errors

- vector: `LinAlgError: Singular matrix for L Cholesky decomposition`
  - box: plb [-8.20201003323745, -8.320048914579257, -2.0003850210341514], pub [8.334855409766911, 8.309412013519617, 1.362617713772141], x0 [0.0, 0.0, 0.6931471805599453]; log joint at plb, pub, centre, x0: [-117368.98206922936, -241.99976900921317, -65.92348299995568, -59.786392195968496]
