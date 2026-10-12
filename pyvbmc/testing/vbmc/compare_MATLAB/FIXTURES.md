# Active importance sampling fixtures

Reference outputs of the MATLAB VBMC toolbox for the three pieces of the noisy
acquisition path, stored as plain NumPy arrays
(`np.load(path, allow_pickle=False)`). Each `.npz` has a `.m` script beside it
that documents how the numbers were made: the script sets up the same
variational posterior, GP and candidate points as the Python test, calls the
MATLAB function, and saves its outputs under the key names used below.
Rerunning one needs MATLAB and the VBMC toolbox, so treat the numbers as
fixed.

All three states are `D = 3` with a `K = 2` mixture, a GP fitted on five
points with two hyperparameter samples, and three candidate points `Xa`.
All three have been in the repository since 2022-11-23.

## `fess.npz` (`fess.m`, `fess_vbmc`)

Fractional effective sample size of the importance-sampling weights.

| key | shape | contents |
| --- | --- | --- |
| `fess_means` | (1, 1) | fESS from a matrix of GP mean predictions |
| `fess_gp` | (1, 1) | fESS from the GP itself |

`test_active_importance_sampling.py::test_fess` checks both at NumPy's default
`isclose` tolerance.

## `activesample_proposalpdf.npz` (`activesample_proposalpdf.m`)

Importance weights and predictive variances for the proposal mixture of the
variational posterior and box-uniforms around the training points, for both
noisy acquisitions. Columns are the two GP hyperparameter samples. The
script writes `activesample_proposalpdf.mat`, to be converted to this
`.npz` with the same keys.

| key | shape | contents |
| --- | --- | --- |
| `ln_weights_viqr` | (3, 2) | log importance weights, VIQR |
| `f_s2_viqr` | (3, 2) | GP predictive variance at `Xa`, VIQR |
| `ln_weights_imiqr` | (3, 2) | log importance weights, IMIQR |
| `f_s2_imiqr` | (3, 2) | GP predictive variance at `Xa`, IMIQR |

`test_active_importance_sampling.py::test_active_sample_proposal_pdf` checks
all four at NumPy's default `allclose` tolerance.

## `log_isbasefun.npz` (`log_isbasefun.m`)

The base importance-sampling proposal log pdf of each acquisition, evaluated
at the three candidate points.

| key | shape | contents |
| --- | --- | --- |
| `y_viqr` | (3, 1) | log proposal density, VIQR, with the variational density added |
| `y_imiqr` | (3, 1) | log proposal density, IMIQR |

`test_active_importance_sampling.py::test_acq_log_f` checks both at
`atol=1e-3`.
