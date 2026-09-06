# Variational posterior fixtures

Values produced by the MATLAB VBMC toolbox and stored as plain NumPy arrays
(`np.load(path, allow_pickle=False)`). MATLAB is needed to make them again,
and the scripts that produced them are not in the repository, so treat the
numbers as fixed.

## `vp-test.npz`

A variational posterior from a MATLAB run in `D = 2` with `K = 50`
components: an input state, not a reference output. In the repository since
2022-11-23.

| key | shape | contents |
| --- | --- | --- |
| `D` | (1, 1) int | number of dimensions, 2 |
| `K` | (1, 1) int | number of mixture components, 50 |
| `mu` | (2, 50) | component means |
| `sigma` | (1, 50) | component scales |
| `lambd` | (2, 1) | per-dimension length scales |
| `w` | (1, 50) | mixture weights |
| `optimize_mu` | (1, 1) int | 1 if the means were being optimized, 0 if held fixed |
| `optimize_sigma` | (1, 1) int | likewise for the scales |
| `optimize_lambd` | (1, 1) int | likewise for the length scales |
| `optimize_weights` | (1, 1) int | likewise for the weights |

All four flags are 1 here.

`get_matlab_vp()` in `test_variational_posterior.py` rebuilds a
`VariationalPosterior` from these arrays with a default
`ParameterTransformer`. Two tests use it, `test_mode_no_orig_flag` and
`test_mode_orig_flag`; both compare `vp.mode()` against the mode MATLAB found
for the same mixture, `[0.0540, -0.1818]`, at `atol=1e-4`. That pinned mode
lives in the test, not in this file.

## `test_moments_no_orig_flag_2_MATLAB.npz`

Mean and covariance MATLAB computes for a `D = 6`, `K = 3` mixture with
non-unit `lambda`. The mixture itself is written out in the test; only the
moments are stored here. In the repository since 2022-11-23.

| key | shape | contents |
| --- | --- | --- |
| `mubar` | (1, 6) | mean of the mixture |
| `sigma` | (6, 6) | covariance of the mixture |

`test_variational_posterior.py::test_moments_no_orig_flag_2` checks both at
NumPy's default `allclose` tolerance; in the transformed space the moments are
analytic, so the two implementations agree closely.
