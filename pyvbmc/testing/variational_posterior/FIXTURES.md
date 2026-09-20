# Variational posterior fixtures

Values produced by the MATLAB VBMC toolbox and stored as plain NumPy arrays
(`np.load(path, allow_pickle=False)`). MATLAB is needed to make them again,
and the scripts that produced them are not in the repository, so treat the
numbers as fixed. The last section describes two pickled posteriors, which
did not come from MATLAB.

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

## `test_vp_save_bounded_py311.pkl`, `test_vp_save_bounded_py312.pkl`

Two pickled `VariationalPosterior` objects, the same posterior written
under Python 3.11.9 and under Python 3.12.6 on 2026-09-20 by PyVBMC at
commit `e3ccd0e`, the last one whose `ParameterTransformer` pickled its
bounded transforms. Each file therefore holds three functions by
value, as bytecode of the Python version that wrote it, which is what the
files are for: whichever interpreter runs the tests, at least one of them
carries bytecode of another Python version. They cannot be made again with
the current code, which no longer stores those functions; an older checkout
and the two interpreters are needed
(`dev/experiments/port_review_20260919/verification/scripts/wave2_xver_make_fixture.py`
is the script that wrote them).

The posterior has `D = 2` and `K = 3`. The first variable has hard bounds 0
and 10 and plausible bounds 2 and 6, the second is unbounded with plausible
bounds -1 and 1, and the bounded transform is the default (probit). In
transformed coordinates `mu = [[-0.8, 0.1, 0.9], [-0.5, 0.0, 0.6]]`,
`sigma = [[0.3, 0.2, 0.4]]`, `lambd = [[1.1], [0.9]]` and
`w = [[0.2, 0.5, 0.3]]`; the generator was seeded with 123.
`_bounded_reference()` in `test_vp_save_and_load.py` builds the same
posterior under the running interpreter.

`test_vp_save_and_load.py::test_vp_saved_under_another_python_version_is_usable`
loads each file, samples from it, compares its density (`rtol=1e-12`) and
its transformed-space moments with the reference, saves it again and checks
that the new file holds no function. Before the transformer rebuilt its
transforms on restoring, sampling from the file of another Python version
ended the interpreter ("Illegal instruction" or a segmentation fault).
