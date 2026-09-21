# Whitening fixtures

Comma-delimited text, read with `np.loadtxt(path, delimiter=",")`: the
inputs and the MATLAB reference outputs of the tests of `warp_input` and
`warp_gp_and_vp` in `test_rotoscaling.py`, in `D = 2`. In the repository
since 2022-03-03 (`4aa4ca97`). MATLAB is needed to make the reference
outputs again, and the script that produced them is not in the repository,
so treat the numbers as fixed.

## Inputs

| file | shape | contents |
| --- | --- | --- |
| `test_warp_input_rands.txt` | (50, 2) | the random numbers from which the tests build the means of a `K = 50` posterior: the first column is multiplied by 10 and the rows are rotated by a fixed angle, so the posterior is elongated along a rotated axis |
| `test_warp_gp_and_vp_gp_X.txt` | (65, 2) | training inputs of the GP that is warped |
| `test_warp_gp_and_vp_gp_y.txt` | (65,) | its training targets |
| `test_warp_gp_and_vp_gp_hyps.txt` | (9,) | its hyperparameters, one sample (squared-exponential ARD covariance, constant Gaussian noise, negative-quadratic mean) |

`test_warp_input_rands.txt` is read by `test_warp_input`,
`test_warp_input_cov_reg`, `test_warp_input_search_cache`,
`test_warp_input_rewrites_every_filled_row_of_the_logger`,
`test_warp_input_inverts_the_search_state_with_the_current_transform` and
`test_warp_gp_and_vp`; the three GP files by `test_warp_gp_and_vp` alone.

## MATLAB reference outputs of `test_warp_gp_and_vp`

| file | shape | contents | tolerance |
| --- | --- | --- | --- |
| `test_warp_gp_and_vp_gp_hyps_new.txt` | (9,) | hyperparameters of the GP after the warp | `allclose` default |
| `test_warp_gp_and_vp_vp_mu.txt` | (2, 50) | means of the warped posterior | `atol=1e-5` |
| `test_warp_gp_and_vp_vp_sigma.txt` | (50,) | its scales | `allclose` default |
| `test_warp_gp_and_vp_vp_lambda.txt` | (2,) | its length scales | `allclose` default |
| `test_warp_gp_and_vp_vp_w.txt` | (50, 50) | its weights: every entry is 0.02, and the test compares the `(1, 50)` weights with it by broadcasting | `allclose` default |
| `test_warp_gp_and_vp_vp_K.txt` | scalar | its number of components, 50 | `allclose` default |

## Not read by any test

`test_warp_input_R_mat.txt` (2, 2) and `test_warp_input_scale.txt` (2,) hold
the rotation matrix and the scales of the warp of that posterior. No test
reads them: `test_warp_input` and `test_warp_gp_and_vp` compare against the
same values written as literals.
