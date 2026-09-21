# VBMC test fixtures

The files directly under this directory. `compare_MATLAB/` has a
`FIXTURES.md` of its own.

## `X.txt`, `y.txt`, `hyp.txt`, `mu.txt`

A small GP and the means of a variational posterior, in `D = 2`: the state
on which MATLAB computed the two gradients below and the scalar values
written in the tests. Inputs, not reference outputs. Comma-delimited text,
read with `np.loadtxt(path, delimiter=",")`.

| file | shape | contents |
| --- | --- | --- |
| `X.txt` | (10, 2) | training inputs of the GP |
| `y.txt` | (10,) | training targets |
| `hyp.txt` | (8, 9) | eight hyperparameter samples, one per row, of a GP with the squared-exponential ARD covariance, constant Gaussian noise and the negative-quadratic mean |
| `mu.txt` | (2, 2) | means of a posterior with `K = 2` components, one per column |

`X.txt` and `mu.txt` are byte for byte the files of the same names under
`../variational_posterior/`.

`test_variational_optimization.py` builds the state in `test_gp_log_joint`,
`_gp_log_joint_fixture` (shared by the tests of `_gp_log_joint` that follow
it), `test_neg_elcbo` and `test_vp_bound_loss`;
`test_variational_optimization_grad_fd.py` builds its finite-difference
checks on the same files, and
`test_variational_optimization_single_sample.py` takes the first row of
`hyp.txt`.

## `dG_gp_log_joint.txt`, `dF.txt`

Gradients computed by the MATLAB VBMC toolbox on the state above, with
respect to the ten variational parameters (`mu`, `sigma`, `lambd` and the
weights of a `D = 2`, `K = 2` posterior).

| file | shape | contents | read by |
| --- | --- | --- | --- |
| `dG_gp_log_joint.txt` | (10,) | gradient of the expected log joint | `test_variational_optimization.py::test_gp_log_joint` |
| `dF.txt` | (10,) | gradient of the negative ELCBO | `test_variational_optimization.py::test_neg_elcbo` |

Both are compared at NumPy's default `allclose` tolerance. The scalar values
MATLAB found on the same state (the expected log joint, its variance, the
negative ELCBO and the entropy) are written in those two tests, not stored
here.

The six text files have been in the repository since 2021-08-25 (`9267816c`,
as `.dat` files; `.txt` since 2022-06-28, in this directory since
2022-11-23). MATLAB is needed to make the gradients again, and the script
that produced them is not in the repository, so treat the numbers as fixed.

## `test_vbmc_save_static.pkl`

A finished `VBMC` run written by `VBMC.save` on 2023-03-08 (`ad96d503`): an
unbounded target in `D = 4` with plausible bounds at plus and minus the
square root of 3, `max_fun_evals = 40`, stopped at iteration 6. What the
file is for is its age: it lacks every attribute and option that `VBMC`,
its options and its iteration history have gained since, so it is the file
on which the tests check that `VBMC.load` and the code after it cope with
an old file: a missing `vectorized_target` reads as False, the recorded
random state is NumPy's global state alone, the recorded GPs carry their
posterior factors, and `optim_state` holds the key names of its day
(`variance_regularized_acqfcn`). A run saved by the current code cannot
replace it.

Load it, never save it again. It holds its target function and PyVBMC's
log-joint wrapper pickled by value, as bytecode of the Python version that
wrote it, and pickling such a function again makes dill disassemble it,
which corrupts memory on Python 3.11: a segmentation fault in the test that
saves, in a later garbage collection, or when the interpreter exits. The one
`save` of this object in the suite
(`test_vbmc_save_and_load.py::test_vbmc_save_load_error_handling`) targets
the existing file, so it raises `FileExistsError` before anything is
pickled. A test that needs a saved run with history makes a short run of its
own, as
`test_vbmc_save_and_load.py::test_load_prefers_the_iterations_map_over_the_saved_live_one`
does.

Read by `test_vbmc_save_and_load.py`, `test_gp_records.py`,
`test_vbmc_seed.py`, `test_vbmc_summary.py` and
`../acquisition_functions/test_acq_legacy_save.py`.
