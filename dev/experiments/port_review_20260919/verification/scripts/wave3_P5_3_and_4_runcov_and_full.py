"""P5-3 and P5-4: the running hyperparameter covariance block of `train_gp`.

Settles:
  * P5-3 (P5i F2, and the same remark inside P5c F8): the guard
    `hyp_dict["full"].shape[1] > 1` tests the number of hyperparameters,
    not the number of sampled rows, so it is true for every model PyVBMC
    builds and the `else` branch that sets `run_cov = None` is dead.
  * P5-4 (P5c F8): when the fit does not sample, `GP.fit` returns
    `sampling_result is None`, `hyp_dict["full"]` keeps the previous
    iteration's chain, and `run_cov` goes on folding it instead of being
    cleared as MATLAB's `hypstruct.runcov = []` does.

Two `train_gp` calls on one 12-point synthetic state: the first with
sampling on, the second with `optim_state["stop_sampling"]` set so that
`gp_s_N` becomes `stable_gp_samples = 0`.
"""

import gpyreg as gpr
import numpy as np

import pyvbmc
from pyvbmc import VBMC
from pyvbmc.vbmc.gaussian_process_train import train_gp

print("pyvbmc.__file__ =", pyvbmc.__file__)
print("gpyreg.__file__ =", gpr.__file__)

D = 2
N = 12
f = lambda x: -np.sum((np.atleast_2d(x) + 1.0) ** 2, axis=1)
options = {"gp_train_n_init": 32, "gp_train_n_init_final": 16}
vbmc = VBMC(
    f,
    np.zeros((1, D)),
    None,
    None,
    np.full((1, D), -1.0),
    np.full((1, D), 1.0),
    options,
)
rng = np.random.default_rng(0)
window = vbmc.optim_state["pub_tran"] - vbmc.optim_state["plb_tran"]
Xs = window * rng.random((N, D)) + vbmc.optim_state["plb_tran"]
ys = f(Xs)
for i in range(N):
    vbmc.function_logger.X_flag[i] = True
    vbmc.function_logger.X[i] = Xs[i]
    vbmc.function_logger.y[i] = ys[i]
    vbmc.function_logger.fun_eval_time[i] = 1e-5
vbmc.optim_state["N"] = N
vbmc.optim_state["n_eff"] = N

hyp_dict = {}
gp, gp_s_N, _, hyp_dict = train_gp(
    hyp_dict,
    vbmc.optim_state,
    vbmc.function_logger,
    vbmc.iteration_history,
    vbmc.options,
    vbmc.optim_state["plb_tran"],
    vbmc.optim_state["pub_tran"],
    rng=1,
)
full1 = hyp_dict["full"]
cov1 = hyp_dict["run_cov"]
print("\ncall 1 (sampling on):")
print("  gp_s_N              =", gp_s_N)
print(
    "  hyp_dict['full'].shape =",
    full1.shape,
    " (rows = pre-thinning samples, cols = hyperparameters)",
)
print("  guard as written  full.shape[1] > 1 ->", full1.shape[1] > 1)
print("  guard on rows     full.shape[0] > 1 ->", full1.shape[0] > 1)
print("  run_cov is None   ->", cov1 is None, " shape", np.shape(cov1))

# Second call: no sampling.
vbmc.optim_state["stop_sampling"] = N
gp, gp_s_N, _, hyp_dict = train_gp(
    hyp_dict,
    vbmc.optim_state,
    vbmc.function_logger,
    vbmc.iteration_history,
    vbmc.options,
    vbmc.optim_state["plb_tran"],
    vbmc.optim_state["pub_tran"],
    rng=2,
)
full2 = hyp_dict["full"]
cov2 = hyp_dict["run_cov"]
print("\ncall 2 (stop_sampling set, gp_s_N = stable_gp_samples):")
print("  gp_s_N                    =", gp_s_N)
print("  hyp_dict['full'] is the same object as before ->", full2 is full1)
print(
    "  hyp_dict['full'] unchanged                    ->",
    np.array_equal(full2, full1),
)
print("  run_cov is None ->", cov2 is None, " shape", np.shape(cov2))
print(
    "  run_cov changed between the two calls ->",
    not np.array_equal(cov1, cov2),
)
print(
    "  MATLAB at this point: hypstruct.full = hyp (one column) and"
    " hypstruct.runcov = []"
)

# What the guard would do on a one-row sample matrix (the shape MATLAB's
# no-sampling case produces), if `full` were written unconditionally.
one_row = np.zeros((1, full1.shape[1]))
with np.errstate(all="ignore"):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = np.cov(one_row.T)
print("\none-row sample matrix, shape", one_row.shape)
print(
    "  guard as written ->",
    one_row.shape[1] > 1,
    "; np.cov(...) all-nan ->",
    bool(np.all(np.isnan(c))),
)
print("  guard on rows    ->", one_row.shape[0] > 1)
