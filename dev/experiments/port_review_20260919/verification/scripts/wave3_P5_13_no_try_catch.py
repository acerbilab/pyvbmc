"""P5-13 (P5c F15): no counterpart of MATLAB's try/catch around the
starting points.

MATLAB `misc/gptrain_vbmc.m:37-50` wraps the whole block that gathers
historical hyperparameter vectors and concatenates them with
`hypstruct.hyp` in `try ... catch hyp0 = hypstruct.hyp; end`, so a
recorded GP of a different shape degrades to "use the summary vector
only". This script shows what PyVBMC does in that state (an unguarded
`np.concatenate`), and checks the routes by which a model of a different
shape could reach `iteration_history["gp"]` today.
"""

import gpyreg as gpr
import numpy as np

import pyvbmc
from pyvbmc import VBMC
from pyvbmc.vbmc.gaussian_process_train import _lean_gp, train_gp

print("pyvbmc.__file__ =", pyvbmc.__file__)
print("gpyreg.__file__ =", gpr.__file__)

D = 2
N = 8
f = lambda x: -np.sum(np.atleast_2d(x) ** 2, axis=1)
v = VBMC(
    f,
    np.zeros((1, D)),
    None,
    None,
    np.full((1, D), -1.0),
    np.full((1, D), 1.0),
    {"gp_train_n_init": 16, "gp_train_n_init_final": 8},
)
rng = np.random.default_rng(0)
window = v.optim_state["pub_tran"] - v.optim_state["plb_tran"]
Xs = window * rng.random((N, D)) + v.optim_state["plb_tran"]
for i in range(N):
    v.function_logger.X_flag[i] = True
    v.function_logger.X[i] = Xs[i]
    v.function_logger.y[i] = f(Xs[i])
    v.function_logger.fun_eval_time[i] = 1e-5
v.optim_state["N"] = N
v.optim_state["n_eff"] = N

# A recorded GP of a *different* model: constant mean (D+3 = 5
# hyperparameters at D = 2) where the live model is negative-quadratic
# (3D+3 = 9).
gp_const = gpr.GP(
    D=D,
    covariance=gpr.covariance_functions.SquaredExponential(),
    mean=gpr.mean_functions.ConstantMean(),
    noise=gpr.noise_functions.GaussianNoise(constant_add=True),
)
gp_const.update(X_new=Xs, y_new=v.function_logger.y[:N], hyp=np.zeros((1, 5)))
print(
    "recorded GP hyperparameter count :",
    gp_const.get_hyperparameters(as_array=True).shape,
)

v.optim_state["iter"] = 1
v.iteration_history.record("gp", _lean_gp(gp_const), 0)
v.iteration_history.record("r_index", np.inf, 0)
v.iteration_history.record("gp_hyp_full", np.zeros((1, 9)), 0)
v.iteration_history.record("sKL", 1.0, 0)
hyp_dict = {"hyp": np.zeros((1, 9))}
try:
    train_gp(
        hyp_dict,
        v.optim_state,
        v.function_logger,
        v.iteration_history,
        v.options,
        v.optim_state["plb_tran"],
        v.optim_state["pub_tran"],
        rng=0,
    )
except Exception as exc:  # noqa: BLE001
    print("train_gp ->", type(exc).__name__ + ":", str(exc).split("\n")[0])
print(
    "MATLAB in the same state: the catch sets hyp0 = hypstruct.hyp and"
    " the fit proceeds."
)

# Which routes could produce such a record in a live run?
print("\nroutes that could change the hyperparameter count mid-run:")
print(
    "  optim_state['gp_mean_fun'] is set once in _init_optim_state and"
    " never rewritten (grep):"
)
import subprocess

out = subprocess.run(
    [
        "git",
        "grep",
        "-n",
        "gp_mean_fun",
        "--",
        "pyvbmc/vbmc",
        "pyvbmc/whitening",
    ],
    cwd=r"C:\Users\luigi\Documents\GitHub\pyvbmc",
    capture_output=True,
    text=True,
).stdout
for line in out.splitlines():
    print("   ", line)
