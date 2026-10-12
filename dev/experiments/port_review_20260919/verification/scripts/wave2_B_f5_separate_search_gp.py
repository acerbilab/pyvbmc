"""Finding B-5: with `separate_search_gp=True` the constant-mean search GP is
trained from, and overwrites, the main `hyp_dict` and `optim_state["sn2_hpd"]`.

Settles, without running `optimize()`:
  * the hyperparameter-vector widths of the two mean functions (`const` and
    `negquad`) at D = 2 and D = 3, which is what the main `train_gp` would
    have to reconcile on the following call;
  * that `np.concatenate` of the `(0, D+3)` starting block built from a
    constant-mean `hyp_dict["hyp"]` with a recorded negative-quadratic GP's
    `(Ns, 3D+3)` hyperparameters raises `ValueError`
    (`gaussian_process_train.py:135` and `:143-149`), before the model-change
    guard at `:161-162` can reset `hyp0`;
  * that the guard really does sit after the concatenation.
"""

import gpyreg as gpr
import numpy as np

import pyvbmc
from pyvbmc.vbmc.gaussian_process_train import _meanfun_name_to_mean_function

print("pyvbmc.__file__ =", pyvbmc.__file__)

for D in (2, 3):
    widths = {}
    for name in ("const", "negquad"):
        mean_f = _meanfun_name_to_mean_function(name)
        gp = gpr.GP(
            D=D,
            covariance=gpr.covariance_functions.SquaredExponential(),
            mean=mean_f,
            noise=gpr.noise_functions.GaussianNoise(constant_add=True),
        )
        widths[name] = int(
            gp.covariance.hyperparameter_count(D)
            + gp.mean.hyperparameter_count(D)
            + gp.noise.hyperparameter_count()
        )
    print(
        f"D={D}: const mean -> {widths['const']} hyperparameters "
        f"(D+3={D+3}); negquad mean -> {widths['negquad']} "
        f"(3D+3={3*D+3})"
    )

    # gaussian_process_train.py:135 with a constant-mean hyp_dict["hyp"]
    hyp_const = np.zeros(widths["const"])
    hyp0 = np.empty((0, np.atleast_2d(hyp_const).T.shape[0]))
    print("  hyp0 starting block:", hyp0.shape)
    recorded = np.zeros((3, widths["negquad"]))  # a recorded negquad GP
    try:
        out = np.concatenate((hyp0, recorded))
        print("  concatenate ->", out.shape)
    except Exception as exc:
        print("  concatenate raises:", type(exc).__name__, exc)
