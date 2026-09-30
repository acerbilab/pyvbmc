"""Traces, for a default PyVBMC run, exactly which of x0, LB, UB, PLB and PUB
of which GP hyperparameters come from gpyreg's bound recommendation, in which
fits, and on which training set.

It replays `_gp_hyp` on a synthetic training set with the default options,
prints the bounds dictionary PyVBMC hands to `gp.set_bounds` (NaN = left to
gpyreg), then the bounds `fit` installs and the plausible bounds it derives,
and marks each entry with its source. It also instruments `GP.fit` to record
how many times it is reached with PyVBMC's recommendation as the starting
point, over three iterations of the real `train_gp`.

Run: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python G2_4_reach_bounds.py
"""

import gpyreg as gpr
import numpy as np

import pyvbmc
from pyvbmc.vbmc.gaussian_process_train import _gp_hyp
from pyvbmc.vbmc.iteration_history import IterationHistory
from pyvbmc.vbmc.options import Options

print("gpyreg:", gpr.__file__)
print("pyvbmc:", pyvbmc.__file__)
np.set_printoptions(precision=5, suppress=True, linewidth=200)

D = 3
rng = np.random.default_rng(21)
N = 36
X = np.column_stack(
    [rng.normal(0, 0.4, N), rng.normal(1, 1.0, N), rng.normal(-3, 4.0, N)]
)
y = (-0.5 * np.sum(X**2, 1) + rng.standard_normal(N) * 0.1).reshape(-1, 1)

options = Options(
    pyvbmc.vbmc.vbmc.__file__.replace(
        "vbmc.py", "option_configs/basic_vbmc_options.ini"
    ),
    evaluation_parameters={"D": D},
)
options.load_options_file(
    pyvbmc.vbmc.vbmc.__file__.replace(
        "vbmc.py", "option_configs/advanced_vbmc_options.ini"
    ),
    evaluation_parameters={"D": D},
)
options.validate_option_names(
    [
        pyvbmc.vbmc.vbmc.__file__.replace(
            "vbmc.py", "option_configs/basic_vbmc_options.ini"
        ),
        pyvbmc.vbmc.vbmc.__file__.replace(
            "vbmc.py", "option_configs/advanced_vbmc_options.ini"
        ),
    ]
)
print("\nrelevant option defaults:")
for k in (
    "upper_gp_length_factor",
    "gp_quadratic_mean_bound",
    "hpd_frac",
    "tol_gp_noise",
):
    print(f"  {k} = {options[k]}")

plb = np.array([[-1.0] * D])
pub = np.array([[1.0] * D])

for level in (0, 1, 2):
    optim_state = {
        "uncertainty_handling_level": level,
        "gp_noise_fun": [1, 1 if level > 0 else 0, 0]
        if level != 1
        else [1, 2, 0],
        "stop_sampling": 0,
        "warmup": True,
        "N": N,
        "vp_K": 2,
        "iter": 0,
    }
    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(
            constant_add=True,
            user_provided_add=optim_state["gp_noise_fun"][1] > 0,
            scale_user_provided=optim_state["gp_noise_fun"][1] == 2,
        ),
    )
    gp2, hyp0, gp_s_N = _gp_hyp(optim_state, options, plb, pub, gp, X, y)
    names = []
    for nm, cnt in (
        gp.covariance.hyperparameter_info(D)
        + gp.noise.hyperparameter_info()
        + gp.mean.hyperparameter_info(D)
    ):
        names += [nm] * cnt
    print(
        f"\n===== uncertainty_handling_level {level} "
        f"(gp_noise_fun {optim_state['gp_noise_fun']}) ====="
    )
    print("  hyperparameter names:", names)
    print("  LB after PyVBMC's set_bounds (nan = left to gpyreg):")
    print("   ", gp2.lower_bounds)
    print("  UB after PyVBMC's set_bounds:")
    print("   ", gp2.upper_bounds)
    lb_before = gp2.lower_bounds.copy()
    ub_before = gp2.upper_bounds.copy()
    # What fit does: fill nan from the recommendation on the FULL set
    gp2.X, gp2.y = X, y
    filled = gp2.get_recommended_bounds(lb_before, ub_before)
    lb_f = np.concatenate(
        [np.atleast_1d(filled[n][0]) for n in dict.fromkeys(names)]
    )
    ub_f = np.concatenate(
        [np.atleast_1d(filled[n][1]) for n in dict.fromkeys(names)]
    )
    cov_i = gp2.covariance.get_bounds_info(X, y)
    mean_i = gp2.mean.get_bounds_info(X, y)
    noise_i = gp2.noise.get_bounds_info(X, y)
    PLB = np.concatenate([cov_i["PLB"], noise_i["PLB"], mean_i["PLB"]])
    PUB = np.concatenate([cov_i["PUB"], noise_i["PUB"], mean_i["PUB"]])
    PLB = np.minimum(np.maximum(PLB, lb_f), ub_f)
    PUB = np.maximum(np.minimum(PUB, ub_f), lb_f)
    print("  per hyperparameter: LB source | UB source | PLB | PUB | hyp0")
    for i, nm in enumerate(names):
        src_lb = "PyVBMC" if np.isfinite(lb_before[i]) else "gpyreg(full X)"
        src_ub = "PyVBMC" if np.isfinite(ub_before[i]) else "gpyreg(full X)"
        print(
            f"    {i:2d} {nm:35s} LB {lb_f[i]:12.5f} [{src_lb:14s}]"
            f" UB {ub_f[i]:12.5f} [{src_ub:14s}]"
            f" PLB {PLB[i]:10.4f} PUB {PUB[i]:10.4f}"
            f" hyp0 {hyp0[i]:10.4f}"
        )
    print(
        "  NOTE: PyVBMC computes its own LB entries on the HPD subset "
        f"(hpd_frac={options['hpd_frac']}); every gpyreg-filled entry and "
        "every PLB/PUB is computed by fit on the FULL training set."
    )

print("\n===== how often PyVBMC's hyp0 is the fit's starting point =====")
calls = {"n": 0, "hyp0_is_recommendation": 0}
recommendation = {"value": None}
orig_fit = gpr.GP.fit


def spy_fit(self, X=None, y=None, s2=None, hyp0=None, options=None, rng=None):
    calls["n"] += 1
    rec = recommendation["value"]
    if hyp0 is not None and rec is not None:
        match = any(np.allclose(row, rec) for row in np.atleast_2d(hyp0))
    else:
        match = False
    calls["hyp0_is_recommendation"] += int(match)
    print(
        f"  fit call {calls['n']}: hyp0 rows = "
        f"{0 if hyp0 is None else np.atleast_2d(hyp0).shape[0]}, "
        f"PyVBMC's recommendation among them: {match}"
    )
    return orig_fit(self, X, y, s2, hyp0=hyp0, options=options, rng=rng)


orig_gp_hyp = _gp_hyp


def spy_gp_hyp(optim_state, options, plb_tran, pub_tran, gp, X, y):
    g, h0, n = orig_gp_hyp(optim_state, options, plb_tran, pub_tran, gp, X, y)
    recommendation["value"] = h0.copy()
    return g, h0, n


import pyvbmc.vbmc.gaussian_process_train as gpt

gpr.GP.fit = spy_fit
gpt._gp_hyp = spy_gp_hyp

hyp_dict = {}
ih = IterationHistory(
    ["gp", "r_index", "gp_hyp_full", "sKL", "N", "n_eff", "timer", "iter"]
)
options.__setitem__("gp_train_n_init", 32, force=True)
options.__setitem__("ns_gp_max_warmup", 0, force=True)
# `train_gp` reads its training set through `_get_training_data`; hand it the
# arrays above instead of building a FunctionLogger.
gpt._get_training_data = lambda fl: (X, y, None, None)
optim_state = {
    "uncertainty_handling_level": 0,
    "gp_noise_fun": [1, 0, 0],
    "stop_sampling": 0,
    "warmup": True,
    "N": N,
    "n_eff": N,
    "vp_K": 2,
    "iter": 0,
    "recompute_var_post": True,
    "gp_mean_fun": "negquad",
    "gp_cov_fun": 1,
    "max_fun_evals": options["max_fun_evals"],
}
rng_run = np.random.default_rng(3)
for it in range(3):
    optim_state["iter"] = it
    gp_out, gp_s_N, sn2, hyp_dict = gpt.train_gp(
        hyp_dict, optim_state, None, ih, options, plb, pub, rng_run
    )
    ih.record("gp", gp_out, it)
    ih.record("r_index", 0.5, it)
    ih.record("sKL", 0.1, it)
    ih.record("gp_hyp_full", np.atleast_2d(hyp_dict["full"]), it)
print(
    "  fits:",
    calls["n"],
    " of which started from PyVBMC's recommendation:",
    calls["hyp0_is_recommendation"],
)
gpr.GP.fit = orig_fit
gpt._gp_hyp = orig_gp_hyp
