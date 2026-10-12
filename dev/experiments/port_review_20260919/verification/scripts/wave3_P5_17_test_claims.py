"""P5-17: two factual claims of the reports' test-adequacy notes.

Settles:
  * whether the hard-coded 16-digit constants of `test_get_hyp_cov` agree
    with a literal transcription of MATLAB's `GetHypCov`
    (`misc/get_GPTrainOptions.m:126-170`) on the test's own inputs, i.e.
    whether the constants are an output snapshot or the MATLAB answer;
  * whether the bounds `test_gp_hyp` passes to `train_gp`
    (`vbmc.plausible_lower_bounds` / `plausible_upper_bounds`, original
    coordinates) equal the transformed bounds `train_gp` documents
    (`optim_state["plb_tran"]` / `["pub_tran"]`) in that fixture.
"""

import numpy as np

import pyvbmc
from pyvbmc import VBMC
from pyvbmc.vbmc.gaussian_process_train import _get_hyp_cov

print("pyvbmc.__file__ =", pyvbmc.__file__)

# ---- the inputs of `_weighted_hyp_cov_inputs()` in the test module ----
history = {
    "gp_hyp_full": [
        np.array([[0.0, 0.0]]),
        np.array([[1.0, 3.0], [4.0, 2.0], [2.0, 5.0]]),
        np.array([[7.0, 8.0, 9.0]]),
        np.array([[10.0, 1.0], [12.0, 4.0]]),
    ],
    "sKL": np.array([1.0, np.exp(1.5), np.exp(0.2), np.exp(2), 1.0]),
    "r_index": np.ones(4),
}
options = {
    "weighted_hyp_cov": True,
    "hyp_run_weight": 0.5,
    "fun_evals_per_iter": 2,
    "tol_skl": 0.5,
    "tol_cov_weight": 0.0,
}
optim_state = {"iter": 4}
hyp_dict = {"hyp": np.zeros(2)}
expected_in_test = np.array(
    [
        [4.580701754385967, 2.717885869909433],
        [2.717885869909433, 4.36639432516309],
    ]
)


def matlab_get_hyp_cov(M, history, options):
    """Literal transcription of misc/get_GPTrainOptions.m:126-170.

    `M` is MATLAB's 1-based `optimState.iter`; the history arrays are the
    0-based Python ones, so `stats.x(k)` is `history["x"][k - 1]`.
    """
    w_list = []
    hyp_list = []
    w = 1.0
    for i in range(1, M):  # for i = 1:optimState.iter-1
        if i > 1:
            # diff_mult = max(1, log(stats.sKL(M-i+1)/(TolsKL*FunEvalsPerIter)))
            skl = history["sKL"][(M - i + 1) - 1]
            diff_mult = max(
                1.0,
                np.log(
                    skl / (options["tol_skl"] * options["fun_evals_per_iter"])
                ),
            )
            w = w * options["hyp_run_weight"] ** (
                options["fun_evals_per_iter"] * diff_mult
            )
        if w < options["tol_cov_weight"]:
            break
        hyp = history["gp_hyp_full"][(M - i) - 1]  # stats.gpHypFull{M-i}
        # MATLAB stores Nhyp x Ns; the Python history stores Ns x Nhyp, so
        # MATLAB's `hyp'` is the Python block as it stands and MATLAB's
        # size(hyp,2) (the sample count) is its number of rows.
        nhyp_samples = hyp.shape[0]
        if not hyp_list or hyp_list[0].shape[1] == hyp.shape[1]:
            hyp_list.append(hyp)
            w_list.append(np.full(nhyp_samples, w / nhyp_samples))
    hyp_list = np.concatenate(hyp_list, axis=0)
    w_list = np.concatenate(w_list)
    w_list = w_list / np.sum(w_list)
    mustar = np.sum(w_list[:, None] * hyp_list, axis=0)
    nhyp = hyp_list.shape[1]
    hypcov = np.zeros((nhyp, nhyp))
    for j in range(hyp_list.shape[0]):
        d = (hyp_list[j, :] - mustar)[:, None]
        hypcov = hypcov + w_list[j] * (d @ d.T)
    return hypcov / (1 - np.sum(w_list**2))


ml = matlab_get_hyp_cov(5, history, options)  # MATLAB iter = python iter + 1
py = _get_hyp_cov(optim_state, history, options, hyp_dict)
print("\nMATLAB GetHypCov transcription:\n", ml)
print("PyVBMC _get_hyp_cov:\n", py)
print("constant hard-coded in test_get_hyp_cov:\n", expected_in_test)
print("max |MATLAB - PyVBMC|            =", np.max(np.abs(ml - py)))
print(
    "max |MATLAB - test constant|     =", np.max(np.abs(ml - expected_in_test))
)

# ---- the bounds test_gp_hyp passes ----
D = 3
f = lambda x: np.sum(np.atleast_2d(x) + 2, axis=1)
v = VBMC(
    f,
    np.ones((2, D)) * 3,
    None,
    None,
    np.ones((1, D)) * -1,
    np.ones((1, D)) * 1,
    {"specify_target_noise": True},
)
print("\ntest_gp_hyp fixture (no hard bounds):")
print(
    "  plausible_lower_bounds =",
    v.plausible_lower_bounds.ravel(),
    " plb_tran =",
    v.optim_state["plb_tran"].ravel(),
)
print(
    "  plausible_upper_bounds =",
    v.plausible_upper_bounds.ravel(),
    " pub_tran =",
    v.optim_state["pub_tran"].ravel(),
)
print(
    "  equal ->",
    np.array_equal(v.plausible_lower_bounds, v.optim_state["plb_tran"])
    and np.array_equal(v.plausible_upper_bounds, v.optim_state["pub_tran"]),
)
