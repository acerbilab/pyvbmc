"""Checks of the policy that steers the GP hyperparameter fit.

The contract is ``misc/get_GPTrainOptions.m`` of MATLAB VBMC, whose
iteration counter is one ahead of ``optim_state["iter"]``.
"""

import numpy as np

from pyvbmc import VBMC
from pyvbmc.vbmc.gaussian_process_train import _get_gp_training_options


def _vbmc(D=2, user_options=None):
    """A VBMC instance whose training policy can be evaluated by hand."""
    options = {"weighted_hyp_cov": False}
    if user_options is not None:
        options.update(user_options)
    return VBMC(
        lambda x: np.sum(x + 2),
        np.ones((2, D)) * 3,
        np.ones((1, D)) * 1,
        np.ones((1, D)) * 5,
        np.ones((1, D)) * 2,
        np.ones((1, D)) * 4,
        options,
    )


def _training_options(vbmc, gp_s_N=8, hyp_dict=None):
    if hyp_dict is None:
        hyp_dict = {"run_cov": None}
    return _get_gp_training_options(
        vbmc.optim_state,
        vbmc.iteration_history,
        vbmc.options,
        hyp_dict,
        gp_s_N,
    )


def test_retrain_branch_starts_at_the_first_recorded_reliability_index():
    """``misc/get_GPTrainOptions.m:109`` is ``iter > 1 &&
    stats.rindex(iter-1) < options.GPRetrainThreshold``. MATLAB's second
    iteration is ``optim_state["iter"] == 1``, so a reliability index
    recorded there and below the threshold already skips the design and
    the optimization."""
    vbmc = _vbmc()
    vbmc.optim_state["n_eff"] = 10
    vbmc.optim_state["iter"] = 1
    vbmc.optim_state["recompute_var_post"] = False
    vbmc.options.__setitem__("gp_retrain_threshold", 10, force=True)
    vbmc.iteration_history.record("r_index", 5.0, 0)

    gp_train = _training_options(vbmc)

    assert gp_train["init_N"] == 0
    assert gp_train["opts_N"] == 0


def test_retrain_branch_is_not_taken_above_the_threshold():
    """At the same iteration, a reliability index at or above the
    threshold leaves the full design and one optimization run
    (``misc/get_GPTrainOptions.m:116-118``)."""
    vbmc = _vbmc()
    vbmc.optim_state["n_eff"] = 10
    vbmc.optim_state["iter"] = 1
    vbmc.optim_state["recompute_var_post"] = False
    vbmc.options.__setitem__("gp_retrain_threshold", 10, force=True)
    vbmc.iteration_history.record("r_index", 50.0, 0)

    gp_train = _training_options(vbmc)

    assert gp_train["init_N"] > 0
    assert gp_train["opts_N"] == 1
