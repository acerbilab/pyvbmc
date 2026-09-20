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


def _matlab_n_init(options, n_eff):
    """The design size of ``misc/get_GPTrainOptions.m:94-100``."""
    a = -(options["gp_train_n_init"] - options["gp_train_n_init_final"])
    b = -3 * a
    c = 3 * a
    d = options["gp_train_n_init"]
    x = (n_eff - options["fun_eval_start"]) / (
        min(options["max_fun_evals"], 1e3) - options["fun_eval_start"]
    )
    return max(round(a * x**3 + b * x**2 + c * x + d), 0)


def test_design_size_follows_the_schedule_past_its_horizon():
    """``misc/get_GPTrainOptions.m:100`` is ``max(round(f(x)),0)``: past
    the end of the schedule's horizon the design shrinks to a handful of
    points and then to none, which also switches off the collection of
    starting points from earlier iterations."""
    vbmc = _vbmc(user_options={"max_fun_evals": 5000})
    assert not vbmc.optim_state.get("budget_active", False)
    vbmc.optim_state["iter"] = 0

    sizes = {}
    for n_eff in (10, 500, 1400, 1500):
        vbmc.optim_state["n_eff"] = n_eff
        sizes[n_eff] = _training_options(vbmc)["init_N"]
        assert sizes[n_eff] == _matlab_n_init(vbmc.options, n_eff)

    # Inside the horizon the schedule is far above any floor; past it the
    # design is first a single point and then empty.
    assert sizes[10] == vbmc.options["gp_train_n_init"]
    assert sizes[500] > 9
    assert sizes[1400] == 1
    assert sizes[1500] == 0


def test_budget_path_clips_the_schedule_at_its_final_size():
    """A run with an active budget clips the schedule's argument to the
    unit interval, so the design never falls below the final size of
    ``gp_train_n_init_final``."""
    vbmc = _vbmc(user_options={"max_fun_evals": 5000})
    vbmc.optim_state["iter"] = 0
    vbmc.optim_state["budget_active"] = True
    vbmc.optim_state["n_eff"] = 1500

    gp_train = _training_options(vbmc)

    assert gp_train["init_N"] == vbmc.options["gp_train_n_init_final"]


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
