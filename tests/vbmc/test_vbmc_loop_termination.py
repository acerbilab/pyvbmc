import numpy as np

from pyvbmc.vbmc import VBMC

fun = lambda x: np.sum(x + 2)


def create_vbmc(
    D: int,
    x0: float,
    lower_bounds: float,
    upper_bounds: float,
    plausible_lower_bounds: float,
    plausible_upper_bounds: float,
    user_options: dict = None,
):
    lb = np.ones((1, D)) * lower_bounds
    ub = np.ones((1, D)) * upper_bounds
    x0_array = np.ones((2, D)) * x0
    plb = np.ones((1, D)) * plausible_lower_bounds
    pub = np.ones((1, D)) * plausible_upper_bounds
    return VBMC(fun, x0_array, lb, ub, plb, pub, user_options)


def test_vbmc_check_termination_conditions_maxfunevals(mocker):
    user_options = {
        "maxfunevals": 10,
        "minfunevals": 5,
        "miniter": 5,
        "maxiter": 100,
        "tolstablecount": 60,
        "funevalsperiter": 5,
    }
    mocker.patch(
        "pyvbmc.vbmc.VBMC._compute_reliability_index",
        return_value=(np.Inf, np.NaN),
    )
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    vbmc.optim_state["func_count"] = 10
    vbmc.optim_state["iter"] = 10
    assert vbmc._check_termination_conditions() == True
    vbmc.optim_state["func_count"] = 9
    assert vbmc._check_termination_conditions() == False


def test_vbmc_check_termination_conditions_maxiter(mocker):
    mocker.patch(
        "pyvbmc.vbmc.VBMC._compute_reliability_index",
        return_value=(np.Inf, np.NaN),
    )
    user_options = {
        "maxfunevals": 10,
        "minfunevals": 5,
        "miniter": 5,
        "maxiter": 100,
    }
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    vbmc.optim_state["func_count"] = 9
    vbmc.optim_state["entropy_switch"] = True
    vbmc.optim_state["iter"] = 100
    assert vbmc._check_termination_conditions() == True
    vbmc.optim_state["iter"] = 99
    assert vbmc._check_termination_conditions() == False


def test_vbmc_check_termination_conditions_prevent_early_termination(mocker):
    user_options = {
        "maxfunevals": 10,
        "minfunevals": 5,
        "miniter": 101,
        "maxiter": 100,
    }
    mocker.patch(
        "pyvbmc.vbmc.VBMC._compute_reliability_index",
        return_value=(np.Inf, np.NaN),
    )
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    vbmc.optim_state["func_count"] = 9
    vbmc.optim_state["iter"] = 100
    vbmc.optim_state["entropy_switch"] = True
    assert vbmc._check_termination_conditions() == False
    user_options = {
        "maxfunevals": 10,
        "minfunevals": 11,
        "miniter": 5,
        "maxiter": 100,
    }
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    vbmc.optim_state["func_count"] = 9
    vbmc.optim_state["iter"] = 100
    vbmc.optim_state["entropy_switch"] = True
    assert vbmc._check_termination_conditions() == False


def test_vbmc_check_termination_conditions_stability(mocker):
    user_options = {
        "maxfunevals": 10,
        "minfunevals": 5,
        "miniter": 5,
        "maxiter": 100,
        "tolimprovement": 0.01,
        "tolstablecount": 60,
        "funevalsperiter": 5,
        "tolstableexcptfrac": 0.2,
    }
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    vbmc.optim_state["func_count"] = 9
    vbmc.optim_state["entropy_switch"] = False
    vbmc.optim_state["iter"] = 99
    vbmc.iteration_history["rindex"] = np.ones(100) * 0.5

    mocker.patch(
        "pyvbmc.vbmc.VBMC._compute_reliability_index",
        return_value=(0.5, 0.005),
    )
    assert vbmc._check_termination_conditions() == True
    mocker.patch(
        "pyvbmc.vbmc.VBMC._compute_reliability_index",
        return_value=(1, 0.005),
    )
    assert vbmc._check_termination_conditions() == False
    mocker.patch(
        "pyvbmc.vbmc.VBMC._compute_reliability_index",
        return_value=(0.5, 0.1),
    )
    assert vbmc._check_termination_conditions() == False
    vbmc.optim_state["iter"] = 9
    mocker.patch(
        "pyvbmc.vbmc.VBMC._compute_reliability_index",
        return_value=(1, 0.005),
    )
    assert vbmc._check_termination_conditions() == False


def test_vbmc_is_finished_stability_entropyswitch(mocker):
    user_options = {
        "maxfunevals": 10,
        "minfunevals": 5,
        "miniter": 5,
        "maxiter": 100,
        "tolimprovement": 0.01,
        "tolstableentropyiters": 6,
        "tolstableexcptfrac": 0.2,
    }
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    vbmc.optim_state["func_count"] = 9
    vbmc.optim_state["entropy_switch"] = True
    vbmc.optim_state["iter"] = 99
    vbmc.iteration_history["rindex"] = np.ones(100) * 0.5
    mocker.patch(
        "pyvbmc.vbmc.VBMC._compute_reliability_index",
        return_value=(0.5, 0.005),
    )
    assert vbmc._check_termination_conditions() == False


def test_vbmc_compute_reliability_index_less_than_3_iter():
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4)
    vbmc.optim_state["iter"] = 2
    rindex, ELCBO_improvement = vbmc._compute_reliability_index(6)
    assert rindex == np.Inf
    assert np.isnan(ELCBO_improvement)


def test_vbmc_compute_reliability_index():
    user_options = {"elcboimproweight": 0, "tolskl": 0.03}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    vbmc.optim_state["iter"] = 49
    vbmc.iteration_history.check_keys = False
    vbmc.iteration_history["elbo"] = np.arange(50) * 10
    vbmc.iteration_history["elbo_sd"] = np.ones(50)
    vbmc.iteration_history["sKL"] = np.ones(50)
    vbmc.iteration_history["funccount"] = np.arange(50) * 10
    rindex, ELCBO_improvement = vbmc._compute_reliability_index(6)
    assert rindex == np.mean([10, 1, 1 / 0.03])
    assert np.isclose(ELCBO_improvement, 1)


def test_is_gp_sampling_finished():
    user_options = {"tolgpvarmcmc": 1e-4}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    vbmc.optim_state["N"] = 300
    vbmc.optim_state["iter"] = 9
    vbmc.optim_state["warmup"] = False
    vbmc.iteration_history = dict()
    vbmc.iteration_history["N"] = np.ones(10)

    # all variances low
    vbmc.iteration_history["gp_sample_var"] = np.ones(10) * 1e-5
    vbmc.optim_state["stop_gp_sampling"] = 0
    vbmc._is_gp_sampling_finished()
    assert vbmc._is_gp_sampling_finished()

    # all variances high
    vbmc.iteration_history["gp_sample_var"] = np.ones(10)
    vbmc.optim_state["stop_gp_sampling"] = 0
    vbmc._is_gp_sampling_finished()
    assert not vbmc._is_gp_sampling_finished()

    # last variance high
    vbmc.iteration_history["gp_sample_var"] = np.ones(10) * 1e-10
    vbmc.iteration_history["gp_sample_var"][-1] = 1e-2
    vbmc.optim_state["stop_gp_sampling"] = 0
    vbmc._is_gp_sampling_finished()
    assert not vbmc._is_gp_sampling_finished()
