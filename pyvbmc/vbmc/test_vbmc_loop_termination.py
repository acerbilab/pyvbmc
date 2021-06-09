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


def test_vbmc_is_finished_maxfunevals():
    user_options = {
        "maxfunevals": 10,
        "minfunevals": 5,
        "miniter": 5,
        "maxiter": 100,
    }
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    vbmc.optim_state["func_count"] = 10
    vbmc.optim_state["iter"] = 10
    assert vbmc._is_finished() == True
    vbmc.optim_state["func_count"] = 9
    assert vbmc._is_finished() == False


def test_vbmc_is_finished_maxiter():
    user_options = {
        "maxfunevals": 10,
        "minfunevals": 5,
        "miniter": 5,
        "maxiter": 100,
    }
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    vbmc.optim_state["func_count"] = 9
    vbmc.optim_state["iter"] = 100
    assert vbmc._is_finished() == True
    vbmc.optim_state["iter"] = 99
    assert vbmc._is_finished() == False

def test_vbmc_is_finished_prevent_early_termination():
    user_options = {
        "maxfunevals": 10,
        "minfunevals": 5,
        "miniter": 101,
        "maxiter": 100,
    }
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    vbmc.optim_state["func_count"] = 9
    vbmc.optim_state["iter"] = 100
    assert vbmc._is_finished() == False
    user_options = {
        "maxfunevals": 10,
        "minfunevals": 11,
        "miniter": 5,
        "maxiter": 100,
    }
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    vbmc.optim_state["func_count"] = 9
    vbmc.optim_state["iter"] = 100
    assert vbmc._is_finished() == False
