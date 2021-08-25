import numpy as np
from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.vbmc import VBMC


def create_vbmc(
    D: int,
    x0: float,
    lower_bounds: float,
    upper_bounds: float,
    plausible_lower_bounds: float,
    plausible_upper_bounds: float,
    user_options: dict = None,
):
    fun = lambda x: np.sum(x + 2)
    lb = np.ones((1, D)) * lower_bounds
    ub = np.ones((1, D)) * upper_bounds
    x0_array = np.ones((2, D)) * x0
    plb = np.ones((1, D)) * plausible_lower_bounds
    pub = np.ones((1, D)) * plausible_upper_bounds
    return VBMC(fun, x0_array, lb, ub, plb, pub, user_options)


def test_final_boost_lambda_options(mocker):
    user_options = {
        "minfinalcomponents": 50,
        "nsent": lambda K: 100 * K ** (2 / 3),
        "nsentfast": lambda K: 0,
        "nsentfine": lambda K: 2 ** 12 * K,
        "nsentboost": lambda K: 100 * K ** (2 / 3) - 10,
        "nsentfastboost": lambda K: 0,
        "nsentfineboost": lambda K: 2 ** 12 * K,
        "nselbo": lambda K: 50 * K,
    }
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    vbmc.iteration_history["gp"] = np.arange(30)
    mocker.patch(
        "pyvbmc.vbmc.vbmc.optimize_vp",
        return_value=(VariationalPosterior(50), None, None)
    )
    vp, elbo, elbo_sd, changedflag = vbmc.finalboost(vbmc.vp, dict())
    assert changedflag


def test_final_boost_fixed_value_options(mocker):
    user_options = {
        "minfinalcomponents": 50,
        "nsent": 1300,
        "nsentfast": 0,
        "nsentfine": 204800,
        "nsentboost": 1300,
        "nsentfastboost": 20,
        "nsentfineboost": 204800,
        "nselbo": 2500,
    }
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    vbmc.iteration_history["gp"] = np.arange(30)
    mocker.patch(
        "pyvbmc.vbmc.vbmc.optimize_vp",
        return_value=(VariationalPosterior(50), None, None)
    )
    vp, elbo, elbo_sd, changedflag = vbmc.finalboost(vbmc.vp, dict())
    assert changedflag


def test_final_boost_fixed_value_options_boost_none(mocker):
    user_options = {
        "minfinalcomponents": 50,
        "nsent": 1300,
        "nsentfast": 0,
        "nsentfine": 204800,
        "nsentboost": [],
        "nsentfastboost": [],
        "nsentfineboost": [],
        "nselbo": 2500,
    }
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    vbmc.iteration_history["gp"] = np.arange(30)
    mocker.patch(
        "pyvbmc.vbmc.vbmc.optimize_vp",
        return_value=(VariationalPosterior(50), None, None)
    )
    vp, elbo, elbo_sd, changedflag = vbmc.finalboost(vbmc.vp, dict())
    assert changedflag


def test_final_boost_no_boost(mocker):
    user_options = {
        "minfinalcomponents": 1,
        "nsent": 1300,
        "nsentfast": 0,
        "nsentfine": 204800,
        "nsentboost": [],
        "nsentfastboost": [],
        "nsentfineboost": [],
    }
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    vbmc.iteration_history["gp"] = np.arange(30)
    mocker.patch(
        "pyvbmc.vbmc.vbmc.optimize_vp",
        return_value=(VariationalPosterior(50), None, None)
    )
    vp, elbo, elbo_sd, changedflag = vbmc.finalboost(vbmc.vp, dict())
    assert changedflag == False
