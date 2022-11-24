import numpy as np

from pyvbmc import VBMC
from pyvbmc.variational_posterior import VariationalPosterior


def create_vbmc(
    D: int,
    x0: float,
    lower_bounds: float,
    upper_bounds: float,
    plausible_lower_bounds: float,
    plausible_upper_bounds: float,
    options: dict = None,
):
    fun = lambda x: np.sum(x + 2)
    lb = np.ones((1, D)) * lower_bounds
    ub = np.ones((1, D)) * upper_bounds
    x0_array = np.ones((2, D)) * x0
    plb = np.ones((1, D)) * plausible_lower_bounds
    pub = np.ones((1, D)) * plausible_upper_bounds
    return VBMC(fun, x0_array, lb, ub, plb, pub, options)


def test_final_boost_lambda_options(mocker):
    options = {
        "min_final_components": 50,
        "ns_ent": lambda K: 100 * K ** (2 / 3),
        "ns_ent_fast": lambda K: 0,
        "ns_ent_fine": lambda K: 2**12 * K,
        "ns_ent_boost": lambda K: 100 * K ** (2 / 3) - 10,
        "ns_ent_fast_boost": lambda K: 0,
        "ns_ent_fine_boost": lambda K: 2**12 * K,
        "ns_elbo": lambda K: 50 * K,
    }
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    vbmc.iteration_history["gp"] = np.arange(30)
    vbmc.vp.stats = dict()
    vbmc.vp.stats["elbo"] = -3
    vbmc.vp.stats["elbo_sd"] = 0.1
    vbmc.vp.stats["stable"] = True
    mocker.patch(
        "pyvbmc.vbmc.vbmc.optimize_vp", return_value=(vbmc.vp, None, None)
    )
    vp, elbo, elbo_sd, changedflag = vbmc.final_boost(vbmc.vp, dict())
    assert changedflag


def test_final_boost_fixed_value_options(mocker):
    options = {
        "min_final_components": 50,
        "ns_ent": 1300,
        "ns_ent_fast": 0,
        "ns_ent_fine": 204800,
        "ns_ent_boost": 1300,
        "ns_ent_fast_boost": 20,
        "ns_ent_fine_boost": 204800,
        "ns_elbo": 2500,
    }
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    vbmc.iteration_history["gp"] = np.arange(30)
    vbmc.vp.stats = dict()
    vbmc.vp.stats["elbo"] = -3
    vbmc.vp.stats["elbo_sd"] = 0.1
    vbmc.vp.stats["stable"] = True
    mocker.patch(
        "pyvbmc.vbmc.vbmc.optimize_vp", return_value=(vbmc.vp, None, None)
    )
    vp, elbo, elbo_sd, changedflag = vbmc.final_boost(vbmc.vp, dict())
    assert changedflag


def test_final_boost_fixed_value_options_boost_none(mocker):
    options = {
        "min_final_components": 50,
        "ns_ent": 1300,
        "ns_ent_fast": 0,
        "ns_ent_fine": 204800,
        "ns_ent_boost": [],
        "ns_ent_fast_boost": [],
        "ns_ent_fine_boost": [],
        "ns_elbo": 2500,
    }
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    vbmc.iteration_history["gp"] = np.arange(30)
    vbmc.vp.stats = dict()
    vbmc.vp.stats["elbo"] = -3
    vbmc.vp.stats["elbo_sd"] = 0.1
    vbmc.vp.stats["stable"] = True
    mocker.patch(
        "pyvbmc.vbmc.vbmc.optimize_vp", return_value=(vbmc.vp, None, None)
    )
    vp, elbo, elbo_sd, changedflag = vbmc.final_boost(vbmc.vp, dict())
    assert changedflag


def test_final_boost_no_boost(mocker):
    options = {
        "min_final_components": 1,
        "ns_ent": 1300,
        "ns_ent_fast": 0,
        "ns_ent_fine": 204800,
        "ns_ent_boost": [],
        "ns_ent_fast_boost": [],
        "ns_ent_fine_boost": [],
    }
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    vbmc.iteration_history["gp"] = np.arange(30)
    vbmc.vp.stats = dict()
    vbmc.vp.stats["elbo"] = -3
    vbmc.vp.stats["elbo_sd"] = 0.1
    vbmc.vp.stats["stable"] = True
    mocker.patch(
        "pyvbmc.vbmc.vbmc.optimize_vp", return_value=(vbmc.vp, None, None)
    )
    vp, elbo, elbo_sd, changedflag = vbmc.final_boost(vbmc.vp, dict())
    assert changedflag == False
