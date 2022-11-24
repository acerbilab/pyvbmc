import sys

import numpy as np
import pytest
import scipy as sp
import scipy.stats

from pyvbmc import VBMC
from pyvbmc.variational_posterior import VariationalPosterior

fun = lambda x: np.sum(x + 2)


def test_vbmc_init_no_x0_PLB_PUB():
    with pytest.raises(ValueError) as execinfo:
        VBMC(fun)
    assert "vbmc:UnknownDims If no starting point is" in execinfo.value.args[0]


def create_vbmc(
    D: int,
    x0: float,
    lower_bounds: float,
    upper_bounds: float,
    plausible_lower_bounds: float,
    plausible_upper_bounds: float,
    options: dict = None,
):
    lb = np.ones((1, D)) * lower_bounds
    ub = np.ones((1, D)) * upper_bounds
    x0_array = np.ones((2, D)) * x0
    plb = np.ones((1, D)) * plausible_lower_bounds
    pub = np.ones((1, D)) * plausible_upper_bounds
    return VBMC(fun, x0_array, lb, ub, plb, pub, options)


def test_vbmc_init_no_x0():
    D = 3
    lb = np.zeros((1, D))
    ub = np.ones((1, D)) * 2
    plb = np.ones((1, D)) * 0.5
    pub = np.ones((1, D)) * 1.5
    vbmc = VBMC(fun, None, lb, ub, plb, pub)
    assert np.all(vbmc.x0 == 0)
    assert vbmc.x0.shape == (1, D)


def test_vbmc_init_no_lb_ub():
    D = 3
    x0 = np.zeros((3, D))
    plb = np.ones((1, D)) * 0.5
    pub = np.ones((1, D)) * 1.5
    vbmc = VBMC(
        fun, x0, plausible_lower_bounds=plb, plausible_upper_bounds=pub
    )
    assert np.all(vbmc.lower_bounds == np.inf * -1)
    assert vbmc.lower_bounds.shape == (1, 3)
    assert np.all(vbmc.upper_bounds == np.inf)
    assert vbmc.upper_bounds.shape == (1, 3)


def test_vbmc_bounds_check_no_PUB_PLB_n0_1():
    D = 3
    lb = np.zeros((1, D))
    ub = np.ones((1, D)) * 2
    x0 = np.ones((1, D))
    _, lb2, ub2, plb, pub = VBMC(fun, x0, lb, ub)._bounds_check(x0, lb, ub)
    assert np.all(lb == lb2)
    assert np.all(ub == ub2)
    assert np.all(plb == lb + 2 * 1e-3)
    assert np.all(pub == ub - 2 * 1e-3)


def test_vbmc_bounds_check_no_PUB_PLB_n0_3():
    D = 3
    lb = np.ones((1, D)) * -2
    ub = np.ones((1, D)) * 2
    x0 = np.concatenate((np.ones((1, D)) * -0.75, np.ones((1, D)) * 0.75))
    _, lb2, ub2, plb, pub = VBMC(fun, x0, lb, ub)._bounds_check(x0, lb, ub)
    assert np.all(lb == lb2)
    assert np.all(ub == ub2)
    assert np.all(plb == np.ones((1, D)) * -1.5)
    assert np.all(pub == np.ones((1, D)) * 1.5)


def test_vbmc_bounds_check_no_PUB_PLB_identical():
    D = 3
    lb = np.ones((1, D)) * -2
    ub = np.ones((1, D)) * 2
    x0 = np.ones((2, D))
    plb = np.ones((1, D))
    _, lb2, ub2, plb, pub = VBMC(fun, x0, lb, ub)._bounds_check(
        x0, lb, ub, plb
    )
    assert np.all(lb == lb2)
    assert np.all(ub == ub2)
    assert np.all(plb == lb + 4 * 1e-3)
    assert np.all(pub == ub - 4 * 1e-3)


def test_vbmc_bounds_check_not_D():
    D = 3
    lb = np.ones((1, D)) * -2
    ub = np.ones((1, D)) * 2
    plb = np.ones((1, D)) * -1
    pub = np.ones((1, D))
    x0 = np.ones((2, D))
    incorrect = np.ones((1, D - 1))
    exception_message = "Bounds must match problem dimension D="
    with pytest.raises(ValueError) as execinfo1:
        VBMC(fun, x0, lb, ub)._bounds_check(x0, incorrect, ub, plb, pub)
    assert exception_message in execinfo1.value.args[0]
    with pytest.raises(ValueError) as execinfo2:
        VBMC(fun, x0, lb, ub)._bounds_check(x0, lb, incorrect, plb, pub)
    assert exception_message in execinfo2.value.args[0]
    with pytest.raises(ValueError) as execinfo3:
        VBMC(fun, x0, lb, ub)._bounds_check(x0, lb, ub, incorrect, pub)
    assert exception_message in execinfo3.value.args[0]
    with pytest.raises(ValueError) as execinfo4:
        VBMC(fun, x0, lb, ub)._bounds_check(x0, lb, ub, plb, incorrect)
    assert exception_message in execinfo4.value.args[0]
    VBMC(fun, x0, lb, ub)._bounds_check(x0, lb, ub, plb, pub)


def test_vbmc_bounds_check_not_vectors():
    D = 3
    lb = np.ones((1, D)) * -2
    ub = np.ones((1, D)) * 2
    plb = np.ones((1, D)) * -1
    pub = np.ones((1, D))
    x0 = np.ones((2, D))
    incorrect = 1
    exception_message = "Bounds must match problem dimension D="
    with pytest.raises(ValueError) as execinfo1:
        VBMC(fun, x0, lb, ub)._bounds_check(x0, incorrect, ub, plb, pub)
    assert exception_message in execinfo1.value.args[0]
    with pytest.raises(ValueError) as execinfo2:
        VBMC(fun, x0, lb, ub)._bounds_check(x0, lb, incorrect, plb, pub)
    assert exception_message in execinfo2.value.args[0]
    with pytest.raises(ValueError) as execinfo3:
        VBMC(fun, x0, lb, ub)._bounds_check(x0, lb, ub, incorrect, pub)
    assert exception_message in execinfo3.value.args[0]
    with pytest.raises(ValueError) as execinfo4:
        VBMC(fun, x0, lb, ub)._bounds_check(x0, lb, ub, plb, incorrect)
    assert exception_message in execinfo4.value.args[0]
    VBMC(fun, x0, lb, ub)._bounds_check(x0, lb, ub, plb, pub)


def test_vbmc_bounds_check_not_row_vectors():
    D = 3
    lb = np.ones((1, D)) * -2
    ub = np.ones((1, D)) * 2
    x0 = np.ones((2, D))
    plb = np.ones((1, D)) * -1
    pub = np.ones((1, D))
    incorrect = np.ones((D, 1))
    VBMC(fun, x0, lb, ub)._bounds_check(x0, -2 * incorrect, ub, plb, pub)
    VBMC(fun, x0, lb, ub)._bounds_check(x0, lb, 2 * incorrect, plb, pub)
    VBMC(fun, x0, lb, ub)._bounds_check(x0, lb, ub, -1 * incorrect, pub)
    VBMC(fun, x0, lb, ub)._bounds_check(x0, lb, ub, plb, 1 * incorrect)
    VBMC(fun, x0, lb, ub)._bounds_check(x0, lb, ub, plb, pub)


def test_vbmc_bounds_check_plb_pub_not_finite():
    D = 3
    lb = np.ones((1, D)) * -2
    ub = np.ones((1, D)) * 2
    x0 = np.ones((2, D))
    plb = np.ones((1, D)) * -1
    pub = np.ones((1, D))
    incorrect = np.array([[1 + 2j, 3 + 4j, 5 + 6j]])
    exception_message = "need to be real valued"
    with pytest.raises(ValueError) as execinfo1:
        VBMC(fun, x0, lb, ub)._bounds_check(x0, incorrect, ub, plb, pub)
    assert exception_message in execinfo1.value.args[0]
    with pytest.raises(ValueError) as execinfo2:
        VBMC(fun, x0, lb, ub)._bounds_check(x0, lb, incorrect, plb, pub)
    assert exception_message in execinfo2.value.args[0]
    with pytest.raises(ValueError) as execinfo3:
        VBMC(fun, x0, lb, ub)._bounds_check(x0, lb, ub, incorrect, pub)
    assert exception_message in execinfo3.value.args[0]
    with pytest.raises(ValueError) as execinfo4:
        VBMC(fun, x0, lb, ub)._bounds_check(x0, lb, ub, plb, incorrect)
    assert exception_message in execinfo4.value.args[0]
    VBMC(fun, x0, lb, ub)._bounds_check(x0, lb, ub, plb, pub)


def test_vbmc_bounds_check_fixed():
    D = 3
    lb = np.ones((1, D)) * -2
    ub = np.ones((1, D)) * 2
    x0 = np.ones((2, D))
    fixed_bound = np.ones((1, D))
    with pytest.raises(ValueError) as execinfo:
        VBMC(fun, x0, lb, ub)._bounds_check(
            x0, fixed_bound, fixed_bound, fixed_bound, fixed_bound
        )
    assert "VBMC does not support fixed" in execinfo.value.args[0]


def test_vbmc_bounds_check_PLB_PUB_different():
    D = 3
    lb = np.ones((1, D)) * -2
    ub = np.ones((1, D)) * 2
    x0 = np.ones((2, D))
    pb = np.ones((1, D))
    with pytest.raises(ValueError) as execinfo:
        VBMC(fun, x0, lb, ub)._bounds_check(x0, lb, ub, pb, pb)
    assert (
        "plausible lower and upper bounds need to be distinct"
        in execinfo.value.args[0]
    )


def test_vbmc_bounds_check_x0_outside_lb_ub():
    D = 3
    lb = np.ones((1, D)) * -2
    ub = np.ones((1, D)) * 2
    x0 = np.ones((2, D))
    plb = np.ones((1, D)) * -1
    pub = np.ones((1, D))
    x0_large = np.ones((3, D)) * 1000
    x0_small = np.ones((3, D)) * -1000
    exception_message = "X0 are not inside the provided hard bounds LB and UB"
    with pytest.raises(ValueError) as execinfo1:
        VBMC(fun, x0, lb, ub)._bounds_check(x0_large, lb, ub, plb, pub)
    assert exception_message in execinfo1.value.args[0]
    with pytest.raises(ValueError) as execinfo2:
        VBMC(fun, x0, lb, ub)._bounds_check(x0_small, lb, ub, plb, pub)
    assert exception_message in execinfo2.value.args[0]


def test_vbmc_bounds_check_ordering():
    D = 3
    lb = np.ones((1, D)) * -2
    ub = np.ones((1, D)) * 2
    x0 = np.ones((2, D))
    plb = np.ones((1, D)) * -1
    pub = np.ones((1, D))
    exception_message = (
        "bounds should respect the ordering LB < PLB < PUB < UB"
    )
    with pytest.raises(ValueError) as execinfo1:
        VBMC(fun, x0, lb, ub)._bounds_check(x0, lb, ub, pub, plb)
    assert exception_message in execinfo1.value.args[0]
    with pytest.raises(ValueError) as execinfo2:
        VBMC(fun, x0, lb, ub)._bounds_check(x0, plb, ub, lb, pub)
    assert exception_message in execinfo2.value.args[0]
    with pytest.raises(ValueError) as execinfo2:
        VBMC(fun, x0, lb, ub)._bounds_check(x0, lb, pub, plb, ub)
    assert exception_message in execinfo2.value.args[0]
    VBMC(fun, x0, lb, ub)._bounds_check(x0, lb, ub, lb, ub)


def test_vbmc_boundcheck_half_bounded():
    D = 3
    lb = np.ones((1, D)) * -2
    ub = np.ones((1, D)) * 2
    x0 = np.ones((2, D)) * 0.5
    plb = np.ones((1, D)) * -1
    pub = np.ones((1, D))
    exception_message = "Variables bounded only below/above are not supported"
    with pytest.raises(ValueError) as execinfo1:
        VBMC(fun, x0, lb, ub)._bounds_check(x0, lb * np.inf, ub, plb, pub)
    assert exception_message in execinfo1.value.args[0]
    with pytest.raises(ValueError) as execinfo2:
        VBMC(fun, x0, lb, ub)._bounds_check(x0, lb, ub * np.inf, plb, pub)
    assert exception_message in execinfo2.value.args[0]


def test_vbmc_boundcheck_hardbounds_too_close():
    D = 3
    lb = np.ones((1, D)) * -2
    ub = np.ones((1, D)) * 2
    x0 = np.ones((2, D)) * 0.5
    identicial = np.zeros((1, D))
    realmin = sys.float_info.min
    exception_message = "vbmc:StrictBoundsTooClose: Hard bounds LB and UB"
    with pytest.raises(ValueError) as execinfo1:
        VBMC(fun, x0, lb, ub)._bounds_check(
            identicial,
            identicial,
            identicial + realmin * 1,
            identicial,
            identicial + realmin * 1,
        )
    assert exception_message in execinfo1.value.args[0]
    # this should be the minimum values with which no exception is being raised
    VBMC(fun, x0, lb, ub)._bounds_check(
        identicial,
        identicial,
        identicial + realmin * 3,
        identicial + realmin * 1,
        identicial + realmin * 2,
    )


def test_vbmc_boundcheck_x0_too_close_to_hardbounds():
    D = 3
    lb = np.ones((1, D)) * -2
    ub = np.ones((1, D)) * 2
    x0 = np.zeros((2, D))
    identicial = np.zeros((1, D))
    realmin = sys.float_info.min
    x0_2, _, _, _, _ = VBMC(fun, x0, lb, ub)._bounds_check(
        x0,
        identicial,
        identicial + realmin * 3,
        identicial + realmin * 1,
        identicial + realmin * 2,
    )
    assert x0_2.shape == x0.shape
    assert np.any(x0_2 != x0)
    assert np.all(np.isclose(x0_2, 1e-3 * realmin * 3, rtol=1e-12, atol=1e-14))


def test_vbmc_boundcheck_plausible_bounds_finite():
    D = 3
    lb = np.ones((1, D)) * -2
    ub = np.ones((1, D)) * 2
    x0 = np.ones((2, D)) * 0.5
    plb = np.zeros((1, D)) + 1e4
    pub = np.ones((1, D))
    exception_message = "PLB and PUB need to be finite."
    with pytest.raises(ValueError) as execinfo1:
        VBMC(fun, x0, lb, ub)._bounds_check(x0, lb, ub, plb, pub * np.inf)
    assert exception_message in execinfo1.value.args[0]
    with pytest.raises(ValueError) as execinfo2:
        VBMC(fun, x0, lb, ub)._bounds_check(x0, lb, ub, plb * np.inf, pub)
    assert exception_message in execinfo2.value.args[0]


def test_vbmc_boundcheck_plausible_bounds_too_close_to_hardbounds():
    D = 3
    lb = np.ones((1, D)) * -2
    ub = np.ones((1, D)) * 2
    x0 = np.zeros((2, D))
    _, _, _, plb2, pub2 = VBMC(fun, x0, lb, ub)._bounds_check(
        x0,
        lb,
        ub,
        lb + 1e-4,
        ub - 1e-4,
    )
    assert plb2.shape == lb.shape
    assert pub2.shape == lb.shape
    assert np.any(plb2 == lb + 1e-3 * 4)
    assert np.any(pub2 == ub - 1e-3 * 4)


def test_vbmc_boundcheck_x0_not_in_plausible_bounds():
    D = 3
    lb = np.ones((1, D)) * -2
    ub = np.ones((1, D)) * 2
    x0 = np.ones((2, D)) * 2
    x0_2, _, _, plb2, pub2 = VBMC(fun, x0, lb, ub)._bounds_check(
        x0 - 1e-10,
        lb,
        ub,
        lb + 1e-10,
        ub - 1e-10,
    )
    assert plb2.shape == lb.shape
    assert pub2.shape == lb.shape
    assert np.any(plb2 == lb + 1e-3 * 4)
    assert np.any(pub2 == ub - 1e-3 * 4)
    assert np.any(x0_2 == pub2)


def test_vbmc_setupvars_no_x0_infinite_bounds():
    D = 3
    lb = np.ones((1, D)) * -np.inf
    ub = np.ones((1, D)) * np.inf
    x0 = np.ones((2, D)) * np.nan
    plb = np.ones((1, D)) * -1.5
    pub = np.ones((1, D)) * -0.5
    vbmc = VBMC(fun, x0, lb, ub, plb, pub)
    assert vbmc.x0.shape == (1, D)
    assert np.all(vbmc.x0 == np.ones((1, D)) * 0)


def test_vbmc_optimstate_integer_vars():
    options = {"integer_vars": np.array([1, 0, 0])}
    D = 3
    lb = np.ones((1, D)) * 1
    ub = np.ones((1, D)) * 5
    x0 = np.ones((2, D)) * 3
    plb = np.ones((1, D)) * 2
    pub = np.ones((1, D)) * 4
    exception_message = "set at +/- 0.5 points from their boundary values"
    with pytest.raises(ValueError) as execinfo1:
        VBMC(fun, x0, lb * -np.inf, ub * np.inf, plb, pub, options)
    assert exception_message in execinfo1.value.args[0]
    lb[0] = -np.inf
    ub[0] = np.inf
    with pytest.raises(ValueError) as execinfo2:
        VBMC(fun, x0, lb, ub, plb, pub, options)
    assert exception_message in execinfo2.value.args[0]
    lb[0] = -10
    ub[0] = 10
    with pytest.raises(ValueError) as execinfo3:
        VBMC(fun, x0, lb, ub, plb, pub, options)
    assert exception_message in execinfo3.value.args[0]
    lb[0] = -10.5
    ub[0] = 10.5
    vbmc = VBMC(fun, x0, lb, ub, plb, pub, options)
    integer_vars = np.full((1, D), False)
    integer_vars[:, 0] = True
    assert np.all(vbmc.optim_state.get("integer_vars") == integer_vars)


def test_vbmc_setupvars_f_vals():
    exception_message = (
        "points in X0 and of their function values as specified"
    )
    with pytest.raises(ValueError) as execinfo1:
        options = {"f_vals": np.zeros((3, 1))}
        create_vbmc(3, 3, 1, 5, 2, 4, options)
    assert exception_message in execinfo1.value.args[0]
    with pytest.raises(ValueError) as execinfo2:
        options = {"f_vals": np.zeros((1, 1))}
        create_vbmc(3, 3, 1, 5, 2, 4, options)
    assert exception_message in execinfo2.value.args[0]

    options = {"f_vals": [1, 2]}
    x0 = np.array(([[1, 2, 3], [3, 4, 3]]))
    D = 3
    lb = np.ones((1, D)) * 1
    ub = np.ones((1, D)) * 5
    plb = np.ones((1, D)) * 2
    pub = np.ones((1, D)) * 4
    vbmc = VBMC(fun, x0, lb, ub, plb, pub, options)
    assert np.all(
        vbmc.optim_state.get("cache").get("y_orig") == options.get("f_vals")
    )
    assert np.all(vbmc.optim_state.get("cache").get("x_orig") is not None)
    assert vbmc.optim_state.get("cache_active")


def test_vbmc_optimstate_gp_functions():
    exception_message = "vbmc:UnknownGPmean:Unknown/unsupported GP mean"
    with pytest.raises(ValueError) as execinfo1:
        options = {"gp_mean_fun": "notvalid"}
        create_vbmc(3, 3, 1, 5, 2, 4, options)
    assert exception_message in execinfo1.value.args[0]
    with pytest.raises(ValueError) as execinfo2:
        options = {"gp_mean_fun": ""}
        create_vbmc(3, 3, 1, 5, 2, 4, options)
    assert exception_message in execinfo2.value.args[0]
    options = {"gp_mean_fun": "const"}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    assert vbmc.optim_state.get("gp_mean_fun") == options.get("gp_mean_fun")
    # uncertainty_handling_level 2
    assert vbmc.optim_state["gp_cov_fun"] == 1
    options = {"specify_target_noise": True}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    # uncertainty_handling_level 1
    assert vbmc.optim_state["gp_noise_fun"] == [1, 1, 0]
    options = {"specify_target_noise": False, "uncertainty_handling": [3]}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    assert vbmc.optim_state["gp_noise_fun"] == [1, 2, 0]
    # uncertainty_handling_level 0
    options = {
        "specify_target_noise": False,
        "uncertainty_handling": [],
        "noise_shaping": True,
    }
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    assert vbmc.optim_state["uncertainty_handling_level"] == 0
    assert vbmc.optim_state["gp_noise_fun"] == [1, 1, 0]
    options = {
        "specify_target_noise": False,
        "uncertainty_handling": [],
        "noise_shaping": False,
    }
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    assert vbmc.optim_state["uncertainty_handling_level"] == 0
    assert vbmc.optim_state["gp_noise_fun"] == [1, 0, 0]


def test_vbmc_optimstate_bounds():
    D = 3
    lb = np.ones((1, D)) * 1
    ub = np.ones((1, D)) * 5
    x0 = np.ones((2, D)) * 3
    plb = np.ones((1, D)) * 2
    pub = np.ones((1, D)) * 4
    vbmc = VBMC(fun, x0, lb, ub, plb, pub)
    assert np.all(vbmc.optim_state["lb_orig"] == lb)
    assert np.all(vbmc.optim_state["ub_orig"] == ub)
    assert np.all(vbmc.optim_state["plb_orig"] == plb)
    assert np.all(vbmc.optim_state["pub_orig"] == pub)
    eps = vbmc.options.get("tol_bound_x") * 4
    assert np.all(vbmc.optim_state["lb_eps_orig"] == lb + eps)
    assert np.all(vbmc.optim_state["ub_eps_orig"] == ub - eps)
    assert np.all(vbmc.optim_state["lb_tran"] == -np.inf)
    assert np.all(vbmc.optim_state["ub_tran"] == np.inf)
    assert np.all(vbmc.optim_state["plb_tran"] == -0.5)
    assert np.all(vbmc.optim_state["pub_tran"] == 0.5)
    assert np.all(vbmc.optim_state["lb_search"] == -2.5)
    assert np.all(vbmc.optim_state["ub_search"] == 2.5)


def test_vbmc_optimstate_constants():
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4)
    assert np.all(vbmc.optim_state["iter"] == -1)
    assert np.all(vbmc.optim_state["sn2_hpd"] == np.inf)
    assert np.all(vbmc.optim_state["last_warping"] == -np.inf)
    assert np.all(vbmc.optim_state["last_successful_warping"] == -np.inf)
    assert np.all(vbmc.optim_state["warping_count"] == 0)
    assert np.all(vbmc.optim_state["recompute_var_post"] == True)
    assert np.all(vbmc.optim_state["warmup_stable_count"] == 0)
    assert np.all(vbmc.optim_state["R"] == np.inf)
    assert np.all(vbmc.optim_state["skip_active_sampling"] == False)
    assert np.all(vbmc.optim_state["run_mean"] == [])
    assert np.all(vbmc.optim_state["run_cov"] == [])
    assert np.all(np.isnan(vbmc.optim_state["last_run_avg"]))
    assert np.all(vbmc.optim_state["vp_K"] == vbmc.vp.K)
    assert np.all(vbmc.optim_state["pruned"] == 0)
    assert np.all(vbmc.optim_state["variance_regularized_acqfcn"] == True)
    assert np.all(vbmc.optim_state["search_cache"] == [])
    assert np.all(vbmc.optim_state["repeated_observations_streak"] == 0)
    assert np.all(vbmc.optim_state["data_trim_list"] == [])
    assert np.all(vbmc.optim_state["run_cov"] == [])


def test_vbmc_optimstate_iter_list():
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4)
    assert np.all(vbmc.optim_state["iter_list"]["u"] == [])
    assert np.all(vbmc.optim_state["iter_list"]["f_val"] == [])
    assert np.all(vbmc.optim_state["iter_list"]["f_sd"] == [])
    assert np.all(vbmc.optim_state["iter_list"]["fhyp"] == [])


def test_vbmc_optimstate_stop_sampling():
    options = {"ns_gp_max": 0}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    assert vbmc.optim_state["stop_sampling"] == np.inf
    options = {"ns_gp_max": 1}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    assert vbmc.optim_state["stop_sampling"] == 0


def test_vbmc_optimstate_warmup():
    options = {"warmup": True}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    assert vbmc.optim_state["warmup"]
    assert vbmc.optim_state["last_warmup"] == np.inf
    options = {"warmup": False}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    assert not vbmc.optim_state["warmup"]
    assert vbmc.optim_state["last_warmup"] == 0


def test_vbmc_optimstate_proposal_fcn():
    options = {"proposal_fcn": fun}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    assert vbmc.optim_state["proposal_fcn"] == fun
    options = {"proposal_fcn": None}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    assert vbmc.optim_state["proposal_fcn"] == "@(x)proposal_vbmc"


def test_vbmc_optimstate_entropy_switch():
    D = 3
    options = {"entropy_switch": False, "det_entropy_min_d": D - 1}
    vbmc = create_vbmc(D, 3, 1, 5, 2, 4, options)
    assert vbmc.optim_state["entropy_switch"] == False
    options = {"entropy_switch": True, "det_entropy_min_d": 1}
    vbmc = create_vbmc(D, 3, 1, 5, 2, 4, options)
    assert vbmc.optim_state["entropy_switch"] == True
    options = {"entropy_switch": True, "det_entropy_min_d": D + 1}
    vbmc = create_vbmc(D, 3, 1, 5, 2, 4, options)
    assert vbmc.optim_state["entropy_switch"] == False


def test_vbmc_optimstate_tol_gp_var():
    options = {"tol_gp_var": 0.0001}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    assert vbmc.optim_state["tol_gp_var"] == options.get("tol_gp_var")
    options = {"tol_gp_var": 0.002}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    assert vbmc.optim_state["tol_gp_var"] == options.get("tol_gp_var")


def test_vbmc_optimstate_max_fun_evals():
    D = 3
    options = {"max_fun_evals": 50 * (2 + D)}
    vbmc = create_vbmc(D, 3, 1, 5, 2, 4, options)
    assert vbmc.optim_state["max_fun_evals"] == options.get("max_fun_evals")
    options = {"max_fun_evals": 10}
    vbmc = create_vbmc(D, 3, 1, 5, 2, 4, options)
    assert vbmc.optim_state["max_fun_evals"] == options.get("max_fun_evals")


def test_vbmc_optimstate_uncertainty_handling_level():
    options = {"specify_target_noise": True}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    assert vbmc.optim_state["uncertainty_handling_level"] == 2
    options = {"specify_target_noise": False, "uncertainty_handling": [3]}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    assert vbmc.optim_state["uncertainty_handling_level"] == 1
    options = {"specify_target_noise": False, "uncertainty_handling": []}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    assert vbmc.optim_state["uncertainty_handling_level"] == 0


def test_vbmc_optimstate_acq_hedge():
    options = {"acq_hedge": True}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    assert vbmc.optim_state["hedge"] == []
    options = {"acq_hedge": False}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    assert "hedge" not in vbmc.optim_state


def test_vbmc_optimstate_entropy_alpha():
    options = {"det_entropy_alpha": False}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    assert np.all(vbmc.optim_state["entropy_alpha"] == False)
    options = {"det_entropy_alpha": True}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    assert np.all(vbmc.optim_state["entropy_alpha"] == True)


def test_vbmc_optimstate_int_mean_fun():
    options = {"gp_int_mean_fun": fun}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    assert np.all(vbmc.optim_state["int_mean_fun"] == fun)


def test_vbmc_optimstate_outwarp_delta():
    options = {"fitness_shaping": False}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    assert vbmc.optim_state["out_warp_delta"] == []
    out_warp_thresh_base = vbmc.options.get("out_warp_thresh_base")
    options = {"fitness_shaping": True}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    assert vbmc.optim_state["out_warp_delta"] == out_warp_thresh_base


def test_vbmc_init_log_joint():
    D = 3
    lb = np.ones((1, D)) * -1
    ub = np.ones((1, D))
    x0_array = np.zeros((1, D))
    plb = np.ones((1, D)) * -0.5
    pub = np.ones((1, D)) * 0.5

    def log_joint(x):
        return x**2 + x + 1

    def sample_prior(n):
        return np.random.normal(size=(n, D))

    vbmc = VBMC(
        log_joint, x0_array, lb, ub, plb, pub, sample_prior=sample_prior
    )
    x = np.random.normal()
    assert vbmc.log_joint is log_joint
    assert vbmc.function_logger.fun is log_joint
    assert vbmc.sample_prior is sample_prior
    with pytest.raises(AttributeError) as e_info:
        vbmc.log_prior
    with pytest.raises(AttributeError) as e_info:
        vbmc.log_likelihood

    def log_lklhd(x):
        return x**2

    def log_prior(x):
        return x + 1

    vbmc = VBMC(
        log_lklhd,
        x0_array,
        lb,
        ub,
        plb,
        pub,
        log_prior=log_prior,
        sample_prior=sample_prior,
    )
    x = np.random.normal()
    assert vbmc.log_joint(x) == log_joint(x)
    assert vbmc.function_logger.fun(x) == log_joint(x)
    assert vbmc.sample_prior is sample_prior
    assert vbmc.log_prior is log_prior
    assert vbmc.log_likelihood is log_lklhd


def test_vbmc_init_log_joint_noisy():
    options = {"specify_target_noise": 2}
    D = 3
    lb = np.ones((1, D)) * -1
    ub = np.ones((1, D))
    x0_array = np.zeros((1, D))
    plb = np.ones((1, D)) * -0.5
    pub = np.ones((1, D)) * 0.5

    def log_joint(x):
        return x**2 + x + 1, 1.0

    vbmc = VBMC(log_joint, x0_array, lb, ub, plb, pub, options=options)
    assert vbmc.log_joint is log_joint
    assert vbmc.function_logger.fun is log_joint
    with pytest.raises(AttributeError) as e_info:
        vbmc.log_prior
    with pytest.raises(AttributeError) as e_info:
        vbmc.log_likelihood

    def log_lklhd(x):
        return x**2, 1.0

    def log_prior(x):
        return x + 1

    vbmc = VBMC(
        log_lklhd,
        x0_array,
        lb,
        ub,
        plb,
        pub,
        log_prior=log_prior,
        options=options,
    )
    x = 5.6
    assert vbmc.log_joint(x) == log_joint(x)
    assert vbmc.function_logger.fun(x) == log_joint(x)
    assert vbmc.log_prior is log_prior
    assert vbmc.log_likelihood is log_lklhd


def test_init_integer_input():
    D = 2
    lb = np.full((1, D), -10)
    ub = np.full((1, D), 10)
    x0_array = np.full((1, D), 0)
    plb = np.full((1, D), -5)
    pub = np.full((1, D), 5)

    def log_joint(x):
        return x**2 + x + 1, 1.0

    vbmc = VBMC(log_joint, x0_array, lb, ub, plb, pub)
    for arr in [
        vbmc.optim_state["cache"]["x_orig"],
        vbmc.optim_state["lb_orig"],
        vbmc.optim_state["ub_orig"],
        vbmc.optim_state["plb_orig"],
        vbmc.optim_state["pub_orig"],
    ]:
        assert arr.dtype == np.float64


def test_init_1D_input():
    D = 2
    lb = np.full((D,), -10)
    ub = np.full((D,), 10)
    x0_array = np.full((D,), 0)
    plb = np.full((D,), -5)
    pub = np.full((D,), 5)

    def log_joint(x):
        return x**2 + x + 1, 1.0

    vbmc = VBMC(log_joint, x0_array, lb, ub, plb, pub)

    assert np.all(
        vbmc.optim_state["cache"]["x_orig"] == x0_array.reshape((1, D))
    )
    assert np.all(vbmc.optim_state["lb_orig"] == lb.reshape((1, D)))
    assert np.all(vbmc.optim_state["ub_orig"] == ub.reshape((1, D)))
    assert np.all(vbmc.optim_state["plb_orig"] == plb.reshape((1, D)))
    assert np.all(vbmc.optim_state["pub_orig"] == pub.reshape((1, D)))


def test__str__and__repr__():
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4)
    vbmc.__str__()
    vbmc.__repr__()
