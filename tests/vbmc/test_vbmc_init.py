import sys

import numpy as np
import pytest
from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.vbmc import VBMC

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
    user_options: dict = None,
):
    lb = np.ones((1, D)) * lower_bounds
    ub = np.ones((1, D)) * upper_bounds
    x0_array = np.ones((2, D)) * x0
    plb = np.ones((1, D)) * plausible_lower_bounds
    pub = np.ones((1, D)) * plausible_upper_bounds
    return VBMC(fun, x0_array, lb, ub, plb, pub, user_options)


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


def test_vbmc_boundscheck_no_PUB_PLB_n0_1():
    D = 3
    lb = np.zeros((1, D))
    ub = np.ones((1, D)) * 2
    x0 = np.ones((1, D))
    _, lb2, ub2, plb, pub = VBMC(fun, x0, lb, ub)._boundscheck(x0, lb, ub)
    assert np.all(lb == lb2)
    assert np.all(ub == ub2)
    assert np.all(plb == lb + 2 * 1e-3)
    assert np.all(pub == ub - 2 * 1e-3)


def test_vbmc_boundscheck_no_PUB_PLB_n0_3():
    D = 3
    lb = np.ones((1, D)) * -2
    ub = np.ones((1, D)) * 2
    x0 = np.concatenate((np.ones((1, D)) * -0.75, np.ones((1, D)) * 0.75))
    _, lb2, ub2, plb, pub = VBMC(fun, x0, lb, ub)._boundscheck(x0, lb, ub)
    assert np.all(lb == lb2)
    assert np.all(ub == ub2)
    assert np.all(plb == np.ones((1, D)) * -1.5)
    assert np.all(pub == np.ones((1, D)) * 1.5)


def test_vbmc_boundscheck_no_PUB_PLB_identical():
    D = 3
    lb = np.ones((1, D)) * -2
    ub = np.ones((1, D)) * 2
    x0 = np.ones((2, D))
    plb = np.ones((1, D))
    _, lb2, ub2, plb, pub = VBMC(fun, x0, lb, ub)._boundscheck(x0, lb, ub, plb)
    assert np.all(lb == lb2)
    assert np.all(ub == ub2)
    assert np.all(plb == lb + 4 * 1e-3)
    assert np.all(pub == ub - 4 * 1e-3)


def test_vbmc_boundscheck_not_D():
    D = 3
    lb = np.ones((1, D)) * -2
    ub = np.ones((1, D)) * 2
    plb = np.ones((1, D)) * -1
    pub = np.ones((1, D))
    x0 = np.ones((2, D))
    incorrect = np.ones((1, D - 1))
    exception_message = "need to be row vectors with D elements"
    with pytest.raises(ValueError) as execinfo1:
        VBMC(fun, x0, lb, ub)._boundscheck(x0, incorrect, ub, plb, pub)
    assert exception_message in execinfo1.value.args[0]
    with pytest.raises(ValueError) as execinfo2:
        VBMC(fun, x0, lb, ub)._boundscheck(x0, lb, incorrect, plb, pub)
    assert exception_message in execinfo2.value.args[0]
    with pytest.raises(ValueError) as execinfo3:
        VBMC(fun, x0, lb, ub)._boundscheck(x0, lb, ub, incorrect, pub)
    assert exception_message in execinfo3.value.args[0]
    with pytest.raises(ValueError) as execinfo4:
        VBMC(fun, x0, lb, ub)._boundscheck(x0, lb, ub, plb, incorrect)
    assert exception_message in execinfo4.value.args[0]
    VBMC(fun, x0, lb, ub)._boundscheck(x0, lb, ub, plb, pub)


def test_vbmc_boundscheck_not_vectors():
    D = 3
    lb = np.ones((1, D)) * -2
    ub = np.ones((1, D)) * 2
    plb = np.ones((1, D)) * -1
    pub = np.ones((1, D))
    x0 = np.ones((2, D))
    incorrect = 1
    exception_message = "need to be row vectors with D elements"
    with pytest.raises(ValueError) as execinfo1:
        VBMC(fun, x0, lb, ub)._boundscheck(x0, incorrect, ub, plb, pub)
    assert exception_message in execinfo1.value.args[0]
    with pytest.raises(ValueError) as execinfo2:
        VBMC(fun, x0, lb, ub)._boundscheck(x0, lb, incorrect, plb, pub)
    assert exception_message in execinfo2.value.args[0]
    with pytest.raises(ValueError) as execinfo3:
        VBMC(fun, x0, lb, ub)._boundscheck(x0, lb, ub, incorrect, pub)
    assert exception_message in execinfo3.value.args[0]
    with pytest.raises(ValueError) as execinfo4:
        VBMC(fun, x0, lb, ub)._boundscheck(x0, lb, ub, plb, incorrect)
    assert exception_message in execinfo4.value.args[0]
    VBMC(fun, x0, lb, ub)._boundscheck(x0, lb, ub, plb, pub)


def test_vbmc_boundscheck_not_row_vectors():
    D = 3
    lb = np.ones((1, D)) * -2
    ub = np.ones((1, D)) * 2
    x0 = np.ones((2, D))
    plb = np.ones((1, D)) * -1
    pub = np.ones((1, D))
    incorrect = np.ones((D, 1))
    exception_message = "need to be row vectors with D elements"
    with pytest.raises(ValueError) as execinfo1:
        VBMC(fun, x0, lb, ub)._boundscheck(x0, incorrect, ub, plb, pub)
    assert exception_message in execinfo1.value.args[0]
    with pytest.raises(ValueError) as execinfo2:
        VBMC(fun, x0, lb, ub)._boundscheck(x0, lb, incorrect, plb, pub)
    assert exception_message in execinfo2.value.args[0]
    with pytest.raises(ValueError) as execinfo3:
        VBMC(fun, x0, lb, ub)._boundscheck(x0, lb, ub, incorrect, pub)
    assert exception_message in execinfo3.value.args[0]
    with pytest.raises(ValueError) as execinfo4:
        VBMC(fun, x0, lb, ub)._boundscheck(x0, lb, ub, plb, incorrect)
    assert exception_message in execinfo4.value.args[0]
    VBMC(fun, x0, lb, ub)._boundscheck(x0, lb, ub, plb, pub)


def test_vbmc_boundscheck_plb_pub_not_finite():
    D = 3
    lb = np.ones((1, D)) * -2
    ub = np.ones((1, D)) * 2
    x0 = np.ones((2, D))
    plb = np.ones((1, D)) * -1
    pub = np.ones((1, D))
    incorrect = np.array([[1 + 2j, 3 + 4j, 5 + 6j]])
    exception_message = "need to be real valued"
    with pytest.raises(ValueError) as execinfo1:
        VBMC(fun, x0, lb, ub)._boundscheck(x0, incorrect, ub, plb, pub)
    assert exception_message in execinfo1.value.args[0]
    with pytest.raises(ValueError) as execinfo2:
        VBMC(fun, x0, lb, ub)._boundscheck(x0, lb, incorrect, plb, pub)
    assert exception_message in execinfo2.value.args[0]
    with pytest.raises(ValueError) as execinfo3:
        VBMC(fun, x0, lb, ub)._boundscheck(x0, lb, ub, incorrect, pub)
    assert exception_message in execinfo3.value.args[0]
    with pytest.raises(ValueError) as execinfo4:
        VBMC(fun, x0, lb, ub)._boundscheck(x0, lb, ub, plb, incorrect)
    assert exception_message in execinfo4.value.args[0]
    VBMC(fun, x0, lb, ub)._boundscheck(x0, lb, ub, plb, pub)


def test_vbmc_boundscheck_fixed():
    D = 3
    lb = np.ones((1, D)) * -2
    ub = np.ones((1, D)) * 2
    x0 = np.ones((2, D))
    fixed_bound = np.ones((1, D))
    with pytest.raises(ValueError) as execinfo:
        VBMC(fun, x0, lb, ub)._boundscheck(
            x0, fixed_bound, fixed_bound, fixed_bound, fixed_bound
        )
    assert "VBMC does not support fixed" in execinfo.value.args[0]


def test_vbmc_boundscheck_PLB_PUB_different():
    D = 3
    lb = np.ones((1, D)) * -2
    ub = np.ones((1, D)) * 2
    x0 = np.ones((2, D))
    pb = np.ones((1, D))
    with pytest.raises(ValueError) as execinfo:
        VBMC(fun, x0, lb, ub)._boundscheck(x0, lb, ub, pb, pb)
    assert (
        "plausible lower and upper bounds need to be distinct"
        in execinfo.value.args[0]
    )


def test_vbmc_boundscheck_x0_outside_lb_ub():
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
        VBMC(fun, x0, lb, ub)._boundscheck(x0_large, lb, ub, plb, pub)
    assert exception_message in execinfo1.value.args[0]
    with pytest.raises(ValueError) as execinfo2:
        VBMC(fun, x0, lb, ub)._boundscheck(x0_small, lb, ub, plb, pub)
    assert exception_message in execinfo2.value.args[0]


def test_vbmc_boundscheck_ordering():
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
        VBMC(fun, x0, lb, ub)._boundscheck(x0, lb, ub, pub, plb)
    assert exception_message in execinfo1.value.args[0]
    with pytest.raises(ValueError) as execinfo2:
        VBMC(fun, x0, lb, ub)._boundscheck(x0, plb, ub, lb, pub)
    assert exception_message in execinfo2.value.args[0]
    with pytest.raises(ValueError) as execinfo2:
        VBMC(fun, x0, lb, ub)._boundscheck(x0, lb, pub, plb, ub)
    assert exception_message in execinfo2.value.args[0]
    VBMC(fun, x0, lb, ub)._boundscheck(x0, lb, ub, lb, ub)


def test_vbmc_boundcheck_half_bounded():
    D = 3
    lb = np.ones((1, D)) * -2
    ub = np.ones((1, D)) * 2
    x0 = np.ones((2, D)) * 0.5
    plb = np.ones((1, D)) * -1
    pub = np.ones((1, D))
    exception_message = "Variables bounded only below/above are not supported"
    with pytest.raises(ValueError) as execinfo1:
        VBMC(fun, x0, lb, ub)._boundscheck(x0, lb * np.inf, ub, plb, pub)
    assert exception_message in execinfo1.value.args[0]
    with pytest.raises(ValueError) as execinfo2:
        VBMC(fun, x0, lb, ub)._boundscheck(x0, lb, ub * np.inf, plb, pub)
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
        VBMC(fun, x0, lb, ub)._boundscheck(
            identicial,
            identicial,
            identicial + realmin * 1,
            identicial,
            identicial + realmin * 1,
        )
    assert exception_message in execinfo1.value.args[0]
    # this should be the minimum values with which no exception is being raised
    VBMC(fun, x0, lb, ub)._boundscheck(
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
    x0_2, _, _, _, _ = VBMC(fun, x0, lb, ub)._boundscheck(
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
        VBMC(fun, x0, lb, ub)._boundscheck(x0, lb, ub, plb, pub * np.inf)
    assert exception_message in execinfo1.value.args[0]
    with pytest.raises(ValueError) as execinfo2:
        VBMC(fun, x0, lb, ub)._boundscheck(x0, lb, ub, plb * np.inf, pub)
    assert exception_message in execinfo2.value.args[0]


def test_vbmc_boundcheck_plausible_bounds_too_close_to_hardbounds():
    D = 3
    lb = np.ones((1, D)) * -2
    ub = np.ones((1, D)) * 2
    x0 = np.zeros((2, D))
    _, _, _, plb2, pub2 = VBMC(fun, x0, lb, ub)._boundscheck(
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
    x0_2, _, _, plb2, pub2 = VBMC(fun, x0, lb, ub)._boundscheck(
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


def test_vbmc_optimstate_integervars():
    user_options = {"integervars": np.array([1, 0, 0])}
    D = 3
    lb = np.ones((1, D)) * 1
    ub = np.ones((1, D)) * 5
    x0 = np.ones((2, D)) * 3
    plb = np.ones((1, D)) * 2
    pub = np.ones((1, D)) * 4
    exception_message = "set at +/- 0.5 points from their boundary values"
    with pytest.raises(ValueError) as execinfo1:
        VBMC(fun, x0, lb * -np.inf, ub * np.inf, plb, pub, user_options)
    assert exception_message in execinfo1.value.args[0]
    lb[0] = -np.inf
    ub[0] = np.inf
    with pytest.raises(ValueError) as execinfo2:
        VBMC(fun, x0, lb, ub, plb, pub, user_options)
    assert exception_message in execinfo2.value.args[0]
    lb[0] = -10
    ub[0] = 10
    with pytest.raises(ValueError) as execinfo3:
        VBMC(fun, x0, lb, ub, plb, pub, user_options)
    assert exception_message in execinfo3.value.args[0]
    lb[0] = -10.5
    ub[0] = 10.5
    vbmc = VBMC(fun, x0, lb, ub, plb, pub, user_options)
    integervars = np.full((1, D), False)
    integervars[:, 0] = True
    assert np.all(vbmc.optim_state.get("integervars") == integervars)


def test_vbmc_setupvars_fvals():
    exception_message = (
        "points in X0 and of their function values as specified"
    )
    with pytest.raises(ValueError) as execinfo1:
        user_options = {"fvals": np.zeros((3, 1))}
        create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    assert exception_message in execinfo1.value.args[0]
    with pytest.raises(ValueError) as execinfo2:
        user_options = {"fvals": np.zeros((1, 1))}
        create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    assert exception_message in execinfo2.value.args[0]

    user_options = {"fvals": [1, 2]}
    x0 = np.array(([[1, 2, 3], [3, 4, 3]]))
    D = 3
    lb = np.ones((1, D)) * 1
    ub = np.ones((1, D)) * 5
    plb = np.ones((1, D)) * 2
    pub = np.ones((1, D)) * 4
    vbmc = VBMC(fun, x0, lb, ub, plb, pub, user_options)
    assert np.all(
        vbmc.optim_state.get("cache").get("y_orig")
        == user_options.get("fvals")
    )
    assert np.all(vbmc.optim_state.get("cache").get("x_orig") is not None)
    assert vbmc.optim_state.get("cache_active")


def test_vbmc_optimstate_gp_functions():
    exception_message = "vbmc:UnknownGPmean:Unknown/unsupported GP mean"
    with pytest.raises(ValueError) as execinfo1:
        user_options = {"gpmeanfun": "notvalid"}
        create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    assert exception_message in execinfo1.value.args[0]
    with pytest.raises(ValueError) as execinfo2:
        user_options = {"gpmeanfun": ""}
        create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    assert exception_message in execinfo2.value.args[0]
    user_options = {"gpmeanfun": "const"}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    assert vbmc.optim_state.get("gp_meanfun") == user_options.get("gpmeanfun")
    # uncertainty_handling_level 2
    assert vbmc.optim_state["gp_covfun"] == 1
    user_options = {"specifytargetnoise": True}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    # uncertainty_handling_level 1
    assert vbmc.optim_state["gp_noisefun"] == [1, 1]
    user_options = {"specifytargetnoise": False, "uncertaintyhandling": []}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    assert vbmc.optim_state["gp_noisefun"] == [1, 2]
    # uncertainty_handling_level 0
    user_options = {
        "specifytargetnoise": False,
        "uncertaintyhandling": [3],
        "noiseshaping": True,
    }
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    assert vbmc.optim_state["uncertainty_handling_level"] == 0
    assert vbmc.optim_state["gp_noisefun"] == [1, 1]
    user_options = {
        "specifytargetnoise": False,
        "uncertaintyhandling": [3],
        "noiseshaping": False,
    }
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    assert vbmc.optim_state["uncertainty_handling_level"] == 0
    assert vbmc.optim_state["gp_noisefun"] == [1, 0]


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
    eps = vbmc.options.get("tolboundx") * 4
    assert np.all(vbmc.optim_state["lb_eps_orig"] == lb + eps)
    assert np.all(vbmc.optim_state["ub_eps_orig"] == ub - eps)
    assert np.all(vbmc.optim_state["lb"] == -np.inf)
    assert np.all(vbmc.optim_state["ub"] == np.inf)
    assert np.all(vbmc.optim_state["plb"] == -0.5)
    assert np.all(vbmc.optim_state["pub"] == 0.5)
    assert np.all(vbmc.optim_state["lb_search"] == -2.5)
    assert np.all(vbmc.optim_state["ub_search"] == 2.5)


def test_vbmc_optimstate_constants():
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4)
    assert np.all(vbmc.optim_state["iter"] == 0)
    assert np.all(vbmc.optim_state["sn2hpd"] == np.inf)
    assert np.all(vbmc.optim_state["last_warping"] == -np.inf)
    assert np.all(vbmc.optim_state["last_successful_warping"] == -np.inf)
    assert np.all(vbmc.optim_state["warping_count"] == 0)
    assert np.all(vbmc.optim_state["recompute_var_post"] == True)
    assert np.all(vbmc.optim_state["warmup_stable_count"] == 0)
    assert np.all(vbmc.optim_state["r"] == np.inf)
    assert np.all(vbmc.optim_state["skip_active_sampling"] == False)
    assert np.all(vbmc.optim_state["run_mean"] == [])
    assert np.all(vbmc.optim_state["run_cov"] == [])
    assert np.all(np.isnan(vbmc.optim_state["last_run_avg"]))
    assert np.all(vbmc.optim_state["vpk"] == vbmc.K)
    assert np.all(vbmc.optim_state["pruned"] == 0)
    assert np.all(vbmc.optim_state["variance_regularized_acqfcn"] == True)
    assert np.all(vbmc.optim_state["search_cache"] == [])
    assert np.all(vbmc.optim_state["vp_repo"] == [])
    assert np.all(vbmc.optim_state["repeated_observations_streak"] == 0)
    assert np.all(vbmc.optim_state["data_trim_list"] == [])
    assert np.all(vbmc.optim_state["run_cov"] == [])


def test_vbmc_optimstate_iterlist():
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4)
    assert np.all(vbmc.optim_state["iterlist"]["u"] == [])
    assert np.all(vbmc.optim_state["iterlist"]["fval"] == [])
    assert np.all(vbmc.optim_state["iterlist"]["fsd"] == [])
    assert np.all(vbmc.optim_state["iterlist"]["fhyp"] == [])


def test_vbmc_optimstate_stop_sampling():
    user_options = {"nsgpmax": 0}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    assert vbmc.optim_state["stop_sampling"] == np.inf
    user_options = {"nsgpmax": 1}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    assert vbmc.optim_state["stop_sampling"] == 0


def test_vbmc_optimstate_warmup():
    user_options = {"warmup": True}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    assert vbmc.optim_state["warmup"]
    assert vbmc.optim_state["last_warmup"] == np.inf
    user_options = {"warmup": False}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    assert not vbmc.optim_state["warmup"]
    assert vbmc.optim_state["last_warmup"] == 0


def test_vbmc_optimstate_proposalfcn():
    user_options = {"proposalfcn": fun}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    assert vbmc.optim_state["proposalfcn"] == fun
    user_options = {"proposalfcn": None}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    assert vbmc.optim_state["proposalfcn"] == "@(x)proposal_vbmc"


def test_vbmc_optimstate_entropy_switch():
    D = 3
    user_options = {"entropyswitch": False, "detentropymind": D - 1}
    vbmc = create_vbmc(D, 3, 1, 5, 2, 4, user_options)
    assert vbmc.optim_state["entropy_switch"] == False
    user_options = {"entropyswitch": True, "detentropymind": 1}
    vbmc = create_vbmc(D, 3, 1, 5, 2, 4, user_options)
    assert vbmc.optim_state["entropy_switch"] == True
    user_options = {"entropyswitch": True, "detentropymind": D + 1}
    vbmc = create_vbmc(D, 3, 1, 5, 2, 4, user_options)
    assert vbmc.optim_state["entropy_switch"] == False


def test_vbmc_optimstate_tol_gp_var():
    user_options = {"tolgpvar": 0.0001}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    assert vbmc.optim_state["tol_gp_var"] == user_options.get("tolgpvar")
    user_options = {"tolgpvar": 0.002}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    assert vbmc.optim_state["tol_gp_var"] == user_options.get("tolgpvar")


def test_vbmc_optimstate_max_fun_evals():
    D = 3
    user_options = {"maxfunevals": 50 * (2 + D)}
    vbmc = create_vbmc(D, 3, 1, 5, 2, 4, user_options)
    assert vbmc.optim_state["max_fun_evals"] == user_options.get("maxfunevals")
    user_options = {"maxfunevals": 10}
    vbmc = create_vbmc(D, 3, 1, 5, 2, 4, user_options)
    assert vbmc.optim_state["max_fun_evals"] == user_options.get("maxfunevals")


def test_vbmc_optimstate_uncertainty_handling_level():
    user_options = {"specifytargetnoise": True}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    assert vbmc.optim_state["uncertainty_handling_level"] == 2
    user_options = {"specifytargetnoise": False, "uncertaintyhandling": []}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    assert vbmc.optim_state["uncertainty_handling_level"] == 1
    user_options = {"specifytargetnoise": False, "uncertaintyhandling": [3]}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    assert vbmc.optim_state["uncertainty_handling_level"] == 0


def test_vbmc_optimstate_acqhedge():
    user_options = {"acqhedge": True}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    assert vbmc.optim_state["hedge"] == []
    user_options = {"acqhedge": False}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    assert "hedge" not in vbmc.optim_state


def test_vbmc_optimstate_delta():
    user_options = {"bandwidth": 1}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    assert np.all(vbmc.optim_state["delta"] == 1)


def test_vbmc_optimstate_entropy_alpha():
    user_options = {"detentropyalpha": False}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    assert np.all(vbmc.optim_state["entropy_alpha"] == False)
    user_options = {"detentropyalpha": True}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    assert np.all(vbmc.optim_state["entropy_alpha"] == True)


def test_vbmc_optimstate_int_meanfun():
    user_options = {"gpintmeanfun": fun}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    assert np.all(vbmc.optim_state["int_meanfun"] == fun)



def test_vbmc_optimstate_outwarp_delta():
    user_options = {"fitnessshaping": False}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    assert vbmc.optim_state["outwarp_delta"] == []
    outwarpthreshbase = vbmc.options.get("outwarpthreshbase")
    user_options = {"fitnessshaping": True}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    assert vbmc.optim_state["outwarp_delta"] == outwarpthreshbase


def test_vbmc_optimize(mocker):
    """
    This is WIP as it should simulate a full run of VBMC later but this requires
    more setup.
    """    
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4)
    mocker.patch(
        "pyvbmc.vbmc.VBMC.finalboost",
        return_value=(VariationalPosterior(3), 10, 10, False),
    )
    mocker.patch(
        "pyvbmc.vbmc.VBMC.determine_best_vp",
        return_value=(VariationalPosterior(3), 10, 10, 1),
    )
    mocker.patch(
        "pyvbmc.vbmc.VBMC._check_warmup_end_conditions",
        return_value=True,
    )
    mocker.patch("pyvbmc.vbmc.VBMC._setup_vbmc_after_warmup")
    vbmc.optimize()
