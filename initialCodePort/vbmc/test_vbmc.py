import sys

import numpy as np
import pytest

from vbmc import VBMC

fun = lambda x: np.sum(x + 2)


def test_vbmc_init_no_x0_PLB_PUB():
    with pytest.raises(ValueError) as execinfo:
        VBMC(fun)
    assert "vbmc:UnknownDims If no starting point is" in execinfo.value.args[0]


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
    _, lb2, ub2, plb, pub = VBMC(fun, x0, lb, ub)._boundscheck(fun, x0, lb, ub)
    assert np.all(lb == lb2)
    assert np.all(ub == ub2)
    assert np.all(plb == lb + 2 * 1e-3)
    assert np.all(pub == ub - 2 * 1e-3)


def test_vbmc_boundscheck_no_PUB_PLB_n0_3():
    D = 3
    lb = np.ones((1, D)) * -2
    ub = np.ones((1, D)) * 2
    x0 = np.concatenate((np.ones((1, D)) * -0.75, np.ones((1, D)) * 0.75))
    _, lb2, ub2, plb, pub = VBMC(fun, x0, lb, ub)._boundscheck(fun, x0, lb, ub)
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
    _, lb2, ub2, plb, pub = VBMC(fun, x0, lb, ub)._boundscheck(
        fun, x0, lb, ub, plb
    )
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
        VBMC(fun, x0, lb, ub)._boundscheck(fun, x0, incorrect, ub, plb, pub)
    assert exception_message in execinfo1.value.args[0]
    with pytest.raises(ValueError) as execinfo2:
        VBMC(fun, x0, lb, ub)._boundscheck(fun, x0, lb, incorrect, plb, pub)
    assert exception_message in execinfo2.value.args[0]
    with pytest.raises(ValueError) as execinfo3:
        VBMC(fun, x0, lb, ub)._boundscheck(fun, x0, lb, ub, incorrect, pub)
    assert exception_message in execinfo3.value.args[0]
    with pytest.raises(ValueError) as execinfo4:
        VBMC(fun, x0, lb, ub)._boundscheck(fun, x0, lb, ub, plb, incorrect)
    assert exception_message in execinfo4.value.args[0]
    VBMC(fun, x0, lb, ub)._boundscheck(fun, x0, lb, ub, plb, pub)


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
        VBMC(fun, x0, lb, ub)._boundscheck(fun, x0, incorrect, ub, plb, pub)
    assert exception_message in execinfo1.value.args[0]
    with pytest.raises(ValueError) as execinfo2:
        VBMC(fun, x0, lb, ub)._boundscheck(fun, x0, lb, incorrect, plb, pub)
    assert exception_message in execinfo2.value.args[0]
    with pytest.raises(ValueError) as execinfo3:
        VBMC(fun, x0, lb, ub)._boundscheck(fun, x0, lb, ub, incorrect, pub)
    assert exception_message in execinfo3.value.args[0]
    with pytest.raises(ValueError) as execinfo4:
        VBMC(fun, x0, lb, ub)._boundscheck(fun, x0, lb, ub, plb, incorrect)
    assert exception_message in execinfo4.value.args[0]
    VBMC(fun, x0, lb, ub)._boundscheck(fun, x0, lb, ub, plb, pub)


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
        VBMC(fun, x0, lb, ub)._boundscheck(fun, x0, incorrect, ub, plb, pub)
    assert exception_message in execinfo1.value.args[0]
    with pytest.raises(ValueError) as execinfo2:
        VBMC(fun, x0, lb, ub)._boundscheck(fun, x0, lb, incorrect, plb, pub)
    assert exception_message in execinfo2.value.args[0]
    with pytest.raises(ValueError) as execinfo3:
        VBMC(fun, x0, lb, ub)._boundscheck(fun, x0, lb, ub, incorrect, pub)
    assert exception_message in execinfo3.value.args[0]
    with pytest.raises(ValueError) as execinfo4:
        VBMC(fun, x0, lb, ub)._boundscheck(fun, x0, lb, ub, plb, incorrect)
    assert exception_message in execinfo4.value.args[0]
    VBMC(fun, x0, lb, ub)._boundscheck(fun, x0, lb, ub, plb, pub)


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
        VBMC(fun, x0, lb, ub)._boundscheck(fun, x0, incorrect, ub, plb, pub)
    assert exception_message in execinfo1.value.args[0]
    with pytest.raises(ValueError) as execinfo2:
        VBMC(fun, x0, lb, ub)._boundscheck(fun, x0, lb, incorrect, plb, pub)
    assert exception_message in execinfo2.value.args[0]
    with pytest.raises(ValueError) as execinfo3:
        VBMC(fun, x0, lb, ub)._boundscheck(fun, x0, lb, ub, incorrect, pub)
    assert exception_message in execinfo3.value.args[0]
    with pytest.raises(ValueError) as execinfo4:
        VBMC(fun, x0, lb, ub)._boundscheck(fun, x0, lb, ub, plb, incorrect)
    assert exception_message in execinfo4.value.args[0]
    VBMC(fun, x0, lb, ub)._boundscheck(fun, x0, lb, ub, plb, pub)


def test_vbmc_boundscheck_fixed():
    D = 3
    lb = np.ones((1, D)) * -2
    ub = np.ones((1, D)) * 2
    x0 = np.ones((2, D))
    fixed_bound = np.ones((1, D))
    with pytest.raises(ValueError) as execinfo:
        VBMC(fun, x0, lb, ub)._boundscheck(
            fun, x0, fixed_bound, fixed_bound, fixed_bound, fixed_bound
        )
    assert "VBMC does not support fixed" in execinfo.value.args[0]


def test_vbmc_boundscheck_PLB_PUB_different():
    D = 3
    lb = np.ones((1, D)) * -2
    ub = np.ones((1, D)) * 2
    x0 = np.ones((2, D))
    pb = np.ones((1, D))
    with pytest.raises(ValueError) as execinfo:
        VBMC(fun, x0, lb, ub)._boundscheck(fun, x0, lb, ub, pb, pb)
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
        VBMC(fun, x0, lb, ub)._boundscheck(fun, x0_large, lb, ub, plb, pub)
    assert exception_message in execinfo1.value.args[0]
    with pytest.raises(ValueError) as execinfo2:
        VBMC(fun, x0, lb, ub)._boundscheck(fun, x0_small, lb, ub, plb, pub)
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
        VBMC(fun, x0, lb, ub)._boundscheck(fun, x0, lb, ub, pub, plb)
    assert exception_message in execinfo1.value.args[0]
    with pytest.raises(ValueError) as execinfo2:
        VBMC(fun, x0, lb, ub)._boundscheck(fun, x0, plb, ub, lb, pub)
    assert exception_message in execinfo2.value.args[0]
    with pytest.raises(ValueError) as execinfo2:
        VBMC(fun, x0, lb, ub)._boundscheck(fun, x0, lb, pub, plb, ub)
    assert exception_message in execinfo2.value.args[0]
    VBMC(fun, x0, lb, ub)._boundscheck(fun, x0, lb, ub, lb, ub)


def test_vbmc_boundcheck_half_bounded():
    D = 3
    lb = np.ones((1, D)) * -2
    ub = np.ones((1, D)) * 2
    x0 = np.ones((2, D)) * 0.5
    plb = np.ones((1, D)) * -1
    pub = np.ones((1, D))
    exception_message = "Variables bounded only below/above are not supported"
    with pytest.raises(ValueError) as execinfo1:
        VBMC(fun, x0, lb, ub)._boundscheck(fun, x0, lb * np.inf, ub, plb, pub)
    assert exception_message in execinfo1.value.args[0]
    with pytest.raises(ValueError) as execinfo2:
        VBMC(fun, x0, lb, ub)._boundscheck(fun, x0, lb, ub * np.inf, plb, pub)
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
            fun,
            identicial,
            identicial,
            identicial + realmin * 1,
            identicial,
            identicial + realmin * 1,
        )
    assert exception_message in execinfo1.value.args[0]
    # this should be the minimum values with which no exception is being raised
    VBMC(fun, x0, lb, ub)._boundscheck(
        fun,
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
        fun,
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
        VBMC(fun, x0, lb, ub)._boundscheck(fun, x0, lb, ub, plb, pub * np.inf)
    assert exception_message in execinfo1.value.args[0]
    with pytest.raises(ValueError) as execinfo2:
        VBMC(fun, x0, lb, ub)._boundscheck(fun, x0, lb, ub, plb * np.inf, pub)
    assert exception_message in execinfo2.value.args[0]


def test_vbmc_boundcheck_plausible_bounds_too_close_to_hardbounds():
    D = 3
    lb = np.ones((1, D)) * -2
    ub = np.ones((1, D)) * 2
    x0 = np.zeros((2, D))
    _, _, _, plb2, pub2 = VBMC(fun, x0, lb, ub)._boundscheck(
        fun,
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
        fun,
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


def test_vbmc_setupvars_integervars():
    user_options = {"integervars": np.array([1, 0, 0])}
    D = 3
    lb = np.ones((1, D)) * 1
    ub = np.ones((1, D)) * 4
    x0 = np.ones((2, D)) * 2
    plb = np.ones((1, D)) * 1
    pub = np.ones((1, D)) * 3
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
    assert np.all(vbmc.optimState.get("integervars") == integervars)


def test_vbmc_setupvars_fvals():
    D = 3
    lb = np.ones((1, D)) * 1
    ub = np.ones((1, D)) * 4
    x0 = np.ones((2, D)) * 2
    plb = np.ones((1, D)) * 1
    pub = np.ones((1, D)) * 3
    exception_message = (
        "points in X0 and of their function values as specified"
    )
    with pytest.raises(ValueError) as execinfo1:
        user_options = {"fvals": np.zeros((3, 1))}
        VBMC(fun, x0, lb, ub, plb, pub, user_options)
    assert exception_message in execinfo1.value.args[0]
    with pytest.raises(ValueError) as execinfo2:
        user_options = {"fvals": np.zeros((1, 1))}
        VBMC(fun, x0, lb, ub, plb, pub, user_options)
    assert exception_message in execinfo2.value.args[0]
    user_options = {"fvals": [1, 2]}
    x0 = np.array(([[1, 2, 3], [3, 4, 3]]))
    VBMC(fun, x0, lb, ub, plb, pub, user_options)


def test_vbmc_setupvars_invalid_gp_mean_function():
    D = 3
    lb = np.ones((1, D)) * 1
    ub = np.ones((1, D)) * 4
    x0 = np.ones((2, D)) * 2
    plb = np.ones((1, D)) * 1
    pub = np.ones((1, D)) * 3
    exception_message = "vbmc:UnknownGPmean:Unknown/unsupported GP mean"
    with pytest.raises(ValueError) as execinfo1:
        user_options = {"gpmeanfun": "notvalid"}
        VBMC(fun, x0, lb, ub, plb, pub, user_options)
    assert exception_message in execinfo1.value.args[0]
    with pytest.raises(ValueError) as execinfo2:
        user_options = {"gpmeanfun": ""}
        VBMC(fun, x0, lb, ub, plb, pub, user_options)
    assert exception_message in execinfo2.value.args[0]
    user_options = {"gpmeanfun": "const"}
    x0 = np.array(([[1, 2, 3], [3, 4, 3]]))
    VBMC(fun, x0, lb, ub, plb, pub, user_options)

    