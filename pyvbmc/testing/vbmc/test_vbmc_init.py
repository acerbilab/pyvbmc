import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy as sp
import scipy.stats

from pyvbmc import VBMC
from pyvbmc.acquisition_functions import AcqFcnLog, AcqFcnVIQR
from pyvbmc.priors import (
    Prior,
    Product,
    SciPy,
    SmoothBox,
    SplineTrapezoidal,
    Trapezoidal,
    UniformBox,
    UserFunction,
    convert_to_prior,
)
from pyvbmc.vbmc.vbmc import _check_prior_covers_bounds

priors = [UniformBox, Trapezoidal, SplineTrapezoidal, SmoothBox, SciPy]
from scipy.stats import lognorm, multivariate_normal, multivariate_t, norm

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


def test_vbmc_bounds_check_scalars_are_replicated():
    """``misc/boundscheck_vbmc.m:6-10`` replicates each of the four bounds
    given as a single value across the variables, and the class docstring
    promises the same."""
    D = 3
    x0 = np.ones((2, D))
    vbmc = VBMC(fun, x0, -2, 2, -1, 1)
    for bound, value in (
        (vbmc.lower_bounds, -2),
        (vbmc.upper_bounds, 2),
        (vbmc.plausible_lower_bounds, -1),
        (vbmc.plausible_upper_bounds, 1),
    ):
        assert bound.shape == (1, D)
        assert np.all(bound == value)


def test_vbmc_scalar_plausible_bounds_without_x0_name_the_problem():
    """Without a starting point the number of variables comes from the
    plausible bounds, so two scalars leave it unknown, and the error says
    that."""
    with pytest.raises(ValueError, match="number of variables"):
        VBMC(fun, None, -10, 10, -1, 1)
    with pytest.raises(ValueError, match="number of variables"):
        VBMC(fun, None, -10, 10, np.float64(-1), np.float64(1))


def test_vbmc_one_scalar_plausible_bound_without_x0_is_replicated():
    """One plausible bound with an entry per variable gives their number,
    and the other, a scalar, is replicated as the docstring promises."""
    D = 3
    vbmc = VBMC(fun, None, -10, 10, np.full((1, D), -1.0), 1)
    assert vbmc.D == D
    assert vbmc.plausible_upper_bounds.shape == (1, D)
    assert np.all(vbmc.plausible_upper_bounds == 1)

    vbmc = VBMC(fun, None, -10, 10, -1, [1.0, 1.0, 1.0])
    assert vbmc.D == D
    assert np.all(vbmc.plausible_lower_bounds == -1)


def test_vbmc_bounds_check_scalars_with_a_degenerate_starting_set():
    """A starting set without width leaves the plausible box without
    width, and the hard bounds take its place, which needs the replicated
    bound to carry one entry per variable."""
    D = 1
    x0 = np.array([[2.0], [2.0]])
    vbmc = VBMC(fun, x0, -10, 10)
    assert vbmc.plausible_lower_bounds.shape == (1, D)
    assert vbmc.plausible_upper_bounds.shape == (1, D)
    assert np.all(vbmc.plausible_lower_bounds < vbmc.plausible_upper_bounds)


def test_vbmc_bounds_check_not_vectors():
    D = 3
    lb = np.ones((1, D)) * -2
    ub = np.ones((1, D)) * 2
    plb = np.ones((1, D)) * -1
    pub = np.ones((1, D))
    x0 = np.ones((2, D))
    incorrect = np.ones((2, D))
    exception_message = "Bounds must match problem dimension D=3."
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
    options = {"integer_vars": np.array([True, False, False])}
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


def _integer_vars_vbmc(value, D=3):
    lb = np.full((1, D), -10.5)
    ub = np.full((1, D), 10.5)
    x0 = np.zeros((1, D))
    plb = np.full((1, D), -2.5)
    pub = np.full((1, D), 2.5)
    return VBMC(fun, x0, lb, ub, plb, pub, {"integer_vars": value})


@pytest.mark.parametrize(
    "value",
    [
        [0, 2],
        (0, 2),
        np.array([0, 2]),
        np.array([2, 0]),
        np.array([True, False, True]),
        [True, False, True],
    ],
)
def test_integer_vars_marks_the_variables_it_names(value):
    """A boolean array is a mask and an integer array holds the 0-based
    indices of the integer variables, as ``misc/setupvars_vbmc.m:15-17``
    reads ``options.IntegerVars`` as indices (1-based there)."""
    vbmc = _integer_vars_vbmc(value)
    assert np.array_equal(
        vbmc.optim_state["integer_vars"], np.array([True, False, True])
    )


@pytest.mark.parametrize("value", [[], (), np.array([]), None])
def test_integer_vars_empty_marks_no_variable(value):
    vbmc = _integer_vars_vbmc(value)
    assert not np.any(vbmc.optim_state["integer_vars"])


def test_integer_vars_all_zeros_and_ones_is_ambiguous():
    """``[1, 0, 1]`` at three variables reads both as a mask and as a list
    of indices, so it is refused rather than guessed."""
    with pytest.raises(ValueError) as execinfo:
        _integer_vars_vbmc(np.array([1, 0, 1]))
    message = execinfo.value.args[0]
    assert "integer_vars" in message
    assert "boolean" in message


@pytest.mark.parametrize(
    "value",
    [
        np.array([1, 3]),  # a 1-based index vector
        np.array([-1]),
        np.array([3]),
        np.array([2, 2]),
        np.array([True, False]),
        np.array([0.0, 2.0]),
        "0, 2",
        np.array([[0, 2]]),
    ],
)
def test_integer_vars_rejects_what_it_cannot_read(value):
    with pytest.raises(ValueError) as execinfo:
        _integer_vars_vbmc(value)
    assert "integer_vars" in execinfo.value.args[0]


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
    options = {"specify_target_noise": False, "uncertainty_handling": True}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    assert vbmc.optim_state["gp_noise_fun"] == [1, 2, 0]
    # uncertainty_handling_level 0
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
    assert np.all(vbmc.optim_state["variance_regularized_acq_fcn"] == True)
    assert "variance_regularized_acqfcn" not in vbmc.optim_state
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
    options = {"specify_target_noise": False, "uncertainty_handling": True}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    assert vbmc.optim_state["uncertainty_handling_level"] == 1
    options = {"specify_target_noise": False, "uncertainty_handling": []}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    assert vbmc.optim_state["uncertainty_handling_level"] == 0


@pytest.mark.parametrize("value", [True, 1, np.True_, np.int64(1)])
def test_uncertainty_handling_true_infers_the_noise_level(value):
    """``uncertainty_handling`` is a boolean, as ``options.UncertaintyHandling``
    is in MATLAB VBMC (``misc/setupvars_vbmc.m:232``), and a target whose
    noise level VBMC infers is handled at level 1."""
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, {"uncertainty_handling": value})
    assert vbmc.optim_state["uncertainty_handling_level"] == 1


@pytest.mark.parametrize(
    "value", [False, 0, np.False_, np.int64(0), [], (), np.array([]), None]
)
def test_uncertainty_handling_off_or_unset_gives_a_noiseless_run(value):
    """False turns the noise handling off and an empty value leaves the
    choice to ``specify_target_noise``, which is off here."""
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, {"uncertainty_handling": value})
    assert vbmc.optim_state["uncertainty_handling_level"] == 0


@pytest.mark.parametrize(
    "value", ["yes", "no", "off", [0], [1], [2], [3], np.array([1, 0]), 2]
)
def test_uncertainty_handling_rejects_other_values(value):
    """A value that is neither a boolean nor empty is refused, and the
    message names what may be written instead."""
    with pytest.raises(ValueError) as execinfo:
        create_vbmc(3, 3, 1, 5, 2, 4, {"uncertainty_handling": value})
    message = execinfo.value.args[0]
    assert "uncertainty_handling" in message
    assert "True or False" in message


@pytest.mark.parametrize("value", [True, 1, np.True_, np.int64(1)])
def test_specify_target_noise_true_takes_the_noise_the_target_returns(value):
    """``specify_target_noise`` is a boolean, as ``SpecifyTargetNoise`` is in
    MATLAB VBMC, and a target that returns its noise estimate is handled at
    level 2."""
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, {"specify_target_noise": value})
    assert vbmc.optim_state["uncertainty_handling_level"] == 2


@pytest.mark.parametrize("value", [False, 0, np.False_, np.int64(0)])
def test_specify_target_noise_false_gives_a_noiseless_run(value):
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, {"specify_target_noise": value})
    assert vbmc.optim_state["uncertainty_handling_level"] == 0


@pytest.mark.parametrize(
    "value", ["yes", "no", "off", [0], [1], 2, 1.0, None, [], np.array([])]
)
def test_specify_target_noise_rejects_other_values(value):
    """A value that is not a boolean is refused, and the message names what
    may be written instead: read by its truth, ``"no"`` or ``[0]`` would
    turn the noise handling on."""
    with pytest.raises(ValueError) as execinfo:
        create_vbmc(3, 3, 1, 5, 2, 4, {"specify_target_noise": value})
    message = execinfo.value.args[0]
    assert "specify_target_noise" in message
    assert "True or False" in message


def test_uncertainty_handling_off_with_specify_target_noise_raises():
    """``misc/setupoptions_vbmc.m:135-137`` refuses a target that supplies
    its own noise estimate while the noise handling is turned off."""
    options = {"specify_target_noise": True, "uncertainty_handling": False}
    with pytest.raises(ValueError) as execinfo:
        create_vbmc(3, 3, 1, 5, 2, 4, options)
    assert "specify_target_noise" in execinfo.value.args[0]


def test_uncertainty_handling_true_with_specify_target_noise_is_level_2():
    """Both set is the one combination MATLAB accepts, and the target's own
    noise estimate wins."""
    options = {"specify_target_noise": True, "uncertainty_handling": True}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    assert vbmc.optim_state["uncertainty_handling_level"] == 2


def test_vbmc_optimstate_acq_hedge():
    """The portfolio of acquisition functions that ``acq_hedge`` asks for
    is not ported, so the option is refused and no state is set up for it;
    the test asserts the refusal, where it used to assert the empty
    portfolio that construction left in ``optim_state``."""
    options = {"acq_hedge": True}
    with pytest.raises(NotImplementedError) as execinfo:
        create_vbmc(3, 3, 1, 5, 2, 4, options)
    assert "acq_hedge" in execinfo.value.args[0]

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
    assert vbmc.prior.sample is sample_prior
    assert vbmc.prior.log_pdf is None
    assert vbmc.log_likelihood is None

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
    assert np.isclose(vbmc.log_joint(x), log_joint(x))
    assert np.isclose(vbmc.function_logger.fun(x), log_joint(x))
    assert vbmc.prior.sample is sample_prior
    assert vbmc.prior.log_pdf is log_prior
    assert vbmc.log_likelihood is log_lklhd


def test_vbmc_init_log_joint_noisy():
    options = {"specify_target_noise": True}
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
    assert vbmc.prior is None
    assert vbmc.log_likelihood is None

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
    # A noisy log-joint returns the value and the noise estimate.
    value, noise = vbmc.log_joint(x)
    expected_value, expected_noise = log_joint(x)
    assert np.isclose(value, expected_value)
    assert np.isclose(noise, expected_noise)
    logged_value, logged_noise = vbmc.function_logger.fun(x)
    assert np.isclose(logged_value, expected_value)
    assert np.isclose(logged_noise, expected_noise)
    assert vbmc.prior.log_pdf is log_prior
    assert vbmc.log_likelihood is log_lklhd


def test_vbmc_init_log_joint_prior():
    D = 3
    # The generic priors of the bounded families live on [0, 1], and the
    # hard bounds have to lie inside the support of the prior.
    lb = np.zeros((1, D))
    ub = np.ones((1, D))
    x0_array = np.full((1, D), 0.5)
    plb = np.full((1, D), 0.1)
    pub = np.full((1, D), 0.9)

    def log_likelihood(x):
        return np.sum(x**2 + x + 1)

    for prior in priors:
        new_prior = prior._generic(D)

        # Init with prior only:
        vbmc = VBMC(
            log_likelihood, x0_array, lb, ub, plb, pub, prior=new_prior
        )
        assert vbmc.prior == new_prior
        x = new_prior.sample(1)
        assert vbmc.log_joint(x) == log_likelihood(x) + new_prior.log_pdf(x)
        # Init with prior and matching log_prior, sample_prior:
        vbmc = VBMC(
            log_likelihood,
            x0_array,
            lb,
            ub,
            plb,
            pub,
            prior=new_prior,
        )
        assert vbmc.prior == new_prior
        x = new_prior.sample(1)
        assert np.isclose(
            vbmc.log_joint(x), log_likelihood(x) + new_prior.log_pdf(x)
        )
    scipy_priors = [
        multivariate_normal(np.zeros(D)),
        multivariate_t(np.zeros(D), df=7),
        [norm(), lognorm(1.0), norm()],
    ]
    for prior in scipy_priors:
        # Init with prior only:
        vbmc = VBMC(
            log_likelihood,
            x0_array,
            lb,
            ub,
            plb,
            pub,
            prior=prior,
        )
        if isinstance(vbmc.prior, SciPy):
            assert vbmc.prior.distribution == prior
        else:
            assert isinstance(vbmc.prior, Product)
            for m, marginal in enumerate(vbmc.prior.marginals):
                assert marginal.distribution is prior[m]
        x = vbmc.prior.sample(1)
        assert np.isclose(
            vbmc.log_joint(x), log_likelihood(x) + vbmc.prior.log_pdf(x)
        )


def test_vbmc_init_log_joint_noisy_prior():
    options = {"specify_target_noise": True}
    D = 3
    # The generic priors of the bounded families live on [0, 1], and the
    # hard bounds have to lie inside the support of the prior.
    lb = np.zeros((1, D))
    ub = np.ones((1, D))
    x0_array = np.full((1, D), 0.5)
    plb = np.full((1, D), 0.1)
    pub = np.full((1, D), 0.9)

    def log_likelihood(x):
        return np.sum(x**2 + x + 1), 1.0

    for prior in priors:
        new_prior = prior._generic(D)

        # Init with prior only:
        vbmc = VBMC(
            log_likelihood,
            x0_array,
            lb,
            ub,
            plb,
            pub,
            prior=new_prior,
            options=options,
        )
        assert vbmc.prior == new_prior
        x = new_prior.sample(1)
        assert np.isclose(
            vbmc.log_joint(x)[0], log_likelihood(x)[0] + new_prior.log_pdf(x)
        )
        assert np.isclose(vbmc.log_joint(x)[1], log_likelihood(x)[1])
        # Init with prior and matching log_prior, sample_prior:
        vbmc = VBMC(
            log_likelihood,
            x0_array,
            lb,
            ub,
            plb,
            pub,
            sample_prior=new_prior.sample,
            prior=new_prior,
            options=options,
        )
        assert vbmc.prior == new_prior
        x = new_prior.sample(1)
        assert np.isclose(
            vbmc.log_joint(x)[0], log_likelihood(x)[0] + new_prior.log_pdf(x)
        )
        assert np.isclose(vbmc.log_joint(x)[1], log_likelihood(x)[1])
    scipy_priors = [
        multivariate_normal(np.zeros(D)),
        multivariate_t(np.zeros(D), df=7),
        [norm(), lognorm(1.0), norm()],
    ]
    for prior in scipy_priors:
        # Init with prior only:
        vbmc = VBMC(
            log_likelihood,
            x0_array,
            lb,
            ub,
            plb,
            pub,
            prior=prior,
            options=options,
        )
        if isinstance(vbmc.prior, SciPy):
            assert vbmc.prior.distribution == prior
        else:
            assert isinstance(vbmc.prior, Product)
            for m, marginal in enumerate(vbmc.prior.marginals):
                assert marginal.distribution is prior[m]
        x = vbmc.prior.sample(1)
        assert np.isclose(
            vbmc.log_joint(x)[0], log_likelihood(x)[0] + vbmc.prior.log_pdf(x)
        )
        assert np.isclose(vbmc.log_joint(x)[1], log_likelihood(x)[1])


def test_vbmc_init_error_handling():
    D = 3
    lb = np.full((1, D), -np.inf)
    ub = np.full((1, D), np.inf)
    x0_array = np.full((1, D), 0.5)
    plb = np.full((1, D), 0.1)
    pub = np.full((1, D), 0.9)

    def log_likelihood(x):
        return np.sum(x**2 + x + 1)

    def log_prior(x):
        return np.sum(x + 1)

    def sample_prior(n):
        return np.random.normal(size=(n, D))

    for prior in priors:
        new_prior = prior._generic(D)
        # Init with prior which is not a pyvbmc.priors.Prior:
        with pytest.raises(TypeError) as err:
            vbmc = VBMC(
                log_likelihood,
                x0_array,
                lb,
                ub,
                plb,
                pub,
                prior=1.0,
            )
        assert (
            "Optional keyword `prior` should be a subclass of `pyvbmc.priors.Prior`, an appropriate `scipy.stats` distribution, or a list of these."
            in err.value.args[0]
        )
        # Init with prior and mismatched log_prior / sample_prior
        with pytest.raises(ValueError) as err:
            vbmc = VBMC(
                log_likelihood,
                x0_array,
                lb,
                ub,
                plb,
                pub,
                sample_prior=sample_prior,
                prior=new_prior,
            )
        assert (
            "If `prior` is provided then `sample_prior` should be `None` or `prior.sample`."
            in err.value.args[0]
        )
        # Test prior with wrong dimension
        with pytest.raises(ValueError) as err:
            vbmc = VBMC(
                log_likelihood,
                x0_array,
                lb,
                ub,
                plb,
                pub,
                prior=SciPy(multivariate_normal(np.zeros(D + 1))),
            )
        assert (
            f"Dimension of `prior` ({D+1}) does not match dimension of model ({D})."
            in err.value.args[0]
        )
        with pytest.raises(ValueError) as err:
            vbmc = VBMC(
                log_likelihood,
                x0_array,
                lb,
                ub,
                plb,
                pub,
                prior=[norm() for __ in range(D + 1)],
            )
        assert (
            f"Dimension of `prior` ({D+1}) does not match dimension of model ({D})."
            in err.value.args[0]
        )


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


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_init_narrow_float_input(dtype):
    D = 2
    lb = np.full((1, D), -10, dtype=dtype)
    ub = np.full((1, D), 10, dtype=dtype)
    x0_array = np.full((1, D), 0, dtype=dtype)
    plb = np.full((1, D), -5, dtype=dtype)
    pub = np.full((1, D), 5, dtype=dtype)

    def log_joint(x):
        return x**2 + x + 1, 1.0

    inputs = [x0_array, lb, ub, plb, pub]
    originals = [array.copy() for array in inputs]
    vbmc = VBMC(log_joint, *inputs)
    for arr in [
        vbmc.x0,
        vbmc.lower_bounds,
        vbmc.upper_bounds,
        vbmc.plausible_lower_bounds,
        vbmc.plausible_upper_bounds,
        vbmc.optim_state["cache"]["x_orig"],
        vbmc.optim_state["lb_orig"],
        vbmc.optim_state["ub_orig"],
        vbmc.optim_state["plb_orig"],
        vbmc.optim_state["pub_orig"],
        vbmc.parameter_transformer.lb_orig,
        vbmc.parameter_transformer.ub_orig,
    ]:
        assert arr.dtype == np.float64
    for supplied, original in zip(inputs, originals):
        assert supplied.dtype == dtype
        assert np.array_equal(supplied, original)


def test_init_does_not_modify_repaired_input_arrays():
    D = 2
    x0 = np.full((1, D), -10.0)
    lb = np.full((1, D), -10.0)
    ub = np.full((1, D), 10.0)
    plb = np.full((1, D), -5.0)
    pub = np.full((1, D), 5.0)
    inputs = [x0, lb, ub, plb, pub]
    originals = [array.copy() for array in inputs]

    VBMC(fun, *inputs)

    for supplied, original in zip(inputs, originals):
        assert np.array_equal(supplied, original)


def test_init_widens_before_inferring_plausible_bounds():
    x0 = np.array([[-60000.0], [60000.0]], dtype=np.float16)
    lb = np.array([[-np.inf]], dtype=np.float16)
    ub = np.array([[np.inf]], dtype=np.float16)
    original_x0 = x0.copy()

    vbmc = VBMC(fun, x0, lb, ub)

    assert np.all(np.isfinite(vbmc.plausible_lower_bounds))
    assert np.all(np.isfinite(vbmc.plausible_upper_bounds))
    assert vbmc.plausible_lower_bounds.dtype == np.float64
    assert vbmc.plausible_upper_bounds.dtype == np.float64
    assert np.array_equal(x0, original_x0)


@pytest.mark.parametrize(
    "true_mean",
    [np.zeros(2), np.zeros((1, 2))],
)
def test_true_diagnostic_uses_independent_generator(mocker, true_mean):
    vbmc = create_vbmc(2, 0.0, -5.0, 5.0, -2.0, 2.0)
    vbmc.options.__setitem__("true_mean", true_mean, force=True)
    vbmc.options.__setitem__("true_cov", np.eye(2), force=True)
    moments = mocker.patch.object(
        VariationalPosterior,
        "moments",
        autospec=True,
        return_value=(np.zeros((1, 2)), np.eye(2)),
    )
    state_before = copy.deepcopy(vbmc.rng.bit_generator.state)

    assert vbmc._compute_true_diagnostic(vbmc.vp) == 0.0

    diagnostic_vp = moments.call_args.args[0]
    assert moments.call_args.args[1:] == (1e6, True, True)
    assert diagnostic_vp.rng is not vbmc.rng
    assert diagnostic_vp.rng.bit_generator.state == state_before
    diagnostic_vp.rng.random()
    assert vbmc.rng.bit_generator.state == state_before


@pytest.mark.parametrize(
    ("true_mean", "true_cov"),
    [([], []), (np.zeros(2), []), ([], np.eye(2))],
)
def test_true_diagnostic_skips_absent_values(true_mean, true_cov):
    vbmc = create_vbmc(2, 0.0, -5.0, 5.0, -2.0, 2.0)
    vbmc.options.__setitem__("true_mean", true_mean, force=True)
    vbmc.options.__setitem__("true_cov", true_cov, force=True)
    assert vbmc._compute_true_diagnostic(vbmc.vp) is None


@pytest.mark.parametrize(
    ("true_mean", "true_cov", "message"),
    [
        (np.zeros((2, 1)), np.eye(2), "true_mean"),
        (np.zeros(2), np.ones((1, 2)), "true_cov"),
    ],
)
def test_true_diagnostic_validates_shapes(true_mean, true_cov, message):
    vbmc = create_vbmc(2, 0.0, -5.0, 5.0, -2.0, 2.0)
    vbmc.options.__setitem__("true_mean", true_mean, force=True)
    vbmc.options.__setitem__("true_cov", true_cov, force=True)
    with pytest.raises(ValueError, match=message):
        vbmc._compute_true_diagnostic(vbmc.vp)


def test_true_diagnostic_skips_nonfinite_values(mocker):
    vbmc = create_vbmc(2, 0.0, -5.0, 5.0, -2.0, 2.0)
    vbmc.options.__setitem__("true_mean", np.array([np.nan, 0.0]), force=True)
    vbmc.options.__setitem__("true_cov", np.eye(2), force=True)
    moments = mocker.patch.object(VariationalPosterior, "moments")
    assert vbmc._compute_true_diagnostic(vbmc.vp) is None
    moments.assert_not_called()


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


def test_init_options_path(tmp_path):
    D = 2
    lb = np.full((D,), -10)
    ub = np.full((D,), 10)
    x0_array = np.full((D,), 0)
    plb = np.full((D,), -5)
    pub = np.full((D,), 5)

    def log_joint(x):
        return x**2 + x + 1, 1.0

    # default options:
    vbmc = VBMC(log_joint, x0_array, lb, ub, plb, pub)
    # Keys from test configs should not be here:
    assert "foo" not in vbmc.options
    assert "bar" not in vbmc.options
    assert "fooD" not in vbmc.options
    # Keys from basic config
    assert vbmc.options["specify_target_noise"] == False
    assert vbmc.options["log_file_name"] is None
    # Keys from advanced config
    assert vbmc.options["sgd_step_size"] == 0.005
    assert vbmc.options["uncertainty_handling"] == []

    abspath = tmp_path.joinpath("user_options.ini")
    abspath.write_text(
        "[UserOptions]\n"
        "# Required stable fcn evals for termination\n"
        "tol_stable_count = 42\n"
        "# Min number of iterations\n"
        "min_iter = 3\n"
    )
    options = {"min_iter": 666}
    for path in [
        abspath,  # absolute Path
        str(abspath),  # absolute Path (string)
    ]:
        vbmc = VBMC(
            log_joint,
            x0_array,
            lb,
            ub,
            plb,
            pub,
            options=options,
            options_path=path,
        )
        # Keys from basic config
        assert vbmc.options["specify_target_noise"] == False  # same as before
        assert vbmc.options["tol_stable_count"] == 42  # overridden by file
        assert vbmc.options["min_iter"] == 666  # `options` beats the file
        # Keys from advanced config
        assert vbmc.options["sgd_step_size"] == 0.005  # same as before


@pytest.mark.parametrize(
    "path",
    [
        Path("option_configs/advanced_vbmc_options.ini"),
        "option_configs/advanced_vbmc_options.ini",
    ],
)
def test_init_options_path_relative_to_the_package(path):
    """A relative ``options_path`` is resolved against the ``pyvbmc/vbmc/``
    directory."""
    D = 2
    vbmc = VBMC(
        fun,
        np.zeros((1, D)),
        np.full((1, D), -10.0),
        np.full((1, D), 10.0),
        np.full((1, D), -5.0),
        np.full((1, D), 5.0),
        options_path=path,
    )
    assert vbmc.options["sgd_step_size"] == 0.005


def _vbmc_with_options_file(tmp_path, lines, options=None):
    D = 2
    path = tmp_path.joinpath("user_options.ini")
    path.write_text("[UserOptions]\n" + "".join(lines))
    return VBMC(
        lambda x: -0.5 * np.sum(x**2),
        np.zeros((1, D)),
        np.full((1, D), -10.0),
        np.full((1, D), 10.0),
        np.full((1, D), -1.0),
        np.full((1, D), 1.0),
        options=options,
        options_path=path,
    )


@pytest.mark.parametrize(
    "line",
    ["specify_target_noise = True\n", "uncertainty_handling = True\n"],
)
def test_options_file_reaches_the_noisy_defaults(tmp_path, line):
    """The defaults that follow the noise handling are settled after every
    source of options has been read, so that a noisy run configured by file
    is configured as one configured by dictionary."""
    by_dictionary = create_vbmc(
        2, 0, -10, 10, -1, 1, {"specify_target_noise": True}
    )
    by_file = _vbmc_with_options_file(tmp_path, ["# noisy\n", line])
    assert by_file.options["max_fun_evals"] == (
        by_dictionary.options["max_fun_evals"]
    )
    assert by_file.options["active_sample_gp_update"] is True
    assert by_file.options["active_sample_vp_update"] is True
    assert isinstance(by_file.options["search_acq_fcn"][0], AcqFcnVIQR)


def test_an_option_set_in_a_file_is_not_overwritten_by_a_default(tmp_path):
    """A value written in the user's file is the user's choice, as one
    passed in the dictionary is, so the noisy defaults leave it alone."""
    vbmc = _vbmc_with_options_file(
        tmp_path,
        [
            "# noisy\n",
            "uncertainty_handling = True\n",
            "# budget\n",
            "max_fun_evals = 33\n",
        ],
    )
    assert vbmc.options["max_fun_evals"] == 33
    assert vbmc.options["active_sample_vp_update"] is True


def _same_random_state(first, second):
    """Whether two legacy global random states are the same state."""
    return (
        first[0] == second[0]
        and np.array_equal(first[1], second[1])
        and first[2:] == second[2:]
    )


def _global_state_around_construction(seed):
    """The global random state before and after one construction, and the
    state that four unsigned draws from the earlier one give."""
    saved = np.random.get_state()
    try:
        np.random.seed(11)
        before = np.random.get_state()
        VBMC(
            fun,
            np.zeros((1, 2)),
            np.full((1, 2), -10.0),
            np.full((1, 2), 10.0),
            np.full((1, 2), -1.0),
            np.full((1, 2), 1.0),
            seed=seed,
        )
        after = np.random.get_state()
        np.random.set_state(before)
        np.random.randint(0, 2**32, size=4, dtype=np.uint32)
        four_draws = np.random.get_state()
    finally:
        np.random.set_state(saved)
    return before, after, four_draws


def test_a_given_seed_leaves_the_global_random_state_where_it_was():
    """The ``seed`` documentation separates the two constructions: a seed
    or a generator is used as it is."""
    before, after, __ = _global_state_around_construction(42)
    assert _same_random_state(before, after)


def test_an_unseeded_construction_advances_the_global_random_state():
    """An unseeded construction derives its generator from the global
    random state, as the ``seed`` documentation says, by drawing the four
    integers of ``pyvbmc.rng.get_rng``."""
    before, after, four_draws = _global_state_around_construction(None)
    assert not _same_random_state(before, after)
    assert _same_random_state(four_draws, after)


def test__str__and__repr__():
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4)
    vbmc.__str__()
    vbmc.__repr__()


class TrackingPrior(Prior):
    def __init__(self, D=2, complex_output=False):
        self.D = D
        self.complex_output = complex_output
        self.calls = []

    def _log_pdf(self, x):
        self.calls.append(x.copy())
        values = -np.sum(x**2, axis=1, keepdims=True)
        if self.complex_output:
            values = values.astype(complex) + 1j
        return values

    def sample(self, n, rng=None):
        return np.zeros((n, self.D))

    @classmethod
    def _generic(cls, D=1):
        return cls(D)


def _vectorized_vbmc(target, *, prior=None, options=None, D=2, bounds=None):
    merged_options = {"vectorized_target": True}
    if options:
        merged_options.update(options)
    if bounds is None:
        bounds = (np.full((1, D), -np.inf), np.full((1, D), np.inf))
    lower_bounds, upper_bounds = bounds
    return VBMC(
        target,
        np.zeros((1, D)),
        lower_bounds,
        upper_bounds,
        np.full((1, D), -1.0),
        np.full((1, D), 1.0),
        options=merged_options,
        prior=prior,
    )


def test_noise_shaping_is_off_by_default():
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4)
    assert vbmc.options["noise_shaping"] is False


def test_noise_shaping_on_is_rejected():
    """Only half of the option is ported, so turning it on would configure
    a run that exists in neither toolbox."""
    with pytest.raises(NotImplementedError) as execinfo:
        create_vbmc(3, 3, 1, 5, 2, 4, options={"noise_shaping": True})
    message = execinfo.value.args[0]
    assert "noise_shaping" in message
    assert "noiseshaping_vbmc.m" in message


class _McmcImportanceSamplingAcq(AcqFcnVIQR):
    """An acquisition that asks for the MCMC step of the importance
    sampler, as a user-supplied one could."""

    def __init__(self):
        super().__init__()
        self.acq_info["mcmc_importance_sampling"] = True


def test_an_acquisition_asking_for_mcmc_importance_sampling_is_rejected():
    """The step that the flag asks for is not ported, so an acquisition
    that sets it is refused where it is supplied."""
    with pytest.raises(NotImplementedError) as execinfo:
        create_vbmc(
            3,
            3,
            1,
            5,
            2,
            4,
            options={"search_acq_fcn": [_McmcImportanceSamplingAcq()]},
        )
    message = execinfo.value.args[0]
    assert "mcmc_importance_sampling" in message
    assert "not ported" in message


@pytest.mark.parametrize(
    "value",
    [AcqFcnLog(), "AcqFcnLog()", None, [], [AcqFcnLog(), 3]],
)
def test_search_acq_fcn_must_be_a_list_of_acquisitions(value):
    """A single acquisition, its name outside a list, and an entry that is
    neither an acquisition nor a string are refused at construction with a
    message that names the option."""
    with pytest.raises(ValueError, match="search_acq_fcn"):
        create_vbmc(3, 3, 1, 5, 2, 4, options={"search_acq_fcn": value})


def test_the_acquisitions_that_do_not_ask_for_it_are_accepted():
    """The shipped acquisitions leave the flag unset, and an entry of
    ``search_acq_fcn`` may also be the name of one."""
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4)
    for acq_fcn in vbmc.options["search_acq_fcn"]:
        assert not acq_fcn.acq_info.get("mcmc_importance_sampling")
    vbmc = create_vbmc(
        3, 3, 1, 5, 2, 4, options={"search_acq_fcn": ["AcqFcnVIQR()"]}
    )
    assert vbmc.options["search_acq_fcn"] == ["AcqFcnVIQR()"]


def test_vectorized_target_option_and_logger_mode():
    vbmc = _vectorized_vbmc(lambda x: np.sum(x, axis=1))
    assert vbmc.options["vectorized_target"] is True
    assert vbmc.function_logger.vectorized_target is True

    with pytest.raises(ValueError, match="must be boolean"):
        _vectorized_vbmc(
            lambda x: np.sum(x, axis=1),
            options={"vectorized_target": 1},
        )


def test_vectorized_likelihood_with_builtin_prior_and_column_output():
    prior = UniformBox(np.full(2, -2.0), np.full(2, 2.0))
    vbmc = _vectorized_vbmc(
        lambda x: np.sum(x, axis=1, keepdims=True),
        prior=prior,
        bounds=(np.full((1, 2), -2.0), np.full((1, 2), 2.0)),
    )
    points = np.array([[0.0, 0.0], [0.5, -0.5]])

    values = vbmc.log_joint(points)

    expected = np.sum(points, axis=1) + prior.log_pdf(points, keepdims=False)
    assert values.shape == (2,)
    assert np.allclose(values, expected)


def test_vectorized_likelihood_evaluates_custom_prior_row_by_row():
    prior = TrackingPrior()
    vbmc = _vectorized_vbmc(lambda x: np.sum(x, axis=1), prior=prior)
    points = np.array([[0.25, -0.5], [0.75, 0.5]])

    values = vbmc.log_joint(points)

    assert values.shape == (2,)
    assert len(prior.calls) == 2
    assert all(call.shape == (1, 2) for call in prior.calls)
    assert np.allclose(
        values,
        np.sum(points, axis=1) - np.sum(points**2, axis=1),
    )


def test_vectorized_cached_log_joint_does_not_apply_prior_twice():
    likelihood_calls = []
    prior = TrackingPrior()

    def likelihood(x):
        likelihood_calls.append(x.copy())
        return np.sum(x, axis=1)

    vbmc = _vectorized_vbmc(likelihood, prior=prior)
    points = np.array([[0.25, -0.5], [0.75, 0.5]])
    cached_log_joint = 123.0

    vbmc.function_logger.batch_call(
        vbmc.parameter_transformer(points),
        [cached_log_joint, np.nan],
    )

    assert vbmc.function_logger.y_orig[0, 0] == cached_log_joint
    assert len(likelihood_calls) == 1
    assert likelihood_calls[0].shape == (1, 2)
    assert np.allclose(likelihood_calls[0][0], points[1])
    assert len(prior.calls) == 1


def test_vectorized_unknown_noise_prior_returns_values_only():
    prior = TrackingPrior()
    vbmc = _vectorized_vbmc(
        lambda x: np.sum(x, axis=1),
        prior=prior,
        options={"uncertainty_handling": True},
    )
    points = np.array([[0.25, -0.5], [0.75, 0.5]])

    log_joint_output = vbmc.log_joint(points)
    _, sds, _ = vbmc.function_logger.batch_call(
        vbmc.parameter_transformer(points)
    )

    assert isinstance(log_joint_output, np.ndarray)
    assert log_joint_output.shape == (2,)
    assert np.array_equal(sds, np.ones(2))


def test_vectorized_noisy_prior_rejects_n_by_two_likelihood():
    vbmc = _vectorized_vbmc(
        lambda x: np.column_stack((np.sum(x, axis=1), np.ones(x.shape[0]))),
        prior=TrackingPrior(),
        options={"specify_target_noise": True},
    )
    with pytest.raises(ValueError, match=r"not an \(N, 2\) ndarray"):
        vbmc.function_logger.batch_call(np.zeros((2, 2)))
    assert vbmc.function_logger.Xn == -1


def test_vectorized_prior_rejects_complex_scalar_with_row_context():
    vbmc = _vectorized_vbmc(
        lambda x: np.sum(x, axis=1),
        prior=TrackingPrior(complex_output=True),
    )
    with pytest.raises(ValueError, match="finite real scalar for row 0"):
        vbmc.log_joint(np.zeros((2, 2)))


def test_log_joint_with_a_user_function_marginal():
    """A list of one-dimensional priors may hold a `UserFunction`, whose
    density is the user's own callable and takes one point."""
    lb = np.array([[-np.inf, 0.0]])
    ub = np.array([[np.inf, 1.0]])
    plb = np.array([[-1.0, 0.2]])
    pub = np.array([[1.0, 0.8]])
    x0_array = np.array([[0.0, 0.5]])

    def log_likelihood(x):
        return np.sum(x**2 + x + 1)

    def log_marginal(x):
        return -0.5 * float(x[0]) ** 2

    box = UniformBox(0.0, 1.0, D=1)
    vbmc = VBMC(
        log_likelihood,
        x0_array,
        lb,
        ub,
        plb,
        pub,
        prior=[UserFunction(log_marginal, D=1), box],
    )

    x = np.array([[0.3, 0.4]])
    expected = (
        log_likelihood(x)
        + log_marginal(x[0, :1])
        + box.log_pdf(x[0, 1:]).item()
    )
    assert np.isclose(vbmc.log_joint(x).item(), expected)


def _log_likelihood_for_prior_bounds(x):
    return np.sum(x**2 + x + 1)


def test_a_prior_narrower_than_the_hard_bounds_is_refused():
    """The log-joint is ``-inf`` where the box reaches outside the support
    of the prior, and a run would stop at the first evaluation it makes
    there."""
    D = 2
    with pytest.raises(ValueError) as err:
        VBMC(
            _log_likelihood_for_prior_bounds,
            np.full((1, D), 5.0),
            np.zeros((1, D)),
            np.full((1, D), 10.0),
            np.full((1, D), 2.0),
            np.full((1, D), 8.0),
            prior=UniformBox(0.0, 1.0, D=D),
        )
    message = err.value.args[0]
    assert "inside the support of `prior`" in message
    assert "coordinate 0 has bounds [0.0, 10.0]" in message
    assert "coordinate 1 has bounds [0.0, 10.0]" in message
    assert "support [0.0, 1.0]" in message


def test_a_prior_whose_support_covers_the_hard_bounds_is_taken():
    """Equality is containment, a wider support is enough, and a prior
    whose support is not finite covers any box."""
    D = 2
    x0_array = np.full((1, D), 5.0)
    lb, ub = np.zeros((1, D)), np.full((1, D), 10.0)
    plb, pub = np.full((1, D), 2.0), np.full((1, D), 8.0)
    unbounded_lb = np.full((1, D), -np.inf)
    unbounded_ub = np.full((1, D), np.inf)

    accepted = [
        # The support is exactly the box.
        (lb, ub, plb, pub, x0_array, UniformBox(0.0, 10.0, D=D)),
        # A support wider than the box.
        (lb, ub, plb, pub, x0_array, UniformBox(-1.0, 11.0, D=D)),
        # A spline trapezoid built from the bounds, as example 5 builds it.
        (
            lb,
            ub,
            plb,
            pub,
            x0_array,
            SplineTrapezoidal(lb, plb, pub, ub),
        ),
        # A product whose shifted marginal covers its coordinate.
        (
            lb,
            ub,
            plb,
            pub,
            x0_array,
            [sp.stats.uniform(loc=-1, scale=12), sp.stats.norm()],
        ),
        # Unbounded parameters with a prior of unbounded support.
        (
            unbounded_lb,
            unbounded_ub,
            np.full((1, D), -1.0),
            np.full((1, D), 1.0),
            np.zeros((1, D)),
            SmoothBox(-1.0, 1.0, 1.0, D=D),
        ),
        (
            unbounded_lb,
            unbounded_ub,
            np.full((1, D), -1.0),
            np.full((1, D), 1.0),
            np.zeros((1, D)),
            multivariate_normal(np.zeros(D)),
        ),
    ]
    for lower, upper, p_lower, p_upper, x0, prior in accepted:
        vbmc = VBMC(
            _log_likelihood_for_prior_bounds,
            x0,
            lower,
            upper,
            p_lower,
            p_upper,
            prior=prior,
        )
        assert vbmc.prior is not None


def _uniform_marginals(lower, upper):
    """The list of marginals the FAQ builds from a pair of hard bounds."""
    return [
        sp.stats.uniform(loc=low, scale=high - low)
        for low, high in zip(np.ravel(lower), np.ravel(upper))
    ]


_BOUND_GRID = np.round(np.arange(-20.0, 20.0 + 1e-9, 1.3), 1)


def test_a_prior_built_from_the_hard_bounds_is_taken():
    """A marginal built as ``uniform(loc=low, scale=high - low)`` has its
    support edge a few units in the last place away from ``high``, and the
    pair of hard bounds it was built from is inside it."""
    for i, low in enumerate(_BOUND_GRID):
        for high in _BOUND_GRID[i + 1 :]:
            prior = convert_to_prior(_uniform_marginals([low], [high]))
            _check_prior_covers_bounds(
                prior, np.array([[low]]), np.array([[high]])
            )


def test_a_prior_built_from_random_hard_bounds_is_taken():
    """The same construction on bounds computed from data, which carry no
    round decimal."""
    rng = np.random.default_rng(20260921)
    for __ in range(300):
        low, high = np.sort(rng.uniform(-20.0, 20.0, size=2))
        prior = convert_to_prior(_uniform_marginals([low], [high]))
        _check_prior_covers_bounds(
            prior, np.array([[low]]), np.array([[high]])
        )


@pytest.mark.parametrize(
    "lower, upper",
    [(-20.0, 0.2), (-1.1, 3.4), (0.9401229776087456, 9.034701816518085)],
)
def test_a_vbmc_built_on_the_faq_prior_is_taken(lower, upper):
    """The whole construction the FAQ recommends: the marginals and the
    hard bounds come from the same pair of numbers."""
    D = 2
    lb, ub = np.full((1, D), lower), np.full((1, D), upper)
    plb = lb + 0.25 * (ub - lb)
    pub = ub - 0.25 * (ub - lb)
    vbmc = VBMC(
        _log_likelihood_for_prior_bounds,
        0.5 * (lb + ub),
        lb,
        ub,
        plb,
        pub,
        prior=_uniform_marginals(lb, ub),
    )
    assert vbmc.prior is not None


@pytest.mark.parametrize("side", ["lower", "upper"])
def test_a_support_short_of_a_hard_bound_is_refused(side):
    """The slack covers rounding alone: a support short of the box by a
    millionth of its range is a prior that is narrower than the bounds."""
    D = 2
    lb, ub = np.zeros((1, D)), np.full((1, D), 10.0)
    gap = 1e-6 * (ub[0, 0] - lb[0, 0])
    if side == "lower":
        support = UniformBox(lb[0, 0] + gap, ub[0, 0], D=D)
    else:
        support = UniformBox(lb[0, 0], ub[0, 0] - gap, D=D)
    with pytest.raises(ValueError) as err:
        _check_prior_covers_bounds(support, lb, ub)
    assert "inside the support of `prior`" in err.value.args[0]


@pytest.mark.parametrize("side", ["lower", "upper"])
def test_an_infinite_hard_bound_against_a_finite_support_is_refused(side):
    """An infinite bound gets no slack, whatever the other bound is."""
    D = 2
    lb, ub = np.zeros((1, D)), np.full((1, D), 10.0)
    if side == "lower":
        lb = np.full((1, D), -np.inf)
    else:
        ub = np.full((1, D), np.inf)
    with pytest.raises(ValueError) as err:
        _check_prior_covers_bounds(UniformBox(0.0, 10.0, D=D), lb, ub)
    assert "inside the support of `prior`" in err.value.args[0]


def test_a_support_given_as_single_values_is_read_as_a_box():
    """A ``Prior`` subclass may report its support as one value for every
    coordinate, which the check reads as the box of the model."""

    class _ScalarSupportPrior(Prior):
        def __init__(self, D):
            self.D = D

        def _support(self):
            return 0.0, 1.0

        def _log_pdf(self, x):
            return np.zeros((x.shape[0], 1))

        def sample(self, n, rng=None):
            return np.zeros((n, self.D))

        @classmethod
        def _generic(cls, D=1):
            return cls(D)

    D = 3
    prior = _ScalarSupportPrior(D)
    _check_prior_covers_bounds(prior, np.zeros((1, D)), np.ones((1, D)))
    with pytest.raises(ValueError) as err:
        _check_prior_covers_bounds(
            prior, np.zeros((1, D)), np.full((1, D), 2.0)
        )
    message = err.value.args[0]
    assert "inside the support of `prior`" in message
    assert "coordinate 2 has bounds [0.0, 2.0]" in message


def test_a_log_prior_callable_is_taken_whatever_the_hard_bounds():
    """A callable states no support, so it covers any box."""
    D = 2
    vbmc = VBMC(
        _log_likelihood_for_prior_bounds,
        np.full((1, D), 5.0),
        np.zeros((1, D)),
        np.full((1, D), 10.0),
        np.full((1, D), 2.0),
        np.full((1, D), 8.0),
        log_prior=lambda x: -0.5 * np.sum(np.asarray(x) ** 2),
    )
    assert vbmc.prior is not None
