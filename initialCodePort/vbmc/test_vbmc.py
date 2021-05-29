import numpy as np
import pytest

from vbmc import VBMC

fun = lambda x: np.sum(x + 2)


def test_vbmc_init_no_x0_PLB_PUB():
    with pytest.raises(ValueError):
        VBMC(fun)


def test_vbmc_boundscheck_no_PUB_PLB_n0_1():
    D = 3
    lb = np.zeros((1, D))
    ub = np.ones((1, D)) * 2
    x0 = np.ones((1, D))
    _, lb2, ub2, plb, pub = VBMC(fun, x0, lb, ub)._boundscheck(fun, x0, lb, ub)
    assert np.all(lb == lb2)
    assert np.all(ub == ub2)
    assert np.all(plb == lb)
    assert np.all(pub == ub)


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
    assert np.all(plb == lb)
    assert np.all(pub == ub)


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
