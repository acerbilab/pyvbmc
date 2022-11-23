import numpy as np
import pytest
import scipy.stats as sps

from pyvbmc import parameter_transformer
from pyvbmc.parameter_transformer import ParameterTransformer

bounded_transform_types = [3, 12, 13]  # Update as needed with new types.
D = 3


def test_init_no_lower_bounds():
    parameter_transformer = ParameterTransformer(D=D)
    assert np.all(np.isinf(parameter_transformer.lb_orig))


def test_init_lower_bounds():
    parameter_transformer = ParameterTransformer(D=D, lb_orig=np.ones((1, D)))
    assert np.all(parameter_transformer.lb_orig == np.ones(D))


def test_init_no_upper_bounds():
    parameter_transformer = ParameterTransformer(D=D)
    assert np.all(np.isinf(parameter_transformer.ub_orig))


def test_init_upper_bounds():
    parameter_transformer = ParameterTransformer(D=D, ub_orig=np.ones((1, D)))
    assert np.all(parameter_transformer.ub_orig == np.ones(D))


def test_init_type_3():
    # logit (default)
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)),
        ub_orig=np.ones((1, D)) * 2,
    )
    assert np.all(parameter_transformer.type == np.ones(D) * 3)

    # logit (keyword)
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)),
        ub_orig=np.ones((1, D)) * 2,
        transform_type="logit",
    )
    assert np.all(parameter_transformer.type == np.ones(D) * 3)

    # logit (number)
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)),
        ub_orig=np.ones((1, D)) * 2,
        transform_type=3,
    )
    assert np.all(parameter_transformer.type == np.ones(D) * 3)

    # probit
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)),
        ub_orig=np.ones((1, D)) * 2,
        transform_type="probit",
    )
    assert np.all(parameter_transformer.type == np.ones(D) * 12)

    # probit (alternate name)
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)),
        ub_orig=np.ones((1, D)) * 2,
        transform_type="norminv",
    )

    # probit (number)
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)),
        ub_orig=np.ones((1, D)) * 2,
        transform_type=12,
    )
    assert np.all(parameter_transformer.type == np.ones(D) * 12)

    # Student's T
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)),
        ub_orig=np.ones((1, D)) * 2,
        transform_type="student4",
    )
    assert np.all(parameter_transformer.type == np.ones(D) * 13)

    # Student's T (number)
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)),
        ub_orig=np.ones((1, D)) * 2,
        transform_type=13,
    )
    assert np.all(parameter_transformer.type == np.ones(D) * 13)

    # Bad transform (keyword)
    with pytest.raises(Exception) as e_info:
        parameter_transformer = ParameterTransformer(
            D=D,
            lb_orig=np.ones((1, D)),
            ub_orig=np.ones((1, D)) * 2,
            transform_type="this_is_not_a_transform_type",
        )
    assert "Unrecognized bounded transform" in e_info.value.args[0]

    # Bad transform (number)
    with pytest.raises(Exception) as e_info:
        parameter_transformer = ParameterTransformer(
            D=D,
            lb_orig=np.ones((1, D)),
            ub_orig=np.ones((1, D)) * 2,
            transform_type=666,
        )
    assert "Unrecognized bounded transform" in e_info.value.args[0]


def test_init_mixed_bounds():
    parameter_transformer = ParameterTransformer(
        D=4,
        lb_orig=np.array([[0.0, -np.inf, -10.0, -np.inf]]),
        ub_orig=np.array([[10.0, np.inf, 0.0, np.inf]]),
    )
    assert np.all(
        parameter_transformer.type == np.array([3, 0, 3, 0], dtype=int)
    )

    parameter_transformer = ParameterTransformer(
        D=4,
        lb_orig=np.array([[0.0, -np.inf, -10.0, -np.inf]]),
        ub_orig=np.array([[10.0, np.inf, 0.0, np.inf]]),
        transform_type="probit",
    )
    assert np.all(
        parameter_transformer.type == np.array([12, 0, 12, 0], dtype=int)
    )

    parameter_transformer = ParameterTransformer(
        D=4,
        lb_orig=np.array([[0.0, -np.inf, -10.0, -np.inf]]),
        ub_orig=np.array([[10.0, np.inf, 0.0, np.inf]]),
        transform_type="student4",
    )
    assert np.all(
        parameter_transformer.type == np.array([13, 0, 13, 0], dtype=int)
    )

    for t in bounded_transform_types:
        parameter_transformer = ParameterTransformer(
            D=4,
            lb_orig=np.array([[0.0, -np.inf, -10.0, -np.inf]]),
            ub_orig=np.array([[10.0, np.inf, 0.0, np.inf]]),
            transform_type=t,
        )
        assert np.all(
            parameter_transformer.type == np.array([t, 0, t, 0], dtype=int)
        )


def test_init_bounds_check():
    with pytest.raises(ValueError):
        ParameterTransformer(
            D=D,
            lb_orig=np.ones((1, D)) * 3,
            ub_orig=np.ones((1, D)) * 2,
        )
    with pytest.raises(ValueError):
        ParameterTransformer(
            D=D,
            lb_orig=np.ones((1, D)) * 0,
            ub_orig=np.ones((1, D)) * 10,
            plb_orig=np.ones((1, D)) * -1,
        )
    with pytest.raises(ValueError):
        ParameterTransformer(
            D=D,
            lb_orig=np.ones((1, D)) * 0,
            ub_orig=np.ones((1, D)) * 10,
            pub_orig=np.ones((1, D)) * 11,
        )
    with pytest.raises(ValueError):
        ParameterTransformer(
            D=D,
            lb_orig=np.ones((1, D)) * 0,
            ub_orig=np.ones((1, D)) * 10,
            plb_orig=np.ones((1, D)) * 100,
            pub_orig=np.ones((1, D)) * -20,
        )


def test_init_mu_inf_bounds():
    parameter_transformer = ParameterTransformer(D=D)
    assert np.all(parameter_transformer.mu == np.zeros(D))


def test_init_delta_inf_bounds():
    parameter_transformer = ParameterTransformer(D=D)
    assert np.all(parameter_transformer.delta == np.ones(D))


def test_init_type3_mu_all_params():
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
        plb_orig=np.ones((1, D)) * 2,
        pub_orig=np.ones((1, D)) * 4,
    )
    parameter_transformer2 = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
    )
    plb = parameter_transformer2(np.ones((1, D)) * 2)
    pub = parameter_transformer2(np.ones((1, D)) * 4)
    mu2 = (plb + pub) * 0.5
    assert np.all(
        np.isclose(parameter_transformer.mu, mu2, rtol=1e-12, atol=1e-14)
    )


def test_init_type3_delta_all_params():
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
        plb_orig=np.ones((1, D)) * 2,
        pub_orig=np.ones((1, D)) * 4,
    )
    parameter_transformer2 = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
    )
    plb = parameter_transformer2(np.ones((1, D)) * 2)
    pub = parameter_transformer2(np.ones((1, D)) * 4)
    delta2 = pub - plb
    assert np.all(
        np.isclose(parameter_transformer.delta, delta2, rtol=1e-12, atol=1e-14)
    )


def test_direct_transform_type3_within():
    # logit
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
    )
    X = np.ones((10, D)) * 3
    Y = parameter_transformer(X)
    Y2 = np.ones((10, D)) * 0.619
    assert np.all(np.isclose(Y, Y2, atol=1e-04))

    # probit
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
        transform_type="probit",
    )
    X = np.ones((10, D)) * 3
    Y = parameter_transformer(X)
    Y2 = np.ones((10, D)) * sps.norm.ppf(13 / 20)
    assert np.all(np.isclose(Y, Y2))

    # student4
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
        transform_type="student4",
    )
    X = np.ones((10, D)) * 3
    Y = parameter_transformer(X)
    Y2 = np.ones((10, D)) * sps.t.ppf(13 / 20, df=4)
    assert np.all(np.isclose(Y, Y2))


def test_direct_transform_type3_within_negative():
    # logit
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
    )
    X = np.ones((10, D)) * -4
    Y = parameter_transformer(X)
    Y2 = np.ones((10, D)) * -0.8473
    assert np.all(np.isclose(Y, Y2))

    # probit
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
        transform_type="probit",
    )
    X = np.ones((10, D)) * -4
    Y = parameter_transformer(X)
    Y2 = np.ones((10, D)) * sps.norm.ppf(6 / 20)
    assert np.all(np.isclose(Y, Y2))

    # student4
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
        transform_type="student4",
    )
    X = np.ones((10, D)) * -4
    Y = parameter_transformer(X)
    Y2 = np.ones((10, D)) * sps.t.ppf(6 / 20, df=4)
    assert np.all(np.isclose(Y, Y2))


def test_direct_transform_type0():
    parameter_transformer = ParameterTransformer(D=D)
    X = np.ones((10, D)) * 3
    Y = parameter_transformer(X)
    assert np.all(Y == X)


def test_direct_transform_type0_negative():
    parameter_transformer = ParameterTransformer(D=D)
    X = np.ones((10, D)) * -4
    Y = parameter_transformer(X)
    assert np.all(Y == X)


def test_inverse_type3_within():
    # logit
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
    )
    Y = np.ones((10, D)) * 3
    X = parameter_transformer.inverse(Y)
    X2 = np.ones((10, D)) * 9.0515
    assert np.all(np.isclose(X, X2))

    # probit
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
        transform_type="probit",
    )
    Y = np.ones((10, D)) * 3
    X = parameter_transformer.inverse(Y)
    X2 = np.ones((10, D)) * sps.norm.cdf(3) * 20 - 10
    assert np.all(np.isclose(X, X2))

    # student4
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
        transform_type="student4",
    )
    Y = np.ones((10, D)) * 3
    X = parameter_transformer.inverse(Y)
    X2 = np.ones((10, D)) * sps.t.cdf(3, df=4) * 20 - 10
    assert np.all(np.isclose(X, X2))


def test_inverse_type3_within_negative():
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
    )
    Y = np.ones((10, D)) * -4
    X = parameter_transformer.inverse(Y)
    X2 = np.ones((10, D)) * -9.6403
    assert np.all(np.isclose(X, X2))

    # probit
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
        transform_type="probit",
    )
    Y = np.ones((10, D)) * -4
    X = parameter_transformer.inverse(Y)
    X2 = np.ones((10, D)) * sps.norm.cdf(-4) * 20 - 10
    assert np.all(np.isclose(X, X2))

    # student4
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
        transform_type="student4",
    )
    Y = np.ones((10, D)) * -4
    X = parameter_transformer.inverse(Y)
    X2 = np.ones((10, D)) * sps.t.cdf(-4, df=4) * 20 - 10
    assert np.all(np.isclose(X, X2))


def test_inverse_type0():
    parameter_transformer = ParameterTransformer(D=D)
    Y = np.ones((10, D)) * 3
    X = parameter_transformer.inverse(Y)
    assert np.all(Y == X)


def test_inverse_type0_negative():
    parameter_transformer = ParameterTransformer(D=D)
    Y = np.ones((10, D)) * -4
    X = parameter_transformer.inverse(Y)
    assert np.all(Y == X)


def test_inverse_type3_min_space():
    # logit
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
    )
    Y = np.ones((10, D)) * -500
    X = parameter_transformer.inverse(Y)
    assert np.allclose(X, np.ones((1, D)) * -10)

    for t in [12, 13]:  # probit, student4
        parameter_transformer = ParameterTransformer(
            D=D,
            lb_orig=np.ones((1, D)) * -10,
            ub_orig=np.ones((1, D)) * 10,
            transform_type=t,
        )
        Y = np.ones((10, D)) * -500
        X = parameter_transformer.inverse(Y)
        assert np.allclose(X, np.ones((1, D)) * -10)


def test_inverse_type3_max_space():
    # logit
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
    )
    Y = np.ones((10, D)) * 3000
    X = parameter_transformer.inverse(Y)
    assert np.allclose(X, np.ones((10, D)) * 10)

    for t in [12, 13]:  # probit, student4
        parameter_transformer = ParameterTransformer(
            D=D,
            lb_orig=np.ones((1, D)) * -10,
            ub_orig=np.ones((1, D)) * 10,
            transform_type=t,
        )
        Y = np.ones((10, D)) * -500
        X = parameter_transformer.inverse(Y)
        assert np.allclose(X, np.ones((1, D)) * -10)


def test_transform_direct_inverse():
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
    )
    X = np.ones((10, D)) * 0.05
    U = parameter_transformer(X)
    X2 = parameter_transformer.inverse(U)
    assert np.all(np.isclose(X, X2, rtol=1e-12, atol=1e-14))

    for t in [12, 13]:  # probit, student4
        parameter_transformer = ParameterTransformer(
            D=D,
            lb_orig=np.ones((1, D)) * -10,
            ub_orig=np.ones((1, D)) * 10,
            transform_type=t,
        )
        X = np.ones((10, D)) * 0.05
        U = parameter_transformer(X)
        X2 = parameter_transformer.inverse(U)
        assert np.all(np.isclose(X, X2, rtol=1e-11, atol=1e-14))


def test_transform_inverse_direct():
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
    )
    U = np.ones((10, D)) * 0.2
    X = parameter_transformer.inverse(U)
    U2 = parameter_transformer(X)
    assert np.all(np.isclose(U, U2, rtol=1e-12, atol=1e-14))

    for t in [12, 13]:  # probit, student4
        parameter_transformer = ParameterTransformer(
            D=D,
            lb_orig=np.ones((1, D)) * -10,
            ub_orig=np.ones((1, D)) * 10,
            transform_type=t,
        )
        U = np.ones((10, D)) * 0.2
        X = parameter_transformer.inverse(U)
        U2 = parameter_transformer(X)
        assert np.all(np.isclose(U, U2, rtol=1e-12, atol=1e-14))


def test_transform_direct_inverse_largeN():
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
    )
    X = np.ones((10 ^ 6, D)) * 0.4
    U = parameter_transformer(X)
    X2 = parameter_transformer.inverse(U)
    assert np.all(np.isclose(X, X2, rtol=1e-12, atol=1e-14))

    for t in [12, 13]:  # probit, student4
        parameter_transformer = ParameterTransformer(
            D=D,
            lb_orig=np.ones((1, D)) * -10,
            ub_orig=np.ones((1, D)) * 10,
            transform_type=t,
        )
        X = np.ones((10 ^ 6, D)) * 0.4
        U = parameter_transformer(X)
        X2 = parameter_transformer.inverse(U)
        assert np.all(np.isclose(X, X2, rtol=1e-11, atol=1e-14))


def test_transform_inverse_direct_largeN():
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
    )
    U = np.ones((10 ^ 6, D)) * 0.11
    X = parameter_transformer.inverse(U)
    U2 = parameter_transformer(X)
    assert np.all(np.isclose(U, U2, rtol=1e-12, atol=1e-14))

    for t in [12, 13]:  # probit, student4
        parameter_transformer = ParameterTransformer(
            D=D,
            lb_orig=np.ones((1, D)) * -10,
            ub_orig=np.ones((1, D)) * 10,
            transform_type=t,
        )
        U = np.ones((10 ^ 6, D)) * 0.11
        X = parameter_transformer.inverse(U)
        U2 = parameter_transformer(X)
        assert np.all(np.isclose(U, U2, rtol=1e-12, atol=1e-14))


def test_log_abs_det_jacobian_type3_within():
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
    )
    U = np.ones((10, D)) * 3
    log_j = parameter_transformer.log_abs_det_jacobian(U)
    log_j2 = np.ones((10)) * -0.3043
    assert np.all(np.isclose(log_j, log_j2, atol=1e-04))


def test_log_abs_det_jacobian_type3_within_negative():
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
    )
    U = np.ones((10, D)) * -4
    log_j = parameter_transformer.log_abs_det_jacobian(U)
    log_j2 = np.ones((10)) * -3.1217
    assert np.all(np.isclose(log_j, log_j2))


def test_log_abs_det_jacobian_type0():
    parameter_transformer = ParameterTransformer(D=D)
    U = np.ones((10, D)) * 5
    log_j = parameter_transformer.log_abs_det_jacobian(U)
    log_j2 = np.ones((10)) * 0
    assert np.all(np.isclose(log_j, log_j2, atol=1e-04))


def test_log_abs_det_jacobian_type0_negative():
    parameter_transformer = ParameterTransformer(D=D)
    U = np.ones((10, D)) * -6
    log_j = parameter_transformer.log_abs_det_jacobian(U)
    log_j2 = np.ones((10)) * 0
    assert np.all(np.isclose(log_j, log_j2))


def test_1D_input_call():
    parameter_transformer = ParameterTransformer(D=D)
    X = np.ones(D)
    Y = parameter_transformer(X)
    assert X.shape == Y.shape
    Y2 = parameter_transformer(x=X)
    assert X.shape == Y2.shape

    for t in [12, 13]:  # probit, student4
        parameter_transformer = ParameterTransformer(D=D, transform_type=t)
        X = np.ones(D)
        Y = parameter_transformer(X)
        assert X.shape == Y.shape
        Y2 = parameter_transformer(x=X)
        assert X.shape == Y2.shape


def test_1D_input_inverse():
    parameter_transformer = ParameterTransformer(D=D)
    Y = np.ones((D))
    X = parameter_transformer.inverse(Y)
    assert X.shape == Y.shape
    X2 = parameter_transformer.inverse(u=Y)
    assert X2.shape == Y.shape

    for t in [12, 13]:  # probit, student4
        parameter_transformer = ParameterTransformer(D=D, transform_type=t)
        Y = np.ones((D))
        X = parameter_transformer.inverse(Y)
        assert X.shape == Y.shape
        X2 = parameter_transformer.inverse(u=Y)
        assert X2.shape == Y.shape


def test_1D_input_log_abs_det_jacobian():
    parameter_transformer = ParameterTransformer(D=D)
    U = np.ones((D))
    log_j = parameter_transformer.log_abs_det_jacobian(U)
    assert np.ndim(log_j) == 0
    log_j2 = parameter_transformer.log_abs_det_jacobian(u=U)
    assert np.ndim(log_j2) == 0

    for t in [12, 13]:  # probit, student4
        parameter_transformer = ParameterTransformer(D=D, transform_type=t)
        U = np.ones((D))
        log_j = parameter_transformer.log_abs_det_jacobian(U)
        assert np.ndim(log_j) == 0
        log_j2 = parameter_transformer.log_abs_det_jacobian(u=U)
        assert np.ndim(log_j2) == 0


def test_bounded_log_abs_det_jacobian_numerically():
    D = np.random.randint(1, 13)
    LB = -2 * np.ones((1, D))
    UB = 2 * np.ones((1, D))
    for t in bounded_transform_types:  # logit, probit, student4
        parameter_transformer = ParameterTransformer(
            D=D, lb_orig=LB, ub_orig=UB, transform_type=t
        )

        dx = 1e-6
        # Pick a random point where the Jacobian is not 1 but not too
        # close to boundaries
        x1 = np.random.choice([-1, 1]) * np.random.uniform(
            0.5 * np.ones((1, D)), 1.5 * np.ones((1, D))
        )
        x2 = x1 + dx * np.ones((1, D))
        x1_t = parameter_transformer(x1)
        x2_t = parameter_transformer(x2)

        vol1 = np.prod(x2 - x1)
        vol2 = np.prod(x2_t - x1_t)

        assert np.isclose(
            vol1 / vol2,
            np.exp(parameter_transformer.log_abs_det_jacobian(x1_t)),
        )


def test_transform_bounded_and_unbounded():
    D = 2
    n = 10
    x1 = np.random.uniform(0, 10, size=(n, 1))
    x2 = np.random.normal(size=(n, 1))
    x = np.hstack([x1, x2])

    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.array([[0.0, -np.inf]]),
        ub_orig=np.array([[10.0, np.inf]]),
    )
    assert np.allclose(
        x, parameter_transformer.inverse(parameter_transformer(x))
    )
    assert np.allclose(
        x,
        parameter_transformer(parameter_transformer.inverse(x)),
    )

    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.array([[0.0, -np.inf]]),
        ub_orig=np.array([[10.0, np.inf]]),
        plb_orig=np.array([[4.5, -np.inf]]),
        pub_orig=np.array([[5.5, np.inf]]),
        transform_type="probit",
    )
    assert np.allclose(
        x, parameter_transformer.inverse(parameter_transformer(x))
    )
    assert np.allclose(
        x,
        parameter_transformer(parameter_transformer.inverse(x)),
    )

    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.array([[0.0, -np.inf]]),
        ub_orig=np.array([[10.0, np.inf]]),
        transform_type="student4",
    )
    assert np.allclose(
        x, parameter_transformer.inverse(parameter_transformer(x))
    )
    assert np.allclose(
        x,
        parameter_transformer(parameter_transformer.inverse(x)),
    )


def test_abs_det_jacobian_bounded_and_unbounded():
    D = 2
    n = 10
    x1 = np.random.uniform(0, 10, size=(n, 1))
    x2 = np.random.normal(size=(n, 1))
    x = np.hstack([x1, x2])

    bounded_transformer = ParameterTransformer(
        D=1,
        lb_orig=np.array([[0.0]]),
        ub_orig=np.array([[10.0]]),
    )
    unbounded_transformer = ParameterTransformer(
        D=1,
        lb_orig=np.array([[-np.inf]]),
        ub_orig=np.array([[np.inf]]),
        plb_orig=np.array([[-0.5]]),
        pub_orig=np.array([[0.5]]),
    )
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.array([[0.0, -np.inf]]),
        ub_orig=np.array([[10.0, np.inf]]),
        plb_orig=np.array([[0.0, -0.5]]),
        pub_orig=np.array([[10.0, 0.5]]),
    )

    j1 = bounded_transformer.log_abs_det_jacobian(x1)
    j2 = unbounded_transformer.log_abs_det_jacobian(x2)
    j = parameter_transformer.log_abs_det_jacobian(x)

    assert np.allclose(j2, 0.0)
    assert np.allclose(j, j1)


def test_boundary_edge_cases():
    D = 4
    for t in bounded_transform_types:
        lb = np.full((1, D), 1000.0)
        ub = np.full((1, D), 1001.0)
        parameter_transformer = ParameterTransformer(
            D=D, lb_orig=lb, ub_orig=ub, transform_type=t
        )
        close_to_lb = np.nextafter(lb, np.inf)
        close_to_ub = np.nextafter(ub, -np.inf)
        close_to_lb_transformed = parameter_transformer(close_to_lb)
        close_to_ub_transformed = parameter_transformer(close_to_ub)
        assert np.all(np.isfinite(close_to_lb_transformed))
        assert np.all(np.isfinite(close_to_ub_transformed))
        assert np.all(
            parameter_transformer.inverse(close_to_lb_transformed)
            == close_to_lb
        )
        assert np.all(
            parameter_transformer.inverse(close_to_ub_transformed)
            == close_to_ub
        )
        big_num = np.sqrt(np.finfo(np.float64).max)
        assert np.all(
            parameter_transformer.inverse(np.full((1, D), -big_num))
            == close_to_lb
        )
        assert np.all(
            parameter_transformer.inverse(np.full((1, D), big_num))
            == close_to_ub
        )

        lb = np.full((1, D), -1000.0)
        ub = np.full((1, D), 0.0)
        parameter_transformer = ParameterTransformer(
            D=D, lb_orig=lb, ub_orig=ub, transform_type=t
        )
        close_to_lb = np.nextafter(lb, np.inf)
        close_to_ub = np.nextafter(ub, -np.inf)
        close_to_lb_transformed = parameter_transformer(close_to_lb)
        close_to_ub_transformed = parameter_transformer(close_to_ub)
        assert np.all(np.isfinite(close_to_lb_transformed))
        assert np.all(np.isfinite(close_to_ub_transformed))
        assert np.allclose(
            parameter_transformer.inverse(close_to_lb_transformed), close_to_lb
        )
        assert np.allclose(
            parameter_transformer.inverse(close_to_ub_transformed), close_to_ub
        )
        big_num = np.sqrt(np.finfo(np.float64).max)
        assert np.all(
            parameter_transformer.inverse(np.full((1, D), -big_num))
            == close_to_lb
        )
        assert np.all(
            parameter_transformer.inverse(np.full((1, D), big_num))
            == close_to_ub
        )


def test_lb_ub_map_to_inf():
    D = 4
    lb_orig = -np.random.normal(scale=100, size=(1, D))
    lb_orig[0, 0], lb_orig[0, 2] = -np.inf, -np.inf  # Mixed bound types
    ub_orig = lb_orig + np.random.lognormal(sigma=2, size=(1, D))
    ub_orig[0, 0], ub_orig[0, 2] = np.inf, np.inf  # Mixed bound types

    for t in bounded_transform_types:
        parameter_transformer = ParameterTransformer(
            D=D,
            lb_orig=lb_orig,
            ub_orig=ub_orig,
            transform_type=t,
        )
        # Hard bounds should map to +- infinity for all variables,
        # both bounded + unbounded, and all types of bounded transforms:
        assert np.all(parameter_transformer(lb_orig) == -np.inf)
        assert np.all(parameter_transformer(ub_orig) == np.inf)


def test__str__and__repr__():
    D = 4
    bounded_names = {
        0: "unbounded",
        3: "logit",
        12: "probit",
        13: "student4",
    }
    for t in bounded_transform_types:
        lb = np.array([[-1.0, -np.inf, -1.0, -np.inf]])
        ub = np.array([[1.0, np.inf, 1.0, np.inf]])
        parameter_transformer = ParameterTransformer(
            D=D, lb_orig=lb, ub_orig=ub, transform_type=t
        )
        string = parameter_transformer.__str__()
        assert bounded_names[t] in string
        parameter_transformer.__repr__()
