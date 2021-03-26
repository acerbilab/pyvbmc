import numpy as np
import pytest

from parameter_transformer import ParameterTransformer

NVARS = 3


def test_init_no_lower_bounds():
    parameter_transformer = ParameterTransformer(nvars=NVARS)
    assert np.all(np.isinf(parameter_transformer.lb_orig))


def test_init_lower_bounds():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS, lower_bounds=np.ones((1, NVARS))
    )
    assert np.all(parameter_transformer.lb_orig == np.ones(NVARS))


def test_init_no_upper_bounds():
    parameter_transformer = ParameterTransformer(nvars=NVARS)
    assert np.all(np.isinf(parameter_transformer.ub_orig))


def test_init_upper_bounds():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS, upper_bounds=np.ones((1, NVARS))
    )
    assert np.all(parameter_transformer.ub_orig == np.ones(NVARS))


def test_init_type_3():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS,
        lower_bounds=np.ones((1, NVARS)),
        upper_bounds=np.ones((1, NVARS)) * 2,
    )
    assert np.all(parameter_transformer.type == np.ones(NVARS) * 3)


def test_init_bounds_check():
    with pytest.raises(ValueError):
        ParameterTransformer(
            nvars=NVARS,
            lower_bounds=np.ones((1, NVARS)) * 3,
            upper_bounds=np.ones((1, NVARS)) * 2,
        )
    with pytest.raises(ValueError):
        ParameterTransformer(
            nvars=NVARS,
            lower_bounds=np.ones((1, NVARS)) * 0,
            upper_bounds=np.ones((1, NVARS)) * 10,
            plausible_lower_bounds=np.ones((1, NVARS)) * -1,
        )
    with pytest.raises(ValueError):
        ParameterTransformer(
            nvars=NVARS,
            lower_bounds=np.ones((1, NVARS)) * 0,
            upper_bounds=np.ones((1, NVARS)) * 10,
            plausible_upper_bounds=np.ones((1, NVARS)) * 11,
        )
    with pytest.raises(ValueError):
        ParameterTransformer(
            nvars=NVARS,
            lower_bounds=np.ones((1, NVARS)) * 0,
            upper_bounds=np.ones((1, NVARS)) * 10,
            plausible_lower_bounds=np.ones((1, NVARS)) * 100,
            plausible_upper_bounds=np.ones((1, NVARS)) * -20,
        )


def test_init_mu_inf_bounds():
    parameter_transformer = ParameterTransformer(nvars=NVARS)
    assert np.all(parameter_transformer.mu == np.zeros(NVARS))


def test_init_delta_inf_bounds():
    parameter_transformer = ParameterTransformer(nvars=NVARS)
    assert np.all(parameter_transformer.delta == np.ones(NVARS))


def test_init_type3_mu_all_params():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS,
        lower_bounds=np.ones((1, NVARS)) * -10,
        upper_bounds=np.ones((1, NVARS)) * 10,
        plausible_lower_bounds=np.ones((1, NVARS)) * 2,
        plausible_upper_bounds=np.ones((1, NVARS)) * 4,
    )
    parameter_transformer2 = ParameterTransformer(
        nvars=NVARS,
        lower_bounds=np.ones((1, NVARS)) * -10,
        upper_bounds=np.ones((1, NVARS)) * 10,
    )
    plb = parameter_transformer2(np.ones((1, NVARS)) * 2)
    pub = parameter_transformer2(np.ones((1, NVARS)) * 4)
    mu2 = (plb + pub) * 0.5
    assert np.all(
        np.isclose(parameter_transformer.mu, mu2, rtol=1e-12, atol=1e-14)
    )


def test_init_type3_delta_all_params():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS,
        lower_bounds=np.ones((1, NVARS)) * -10,
        upper_bounds=np.ones((1, NVARS)) * 10,
        plausible_lower_bounds=np.ones((1, NVARS)) * 2,
        plausible_upper_bounds=np.ones((1, NVARS)) * 4,
    )
    parameter_transformer2 = ParameterTransformer(
        nvars=NVARS,
        lower_bounds=np.ones((1, NVARS)) * -10,
        upper_bounds=np.ones((1, NVARS)) * 10,
    )
    plb = parameter_transformer2(np.ones((1, NVARS)) * 2)
    pub = parameter_transformer2(np.ones((1, NVARS)) * 4)
    delta2 = plb - pub
    assert np.all(
        np.isclose(parameter_transformer.delta, delta2, rtol=1e-12, atol=1e-14)
    )


def test_direct_transform_type3_within():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS,
        lower_bounds=np.ones((1, NVARS)) * -10,
        upper_bounds=np.ones((1, NVARS)) * 10,
    )
    X = np.ones((10, NVARS)) * 3
    Y = parameter_transformer(X)
    Y2 = np.ones((10, NVARS)) * 0.619
    assert np.all(np.isclose(Y, Y2, atol=1e-04))


def test_direct_transform_type3_within_negative():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS,
        lower_bounds=np.ones((1, NVARS)) * -10,
        upper_bounds=np.ones((1, NVARS)) * 10,
    )
    X = np.ones((10, NVARS)) * -4
    Y = parameter_transformer(X)
    Y2 = np.ones((10, NVARS)) * -0.8473
    assert np.all(np.isclose(Y, Y2))


def test_direct_transform_type0():
    parameter_transformer = ParameterTransformer(nvars=NVARS)
    X = np.ones((10, NVARS)) * 3
    Y = parameter_transformer(X)
    assert np.all(Y == X)


def test_direct_transform_type0_negative():
    parameter_transformer = ParameterTransformer(nvars=NVARS)
    X = np.ones((10, NVARS)) * -4
    Y = parameter_transformer(X)
    assert np.all(Y == X)


def test_inverse_type3_within():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS,
        lower_bounds=np.ones((1, NVARS)) * -10,
        upper_bounds=np.ones((1, NVARS)) * 10,
    )
    Y = np.ones((10, NVARS)) * 3
    X = parameter_transformer.inverse(Y)
    X2 = np.ones((10, NVARS)) * 9.0515
    assert np.all(np.isclose(X, X2))


def test_inverse_type3_within_negative():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS,
        lower_bounds=np.ones((1, NVARS)) * -10,
        upper_bounds=np.ones((1, NVARS)) * 10,
    )
    Y = np.ones((10, NVARS)) * -4
    X = parameter_transformer.inverse(Y)
    X2 = np.ones((10, NVARS)) * -9.6403
    assert np.all(np.isclose(X, X2))


def test_inverse_type0():
    parameter_transformer = ParameterTransformer(nvars=NVARS)
    Y = np.ones((10, NVARS)) * 3
    X = parameter_transformer.inverse(Y)
    assert np.all(Y == X)


def test_inverse_type0_negative():
    parameter_transformer = ParameterTransformer(nvars=NVARS)
    Y = np.ones((10, NVARS)) * -4
    X = parameter_transformer.inverse(Y)
    assert np.all(Y == X)


def test_inverse_type3_min_space():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS,
        lower_bounds=np.ones((1, NVARS)) * -10,
        upper_bounds=np.ones((1, NVARS)) * 10,
    )
    Y = np.ones((10, NVARS)) * -500
    X = parameter_transformer.inverse(Y)
    assert np.all(X == np.ones((1, NVARS)) * -10)


def test_inverse_type3_max_space():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS,
        lower_bounds=np.ones((1, NVARS)) * -10,
        upper_bounds=np.ones((1, NVARS)) * 10,
    )
    Y = np.ones((10, NVARS)) * 3000
    X = parameter_transformer.inverse(Y)
    assert np.all(X == np.ones((10, NVARS)) * 10)


def test_transform_direct_inverse():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS,
        lower_bounds=np.ones((1, NVARS)) * -10,
        upper_bounds=np.ones((1, NVARS)) * 10,
    )
    X = np.ones((10, NVARS)) * 0.05
    U = parameter_transformer(X)
    X2 = parameter_transformer.inverse(U)
    assert np.all(np.isclose(X, X2, rtol=1e-12, atol=1e-14))


def test_transform_inverse_direct():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS,
        lower_bounds=np.ones((1, NVARS)) * -10,
        upper_bounds=np.ones((1, NVARS)) * 10,
    )
    U = np.ones((10, NVARS)) * 0.2
    X = parameter_transformer.inverse(U)
    U2 = parameter_transformer(X)
    assert np.all(np.isclose(U, U2, rtol=1e-12, atol=1e-14))


def test_transform_direct_inverse_largeN():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS,
        lower_bounds=np.ones((1, NVARS)) * -10,
        upper_bounds=np.ones((1, NVARS)) * 10,
    )
    X = np.ones((10 ^ 6, NVARS)) * 0.4
    U = parameter_transformer(X)
    X2 = parameter_transformer.inverse(U)
    assert np.all(np.isclose(X, X2, rtol=1e-12, atol=1e-14))


def test_transform_inverse_direct_largeN():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS,
        lower_bounds=np.ones((1, NVARS)) * -10,
        upper_bounds=np.ones((1, NVARS)) * 10,
    )
    U = np.ones((10 ^ 6, NVARS)) * 0.11
    X = parameter_transformer.inverse(U)
    U2 = parameter_transformer(X)
    assert np.all(np.isclose(U, U2, rtol=1e-12, atol=1e-14))


def test_log_abs_det_jacobian_type3_within():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS,
        lower_bounds=np.ones((1, NVARS)) * -10,
        upper_bounds=np.ones((1, NVARS)) * 10,
    )
    U = np.ones((10, NVARS)) * 3
    log_j = parameter_transformer.log_abs_det_jacobian(U)
    log_j2 = np.ones((10)) * -0.3043
    assert np.all(np.isclose(log_j, log_j2, atol=1e-04))


def test_log_abs_det_jacobian_type3_within_negative():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS,
        lower_bounds=np.ones((1, NVARS)) * -10,
        upper_bounds=np.ones((1, NVARS)) * 10,
    )
    U = np.ones((10, NVARS)) * -4
    log_j = parameter_transformer.log_abs_det_jacobian(U)
    log_j2 = np.ones((10)) * -3.1217
    assert np.all(np.isclose(log_j, log_j2))


def test_log_abs_det_jacobian_type0():
    parameter_transformer = ParameterTransformer(nvars=NVARS)
    U = np.ones((10, NVARS)) * 5
    log_j = parameter_transformer.log_abs_det_jacobian(U)
    log_j2 = np.ones((10)) * 0
    assert np.all(np.isclose(log_j, log_j2, atol=1e-04))


def test_log_abs_det_jacobian_type0_negative():
    parameter_transformer = ParameterTransformer(nvars=NVARS)
    U = np.ones((10, NVARS)) * -6
    log_j = parameter_transformer.log_abs_det_jacobian(U)
    log_j2 = np.ones((10)) * 0
    assert np.all(np.isclose(log_j, log_j2))


def test_1D_input_call():
    parameter_transformer = ParameterTransformer(nvars=NVARS)
    X = np.ones(NVARS)
    Y = parameter_transformer(X)
    assert X.shape == Y.shape
    Y2 = parameter_transformer(x=X)
    assert X.shape == Y2.shape


def test_1D_input_inverse():
    parameter_transformer = ParameterTransformer(nvars=NVARS)
    Y = np.ones((NVARS))
    X = parameter_transformer.inverse(Y)
    assert X.shape == Y.shape
    X2 = parameter_transformer.inverse(u=Y)
    assert X2.shape == Y.shape


def test_1D_input_log_abs_det_jacobian_():
    parameter_transformer = ParameterTransformer(nvars=NVARS)
    U = np.ones((NVARS))
    log_j = parameter_transformer.log_abs_det_jacobian(U)
    assert np.ndim(log_j) == 0
    log_j2 = parameter_transformer.log_abs_det_jacobian(u=U)
    assert np.ndim(log_j2) == 0