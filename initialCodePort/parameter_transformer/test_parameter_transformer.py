import pytest
import numpy as np
from parameter_transformer import ParameterTransformer

"""
    toDo:
    - test init < < for bounds
"""

"""
__Init__ method
"""
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


def test_init_mu_inf_bounds():
    parameter_transformer = ParameterTransformer(nvars=NVARS)
    assert np.all(parameter_transformer.mu == np.zeros(NVARS))


def test_init_delta_inf_bounds():
    parameter_transformer = ParameterTransformer(nvars=NVARS)
    assert np.all(parameter_transformer.delta == np.ones(NVARS))


def test_direct_transform_type3_within():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS,
        lower_bounds=np.ones((1, NVARS)) * -10,
        upper_bounds=np.ones((1, NVARS)) * 10,
    )
    X = np.ones((10, NVARS)) * 3
    Y = parameter_transformer.direct_transform(X)
    Y2 = np.ones((10, NVARS)) * 0.619
    assert np.all(np.isclose(Y, Y2, atol=1e-04))


def test_direct_transform_type3_within_negative():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS,
        lower_bounds=np.ones((1, NVARS)) * -10,
        upper_bounds=np.ones((1, NVARS)) * 10,
    )
    X = np.ones((10, NVARS)) * -4
    Y = parameter_transformer.direct_transform(X)
    Y2 = np.ones((10, NVARS)) * -0.8473
    assert np.all(np.isclose(Y, Y2))


def test_direct_transform_type0():
    parameter_transformer = ParameterTransformer(nvars=NVARS)
    X = np.ones((10, NVARS)) * 3
    Y = parameter_transformer.direct_transform(X)
    assert np.all(Y == X)


def test_direct_transform_type0_negative():
    parameter_transformer = ParameterTransformer(nvars=NVARS)
    X = np.ones((10, NVARS)) * -4
    Y = parameter_transformer.direct_transform(X)
    assert np.all(Y == X)


def test_inverse_transform_type3_within():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS,
        lower_bounds=np.ones((1, NVARS)) * -10,
        upper_bounds=np.ones((1, NVARS)) * 10,
    )
    Y = np.ones((10, NVARS)) * 3
    X = parameter_transformer.inverse_transform(Y)
    X2 = np.ones((10, NVARS)) * 9.0515
    assert np.all(np.isclose(X, X2))


def test_inverse_transform_type3_within_negative():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS,
        lower_bounds=np.ones((1, NVARS)) * -10,
        upper_bounds=np.ones((1, NVARS)) * 10,
    )
    Y = np.ones((10, NVARS)) * -4
    X = parameter_transformer.inverse_transform(Y)
    X2 = np.ones((10, NVARS)) * -9.6403
    assert np.all(np.isclose(X, X2))


def test_inverse_transform_type0():
    parameter_transformer = ParameterTransformer(nvars=NVARS)
    Y = np.ones((10, NVARS)) * 3
    X = parameter_transformer.inverse_transform(Y)
    assert np.all(Y == X)


def test_inverse_transform_type0_negative():
    parameter_transformer = ParameterTransformer(nvars=NVARS)
    Y = np.ones((10, NVARS)) * -4
    X = parameter_transformer.inverse_transform(Y)
    assert np.all(Y == X)


def test_inverse_transform_type3_min_space():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS,
        lower_bounds=np.ones((1, NVARS)) * -10,
        upper_bounds=np.ones((1, NVARS)) * 10,
    )
    Y = np.ones((10, NVARS)) * -500
    X = parameter_transformer.inverse_transform(Y)
    assert np.all(X == np.ones((1, NVARS)) * -10)


def test_inverse_transform_type3_max_space():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS,
        lower_bounds=np.ones((1, NVARS)) * -10,
        upper_bounds=np.ones((1, NVARS)) * 10,
    )
    Y = np.ones((10, NVARS)) * 3000
    X = parameter_transformer.inverse_transform(Y)
    assert np.all(X == np.ones((10, NVARS)) * 10)

def test_log_jacobian_type3_within():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS,
        lower_bounds=np.ones((1, NVARS)) * -10,
        upper_bounds=np.ones((1, NVARS)) * 10,
    )
    U = np.ones((10, NVARS)) * 3
    log_j = parameter_transformer.log_jacobian(U)
    log_j2 = np.ones((10)) * -0.3043
    assert np.all(np.isclose(log_j, log_j2, atol=1e-04))


def test_log_jacobian_type3_within_negative():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS,
        lower_bounds=np.ones((1, NVARS)) * -10,
        upper_bounds=np.ones((1, NVARS)) * 10,
    )
    U = np.ones((10, NVARS)) * -4
    log_j = parameter_transformer.log_jacobian(U)
    log_j2 = np.ones((10)) * -3.1217
    assert np.all(np.isclose(log_j, log_j2))

def test_log_jacobian_type0():
    parameter_transformer = ParameterTransformer(nvars=NVARS)
    U = np.ones((10, NVARS)) * 5
    log_j = parameter_transformer.log_jacobian(U)
    log_j2 = np.ones((10)) * 0
    assert np.all(np.isclose(log_j, log_j2, atol=1e-04))


def test_log_jacobian_type0_negative():
    parameter_transformer = ParameterTransformer(nvars=NVARS)
    U = np.ones((10, NVARS)) * -6
    log_j = parameter_transformer.log_jacobian(U)
    log_j2 = np.ones((10)) * 0
    assert np.all(np.isclose(log_j, log_j2))