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
    assert np.all(np.isinf(parameter_transformer.lower_bounds_orig))


def test_init_lower_bounds():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS, lower_bounds=np.ones(NVARS)
    )
    assert np.all(parameter_transformer.lower_bounds_orig == np.ones(NVARS))


def test_init_no_upper_bounds():
    parameter_transformer = ParameterTransformer(nvars=NVARS)
    assert np.all(np.isinf(parameter_transformer.upper_bounds_orig))


def test_init_upper_bounds():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS, upper_bounds=np.ones(NVARS)
    )
    assert np.all(parameter_transformer.upper_bounds_orig == np.ones(NVARS))

def test_init_type_1():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS,
        lower_bounds=np.ones(NVARS) * -np.inf,
        upper_bounds=np.ones(NVARS) * np.inf,
    )
    assert np.all(parameter_transformer.type == np.zeros(NVARS))


def test_init_type_1():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS, lower_bounds=np.ones(NVARS), upper_bounds=np.ones(NVARS) * np.inf
    )
    assert np.all(parameter_transformer.type == np.ones(NVARS) * 1)


def test_init_type_2():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS, lower_bounds=np.ones(NVARS) * -np.inf, upper_bounds=np.ones(NVARS)
    )
    assert np.all(parameter_transformer.type == np.ones(NVARS) * 2)


def test_init_type_3():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS, lower_bounds=np.ones(NVARS), upper_bounds=np.ones(NVARS)*2
    )
    assert np.all(parameter_transformer.type == np.ones(NVARS) * 3)

def test_init_mu_inf_bounds():
    parameter_transformer = ParameterTransformer(nvars=NVARS)
    assert np.all(parameter_transformer.mu == np.zeros(NVARS))

def test_init_delta_inf_bounds():
    parameter_transformer = ParameterTransformer(nvars=NVARS)
    assert np.all(parameter_transformer.delta == np.ones(NVARS))
