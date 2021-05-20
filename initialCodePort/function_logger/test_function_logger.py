import numpy as np
import pytest

from function_logger import FunctionLogger
from parameter_transformer.parameter_transformer import ParameterTransformer

non_noisy_function = lambda x: np.sum(x + 2)
noisy_function = lambda x: (np.sum(x + 2), np.sum(x))


def test_call_index():
    f_logger = FunctionLogger(non_noisy_function, 3, False, 0)
    _, _, idx = f_logger(np.array([3, 4, 5]))
    assert idx == 0


def test_call_fval():
    x = np.array([3, 4, 5])
    f_logger = FunctionLogger(non_noisy_function, 3, False, 0)
    fval, _, _ = f_logger(x)
    assert np.all(fval == non_noisy_function(x))


def test_call_noisy_function_level_1():
    x = np.array([3, 4, 5])
    f_logger = FunctionLogger(non_noisy_function, 3, True, 1)
    fval, fsd, idx = f_logger(x)
    fval2 = non_noisy_function(x)
    assert fval == fval2
    assert fsd == 1
    assert idx == 0
    assert f_logger.S[0] == 1


def test_call_noisy_function_level_2():
    x = np.array([3, 4, 5])
    f_logger = FunctionLogger(noisy_function, 3, True, 2)
    fval, fsd, idx = f_logger(x)
    fval2, fsd2 = noisy_function(x)
    assert fval == fval2
    assert fsd == fsd2
    assert idx == 0
    assert f_logger.S[0] == fsd2


def test_call_expand_cache():
    x = np.array([3, 4, 5])
    f_logger = FunctionLogger(non_noisy_function, 3, False, 0, cache_size=1)
    assert f_logger.x.shape[0] == 1
    f_logger(x)
    f_logger(x * 2)
    y = non_noisy_function(x * 2)
    assert np.all(f_logger.y_orig[1] == y)


def test_add_expand_cache():
    x = np.array([3, 4, 5])
    y = non_noisy_function(x)
    f_logger = FunctionLogger(non_noisy_function, 3, False, 0, cache_size=1)
    assert f_logger.x.shape[0] == 1
    f_logger.add(x, y, None)
    f_logger.add(x*2, y, None)
    assert np.all(f_logger.x[1] == x*2)
    assert np.all(f_logger.y_orig[1] == y)
    assert f_logger.cache_count == 2


def test_add_no_fsd():
    x = np.array([3, 4, 5])
    y = non_noisy_function(x)
    f_logger = FunctionLogger(non_noisy_function, 3, True, 1, cache_size=1)
    f_logger.add(x, y, None)
    f_logger.add(x*2, y, None)
    assert np.all(f_logger.x[1] == x*2)
    assert np.all(f_logger.y_orig[1] == y)
    assert np.all(f_logger.S[1] == 1)


def test_call_record_stats():
    x = np.array([3, 4, 5])
    f_logger = FunctionLogger(non_noisy_function, 3, False, 0)
    for i in range(10):
        f_logger(x * i)
    assert f_logger.total_fun_evaltime == np.sum(f_logger.fun_evaltime)
    assert np.all(f_logger.x_orig[9] == x * 9)
    assert f_logger.y_orig[9] == non_noisy_function(x * 9)
    assert f_logger.ymax == non_noisy_function(x * 9)
    assert np.sum(f_logger.X_flag) == 10
    assert f_logger.Xn == 9
    assert f_logger.cache_count == 0


def test_add_record_stats():
    x = np.array([3, 4, 5])
    f_logger = FunctionLogger(non_noisy_function, 3, False, 0)
    for i in range(10):
        y = non_noisy_function(x * i)
        f_logger.add(x * i, y, None)
    assert f_logger.total_fun_evaltime == 0
    assert np.sum(f_logger.fun_evaltime[:11]) == 0
    assert np.all(f_logger.x_orig[9] == x * 9)
    assert f_logger.y_orig[9] == non_noisy_function(x * 9)
    assert f_logger.ymax == non_noisy_function(x * 9)
    assert np.sum(f_logger.X_flag) == 10
    assert f_logger.Xn == 9
    assert f_logger.func_count == 0
    assert f_logger.cache_count == 10


def test_record_duplicate_existing_already():
    x = np.array([3, 4, 5])
    f_logger = FunctionLogger(
        non_noisy_function, 3, False, 0, 500, ParameterTransformer(3)
    )
    f_logger._record(x, x, 9, None, 10)
    f_logger.x[1] = x
    with pytest.raises(ValueError):
        f_logger._record(x, x, 9, None, 10)


def test_record_duplicate():
    x = np.array([3, 4, 5])
    f_logger = FunctionLogger(
        non_noisy_function, 3, False, 0, 500, ParameterTransformer(3)
    )
    f_logger._record(x * 2, x * 2, 18, None, 9)
    f_logger._record(x, x, 9, None, 9)
    _, idx = f_logger._record(x, x, 1, None, 1)
    assert idx == 1
    assert f_logger.Xn == 1
    assert f_logger.nevals[0] == 1
    assert f_logger.nevals[1] == 2
    assert np.all(f_logger.x[1] == x)
    assert f_logger.y[1] == 5
    assert f_logger.y_orig[1] == 5
    assert f_logger.fun_evaltime[1] == 5


def test_record_duplicate_fsd():
    x = np.array([3, 4, 5])
    f_logger = FunctionLogger(
        non_noisy_function, 3, True, 1, 500, ParameterTransformer(3)
    )
    f_logger._record(x * 2, x * 2, 18, 2, 9)
    f_logger._record(x, x, 9, 3, 9)
    _, idx = f_logger._record(x, x, 1, 3, 1)
    assert idx == 1
    assert f_logger.Xn == 1
    assert f_logger.nevals[0] == 1
    assert f_logger.nevals[1] == 2
    assert np.all(f_logger.x[1] == x)
    assert np.isclose(f_logger.y[1], 5, rtol=1e-12, atol=1e-14)
    assert np.isclose(f_logger.y_orig[1], 5, rtol=1e-12, atol=1e-14)
    assert f_logger.fun_evaltime[1] == 5
    assert f_logger.S[1] ==  1/np.sqrt(1/9 + 1/9)


def test_finalize():
    x = np.array([3, 4, 5])
    f_logger = FunctionLogger(non_noisy_function, 3, False, 0)
    for i in range(10):
        f_logger(x * i)
    f_logger.finalize()
    assert f_logger.total_fun_evaltime == np.sum(f_logger.fun_evaltime)
    assert np.all(f_logger.x_orig[9] == x * 9)
    assert f_logger.y_orig[9] == non_noisy_function(x * 9)
    assert f_logger.ymax == non_noisy_function(x * 9)
    assert np.sum(f_logger.X_flag) == 10
    assert f_logger.Xn == 9
    assert f_logger.func_count == 10
    assert f_logger.x_orig.shape[0] == 10
    assert f_logger.y_orig.shape[0] == 10
    assert f_logger.x.shape[0] == 10
    assert f_logger.y.shape[0] == 10
    assert f_logger.X_flag.shape[0] == 10
    assert f_logger.fun_evaltime.shape[0] == 10

    # noise level 2
    f_logger = FunctionLogger(noisy_function, 3, True, 2)
    for i in range(10):
        f_logger(x * i)
    f_logger.finalize()
    fval, fsd = noisy_function(x * 9)
    assert f_logger.S[9] == fsd
    assert f_logger.y_orig[9] == fval
    assert f_logger.S.shape[0] == 10


def test_call_parameter_transform_no_constraints():
    x = np.array([3, 4, 5])
    parameter_transformer = ParameterTransformer(3)
    f_logger = FunctionLogger(
        non_noisy_function, 3, False, 0, 500, parameter_transformer
    )
    fval, _, _ = f_logger(x)
    assert np.all(f_logger.x[0] == f_logger.x_orig[0])
    assert np.all(f_logger.y[0] == f_logger.y_orig[0])
    assert np.all(fval == non_noisy_function(x))


def test_add_parameter_transform():
    x = np.array([3, 4, 5])
    parameter_transformer = ParameterTransformer(3)
    f_logger = FunctionLogger(
        non_noisy_function, 3, False, 0, 500, parameter_transformer
    )
    fval_orig = non_noisy_function(x)
    f_logger.add(x, fval_orig)
    assert np.all(f_logger.x[0] == f_logger.x_orig[0])
    assert np.all(f_logger.y[0] == f_logger.y_orig[0])
    assert np.all(f_logger.y_orig[0] == fval_orig)


def test_call_invalid_func_value():
    x = np.array([3, 4, 5])
    return_inf_function = lambda x: x * np.inf
    f_logger = FunctionLogger(return_inf_function, 3, False, 0)
    with pytest.raises(ValueError):
        f_logger(x)


def test_call_invalid_sd_value():
    x = np.array([3, 4, 5])
    return_inf_function = lambda x: (np.sum(x), np.inf * 1)
    f_logger = FunctionLogger(return_inf_function, 3, True, 2)
    with pytest.raises(ValueError):
        f_logger(x)


def test_call_function_error():
    x = np.array([3, 4, 5])

    def error_function(x):
        return 3 / 0

    f_logger = FunctionLogger(error_function, 3, False, 0)
    with pytest.raises(ZeroDivisionError) as err:
        f_logger(x)
    assert "FunctionLogger:FuncError" in str(err.value)


def test_add_invalid_func_value():
    x = np.array([3, 4, 5])
    f_logger = FunctionLogger(non_noisy_function, 3, False, 0)
    with pytest.raises(ValueError):
        f_logger.add(x, np.inf)


def test_add_invalid_sd_value():
    x = np.array([3, 4, 5])
    f_logger = FunctionLogger(noisy_function, 3, True, 2)
    with pytest.raises(ValueError):
        f_logger.add(x, 3, np.inf)