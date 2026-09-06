import numpy as np
import pytest

from pyvbmc.function_logger import FunctionLogger
from pyvbmc.parameter_transformer import ParameterTransformer

non_noisy_function = lambda x: np.sum(x + 2)
noisy_function = lambda x: (np.sum(x + 2), np.sum(x))
noisy_function_2 = lambda x: (np.sum(x + 2) + 0.5, np.sum(x) + 0.25)


def test_call_index():
    f_logger = FunctionLogger(non_noisy_function, 3, False, 0)
    _, _, idx = f_logger(np.array([3, 4, 5]))
    assert idx == 0


def test_call_f_val():
    x = np.array([3, 4, 5])
    f_logger = FunctionLogger(non_noisy_function, 3, False, 0)
    f_val, _, _ = f_logger(x)
    assert np.all(f_val == non_noisy_function(x))


def test_call_noisy_function_level_1():
    x = np.array([3, 4, 5])
    f_logger = FunctionLogger(non_noisy_function, 3, True, 1)
    f_val, f_sd, idx = f_logger(x)
    f_val2 = non_noisy_function(x)
    assert f_val == f_val2
    assert f_sd == 1
    assert idx == 0
    assert f_logger.S[0] == 1


def test_call_noisy_function_level_2():
    x = np.array([3, 4, 5])
    f_logger = FunctionLogger(noisy_function, 3, True, 2)
    f_val, f_sd, idx = f_logger(x)
    f_val2, f_sd2 = noisy_function(x)
    assert f_val == f_val2
    assert f_sd == f_sd2
    assert idx == 0
    assert f_logger.S[0] == f_sd2


def test_call_duplicate_noisy_function_level_2():
    x = np.array([3, 4, 5])
    f_logger = FunctionLogger(noisy_function, 3, True, 2)
    f_val_1, f_sd_1, idx_1 = f_logger(x)
    assert f_logger.y[0, 0] == f_val_1
    assert f_logger.S[0, 0] == f_sd_1
    assert f_val_1, f_sd_1 == noisy_function(x)

    # Change the function to simulate another different observation:
    f_logger.fun = noisy_function_2
    f_val_2, f_sd_2, idx_2 = f_logger(x)
    f_val_meas, f_sd_meas = noisy_function_2(x)
    assert f_sd_2 == f_sd_meas

    assert idx_1 == 0
    assert idx_2 == 0
    tau_1 = 1 / f_sd_1**2
    tau_2 = 1 / f_sd_2**2
    assert f_logger.S[0] == 1 / np.sqrt(tau_1 + tau_2)
    assert f_logger.y_orig[0, 0] == f_val_2
    assert f_logger.y_orig[0] == (tau_1 * f_val_1 + tau_2 * f_val_meas) / (
        tau_1 + tau_2
    )
    assert f_logger.y[0] == f_logger.y_orig[0]
    assert f_logger.S[0] == 1 / np.sqrt((1 / f_sd_1**2) + (1 / f_sd_2**2))


def test_call_expand_cache():
    x = np.array([3, 4, 5])
    f_logger = FunctionLogger(non_noisy_function, 3, False, 0, cache_size=1)
    assert f_logger.X.shape[0] == 1
    f_logger(x)
    f_logger(x * 2)
    y = non_noisy_function(x * 2)
    assert np.all(f_logger.y_orig[1] == y)


def test_add_expand_cache():
    x = np.array([3, 4, 5])
    y = non_noisy_function(x)
    f_logger = FunctionLogger(non_noisy_function, 3, False, 0, cache_size=1)
    assert f_logger.X.shape[0] == 1
    f_logger.add(x, y, None)
    f_logger.add(x * 2, y, None)
    assert np.all(f_logger.X[1] == x * 2)
    assert np.all(f_logger.y_orig[1] == y)
    assert f_logger.cache_count == 2


def test_add_record_funtime():
    x = np.array([3, 4, 5])
    y = non_noisy_function(x)
    f_logger = FunctionLogger(non_noisy_function, 3, False, 0)
    f_logger.add(x, y, None, 10)
    f_logger.add(x * 2, y, None, 10)
    f_logger.add(x * 3, y, None)
    assert f_logger.total_fun_eval_time == np.nansum(f_logger.fun_eval_time)
    assert f_logger.total_fun_eval_time == 20


def test_add_no_f_sd():
    x = np.array([3, 4, 5])
    y = non_noisy_function(x)
    f_logger = FunctionLogger(non_noisy_function, 3, True, 1, cache_size=1)
    f_logger.add(x, y, None)
    f_logger.add(x * 2, y, None)
    assert np.all(f_logger.X[1] == x * 2)
    assert np.all(f_logger.y_orig[1] == y)
    assert np.all(f_logger.S[1] == 1)


def test_call_record_stats():
    x = np.array([3, 4, 5])
    f_logger = FunctionLogger(non_noisy_function, 3, False, 0)
    for i in range(10):
        f_logger(x * i)
    assert f_logger.total_fun_eval_time == np.nansum(f_logger.fun_eval_time)
    assert np.all(f_logger.X_orig[9] == x * 9)
    assert f_logger.y_orig[9] == non_noisy_function(x * 9)
    assert f_logger.y_max == non_noisy_function(x * 9)
    assert np.sum(f_logger.X_flag) == 10
    assert f_logger.Xn == 9
    assert f_logger.cache_count == 0


def test_add_record_stats():
    x = np.array([3, 4, 5])
    f_logger = FunctionLogger(non_noisy_function, 3, False, 0)
    for i in range(10):
        y = non_noisy_function(x * i)
        f_logger.add(x * i, y, None)
    assert f_logger.total_fun_eval_time == 0
    assert np.nansum(f_logger.fun_eval_time) == 0
    assert np.all(f_logger.X_orig[9] == x * 9)
    assert f_logger.y_orig[9] == non_noisy_function(x * 9)
    assert f_logger.y_max == non_noisy_function(x * 9)
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
    f_logger.X[1] = x
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
    assert f_logger.n_evals[0] == 1
    assert f_logger.n_evals[1] == 2
    assert np.all(f_logger.X[1] == x)
    assert f_logger.y[1] == 5
    assert f_logger.y_orig[1] == 5
    assert f_logger.fun_eval_time[1] == 5


def test_record_duplicate_f_sd():
    x = np.array([3, 4, 5])
    f_logger = FunctionLogger(
        non_noisy_function, 3, True, 1, 500, ParameterTransformer(3)
    )
    f_logger._record(x * 2, x * 2, 18, 2, 9)
    f_logger._record(x, x, 9, 3, 9)
    _, idx = f_logger._record(x, x, 1, 3, 1)
    assert idx == 1
    assert f_logger.Xn == 1
    assert f_logger.n_evals[0] == 1
    assert f_logger.n_evals[1] == 2
    assert np.all(f_logger.X[1] == x)
    assert np.isclose(f_logger.y[1], 5, rtol=1e-12, atol=1e-14)
    assert np.isclose(f_logger.y_orig[1], 5, rtol=1e-12, atol=1e-14)
    assert f_logger.fun_eval_time[1] == 5
    assert f_logger.S[1] == 1 / np.sqrt(1 / 9 + 1 / 9)


def test_finalize():
    x = np.array([3, 4, 5])
    f_logger = FunctionLogger(non_noisy_function, 3, False, 0)
    for i in range(10):
        f_logger(x * i)
    f_logger.finalize()
    assert f_logger.total_fun_eval_time == np.sum(f_logger.fun_eval_time)
    assert np.all(f_logger.X_orig[9] == x * 9)
    assert f_logger.y_orig[9] == non_noisy_function(x * 9)
    assert f_logger.y_max == non_noisy_function(x * 9)
    assert np.sum(f_logger.X_flag) == 10
    assert f_logger.Xn == 9
    assert f_logger.func_count == 10
    assert f_logger.X_orig.shape[0] == 10
    assert f_logger.y_orig.shape[0] == 10
    assert f_logger.X.shape[0] == 10
    assert f_logger.y.shape[0] == 10
    assert f_logger.X_flag.shape[0] == 10
    assert f_logger.fun_eval_time.shape[0] == 10

    # noise level 2
    f_logger = FunctionLogger(noisy_function, 3, True, 2)
    for i in range(1, 10):
        f_logger(x * i)
    f_logger.finalize()
    f_val, f_sd = noisy_function(x * 9)
    assert f_logger.S[8] == f_sd
    assert f_logger.y_orig[8] == f_val
    assert f_logger.S.shape[0] == 9


def test_call_parameter_transform_no_constraints():
    x = np.array([3, 4, 5])
    parameter_transformer = ParameterTransformer(3)
    f_logger = FunctionLogger(
        non_noisy_function, 3, False, 0, 500, parameter_transformer
    )
    f_val, _, _ = f_logger(x)
    assert np.all(f_logger.X[0] == f_logger.X_orig[0])
    assert np.all(f_logger.y[0] == f_logger.y_orig[0])
    assert np.all(f_val == non_noisy_function(x))


def test_add_parameter_transform():
    x = np.array([3, 4, 5])
    parameter_transformer = ParameterTransformer(3)
    f_logger = FunctionLogger(
        non_noisy_function, 3, False, 0, 500, parameter_transformer
    )
    f_val_orig = non_noisy_function(x)
    f_logger.add(x, f_val_orig)
    assert np.all(f_logger.X[0] == f_logger.X_orig[0])
    assert np.all(f_logger.y[0] == f_logger.y_orig[0])
    assert np.all(f_logger.y_orig[0] == f_val_orig)


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
        x = np.array([x])
        return x @ x

    f_logger = FunctionLogger(error_function, 3, False, 0)
    with pytest.raises(ValueError) as err:
        f_logger(x)
    assert "FunctionLogger:FuncError" in str(err.value)


def test_call_non_scalar_return():
    """
    Test that the FunctionLogger is robust against a function returning an array
    of size 1 instead of just a float.
    """
    x = np.array([3, 4, 5])
    f_logger = FunctionLogger(lambda x: np.array([np.sum(x)]), 3, False, 0)
    f_logger(x)
    f_logger(x * 2)
    y = np.sum(x * 2)
    assert np.all(f_logger.y_orig[1] == y)


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


def test__str__():
    x = np.array([3, 4, 5])
    f_logger = FunctionLogger(non_noisy_function, 3, False, 0)
    for i in range(10):
        f_logger(x * i)
    f_logger.__str__()
    f_logger.__repr__()


def test_batch_call_requires_vectorized_mode():
    f_logger = FunctionLogger(non_noisy_function, 3, False, 0)
    with pytest.raises(RuntimeError, match="vectorized_target=True"):
        f_logger.batch_call(np.ones((2, 3)))


def test_vectorized_call_keeps_two_dimensional_target_contract():
    calls = []

    def target(x):
        calls.append(x.copy())
        return np.sum(x, axis=1)

    f_logger = FunctionLogger(target, 3, False, 0, vectorized_target=True)
    value, sd, idx = f_logger(np.array([1.0, 2.0, 3.0]))

    assert calls[0].shape == (1, 3)
    assert value == 6
    assert sd is None
    assert idx == 0


@pytest.mark.parametrize("column_output", [False, True])
def test_batch_call_output_shape_and_order(column_output):
    calls = []

    def target(x):
        calls.append(x.copy())
        values = np.sum(x, axis=1)
        return values[:, None] if column_output else values

    f_logger = FunctionLogger(
        target, 2, False, 0, cache_size=1, vectorized_target=True
    )
    x = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [2.0, 2.0]])
    values, sds, indices = f_logger.batch_call(
        x, np.array([10.0, np.nan, np.nan, np.nan])
    )

    assert np.array_equal(calls[0], x[[1, 2, 3]])
    assert np.array_equal(values, [10.0, 2.0, 5.0, 4.0])
    assert sds is None
    assert np.array_equal(indices, [0, 1, 0, 2])
    assert np.array_equal(f_logger.X[:3], x[[0, 1, 3]])
    assert f_logger.func_count == 3
    assert f_logger.cache_count == 1
    assert f_logger.X.shape[0] > 1


def test_batch_call_all_cached_skips_target():
    def target(x):
        raise AssertionError("target should not be called")

    f_logger = FunctionLogger(target, 1, False, 0, vectorized_target=True)
    values, sds, indices = f_logger.batch_call(
        np.array([[1.0], [2.0]]), [3.0, 4.0]
    )

    assert np.array_equal(values, [3.0, 4.0])
    assert sds is None
    assert np.array_equal(indices, [0, 1])
    assert f_logger.func_count == 0
    assert f_logger.cache_count == 2


def test_batch_call_transforms_all_target_inputs_to_original_space():
    target_inputs = []
    transformer = ParameterTransformer(
        2,
        np.full((1, 2), -2.0),
        np.full((1, 2), 2.0),
        np.full((1, 2), -1.0),
        np.full((1, 2), 1.0),
    )

    def target(x):
        target_inputs.append(x.copy())
        return np.sum(x, axis=1)

    f_logger = FunctionLogger(
        target,
        2,
        False,
        0,
        parameter_transformer=transformer,
        vectorized_target=True,
    )
    x_orig = np.array([[-0.5, 0.25], [0.75, -0.25]])
    f_logger.batch_call(transformer(x_orig))

    assert np.allclose(target_inputs[0], x_orig)
    assert np.allclose(f_logger.X_orig[:2], x_orig)


def test_batch_call_noisy_pair_and_unknown_noise():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])

    def noisy_target(points):
        return np.sum(points, axis=1), np.array([[0.5], [1.5]])

    provided = FunctionLogger(noisy_target, 2, True, 2, vectorized_target=True)
    values, sds, _ = provided.batch_call(x)
    assert np.array_equal(values, [3.0, 7.0])
    assert np.array_equal(sds, [0.5, 1.5])

    unknown = FunctionLogger(
        lambda points: np.sum(points, axis=1),
        2,
        True,
        1,
        vectorized_target=True,
    )
    _, sds, _ = unknown.batch_call(x)
    assert np.array_equal(sds, np.ones(2))


def test_batch_call_rejects_noisy_n_by_two_without_mutation():
    f_logger = FunctionLogger(
        lambda x: np.column_stack((np.sum(x, axis=1), np.ones(x.shape[0]))),
        2,
        True,
        2,
        vectorized_target=True,
    )
    with pytest.raises(ValueError, match=r"not an \(N, 2\) ndarray"):
        f_logger.batch_call(np.ones((2, 2)))
    assert f_logger.Xn == -1
    assert f_logger.func_count == 0
    assert f_logger.cache_count == 0


@pytest.mark.parametrize(
    "output, message",
    [
        (1.0, "shape"),
        (np.ones((2, 2)), "shape"),
        (np.array([1.0, np.inf]), "invalid rows: \\[1\\]"),
        (np.array([1.0, 2.0j]), "real values"),
    ],
)
def test_batch_call_validates_complete_output_before_recording(
    output, message
):
    f_logger = FunctionLogger(
        lambda x: output, 2, False, 0, vectorized_target=True
    )
    with pytest.raises(ValueError, match=message):
        f_logger.batch_call(np.ones((2, 2)))
    assert f_logger.Xn == -1
    assert f_logger.func_count == 0
    assert f_logger.cache_count == 0


def test_batch_call_validates_inputs_and_sd_before_recording():
    with pytest.raises(ValueError, match="must be a boolean"):
        FunctionLogger(np.sum, 2, False, 0, vectorized_target=1)

    f_logger = FunctionLogger(
        lambda x: (np.ones(2), np.array([1.0, 0.0])),
        2,
        True,
        2,
        vectorized_target=True,
    )
    with pytest.raises(ValueError, match="positive values"):
        f_logger.batch_call(np.ones((2, 2)))
    assert f_logger.Xn == -1

    with pytest.raises(ValueError, match="nonempty shape"):
        f_logger.batch_call(np.ones(2))
    with pytest.raises(ValueError, match="invalid rows"):
        f_logger.batch_call(np.ones((2, 2)), [1.0, np.inf])


@pytest.mark.parametrize(
    "x",
    [
        np.array([[1.0], ["bad"]], dtype=object),
        np.array([[1.0], [np.nan]]),
        np.array([[1.0], [np.inf]]),
        np.array([[1.0], [2.0j]]),
    ],
)
def test_batch_call_validates_all_coordinates_before_mutation(x):
    calls = []

    def target(points):
        calls.append(points)
        return np.zeros(points.shape[0])

    f_logger = FunctionLogger(target, 1, False, 0, vectorized_target=True)
    arrays_before = {
        name: getattr(f_logger, name).copy()
        for name in ("X", "X_orig", "y", "y_orig", "X_flag", "n_evals")
    }

    with pytest.raises(ValueError, match=r"invalid rows: \[1\]"):
        f_logger.batch_call(x, [1.0, 2.0])

    assert calls == []
    assert f_logger.Xn == -1
    assert f_logger.func_count == 0
    assert f_logger.cache_count == 0
    for name, before in arrays_before.items():
        assert np.array_equal(getattr(f_logger, name), before, equal_nan=True)


def test_batch_call_validates_inverse_transformed_coordinates_before_mutation():
    class InvalidInverse:
        def inverse(self, x):
            result = x.copy()
            result[-1, 0] = np.inf
            return result

    calls = []

    def target(points):
        calls.append(points)
        return np.zeros(points.shape[0])

    f_logger = FunctionLogger(
        target,
        1,
        False,
        0,
        parameter_transformer=InvalidInverse(),
        vectorized_target=True,
    )
    with pytest.raises(ValueError, match=r"invalid rows: \[1\]"):
        f_logger.batch_call(np.array([[1.0], [2.0]]), [1.0, 2.0])

    assert calls == []
    assert f_logger.Xn == -1
    assert f_logger.func_count == 0
    assert f_logger.cache_count == 0


def test_batch_call_distributes_target_time(mocker):
    mocker.patch("pyvbmc.timer.timer.time.time", side_effect=[10.0, 16.0])
    f_logger = FunctionLogger(
        lambda x: np.sum(x, axis=1),
        1,
        False,
        0,
        vectorized_target=True,
    )
    f_logger.batch_call(np.array([[1.0], [2.0], [3.0]]))

    assert np.array_equal(f_logger.fun_eval_time[:3, 0], [2.0, 2.0, 2.0])
    assert f_logger.total_fun_eval_time == 6.0


def test_batch_call_function_error_has_batch_context():
    def target(x):
        raise RuntimeError("model failed")

    f_logger = FunctionLogger(target, 2, False, 0, vectorized_target=True)
    with pytest.raises(RuntimeError) as err:
        f_logger.batch_call(np.ones((3, 2)))
    assert "FunctionLogger:FuncError" in str(err.value)
    assert "3 batch rows" in str(err.value)
    assert "(3, 2)" in str(err.value)
