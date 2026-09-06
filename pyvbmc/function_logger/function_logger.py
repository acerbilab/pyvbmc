from copy import deepcopy
from textwrap import indent

import numpy as np

from pyvbmc.formatting import full_repr
from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.timer import Timer


def _validate_vectorized_output(
    value,
    n_rows: int,
    name: str,
    *,
    positive: bool = False,
    allow_nan: bool = False,
):
    """Validate and flatten one output of a vectorized target."""
    array = np.asarray(value)
    if array.shape == (n_rows, 1):
        array = array[:, 0]
    elif array.shape != (n_rows,):
        raise ValueError(
            f"{name} must have shape ({n_rows},) or ({n_rows}, 1), "
            f"but has shape {array.shape}."
        )

    if not np.isrealobj(array):
        rows = np.flatnonzero(~np.isreal(array)).tolist()
        raise ValueError(
            f"{name} must contain real values; invalid rows: {rows}."
        )
    try:
        finite = np.isfinite(array)
        nan = np.isnan(array)
    except TypeError as err:
        raise ValueError(f"{name} must contain numeric values.") from err
    if allow_nan:
        invalid = ~(finite | nan)
    else:
        invalid = ~finite
    if np.any(invalid):
        rows = np.flatnonzero(invalid).tolist()
        raise ValueError(
            f"{name} must contain finite real values"
            + (" or NaN markers" if allow_nan else "")
            + f"; invalid rows: {rows}."
        )
    if positive and np.any(array <= 0):
        rows = np.flatnonzero(array <= 0).tolist()
        raise ValueError(
            f"{name} must contain positive values; invalid rows: {rows}."
        )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        converted = np.asarray(array, dtype=float)
    converted_finite = np.isfinite(converted)
    if allow_nan:
        converted_invalid = ~(converted_finite | np.isnan(converted))
    else:
        converted_invalid = ~converted_finite
    if np.any(converted_invalid):
        rows = np.flatnonzero(converted_invalid).tolist()
        raise ValueError(
            f"{name} must remain finite when converted to float64; "
            f"invalid rows: {rows}."
        )
    if positive and np.any(converted <= 0):
        rows = np.flatnonzero(converted <= 0).tolist()
        raise ValueError(
            f"{name} must remain positive when converted to float64; "
            f"invalid rows: {rows}."
        )
    return converted


def _validate_batch_coordinates(value, n_rows: int, D: int, name: str):
    """Validate and convert a complete batch of coordinates."""
    array = np.asarray(value)
    if array.shape != (n_rows, D):
        raise ValueError(
            f"{name} must have shape ({n_rows}, {D}), "
            f"but has shape {array.shape}."
        )
    if not np.isrealobj(array):
        real_rows = np.all(np.isreal(array), axis=1)
        rows = np.flatnonzero(~real_rows).tolist()
        raise ValueError(
            f"{name} must contain real coordinates; invalid rows: {rows}."
        )
    try:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            converted = np.asarray(array, dtype=float)
    except (OverflowError, TypeError, ValueError) as err:
        invalid_rows = []
        for row in range(n_rows):
            try:
                np.asarray(array[row], dtype=float)
            except (OverflowError, TypeError, ValueError):
                invalid_rows.append(row)
        raise ValueError(
            f"{name} must contain numeric coordinates; "
            f"invalid rows: {invalid_rows}."
        ) from err
    finite_rows = np.all(np.isfinite(converted), axis=1)
    if not np.all(finite_rows):
        rows = np.flatnonzero(~finite_rows).tolist()
        raise ValueError(
            f"{name} must contain finite float64 coordinates; "
            f"invalid rows: {rows}."
        )
    return converted


def _validate_vectorized_target_output(
    output, n_rows: int, uncertainty_handling_level: int
):
    """Normalize the value and optional SD arrays returned by a target."""
    if uncertainty_handling_level == 2:
        if isinstance(output, np.ndarray):
            if output.shape == (n_rows, 2):
                raise ValueError(
                    "A vectorized noisy target must return a pair "
                    "(values, SDs), not an (N, 2) ndarray."
                )
            raise ValueError(
                "A vectorized noisy target must return a pair "
                "(values, SDs)."
            )
        if not isinstance(output, (tuple, list)) or len(output) != 2:
            raise ValueError(
                "A vectorized noisy target must return a pair "
                "(values, SDs)."
            )
        values = _validate_vectorized_output(
            output[0], n_rows, "Vectorized target values"
        )
        sds = _validate_vectorized_output(
            output[1],
            n_rows,
            "Vectorized target SDs",
            positive=True,
        )
    else:
        values = _validate_vectorized_output(
            output, n_rows, "Vectorized target values"
        )
        sds = (
            np.ones(n_rows, dtype=float)
            if uncertainty_handling_level == 1
            else None
        )
    return values, sds


class FunctionLogger:
    """
    Class that evaluates a function and caches its values.

    Parameters
    ----------
    fun : callable
        The function to be logged.
        `fun` must take a vector input and return a scalar value and,
        optionally, the (estimated) SD of the returned value (if the
        function fun is stochastic). If ``vectorized_target`` is true,
        ``fun`` instead takes an ``(N, D)`` array and returns an ``(N,)`` or
        ``(N, 1)`` array, or a pair of those arrays for user-provided noise.
    D : int
        The number of dimensions that the function takes as input.
    noise_flag : bool
        Whether the function fun is stochastic or not.
    uncertainty_handling_level : {0, 1, 2}
        The uncertainty handling level which can be one of
        (0: none; 1: unknown noise level; 2: user-provided noise).
    cache_size : int, optional
        The initial size of caching table (default 500).
    parameter_transformer : ParameterTransformer, optional
        A ParameterTransformer is required to transform the parameters
        between constrained and unconstrained space, by default None.
    vectorized_target : bool, optional
        Whether ``fun`` accepts a two-dimensional batch and returns one value
        per row. Default ``False``.
    """

    def __init__(
        self,
        fun,
        D: int,
        noise_flag: bool,
        uncertainty_handling_level: int,
        cache_size: int = 500,
        parameter_transformer: ParameterTransformer = None,
        vectorized_target: bool = False,
    ):
        if not isinstance(vectorized_target, (bool, np.bool_)):
            raise ValueError("vectorized_target must be a boolean.")
        self.fun = fun
        self.D: int = D
        self.noise_flag: bool = noise_flag
        self.uncertainty_handling_level: int = uncertainty_handling_level
        self.transform_parameters = parameter_transformer is not None
        self.parameter_transformer = parameter_transformer
        self.vectorized_target = bool(vectorized_target)

        self.func_count: int = 0
        self.cache_count: int = 0
        self.X_orig = np.full([cache_size, self.D], np.nan)
        self.y_orig = np.full([cache_size, 1], np.nan)
        self.X = np.full([cache_size, self.D], np.nan)
        self.y = np.full([cache_size, 1], np.nan)
        self.y_max = np.nan
        self.n_evals = np.full([cache_size, 1], 0)

        if self.noise_flag:
            self.S = np.full([cache_size, 1], np.nan)

        self.Xn: int = -1  # Last filled entry
        # Use 1D array since this is a boolean mask.
        self.X_flag = np.full((cache_size,), False, dtype=bool)
        self.y_max = float("-Inf")
        self.fun_eval_time = np.full([cache_size, 1], np.nan)
        self.total_fun_eval_time = 0

    def __call__(self, x: np.ndarray):
        """
        Evaluates the function FUN at x and caches values.

        Parameters
        ----------
        x : np.ndarray
            The point at which the function will be evaluated. The shape of x
            should be (1, D) or (D,).

        Returns
        -------
        f_val : float
            The result of the evaluation.
        SD : float
            The (estimated) SD of the returned value.
        idx : int
            The index of the last updated entry.

        Raises
        ------
        ValueError
            If the input cannot be coerced to 1-D.
        ValueError
            Raise if the function value is not a finite real-valued scalar.
        ValueError
            Raise if the (estimated) SD (second function output)
            is not a finite, positive real-valued scalar.
        """

        if getattr(self, "vectorized_target", False):
            x_shape_orig = x.shape
            if x.ndim > 1:
                x = x.squeeze()
            if x.ndim == 0:
                x = np.atleast_1d(x)
            if x.size != x.shape[0]:
                raise ValueError(
                    "Input should be one-dimensional but has shape "
                    f"{x_shape_orig}."
                )
            values, sds, indices = self.batch_call(x.reshape(1, -1))
            sd = None if sds is None else sds[0]
            return values[0], sd, indices[0]

        timer = Timer()
        x_shape_orig = x.shape
        if x.ndim > 1:
            x = x.squeeze()
        if x.ndim == 0:
            x = np.atleast_1d(x)
        if x.size != x.shape[0]:
            raise ValueError(
                f"Input should be one-dimensional but has shape {x_shape_orig}."
            )
        # Convert back to original space
        if self.transform_parameters:
            x_orig = self.parameter_transformer.inverse(
                np.reshape(x, (1, x.shape[0]))
            )[0]
        else:
            x_orig = x

        timer.start_timer("fun_time")
        try:
            if self.noise_flag and self.uncertainty_handling_level == 2:
                f_val_orig, f_sd = self.fun(x_orig)
            else:
                f_val_orig = self.fun(x_orig)
                if self.noise_flag:
                    f_sd = 1
                else:
                    f_sd = None
            if isinstance(f_val_orig, np.ndarray):
                # f_val_orig can only be an array with size 1
                f_val_orig = f_val_orig.item()

        except Exception as err:
            err.args += (
                "FunctionLogger:FuncError "
                + "Error in executing the logged function"
                + "with input: "
                + str(x_orig),
            )
            raise
        timer.stop_timer("fun_time")

        # if f_val is an array with only one element, extract that element
        if not np.isscalar(f_val_orig) and np.size(f_val_orig) == 1:
            f_val_orig = np.array(f_val_orig).flat[0]

        # Check function value
        if (
            not np.isscalar(f_val_orig)
            or not np.isfinite(f_val_orig)
            or not np.isreal(f_val_orig)
        ):
            error_message = """FunctionLogger:InvalidFuncValue:
            The returned function value must be a finite real-valued scalar
            (returned value {})"""
            raise ValueError(error_message.format(str(f_val_orig)))

        # Check returned function SD
        if self.noise_flag and (
            not np.isscalar(f_sd)
            or not np.isfinite(f_sd)
            or not np.isreal(f_sd)
            or f_sd <= 0.0
        ):
            error_message = """FunctionLogger:InvalidNoiseValue
                The returned estimated SD (second function output)
                must be a finite, positive real-valued scalar
                (returned SD: {})"""
            raise ValueError(error_message.format(str(f_sd)))

        # record timer stats
        funtime = timer.get_duration("fun_time")

        self.func_count += 1
        f_val, idx = self._record(x_orig, x, f_val_orig, f_sd, funtime)

        # optimstate.N = self.Xn
        # optimstate.N_eff = np.sum(self.n_evals[self.X_flag])
        # optimState.totalfunevaltime = optimState.totalfunevaltime + t;
        return f_val, f_sd, idx

    def batch_call(self, x: np.ndarray, f_vals=None):
        """Evaluate and cache a batch of points in input order.

        Parameters
        ----------
        x : np.ndarray
            A nonempty array with shape ``(N, D)`` in transformed
            coordinates.
        f_vals : array-like, optional
            Already evaluated original-space values. It must have length
            ``N``; NaN rows are evaluated by the target.

        Returns
        -------
        values : np.ndarray
            The recorded values in transformed coordinates, with shape
            ``(N,)``.
        SDs : np.ndarray or None
            The target SD for each row when noise handling is enabled,
            otherwise ``None``.
        indices : np.ndarray
            Cache indices corresponding to the input rows.

        Raises
        ------
        RuntimeError
            If this logger was not created with ``vectorized_target=True``.
        ValueError
            If the input or any supplied or returned value has an invalid
            shape or value. Target outputs are fully validated before any
            row is recorded.
        """
        if not getattr(self, "vectorized_target", False):
            raise RuntimeError(
                "batch_call requires FunctionLogger(vectorized_target=True)."
            )

        x = np.asarray(x)
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != self.D:
            raise ValueError(
                f"Batch input must have nonempty shape (N, {self.D}), "
                f"but has shape {x.shape}."
            )
        n_rows = x.shape[0]
        x = _validate_batch_coordinates(x, n_rows, self.D, "Batch input")
        if f_vals is None:
            cached_values = np.full(n_rows, np.nan)
        else:
            cached_values = _validate_vectorized_output(
                f_vals,
                n_rows,
                "Cached function values",
                allow_nan=True,
            )

        if self.transform_parameters:
            x_orig = self.parameter_transformer.inverse(x)
        else:
            x_orig = x
        x_orig = _validate_batch_coordinates(
            x_orig, n_rows, self.D, "Inverse-transformed batch input"
        )

        missing = np.isnan(cached_values)
        n_missing = int(np.sum(missing))
        measured_values = np.empty(n_missing, dtype=float)
        measured_sds = (
            np.empty(n_missing, dtype=float) if self.noise_flag else None
        )
        time_per_row = np.nan
        if n_missing:
            timer = Timer()
            timer.start_timer("fun_time")
            target_x = x_orig[missing]
            try:
                output = self.fun(target_x)
            except Exception as err:
                err.args += (
                    "FunctionLogger:FuncError Error in executing the logged "
                    f"function for {n_missing} batch rows with input shape "
                    f"{target_x.shape}: {target_x}",
                )
                raise
            timer.stop_timer("fun_time")
            measured_values, measured_sds = _validate_vectorized_target_output(
                output, n_missing, self.uncertainty_handling_level
            )
            time_per_row = timer.get_duration("fun_time") / n_missing

        values = np.empty(n_rows, dtype=float)
        sds = np.empty(n_rows, dtype=float) if self.noise_flag else None
        indices = np.empty(n_rows, dtype=int)
        measured_idx = 0
        for row in range(n_rows):
            if missing[row]:
                f_sd = (
                    None
                    if measured_sds is None
                    else measured_sds[measured_idx]
                )
                self.func_count += 1
                f_val, idx = self._record(
                    x_orig[row],
                    x[row],
                    measured_values[measured_idx],
                    f_sd,
                    time_per_row,
                )
                measured_idx += 1
            else:
                f_val, f_sd, idx = self.add(x[row], cached_values[row])
            values[row] = np.asarray(f_val).item()
            if sds is not None:
                sds[row] = f_sd
            indices[row] = idx
        return values, sds, indices

    def add(
        self,
        x: np.ndarray,
        f_val_orig: float,
        f_sd: float = None,
        fun_eval_time=np.nan,
    ):
        """
        Add an previously evaluated function sample to the function cache.

        Parameters
        ----------
        x : np.ndarray
            The point at which the function has been evaluated. The shape of x
            should be (1, D) or (D,).
        f_val_orig : float
            The result of the evaluation of the function.
        f_sd : float, optional
            The (estimated) SD of the returned value (if heteroskedastic noise
            handling is on) of the evaluation of the function, by default None.
        fun_eval_time : float
            The duration of the time it took to evaluate the function,
            by default np.nan.

        Returns
        -------
        f_val : float
            The result of the evaluation.
        SD : float
            The (estimated) SD of the returned value.
        idx : int
            The index of the last updated entry.

        Raises
        ------
        ValueError
            If the input cannot be coerced to 1-D.
        ValueError
            Raise if the function value is not a finite real-valued scalar.
        ValueError
            Raise if the (estimated) SD (second function output)
            is not a finite, positive real-valued scalar.
        """
        x_shape_orig = x.shape
        if x.ndim > 1:
            x = x.squeeze()
        if x.ndim == 0:
            x = np.atleast_1d(x)
        if x.size != x.shape[0]:
            raise ValueError(
                f"Input should be one-dimensional but has shape {x_shape_orig}."
            )
        # Convert back to original space
        if self.transform_parameters:
            x_orig = self.parameter_transformer.inverse(
                np.reshape(x, (1, x.shape[0]))
            )[0]
        else:
            x_orig = x

        if self.noise_flag:
            if f_sd is None:
                f_sd = 1
        else:
            f_sd = None

        # Check function value
        if (
            not np.isscalar(f_val_orig)
            or not np.isfinite(f_val_orig)
            or not np.isreal(f_val_orig)
        ):
            error_message = """FunctionLogger:InvalidFuncValue:
            The returned function value must be a finite real-valued scalar
            (returned value {})"""
            raise ValueError(error_message.format(str(f_val_orig)))

        # Check returned function SD
        if self.noise_flag and (
            not np.isscalar(f_sd)
            or not np.isfinite(f_sd)
            or not np.isreal(f_sd)
            or f_sd <= 0.0
        ):
            error_message = """FunctionLogger:InvalidNoiseValue
                The returned estimated SD (second function output)
                must be a finite, positive real-valued scalar
                (returned SD:{})"""
            raise ValueError(error_message.format(str(f_sd)))

        self.cache_count += 1
        f_val, idx = self._record(x_orig, x, f_val_orig, f_sd, fun_eval_time)
        return f_val, f_sd, idx

    def finalize(self):
        """
        Remove unused caching entries.
        """
        self.X_orig = self.X_orig[: self.Xn + 1]
        self.y_orig = self.y_orig[: self.Xn + 1]

        # in the original matlab version X and Y get deleted
        self.X = self.X[: self.Xn + 1]
        self.y = self.y[: self.Xn + 1]

        if self.noise_flag:
            self.S = self.S[: self.Xn + 1]
        self.X_flag = self.X_flag[: self.Xn + 1]
        self.fun_eval_time = self.fun_eval_time[: self.Xn + 1]

    def _expand_arrays(self, resize_amount: int = None):
        """
        A private function to extend the rows of the object attribute arrays.

        Parameters
        ----------
        resize_amount : int, optional
            The number of additional rows, by default expand current table
            by 50%.
        """

        if resize_amount is None:
            resize_amount = int(np.max((np.ceil(self.Xn / 2), 1)))

        self.X_orig = np.append(
            self.X_orig, np.full([resize_amount, self.D], np.nan), axis=0
        )
        self.y_orig = np.append(
            self.y_orig, np.full([resize_amount, 1], np.nan), axis=0
        )
        self.X = np.append(
            self.X, np.full([resize_amount, self.D], np.nan), axis=0
        )
        self.y = np.append(self.y, np.full([resize_amount, 1], np.nan), axis=0)

        if self.noise_flag:
            self.S = np.append(
                self.S, np.full([resize_amount, 1], np.nan), axis=0
            )
        self.X_flag = np.append(
            self.X_flag, np.full((resize_amount,), False, dtype=bool)
        )
        self.fun_eval_time = np.append(
            self.fun_eval_time, np.full([resize_amount, 1], np.nan), axis=0
        )
        self.n_evals = np.append(
            self.n_evals, np.full([resize_amount, 1], 0), axis=0
        )

    def _record(
        self,
        x_orig: float,
        x: float,
        f_val_orig: float,
        f_sd: float,
        fun_eval_time: float,
    ):
        """
        A private method to save function values to class attributes.

        Parameters
        ----------
        x_orig : float
            The point at which the function has been evaluated
            (in original space).
        x : float
            The point at which the function has been evaluated
            (in transformed space).
        f_val_orig : float
            The result of the evaluation.
        f_sd : float
            The (estimated) SD of the returned value
            (if heteroskedastic noise handling is on).
        fun_eval_time : float
            The duration of the time it took to evaluate the function.

        Returns
        -------
        f_val : float
            The result of the evaluation.
        idx : int
            The index of the last updated entry.

        Raises
        ------
        ValueError
            Raise if there is more than one match for a duplicate entry.
        """
        duplicate_flag = (self.X == x).all(axis=1)
        if np.any(duplicate_flag):
            if np.sum(duplicate_flag) > 1:
                raise ValueError("More than one match for duplicate entry.")
            idx = np.argwhere(duplicate_flag)[0, 0]
            N = self.n_evals[idx]
            if f_sd is not None:
                tau_n = 1 / self.S[idx] ** 2
                tau_1 = 1 / f_sd**2
                self.y_orig[idx] = (
                    tau_n * self.y_orig[idx] + tau_1 * f_val_orig
                ) / (tau_n + tau_1)
                self.S[idx] = 1 / np.sqrt(tau_n + tau_1)
            else:
                self.y_orig[idx] = (N * self.y_orig[idx] + f_val_orig) / (
                    N + 1
                )

            f_val = self.y_orig[idx]
            if self.transform_parameters:
                f_val += self.parameter_transformer.log_abs_det_jacobian(x)
            self.y[idx] = f_val
            self.fun_eval_time[idx] = (
                N * self.fun_eval_time[idx] + fun_eval_time
            ) / (N + 1)
            self.n_evals[idx] += 1
            return f_val, idx
        else:
            self.Xn += 1
            if self.Xn > self.X_orig.shape[0] - 1:
                self._expand_arrays()

            # record function time
            if not np.isnan(fun_eval_time):
                self.fun_eval_time[self.Xn] = fun_eval_time
                self.total_fun_eval_time += fun_eval_time

            self.X_orig[self.Xn] = x_orig
            self.X[self.Xn] = x
            self.y_orig[self.Xn] = f_val_orig
            f_val = f_val_orig
            if self.transform_parameters:
                f_val += self.parameter_transformer.log_abs_det_jacobian(
                    np.reshape(x, (1, x.shape[0]))
                )[0]
            self.y[self.Xn] = f_val
            if f_sd is not None:
                self.S[self.Xn] = f_sd
            self.X_flag[self.Xn] = True
            self.n_evals[self.Xn] += 1
            self.y_max = np.nanmax(self.y[self.X_flag])
            return f_val, self.Xn

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)

        # Avoid infinite recursion in deepcopy
        memo[id(self)] = result
        # Copy class properties:
        for k, v in self.__dict__.items():
            if k == "fun":  # Avoid deepcopy of log-joint function
                # (interferes with benchmark logging)
                setattr(result, k, v)
            else:
                setattr(result, k, deepcopy(v, memo))
        return result

    def __str__(self, arr_size_thresh=10):
        """Print a string summary."""
        return "FunctionLogger:" + indent(
            f"""
function = {self.fun},
dimension = {self.D},
noisy = {self.noise_flag},
num. evaluations = {self.func_count},
y max = {self.y_max},
fun. eval. time = {self.total_fun_eval_time}""",
            "    ",
        )

    def __repr__(self, arr_size_thresh=10, expand=False):
        """Construct a detailed string summary.

        Parameters
        ----------
        arr_size_thresh : float, optional
            If ``obj`` is an array whose product of dimensions is less than
            ``arr_size_thresh``, print the full array. Otherwise print only the
            shape. Default `10`.
        expand : bool, optional
            If ``expand`` is `False`, then describe any complex child
            attributes of the object by their name and memory location.
            Otherwise, recursively expand the child attributes into their own
            representations. Default `False`.

        Returns
        -------
        string : str
            The string representation of ``self``.
        """
        return full_repr(
            self,
            "FunctionLogger",
            expand=expand,
            arr_size_thresh=arr_size_thresh,
        )
