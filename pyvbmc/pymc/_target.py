"""PyMC model adapter for PyVBMC."""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping
from numbers import Integral

import numpy as np

from pyvbmc.rng import get_rng
from pyvbmc.vbmc._bounds import _effective_bounds, _normalize_bounds

from . import _compat, _plausible

_PRIOR_DRAWS = 4000
_TRANSFORM_KEYS = {"log", "logodds", "interval", "interval_base"}
_LOGGER = logging.getLogger("pyvbmc.pymc")


def _compatibility_error(pm, capability, exc=None):
    """Build an actionable error for an undocumented PyMC API mismatch."""
    error = ImportError(
        f"PyMC {getattr(pm, '__version__', 'unknown')} does not provide "
        f"the {capability} required by PyVBMC. This adapter was tested "
        f"with {_compat.TESTED_RANGE}."
    )
    if exc is not None:
        error.__cause__ = exc
    return error


def _as_float64(value, label):
    """Return an array only when its numerical dtype is exactly float64."""
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float64):
        raise _compat.UnsupportedModel(
            f"{label} is {array.dtype}; PyMC targets require float64 "
            "free variables, value variables, and density derivatives."
        )
    return array


def _supported_transform_classes(pm):
    """Return the validated exact-type transform registry."""
    try:
        classes = _compat.transform_classes()
    except ImportError as exc:
        if _compat.TESTED_RANGE in str(exc):
            raise
        raise _compatibility_error(
            pm, "supported transform classes", exc
        ) from exc
    if set(classes) != _TRANSFORM_KEYS or any(
        not isinstance(cls, type) for cls in classes.values()
    ):
        raise _compatibility_error(pm, "supported transform classes")
    return classes


def _validate_source_transform_calls(model, classes, pm):
    """Validate transform call signatures before PyMC invokes them."""
    kind_by_type = {cls: kind for kind, cls in classes.items()}
    for rv in model.free_RVs:
        transform = model.rvs_to_transforms[rv]
        if transform is None:
            continue
        kind = kind_by_type.get(type(transform))
        if kind is None:
            raise _compat.UnsupportedModel(
                f"{rv.name}: unsupported transform "
                f"{type(transform).__name__}."
            )
        if kind in {"interval", "interval_base"}:
            args_fn = getattr(transform, "args_fn", None)
            if args_fn is None:
                raise _compatibility_error(pm, "interval transform args_fn")
            try:
                args_fn(*rv.owner.inputs)
            except TypeError as exc:
                raise _compat.UnsupportedModel(
                    f"{rv.name}: interval transform args_fn could not "
                    "be applied to its distribution inputs."
                ) from exc
            except (AttributeError, ValueError) as exc:
                raise _compatibility_error(
                    pm, "interval transform args_fn", exc
                ) from exc
        value_var = model.rvs_to_values[rv]
        for method_name in ("forward", "backward"):
            method = getattr(transform, method_name, None)
            if method is None:
                raise _compatibility_error(
                    pm, "transform forward/backward methods"
                )
            try:
                method(value_var, *rv.owner.inputs)
            except TypeError as exc:
                raise _compat.UnsupportedModel(
                    f"{rv.name}: {type(transform).__name__} could not be "
                    "applied to its distribution inputs."
                ) from exc
            except AttributeError as exc:
                raise _compatibility_error(
                    pm, "transform forward/backward methods", exc
                ) from exc


def _has_random_ancestor(expression):
    """Whether an interval limit depends on a random variable."""
    try:
        from pymc.distributions.distribution import SymbolicRandomVariable
        from pytensor.graph.traversal import ancestors
        from pytensor.tensor.random.op import RandomVariable
    except (AttributeError, ImportError) as exc:
        pm, _, _ = _compat.import_pymc()
        raise _compatibility_error(
            pm, "random-variable graph types", exc
        ) from exc

    return any(
        node.owner is not None
        and isinstance(node.owner.op, (RandomVariable, SymbolicRandomVariable))
        for node in ancestors([expression])
    )


def _interval_limit(rv, expression, shape, default):
    """Evaluate one fixed interval limit on the frozen model graph."""
    if expression is None:
        return np.full(shape, default, dtype=np.float64)
    if hasattr(expression, "eval"):
        if _has_random_ancestor(expression):
            raise _compat.UnsupportedModel(
                f"{rv.name}: an interval limit depends on another random "
                "variable, so it has no fixed value."
            )
        expression = expression.eval()
    try:
        return np.broadcast_to(
            np.asarray(expression, dtype=np.float64), shape
        ).copy()
    except (TypeError, ValueError) as exc:
        raise _compat.UnsupportedModel(
            f"{rv.name}: an interval limit cannot be broadcast to its shape."
        ) from exc


class PyMCTarget:
    """Represent a supported PyMC model as a flat PyVBMC target.

    The adapter snapshots the model inputs at construction, maps supported
    constrained variables to coordinates accepted by PyVBMC, and prepares a
    finite starting point and plausible box. Setup density evaluations are
    retained for reuse by :class:`pyvbmc.VBMC`.

    Parameters
    ----------
    model : pymc.Model
        Model whose free variables define the target, in ``model.free_RVs``
        order.
    plausible_bounds : mapping, optional
        Mapping from every free-variable name to ``(lower, upper)`` values in
        model coordinates. Values must broadcast to the variable shape and be
        strictly inside its support. By default a Laplace box with prior-width
        fallbacks is constructed.
    start : mapping, optional
        Starting point in model coordinates. It must contain every free
        variable and lie strictly inside its support. Supplying it skips the
        mode search.
    seed : None, int, SeedSequence or numpy.random.Generator, optional
        Random generator or seed for prior fallback draws and stochastic
        ``initval="prior"`` strategies. A generator is used as is. With
        ``None``, a generator is derived from NumPy's legacy global state.
    setup_budget : int, optional
        Positive function-equivalent setup allowance. The default is
        ``20 + 5 * D``. Actual work, including rejected search trials and a
        ``D``-unit Hessian evaluation, determines :attr:`setup_cost`.

    Attributes
    ----------
    D : int
        Dimension of the flat PyVBMC coordinates.
    names, value_names, coordinate_names : list of str
        Model variable names, PyVBMC value-variable names, and flattened
        coordinate labels.
    shapes : list of tuple
        Model-variable shapes in layout order.
    sizes : list of int
        Flattened sizes in layout order.
    kept : dict
        Model-variable names mapped to the class names of transforms retained
        in PyVBMC coordinates.
    support : dict
        Model-variable names mapped to ``(lower, upper)`` support arrays.
    x0, lb, ub, plb, pub : numpy.ndarray
        Float64 row vectors of shape ``(1, D)``.
    setup_evaluations : tuple of numpy.ndarray
        Finite setup observations ``(X, y)`` in call order.
    plausible_info : dict
        Setup route, diagnostics, and exact cost accounting.
    model : pymc.Model
        The original model, retained for posterior prediction. Numerical
        target calls use a private frozen snapshot.

    Raises
    ------
    ImportError
        If the PyMC extra is absent or the installed PyMC API is incompatible.
    UnsupportedModel
        If a free variable, transform, support, or dtype is unsupported.
    ValueError
        If setup arguments, bounds, or the final starting density are invalid.
    """

    def __init__(
        self,
        model,
        *,
        plausible_bounds=None,
        start=None,
        seed=None,
        setup_budget=None,
    ):
        pm, pytensor, pt = _compat.import_pymc()
        try:
            _compat.check_model(model)
        except ImportError as exc:
            if _compat.TESTED_RANGE in str(exc):
                raise
            raise _compatibility_error(pm, "required Model API", exc) from exc
        if getattr(pytensor.config, "floatX", None) != "float64":
            raise _compat.UnsupportedModel(
                f"PyTensor floatX is {pytensor.config.floatX}; PyMC targets "
                "require float64 evaluation."
            )

        for rv in model.free_RVs:
            if not np.issubdtype(np.dtype(rv.dtype), np.floating):
                raise _compat.UnsupportedModel(
                    f"{rv.name} is {rv.dtype}: VBMC needs continuous parameters."
                )
            if np.dtype(rv.dtype) != np.dtype(np.float64):
                raise _compat.UnsupportedModel(
                    f"{rv.name} is {rv.dtype}; PyMC targets require float64 "
                    "free variables."
                )
            try:
                value_var = model.rvs_to_values[rv]
            except KeyError as exc:
                raise _compatibility_error(
                    pm, f"value variable for {rv.name}", exc
                ) from exc
            if np.dtype(value_var.dtype) != np.dtype(np.float64):
                raise _compat.UnsupportedModel(
                    f"{value_var.name} is {value_var.dtype}; PyMC targets "
                    "require float64 value variables."
                )

        classes = _supported_transform_classes(pm)
        _validate_source_transform_calls(model, classes, pm)

        self.model = model
        self._rng = get_rng(seed)
        self._snapshot, original_names = _compat.snapshot_model(model)
        self.names = list(original_names)
        if not self.names:
            raise _compat.UnsupportedModel(
                "The PyMC model has no free variables for PyVBMC."
            )

        kind_by_type = {cls: kind for kind, cls in classes.items()}

        snapshot_by_name = {rv.name: rv for rv in self._snapshot.free_RVs}
        missing = [name for name in self.names if name not in snapshot_by_name]
        if missing:
            raise _compatibility_error(
                pm, f"snapshot free variables {', '.join(missing)}"
            )
        ordered_rvs = [snapshot_by_name[name] for name in self.names]

        prior_init = any(
            isinstance(value, str) and value == "prior"
            for value in self._snapshot.rvs_to_initial_values.values()
        )
        self._initial_seed = (
            int(self._rng.integers(0, 2**32, dtype=np.uint32))
            if prior_init
            else None
        )
        initial = self._snapshot.initial_point(random_seed=self._initial_seed)

        self.support = {}
        transform_kinds = {}
        untransform = []
        for rv in ordered_rvs:
            if not np.issubdtype(np.dtype(rv.dtype), np.floating):
                raise _compat.UnsupportedModel(
                    f"{rv.name} is {rv.dtype}: VBMC needs continuous parameters."
                )
            if np.dtype(rv.dtype) != np.dtype(np.float64):
                raise _compat.UnsupportedModel(
                    f"{rv.name} is {rv.dtype}; PyMC targets require float64 "
                    "free variables."
                )
            value_var = self._snapshot.rvs_to_values[rv]
            if np.dtype(value_var.dtype) != np.dtype(np.float64):
                raise _compat.UnsupportedModel(
                    f"{value_var.name} is {value_var.dtype}; PyMC targets "
                    "require float64 value variables."
                )
            if value_var.name not in initial:
                raise _compatibility_error(
                    pm, f"initial value for {value_var.name}"
                )
            shape = _as_float64(initial[value_var.name], value_var.name).shape
            transform = self._snapshot.rvs_to_transforms[rv]
            try:
                default = _compat.default_transform(rv.owner.op, rv)
            except ImportError as exc:
                if _compat.TESTED_RANGE in str(exc):
                    raise
                raise _compatibility_error(
                    pm, "default-transform registry", exc
                ) from exc

            if transform is None:
                if default is not None:
                    raise _compat.UnsupportedModel(
                        f"{rv.name}: its {type(default).__name__} was "
                        "suppressed at construction."
                    )
                if not _compat.is_known_real_line(rv.owner.op):
                    raise _compat.UnsupportedModel(
                        f"{rv.name}: support cannot be inferred for "
                        f"{type(rv.owner.op).__name__} without a transform."
                    )
                kind = None
                lower = np.full(shape, -np.inf, dtype=np.float64)
                upper = np.full(shape, np.inf, dtype=np.float64)
            else:
                kind = kind_by_type.get(type(transform))
                if kind is None:
                    raise _compat.UnsupportedModel(
                        f"{rv.name}: unsupported transform "
                        f"{type(transform).__name__}."
                    )
                if kind == "log":
                    lower = np.zeros(shape, dtype=np.float64)
                    upper = np.full(shape, np.inf, dtype=np.float64)
                elif kind == "logodds":
                    lower = np.zeros(shape, dtype=np.float64)
                    upper = np.ones(shape, dtype=np.float64)
                else:
                    args_fn = getattr(transform, "args_fn", None)
                    if args_fn is None:
                        raise _compatibility_error(
                            pm, "interval transform args_fn"
                        )
                    try:
                        lower_expr, upper_expr = args_fn(*rv.owner.inputs)
                    except TypeError as exc:
                        raise _compat.UnsupportedModel(
                            f"{rv.name}: interval transform args_fn could not "
                            "be applied to its distribution inputs."
                        ) from exc
                    except (AttributeError, ValueError) as exc:
                        raise _compatibility_error(
                            pm, "interval transform args_fn", exc
                        ) from exc
                    lower = _interval_limit(rv, lower_expr, shape, -np.inf)
                    upper = _interval_limit(rv, upper_expr, shape, np.inf)
            if np.any(lower >= upper):
                raise _compat.UnsupportedModel(
                    f"{rv.name}: support bounds are not strictly ordered."
                )
            self.support[rv.name] = (lower, upper)
            transform_kinds[rv.name] = kind

            finite_lower = np.isfinite(lower)
            finite_upper = np.isfinite(upper)
            two_sided = finite_lower & finite_upper
            one_sided = finite_lower ^ finite_upper
            if transform is None or np.all(two_sided):
                untransform.append(rv)
            elif not np.all(one_sided):
                raise _compat.UnsupportedModel(
                    f"{rv.name}: its coordinates mix one-sided and other "
                    "bounds."
                )

        try:
            self._partial = _compat.remove_value_transforms(
                self._snapshot, vars=untransform
            )
        except ImportError as exc:
            if _compat.TESTED_RANGE in str(exc):
                raise
            raise _compatibility_error(
                pm, "value-transform removal", exc
            ) from exc

        partial_by_name = {rv.name: rv for rv in self._partial.free_RVs}
        missing = [name for name in self.names if name not in partial_by_name]
        if missing:
            raise _compatibility_error(
                pm, f"partially untransformed variables {', '.join(missing)}"
            )
        self._ordered_rvs = [partial_by_name[name] for name in self.names]
        self.value_names = [
            self._partial.rvs_to_values[rv].name for rv in self._ordered_rvs
        ]
        point = self._partial.initial_point(random_seed=self._initial_seed)
        missing = [name for name in self.value_names if name not in point]
        if missing:
            raise _compatibility_error(
                pm, f"initial values for {', '.join(missing)}"
            )
        self.shapes = [
            _as_float64(point[name], name).shape for name in self.value_names
        ]
        self.sizes = [int(np.prod(shape)) for shape in self.shapes]
        self.D = int(sum(self.sizes))
        self._offsets = np.concatenate(
            ([0], np.cumsum(self.sizes, dtype=int))
        ).astype(int)

        self.kept = {}
        self._maps = {}
        for name, rv, value_name, shape in zip(
            self.names, self._ordered_rvs, self.value_names, self.shapes
        ):
            transform = self._partial.rvs_to_transforms[rv]
            if transform is None:
                continue
            kind = kind_by_type.get(type(transform))
            if kind is None:
                raise _compat.UnsupportedModel(
                    f"{name}: unsupported transform {type(transform).__name__}."
                )
            input_var = pt.TensorType(
                "float64", shape=(None,) * (len(shape) + 1)
            )(f"{name}_pyvbmc_value")
            try:
                backward_expr = transform.backward(input_var, *rv.owner.inputs)
                forward_expr = transform.forward(
                    backward_expr, *rv.owner.inputs
                )
            except TypeError as exc:
                raise _compat.UnsupportedModel(
                    f"{name}: {type(transform).__name__} could not be applied "
                    "to its distribution inputs."
                ) from exc
            except AttributeError as exc:
                raise _compatibility_error(
                    pm, "transform forward/backward methods", exc
                ) from exc
            if (
                backward_expr.dtype != "float64"
                or forward_expr.dtype != "float64"
            ):
                raise _compat.UnsupportedModel(
                    f"{name}: transform maps must evaluate in float64."
                )
            backward = pytensor.function([input_var], backward_expr)
            try:
                forward_graph = transform.forward(input_var, *rv.owner.inputs)
            except TypeError as exc:
                raise _compat.UnsupportedModel(
                    f"{name}: {type(transform).__name__} could not be applied "
                    "to its distribution inputs."
                ) from exc
            except AttributeError as exc:
                raise _compatibility_error(
                    pm, "transform forward/backward methods", exc
                ) from exc
            forward = pytensor.function([input_var], forward_graph)
            probe = _as_float64(
                point[value_name], f"initial value for {value_name}"
            ).reshape((1, *shape))
            observed = _as_float64(backward(probe), f"{name} backward map")
            expected = _compat.closed_form(
                kind,
                probe,
                self.support[name][0],
                self.support[name][1],
            )
            round_trip = _as_float64(forward(observed), f"{name} forward map")
            if not np.allclose(
                observed, expected, rtol=1e-10, atol=1e-10
            ) or not np.allclose(round_trip, probe, rtol=1e-10, atol=1e-10):
                raise _compatibility_error(
                    pm, f"closed-form check for {type(transform).__name__}"
                )
            self.kept[name] = type(transform).__name__
            self._maps[name] = {
                "kind": kind,
                "forward": forward,
                "backward": backward,
            }

        lower_parts = []
        upper_parts = []
        for name, size in zip(self.names, self.sizes):
            if name in self.kept:
                lower_parts.append(np.full(size, -np.inf, dtype=np.float64))
                upper_parts.append(np.full(size, np.inf, dtype=np.float64))
            else:
                lower_parts.append(self.support[name][0].reshape(-1))
                upper_parts.append(self.support[name][1].reshape(-1))
        self.lb = np.concatenate(lower_parts).reshape(1, self.D)
        self.ub = np.concatenate(upper_parts).reshape(1, self.D)

        self._capture_export_metadata()
        self._compile_density(pm)

        start_values = None
        if start is not None:
            start_values = self._validate_model_point(start, "start")
            x_initial = self._from_validated_model_variables(start_values)
        else:
            x_initial = self.flatten(point)

        explicit_plausible = None
        if plausible_bounds is not None:
            explicit_plausible = self._validate_plausible_bounds(
                plausible_bounds
            )

        need_derivatives = start is None or plausible_bounds is None
        gradient_fn = None
        hessian_fn = None
        if need_derivatives:
            gradient_fn = self._compile_gradient(pm)
        if gradient_fn is not None and plausible_bounds is None:
            hessian_fn = self._compile_hessian(pm)

        has_gradient = gradient_fn is not None
        search_requested = start is None and has_gradient
        hessian_cost = self.D if hessian_fn is not None else 0
        minimum = hessian_cost + (2 if search_requested else 1)
        if setup_budget is None:
            setup_budget = 20 + 5 * self.D
        elif (
            isinstance(setup_budget, (bool, np.bool_))
            or not isinstance(setup_budget, Integral)
            or setup_budget < 1
        ):
            raise ValueError("setup_budget must be a positive integer.")
        self.setup_budget = int(setup_budget)
        if self.setup_budget < minimum:
            raise ValueError(
                "setup_budget is insufficient for the selected setup route: "
                f"at least {minimum} function-equivalent units are required."
            )

        start_moved = np.zeros(self.D, dtype=bool)
        if start is None:
            x_start, moved = _plausible.move_inside(
                x_initial, self.lb, self.ub
            )
            start_moved |= moved
        else:
            x_start = self._normalize_explicit_start(x_initial)

        X_setup = np.empty((0, self.D), dtype=np.float64)
        y_setup = np.empty(0, dtype=np.float64)
        n_target_calls = 0
        cap_reached = False
        mode = None
        if search_requested:
            search_lower, search_upper = self._search_bounds()
            max_search = min(
                19 + 4 * self.D,
                self.setup_budget - hessian_cost - 1,
            )

            def value_and_grad(x):
                value, gradient = gradient_fn(self.unflatten(x))
                value_array = _as_float64(value, "compiled log density")
                gradient_array = _as_float64(
                    gradient, "compiled log-density gradient"
                )
                if value_array.size != 1 or gradient_array.size != self.D:
                    raise _compatibility_error(
                        pm, "ordered scalar density and gradient outputs"
                    )
                return float(
                    value_array.reshape(-1)[0]
                ), gradient_array.reshape(-1)

            try:
                (
                    x0,
                    X_setup,
                    y_setup,
                    n_target_calls,
                    cap_reached,
                ) = _plausible.search_mode(
                    value_and_grad,
                    x_start,
                    search_lower,
                    search_upper,
                    max_search,
                )
            except _compat.UnsupportedModel:
                raise
            except ValueError as exc:
                raise ValueError(
                    f"Mode search failed for model variables "
                    f"{self._describe_point(x_start)}: {exc}"
                ) from exc
            boundary = np.isfinite(search_lower) & (
                (x0 <= search_lower) | (x0 >= search_upper)
            )
            start_moved |= boundary
            mode = self._single_model_point(x0)
            start_kind = "mode"
        else:
            x0 = x_start
            start_kind = "user" if start is not None else "initial"

        hessian_stats = {"calls": 0}

        def evaluate_hessian():
            if hessian_fn is None:
                return None
            hessian_stats["calls"] += 1
            value = _as_float64(
                hessian_fn(self.unflatten(x0)), "compiled log-density Hessian"
            )
            if value.shape != (self.D, self.D):
                raise _compatibility_error(pm, "ordered Hessian output")
            return value

        prior_cache = []

        def prior_draws():
            if not prior_cache:
                prior_cache.append(self._prior_draws())
            return prior_cache[0]

        curvature_mask = np.zeros(self.D, dtype=bool)
        location_mask = np.zeros(self.D, dtype=bool)
        clipped_mask = np.zeros(self.D, dtype=bool)
        if explicit_plausible is not None:
            plb, pub = explicit_plausible
            route = "explicit"
        elif has_gradient:
            (
                x0,
                plb,
                pub,
                curvature_mask,
                location_mask,
                clipped_mask,
            ) = _plausible.laplace_box(
                x0,
                evaluate_hessian,
                self.lb,
                self.ub,
                prior_draws,
                k=3.0,
                check_location=start is None,
            )
            route = "laplace"
        else:
            plb, pub = _plausible.quantile_box(prior_draws())
            plb, pub, clipped_mask = _plausible.clip_inside(
                plb, pub, self.lb, self.ub, x0
            )
            route = "prior"

        (
            self.x0,
            self.lb,
            self.ub,
            self.plb,
            self.pub,
        ) = _normalize_bounds(
            np.asarray(x0, dtype=np.float64).reshape(1, self.D),
            self.lb,
            self.ub,
            np.asarray(plb, dtype=np.float64).reshape(1, self.D),
            np.asarray(pub, dtype=np.float64).reshape(1, self.D),
            logger=_LOGGER,
        )

        exact = np.all(X_setup == self.x0, axis=1) if len(X_setup) else []
        if np.any(exact):
            final_value = float(y_setup[np.flatnonzero(exact)[-1]])
        else:
            final_value = self.log_joint(self.x0)
            n_target_calls += 1
            X_setup = np.vstack((X_setup, self.x0))
            y_setup = np.append(y_setup, final_value)
        if not np.isfinite(final_value):
            raise ValueError(
                f"The starting log density is non-finite for model variables "
                f"{self._describe_point(self.x0)}."
            )

        n_hessian_calls = hessian_stats["calls"]
        actual_hessian_cost = n_hessian_calls * self.D
        n_evaluations = n_target_calls + actual_hessian_cost
        if n_evaluations > self.setup_budget:
            raise RuntimeError("PyMC target setup exceeded setup_budget.")

        self.setup_evaluations = (
            np.asarray(X_setup, dtype=np.float64).reshape(-1, self.D),
            np.asarray(y_setup, dtype=np.float64).reshape(-1),
        )
        self.plausible_info = {
            "route": route,
            "start": start_kind,
            "mode": mode,
            "n_evaluations": int(n_evaluations),
            "n_target_calls": int(n_target_calls),
            "n_hessian_calls": int(n_hessian_calls),
            "hessian_cost": int(actual_hessian_cost),
            "cap_reached": bool(cap_reached),
            "start_moved": self._coordinate_mask_names(start_moved),
            "curvature": self._coordinate_mask_names(curvature_mask),
            "location_outside_prior": self._coordinate_mask_names(
                location_mask
            ),
            "clipped": self._coordinate_mask_names(clipped_mask),
        }
        self._report_setup_warnings()

    @property
    def setup_cost(self):
        """Actual function-equivalent setup cost."""
        return self.plausible_info["n_evaluations"]

    def _capture_export_metadata(self):
        """Copy variable dimensions, coordinates, and flat labels."""
        source_dims = getattr(self._snapshot, "named_vars_to_dims", {})
        source_coords = getattr(self._snapshot, "coords", {})
        reserved_dims = {
            "chain",
            "draw",
            *self.names,
            *source_coords.keys(),
        }
        for declared in source_dims.values():
            reserved_dims.update(dim for dim in declared if dim is not None)
        self.dims = {}
        self.coords = {}
        coordinate_names = []
        for name, value_name, shape in zip(
            self.names, self.value_names, self.shapes
        ):
            raw_dims = tuple(source_dims.get(name, ()))
            if raw_dims and len(raw_dims) != len(shape):
                pm, _, _ = _compat.import_pymc()
                raise _compatibility_error(
                    pm, f"dimension metadata for {name}"
                )
            if not raw_dims:
                raw_dims = (None,) * len(shape)
            event_dims = []
            for axis, dim in enumerate(raw_dims):
                if dim is not None:
                    event_dims.append(dim)
                    continue
                base = f"{name}_dim_{axis}"
                generated = base
                suffix = 1
                while generated in reserved_dims:
                    generated = f"{base}_{suffix}"
                    suffix += 1
                reserved_dims.add(generated)
                event_dims.append(generated)
            event_dims = tuple(event_dims)
            self.dims[name] = event_dims
            for dim in raw_dims:
                if (
                    dim is not None
                    and dim in source_coords
                    and source_coords[dim] is not None
                ):
                    self.coords[dim] = np.array(source_coords[dim], copy=True)
            if not shape:
                coordinate_names.append(value_name)
                continue
            for index in np.ndindex(shape):
                labels = []
                for axis, item in enumerate(index):
                    dim = raw_dims[axis]
                    if dim is not None and dim in self.coords:
                        labels.append(str(self.coords[dim][item]))
                    else:
                        labels.append(str(item))
                coordinate_names.append(f"{value_name}[{','.join(labels)}]")
        self.coordinate_names = coordinate_names

        from pyvbmc.variational_posterior._arviz import _structured_layout

        _structured_layout(
            self.D,
            dict(zip(self.names, self.shapes)),
            self.dims,
            self.coords,
        )

    def _compile_density(self, pm):
        """Compile and validate the two repeatedly evaluated densities."""
        logp_graph = self._partial.logp(jacobian=True)
        plain_graph = self._partial.logp(jacobian=False)
        if logp_graph.dtype != "float64" or plain_graph.dtype != "float64":
            raise _compat.UnsupportedModel(
                "Compiled PyMC log densities must have float64 outputs."
            )
        self._logp = self._partial.compile_logp(jacobian=True)
        self._logp_plain = self._partial.compile_logp(jacobian=False)

    def _compile_gradient(self, pm):
        """Compile the ordered joint value/gradient, or return None."""
        no_derivative = _compat.missing_derivative_errors()

        try:
            logp = self._partial.logp(jacobian=True)
            gradient = self._partial.dlogp(
                vars=self._ordered_rvs, jacobian=True
            )
            if logp.dtype != "float64" or gradient.dtype != "float64":
                raise _compat.UnsupportedModel(
                    "Compiled PyMC density and gradient must be float64."
                )
            return self._partial.compile_fn(
                [logp, gradient],
                inputs=self._partial.value_vars,
                on_unused_input="ignore",
            )
        except no_derivative:
            return None

    def _compile_hessian(self, pm):
        """Compile the ordered exact Hessian, or return None."""
        no_derivative = _compat.missing_derivative_errors()

        try:
            hessian = self._partial.d2logp(
                vars=self._ordered_rvs,
                jacobian=True,
                negate_output=False,
            )
            if hessian.dtype != "float64":
                raise _compat.UnsupportedModel(
                    "Compiled PyMC Hessian must be float64."
                )
            return self._partial.compile_fn(
                hessian,
                inputs=self._partial.value_vars,
                on_unused_input="ignore",
                mode="FAST_COMPILE",
            )
        except no_derivative:
            return None

    def _validate_model_mapping(self, values, parameter):
        if not isinstance(values, Mapping):
            raise ValueError(
                f"{parameter} must be a mapping over model variables."
            )
        missing = [name for name in self.names if name not in values]
        unknown = [name for name in values if name not in self.names]
        if missing:
            raise ValueError(
                f"{parameter} is missing model variable {missing[0]!r}."
            )
        if unknown:
            raise ValueError(
                f"{parameter} contains unknown model variable {unknown[0]!r}."
            )

    def _validate_model_point(self, values, parameter):
        self._validate_model_mapping(values, parameter)
        result = {}
        for name, shape in zip(self.names, self.shapes):
            try:
                value = np.broadcast_to(
                    np.asarray(values[name], dtype=np.float64), shape
                ).copy()
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{parameter} value for {name!r} must match shape {shape}."
                ) from exc
            lower, upper = self.support[name]
            if (
                np.any(~np.isfinite(value))
                or np.any(value <= lower)
                or np.any(value >= upper)
            ):
                raise ValueError(
                    f"{parameter} value for {name!r} must be finite and "
                    "strictly inside its support."
                )
            result[name] = value
        return result

    def _validate_plausible_bounds(self, bounds):
        self._validate_model_mapping(bounds, "plausible_bounds")
        lower_values = {}
        upper_values = {}
        for name, shape in zip(self.names, self.shapes):
            try:
                pair = tuple(bounds[name])
            except TypeError as exc:
                raise ValueError(
                    f"plausible_bounds for {name!r} must be a lower/upper pair."
                ) from exc
            if len(pair) != 2:
                raise ValueError(
                    f"plausible_bounds for {name!r} must be a lower/upper pair."
                )
            try:
                lower = np.broadcast_to(
                    np.asarray(pair[0], dtype=np.float64), shape
                ).copy()
                upper = np.broadcast_to(
                    np.asarray(pair[1], dtype=np.float64), shape
                ).copy()
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"plausible_bounds for {name!r} must match shape {shape}."
                ) from exc
            support_lower, support_upper = self.support[name]
            if (
                np.any(~np.isfinite(lower))
                or np.any(~np.isfinite(upper))
                or np.any(lower >= upper)
                or np.any(lower <= support_lower)
                or np.any(upper >= support_upper)
            ):
                raise ValueError(
                    f"plausible_bounds for {name!r} must be finite, ordered, "
                    "and strictly inside its support."
                )
            lower_values[name] = lower
            upper_values[name] = upper
        lower = self._from_validated_model_variables(lower_values)
        upper = self._from_validated_model_variables(upper_values)
        return np.minimum(lower, upper), np.maximum(lower, upper)

    def _normalize_explicit_start(self, x):
        """Apply VBMC's numerical hard-bound margin before curvature."""
        lower, upper = _effective_bounds(self.lb, self.ub)
        normalized = np.maximum(np.minimum(x, upper.ravel()), lower.ravel())
        if np.any(normalized != x):
            _LOGGER.warning(
                "vbmc:InitialPointsTooClosePB: The starting points X0 are "
                "on or numerically too close to the hard bounds LB and UB. "
                "Moving the initial points more inside..."
            )
        return normalized

    def _search_bounds(self):
        lower = self.lb.ravel().copy()
        upper = self.ub.ravel().copy()
        finite = np.isfinite(lower) & np.isfinite(upper)
        width = upper[finite] - lower[finite]
        lower[finite] += 2e-3 * width
        upper[finite] -= 2e-3 * width
        return lower, upper

    def _prior_draws(self):
        """Draw and flatten the frozen model's free-variable prior."""
        pm, _, _ = _compat.import_pymc()
        by_name = {rv.name: rv for rv in self._snapshot.free_RVs}
        variables = [by_name[name] for name in self.names]
        draws = pm.draw(
            variables,
            draws=_PRIOR_DRAWS,
            random_seed=self._rng,
        )
        if len(self.names) == 1 and not isinstance(draws, (list, tuple)):
            draws = [draws]
        original = {
            name: np.asarray(value, dtype=np.float64)
            for name, value in zip(self.names, draws)
        }
        return self._flat_model_draws(original)

    def _flat_model_draws(self, values):
        n_draws = len(np.asarray(values[self.names[0]]))
        blocks = []
        for name, value_name, shape in zip(
            self.names, self.value_names, self.shapes
        ):
            value = np.asarray(values[name], dtype=np.float64).reshape(
                (n_draws, *shape)
            )
            if name in self._maps:
                value = _as_float64(
                    self._maps[name]["forward"](value),
                    f"{name} forward map",
                )
            blocks.append(value.reshape(n_draws, -1))
        return np.column_stack(blocks).astype(np.float64, copy=False)

    def _from_validated_model_variables(self, values):
        value_point = {}
        for name, value_name in zip(self.names, self.value_names):
            value = np.asarray(values[name], dtype=np.float64)
            if name in self._maps:
                value = _as_float64(
                    self._maps[name]["forward"](value[None]),
                    f"{name} forward map",
                )[0]
            value_point[value_name] = value
        return self.flatten(value_point)

    def flatten(self, values):
        """Flatten a mapping over value-variable names in C order.

        Parameters
        ----------
        values : mapping
            One array of the declared shape for every name in `value_names`.

        Returns
        -------
        x : numpy.ndarray
            Float64 vector of shape ``(D,)``.
        """
        if not isinstance(values, Mapping):
            raise ValueError("values must be a mapping over value_names.")
        missing = [name for name in self.value_names if name not in values]
        unknown = [name for name in values if name not in self.value_names]
        if missing or unknown:
            detail = missing[0] if missing else unknown[0]
            raise ValueError(f"values do not match value_names: {detail!r}.")
        blocks = []
        for name, shape in zip(self.value_names, self.shapes):
            value = np.asarray(values[name], dtype=np.float64)
            if value.shape != shape:
                raise ValueError(
                    f"values entry {name!r} must have shape {shape}."
                )
            blocks.append(value.reshape(-1))
        return np.concatenate(blocks).astype(np.float64, copy=False)

    def unflatten(self, x):
        """Convert one flat PyVBMC point to a value-variable mapping."""
        vector = self._point_vector(x)
        return {
            name: vector[self._offsets[i] : self._offsets[i + 1]].reshape(
                shape
            )
            for i, (name, shape) in enumerate(
                zip(self.value_names, self.shapes)
            )
        }

    def _point_vector(self, x):
        array = np.asarray(x, dtype=np.float64)
        if array.shape == (1, self.D):
            array = array[0]
        elif array.shape != (self.D,):
            raise ValueError(
                f"x must have shape ({self.D},) or (1, {self.D})."
            )
        return array

    def to_model_variables(self, X):
        """Map flat PyVBMC draws to arrays over the model's variables.

        Parameters
        ----------
        X : numpy.ndarray
            Float64-compatible array of shape ``(n, D)``.

        Returns
        -------
        values : dict
            Variable names mapped to arrays of shape ``(n, *shape)``.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.D:
            raise ValueError(f"X must have shape (n, {self.D}).")
        result = {}
        for i, (name, shape) in enumerate(zip(self.names, self.shapes)):
            value = X[:, self._offsets[i] : self._offsets[i + 1]].reshape(
                (len(X), *shape)
            )
            if name in self._maps:
                value = _as_float64(
                    self._maps[name]["backward"](value),
                    f"{name} backward map",
                )
            lower, upper = self.support[name]
            if (
                np.any(~np.isfinite(value))
                or np.any(value <= lower)
                or np.any(value >= upper)
            ):
                raise ValueError(
                    f"Mapped model values for {name!r} must be finite and "
                    "strictly inside its support."
                )
            result[name] = value
        return result

    def from_model_variables(self, point):
        """Map one model-variable point to a flat PyVBMC vector."""
        values = self._validate_model_point(point, "point")
        return self._from_validated_model_variables(values)

    def _single_model_point(self, x):
        values = self.to_model_variables(self._point_vector(x)[None])
        return {name: value[0].copy() for name, value in values.items()}

    def _describe_point(self, x):
        values = self._single_model_point(x)
        return ", ".join(
            f"{name}={np.asarray(value).tolist()}"
            for name, value in values.items()
        )

    def _evaluate_density(self, x, compiled, label):
        vector = self._point_vector(x)
        if np.any(vector <= self.lb.ravel()) or np.any(
            vector >= self.ub.ravel()
        ):
            return -np.inf
        value = _as_float64(compiled(self.unflatten(vector)), label)
        if value.size != 1:
            pm, _, _ = _compat.import_pymc()
            raise _compatibility_error(pm, "scalar compiled log density")
        scalar = float(value.reshape(-1)[0])
        if not np.isfinite(scalar):
            raise ValueError(
                "Non-finite log density for model variables "
                f"{self._describe_point(vector)}."
            )
        return scalar

    def log_joint(self, x):
        """Evaluate the model log joint, including retained Jacobians.

        A point on or outside a hard bound returns ``-inf``. A non-finite
        density strictly inside the box raises a `ValueError` that identifies
        the model variables and their values.
        """
        return self._evaluate_density(x, self._logp, "compiled log density")

    def log_joint_no_jacobian(self, x):
        """Evaluate the model-coordinate log density without Jacobians."""
        return self._evaluate_density(
            x, self._logp_plain, "compiled log density without Jacobian"
        )

    def _coordinate_mask_names(self, mask):
        mask = np.asarray(mask, dtype=bool).reshape(self.D)
        return [
            name
            for name, selected in zip(self.coordinate_names, mask)
            if selected
        ]

    def _report_setup_warnings(self):
        info = self.plausible_info
        if info["start_moved"]:
            _LOGGER.warning(
                "Moved automatic start coordinates inside hard bounds: %s",
                ", ".join(info["start_moved"]),
            )
        if info["curvature"]:
            _LOGGER.warning(
                "Used prior widths for coordinates with unusable curvature: %s",
                ", ".join(info["curvature"]),
            )
        if info["location_outside_prior"]:
            _LOGGER.warning(
                "Mode lies outside the prior location interval for: %s",
                ", ".join(info["location_outside_prior"]),
            )
        if info["clipped"]:
            _LOGGER.warning(
                "Clipped automatic plausible bounds for: %s",
                ", ".join(info["clipped"]),
            )
        if info["cap_reached"]:
            _LOGGER.warning("Mode search reached its setup call cap.")
        if info["route"] == "prior":
            origin = (
                "the supplied starting point"
                if info["start"] == "user"
                else "the model's initial point"
            )
            _LOGGER.warning(
                "The model gradient is unavailable; using %s and "
                "prior-quantile plausible bounds.",
                origin,
            )

    def to_arviz(self, vp, n_samples=1000):
        """Export variational draws in the PyMC model's coordinates.

        Parameters
        ----------
        vp : VariationalPosterior
            Posterior fitted with this target's frozen model and data, in
            its flat coordinate layout. Only dimension compatibility is
            checked. After loading a run, use
            ``loaded.target.to_arviz(loaded.vp)``.
        n_samples : int, optional
            Positive number of independent draws, default 1000.

        Returns
        -------
        data : xarray.DataTree
            One ``posterior`` group with the model's variable names, shapes,
            dimensions, and coordinates.

        Raises
        ------
        ImportError
            If the ArviZ data API is unavailable.
        ValueError
            If the sample count, posterior dimension, or stored layout is
            invalid. Validation errors leave ``vp.rng`` unchanged.
        """
        if (
            isinstance(n_samples, (bool, np.bool_))
            or not isinstance(n_samples, Integral)
            or n_samples < 1
        ):
            raise ValueError("n_samples must be a positive integer.")
        if not hasattr(vp, "D") or vp.D != self.D:
            raise ValueError(
                f"vp.D must equal target.D ({self.D}) for PyMC export."
            )

        from pyvbmc.variational_posterior._arviz import (
            _structured_layout,
            datatree_from_arrays,
            require_from_dict,
        )

        _structured_layout(
            self.D,
            dict(zip(self.names, self.shapes)),
            self.dims,
            self.coords,
        )
        require_from_dict()
        rng = getattr(vp, "rng", None)
        rng_state = (
            copy.deepcopy(rng.bit_generator.state)
            if rng is not None and hasattr(rng, "bit_generator")
            else None
        )
        try:
            samples, _ = vp.sample(n_samples, orig_flag=True)
            values = self.to_model_variables(samples)
            arrays = {name: value[None] for name, value in values.items()}
            return datatree_from_arrays(
                arrays,
                dims=self.dims,
                coords=self.coords,
                attrs={
                    "inference_library": "pyvbmc",
                    "inference_method": "variational approximation",
                    "sample_type": "independent",
                    "parameter_space": "model",
                },
            )
        except Exception:
            if rng_state is not None:
                rng.bit_generator.state = rng_state
            raise

    def __str__(self):
        """Return a table describing the adapter's flat coordinates."""
        rows = [
            "variable | shape | VBMC coordinates | kept transform | hard bounds | plausible bounds | start"
        ]
        for i, (name, shape) in enumerate(zip(self.names, self.shapes)):
            start, stop = self._offsets[i], self._offsets[i + 1]
            rows.append(
                f"{name} | {shape} | "
                f"{', '.join(self.coordinate_names[start:stop])} | "
                f"{self.kept.get(name, '-')} | "
                f"[{self.lb.ravel()[start:stop]}, {self.ub.ravel()[start:stop]}] | "
                f"[{self.plb.ravel()[start:stop]}, {self.pub.ravel()[start:stop]}] | "
                f"{self.x0.ravel()[start:stop]}"
            )
        return "\n".join(rows)

    def __deepcopy__(self, memo):
        """Copy models safely while preserving compiled numerical callables."""
        copied = type(self).__new__(type(self))
        memo[id(self)] = copied
        shared = {
            "model",
            "_snapshot",
            "_partial",
            "_ordered_rvs",
            "_maps",
            "_logp",
            "_logp_plain",
        }
        for name, value in self.__dict__.items():
            setattr(
                copied,
                name,
                value if name in shared else copy.deepcopy(value, memo),
            )
        copied.model = _compat.clone_model(self.model)
        copied._snapshot = _compat.clone_model(self._snapshot)
        copied._partial = _compat.clone_model(self._partial)
        partial_by_name = {rv.name: rv for rv in copied._partial.free_RVs}
        copied._ordered_rvs = [partial_by_name[name] for name in copied.names]
        return copied


UnsupportedModel = _compat.UnsupportedModel

__all__ = ["PyMCTarget", "UnsupportedModel"]
