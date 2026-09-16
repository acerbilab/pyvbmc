"""Private helpers for exporting posterior arrays to ArviZ."""

import sys
from collections.abc import Mapping
from math import prod
from numbers import Integral

import numpy as np


def _structured_layout(D, variables, dims, coords):
    """Validate and normalize a structured posterior variable layout."""
    if not isinstance(variables, Mapping):
        raise ValueError("variables must be a mapping of names to shapes.")
    if dims is not None and not isinstance(dims, Mapping):
        raise ValueError("dims must be a mapping of variable names to axes.")
    if coords is not None and not isinstance(coords, Mapping):
        raise ValueError("coords must be a mapping of dimensions to values.")

    dimensions = {} if dims is None else dims
    unknown_dims = set(dimensions).difference(variables)
    if unknown_dims:
        raise ValueError("dims keys must name entries in variables.")

    layout = []
    variable_names = set()
    dimension_sizes = {}
    offset = 0
    for name, raw_shape in variables.items():
        if (
            not isinstance(name, str)
            or not name.strip()
            or name in variable_names
            or name in {"chain", "draw"}
        ):
            raise ValueError(
                "variables must have unique, nonempty string names, "
                "excluding 'chain' and 'draw'."
            )
        variable_names.add(name)

        if isinstance(raw_shape, (bool, np.bool_)):
            raise ValueError(
                "variables shapes must contain positive integer axes."
            )
        if isinstance(raw_shape, Integral):
            shape = (int(raw_shape),)
        else:
            try:
                shape = tuple(raw_shape)
            except TypeError as exc:
                raise ValueError(
                    "variables shapes must be an integer or a sequence of integers."
                ) from exc
        if any(
            isinstance(axis, (bool, np.bool_))
            or not isinstance(axis, Integral)
            or axis < 1
            for axis in shape
        ):
            raise ValueError(
                "variables shapes must contain positive integer axes."
            )
        shape = tuple(int(axis) for axis in shape)

        if name in dimensions:
            raw_dims = dimensions[name]
            if isinstance(raw_dims, str):
                raise ValueError(
                    "dims entries must contain one dimension name per axis."
                )
            try:
                event_dims = tuple(raw_dims)
            except TypeError as exc:
                raise ValueError(
                    "dims entries must contain one dimension name per axis."
                ) from exc
            if len(event_dims) != len(shape):
                raise ValueError(
                    "dims entries must contain one dimension name per axis."
                )
        else:
            event_dims = tuple(
                f"{name}_dim_{axis}" for axis in range(len(shape))
            )
        if any(
            not isinstance(dim, str)
            or not dim.strip()
            or dim in {"chain", "draw"}
            for dim in event_dims
        ):
            raise ValueError(
                "dims must contain nonempty strings excluding 'chain' and 'draw'."
            )
        if len(set(event_dims)) != len(event_dims):
            raise ValueError(
                "dims must not repeat a dimension within one variable."
            )
        for dim, axis_size in zip(event_dims, shape):
            if dim in dimension_sizes and dimension_sizes[dim] != axis_size:
                raise ValueError(
                    "dims shared by variables must have one consistent length."
                )
            dimension_sizes[dim] = axis_size

        layout.append((name, offset, shape, event_dims))
        offset += prod(shape)

    if offset != D:
        raise ValueError("variables sizes must sum to D.")
    if variable_names.intersection(dimension_sizes):
        raise ValueError(
            "variables names must differ from every name in dims, "
            "including generated dimension names."
        )

    if coords is not None:
        for dim, values in coords.items():
            if dim not in dimension_sizes:
                raise ValueError(
                    "coords keys must name dimensions used by variables."
                )
            try:
                coord_values = np.asarray(values)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "coords values must be one-dimensional array-like values."
                ) from exc
            if (
                coord_values.ndim != 1
                or coord_values.shape[0] != dimension_sizes[dim]
            ):
                raise ValueError(
                    "coords values must match their dimension lengths."
                )
    return layout


def require_from_dict():
    """Return ArviZ's ``from_dict`` converter or raise an actionable error."""
    if sys.version_info < (3, 12):
        raise ImportError(
            "ArviZ DataTree export requires Python 3.12+; "
            "use Python 3.12+ and install pyvbmc[arviz]."
        )
    try:
        from arviz_base import from_dict
    except ModuleNotFoundError as exc:
        if exc.name != "arviz_base":
            raise
        raise ImportError(
            "ArviZ DataTree export requires arviz-base; "
            "install pyvbmc[arviz] on Python 3.12+."
        ) from exc
    return from_dict


def datatree_from_arrays(arrays, *, dims=None, coords=None, attrs=None):
    """Build a one-chain ArviZ DataTree from posterior sample arrays."""
    if not isinstance(arrays, Mapping) or not arrays:
        raise ValueError("arrays must be a nonempty mapping.")

    posterior = {}
    n_samples = None
    for name, values in arrays.items():
        array = np.asarray(values)
        if array.ndim < 2 or array.shape[0] != 1:
            raise ValueError(
                "arrays values must have shape (1, n_samples, *shape)."
            )
        if n_samples is None:
            n_samples = array.shape[1]
        elif array.shape[1] != n_samples:
            raise ValueError("arrays must have one shared sample count.")
        posterior[name] = array

    export_coords = {} if coords is None else dict(coords)
    export_coords.update(
        {"chain": np.array([0]), "draw": np.arange(n_samples)}
    )
    from_dict = require_from_dict()
    return from_dict(
        {"posterior": posterior},
        sample_dims=["chain", "draw"],
        coords=export_coords,
        dims=dims,
        attrs={"posterior": attrs},
    )
