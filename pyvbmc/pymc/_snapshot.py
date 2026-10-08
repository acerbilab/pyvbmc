"""Construction of fixed PyMC model snapshots."""

from __future__ import annotations

import copy

import numpy as np

from pyvbmc.pymc._compat import (
    UnsupportedModel,
    _version_error,
    check_model,
    import_pymc,
)


def _without_initial_values(model, operation):
    saved = dict(model.rvs_to_initial_values)
    try:
        for rv in saved:
            model.rvs_to_initial_values[rv] = None
        result = operation(model)
    finally:
        model.rvs_to_initial_values.update(saved)
    return result, {
        rv.name: copy.deepcopy(value) for rv, value in saved.items()
    }


def _is_numeric_dtype(dtype) -> bool:
    return np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)


def _is_numeric_shared(value, SharedVariable) -> bool:
    if not isinstance(value, SharedVariable):
        return False
    try:
        dtype = np.dtype(value.dtype)
    except (AttributeError, TypeError):
        return False
    return _is_numeric_dtype(dtype)


def _is_numeric_array_constant(value, Constant) -> bool:
    """Return whether a graph constant holds a numeric NumPy array.

    PyTensor wraps an array written into a model graph without copying it,
    and cloning a graph keeps its constants, so such a constant's data stays
    the caller's array.
    """
    return (
        isinstance(value, Constant)
        and isinstance(value.data, np.ndarray)
        and _is_numeric_dtype(value.data.dtype)
    )


def _copied_constant(value):
    return type(value)(value.type, copy.deepcopy(value.data), name=value.name)


def _model_roots(model):
    return [
        *model.free_RVs,
        *model.observed_RVs,
        *model.deterministics,
        *model.potentials,
        *model.data_vars,
        *model.rvs_to_values.values(),
        *model.dim_lengths.values(),
    ]


def _numeric_shared_inputs(model):
    """Return non-RNG numeric shared inputs reachable from a model."""
    pm, _, _ = import_pymc()
    try:
        from pymc.pytensorf import find_rng_nodes
        from pytensor.compile import SharedVariable
        from pytensor.graph.traversal import graph_inputs
    except (AttributeError, ImportError) as exc:
        raise _version_error(pm, "snapshot graph traversal", exc)

    roots = _model_roots(model)
    rngs = set(find_rng_nodes(roots))
    return tuple(
        value
        for value in graph_inputs(roots)
        if value not in rngs and _is_numeric_shared(value, SharedVariable)
    )


def _numeric_array_constants(model):
    """Return the numeric array constants of a model's graph.

    The traversal does not enter the inner graphs of operations, such as
    the body of a scan loop.
    """
    pm, _, _ = import_pymc()
    try:
        from pytensor.graph.basic import Constant
        from pytensor.graph.traversal import graph_inputs
    except (AttributeError, ImportError) as exc:
        raise _version_error(pm, "snapshot graph traversal", exc)

    return tuple(
        value
        for value in graph_inputs(_model_roots(model))
        if _is_numeric_array_constant(value, Constant)
    )


def copy_array_constants(outputs, owned=()):
    """Return graphs computing `outputs` from copies of their arrays.

    Every numeric array constant of the graphs, other than those in
    `owned`, is replaced with a copy of itself; the inner graphs of
    operations are not entered. PyMC builds a log density, and a transform
    its maps, when they are requested, so the arrays that a ``CustomDist``
    log-density function or a transform's bounds function writes into them
    never pass through the snapshot.
    """
    pm, _, _ = import_pymc()
    try:
        from pytensor.graph.basic import Constant
        from pytensor.graph.replace import clone_replace
        from pytensor.graph.traversal import graph_inputs
    except (AttributeError, ImportError) as exc:
        raise _version_error(pm, "graph traversal for copied constants", exc)

    owned = set(owned)
    replacements = {
        value: _copied_constant(value)
        for value in graph_inputs(outputs)
        if _is_numeric_array_constant(value, Constant) and value not in owned
    }
    if not replacements:
        return list(outputs)
    try:
        return clone_replace(
            list(outputs), replace=replacements, rebuild_strict=False
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise _version_error(
            pm, "graph reconstruction for copied constants", exc
        )


def snapshot_model(model):
    """Freeze registered and unregistered numeric inputs of a PyMC model.

    Numeric shared variables become constants holding copies of their
    values, and array constants are replaced with copies of themselves, so
    that a later change to the caller's arrays or shared variables reaches
    neither the snapshot, nor its compiled functions, nor a copy of it
    restored from a pickle. The constants inside an op's inner graph, such
    as the body of a scan loop, stay shared: PyTensor interns the nodes of
    inner graphs by the values of their constants, so a copy resolves to
    the original nodes.
    """
    check_model(model)
    pm, _, _ = import_pymc()
    source_constants = set(_numeric_array_constants(model))
    original_names = tuple(rv.name for rv in model.free_RVs)
    if any(name is None for name in original_names) or len(
        set(original_names)
    ) != len(original_names):
        raise UnsupportedModel("Free variables must have unique names.")

    try:
        from pymc.model.fgraph import (
            ModelFreeRV,
            fgraph_from_model,
            model_from_fgraph,
        )
        from pymc.model.transform import freeze_dims_and_data
        from pymc.pytensorf import find_rng_nodes
        from pytensor.compile import SharedVariable
        from pytensor.graph import FunctionGraph
        from pytensor.graph.basic import Constant
        from pytensor.graph.replace import clone_replace
        from pytensor.graph.traversal import graph_inputs
    except (AttributeError, ImportError) as exc:
        raise _version_error(pm, "snapshot graph reconstruction", exc)

    try:
        frozen = freeze_dims_and_data(model)
    except NotImplementedError as exc:
        raise UnsupportedModel(
            "PyMC targets support only constant or strategy-string initial values."
        ) from exc
    except (AttributeError, KeyError, TypeError) as exc:
        raise _version_error(pm, "registered data and dimension freezing", exc)

    try:
        (fgraph, _), initial_values = _without_initial_values(
            frozen, fgraph_from_model
        )
        roots = [*fgraph.outputs, *fgraph._dim_lengths.values()]
        rngs = set(find_rng_nodes(roots))
        replacements = {}
        for value in graph_inputs(roots):
            if value in rngs:
                continue
            if _is_numeric_shared(value, SharedVariable):
                replacements[value] = value.type.constant_type(
                    type=value.type,
                    data=copy.deepcopy(value.get_value()),
                    name=value.name,
                )
            elif _is_numeric_array_constant(value, Constant):
                replacements[value] = _copied_constant(value)

        outputs = clone_replace(
            fgraph.outputs,
            replace=replacements,
            rebuild_strict=False,
        )
        rebuilt = FunctionGraph(outputs=outputs, clone=False)
        rebuilt._coords = copy.deepcopy(fgraph._coords)
        rebuilt._dim_lengths = {
            name: clone_replace(
                [value], replace=replacements, rebuild_strict=False
            )[0]
            for name, value in fgraph._dim_lengths.items()
        }

        value_replacements = []
        for node in rebuilt.apply_nodes:
            if not isinstance(node.op, ModelFreeRV):
                continue
            rv, old_value, *_ = node.inputs
            transform = node.op.transform
            if transform is None:
                new_value = rv.type()
            else:
                new_value = transform.forward(rv, *rv.owner.inputs).type()
            new_value.name = old_value.name
            value_replacements.append((old_value, new_value))
        rebuilt.replace_all(value_replacements, import_missing=True)
        snapshot = model_from_fgraph(rebuilt, mutate_fgraph=True)
        for name, value in initial_values.items():
            snapshot.set_initval(snapshot[name], copy.deepcopy(value))
    except UnsupportedModel:
        raise
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise _version_error(pm, "snapshot graph reconstruction", exc)

    rebuilt_names = tuple(rv.name for rv in snapshot.free_RVs)
    if set(rebuilt_names) != set(original_names) or len(rebuilt_names) != len(
        original_names
    ):
        raise _version_error(
            pm, "free-variable preservation during snapshotting"
        )
    residual = _numeric_shared_inputs(snapshot)
    if residual:
        names = ", ".join(value.name or "<unnamed>" for value in residual)
        raise _version_error(
            pm,
            f"freezing numeric shared model inputs ({names})",
        )
    retained = [
        value
        for value in _numeric_array_constants(snapshot)
        if value in source_constants
    ]
    if retained:
        names = ", ".join(
            value.name or f"array of shape {value.data.shape}"
            for value in retained
        )
        raise _version_error(
            pm,
            f"copying numeric model constants ({names})",
        )
    return snapshot, original_names


__all__ = ["copy_array_constants", "snapshot_model"]
