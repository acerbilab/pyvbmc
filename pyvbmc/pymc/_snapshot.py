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


def _is_numeric_shared(value, SharedVariable) -> bool:
    if not isinstance(value, SharedVariable):
        return False
    try:
        dtype = np.dtype(value.dtype)
    except (AttributeError, TypeError):
        return False
    return np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)


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


def snapshot_model(model):
    """Freeze registered and unregistered numeric inputs of a PyMC model."""
    check_model(model)
    pm, _, _ = import_pymc()
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
        replacements = {
            value: value.type.constant_type(
                type=value.type,
                data=copy.deepcopy(value.get_value()),
                name=value.name,
            )
            for value in graph_inputs(roots)
            if value not in rngs and _is_numeric_shared(value, SharedVariable)
        }

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
    return snapshot, original_names


__all__ = ["snapshot_model"]
