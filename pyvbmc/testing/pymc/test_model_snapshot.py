import numpy as np
import pytest

pm = pytest.importorskip("pymc")
pytest.importorskip("arviz_base")
import pytensor

from pyvbmc.pymc import _snapshot
from pyvbmc.pymc._compat import remove_value_transforms, snapshot_model
from pyvbmc.pymc._snapshot import (
    _numeric_array_constants,
    _numeric_shared_inputs,
)


def _partially_untransform(model):
    return remove_value_transforms(model, vars=[model["x"]])


def test_snapshot_freezes_registered_and_unregistered_shared_inputs():
    observed = np.array([0.0, 1.0])
    labels = ["a", "b"]
    raw_scale = pytensor.shared(np.asarray(1.0), name="raw_scale")
    raw_upper = pytensor.shared(np.asarray(2.0), name="raw_upper")

    with pm.Model(coords={"observation": labels}) as model:
        data = pm.Data("data", observed, dims="observation")
        pm.Uniform("x", lower=0.0, upper=raw_upper, initval=0.7)
        pm.Normal("beta", initval="support_point")
        pm.Normal(
            "y",
            mu=model["x"] + model["beta"],
            sigma=raw_scale,
            observed=data,
            dims="observation",
        )

    snapshot, names = snapshot_model(model)
    partial = _partially_untransform(snapshot)
    point = partial.initial_point()
    logp = partial.compile_logp(mode="FAST_COMPILE")
    before = float(logp(point))

    assert names == ("x", "beta")
    assert set(rv.name for rv in partial.free_RVs) == set(names)
    assert point["x"] == pytest.approx(0.7)
    assert partial.rvs_to_initial_values[partial["beta"]] == "support_point"
    assert partial.coords["observation"] == ("a", "b")
    assert not _numeric_shared_inputs(partial)

    observed[:] = 50.0
    labels[:] = ["caller-a", "caller-b"]
    raw_scale.set_value(np.asarray(3.0))
    raw_upper.set_value(np.asarray(5.0))
    with model:
        pm.set_data(
            {"data": np.array([10.0, 11.0])},
            coords={"observation": ["new-a", "new-b"]},
        )

    assert float(logp(point)) == before
    assert partial.coords["observation"] == ("a", "b")

    fresh, fresh_names = snapshot_model(model)
    fresh_partial = _partially_untransform(fresh)
    fresh_logp = fresh_partial.compile_logp(mode="FAST_COMPILE")
    assert fresh_names == names
    assert float(fresh_logp(fresh_partial.initial_point())) != before


def test_snapshot_copies_arrays_written_into_the_graph():
    covariates = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    weights = np.array([1.0, 2.0])
    with pm.Model() as model:
        x = pm.Normal("x", 0.0, 2.0, shape=2)
        pm.Potential("penalty", -(weights * x**2).sum())
        pm.Normal("y", covariates @ x, 1.0, observed=np.zeros(3))

    snapshot, _ = snapshot_model(model)
    assert not any(
        np.shares_memory(constant.data, array)
        for constant in _numeric_array_constants(snapshot)
        for array in (covariates, weights)
    )
    point = {"x": np.array([0.3, 0.7])}
    logp = snapshot.compile_logp(mode="FAST_COMPILE")
    before = float(logp(point))
    source_before = float(model.compile_logp(mode="FAST_COMPILE")(point))
    assert source_before == before

    covariates[:, 1] += 10.0
    weights[:] = 5.0

    assert float(logp(point)) == before
    rebuilt = snapshot.compile_logp(mode="FAST_COMPILE")
    assert float(rebuilt(point)) == before
    # The caller's model keeps its arrays and so sees the change.
    source = model.compile_logp(mode="FAST_COMPILE")
    assert float(source(point)) != before


def test_snapshot_refuses_to_keep_the_callers_arrays(monkeypatch):
    covariates = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    with pm.Model() as model:
        x = pm.Normal("x", 0.0, 2.0, shape=2)
        pm.Normal("y", covariates @ x, 1.0, observed=np.zeros(3))

    monkeypatch.setattr(_snapshot, "_copied_constant", lambda value: value)
    with pytest.raises(ImportError, match="copying numeric model constants"):
        snapshot_model(model)
