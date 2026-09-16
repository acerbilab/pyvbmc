import numpy as np
import pytest

pm = pytest.importorskip("pymc")
pytest.importorskip("arviz_base")
import pytensor

from pyvbmc.pymc._compat import remove_value_transforms, snapshot_model
from pyvbmc.pymc._snapshot import _numeric_shared_inputs


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
