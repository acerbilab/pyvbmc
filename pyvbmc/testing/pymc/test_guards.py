import numpy as np
import pytest

pm = pytest.importorskip("pymc")
pytest.importorskip("arviz_base")

from pyvbmc.pymc import PyMCTarget
from pyvbmc.pymc import _target as target_module
from pyvbmc.pymc._compat import (
    TESTED_RANGE,
    UnsupportedModel,
    check_model,
    clone_model,
    closed_form,
    default_transform,
    is_known_real_line,
    remove_value_transforms,
    transform_classes,
)


def _normal_model():
    with pm.Model() as model:
        pm.Normal("x")
    return model


def _explicit_normal(model):
    return PyMCTarget(
        model,
        start={"x": 0.0},
        plausible_bounds={"x": (-1.0, 1.0)},
        seed=1,
    )


def test_check_model_reports_tested_range():
    with pytest.raises(ImportError, match=TESTED_RANGE):
        check_model(object())


def test_transform_class_registry_uses_exact_types():
    classes = transform_classes()
    assert set(classes) == {"log", "logodds", "interval", "interval_base"}
    assert classes["interval"] is not classes["interval_base"]

    class IntervalSubclass(classes["interval"]):
        pass

    transform = IntervalSubclass(lower=0.0, upper=1.0)
    assert type(transform) not in classes.values()


@pytest.mark.parametrize(
    ("kind", "value", "lower", "upper", "expected"),
    [
        ("log", 0.0, None, None, 1.0),
        ("logodds", 0.0, None, None, 0.5),
        ("interval", 0.0, 2.0, 4.0, 3.0),
        ("interval", 0.0, 2.0, np.inf, 3.0),
        ("interval", 0.0, -np.inf, 4.0, 3.0),
    ],
)
def test_closed_form_backward_maps(kind, value, lower, upper, expected):
    actual = closed_form(kind, value, lower=lower, upper=upper)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)


def test_closed_form_handles_mixed_one_sided_interval_coordinates():
    actual = closed_form(
        "interval",
        np.zeros(3),
        lower=np.array([2.0, -np.inf, 1.0]),
        upper=np.array([np.inf, 4.0, 5.0]),
    )
    np.testing.assert_allclose(actual, [3.0, 3.0, 3.0])


def test_closed_form_rejects_unbounded_interval_coordinate():
    with pytest.raises(
        ValueError, match="Every interval-transform coordinate"
    ):
        closed_form(
            "interval",
            np.zeros(2),
            lower=np.array([0.0, -np.inf]),
            upper=np.array([np.inf, np.inf]),
        )


def test_default_transform_and_real_line_registry_are_conservative():
    with pm.Model() as model:
        normal = pm.Normal("normal")
        positive = pm.HalfNormal("positive")

    assert default_transform(normal.owner.op, normal) is None
    assert is_known_real_line(normal.owner.op)
    assert default_transform(positive.owner.op, positive) is not None
    assert not is_known_real_line(positive.owner.op)
    assert not is_known_real_line(object())


def test_remove_value_transforms_preserves_initial_values():
    with pm.Model() as model:
        bounded = pm.Uniform("bounded", 0.0, 2.0, initval=0.7)
        pm.Normal("normal", initval="support_point")

    result = remove_value_transforms(model, vars=[bounded])
    point = result.initial_point()
    assert point["bounded"] == pytest.approx(0.7)
    assert result.rvs_to_initial_values[result["normal"]] == "support_point"


def test_clone_model_preserves_initial_values():
    with pm.Model() as model:
        pm.Uniform("bounded", 0.0, 2.0, initval=0.7)
        pm.Normal("normal", initval="support_point")

    result = clone_model(model)
    point = result.initial_point()
    assert point["bounded_interval__"] == pytest.approx(
        np.log(0.7 / (2.0 - 0.7))
    )
    assert result.rvs_to_initial_values[result["normal"]] == "support_point"
    assert model.rvs_to_initial_values[model["bounded"]] == pytest.approx(0.7)


def test_unsupported_model_is_value_error():
    assert issubclass(UnsupportedModel, ValueError)


def test_constructor_wraps_default_transform_registry_error(monkeypatch):
    def unavailable(*args):
        raise ImportError("registry moved")

    monkeypatch.setattr(
        target_module._compat, "default_transform", unavailable
    )
    with pytest.raises(ImportError, match=TESTED_RANGE):
        _explicit_normal(_normal_model())


def test_constructor_rejects_incomplete_transform_registry(monkeypatch):
    classes = transform_classes()
    classes.pop("logodds")
    monkeypatch.setattr(
        target_module._compat, "transform_classes", lambda: classes
    )
    with pytest.raises(ImportError, match=TESTED_RANGE):
        _explicit_normal(_normal_model())


def test_constructor_checks_model_capabilities():
    with pytest.raises(ImportError, match=TESTED_RANGE):
        PyMCTarget(object())


def test_constructor_guards_missing_interval_args_fn(monkeypatch):
    with pm.Model() as model:
        variable = pm.Uniform("u", 0.0, 1.0)
    transform = model.rvs_to_transforms[variable]
    monkeypatch.delattr(transform, "args_fn")
    with pytest.raises(ImportError, match=TESTED_RANGE):
        PyMCTarget(
            model,
            start={"u": 0.5},
            plausible_bounds={"u": (0.2, 0.8)},
            seed=2,
        )


def test_transform_call_type_error_names_unsupported_variable(monkeypatch):
    with pm.Model() as model:
        variable = pm.Uniform("u", 0.0, 1.0)
    transform = model.rvs_to_transforms[variable]
    monkeypatch.setattr(transform, "args_fn", lambda: (0.0, 1.0))
    with pytest.raises(
        UnsupportedModel, match=r"u.*args_fn.*distribution inputs"
    ):
        PyMCTarget(model, seed=2)


def test_transform_map_type_error_names_unsupported_variable(monkeypatch):
    with pm.Model() as model:
        variable = pm.HalfNormal("sigma", 2.0)
    transform = model.rvs_to_transforms[variable]
    monkeypatch.setattr(transform, "backward", lambda value: value)
    with pytest.raises(
        UnsupportedModel, match=r"sigma.*LogTransform.*distribution inputs"
    ):
        PyMCTarget(model, seed=2)


def test_constructor_rejects_unregistered_transform_subclass():
    class IntervalSubclass(pm.distributions.transforms.Interval):
        pass

    transform = IntervalSubclass(lower=0.0, upper=1.0)
    with pm.Model() as model:
        pm.Normal("x", transform=transform)
    with pytest.raises(
        UnsupportedModel, match="unsupported transform IntervalSubclass"
    ):
        PyMCTarget(model, seed=3)


def test_constructor_checks_transform_against_closed_form(monkeypatch):
    monkeypatch.setattr(
        target_module._compat,
        "closed_form",
        lambda *args, **kwargs: np.asarray(99.0),
    )
    with pm.Model() as model:
        pm.HalfNormal("sigma", 2.0)
    with pytest.raises(ImportError, match=TESTED_RANGE):
        PyMCTarget(
            model,
            start={"sigma": 1.0},
            plausible_bounds={"sigma": (0.2, 2.0)},
            seed=4,
        )


def test_transform_check_uses_partial_initial_point(monkeypatch):
    real_closed_form = target_module._compat.closed_form
    probes = []

    def recording_closed_form(kind, value, lower=None, upper=None):
        probes.append(np.array(value, copy=True))
        return real_closed_form(kind, value, lower, upper)

    monkeypatch.setattr(
        target_module._compat, "closed_form", recording_closed_form
    )
    initial_value = np.exp(0.37)
    with pm.Model() as model:
        pm.HalfNormal("sigma", 2.0, initval=initial_value)
    PyMCTarget(
        model,
        start={"sigma": initial_value},
        plausible_bounds={"sigma": (0.2, 3.0)},
        setup_budget=1,
        seed=4,
    )
    assert len(probes) == 1
    np.testing.assert_allclose(probes[0], [0.37], rtol=0, atol=1e-14)


def test_constructor_rejects_float32_value_variable():
    import pytensor.tensor as pt

    model = _normal_model()
    rv = model.free_RVs[0]
    model.rvs_to_values[rv] = pt.scalar("x", dtype="float32")
    with pytest.raises(
        UnsupportedModel, match=r"x is float32.*value variables"
    ):
        PyMCTarget(model, seed=5)


def test_float32_graph_constant_does_not_reject_float64_model():
    import pytensor.tensor as pt

    fixed = pt.constant(np.float32(1.0))
    with pm.Model() as model:
        pm.Normal("x", mu=fixed, sigma=2.0)
    target = _explicit_normal(model)
    assert np.isfinite(target.log_joint(target.x0))
