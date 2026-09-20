"""Unsupported PyMC model structures are rejected by variable name."""

import numpy as np
import pytest

pm = pytest.importorskip("pymc")
pytest.importorskip("arviz_base")

from pyvbmc.pymc import PyMCTarget, UnsupportedModel
from pyvbmc.testing.pymc.models import no_gradient_model, rejected_models


@pytest.fixture(scope="module")
def models():
    return rejected_models()


@pytest.mark.parametrize(
    ("key", "pattern"),
    [
        ("discrete", r"k.*int64"),
        ("simplex", r"w.*unsupported transform SimplexTransform"),
        ("suppressed", r"s.*suppressed"),
        ("random_bounds", r"h.*depends on another random variable"),
        (
            "symbolic_random_bounds",
            r"symbolic_bound.*depends on another random variable",
        ),
        ("ordered", r"ordered.*unsupported transform Ordered"),
        ("zero_sum", r"zero_sum.*unsupported transform ZeroSumTransform"),
        ("mixed", r"mixed.*mix"),
        ("shifted_log_support", r"shifted.*density is zero"),
    ],
)
def test_rejected_model_families(models, key, pattern):
    with pytest.raises(UnsupportedModel, match=pattern):
        PyMCTarget(models[key], seed=50)


def test_unknown_custom_free_variable_support_is_rejected():
    def logp(value):
        return -0.5 * value**2

    def random(rng=None, size=None):
        return np.asarray(rng.normal(size=size))

    with pm.Model() as model:
        pm.CustomDist("mystery", logp=logp, random=random)
    with pytest.raises(
        UnsupportedModel, match=r"mystery.*support cannot be inferred"
    ):
        PyMCTarget(model, seed=52)


def test_log_support_reaching_zero_is_accepted():
    with pm.Model() as model:
        pm.Wald("x", mu=1.0, lam=2.0, alpha=0.0)
    target = PyMCTarget(model, seed=54)
    lower, upper = target.support["x"]
    assert float(lower) == 0.0
    assert np.isposinf(float(upper))
    assert target.kept == {"x": "LogTransform"}
    assert np.isfinite(target.log_joint(target.x0))


def test_recognized_prior_with_custom_likelihood_is_accepted():
    target = PyMCTarget(no_gradient_model(), seed=53)
    assert target.names == ["x"]
    assert target.plausible_info["route"] == "prior"
    assert np.isfinite(target.log_joint(target.x0))
