"""Structured posterior export from PyMC target coordinates."""

import copy

import numpy as np
import pytest

pm = pytest.importorskip("pymc")
pytest.importorskip("arviz_base")

from pyvbmc import VariationalPosterior
from pyvbmc.pymc import PyMCTarget
from pyvbmc.testing._dtype import assert_float64
from pyvbmc.testing.pymc.models import vector_model


@pytest.fixture(scope="module")
def vector_target():
    spec = vector_model(np.random.default_rng(712))
    return spec, PyMCTarget(spec["model"], seed=713)


def _posterior(target, seed):
    vp = VariationalPosterior(target.D, 2, x0=target.x0, rng=seed)
    vp.sigma[:] = 0.2
    return vp


def test_export_layout_values_and_rng_stream(vector_target):
    _, target = vector_target
    vp = _posterior(target, 18)
    reference = _posterior(target, 18)
    flat, _ = reference.sample(40, orig_flag=True)

    data = target.to_arviz(vp, 40)
    posterior = data["posterior"]
    assert set(posterior.data_vars) == {"beta", "sigma"}
    assert posterior.beta.dims == ("chain", "draw", "coef")
    assert posterior.beta.shape == (1, 40, 2)
    assert posterior.sigma.dims == ("chain", "draw")
    assert posterior.sigma.shape == (1, 40)
    np.testing.assert_array_equal(
        posterior.coef.values, ["intercept", "slope"]
    )
    np.testing.assert_array_equal(posterior.beta.values[0], flat[:, :2])
    np.testing.assert_allclose(
        posterior.sigma.values[0], np.exp(flat[:, 2]), rtol=0, atol=1e-15
    )
    assert np.all(posterior.sigma.values > 0)
    assert posterior.attrs["parameter_space"] == "model"
    assert posterior.attrs["inference_library"] == "pyvbmc"
    assert vp.rng.bit_generator.state == reference.rng.bit_generator.state
    assert_float64(
        {name: posterior[name].values for name in target.names},
        path="posterior",
        min_leaves=2,
    )


def test_export_drives_conditioned_deterministics_and_predictions(
    vector_target,
):
    spec, target = vector_target
    data = target.to_arviz(_posterior(target, 29), 40)
    beta = data["posterior"].beta.values
    expected_mu = np.einsum("nd,csd->csn", spec["X"], beta)

    deterministics = pm.compute_deterministics(data, model=target.model)
    np.testing.assert_allclose(
        deterministics.mu.values, expected_mu, rtol=0, atol=1e-12
    )

    predictive = pm.sample_posterior_predictive(
        data,
        model=target.model,
        var_names=["mu", "y"],
        random_seed=30,
        progressbar=False,
    )
    group = predictive["posterior_predictive"]
    assert group.y.shape == (1, 40, len(spec["y"]))
    np.testing.assert_allclose(
        group.mu.values, expected_mu, rtol=0, atol=1e-12
    )


def test_invalid_export_does_not_advance_posterior_rng(vector_target):
    _, target = vector_target
    vp = VariationalPosterior(target.D + 1, rng=31)
    state = copy.deepcopy(vp.rng.bit_generator.state)
    with pytest.raises(ValueError, match=r"vp.D must equal target.D"):
        target.to_arviz(vp, 5)
    assert vp.rng.bit_generator.state == state

    valid = _posterior(target, 32)
    state = copy.deepcopy(valid.rng.bit_generator.state)
    with pytest.raises(ValueError, match="positive integer"):
        target.to_arviz(valid, 0)
    assert valid.rng.bit_generator.state == state
