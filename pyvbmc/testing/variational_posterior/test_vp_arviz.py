"""ArviZ export preserves the existing sampler and its random stream."""

import copy
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("arviz_base")
xr = pytest.importorskip("xarray")

from pyvbmc import VariationalPosterior
from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.testing._dtype import assert_float64


def make_vp(seed=17, K=2):
    transformer = ParameterTransformer(
        2,
        lb_orig=np.array([[0.0, -np.inf]]),
        ub_orig=np.array([[1.0, np.inf]]),
        plb_orig=np.array([[0.2, -2.0]]),
        pub_orig=np.array([[0.8, 2.0]]),
        transform_type="probit",
    )
    vp = VariationalPosterior(
        2, K, parameter_transformer=transformer, rng=seed
    )
    vp.sigma[:] = 0.7
    return vp


@pytest.mark.parametrize("orig_flag", [True, False])
@pytest.mark.parametrize("K", [1, 2])
def test_export_matches_sample_and_stream(orig_flag, K):
    vp, reference = make_vp(K=K), make_vp(K=K)
    expected, _ = reference.sample(31, orig_flag=orig_flag)
    data = vp.to_arviz(31, orig_flag=orig_flag)
    assert isinstance(data, xr.DataTree)
    assert list(data.children) == ["posterior"]
    posterior = data["posterior"]
    assert set(posterior.data_vars) == {"x_0", "x_1"}
    for i in range(2):
        value = posterior[f"x_{i}"]
        assert value.dims == ("chain", "draw")
        assert value.shape == (1, 31)
        np.testing.assert_array_equal(value.values[0], expected[:, i])
    assert vp.rng.bit_generator.state == reference.rng.bit_generator.state
    assert posterior.attrs["inference_library"] == "pyvbmc"
    assert posterior.attrs["sample_type"] == "independent"
    if orig_flag:
        assert np.all((posterior.x_0.values > 0) & (posterior.x_0.values < 1))
    assert_float64(vp)


def test_names_and_independence():
    vp = make_vp()
    data = vp.to_arviz(np.int64(3), var_names=["probability", "location"])
    before = data["posterior"].location.values.copy()
    vp.mu[:] = 100
    np.testing.assert_array_equal(data["posterior"].location.values, before)
    assert set(data["posterior"].data_vars) == {"probability", "location"}


def test_explicit_sample_dimensions_ignore_arviz_defaults():
    from arviz_base import rc_context

    with rc_context({"data.sample_dims": ["sample"]}):
        data = make_vp().to_arviz(4)
    assert data["posterior"].x_0.dims == ("chain", "draw")


def test_saved_legacy_posterior():
    vp = VariationalPosterior.load(
        Path(__file__).with_name("test_vp_save_static.pkl")
    )
    vp.rng = 15
    data = vp.to_arviz(3)
    assert len(data["posterior"].data_vars) == vp.D


@pytest.mark.parametrize(
    "count", [0, -1, 1.5, True, np.bool_(True), "2", None]
)
def test_bad_sample_count_does_not_draw(count):
    vp = make_vp()
    state = copy.deepcopy(vp.rng.bit_generator.state)
    with pytest.raises(ValueError, match="positive integer"):
        vp.to_arviz(count)
    assert vp.rng.bit_generator.state == state


@pytest.mark.parametrize(
    "names",
    [
        "ab",
        ["a"],
        ["a", "a"],
        ["", "b"],
        ["a", " "],
        ["chain", "b"],
        ["a", "draw"],
        ["a", 1],
        None,
        3,
    ],
)
def test_bad_names_do_not_draw(names):
    if names is None:
        names = [[], "b"]
    vp = make_vp()
    state = copy.deepcopy(vp.rng.bit_generator.state)
    with pytest.raises(ValueError, match="var_names"):
        vp.to_arviz(3, var_names=names)
    assert vp.rng.bit_generator.state == state
