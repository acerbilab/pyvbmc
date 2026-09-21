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
    "count",
    [
        0,
        -1,
        1.5,
        True,
        np.bool_(True),
        np.array(True),
        "2",
        None,
        np.inf,
        np.nan,
        [3],
        np.array([3]),
    ],
)
def test_bad_sample_count_does_not_draw(count):
    vp = make_vp()
    state = copy.deepcopy(vp.rng.bit_generator.state)
    with pytest.raises(ValueError, match="positive integer"):
        vp.to_arviz(count)
    assert vp.rng.bit_generator.state == state


@pytest.mark.parametrize(
    "count", [7.0, 7e0, np.float64(7), np.int32(7), np.array(7), np.array(7.0)]
)
def test_a_whole_number_in_any_scalar_is_the_count(count):
    """The export takes the counts that ``sample`` takes, a whole number
    written as a float among them, and draws what the integer draws."""
    vp, reference = make_vp(), make_vp()
    expected, _ = reference.sample(7)
    posterior = vp.to_arviz(count)["posterior"]
    assert posterior.x_0.shape == (1, 7)
    np.testing.assert_array_equal(posterior.x_0.values[0], expected[:, 0])
    assert vp.rng.bit_generator.state == reference.rng.bit_generator.state


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


def make_structured_vp(D, seed=29):
    lb = np.full((1, D), -np.inf)
    ub = np.full((1, D), np.inf)
    plb = np.full((1, D), -2.0)
    pub = np.full((1, D), 2.0)
    lb[0, 0], ub[0, 0] = 0.0, 1.0
    plb[0, 0], pub[0, 0] = 0.2, 0.8
    transformer = ParameterTransformer(
        D,
        lb_orig=lb,
        ub_orig=ub,
        plb_orig=plb,
        pub_orig=pub,
        transform_type="probit",
    )
    vp = VariationalPosterior(
        D, 2, parameter_transformer=transformer, rng=seed
    )
    vp.sigma[:] = 0.6
    return vp


@pytest.mark.parametrize("orig_flag", [True, False])
def test_structured_export_matches_sample_and_stream(orig_flag):
    vp = make_structured_vp(3)
    reference = make_structured_vp(3)
    expected, _ = reference.sample(19, orig_flag=orig_flag)

    data = vp.to_arviz(
        19,
        orig_flag=orig_flag,
        variables={"beta": (2,), "sigma": ()},
        dims={"beta": ["coef"]},
        coords={"coef": ["a", "b"]},
    )

    posterior = data["posterior"]
    assert posterior.beta.dims == ("chain", "draw", "coef")
    assert posterior.beta.shape == (1, 19, 2)
    assert posterior.sigma.dims == ("chain", "draw")
    assert posterior.sigma.shape == (1, 19)
    np.testing.assert_array_equal(posterior.beta.values[0], expected[:, :2])
    np.testing.assert_array_equal(posterior.sigma.values[0], expected[:, 2])
    np.testing.assert_array_equal(posterior.coef.values, ["a", "b"])
    assert vp.rng.bit_generator.state == reference.rng.bit_generator.state
    expected_space = "original" if orig_flag else "internal"
    assert posterior.attrs["parameter_space"] == expected_space


def test_structured_matrix_uses_c_order_and_accepts_list_shape():
    vp = make_structured_vp(4)
    reference = make_structured_vp(4)
    expected, _ = reference.sample(7)

    data = vp.to_arviz(7, variables={"matrix": [2, 2]})

    matrix = data["posterior"].matrix
    assert matrix.dims == (
        "chain",
        "draw",
        "matrix_dim_0",
        "matrix_dim_1",
    )
    assert matrix.shape == (1, 7, 2, 2)
    for i in range(2):
        for j in range(2):
            np.testing.assert_array_equal(
                matrix.values[0, :, i, j], expected[:, 2 * i + j]
            )


@pytest.mark.parametrize(
    "kwargs, parameter",
    [
        (
            {"var_names": ["a", "b", "c", "d"], "variables": {"x": 4}},
            "var_names",
        ),
        ({"variables": {"x": 3}}, "variables"),
        (
            {"variables": {"x": 4}, "dims": {"x": ["row", "column"]}},
            "dims",
        ),
        (
            {"variables": {"x": 4}, "dims": {"missing": ["axis"]}},
            "dims",
        ),
        (
            {
                "variables": {"matrix": (2, 2)},
                "dims": {"matrix": ["row", "column"]},
                "coords": {"row": [0, 1, 2]},
            },
            "coords",
        ),
        (
            {"variables": {"x": 4}, "coords": {"unused": [0, 1, 2, 3]}},
            "coords",
        ),
        (
            {
                "variables": {"beta": 2, "coef": 2},
                "dims": {"beta": ["coef"], "coef": ["item"]},
            },
            "variables",
        ),
        ({"dims": {"x": ["axis"]}}, "dims"),
        ({"coords": {"axis": [0, 1, 2, 3]}}, "coords"),
        ({"variables": {"x": (True, 4)}}, "variables"),
        ({"variables": {"x": (0,), "y": 4}}, "variables"),
        ({"variables": {"x": (-1,), "y": 5}}, "variables"),
        (
            {
                "variables": {"a": 1, "b": 3},
                "dims": {"a": ["shared"], "b": ["shared"]},
            },
            "dims",
        ),
        ({"variables": {"a": 3, "a_dim_0": ()}}, "variables"),
        (
            {
                "variables": {"matrix": (2, 2)},
                "dims": {"matrix": ["coef", "coef"]},
            },
            "dims",
        ),
        (
            {
                "variables": {"x": 4},
                "dims": {"x": ["axis"]},
                "coords": {"axis": 1},
            },
            "coords",
        ),
        (
            {
                "variables": {"matrix": (2, 2)},
                "dims": {"matrix": ["row", "column"]},
                "coords": {"row": [[0, 1]]},
            },
            "coords",
        ),
    ],
)
def test_bad_structured_layout_does_not_draw(kwargs, parameter):
    vp = make_structured_vp(4)
    state = copy.deepcopy(vp.rng.bit_generator.state)
    with pytest.raises(ValueError, match=parameter):
        vp.to_arviz(3, **kwargs)
    assert vp.rng.bit_generator.state == state
