"""Tests of ``pyvbmc.svbmc.utils``; these need neither torch nor fixtures."""

import matplotlib
import numpy as np
import pytest
from matplotlib.figure import Figure

from pyvbmc.svbmc.utils import find_init_bounds, overlay_corner_plot


@pytest.fixture(scope="module", autouse=True)
def agg_backend():
    """Draw into a head-less canvas, and put the backend back afterwards."""
    from matplotlib import pyplot as plt

    previous = matplotlib.get_backend()
    plt.switch_backend("Agg")
    yield
    plt.switch_backend(previous)


def _close(fig):
    from matplotlib import pyplot as plt

    plt.close(fig)


def test_find_init_bounds_uses_hard_bounds():
    lb, ub = find_init_bounds(LB=[-1, -2], UB=[1, 2])
    np.testing.assert_array_equal(lb, [-1, -2])
    np.testing.assert_array_equal(ub, [1, 2])


def test_find_init_bounds_prefers_plausible_bounds():
    lb, ub = find_init_bounds(
        LB=[-10, -10], UB=[10, 10], PLB=[0, -1], PUB=[2, 1]
    )
    np.testing.assert_array_equal(lb, [0, -1])
    np.testing.assert_array_equal(ub, [2, 1])


def test_find_init_bounds_without_hard_bounds():
    lb, ub = find_init_bounds(PLB=[0, 0], PUB=[2, 2])
    np.testing.assert_array_equal(lb, [0, 0])
    np.testing.assert_array_equal(ub, [2, 2])


def test_find_init_bounds_broadcasts_scalars():
    lb, ub = find_init_bounds(LB=-2, UB=2, PLB=[-1, -1], PUB=[1, 1])
    np.testing.assert_array_equal(lb, [-1, -1])
    np.testing.assert_array_equal(ub, [1, 1])
    lb, ub = find_init_bounds(LB=-2, UB=np.float64(3.0), PLB=[-1, -1])
    np.testing.assert_array_equal(lb, [-1, -1])
    np.testing.assert_array_equal(ub, [3, 3])


def test_find_init_bounds_without_any_bound():
    with pytest.raises(ValueError, match="Cannot infer dimensionality"):
        find_init_bounds()


def test_find_init_bounds_length_mismatch():
    with pytest.raises(ValueError, match="expected a scalar or 2"):
        find_init_bounds(LB=[-1, -1], UB=[1])


def test_find_init_bounds_infinite_bound_without_plausible():
    with pytest.raises(ValueError, match="infinite bound"):
        find_init_bounds(LB=[-np.inf], UB=[np.inf])
    with pytest.raises(ValueError, match="infinite bound"):
        find_init_bounds(
            LB=[-np.inf], UB=[np.inf], PLB=[-np.inf], PUB=[np.inf]
        )


def test_find_init_bounds_crossed_bounds():
    with pytest.raises(ValueError, match="lower bound .* >= upper bound"):
        find_init_bounds(LB=[1, 0], UB=[-1, 1])


def test_overlay_corner_plot_returns_figure():
    rng = np.random.default_rng(0)
    samples = [
        rng.standard_normal((100, 2)),
        rng.standard_normal((80, 2)) + np.array([2.0, -1.0]),
    ]
    fig = overlay_corner_plot(
        samples, labels=["A", "B"], colors=["C0", "C1"], bins=15
    )
    assert isinstance(fig, Figure)
    # A 2-D corner plot has D * (D + 1) / 2 + D(D-1)/2 = 4 panels.
    assert len(fig.axes) == 4
    legend = fig.axes[0].get_legend()
    assert legend is not None
    assert [t.get_text() for t in legend.get_texts()] == ["A", "B"]
    _close(fig)


def test_overlay_corner_plot_default_labels():
    rng = np.random.default_rng(1)
    fig = overlay_corner_plot([rng.standard_normal((50, 2))])
    legend = fig.axes[0].get_legend()
    assert [t.get_text() for t in legend.get_texts()] == ["Run 1"]
    _close(fig)


def test_overlay_corner_plot_empty_samples():
    with pytest.raises(ValueError, match="at least one array"):
        overlay_corner_plot([])


def test_overlay_corner_plot_dimension_mismatch():
    rng = np.random.default_rng(2)
    with pytest.raises(ValueError, match="same dimensionality"):
        overlay_corner_plot(
            [rng.standard_normal((10, 2)), rng.standard_normal((10, 3))]
        )


def test_overlay_corner_plot_label_mismatch():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match="`labels` length"):
        overlay_corner_plot(
            [rng.standard_normal((20, 2)), rng.standard_normal((20, 2))],
            labels=["only-one"],
        )
