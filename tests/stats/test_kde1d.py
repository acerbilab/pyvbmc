import numpy as np
import pytest
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from scipy.stats import norm, uniform

from pyvbmc.stats import kde_1d


def test_kde_shapes():
    samples = np.concatenate(
        (
            np.random.randn(100, 1),
            np.random.randn(100, 1) * 2 + 35,
            np.random.randn(100, 1) + 55,
        )
    )
    density, xmesh, bandwidth = kde_1d(
        samples, 2**14, min(samples) - 5, max(samples) + 5
    )
    assert density.shape == (2**14,)
    assert xmesh.shape == (2**14,)
    assert bandwidth.shape == (1,)


def test_kde_n_not_power_of_two():
    samples = np.concatenate(
        (
            np.random.randn(100, 1),
            np.random.randn(100, 1) * 2 + 35,
            np.random.randn(100, 1) + 55,
        )
    )
    density, xmesh, bandwidth = kde_1d(
        samples, 2**14 - 10, min(samples) - 5, max(samples) + 5
    )
    assert density.shape == (2**14,)
    assert xmesh.shape == (2**14,)
    assert bandwidth.shape == (1,)


def test_kde_no_bounds():
    samples = np.concatenate(
        (
            np.random.randn(100, 1),
            np.random.randn(100, 1) * 2 + 35,
            np.random.randn(100, 1) + 55,
        )
    )
    density, xmesh, bandwidth = kde_1d(samples, 2**14 - 10)
    assert density.shape == (2**14,)
    assert xmesh.shape == (2**14,)
    assert bandwidth.shape == (1,)


def test_kde_n_negative():
    samples = np.concatenate(
        (
            np.random.randn(100, 1),
            np.random.randn(100, 1) * 2 + 35,
            np.random.randn(100, 1) + 55,
        )
    )
    with pytest.raises(ValueError):
        kde_1d(samples, 0)


def test_kde_bounds_switched():
    samples = np.concatenate(
        (
            np.random.randn(100, 1),
            np.random.randn(100, 1) * 2 + 35,
            np.random.randn(100, 1) + 55,
        )
    )
    with pytest.raises(ValueError):
        kde_1d(samples, 2**14, max(samples), min(samples))


def mtv(xmesh: np.ndarray, yy1: np.ndarray, yy2: np.ndarray):
    """
    mtv Marginal Total Variation distances between two pdfs
    """
    mtv = 0
    f = lambda x: np.abs(
        interp1d(
            xmesh,
            yy1,
            kind="cubic",
            fill_value=np.array([0]),
            bounds_error=False,
        )(x)
        - interp1d(
            xmesh,
            yy2,
            kind="cubic",
            fill_value=np.array([0]),
            bounds_error=False,
        )(x)
    )
    bb = np.sort(np.array([xmesh[0], xmesh[-1], xmesh[0], xmesh[-1]]))
    for j in range(3):
        xx_range = np.linspace(bb[j], bb[j + 1], num=int(1e5))
        mtv = mtv + 0.5 * trapezoid(f(xx_range)) * (xx_range[1] - xx_range[0])
    return mtv


def test_kde_density_valid_input_one_gaussian():
    samples = norm.rvs(loc=0, scale=1, size=int(1e5))
    density_kde, xmesh, _ = kde_1d(samples, 2**14)
    density_gaussian = norm.pdf(xmesh, loc=0, scale=1)
    assert mtv(xmesh, density_kde, density_gaussian) < 0.03


def test_kde_density_valid_input_two_gaussians():
    samples = np.concatenate(
        (
            norm.rvs(loc=0, scale=1, size=int(1e5 * 0.5)),
            norm.rvs(loc=10, scale=1, size=int(1e5 * 0.5)),
        )
    )
    density_kde, xmesh, _ = kde_1d(samples, 2**14)
    density_gaussian = 0.5 * (
        norm.pdf(xmesh, loc=0, scale=1) + norm.pdf(xmesh, loc=10, scale=1)
    )
    assert mtv(xmesh, density_kde, density_gaussian) < 0.03


def test_kde_density_valid_input_uniform():
    samples = uniform.rvs(loc=0, scale=1, size=int(1e5))
    density_kde, xmesh, _ = kde_1d(samples, 2**14)
    density_uniform = uniform.pdf(xmesh, loc=0, scale=1)
    assert mtv(xmesh, density_kde, density_uniform) < 0.03
