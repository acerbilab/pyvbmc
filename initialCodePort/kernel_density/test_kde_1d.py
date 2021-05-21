import numpy as np
import pytest

from kernel_density import kde1d


def test_kde_shapes():
    samples = np.concatenate(
        (
            np.random.randn(100, 1),
            np.random.randn(100, 1) * 2 + 35,
            np.random.randn(100, 1) + 55,
        )
    )
    density, xmesh, bandwidth = kde1d(
        samples, 2 ** 14, min(samples) - 5, max(samples) + 5
    )
    assert density.shape == (2 ** 14,)
    assert xmesh.shape == (2 ** 14,)
    assert bandwidth.shape == (1,)


def test_kde_n_not_power_of_two():
    samples = np.concatenate(
        (
            np.random.randn(100, 1),
            np.random.randn(100, 1) * 2 + 35,
            np.random.randn(100, 1) + 55,
        )
    )
    density, xmesh, bandwidth = kde1d(
        samples, 2 ** 14 - 10, min(samples) - 5, max(samples) + 5
    )
    assert density.shape == (2 ** 14,)
    assert xmesh.shape == (2 ** 14,)
    assert bandwidth.shape == (1,)


def test_kde_no_bounds():
    samples = np.concatenate(
        (
            np.random.randn(100, 1),
            np.random.randn(100, 1) * 2 + 35,
            np.random.randn(100, 1) + 55,
        )
    )
    density, xmesh, bandwidth = kde1d(samples, 2 ** 14 - 10)
    assert density.shape == (2 ** 14,)
    assert xmesh.shape == (2 ** 14,)
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
        kde1d(samples, 0)


def test_kde_bounds_switched():
    samples = np.concatenate(
        (
            np.random.randn(100, 1),
            np.random.randn(100, 1) * 2 + 35,
            np.random.randn(100, 1) + 55,
        )
    )
    with pytest.raises(ValueError):
        kde1d(samples, 2 ** 14, max(samples), min(samples))


def test_kde_density_valid_input():
    samples = np.random.randn(1000, 1)
    density, xmesh, bandwidth = kde1d(
        samples, 2 ** 14, min(samples), max(samples)
    )
    assert xmesh[0] == min(samples)
    assert xmesh[-1] == max(samples)