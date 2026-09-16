"""Package parity with the historical S-VBMC shrinkage calculation.

This developer check uses shipped posterior fixtures and performs no VBMC
run. It deliberately imports the scientific reference functions from
``svbmc_shrink_elbo.py`` rather than duplicating their covariance and
method-of-moments rules.
"""

import logging

import numpy as np
import pytest

pytest.importorskip("torch")

from svbmc_shrink_elbo import (  # noqa: E402
    moments,
    run_estimates,
    shrink_diagonal,
    shrink_full,
)

from pyvbmc.svbmc import SVBMC  # noqa: E402
from pyvbmc.svbmc._elbo_shrinkage import _two_level_shrinkage  # noqa: E402
from pyvbmc.testing.svbmc._fixtures import load_group  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def quiet_logger():
    logger = logging.getLogger("SVBMC")
    previous = logger.level
    logger.setLevel(logging.WARNING)
    yield
    logger.setLevel(previous)


@pytest.mark.parametrize(
    "group",
    ["normal_D1", "bounded_D2", "corr_D3", "upstream_GMM_noisy"],
)
def test_package_matches_historical_two_level_full(group):
    vps = load_group(group, rng=0)[0]
    stacked = SVBMC(vps, seed=0)
    offsets = np.concatenate([[0], np.cumsum(stacked.K)])
    jacobian = np.ravel(stacked._jacobian_corrections)
    runs = [
        run_estimates(vp, jacobian[offsets[m] : offsets[m + 1]])
        for m, vp in enumerate(stacked.vp_list)
    ]
    own = [
        np.ravel(vp.w) / np.sum(vp.w, dtype=np.float64)
        for vp in stacked.vp_list
    ]
    levels = np.array(
        [float(o @ I) for o, (I, _, _) in zip(own, runs)],
        dtype=np.float64,
    )
    level_variance = np.array(
        [float(o @ C @ o) for o, (_, _, C) in zip(own, runs)],
        dtype=np.float64,
    )
    run_mean, run_tau2, _, _ = moments(levels, np.diag(level_variance))
    shrunk_levels, _ = shrink_diagonal(
        levels, level_variance, run_mean, run_tau2
    )
    corrected = []
    shares = []
    for m, (I, _, covariance) in enumerate(runs):
        mean, tau2, spread, noise = moments(I, covariance)
        within, _ = shrink_full(I, covariance, mean, tau2)
        corrected.append(within + (shrunk_levels[m] - levels[m]))
        shares.append(None if spread <= 0.0 else noise / spread)
    corrected = np.concatenate(corrected)

    selected = np.arange(1, corrected.size + 1, dtype=np.float64)
    selected /= np.sum(selected, dtype=np.float64)
    entropy = 1.23456789
    expected = float(selected @ corrected + entropy)
    masses = np.array(
        [
            np.sum(selected[offsets[m] : offsets[m + 1]])
            for m in range(stacked.M)
        ]
    )
    defined = np.array([share is not None for share in shares])
    expected_share = (
        float(
            masses[defined]
            @ np.asarray([share for share in shares if share is not None])
            / masses[defined].sum()
        )
        if np.any(defined & (masses > 0.0))
        else None
    )

    actual, actual_share = _two_level_shrinkage(
        [vp.stats["I_sk"] for vp in stacked.vp_list],
        [vp.stats["J_sjk"] for vp in stacked.vp_list],
        [I for I, _, _ in runs],
        own,
        selected,
        entropy,
    )

    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-10)
    assert actual_share == pytest.approx(expected_share, rel=1e-12, abs=1e-10)
