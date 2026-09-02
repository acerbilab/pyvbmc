"""Tests for the ``seed`` argument and the random generator threading.

``VBMC(seed=...)`` must make a run reproducible end to end, including the
parts that still draw from NumPy's global random state (the ``gpyreg``
hyperparameter fit and the ``cma`` noise handler), and ``seed=None`` must keep
the legacy behaviour in which ``np.random.seed`` before construction fixes
the run.
"""

import copy
import logging
from pathlib import Path

import numpy as np
import pytest

from pyvbmc import VBMC
from pyvbmc.variational_posterior import VariationalPosterior

base_path = Path(__file__).parent
D = 2


@pytest.fixture(autouse=True)
def _restore_global_random_state():
    """Keep the tests here from shifting the global stream of later tests."""
    state = np.random.get_state()
    yield
    np.random.set_state(state)


def _log_density(x):
    x = np.asarray(x).reshape(-1)
    return float(-0.5 * np.sum(x**2))


def _make_vbmc(seed, **options):
    opts = {"max_iter": 2, "display": "off", "do_final_boost": False}
    opts.update(options)
    return VBMC(
        _log_density,
        np.zeros((1, D)),
        -5 * np.ones((1, D)),
        5 * np.ones((1, D)),
        -2 * np.ones((1, D)),
        2 * np.ones((1, D)),
        options=opts,
        seed=seed,
    )


def test_seed_fixes_initialization():
    vbmc_1 = _make_vbmc(0)
    vbmc_2 = _make_vbmc(0)
    assert np.array_equal(vbmc_1.vp.mu, vbmc_2.vp.mu)
    assert vbmc_1.rng.bit_generator.state == vbmc_2.rng.bit_generator.state
    # The VP shares the instance's generator.
    assert vbmc_1.vp.rng is vbmc_1.rng
    # Different seeds give different jitter.
    assert not np.array_equal(vbmc_1.vp.mu, _make_vbmc(1).vp.mu)


def test_seed_accepts_generator():
    rng = np.random.default_rng(3)
    vbmc = _make_vbmc(rng)
    assert vbmc.rng is rng


def test_seed_none_follows_global_seed():
    """Legacy contract: seeding NumPy globally still fixes the run."""
    np.random.seed(5)
    vbmc_1 = _make_vbmc(None)
    np.random.seed(5)
    vbmc_2 = _make_vbmc(None)
    assert np.array_equal(vbmc_1.vp.mu, vbmc_2.vp.mu)
    assert vbmc_1.rng.bit_generator.state == vbmc_2.rng.bit_generator.state


def test_seed_none_does_not_reseed_global_state():
    """Construction only consumes the draws that derive the generator (see
    ``pyvbmc.rng.get_rng``); it does not reseed the global state."""
    np.random.seed(11)
    _make_vbmc(None)
    after_construction = np.random.random()
    np.random.seed(11)
    np.random.randint(0, 2**32, size=4, dtype=np.uint32)
    assert np.random.random() == after_construction


def test_vp_deepcopy_shares_generator():
    vp = VariationalPosterior(D, 2, rng=np.random.default_rng(0))
    vp_copy = copy.deepcopy(vp)
    assert vp_copy.rng is vp.rng
    assert vp_copy.parameter_transformer is not vp.parameter_transformer
    # Draws from the copy advance the shared stream.
    x_copy, _ = vp_copy.sample(3, orig_flag=False)
    x, _ = vp.sample(3, orig_flag=False)
    assert not np.array_equal(x_copy, x)


def test_vp_sample_reproducible_with_seed():
    x_1, _ = VariationalPosterior(D, 3, rng=7).sample(10)
    x_2, _ = VariationalPosterior(D, 3, rng=7).sample(10)
    assert np.array_equal(x_1, x_2)


def test_seed_fixes_optimization():
    """Two short runs with the same seed are identical, even if the global
    random state is touched in between construction and optimization."""
    vbmc_1 = _make_vbmc(42)
    np.random.seed(999)
    vp_1, results_1 = vbmc_1.optimize()

    np.random.seed(12345)
    vbmc_2 = _make_vbmc(42)
    vp_2, results_2 = vbmc_2.optimize()

    assert results_1["elbo"] == results_2["elbo"]
    assert results_1["elbo_sd"] == results_2["elbo_sd"]
    assert np.array_equal(vp_1.mu, vp_2.mu)
    assert np.array_equal(vp_1.w, vp_2.w)
    assert np.array_equal(
        vbmc_1.function_logger.X_orig[vbmc_1.function_logger.X_flag],
        vbmc_2.function_logger.X_orig[vbmc_2.function_logger.X_flag],
    )
    # The returned posterior keeps sharing the instance's generator.
    assert vp_1.rng is vbmc_1.rng


def test_load_legacy_random_state_warns(caplog):
    """Files saved before ``vbmc.rng`` existed store only the global state."""
    vbmc = VBMC.load(
        base_path.joinpath("test_vbmc_save_static.pkl"), iteration=0
    )
    assert isinstance(vbmc.rng, np.random.Generator)
    assert vbmc.vp.rng is vbmc.rng

    with caplog.at_level(logging.WARNING, logger="VBMC"):
        vbmc = VBMC.load(
            base_path.joinpath("test_vbmc_save_static.pkl"),
            iteration=0,
            set_random_state=True,
        )
    assert "before VBMC had its own random generator" in caplog.text
    assert vbmc.vp.rng is vbmc.rng
