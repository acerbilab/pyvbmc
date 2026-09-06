"""Tests for the ``seed`` argument and the random generator threading.

``VBMC(seed=...)`` must make a run reproducible end to end without touching
NumPy's global random state (the ``gpyreg`` hyperparameter fit receives the
generator, the ``cma`` noise handler subclass draws from it), and
``seed=None`` must keep the legacy behaviour in which ``np.random.seed``
before construction fixes the run.

This module holds the suite's short end-to-end runs (two iterations at
``D = 2``). One of them is shared through the ``seeded_run`` fixture, and
the live check of the dtype canary (``pyvbmc.testing._dtype``) rides on
it rather than adding a run.
"""

import copy
import logging
from pathlib import Path

import numpy as np
import pytest

from pyvbmc import VBMC
from pyvbmc.testing import (
    assert_float64,
    assert_manifest_float64,
    load_bearing_arrays,
)
from pyvbmc.variational_posterior import VariationalPosterior

base_path = Path(__file__).parent
D = 2
# Fewest dtype leaves the walk of a finished instance may find: half the
# 394 measured on the shared run (dev/plans/stage0-dtype-canary.md).
LIVE_MIN_LEAVES = 197


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


@pytest.fixture(scope="module")
def seeded_run():
    """One short seeded run, shared by the tests that need a finished
    instance (the suite holds few full ``optimize()`` runs; AGENTS.md).
    Module-scoped, so it runs outside the per-test snapshot of the global
    random state and takes its own."""
    state = np.random.get_state()
    vbmc = _make_vbmc(42)
    vp, results = vbmc.optimize()
    np.random.set_state(state)
    return vbmc, vp, results


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


def test_seed_fixes_optimization(seeded_run):
    """Two short runs with the same seed are identical: the shared run,
    and a second one with the global random state reseeded before its
    construction and again between construction and optimization."""
    vbmc_1, vp_1, results_1 = seeded_run

    np.random.seed(12345)
    vbmc_2 = _make_vbmc(42)
    np.random.seed(999)
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


def test_seeded_run_state_is_float64(seeded_run):
    """The dtype canary on a live run: every floating-point array
    reachable from the instance and the results is float64 (the GP
    factors, the recorded history, ``optim_state`` and the transformer
    included), and the arrays the numerics ride on are present."""
    vbmc, vp, results = seeded_run
    assert_float64(vbmc, "vbmc", min_leaves=LIVE_MIN_LEAVES)
    # A handful of scalars, so no floor: an offender fails, nothing else.
    assert_float64(results, "results", min_leaves=0)
    arrays = load_bearing_arrays(
        vp=vp,
        gp=vbmc.gp,
        logger=vbmc.function_logger,
        pt=vbmc.parameter_transformer,
    )
    for k in (
        "lb_orig",
        "ub_orig",
        "plb_orig",
        "pub_orig",
        "lb_tran",
        "ub_tran",
        "plb_tran",
        "pub_tran",
    ):
        arrays[f"optim_state[{k!r}]"] = vbmc.optim_state[k]
    arrays["optim_state['cache']['x_orig']"] = vbmc.optim_state["cache"][
        "x_orig"
    ]
    assert_manifest_float64(arrays)


def test_seeded_run_leaves_global_state_untouched():
    """Every draw of a run comes from ``vbmc.rng``: a seeded run neither
    reads nor writes NumPy's global random state (the GP fit and the CMA-ES
    noise handler included)."""
    np.random.seed(7)
    before = np.random.get_state()
    vbmc = _make_vbmc(3)
    vbmc.optimize()
    after = np.random.get_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert "legacy" not in vbmc.random_state


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
