"""The GP records of the iteration history: lean copies and their restore.

``VBMC.optimize`` records each iteration's GP without its posterior
factors, and ``VBMC.get_gp`` gives back a copy with the factors recomputed.
These tests check the two halves against each other on stored algorithm
states, and the public method on a saved instance.
"""

import copy
from pathlib import Path

import gpyreg as gpr
import numpy as np
import pytest

from pyvbmc import VBMC
from pyvbmc.testing.oracles._state import build_state, load_snapshot
from pyvbmc.vbmc.gaussian_process_train import _lean_gp, _restore_gp_posteriors

BASE_PATH = Path(__file__).parent
FIXTURES = Path(__file__).parents[1] / "oracles" / "fixtures"

# One state with several hyperparameter samples, one with a single sample,
# and one with input-dependent noise (``s2``).
SNAPSHOTS = [
    "normal_D2_warmup",
    "normal_D2_singlesample",
    "rosenbrock_D2_noise1_viqr",
]

POSTERIOR_ARRAYS = ("hyp", "alpha", "sW", "L")
POSTERIOR_SCALARS = ("sn2_mult", "L_chol")


@pytest.fixture(params=SNAPSHOTS)
def gp(request):
    """A GP rebuilt from one of the stored algorithm states."""
    if not (FIXTURES / (request.param + ".json")).exists():
        pytest.skip(f"no oracle fixture {request.param} under {FIXTURES}")
    return build_state(load_snapshot(FIXTURES / request.param))["gp"]


def make_vbmc():
    """A VBMC instance that has not been optimized."""
    D = 3
    return VBMC(
        lambda x: np.sum(x + 2),
        np.ones((2, D)) * 3,
        np.ones((1, D)) * 1,
        np.ones((1, D)) * 5,
        np.ones((1, D)) * 2,
        np.ones((1, D)) * 4,
    )


def test_lean_gp_drops_the_factors_and_keeps_the_rest(gp):
    """The lean copy keeps data and hyperparameters, not the factors."""
    posteriors_before = list(gp.posteriors)
    hyp_before = gp.get_hyperparameters(as_array=True)

    lean = _lean_gp(gp)

    assert len(lean.posteriors) == len(posteriors_before)
    for posterior in lean.posteriors:
        assert posterior.alpha is None
        assert posterior.L is None
        assert posterior.sW is None
    assert np.array_equal(lean.get_hyperparameters(as_array=True), hyp_before)
    assert np.array_equal(lean.X, gp.X)
    assert np.array_equal(lean.y, gp.y)
    if gp.s2 is None:
        assert lean.s2 is None
    else:
        assert np.array_equal(lean.s2, gp.s2)
    assert lean.temporary_data == {}

    # The original is untouched: same posterior objects, still with factors.
    assert lean.posteriors is not gp.posteriors
    assert list(gp.posteriors) == posteriors_before
    for posterior in gp.posteriors:
        assert posterior.alpha is not None
        assert posterior.L is not None
        assert posterior.sW is not None
    assert np.array_equal(gp.get_hyperparameters(as_array=True), hyp_before)


def test_restore_reproduces_the_factors_exactly(gp):
    """The restored factors are bit for bit the ones that were dropped."""
    # The deep copy is the one the iteration history makes when recording.
    restored = _restore_gp_posteriors(copy.deepcopy(_lean_gp(gp)))

    assert len(restored.posteriors) == len(gp.posteriors)
    for new, old in zip(restored.posteriors, gp.posteriors):
        for name in POSTERIOR_ARRAYS:
            assert np.array_equal(getattr(new, name), getattr(old, name))
        for name in POSTERIOR_SCALARS:
            assert getattr(new, name) == getattr(old, name)

    x_star = gp.X[:5] + 0.1
    for restored_out, old_out in zip(
        restored.predict(x_star), gp.predict(x_star)
    ):
        assert np.array_equal(restored_out, old_out)


def test_restore_is_a_no_op_on_a_complete_gp(gp):
    """A GP that carries its factors is returned as it is."""
    posteriors_before = list(gp.posteriors)
    factors_before = [(p.alpha, p.L, p.sW, p.hyp) for p in gp.posteriors]

    restored = _restore_gp_posteriors(gp)

    assert restored is gp
    assert list(gp.posteriors) == posteriors_before
    for posterior, factors in zip(gp.posteriors, factors_before):
        assert posterior.alpha is factors[0]
        assert posterior.L is factors[1]
        assert posterior.sW is factors[2]
        assert posterior.hyp is factors[3]


def test_get_gp_returns_a_usable_copy():
    """``get_gp`` completes a record without changing it."""
    vbmc = VBMC.load(BASE_PATH.joinpath("test_vbmc_save_static.pkl"))
    stored = vbmc.iteration_history["gp"][3]
    stored_before = copy.deepcopy(stored)

    gp = vbmc.get_gp(3)

    assert gp is not stored
    for posterior in gp.posteriors:
        assert posterior.alpha is not None
        assert posterior.L is not None
        assert posterior.sW is not None
    assert np.array_equal(gp.X, stored.X)
    assert np.array_equal(
        gp.get_hyperparameters(as_array=True),
        stored.get_hyperparameters(as_array=True),
    )
    # The record itself is the same object, with the same contents.
    assert vbmc.iteration_history["gp"][3] is stored
    assert np.array_equal(stored.X, stored_before.X)
    assert np.array_equal(stored.y, stored_before.y)
    assert np.array_equal(
        stored.get_hyperparameters(as_array=True),
        stored_before.get_hyperparameters(as_array=True),
    )
    for posterior, before in zip(stored.posteriors, stored_before.posteriors):
        for name in POSTERIOR_ARRAYS:
            assert np.array_equal(
                getattr(posterior, name), getattr(before, name)
            )


@pytest.mark.parametrize("iteration", [-1, 7])
def test_get_gp_rejects_an_index_outside_the_history(iteration):
    vbmc = VBMC.load(BASE_PATH.joinpath("test_vbmc_save_static.pkl"))
    with pytest.raises(ValueError) as err:
        vbmc.get_gp(iteration)
    assert f"Specified iteration ({iteration})" in err.value.args[0]


def test_get_gp_without_a_recorded_gp():
    vbmc = make_vbmc()
    assert vbmc.iteration_history["gp"] is None
    with pytest.raises(ValueError) as err:
        vbmc.get_gp(0)
    assert "No Gaussian process has been recorded" in err.value.args[0]


def test_lean_gp_of_a_gp_without_posteriors():
    """A GP that was never fitted has no posteriors; the lean copy and the
    restore leave it that way."""
    gp_empty = gpr.GP(
        D=2,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    assert gp_empty.posteriors is None
    lean = _lean_gp(gp_empty)
    assert lean is not gp_empty
    assert lean.posteriors is None
    assert lean.temporary_data == {}
    assert _restore_gp_posteriors(lean) is lean
    assert lean.posteriors is None
