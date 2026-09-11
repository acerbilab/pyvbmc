"""Construction of :class:`~pyvbmc.svbmc.SVBMC`: validation, filters, logs.

The posteriors here are synthetic: the constructor only reads the mixture
shapes, the transformer and the ``stable``, ``elbo``, ``I_sk`` and
``J_sjk`` statistics, so a fitted run is not needed to exercise it.
"""

import logging

import numpy as np
import pytest

pytest.importorskip("torch")

from pyvbmc import VariationalPosterior  # noqa: E402
from pyvbmc.svbmc import SVBMC  # noqa: E402
from pyvbmc.testing.svbmc._fixtures import load_group  # noqa: E402


def make_vp(D=2, K=1, stable=True, J=0.01, elbo=0.0, Ns=3, seed=0):
    """A posterior carrying the statistics stacking filters on."""
    vp = VariationalPosterior(D, K, np.zeros((1, D)), rng=seed)
    vp.stats = {
        "stable": stable,
        "elbo": float(elbo),
        "elbo_sd": 0.1,
        "I_sk": np.full((Ns, K), float(elbo)),
        "J_sjk": np.full((Ns, K, K), float(J)),
    }
    return vp


@pytest.fixture(autouse=True)
def restore_global_random_state():
    """Unseeded constructions draw their seed from NumPy's global state."""
    state = np.random.get_state()
    yield
    np.random.set_state(state)


@pytest.fixture(autouse=True)
def quiet_logger():
    """Keep progress messages out of the report, except where tested."""
    logger = logging.getLogger("SVBMC")
    previous = logger.level
    logger.setLevel(logging.WARNING)
    yield
    logger.setLevel(previous)


# --------------------------------------------------------------------- #
# s_max and stability filters                                           #
# --------------------------------------------------------------------- #
def test_keeps_only_stable_runs_below_s_max():
    kept = [make_vp(J=0.01), make_vp(J=0.01)]
    dropped = [make_vp(J=0.09), make_vp(stable=False, J=0.0)]
    stacked = SVBMC(kept + dropped, s_max=0.2, M_min=2)
    assert stacked.M == 2
    assert stacked.K == [1, 1]
    assert stacked.D == 2
    assert [id(vp) for vp in stacked.vp_list] == [id(vp) for vp in kept]


def test_s_max_comparison_is_strict():
    s_max = 0.2
    boundary = make_vp(J=s_max**2)  # sqrt(max(J_sjk)) == s_max
    with pytest.raises(ValueError, match="at least 1 well-converged"):
        SVBMC([boundary], s_max=s_max, M_min=1)


def test_fitted_runs_survive_the_default_filters():
    vps = load_group("upstream_GMM", rng=0)[0]
    stacked = SVBMC(vps, seed=0)
    assert stacked.M == len(vps) == 10
    assert stacked.K == [50] * 10
    assert all(bool(vp.stats["stable"]) for vp in vps)
    largest = max(np.sqrt(np.max(vp.stats["J_sjk"])) for vp in vps)
    assert largest < np.sqrt(5)
    # A tolerance below what any run achieves keeps nothing.
    with pytest.raises(ValueError, match="well-converged"):
        SVBMC(vps, s_max=1e-6)


def test_too_few_survivors_raises():
    runs = [make_vp(J=0.0), make_vp(J=100.0)]
    with pytest.raises(ValueError) as excinfo:
        SVBMC(runs, s_max=0.2, M_min=2)
    message = str(excinfo.value)
    assert "at least 2 well-converged VBMC runs" in message
    assert "got 1" in message


# --------------------------------------------------------------------- #
# M_min                                                                 #
# --------------------------------------------------------------------- #
def test_M_min_as_proportion():
    runs = [make_vp(), make_vp(), make_vp(J=100.0)]
    # 2/3 of three runs is 2.0, and two runs survive the s_max filter.
    stacked = SVBMC(runs, M_min=2 / 3)
    assert stacked.M == 2
    with pytest.raises(ValueError, match="at least 3 well-converged"):
        SVBMC(runs, M_min=1)


def test_M_min_as_count():
    runs = [make_vp(), make_vp(), make_vp(J=100.0)]
    assert SVBMC(runs, M_min=2).M == 2
    with pytest.raises(ValueError, match="at least 3 well-converged"):
        SVBMC(runs, M_min=3)


def test_M_min_equal_to_number_of_runs():
    runs = [make_vp(), make_vp()]
    assert SVBMC(runs, M_min=2).M == 2


@pytest.mark.parametrize("M_min", [0, -0.1, -3])
def test_M_min_non_positive_raises(M_min):
    with pytest.raises(ValueError, match="positive number"):
        SVBMC([make_vp(), make_vp()], M_min=M_min)


def test_M_min_above_number_of_runs_warns_and_clamps():
    runs = [make_vp(), make_vp()]
    with pytest.warns(UserWarning, match="at most `len"):
        stacked = SVBMC(runs, M_min=10)
    assert stacked.M == 2


def test_M_min_non_integer_count_warns_and_rounds():
    runs = [make_vp(), make_vp(), make_vp()]
    with pytest.warns(UserWarning, match="rounding"):
        stacked = SVBMC(runs, M_min=2.4)
    assert stacked.M == 3
    # 2.4 rounds to 2, so two survivors are still enough.
    with pytest.warns(UserWarning, match="rounding"):
        stacked = SVBMC(runs[:2] + [make_vp(J=100.0)], M_min=2.4)
    assert stacked.M == 2


# --------------------------------------------------------------------- #
# Input validation                                                      #
# --------------------------------------------------------------------- #
def test_unfinished_posterior_raises():
    # A posterior VBMC has not finished has no statistics at all.
    vp = VariationalPosterior(2, 1, np.zeros((1, 2)), rng=0)
    assert vp.stats is None
    with pytest.raises(ValueError, match="no statistics"):
        SVBMC([make_vp(), vp], seed=0)


def test_unstable_run_with_nonfinite_statistics_is_dropped():
    bad = make_vp(stable=False, seed=3)
    bad.stats["I_sk"][:] = np.nan
    bad.stats["J_sjk"][:] = np.nan
    bad.stats["elbo"] = np.nan
    stacked = SVBMC([make_vp(seed=1), make_vp(seed=2), bad], M_min=2, seed=0)
    assert stacked.M == 2


def test_stable_run_with_nonfinite_statistics_raises():
    bad = make_vp(seed=3)
    bad.stats["I_sk"][0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        SVBMC([make_vp(seed=1), bad], seed=0)


def test_empty_list_raises():
    with pytest.raises(ValueError, match="is empty"):
        SVBMC([])


def test_non_posterior_element_raises():
    with pytest.raises(TypeError, match="not a fitted VariationalPosterior"):
        SVBMC([make_vp(), object()])


def test_mismatched_dimension_raises():
    with pytest.raises(ValueError, match="stack only posteriors of the same"):
        SVBMC([make_vp(D=2), make_vp(D=3)])


def test_component_count_mismatch_raises():
    vp = make_vp(K=3)
    vp.w = np.ones((1, 2)) / 2
    with pytest.raises(ValueError, match="components in `mu`"):
        SVBMC([vp])


def test_missing_statistic_raises():
    vp = make_vp()
    del vp.stats["elbo"]
    with pytest.raises(ValueError, match="lacks 'elbo'"):
        SVBMC([vp])


def test_nonfinite_statistics_raise():
    vp = make_vp()
    vp.stats["I_sk"] = vp.stats["I_sk"].copy()
    vp.stats["I_sk"][0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        SVBMC([vp])

    vp = make_vp()
    vp.stats["elbo"] = np.inf
    with pytest.raises(ValueError, match="nonfinite"):
        SVBMC([vp])


def test_malformed_statistic_shapes_raise():
    vp = make_vp(K=2)
    vp.stats["J_sjk"] = np.zeros((3, 2, 3))
    with pytest.raises(ValueError, match=r"'J_sjk'\S* should have shape"):
        SVBMC([vp])

    vp = make_vp(K=2)
    vp.stats["I_sk"] = np.zeros(2)
    with pytest.raises(ValueError, match=r"'I_sk'\S* should have shape"):
        SVBMC([vp])


# --------------------------------------------------------------------- #
# Logging                                                               #
# --------------------------------------------------------------------- #
def test_construction_logs_progress(caplog):
    caplog.set_level(logging.INFO, logger="SVBMC")
    SVBMC([make_vp(), make_vp()])
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "SVBMC" and record.levelno == logging.INFO
    ]
    assert any("well-converged runs" in message for message in messages)


def test_user_set_level_is_respected(caplog):
    # The autouse fixture has already put the logger at WARNING; the
    # constructor must not raise it back to INFO.
    caplog.set_level(logging.DEBUG)
    SVBMC([make_vp(), make_vp()])
    assert logging.getLogger("SVBMC").level == logging.WARNING
    assert not [
        record
        for record in caplog.records
        if record.name == "SVBMC" and record.levelno < logging.WARNING
    ]
