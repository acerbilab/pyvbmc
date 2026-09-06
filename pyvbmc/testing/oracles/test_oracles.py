"""Stage-level oracles: the current code against stored reference outputs.

Every fixture under ``fixtures/`` is one algorithm state (GP, VP,
transformer, function logger, ``optim_state``, options, a candidate set)
saved as plain arrays by ``dev/scripts/make_oracle_fixtures.py``, together
with the outputs of the numerical stages computed from that state. The
tests rebuild the state through the public constructors and recompute.

A failure means the numerics changed, or a dtype did: every output of an
oracle is checked to be float64 before the cast that stores and compares
it (``cast_outputs``), the state an oracle worked on is walked for stray
floating-point dtypes, and the arrays the numerics ride on are asserted
present and float64 on every rebuilt state (the dtype canary,
``pyvbmc.testing._dtype``). If a numerical change is intended (a new
baseline), regenerate the fixtures with the generator; never loosen a
tolerance to make a refactor pass. The ``active_sample_step`` oracle needs
the benchmark targets in ``dev/scripts`` (a repository checkout). It and
``gp_fit`` run only on the platform that generated the fixture (a CMA-ES
search or a slice-sampling chain turns BLAS rounding differences into
different decisions; set ``PYVBMC_ORACLES_ALL=1`` to force them
elsewhere); the ``entmc`` and ``neg_elcbo`` oracles depend on the order of
the Monte Carlo draws and are re-baselined deliberately when that order
changes (see the plan in ``dev/plans/``).
"""

import os
import platform
import sys
from pathlib import Path

import numpy as np
import pytest

from pyvbmc.testing import (
    assert_float64,
    assert_manifest_float64,
    load_bearing_arrays,
)
from pyvbmc.testing.oracles._oracles import (
    ORACLES,
    PLATFORM_BOUND,
    applicable,
    cast_outputs,
    compare,
    format_rows,
)
from pyvbmc.testing.oracles._state import (
    build_state,
    load_snapshot,
    snapshot_names,
)

FIXTURES = Path(__file__).parent / "fixtures"
DEV_SCRIPTS = Path(__file__).resolve().parents[3] / "dev" / "scripts"
NAMES = snapshot_names(FIXTURES) if FIXTURES.exists() else []

# Outputs that may be the Python integer 0: a variance whose branch did
# not run (the entropy variance is never computed; the per-sample
# variance needs more than one hyperparameter sample). Every other output
# must be float64.
INTEGER_PLACEHOLDERS = frozenset({"varH", "var_ss", "varG_ss"})
# The rebuilt objects and the candidate set: what an oracle works on.
# ``ref`` and ``meta`` are the fixture's own and are not walked.
STATE_KEYS = ("pt", "vp", "gp", "logger", "optim_state", "options", "cand")
# Fewest dtype leaves a walk of one rebuilt state may find: half the
# smallest count measured over ``STATE_KEYS``, 62 on the single-sample
# snapshot (dev/plans/stage0-dtype-canary.md).
STATE_MIN_LEAVES = 31


def _target(meta):
    """The benchmark target of a fixture, or ``None`` outside a checkout."""
    if not (DEV_SCRIPTS / "benchmark_targets.py").exists():
        return None
    if str(DEV_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(DEV_SCRIPTS))
    from benchmark_targets import find_config

    return find_config(meta["config"]).make(seed=meta["problem_seed"]).fun


def _assert_outputs_float64(label, out):
    bad = [
        (k, np.asarray(v).dtype.name)
        for k, v in out.items()
        if not (k in INTEGER_PLACEHOLDERS and isinstance(v, int))
        and np.asarray(v).dtype != np.float64
    ]
    assert not bad, f"{label}: outputs that are not float64: {bad}"


@pytest.fixture(scope="module")
def snapshots():
    return {name: load_snapshot(FIXTURES / name) for name in NAMES}


def test_fixtures_present():
    assert NAMES, f"no oracle fixtures under {FIXTURES}"


@pytest.mark.parametrize("oracle", sorted(ORACLES))
@pytest.mark.parametrize("name", NAMES)
def test_oracle(snapshots, name, oracle):
    snap = snapshots[name]
    if oracle not in snap["ref"]:
        pytest.skip(f"{oracle} not applicable to {name}")
    fun = _target(snap["meta"]) if oracle == "active_sample_step" else None
    if oracle == "active_sample_step" and fun is None:
        pytest.skip("benchmark targets (dev/scripts) not available")
    if oracle in PLATFORM_BOUND:
        # A CMA-ES search or a slice-sampling chain amplifies BLAS rounding
        # differences into different decisions: the result reproduces only
        # on the platform that generated the fixture (seen on the first CI
        # run: Ubuntu's search picked points 1.3 away). Same-machine
        # determinism check.
        here, there = platform.platform(), snap["meta"].get("platform")
        if here != there and not os.environ.get("PYVBMC_ORACLES_ALL"):
            pytest.skip(
                f"{oracle} is platform-bound (fixture: {there}, "
                f"here: {here}); set PYVBMC_ORACLES_ALL=1 to force"
            )
    orc = ORACLES[oracle]
    state = build_state(snap, fun=fun)
    assert orc.applies(state), f"{oracle} stored but not applicable now"
    raw = orc.fn(state, snap["meta"]["oracle_seed"])
    _assert_outputs_float64(f"{name}/{oracle}", raw)
    # The state the oracle worked on, scratch arrays included.
    walked = {k: state[k] for k in STATE_KEYS}
    assert_float64(walked, f"{name}/{oracle} state", STATE_MIN_LEAVES)
    out = cast_outputs(raw)
    rows = compare(snap["ref"][oracle], out, orc.rtol, orc.atol)
    assert all(r[3] for r in rows), f"{name}/{oracle}:\n{format_rows(rows)}"


def test_fixture_complete(snapshots):
    """Every oracle applicable to a snapshot has a stored reference, so a
    half-written fixture cannot degrade into a green, all-skipped run."""
    for name, snap in snapshots.items():
        needed = set(applicable(build_state(snap)))
        missing = needed - set(snap["ref"])
        assert not missing, f"{name}: no reference for {sorted(missing)}"


def test_rebuilt_state_arrays_are_float64(snapshots):
    """The load-bearing arrays of every rebuilt state are present and
    float64. The GP posterior factors are recomputed by ``build_gp``; the
    rest is the fixture's own dtype, so this also pins the fixtures."""
    for name, snap in snapshots.items():
        state = build_state(snap)
        arrays = load_bearing_arrays(
            vp=state["vp"],
            gp=state["gp"],
            logger=state["logger"],
            pt=state["pt"],
        )
        arrays["cand['Xs']"] = state["cand"]["Xs"]
        assert_manifest_float64({f"{name}: {k}": v for k, v in arrays.items()})


def test_state_rebuilds(snapshots):
    """The rebuilt GP posteriors and VP are self-consistent."""
    for name, snap in snapshots.items():
        state = build_state(snap)
        gp, vp = state["gp"], state["vp"]
        assert len(gp.posteriors) == snap["gp"]["Ns"], name
        assert vp.K == snap["vp"]["K"] and vp.D == snap["vp"]["D"], name
        assert np.isclose(np.sum(vp.w), 1.0), name
        for p in gp.posteriors:
            assert np.all(np.isfinite(p.alpha)), name
