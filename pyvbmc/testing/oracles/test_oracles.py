"""Stage-level oracles: the current code against stored reference outputs.

Every fixture under ``fixtures/`` is one algorithm state (GP, VP,
transformer, function logger, ``optim_state``, options, a candidate set)
saved as plain arrays by ``dev/scripts/make_oracle_fixtures.py``, together
with the outputs of the numerical stages computed from that state. The
tests rebuild the state through the public constructors and recompute.

A failure means the numerics changed. If the change is intended (a new
baseline), regenerate the fixtures with the generator; never loosen a
tolerance to make a refactor pass. The ``active_sample_step`` oracle needs
the benchmark targets in ``dev/scripts`` (a repository checkout) and runs
only on the platform that generated the fixture (a CMA-ES search turns
BLAS rounding differences into different chosen points; set
``PYVBMC_ORACLES_ALL=1`` to force it elsewhere); the ``entmc`` and
``neg_elcbo`` oracles depend on the order of the Monte Carlo draws and are
re-baselined deliberately when that order changes (see the plan in
``dev/plans/``).
"""

import os
import platform
import sys
from pathlib import Path

import numpy as np
import pytest

from pyvbmc.testing.oracles._oracles import (
    ORACLES,
    applicable,
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


def _target(meta):
    """The benchmark target of a fixture, or ``None`` outside a checkout."""
    if not (DEV_SCRIPTS / "benchmark_targets.py").exists():
        return None
    if str(DEV_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(DEV_SCRIPTS))
    from benchmark_targets import find_config

    return find_config(meta["config"]).make(seed=meta["problem_seed"]).fun


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
    if oracle == "active_sample_step":
        if fun is None:
            pytest.skip("benchmark targets (dev/scripts) not available")
        # A full CMA-ES search amplifies BLAS rounding differences into
        # different decisions: the chosen points reproduce only on the
        # platform that generated the fixture (seen on the first CI run:
        # Ubuntu picked points 1.3 away). Same-machine determinism check.
        here, there = platform.platform(), snap["meta"].get("platform")
        if here != there and not os.environ.get("PYVBMC_ORACLES_ALL"):
            pytest.skip(
                f"active_sample_step is platform-bound (fixture: {there}, "
                f"here: {here}); set PYVBMC_ORACLES_ALL=1 to force"
            )
    orc = ORACLES[oracle]
    state = build_state(snap, fun=fun)
    assert orc.applies(state), f"{oracle} stored but not applicable now"
    out = orc(state, snap["meta"]["oracle_seed"])
    rows = compare(snap["ref"][oracle], out, orc.rtol, orc.atol)
    assert all(r[3] for r in rows), f"{name}/{oracle}:\n{format_rows(rows)}"


def test_fixture_complete(snapshots):
    """Every oracle applicable to a snapshot has a stored reference, so a
    half-written fixture cannot degrade into a green, all-skipped run."""
    for name, snap in snapshots.items():
        needed = set(applicable(build_state(snap)))
        missing = needed - set(snap["ref"])
        assert not missing, f"{name}: no reference for {sorted(missing)}"


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
