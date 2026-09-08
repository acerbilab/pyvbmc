"""Replay the three authentic pre-``train_gp`` history captures."""

import os
from pathlib import Path

import numpy as np
import pytest

from pyvbmc.testing import assert_float64
from pyvbmc.testing.oracles._gp_fit_history import (
    history_block_summary,
    portable_outputs,
    replay,
    same_platform,
)
from pyvbmc.testing.oracles._state import load_snapshot, snapshot_names

FIXTURES = Path(__file__).parent / "fixtures" / "gp_fit_history"
NAMES = snapshot_names(FIXTURES) if FIXTURES.exists() else []
EXPECTED = {
    "early_sampled",
    "later_changing_ns",
    "noisy_nonuniform_weights",
}


@pytest.fixture(scope="module")
def captures():
    return {name: load_snapshot(FIXTURES / name) for name in NAMES}


def _assert_outputs(expected, actual):
    assert set(actual) == set(expected)
    for key in expected:
        np.testing.assert_allclose(
            actual[key], expected[key], rtol=1e-10, atol=1e-12
        )


def test_gp_fit_history_fixtures_present():
    assert set(NAMES) == EXPECTED


@pytest.mark.parametrize("name", NAMES)
def test_gp_fit_history_covariance_and_gp_fit_widths(captures, name):
    snap = captures[name]
    ref = snap["ref"]["portable"]
    out = portable_outputs(
        snap["pre"], snap["meta"]["hyp_n"], snap["meta"]["gp_s_N"]
    )
    assert out["covariance"] is not None
    assert out["gp_fit_widths"] is not None
    assert np.all(np.isfinite(out["gp_fit_widths"]))
    _assert_outputs(ref, out)
    assert_float64(out, f"{name}/portable", min_leaves=2)
    assert_float64(snap["pre"], f"{name}/pre", min_leaves=10)


def test_gp_fit_history_regimes(captures):
    early = captures["early_sampled"]
    assert set(early["meta"]["sources"]) == {
        "train_gp",
        "gpyreg.GP.fit",
        "capture_helper",
        "capture_script",
    }
    assert all(
        source["path"] and len(source["sha256"]) == 64
        for source in early["meta"]["sources"].values()
    )
    assert early["meta"]["capture_iteration"] > 0
    assert early["meta"]["gp_s_N"] > 0

    for name, snap in captures.items():
        sampler = snap["ref"]["sampler_widths"]
        assert_float64(sampler, f"{name}/stored sampler widths", min_leaves=2)
        assert np.any(
            sampler["effective_widths"] < sampler["widths_default"]
        ), f"{name}: history did not reduce a SliceSampler width"

    later = captures["later_changing_ns"]
    later_summary = history_block_summary(later["pre"], later["meta"]["hyp_n"])
    assert len(set(later_summary["sample_counts"])) > 1

    noisy = captures["noisy_nonuniform_weights"]
    noisy_summary = history_block_summary(noisy["pre"], noisy["meta"]["hyp_n"])
    assert noisy["pre"]["logger"]["noise_flag"]
    assert len(noisy_summary["weights"]) > 1
    assert not np.allclose(
        noisy_summary["weights"], noisy_summary["weights"][0]
    )


@pytest.mark.parametrize("name", NAMES)
def test_gp_fit_history_replay(captures, name):
    snap = captures[name]
    if not same_platform(snap) and not os.environ.get("PYVBMC_ORACLES_ALL"):
        pytest.skip(
            "stochastic GP fit is platform-bound; set PYVBMC_ORACLES_ALL=1"
        )
    out, observation = replay(snap)
    _assert_outputs(snap["ref"]["fit"], out)
    np.testing.assert_array_equal(
        observation["gp_fit_widths"],
        snap["ref"]["portable"]["gp_fit_widths"],
    )
    _assert_outputs(
        snap["ref"]["sampler_widths"],
        {
            "effective_widths": observation["effective_widths"],
            "widths_default": observation["widths_default"],
        },
    )
    assert_float64(out, f"{name}/fit", min_leaves=5)
    assert_float64(observation, f"{name}/observed widths", min_leaves=3)
