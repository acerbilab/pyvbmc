import copy
from pathlib import Path

import numpy as np
import pytest

from pyvbmc.testing.oracles._oracles import compare, oracle_error_scale
from pyvbmc.testing.oracles._state import load_snapshot

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def corr_snapshot():
    return load_snapshot(FIXTURES / "corr_D5_warped")


def _reference_and_scale(snapshot, oracle="acq_AcqFcn"):
    reference = np.asarray(snapshot["ref"][oracle]["acq"])
    scale = oracle_error_scale(snapshot, oracle)["acq"]
    return reference, scale


def test_conditioned_recorded_scale_accepts_rounding_not_material_error(
    corr_snapshot,
):
    reference, scale = _reference_and_scale(corr_snapshot)
    index = 252
    assert reference[index] == -1.4157800769380085e-72
    rounding = reference.copy()
    rounding[index] = -1.4193518562966716e-72
    raw = compare({"acq": reference}, {"acq": rounding}, 1e-3, 0.0)
    conditioned = compare(
        {"acq": reference},
        {"acq": rounding},
        1e-3,
        0.0,
        error_scale={"acq": scale},
    )
    assert not raw[0][3]
    assert conditioned[0][3]

    material = reference.copy()
    material[index] *= 1 + 2e-3 * scale[index]
    conditioned = compare(
        {"acq": reference},
        {"acq": material},
        1e-3,
        0.0,
        error_scale={"acq": scale},
    )
    assert not conditioned[0][3]

    missing_penalty = reference.copy()
    missing_penalty[index] = -4.642303304721911e-9
    conditioned = compare(
        {"acq": reference},
        {"acq": missing_penalty},
        1e-3,
        0.0,
        error_scale={"acq": scale},
    )
    assert not conditioned[0][3]


def test_unregularized_and_missing_scale_keep_original_gate(corr_snapshot):
    reference, scale = _reference_and_scale(corr_snapshot)
    floor = np.quantile(np.abs(reference[np.isfinite(reference)]), 0.25)
    candidates = np.flatnonzero(
        (scale == 1) & (np.abs(reference) > max(floor, np.finfo(float).tiny))
    )
    index = candidates[np.argmax(np.abs(reference[candidates]))]
    output = reference.copy()
    output[index] *= 1 + 2e-3

    supplied = compare(
        {"acq": reference},
        {"acq": output},
        1e-3,
        0.0,
        error_scale={"acq": scale},
    )
    missing = compare(
        {"acq": reference},
        {"acq": output},
        1e-3,
        0.0,
        error_scale={"other": np.ones_like(reference)},
    )
    assert not supplied[0][3]
    assert not missing[0][3]


def test_oracle_error_scale_scope_and_key_precedence(corr_snapshot):
    regular = oracle_error_scale(corr_snapshot, "acq_AcqFcn")["acq"]
    noisy = oracle_error_scale(corr_snapshot, "acq_AcqFcnNoisy")["acq"]
    zero = corr_snapshot["ref"]["acq_AcqFcn"]["acq"] == 0
    assert np.any(zero)
    assert np.all(regular[zero] == 1)
    common = (regular > 1) & (noisy > 1)
    assert np.any(common)
    assert np.all(noisy[common] == regular[common] + 1)
    assert oracle_error_scale(corr_snapshot, "acq_AcqFcnVIQR") is None

    canonical_false = copy.deepcopy(corr_snapshot)
    canonical_false["optim_state"]["variance_regularized_acq_fcn"] = False
    assert oracle_error_scale(canonical_false, "acq_AcqFcn") is None

    inactive = copy.deepcopy(corr_snapshot)
    inactive["optim_state"].pop("variance_regularized_acqfcn")
    assert oracle_error_scale(inactive, "acq_AcqFcn") is None


def test_oracle_error_scale_requires_valid_penalty(corr_snapshot):
    missing = copy.deepcopy(corr_snapshot)
    missing["optim_state"].pop("tol_gp_var")
    with pytest.raises(KeyError, match="tol_gp_var"):
        oracle_error_scale(missing, "acq_AcqFcn")

    for value in (0.0, np.nan):
        invalid = copy.deepcopy(corr_snapshot)
        invalid["optim_state"]["tol_gp_var"] = value
        with pytest.raises(ValueError, match="tol_gp_var"):
            oracle_error_scale(invalid, "acq_AcqFcn")


def test_nonfinite_reference_patterns_remain_exact():
    reference = np.array([np.inf, -np.inf, np.nan, 1.0])
    matching = reference.copy()
    scale = np.full_like(reference, 1e6)
    rows = compare(
        {"x": reference},
        {"x": matching},
        1e-3,
        0.0,
        error_scale={"x": scale},
    )
    assert rows[0][3]

    for index, value in ((0, -np.inf), (1, np.inf), (2, 0.0)):
        wrong_pattern = matching.copy()
        wrong_pattern[index] = value
        rows = compare(
            {"x": reference},
            {"x": wrong_pattern},
            1e-3,
            0.0,
            error_scale={"x": scale},
        )
        assert not rows[0][3]


@pytest.mark.parametrize(
    "bad_scale",
    [np.ones(2), np.array([np.nan]), np.array([0.5])],
)
def test_compare_rejects_invalid_error_scale(bad_scale):
    with pytest.raises(ValueError, match="error scale"):
        compare(
            {"x": np.ones(1)},
            {"x": np.ones(1)},
            1e-3,
            0.0,
            error_scale={"x": bad_scale},
        )


def test_exact_comparison_ignores_error_scale_and_detects_one_ulp():
    reference = np.array([1.0])
    output = np.nextafter(reference, np.inf)
    rows = compare(
        {"x": reference},
        {"x": output},
        0.0,
        0.0,
        error_scale={"x": np.array([np.inf])},
    )
    assert not rows[0][3]
