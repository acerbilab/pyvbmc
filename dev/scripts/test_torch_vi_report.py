"""Checks for scientific comparison semantics in the artifact reader."""

import hashlib
import json

import numpy as np
import pytest
from torch_vi_report import _fine_comparison, _load_runs, _tolerance_comparison


def _array(path, name, values):
    value = np.asarray(values, dtype=np.float64)
    np.savez(path / f"{name}.npz", value=value)
    return {
        "archive": f"{name}.npz",
        "npz": "value",
        "shape": list(value.shape),
        "dtype": "float64",
        "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
    }


@pytest.mark.parametrize("offset_only", (True, False))
def test_logweight_gauge_does_not_hide_stale_relative_weights(
    tmp_path, offset_only
):
    expected = _array(tmp_path, "expected", [[-2.0, -1.0, 0.0]])
    actual = _array(
        tmp_path,
        "actual",
        [2.0, 3.0, 4.0] if offset_only else [4.0, 4.0, 4.0],
    )
    result = _tolerance_comparison(
        actual,
        expected,
        tmp_path,
        tmp_path,
        rtol=1e-10,
        atol=1e-12,
        canonicalize_singleton_vector=True,
        center_logweights=True,
    )
    assert (result["status"] == "matched") is offset_only
    assert result["raw_max_abs_before_gauge"] > 0
    assert result["shape_canonicalized"]


@pytest.mark.parametrize("same_shape", (True, False))
def test_recovered_scores_require_matching_draw_dimensions(
    tmp_path, same_shape
):
    for backend in ("numpy", "torch"):
        total = 12 if backend == "numpy" or same_shape else 18
        run = tmp_path / "runs" / backend
        run.mkdir(parents=True)
        rows = [
            {
                "seed": seed,
                "elbo": float(seed) + (1e-7 if backend == "torch" else 0),
                "samples_per_component": 6,
                "samples_total": total,
                "I_sk_shape": [1, total // 6],
                "J_sjk_shape": [1, total // 6, total // 6],
            }
            for seed in range(5)
        ]
        (run / "result.json").write_text(
            json.dumps(
                {
                    "backend": backend,
                    "device": "cpu",
                    "case": "warped",
                    "seed": 1703,
                    "rescored_candidates": {
                        "aliases": {},
                        "common_streams": True,
                        "scores": {"final:0": rows},
                    },
                }
            ),
            encoding="utf-8",
        )
    records = _load_runs(tmp_path, "auxiliary_recovery")
    comparison = _fine_comparison(records[0], records[1])
    assert comparison["available"]
    assert comparison["paired_claim_valid"] is same_shape
    assert comparison["paired_torch_minus_numpy"]["n"] == (
        5 if same_shape else 0
    )
