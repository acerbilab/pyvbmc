"""Focused synthetic checks for the repository-only golden replay gate."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEV_SCRIPTS = REPO_ROOT / "dev" / "scripts"
if not (DEV_SCRIPTS / "golden_replay.py").exists():
    pytest.skip(
        "golden replay development scripts are absent",
        allow_module_level=True,
    )
sys.path.insert(0, str(DEV_SCRIPTS))
import golden_replay  # noqa: E402

EXPECTED_NPZ_KEYS = {
    "iter",
    "elbo",
    "elbo_sd",
    "sKL",
    "r_index",
    "stable",
    "warmup",
    "Ns_gp",
    "func_count",
    "n_eff",
    "pruned",
    "K",
    "N",
    "warped",
    "timer",
    "pt_mu",
    "pt_delta",
    "pt_scale",
    "pt_R",
    "gp_hyp",
    "gp_hyp_iter",
    "vp_w",
    "vp_mu",
    "vp_sigma",
    "vp_iter",
    "vp_lambd",
    "X_orig",
    "y_orig",
    "X_init",
    "y_init",
    "final_w",
    "final_mu",
    "final_sigma",
    "final_lambd",
    "post_mean",
    "post_cov",
}


def _arrays():
    per_iter = {
        key: np.array([value, value + 1.0])
        for key, value in {
            "elbo": -2.0,
            "elbo_sd": 0.1,
            "sKL": 0.2,
            "r_index": 0.3,
            "stable": 0.0,
            "warmup": 0.0,
            "Ns_gp": 2.0,
            "func_count": 2.0,
            "n_eff": 4.0,
            "pruned": 0.0,
            "K": 1.0,
            "N": 2.0,
            "warped": 0.0,
        }.items()
    }
    per_iter["iter"] = np.arange(2)
    per_iter["func_count"] = np.array([2.0, 3.0])
    per_iter["Ns_gp"] = np.array([3.0, 2.0])
    per_iter["K"] = np.ones(2)
    per_iter["N"] = np.array([2.0, 3.0])
    per_iter["warped"] = np.zeros(2)
    arrays = {
        **per_iter,
        "timer": np.arange(4.0).reshape(2, 2),
        "pt_mu": np.zeros((2, 1)),
        "pt_delta": np.ones((2, 1)),
        "pt_scale": np.ones((2, 1)),
        "pt_R": np.ones((2, 1, 1)),
        "gp_hyp": np.arange(4.0).reshape(2, 2),
        "gp_hyp_iter": np.arange(2),
        "vp_w": np.ones(2),
        "vp_mu": np.arange(2.0).reshape(2, 1),
        "vp_sigma": np.ones(2),
        "vp_iter": np.arange(2),
        "vp_lambd": np.ones((2, 1)),
        "X_orig": np.arange(3.0).reshape(3, 1),
        "y_orig": -np.arange(3.0),
        "X_init": np.arange(2.0).reshape(2, 1),
        "y_init": -np.arange(2.0),
        "final_w": np.ones(1),
        "final_mu": np.zeros((1, 1)),
        "final_sigma": np.ones(1),
        "final_lambd": np.ones(1),
        "post_mean": np.zeros((1, 1)),
        "post_cov": np.ones((1, 1)),
    }
    assert set(arrays) == EXPECTED_NPZ_KEYS
    return arrays


def _final():
    return {
        "elbo": -1.0,
        "elbo_sd": 0.05,
        "best_iter": 1,
        "iterations": 2,
        "func_count": 3,
        "final_K": 1,
        "final_N": 3,
        "min_Ns_gp": 2,
        "n_warps": 0,
        "success_flag": True,
        "message": "done",
        "wall_s": 1.0,
        "target_eval_s": 0.1,
        "elbo_err": 0.1,
        "gskl": 0.2,
        "mmtv": 0.3,
        "rmse": 0.4,
        "moment_method": "affine",
        "peak_rss_mb": 10.0,
    }


def _copy_arrays(arrays):
    return {key: value.copy() for key, value in arrays.items()}


def _change_first(array):
    changed = array.copy()
    changed.flat[0] += 1
    return changed


@pytest.mark.parametrize("key", sorted(EXPECTED_NPZ_KEYS - {"timer"}))
def test_every_non_timer_array_participates_in_exact_identity(key):
    reference = _arrays()
    new = _copy_arrays(reference)
    new[key] = _change_first(new[key])

    result = golden_replay.compare_traces(reference, new)

    differences = {
        **result["loop_differences"],
        **result["final_array_differences"],
    }
    assert differences[key] == "value"
    assert not result["stored_arrays_identical"]


def test_array_shape_and_timer_rules_and_nan_sentinels():
    reference = _arrays()
    new = _copy_arrays(reference)
    new["elbo"] = new["elbo"][:-1]
    result = golden_replay.compare_traces(reference, new)
    assert result["loop_differences"]["elbo"].startswith("shape ")

    new = _copy_arrays(reference)
    new["timer"] = np.zeros((9, 7))
    result = golden_replay.compare_traces(reference, new)
    assert result["stored_arrays_identical"]

    reference["sKL"][0] = np.nan
    new = _copy_arrays(reference)
    assert golden_replay.compare_traces(reference, new)[
        "stored_arrays_identical"
    ]
    reference["final_mu"][0, 0] = np.nan
    new = _copy_arrays(reference)
    assert not golden_replay.compare_traces(reference, new)[
        "stored_arrays_identical"
    ]


@pytest.mark.parametrize("key", golden_replay.SEMANTIC_FINAL_KEYS)
def test_every_semantic_sidecar_field_participates_in_identity(key):
    reference = _final()
    new = copy.deepcopy(reference)
    if isinstance(new[key], bool):
        new[key] = not new[key]
    elif isinstance(new[key], str):
        new[key] += " changed"
    elif isinstance(new[key], int):
        new[key] += 1
    else:
        new[key] = np.nextafter(new[key], np.inf)

    differences = golden_replay._semantic_final_differences(reference, new)

    assert differences == {key: "value"}


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("iterations", 3),
        ("func_count", 4),
        ("final_K", 2),
        ("final_N", 4),
        ("min_Ns_gp", 1),
        ("n_warps", 1),
        ("best_iter", 2),
    ],
)
def test_sidecar_counts_are_cross_checked_against_npz_semantics(key, value):
    final = _final()
    final[key] = value

    issues = golden_replay._trace_consistency_issues(_arrays(), final)

    assert key in issues


def test_returned_transformer_missing_is_explicitly_not_certifiable():
    arrays = _arrays()
    result = golden_replay.compare_traces(arrays, _copy_arrays(arrays))
    assert result["stored_arrays_identical"]
    assert not result["returned_transformer_certifiable"]
    assert "not certifiable" in result["returned_transformer_coverage"]

    reference = _copy_arrays(arrays)
    new = _copy_arrays(arrays)
    transformer = {
        "final_pt_mu": np.zeros(1),
        "final_pt_delta": np.ones(1),
        "final_pt_scale": np.ones(1),
        "final_pt_R": np.eye(1),
    }
    reference.update(transformer)
    new.update(_copy_arrays(transformer))
    result = golden_replay.compare_traces(reference, new)
    assert result["returned_transformer_certifiable"]
    assert result["final_arrays_identical"]
    new["final_pt_mu"][0] = np.inf
    assert (
        golden_replay._final_validity_issues(new, _final())["final_pt_mu"]
        == "non-finite returned array"
    )


def _write_run(directory, arrays, final):
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "synthetic_seed0.npz", **arrays)
    (directory / "synthetic_seed0.json").write_text(
        json.dumps({"final": final}), encoding="utf-8"
    )


def test_compare_run_classifies_same_loop_changed_final_and_applies_fence(
    tmp_path,
):
    baseline = tmp_path / "baseline"
    out = tmp_path / "out"
    reference = _final()
    changed = copy.deepcopy(reference)
    changed["message"] = "different return"
    changed["gskl"] = 10.0
    _write_run(baseline, _arrays(), reference)
    _write_run(out, _arrays(), changed)
    pop = {
        "synthetic": {
            "func_count": np.array([3.0, 3.0, 3.0, 3.0]),
            "elbo_err": np.array([0.1, 0.1, 0.1, 0.1]),
            "gskl": np.array([0.2, 0.2, 0.2, 0.2]),
            "mmtv": np.array([0.3, 0.3, 0.3, 0.3]),
        }
    }

    row = golden_replay.compare_run(
        "synthetic", 0, out, baseline, tmp_path / "sidecars", pop
    )

    assert row["loop_identical"]
    assert not row["identical"]
    assert row["semantic_final_differences"] == {"message": "value"}
    assert row["verdict"].startswith("same loop, changed final")
    assert row["outside"] == ["gskl"]
    assert row["flagged"]


def test_timer_and_report_only_metadata_do_not_part_identity(tmp_path):
    baseline = tmp_path / "baseline"
    out = tmp_path / "out"
    reference_arrays = _arrays()
    new_arrays = _copy_arrays(reference_arrays)
    new_arrays["timer"] += 1000
    reference_final = _final()
    new_final = copy.deepcopy(reference_final)
    new_final["wall_s"] = 999.0
    new_final["peak_rss_mb"] = 999.0
    _write_run(baseline, reference_arrays, reference_final)
    _write_run(out, new_arrays, new_final)

    row = golden_replay.compare_run(
        "synthetic", 0, out, baseline, tmp_path / "sidecars", {}
    )

    assert row["identical"]
    assert row["verdict"].startswith("identical stored loop and final")
    assert not row["flagged"]


def test_semantics_use_sidecar_sibling_of_selected_baseline_trace(tmp_path):
    baseline = tmp_path / "baseline"
    out = tmp_path / "out"
    sidecars = tmp_path / "population_sidecars"
    reference = _final()
    population_reference = copy.deepcopy(reference)
    population_reference["message"] = "from a different population"
    _write_run(baseline, _arrays(), reference)
    _write_run(out, _arrays(), reference)
    sidecars.mkdir()
    (sidecars / "synthetic_seed0.json").write_text(
        json.dumps({"final": population_reference}), encoding="utf-8"
    )

    row = golden_replay.compare_run(
        "synthetic", 0, out, baseline, sidecars, {}
    )

    assert row["semantic_final_differences"] == {}
    assert row["identical"]


def test_missing_selected_baseline_sidecar_is_not_population_certified(
    tmp_path,
):
    baseline = tmp_path / "baseline"
    out = tmp_path / "out"
    sidecars = tmp_path / "population_sidecars"
    baseline.mkdir()
    np.savez_compressed(baseline / "synthetic_seed0.npz", **_arrays())
    _write_run(out, _arrays(), _final())
    sidecars.mkdir()
    (sidecars / "synthetic_seed0.json").write_text(
        json.dumps({"final": _final()}), encoding="utf-8"
    )

    row = golden_replay.compare_run(
        "synthetic", 0, out, baseline, sidecars, {}
    )

    assert row["loop_identical"]
    assert not row["semantic_final_certifiable"]
    assert row["semantic_final_identical"] is None
    assert not row["identical"]
    assert row["identity_not_certifiable"] == [
        "returned transformer not certifiable (absent from reference and "
        "new trace)",
        "semantic reference sidecar absent",
    ]


def test_nonfinite_semantic_final_is_flagged_without_population(tmp_path):
    baseline = tmp_path / "baseline"
    out = tmp_path / "out"
    reference = _final()
    new = copy.deepcopy(reference)
    new["elbo"] = np.nan
    _write_run(baseline, _arrays(), reference)
    _write_run(out, _arrays(), new)

    row = golden_replay.compare_run(
        "synthetic", 0, out, baseline, tmp_path / "sidecars", {}
    )

    assert row["final_validity_issues"]["new"] == {
        "elbo": "non-finite or non-numeric value nan"
    }
    assert row["flagged"]
    assert "NONFINITE final output" in row["verdict"]


def test_finals_only_still_flags_nonfinite_semantic_final(tmp_path):
    baseline = tmp_path / "baseline"
    out = tmp_path / "out"
    baseline.mkdir()
    (baseline / "synthetic_seed0.json").write_text(
        json.dumps({"final": _final()}), encoding="utf-8"
    )
    new = _final()
    new["elbo_sd"] = np.inf
    _write_run(out, _arrays(), new)

    row = golden_replay.compare_run(
        "synthetic", 0, out, baseline, tmp_path / "sidecars", {}
    )

    assert "elbo_sd" in row["final_validity_issues"]["new"]
    assert row["flagged"]


@pytest.mark.parametrize("key", sorted(golden_replay.FINAL_ARRAY_KEYS))
def test_nonfinite_returned_arrays_are_flagged(key, tmp_path):
    baseline = tmp_path / "baseline"
    out = tmp_path / "out"
    new_arrays = _arrays()
    new_arrays[key].flat[0] = np.inf
    _write_run(baseline, _arrays(), _final())
    _write_run(out, new_arrays, _final())

    row = golden_replay.compare_run(
        "synthetic", 0, out, baseline, tmp_path / "sidecars", {}
    )

    assert row["final_validity_issues"]["new"][key] == (
        "non-finite returned array"
    )
    assert row["flagged"]
