"""Recording and resume contracts; no full inference runs."""

import copy
import json
from contextlib import nullcontext
from unittest.mock import patch

import dill
import numpy as np
import population_run as runner
import pytest

from pyvbmc import VBMC


def make_vbmc(skip=False):
    vbmc = VBMC(
        lambda x: -np.sum(x**2),
        np.zeros((1, 2)),
        np.full((1, 2), -np.inf),
        np.full((1, 2), np.inf),
        np.full((1, 2), -2.0),
        np.full((1, 2), 2.0),
        seed=123,
        options={
            "display": "off",
            "performance_calibration": "off",
            "min_final_components": 1 if skip else 5,
            "ns_ent_boost": [],
            "ns_ent_fast_boost": [],
            "ns_ent_fine_boost": [],
        },
    )
    vbmc.vp.stats = {"elbo": -10.0, "elbo_sd": 0.1, "stable": True}
    return vbmc


@pytest.mark.parametrize(
    "candidate_elbo,skip,accepted",
    [(-9.0, False, True), (-12.0, False, False), (-9.0, True, False)],
)
def test_capture_preserves_production_guard_and_rng(
    candidate_elbo, skip, accepted
):
    def optimize(options, optim_state, vp, gp, *args):
        assert options["weight_penalty"] == 0
        vp.mu += vp.rng.normal(size=vp.mu.shape)
        vp.stats = {"elbo": candidate_elbo, "elbo_sd": 0.1, "stable": False}
        return vp, None, None

    outcomes = []
    for recording in (False, True):
        vbmc = make_vbmc(skip)
        global_before = copy.deepcopy(np.random.get_state())
        with patch.object(runner.VBMC_MODULE, "optimize_vp", optimize):
            with (
                runner.BoostCapture() if recording else nullcontext()
            ) as capture:
                outcome = vbmc.final_boost(vbmc.vp, object())
        assert outcome[3] == accepted
        outcomes.append((outcome, copy.deepcopy(vbmc.rng.bit_generator.state)))
        global_after = np.random.get_state()
        for a, b in zip(global_before, global_after):
            np.testing.assert_equal(a, b)
    a, b = outcomes
    np.testing.assert_array_equal(a[0][0].mu, b[0][0].mu)
    assert a[0][1:] == b[0][1:]
    assert a[1] == b[1]
    state = capture.state
    assert state["attempted"] == (not skip)
    assert state["accepted"] == accepted
    assert state["pre"].stats["elbo"] == -10
    if not skip:
        assert state["candidate"].stats["elbo"] == candidate_elbo
    # Recorded VPs are independent, including their generators and transforms.
    before = copy.deepcopy(vbmc.rng.bit_generator.state)
    state["pre"].rng.normal()
    assert vbmc.rng.bit_generator.state == before
    assert (
        state["returned"].parameter_transformer
        is not vbmc.vp.parameter_transformer
    )


def test_capture_restores_hooks_on_failure():
    vbmc = make_vbmc()
    original = VBMC.final_boost
    with patch.object(
        runner.VBMC_MODULE,
        "optimize_vp",
        side_effect=RuntimeError("test failure"),
    ) as mocked:
        with pytest.raises(RuntimeError, match="test failure"):
            with runner.BoostCapture():
                vbmc.final_boost(vbmc.vp, object())
        assert runner.VBMC_MODULE.optimize_vp is mocked
    assert VBMC.final_boost is original


def write_case(tmp_path):
    tag = "normal_D5_seed0"
    expected = {
        "candidate_sha": "test",
        "options": {"display": "off", "tol_elcbo_boost": 0.1},
    }
    side = {
        "label": "normal_D5",
        "seed": 0,
        "requested_options": expected["options"],
        "effective_options": expected["options"],
        "final": {
            k: 0.1 for k in (*runner.golden_trace.METRICS, "elbo", "elbo_sd")
        },
    }
    runner.write_json(tmp_path / f"{tag}.json", side)
    np.savez(tmp_path / f"{tag}.npz", final_mu=np.zeros((2, 1)))
    vp = make_vbmc().vp
    side["final"].update(runner.scores(vp))
    runner.write_json(tmp_path / f"{tag}.json", side)
    state = {
        "pre": vp,
        "candidate": None,
        "returned": vp,
        "attempted": False,
        "accepted": False,
        "tolerance": 0.1,
        "boost_options": None,
        "boost_call": None,
    }
    (tmp_path / f"{tag}.boost.pkl").write_bytes(dill.dumps(state))
    report = {
        k: state[k]
        for k in (
            "attempted",
            "accepted",
            "tolerance",
            "boost_options",
            "boost_call",
        )
    }
    report["scores"] = {
        k: runner.scores(state[k]) for k in ("pre", "candidate", "returned")
    }
    report["metrics"] = {
        k: {
            "elbo_err": 0.1,
            "gskl": 0.1,
            "mmtv": 0.1,
            "rmse": 0.1,
            "post_mean": [[0, 0]],
            "post_cov": [[1, 0], [0, 1]],
            "moment_method": "affine",
        }
        for k in ("pre", "returned")
    }
    runner.write_json(runner.case_path(tmp_path, tag, ".boost.json"), report)
    done = {
        "identity": expected,
        "hashes": {
            s: runner.sha256(runner.case_path(tmp_path, tag, s))
            for s in runner.SUFFIXES
        },
    }
    runner.write_json(runner.case_path(tmp_path, tag, ".complete.json"), done)
    return tag, expected


@pytest.mark.parametrize(
    "damage", ["missing", "tampered", "identity", "nan", "options"]
)
def test_resume_rejects_invalid_case(tmp_path, damage):
    tag, expected = write_case(tmp_path)
    runner.validate_case(tmp_path, tag, expected)
    if damage == "missing":
        (tmp_path / f"{tag}.boost.pkl").unlink()
    elif damage == "tampered":
        (tmp_path / f"{tag}.npz").write_bytes(b"broken")
    elif damage == "identity":
        expected["candidate_sha"] = "different"
    else:
        path = tmp_path / f"{tag}.json"
        side = json.loads(path.read_text())
        if damage == "nan":
            side["final"]["gskl"] = float("nan")
        else:
            side["effective_options"]["display"] = "iter"
        runner.write_json(path, side)
        marker = runner.case_path(tmp_path, tag, ".complete.json")
        done = json.loads(marker.read_text())
        done["hashes"][".json"] = runner.sha256(path)
        runner.write_json(marker, done)
    with pytest.raises((RuntimeError, FileNotFoundError)):
        runner.validate_case(tmp_path, tag, expected)


def test_auxiliary_records_do_not_enter_population(tmp_path):
    write_case(tmp_path)
    population = runner.golden_trace.load_population(tmp_path)
    assert population["normal_D5"]["seeds"] == [0]


def test_resume_rejects_empty_boost_report_even_with_matching_hash(tmp_path):
    tag, expected = write_case(tmp_path)
    report = runner.case_path(tmp_path, tag, ".boost.json")
    runner.write_json(report, {})
    marker = runner.case_path(tmp_path, tag, ".complete.json")
    done = json.loads(marker.read_text())
    done["hashes"][".boost.json"] = runner.sha256(report)
    runner.write_json(marker, done)
    with pytest.raises(RuntimeError, match="inconsistent boost"):
        runner.validate_case(tmp_path, tag, expected)


def test_optional_candidate_metric_failure_is_retained():
    with patch.object(
        runner, "metrics", side_effect=ValueError("invalid candidate")
    ):
        assert runner.diagnostic_metrics(None, make_vbmc().vp) == {
            "error": "ValueError: invalid candidate"
        }
        with pytest.raises(ValueError):
            runner.diagnostic_metrics(None, make_vbmc().vp, required=True)


def test_unready_manifest_cannot_launch():
    with pytest.raises(RuntimeError, match="not marked ready"):
        runner.run(None, {"launch_ready": False})
