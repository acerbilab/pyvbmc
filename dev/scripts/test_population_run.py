"""Recording, resume and campaign-contract checks; no full inference runs.

The boost capture and the records of campaigns run before array mode are
checked on hand-made posteriors. The array-mode subcommands run on a
campaign of the ``smoke`` suite's ``normal_D2`` whose identity is fixed by
the test (the real one needs clean trees) and whose cases are written by
hand or by a stand-in for the run, so that the contract's states (the
early exit, the refusals, a stop by SIGTERM, a failure) and ``verify``,
``rescore`` and ``summarize`` are exercised without VBMC.
"""

import copy
import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import dill
import numpy as np
import population_run as runner
import pytest

from pyvbmc import VBMC

contract = runner.contract


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


def test_validate_case_compares_transformers_by_their_data(tmp_path):
    # An accepted boost returns a copy of the candidate: equal transformers
    # whose bounded transforms are different functions once unpickled.
    tag, expected = write_case(tmp_path)
    state = dill.loads((tmp_path / f"{tag}.boost.pkl").read_bytes())
    state["candidate"] = clone = copy.deepcopy(state["returned"])
    state.update(attempted=True, accepted=True, boost_options={})
    state["boost_options"]["weight_penalty"] = 0
    state["boost_call"] = {"n_fast_opts": 1, "n_slow_opts": 1, "K_new": 50}
    assert (
        clone.parameter_transformer._bounded_transforms
        != state["returned"].parameter_transformer._bounded_transforms
    )
    (tmp_path / f"{tag}.boost.pkl").write_bytes(dill.dumps(state))
    report_path = runner.case_path(tmp_path, tag, ".boost.json")
    report = json.loads(report_path.read_text())
    report.update(
        attempted=True,
        accepted=True,
        boost_options=state["boost_options"],
        boost_call=state["boost_call"],
    )
    report["scores"]["candidate"] = runner.scores(clone)
    report["metrics"]["candidate"] = report["metrics"]["pre"]
    runner.write_json(report_path, report)
    marker = runner.case_path(tmp_path, tag, ".complete.json")
    done = json.loads(marker.read_text())
    for suffix in (".boost.pkl", ".boost.json"):
        done["hashes"][suffix] = runner.sha256(
            runner.case_path(tmp_path, tag, suffix)
        )
    runner.write_json(marker, done)
    runner.validate_case(tmp_path, tag, expected)


# --------------------------------------------------------------------------
# Array mode
# --------------------------------------------------------------------------

LABEL = "normal_D2"
FAKE_IDENTITY = {
    "contract": contract.CONTRACT_VERSION,
    "source": {
        "trees": {
            name: {"commit": "0" * 40, "clean": True}
            for name in ("harness", "pyvbmc", "gpyreg")
        },
        "files": {"dev/scripts/population_run.py": "1" * 64},
        "versions": {"python": "3", "numpy": "2", "scipy": "1", "cma": "4"},
    },
    "imports": {"trees": {}, "modules": {}, "harness_modules": {}},
    "host": {"hostname": "test"},
}


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    """A prepared campaign: normal_D2 at seeds 0-2, with a fixed identity."""
    monkeypatch.setenv("PYVBMC_GPYREG_SOURCE", str(tmp_path / "gpyreg"))
    monkeypatch.delenv("PYVBMC_SOURCE", raising=False)
    monkeypatch.setattr(
        runner, "this_identity", lambda host=True: copy.deepcopy(FAKE_IDENTITY)
    )
    monkeypatch.setattr(contract, "pip_freeze", lambda: ["pyvbmc==0"])
    out = tmp_path / "campaign"
    assert (
        runner.main(
            [
                "prepare",
                "--out",
                str(out),
                "--suite",
                "smoke",
                "--labels",
                LABEL,
                "--seeds",
                "0-2",
                "--arm",
                "after",
            ]
        )
        == 0
    )
    return out


def line_of(seed):
    return f"{LABEL}/{LABEL}_seed{seed} {LABEL} {seed}"


def files_of(out, seed):
    return {
        key: out / rel for key, rel in runner.case_files(LABEL, seed).items()
    }


def complete_case(out, seed, exact_metrics=False):
    """Write a verified-looking case by hand, with a real posterior.

    With ``exact_metrics`` the sidecar's metrics are the posterior's own,
    so that the rescoring reproduces them.
    """
    vp = make_vbmc().vp
    files = files_of(out, seed)
    for path in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    transformer = vp.parameter_transformer
    np.savez_compressed(
        files["trace"],
        final_w=np.ravel(vp.w),
        final_mu=np.asarray(vp.mu),
        final_sigma=np.ravel(vp.sigma),
        final_lambd=np.ravel(vp.lambd),
        pt_mu=transformer.mu[None],
        pt_delta=transformer.delta[None],
        pt_scale=np.ones((1, vp.D)),
        pt_R=np.eye(vp.D)[None],
    )
    extras = runner.posterior_extras(vp)
    np.savez(files["posterior"], **extras)
    options = contract.read_json(out / "manifest.json")["options"]
    final = {k: 0.1 for k in runner.golden_trace.METRICS}
    final.update(
        runner.scores(vp),
        best_iter=0,
        rmse=0.1,
        iterations=1,
        final_K=vp.K,
        n_warps=0,
        wall_s=1.0,
    )
    if exact_metrics:
        problem = runner.find_config(LABEL).make(seed=seed)
        with np.load(files["trace"]) as trace:
            rebuilt = runner.returned_posterior(trace, extras, problem, 0)
        found = runner.rescore_metrics(problem, rebuilt, final["elbo"])
        final.update({k: found[k] for k in runner.RESCORED_METRICS})
    side = {
        "label": LABEL,
        "seed": seed,
        "requested_options": options,
        "effective_options": options,
        "final": final,
        "provenance": {
            "source": FAKE_IDENTITY["source"],
            "imports": FAKE_IDENTITY["imports"],
        },
    }
    files["sidecar"].write_text(json.dumps(side, indent=1))
    state = {
        "pre": vp,
        "candidate": None,
        "returned": vp,
        "attempted": False,
        "accepted": False,
        "tolerance": options["tol_elcbo_boost"],
        "boost_options": None,
        "boost_call": None,
    }
    files["boost_state"].write_bytes(dill.dumps(state))
    metric = {
        "elbo_err": 0.1,
        "gskl": 0.1,
        "mmtv": 0.1,
        "rmse": 0.1,
        "post_mean": [[0, 0]],
        "post_cov": [[1, 0], [0, 1]],
        "moment_method": "affine",
    }
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
    report["metrics"] = {"pre": metric, "returned": metric}
    runner.write_json(files["boost_report"], report)
    tag = contract.case_tag(line_of(seed))
    contract.write_completion(
        out,
        tag,
        line_of(seed),
        list(files.values()),
        copy.deepcopy(FAKE_IDENTITY),
        time.time(),
        1.0,
        {"boost": {"attempted": False, "accepted": False}},
    )
    return tag, files


def test_case_lines_subsets_and_files(campaign):
    manifest = contract.read_json(campaign / "manifest.json")
    lines = runner.case_lines(manifest)
    assert lines == [line_of(seed) for seed in range(3)]
    assert [runner.parse_case(line)[1:] for line in lines] == [
        (LABEL, 0),
        (LABEL, 1),
        (LABEL, 2),
    ]
    assert runner.subset_indices(manifest, "canary") == [1]
    assert runner.subset_indices(manifest, LABEL) == [1, 2, 3]
    assert runner.subset_indices(manifest, "noiseless") == [1, 2, 3]
    with pytest.raises(SystemExit):
        runner.subset_indices(manifest, "noisy")
    with pytest.raises(contract.ContractError):
        runner.parse_case(f"{LABEL}/{LABEL}_seed1 {LABEL} 2")
    # The sidecars of a configuration's directory are all the population
    # discovery of golden_trace.py finds; the boost files lie apart.
    rels = runner.case_files(LABEL, 0).values()
    assert sorted(r for r in rels if r.endswith(".json")) == [
        f"boost/{LABEL}/{LABEL}_seed0.boost.json",
        f"{LABEL}/{LABEL}_seed0.json",
    ]


def test_prepare_records_the_campaign(campaign, capsys):
    manifest = contract.read_json(campaign / "manifest.json")
    assert manifest["allocation"] == {
        "suite": "smoke",
        "labels": [LABEL],
        "seeds": [0, 1, 2],
    }
    assert manifest["options"] == runner.DEFAULT_OPTIONS
    assert manifest["identity"] == FAKE_IDENTITY
    assert manifest["arm"] == "after"
    # The harness checkout's own package is the release code: it rescores.
    assert manifest["finishing_steps"] == [["summarize"], ["rescore"]]
    family = manifest["confirmatory"]
    assert family["labels"] == [LABEL]
    assert family["signed_rank"] == ["elbo_err", "gskl", "mmtv"]
    assert family["tests"] == 4 and "Holm" in family["statement"]
    assert set(manifest["site"]) == set(contract.SETTINGS)
    runner.main(["cases", "--out", str(campaign), "--subset", "canary"])
    assert capsys.readouterr().out == f"1 {line_of(0)}\n"
    # Preparing again changes nothing; preparing otherwise is refused.
    before = (campaign / "manifest.json").read_bytes()
    base = ["prepare", "--out", str(campaign), "--suite", "smoke"]
    assert (
        runner.main(
            [*base, "--labels", LABEL, "--seeds", "0-2", "--arm", "after"]
        )
        == 0
    )
    assert (campaign / "manifest.json").read_bytes() == before
    with pytest.raises(SystemExit, match="prepared otherwise"):
        runner.main([*base, "--labels", LABEL, "--seeds", "0-3"])


def test_prepare_pairs_arms(campaign, tmp_path, monkeypatch):
    base = ["--suite", "smoke", "--labels", LABEL, "--seeds", "0-2"]
    after = tmp_path / "after"
    runner.main(
        ["prepare", "--out", str(after), *base, "--pair", str(campaign)]
    )
    steps = contract.read_json(after / "manifest.json")["finishing_steps"]
    assert steps == [
        ["summarize"],
        ["rescore", "--campaign", campaign.resolve().as_posix()],
    ]
    with pytest.raises(SystemExit, match="another allocation"):
        runner.main(
            [
                "prepare",
                "--out",
                str(tmp_path / "other"),
                "--suite",
                "smoke",
                "--labels",
                LABEL,
                "--seeds",
                "0-4",
                "--pair",
                str(campaign),
            ]
        )
    # A campaign of other code than the harness checkout's does not rescore.
    before_tree = tmp_path / "before_tree"
    before_tree.mkdir()
    monkeypatch.setenv("PYVBMC_SOURCE", str(before_tree))
    before = tmp_path / "before"
    runner.main(["prepare", "--out", str(before), *base])
    assert contract.read_json(before / "manifest.json")["finishing_steps"] == [
        ["summarize"]
    ]
    with pytest.raises(SystemExit, match="only a campaign"):
        runner.main(
            [
                "prepare",
                "--out",
                str(tmp_path / "x"),
                *base,
                "--pair",
                str(campaign),
            ]
        )


def test_confirmatory_family_override_and_refusals():
    family = runner.confirmatory_family(
        {"labels": ["b"], "signed_rank": ["gskl"], "alpha": 0.01},
        ["a", "b"],
    )
    assert family["tests"] == 2 and family["alpha"] == 0.01
    assert runner.confirmatory_family(family, ["a", "b"]) == family
    for spec in (
        {"labels": ["c"]},
        {"signed_rank": ["rmse"]},
        {"correction": "bonferroni"},
        {"alpha": 1.5},
        {"primary": ["x"]},
        {"signed_rank": [], "mcnemar_usability": False},
    ):
        with pytest.raises(SystemExit):
            runner.confirmatory_family(spec, ["a", "b"])


def fail_if_run(*args, **kwargs):
    raise AssertionError("the case must not run")


def test_worker_exits_early_on_a_completed_case(campaign, monkeypatch):
    tag = contract.case_tag(line_of(0))
    contract.write_json(contract.record_path(campaign, tag), {"tag": tag})
    monkeypatch.setattr(runner, "run_case", fail_if_run)
    args = ["worker", "--out", str(campaign), "--case", line_of(0)]
    assert runner.main(args) == 0


def test_worker_refuses_a_live_claim_and_leaves_the_files(
    campaign, monkeypatch
):
    tag = contract.case_tag(line_of(0))
    partial = files_of(campaign, 0)["trace"]
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"left by another attempt")
    claim = contract.claim_path(campaign, tag)
    contract.write_json(
        claim,
        {
            "tag": tag,
            "job": None,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "started": contract.now(),
            "token": "someone else",
        },
    )
    monkeypatch.setattr(runner, "run_case", fail_if_run)
    args = ["worker", "--out", str(campaign), "--case", line_of(0)]
    assert runner.main(args) == contract.EXIT_CLAIMED
    assert partial.read_bytes() == b"left by another attempt"
    assert contract.read_json(claim)["token"] == "someone else"


def test_worker_refuses_another_identity_and_leaves_the_files(
    campaign, monkeypatch
):
    partial = files_of(campaign, 0)["trace"]
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"left")
    other = copy.deepcopy(FAKE_IDENTITY)
    other["source"]["trees"]["pyvbmc"]["commit"] = "f" * 40
    monkeypatch.setattr(runner, "this_identity", lambda host=True: other)
    monkeypatch.setattr(runner, "run_case", fail_if_run)
    args = ["worker", "--out", str(campaign), "--case", line_of(0)]
    assert runner.main(args) == contract.EXIT_IDENTITY
    assert partial.read_bytes() == b"left"
    assert not contract.claim_path(
        campaign, contract.case_tag(line_of(0))
    ).exists()


def test_worker_stopped_by_sigterm_cleans_up(campaign, monkeypatch):
    def stopped(out, label, seed, options, identity):
        files = files_of(Path(out), seed)
        files["trace"].parent.mkdir(parents=True, exist_ok=True)
        files["trace"].write_bytes(b"partial")
        signal.raise_signal(signal.SIGTERM)
        time.sleep(5)
        raise AssertionError("SIGTERM did not stop the run")

    monkeypatch.setattr(runner, "run_case", stopped)
    tag = contract.case_tag(line_of(1))
    args = ["worker", "--out", str(campaign), "--case", line_of(1)]
    assert runner.main(args) == 128 + signal.SIGTERM
    assert not files_of(campaign, 1)["trace"].exists()
    assert not contract.error_path(campaign, tag).exists()
    assert not contract.claim_path(campaign, tag).exists()
    assert not contract.record_path(campaign, tag).exists()


def test_worker_failure_keeps_the_traceback_of_run_task(campaign, monkeypatch):
    def failed(label, seed, options, directory):
        stem = f"{label}_seed{seed}"
        (Path(directory) / f"{stem}.npz").write_bytes(b"partial")
        (Path(directory) / f"{stem}.error.txt").write_text(
            "Traceback (most recent call last):\nValueError: the target broke\n"
        )
        return {"tag": stem, "ok": False, "wall_s": 0.0}

    monkeypatch.setattr(runner.golden_trace, "run_task", failed)
    tag = contract.case_tag(line_of(2))
    args = ["worker", "--out", str(campaign), "--case", line_of(2)]
    assert runner.main(args) == 1
    text = contract.error_path(campaign, tag).read_text()
    assert "golden_trace.run_task failed" in text
    assert "ValueError: the target broke" in text
    assert not files_of(campaign, 2)["trace"].exists()
    assert not contract.claim_path(campaign, tag).exists()


def test_worker_success_writes_the_record(campaign, monkeypatch):
    def succeeded(out, label, seed, options, identity):
        files = files_of(Path(out), seed)
        for path in files.values():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(path.name)
        return list(files.values()), {"boost": {"attempted": True}}

    monkeypatch.setattr(runner, "run_case", succeeded)
    tag = contract.case_tag(line_of(0))
    args = ["worker", "--out", str(campaign), "--case", line_of(0)]
    assert runner.main(args) == 0
    record = contract.read_json(contract.record_path(campaign, tag))
    assert sorted(record["artifacts"]) == sorted(
        runner.case_files(LABEL, 0).values()
    )
    assert record["boost"] == {"attempted": True}
    assert not contract.claim_path(campaign, tag).exists()
    unknown = ["worker", "--out", str(campaign), "--case", line_of(7)]
    assert runner.main(unknown) == runner.EXIT_USAGE


def verification(out):
    report = contract.read_json(out / "verification.json")
    return report, {case["index"]: case for case in report["cases"]}


def test_verify_reconciles_every_state(campaign):
    complete_case(campaign, 0)
    tag = contract.case_tag(line_of(1))
    contract.error_path(campaign, tag).write_text("x: RuntimeError: boom\n")
    assert runner.main(["verify", "--out", str(campaign)]) == 0
    report, cases = verification(campaign)
    assert [cases[i]["status"] for i in (1, 2, 3)] == [
        "verified",
        "failed",
        "missing",
    ]
    assert cases[1]["boost"] == {"attempted": False, "accepted": False}
    assert report["manifest_sha256"] == runner.sha256(
        campaign / "manifest.json"
    )
    # Files without a record or a claim, and a file no case owns.
    files_of(campaign, 2)["trace"].write_bytes(b"x")
    (campaign / LABEL / "notes.txt").write_text("x")
    assert runner.main(["verify", "--out", str(campaign)]) == 1
    report, cases = verification(campaign)
    assert cases[3]["status"] == "partial"
    assert report["stray"] == [f"{LABEL}/notes.txt"]
    # A stale claim beside them: the case was interrupted.
    contract.write_json(
        contract.claim_path(campaign, contract.case_tag(line_of(2))),
        {"job": None, "host": socket.gethostname(), "pid": 2**30},
    )
    (campaign / LABEL / "notes.txt").unlink()
    assert runner.main(["verify", "--out", str(campaign)]) == 0
    assert verification(campaign)[1][3]["status"] == "interrupted"


def test_verify_fails_a_changed_or_inconsistent_case(campaign):
    _, files = complete_case(campaign, 0)
    side = json.loads(files["sidecar"].read_text())
    side["final"]["gskl"] = 0.2
    files["sidecar"].write_text(json.dumps(side, indent=1))
    assert runner.main(["verify", "--out", str(campaign)]) == 1
    case = verification(campaign)[1][1]
    assert case["status"] == "verify_failed"
    assert "differs from its recorded SHA-256" in case["error"]


def test_verify_checks_the_rebuilt_posterior(campaign):
    tag, files = complete_case(campaign, 0)
    # Arrays that claim a rescaling the returned posterior does not have.
    np.savez(
        files["posterior"],
        bounded_types=np.array([12]),
        scale_is_none=np.asarray(False),
        rotation_is_none=np.asarray(True),
    )
    record_path = contract.record_path(campaign, tag)
    record = contract.read_json(record_path)
    rel = runner.case_files(LABEL, 0)["posterior"]
    record["artifacts"][rel]["sha256"] = runner.sha256(files["posterior"])
    contract.write_json(record_path, record)
    assert runner.main(["verify", "--out", str(campaign)]) == 1
    case = verification(campaign)[1][1]
    assert "parameter_transformer.scale" in case["error"]


def test_verify_refuses_other_trees(campaign, monkeypatch):
    other = copy.deepcopy(FAKE_IDENTITY)
    other["source"]["trees"]["gpyreg"]["commit"] = "e" * 40
    monkeypatch.setattr(runner, "this_identity", lambda host=True: other)
    assert (
        runner.main(["verify", "--out", str(campaign)])
        == contract.EXIT_IDENTITY
    )
    assert not (campaign / "verification.json").exists()


def test_rescore_and_summarize_a_verified_campaign(campaign, tmp_path):
    complete_case(campaign, 0, exact_metrics=True)
    assert runner.main(["verify", "--out", str(campaign)]) == 0
    assert runner.main(["summarize", "--out", str(campaign)]) == 0
    summary = (campaign / "summary.md").read_text()
    assert f"| {LABEL} | 1 |" in summary
    assert "verified 1, missing 2" in summary
    assert "skipped 1" in summary
    assert runner.main(["rescore", "--out", str(campaign)]) == 0
    report = contract.read_json(
        campaign / "rescored" / f"{campaign.name}.json"
    )
    assert report["counts"]["rescored"] == 1
    assert report["counts"]["not_verified"] == 2
    assert report["counts"]["equal_to_in_run"] == {
        k: 1 for k in runner.RESCORED_METRICS
    }
    case = report["cases"][f"{LABEL}_seed0"]
    assert case["moment_method"] == "affine"
    assert report["campaign"]["verification_sha256"] == runner.sha256(
        campaign / "verification.json"
    )
    assert report["rescoring"]["identity"] == FAKE_IDENTITY
    # A sidecar changed after verification is refused.
    path = files_of(campaign, 0)["sidecar"]
    path.write_text(path.read_text() + " ")
    with pytest.raises(contract.ContractError, match="not the file"):
        runner.main(["rescore", "--out", str(campaign)])


def test_rescore_runs_in_the_release_code_alone(
    campaign, tmp_path, monkeypatch
):
    other = tmp_path / "other_tree"
    other.mkdir()
    monkeypatch.setenv("PYVBMC_SOURCE", str(other))
    assert (
        runner.main(["rescore", "--out", str(campaign)])
        == contract.EXIT_IDENTITY
    )


def test_returned_posterior_rebuilds_a_posterior_exactly():
    vp = make_vbmc().vp
    problem = runner.find_config(LABEL).make(seed=0)
    transformer = vp.parameter_transformer
    trace = {
        "final_w": np.ravel(vp.w),
        "final_mu": np.asarray(vp.mu),
        "final_sigma": np.ravel(vp.sigma),
        "final_lambd": np.ravel(vp.lambd),
        "pt_mu": np.stack([np.full(2, 7.0), transformer.mu]),
        "pt_delta": np.stack([np.ones(2), transformer.delta]),
        "pt_scale": np.ones((2, 2)),
        "pt_R": np.stack([np.eye(2)] * 2),
    }
    extras = runner.posterior_extras(vp)
    rebuilt = runner.returned_posterior(trace, extras, problem, 1)
    assert runner.posterior_differences(rebuilt, vp) == []
    # The iteration matters, and so does every array.
    wrong = runner.returned_posterior(trace, extras, problem, 0)
    assert runner.posterior_differences(wrong, vp) == [
        "parameter_transformer.delta",
        "parameter_transformer.mu",
    ]
    rebuilt.sigma = rebuilt.sigma * 2
    assert runner.posterior_differences(rebuilt, vp) == ["sigma"]


def test_two_trees_keep_the_package_tree_first(tmp_path):
    # A stand-in package tree: importing population_run must take PyVBMC
    # from it, although golden_trace and profile_run put the harness
    # checkout first on sys.path when they are imported.
    tree = tmp_path / "tree"
    (tree / "pyvbmc" / "vbmc").mkdir(parents=True)
    (tree / "pyvbmc" / "__init__.py").write_text(
        "from .vbmc.vbmc import VBMC\n"
    )
    (tree / "pyvbmc" / "vbmc" / "__init__.py").write_text("")
    (tree / "pyvbmc" / "vbmc" / "vbmc.py").write_text(
        "class VBMC:\n    pass\n\n\ndef optimize_vp(*args):\n    pass\n"
    )
    script = textwrap.dedent(
        """
        import sys
        from pathlib import Path
        import population_run
        import pyvbmc
        tree = Path(sys.argv[1]).resolve()
        assert Path(pyvbmc.__file__).resolve().parents[1] == tree, pyvbmc
        assert Path(sys.path[0]).resolve() == tree, sys.path[:3]
        assert Path(sys.path[1]).resolve() == population_run.ROOT
        print("ok")
        """
    )
    env = dict(os.environ, PYVBMC_SOURCE=str(tree))
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", script, str(tree)],
        cwd=runner.HERE,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
