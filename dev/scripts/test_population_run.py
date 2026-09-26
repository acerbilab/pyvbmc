"""Recording, resume and campaign-contract checks of the population harness.

The boost capture and the records of campaigns run before array mode are
checked on hand-made posteriors. The array-mode subcommands run on
campaigns of the ``smoke`` suite's ``normal_D2`` whose identity is fixed by
the test (the real one needs clean trees) and whose cases are written by
hand or by a stand-in for the run, so that the contract's states (the
early exit, the refusals, a stop by SIGTERM, a failure), ``verify``,
``rescore``, ``summarize``, the comparison of two arms that the harness
verified and rescored, and a campaign's use as the envelope population of
``golden_replay.py --sidecars`` are exercised without VBMC. One such campaign,
prepared at a stand-in site (``campaign_slurm_stubs.FakeSite``), has its
tracked copies redacted.

Two real runs of one case (``normal_D2`` at seed 0, a few seconds of
inference each), made once for the module through the command line, one by
``run`` with its finishing steps and one by ``worker --case``, give the
artifacts that the rescoring must reproduce and that the two paths must
share, and the run that a stand-in replays to seed a failed check. They
take the gpyreg checkout from ``PYVBMC_GPYREG_SOURCE``, or else the git
checkout the imported gpyreg lies in, and skip without either.
"""

import copy
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import textwrap
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import campaign_slurm_stubs as stubs
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
    # Transformers whose data differ still differ.
    clone.parameter_transformer.mu = clone.parameter_transformer.mu + 1.0
    (tmp_path / f"{tag}.boost.pkl").write_bytes(dill.dumps(state))
    done["hashes"][".boost.pkl"] = runner.sha256(tmp_path / f"{tag}.boost.pkl")
    runner.write_json(marker, done)
    with pytest.raises(AssertionError):
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
#: The identity of an arm of other code than the harness checkout's: its
#: package tree and its gpyreg are at other commits.
OTHER_IDENTITY = copy.deepcopy(FAKE_IDENTITY)
OTHER_IDENTITY["source"]["trees"]["pyvbmc"]["commit"] = "b" * 40
OTHER_IDENTITY["source"]["trees"]["gpyreg"]["commit"] = "c" * 40
#: The allocation of every campaign of the module but the real runs'.
ARGUMENTS = ["--suite", "smoke", "--labels", LABEL, "--seeds", "0-2"]


def use_identity(monkeypatch, identity):
    """Make ``identity`` this process's (the real one needs clean trees)."""
    monkeypatch.setattr(
        runner, "this_identity", lambda host=True: copy.deepcopy(identity)
    )


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    """A prepared campaign: normal_D2 at seeds 0-2, with a fixed identity."""
    monkeypatch.setenv("PYVBMC_GPYREG_SOURCE", str(tmp_path / "gpyreg"))
    monkeypatch.delenv("PYVBMC_SOURCE", raising=False)
    use_identity(monkeypatch, FAKE_IDENTITY)
    monkeypatch.setattr(contract, "pip_freeze", lambda: ["pyvbmc==0"])
    out = tmp_path / "campaign"
    assert (
        runner.main(
            ["prepare", "--out", str(out), *ARGUMENTS, "--arm", "after"]
        )
        == 0
    )
    return out


def prepare_other_arm(tmp_path, monkeypatch, name, identity, extra=()):
    """Prepare, at ``tmp_path / name``, an arm whose package tree is not
    the harness checkout (``PYVBMC_SOURCE`` names a stand-in) and whose
    identity is ``identity``; this process's identity is then
    :data:`FAKE_IDENTITY` again, with ``PYVBMC_SOURCE`` unset."""
    tree = tmp_path / f"{name}_tree"
    tree.mkdir(exist_ok=True)
    monkeypatch.setenv("PYVBMC_SOURCE", str(tree))
    use_identity(monkeypatch, identity)
    out = tmp_path / name
    arguments = [*ARGUMENTS, "--arm", name, *extra]
    assert runner.main(["prepare", "--out", str(out), *arguments]) == 0
    monkeypatch.delenv("PYVBMC_SOURCE")
    use_identity(monkeypatch, FAKE_IDENTITY)
    return out


def line_of(seed):
    return f"{LABEL}/{LABEL}_seed{seed} {LABEL} {seed}"


def files_of(out, seed):
    return {
        key: out / rel for key, rel in runner.case_files(LABEL, seed).items()
    }


def complete_case(out, seed, exact_metrics=False, identity=None, shift=0.0):
    """Write a verified-looking case by hand, with a real posterior.

    With ``exact_metrics`` the sidecar's metrics, and the boost report's of
    the returned posterior, are those that ``benchmark_targets.metrics``
    gives the posterior itself, as ``golden_trace.run_task`` computes them,
    so that a rescoring, which rebuilds the posterior from its plain
    arrays, reproduces them; otherwise they are 0.1. ``identity`` is the
    one its record and its sidecar's provenance hold,
    :data:`FAKE_IDENTITY` by default; ``shift`` moves the posterior's
    means.
    """
    identity = FAKE_IDENTITY if identity is None else identity
    vp = make_vbmc().vp
    vp.mu = vp.mu + shift
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
        success_flag=True,
        peak_rss_mb=100.0,
    )
    metric = {
        "elbo_err": 0.1,
        "gskl": 0.1,
        "mmtv": 0.1,
        "rmse": 0.1,
        "post_mean": [[0, 0]],
        "post_cov": [[1, 0], [0, 1]],
        "moment_method": "affine",
    }
    if exact_metrics:
        problem = runner.find_config(LABEL).make(seed=seed)
        found = runner.jsonable(runner.metrics(problem, vp, final["elbo"]))
        final.update({k: found[k] for k in runner.RESCORED_METRICS})
        metric = {k: found[k] for k in metric}
    side = {
        "label": LABEL,
        "seed": seed,
        "requested_options": options,
        "effective_options": options,
        "final": final,
        "provenance": {
            "source": identity["source"],
            "imports": identity["imports"],
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
        copy.deepcopy(identity),
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
    # The harness checkout's own package is the release code: it rescores,
    # and its tracked copies hold the rescored metrics and their record.
    assert manifest["finishing_steps"] == [["summarize"], ["rescore"]]
    assert manifest["tracked_copies"] == runner.tracked_copies(True)
    assert runner.RESCORING in manifest["tracked_copies"]["files"]
    family = manifest["confirmatory"]
    assert family["labels"] == [LABEL]
    assert family["signed_rank"] == ["elbo_err", "gskl", "mmtv"]
    assert family["tests"] == 4 and "Holm" in family["statement"]
    assert set(manifest["site"]) == set(contract.SETTINGS)
    runner.main(["cases", "--out", str(campaign), "--subset", "canary"])
    assert capsys.readouterr().out == f"1 {line_of(0)}\n"
    # Preparing again changes nothing; preparing otherwise is refused.
    before = (campaign / "manifest.json").read_bytes()
    base = ["prepare", "--out", str(campaign), *ARGUMENTS]
    assert runner.main([*base, "--arm", "after"]) == 0
    assert (campaign / "manifest.json").read_bytes() == before
    with pytest.raises(SystemExit, match="prepared otherwise"):
        runner.main([*base[:-1], "0-3"])


def test_prepare_pairs_arms(campaign, tmp_path, monkeypatch):
    before = prepare_other_arm(tmp_path, monkeypatch, "before", OTHER_IDENTITY)
    manifest = contract.read_json(before / "manifest.json")
    # A campaign of other code than the harness checkout's does not rescore.
    assert manifest["finishing_steps"] == [["summarize"]]
    assert manifest["tracked_copies"] == runner.tracked_copies(False)
    after = tmp_path / "after"
    runner.main(
        ["prepare", "--out", str(after), *ARGUMENTS, "--pair", str(before)]
    )
    steps = contract.read_json(after / "manifest.json")["finishing_steps"]
    assert steps == [
        ["summarize"],
        ["rescore", "--campaign", before.resolve().as_posix()],
    ]

    def paired_with(other, *extra):
        out = tmp_path / f"with_{other.name}{len(extra)}"
        return runner.main(
            ["prepare", "--out", str(out), *ARGUMENTS, *extra]
            + ["--pair", str(other)]
        )

    with pytest.raises(SystemExit, match="another allocation"):
        paired_with(before, "--seeds", "0-4")
    # The campaign fixture holds the code of this process.
    with pytest.raises(SystemExit, match="the code with itself"):
        paired_with(campaign)
    # The arms differ in their code alone.
    for name, change, message in (
        ("harness", ("trees", "harness", "commit"), "harness checkouts"),
        ("files", ("files", "dev/scripts/population_run.py"), "harness files"),
        ("versions", ("versions", "numpy"), "environment versions"),
    ):
        identity = copy.deepcopy(OTHER_IDENTITY)
        part = identity["source"]
        for key in change[:-1]:
            part = part[key]
        part[change[-1]] = "d" * 40
        other = prepare_other_arm(tmp_path, monkeypatch, name, identity)
        with pytest.raises(SystemExit, match=message):
            paired_with(other)
    monkeypatch.setenv("PYVBMC_SOURCE", str(tmp_path / "before_tree"))
    with pytest.raises(SystemExit, match="only a campaign"):
        paired_with(before)


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


def live_claim(campaign, seed):
    """A claim of this process's, which is live while it runs."""
    tag = contract.case_tag(line_of(seed))
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
    return claim


def test_worker_refuses_a_live_claim_and_leaves_the_files(
    campaign, monkeypatch
):
    partial = files_of(campaign, 0)["trace"]
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"left by another attempt")
    claim = live_claim(campaign, 0)
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
    use_identity(monkeypatch, OTHER_IDENTITY)
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


def test_worker_refuses_a_directory_that_is_not_its_campaign(tmp_path, capsys):
    """A directory without a manifest, or with another harness's, exits 64
    and is left as it is."""
    empty = tmp_path / "empty"
    empty.mkdir()
    args = ["worker", "--out", str(empty), "--case", line_of(0)]
    assert runner.main(args) == runner.EXIT_USAGE
    assert "holds no readable manifest.json" in capsys.readouterr().out
    assert list(empty.iterdir()) == []
    other = tmp_path / "other"
    other.mkdir()
    contract.write_json(
        other / "manifest.json",
        {"campaign": "svbmc_pool", "contract": contract.CONTRACT_VERSION},
    )
    args = ["worker", "--out", str(other), "--case", line_of(0)]
    assert runner.main(args) == runner.EXIT_USAGE
    assert "is not a manifest of population_run" in capsys.readouterr().out
    assert sorted(path.name for path in other.iterdir()) == ["manifest.json"]


def verification(out):
    report = contract.read_json(out / "verification.json")
    return report, {case["index"]: case for case in report["cases"]}


def test_verify_reconciles_every_state(campaign):
    tag, _ = complete_case(campaign, 0)
    failed = contract.case_tag(line_of(1))
    contract.error_path(campaign, failed).write_text("x: RuntimeError: boom\n")
    assert runner.main(["verify", "--out", str(campaign)]) == 0
    report, cases = verification(campaign)
    assert [cases[i]["status"] for i in (1, 2, 3)] == [
        "verified",
        "failed",
        "missing",
    ]
    assert cases[1]["boost"] == {"attempted": False, "accepted": False}
    # The record verify checked, for the readers of the report.
    assert cases[1]["record_sha256"] == runner.sha256(
        contract.record_path(campaign, tag)
    )
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
    use_identity(monkeypatch, OTHER_IDENTITY)
    assert (
        runner.main(["verify", "--out", str(campaign)])
        == contract.EXIT_IDENTITY
    )
    assert not (campaign / "verification.json").exists()


def test_summarize_reads_the_verified_cases_alone(campaign):
    complete_case(campaign, 0)
    # A case in flight has written its sidecar, and holds a live claim.
    _, files = complete_case(campaign, 1)
    contract.record_path(campaign, contract.case_tag(line_of(1))).unlink()
    live_claim(campaign, 1)
    contract.error_path(campaign, contract.case_tag(line_of(2))).write_text(
        "x: RuntimeError: boom\n"
    )
    assert runner.main(["verify", "--out", str(campaign)]) == 0
    assert [case["status"] for case in verification(campaign)[0]["cases"]] == [
        "verified",
        "in_flight",
        "failed",
    ]
    assert files["sidecar"].exists()
    assert runner.main(["summarize", "--out", str(campaign)]) == 0
    summary = (campaign / "summary.md").read_text()
    assert f"| {LABEL} | 1 | 1 |" in summary
    assert "verified 1, failed 1, in flight 1" in summary
    assert "skipped 1" in summary


def test_rescore_and_its_resumption(campaign, monkeypatch):
    complete_case(campaign, 0, exact_metrics=True)
    assert runner.main(["verify", "--out", str(campaign)]) == 0
    assert runner.main(["rescore", "--out", str(campaign)]) == 0
    path = campaign / "rescored" / f"{campaign.name}.json"
    report = contract.read_json(path)
    assert report["counts"] == {
        "rescored": 1,
        "reused": 0,
        "not_verified": 2,
        "equal_to_in_run": {k: 1 for k in runner.RESCORED_METRICS},
        "nonfinite": {k: 0 for k in runner.RESCORED_METRICS},
    }
    case = report["cases"][f"{LABEL}_seed0"]
    assert case["moment_method"] == "affine"
    assert all(case["finite"].values())
    assert report["campaign"]["verification_sha256"] == runner.sha256(
        campaign / "verification.json"
    )
    assert report["rescoring"]["identity"] == FAKE_IDENTITY
    # The rescoring's record binds the file by its SHA-256.
    record = contract.read_json(campaign / runner.RESCORING)
    assert record["identity"] == FAKE_IDENTITY
    assert record["rescored"][campaign.name]["file"] == (
        f"rescored/{campaign.name}.json"
    )
    assert record["rescored"][campaign.name]["sha256"] == runner.sha256(path)
    # A second rescoring takes the case from its work file.
    rescore_case = runner.rescore_case
    monkeypatch.setattr(runner, "rescore_case", fail_if_run)
    assert runner.main(["rescore", "--out", str(campaign)]) == 0
    again = contract.read_json(path)
    assert again["counts"]["reused"] == 1
    assert again["cases"] == report["cases"]
    # A work file made by other code is rescored anew.
    work = campaign / "rescored" / f"{campaign.name}.parts" / f"{LABEL}.json"
    part = contract.read_json(work)
    part["rescoring_source"] = OTHER_IDENTITY["source"]
    contract.write_json(work, part)
    with pytest.raises(AssertionError, match="must not run"):
        runner.main(["rescore", "--out", str(campaign)])
    monkeypatch.setattr(runner, "rescore_case", rescore_case)
    assert runner.main(["rescore", "--out", str(campaign)]) == 0
    assert contract.read_json(path)["counts"]["reused"] == 0
    # A sidecar changed after verification is refused.
    side = files_of(campaign, 0)["sidecar"]
    side.write_text(side.read_text() + " ")
    with pytest.raises(contract.ContractError, match="not the file"):
        runner.main(["rescore", "--out", str(campaign)])


def test_readers_refuse_a_verification_older_than_the_campaign(campaign):
    tag, _ = complete_case(campaign, 0, exact_metrics=True)
    assert runner.main(["verify", "--out", str(campaign)]) == 0
    # A case completes after the verification, as after a canary's finish.
    complete_case(campaign, 1, exact_metrics=True)
    for step in ("rescore", "summarize"):
        with pytest.raises(contract.ContractError, match="is older than"):
            runner.main([step, "--out", str(campaign)])
    assert runner.main(["verify", "--out", str(campaign)]) == 0
    assert runner.main(["rescore", "--out", str(campaign)]) == 0
    report = contract.read_json(
        campaign / "rescored" / f"{campaign.name}.json"
    )
    assert report["counts"]["rescored"] == 2
    # A record rewritten after the verification is not the one it checked.
    record = contract.record_path(campaign, tag)
    record.write_text(record.read_text() + " ")
    with pytest.raises(contract.ContractError, match="not the record"):
        runner.main(["rescore", "--out", str(campaign)])


def test_rescore_refuses_another_identity_than_the_manifests(
    campaign, monkeypatch
):
    complete_case(campaign, 0, exact_metrics=True)
    assert runner.main(["verify", "--out", str(campaign)]) == 0
    other = copy.deepcopy(FAKE_IDENTITY)
    other["source"]["versions"]["numpy"] = "3"
    use_identity(monkeypatch, other)
    assert (
        runner.main(["rescore", "--out", str(campaign)])
        == contract.EXIT_IDENTITY
    )
    assert not (campaign / runner.RESCORING).exists()
    assert not (campaign / "rescored").exists()


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


def test_rescore_without_the_gpyreg_checkout_is_refused(
    campaign, monkeypatch, capsys
):
    """Without ``PYVBMC_GPYREG_SOURCE`` no identity can be established."""
    monkeypatch.delenv("PYVBMC_GPYREG_SOURCE")
    assert (
        runner.main(["rescore", "--out", str(campaign)])
        == contract.EXIT_IDENTITY
    )
    printed = capsys.readouterr().out
    assert "rescore refused: no identity" in printed
    assert "PYVBMC_GPYREG_SOURCE is not set" in printed
    assert not (campaign / "rescored").exists()


def test_rescore_requires_its_own_code_to_reproduce_the_in_run_metrics(
    campaign,
):
    # In-run metrics of 0.1, which the posterior does not score.
    complete_case(campaign, 0)
    assert runner.main(["verify", "--out", str(campaign)]) == 0
    with pytest.raises(contract.ContractError, match="differ from their in"):
        runner.main(["rescore", "--out", str(campaign)])
    assert not (campaign / "rescored" / f"{campaign.name}.json").exists()
    assert not (campaign / runner.RESCORING).exists()
    work = campaign / "rescored" / f"{campaign.name}.parts" / f"{LABEL}.json"
    entry = contract.read_json(work)["cases"][f"{LABEL}_seed0"]
    assert entry["equal_to_in_run"]["gskl"] is False


def sited_identity(site, node=None):
    """:data:`FAKE_IDENTITY` with the host part and the paths of ``site``:
    of its login node, or with ``node`` of an array task there."""
    identity = copy.deepcopy(FAKE_IDENTITY)
    identity["imports"] = {
        "trees": {
            name: {"path": "", "dirty": []}
            for name in ("harness", "pyvbmc", "gpyreg")
        },
        "modules": {"pyvbmc": "", "gpyreg": ""},
        "installed_metadata_versions": {"pyvbmc": None, "gpyreg": None},
    }
    identity = site.plant(identity, node=node)
    identity["imports"]["harness_modules"] = {
        "golden_trace": str(site.home / "src" / "golden_trace.py")
    }
    return identity


@pytest.fixture
def sited(tmp_path, monkeypatch):
    """``(site, campaign)``: the campaign of :func:`campaign`, prepared at
    a stand-in site (``campaign_slurm_stubs.FakeSite``) in its operator's
    home, with the site's settings and the login node's identity."""
    site = stubs.FakeSite(tmp_path)
    for name, value in site.site_block(
        "dev/scripts/population_run.py"
    ).items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    use_identity(monkeypatch, sited_identity(site))
    monkeypatch.setattr(
        contract,
        "pip_freeze",
        lambda: ["pyvbmc==0", f"gpyreg @ file://{site.gpyreg.as_posix()}"],
    )
    out = site.home / "runs" / "population_after"
    assert runner.main(["prepare", "--out", str(out), *ARGUMENTS]) == 0
    return site, out


def test_the_tracked_copies_of_a_campaign_are_redacted(sited, tmp_path):
    site, out = sited
    manifest = contract.read_json(out / "manifest.json")
    assert manifest["tracked_copies"] == runner.tracked_copies(True)
    tag, files = complete_case(
        out,
        0,
        exact_metrics=True,
        identity=sited_identity(site, site.nodes[0]),
    )
    assert runner.main(["verify", "--out", str(out)]) == 0
    assert runner.main(["summarize", "--out", str(out)]) == 0
    assert runner.main(["rescore", "--out", str(out)]) == 0
    site.write_slurm(out)
    assert site.leaks(out)
    target = tmp_path / "handback" / "population_after"
    contract.redact(
        out,
        target,
        operator=site.operator(),
        environ={},
        host="fakelogin9",
        say=lambda message: None,
    )
    assert site.leaks(target) == []
    rels = runner.case_files(LABEL, 0)
    names = sorted(
        p.relative_to(target).as_posix()
        for p in target.rglob("*")
        if p.is_file()
    )
    # Every verified case's record, sidecar and boost report, and none of
    # the unverified ones; the traces, the pickles and the rescoring's work
    # files stay in the archive.
    assert names == sorted(
        [
            "manifest.json",
            "verification.json",
            "summary.md",
            "rescored/population_after.json",
            runner.RESCORING,
            "redaction.json",
            f"records/{tag}.complete.json",
            rels["sidecar"],
            rels["boost_report"],
        ]
    )
    side = json.loads((target / rels["sidecar"]).read_text())
    raw = json.loads(files["sidecar"].read_text())
    assert side["final"] == raw["final"]
    trees = side["provenance"]["imports"]["trees"]
    assert trees["pyvbmc"]["path"].startswith("~")
    assert trees["gpyreg"]["path"] == "$PYVBMC_GPYREG_SOURCE"
    record = contract.read_json(contract.record_path(target, tag))
    assert record["identity"]["host"]["hostname"] == site.family
    for key in ("sidecar", "boost_report"):
        assert (
            contract.source_sha256(target, rels[key])
            == record["artifacts"][rels[key]]["sha256"]
        )
    rescored = contract.read_json(target / "rescored/population_after.json")
    assert rescored["campaign"]["path"].startswith("~")
    assert rescored["rescoring"]["identity"]["host"]["hostname"] == "login"
    # The rescoring's record names the SHA-256 of the file as it was
    # written, which the copy's source SHA-256 is.
    rescoring = contract.read_json(target / runner.RESCORING)
    assert rescoring["identity"]["host"]["hostname"] == "login"
    assert rescoring["rescored"]["population_after"]["sha256"] == (
        contract.source_sha256(target, "rescored/population_after.json")
    )
    # A sidecar field the redaction knows nothing of, holding the username.
    raw["notes"] = f"run by {site.user}"
    files["sidecar"].write_text(json.dumps(raw, indent=1))
    again = tmp_path / "handback" / "again"
    with pytest.raises(contract.ContractError) as refusal:
        contract.redact(
            out,
            again,
            operator=site.operator(),
            environ={},
            say=lambda message: None,
        )
    assert rels["sidecar"] in str(refusal.value)
    assert "the username" in str(refusal.value)
    assert not again.exists()


# --------------------------------------------------------------------------
# A population as the envelope of golden_replay.py
# --------------------------------------------------------------------------


def test_golden_replay_reads_the_envelopes_of_either_layout(
    campaign, tmp_path
):
    import golden_replay

    complete_case(campaign, 0)
    complete_case(campaign, 1)
    # A case in flight has written its sidecar.
    _, files = complete_case(campaign, 2)
    contract.record_path(campaign, contract.case_tag(line_of(2))).unlink()
    live_claim(campaign, 2)
    assert runner.main(["verify", "--out", str(campaign)]) == 0
    # A campaign directory, as its tracked copies: one directory per
    # configuration, and the verified cases alone.
    found = golden_replay.load_envelopes(campaign, [LABEL, "normal_D5"])
    assert sorted(found) == [LABEL]
    assert sorted(found[LABEL]["seeds"]) == [0, 1]
    assert golden_replay.sidecar_directory(campaign, LABEL) == campaign / LABEL
    # A flat directory, as dev/golden/baseline/ is: every sidecar.
    flat = tmp_path / "flat"
    flat.mkdir()
    for seed in range(3):
        shutil.copy(files_of(campaign, seed)["sidecar"], flat)
    found = golden_replay.load_envelopes(flat, [LABEL])
    assert sorted(found[LABEL]["seeds"]) == [0, 1, 2]
    np.testing.assert_array_equal(found[LABEL]["gskl"], [0.1] * 3)
    assert golden_replay.sidecar_directory(flat, LABEL) == flat


def test_golden_replay_refuses_a_configuration_without_a_population(
    campaign, tmp_path
):
    import golden_replay

    complete_case(campaign, 0)
    assert runner.main(["verify", "--out", str(campaign)]) == 0
    empty = tmp_path / "empty"
    empty.mkdir()
    out = tmp_path / "replay"

    def replay(labels, sidecars):
        return golden_replay.main(
            ["--configs", labels, "--sidecars", str(sidecars)]
            + ["--report-only", "--out", str(out)]
        )

    for labels, sidecars, missing in (
        (LABEL, empty, LABEL),
        (f"{LABEL},normal_D5", campaign, "normal_D5"),
        (LABEL, tmp_path / "absent", None),
    ):
        with pytest.raises(SystemExit) as refusal:
            replay(labels, sidecars)
        assert str(missing or "is not a directory") in str(refusal.value)
    # With a population, the replay goes on: here nothing is left to
    # report, which is its own failure.
    assert replay(LABEL, campaign) == 1


# --------------------------------------------------------------------------
# Two arms, verified and rescored by the harness, compared
# --------------------------------------------------------------------------


@pytest.fixture
def arms(tmp_path, monkeypatch):
    """``(before, after)``: two verified arms on one allocation, the first
    of other code (:data:`OTHER_IDENTITY`), whose posteriors lie elsewhere,
    and the second of the harness checkout's, prepared with ``--pair``."""
    monkeypatch.setenv("PYVBMC_GPYREG_SOURCE", str(tmp_path / "gpyreg"))
    monkeypatch.delenv("PYVBMC_SOURCE", raising=False)
    monkeypatch.setattr(contract, "pip_freeze", lambda: ["pyvbmc==0"])
    before = prepare_other_arm(tmp_path, monkeypatch, "before", OTHER_IDENTITY)
    for seed in range(3):
        complete_case(
            before,
            seed,
            exact_metrics=True,
            identity=OTHER_IDENTITY,
            shift=0.2 * (seed + 1),
        )
    use_identity(monkeypatch, OTHER_IDENTITY)
    assert runner.main(["verify", "--out", str(before)]) == 0
    use_identity(monkeypatch, FAKE_IDENTITY)
    after = tmp_path / "after"
    assert (
        runner.main(
            ["prepare", "--out", str(after), *ARGUMENTS]
            + ["--arm", "after", "--pair", str(before)]
        )
        == 0
    )
    for seed in range(3):
        complete_case(after, seed, exact_metrics=True)
    assert runner.main(["verify", "--out", str(after)]) == 0
    return before, after


def finish(out):
    """Run the finishing steps of a verified campaign, as the finish does."""
    manifest = contract.read_json(out / "manifest.json")
    for step in contract.finishing_steps(manifest):
        assert runner.main([*step, "--out", str(out)]) == 0, step


def test_two_arms_verified_rescored_and_compared(arms, tmp_path):
    import analyze_population_run as analysis

    before, after = arms
    finish(after)
    record = contract.read_json(after / runner.RESCORING)
    assert sorted(record["rescored"]) == ["after", "before"]
    result = analysis.analyze_arms(before, after, None, tmp_path / "report")
    assert result["paired_cases"] == 3
    assert result["arms"]["reference"]["arm"] == "before"
    assert result["arms"]["candidate"]["rescored_equal_to_in_run"] == {
        k: 3 for k in runner.RESCORED_METRICS
    }
    tests = result["confirmatory_tests"]
    assert len(tests) == result["confirmatory_family"]["tests"] == 4
    assert all(t["computed"] and t["n_pairs"] == 3 for t in tests)
    # The arms' posteriors differ, and so do their metrics.
    gskl = next(t for t in tests if t["metric"] == "gskl")
    assert gskl["tied"] < 3
    assert (tmp_path / "report" / "assessment.json").is_file()


def test_rescore_counts_rescored_metrics_that_are_not_finite(
    arms, tmp_path, monkeypatch
):
    # The other arm's posteriors score no finite gsKL with the release
    # code: the rescoring records it, where it would refuse its own arm.
    before, _ = arms
    rescore_metrics = runner.rescore_metrics

    def infinite(problem, vp, elbo):
        return {**rescore_metrics(problem, vp, elbo), "gskl": float("inf")}

    monkeypatch.setattr(runner, "rescore_metrics", infinite)
    report = runner.rescore_campaign(
        before, copy.deepcopy(FAKE_IDENTITY), tmp_path / "parts"
    )
    assert report["counts"]["nonfinite"]["gskl"] == 3
    assert report["counts"]["equal_to_in_run"]["gskl"] == 0
    case = report["cases"][f"{LABEL}_seed1"]
    assert case["finite"] == {k: k != "gskl" for k in runner.RESCORED_METRICS}


def test_the_comparison_refuses_a_rescoring_of_other_code(arms, tmp_path):
    import analyze_population_run as analysis

    before, after = arms
    finish(after)
    path = after / runner.RESCORING
    record = contract.read_json(path)
    record["identity"]["source"]["versions"]["numpy"] = "3"
    contract.write_json(path, record)
    with pytest.raises(AssertionError, match="another process"):
        analysis.analyze_arms(before, after, None, tmp_path / "report")


def test_the_comparison_refuses_arms_of_other_families(tmp_path, monkeypatch):
    import analyze_population_run as analysis

    monkeypatch.setenv("PYVBMC_GPYREG_SOURCE", str(tmp_path / "gpyreg"))
    monkeypatch.setattr(contract, "pip_freeze", lambda: ["pyvbmc==0"])
    spec = tmp_path / "family.json"
    contract.write_json(spec, {"signed_rank": ["gskl"]})
    before = prepare_other_arm(
        tmp_path,
        monkeypatch,
        "before",
        OTHER_IDENTITY,
        extra=["--confirmatory", str(spec)],
    )
    for seed in range(3):
        complete_case(
            before, seed, exact_metrics=True, identity=OTHER_IDENTITY
        )
    use_identity(monkeypatch, OTHER_IDENTITY)
    assert runner.main(["verify", "--out", str(before)]) == 0
    use_identity(monkeypatch, FAKE_IDENTITY)
    after = tmp_path / "after"
    with pytest.raises(SystemExit, match="another confirmatory"):
        runner.main(
            ["prepare", "--out", str(after), *ARGUMENTS]
            + ["--pair", str(before)]
        )
    # Prepared without the pair, rescored with the other arm all the same.
    assert runner.main(["prepare", "--out", str(after), *ARGUMENTS]) == 0
    for seed in range(3):
        complete_case(after, seed, exact_metrics=True)
    assert runner.main(["verify", "--out", str(after)]) == 0
    rescore = ["rescore", "--out", str(after), "--campaign", str(before)]
    assert runner.main(rescore) == 0
    with pytest.raises(AssertionError, match="different confirmatory"):
        analysis.analyze_arms(before, after, None, tmp_path / "report")


# --------------------------------------------------------------------------
# The returned posterior as plain arrays
# --------------------------------------------------------------------------


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


@pytest.mark.parametrize("rotoscaled", [False, True])
@pytest.mark.parametrize("transform", ["logit", "probit"])
def test_returned_posterior_rebuilds_bounded_and_rotoscaled_posteriors(
    transform, rotoscaled
):
    """A posterior with bounded, unbounded, rescaled and rotated
    coordinates, as a run returns it after a warp, rebuilds exactly: the
    same data and the same density in the original space."""
    from pyvbmc.parameter_transformer import ParameterTransformer
    from pyvbmc.variational_posterior import VariationalPosterior

    D, K = 3, 4
    rng = np.random.default_rng(5)
    lb, ub = np.array([[-np.inf, 0.0, -3.0]]), np.array([[np.inf, 5.0, 2.0]])
    plb, pub = np.array([[-2.0, 0.5, -2.0]]), np.array([[2.0, 4.0, 1.0]])
    scale = rotation = None
    if rotoscaled:
        scale = rng.uniform(0.5, 2.0, D)
        rotation = np.linalg.qr(rng.normal(size=(D, D)))[0]
    transformer = ParameterTransformer(
        D,
        lb,
        ub,
        plb,
        pub,
        scale=scale,
        rotation_matrix=rotation,
        transform_type=transform,
    )
    vp = VariationalPosterior(
        D, K, parameter_transformer=transformer, rng=np.random.default_rng(1)
    )
    vp.w = rng.dirichlet(np.ones(K)).reshape(1, K)
    vp.mu = rng.normal(size=(D, K))
    vp.sigma = rng.uniform(0.2, 1.0, (1, K))
    vp.lambd = rng.uniform(0.5, 1.5, (D, 1))
    returned = vp.parameter_transformer
    trace = {
        "final_w": np.ravel(vp.w),
        "final_mu": vp.mu,
        "final_sigma": np.ravel(vp.sigma),
        "final_lambd": np.ravel(vp.lambd),
        # As golden_trace.run_task stores them: ones and the identity where
        # the transformer neither rescales nor rotates.
        "pt_mu": returned.mu[None],
        "pt_delta": returned.delta[None],
        "pt_scale": np.ones((1, D)) if scale is None else returned.scale[None],
        "pt_R": (np.eye(D) if rotation is None else returned.R_mat)[None],
    }
    extras = runner.posterior_extras(vp)
    assert (
        int(extras["bounded_types"][0])
        == {"logit": 3, "probit": 12}[transform]
    )
    assert bool(extras["rotation_is_none"]) == (not rotoscaled)
    problem = SimpleNamespace(D=D, lb=lb, ub=ub)
    rebuilt = runner.returned_posterior(trace, extras, problem, 0)
    assert runner.posterior_differences(rebuilt, vp) == []
    x = vp.sample(200, orig_flag=True)[0]
    assert np.all((x > lb) & (x < ub))
    for log_flag in (False, True):
        np.testing.assert_array_equal(
            rebuilt.pdf(x, orig_flag=True, log_flag=log_flag),
            vp.pdf(x, orig_flag=True, log_flag=log_flag),
        )
    # Arrays that name another bounded transform, or deny the rotation and
    # the rescaling, rebuild another posterior.
    other = {"logit": 12, "probit": 3}[transform]
    wrong = dict(extras, bounded_types=np.array([other]))
    assert (
        "parameter_transformer.bounded_types"
        in runner.posterior_differences(
            runner.returned_posterior(trace, wrong, problem, 0), vp
        )
    )
    if rotoscaled:
        wrong = dict(
            extras,
            scale_is_none=np.asarray(True),
            rotation_is_none=np.asarray(True),
        )
        assert runner.posterior_differences(
            runner.returned_posterior(trace, wrong, problem, 0), vp
        ) == ["parameter_transformer.R_mat", "parameter_transformer.scale"]


def test_recorded_options_are_the_same_in_every_process():
    # Two processes record equal options alike: no memory address, and a
    # set in sorted order.
    class Thing:
        pass

    value = {"f": lambda x: x, "o": Thing(), "s": {"b", "a", "c"}}
    recorded = runner.jsonable(value)
    assert recorded["s"] == ["a", "b", "c"]
    assert " at 0x" not in recorded["f"] and " at 0x" not in recorded["o"]
    assert recorded["o"].endswith("Thing object>")


# --------------------------------------------------------------------------
# PYVBMC_SOURCE
# --------------------------------------------------------------------------


def stand_in_tree(tmp_path):
    """A stand-in package tree, whose ``pyvbmc`` holds what the harness
    imports of it."""
    tree = tmp_path / "tree"
    (tree / "pyvbmc" / "vbmc").mkdir(parents=True)
    (tree / "pyvbmc" / "__init__.py").write_text(
        "from .vbmc.vbmc import VBMC\n"
    )
    (tree / "pyvbmc" / "vbmc" / "__init__.py").write_text("")
    (tree / "pyvbmc" / "vbmc" / "vbmc.py").write_text(
        "class VBMC:\n    pass\n\n\ndef optimize_vp(*args):\n    pass\n"
    )
    return tree


def run_python(script, tree):
    """Run ``script`` in a fresh interpreter in this directory, with
    ``PYVBMC_SOURCE`` naming ``tree``; return its output."""
    env = dict(os.environ, PYVBMC_SOURCE=str(tree))
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script), str(tree)],
        cwd=runner.HERE,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_the_harness_run_as_a_script_takes_the_package_tree(tmp_path):
    # The harness modules put the harness checkout first on sys.path when
    # they are imported, after the harness has imported PyVBMC from the
    # tree PYVBMC_SOURCE names.
    tree = stand_in_tree(tmp_path)
    script = """
        import runpy
        import sys
        from pathlib import Path
        tree = Path(sys.argv[1]).resolve()
        sys.argv = ["population_run.py", "--help"]
        try:
            runpy.run_path("population_run.py", run_name="__main__")
        except SystemExit as stop:
            assert stop.code == 0, stop.code
        import golden_trace
        import pyvbmc
        assert Path(pyvbmc.__file__).resolve().parents[1] == tree, pyvbmc
        here = Path.cwd().resolve()
        assert Path(golden_trace.__file__).resolve().parent == here
        print("ok")
        """
    assert run_python(script, tree).endswith("ok")


@pytest.mark.parametrize("holds", ["no package", "a package that fails"])
def test_a_package_tree_that_does_not_import_leaves_no_identity(
    tmp_path, holds
):
    """Run as a script, the harness imports PyVBMC from the tree that
    ``PYVBMC_SOURCE`` names before anything else. From a tree that holds no
    package, or one whose import fails, no identity can be established:
    the worker exits 78 and writes nothing, an error file included."""
    tree = tmp_path / "tree"
    tree.mkdir()
    if holds == "a package that fails":
        (tree / "pyvbmc").mkdir()
        (tree / "pyvbmc" / "__init__.py").write_text(
            "raise ImportError('a tree that does not import')\n"
        )
    out = tmp_path / "campaign"
    out.mkdir()
    (out / "manifest.json").write_text("{}\n")
    env = dict(os.environ, PYVBMC_SOURCE=str(tree))
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-u", str(runner.HERE / "population_run.py")]
        + ["worker", "--out", str(out), "--case", line_of(0)],
        cwd=runner.ROOT,
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    assert result.returncode == contract.EXIT_IDENTITY, (
        result.stdout + result.stderr
    )
    assert "cannot be imported from the campaign's package tree" in (
        result.stderr
    )
    assert sorted(path.name for path in out.iterdir()) == ["manifest.json"]


@pytest.mark.parametrize(
    "imports",
    [
        # make_oracle_fixtures.py, the gate of the oracles
        "from benchmark_targets import find_config\n"
        "from profile_run import git_info, pkg_version\n"
        "import golden_trace\n",
        # svbmc_pool_io.py, the pool readers
        "from profile_run import effective_options, jsonable\n",
        # analyze_population_run.py and reference_join.py
        "import population_run\n",
    ],
)
def test_importers_of_the_harness_modules_ignore_PYVBMC_SOURCE(
    tmp_path, imports
):
    tree = stand_in_tree(tmp_path)
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "root = Path.cwd().resolve().parents[1]\n"
        "sys.path.insert(0, str(Path.cwd()))\n"
        "sys.path.insert(0, str(root))\n" + imports + "import pyvbmc\n"
        "assert Path(pyvbmc.__file__).resolve().parents[1] == root, pyvbmc\n"
        "print('ok')\n"
    )
    assert run_python(script, tree) == "ok"


# --------------------------------------------------------------------------
# Real runs
# --------------------------------------------------------------------------


def gpyreg_checkout():
    """The gpyreg checkout of the real runs: the one
    ``PYVBMC_GPYREG_SOURCE`` names, else the git checkout the imported
    gpyreg lies in; None without either."""
    named = os.environ.get("PYVBMC_GPYREG_SOURCE")
    if named:
        return Path(named).resolve()
    import gpyreg

    top = Path(gpyreg.__file__).resolve().parents[1]
    return top if (top / ".git").exists() else None


def harness_environment(gpyreg):
    """This process's environment outside any Slurm task and campaign,
    with the campaign's thread settings and ``gpyreg``."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("SLURM")
        and key.upper() not in contract.SETTINGS
        and key != "PYTHONPATH"
    }
    env.update({key: "1" for key in runner.THREAD_KEYS})
    env.update(MPLBACKEND="Agg", PYVBMC_GPYREG_SOURCE=str(gpyreg))
    return env


def harness(*args, env):
    result = subprocess.run(
        [sys.executable, "-u", str(runner.HERE / "population_run.py")]
        + [str(arg) for arg in args],
        cwd=runner.ROOT,
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture(scope="module")
def real(tmp_path_factory):
    """``(ran, arrayed)``: one case run by ``run``, its finishing steps
    included, and by ``worker --case``, in two campaigns prepared alike."""
    gpyreg = gpyreg_checkout()
    if gpyreg is None:
        pytest.skip(
            "PYVBMC_GPYREG_SOURCE is unset and the imported gpyreg lies in "
            "no git checkout"
        )
    env = harness_environment(gpyreg)
    root = tmp_path_factory.mktemp("real")
    ran, arrayed = root / "ran", root / "arrayed"
    arguments = ["--suite", "smoke", "--labels", LABEL, "--seeds", "0"]
    for out in (ran, arrayed):
        harness("prepare", "--out", out, *arguments, "--arm", "after", env=env)
    harness("run", "--out", ran, env=env)
    harness("worker", "--out", arrayed, "--case", line_of(0), env=env)
    return ran, arrayed


def test_run_verifies_and_rescores_its_real_cases_exactly(real):
    ran, _ = real
    report, cases = verification(ran)
    assert report["counts"]["verified"] == 1 and report["exit_code"] == 0
    rels = runner.case_files(LABEL, 0)
    side = contract.read_json(ran / rels["sidecar"])
    # The package is the harness checkout's, and so is every tree.
    source = side["meta"]["pyvbmc_source"]
    assert Path(source["path"]).parent == runner.ROOT
    record = contract.read_json(contract.record_path(ran, cases[1]["tag"]))
    assert side["provenance"] == {
        "source": record["identity"]["source"],
        "imports": record["identity"]["imports"],
    }
    # The in-run metrics are golden_trace.run_task's, of the posterior the
    # run returned; the rescoring's are of the posterior rebuilt from the
    # plain arrays, by the same code.
    rescored = contract.read_json(ran / "rescored" / f"{ran.name}.json")
    case = rescored["cases"][f"{LABEL}_seed0"]
    assert case["metrics"] == {
        key: side["final"][key] for key in runner.RESCORED_METRICS
    }
    assert all(case["equal_to_in_run"].values())
    assert np.all(np.isfinite(list(case["metrics"].values())))
    record = contract.read_json(ran / runner.RESCORING)
    assert record["rescored"][ran.name]["sha256"] == runner.sha256(
        ran / "rescored" / f"{ran.name}.json"
    )
    assert f"| {LABEL} | 1 | 0 |" in (ran / "summary.md").read_text()


#: What says when a case ran and how long it took, in its sidecar.
TIMING_FINAL = ("wall_s", "target_eval_s", "peak_rss_mb")
TIMING_META = ("started", "finished", "pid")


def sidecar_content(path):
    """A sidecar without its timings and process id."""
    side = contract.read_json(path)
    for key in TIMING_FINAL:
        side["final"].pop(key)
    for key in TIMING_META:
        side["meta"].pop(key)
    return side


def test_the_array_worker_and_run_give_identical_artifacts(real):
    """One case, run by ``run`` in one campaign and by ``worker --case`` in
    another prepared alike: every stored array but the timer's is equal bit
    for bit, the boost capture and its report are equal, and the sidecar
    and the record differ in their timings and the process alone."""
    ran, arrayed = real
    rels = runner.case_files(LABEL, 0)
    for key in ("trace", "posterior"):
        with np.load(ran / rels[key], allow_pickle=False) as a, np.load(
            arrayed / rels[key], allow_pickle=False
        ) as b:
            assert sorted(a.files) == sorted(b.files)
            for name in a.files:
                if name == "timer":
                    continue
                assert a[name].dtype == b[name].dtype, name
                assert a[name].shape == b[name].shape, name
                assert a[name].tobytes() == b[name].tobytes(), name
    assert sidecar_content(ran / rels["sidecar"]) == sidecar_content(
        arrayed / rels["sidecar"]
    )
    assert contract.read_json(
        ran / rels["boost_report"]
    ) == contract.read_json(arrayed / rels["boost_report"])
    captures = []
    for out in (ran, arrayed):
        with (out / rels["boost_state"]).open("rb") as stream:
            captures.append(dill.load(stream))
    a, b = captures
    assert set(a) == set(b)
    for key in a:
        if key in ("pre", "candidate", "returned"):
            if a[key] is None:
                assert b[key] is None, key
                continue
            assert runner.posterior_differences(a[key], b[key]) == [], key
            assert a[key].stats.keys() == b[key].stats.keys(), key
            np.testing.assert_equal(dict(a[key].stats), dict(b[key].stats))
            assert (
                a[key].rng.bit_generator.state
                == b[key].rng.bit_generator.state
            ), key
        else:
            np.testing.assert_equal(a[key], b[key])
    tag = contract.case_tag(line_of(0))
    records = [
        contract.read_json(contract.record_path(out, tag))
        for out in (ran, arrayed)
    ]
    for key in ("harness", "label", "seed", "boost", "case", "tag"):
        assert records[0][key] == records[1][key], key
    assert records[0]["identity"]["source"] == records[1]["identity"]["source"]


@pytest.mark.parametrize(
    "seeded, message",
    [
        (lambda side: None, None),
        (
            lambda side: side["effective_options"].update(tol_elcbo_boost=0.2),
            "incorrect effective tol_elcbo_boost",
        ),
        (
            lambda side: side["final"].update(elbo_sd=float("nan")),
            "nonfinite elbo_sd",
        ),
    ],
    ids=["unchanged", "option_not_applied", "nonfinite_elbo_sd"],
)
def test_a_case_that_fails_its_checks_fails_in_the_worker(
    real, campaign, monkeypatch, seeded, message
):
    """The real run replayed through the worker, with a fault seeded into
    its sidecar: a case that ``verify`` would fail gets an error file and
    no completion record, and the case unchanged completes."""
    ran, _ = real
    stored = runner.case_files(LABEL, 0)
    with (ran / stored["boost_state"]).open("rb") as stream:
        state = dill.load(stream)

    def run_task(label, seed, options, directory):
        for key in ("trace", "sidecar"):
            shutil.copyfile(
                ran / stored[key], Path(directory) / Path(stored[key]).name
            )
        path = Path(directory) / Path(stored["sidecar"]).name
        side = json.loads(path.read_text())
        seeded(side)
        path.write_text(json.dumps(side, indent=1))
        return {"tag": f"{label}_seed{seed}", "ok": True}

    class Capture:
        def __enter__(self):
            self.state = copy.deepcopy(state)
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(runner, "BoostCapture", Capture)
    monkeypatch.setattr(runner.golden_trace, "run_task", run_task)
    tag = contract.case_tag(line_of(0))
    args = ["worker", "--out", str(campaign), "--case", line_of(0)]
    code = runner.main(args)
    if message is None:
        assert code == 0
        assert contract.record_path(campaign, tag).is_file()
        return
    assert code == 1
    assert message in contract.error_path(campaign, tag).read_text()
    assert not contract.record_path(campaign, tag).exists()
    assert not files_of(campaign, 0)["sidecar"].exists()
    assert runner.main(["verify", "--out", str(campaign)]) == 0
    assert verification(campaign)[1][1]["status"] == "failed"
