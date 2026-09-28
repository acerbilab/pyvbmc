"""Checks of the promotion of the release gate's reference (``reference_promote.py``).

A promotion runs under a temporary root that holds copies of what it reads
and rewrites in the checkout (``AGENTS.md``, ``dev/README.md``,
``dev/golden/README.md`` and ``baseline/``, the previous promotion's
manifest, ``golden_replay.py`` and ``analyze_population_run.py``) and the
records of a small release gate: two campaigns of array mode on the
``smoke`` suite's ``normal_D2`` at seeds 0-2, written as
``test_population_run.py`` writes them at a stand-in site
(``campaign_slurm_stubs.FakeSite``), an after arm of the harness checkout's
code and a before arm of other code, verified and rescored, redacted into
the root as a hand-back puts them in the repository, and assessed there. The fingerprints and the replay of the defaults are real
runs of ``normal_D2`` at seed 0 (seconds each) through ``golden_replay.py``;
the gate runs are those of the stand-in gate script of
``test_seeded_gate_runs.py``. The identities name the checkout's commit and
its gpyreg's; the checks of clean checkouts are off, and git's tracking of
the copies is taken as given, so that the module runs on a working tree
with changes. The tests that record skip without a gpyreg checkout.

The checkout's own passages are checked against the text the promotion was
written against: an edit to one of them fails here until its template and
its SHA-256 in ``PASSAGES`` follow it.
"""

import copy
import json
import shutil
import socket
from pathlib import Path

import campaign_slurm_stubs as stubs
import pytest
import reference_promote as promote
import seeded_gate_runs as gates
import test_population_run as tp
import test_seeded_gate_runs as tg

runner = promote.runner
contract = promote.contract
LABEL = tp.LABEL
NAME = "reference_3_test"
#: What the root copies from the checkout.
FILES = (
    "AGENTS.md",
    "dev/README.md",
    "dev/golden/README.md",
    "dev/scripts/golden_replay.py",
    "dev/scripts/analyze_population_run.py",
    f"{promote.PREVIOUS_RECORD}/{promote.MANIFEST}",
)
GATE = "dev/experiments/release_gate_test"


# --------------------------------------------------------------------------
# The checkout's passages, and the pure parts
# --------------------------------------------------------------------------


def test_the_passages_are_those_the_promotion_rewrites():
    assert promote.passage_problems(promote.REPO) == []
    for file, old, _ in promote.REPLACEMENTS:
        text, _ = promote.read_text(promote.REPO / file)
        assert text.count(old) == 1, (file, old)
    file, start, end = promote.OWN_ENTRY
    promote.span(promote.read_text(promote.REPO / file)[0], start, end)


def test_reflow_wraps_without_opening_a_markdown_block():
    text = promote.reflow(
        "- **Item.** "
        + "word " * 13
        + "75 (D + 2) evaluations and\nmore words 1. here "
        + "x " * 30
        + "- y\n\n| a | b |\n|---|---|\n\n```console\nkeep   this\n```\n",
        width=40,
    )
    lines = text.split("\n")
    assert lines[0].startswith("- **Item.**")
    body = lines[1 : lines.index("")]
    assert all(line.startswith("  ") and len(line) <= 45 for line in body)
    assert any("75 (D + 2)" in line for line in lines)
    for line in body:
        assert not promote.BLOCK_MARKER.match(line.strip()), line
    assert "| a | b |" in lines and "keep   this" in lines


def test_fingerprints_outside_their_envelopes_are_judged_together():
    # At the rate of the reference of 2026-09-13, 4.2 %, four of 24
    # fingerprints outside their envelopes are plausible and five are not.
    rate = 42 / 990
    assert promote.outside_probability(0, 24, rate) == 1.0
    assert promote.outside_probability(4, 24, rate) > promote.FINGERPRINT_ALPHA
    assert promote.outside_probability(5, 24, rate) < promote.FINGERPRINT_ALPHA
    # A population none of whose runs lies outside its envelope admits no
    # fingerprint outside.
    assert promote.outside_probability(1, 24, 0.0) == 0.0
    # One run of eight far beyond the others' Q3 + 3 IQR.
    envelopes = {
        "a": {
            "seeds": list(range(8)),
            "elbo_err": promote.np.array([0.1] * 7 + [5.0]),
            "gskl": promote.np.full(8, 0.1),
            "mmtv": promote.np.full(8, 0.1),
        }
    }
    assert promote.outlier_rate(envelopes) == 1 / 8


# --------------------------------------------------------------------------
# A small release gate
# --------------------------------------------------------------------------


def identities():
    """The after arm's identity, of this checkout and its gpyreg, and the
    before arm's, of other code."""
    head = contract.git(promote.REPO, "rev-parse", "HEAD")
    gpyreg = contract.git(gates.gpyreg_tree(), "rev-parse", "HEAD")
    after = copy.deepcopy(tp.FAKE_IDENTITY)
    for tree in ("harness", "pyvbmc"):
        after["source"]["trees"][tree]["commit"] = head
    after["source"]["trees"]["gpyreg"]["commit"] = gpyreg
    before = copy.deepcopy(after)
    before["source"]["trees"]["pyvbmc"]["commit"] = "b" * 40
    before["source"]["trees"]["gpyreg"]["commit"] = "c" * 40
    return after, before


def effective(seed):
    """The effective options of a run of the after arm, as
    ``golden_trace.run_task`` records them with the harness's options."""
    from benchmark_targets import find_config
    from profile_run import effective_options

    from pyvbmc import VBMC

    args, options = find_config(LABEL).make(seed=seed).vbmc_args()
    options.update(
        display="off",
        plot=False,
        print_iteration_header=False,
        performance_calibration="off",
    )
    options.update(runner.DEFAULT_OPTIONS)
    vbmc = VBMC(*args, options=dict(options), seed=seed)
    return effective_options(vbmc, options.keys())


def write_arm_case(site, out, seed, identity, shift=0.0):
    """A verified-looking case (``test_population_run.complete_case``)
    whose sidecar holds the effective options of a real run, recorded on a
    compute node of the stand-in ``site``."""
    tag, files = tp.complete_case(
        out, seed, exact_metrics=True, identity=identity, shift=shift
    )
    side = json.loads(files["sidecar"].read_text())
    side["effective_options"] = effective(seed)
    side["noise_sd"] = None
    side["D"] = 2
    files["sidecar"].write_text(json.dumps(side, indent=1))
    line = tp.line_of(seed)
    node = site.nodes[seed % len(site.nodes)]
    contract.write_completion(
        out,
        tag,
        line,
        list(files.values()),
        site.plant(copy.deepcopy(identity), node=node, task=str(seed + 1)),
        0.0,
        1.0,
        {"boost": {"attempted": False, "accepted": False}},
    )


def use_constants(patch):
    """The promotion of this module's small gate: its allocation and
    defaults, the stand-in gate script, no check of clean checkouts, and
    the copies taken as tracked."""
    patch.setattr(promote, "SUITE", "smoke")
    patch.setattr(promote, "LABELS", [LABEL])
    patch.setattr(promote, "SEEDS", "0-2")
    patch.setattr(promote, "DEFAULT_CONFIGS", (LABEL,))
    patch.setattr(promote, "clean_required", lambda: False)
    patch.setattr(promote, "is_tracked", lambda path: True)
    patch.setattr(gates, "GATE_RUNS", tg.RUNS)


def redact(site, campaign, target):
    """The tracked copies of ``campaign``, run at the stand-in ``site``,
    redacted as its operator redacts them."""
    contract.redact(
        campaign,
        target,
        operator=site.operator(),
        environ={},
        host="fakelogin9",
        say=lambda m: None,
    )


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """The root of a promotion before ``prepare``, with the local
    directories of the raw arms, the fingerprints and the gate runs under
    ``local/``."""
    try:
        gates.gpyreg_tree()
    except contract.IdentityError as error:
        pytest.skip(f"no gpyreg checkout: {error}")
    base = tmp_path_factory.mktemp("world")
    root = base / "root"
    for rel in FILES:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(promote.REPO / rel, root / rel)
    shutil.copytree(promote.REPO / promote.BASELINE, root / promote.BASELINE)
    gate = root / GATE
    gate.mkdir(parents=True)
    (gate / "README.md").write_text("The release gate's records.\n")
    # The arms run at a stand-in site, in its operator's home.
    site = stubs.FakeSite(base)
    raw = site.home / "runs"
    raw.mkdir(parents=True)
    after_identity, before_identity = (site.plant(i) for i in identities())
    script = tg.stand_in(base)
    assert (
        gates.main(
            ["run", "--out", str(base / "local/gates")]
            + ["--script", str(script), "--allow-dirty"]
        )
        == 0
    )
    with pytest.MonkeyPatch.context() as patch:
        use_constants(patch)
        block = site.site_block("dev/scripts/population_run.py")
        for name, value in block.items():
            if value is None:
                patch.delenv(name, raising=False)
            else:
                patch.setenv(name, value)
        patch.delenv("PYVBMC_SOURCE", raising=False)
        patch.setattr(contract, "pip_freeze", lambda: ["pyvbmc==0"])
        before = tp.prepare_other_arm(
            raw, patch, "population_before", before_identity
        )
        for seed in range(3):
            write_arm_case(
                site, before, seed, before_identity, 0.2 * (seed + 1)
            )
        tp.use_identity(patch, before_identity)
        assert runner.main(["verify", "--out", str(before)]) == 0
        tp.finish(before)
        tp.use_identity(patch, after_identity)
        after = raw / "population_after"
        assert (
            runner.main(
                ["prepare", "--out", str(after), *tp.ARGUMENTS]
                + ["--arm", "after", "--pair", str(before)]
            )
            == 0
        )
        for seed in range(3):
            write_arm_case(site, after, seed, after_identity)
        assert runner.main(["verify", "--out", str(after)]) == 0
        tp.finish(after)
        for arm in (before, after):
            redact(site, arm, gate / arm.name)
        promote.analysis.analyze_arms(
            gate / "population_before",
            gate / "population_after",
            None,
            gate / "assessment",
        )
        fingerprints = base / "local/fingerprints"
        assert (
            promote.main(
                ["fingerprints", "--after", str(gate / "population_after")]
                + ["--out", str(fingerprints)],
                root=root,
            )
            == 0
        )
    assert site.leaks(gate) == []
    return {
        "base": base,
        "script": str(script.resolve()),
        "raw_after": raw.relative_to(base) / "population_after",
    }


@pytest.fixture
def gate(world, tmp_path, monkeypatch):
    """A copy of the world, with the module's constants in force."""
    base = tmp_path / "world"
    shutil.copytree(world["base"], base)
    use_constants(monkeypatch)
    monkeypatch.setattr(gates, "GATE_SCRIPT", world["script"])
    root = base / "root"
    return {
        "root": root,
        "after": root / GATE / "population_after",
        "raw_after": base / world["raw_after"],
        "assessment": root / GATE / "assessment",
        "record": root / "dev/golden/promotion_test",
        "fingerprints": base / "local/fingerprints",
        "gates": base / "local/gates",
        "replay": base / "local/replay",
    }


def prepare(gate, *extra, accepted=None, after=None):
    accepted = accepted or promote.sha256(
        gate["assessment"] / "assessment.json", True
    )
    return promote.main(
        [
            "prepare",
            "--after",
            str(after or gate["after"]),
            "--assessment",
            str(gate["assessment"]),
            "--accepted-assessment",
            accepted,
            "--fingerprints",
            str(gate["fingerprints"]),
            "--gate-runs",
            str(gate["gates"]),
            "--record",
            str(gate["record"]),
            "--name",
            NAME,
            *extra,
        ],
        root=gate["root"],
    )


def replay(gate):
    return promote.main(
        ["replay", "--record", str(gate["record"])]
        + ["--out", str(gate["replay"])],
        root=gate["root"],
    )


def publish(gate, *extra):
    return promote.main(
        ["publish", "--record", str(gate["record"])]
        + ["--final-replay", str(gate["replay"]), *extra],
        root=gate["root"],
    )


def write_record_readme(gate):
    (gate["record"] / "README.md").write_text(
        f"# Golden reference promotion: {NAME}\n", encoding="utf-8"
    )


def snapshot(root):
    """Every file under ``root`` and its bytes."""
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in Path(root).rglob("*")
        if p.is_file()
    }


def flat(text):
    return " ".join(text.split())


# --------------------------------------------------------------------------
# The promotion
# --------------------------------------------------------------------------


def test_the_fingerprints_meet_the_envelope_of_the_after_arm(gate):
    report = contract.read_json(gate["fingerprints"] / "replay.json")
    [row] = report["rows"]
    assert (row["label"], row["seed"]) == (LABEL, 0)
    assert row["verdict"].startswith("finals only")
    assert not row["flagged"] and row["outside"] == []
    assert f"baseline `{promote.NO_TRACES}`" in (
        gate["fingerprints"] / "replay.md"
    ).read_text(encoding="utf-8")
    # Made once: a directory in use is refused.
    assert (
        promote.main(
            ["fingerprints", "--after", str(gate["after"])]
            + ["--out", str(gate["fingerprints"])],
            root=gate["root"],
        )
        == 1
    )


def test_prepare_replay_and_publish(gate):
    root = gate["root"]
    before = snapshot(root / "dev/golden/baseline")
    assert prepare(gate) == 0
    record = gate["record"]
    validation = contract.read_json(record / promote.VALIDATION)
    assert validation["status"] == "prepared"
    assert validation["reference"] == NAME
    assert validation["population"]["runs"] == 3
    # Three seeds give the KS screen too few runs to test; the count is
    # read from the report, whose real form the previous promotion's has.
    assert validation["population"]["even_odd_tests"] == 0
    previous = promote.REPO / promote.PREVIOUS_RECORD / "even_vs_odd.md"
    assert promote.ks_tests(previous.read_text(encoding="utf-8")) == 92
    assert validation["population"]["even_odd_flagged"] == []
    assert validation["fingerprints"]["cases"] == 1
    assert validation["fingerprints"]["outside_envelope"] == {}
    assert validation["gate_runs"]["identical"] is True
    assert validation["gate_runs"]["host"] == socket.gethostname()
    assert validation["previous"]["traces"] == "absent from this machine"
    assert validation["assessment"]["before"]["pyvbmc"] == "b" * 40
    for name in (
        "previous_reference_README.md",
        "fingerprint_replay.json",
        "fingerprint_replay.md",
        gates.RECORD,
        "even_vs_odd.md",
        promote.MANIFEST,
    ):
        assert (record / name).is_file(), name
    assert promote.sha256(
        record / "previous_reference_README.md", True
    ) == promote.sha256(promote.REPO / "dev/golden/README.md", True)
    traces = root / promote.GOLDEN_RUNS / f"{NAME}_fingerprints"
    stem = f"{LABEL}_seed0"
    for suffix in (".json", ".npz"):
        assert promote.sha256(traces / f"{stem}{suffix}") == promote.sha256(
            gate["fingerprints"] / f"{stem}{suffix}"
        )
    assert (traces / "gate_runs" / gates.RECORD).is_file()
    # prepare changes nothing tracked but the record.
    assert snapshot(root / "dev/golden/baseline") == before
    # Again: the copies match, and the record is written anew under the
    # name it has.
    assert (
        promote.main(
            [
                "prepare",
                "--after",
                str(gate["after"]),
                "--assessment",
                str(gate["assessment"]),
                "--accepted-assessment",
                promote.sha256(gate["assessment"] / "assessment.json", True),
                "--fingerprints",
                str(gate["fingerprints"]),
                "--gate-runs",
                str(gate["gates"]),
                "--record",
                str(record),
            ],
            root=root,
        )
        == 0
    )
    assert contract.read_json(record / promote.VALIDATION)["reference"] == NAME

    assert replay(gate) == 0
    # The documents link to the record's README.
    assert publish(gate) == 1
    write_record_readme(gate)
    eols = {
        file: promote.read_text(root / file)[1]
        for file in ("AGENTS.md", "dev/README.md", "dev/golden/README.md")
    }
    public = "the asset `population_after_traces.tar.zst` of the release"
    assert publish(gate, "--public-traces", public) == 0

    baseline = root / promote.BASELINE
    assert sorted(p.name for p in baseline.iterdir()) == sorted(
        [f"{LABEL}_seed{seed}.json" for seed in range(3)] + ["summary.md"]
    )
    for seed in range(3):
        rel = runner.case_files(LABEL, seed)["sidecar"]
        assert (baseline / f"{LABEL}_seed{seed}.json").read_bytes() == (
            gate["after"] / rel
        ).read_bytes()
    assert (
        (baseline / "summary.md")
        .read_text()
        .startswith(f"# Golden population {NAME}")
    )
    module = load_module(root / "dev/scripts/golden_replay.py")
    assert module.DEFAULT_BASELINE.name == f"{NAME}_fingerprints"
    assert module.DEFAULT_CONFIGS == (LABEL,)
    assert (
        "student_D4" not in (root / "dev/scripts/golden_replay.py").read_text()
    )
    for file, eol in eols.items():
        assert promote.read_text(root / file)[1] == eol
    agents = promote.passage(root, "agents")
    assert f"`{NAME}`" in agents and promote.PREVIOUS not in agents
    text = promote.read_text(root / "dev/README.md")[0]
    reference = text[
        text.index("The current golden reference is") : text.index(
            promote.PASSAGES["readme_reference"][2]
        )
    ]
    assert reference.endswith(".\n\n")
    assert "scripts/reference_promote.py" not in text
    assert "- `scripts/seeded_gate_runs.py` —" in text
    entry = promote.passage(root, "readme_replay")
    readme = promote.read_text(root / "dev/golden/README.md")[0]
    for passage, phrases in (
        (
            reference,
            [
                f"`{NAME}`: **3 runs of the 1",
                "at seeds 0–2",
                "(golden/promotion_test/README.md)",
                "which the PI accepted",
                "The one default cases",
            ],
        ),
        (entry, [f"`scripts/runs/golden/{NAME}_fingerprints/`"]),
        (
            readme,
            [
                f"**`{NAME}`: 3 runs",
                "(promotion_test/README.md)",
                "(../experiments/release_gate_test/population_after)",
                "(../experiments/release_gate_test/README.md)",
                "only the repository's collaborators see",
                f"They are published as {public}.",
                "none lies outside its configuration's accuracy envelope",
                "0 of the 1 equal the cluster's run of seed 0",
                "the same runs of the code of `bbbbbbbb`",
                "A machine gets replay fingerprints of its own",
            ],
        ),
    ):
        for phrase in phrases:
            assert phrase in flat(passage), phrase
    for passage in (agents, reference, entry, readme):
        for line in passage.split("\n"):
            assert not line.startswith(("+ ", "1. ")), line
    for file, old, new in promote.REPLACEMENTS:
        text = promote.read_text(root / file)[0]
        assert old not in text
        assert new.format(name=NAME, traces=f"{NAME}_fingerprints") in text
    assert (record / "promote.py").read_bytes() == (
        promote.HERE / "reference_promote.py"
    ).read_bytes()
    validation = contract.read_json(record / promote.VALIDATION)
    assert validation["status"] == "promoted"
    assert validation["publication"]["final_replay"]["identical"] == 1
    assert validation["publication"]["public_traces"] == public
    assert (record / "final_replay.json").is_file()
    # Once only.
    assert publish(gate) == 1


def load_module(path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("promoted_replay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_refuses_and_writes_nothing(gate, monkeypatch, capsys):
    root = gate["root"]

    def refused(*phrases, **kwargs):
        before = snapshot(root)
        assert prepare(gate, **kwargs) == 1
        out = capsys.readouterr().out
        for phrase in phrases:
            assert phrase in out, (phrase, out)
        assert snapshot(root) == before

    refused("is not the accepted assessment", accepted="0" * 64)
    # The campaign directory in place of its redacted copies, where a
    # hand-back would unpack it.
    raw = root / "dev/scripts/runs/population_after"
    shutil.copytree(gate["raw_after"], raw)
    refused("holds no redaction.json", after=raw)
    with monkeypatch.context() as patch:
        patch.setattr(promote, "is_tracked", lambda path: False)
        refused("is not tracked by git")
    # This process's code, or the fingerprints', is not the after arm's.
    with monkeypatch.context() as patch:
        patch.setattr(promote, "numerics_differ", lambda a, b: ["pyvbmc/x.py"])
        refused("prepare: HEAD's code differs from the after arm's")
    # A fingerprint judged against another population, and one outside its
    # envelope where no run of the population lies outside its own.
    path = gate["fingerprints"] / "replay.json"
    saved = path.read_bytes()
    report = json.loads(saved)
    report["rows"][0]["pop_fence"]["gskl"] += 1.0
    path.write_text(json.dumps(report))
    refused("judged by another population")
    report = json.loads(saved)
    report["rows"][0]["outside"] = ["gskl"]
    path.write_text(json.dumps(report))
    refused("1 of 1 fingerprints lie outside their envelopes")
    path.write_bytes(saved)
    # A fingerprint of another gpyreg, and one of other options.
    side_path = gate["fingerprints"] / f"{LABEL}_seed0.json"
    saved = side_path.read_bytes()
    side = json.loads(saved)
    side["meta"]["gpyreg_source"]["git"]["sha"] = "d" * 7
    side_path.write_text(json.dumps(side))
    refused("the fingerprint's gpyreg is not the after arm's")
    side = json.loads(saved)
    side["effective_options"]["max_fun_evals"] += 1
    side_path.write_text(json.dumps(side))
    refused("other options than the after arm's runs: ['max_fun_evals']")
    side_path.write_bytes(saved)
    # Gate runs of another machine.
    with monkeypatch.context() as patch:
        patch.setattr(promote.socket, "gethostname", lambda: "elsewhere")
        refused("not on this machine")
    # An assessment of another candidate, accepted as it is.
    path = gate["assessment"] / "assessment.json"
    saved = path.read_bytes()
    assessment = json.loads(saved)
    assessment["arms"]["candidate"]["name"] = "elsewhere"
    path.write_text(json.dumps(assessment))
    refused(
        "does not assess population_after",
        accepted=promote.sha256(path, True),
    )
    path.write_bytes(saved)
    # A baseline that no longer holds the previous reference.
    (root / promote.BASELINE / "normal_D5_seed0.json").unlink()
    refused("does not hold the sidecars")


def test_the_gate_runs_must_come_from_clean_checkouts(gate, monkeypatch):
    campaign, _, _ = promote.read_after(gate["after"], {})
    monkeypatch.setattr(promote, "clean_required", lambda: True)
    with pytest.raises(promote.PromotionError, match="dirty checkout"):
        promote.check_gate_runs(gate["gates"], campaign)


def test_cases_not_verified_need_a_ruling(gate, monkeypatch):
    load = promote.analysis.load_array_campaign

    def with_a_failure(*args):
        campaign = load(*args)
        stem = f"{LABEL}_seed2"
        del campaign["rows"][stem]
        campaign["cases"][stem] = (LABEL, 2, "failed")
        return campaign

    monkeypatch.setattr(
        promote.analysis, "load_array_campaign", with_a_failure
    )
    with pytest.raises(promote.PromotionError, match="without a ruling"):
        promote.read_after(gate["after"], {})
    rulings = {f"{LABEL}/{LABEL}_seed2": "a node failure, not the code's"}
    _, sidecars, _ = promote.read_after(gate["after"], rulings)
    assert sorted(sidecars) == [f"{LABEL}_seed0", f"{LABEL}_seed1"]
    with pytest.raises(promote.PromotionError, match="verified or unknown"):
        promote.read_after(
            gate["after"], {**rulings, f"{LABEL}/{LABEL}_seed0": "x"}
        )


def test_publish_refuses_and_writes_nothing(gate, capsys):
    root = gate["root"]
    assert prepare(gate) == 0
    assert replay(gate) == 0
    write_record_readme(gate)

    def refused(phrase):
        before = snapshot(root)
        assert publish(gate) == 1
        out = capsys.readouterr().out
        assert phrase in out, out
        assert snapshot(root) == before

    # A passage edited since the script was written.
    agents = root / "AGENTS.md"
    saved = agents.read_bytes()
    agents.write_bytes(
        saved.replace(b"step by step with its stored trace", b"step by step")
    )
    refused("AGENTS.md: the passage '- **Trajectories.**' has changed")
    agents.write_bytes(saved)
    # A line to replace that is no longer there once.
    path = root / "dev/scripts/analyze_population_run.py"
    saved = path.read_bytes()
    path.write_bytes(saved + promote.REPLACEMENTS[-1][1].encode())
    refused("times")
    path.write_bytes(saved)
    # A sidecar of the after arm changed since prepare.
    rel = runner.case_files(LABEL, 1)["sidecar"]
    side_path = gate["after"] / rel
    saved = side_path.read_bytes()
    side_path.write_bytes(saved.replace(b'"seed": 1', b'"seed":  1'))
    refused("is not the prepared sidecar")
    side_path.write_bytes(saved)
    # A replay that is not identical.
    path = gate["replay"] / "replay.json"
    report = json.loads(path.read_text())
    report["rows"][0]["identical"] = False
    path.write_text(json.dumps(report))
    refused("the replay is not identical")
