"""Checks of the promotion of the release gate's reference (``reference_promote.py``).

A promotion runs under a temporary root that holds copies of what it reads
and rewrites in the checkout (``AGENTS.md``, ``dev/README.md``,
``dev/golden/README.md`` and ``baseline/``, the previous promotion's
manifest, ``golden_replay.py`` and ``analyze_population_run.py``) and the
records of a small release gate: two campaigns of array mode on the
``smoke`` suite's ``normal_D2`` at seeds 0-2, written as
``test_population_run.py`` writes them, an after arm of the harness
checkout's code and a before arm of other code, verified, rescored and
assessed. The fingerprints and the replay of the defaults are real runs of
``normal_D2`` at seed 0 (seconds each) through ``golden_replay.py``; the
gate runs are those of the stand-in gate script of
``test_seeded_gate_runs.py``. The identities name the checkout's commit and
its gpyreg's, and the checks of clean checkouts are off, so that the module
runs on a working tree with changes. The tests that record skip without a
gpyreg checkout.

The checkout's own passages are checked against the text the promotion was
written against: an edit to one of them fails here until its template and
its SHA-256 in ``PASSAGES`` follow it.
"""

import copy
import json
import shutil
import socket
from pathlib import Path

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
# The checkout's passages
# --------------------------------------------------------------------------


def test_the_passages_are_those_the_promotion_rewrites():
    assert promote.passage_problems(promote.REPO) == []
    for file, old, _ in promote.REPLACEMENTS:
        text, _ = promote.read_text(promote.REPO / file)
        assert text.count(old) == 1, (file, old)


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


def write_arm_case(out, seed, identity, shift=0.0):
    """A verified-looking case (``test_population_run.complete_case``)
    whose sidecar holds the effective options of a real run."""
    tag, files = tp.complete_case(
        out, seed, exact_metrics=True, identity=identity, shift=shift
    )
    side = json.loads(files["sidecar"].read_text())
    side["effective_options"] = effective(seed)
    side["noise_sd"] = None
    side["D"] = 2
    files["sidecar"].write_text(json.dumps(side, indent=1))
    line = tp.line_of(seed)
    contract.write_completion(
        out,
        tag,
        line,
        list(files.values()),
        copy.deepcopy(identity),
        0.0,
        1.0,
        {"boost": {"attempted": False, "accepted": False}},
    )


def use_constants(patch):
    """The promotion of this module's small gate: its allocation and
    defaults, the stand-in gate script, and no check of clean checkouts."""
    patch.setattr(promote, "SUITE", "smoke")
    patch.setattr(promote, "LABELS", [LABEL])
    patch.setattr(promote, "SEEDS", "0-2")
    patch.setattr(promote, "DEFAULT_CONFIGS", (LABEL,))
    patch.setattr(promote, "clean_required", lambda: False)
    patch.setattr(gates, "GATE_RUNS", tg.RUNS)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """The root of a promotion before ``prepare``, with the local
    directories of the fingerprints and the gate runs under ``local/``."""
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
    after_identity, before_identity = identities()
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
        patch.setenv("PYVBMC_GPYREG_SOURCE", str(base / "gpyreg"))
        patch.delenv("PYVBMC_SOURCE", raising=False)
        patch.setattr(contract, "pip_freeze", lambda: ["pyvbmc==0"])
        before = tp.prepare_other_arm(
            gate, patch, "population_before", before_identity
        )
        for seed in range(3):
            write_arm_case(before, seed, before_identity, 0.2 * (seed + 1))
        tp.use_identity(patch, before_identity)
        assert runner.main(["verify", "--out", str(before)]) == 0
        tp.use_identity(patch, after_identity)
        after = gate / "population_after"
        assert (
            runner.main(
                ["prepare", "--out", str(after), *tp.ARGUMENTS]
                + ["--arm", "after", "--pair", str(before)]
            )
            == 0
        )
        for seed in range(3):
            write_arm_case(after, seed, after_identity)
        assert runner.main(["verify", "--out", str(after)]) == 0
        tp.finish(after)
        promote.analysis.analyze_arms(before, after, None, gate / "assessment")
        fingerprints = base / "local/fingerprints"
        assert (
            promote.main(
                ["fingerprints", "--after", str(after)]
                + ["--out", str(fingerprints)],
                root=root,
            )
            == 0
        )
    return {"base": base, "script": str(script.resolve())}


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
        "assessment": root / GATE / "assessment",
        "record": root / "dev/golden/promotion_test",
        "fingerprints": base / "local/fingerprints",
        "gates": base / "local/gates",
        "replay": base / "local/replay",
    }


def prepare(gate, *extra, accepted=None):
    accepted = accepted or promote.sha256(
        gate["assessment"] / "assessment.json"
    )
    return promote.main(
        [
            "prepare",
            "--after",
            str(gate["after"]),
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


def publish(gate):
    return promote.main(
        ["publish", "--record", str(gate["record"])]
        + ["--final-replay", str(gate["replay"])],
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
    assert validation["population"]["even_odd_flagged"] == []
    assert validation["fingerprints"]["cases"] == 1
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
    # Again: the copies match, and the record is written anew.
    assert prepare(gate) == 0

    assert replay(gate) == 0
    # The documents link to the record's README.
    assert publish(gate) == 1
    write_record_readme(gate)
    eols = {
        file: promote.read_text(root / file)[1]
        for file in ("AGENTS.md", "dev/README.md", "dev/golden/README.md")
    }
    assert publish(gate) == 0

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
    entry = promote.passage(root, "readme_replay")
    readme = promote.read_text(root / "dev/golden/README.md")[0]
    for passage, phrases in (
        (
            reference,
            [
                f"`{NAME}`: **3 runs of the 1",
                "(golden/promotion_test/README.md)",
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
                "0 of the 1 equal the cluster's run of seed 0",
            ],
        ),
    ):
        flat = " ".join(passage.split())
        for phrase in phrases:
            assert phrase in flat, phrase
    for passage in (agents, reference, entry, readme):
        for line in passage.split("\n"):
            assert not line.startswith(("+ ", "1. ")), line
    for file, old, new in promote.REPLACEMENTS:
        text = promote.read_text(root / file)[0]
        assert old not in text
        assert new.format(name=NAME, traces=f"{NAME}_fingerprints") in text
    validation = contract.read_json(record / promote.VALIDATION)
    assert validation["status"] == "promoted"
    assert validation["publication"]["final_replay"]["identical"] == 1
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
    before = snapshot(root)
    assert prepare(gate, accepted="0" * 64) == 1
    assert "is not the accepted assessment" in capsys.readouterr().out
    # A fingerprint judged against another population.
    path = gate["fingerprints"] / "replay.json"
    saved = path.read_bytes()
    report = json.loads(saved)
    report["rows"][0]["pop_fence"]["gskl"] += 1.0
    path.write_text(json.dumps(report))
    assert prepare(gate) == 1
    assert "judged by another population" in capsys.readouterr().out
    path.write_bytes(saved)
    # Gate runs of another machine.
    with monkeypatch.context() as patch:
        patch.setattr(promote.socket, "gethostname", lambda: "elsewhere")
        assert prepare(gate) == 1
    assert "not on this machine" in capsys.readouterr().out
    assert snapshot(root) == before
    # A baseline that no longer holds the previous reference.
    (root / promote.BASELINE / "normal_D5_seed0.json").unlink()
    before = snapshot(root)
    assert prepare(gate) == 1
    assert "does not hold the sidecars" in capsys.readouterr().out
    assert snapshot(root) == before


def test_cases_not_verified_need_a_ruling(gate, monkeypatch, tmp_path):
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
    campaign, sidecars, files = promote.read_after(gate["after"], rulings)
    assert sorted(sidecars) == [f"{LABEL}_seed0", f"{LABEL}_seed1"]
    with pytest.raises(promote.PromotionError, match="verified or unknown"):
        promote.read_after(
            gate["after"], {**rulings, f"{LABEL}/{LABEL}_seed0": "x"}
        )


def test_publish_refuses_a_changed_passage_or_replay(gate, capsys):
    root = gate["root"]
    assert prepare(gate) == 0
    assert replay(gate) == 0
    write_record_readme(gate)
    agents = root / "AGENTS.md"
    saved = agents.read_bytes()
    agents.write_bytes(
        saved.replace(b"step by step with its stored trace", b"step by step")
    )
    before = snapshot(root)
    assert publish(gate) == 1
    out = capsys.readouterr().out
    assert "AGENTS.md: the passage '- **Trajectories.**' has changed" in out
    assert snapshot(root) == before
    agents.write_bytes(saved)
    before = snapshot(root)
    path = gate["replay"] / "replay.json"
    report = json.loads(path.read_text())
    report["rows"][0]["identical"] = False
    path.write_text(json.dumps(report))
    assert publish(gate) == 1
    assert "the replay is not identical" in capsys.readouterr().out
    assert snapshot(root) == before
