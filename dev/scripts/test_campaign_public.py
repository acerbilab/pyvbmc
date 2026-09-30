"""Checks of the public asset of a campaign (``campaign_public.py``).

A campaign of ``population_run.py`` on the ``smoke`` suite's ``normal_D2``
is prepared at a stand-in site (``campaign_slurm_stubs.FakeSite``), two of
its three cases written as ``test_population_run.py`` writes them, verified,
finished, and redacted into tracked copies as a hand-back redacts them; the
asset is built from the campaign and those copies with the site's operator.
"""

import copy
import io
import json
import shutil
import subprocess
import tarfile

import campaign_public as public
import campaign_slurm_stubs as stubs
import numpy as np
import pytest
import test_population_run as tp

runner = tp.runner
contract = tp.contract
LABEL = tp.LABEL


def finished_campaign(tmp_path, monkeypatch, plant=None, spec=None):
    """``(site, campaign, copies)``: a finished campaign and its tracked
    copies. ``plant(files)`` changes the artifacts of seed 1 before its
    completion record; ``spec`` replaces the manifest's declaration of the
    tracked copies before the redaction."""
    site = stubs.FakeSite(tmp_path)
    for name, value in site.site_block(
        "dev/scripts/population_run.py"
    ).items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    tp.use_identity(monkeypatch, tp.sited_identity(site))
    monkeypatch.setattr(
        contract,
        "pip_freeze",
        lambda: ["pyvbmc==0", f"gpyreg @ file://{site.gpyreg.as_posix()}"],
    )
    out = site.home / "runs" / "population_after"
    assert runner.main(["prepare", "--out", str(out), *tp.ARGUMENTS]) == 0
    for seed in (0, 1):
        identity = tp.sited_identity(site, site.nodes[seed])
        tag, files = tp.complete_case(
            out, seed, exact_metrics=True, identity=identity
        )
        if seed == 1 and plant is not None:
            plant(files)
            contract.write_completion(
                out,
                tag,
                tp.line_of(seed),
                list(files.values()),
                copy.deepcopy(identity),
                0.0,
                1.0,
                {"boost": {"attempted": False, "accepted": False}},
            )
    assert runner.main(["verify", "--out", str(out)]) == 0
    tp.finish(out)
    site.write_slurm(out)
    if spec is not None:
        manifest = contract.read_json(out / "manifest.json")
        manifest["tracked_copies"] = spec
        contract.write_json(out / "manifest.json", manifest)
    copies = tmp_path / "handback" / "population_after"
    contract.redact(
        out,
        copies,
        operator=site.operator(),
        environ={},
        host="fakelogin9",
        say=lambda message: None,
    )
    return site, out, copies


def build(site, campaign, copies, out, **kwargs):
    return public.build(
        campaign,
        copies,
        out,
        operator=site.operator(),
        environ={},
        host="fakelogin9",
        say=lambda message: None,
        **kwargs,
    )


def members(directory):
    """``{name: bytes}`` of the asset in ``directory``."""
    listing = next(directory.glob("*.sha256")).read_text().splitlines()
    parts = [directory / line.split("  ")[1] for line in listing]
    data = b"".join(part.read_bytes() for part in parts)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        return {m.name: archive.extractfile(m).read() for m in archive}


def test_the_asset_holds_the_copies_and_the_numeric_files(
    tmp_path, monkeypatch
):
    site, campaign, copies = finished_campaign(tmp_path, monkeypatch)
    out = tmp_path / "public"
    record = build(site, campaign, copies, out)
    found = members(out)
    prefix = f"{campaign.name}/"
    assert all(name.startswith(prefix) for name in found)
    names = {name[len(prefix) :] for name in found}
    copied = {
        p.relative_to(copies).as_posix()
        for p in copies.rglob("*")
        if p.is_file()
    }
    numeric = {
        runner.case_files(LABEL, seed)[key]
        for seed in (0, 1)
        for key in ("trace", "posterior")
    }
    assert names == copied | numeric | {public.PUBLIC}
    # The copies as they are, the numeric files as the campaign holds them,
    # no pickle.
    for name in copied:
        assert found[prefix + name] == (copies / name).read_bytes(), name
    for name in numeric:
        assert found[prefix + name] == (campaign / name).read_bytes(), name
    assert not any(name.endswith(".pkl") for name in names)
    assert record["left_out"] == {".boost.pkl": 2}
    assert record["kinds"] == {"numeric": 4, "tracked copy": len(copied) - 1}
    assert record == json.loads(found[prefix + public.PUBLIC])
    for name, entry in record["files"].items():
        assert public.sha256(found[prefix + name]) == entry["sha256"]
    # Nothing of the site remains in any file of the asset.
    extracted = tmp_path / "extracted"
    for name, data in found.items():
        (extracted / name).parent.mkdir(parents=True, exist_ok=True)
        (extracted / name).write_bytes(data)
    assert site.leaks(extracted) == []
    assert site.leaks(campaign)  # which the campaign itself holds
    assert public.check(out) == []
    assert public.main(["check", str(out)]) == 0
    # Once: an asset of the campaign in the directory already is refused.
    with pytest.raises(contract.ContractError, match="already"):
        build(site, campaign, copies, out)


def test_an_asset_in_parts_and_a_damaged_part(tmp_path, monkeypatch):
    site, campaign, copies = finished_campaign(tmp_path, monkeypatch)
    out = tmp_path / "public"
    build(site, campaign, copies, out, part_size=4096)
    parts = sorted(out.glob(f"{public.asset_name(campaign)}.[0-9]*"))
    assert len(parts) > 1
    assert all(part.stat().st_size <= 4096 for part in parts)
    assert public.check(out) == []
    data = bytearray(parts[1].read_bytes())
    data[0] ^= 0xFF
    parts[1].write_bytes(bytes(data))
    assert public.check(out) == [
        f"{parts[1].name} is not the part its listing hashes"
    ]
    assert public.main(["check", str(out)]) == 1


def write_small_asset(out, name, files=None, built_by=None):
    """A whole asset ``name`` in ``out``: ``files`` (a tracked copy
    ``a.txt`` by default), a ``redaction.json`` that lists ``a.txt`` as a
    copy, and ``public.json`` with ``built_by`` (this code's by default),
    in one part, with its listing."""
    files = {"a.txt": b"x\n"} if files is None else dict(files)
    files[contract.REDACTION] = json.dumps({"files": {"a.txt": {}}}).encode()
    record = {
        "built_by": public.building_code() if built_by is None else built_by,
        "files": {n: {"sha256": public.sha256(d)} for n, d in files.items()},
    }
    members = {f"{name}/{n}": d for n, d in files.items()}
    members[f"{name}/{public.PUBLIC}"] = json.dumps(record).encode("utf-8")
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for member, content in members.items():
            info = tarfile.TarInfo(member)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    part = out / f"{name}.public.tar.gz.000"
    part.write_bytes(stream.getvalue())
    (out / f"{name}.public.tar.gz.sha256").write_text(
        f"{public.sha256(part.read_bytes())}  {part.name}\n", encoding="utf-8"
    )
    return part


def test_the_check_reads_every_asset_of_a_directory(tmp_path, monkeypatch):
    """A release's directory holds the asset of each campaign; the check
    reads them all, and a damaged one fails it."""
    site, campaign, copies = finished_campaign(tmp_path, monkeypatch)
    out = tmp_path / "public"
    build(site, campaign, copies, out)
    part = write_small_asset(out, "other")
    assert public.assets(out) == sorted(
        ["other.public.tar.gz", public.asset_name(campaign)]
    )
    assert public.check(out) == []
    assert public.main(["check", str(out)]) == 0
    # A part that no listing names, as a failed build may leave one.
    stray = out / "third.public.tar.gz.000"
    stray.write_bytes(b"\0")
    assert public.check(out) == [
        f"{stray.name} is a part that no listing names"
    ]
    stray.unlink()
    part.write_bytes(part.read_bytes() + b"\0")
    assert public.check(out) == [
        f"{part.name} is not the part its listing hashes"
    ]
    assert public.check(tmp_path / "nothing") == [
        f"{tmp_path / 'nothing'} holds no asset listing"
    ]


def test_the_check_applies_the_rules_that_need_no_campaign(tmp_path):
    """An asset built by other code, a member that is no copy and no .npz
    or .json file, a numeric file that holds text and an absolute path in a
    copy each fail the check."""
    out = tmp_path / "public"
    out.mkdir()
    write_small_asset(out, "other", built_by={"commit": None, "sha256": {}})
    [problem] = public.check(out)
    assert "was built by other code than this check's" in problem
    assert "campaign_contract.py" in problem and "build it again" in problem
    for path in out.iterdir():
        path.unlink()
    buffer = io.BytesIO()
    np.savez(buffer, words=np.array(["a"]))
    write_small_asset(
        out,
        "other",
        files={
            "a.txt": b"made in /home/someone/runs\n",
            "run.boost.pkl": b"\x80\x04.",
            "t.npz": buffer.getvalue(),
        },
    )
    assert public.check(out) == [
        "other/a.txt names the absolute path /home/someone/runs",
        "other/run.boost.pkl is neither a tracked copy nor a .npz or .json "
        "file",
        "other/t.npz: words: an array of kind 'U'",
    ]


def test_the_asset_holds_each_case_record_where_no_copy_does(
    tmp_path, monkeypatch
):
    """A harness whose tracked copies hold no case records (the pools'):
    the asset holds each verified case's completion record, redacted,
    which ties the case's published files to their source hashes; it
    records the code that built it and the exemptions it applied, and
    passes the check."""
    site, campaign, copies = finished_campaign(
        tmp_path, monkeypatch, spec={"files": ["summary.md"]}
    )
    out = tmp_path / "public"
    record = build(site, campaign, copies, out)
    assert record["kinds"]["record"] == 2
    found = members(out)
    for seed in (0, 1):
        tag = f"{LABEL}/{LABEL}_seed{seed}"
        name = contract.record_path(campaign, tag).relative_to(campaign)
        name = name.as_posix()
        text = found[f"{campaign.name}/{name}"].decode("utf-8")
        assert site.user not in text and site.family in text
        assert record["files"][name]["source_sha256"] == public.sha256(
            (campaign / name).read_bytes()
        )
        # Each published artifact's source hash is the one its record lists.
        listed = json.loads(text)["artifacts"]
        published = [a for a in listed if a in record["files"]]
        assert published
        for artifact in published:
            assert (
                record["files"][artifact]["source_sha256"]
                == listed[artifact]["sha256"]
            )
    assert record["built_by"] == public.building_code()
    assert record["exemptions"] == {
        "allowed": {},
        "allowed_beyond_the_copies": [],
        "paths": [],
    }
    assert public.check(out) == []


def test_a_json_artifact_that_is_no_tracked_copy_is_redacted(
    tmp_path, monkeypatch
):
    # A harness whose tracked copies are not its cases' JSON files (the
    # pools', say): the sidecars enter the asset redacted.
    spec = {
        "files": ["summary.md"],
        "cases": {"record": True, "artifacts": ["*.boost.json"]},
    }
    site, campaign, copies = finished_campaign(
        tmp_path, monkeypatch, spec=spec
    )
    sidecar = runner.case_files(LABEL, 0)["sidecar"]
    assert not (copies / sidecar).exists()
    raw = (campaign / sidecar).read_text()
    assert str(site.home) in raw.replace("\\\\", "\\") or site.user in raw
    out = tmp_path / "public"
    record = build(site, campaign, copies, out)
    assert record["kinds"]["redacted"] == 2
    found = members(out)
    redacted = json.loads(found[f"{campaign.name}/{sidecar}"])
    assert redacted["final"] == json.loads(raw)["final"]
    assert site.user not in json.dumps(redacted)
    assert record["files"][sidecar]["source_sha256"] == public.sha256(
        (campaign / sidecar).read_bytes()
    )


def test_an_npz_that_holds_more_than_numbers_is_refused(tmp_path, monkeypatch):
    def plant(files):
        with np.load(files["trace"]) as trace:
            arrays = dict(trace)
        np.savez(files["trace"], **arrays, note=np.array(["a word"]))

    site, campaign, copies = finished_campaign(
        tmp_path, monkeypatch, plant=plant
    )
    out = tmp_path / "public"
    with pytest.raises(contract.ContractError, match="note: an array of kind"):
        build(site, campaign, copies, out)
    assert not out.exists() or not any(out.iterdir())
    # Objects, a name that holds the operator's username, and a file that
    # is not an archive, each tested on its own.
    read = contract._read_campaign(campaign)
    redaction = contract.redaction_rules(
        campaign,
        read["manifest"],
        read["documents"],
        site.operator(),
        {},
        (),
        "fakelogin9",
    )
    buffer = io.BytesIO()
    np.savez(
        buffer,
        objects=np.array([{"a": 1}], dtype=object),
        **{f"run of {site.user}": np.zeros(2)},
    )
    problems = public.npz_problems(buffer.getvalue(), redaction)
    assert problems[0] == "objects: an array of objects"
    assert "the username" in problems[1] and site.user in problems[1]
    assert public.npz_problems(b"not a zip", redaction)[0].startswith(
        "not a readable .npz file"
    )


def test_copies_of_another_state_are_refused(tmp_path, monkeypatch):
    site, campaign, copies = finished_campaign(tmp_path, monkeypatch)
    out = tmp_path / "public"
    summary = campaign / "summary.md"
    summary.write_text(summary.read_text() + "\n")
    with pytest.raises(contract.ContractError, match="redact the campaign"):
        build(site, campaign, copies, out)
    shutil.rmtree(copies / "rescored")
    with pytest.raises(contract.ContractError, match="redact the campaign"):
        build(site, campaign, copies, out)
    assert not out.exists()


def test_the_wrapper_parses():
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("no bash")
    script = public.HERE / "hpc" / "campaign_public.sh"
    result = subprocess.run(
        [bash, "-n", script.as_posix()], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    usage = subprocess.run(
        [bash, script.as_posix()], capture_output=True, text=True
    )
    assert usage.returncode == 64
    assert "campaign_public.sh CAMPAIGN_DIR COPIES_DIR OUT_DIR" in usage.stderr
