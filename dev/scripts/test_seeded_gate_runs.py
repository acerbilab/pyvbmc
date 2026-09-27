"""Checks of the wrapper of the seeded gate runs (``seeded_gate_runs.py``).

The recordings run a stand-in gate script, written by the test, whose runs
draw a few numbers from their seed in place of a VBMC run, so that two
recordings take seconds; one stand-in draws from the process id as well, so
that its recordings differ. The real gate script is imported to check that
the wrapper reads its runs and adds the option to each, without running
them. The recordings take the gpyreg checkout from
``PYVBMC_GPYREG_SOURCE``, or else from the git checkout the imported gpyreg
lies in, and the tests that record skip without either.
"""

import copy
import textwrap

import numpy as np
import pytest
import seeded_gate_runs as gates

contract = gates.contract

STAND_IN = textwrap.dedent(
    """
    import os

    import numpy as np

    PERTURB = {perturb}


    def build_runs():
        return {{
            "first_run": (None, 2, 0, 1, 0, 1, 0, {{"display": "off"}}, 3),
            "second_run": (None, 3, 0, 1, 0, 1, 0, {{}}, 4, None),
        }}


    def record(name, spec):
        options, seed = spec[7], spec[8]
        assert options["performance_calibration"] == "off", options
        rng = np.random.default_rng(seed + (os.getpid() if PERTURB else 0))
        n = 4 + seed
        out = {{
            "elbo": rng.standard_normal(n),
            "func_count": np.arange(1, n + 1) * 5.0,
            "final_elbo": np.array([-1.5, 0.01]),
            "best_iter": np.array([n - 1]),
            "n_warps": np.array([1]),
        }}
        print(f"{{name}}: {{n}} iterations", flush=True)
        return {{f"{{name}}/{{k}}": v for k, v in out.items()}}


    def compare(path_a, path_b):
        a, b = np.load(path_a), np.load(path_b)
        keys = sorted(set(a.files) | set(b.files))
        differing = [
            key
            for key in keys
            if key not in a.files
            or key not in b.files
            or a[key].tobytes() != b[key].tobytes()
        ]
        print(f"{{len(keys)}} arrays compared, {{len(differing)}} differ")
        return 1 if differing else 0
    """
)
RUNS = ("first_run", "second_run")


@pytest.fixture(scope="module")
def gpyreg_checkout():
    try:
        return gates.gpyreg_tree()
    except contract.IdentityError as error:
        pytest.skip(f"no gpyreg checkout: {error}")


def stand_in(tmp_path, perturb=False):
    path = tmp_path / ("perturbed.py" if perturb else "stand_in.py")
    path.write_text(STAND_IN.format(perturb=perturb), encoding="utf-8")
    return path


def record_runs(tmp_path, script, name="gates"):
    out = tmp_path / name
    code = gates.main(
        ["run", "--out", str(out), "--script", str(script), "--allow-dirty"]
    )
    return code, out


def test_the_real_gate_script_takes_the_option_in_every_run():
    gate = gates.load_script(gates.ROOT / gates.GATE_SCRIPT)
    runs = gate.build_runs()
    assert tuple(runs) == gates.GATE_RUNS
    changed = gates.with_options_added(runs)
    for name, spec in runs.items():
        options = changed[name][gates.OPTIONS_INDEX]
        assert options == {**spec[gates.OPTIONS_INDEX], **gates.OPTIONS_ADDED}
        assert "performance_calibration" not in spec[gates.OPTIONS_INDEX]
        # Everything else in the specification is the gate script's.
        assert [
            v for i, v in enumerate(changed[name]) if i != gates.OPTIONS_INDEX
        ] == [v for i, v in enumerate(spec) if i != gates.OPTIONS_INDEX]


def test_an_option_the_gate_script_sets_otherwise_is_refused():
    runs = {
        "a": (None, 2, 0, 1, 0, 1, 0, {"performance_calibration": "cached"}, 1)
    }
    with pytest.raises(ValueError, match="performance_calibration"):
        gates.with_options_added(runs)
    runs = {"a": (None, 2, 0, 1, 0, 1, 0, None, 1)}
    with pytest.raises(TypeError, match="options dict"):
        gates.with_options_added(runs)


def test_two_recordings_identical_with_their_provenance(
    tmp_path, gpyreg_checkout, monkeypatch
):
    script = stand_in(tmp_path)
    code, out = record_runs(tmp_path, script)
    assert code == 0
    record = contract.read_json(out / gates.RECORD)
    assert record["identical"] and record["exit_code"] == 0
    assert record["comparison"]["identical"]
    assert "0 differ" in record["comparison"]["output"]
    assert record["script"] == {
        "path": str(script.resolve()),
        "sha256": contract.sha256_file(script),
    }
    head = contract.git(gates.ROOT, "rev-parse", "HEAD")
    for name in gates.RECORDINGS:
        entry = record["recordings"][name]
        child = entry["record"]
        assert child == contract.read_json(out / f"{name}.json")
        for suffix in ("npz", "json", "log"):
            assert entry["files"][suffix]["sha256"] == contract.sha256_file(
                out / f"{name}.{suffix}"
            )
        identity = child["identity"]
        # The commit, the gpyreg checkout, the threads and the host.
        assert identity["source"]["trees"]["harness"]["commit"] == head
        assert identity["imports"]["trees"]["gpyreg"]["path"] == str(
            gpyreg_checkout
        )
        assert identity["imports"]["modules"]["pyvbmc"] == str(
            gates.ROOT / "pyvbmc"
        )
        assert identity["host"]["threads"] == {
            key: "1" for key in gates.THREAD_KEYS
        }
        assert identity["host"]["hostname"]
        assert "blas" in identity["host"]
        assert child["options_added"] == gates.OPTIONS_ADDED
        assert list(child["runs"]) == list(RUNS)
        with np.load(out / f"{name}.npz") as arrays:
            arrays = dict(arrays)
        for run in RUNS:
            assert child["runs"][run]["sha256"] == gates.run_digest(
                arrays, run
            )
            assert (
                child["runs"][run]["best_iter"]
                == len(arrays[f"{run}/elbo"]) - 1
            )
        # The log holds what the recording printed.
        log = (out / f"{name}.log").read_text(encoding="utf-8")
        assert "first_run: 7 iterations" in log
    # The stand-in holds other runs than the gate script: the check says so,
    # and nothing else.
    _, problems = gates.check_record(out)
    assert problems == [
        f"{name} holds the runs of another gate script"
        for name in gates.RECORDINGS
    ]
    monkeypatch.setattr(gates, "GATE_RUNS", RUNS)
    assert gates.check_record(out)[1] == []
    assert gates.main(["check", str(out)]) == 0


def test_the_check_finds_a_changed_file_or_script(
    tmp_path, gpyreg_checkout, monkeypatch
):
    monkeypatch.setattr(gates, "GATE_RUNS", RUNS)
    script = stand_in(tmp_path)
    code, out = record_runs(tmp_path, script)
    assert code == 0
    saved = (out / "second.npz").read_bytes()
    with np.load(out / "first.npz") as arrays:
        arrays = dict(arrays)
    arrays["first_run/elbo"] = arrays["first_run/elbo"] + 1e-15
    np.savez(out / "second.npz", **arrays)
    problems = gates.check_record(out)[1]
    assert problems == ["second.npz is not the file recorded"]
    assert gates.main(["check", str(out)]) == 1
    (out / "second.npz").write_bytes(saved)
    script.write_text(script.read_text() + "\n# changed\n")
    problems = gates.check_record(out)[1]
    assert problems == [
        f"the gate script {record_script(out)} is not the one the "
        "recordings ran"
    ]


def record_script(out):
    return contract.read_json(out / gates.RECORD)["script"]["path"]


def test_recordings_that_differ_fail(tmp_path, gpyreg_checkout, monkeypatch):
    monkeypatch.setattr(gates, "GATE_RUNS", RUNS)
    code, out = record_runs(tmp_path, stand_in(tmp_path, perturb=True))
    assert code == gates.EXIT_FAILED
    record = contract.read_json(out / gates.RECORD)
    assert record["identical"] is False
    assert not record["comparison"]["identical"]
    assert record["same_source_identity"]
    assert gates.check_record(out)[1][0] == (
        "the record is not of two identical recordings"
    )


def test_a_failed_recording_stops_the_run(tmp_path, gpyreg_checkout):
    script = tmp_path / "broken.py"
    script.write_text("def build_runs():\n    raise RuntimeError('boom')\n")
    code, out = record_runs(tmp_path, script)
    assert code == 1
    record = contract.read_json(out / gates.RECORD)
    assert record["exit_codes"] == {"first": 1}
    assert record["identical"] is False and "comparison" not in record
    assert "RuntimeError: boom" in (out / "first.log").read_text()
    assert not (out / "second.log").exists()


def test_a_directory_in_use_or_a_dirty_checkout_is_refused(
    tmp_path, monkeypatch, capsys
):
    out = tmp_path / "used"
    out.mkdir()
    (out / "notes.txt").write_text("x")
    assert gates.main(["run", "--out", str(out)]) == gates.EXIT_USAGE
    assert "is not empty" in capsys.readouterr().out
    identity = {
        "source": {
            "trees": {
                "harness": {"commit": "0" * 40, "clean": False},
                "gpyreg": {"commit": "1" * 40, "clean": True},
            }
        },
        "imports": {"trees": {"gpyreg": {"path": str(tmp_path)}}},
    }
    monkeypatch.setattr(
        gates, "this_identity", lambda host=True: copy.deepcopy(identity)
    )
    fresh = tmp_path / "fresh"
    assert gates.main(["run", "--out", str(fresh)]) == gates.EXIT_IDENTITY
    assert "['harness'] are not clean" in capsys.readouterr().out
    assert not fresh.exists()


def test_a_process_without_a_gpyreg_checkout_has_no_identity(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("PYVBMC_GPYREG_SOURCE", str(tmp_path / "nowhere"))
    code = gates.main(["run", "--out", str(tmp_path / "out")])
    assert code == gates.EXIT_IDENTITY
    assert "no identity" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()
