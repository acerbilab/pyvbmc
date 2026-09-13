"""Prepare or publish the approved 2026-09-13 golden reference.

Run with the repository venv on the generating machine. ``prepare`` checks
the frozen campaign records, preserves the previous reference, and assembles
the new population without changing the active sidecars. Run golden_replay.py
against the prepared population before ``publish``; publication requires five
identical default replays and a clean even/odd check.
"""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

if sys.flags.optimize:
    raise RuntimeError("Run without -O: integrity assertions must be enabled")

ROOT = Path(__file__).resolve().parents[3]
RECORD = Path(__file__).resolve().parent
RUNS = ROOT / "dev/scripts/runs"
FIRST = RUNS / "population_overnight_20260910"
PREVIOUS = RUNS / "golden/reference_990_20260912"
COMBINED = RUNS / "golden/reference_990_20260913"
SIDECARS = ROOT / "dev/golden/baseline"
CAMPAIGNS = (
    "population_overnight_20260910",
    "population_extension_20260911",
    "population_completion_20260912",
)

os.environ["PYVBMC_GPYREG_SOURCE"] = (FIRST / "gpyreg").as_posix()
sys.path.insert(0, str(FIRST / "source/dev/scripts"))
import population_run as runner  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "join_helpers", ROOT / "dev/scripts/reference_join.py"
)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def prepare():
    previous_manifest = (
        ROOT / "dev/golden/realdata_extension_20260912/sha256_manifest.json"
    )
    previous = helpers.verify_previous(PREVIOUS, previous_manifest)
    original = read(
        ROOT / "dev/golden/noisy_extension_20260907/sha256_manifest.json"
    )
    if helpers.tags_of(SIDECARS, ".json") != set(previous["files"]):
        raise RuntimeError("Active allocation differs from previous reference")
    for tag, hashes in previous["files"].items():
        assert (
            helpers.sha256(SIDECARS / f"{tag}.json", True)
            == hashes["json_sha256"]
        )
    assert (
        helpers.sha256(SIDECARS / "summary.md", True)
        == previous["summary_sha256"]
    )

    assessment_path = RUNS / CAMPAIGNS[-1] / "assessment/assessment.json"
    assessment = read(assessment_path)
    assert (
        helpers.sha256(assessment_path)
        == "dd4b97cbe86fdd6bb5350ca8005d92237fdf451cad498e448073e3b82a9799fe"
    )
    sources, artifacts = {}, {}
    for campaign in CAMPAIGNS:
        folder = RUNS / campaign / "results"
        launch = read(folder / "launch.json")
        expected = {
            f"{config['label']}_seed{seed}"
            for config in launch["manifest"]["allocation"]
            for seed in config["seeds"]
        }
        assert not sources.keys() & expected
        assert expected == set(read(folder / "finished.json")["completed"])
        assert expected == helpers.tags_of(folder, ".json")
        assert not read(folder / "status.json")["failed"]
        assert not list(folder.glob("*.error.txt"))
        assert launch["identity"] == assessment["candidate_identity"]
        for tag in sorted(expected):
            completion = runner.validate_case(folder, tag, launch["identity"])
            artifacts[tag] = completion["hashes"]
            sources[tag] = folder
        print(f"Validated {campaign}: {len(expected)} cases", flush=True)
    assert set(sources) == set(original["files"]) and len(sources) == 870
    retained = set(previous["files"]) - set(sources)
    assert len(retained) == 120

    historical_readme = RECORD / "previous_reference_README.md"
    if not historical_readme.exists():
        shutil.copyfile(ROOT / "dev/golden/README.md", historical_readme)
    if not (PREVIOUS / "README.md").exists():
        shutil.copyfile(historical_readme, PREVIOUS / "README.md")
    COMBINED.mkdir(exist_ok=True)
    assert helpers.tags_of(COMBINED, ".json") <= set(previous["files"])
    assert helpers.tags_of(COMBINED, ".npz") <= set(previous["files"])
    for tag in sorted(previous["files"]):
        source = sources.get(tag, PREVIOUS)
        if (COMBINED / f"{tag}.json").exists() or (
            COMBINED / f"{tag}.npz"
        ).exists():
            for suffix in (".json", ".npz"):
                digest = helpers.sha256(COMBINED / f"{tag}{suffix}")
                assert digest == helpers.sha256(source / f"{tag}{suffix}")
                if tag in artifacts:
                    assert digest == artifacts[tag][suffix]
            with zipfile.ZipFile(COMBINED / f"{tag}.npz") as archive:
                assert archive.testzip() is None
        else:
            helpers.copy_pair(source, COMBINED, tag, artifacts.get(tag))
    runner.golden_trace.cmd_summary(argparse.Namespace(dir=str(COMBINED)))
    files = {
        tag: {
            "json_sha256": helpers.sha256(COMBINED / f"{tag}.json", True),
            "npz_sha256": helpers.sha256(COMBINED / f"{tag}.npz"),
        }
        for tag in sorted(previous["files"])
    }
    assert all(files[tag] == previous["files"][tag] for tag in retained)
    helpers.verify_previous(PREVIOUS, previous_manifest)
    manifest = {
        "population": COMBINED.name,
        "files": files,
        "summary_sha256": helpers.sha256(COMBINED / "summary.md", True),
        "text_hash_normalization": helpers.NORMALIZATION,
    }
    write(RECORD / "sha256_manifest.json", manifest)
    population = runner.golden_trace.load_population(COMBINED)
    even, odd = helpers.split_population(population)
    report, flagged = runner.golden_trace.compare_populations(even, odd)
    (RECORD / "even_vs_odd.md").write_text(
        f"# Even/odd check: {COMBINED.name}\n\n{report}\n", encoding="utf-8"
    )
    validation = {
        "population": COMBINED.name,
        "previous_population": PREVIOUS.name,
        "previous_manifest_sha256": helpers.sha256(previous_manifest, True),
        "previous_pairs_preserved": 990,
        "candidate_pairs": 870,
        "realdata_pairs_unchanged": 120,
        "pairs": len(files),
        "configs": {
            key: len(value["seeds"])
            for key, value in sorted(population.items())
        },
        "npz_zip_integrity_passed": len(files),
        "all_pairs_byte_identical_to_sources": True,
        "candidate_identity": assessment["candidate_identity"],
        "candidate_artifacts_sha256": artifacts,
        "assessment_sha256": helpers.sha256(assessment_path),
        "even_odd_tests": 4 * len(population),
        "even_odd_flagged": sorted(flagged),
        "promotion_script_sha256": helpers.sha256(Path(__file__)),
        "status": "prepared",
    }
    write(RECORD / "validation.json", validation)
    print(
        f"Prepared {len(files)} pairs; even/odd flags: {flagged}", flush=True
    )
    if flagged:
        raise RuntimeError("Even/odd flags require review")


def publish():
    validation = read(RECORD / "validation.json")
    assert not validation["even_odd_flagged"]
    replay = read(RECORD / "final_replay.json")
    assert replay["calibration_budget"] is None
    assert replay["git"]["dirty"] is False
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    assert head.startswith(replay["git"]["sha"])
    subprocess.run(
        [
            "git",
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            "pyvbmc",
            "dev/scripts/golden_trace.py",
            "dev/scripts/benchmark_targets.py",
            "dev/scripts/profile_run.py",
        ],
        cwd=ROOT,
        check=True,
    )
    assert f"baseline `{COMBINED.name}`" in (
        RECORD / "final_replay.md"
    ).read_text(encoding="utf-8")
    rows = replay["rows"]
    expected = {
        (label, 0)
        for label in (
            "normal_D5",
            "banana_D2",
            "halfnormal_D2",
            "cigar_D4",
            "rosenbrock_D2_noise1",
        )
    }
    assert (
        len(rows) == 5 and {(r["label"], r["seed"]) for r in rows} == expected
    )
    assert replay["threads"] == 1
    assert all(
        row["ok"]
        and row["identical"]
        and row["initial_design_ok"] is True
        and row["design"] == "identical (X_init in both traces)"
        and not row["flagged"]
        for row in rows
    )
    manifest = helpers.verify_previous(
        COMBINED, RECORD / "sha256_manifest.json"
    )
    previous = helpers.verify_previous(
        PREVIOUS,
        ROOT / "dev/golden/realdata_extension_20260912/sha256_manifest.json",
    )
    for tag, hashes in previous["files"].items():
        assert (
            helpers.sha256(SIDECARS / f"{tag}.json", True)
            == hashes["json_sha256"]
        )
    assert (
        helpers.sha256(SIDECARS / "summary.md", True)
        == previous["summary_sha256"]
    )
    assert helpers.tags_of(SIDECARS, ".json") == set(manifest["files"])
    for tag, hashes in manifest["files"].items():
        shutil.copyfile(COMBINED / f"{tag}.json", SIDECARS / f"{tag}.json")
        assert (
            helpers.sha256(SIDECARS / f"{tag}.json", True)
            == hashes["json_sha256"]
        )
    shutil.copyfile(COMBINED / "summary.md", SIDECARS / "summary.md")
    assert (
        helpers.sha256(SIDECARS / "summary.md", True)
        == manifest["summary_sha256"]
    )
    validation["status"] = "promoted"
    validation["publication_script_sha256"] = helpers.sha256(Path(__file__))
    validation["exact_replay"] = {
        "cases": 5,
        "identical": 5,
        "initial_design_identical": 5,
        "flags": 0,
        "report_sha256": helpers.sha256(RECORD / "final_replay.json"),
        "markdown_sha256": helpers.sha256(RECORD / "final_replay.md"),
        "code_head": head,
        "baseline": COMBINED.name,
        "sidecars": COMBINED.name,
    }
    write(RECORD / "validation.json", validation)
    print("Published 990 sidecars and summary", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "publish"))
    args = parser.parse_args()
    {"prepare": prepare, "publish": publish}[args.action]()
