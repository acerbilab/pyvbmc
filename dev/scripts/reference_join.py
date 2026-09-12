"""Join a finished `population_run.py` campaign to the golden reference.

The extension pattern of ``dev/golden/noisy_extension_20260907/README.md``
as one command::

    python dev/scripts/reference_join.py join \
        --campaign dev/scripts/runs/population_realdata_20260912 \
        --previous dev/scripts/runs/golden/reference_870_20260907 \
        --previous-manifest dev/golden/noisy_extension_20260907/sha256_manifest.json \
        --combined dev/scripts/runs/golden/reference_990_20260912 \
        --record dev/golden/realdata_extension_20260912
    python dev/scripts/reference_join.py record-replay \
        --record dev/golden/realdata_extension_20260912 \
        --combined dev/scripts/runs/golden/reference_990_20260912 \
        --replay-dir <dir>

``join`` revalidates every case of the campaign against its completion
record (the launcher's own check, repeated), checks the previous reference
against its tracked SHA256 manifest, refuses any overlap between the two,
copies both populations byte for byte into the new combined directory
(checking the ZIP integrity of every copied trace), writes the combined
population's summary, writes the new SHA256 manifest and checks that it
preserves every entry of the previous one, then copies the new sidecars and
the summary into the tracked ``dev/golden/baseline`` and writes the even/odd
null check and a validation record under ``--record``, and prints the
per-configuration table for the README. The validation record pins the
frozen launcher, the frozen helper modules and every artifact of every new
case (the boost captures stay gitignored with the campaign). It exits nonzero when the null check flags a configuration; every
file is written first. ``record-replay`` copies a finished
``golden_replay.py`` report (run afterwards with the combined directory as
its baseline and sidecars) into the record and adds its counts to the
validation record.

Run from the repository root with the project venv, as one process, after
the campaign's ``finished.json`` exists. Nothing is regenerated: the
traces are the campaign's, and a mismatch stops the join for inspection.
"""

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import population_run as runner  # isort: skip  (single-thread env first)

# isort: split
import numpy as np

golden_trace = runner.golden_trace
NORMALIZATION = (
    "JSON and Markdown: line endings normalized to LF; NPZ: raw bytes."
)
USABLE = {"elbo_err": 1.0, "gskl": 1.0, "mmtv": 0.2}


def sha256(path, normalize_text=False):
    data = Path(path).read_bytes()
    if normalize_text:
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def tags_of(directory, suffix):
    return {
        p.name[: -len(suffix)]
        for p in Path(directory).glob(f"*_seed*{suffix}")
    }


def verify_previous(previous, manifest_path):
    """Every pair of the previous reference is as its tracked manifest says."""
    manifest = read_json(manifest_path)
    files = manifest["files"]
    if tags_of(previous, ".json") != set(files) or tags_of(
        previous, ".npz"
    ) != set(files):
        raise RuntimeError(
            "previous reference allocation differs from its manifest"
        )
    if list(Path(previous).glob("*.error.txt")):
        raise RuntimeError("previous reference has error files")
    for tag, hashes in files.items():
        if sha256(previous / f"{tag}.json", True) != hashes["json_sha256"]:
            raise RuntimeError(f"previous JSON hash mismatch: {tag}")
        if sha256(previous / f"{tag}.npz") != hashes["npz_sha256"]:
            raise RuntimeError(f"previous NPZ hash mismatch: {tag}")
    if sha256(previous / "summary.md", True) != manifest["summary_sha256"]:
        raise RuntimeError("previous summary differs from its manifest")
    return manifest


def verify_campaign(campaign):
    """Repeat the launcher's completion check on every case of the campaign."""
    campaign = Path(campaign)
    results = campaign / "results"
    manifest = read_json(campaign / "launch_manifest.json")
    launch = read_json(results / "launch.json")
    if launch["manifest"] != manifest:
        raise RuntimeError(
            "launch.json manifest differs from launch_manifest.json"
        )
    finished = read_json(results / "finished.json")
    status = read_json(results / "status.json")
    expected = {
        f"{c['label']}_seed{s}"
        for c in manifest["allocation"]
        for s in c["seeds"]
    }
    if len(expected) != manifest["candidate_run_count"]:
        raise RuntimeError("allocation size differs from candidate_run_count")
    if set(finished["completed"]) != expected or status["failed"]:
        raise RuntimeError("campaign is not complete without failures")
    if finished["flagged_configurations"]:
        raise RuntimeError(
            "the launcher's closing comparison flagged configurations, so "
            "the campaign overlaps the reference or is not the manifest's"
        )
    if list(results.glob("*.error.txt")):
        raise RuntimeError("campaign has error files")
    identity = launch["identity"]
    frozen_root = Path(identity["pyvbmc_import"]).resolve().parents[1]
    frozen_launcher = frozen_root / "dev" / "scripts" / "population_run.py"
    if sha256(frozen_launcher) != sha256(Path(runner.__file__)):
        raise RuntimeError(
            "the validating population_run.py differs from the campaign's "
            f"frozen launcher {frozen_launcher}"
        )
    records = {}
    for tag in sorted(expected):
        records[tag] = runner.validate_case(results, tag, identity)
        side = read_json(results / f"{tag}.json")
        meta = side["meta"]
        if meta["git"]["dirty"] is not False:
            raise RuntimeError(f"{tag}: dirty source provenance")
        if not identity["candidate_sha"].startswith(meta["git"]["sha"]):
            raise RuntimeError(f"{tag}: source provenance mismatch")
        if meta["threads"] != identity["threads"]:
            raise RuntimeError(f"{tag}: thread provenance mismatch")
        for name, version in identity["dependencies"].items():
            if name in meta and meta[name] != version:
                raise RuntimeError(f"{tag}: {name} provenance mismatch")
        if meta["python"] != identity["python"]:
            raise RuntimeError(f"{tag}: Python provenance mismatch")
        budget = side["effective_options"].get("max_fun_evals")
        if not 0 < side["final"]["func_count"] <= budget:
            raise RuntimeError(f"{tag}: evaluation count outside the budget")
    provenance = {
        "frozen_root": str(frozen_root),
        "launcher_sha256": sha256(frozen_launcher),
        "supporting_source_sha256": {
            name: sha256(path)
            for name, path in sorted(identity["helper_imports"].items())
        },
        "new_case_artifacts_sha256": {
            tag: records[tag]["hashes"] for tag in sorted(records)
        },
    }
    return manifest, identity, records, provenance


def copy_pair(source, target, tag, expected_raw=None):
    for suffix in (".json", ".npz"):
        src, dst = source / f"{tag}{suffix}", target / f"{tag}{suffix}"
        if dst.exists():
            raise RuntimeError(f"refusing to overwrite {dst}")
        shutil.copyfile(src, dst)
        digest = sha256(dst)
        if digest != sha256(src):
            raise RuntimeError(f"copy differs from source: {dst}")
        if expected_raw is not None and digest != expected_raw[suffix]:
            raise RuntimeError(f"copy differs from completion record: {dst}")
    with zipfile.ZipFile(target / f"{tag}.npz") as archive:
        if archive.testzip() is not None:
            raise RuntimeError(f"NPZ integrity failed: {tag}")


def split_population(population):
    even, odd = {}, {}
    for label, entry in population.items():
        seeds = np.array(entry["seeds"])
        for target, mask in ((even, seeds % 2 == 0), (odd, seeds % 2 == 1)):
            sub = {"seeds": list(seeds[mask]), "rows": [], "fails": 0}
            for key, value in entry.items():
                if isinstance(value, np.ndarray):
                    sub[key] = value[mask]
            target[label] = sub
    return even, odd


def config_table(results, allocation):
    """Per-configuration outcomes of the new runs, as the README reports them."""
    table = {}
    for config in allocation:
        rows = [
            read_json(results / f"{config['label']}_seed{s}.json")
            for s in config["seeds"]
        ]
        finals = [r["final"] for r in rows]
        first = rows[0]
        table[config["label"]] = {
            "problem": first["problem"],
            "D": first["D"],
            "noise_sd": first["noise_sd"],
            "max_fun_evals": first["effective_options"]["max_fun_evals"],
            "seeds": [config["seeds"][0], config["seeds"][-1]],
            "runs": len(rows),
            "converged": sum(bool(f["success_flag"]) for f in finals),
            "reached_budget": sum(
                f["func_count"] >= first["effective_options"]["max_fun_evals"]
                for f in finals
            ),
            "usable": sum(
                all(f[k] < v for k, v in USABLE.items()) for f in finals
            ),
            "median_func_count": float(
                np.median([f["func_count"] for f in finals])
            ),
            "mean_minutes": float(np.mean([f["wall_s"] for f in finals]) / 60),
            "first_started": min(r["meta"]["started"] for r in rows),
            "last_finished": max(r["meta"]["finished"] for r in rows),
        }
    return table


def cmd_join(args):
    campaign = Path(args.campaign).resolve()
    results = campaign / "results"
    previous = Path(args.previous).resolve()
    combined = Path(args.combined).resolve()
    record = Path(args.record).resolve()
    sidecars = Path(args.sidecars).resolve()
    if combined.exists():
        raise RuntimeError(f"combined directory exists: {combined}")
    for name in ("sha256_manifest.json", "validation.json", "even_vs_odd.md"):
        if (record / name).exists():
            raise RuntimeError(f"record exists: {record / name}")
    steps = []

    manifest, identity, records, provenance = verify_campaign(campaign)
    steps.append(f"validated_{len(records)}_campaign_cases")
    previous_manifest = verify_previous(previous, args.previous_manifest)
    steps.append(
        f"previous_{len(previous_manifest['files'])}_pairs_hash_check"
    )
    new_tags = set(records)
    if new_tags & set(previous_manifest["files"]):
        raise RuntimeError("campaign overlaps the previous reference")
    tracked_before = tags_of(sidecars, ".json")
    if tracked_before != set(previous_manifest["files"]):
        raise RuntimeError(
            "tracked sidecars differ from the previous manifest"
        )

    combined.mkdir(parents=True)
    for tag in previous_manifest["files"]:
        copy_pair(previous, combined, tag)
    for tag in sorted(new_tags):
        copy_pair(results, combined, tag, records[tag]["hashes"])
    steps.append("combined_copy_and_zip_integrity")

    golden_trace.cmd_summary(argparse.Namespace(dir=str(combined)))
    steps.append("combined_summary")

    files = {
        tag: {
            "json_sha256": sha256(combined / f"{tag}.json", True),
            "npz_sha256": sha256(combined / f"{tag}.npz"),
        }
        for tag in sorted(set(previous_manifest["files"]) | new_tags)
    }
    for tag, hashes in previous_manifest["files"].items():
        if files[tag] != hashes:
            raise RuntimeError(f"previous manifest entry not preserved: {tag}")
    record.mkdir(parents=True, exist_ok=True)
    write_json(
        record / "sha256_manifest.json",
        {
            "population": combined.name,
            "files": files,
            "summary_sha256": sha256(combined / "summary.md", True),
            "text_hash_normalization": NORMALIZATION,
        },
    )
    steps.append("sha256_manifest")

    for tag in sorted(new_tags):
        shutil.copyfile(combined / f"{tag}.json", sidecars / f"{tag}.json")
    shutil.copyfile(combined / "summary.md", sidecars / "summary.md")
    for tag in sorted(set(previous_manifest["files"]) | new_tags):
        if sha256(sidecars / f"{tag}.json", True) != sha256(
            combined / f"{tag}.json", True
        ):
            raise RuntimeError(f"tracked sidecar differs from combined: {tag}")
    steps.append("published_sidecars_and_summary")

    population = golden_trace.load_population(combined)
    even, odd = split_population(population)
    text, flagged = golden_trace.compare_populations(even, odd)
    n_tests = sum(
        1
        for line in text.splitlines()
        if re.match(r"^\| \S+ \| (elbo_err|gskl|mmtv|func_count) \|", line)
    )
    (record / "even_vs_odd.md").write_text(
        f"# Null check (even vs odd seeds) on {combined.name}\n\n{text}\n",
        encoding="utf-8",
    )
    steps.append("even_vs_odd_null_check")

    statuses = {}
    git_provenance = {}
    for tag in files:
        side = read_json(combined / f"{tag}.json")
        key = (bool(side["final"]["success_flag"]), side["final"]["message"])
        statuses[key] = statuses.get(key, 0) + 1
        git = (side["meta"]["git"]["sha"], bool(side["meta"]["git"]["dirty"]))
        git_provenance[git] = git_provenance.get(git, 0) + 1
    configs = {
        label: len(entry["seeds"]) for label, entry in population.items()
    }
    table = config_table(results, manifest["allocation"])
    validation = {
        "population": combined.name,
        "combined_path": str(combined),
        "pairs": len(files),
        "configs": dict(sorted(configs.items())),
        "previous_population": previous.name,
        "previous_manifest": Path(args.previous_manifest).as_posix(),
        "previous_manifest_sha256": sha256(args.previous_manifest, True),
        "previous_pairs_unchanged": len(previous_manifest["files"]),
        "all_combined_files_byte_identical_to_sources": True,
        "npz_zip_integrity_passed": len(files),
        "npz_bytes": sum(
            (combined / f"{tag}.npz").stat().st_size for tag in files
        ),
        "statuses": [
            {"success": s, "message": m, "count": c}
            for (s, m), c in sorted(statuses.items(), key=lambda i: -i[1])
        ],
        "provenance": [
            {"sha": s, "dirty": d, "count": c}
            for (s, d), c in sorted(
                git_provenance.items(), key=lambda i: -i[1]
            )
        ],
        "noisy_pairs": int(
            sum(
                n
                for label, n in configs.items()
                if read_json(
                    combined
                    / f"{label}_seed{population[label]['seeds'][0]}.json"
                )["noise_sd"]
            )
        ),
        "new_pairs": len(new_tags),
        "new_configs": table,
        "new_campaign": str(campaign),
        "new_campaign_stage": manifest.get("stage"),
        "new_source_sha": identity["candidate_sha"],
        "new_numerical_base_sha": identity["numerical_base_sha"],
        "gpyreg_sha": identity["gpyreg_sha"],
        "frozen_root": provenance["frozen_root"],
        "launcher_sha256": provenance["launcher_sha256"],
        "joiner_sha256": sha256(Path(__file__)),
        "supporting_source_sha256": provenance["supporting_source_sha256"],
        "new_case_artifacts_sha256": provenance["new_case_artifacts_sha256"],
        "runtime_contract": {
            k: identity[k]
            for k in (
                "pyvbmc_import",
                "gpyreg_import",
                "helper_imports",
                "python",
                "dependencies",
                "threads",
                "options",
            )
        },
        "new_results": {
            "completed_pairs": len(new_tags),
            "first_started": min(t["first_started"] for t in table.values()),
            "last_finished": max(t["last_finished"] for t in table.values()),
            "summed_optimizer_hours": float(
                sum(
                    read_json(results / f"{tag}.json")["final"]["wall_s"]
                    for tag in new_tags
                )
                / 3600
            ),
            "campaign_span_hours": (
                time.mktime(
                    time.strptime(
                        max(t["last_finished"] for t in table.values()),
                        "%Y-%m-%d %H:%M:%S",
                    )
                )
                - time.mktime(
                    time.strptime(
                        min(t["first_started"] for t in table.values()),
                        "%Y-%m-%d %H:%M:%S",
                    )
                )
            )
            / 3600,
            "supervisor_last_pass_hours": read_json(results / "finished.json")[
                "elapsed_seconds"
            ]
            / 3600,
        },
        "published_sidecars": len(tags_of(sidecars, ".json")),
        "null_check": {
            "ks_tests": n_tests,
            "holm_alpha": 0.05,
            "flagged_configs": len(flagged),
            "flagged": sorted(flagged),
            "report": "even_vs_odd.md",
        },
        "completed_steps": steps,
        "joined": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    write_json(record / "validation.json", validation)

    print(
        f"\njoined {len(new_tags)} pairs to {len(previous_manifest['files'])}: {combined.name}"
    )
    print(
        "\n| New configuration | Runs | Converged | Reached budget | Usable | Median evals | Mean optimizer time |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    for label, t in table.items():
        print(
            f"| {label} | {t['runs']} | {t['converged']} | {t['reached_budget']} |"
            f" {t['usable']}/{t['runs']} | {t['median_func_count']:.0f} |"
            f" {t['mean_minutes']:.3f} min |"
        )
    print(f"\nnull check: {n_tests} KS tests, flagged {sorted(flagged)}")
    return 1 if flagged else 0


def cmd_record_replay(args):
    record = Path(args.record).resolve()
    replay = Path(args.replay_dir).resolve()
    validation_path = record / "validation.json"
    validation = read_json(validation_path)
    if "final_replay" in validation:
        raise RuntimeError("validation record already holds a replay")
    report = read_json(replay / "replay.json")
    rows = report["rows"]
    if report["git"]["dirty"]:
        raise RuntimeError("the replay ran on a dirty checkout")
    combined = Path(args.combined).resolve()
    header = (replay / "replay.md").read_text(encoding="utf-8").splitlines()[2]
    if f"baseline `{combined.name}`" not in header:
        raise RuntimeError(
            f"the replay's baseline is not {combined.name}: {header}"
        )
    if validation["population"] != combined.name:
        raise RuntimeError("the record is not for this combined population")
    summary = {
        "cases": len(rows),
        "stored_loop_and_final_exact": sum(
            r["verdict"].startswith("identical stored loop and final")
            for r in rows
        ),
        "initial_designs_exact": sum(
            "initial design identical" in r["verdict"] for r in rows
        ),
        "returned_transformers_certifiable": sum(
            "returned transformer not certifiable" not in r["verdict"]
            for r in rows
        ),
        "flagged": sum(bool(r.get("flagged")) for r in rows),
        "minutes": report["minutes"],
        "git": report["git"],
        "threads": report["threads"],
        "baseline": combined.name,
        "cases_run": [f"{r['label']}_seed{r['seed']}" for r in rows],
        "report": "final_replay.json",
        "table": "final_replay.md",
    }
    for name in ("replay.md", "replay.json"):
        shutil.copyfile(replay / name, record / f"final_{name}")
    validation["final_replay"] = summary
    validation["completed_steps"].append("final_replay_of_new_configurations")
    write_json(validation_path, validation)
    print(json.dumps(summary, indent=2))
    return 1 if summary["flagged"] else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    join = sub.add_parser(
        "join", help="join a finished campaign to the reference"
    )
    join.add_argument("--campaign", required=True)
    join.add_argument("--previous", required=True)
    join.add_argument("--previous-manifest", required=True)
    join.add_argument("--combined", required=True)
    join.add_argument("--record", required=True)
    join.add_argument(
        "--sidecars", default=str(HERE.parents[1] / "dev/golden/baseline")
    )
    join.set_defaults(func=cmd_join)
    replay = sub.add_parser(
        "record-replay", help="add a finished replay to the record"
    )
    replay.add_argument("--record", required=True)
    replay.add_argument("--combined", required=True)
    replay.add_argument("--replay-dir", required=True)
    replay.set_defaults(func=cmd_record_replay)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
