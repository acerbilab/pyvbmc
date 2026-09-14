"""Generate a pool of independent VBMC runs for the S-VBMC campaign.

The pools are the input of the ELBO debiasing estimator and of the
stacking comparison against the original S-VBMC; the campaign, its
conditions, seeds, filters and stopping rule are
``dev/plans/svbmc-benchmark-campaign.md``. Each run is one fresh process
on one configuration and seed of the ``svbmc_pool`` suite of
``benchmark_targets.py``, saved as the per-run artifact of that plan
(``svbmc_pool_io.save_run``: the returned posterior with all of its
statistics, the GP behind them, the transformer, every evaluation, the
run's state and its metadata) and recorded in a hash-verified completion
record that permits resumption.

Sub-commands::

    python dev/scripts/svbmc_pool_run.py prepare --out DIR \\
        --suite svbmc_pool --target 60 --max-seeds 90 \\
        --allocation ring_D2_noise3_svbmc=40/80 \\
        --allocation gmm_D2_svbmc=30/45
    python dev/scripts/svbmc_pool_run.py prepare --out DIR ... \\
        --ready --authorized-by NAME
    python -u dev/scripts/svbmc_pool_run.py run --out DIR --pilot-seeds 3
    python -u dev/scripts/svbmc_pool_run.py run --out DIR
    python dev/scripts/svbmc_pool_run.py worker --out DIR --label L --seed S
    python dev/scripts/svbmc_pool_run.py summarize --out DIR

``prepare`` fixes the allocation, the run options and the identity of the
code and environment the pool is generated with, and writes them unready;
``prepare --ready --authorized-by NAME`` records who authorized the
launch and when. Of the ``svbmc_pool`` suite it allocates the campaign's
five pool conditions (``POOL_LABELS``); the suite's sixth entry is the
extension condition and is allocated only when ``--only`` names it.
Run on a directory that already holds a manifest, ``prepare`` marks the
campaign ready and revises its allocation; nothing else about it can
change. Filtered targets and seed caps may be revised and a condition
added, a condition that already has runs may neither be dropped nor
given a different first seed, and the previous allocation, the time and
``--authorized-by NAME`` are appended to ``allocation_history``, which is
how the pilot's revision of the seed caps is recorded.

``run`` refuses an unready manifest or an identity that
differs from the prepared one, takes the campaign lock, and walks the
conditions in manifest order, seeds upward, stopping each condition at
its filtered target or its seed cap. A failing case writes
``<tag>.error.txt``, counts towards the seed cap and the sweep continues;
an artifact file of a case with neither a completion record nor an error
file stops the sweep for inspection. The ``<tag>.log`` an interrupted
sweep leaves behind is not such a file: it is truncated when the case
runs again.

gpyreg is pinned to the frozen worktree the manifest names: every
process prepends it to ``sys.path`` before PyVBMC is imported (through
``PYVBMC_GPYREG_SOURCE``, which the launcher also passes to its
children) and refuses to run when ``gpyreg`` resolves elsewhere. PyVBMC
itself is this checkout: the identity records its commit, whether the
package or the suite module carries uncommitted changes, and the hashes
of the suite module and of the two pool scripts, and ``run``
refuses to generate a pool from a dirty tree unless the manifest was
prepared with ``--allow-dirty``, which records the relaxation.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
for _key in THREAD_KEYS:
    os.environ[_key] = "1"
os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
# Honour a gpyreg pin already in the environment (a child of `run`, or an
# operator following `population_run.py`); `activate_gpyreg` installs the
# manifest's pin for a process started without it. Either way this
# happens before the first PyVBMC import, which is why nothing under
# `pyvbmc` or `svbmc_pool_io` is imported at module level here.
if os.environ.get("PYVBMC_GPYREG_SOURCE"):
    sys.path.insert(1, os.environ["PYVBMC_GPYREG_SOURCE"])

import numpy as np  # noqa: E402
from filelock import FileLock  # noqa: E402

SUITE_MODULE = ROOT / "dev" / "scripts" / "benchmark_targets.py"
IO_MODULE = HERE / "svbmc_pool_io.py"
RUNNER_MODULE = Path(__file__).resolve()
DEFAULT_GPYREG = (
    ROOT / "dev" / "scripts" / "runs" / "svbmc_pool_20260913" / "gpyreg"
)
#: The conditions the campaign allocates out of the ``svbmc_pool`` suite.
#: The suite also holds the extension condition
#: ``multisensory_s1_D6_noise1.3_svbmc``, which enters a pool only when
#: ``prepare --only`` names it.
POOL_LABELS = (
    "multisensory_s1_D6_noise3_svbmc",
    "rosenbrock_D2_noise3_svbmc",
    "gmm_D2_noise3_svbmc",
    "ring_D2_noise3_svbmc",
    "gmm_D2_svbmc",
)
BASE_OPTIONS = {
    "display": "off",
    "plot": False,
    "print_iteration_header": False,
    "performance_calibration": "off",
}
# The pool's usability thresholds, the house convention of the benchmark
# suite: evidence error, Gaussianized symmetrized KL, marginal TV.
THRESHOLDS = {"elbo_err": 1.0, "gskl": 1.0, "mmtv": 0.2}
#: The manifest fields a campaign directory is bound to for its whole
#: life. The allocation is not one of them: the pilot revises targets and
#: seed caps in place (``cmd_prepare``), which leaves every run already
#: generated valid.
STRUCTURAL_KEYS = (
    "campaign",
    "suite",
    "options",
    "gpyreg_source",
    "identity",
    "allow_dirty",
)
#: The suffixes of one case's artifact files. A ``<tag>.log`` is not one:
#: every case that starts writes one, and it is truncated on a rerun.
ARTIFACT_SUFFIXES = (".npz", ".json", ".vbmc.pkl")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def git(path, *args):
    return subprocess.check_output(
        ["git", "-C", str(path), *args], text=True
    ).strip()


def activate_gpyreg(source):
    """Pin gpyreg to ``source`` for this process and its children.

    Sets ``PYVBMC_GPYREG_SOURCE`` and prepends the directory to
    ``sys.path``, which wins over an editable install. Raises when the
    directory holds no ``gpyreg`` package, or when gpyreg is already
    imported from somewhere else.
    """
    source = Path(source).resolve()
    if not (source / "gpyreg").is_dir():
        raise RuntimeError(f"no gpyreg package under {source}")
    os.environ["PYVBMC_GPYREG_SOURCE"] = str(source)
    if str(source) not in sys.path:
        sys.path.insert(1, str(source))
    module = sys.modules.get("gpyreg")
    if module is not None:
        actual = Path(module.__file__).resolve().parent
        if actual != source / "gpyreg":
            raise RuntimeError(
                f"gpyreg was already imported from {actual}, not the "
                f"campaign's frozen source {source / 'gpyreg'}"
            )
    return str(source)


def identity(gpyreg_source):
    """The code and environment every process of the campaign must share.

    The environment of ``svbmc_pool_io`` (commits, import paths,
    versions, threads, host) plus the state of the working tree: whether
    the package or the suite module carries uncommitted changes, whether
    the frozen gpyreg worktree is clean, and the hashes of the three
    modules a run is generated by (the suite, this harness and its
    artifact layer), which the package commit does not cover while they
    are uncommitted.
    """
    import svbmc_pool_io as pool_io

    source = Path(gpyreg_source).resolve()
    record = pool_io.environment(source)
    record["gpyreg_source"] = str(source)
    record["gpyreg_clean"] = not git(source, "status", "--porcelain")
    record["pyvbmc_dirty"] = git(ROOT, "status", "--porcelain", "--", "pyvbmc")
    record["suite_module_dirty"] = git(
        ROOT, "status", "--porcelain", "--", str(SUITE_MODULE)
    )
    record["suite_module_sha256"] = pool_io.sha256(SUITE_MODULE)
    record["io_module_sha256"] = pool_io.sha256(IO_MODULE)
    record["runner_sha256"] = pool_io.sha256(RUNNER_MODULE)
    return record


def identity_differences(actual, expected):
    keys = set(actual) | set(expected)
    return sorted(k for k in keys if actual.get(k) != expected.get(k))


def check_tree(record, allow_dirty):
    """Refuse a dirty numerical source unless the manifest allows it."""
    if not record["gpyreg_clean"]:
        raise RuntimeError(
            "the frozen gpyreg worktree has uncommitted changes; the pin "
            "is what makes the pool reproducible"
        )
    dirty = [
        name
        for name, value in (
            ("pyvbmc/", record["pyvbmc_dirty"]),
            ("dev/scripts/benchmark_targets.py", record["suite_module_dirty"]),
        )
        if value
    ]
    if dirty and not allow_dirty:
        raise RuntimeError(
            "uncommitted changes in "
            + ", ".join(dirty)
            + "; commit them, or prepare the manifest with --allow-dirty "
            "to record that the pool was generated from a dirty tree"
        )
    return dirty


# --------------------------------------------------------------------------
# prepare
# --------------------------------------------------------------------------


def parse_override(item):
    """One ``--allocation LABEL=TARGET/MAXSEEDS`` argument."""
    label, assigned, numbers = item.partition("=")
    target, divided, cap = numbers.partition("/")
    if not (label and assigned and divided) or not (
        target.isdigit() and cap.isdigit()
    ):
        raise RuntimeError(
            f"--allocation {item!r} is not LABEL=TARGET/MAXSEEDS with two "
            "non-negative integers"
        )
    return label, (int(target), int(cap))


def allocation(args, configs):
    """One entry per condition: label, first seed, seed cap, filter target."""
    overrides = dict(parse_override(item) for item in args.allocation or [])
    entries = []
    for cfg in configs:
        if cfg.noise_sd is None:
            target = args.control_target or args.target
            cap = args.control_max_seeds or args.max_seeds
        else:
            target, cap = args.target, args.max_seeds
        target, cap = overrides.get(cfg.label, (target, cap))
        if target > cap:
            raise RuntimeError(
                f"{cfg.label}: filtered target {target} exceeds the seed "
                f"cap {cap}"
            )
        entries.append(
            {
                "label": cfg.label,
                "seed_start": int(args.seed_start),
                "max_seeds": int(cap),
                "target_filtered": int(target),
            }
        )
    unknown = set(overrides) - {e["label"] for e in entries}
    if unknown:
        raise RuntimeError(f"--allocation names unknown labels: {unknown}")
    return entries


def select_configs(args, configs):
    """The conditions to allocate, and the extension conditions among them.

    Without ``--only`` the ``svbmc_pool`` suite contributes exactly the
    campaign's five pool conditions and any other suite all of its
    entries; ``--only`` selects by label and is the only way an extension
    condition of the pool suite enters an allocation.
    """
    by_label = {config.label: config for config in configs}
    if args.only:
        wanted = [s.strip() for s in args.only.split(",") if s.strip()]
    elif args.suite == "svbmc_pool":
        wanted = list(POOL_LABELS)
    else:
        wanted = list(by_label)
    missing = [label for label in wanted if label not in by_label]
    if missing:
        raise RuntimeError(f"not in suite {args.suite}: {missing}")
    extensions = [
        label
        for label in wanted
        if args.suite == "svbmc_pool" and label not in POOL_LABELS
    ]
    return [by_label[label] for label in wanted], extensions


def condition_runs(out, label):
    """The tags of one condition that left a completion record or an error."""
    out = Path(out)
    tags = {
        path.name[: -len(".complete.json")]
        for path in (out / "records").glob(f"{label}_seed*.complete.json")
    }
    tags |= {
        path.name[: -len(".error.txt")]
        for path in out.glob(f"{label}_seed*.error.txt")
    }
    return sorted(tags, key=lambda tag: int(tag.rsplit("seed", 1)[1]))


def revised_history(out, previous, current, authorized_by):
    """Accept a revision of a prepared campaign's allocation, and log it.

    Filtered targets and seed caps may change and a condition may be
    added; a condition that already holds runs may neither be dropped nor
    given a different first seed. Returns the ``allocation_history`` to
    store beside the new allocation: the previous one, the time and who
    authorized the revision.
    """
    if not authorized_by:
        raise RuntimeError(
            "revising the allocation of a prepared campaign needs "
            "--authorized-by NAME"
        )
    allocated = {entry["label"]: entry for entry in current}
    for entry in previous["allocation"]:
        label = entry["label"]
        runs = condition_runs(out, label)
        if not runs:
            continue
        if label not in allocated:
            raise RuntimeError(
                f"{label} holds {len(runs)} runs in this directory and "
                "cannot be dropped from the allocation"
            )
        if allocated[label]["seed_start"] != entry["seed_start"]:
            raise RuntimeError(
                f"{label} holds {len(runs)} runs from seed "
                f"{entry['seed_start']}; its first seed cannot change"
            )
    return list(previous.get("allocation_history", [])) + [
        {
            "allocation": previous["allocation"],
            "revised": time.strftime("%Y-%m-%d %H:%M:%S"),
            "authorized_by": authorized_by,
        }
    ]


def cmd_prepare(args):
    source = activate_gpyreg(args.gpyreg_source)
    from benchmark_targets import suite_configs

    configs, extensions = select_configs(args, suite_configs(args.suite))
    manifest = {
        "campaign": "svbmc_pool",
        "suite": args.suite,
        "options": dict(BASE_OPTIONS),
        "allocation": allocation(args, configs),
        "gpyreg_source": source,
        "identity": identity(source),
        "allow_dirty": bool(args.allow_dirty),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "launch_ready": False,
        "authorized_by": None,
        "authorized_at": None,
        "allocation_history": [],
    }
    out = args.out.resolve()
    path = out / "manifest.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        differing = [
            k for k in STRUCTURAL_KEYS if previous.get(k) != manifest[k]
        ]
        if differing:
            raise RuntimeError(
                f"{path} exists and differs in {differing}; prepare a new "
                "directory instead of changing a campaign in place"
            )
        manifest["created"] = previous["created"]
        manifest["launch_ready"] = previous["launch_ready"]
        manifest["authorized_by"] = previous["authorized_by"]
        manifest["authorized_at"] = previous["authorized_at"]
        manifest["allocation_history"] = list(
            previous.get("allocation_history", [])
        )
        if manifest["allocation"] != previous["allocation"]:
            manifest["allocation_history"] = revised_history(
                out, previous, manifest["allocation"], args.authorized_by
            )
            print(
                f"{path}: allocation revised, authorized by "
                f"{args.authorized_by}",
                flush=True,
            )
    for label in extensions:
        print(
            f"{label}: extension condition of the campaign, allocated "
            "because --only names it",
            flush=True,
        )
    if args.ready:
        if not args.authorized_by:
            raise RuntimeError("--ready needs --authorized-by NAME")
        manifest["launch_ready"] = True
        manifest["authorized_by"] = args.authorized_by
        manifest["authorized_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(path, manifest)
    print(f"{path}: {len(manifest['allocation'])} conditions", flush=True)
    for entry in manifest["allocation"]:
        print(
            f"  {entry['label']}: seeds {entry['seed_start']}.."
            f"{entry['seed_start'] + entry['max_seeds'] - 1}, "
            f"target {entry['target_filtered']} filtered",
            flush=True,
        )
    print(f"launch_ready: {manifest['launch_ready']}", flush=True)
    return 0


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


def validate_case(out, tag, expected):
    """Re-verify a completed case; never silently skip a changed one."""
    import svbmc_pool_io as pool_io

    done = json.loads(
        pool_io.record_path(out, tag).read_text(encoding="utf-8")
    )
    if done["identity"] != expected:
        raise RuntimeError(
            f"{tag}: recorded identity differs in "
            f"{identity_differences(done['identity'], expected)}"
        )
    if not set(pool_io.SUFFIXES) <= set(done["hashes"]):
        raise RuntimeError(f"{tag}: incomplete file manifest")
    for suffix, digest in done["hashes"].items():
        path = Path(out) / f"{tag}{suffix}"
        if not path.exists():
            raise RuntimeError(f"{tag}: missing {suffix}")
        if pool_io.sha256(path) != digest:
            raise RuntimeError(f"{tag}: changed {suffix}")
    if done["tag"] != tag:
        raise RuntimeError(f"{tag}: record belongs to {done['tag']}")
    return done


def partial_artifacts(out, tag):
    """The artifact files one case left behind, if any."""
    return [
        f"{tag}{suffix}"
        for suffix in ARTIFACT_SUFFIXES
        if (Path(out) / f"{tag}{suffix}").exists()
    ]


def case_state(out, tag, expected):
    """``("done", record)``, ``("failed", None)``, ``("partial", files)`` or
    ``("new", None)`` for one case."""
    import svbmc_pool_io as pool_io

    if pool_io.record_path(out, tag).exists():
        return "done", validate_case(out, tag, expected)
    if (Path(out) / f"{tag}.error.txt").exists():
        return "failed", None
    leftover = partial_artifacts(out, tag)
    if leftover:
        return "partial", leftover
    return "new", None


def condition_plan(entry, pilot_seeds):
    """``(seed_cap, filtered_target)`` for one condition of this sweep."""
    if pilot_seeds:
        return int(pilot_seeds), None
    return int(entry["max_seeds"]), int(entry["target_filtered"])


def cmd_run(args):
    out = args.out.resolve()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("launch_ready"):
        raise RuntimeError(
            "manifest is not marked ready for launch "
            "(prepare --ready --authorized-by NAME)"
        )
    source = activate_gpyreg(manifest["gpyreg_source"])
    expected = identity(source)
    check_tree(expected, manifest.get("allow_dirty"))
    differing = identity_differences(expected, manifest["identity"])
    if differing:
        raise RuntimeError(
            f"identity differs from the manifest in {differing}; the pool "
            "must be generated with the code it was prepared with"
        )
    launch = out / "launch.json"
    structural = {k: manifest[k] for k in STRUCTURAL_KEYS}
    if launch.exists():
        previous = json.loads(launch.read_text(encoding="utf-8"))
        started_with = {
            k: previous["manifest"].get(k) for k in STRUCTURAL_KEYS
        }
        if previous["identity"] != expected or started_with != structural:
            raise RuntimeError(
                "an existing campaign in this directory belongs to a "
                "different setup"
            )
    else:
        write_json(
            launch,
            {
                "identity": expected,
                "manifest": structural,
                "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
    started = time.time()
    conditions = {}
    completed, failed, failed_now = [], [], []
    status = {
        "pid": os.getpid(),
        "started": started,
        "pilot_seeds": args.pilot_seeds,
        "active": None,
        "conditions": conditions,
        "completed": completed,
        "failed": failed,
        "failed_this_sweep": failed_now,
    }

    def publish(active):
        status["active"] = active
        status["updated"] = time.time()
        write_json(out / "status.json", status)

    for entry in manifest["allocation"]:
        label = entry["label"]
        cap, target = condition_plan(entry, args.pilot_seeds)
        counts = {"seeds_run": 0, "filtered": 0, "failed": 0, "seeds": []}
        conditions[label] = counts
        seed = int(entry["seed_start"])
        while counts["seeds_run"] < cap and (
            target is None or counts["filtered"] < target
        ):
            current, seed = seed, seed + 1
            tag = f"{label}_seed{current}"
            state, detail = case_state(out, tag, expected)
            if state == "partial":
                publish(None)
                raise RuntimeError(
                    f"{tag}: incomplete prior attempt ("
                    + ", ".join(detail)
                    + ") without a completion record or an error file; "
                    "inspect before retrying, and delete those files to "
                    "run the case again"
                )
            counts["seeds_run"] += 1
            counts["seeds"].append(current)
            if state == "failed":
                counts["failed"] += 1
                failed.append(tag)
                print(f"SKIP {tag} (failed earlier)", flush=True)
                continue
            if state == "done":
                record = detail
                counts["filtered"] += bool(record["verdict"]["passes"])
                completed.append(tag)
                print(
                    f"SKIP {tag} (complete, "
                    f"{'passes' if record['verdict']['passes'] else 'filtered out'})",
                    flush=True,
                )
                continue
            publish(tag)
            print(
                f"START {tag} ({counts['filtered']}/"
                f"{'-' if target is None else target} filtered, "
                f"{counts['seeds_run']}/{cap} seeds)",
                flush=True,
            )
            case_started = time.time()
            command = [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "worker",
                "--out",
                str(out),
                "--label",
                label,
                "--seed",
                str(current),
            ]
            if args.save_vbmc:
                command.append("--save-vbmc")
            child_env = dict(os.environ)
            child_env.update({k: "1" for k in THREAD_KEYS})
            child_env["MPLBACKEND"] = "Agg"
            child_env["PYVBMC_GPYREG_SOURCE"] = source
            log_path = out / f"{tag}.log"
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command,
                    cwd=str(ROOT),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=child_env,
                )
            if result.returncode:
                tail = log_path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()[-40:]
                (out / f"{tag}.error.txt").write_text(
                    f"exit code {result.returncode}\n"
                    + "\n".join(tail)
                    + "\n",
                    encoding="utf-8",
                )
                counts["failed"] += 1
                failed.append(tag)
                failed_now.append(tag)
                print(f"FAILED {tag}; see {tag}.error.txt", flush=True)
            else:
                record = validate_case(out, tag, expected)
                counts["filtered"] += bool(record["verdict"]["passes"])
                completed.append(tag)
                print(
                    f"DONE {tag} in "
                    f"{(time.time() - case_started) / 60:.1f} min; "
                    f"{'passes' if record['verdict']['passes'] else 'filtered out'}"
                    f"; elapsed {(time.time() - started) / 3600:.2f} h",
                    flush=True,
                )
            publish(None)
    publish(None)
    write_json(
        out / "finished.json",
        {
            "conditions": conditions,
            "completed": completed,
            "failed": failed,
            "failed_this_sweep": failed_now,
            "pilot_seeds": args.pilot_seeds,
            "elapsed_seconds": time.time() - started,
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    # A case that failed in an earlier sweep is known and was skipped;
    # only a failure of this sweep makes it fail.
    return 1 if failed_now else 0


# --------------------------------------------------------------------------
# worker
# --------------------------------------------------------------------------


def cmd_worker(args):
    out = args.out.resolve()
    launch = out / "launch.json"
    if launch.exists():
        loaded = json.loads(launch.read_text(encoding="utf-8"))
        manifest, expected = loaded["manifest"], loaded["identity"]
    else:  # invoked per case, outside a `run` sweep
        manifest = json.loads(
            (out / "manifest.json").read_text(encoding="utf-8")
        )
        expected = manifest["identity"]
    source = activate_gpyreg(manifest["gpyreg_source"])
    actual = identity(source)
    differing = identity_differences(actual, expected)
    if differing:
        raise RuntimeError(f"worker identity differs in {differing}")

    import svbmc_pool_io as pool_io
    from benchmark_targets import find_config, metrics
    from profile_run import jsonable

    from pyvbmc import VBMC

    tag = f"{args.label}_seed{args.seed}"
    started = time.time()
    cfg = find_config(args.label)
    problem = cfg.make(seed=args.seed)
    vbmc_args, options = problem.vbmc_args()
    options.update(manifest["options"])
    vbmc = VBMC(*vbmc_args, options=options, seed=args.seed)
    clock = time.perf_counter()
    vp, results = vbmc.optimize()
    wall = time.perf_counter() - clock
    measured = jsonable(metrics(problem, vp, results["elbo"]))
    verdict = pool_io.filter_verdict(vp)
    report = pool_io.save_run(
        out,
        tag,
        vbmc,
        results,
        problem,
        cfg,
        args.seed,
        {"wall_s": wall},
        meta_extra={"metrics": measured, "filter": verdict},
    )
    hashes = dict(report["hashes"])
    if args.save_vbmc:
        vbmc.save(out / f"{tag}.vbmc.pkl", overwrite=True)
        hashes[".vbmc.pkl"] = pool_io.sha256(out / f"{tag}.vbmc.pkl")
    write_json(
        pool_io.record_path(out, tag),
        {
            "tag": tag,
            "label": args.label,
            "seed": int(args.seed),
            "identity": actual,
            "elapsed_seconds": time.time() - started,
            "wall_s": wall,
            "target_eval_s": float(vbmc.function_logger.total_fun_eval_time),
            "K": int(vp.K),
            "func_count": int(results["func_count"]),
            "elbo": float(results["elbo"]),
            "elbo_sd": float(results["elbo_sd"]),
            "success_flag": bool(results["success_flag"]),
            "verdict": verdict,
            "metrics": measured,
            "verification": {
                "checks": report["checks"],
                "differences": report["differences"],
                "bytes": report["bytes"],
            },
            "hashes": hashes,
        },
    )
    print(
        f"{tag}: elbo {results['elbo']:.3f} +- {results['elbo_sd']:.3f}, "
        f"K {vp.K}, {results['func_count']} evaluations, "
        f"{wall / 60:.1f} min, "
        f"{'passes' if verdict['passes'] else 'filtered out'} "
        f"(stable {verdict['stable']}, "
        f"max J_sjk {verdict['max_J_sjk']:.3f})",
        flush=True,
    )
    return 0


# --------------------------------------------------------------------------
# summarize
# --------------------------------------------------------------------------


def quartiles(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    if not values:
        return {"n": 0, "median": None, "q1": None, "q3": None}
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return {
        "n": len(values),
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
    }


def condition_summary(out, entry):
    """One condition of a pool, counted from the cases it holds.

    Every case of the condition that left a completion record or an error
    file is counted, whatever seed cap the sweeps that generated them
    worked under; the allocation contributes the first seed, the cap and
    the filtered target the counts are read against.
    """
    import svbmc_pool_io as pool_io

    label = entry["label"]
    records, failures = [], []
    for tag in condition_runs(out, label):
        path = pool_io.record_path(out, tag)
        if path.exists():
            records.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            first = (
                (Path(out) / f"{tag}.error.txt")
                .read_text(encoding="utf-8", errors="replace")
                .strip()
                .splitlines()
            )
            failures.append({"tag": tag, "reason": first[-1] if first else ""})
    passed = [r for r in records if r["verdict"]["passes"]]
    seeds_run = len(records) + len(failures)
    usable = [
        r
        for r in passed
        if all(
            np.isfinite(r["metrics"][k]) and r["metrics"][k] < bound
            for k, bound in THRESHOLDS.items()
        )
    ]
    summary = {
        "label": label,
        "seed_start": entry["seed_start"],
        "seed_cap": int(entry["max_seeds"]),
        "target_filtered": int(entry["target_filtered"]),
        "seeds_run": seeds_run,
        "completed": len(records),
        "failed": len(failures),
        "failures": failures,
        "filtered": len(passed),
        "pass_rate": (len(passed) / len(records)) if records else None,
        "unstable": sum(1 for r in records if not r["verdict"]["stable"]),
        "above_s_max": sum(
            1
            for r in records
            if r["verdict"]["stable"] and not r["verdict"]["passes"]
        ),
        "usable_fraction": (len(usable) / len(passed)) if passed else None,
        "wall_minutes": quartiles([r["wall_s"] / 60 for r in records]),
        "func_count": quartiles([r["func_count"] for r in records]),
        "K": quartiles([r["K"] for r in passed]),
        "max_J_sjk": quartiles([r["verdict"]["max_J_sjk"] for r in records]),
    }
    for key in ("elbo_err", "gskl", "mmtv", "rmse"):
        summary[key] = quartiles([r["metrics"][key] for r in passed])
    return summary


def summary_markdown(summary):
    gpyreg_commit = summary["identity"]["gpyreg_commit"] or "unknown"
    lines = [
        f"# S-VBMC pool: {summary['directory']}",
        "",
        f"Generated {summary['generated']} from "
        f"`{summary['identity']['pyvbmc_commit']}` "
        f"(gpyreg `{gpyreg_commit[:12]}`) on "
        f"{summary['identity']['hostname']}. Filters: stable and "
        "`sqrt(max J_sjk) < sqrt(5)`; the counts are of the runs the "
        "directory holds, against the allocation's seed caps and filtered "
        "targets, and the metric quartiles are over the filtered runs, "
        "the wall times over every completed run.",
        "",
    ]
    if summary["pilot_seeds"]:
        lines += [
            f"The last sweep ran with `--pilot-seeds "
            f"{summary['pilot_seeds']}`, which stops every condition at "
            "that many seeds whatever its filtered target.",
            "",
        ]
    lines += [
        "| condition | seeds | filtered | pass rate | usable | "
        "wall min (med [IQR]) | evals | K | elbo_err | gskl | mmtv |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def cell(q, digits=2):
        if q["median"] is None:
            return "-"
        return (
            f"{q['median']:.{digits}f} "
            f"[{q['q1']:.{digits}f}, {q['q3']:.{digits}f}]"
        )

    for condition in summary["conditions"]:
        rate = condition["pass_rate"]
        usable = condition["usable_fraction"]
        lines.append(
            "| {label} | {seeds}/{cap} | {filtered}/{target} | {rate} | "
            "{usable} | {wall} | {evals} | {K} | {elbo} | {gskl} | "
            "{mmtv} |".format(
                label=condition["label"],
                seeds=condition["seeds_run"],
                cap=condition["seed_cap"],
                filtered=condition["filtered"],
                target=condition["target_filtered"],
                rate="-" if rate is None else f"{rate:.2f}",
                usable="-" if usable is None else f"{usable:.2f}",
                wall=cell(condition["wall_minutes"], 1),
                evals=cell(condition["func_count"], 0),
                K=cell(condition["K"], 0),
                elbo=cell(condition["elbo_err"]),
                gskl=cell(condition["gskl"]),
                mmtv=cell(condition["mmtv"], 3),
            )
        )
    failures = [
        f"- `{f['tag']}`: {f['reason']}"
        for condition in summary["conditions"]
        for f in condition["failures"]
    ]
    if failures:
        lines += ["", "## Failed runs", ""] + failures
    return "\n".join(lines) + "\n"


def cmd_summarize(args):
    out = args.out.resolve()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    finished = out / "finished.json"
    pilot_seeds = None
    if finished.exists():
        pilot_seeds = json.loads(finished.read_text(encoding="utf-8")).get(
            "pilot_seeds"
        )
    summary = {
        "campaign": manifest["campaign"],
        "directory": out.name,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        # What the last sweep was told to do, for the reader; the counts
        # below come from the cases the directory holds.
        "pilot_seeds": pilot_seeds,
        "identity": manifest["identity"],
        "options": manifest["options"],
        "conditions": [
            condition_summary(out, entry) for entry in manifest["allocation"]
        ],
    }
    summary["totals"] = {
        "seeds_run": sum(c["seeds_run"] for c in summary["conditions"]),
        "filtered": sum(c["filtered"] for c in summary["conditions"]),
        "failed": sum(c["failed"] for c in summary["conditions"]),
    }
    write_json(out / "summary.json", summary)
    text = summary_markdown(summary)
    (out / "summary.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
    return 0


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="write the campaign manifest")
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--suite", default="svbmc_pool")
    prepare.add_argument(
        "--only",
        help="comma-separated subset of labels; without it the svbmc_pool "
        "suite allocates its five pool conditions and any other suite all "
        "of its entries",
    )
    prepare.add_argument("--target", type=int, default=60)
    prepare.add_argument("--max-seeds", type=int, default=90)
    prepare.add_argument("--seed-start", type=int, default=1000)
    prepare.add_argument("--control-target", type=int)
    prepare.add_argument("--control-max-seeds", type=int)
    prepare.add_argument(
        "--allocation",
        action="append",
        metavar="LABEL=TARGET/MAXSEEDS",
        help="per-condition allocation, repeatable",
    )
    prepare.add_argument("--gpyreg-source", type=Path, default=DEFAULT_GPYREG)
    prepare.add_argument(
        "--allow-dirty",
        action="store_true",
        help="record that the pool may be generated from a dirty tree",
    )
    prepare.add_argument("--ready", action="store_true")
    prepare.add_argument("--authorized-by")

    runner = sub.add_parser("run", help="generate the pool")
    runner.add_argument("--out", type=Path, required=True)
    runner.add_argument(
        "--pilot-seeds",
        type=int,
        help="run exactly this many seeds per condition, ignoring the "
        "filtered target",
    )
    runner.add_argument("--save-vbmc", action="store_true")

    worker = sub.add_parser("worker", help="one run (one fresh process)")
    worker.add_argument("--out", type=Path, required=True)
    worker.add_argument("--label", required=True)
    worker.add_argument("--seed", type=int, required=True)
    worker.add_argument("--save-vbmc", action="store_true")

    summarize = sub.add_parser("summarize", help="summarize a pool")
    summarize.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "prepare":
        return cmd_prepare(args)
    if args.command == "worker":
        return cmd_worker(args)
    if args.command == "summarize":
        return cmd_summarize(args)
    # Process-scoped request: permit display sleep, prevent idle system
    # sleep while a long sweep is working.
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        args.out.mkdir(parents=True, exist_ok=True)
        with FileLock(str(args.out / "campaign.lock"), timeout=0):
            return cmd_run(args)
    finally:
        if sys.platform == "win32":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    sys.exit(main())
