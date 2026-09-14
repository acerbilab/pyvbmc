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
        --suite svbmc_pool
    python -u dev/scripts/svbmc_pool_run.py run --out DIR --pilot-seeds 3
    python -u dev/scripts/svbmc_pool_run.py run --out DIR
    python dev/scripts/svbmc_pool_run.py cases --out DIR
    python dev/scripts/svbmc_pool_run.py worker --out DIR --label L --seed S
    python dev/scripts/svbmc_pool_run.py select --out DIR
    python dev/scripts/svbmc_pool_run.py summarize --out DIR
    python dev/scripts/svbmc_pool_run.py verify --out DIR

``prepare`` writes the manifest: the allocation, the run options and the
identity of the code and environment the pool is generated with. It
allocates every condition of the named suite, which for ``svbmc_pool`` is
the campaign's eight pool conditions (``POOL_LABELS``); ``--only``
allocates a subset of them. The defaults are
the campaign's approved allocation, so the command above needs no
allocation flags: 100 filtered runs per noisy condition and 50 per
noiseless control, seed caps 150, 200 for the ring (``DEFAULT_SEED_CAPS``)
and 75 for the controls. The narrower the flag, the later it wins:
``--target`` and ``--max-seeds`` set every condition including the
controls, ``--control-target`` and ``--control-max-seeds`` the controls
alone, ``--allocation LABEL=TARGET/MAXSEEDS`` one condition.
Run on a directory that already holds a manifest, ``prepare`` revises the
allocation; nothing else about the campaign can change. Filtered targets
and seed caps may be revised and a condition added, a condition that
already has runs may neither be dropped nor given a different first seed,
and the previous allocation and the time are appended to
``allocation_history``, which is how the pilot's revision of the seed caps
is recorded.

``run`` is the laptop supervisor: it refuses a source identity that
differs from the prepared one, takes the campaign lock, and walks the
conditions in manifest order, seeds upward, one
worker subprocess at a time, stopping each condition at its filtered
target or its seed cap. A failing case leaves ``<tag>.error.txt`` — the
one its own worker wrote, or one the supervisor writes from the case log
when the worker died before it could —, counts towards the seed cap, and
the sweep continues; an artifact file of a case with neither a completion
record nor an error file stops the sweep for inspection. The ``<tag>.log``
an interrupted sweep leaves behind is not such a file: it is truncated
when the case runs again. A sweep that runs
to its targets leaves exactly the runs ``select`` would choose, so the
comparison reads the same pool either way; a sweep run with
``--pilot-seeds`` or under a lowered target leaves more than the target,
and ``select`` is then what says which of them the pool is.

``cases`` prints every ``label seed`` pair of the allocation over the
full seed range, and ``worker`` runs one of them in one fresh process, so
a cluster can generate the same pool as an array job, one task per case.
The plan's section "Cluster generation" owns that hand-over; the shape of
it is three separate pieces. ``dev/scripts/hpc/`` holds the scripts that
implement this sketch (its README is the operator's guide). On the login
node, once the campaign is prepared, write the case list and submit the
array::

    python dev/scripts/svbmc_pool_run.py cases --out $DIR > $DIR/cases.txt
    wc -l < $DIR/cases.txt      # 1100 for the campaign's allocation
    sbatch --array=1-1000%50 --cpus-per-task=1 --mem=2G pool_task.sh
    sbatch --array=1-100%50 --export=ALL,INDEX_OFFSET=1000 \\
        --cpus-per-task=1 --mem=2G pool_task.sh

Two submissions because Slurm's ``MaxArraySize`` defaults to 1001 and caps
the largest index rather than the number of tasks: the valid indices are
0..1000, which the allocation's 1100 cases exceed. The second submission
therefore runs indices 1..100 again and has the script add
``INDEX_OFFSET`` to them, so those tasks read lines 1001..1100 of the case
list; ``scontrol show config | grep MaxArraySize`` gives the site's own
limit, and the ``%50`` throttle caps how many tasks run at once. One core
and under 2 GB per case is enough.

``pool_task.sh`` is the array script, run once per task by Slurm. Its
body is these lines, which belong in the script and nowhere else; the
line of the case list one task reads is its array index plus the offset
its submission exported, which is nothing for the first chunk::

    #!/bin/bash
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    export MPLBACKEND=Agg PYVBMC_GPYREG_SOURCE=<the manifest's gpyreg_source>
    LINE=$((SLURM_ARRAY_TASK_ID + ${INDEX_OFFSET:-0}))
    CASE=$(sed -n "${LINE}p" $DIR/cases.txt)
    python -u dev/scripts/svbmc_pool_run.py worker --out $DIR \\
        --label ${CASE% *} --seed ${CASE#* }

When the array is done, on any machine that can read ``$DIR``::

    python dev/scripts/svbmc_pool_run.py select --out $DIR
    python dev/scripts/svbmc_pool_run.py summarize --out $DIR

An array runs every seed of the range rather than stopping at the
filtered target, and ``select`` then applies the campaign's stopping rule
after the fact: per condition it takes the lowest-seed runs that pass the
filters, up to the filtered target, and writes them to ``selection.json``,
which is the authoritative definition of the filtered pool and what the
stacking comparison reads. A task that fails leaves ``<tag>.error.txt``
naming the case and carrying the end of its traceback, deletes whatever
artifact files it had begun so that no truncated artifact is mistaken for
a run, and exits non-zero, so Slurm records the failure too; the case is
rerun or left out, and nothing else needs doing about it. ``run``'s
campaign lock, its ``status.json`` and its sequential stopping rule belong
to the laptop sweep alone; no array task writes any of them.

``verify`` is what reads an array's output before ``select``. It re-checks
every completed case against its completion record (the recorded identity,
which must match the manifest's source identity, and the hashes of both
files) and then post hoc with ``svbmc_pool_io.verify_run``, the
recomputation gate on the rebuilt posterior and GP included, and it
reconciles the allocation case by case: the ``failed`` cases with the last
line of their error file, the ``partial`` artifacts left without a record
or an error file, and the ``missing`` cases, which is what a task Slurm
kills for its time limit or its memory leaves behind, printed with the
array index that resubmits them; artifact files the allocation does not
name are reported as stray. The report is ``verification.json``. The
command exits non-zero when an artifact fails its checks or the directory
holds a partial or a stray file, and zero for failed and missing cases,
which are reported and then rerun or left out. A pool copied to another
machine is verified with ``--gpyreg-source`` naming a local checkout of the
pinned library instead of the manifest's path, which is accepted only at
the manifest's gpyreg commit.

gpyreg is pinned to the frozen worktree the manifest names: every
process prepends it to ``sys.path`` before PyVBMC is imported (through
``PYVBMC_GPYREG_SOURCE``, which the launcher also passes to its
children) and refuses to run when ``gpyreg`` resolves elsewhere. PyVBMC
itself is this checkout: the identity's ``source`` half records its
commit, whether the package or the suite module carries uncommitted
changes, and the hashes of the suite module and of the two pool scripts.
``prepare``, ``run``, ``worker`` and the resumption check compare that
half alone, so that one pool is generated by one code and library state
while any node may run any case; the ``host`` half (hostname, platform,
interpreter, import paths, threads) is recorded and never compared.
``run`` refuses to generate a pool from a dirty tree unless the manifest
was prepared with ``--allow-dirty``, which records the relaxation.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
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
    ROOT / "dev" / "scripts" / "runs" / "svbmc_pool_20260913" / "gpyreg_1.2.1"
)
#: The campaign's pool conditions, which are the whole ``svbmc_pool``
#: suite of ``benchmark_targets.py`` in its order. ``prepare`` allocates
#: every entry of the suite it is given, so this is what it allocates
#: without ``--only``.
POOL_LABELS = (
    "multisensory_s1_D6_noise3_svbmc",
    "multisensory_s1_D6_noise1.3_svbmc",
    "rosenbrock_D2_noise3_svbmc",
    "gmm_D2_noise3_svbmc",
    "ring_D2_noise3_svbmc",
    "student_D8_noise3_svbmc",
    "gmm_D2_svbmc",
    "multisensory_s1_D6_svbmc",
)
#: The campaign's approved allocation, the defaults of ``prepare``: 100
#: filtered runs per noisy condition and 50 per noiseless control, seed
#: caps 150 and 75.
DEFAULT_TARGET, DEFAULT_MAX_SEEDS = 100, 150
DEFAULT_CONTROL_TARGET, DEFAULT_CONTROL_MAX_SEEDS = 50, 75
#: Seed caps that depart from those defaults, per condition. The ring is
#: the campaign's most expensive condition and the one whose runs most
#: often fail the filters, so its seeds are over-provisioned.
#: ``--allocation LABEL=TARGET/MAXSEEDS`` overrides both numbers and this
#: table with them.
DEFAULT_SEED_CAPS = {"ring_D2_noise3_svbmc": 200}
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


#: The identity fields every process of one campaign must agree on.
#: ``svbmc_pool_io.environment`` provides the commits and library
#: versions; :func:`identity` adds the working-tree state and the module
#: hashes. Everything else an identity record holds says where the
#: process ran and is never compared.
SOURCE_KEYS = (
    "python",
    "pyvbmc_commit",
    "pyvbmc_dirty",
    "gpyreg_commit",
    "gpyreg_clean",
    "suite_module_dirty",
    "suite_module_sha256",
    "io_module_sha256",
    "runner_sha256",
    "numpy",
    "scipy",
)


def identity(gpyreg_source):
    """What a pool is generated by, and where the process generating it ran.

    Returns ``{"source": ..., "host": ...}``. ``source`` is the code and
    library state every process of one campaign must share, so that a
    resumed or distributed campaign cannot mix versions: the commits and
    versions of ``svbmc_pool_io.environment`` plus the state of the
    working tree (whether the package or the suite module carries
    uncommitted changes, whether the frozen gpyreg worktree is clean) and
    the hashes of the three modules a run is generated by (the suite,
    this harness and its artifact layer), which the package commit does
    not cover while they are uncommitted. ``host`` is where the process
    ran, recorded and never compared, so that any node may run any case
    of one campaign.
    """
    import svbmc_pool_io as pool_io

    directory = Path(gpyreg_source).resolve()
    record = pool_io.environment(directory)
    record["source"].update(
        {
            "pyvbmc_dirty": git(ROOT, "status", "--porcelain", "--", "pyvbmc"),
            "gpyreg_clean": not git(directory, "status", "--porcelain"),
            "suite_module_dirty": git(
                ROOT, "status", "--porcelain", "--", str(SUITE_MODULE)
            ),
            "suite_module_sha256": pool_io.sha256(SUITE_MODULE),
            "io_module_sha256": pool_io.sha256(IO_MODULE),
            "runner_sha256": pool_io.sha256(RUNNER_MODULE),
        }
    )
    record["host"]["gpyreg_source"] = str(directory)
    return record


def identity_source(record):
    """The compared half of an identity record, in either stored shape.

    Records written before the split hold one flat mapping of both
    halves; selecting :data:`SOURCE_KEYS` out of it yields the same
    ``source`` the split writes, so a campaign generated by an earlier
    version of this harness is still readable here.
    """
    if "source" in record:
        return record["source"]
    return {key: record[key] for key in SOURCE_KEYS if key in record}


def identity_host(record):
    """The recorded-only half of an identity record, in either shape."""
    if "host" in record:
        return record["host"]
    return {k: v for k, v in record.items() if k not in SOURCE_KEYS}


def identity_differences(actual, expected):
    keys = set(actual) | set(expected)
    return sorted(k for k in keys if actual.get(k) != expected.get(k))


def structural_differences(previous, manifest):
    """The fields a campaign directory is bound to that a mapping changes.

    Of the identity only the source half counts, so that a campaign
    prepared on one machine and generated on another is one campaign.
    """
    differing = []
    for key in STRUCTURAL_KEYS:
        before, now = previous.get(key), manifest[key]
        if key == "identity":
            before, now = (
                identity_source(before or {}),
                identity_source(now),
            )
        if before != now:
            differing.append(key)
    return differing


def check_tree(record, allow_dirty):
    """Refuse a dirty numerical source unless the manifest allows it."""
    record = identity_source(record)
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
    """One entry per condition: label, first seed, seed cap, filter target.

    Asked for nothing, every condition takes the campaign's approved
    allocation: the noiseless controls the ``DEFAULT_CONTROL_*`` numbers,
    every other condition ``DEFAULT_TARGET`` and ``DEFAULT_MAX_SEEDS``,
    and the conditions of :data:`DEFAULT_SEED_CAPS` their own seed cap.
    The narrower the flag, the later it wins: ``--target`` and
    ``--max-seeds`` set every condition including the controls,
    ``--control-target`` and ``--control-max-seeds`` the controls alone,
    ``--allocation LABEL=TARGET/MAXSEEDS`` one condition.
    """
    overrides = dict(parse_override(item) for item in args.allocation or [])
    asked = {
        "target": args.target is not None,
        "max_seeds": args.max_seeds is not None,
    }
    noisy_target = DEFAULT_TARGET if args.target is None else args.target
    noisy_cap = DEFAULT_MAX_SEEDS if args.max_seeds is None else args.max_seeds
    control_target = (
        args.control_target
        if args.control_target is not None
        else (noisy_target if asked["target"] else DEFAULT_CONTROL_TARGET)
    )
    control_cap = (
        args.control_max_seeds
        if args.control_max_seeds is not None
        else (noisy_cap if asked["max_seeds"] else DEFAULT_CONTROL_MAX_SEEDS)
    )
    entries = []
    for cfg in configs:
        if cfg.noise_sd is None:
            target, cap = control_target, control_cap
        else:
            target, cap = noisy_target, noisy_cap
        if not asked["max_seeds"]:
            cap = DEFAULT_SEED_CAPS.get(cfg.label, cap)
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
    """The conditions to allocate: every entry of the suite, or ``--only``.

    ``--only`` selects a subset by label, in the order it names them.
    """
    by_label = {config.label: config for config in configs}
    wanted = (
        [s.strip() for s in args.only.split(",") if s.strip()]
        if args.only
        else list(by_label)
    )
    missing = [label for label in wanted if label not in by_label]
    if missing:
        raise RuntimeError(f"not in suite {args.suite}: {missing}")
    return [by_label[label] for label in wanted]


def read_manifest(out):
    """The manifest of a campaign directory."""
    path = Path(out) / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


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


def revised_history(out, previous, current):
    """Accept a revision of a prepared campaign's allocation, and log it.

    Filtered targets and seed caps may change and a condition may be
    added; a condition that already holds runs may neither be dropped nor
    given a different first seed. Returns the ``allocation_history`` to
    store beside the new allocation: the previous allocation and the time
    it was replaced.
    """
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
        }
    ]


def cmd_prepare(args):
    source = activate_gpyreg(args.gpyreg_source)
    from benchmark_targets import suite_configs

    configs = select_configs(args, suite_configs(args.suite))
    manifest = {
        "campaign": "svbmc_pool",
        "suite": args.suite,
        "options": dict(BASE_OPTIONS),
        "allocation": allocation(args, configs),
        "gpyreg_source": source,
        "identity": identity(source),
        "allow_dirty": bool(args.allow_dirty),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "allocation_history": [],
    }
    out = args.out.resolve()
    path = out / "manifest.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        differing = structural_differences(previous, manifest)
        if differing:
            raise RuntimeError(
                f"{path} exists and differs in {differing}; prepare a new "
                "directory instead of changing a campaign in place"
            )
        manifest["created"] = previous["created"]
        manifest["allocation_history"] = list(
            previous.get("allocation_history", [])
        )
        if manifest["allocation"] != previous["allocation"]:
            manifest["allocation_history"] = revised_history(
                out, previous, manifest["allocation"]
            )
            print(f"{path}: allocation revised", flush=True)
    write_json(path, manifest)
    print(f"{path}: {len(manifest['allocation'])} conditions", flush=True)
    for entry in manifest["allocation"]:
        print(
            f"  {entry['label']}: seeds {entry['seed_start']}.."
            f"{entry['seed_start'] + entry['max_seeds'] - 1}, "
            f"target {entry['target_filtered']} filtered",
            flush=True,
        )
    return 0


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


def validate_case(out, tag, expected):
    """Re-verify a completed case; never silently skip a changed one.

    Only the source half of the identity is compared, so a case generated
    on another node of a distributed campaign is accepted and one
    generated by other code is not.
    """
    import svbmc_pool_io as pool_io

    done = json.loads(
        pool_io.record_path(out, tag).read_text(encoding="utf-8")
    )
    differing = identity_differences(
        identity_source(done["identity"]), identity_source(expected)
    )
    if differing:
        raise RuntimeError(f"{tag}: recorded identity differs in {differing}")
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
    manifest = read_manifest(out)
    source = activate_gpyreg(manifest["gpyreg_source"])
    expected = identity(source)
    check_tree(expected, manifest.get("allow_dirty"))
    differing = identity_differences(
        identity_source(expected), identity_source(manifest["identity"])
    )
    if differing:
        raise RuntimeError(
            f"identity differs from the manifest in {differing}; the pool "
            "must be generated with the code it was prepared with"
        )
    launch = out / "launch.json"
    structural = {k: manifest[k] for k in STRUCTURAL_KEYS}
    if launch.exists():
        previous = json.loads(launch.read_text(encoding="utf-8"))
        differing = structural_differences(
            previous["manifest"], manifest
        ) + identity_differences(
            identity_source(previous["identity"]), identity_source(expected)
        )
        if differing:
            raise RuntimeError(
                "an existing campaign in this directory belongs to a "
                f"different setup (differs in {sorted(set(differing))})"
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
                # The worker records its own failure; the supervisor
                # writes the record only for a worker that died before it
                # could, so that the case is never counted as unfinished.
                error_file = out / f"{tag}.error.txt"
                if not error_file.exists():
                    tail = log_path.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()[-40:]
                    error_file.write_text(
                        f"{tag}: worker exited with code "
                        f"{result.returncode} leaving no error file\n"
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


def record_failure(out, tag, error):
    """Leave one failed case as ``<tag>.error.txt`` and nothing else.

    The file names the case, the exception and the end of the traceback,
    and every artifact file the case had begun is deleted, so that a case
    which died between writing its artifact and writing its completion
    record cannot leave a truncated run behind for ``select`` or the
    comparison to read. A sweep and a Slurm array both read this file as
    the whole of what a failed case leaves.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tail = traceback.format_exc().splitlines()[-40:]
    (out / f"{tag}.error.txt").write_text(
        f"{tag}: {type(error).__name__}: {error}\n" + "\n".join(tail) + "\n",
        encoding="utf-8",
    )
    for suffix in ARTIFACT_SUFFIXES:
        (out / f"{tag}{suffix}").unlink(missing_ok=True)


def cmd_worker(args):
    """One case in one fresh process, recording its own failure.

    A case that raises leaves ``<tag>.error.txt`` and no artifact, and the
    process exits non-zero so that whatever launched it — a sweep, a Slurm
    array — sees the failure; one that succeeds removes the error file of
    an earlier attempt.
    """
    out = args.out.resolve()
    tag = f"{args.label}_seed{args.seed}"
    # Outside the guard below: a directory that holds no manifest is not a
    # case that failed, so it must leave nothing behind.
    manifest = read_manifest(out)
    try:
        worker_case(args, out, tag, manifest)
    except Exception as error:
        record_failure(out, tag, error)
        raise
    (out / f"{tag}.error.txt").unlink(missing_ok=True)
    return 0


def worker_case(args, out, tag, manifest):
    out = Path(out)
    launch = out / "launch.json"
    if launch.exists():
        loaded = json.loads(launch.read_text(encoding="utf-8"))
        manifest, expected = loaded["manifest"], loaded["identity"]
    else:  # invoked per case, outside a `run` sweep
        expected = manifest["identity"]
    source = activate_gpyreg(manifest["gpyreg_source"])
    actual = identity(source)
    # The source half alone, so that any node of an array job may run any
    # case while no node may run one with different code.
    differing = identity_differences(
        identity_source(actual), identity_source(expected)
    )
    if differing:
        raise RuntimeError(f"worker identity differs in {differing}")

    import svbmc_pool_io as pool_io
    from benchmark_targets import find_config, metrics
    from profile_run import jsonable

    from pyvbmc import VBMC

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


# --------------------------------------------------------------------------
# cases and select
# --------------------------------------------------------------------------


def manifest_cases(manifest):
    """Every ``(label, seed)`` of an allocation, in condition order.

    One case per seed of each condition's whole range, whatever the
    filtered target: an array job runs them all and :func:`cmd_select`
    applies the stopping rule afterwards.
    """
    return [
        (entry["label"], seed)
        for entry in manifest["allocation"]
        for seed in range(
            int(entry["seed_start"]),
            int(entry["seed_start"]) + int(entry["max_seeds"]),
        )
    ]


def cmd_cases(args):
    """Print the cases of a campaign, one per line, for an array job.

    The line number is the array index: line ``i`` names the case that
    ``worker --out DIR --label L --seed S`` generates. The count goes to
    standard error, so that redirecting standard output yields exactly the
    case list.
    """
    out = args.out.resolve()
    manifest = read_manifest(out)
    cases = manifest_cases(manifest)
    if args.format == "json":
        print(
            json.dumps(
                {
                    "campaign": manifest["campaign"],
                    "directory": str(out),
                    "count": len(cases),
                    "cases": [
                        {"index": index, "label": label, "seed": seed}
                        for index, (label, seed) in enumerate(cases, start=1)
                    ],
                },
                indent=2,
            ),
            flush=True,
        )
    else:
        for label, seed in cases:
            print(f"{label} {seed}", flush=True)
        print(f"{len(cases)} cases", file=sys.stderr, flush=True)
    return 0


def condition_selection(out, entry, target):
    """The campaign's stopping rule applied to one condition's records.

    The completed runs are walked in seed order and the ones that pass the
    filters are taken until ``target`` of them are in hand, the paper's
    "lowest indices" rule. Returns the selected tags, how many seeds were
    scanned to find them, and the shortfall when the records run out
    first. ``pass_rate_scanned`` is the selected runs over those scanned
    seeds, which is not the condition's pass rate: the scan stops at the
    target rather than at the end of the condition, and a seed whose case
    failed is scanned like any other. The pass rate over every completed
    case is ``summarize``'s.
    """
    import svbmc_pool_io as pool_io

    label = entry["label"]
    selected, scanned, failed = [], 0, 0
    for tag in condition_runs(out, label):
        if len(selected) >= target:
            break
        scanned += 1
        path = pool_io.record_path(out, tag)
        if not path.exists():  # the case failed and left an error file
            failed += 1
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["verdict"]["passes"]:
            selected.append({"tag": tag, "seed": int(record["seed"])})
    return {
        "label": label,
        "seed_start": int(entry["seed_start"]),
        "seed_cap": int(entry["max_seeds"]),
        "target_filtered": int(target),
        "selected": len(selected),
        "shortfall": max(0, int(target) - len(selected)),
        "seeds_scanned": scanned,
        "failed_while_scanning": failed,
        "pass_rate_scanned": (len(selected) / scanned) if scanned else None,
        "last_seed": selected[-1]["seed"] if selected else None,
        "runs": selected,
    }


def selection_markdown(selection):
    lines = [
        f"# S-VBMC filtered pool: {selection['directory']}",
        "",
        f"Written {selection['generated']} from the completion records of "
        "the campaign. Per condition the runs are walked in seed order and "
        "the ones that pass the filters (stable and `sqrt(max J_sjk) < "
        "sqrt(5)`) are taken until the filtered target is met, so the pool "
        "is the lowest-seed runs that pass, whatever order the cases were "
        "generated in. The stacking comparison reads this file.",
        "",
        "The walk stops as soon as the target is met, so the seeds it "
        "scanned are a prefix of the condition and the seeds beyond them "
        "were never looked at; a seed whose case failed is scanned like "
        "any other and leaves no run. `pass rate over the scanned seeds` "
        "is the selected runs over that prefix, failures included, and is "
        "therefore not the condition's pass rate, which `summarize` "
        "reports over every completed case.",
        "",
    ]
    if selection["target_override"] is not None:
        lines += [
            f"Every condition was selected to {selection['target_override']} "
            "runs, not to the manifest's filtered target.",
            "",
        ]
    lines += [
        "| condition | selected | target | shortfall | seeds scanned | "
        "pass rate over the scanned seeds | last seed |",
        "|---|---|---|---|---|---|---|",
    ]
    for condition in selection["conditions"]:
        rate = condition["pass_rate_scanned"]
        lines.append(
            "| {label} | {selected} | {target} | {shortfall} | {scanned} | "
            "{rate} | {last} |".format(
                label=condition["label"],
                selected=condition["selected"],
                target=condition["target_filtered"],
                shortfall=condition["shortfall"] or "-",
                scanned=condition["seeds_scanned"],
                rate="-" if rate is None else f"{rate:.2f}",
                last=(
                    "-"
                    if condition["last_seed"] is None
                    else condition["last_seed"]
                ),
            )
        )
    return "\n".join(lines) + "\n"


def cmd_select(args):
    """Write the filtered pool of a campaign: ``selection.json`` and ``.md``.

    The authoritative definition of the filtered pool, applied after the
    fact to whatever runs the directory holds. ``run``'s sequential
    stopping rule produces the same set when it walks the seeds in order;
    an array job that runs every seed of the range needs this step.
    """
    out = args.out.resolve()
    manifest = read_manifest(out)
    conditions = [
        condition_selection(
            out,
            entry,
            entry["target_filtered"] if args.target is None else args.target,
        )
        for entry in manifest["allocation"]
    ]
    selection = {
        "campaign": manifest["campaign"],
        "directory": str(out),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target_override": args.target,
        "identity": manifest["identity"],
        "totals": {
            "selected": sum(c["selected"] for c in conditions),
            "shortfall": sum(c["shortfall"] for c in conditions),
        },
        "conditions": conditions,
    }
    write_json(out / "selection.json", selection)
    text = selection_markdown(selection)
    (out / "selection.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
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
    source = identity_source(summary["identity"])
    host = identity_host(summary["identity"])
    gpyreg_commit = source["gpyreg_commit"] or "unknown"
    lines = [
        f"# S-VBMC pool: {summary['directory']}",
        "",
        f"Generated {summary['generated']}. The campaign was prepared on "
        f"{host['hostname']} at `{source['pyvbmc_commit']}` "
        f"(gpyreg `{gpyreg_commit[:12]}`), the code every case was "
        "generated with; each case records the host that ran it. "
        "Filters: stable and "
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
    manifest = read_manifest(out)
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
# verify
# --------------------------------------------------------------------------

#: The case states :func:`cmd_verify` counts, in the totals and per
#: condition. ``verify_failed`` is a case whose own verification raised,
#: a fault of the stored artifact rather than of the allocation; it is
#: counted with the others so that a condition's counts add up to its
#: cases.
VERIFY_STATUSES = (
    "verified",
    "failed",
    "partial",
    "missing",
    "verify_failed",
)


def error_reason(out, tag):
    """The last non-empty line of one failed case's error file."""
    lines = (
        (Path(out) / f"{tag}.error.txt")
        .read_text(encoding="utf-8", errors="replace")
        .strip()
        .splitlines()
    )
    return lines[-1] if lines else ""


def stray_tags(out, tags):
    """The artifact tags of a directory that the allocation does not name.

    Only what a case leaves is looked at, its ``<tag>.npz``, its optional
    ``<tag>.vbmc.pkl`` and its completion record, so that the campaign's
    own files (the manifest, the reports, the case list, the lock, a log
    or an error file) and the directories beside them (``records/``,
    ``slurm/``) are never mistaken for a run.
    """
    out = Path(out)
    candidates = {path.name[: -len(".npz")] for path in out.glob("*.npz")}
    candidates |= {
        path.name[: -len(".vbmc.pkl")] for path in out.glob("*.vbmc.pkl")
    }
    candidates |= {
        path.name[: -len(".complete.json")]
        for path in (out / "records").glob("*.complete.json")
    }
    return sorted(candidates - set(tags))


def status_counts(cases, statuses):
    return {
        status: sum(1 for case in cases if case["status"] == status)
        for status in statuses
    }


def case_verification(out, tag, expected, pool_io):
    """One case of the allocation, re-checked against what it left behind.

    The guard is broad on purpose: ``case_state`` raises ``RuntimeError``
    for a record that no longer describes its files, and ``verify_run`` on
    a truncated ``.npz`` raises whatever the codec beneath it raises
    (``zlib.error``, ``OSError``, ``KeyError``, ``ValueError``). Every one
    of them is this case failing its verification, not the command.
    """
    try:
        state, detail = case_state(out, tag, expected)
        if state == "done":
            report = pool_io.verify_run(Path(out) / tag)
            return {
                "status": "verified",
                "passes": bool(detail["verdict"]["passes"]),
                "differences": report["differences"],
            }
        if state == "failed":
            return {"status": "failed", "reason": error_reason(out, tag)}
        if state == "partial":
            return {"status": "partial", "files": detail}
        return {"status": "missing"}
    except Exception as error:
        return {
            "status": "verify_failed",
            "error": f"{type(error).__name__}: {error}",
        }


def case_line(case):
    """One reported case: ``index tag status``, and what it left."""
    line = f"{case['index']} {case['tag']} {case['status']}"
    detail = {
        "failed": case.get("reason"),
        "partial": ", ".join(case.get("files", [])),
        "verify_failed": case.get("error"),
    }.get(case["status"])
    return f"{line}: {detail}" if detail else line


def cmd_verify(args):
    """Re-check every artifact of a campaign and reconcile the allocation.

    The post-hoc verification of a pool an array job generated, and what
    makes ``select`` and ``summarize`` trustworthy on one: those two read
    the cases that left a completion record or an error file, and a task
    Slurm killed leaves neither, so it would silently be left out. Every
    case of the allocation is placed instead — ``verified``, ``failed``,
    ``partial``, ``missing``, or ``verify_failed`` when the check itself
    raised — and every artifact file the allocation does not name is
    reported as stray.
    """
    out = args.out.resolve()
    manifest = read_manifest(out)
    if args.gpyreg_source is not None:
        try:
            commit = git(args.gpyreg_source, "rev-parse", "HEAD")
            dirty = git(args.gpyreg_source, "status", "--porcelain")
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError(
                f"{args.gpyreg_source} is not a git checkout; --gpyreg-source "
                "names a clone of gpyreg at the manifest's commit"
            ) from error
        pinned = identity_source(manifest["identity"])["gpyreg_commit"]
        if commit != pinned:
            raise RuntimeError(
                f"{args.gpyreg_source} is at gpyreg commit {commit}, not "
                f"the manifest's {pinned}; the pool was generated by that "
                "commit and is verified against it"
            )
        if dirty:
            raise RuntimeError(
                f"{args.gpyreg_source} has uncommitted changes; the pool is "
                "verified against the pinned library as committed"
            )
    source = activate_gpyreg(args.gpyreg_source or manifest["gpyreg_source"])
    # The manifest's identity, not this process's: a pool is verified
    # against the code it was prepared with, wherever it is read.
    expected = manifest["identity"]

    import svbmc_pool_io as pool_io

    cases = []
    for index, (label, seed) in enumerate(manifest_cases(manifest), start=1):
        tag = f"{label}_seed{seed}"
        cases.append(
            {
                "index": index,
                "tag": tag,
                "label": label,
                "seed": int(seed),
                **case_verification(out, tag, expected, pool_io),
            }
        )
    stray = stray_tags(out, [case["tag"] for case in cases])
    counts = status_counts(cases, VERIFY_STATUSES)
    counts["stray"] = len(stray)
    by_label = {}
    for case in cases:
        by_label.setdefault(case["label"], []).append(case)
    conditions = [
        {
            "label": entry["label"],
            **status_counts(by_label.get(entry["label"], []), VERIFY_STATUSES),
        }
        for entry in manifest["allocation"]
    ]
    write_json(
        out / "verification.json",
        {
            "campaign": manifest["campaign"],
            "directory": str(out),
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpyreg_source": source,
            "identity": manifest["identity"],
            "counts": counts,
            "conditions": conditions,
            "stray": stray,
            "cases": cases,
        },
    )
    print("| condition | " + " | ".join(VERIFY_STATUSES) + " |", flush=True)
    print("|" + "---|" * (len(VERIFY_STATUSES) + 1), flush=True)
    for condition in conditions:
        cells = " | ".join(
            str(condition[key]) for key in ("label",) + VERIFY_STATUSES
        )
        print(f"| {cells} |", flush=True)
    print(
        f"stray: {counts['stray']}"
        + (f" ({', '.join(stray)})" if stray else ""),
        flush=True,
    )
    for case in cases:
        if case["status"] != "verified":
            print(case_line(case), flush=True)
    print(
        f"{len(cases)} cases: "
        + ", ".join(f"{key} {value}" for key, value in counts.items()),
        flush=True,
    )
    # A failed case is rerun or left out and a missing one is resubmitted,
    # both of which the operator reads off this report; an artifact that
    # fails its checks, a partial one and a stray file are the faults that
    # must be settled before the pool is read.
    fatal = ("verify_failed", "partial", "stray")
    return 1 if any(counts[key] for key in fatal) else 0


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
        help="comma-separated subset of the suite's labels; without it "
        "every entry of the suite is allocated",
    )
    prepare.add_argument(
        "--target",
        type=int,
        help="filtered runs per condition, the controls included; without "
        f"it {DEFAULT_TARGET} per noisy condition and "
        f"{DEFAULT_CONTROL_TARGET} per noiseless one",
    )
    prepare.add_argument(
        "--max-seeds",
        type=int,
        help="seeds per condition, the controls included; without it "
        f"{DEFAULT_MAX_SEEDS} per noisy condition, "
        + ", ".join(
            f"{cap} for {label}" for label, cap in DEFAULT_SEED_CAPS.items()
        )
        + f" and {DEFAULT_CONTROL_MAX_SEEDS} per noiseless one",
    )
    prepare.add_argument("--seed-start", type=int, default=1000)
    prepare.add_argument(
        "--control-target",
        type=int,
        help="filtered runs per noiseless condition, overriding --target "
        f"for those (default {DEFAULT_CONTROL_TARGET})",
    )
    prepare.add_argument(
        "--control-max-seeds",
        type=int,
        help="seeds per noiseless condition, overriding --max-seeds for "
        f"those (default {DEFAULT_CONTROL_MAX_SEEDS})",
    )
    prepare.add_argument(
        "--allocation",
        action="append",
        metavar="LABEL=TARGET/MAXSEEDS",
        help="per-condition allocation, repeatable; overrides every "
        "default above for the condition it names",
    )
    prepare.add_argument("--gpyreg-source", type=Path, default=DEFAULT_GPYREG)
    prepare.add_argument(
        "--allow-dirty",
        action="store_true",
        help="record that the pool may be generated from a dirty tree",
    )

    runner = sub.add_parser("run", help="generate the pool")
    runner.add_argument("--out", type=Path, required=True)
    runner.add_argument(
        "--pilot-seeds",
        type=int,
        help="run exactly this many seeds per condition, ignoring the "
        "filtered target",
    )
    runner.add_argument("--save-vbmc", action="store_true")

    cases = sub.add_parser(
        "cases", help="print every (label, seed) of the allocation"
    )
    cases.add_argument("--out", type=Path, required=True)
    cases.add_argument(
        "--format",
        choices=("lines", "json"),
        default="lines",
        help="one `label seed` per line (the array index is the line "
        "number, and the count goes to standard error), or one JSON object",
    )

    worker = sub.add_parser("worker", help="one run (one fresh process)")
    worker.add_argument("--out", type=Path, required=True)
    worker.add_argument("--label", required=True)
    worker.add_argument("--seed", type=int, required=True)
    worker.add_argument("--save-vbmc", action="store_true")

    select = sub.add_parser("select", help="write the filtered pool")
    select.add_argument("--out", type=Path, required=True)
    select.add_argument(
        "--target",
        type=int,
        help="select this many filtered runs per condition instead of the "
        "manifest's filtered target",
    )

    summarize = sub.add_parser("summarize", help="summarize a pool")
    summarize.add_argument("--out", type=Path, required=True)

    verify = sub.add_parser(
        "verify",
        help="re-check every artifact and reconcile the allocation",
    )
    verify.add_argument("--out", type=Path, required=True)
    verify.add_argument(
        "--gpyreg-source",
        type=Path,
        help="a gpyreg checkout to use instead of the manifest's path, for "
        "a pool copied to another machine; it must be at the manifest's "
        "gpyreg commit",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "prepare":
        return cmd_prepare(args)
    if args.command == "cases":
        return cmd_cases(args)
    if args.command == "worker":
        return cmd_worker(args)
    if args.command == "select":
        return cmd_select(args)
    if args.command == "summarize":
        return cmd_summarize(args)
    if args.command == "verify":
        return cmd_verify(args)
    # The sequential sweep below is the laptop supervisor: the campaign
    # lock, `status.json` and the idle-sleep request are its own, and no
    # other sub-command takes them. Process-scoped request: permit display
    # sleep, prevent idle system sleep while a long sweep is working.
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
