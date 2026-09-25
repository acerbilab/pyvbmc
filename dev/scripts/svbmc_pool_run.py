"""Generate a pool of independent VBMC runs for the S-VBMC campaign.

The pools are the input of the ELBO debiasing estimator and of the
stacking comparison against the original S-VBMC; the campaign, its
conditions, seeds, filters and stopping rule are
``dev/plans/svbmc-benchmark-campaign.md``, and how the release pools run on
a Slurm cluster is ``dev/plans/slurm-benchmark-support.md``. Each run is
one fresh process on one configuration and seed of the ``svbmc_pool``
suite of ``benchmark_targets.py``, saved as the per-run artifact of the
campaign plan (``svbmc_pool_io.save_run``: the returned posterior with all
of its statistics, the GP behind them, the transformer, every evaluation,
the run's state and its metadata) and recorded in a hash-verified
completion record.

The harness meets the campaign contract of the Slurm plan
(``campaign_contract.py``), so the driver under ``dev/scripts/hpc/``
(``campaign_submit.sh``, ``campaign_task.sbatch``, ``campaign_finish.sh``,
with ``HARNESS=dev/scripts/svbmc_pool_run.py``) runs a pool as Slurm job
arrays. Sub-commands::

    python dev/scripts/svbmc_pool_run.py prepare --out DIR \\
        --gpyreg-source PATH [--suite svbmc_pool] [allocation flags]
    python dev/scripts/svbmc_pool_run.py cases --out DIR [--subset LABEL]
    python dev/scripts/svbmc_pool_run.py worker --out DIR --case LINE
    python dev/scripts/svbmc_pool_run.py verify --out DIR
    python dev/scripts/svbmc_pool_run.py select --out DIR
    python dev/scripts/svbmc_pool_run.py summarize --out DIR
    python -u dev/scripts/svbmc_pool_run.py run --out DIR [--pilot-seeds K]

``prepare`` writes the manifest: the allocation, the run options, the
identity of the code the pool is generated with, the operator settings of
the Slurm plan (the ``site`` block), the environment's ``pip freeze`` and
the finishing steps, ``select`` and then ``summarize``. ``--gpyreg-source``
names the gpyreg checkout every process of the campaign imports, and must
name the directory of ``PYVBMC_GPYREG_SOURCE`` where that is set. The
identity is ``campaign_contract.identity`` over two trees, the whole
harness checkout (``harness``) and gpyreg: their commits and clean states,
the SHA-256 of this harness, ``svbmc_pool_io.py``, ``benchmark_targets.py``,
``campaign_contract.py`` and the data directory ``dev/scripts/data``, and
the imported versions of Python, NumPy, SciPy and cma; ``pyvbmc`` must be
imported from the harness checkout and ``gpyreg`` from its checkout. A
gpyreg checkout with any uncommitted or untracked change is refused, and so
is a dirty harness checkout unless ``--allow-dirty``, which the manifest
records. It allocates every condition of the named suite, which for
``svbmc_pool`` is the campaign's eight pool conditions (``POOL_LABELS``);
``--only`` allocates a subset of them. The defaults are the campaign's
approved allocation: 100 filtered runs per noisy condition and 50 per
noiseless control, seed caps 150, 200 for the ring
(``DEFAULT_SEED_CAPS``) and 75 for the controls. The narrower the flag,
the later it wins: ``--target`` and ``--max-seeds`` set every condition
including the controls, ``--control-target`` and ``--control-max-seeds``
the controls alone, ``--allocation LABEL=TARGET/MAXSEEDS`` one condition.
The release pools are ``--target 320 --max-seeds 350 --allocation
ring_D2_noise3_svbmc=320/480``, 2930 cases. Run on a directory that
already holds a manifest, ``prepare`` revises the allocation and nothing
else: filtered targets and seed caps may be revised and a condition added,
a condition that already has runs may neither be dropped nor given a
different first seed, and the previous allocation and the time are
appended to ``allocation_history``. The driver writes the case list once,
so a campaign that has been submitted is not revised.

``cases`` prints one line per case, its tag ``<label>/<label>_seed<seed>``,
for every seed of each condition's range whatever its filtered target,
conditions in manifest order; line ``i`` is case index ``i``. ``--subset
LABEL`` prints ``<index> <line>`` for the cases of one condition, which the
driver submits as its own array with its own time limit.

``worker --case LINE`` runs one case in one fresh process through
``campaign_contract.run_worker``: it exits 0 at once when the case has a
completion record, refuses a source identity that differs from the
manifest's (exit 78) and a case that a live task has claimed (exit 75),
removes what an interrupted attempt left, runs the case and writes its
artifact and its completion record. A case that raises leaves
``<tag>.error.txt`` with the traceback and no artifact and exits 1; a
SIGTERM or SIGINT removes the partial files and exits 128 plus the
signal's number, so that ``verify`` reports the case as missing. A line
that is not a case of the allocation is refused with exit 64 and touches
nothing.

``run`` is the workstation supervisor: it refuses a source identity that
differs from the manifest's, takes the campaign lock and walks the
conditions in manifest order, seeds upward, one ``worker`` subprocess at a
time, stopping each condition at its filtered target or its seed cap
(``--pilot-seeds K`` runs exactly ``K`` seeds of each instead). A case
that failed counts towards the seed cap and the sweep goes on; a worker
that dies without writing its error file gets one from the tail of its
``<tag>.log``, and its partial files are removed. The sweep stops for
inspection at a case whose artifact files lie without a record, an error
file or a claim, and at a case that another process holds. A sweep that
runs to its targets leaves exactly the runs ``select`` would choose; one
run with ``--pilot-seeds`` or under a lowered target leaves more, and
``select`` then says which of them the pool is. The campaign lock,
``status.json`` and ``finished.json`` belong to ``run`` alone.

``select`` applies the campaign's stopping rule after the fact: per
condition it takes the lowest-seed runs that pass the filters, up to the
filtered target, and writes them to ``selection.json`` (per condition in
``conditions``, the selected ``tag`` and ``seed`` pairs in seed order in
``runs``), the authoritative definition of the filtered pool, which the
stacking comparison reads. ``summarize`` writes the per-condition counts,
pass rates, convergence (``success_flag``, ``convergence_status``,
``message``, ``r_index``, ``iterations``), wall times and metric quartiles
of every case the directory holds, to ``summary.json`` and ``summary.md``.

``verify`` re-checks every completed case against its record and the
manifest, and reconciles the allocation case by case with
``campaign_contract.reconcile``: ``verified``, ``verify_failed``,
``failed`` (an error file), ``in_flight`` (a live claim), ``interrupted``
(artifact files and a stale claim, which a task killed outright leaves),
``partial`` (artifact files with neither a record nor a claim),
``missing``, and the stray files no case owns. A record passes when its
source identity equals the manifest's, every artifact it lists has its
SHA-256, and, where the manifest's ``site`` block names ``NODE_FEATURE``,
its host part shows that feature and one physical core; the artifact must
then pass ``svbmc_pool_io.verify_run``, the recomputation gate on the
rebuilt posterior and GP included. The report is ``verification.json``;
the command exits non-zero when a case fails its verification or the
directory holds a partial case or a stray file, and zero for failed,
interrupted and missing cases, which are resubmitted or left out. A pool
copied to another machine is verified with ``--gpyreg-source`` naming a
local checkout of the pinned library instead of the manifest's path, which
is accepted only at the manifest's gpyreg commit. On that other machine the
recomputation is no longer bit-identical: its BLAS rounds differently, and
the rebuilt posterior factors carry that rounding amplified by the
condition number of each run's GP, up to a relative 0.3 on the
ill-conditioned GPs of the noisy Rosenbrock condition, whose training sets
hold far-tail evaluations. The gate allows each artifact its own amplified
rounding (``svbmc_pool_io.verify_run``; ``--rounding-factor`` scales the
allowance, 0 restores the absolute gate) and the report carries every
case's relative differences and condition number, per condition their
maxima, and the identity of the verifying checkout next to the pool's.

A campaign directory holds, per condition ``<label>``,
``<label>/<label>_seed<seed>.npz`` and ``.json`` (the artifact; ``run
--save-vbmc`` adds ``.vbmc.pkl``), ``.error.txt`` for a failed case and
``.log`` for a case ``run`` started, and beside them the contract's
``records/<tag>.complete.json`` and ``claims/<tag>``. A completion record
holds the contract's fields (the case, the identity with its host part,
the times, the SHA-256 and size of every artifact file) and the run's own:
``label``, ``seed``, ``wall_s``, ``target_eval_s``, ``K``,
``func_count``, ``elbo``, ``elbo_sd``, ``success_flag``,
``convergence_status``, ``message``, ``r_index``, ``iterations``,
``verdict`` (the filters), ``metrics`` (against the truth) and
``verification`` (the checks the artifact passed when it was saved).

Pools prepared before the campaign contract, among them
``dev/experiments/svbmc_pool/pool_20260914/``, keep their flat layout,
recognised by a manifest without a ``contract`` field: artifacts
``<label>_seed<seed>.*`` at the top of the directory, records
``records/<label>_seed<seed>.complete.json`` holding the hashes by suffix
and the flat identity of :func:`identity`. ``verify``, ``select``,
``summarize`` and ``cases`` read such a pool as it is, with the checks it
was generated under; ``prepare``, ``worker`` and ``run`` refuse it.

gpyreg is pinned to the checkout the manifest names: every process
prepends it to ``sys.path`` before PyVBMC is imported (through
``PYVBMC_GPYREG_SOURCE``, which ``run`` also passes to its workers) and
refuses to run when ``gpyreg`` resolves elsewhere.
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
# Honour a gpyreg pin already in the environment (the campaign's
# environment script, a child of `run`); `activate_gpyreg` installs the
# manifest's pin for a process started without it. Either way this
# happens before the first PyVBMC import, which is why nothing under
# `pyvbmc` or `svbmc_pool_io` is imported at module level here.
if os.environ.get("PYVBMC_GPYREG_SOURCE"):
    sys.path.insert(1, os.environ["PYVBMC_GPYREG_SOURCE"])

import campaign_contract as contract  # noqa: E402
import numpy as np  # noqa: E402
from filelock import FileLock  # noqa: E402

SUITE_MODULE = ROOT / "dev" / "scripts" / "benchmark_targets.py"
IO_MODULE = HERE / "svbmc_pool_io.py"
RUNNER_MODULE = Path(__file__).resolve()
#: The frozen gpyreg 1.2.1 worktree against which ``pool_20260914`` is
#: read, at its place under the gitignored ``dev/scripts/runs/`` (listed in
#: ``dev/scripts/runs/LOCAL.md``). This harness takes its gpyreg checkout
#: as an argument; the scripts that default to this path import it here.
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
#: life. The allocation is not one of them: a revision of targets and
#: seed caps (``cmd_prepare``) leaves every run already generated valid.
STRUCTURAL_KEYS = (
    "campaign",
    "suite",
    "options",
    "gpyreg_source",
    "identity",
    "allow_dirty",
)
#: The suffixes of one case's artifact files. A ``<tag>.log`` is not one:
#: every case that ``run`` starts writes one, and it is truncated on a
#: rerun.
ARTIFACT_SUFFIXES = (".npz", ".json", ".vbmc.pkl")
#: The files and directories whose SHA-256 the source identity holds,
#: relative to the harness checkout.
IDENTITY_FILES = (
    "dev/scripts/svbmc_pool_run.py",
    "dev/scripts/svbmc_pool_io.py",
    "dev/scripts/benchmark_targets.py",
    "dev/scripts/campaign_contract.py",
    "dev/scripts/data",
)
#: The packages whose import path the identity records, each with the
#: tree it must be imported from.
IDENTITY_MODULES = {"pyvbmc": "harness", "gpyreg": "gpyreg"}
#: The finishing steps the driver runs on a verified directory.
FINISHING_STEPS = [["select"], ["summarize"]]
#: Exit code of a worker given a line that is not a case of the
#: allocation, or a directory it may not generate cases in (``EX_USAGE``).
EXIT_USAGE = 64
#: The convergence fields of a run's results that its completion record
#: holds, beside ``success_flag``.
CONVERGENCE_KEYS = ("convergence_status", "message", "r_index", "iterations")

write_json = contract.write_json


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


# --------------------------------------------------------------------------
# Layouts
# --------------------------------------------------------------------------


def is_contract(manifest):
    """Whether a manifest was prepared under the campaign contract.

    A pool prepared before it (the flat layout of ``pool_20260914``) has
    no ``contract`` field.
    """
    return "contract" in manifest


def case_tag(label, seed, contract_layout=True):
    """The tag of one case: ``<label>/<label>_seed<seed>``, or in the flat
    layout ``<label>_seed<seed>``."""
    name = f"{label}_seed{int(seed)}"
    return f"{label}/{name}" if contract_layout else name


def tag_seed(tag):
    """The seed a tag of either layout names."""
    return int(tag.rsplit("_seed", 1)[1])


def manifest_gpyreg_commit(manifest):
    """The gpyreg commit a manifest of either layout pins."""
    identity = manifest["identity"]
    if is_contract(manifest):
        return identity["source"]["trees"]["gpyreg"]["commit"]
    return identity_source(identity).get("gpyreg_commit")


def pinned_gpyreg_source(source, manifest):
    """A local gpyreg checkout accepted in place of the manifest's path.

    A pool's manifest names the checkout it was generated against as an
    absolute path on the machine that generated it, which a copy of the
    pool on another machine cannot use. ``source`` is accepted instead
    when it is a clean git checkout at the manifest's gpyreg commit, the
    library the pool's numerics were produced by; anything else raises
    ``RuntimeError`` naming what differs. Returns the resolved path,
    ready for :func:`activate_gpyreg`.
    """
    source = Path(source).resolve()
    try:
        commit = git(source, "rev-parse", "HEAD")
        dirty = git(source, "status", "--porcelain")
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"{source} is not a git checkout; --gpyreg-source names a clone "
            "of gpyreg at the manifest's commit"
        ) from error
    pinned = manifest_gpyreg_commit(manifest)
    if commit != pinned:
        raise RuntimeError(
            f"{source} is at gpyreg commit {commit}, not the manifest's "
            f"{pinned}; the pool was generated by that commit and is read "
            "against it"
        )
    if dirty:
        raise RuntimeError(
            f"{source} has uncommitted changes; the pool is read against "
            "the pinned library as committed"
        )
    return str(source)


# --------------------------------------------------------------------------
# Identities
# --------------------------------------------------------------------------


def campaign_identity(gpyreg_source, host=True):
    """The identity of this process under the campaign contract.

    ``campaign_contract.identity`` over the harness checkout and the gpyreg
    checkout, with the files of :data:`IDENTITY_FILES` and the import paths
    of :data:`IDENTITY_MODULES`: ``source`` (compared by every worker),
    ``imports`` and, with ``host``, the host part (recorded only).
    """
    return contract.identity(
        {"harness": ROOT, "gpyreg": Path(gpyreg_source).resolve()},
        IDENTITY_FILES,
        modules=IDENTITY_MODULES,
        host=host,
    )


def dirty_trees(record, allow_dirty):
    """Refuse a dirty source tree of a contract identity.

    The gpyreg checkout must be clean, since the pin is what makes the
    pool reproducible; the harness checkout may be dirty only with
    ``allow_dirty``. Returns the names of the dirty trees.
    """
    trees = record["source"]["trees"]
    status = record.get("imports", {}).get("trees", {})

    def detail(name):
        lines = (status.get(name) or {}).get("dirty") or []
        more = f" and {len(lines) - 5} more" if len(lines) > 5 else ""
        return "; ".join(lines[:5]) + more

    if not trees["gpyreg"]["clean"]:
        raise RuntimeError(
            "the gpyreg checkout has uncommitted or untracked changes "
            f"({detail('gpyreg')}); the pin is what makes the pool "
            "reproducible"
        )
    dirty = [name for name, tree in trees.items() if not tree["clean"]]
    if dirty and not allow_dirty:
        raise RuntimeError(
            "uncommitted or untracked changes in the "
            + ", ".join(f"{name} checkout ({detail(name)})" for name in dirty)
            + "; commit them, or prepare with --allow-dirty to record that "
            "the pool is generated from a dirty tree"
        )
    return dirty


#: The fields of the flat identity (:func:`identity`) that the pools
#: prepared before the campaign contract compared between processes.
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
    """The flat identity: what a pool of the flat layout was generated by.

    Returns ``{"source": ..., "host": ...}``: the commits and versions of
    ``svbmc_pool_io.environment`` plus the state of the working tree
    (whether the package or the suite module carries uncommitted changes,
    whether the gpyreg checkout is clean) and the hashes of the suite, this
    harness and its artifact layer; ``host`` is where the process ran. The
    pools prepared before the campaign contract compare its ``source``
    half, and the analysis scripts record it as their provenance
    (:func:`analysis_sources`).
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


def analysis_sources(script, cells, pool, gpyreg_source):
    """The provenance record of a script that scores recorded cells.

    The script's path and hash, the cells file's path and hash, the pool
    directory, this process's identity (commits, versions, working-tree
    state, where gpyreg resolved) and the thread settings.
    """
    import svbmc_pool_io as pool_io

    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": {
            "path": str(Path(script).resolve()),
            "sha256": pool_io.sha256(script),
        },
        "cells": {
            "path": str(Path(cells).resolve()),
            "sha256": pool_io.sha256(cells),
        },
        "pool": str(Path(pool).resolve()),
        "environment": identity(gpyreg_source),
        "threads": {key: os.environ.get(key) for key in THREAD_KEYS},
    }


def identity_source(record):
    """The compared half of a flat identity record, in either stored shape.

    Records written before the split into halves hold one flat mapping of
    both; selecting :data:`SOURCE_KEYS` out of it yields the same
    ``source`` the split writes.
    """
    if "source" in record:
        return record["source"]
    return {key: record[key] for key in SOURCE_KEYS if key in record}


def identity_host(record):
    """The recorded-only half of a flat identity record, in either shape."""
    if "host" in record:
        return record["host"]
    return {k: v for k, v in record.items() if k not in SOURCE_KEYS}


def identity_differences(actual, expected):
    keys = set(actual) | set(expected)
    return sorted(k for k in keys if actual.get(k) != expected.get(k))


def identity_summary(record, contract_layout):
    """``(pyvbmc commit, gpyreg commit, hostname)`` of an identity."""
    if contract_layout:
        trees = record["source"]["trees"]
        return (
            trees["harness"]["commit"],
            trees["gpyreg"]["commit"],
            (record.get("host") or {}).get("hostname"),
        )
    source, host = identity_source(record), identity_host(record)
    return (
        source.get("pyvbmc_commit"),
        source.get("gpyreg_commit"),
        host.get("hostname"),
    )


def structural_differences(previous, manifest):
    """The fields a campaign directory is bound to that a mapping changes.

    Of the identity only the source part counts, so that a campaign
    prepared on one machine and generated on another is one campaign; its
    differing keys are named.
    """
    differing = []
    for key in STRUCTURAL_KEYS:
        before, now = previous.get(key), manifest[key]
        if key == "identity":
            changed = contract.source_differences(now, before)
            if changed:
                differing.append(f"identity ({', '.join(changed)})")
        elif before != now:
            differing.append(key)
    return differing


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


def condition_runs(out, label, contract_layout=True):
    """The tags of one condition that left a completion record or an error.

    Only the directories of the condition are listed: under the contract,
    ``records/<label>/`` and ``<label>/``; in the flat layout, ``records/``
    and the top of the directory. Sorted by seed.
    """
    out = Path(out)
    name = f"{label}_seed*"
    if contract_layout:
        records, errors = out / "records" / label, out / label
        prefix = f"{label}/"
    else:
        records, errors, prefix = out / "records", out, ""
    tags = {
        prefix + path.name[: -len(".complete.json")]
        for path in records.glob(f"{name}.complete.json")
    }
    tags |= {
        prefix + path.name[: -len(".error.txt")]
        for path in errors.glob(f"{name}.error.txt")
    }
    return sorted(tags, key=tag_seed)


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
        runs = condition_runs(out, label, is_contract(previous))
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
    out = args.out.resolve()
    path = out / "manifest.json"
    previous = read_manifest(out) if path.exists() else None
    if previous is not None and not is_contract(previous):
        raise RuntimeError(
            f"{path} is a pool of the flat layout, prepared before the "
            "campaign contract; it is verified, selected and summarized "
            "here, and never revised or extended"
        )
    requested = Path(args.gpyreg_source).resolve()
    pinned = os.environ.get("PYVBMC_GPYREG_SOURCE")
    if pinned and Path(pinned).resolve() != requested:
        raise RuntimeError(
            f"--gpyreg-source {requested} is not PYVBMC_GPYREG_SOURCE "
            f"{Path(pinned).resolve()}; every process of the campaign "
            "imports one gpyreg checkout"
        )
    source = activate_gpyreg(requested)
    from benchmark_targets import suite_configs

    configs = select_configs(args, suite_configs(args.suite))
    manifest = {
        "campaign": "svbmc_pool",
        "contract": contract.CONTRACT_VERSION,
        "suite": args.suite,
        "options": dict(BASE_OPTIONS),
        "allocation": allocation(args, configs),
        "gpyreg_source": source,
        "identity": campaign_identity(source),
        "allow_dirty": bool(args.allow_dirty),
    }
    if previous is not None:
        differing = structural_differences(previous, manifest)
        if differing:
            raise RuntimeError(
                f"{path} exists and differs in {differing}; prepare a new "
                "directory instead of changing a campaign in place"
            )
    dirty = dirty_trees(manifest["identity"], manifest["allow_dirty"])
    if previous is None:
        manifest.update(
            {
                "site": contract.site_block(),
                "pip_freeze": contract.pip_freeze(),
                "finishing_steps": FINISHING_STEPS,
                "created": contract.now(),
                "allocation_history": [],
            }
        )
    elif manifest["allocation"] != previous["allocation"]:
        history = revised_history(out, previous, manifest["allocation"])
        manifest = dict(
            previous,
            allocation=manifest["allocation"],
            allocation_history=history,
        )
        print(f"{path}: allocation revised", flush=True)
    else:
        manifest = previous
    write_json(path, manifest)
    cases = manifest_cases(manifest)
    print(
        f"{path}: {len(manifest['allocation'])} conditions, "
        f"{len(cases)} cases"
        + (
            f"; generated from a dirty {', '.join(dirty)} tree"
            if dirty
            else ""
        ),
        flush=True,
    )
    for entry in manifest["allocation"]:
        print(
            f"  {entry['label']}: seeds {entry['seed_start']}.."
            f"{entry['seed_start'] + entry['max_seeds'] - 1}, "
            f"target {entry['target_filtered']} filtered",
            flush=True,
        )
    return 0


# --------------------------------------------------------------------------
# cases
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


def case_lines(manifest):
    """The case list: line ``i`` is case index ``i``.

    Under the contract a line is the case's tag; in the flat layout it is
    ``<label> <seed>``, the case list of the pools prepared before it.
    """
    if is_contract(manifest):
        return [
            case_tag(label, seed) for label, seed in manifest_cases(manifest)
        ]
    return [f"{label} {seed}" for label, seed in manifest_cases(manifest)]


def allocated_case(manifest, line):
    """``(label, seed)`` of a line of the case list, or None."""
    for case, candidate in zip(manifest_cases(manifest), case_lines(manifest)):
        if candidate == line:
            return case
    return None


def cmd_cases(args):
    """Print the cases of a campaign, one per line, for the driver.

    The line number is the case index. With ``--subset LABEL``, one
    condition's cases as ``<index> <line>``. The count goes to standard
    error, so that redirecting standard output yields exactly the list.
    """
    out = args.out.resolve()
    manifest = read_manifest(out)
    cases = manifest_cases(manifest)
    lines = case_lines(manifest)
    if args.format == "json":
        print(
            json.dumps(
                {
                    "campaign": manifest["campaign"],
                    "directory": str(out),
                    "count": len(cases),
                    "cases": [
                        {
                            "index": index,
                            "line": line,
                            "label": label,
                            "seed": seed,
                        }
                        for index, ((label, seed), line) in enumerate(
                            zip(cases, lines), start=1
                        )
                    ],
                },
                indent=2,
            ),
            flush=True,
        )
        return 0
    if args.subset is not None:
        labels = [entry["label"] for entry in manifest["allocation"]]
        if args.subset not in labels:
            print(
                f"no subset {args.subset!r}: the subsets are the "
                f"conditions of the allocation, {labels}",
                file=sys.stderr,
            )
            return 1
        chosen = [
            f"{index} {line}"
            for index, ((label, _), line) in enumerate(
                zip(cases, lines), start=1
            )
            if label == args.subset
        ]
        for line in chosen:
            print(line)
        print(f"{len(chosen)} cases", file=sys.stderr, flush=True)
        return 0
    for line in lines:
        print(line)
    print(f"{len(cases)} cases", file=sys.stderr, flush=True)
    return 0


# --------------------------------------------------------------------------
# worker
# --------------------------------------------------------------------------


def case_files(out, tag):
    """Every artifact path one case may write."""
    return [Path(out) / f"{tag}{suffix}" for suffix in ARTIFACT_SUFFIXES]


def partial_artifacts(out, tag):
    """The artifact files one case left behind, relative to ``out``."""
    return [
        f"{tag}{suffix}"
        for suffix in ARTIFACT_SUFFIXES
        if (Path(out) / f"{tag}{suffix}").exists()
    ]


def convergence(results):
    """The convergence fields of a run's results, as plain values."""
    r_index = results.get("r_index")
    return {
        "convergence_status": str(results["convergence_status"]),
        "message": str(results["message"]),
        "r_index": None if r_index is None else float(r_index),
        "iterations": int(results["iterations"]),
    }


def run_case(out, manifest, tag, label, seed, save_vbmc=False):
    """Run one case and write its artifact; ``(artifact paths, fields)``.

    The fields are the run's own part of the completion record.
    """
    import svbmc_pool_io as pool_io
    from benchmark_targets import find_config, metrics
    from profile_run import jsonable

    from pyvbmc import VBMC

    base = Path(out) / tag
    cfg = find_config(label)
    problem = cfg.make(seed=seed)
    vbmc_args, options = problem.vbmc_args()
    options.update(manifest["options"])
    vbmc = VBMC(*vbmc_args, options=options, seed=seed)
    clock = time.perf_counter()
    vp, results = vbmc.optimize()
    wall = time.perf_counter() - clock
    measured = jsonable(metrics(problem, vp, results["elbo"]))
    verdict = pool_io.filter_verdict(vp)
    base.parent.mkdir(parents=True, exist_ok=True)
    report = pool_io.save_run(
        base.parent,
        base.name,
        vbmc,
        results,
        problem,
        cfg,
        seed,
        {"wall_s": wall},
        meta_extra={"metrics": measured, "filter": verdict},
    )
    artifacts = [
        base.with_name(f"{base.name}{suffix}") for suffix in pool_io.SUFFIXES
    ]
    if save_vbmc:
        pickled = base.with_name(f"{base.name}.vbmc.pkl")
        vbmc.save(pickled, overwrite=True)
        artifacts.append(pickled)
    fields = {
        "label": label,
        "seed": int(seed),
        "wall_s": wall,
        "target_eval_s": float(vbmc.function_logger.total_fun_eval_time),
        "K": int(vp.K),
        "func_count": int(results["func_count"]),
        "elbo": float(results["elbo"]),
        "elbo_sd": float(results["elbo_sd"]),
        "success_flag": bool(results["success_flag"]),
        **convergence(results),
        "verdict": verdict,
        "metrics": measured,
        "verification": {
            "checks": report["checks"],
            "differences": report["differences"],
            "bytes": report["bytes"],
        },
    }
    print(
        f"{tag}: elbo {results['elbo']:.3f} +- {results['elbo_sd']:.3f}, "
        f"K {vp.K}, {results['func_count']} evaluations, "
        f"{results['iterations']} iterations, {wall / 60:.1f} min, "
        f"convergence {results['convergence_status']}, "
        f"{'passes' if verdict['passes'] else 'filtered out'} "
        f"(stable {verdict['stable']}, "
        f"max J_sjk {verdict['max_J_sjk']:.3f})",
        flush=True,
    )
    return artifacts, fields


def cmd_worker(args):
    """One case in one fresh process, through ``run_worker``."""
    out = args.out.resolve()
    manifest = read_manifest(out)
    if not is_contract(manifest):
        print(
            f"{out} is a pool of the flat layout, prepared before the "
            "campaign contract; no case runs in it",
            file=sys.stderr,
        )
        return EXIT_USAGE
    case = allocated_case(manifest, args.case)
    if case is None:
        print(
            f"{args.case!r} is not a case of the allocation of {out}; "
            "the worker takes a line of `cases`",
            file=sys.stderr,
        )
        return EXIT_USAGE
    label, seed = case
    tag = contract.case_tag(args.case)
    source = manifest["gpyreg_source"]

    def this_identity():
        return campaign_identity(activate_gpyreg(source))

    def run(identity):
        return run_case(out, manifest, tag, label, seed, args.save_vbmc)

    return contract.run_worker(
        out,
        args.case,
        manifest["identity"],
        this_identity,
        run,
        lambda: case_files(out, tag),
    )


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


def check_record(out, tag, expected, node_feature=None):
    """Re-check one completion record; raise ``CompletionError``.

    ``campaign_contract.check_completion`` (the source identity, the
    artifact files and their SHA-256, and with ``node_feature`` the host
    part) with both artifact files required, and the record's label and
    seed those of its tag.
    """
    record = contract.check_completion(
        out,
        tag,
        expected,
        required=[f"{tag}.npz", f"{tag}.json"],
        node_feature=node_feature,
    )
    label = tag.split("/", 1)[0]
    problems = []
    if record.get("label") != label or record.get("seed") != tag_seed(tag):
        problems.append(
            f"the record is of {record.get('label')} seed "
            f"{record.get('seed')}"
        )
    if not isinstance(record.get("verdict"), dict):
        problems.append("the record holds no filter verdict")
    if problems:
        raise contract.CompletionError(tag, problems)
    return record


def case_state(out, tag, expected):
    """The state of one case as ``run`` sees it: ``(state, detail)``.

    ``done`` (a record that passes :func:`check_record`, the record),
    ``in_flight`` (a live claim), ``interrupted`` (artifact files and a
    stale claim, which the worker takes over), ``partial`` (artifact
    files with neither a record nor a claim), ``failed`` (an error file)
    or ``new``.
    """
    if contract.record_path(out, tag).exists():
        return "done", check_record(out, tag, expected)
    claim = contract.claim_status(out, tag)
    if claim["state"] == "live":
        return "in_flight", claim
    files = partial_artifacts(out, tag)
    if files:
        return (
            "interrupted" if claim["state"] == "stale" else "partial"
        ), files
    if contract.error_path(out, tag).exists():
        return "failed", None
    return "new", None


def condition_plan(entry, pilot_seeds):
    """``(seed_cap, filtered_target)`` for one condition of this sweep."""
    if pilot_seeds:
        return int(pilot_seeds), None
    return int(entry["max_seeds"]), int(entry["target_filtered"])


def died_worker(out, tag, returncode, log_path, pid):
    """What ``run`` leaves of a worker that died without its error file.

    The error file, from the tail of the case's log; the partial artifact
    files are removed, and the dead worker's own claim with them, so that
    the case counts as failed.
    """
    error_file = contract.error_path(out, tag)
    tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    error_file.write_text(
        f"{tag}: worker exited with code {returncode} leaving no error "
        "file\n" + "\n".join(tail[-40:]) + "\n",
        encoding="utf-8",
    )
    for path in case_files(out, tag):
        path.unlink(missing_ok=True)
    claim = contract.claim_path(out, tag)
    try:
        record = contract.read_json(claim)
    except (OSError, ValueError):
        return
    if record.get("pid") == pid and not record.get("job"):
        claim.unlink(missing_ok=True)


def cmd_run(args):
    out = args.out.resolve()
    manifest = read_manifest(out)
    if not is_contract(manifest):
        raise RuntimeError(
            f"{out} is a pool of the flat layout, prepared before the "
            "campaign contract; it is never extended"
        )
    source = activate_gpyreg(manifest["gpyreg_source"])
    expected = manifest["identity"]
    differing = contract.source_differences(
        campaign_identity(source, host=False), expected
    )
    if differing:
        raise RuntimeError(
            f"identity differs from the manifest in {differing}; the pool "
            "must be generated with the code it was prepared with"
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
            tag = case_tag(label, current)
            state, detail = case_state(out, tag, expected)
            if state == "partial":
                publish(None)
                raise RuntimeError(
                    f"{tag}: incomplete prior attempt ("
                    + ", ".join(detail)
                    + ") without a completion record, an error file or a "
                    "claim; inspect before retrying, and delete those files "
                    "to run the case again"
                )
            if state == "in_flight":
                publish(None)
                raise RuntimeError(
                    f"{tag}: claimed by {detail['owner']} since "
                    f"{detail['started']} ({detail['detail']}); another "
                    "process is running it"
                )
            counts["seeds_run"] += 1
            counts["seeds"].append(current)
            if state == "failed":
                counts["failed"] += 1
                failed.append(tag)
                print(f"SKIP {tag} (failed earlier)", flush=True)
                continue
            if state == "done":
                passes = bool(detail["verdict"]["passes"])
                counts["filtered"] += passes
                completed.append(tag)
                print(
                    f"SKIP {tag} (complete, "
                    f"{'passes' if passes else 'filtered out'})",
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
                "--case",
                tag,
            ]
            if args.save_vbmc:
                command.append("--save-vbmc")
            child_env = dict(os.environ)
            child_env.update({k: "1" for k in THREAD_KEYS})
            child_env["MPLBACKEND"] = "Agg"
            child_env["PYVBMC_GPYREG_SOURCE"] = source
            log_path = out / f"{tag}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as log:
                child = subprocess.Popen(
                    command,
                    cwd=str(ROOT),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=child_env,
                )
                returncode = child.wait()
            if (
                returncode
                in (
                    contract.EXIT_CLAIMED,
                    contract.EXIT_IDENTITY,
                    EXIT_USAGE,
                )
                or returncode > 128
            ):
                publish(None)
                raise RuntimeError(
                    f"{tag}: the worker exited with code {returncode} "
                    f"(see {log_path}); the sweep stops"
                )
            if returncode:
                # The worker records its own failure; the supervisor
                # writes the record only for a worker that died before it
                # could, so that the case is never counted as unfinished.
                if not contract.error_path(out, tag).exists():
                    died_worker(out, tag, returncode, log_path, child.pid)
                counts["failed"] += 1
                failed.append(tag)
                failed_now.append(tag)
                print(f"FAILED {tag}; see {tag}.error.txt", flush=True)
            else:
                record = check_record(out, tag, expected)
                passes = bool(record["verdict"]["passes"])
                counts["filtered"] += passes
                completed.append(tag)
                print(
                    f"DONE {tag} in "
                    f"{(time.time() - case_started) / 60:.1f} min; "
                    f"{'passes' if passes else 'filtered out'}"
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
# select
# --------------------------------------------------------------------------


def read_record(out, tag):
    path = Path(out) / "records" / f"{tag}.complete.json"
    return json.loads(path.read_text(encoding="utf-8"))


def condition_selection(out, entry, target, contract_layout=True):
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
    label = entry["label"]
    selected, scanned, failed = [], 0, 0
    for tag in condition_runs(out, label, contract_layout):
        if len(selected) >= target:
            break
        scanned += 1
        if not (Path(out) / "records" / f"{tag}.complete.json").exists():
            failed += 1  # the case failed and left an error file
            continue
        record = read_record(out, tag)
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
            is_contract(manifest),
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


def tally(values):
    """``{value: count}`` of the values that are not None, most common first."""
    counts = {}
    for value in values:
        if value is not None:
            counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def condition_summary(out, entry, contract_layout=True):
    """One condition of a pool, counted from the cases it holds.

    Every case of the condition that left a completion record or an error
    file is counted, whatever seed cap the sweeps that generated them
    worked under; the allocation contributes the first seed, the cap and
    the filtered target the counts are read against. The convergence
    counts and quartiles are over the completed runs whose record holds
    the field (the flat layout's records hold ``success_flag`` alone).
    """
    label = entry["label"]
    records, failures = [], []
    for tag in condition_runs(out, label, contract_layout):
        if (Path(out) / "records" / f"{tag}.complete.json").exists():
            records.append(read_record(out, tag))
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
        "success_flag": sum(1 for r in records if r.get("success_flag")),
        "convergence_status": tally(
            r.get("convergence_status") for r in records
        ),
        "messages": tally(r.get("message") for r in records),
        "iterations": quartiles([r.get("iterations") for r in records]),
        "r_index": quartiles([r.get("r_index") for r in records]),
        "wall_minutes": quartiles([r["wall_s"] / 60 for r in records]),
        "func_count": quartiles([r["func_count"] for r in records]),
        "K": quartiles([r["K"] for r in passed]),
        "max_J_sjk": quartiles([r["verdict"]["max_J_sjk"] for r in records]),
    }
    for key in ("elbo_err", "gskl", "mmtv", "rmse"):
        summary[key] = quartiles([r["metrics"][key] for r in passed])
    return summary


def summary_markdown(summary, contract_layout=True):
    pyvbmc_commit, gpyreg_commit, hostname = identity_summary(
        summary["identity"], contract_layout
    )
    lines = [
        f"# S-VBMC pool: {summary['directory']}",
        "",
        f"Generated {summary['generated']}. The campaign was prepared on "
        f"{hostname} at `{pyvbmc_commit}` "
        f"(gpyreg `{(gpyreg_commit or 'unknown')[:12]}`), the code every "
        "case was generated with; each case records the host that ran it. "
        "Filters: stable and "
        "`sqrt(max J_sjk) < sqrt(5)`; the counts are of the runs the "
        "directory holds, against the allocation's seed caps and filtered "
        "targets, and the metric quartiles are over the filtered runs, "
        "the wall times over every completed run. `success` counts the "
        "completed runs whose `success_flag` is true and `converged` those "
        "whose `convergence_status` is `probable`, where the records hold "
        "it.",
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
        "| condition | seeds | filtered | pass rate | usable | success | "
        "converged | wall min (med [IQR]) | evals | K | elbo_err | gskl | "
        "mmtv |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
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
        statuses = condition["convergence_status"]
        lines.append(
            "| {label} | {seeds}/{cap} | {filtered}/{target} | {rate} | "
            "{usable} | {success}/{completed} | {converged} | {wall} | "
            "{evals} | {K} | {elbo} | {gskl} | {mmtv} |".format(
                label=condition["label"],
                seeds=condition["seeds_run"],
                cap=condition["seed_cap"],
                filtered=condition["filtered"],
                target=condition["target_filtered"],
                rate="-" if rate is None else f"{rate:.2f}",
                usable="-" if usable is None else f"{usable:.2f}",
                success=condition["success_flag"],
                completed=condition["completed"],
                converged=(
                    f"{statuses.get('probable', 0)}/{sum(statuses.values())}"
                    if statuses
                    else "-"
                ),
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
    contract_layout = is_contract(manifest)
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
            condition_summary(out, entry, contract_layout)
            for entry in manifest["allocation"]
        ],
    }
    summary["totals"] = {
        "seeds_run": sum(c["seeds_run"] for c in summary["conditions"]),
        "filtered": sum(c["filtered"] for c in summary["conditions"]),
        "failed": sum(c["failed"] for c in summary["conditions"]),
        "success_flag": sum(c["success_flag"] for c in summary["conditions"]),
    }
    write_json(out / "summary.json", summary)
    text = summary_markdown(summary, contract_layout)
    (out / "summary.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
    return 0


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------

#: Per-condition extremes of the recomputation gate over the verified
#: cases: the largest relative difference of the recomputed statistics
#: from the stored ones, and the largest condition number of a run's GP.
GATE_COLUMNS = ("max_relative", "max_condition_number")
#: The case states of the flat layout's ``verify``, in the totals and per
#: condition. ``verify_failed`` is a case whose own verification raised,
#: a fault of the stored artifact rather than of the allocation.
FLAT_STATUSES = (
    "verified",
    "failed",
    "partial",
    "missing",
    "verify_failed",
)


def status_counts(cases, statuses):
    return {
        status: sum(1 for case in cases if case["status"] == status)
        for status in statuses
    }


def gate_extremes(cases):
    verified = [case for case in cases if case["status"] == "verified"]
    return {
        "max_relative": max(
            (max(case["relative"].values()) for case in verified),
            default=0.0,
        ),
        "max_condition_number": max(
            (case["condition_number"] for case in verified), default=0.0
        ),
    }


def gate_fields(report, passes):
    """The fields of one verified case: its verdict and the gate's report."""
    return {
        "passes": bool(passes),
        "differences": report["differences"],
        "relative": report["relative"],
        "condition_number": report["condition_number"],
    }


def verification_source(args, manifest):
    """The gpyreg checkout ``verify`` reads a pool against, activated."""
    if args.gpyreg_source is not None:
        source = pinned_gpyreg_source(args.gpyreg_source, manifest)
    else:
        source = manifest["gpyreg_source"]
    return activate_gpyreg(source)


def rounding(args):
    import svbmc_pool_io as pool_io

    if args.rounding_factor is None:
        return pool_io.ROUNDING_FACTOR
    return float(args.rounding_factor)


def contract_strays(out, manifest):
    """The artifact files of a contract directory that no case owns.

    Only the directories of the allocated conditions and the top of the
    directory are listed: an artifact file of a tag the allocation does
    not name, and an artifact file of the flat layout at the top.
    """
    out = Path(out)
    tags = set(case_lines(manifest))
    stray = []
    for entry in manifest["allocation"]:
        label = entry["label"]
        for path in sorted((out / label).glob("*")):
            name = path.name
            if not path.is_file() or name.startswith("."):
                continue
            for suffix in ARTIFACT_SUFFIXES:
                if name.endswith(suffix):
                    if f"{label}/{name[: -len(suffix)]}" not in tags:
                        stray.append(f"{label}/{name}")
                    break
    for suffix in (".npz", ".vbmc.pkl"):
        stray += [path.name for path in sorted(out.glob(f"*{suffix}"))]
    return sorted(stray)


def print_verification(conditions, columns, counts, stray, cases, line):
    print("| condition | " + " | ".join(columns) + " |", flush=True)
    print("|" + "---|" * (len(columns) + 1), flush=True)
    for condition in conditions:
        cells = [condition["label"]]
        cells += [
            (
                f"{condition[key]:.2e}"
                if key in GATE_COLUMNS
                else str(condition[key])
            )
            for key in columns
        ]
        print("| " + " | ".join(cells) + " |", flush=True)
    print(
        f"stray: {len(stray)}" + (f" ({', '.join(stray)})" if stray else ""),
        flush=True,
    )
    for case in cases:
        if case["status"] != "verified":
            print(line(case), flush=True)
    print(
        f"{len(cases)} cases: "
        + ", ".join(f"{key} {value}" for key, value in counts.items()),
        flush=True,
    )


def contract_case_line(case):
    """One reported case: ``index tag status``, and what explains it."""
    text = f"{case['index']} {case['tag']} {case['status']}"
    detail = {
        "verify_failed": case.get("error"),
        "failed": case.get("reason"),
        "in_flight": (case.get("claim") or {}).get("detail"),
        "interrupted": ", ".join(case.get("files", [])),
        "partial": ", ".join(case.get("files", [])),
    }.get(case["status"])
    return f"{text}: {detail}" if detail else text


def verify_contract(args, out, manifest):
    """``verify`` on a campaign directory of the contract."""
    source = verification_source(args, manifest)
    # The manifest's identity, not this process's: a pool is verified
    # against the code it was prepared with, wherever it is read. The
    # verifier's own identity is recorded next to it, so that the report
    # says which code and libraries recomputed the artifacts.
    expected = manifest["identity"]
    verifier = campaign_identity(source)
    verifier_differs_in = contract.source_differences(verifier, expected)
    if verifier_differs_in:
        print(
            "verifying with a source identity that differs from the "
            f"pool's in {verifier_differs_in}; recorded, not compared",
            flush=True,
        )

    import svbmc_pool_io as pool_io

    rounding_factor = rounding(args)
    node_feature = (manifest.get("site") or {}).get("NODE_FEATURE")
    lines = case_lines(manifest)
    by_tag = dict(zip(lines, manifest_cases(manifest)))

    def check(tag):
        record = check_record(out, tag, expected, node_feature)
        report = pool_io.verify_run(
            out / tag,
            rounding_factor=rounding_factor,
            record=contract.record_path(out, tag),
        )
        return gate_fields(report, record["verdict"]["passes"])

    report = contract.reconcile(
        out,
        lines,
        check,
        lambda tag: partial_artifacts(out, tag),
        stray=contract_strays(out, manifest),
    )
    cases = []
    for case in report["cases"]:
        label, seed = by_tag[case["tag"]]
        cases.append(
            {
                "index": case["index"],
                "tag": case["tag"],
                "label": label,
                "seed": int(seed),
                **{
                    k: v
                    for k, v in case.items()
                    if k not in ("index", "tag", "case")
                },
            }
        )
    by_label = {}
    for case in cases:
        by_label.setdefault(case["label"], []).append(case)
    conditions = [
        {
            "label": entry["label"],
            **status_counts(
                by_label.get(entry["label"], []), contract.STATUSES
            ),
            **gate_extremes(by_label.get(entry["label"], [])),
        }
        for entry in manifest["allocation"]
    ]
    write_json(
        out / "verification.json",
        {
            "campaign": manifest["campaign"],
            "contract": report["contract"],
            "directory": str(out),
            "generated": report["generated"],
            "gpyreg_source": source,
            "identity": expected,
            "verifier": verifier,
            "verifier_differs_in": verifier_differs_in,
            "rounding_factor": rounding_factor,
            "node_feature": node_feature,
            "counts": report["counts"],
            "conditions": conditions,
            "stray": report["stray"],
            "cases": cases,
            "exit_code": report["exit_code"],
        },
    )
    print_verification(
        conditions,
        contract.STATUSES + GATE_COLUMNS,
        report["counts"],
        report["stray"],
        cases,
        contract_case_line,
    )
    return report["exit_code"]


def validate_flat_case(out, tag, expected):
    """Re-verify a completed case of the flat layout against its record.

    Only the source half of the flat identity is compared, so a case
    generated on another node is accepted and one generated by other code
    is not.
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


def flat_case_verification(out, tag, expected, pool_io, rounding_factor):
    """One case of a flat allocation, re-checked against what it left.

    The guard is broad on purpose: ``validate_flat_case`` raises
    ``RuntimeError`` for a record that no longer describes its files, and
    ``verify_run`` on a truncated ``.npz`` raises whatever the codec
    beneath it raises (``zlib.error``, ``OSError``, ``KeyError``,
    ``ValueError``). Every one of them is this case failing its
    verification, not the command.
    """
    try:
        if pool_io.record_path(out, tag).exists():
            record = validate_flat_case(out, tag, expected)
            report = pool_io.verify_run(
                Path(out) / tag, rounding_factor=rounding_factor
            )
            return {
                "status": "verified",
                **gate_fields(report, record["verdict"]["passes"]),
            }
        error = Path(out) / f"{tag}.error.txt"
        if error.exists():
            lines = (
                error.read_text(encoding="utf-8", errors="replace")
                .strip()
                .splitlines()
            )
            return {"status": "failed", "reason": lines[-1] if lines else ""}
        leftover = partial_artifacts(out, tag)
        if leftover:
            return {"status": "partial", "files": leftover}
        return {"status": "missing"}
    except Exception as error:
        return {
            "status": "verify_failed",
            "error": f"{type(error).__name__}: {error}",
        }


def flat_stray_tags(out, tags):
    """The artifact tags of a flat directory that the allocation does not
    name: its ``<tag>.npz``, ``<tag>.vbmc.pkl`` and completion records."""
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


def flat_case_line(case):
    line = f"{case['index']} {case['tag']} {case['status']}"
    detail = {
        "failed": case.get("reason"),
        "partial": ", ".join(case.get("files", [])),
        "verify_failed": case.get("error"),
    }.get(case["status"])
    return f"{line}: {detail}" if detail else line


def verify_flat(args, out, manifest):
    """``verify`` on a pool of the flat layout, with its own checks.

    Its records hold the flat identity and the hashes by suffix, so each
    is checked by :func:`validate_flat_case`, and the states are those the
    flat layout can hold: verified, failed, partial, missing and
    verify_failed, and the stray artifact tags.
    """
    source = verification_source(args, manifest)
    expected = manifest["identity"]
    verifier = identity(source)
    verifier_differs_in = identity_differences(
        identity_source(verifier), identity_source(expected)
    )
    if verifier_differs_in:
        print(
            "verifying with a source identity that differs from the "
            f"pool's in {verifier_differs_in}; recorded, not compared",
            flush=True,
        )

    import svbmc_pool_io as pool_io

    rounding_factor = rounding(args)
    cases = []
    for index, (label, seed) in enumerate(manifest_cases(manifest), start=1):
        tag = case_tag(label, seed, contract_layout=False)
        cases.append(
            {
                "index": index,
                "tag": tag,
                "label": label,
                "seed": int(seed),
                **flat_case_verification(
                    out, tag, expected, pool_io, rounding_factor
                ),
            }
        )
    stray = flat_stray_tags(out, [case["tag"] for case in cases])
    counts = status_counts(cases, FLAT_STATUSES)
    counts["stray"] = len(stray)
    by_label = {}
    for case in cases:
        by_label.setdefault(case["label"], []).append(case)
    conditions = [
        {
            "label": entry["label"],
            **status_counts(by_label.get(entry["label"], []), FLAT_STATUSES),
            **gate_extremes(by_label.get(entry["label"], [])),
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
            "identity": expected,
            "verifier": verifier,
            "verifier_differs_in": verifier_differs_in,
            "rounding_factor": rounding_factor,
            "counts": counts,
            "conditions": conditions,
            "stray": stray,
            "cases": cases,
        },
    )
    print_verification(
        conditions,
        FLAT_STATUSES + GATE_COLUMNS,
        counts,
        stray,
        cases,
        flat_case_line,
    )
    # A failed case is rerun or left out and a missing one is resubmitted,
    # both of which the operator reads off this report; an artifact that
    # fails its checks, a partial one and a stray file are the faults that
    # must be settled before the pool is read.
    fatal = ("verify_failed", "partial", "stray")
    return 1 if any(counts[key] for key in fatal) else 0


def cmd_verify(args):
    """Re-check every artifact of a campaign and reconcile the allocation.

    What makes ``select`` and ``summarize`` trustworthy on a pool: those
    two read the cases that left a completion record or an error file,
    and a task that Slurm stopped leaves neither, so it would silently be
    left out. Every case of the allocation is placed instead, and every
    artifact file the allocation does not name is reported as stray.
    """
    out = args.out.resolve()
    manifest = read_manifest(out)
    if is_contract(manifest):
        return verify_contract(args, out, manifest)
    return verify_flat(args, out, manifest)


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
    prepare.add_argument(
        "--gpyreg-source",
        type=Path,
        required=True,
        help="the gpyreg checkout every process of the campaign imports: "
        "the top of a clean git checkout, the directory of "
        "PYVBMC_GPYREG_SOURCE where that is set",
    )
    prepare.add_argument(
        "--allow-dirty",
        action="store_true",
        help="record that the pool may be generated from a harness "
        "checkout with uncommitted or untracked changes",
    )

    runner = sub.add_parser(
        "run", help="generate the pool on a workstation, one case at a time"
    )
    runner.add_argument("--out", type=Path, required=True)
    runner.add_argument(
        "--pilot-seeds",
        type=int,
        help="run exactly this many seeds per condition, ignoring the "
        "filtered target",
    )
    runner.add_argument("--save-vbmc", action="store_true")

    cases = sub.add_parser(
        "cases", help="print the case list, one tag per line"
    )
    cases.add_argument("--out", type=Path, required=True)
    cases.add_argument(
        "--subset",
        metavar="LABEL",
        help="print '<index> <line>' for the cases of one condition",
    )
    cases.add_argument(
        "--format",
        choices=("lines", "json"),
        default="lines",
        help="one case per line (the case index is the line number, and "
        "the count goes to standard error), or one JSON object",
    )

    worker = sub.add_parser("worker", help="one case (one fresh process)")
    worker.add_argument("--out", type=Path, required=True)
    worker.add_argument(
        "--case", required=True, help="the case's line of `cases`"
    )
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
    verify.add_argument(
        "--rounding-factor",
        type=float,
        help="multiples of eps times the condition number of a run's GP "
        "that the recomputation gate allows, relative to the largest "
        "stored value, on top of its absolute 1e-8 (default 100; 0 keeps "
        "the absolute gate alone, which a recomputation on the "
        "generating machine meets exactly)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    # The driver reads the case list in bash, which takes a carriage
    # return for part of a word.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(newline="\n")
        except AttributeError:
            pass
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
    # The sequential sweep below is the workstation supervisor: the
    # campaign lock, `status.json` and the idle-sleep request are its own,
    # and no other sub-command takes them. Process-scoped request: permit
    # display sleep, prevent idle system sleep while a long sweep works.
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
