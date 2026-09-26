"""Matched comparison of the two S-VBMC implementations on pooled runs.

The integrated ``pyvbmc.svbmc.SVBMC`` and the original standalone ``svbmc``
0.1.1 stack the same subsets of the same finished VBMC runs, cell by cell,
and every cell records the optimized weights, every ELBO variant, the
entropy, the seconds spent, and the quality of 100 000 draws from the
stacked posterior against the target's truth. The design, the conditions,
the grid and the acceptance criteria are
``dev/plans/svbmc-benchmark-campaign.md``, sections "Stacking comparison",
"Metrics" and "Acceptance criteria for the comparison"; the pools come from
``svbmc_pool_run.py``.

Every ELBO an arm reports is scored by its bias against ``elbo_mc``, the
Monte Carlo ELBO of the posterior that arm's fit produced::

    elbo_mc = e_log_joint_mc + entropy_ref
    bias_<variant> = elbo_<variant> - elbo_mc
    kl_gap = ln Z - elbo_mc

``e_log_joint_mc`` is the mean of the target's noiseless log density over
``N_LOG_JOINT`` of the stack's own draws, and ``entropy_ref`` the entropy
of the stacked mixture at that arm's final weights, estimated by this
script rather than taken from the arm: one estimator for both arms, the
integrated class's ``SVBMC.stacked_entropy`` on the cell's posteriors,
``N_ENTROPY_REF`` draws per component in ``N_ENTROPY_BATCHES`` independent
batches, from a generator seeded by the cell. An arm's own reported
entropy is estimated at the weights that arm selected, from its own draws
(the optimizer's 20 per component for the original, a fresh 100 for the
integrated class), so a reference built on it would move with the arm and
credit whichever entropy estimate came out high; the biases would then not
be comparable between the arms. Both terms of ``elbo_mc`` carry their Monte
Carlo standard deviation, and ``kl_gap`` measures how far the stacked
posterior itself is from the target, in evidence units. The error against
``ln Z``, ``elbo_err_<variant>``, stays in the outputs as a descriptive
column: it adds the estimator's bias to that gap and so cannot rank
estimators.

Integrated-arm cells record the additive ``shrunk_two_level`` estimate when
its numerical calculation is available, plus ``shrinkage_available`` and the
nullable ``shrinkage_noise_share`` diagnostic. Summaries report its bias and
the numbers of contributing, unavailable and older unrecorded cells. Its
bootstrap uses an independent deterministic stream, so adding it does not
move historical intervals or paired checks.

The comparison runs in one of two ways, which give the same cells and the
same summaries but for the seconds they measure. On one machine, the flags
below without a subcommand run every cell in one process. On a Slurm
cluster, the subcommands meet the campaign contract of
``dev/plans/slurm-benchmark-support.md`` (``campaign_contract.py``), which
the driver under ``dev/scripts/hpc/`` runs as array tasks::

    python -u dev/scripts/svbmc_pool_stack.py \\
        --pool DIR --out DIR [--conditions L1,L2] [--M 2,4,8,16] \\
        [--repetitions 20,20,20,10] [--arms both|integrated|A1,A2,...] \\
        [--seed 0] [--max-steps 500] [--gpyreg-source DIR] [--overwrite]
    python -u dev/scripts/svbmc_pool_stack.py \\
        --fixtures upstream_Ring --out DIR --M 2,3 --repetitions 2,2 \\
        --max-steps 3
    python dev/scripts/svbmc_pool_stack.py --summarize-only --out DIR \\
        [--from-results R1.json --from-results R2.json]

    python dev/scripts/svbmc_pool_stack.py prepare --out DIR \\
        (--pool DIR ... | --fixtures G1,G2) [--conditions L1,L2] \\
        [--M ...] [--repetitions ...] [--arms ...] [--seed 0] \\
        [--max-steps 500] [--split-from 16] [--allow-dirty]
    python dev/scripts/svbmc_pool_stack.py cases --out DIR [--subset NAME]
    python -u dev/scripts/svbmc_pool_stack.py worker --out DIR --case LINE
    python dev/scripts/svbmc_pool_stack.py verify --out DIR
    python -u dev/scripts/svbmc_pool_stack.py assemble --out DIR

What every invocation that runs a cell needs from its environment: Torch,
importable in the process (the campaign environment holds CPU Torch
2.14.0; on the developer's machine it is the overlay ``<BASELINE_DIR>/deps``
on ``PYTHONPATH``); ``BASELINE_DIR``, for a run of the original arm; and
the gpyreg checkout both arms import, ``PYVBMC_GPYREG_SOURCE`` (or, for the
single-process run, ``--gpyreg-source`` or the pool manifest's path).

A condition's filtered pool is the runs its pool directory's
``selection.json`` names, written by ``svbmc_pool_run.py select``, which
is the campaign's stopping rule applied to the generated runs; in the
single-process run, a directory without one contributes every run whose
completion record says it passed the filters, while a campaign's
``prepare`` requires the selection and checks it (below). The line
printed for each condition, ``sources.json`` and the campaign manifest say
which of the two it was. A run's files are
``<pool>/<tag>.npz`` and ``<pool>/<tag>.json``, whatever the tag holds: a
flat ``<label>_seed<seed>`` in the September pools,
``<label>/<label>_seed<seed>`` in the pools of the campaign contract. The
SHA-256 of both files is taken when the pools are read, and every process
that rebuilds a run's posterior checks it first.

A cell is one subset of ``M`` runs of a condition's filtered pool. The
subsets are disjoint within one ``M``: repetition ``r`` takes the ``r``-th
block of ``M`` runs of a permutation of the condition's runs drawn by
``np.random.default_rng([seed, condition_index, M])``, so that a condition
whose pool holds ``n`` runs gives at most ``n // M`` repetitions at ``M``
(the cells beyond are listed as skipped), and the permutations of
different ``M`` are independent. ``condition_index`` is the condition's
place among every label the pools hold, so that the subsets do not depend
on which conditions an invocation selects. ``np.random.SeedSequence`` on
``[seed, condition_index, M, repetition]`` yields the cell's seed, and
spawning that sequence yields one seed per entry of the subset, with which
both arms rebuild the cell's posteriors: the original draws each run's
block of samples from that run's own generator, which one shared seed
would couple. Both arms receive the subset in the same order and never run
at the same time; the integrated arm runs first on the even repetitions
and the original on the odd ones.

``--arms`` names the arms of every ``M``: ``both``, ``integrated`` (the
integrated class alone, for the larger-``M`` regime where the original's
cost, quadratic in ``M`` and two to four times the integrated arm's, is not
worth paying), or one of the two per ``M``. A cell of the integrated arm
alone carries no paired quantity, and the summaries carry that arm's
medians, biases and headline-bias growth there. The integrated arm runs in
the process that runs the cell, seeded through ``SVBMC(seed=cell_seed)``.
The original arm runs in one long-lived subprocess, seeded by
``np.random.seed(cell_seed)`` immediately before construction, because it
draws its entropy samples from NumPy's global legacy stream and its
posterior samples from the input posteriors' own generators. That
subprocess alone has the baseline's ``src`` directory on its import path;
the process that starts it refuses to have it on its own, so that no
``import svbmc`` there can reach the pinned upstream package.

``BASELINE_DIR`` names the original S-VBMC checkout, or a directory that
holds it as its one subdirectory with ``src/svbmc`` (the layout that
``dev/experiments/svbmc_pool/baseline_environment.json`` describes, with
the Torch overlay ``deps`` beside the checkout ``source``). A process that
runs the original arm verifies the checkout by its content, wherever it
lies: the recorded commit, a clean tree, the recorded SHA-256 of the
committed content of every file of ``src/svbmc`` (with CRLF read as LF,
since Git on Windows may convert line endings on checkout) and no other
file there, and the recorded Torch version in the process. A run of the integrated arm alone neither needs nor verifies
it. Both arms import gpyreg from one checkout, whose commit must be the one
every pool's manifest names (as ``identity.source.gpyreg_commit`` in the
September pools, as the gpyreg tree of the contract's identity in the
campaign pools), with a clean tree.

``--fixtures`` presents the shipped S-VBMC posterior fixtures
(``pyvbmc/testing/svbmc/fixtures/``) as pools, one condition per fixture
group, scored against the ported target the group's runs used. It needs no
pool directory and is what ``test_svbmc_pool_stack.py`` exercises.

The single-process run writes under ``--out``: ``cells.jsonl`` (one line
per finished cell, written as the sweep goes), ``results.json`` (every
cell, the single-run rows and the settings), ``summary.json`` and
``summary.md`` (per condition and ``M``: medians with 10 000-resample
bootstrap 95 % intervals of the biases, the KL gap, the metrics, the
maximum weight difference and the runtime ratio, the paired differences
integrated minus original with exact signed-rank tests on the metrics,
and criterion 3's two gates, ``headline_bias_not_worse`` per ``M`` and
``headline_bias_growth`` per condition), ``sources.json`` (both arms'
commits, working-tree state, import paths, versions and thread settings,
the baseline's verification, the pools' or fixtures' identities and this
file's SHA-256) and ``original_arm.log`` (everything the original arm's
subprocess wrote to its standard streams). ``--summarize-only`` rebuilds
``summary.json`` and ``summary.md`` from a finished ``results.json``
without running a cell. It describes that comparison by the settings the
file records — the draw counts, the entropy reference, the bootstrap and
the stacking call — and falls back to this module's constants only for a
field a results file written before it existed does not carry, so a
rebuilt summary never attributes today's constants to an older run.

The campaign. ``prepare`` reads the pools and writes ``manifest.json``:
the pools with their selections and every selected run's hashes, the
``M`` values, repetition counts and arms, the seed, every planned cell
with its subset and seeds, the baseline's record and verification, the
source identity (the harness checkout, gpyreg and, when a cell runs the
original arm, the baseline checkout; the harness modules, the targets
module and its data, the baseline record; the versions of Python, NumPy,
SciPy, cma and Torch), the operator settings, the environment's
``pip freeze``, the finishing step and the tracked copies
(``TRACKED_COPIES``: the assembled outputs, which
``campaign_contract.py redact`` writes for the repository). Its defaults
are the release gate's grid (``RELEASE_M``, ``RELEASE_REPETITIONS``,
``RELEASE_ARMS``). It stacks a pool only once the pool has verified and
been selected after that verification: ``svbmc_pool_run.py``'s
``stackable_selection`` requires the pool's ``verification.json`` to have
passed, its ``selection.json`` to record the SHA-256 of the manifest and of
that report, and every selected condition to be the stopping rule applied
to the report's cases, so that no seed before a selected run is left
unfinished; the manifest records each pool's report by its SHA-256. It
refuses a dirty gpyreg checkout, and a dirty harness checkout unless
``--allow-dirty``, which the manifest records. A task, one line of
``cases``, computes every repetition of one condition and ``M`` below
``--split-from`` (``<condition>/M<M>``) and one cell at ``M`` from it on
(``<condition>/M<M>_r<r>``), writing one file per cell,
``<condition>/M<M>_r<r>.cell.json``; it runs ``warm_up`` before its first
timed cell and both arms of every two-arm cell, so that a cell's runtime
ratio is measured on one node. ``worker`` runs one task under the claim
and refusals of the contract; the identity it compares leaves the baseline
out for a task of the integrated arm alone, and a line that is not a case
of the campaign is refused with exit 64, touching nothing. The error of a
task whose original arm's subprocess died gives the subprocess's exit
code, and the signal that killed it where one did (SIGKILL, exit code -9,
for an out-of-memory kill). A task's completion record adds the original
arm's import paths, the baseline's verification, the warm-up's seconds
and the peak resident memory of the task's process and of the original
arm's subprocess. ``verify`` re-checks every record and
cell file against the manifest's plan and reconciles the allocation.
``assemble``, the finishing step, runs on a campaign whose every task
verifies and writes the outputs the single-process run writes, but for
``cells.jsonl`` and ``original_arm.log``: ``results.json`` from the cell
files in the plan's order, the single-run rows from the pool records the
manifest holds, the bootstrap drawn in that order, and ``sources.json``
with the tasks' hosts and memory. It never merges a partial campaign.
``cases --subset`` takes ``M<M>`` (the tasks at one ``M``), ``both`` (the
tasks that run the original arm) and ``integrated`` (those that do not).
"""

import argparse
import contextlib
import copy
import inspect
import json
import logging
import os
import platform
import signal
import subprocess
import sys
import time
import traceback
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

# The pool generator's module body pins BLAS to one thread and selects the
# non-interactive matplotlib backend, both of which must happen before
# NumPy is imported, so its import stays above the one below; the contract
# module imports only the standard library.
import campaign_contract as contract  # noqa: E402
from svbmc_pool_run import (  # noqa: E402
    EXIT_USAGE,
    THREAD_KEYS,
    activate_gpyreg,
    dirty_trees,
    identity,
    stackable_selection,
)

# isort: split
import numpy as np  # noqa: E402

BASELINE_RECORD = (
    ROOT / "dev" / "experiments" / "svbmc_pool" / "baseline_environment.json"
)
#: The stacking call both arms make; ``n_samples_final`` is integrated-only.
N_SAMPLES = 20
LEARNING_RATE = 0.1
VERSION = "all-weights"
N_SAMPLES_FINAL = 100
#: Draws scored per cell, and how many of them the Monte Carlo expected log
#: joint of the plan's ``elbo_mc`` averages over.
N_DRAWS = 100_000
N_LOG_JOINT = 10_000
#: The entropy reference of ``elbo_mc``: draws per component, split into
#: independent batches whose spread is the estimate's standard error. The
#: mean of the batches is the estimate at the full draw count, because the
#: estimator averages over each component's draws and the batches are
#: equal and exhaust the total; batching bounds the peak memory of the
#: draw matrix, which grows as ``K_total ** 2`` times the draws per
#: component, and more batches buy degrees of freedom for the standard
#: error at the same total cost.
N_ENTROPY_REF = 200
N_ENTROPY_BATCHES = 8
N_ENTROPY_PER_BATCH = N_ENTROPY_REF // N_ENTROPY_BATCHES
assert N_ENTROPY_PER_BATCH * N_ENTROPY_BATCHES == N_ENTROPY_REF, (
    "the batches must be equal and exhaust the draws, or their mean is "
    "not the estimate at N_ENTROPY_REF draws per component"
)
#: Each arm's entropy reference is drawn from a generator of its own,
#: ``default_rng([cell_seed, 11])``, so that the two arms' references come
#: from the same component draws and differ only through the weights.
ENTROPY_REF_STREAM = 11
#: Criterion 3 of the plan: the bound on how much the median bias of the
#: integrated headline may grow from the smallest ``M`` to the largest.
GROWTH_BOUND = 0.5
BOOTSTRAP_RESAMPLES = 10_000
#: Independent summary stream for the additive shrinkage estimator. Keeping
#: it separate preserves every historical bootstrap draw and paired check.
SHRINKAGE_BOOTSTRAP_STREAM = 31
#: Criterion 2 of the plan: the metrics tested for equivalence, and the
#: level of the Holm correction over all condition-and-``M`` cells.
TEST_METRICS = ("mmtv", "gskl")
ALPHA = 0.05
#: The constants ``results.json["settings"]`` records, and so what a
#: summary describes a comparison by. The value is the fallback for a
#: results file written before the field existed; the summary of an
#: earlier comparison must report the settings that comparison ran under,
#: not whatever this module's constants happen to be today.
SETTING_DEFAULTS = {
    "n_samples": N_SAMPLES,
    "lr": LEARNING_RATE,
    "version": VERSION,
    "n_samples_final": N_SAMPLES_FINAL,
    "n_draws": N_DRAWS,
    "n_log_joint": N_LOG_JOINT,
    "n_entropy_ref": N_ENTROPY_REF,
    "n_entropy_batches": N_ENTROPY_BATCHES,
    "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    "arms": ["integrated", "original"],
}
#: The ported target each fixture group's runs were fitted to, so that
#: ``--fixtures`` scores the stacked posteriors against a truth.
FIXTURE_PROBLEMS = {
    "upstream_GMM": "gmm_D2_svbmc",
    "upstream_GMM_noisy": "gmm_D2_noise3_svbmc",
    "upstream_Ring": "ring_D2_noise3_svbmc",
}
ARMS = ("integrated", "original")
#: What ``--arms`` names for one ``M``: the arms its cells run.
ARM_SETS = {"both": ARMS, "integrated": ("integrated",)}
#: The variant each arm reports as its ELBO; criterion 3 compares the
#: biases of these two.
HEADLINE_VARIANT = {"integrated": "headline", "original": "estimated"}
#: The suffixes of one pool run's two files.
RUN_SUFFIXES = (".npz", ".json")

#: The name a campaign manifest of this harness carries.
CAMPAIGN = "svbmc_pool_stack"
#: The release gate's grid (``dev/plans/slurm-benchmark-support.md``,
#: decision 6): both arms at ``M`` = 2, 4, 8 and 16, the integrated arm
#: alone at 3, 5 and 32, 20 repetitions up to ``M`` = 8 and 10 beyond.
RELEASE_M = (2, 3, 4, 5, 8, 16, 32)
RELEASE_REPETITIONS = (20, 20, 20, 20, 20, 10, 10)
RELEASE_ARMS = (
    "both",
    "integrated",
    "both",
    "integrated",
    "both",
    "both",
    "integrated",
)
#: The ``M`` from which a campaign task holds one cell rather than every
#: repetition of its condition and ``M``.
SPLIT_FROM = 16
CELL_SUFFIX = ".cell.json"
#: What of a finished campaign enters the repository, beside its manifest
#: and verification report (``campaign_contract.tracked_copies``): the
#: assembled outputs. ``results.json`` holds every cell, so the cell files
#: stay in the archive.
TRACKED_COPIES = {
    "files": ["results.json", "summary.json", "summary.md", "sources.json"]
}
#: The files, relative to the harness checkout, whose SHA-256 a campaign's
#: source identity holds: this harness and the modules it runs, the targets
#: module and the data it reads, and the baseline's record.
HARNESS_FILES = (
    "dev/scripts/svbmc_pool_stack.py",
    "dev/scripts/campaign_contract.py",
    "dev/scripts/svbmc_pool_run.py",
    "dev/scripts/svbmc_pool_io.py",
    "dev/scripts/benchmark_targets.py",
    "dev/scripts/profile_run.py",
    "dev/scripts/golden_trace.py",
    "dev/scripts/data",
    "dev/experiments/svbmc_pool/baseline_environment.json",
)
#: The top-level names of a campaign directory that belong to the contract,
#: which no condition may take for its subdirectory.
RESERVED_NAMES = frozenset({"records", "claims", "slurm", "subsets", "tmp"})
#: The keys of a planned cell that every cell file must repeat.
PLAN_KEYS = (
    "condition",
    "condition_index",
    "M",
    "repetition",
    "indices",
    "entries",
    "seeds",
    "cell_seed",
    "entry_seeds",
    "first_arm",
)


# --------------------------------------------------------------------------
# Baseline environment
# --------------------------------------------------------------------------


def baseline_checkout(directory=None):
    """The original S-VBMC checkout that ``BASELINE_DIR`` names.

    ``directory`` (``BASELINE_DIR`` by default) is the checkout itself,
    holding ``src/svbmc``, or a directory whose one subdirectory holding
    ``src/svbmc`` is the checkout. Raises
    :class:`campaign_contract.IdentityError` otherwise, since a process
    that runs the original arm cannot establish its identity without it.
    """
    if directory is None:
        directory = os.environ.get("BASELINE_DIR")
    if not directory:
        raise contract.IdentityError(
            "BASELINE_DIR is not set; it names the original S-VBMC "
            "checkout, which the original arm imports"
        )
    directory = Path(directory).resolve()
    if (directory / "src" / "svbmc").is_dir():
        return directory
    found = (
        [
            child
            for child in sorted(directory.iterdir())
            if (child / "src" / "svbmc").is_dir()
        ]
        if directory.is_dir()
        else []
    )
    if len(found) != 1:
        raise contract.IdentityError(
            f"BASELINE_DIR={directory} holds no src/svbmc and "
            f"{len(found)} subdirectories that do; it names the original "
            "S-VBMC checkout or the directory that holds it"
        )
    return found[0]


def verify_baseline(record, checkout):
    """Re-verify the original-S-VBMC baseline by its content; return a report.

    ``record`` is ``baseline_environment.json``. The checkout must sit at
    the recorded commit with a clean tree, every file the record lists
    under ``src/svbmc`` must hold its committed content (the SHA-256 of
    ``files_sha256_committed``, taken with CRLF read as LF, since Git on
    Windows may convert the line endings on checkout) and no other file
    may lie there (``__pycache__`` aside), and the Torch this process
    imports must be the recorded version. Where the checkout lies does not
    matter, so that a recreation on another machine verifies. Raises
    :class:`campaign_contract.IdentityError` naming every mismatch: the
    baseline is the comparison's other arm, so a run that cannot identify
    it must not start.
    """
    import hashlib

    import torch

    baseline = record["baseline"]
    checkout = Path(checkout).resolve()
    package = checkout / "src" / "svbmc"
    failures = []
    state, where = contract.tree_state(checkout)
    if state["commit"] != baseline["commit"]:
        failures.append(
            f"commit {state['commit']}, recorded {baseline['commit']}"
        )
    if not state["clean"]:
        failures.append(
            "the checkout has uncommitted or untracked changes: "
            + "; ".join(where["dirty"])
        )
    recorded = baseline["files_sha256_committed"]
    present = sorted(
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    unrecorded = [name for name in present if name not in recorded]
    if unrecorded:
        failures.append(f"files the record does not list: {unrecorded}")
    files = {}
    for name, stored in sorted(recorded.items()):
        path = package / name
        if not path.is_file():
            failures.append(f"{name} is missing")
            continue
        content = path.read_bytes().replace(b"\r\n", b"\n")
        files[name] = hashlib.sha256(content).hexdigest()
        if files[name] != stored["sha256"]:
            failures.append(f"{name} differs from the recorded hash")
    if torch.__version__ != record["torch"]["version"]:
        failures.append(
            f"Torch {torch.__version__}, recorded {record['torch']['version']}"
        )
    if failures:
        raise contract.IdentityError(
            f"the baseline at {checkout} does not verify against "
            f"{BASELINE_RECORD.relative_to(ROOT).as_posix()}: "
            + "; ".join(failures)
        )
    return {
        "record": BASELINE_RECORD.relative_to(ROOT).as_posix(),
        "checkout": str(checkout),
        "commit": state["commit"],
        "clean": True,
        "files_sha256_committed": files,
        "torch_version": torch.__version__,
        "torch_import": str(Path(torch.__file__).resolve()),
        "verified": contract.now(),
    }


def read_baseline_record():
    return contract.read_json(BASELINE_RECORD)


def refuse_upstream_on_path(upstream):
    """This process must not be able to ``import svbmc`` from ``upstream``."""
    entries = [
        Path(p).resolve()
        for p in (
            *sys.path,
            *os.environ.get("PYTHONPATH", "").split(os.pathsep),
        )
        if p
    ]
    if Path(upstream).resolve() in entries:
        raise RuntimeError(
            f"{upstream} is on this process's import path; only the "
            "original arm's subprocess may carry it, so that the integrated "
            "class is the only S-VBMC this process can import"
        )


def import_torch():
    """Import Torch for this process, single-threaded; say where to get it."""
    try:
        import torch
    except ImportError as error:
        raise contract.IdentityError(
            "the integrated class needs Torch: the campaign environment "
            "holds it, and on the developer's machine PYTHONPATH carries "
            "the overlay <BASELINE_DIR>/deps"
        ) from error
    torch.set_num_threads(1)
    logging.getLogger("SVBMC").setLevel(logging.WARNING)
    return torch


# --------------------------------------------------------------------------
# Pools, entries and problems
# --------------------------------------------------------------------------


def pool_gpyreg_commit(manifest):
    """The gpyreg commit a pool manifest names, or None.

    The September pools record it in a flat identity,
    ``identity.source.gpyreg_commit``; the pools of the campaign contract
    record it as the ``gpyreg`` tree of ``campaign_contract.identity``,
    ``identity.source.trees.gpyreg.commit``.
    """
    source = (manifest.get("identity") or {}).get("source") or {}
    if source.get("gpyreg_commit"):
        return source["gpyreg_commit"]
    tree = (source.get("trees") or {}).get("gpyreg") or {}
    return tree.get("commit")


def check_gpyreg_against_pools(source, manifests):
    """Refuse a gpyreg checkout other than the pools' own library.

    ``manifests`` holds ``(path, manifest)`` of every pool read. ``source``
    must hold the ``gpyreg`` package and be a clean checkout at the commit
    every manifest names (:func:`pool_gpyreg_commit`), the library the
    pools' numerics were produced by; anything else raises
    ``RuntimeError`` naming what differs. Returns the commit.
    """
    source = Path(source).resolve()
    if not (source / "gpyreg").is_dir():
        raise RuntimeError(f"no gpyreg package under {source}")
    state, _ = contract.tree_state(source)
    if not state["clean"]:
        raise RuntimeError(
            f"{source} has uncommitted changes; the pool is read against "
            "the pinned library as committed"
        )
    for path, manifest in manifests:
        pinned = pool_gpyreg_commit(manifest)
        if pinned is None:
            raise RuntimeError(
                f"{path} names no gpyreg commit, so the library its pool "
                "was generated by is unknown"
            )
        if pinned != state["commit"]:
            raise RuntimeError(
                f"{source} is at gpyreg commit {state['commit']}, not the "
                f"manifest's {pinned}; the pool was generated by that "
                "commit and is read against it"
            )
    return state["commit"]


def run_files(base):
    """``{suffix: path}`` of the two files of a run or fixture at ``base``.

    ``base`` is the path without suffix, ``<pool>/<tag>`` for a pool run;
    the suffix is appended to the name, which may hold a dot.
    """
    base = Path(base)
    return {s: base.parent / (base.name + s) for s in RUN_SUFFIXES}


def pool_entry(directory, tag, origin="the pool directory"):
    """One filtered run of a pool, from its completion record.

    ``origin`` names the list the tag came from, so that a run whose
    record or artifact the directory does not hold is reported as the
    wrong entry of that list rather than as an absent file. A pool is
    generated once and read many times, and the case that leads here is a
    selection copied without the artifacts it names, or one whose run
    failed after the selection was written. The entry carries the SHA-256
    of the run's two files, which :func:`load_entry` checks.
    """
    import svbmc_pool_io as pool_io

    record_file = pool_io.record_path(directory, tag)
    if not record_file.exists():
        raise RuntimeError(
            f"{origin} names {tag}, which has no completion record "
            f"({record_file}): the case failed or was never generated"
        )
    record = json.loads(record_file.read_text(encoding="utf-8"))
    files = run_files(directory / record["tag"])
    missing = [path.name for path in files.values() if not path.exists()]
    if missing:
        raise RuntimeError(
            f"{origin} names {tag}, whose artifact {directory} does not "
            f"hold: {', '.join(missing)} missing"
        )
    return {
        "kind": "run",
        "name": record["tag"],
        "path": str(directory / record["tag"]),
        "seed": int(record["seed"]),
        "metrics": record["metrics"],
        "passes": bool(record["verdict"]["passes"]),
        "label": record["label"],
        "sha256": {s: contract.sha256_file(p) for s, p in files.items()},
    }


def passing_record_tags(directory, label):
    """The tags of a condition's completion records, in either layout.

    A September pool keeps its records flat,
    ``records/<label>_seed<seed>.complete.json``; a pool of the campaign
    contract keeps them in one subdirectory per condition,
    ``records/<label>/<label>_seed<seed>.complete.json``.
    """
    records = directory / "records"
    tags = []
    for pattern in (
        f"{label}_seed*.complete.json",
        f"{label}/{label}_seed*.complete.json",
    ):
        for path in sorted(records.glob(pattern)):
            relative = path.relative_to(records).as_posix()
            tags.append(relative[: -len(".complete.json")])
    return tags


def pool_conditions(pool_dirs, only=None, gpyreg_source=None):
    """The filtered runs of every condition, ordered by seed.

    One entry per run of the condition's filtered pool, carrying the
    artifact path (without suffix), the SHA-256 of its two files and the
    metrics recorded for that single run, which are the comparison's
    ``M = 1`` rows.

    The pools must have been generated against one gpyreg source. With
    ``gpyreg_source`` given, a local checkout that the caller has already
    checked against every manifest's gpyreg commit, the paths the
    manifests name are not compared, since they belong to the machines
    that generated the pools.

    The filtered pool of a directory that holds a ``selection.json``
    (``svbmc_pool_run.py select``) is the runs that file names: the
    campaign's stopping rule applied once to the whole directory, which is
    what a pool generated as an array job needs and what a sequential
    sweep reproduces. Without one, every run whose record says it passed
    the filters is taken, and the line printed for the condition says
    which of the two it was.

    Returns the selected conditions, the pools' identities, and every
    label the pools allocate (which conditions are selected does not
    change that list, and it is what indexes a condition when its subsets
    are drawn).
    """
    conditions, identities, labels = {}, [], []
    for directory in pool_dirs:
        directory = Path(directory).resolve()
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        selection_path = directory / "selection.json"
        selection = (
            json.loads(selection_path.read_text(encoding="utf-8"))
            if selection_path.exists()
            else None
        )
        selected = {
            entry["label"]: entry
            for entry in (selection["conditions"] if selection else [])
        }
        identity_record = {
            "directory": str(directory),
            "manifest_sha256": contract.sha256_file(manifest_path),
            "identity": manifest["identity"],
            "gpyreg_source": manifest.get("gpyreg_source"),
            "gpyreg_commit": pool_gpyreg_commit(manifest),
            "options": manifest.get("options"),
            "selection": {
                "path": str(selection_path) if selection else None,
                "generated": selection["generated"] if selection else None,
                "conditions": {},
            },
            "selection_sha256": (
                contract.sha256_file(selection_path) if selection else None
            ),
        }
        identities.append(identity_record)
        for allocated in manifest["allocation"]:
            label = allocated["label"]
            labels.append(label)
            if only and label not in only:
                continue
            if label in conditions:
                raise RuntimeError(
                    f"{label} is allocated in more than one pool directory"
                )
            if label in selected:
                origin = "selection.json"
                entries = [
                    pool_entry(directory, run["tag"], str(selection_path))
                    for run in selected[label]["runs"]
                ]
                wrong = [e["name"] for e in entries if not e["passes"]]
                if wrong:
                    raise RuntimeError(
                        f"{selection_path} selects runs that their records "
                        f"say do not pass the filters: {wrong}"
                    )
            else:
                origin = "every passing record"
                entries = []
                for tag in passing_record_tags(directory, label):
                    entry = pool_entry(
                        directory,
                        tag,
                        f"the completion records under {directory}",
                    )
                    if entry["label"] == label and entry["passes"]:
                        entries.append(entry)
            identity_record["selection"]["conditions"][label] = origin
            conditions[label] = sorted(entries, key=lambda e: e["seed"])
    missing = [label for label in (only or ()) if label not in conditions]
    if missing:
        raise RuntimeError(f"no pool holds the conditions {missing}")
    sources = {entry["gpyreg_source"] for entry in identities}
    if gpyreg_source is None and len(sources) > 1:
        raise RuntimeError(
            f"the pools were generated against different gpyreg sources: "
            f"{sorted(sources, key=str)}"
        )
    return conditions, identities, sorted(set(labels))


def fixture_conditions(groups):
    """Fixture groups presented as pools: one condition per group.

    The shipped posteriors carry no artifact path, so an entry names the
    fixture instead; the single-run metrics are computed on demand. A
    fixture belongs to the group its sidecar records, not to every group
    whose name is a prefix of its own. Only the posteriors that pass the
    pool's filters are kept, as in a pool.
    """
    from svbmc_pool_io import filter_verdict

    from pyvbmc.testing.svbmc._fixtures import (
        FIXTURES_DIR,
        _group_of,
        fixture_names,
        load_vp,
    )

    conditions = {}
    for group in groups:
        if group not in FIXTURE_PROBLEMS:
            raise RuntimeError(
                f"no ported target is recorded for the fixture group "
                f"{group!r}; known groups: {sorted(FIXTURE_PROBLEMS)}"
            )
        entries = []
        for index, name in enumerate(sorted(fixture_names())):
            if _group_of(name, FIXTURES_DIR) != group:
                continue
            vp, _ = load_vp(name, rng=0)
            if filter_verdict(vp)["passes"]:
                entries.append(
                    {
                        "kind": "fixture",
                        "name": name,
                        "path": None,
                        "seed": index,
                        "metrics": None,
                        "sha256": {
                            s: contract.sha256_file(p)
                            for s, p in run_files(FIXTURES_DIR / name).items()
                        },
                    }
                )
        if not entries:
            raise RuntimeError(f"no fixture of group {group!r} passes")
        conditions[group] = entries
    return conditions


def fixture_sources(conditions):
    """The files behind every ``--fixtures`` condition, with their hashes."""
    from pyvbmc.testing.svbmc._fixtures import FIXTURES_DIR

    return [
        {
            "condition": condition,
            "directory": str(FIXTURES_DIR),
            "fixtures": [
                {"name": entry["name"], "sha256": entry["sha256"]}
                for entry in entries
            ],
        }
        for condition, entries in conditions.items()
    ]


def entry_base(entry):
    """The path without suffix of one entry's two files."""
    if entry["kind"] == "fixture":
        from pyvbmc.testing.svbmc._fixtures import FIXTURES_DIR

        return FIXTURES_DIR / entry["name"]
    return Path(entry["path"])


def check_entry(entry):
    """Refuse an entry whose files differ from the hashes it carries.

    An entry read from a pool carries the SHA-256 of both files, taken
    when the pool was read (for a campaign, at ``prepare``), so that a
    posterior is never rebuilt from a file that changed since.
    """
    expected = entry.get("sha256")
    if not expected:
        return
    for suffix, path in run_files(entry_base(entry)).items():
        if contract.sha256_file(path) != expected[suffix]:
            raise RuntimeError(
                f"{path} differs from the SHA-256 recorded when its pool "
                "was read"
            )


def load_posterior(base, rng=None):
    """Rebuild the returned posterior of one stored pool run.

    ``base`` is the run's path without suffix. The posterior and its
    transformer are rebuilt through the codec's public constructors, as
    ``svbmc_pool_io.load_run`` rebuilds them, without the GP, the logger
    and the state, which stacking does not read. ``rng`` (an integer seed
    or a generator) becomes the posterior's generator, the codec's fixed
    seed without it.
    """
    from pyvbmc.testing.oracles._state import (
        build_transformer,
        build_vp,
        load_snapshot,
    )

    snapshot = load_snapshot(base)
    return build_vp(
        snapshot["vp"],
        build_transformer(snapshot["pt"]),
        rng=None if rng is None else np.random.default_rng(rng),
    )


def load_entry(entry, rng):
    """Rebuild one pool entry's posterior with its own generator."""
    check_entry(entry)
    if entry["kind"] == "fixture":
        from pyvbmc.testing.svbmc._fixtures import load_vp

        return load_vp(entry["name"], rng=rng)[0]
    return load_posterior(entry["path"], rng=rng)


class Problems:
    """The target of every condition, with its exact reference draws.

    A condition's label is a suite configuration; a fixture group maps to
    the ported target its runs were fitted to. The 100 000 reference draws
    that ``sample_metrics`` scores against are drawn once per problem, so
    that every cell and both arms are compared with the same reference;
    they are drawn with the generator ``sample_metrics`` uses when it is
    given no reference of its own.
    """

    def __init__(self):
        self._problems, self._references = {}, {}

    def get(self, condition, kind):
        from benchmark_targets import find_config

        if condition not in self._problems:
            label = (
                FIXTURE_PROBLEMS[condition] if kind == "fixture" else condition
            )
            self._problems[condition] = find_config(label).make(seed=0)
        return self._problems[condition]

    def reference(self, condition, kind):
        from benchmark_targets import DIAG_TV_SAMPLES, sample_metrics

        problem = self.get(condition, kind)
        if condition not in self._references:
            seed = inspect.signature(sample_metrics).parameters["seed"].default
            self._references[condition] = (
                None
                if problem.sampler is None
                else problem.sampler(
                    DIAG_TV_SAMPLES, np.random.default_rng(seed)
                )
            )
        return self._references[condition]


def single_run_rows(condition, kind, entries, problems):
    """The ``M = 1`` rows of one condition: every filtered run's metrics."""
    from benchmark_targets import metrics

    problem = problems.get(condition, kind)
    rows = []
    for entry in entries:
        measured = entry["metrics"]
        if measured is None:
            vp = load_entry(entry, rng=0)
            measured = metrics(problem, vp, float(vp.stats["elbo"]))
        row = {
            key: float(measured[key])
            for key in ("elbo_err", "gskl", "mmtv")
            if key in measured
        }
        row["gskl_normalized"] = row["gskl"] / problem.D
        row["name"] = entry["name"]
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# The two arms
# --------------------------------------------------------------------------


def log_joint_subsample(n, cell_seed, arm):
    """Which draws the Monte Carlo expected log joint averages over.

    Both implementations return their draws grouped by input posterior, and
    the original concatenates one block per run without shuffling, so the
    first ``N_LOG_JOINT`` rows would measure the stack's first runs rather
    than the stack. The subsample is drawn without replacement from a
    generator derived from the cell's seed and the arm's position, so the
    same procedure runs on both arms and either is reproducible from the
    cell's record.
    """
    return np.random.default_rng([int(cell_seed), ARMS.index(arm)]).choice(
        n, min(N_LOG_JOINT, n), replace=False
    )


def stacked_outcome(problem, reference, samples, elbos, seed, arm):
    """Everything a fitted stack is scored on, for either arm.

    The expected log joint of ``elbo_mc`` is the mean of the target's
    noiseless log density over the subsampled draws, with the standard
    error of that mean. The entropy term of ``elbo_mc`` is deliberately
    not the arm's own, so it is not added here: the controller estimates
    it for both arms with one estimator once a cell's two fits are in, and
    :func:`add_reference` then completes this record with ``entropy_ref``,
    ``elbo_mc``, the biases and the KL gap.
    """
    from benchmark_targets import sample_metrics
    from profile_run import jsonable

    started = time.perf_counter()
    samples = np.asarray(samples)
    index = log_joint_subsample(len(samples), seed, arm)
    values = np.asarray(problem.log_density_vec(samples[index]), dtype=float)
    measured = sample_metrics(problem, samples, elbos, reference=reference)
    return {
        "elbos": {k: float(v) for k, v in elbos.items()},
        "e_log_joint_mc": float(np.mean(values)),
        "e_log_joint_mc_sd": float(
            np.std(values, ddof=1) / np.sqrt(values.size)
        ),
        "n_log_joint": int(index.size),
        "metrics": jsonable(measured),
        "metrics_seconds": time.perf_counter() - started,
    }


def integrated_elbo_report(stacked):
    """Numeric ELBOs and nullable shrinkage diagnostics for one stack."""
    details = stacked.elbo_details
    elbos = {
        "headline": float(stacked.elbo),
        "raw": float(details["raw"]),
        "capped_I_median": float(details["capped_I_median"]),
        "capped_E_median": float(details["capped_E_median"]),
        "naive": float(details["naive"]),
    }
    available = details["shrunk_two_level"] is not None
    if available:
        elbos["shrunk_two_level"] = float(details["shrunk_two_level"])
    share = details["shrinkage_noise_share"]
    return elbos, bool(available), None if share is None else float(share)


def fit_integrated(entries, seeds, cell_seed, max_steps, problem, reference):
    """One cell of the integrated arm, in this process.

    Returns the cell's record and the fitted object, which
    :func:`add_reference` then uses as the cell's entropy estimator: it
    holds the cell's posteriors, filtered as both arms filter them, in the
    order both arms weight them.
    """
    from pyvbmc.svbmc import SVBMC

    started = time.perf_counter()
    vps = [load_entry(e, rng=s) for e, s in zip(entries, seeds)]
    stacked = SVBMC(vps, seed=cell_seed)
    construction_seconds = time.perf_counter() - started
    started = time.perf_counter()
    stacked.optimize(
        n_samples=N_SAMPLES,
        lr=LEARNING_RATE,
        max_steps=max_steps,
        version=VERSION,
        n_samples_final=N_SAMPLES_FINAL,
    )
    optimize_seconds = time.perf_counter() - started
    started = time.perf_counter()
    samples = stacked.sample(N_DRAWS)
    sample_seconds = time.perf_counter() - started
    details = stacked.elbo_details
    elbos, shrinkage_available, shrinkage_noise_share = integrated_elbo_report(
        stacked
    )
    outcome = stacked_outcome(
        problem,
        reference,
        samples,
        elbos,
        cell_seed,
        "integrated",
    )
    outcome.update(
        {
            "arm": "integrated",
            "headline": "headline",
            "M_used": int(stacked.M),
            "K": [int(k) for k in stacked.K],
            "w": np.ravel(stacked.w).tolist(),
            "entry_seeds": [int(s) for s in seeds],
            "entropy": float(stacked.entropy),
            "elbo_sd": float(stacked.elbo_sd),
            "cap_amount": float(details["cap_amount"]),
            "entropy_sd": float(details["entropy_sd"]),
            "gp_sd": float(details["gp_sd"]),
            "noisy": bool(details["noisy"]),
            "noise_status_source": list(details["noise_status_source"]),
            "shrinkage_available": shrinkage_available,
            "shrinkage_noise_share": shrinkage_noise_share,
            "construction_seconds": construction_seconds,
            "optimize_seconds": optimize_seconds,
            "sample_seconds": sample_seconds,
        }
    )
    return outcome, stacked


def fit_original(entries, seeds, cell_seed, max_steps, problem, reference):
    """One cell of the original arm; runs inside the worker process."""
    import svbmc

    started = time.perf_counter()
    vps = [load_entry(e, rng=s) for e, s in zip(entries, seeds)]
    # The original draws its entropy samples from NumPy's global legacy
    # stream, so the cell's seed is set immediately before construction.
    np.random.seed(cell_seed)
    stacked = svbmc.SVBMC(vps, s_max=float(np.sqrt(5)), M_min=2 / 3)
    construction_seconds = time.perf_counter() - started
    started = time.perf_counter()
    stacked.optimize(
        n_samples=N_SAMPLES,
        lr=LEARNING_RATE,
        max_steps=max_steps,
        version=VERSION,
    )
    optimize_seconds = time.perf_counter() - started
    started = time.perf_counter()
    samples = stacked.sample(N_DRAWS)
    sample_seconds = time.perf_counter() - started
    outcome = stacked_outcome(
        problem,
        reference,
        samples,
        {k: float(v) for k, v in stacked.elbo.items()},
        cell_seed,
        "original",
    )
    outcome.update(
        {
            "arm": "original",
            "headline": "estimated",
            "M_used": int(stacked.M),
            "K": [int(k) for k in stacked.K],
            "w": np.ravel(stacked.w).tolist(),
            "entry_seeds": [int(s) for s in seeds],
            "entropy": float(stacked.entropy),
            "construction_seconds": construction_seconds,
            "optimize_seconds": optimize_seconds,
            "sample_seconds": sample_seconds,
        }
    )
    return outcome


# --------------------------------------------------------------------------
# The Monte Carlo ELBO reference
# --------------------------------------------------------------------------


def entropy_reference(stacked, w, rng):
    """Entropy of the stacked mixture at given weights, and its MC error.

    ``stacked`` is the cell's integrated object, used here only as an
    estimator of the mixture its posteriors define: ``stacked_entropy``
    draws from every component through the object's generator and averages
    the mixture's log density over those draws, so the estimate depends on
    the weights and the posteriors and not on which arm produced the
    weights. The caller's generator serves for the length of the call, so
    the draws come from the cell's seed rather than from wherever an arm
    left the object's own stream.

    ``N_ENTROPY_BATCHES`` independent batches of ``N_ENTROPY_PER_BATCH``
    draws per component are averaged. The estimator averages over each
    component's draws, so that mean is the estimate at ``N_ENTROPY_REF``
    draws per component, and the spread of the batches gives it a standard
    error without a second estimator.
    """
    import torch

    weights = np.asarray(w, dtype=np.float64).ravel()
    K_total = int(np.sum(stacked.K))
    if weights.size != K_total:
        raise RuntimeError(
            f"the weights hold {weights.size} components but the cell's "
            f"stacked posterior holds {K_total}: the arms and the entropy "
            "reference did not retain the same runs"
        )
    tensor = torch.as_tensor(weights.reshape(1, -1), dtype=torch.float64)
    estimates = []
    previous, stacked.rng = stacked.rng, rng
    try:
        with torch.no_grad():
            for _ in range(N_ENTROPY_BATCHES):
                H, _corrections = stacked.stacked_entropy(
                    tensor, N_ENTROPY_PER_BATCH
                )
                estimates.append(float(H.item()))
    finally:
        stacked.rng = previous
    return {
        "entropy_ref": float(np.mean(estimates)),
        "entropy_ref_sd": float(
            np.std(estimates, ddof=1) / np.sqrt(N_ENTROPY_BATCHES)
        ),
        "entropy_ref_batches": estimates,
    }


def add_reference(row, stacked, problem, cell_seed):
    """Complete a cell's records with the Monte Carlo ELBO reference.

    Each arm is scored against the ELBO of the posterior it produced:
    its own expected log joint (already recorded) plus the entropy of the
    stacked mixture at the weights it returned, estimated here for both
    arms by one estimator. Each arm's reference starts from its own
    generator on the same seed, so both are computed from the same
    component draws and differ only through the weights, and the
    alternation of which arm is fitted first leaves the reference
    unchanged.
    """
    for arm in row["arms"]:
        outcome = row["arms"][arm]
        rng = np.random.default_rng([int(cell_seed), ENTROPY_REF_STREAM])
        outcome.update(entropy_reference(stacked, outcome["w"], rng))
        elbo_mc = float(outcome["e_log_joint_mc"] + outcome["entropy_ref"])
        outcome["elbo_mc"] = elbo_mc
        outcome["elbo_mc_sd"] = float(
            np.hypot(outcome["e_log_joint_mc_sd"], outcome["entropy_ref_sd"])
        )
        outcome["elbos"]["mc"] = elbo_mc
        ln_Z = problem.ln_Z
        outcome["kl_gap"] = (
            float("nan") if ln_Z is None else float(ln_Z - elbo_mc)
        )
        outcome["metrics"]["elbo_err_mc"] = (
            float("nan") if ln_Z is None else float(abs(elbo_mc - ln_Z))
        )
        outcome["bias"] = {
            variant: float(value - elbo_mc)
            for variant, value in outcome["elbos"].items()
            if variant != "mc"
        }
    return row


# --------------------------------------------------------------------------
# The original arm's subprocess
# --------------------------------------------------------------------------


def peak_rss(children=False):
    """The peak resident memory in bytes of this process, or of its children.

    On Linux ``getrusage``: ``children`` gives the largest of the
    terminated children this process has waited for. Elsewhere the
    process's own peak working set through ``psutil`` where it is
    installed, and None for the children, which :class:`OriginalArm`
    measures itself before its subprocess exits.
    """
    if sys.platform.startswith("linux"):
        import resource

        who = resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF
        return int(resource.getrusage(who).ru_maxrss) * 1024  # KiB on Linux
    if children:
        return None
    return process_peak_rss(os.getpid())


def process_peak_rss(pid, descendants=False):
    """A running process's peak working set through ``psutil``, or None.

    With ``descendants``, the largest over the process and every process it
    started: a virtual environment's ``python.exe`` on Windows is a
    launcher whose child is the interpreter.
    """
    try:
        import psutil

        process = psutil.Process(pid)
        processes = [process]
        if descendants:
            processes += process.children(recursive=True)
    except Exception:  # psutil absent, or the process gone
        return None
    peaks = []
    for each in processes:
        try:
            peak = getattr(each.memory_info(), "peak_wset", None)
        except Exception:  # the process ended meanwhile
            continue
        if peak is not None:
            peaks.append(int(peak))
    return max(peaks) if peaks else None


def campaign_identity(gpyreg_source, checkout=None, host=True):
    """The campaign contract's identity of a process of this harness.

    The trees are the harness checkout, gpyreg and, for a process that
    runs the original arm, the baseline checkout; the files are
    :data:`HARNESS_FILES`; PyVBMC must be imported from the harness
    checkout and gpyreg from its tree. Torch is imported first, so that
    its version is part of the source identity.
    """
    trees = {"harness": ROOT, "gpyreg": Path(gpyreg_source)}
    if checkout is not None:
        trees["baseline"] = Path(checkout)
    return contract.identity(
        trees,
        HARNESS_FILES,
        modules={"pyvbmc": "harness", "gpyreg": "gpyreg"},
        host=host,
    )


def serve_original(gpyreg_source, checkout):
    """Serve stacking requests for the original implementation.

    Requests and replies are one JSON object per line. The original
    implementation prints progress and warnings, so file descriptor 1 is
    pointed at stderr (which the controller captures into a log, or passes
    on to its own) and the protocol keeps a private copy of the real
    stdout. The stop signals are ignored here: the controller receives
    them, and ends this process itself once it has cleaned up.
    """
    for signum in contract.STOP_SIGNALS:
        signal.signal(signum, signal.SIG_IGN)
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr

    activate_gpyreg(gpyreg_source)
    torch = import_torch()
    import svbmc

    def send(payload):
        protocol.write(json.dumps(payload) + "\n")
        protocol.flush()

    source_identity = campaign_identity(gpyreg_source, checkout, host=False)
    contract.module_origin("svbmc", checkout)
    send(
        {
            "ready": True,
            "svbmc_version": svbmc.__version__,
            "svbmc_import": str(Path(svbmc.__file__).resolve()),
            "torch_version": torch.__version__,
            "torch_import": str(Path(torch.__file__).resolve()),
            "torch_threads": torch.get_num_threads(),
            "executable": sys.executable,
            "environment": identity(gpyreg_source),
            "identity": source_identity,
        }
    )

    problems = Problems()
    for line in sys.stdin:
        request = json.loads(line)
        if request["op"] == "quit":
            send({"bye": True})
            break
        try:
            condition, kind = request["condition"], request["kind"]
            send(
                fit_original(
                    request["entries"],
                    [int(s) for s in request["entry_seeds"]],
                    int(request["cell_seed"]),
                    int(request["max_steps"]),
                    problems.get(condition, kind),
                    problems.reference(condition, kind),
                )
            )
        except Exception:  # reported to the controller, which stops
            send({"error": traceback.format_exc()})


class OriginalArm:
    """The original implementation, served by one long-lived subprocess.

    The subprocess's ``PYTHONPATH`` is this process's with the baseline
    checkout's ``src`` appended, so that it finds the same Torch and the
    pinned upstream package. A context manager: leaving the block normally
    asks the subprocess to quit, leaving it on an exception (a stop signal
    among them) kills it at once, since a fit in progress could outlast
    the time Slurm leaves before its SIGKILL. ``peak_rss_bytes`` is the
    subprocess's peak resident memory once it has ended, where the
    platform reports it.
    """

    def __init__(self, gpyreg_source, checkout, log_path=None):
        checkout = Path(checkout)
        environment = dict(os.environ)
        inherited = [
            p for p in environment.get("PYTHONPATH", "").split(os.pathsep) if p
        ]
        environment["PYTHONPATH"] = os.pathsep.join(
            [*inherited, str(checkout / "src")]
        )
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYVBMC_GPYREG_SOURCE"] = str(gpyreg_source)
        environment["MPLBACKEND"] = "Agg"
        environment.update({key: "1" for key in THREAD_KEYS})
        self.pythonpath = environment["PYTHONPATH"]
        self.peak_rss_bytes = None
        self.log = (
            None
            if log_path is None
            else Path(log_path).open("w", encoding="utf-8")
        )
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "--serve-original",
                str(gpyreg_source),
                str(checkout),
            ],
            cwd=str(ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            env=environment,
            text=True,
            bufsize=1,
        )
        try:
            self.info = self._read()
            if not self.info.get("ready"):
                raise RuntimeError(
                    f"the original arm did not start: {self.info}"
                )
        except BaseException:
            self.kill()
            raise

    def __enter__(self):
        return self

    def __exit__(self, kind, value, trace):
        if kind is None:
            self.close()
        else:
            self.kill()
        return False

    def _read(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(
                f"the original arm's subprocess {self._ending()}; its "
                "traceback, if it wrote one, is in its log or on this "
                "process's standard error"
            )
        return json.loads(line)

    def _ending(self):
        """How the subprocess ended, once its standard output has closed:
        its exit code, or the signal that killed it (a SIGKILL, -9, is
        what the out-of-memory killer sends)."""
        try:
            code = self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            return "closed its output and is still running"
        if code < 0:
            try:
                name = signal.Signals(-code).name
            except ValueError:
                name = f"signal {-code}"
            return f"was killed by {name} (exit code {code})"
        return f"exited with code {code}"

    def request(self, payload):
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()
        reply = self._read()
        if "error" in reply:
            raise RuntimeError("the original arm failed:\n" + reply["error"])
        return reply

    def _finish(self):
        if not sys.platform.startswith("linux"):
            return
        # Every child this process has waited for; the subprocess is by far
        # the largest of them (the others are git and pip queries).
        self.peak_rss_bytes = peak_rss(children=True)

    def close(self):
        """Ask the subprocess to quit and wait for it."""
        if self.process.poll() is None:
            self.peak_rss_bytes = process_peak_rss(
                self.process.pid, descendants=True
            )
            try:
                self.request({"op": "quit"})
            except (OSError, ValueError, RuntimeError):
                pass
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self._finish()
        self._close_log()

    def kill(self):
        """End the subprocess at once."""
        if self.process.poll() is None:
            self.peak_rss_bytes = process_peak_rss(
                self.process.pid, descendants=True
            )
            self.process.kill()
            self.process.wait()
            self._finish()
        self._close_log()

    def _close_log(self):
        if self.log is not None and not self.log.closed:
            self.log.close()

    def summary(self):
        """What a record keeps of the subprocess: its imports and memory."""
        keys = (
            "svbmc_version",
            "svbmc_import",
            "torch_version",
            "torch_import",
            "torch_threads",
            "executable",
        )
        return {
            **{key: self.info[key] for key in keys},
            "pythonpath": self.pythonpath,
            "peak_rss_bytes": self.peak_rss_bytes,
        }


# --------------------------------------------------------------------------
# The cells
# --------------------------------------------------------------------------


def entry_seeds(cell_seed, M):
    """One generator seed per entry of a cell, from the cell's seed.

    Every posterior of a cell is rebuilt with its own seed, in both arms:
    the original implementation draws each run's block of samples from that
    run's own generator, so one seed shared by the whole subset would
    couple the blocks.
    """
    return [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in np.random.SeedSequence(int(cell_seed)).spawn(int(M))
    ]


def arm_sets(value, grid):
    """The arm set of every ``M``: ``both`` or ``integrated``, per ``M``.

    ``value`` is one name for every ``M``, or a comma-separated name per
    ``M`` of ``grid``; raises ``ValueError`` otherwise.
    """
    names = [v.strip() for v in str(value).split(",") if v.strip()]
    if len(names) == 1:
        names = names * len(grid)
    unknown = [name for name in names if name not in ARM_SETS]
    if unknown or len(names) != len(grid):
        raise ValueError(
            f"--arms takes {' or '.join(ARM_SETS)}, once or once per M "
            f"({len(grid)} values); got {value!r}"
        )
    return names


def describe_arms(grid, arms_by_M):
    """The arms of a grid in words, as the run prints them."""
    parts = []
    for name, words in (
        ("both", "both arms"),
        ("integrated", "the integrated arm alone"),
    ):
        values = [str(M) for M, arms in zip(grid, arms_by_M) if arms == name]
        if values:
            parts.append(f"{words} at M = {', '.join(values)}")
    return "; ".join(parts)


def plan_cells(conditions, grid, repetitions, seed, indices, arms_by_M=None):
    """Every cell of the comparison, in the order it is run and assembled.

    The subsets of one condition and ``M`` are disjoint: repetition ``r``
    takes the ``r``-th block of ``M`` runs of a permutation of the
    condition's runs drawn by ``default_rng([seed, index, M])``, and a
    pool of ``n`` runs gives at most ``n // M`` repetitions. Every
    ``(condition, M)`` that gives fewer than the requested repetitions is
    listed in the second return value with the pool size and the number
    drawn. ``indices`` gives each condition its place among all the labels
    the pools hold, so that the subsets drawn for a condition do not
    depend on which conditions this invocation selected. ``arms_by_M``
    (``both`` by default) names the arms of each ``M``; a two-arm cell
    runs the integrated arm first on an even repetition and the original
    on an odd one.
    """
    arms_by_M = ["both"] * len(grid) if arms_by_M is None else arms_by_M
    plan, skipped = [], []
    for condition, entries in conditions.items():
        index = indices[condition]
        n = len(entries)
        for M, R, arms in zip(grid, repetitions, arms_by_M):
            M, R = int(M), int(R)
            drawn = min(R, n // M)
            if drawn < R:
                skipped.append(
                    {
                        "condition": condition,
                        "M": M,
                        "pool": n,
                        "requested": R,
                        "drawn": drawn,
                    }
                )
            if not drawn:
                continue
            order = np.random.default_rng([seed, index, M]).permutation(n)
            arm_set = list(ARM_SETS[arms])
            for repetition in range(drawn):
                key = [seed, index, M, repetition]
                block = order[repetition * M : (repetition + 1) * M]
                subset = sorted(int(i) for i in block)
                cell_seed = int(
                    np.random.SeedSequence(key).generate_state(
                        1, dtype=np.uint32
                    )[0]
                )
                plan.append(
                    {
                        "condition": condition,
                        "condition_index": index,
                        "M": M,
                        "repetition": repetition,
                        "indices": subset,
                        "entries": [entries[i]["name"] for i in subset],
                        "seeds": [entries[i]["seed"] for i in subset],
                        "cell_seed": cell_seed,
                        "entry_seeds": entry_seeds(cell_seed, M),
                        "first_arm": arm_set[repetition % len(arm_set)],
                        "arm_set": arm_set,
                    }
                )
    return plan, skipped


def comparison_settings(seed, grid, repetitions, arms_by_M, max_steps, kind):
    """The settings ``results.json`` records for a comparison."""
    return {
        "seed": seed,
        "M": [int(M) for M in grid],
        "repetitions": [int(R) for R in repetitions],
        "arms_by_M": list(arms_by_M),
        "subsets": "disjoint",
        "max_steps": max_steps,
        "n_samples": N_SAMPLES,
        "lr": LEARNING_RATE,
        "version": VERSION,
        "n_samples_final": N_SAMPLES_FINAL,
        "n_draws": N_DRAWS,
        "n_log_joint": N_LOG_JOINT,
        "n_entropy_ref": N_ENTROPY_REF,
        "n_entropy_batches": N_ENTROPY_BATCHES,
        "growth_bound": GROWTH_BOUND,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "alpha": ALPHA,
        "kind": kind,
        "arms": [
            arm
            for arm in ARMS
            if any(arm in ARM_SETS[arms] for arms in arms_by_M)
        ],
    }


def fit_cell(
    arm,
    server,
    condition,
    kind,
    entries,
    seeds,
    cell_seed,
    max_steps,
    problem,
    ref,
):
    """One arm of one cell: in this process, or through the subprocess.

    Returns the arm's record and, for the integrated arm, the fitted
    object that :func:`add_reference` scores the cell with; the original
    arm's object lives in its subprocess and never crosses back.
    """
    if arm == "integrated":
        return fit_integrated(
            entries, seeds, cell_seed, max_steps, problem, ref
        )
    return (
        server.request(
            {
                "op": "fit",
                "condition": condition,
                "kind": kind,
                "entries": entries,
                "entry_seeds": seeds,
                "cell_seed": cell_seed,
                "max_steps": max_steps,
            }
        ),
        None,
    )


def warm_up(cell, conditions, kind, server, problems, arms):
    """One short discarded fit per arm, so no cell pays the first-call cost.

    Both implementations build their Torch graph, import what they need
    lazily and touch the sampling and metric paths on their first fit, which
    would otherwise land on the first recorded cell and inflate its
    optimization seconds. The fit is two Adam steps on the first two runs of
    ``cell`` and nothing about it is recorded; every recorded cell
    rebuilds its posteriors and reseeds both arms, so it leaves no trace.
    Returns the seconds of each arm's warm-up.
    """
    condition = cell["condition"]
    entries = [conditions[condition][i] for i in cell["indices"][:2]]
    seconds = {}
    for arm in arms:
        started = time.perf_counter()
        fit_cell(
            arm,
            server,
            condition,
            kind,
            entries,
            cell["entry_seeds"][:2],
            cell["cell_seed"],
            2,
            problems.get(condition, kind),
            problems.reference(condition, kind),
        )
        seconds[arm] = time.perf_counter() - started
        print(f"warmed up {arm} in {seconds[arm]:.1f} s", flush=True)
    return seconds


def check_pairing(where, outcomes):
    """Both arms filter the subset with the same rule, so they must have
    retained the same runs with the same component counts; a difference
    means the cell's two fits are not of the same mixture and neither the
    weight difference nor the shared entropy reference would mean
    anything. Returns the maximum absolute weight difference."""
    weights = [np.asarray(outcomes[arm]["w"]) for arm in ARMS]
    if weights[0].shape != weights[1].shape:
        raise RuntimeError(
            f"{where}: the arms returned weight vectors of "
            f"different lengths ({weights[0].size} and "
            f"{weights[1].size}); the cell is not paired"
        )
    for key in ("K", "M_used"):
        if outcomes["integrated"][key] != outcomes["original"][key]:
            raise RuntimeError(
                f"{where}: the arms retained different runs "
                f"({key} {outcomes['integrated'][key]} against "
                f"{outcomes['original'][key]}); the cell is not "
                "paired"
            )
    return float(np.max(np.abs(weights[0] - weights[1])))


def run_cell(cell, conditions, kind, server, problems, max_steps):
    """Run one planned cell, its arms one at a time; return its row.

    With both arms, the cell's ``first_arm`` goes first and the row
    records their pairing; with the integrated arm alone, the row carries
    that arm's record and no paired quantity.
    """
    condition = cell["condition"]
    entries = [conditions[condition][i] for i in cell["indices"]]
    problem = problems.get(condition, kind)
    reference = problems.reference(condition, kind)
    arms = cell["arm_set"]
    outcomes, stacked = {}, None
    for arm in (
        cell["first_arm"],
        *[a for a in arms if a != cell["first_arm"]],
    ):
        outcomes[arm], fitted = fit_cell(
            arm,
            server,
            condition,
            kind,
            entries,
            cell["entry_seeds"],
            cell["cell_seed"],
            max_steps,
            problem,
            reference,
        )
        stacked = fitted if arm == "integrated" else stacked
    row = {key: value for key, value in cell.items() if key != "arm_set"}
    row["arms"] = outcomes
    where = f"{condition} M={cell['M']} r={cell['repetition']}"
    row["max_abs_dw"] = (
        check_pairing(where, outcomes) if set(arms) == set(ARMS) else None
    )
    reference_started = time.perf_counter()
    add_reference(row, stacked, problem, cell["cell_seed"])
    row["reference_seconds"] = time.perf_counter() - reference_started
    if set(arms) == set(ARMS):
        print(
            f"DONE  {where}: "
            f"max|dw| {row['max_abs_dw']:.4g}, "
            f"{outcomes['integrated']['optimize_seconds']:.1f} s vs "
            f"{outcomes['original']['optimize_seconds']:.1f} s, bias "
            f"{outcomes['integrated']['bias']['headline']:+.3f} vs "
            f"{outcomes['original']['bias']['estimated']:+.3f} "
            f"(reference {row['reference_seconds']:.1f} s)",
            flush=True,
        )
    else:
        print(
            f"DONE  {where}: "
            + ", ".join(
                f"{arm} {outcomes[arm]['optimize_seconds']:.1f} s, "
                f"bias "
                f"{outcomes[arm]['bias'][HEADLINE_VARIANT[arm]]:+.3f}"
                for arm in arms
            )
            + f" (reference {row['reference_seconds']:.1f} s)",
            flush=True,
        )
    return row


def run_cells(plan, conditions, kind, server, problems, max_steps, out):
    """Run every planned cell in order; return the rows.

    Each finished row is also appended to ``<out>/cells.jsonl`` as the
    sweep goes.
    """
    progress = (out / "cells.jsonl").open("w", encoding="utf-8")
    rows = []
    started = time.time()
    try:
        for number, cell in enumerate(plan):
            print(
                f"START {cell['condition']} M={cell['M']} "
                f"r={cell['repetition']} ({number + 1}/{len(plan)}, "
                f"{cell['first_arm']} first, "
                f"{(time.time() - started) / 60:.1f} min elapsed)",
                flush=True,
            )
            row = run_cell(cell, conditions, kind, server, problems, max_steps)
            rows.append(row)
            progress.write(json.dumps(row) + "\n")
            progress.flush()
    finally:
        progress.close()
    return rows


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------


#: The per-arm fields of a cell that every summary reads and that cells
#: recorded before the Monte Carlo ELBO reference (2026-09-14) do not
#: carry.
REFERENCE_FIELDS = (
    "entropy_ref",
    "entropy_ref_sd",
    "e_log_joint_mc",
    "elbo_mc",
    "elbo_mc_sd",
    "kl_gap",
    "bias",
)


def require_reference_fields(rows, path):
    """Refuse cells that predate the Monte Carlo ELBO reference.

    The summaries are stated as biases against ``elbo_mc``, which a
    comparison run before that reference existed never recorded. Such a
    ``results.json`` cannot be summarized at all, only regenerated, so say
    which field is missing rather than failing on the first lookup.
    """
    for row in rows:
        for arm, outcome in row["arms"].items():
            missing = [k for k in REFERENCE_FIELDS if k not in outcome]
            if missing:
                raise RuntimeError(
                    f"{path}: the {arm} arm of {row['condition']} "
                    f"M={row['M']} r={row['repetition']} records no "
                    f"{missing[0]}. This file predates the Monte Carlo ELBO "
                    "reference every summary is stated against (the bias of "
                    "each estimate relative to `elbo_mc`), which cannot be "
                    "recovered from the recorded cells: rerun the "
                    "comparison to summarize it."
                )


def setting(settings, key):
    """One setting of the comparison being summarized.

    The recorded value, or this module's constant when the results file
    predates the field, so that a summary rebuilt later describes the run
    it summarizes rather than the code that rebuilt it.
    """
    value = settings.get(key)
    return SETTING_DEFAULTS[key] if value is None else value


def bootstrap_median(values, rng, resamples=BOOTSTRAP_RESAMPLES):
    """Median of ``values`` with a percentile bootstrap 95 % interval."""
    finite = np.asarray(
        [v for v in values if v is not None and np.isfinite(v)], dtype=float
    )
    if finite.size == 0:
        return {"n": 0, "median": None, "lo": None, "hi": None}
    draws = np.median(
        finite[rng.integers(0, finite.size, size=(resamples, finite.size))],
        axis=1,
    )
    return {
        "n": int(finite.size),
        "median": float(np.median(finite)),
        "lo": float(np.percentile(draws, 2.5)),
        "hi": float(np.percentile(draws, 97.5)),
    }


def arm_values(rows, arm, key):
    return [row["arms"][arm]["metrics"].get(key) for row in rows]


def headline_errors(rows, arm):
    return [
        row["arms"][arm]["metrics"].get(
            "elbo_err_" + row["arms"][arm]["headline"]
        )
        for row in rows
    ]


def arm_field(rows, arm, key):
    """One recorded quantity of one arm, cell by cell."""
    return [row["arms"][arm][key] for row in rows]


def bias_values(rows, arm, variant):
    """The bias of one ELBO variant of one arm against its ``elbo_mc``."""
    return [row["arms"][arm]["bias"].get(variant) for row in rows]


def paired_row(row):
    """Whether a cell ran both arms, so that it has paired quantities."""
    return all(arm in row["arms"] for arm in ARMS)


def arms_present(rows):
    """The arms every cell of a set ran, in ``ARMS`` order."""
    return [arm for arm in ARMS if all(arm in row["arms"] for row in rows)]


def paired(integrated, original):
    """Differences integrated minus original, NaN where either is missing."""
    return [
        float("nan") if a is None or b is None else float(a) - float(b)
        for a, b in zip(integrated, original)
    ]


def runtime_ratios(rows):
    """Integrated over original optimization seconds, on the paired cells."""
    return [
        row["arms"]["integrated"]["optimize_seconds"]
        / row["arms"]["original"]["optimize_seconds"]
        for row in rows
        if paired_row(row)
    ]


def exact_signed_rank(delta):
    """Two-sided exact signed-rank test over all sign assignments.

    Zero differences are dropped, tied absolute differences receive
    midranks, and the null distribution of the positive-rank sum is
    enumerated by dynamic programming over the midranks, so ties are
    handled exactly at any sample size. Returns the smaller rank sum and
    the p-value. ``dev/scripts/analyze_population_run.py`` holds the same
    procedure; importing it here would import PyVBMC and gpyreg through
    ``population_run``, which the summaries must not depend on.
    """
    from scipy import stats

    d = np.asarray(delta, dtype=float)
    d = d[d != 0]
    m = len(d)
    if m < 2:
        return 0.0, 1.0
    if m > 62:
        raise ValueError("exact enumeration counts exceed int64 beyond 62")
    ranks = stats.rankdata(np.abs(d))
    units = np.rint(2 * ranks).astype(np.int64)  # midranks are half-integers
    assert np.array_equal(units, 2 * ranks)
    total = int(units.sum())
    counts = np.zeros(total + 1, dtype=np.int64)
    counts[0] = 1
    for unit in units:
        shifted = counts.copy()
        shifted[unit:] += counts[: total + 1 - unit]
        counts = shifted
    positive = int(units[d > 0].sum())
    assignments = float(2**m)
    less = counts[: positive + 1].sum() / assignments
    greater = counts[positive:].sum() / assignments
    statistic = float(min(ranks[d > 0].sum(), ranks[d < 0].sum()))
    return statistic, float(min(1.0, 2 * min(less, greater)))


def holm(tests, alpha=ALPHA):
    """Annotate one test family in place with Holm-adjusted p-values.

    ``integrated_worse`` is the criterion's flag: the family rejects and
    the median paired difference is positive, the integrated class being
    the worse of the two on that cell.
    """
    import golden_trace

    order = np.argsort([test["pvalue"] for test in tests])
    adjusted = 0.0
    for rank, index in enumerate(order):
        adjusted = float(
            max(
                adjusted,
                min(1.0, (len(tests) - rank) * tests[index]["pvalue"]),
            )
        )
        median = tests[index]["median_paired_difference"]
        tests[index]["holm_adjusted_pvalue"] = adjusted
        tests[index]["holm_rejected"] = bool(adjusted <= alpha)
        tests[index]["integrated_worse"] = bool(
            adjusted <= alpha and median is not None and median > 0
        )
    assert [test["holm_rejected"] for test in tests] == list(
        golden_trace._holm([test["pvalue"] for test in tests], alpha)
    )
    return tests


def signed_rank_test(condition, M, metric, rows):
    """The paired test of one metric on one condition-and-``M`` cell set."""
    delta = np.asarray(
        [
            d
            for d in paired(
                arm_values(rows, "integrated", metric),
                arm_values(rows, "original", metric),
            )
            if np.isfinite(d)
        ],
        dtype=float,
    )
    statistic, pvalue = exact_signed_rank(delta)
    return {
        "condition": condition,
        "M": int(M),
        "metric": metric,
        "method": "exact signed-rank over all sign assignments",
        "n_pairs": int(delta.size),
        "improved": int(np.sum(delta < 0)),
        "worsened": int(np.sum(delta > 0)),
        "tied": int(np.sum(delta == 0)),
        "median_paired_difference": (
            float(np.median(delta)) if delta.size else None
        ),
        "statistic": statistic,
        "pvalue": pvalue,
    }


def equivalence_tests(rows, metrics=TEST_METRICS, alpha=ALPHA):
    """Criterion 2: the paired differences per condition and ``M``.

    One exact signed-rank test on the per-cell paired differences
    (integrated minus original) of every condition-and-``M`` cell set, with
    one Holm family per metric over all of those cells. Cell sets the
    original arm did not run have no pairs and no test.
    """
    rows = [row for row in rows if paired_row(row)]
    if not rows:
        return []
    tests = []
    for metric in metrics:
        family = [
            signed_rank_test(
                condition,
                M,
                metric,
                [
                    r
                    for r in rows
                    if r["condition"] == condition and r["M"] == M
                ],
            )
            for condition in dict.fromkeys(row["condition"] for row in rows)
            for M in sorted(
                {r["M"] for r in rows if r["condition"] == condition}
            )
        ]
        tests += holm(family, alpha)
    return tests


def median_bias(arms, arm):
    """The median bias of one arm's headline, or ``None`` when it has none
    or the arm did not run on that cell set."""
    outcome = arms.get(arm)
    entry = (
        None if outcome is None else outcome["bias"].get(HEADLINE_VARIANT[arm])
    )
    return None if entry is None else entry["median"]


def headline_bias_gate(arms):
    """Criterion 3's first clause, on one condition-and-``M`` cell set.

    The integrated class's headline must be no further from the stacked
    posterior's own Monte Carlo ELBO, in the median over the cells, than
    the estimate the original implementation reports.
    """
    integrated, original = median_bias(arms, "integrated"), median_bias(
        arms, "original"
    )
    return {
        "median_bias_headline": integrated,
        "median_bias_estimated": original,
        "headline_bias_not_worse": (
            None
            if integrated is None or original is None
            else bool(abs(integrated) <= abs(original))
        ),
    }


def headline_bias_growth(entries):
    """Criterion 3's second clause, on one condition's cell sets.

    How much the median bias of the integrated headline grows from the
    smallest ``M`` of the condition to the largest, which the criterion
    bounds by ``GROWTH_BOUND`` nats; ``growth_within_bound`` is whether it
    stays under it. The headline is the capped value on a
    noisy stack, which is the case the bound was written for, and the raw
    value where no cap applies, as on the noiseless control. A comparison
    run at a single ``M`` reports a growth of zero, which meets the bound
    by construction.
    """
    by_M = sorted(entries, key=lambda entry: entry["M"])
    first, last = (
        median_bias(entry["arms"], "integrated")
        for entry in (by_M[0], by_M[-1])
    )
    growth = None if first is None or last is None else float(last - first)
    return {
        "M_min": by_M[0]["M"],
        "M_max": by_M[-1]["M"],
        "bias_at_M_min": first,
        "bias_at_M_max": last,
        "growth": growth,
        "bound": GROWTH_BOUND,
        "growth_within_bound": (
            None if growth is None else bool(growth < GROWTH_BOUND)
        ),
    }


def cell_summary(
    rows, rng, resamples=BOOTSTRAP_RESAMPLES, *, shrinkage_rng=None
):
    """The arms a cell set ran, their paired differences and the ratios.

    ``arms_present`` names the arms every cell of the set ran; the paired
    differences, the weight difference, the runtime ratio and criterion
    3's gate exist only when both did.
    """
    boot = partial(bootstrap_median, rng=rng, resamples=resamples)
    keys = ("mmtv", "gskl", "gskl_normalized")
    present = arms_present(rows)
    summary = {
        "cells": len(rows),
        "arms_present": present,
        "arms": {},
        "paired": {},
    }
    for arm in present:
        entry = {key: boot(arm_values(rows, arm, key)) for key in keys}
        entry["elbo_err_headline"] = boot(headline_errors(rows, arm))
        entry["elbo_err_mc"] = boot(arm_values(rows, arm, "elbo_err_mc"))
        for key in (
            "entropy",
            "entropy_ref",
            "entropy_ref_sd",
            "e_log_joint_mc",
            "elbo_mc",
            "elbo_mc_sd",
            "kl_gap",
            "optimize_seconds",
        ):
            entry[key] = boot(arm_field(rows, arm, key))
        variants = sorted(
            {v for row in rows for v in row["arms"][arm]["bias"]}
            - {"shrunk_two_level"}
        )
        entry["bias"] = {
            variant: boot(bias_values(rows, arm, variant))
            for variant in variants
        }
        if arm == "integrated":
            outcomes = [row["arms"][arm] for row in rows]
            recorded = [
                outcome
                for outcome in outcomes
                if "shrinkage_available" in outcome
            ]
            if recorded:
                if shrinkage_rng is None:
                    shrinkage_rng = np.random.default_rng(
                        SHRINKAGE_BOOTSTRAP_STREAM
                    )
                entry["bias"]["shrunk_two_level"] = bootstrap_median(
                    bias_values(rows, arm, "shrunk_two_level"),
                    shrinkage_rng,
                    resamples,
                )
            entry["shrinkage_availability"] = {
                "contributing_cells": entry["bias"].get(
                    "shrunk_two_level", {"n": 0}
                )["n"],
                "unavailable_cells": sum(
                    outcome.get("shrinkage_available") is False
                    for outcome in outcomes
                ),
                "not_recorded_cells": len(outcomes) - len(recorded),
            }
        summary["arms"][arm] = entry
    summary["criterion3"] = headline_bias_gate(summary["arms"])
    if set(present) == set(ARMS):
        summary["paired"]["bias_headline_minus_estimated"] = boot(
            paired(
                bias_values(
                    rows, "integrated", HEADLINE_VARIANT["integrated"]
                ),
                bias_values(rows, "original", HEADLINE_VARIANT["original"]),
            ),
        )
        for key in keys:
            summary["paired"][key] = boot(
                paired(
                    arm_values(rows, "integrated", key),
                    arm_values(rows, "original", key),
                ),
            )
        summary["paired"]["elbo_err_headline"] = boot(
            paired(
                headline_errors(rows, "integrated"),
                headline_errors(rows, "original"),
            ),
        )
    summary["max_abs_dw"] = boot([row["max_abs_dw"] for row in rows])
    summary["runtime_ratio"] = boot(runtime_ratios(rows))
    return summary


def build_summary(rows, singles, settings, rng):
    """Per condition and ``M``, with the single-run medians per condition.

    Each condition also carries an ``all_M`` aggregate over its cells of
    the runtime ratio and the maximum weight difference, which is the level
    criteria 1 and 4 are stated at, and each condition-and-``M`` entry
    carries its two equivalence tests. The tests are also listed flat, in
    the two Holm families they were corrected in.

    The bootstrap draws as many resamples as ``settings`` records, so that
    rebuilding the summary of an earlier comparison gives the summary that
    comparison wrote rather than one this module's constants imply.
    """
    resamples = int(setting(settings, "bootstrap_resamples"))
    boot = partial(bootstrap_median, rng=rng, resamples=resamples)
    shrinkage_rng = np.random.default_rng(
        [int(settings["seed"]), SHRINKAGE_BOOTSTRAP_STREAM]
    )
    conditions = []
    for condition in dict.fromkeys(row["condition"] for row in rows):
        here = [row for row in rows if row["condition"] == condition]
        entry = {
            "condition": condition,
            "M": [
                dict(
                    cell_summary(
                        [r for r in here if r["M"] == M],
                        rng,
                        resamples,
                        shrinkage_rng=shrinkage_rng,
                    ),
                    M=M,
                )
                for M in sorted({r["M"] for r in here})
            ],
            "all_M": {
                "cells": len(here),
                "runtime_ratio": boot(runtime_ratios(here)),
                "max_abs_dw": boot([row["max_abs_dw"] for row in here]),
            },
            "single_run": {
                key: boot([row[key] for row in singles.get(condition, [])])
                for key in ("elbo_err", "gskl", "gskl_normalized", "mmtv")
            },
        }
        # Reads the per-`M` summaries the entry already holds.
        entry["headline_bias_growth"] = headline_bias_growth(entry["M"])
        conditions.append(entry)
    tests = equivalence_tests(rows)
    for condition in conditions:
        for entry in condition["M"]:
            entry["tests"] = {
                test["metric"]: test
                for test in tests
                if test["condition"] == condition["condition"]
                and test["M"] == entry["M"]
            }
    return {
        "campaign": "svbmc_pool",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "settings": settings,
        "alpha": ALPHA,
        "equivalence_tests": tests,
        "conditions": conditions,
    }


def number(value, digits=3):
    """One number of a table cell, or a dash when it is missing."""
    return "-" if value is None else f"{value:.{digits}f}"


def spaced(count):
    """A count as the prose writes it, thousands separated by a space."""
    return f"{count:,}".replace(",", " ")


def flag(value):
    """One boolean of a table cell, or a dash when it is undetermined."""
    return "-" if value is None else ("yes" if value else "**no**")


def summary_markdown(summary):
    def cell(entry, digits=3):
        if entry["median"] is None:
            return "-"
        return (
            f"{entry['median']:.{digits}f} "
            f"[{entry['lo']:.{digits}f}, {entry['hi']:.{digits}f}]"
        )

    def optional(mapping, *keys, digits=3):
        """A bootstrap cell reached through ``keys``, or a dash when the
        mapping or any key is missing (an arm that did not run)."""
        value = mapping
        for key in keys:
            value = None if value is None else value.get(key)
        return "-" if value is None else cell(value, digits)

    settings = summary["settings"]
    lines = [
        "# S-VBMC stacking comparison: integrated against original 0.1.1",
        "",
        f"Generated {summary['generated']}. Every cell stacks the same "
        f"subset of filtered pool runs with both implementations, "
        f"`n_samples={setting(settings, 'n_samples')}`, "
        f"`lr={setting(settings, 'lr')}`, "
        f"`max_steps={settings['max_steps']}`, "
        f"`version=\"{setting(settings, 'version')}\"`, "
        f"subsets drawn with seed {settings['seed']}"
        + (
            ", disjoint within each `M`"
            if settings.get("subsets") == "disjoint"
            else ""
        )
        + ". Medians over the "
        f"repetitions of a cell with a "
        f"{spaced(setting(settings, 'bootstrap_resamples'))}"
        "-resample bootstrap 95 % "
        "interval; `d` columns are the paired difference integrated minus "
        "original, so a negative value favours the integrated class. The "
        "headline ELBO is the capped value for the integrated class on "
        "noisy stacks and the raw estimate for the original, which is what "
        "each implementation reports. gsKL is the house convention (no "
        "`1/D` factor).",
        "",
        "Every reported ELBO is scored by its bias against `elbo_mc = "
        "e_log_joint_mc + entropy_ref`, the Monte Carlo ELBO of the "
        "posterior that arm's fit produced. `e_log_joint_mc` is the mean "
        "of the target's noiseless log density over "
        f"{spaced(setting(settings, 'n_log_joint'))} "
        "of the stack's own draws; `entropy_ref` is the entropy of the "
        "stacked "
        "mixture at that arm's final weights, estimated here for both arms "
        "by one estimator (the integrated class's `stacked_entropy`, "
        f"{setting(settings, 'n_entropy_ref')} draws per component in "
        f"{setting(settings, 'n_entropy_batches')} "
        "batches, from a generator seeded by the cell, the same draws for "
        "both arms). The reference is arm-independent by construction: an "
        "arm's own reported entropy is estimated at the weights that arm "
        "selected, from its own draws (the optimizer's "
        f"{setting(settings, 'n_samples')} per component for the original, "
        f"a fresh {setting(settings, 'n_samples_final')} for the "
        "integrated class), so a "
        "reference built on it would move with the arm and credit "
        "whichever entropy estimate came out high. `kl_gap = ln Z - "
        "elbo_mc` is how far the stacked posterior itself is from the "
        "target, in evidence units; the `err` columns further below are "
        "the error against `ln Z`, descriptive only.",
    ]
    if settings.get("merged_from"):
        lines += [
            "",
            "Cells of several comparison runs summarized together: "
            + "; ".join(
                f"`{Path(part['path']).name}` from `{part['path']}` "
                f"(M {', '.join(str(M) for M in part['M'])}; "
                f"{' and '.join(part['arms'])})"
                for part in settings["merged_from"]
            )
            + ". A cell set that only the integrated arm ran has no "
            "paired quantity, no weight difference, no runtime ratio and "
            "no equivalence test, shown as dashes.",
        ]
    for condition in summary["conditions"]:
        single = condition["single_run"]
        aggregate = condition["all_M"]
        lines += [
            "",
            f"## {condition['condition']}",
            "",
            f"Single runs (`M = 1`, n = {single['mmtv']['n']}): "
            f"MMTV {cell(single['mmtv'])}, gsKL {cell(single['gskl'], 2)}, "
            f"evidence error {cell(single['elbo_err'], 2)}.",
            "",
            f"All {aggregate['cells']} cells of this condition: runtime "
            f"ratio {cell(aggregate['runtime_ratio'])}, "
            f"max\\|dw\\| {cell(aggregate['max_abs_dw'], 4)}.",
            "",
            "Bias of every reported ELBO against `elbo_mc` (criterion 3), "
            "and the KL gap of the stacked posterior. The shrinkage column "
            "shows its contributing and numerically unavailable cell counts; "
            "older cells that did not record shrinkage are counted separately. "
            "`d(head-est)` is the "
            "paired difference of the two arms' headline biases; the last "
            "column is the criterion's gate, `|median bias headline| <= "
            "|median bias estimated|`.",
            "",
            "| M | cells | bias headline int | bias raw int | bias shrink int "
            "(contributing / unavailable / not recorded) | "
            "bias estimated orig | bias debiased_I orig | d(head-est) | "
            "KL gap int | KL gap orig | not worse |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for entry in condition["M"]:
            integrated = entry["arms"]["integrated"]
            original = entry["arms"].get("original")
            gate = entry["criterion3"]["headline_bias_not_worse"]
            lines.append(
                "| {M} | {cells} | {bh} | {br} | {bs} ({sc} / {su} / {so}) "
                "| {be} | {bd} | {dp} | "
                "{ki} | {ko} | {gate} |".format(
                    M=entry["M"],
                    cells=entry["cells"],
                    bh=cell(integrated["bias"]["headline"]),
                    br=cell(integrated["bias"]["raw"]),
                    bs=optional(integrated, "bias", "shrunk_two_level"),
                    sc=integrated["shrinkage_availability"][
                        "contributing_cells"
                    ],
                    su=integrated["shrinkage_availability"][
                        "unavailable_cells"
                    ],
                    so=integrated["shrinkage_availability"][
                        "not_recorded_cells"
                    ],
                    be=optional(original, "bias", "estimated"),
                    bd=optional(original, "bias", "debiased_I_median"),
                    dp=optional(
                        entry["paired"], "bias_headline_minus_estimated"
                    ),
                    ki=cell(integrated["kl_gap"]),
                    ko=optional(original, "kl_gap"),
                    gate=flag(gate),
                )
            )
        growth = condition["headline_bias_growth"]
        lines += [
            "",
            f"Growth of the integrated headline's median bias from "
            f"`M = {growth['M_min']}` to `M = {growth['M_max']}`: "
            f"{number(growth['growth'])} nats "
            f"({number(growth['bias_at_M_min'])} to "
            f"{number(growth['bias_at_M_max'])}), below the "
            f"{growth['bound']}-nat bound: "
            f"{flag(growth['growth_within_bound'])}. The headline is the "
            "capped value on a noisy stack and the raw value where no cap "
            "applies.",
            "",
            "The reference and the arms' own entropies. `H ref` is the "
            "arm-independent estimate at that arm's weights, `H` what the "
            "arm reports; `sd` columns are medians of the per-cell Monte "
            "Carlo standard deviation of `elbo_mc`.",
            "",
            "| M | H ref int | H int | H ref orig | H orig | elbo_mc int | "
            "sd | elbo_mc orig | sd |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for entry in condition["M"]:
            integrated = entry["arms"]["integrated"]
            original = entry["arms"].get("original")
            lines.append(
                "| {M} | {hri} | {hi} | {hro} | {ho} | {mi} | {si} | {mo} "
                "| {so} |".format(
                    M=entry["M"],
                    hri=cell(integrated["entropy_ref"]),
                    hi=cell(integrated["entropy"]),
                    hro=optional(original, "entropy_ref"),
                    ho=optional(original, "entropy"),
                    mi=cell(integrated["elbo_mc"]),
                    si=number(integrated["elbo_mc_sd"]["median"]),
                    mo=optional(original, "elbo_mc"),
                    so=(
                        "-"
                        if original is None
                        else number(original["elbo_mc_sd"]["median"])
                    ),
                )
            )
        lines += [
            "",
            "Posterior quality, weights and runtime. The `err` columns are "
            "the error of each arm's headline against `ln Z`, descriptive "
            "only: they mix the estimator's bias with the stack's KL gap, "
            "so a small value can arise by cancellation and cannot rank "
            "estimators.",
            "",
            "| M | cells | MMTV int | MMTV orig | dMMTV | gsKL int | "
            "gsKL orig | dgsKL | err int | err orig | derr | max\\|dw\\| | "
            "opt s int | opt s orig | ratio |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for entry in condition["M"]:
            integrated = entry["arms"]["integrated"]
            original = entry["arms"].get("original")
            lines.append(
                "| {M} | {cells} | {mi} | {mo} | {dm} | {gi} | {go} | {dg} "
                "| {ei} | {eo} | {de} | {dw} | {ti} | {to} | {ratio} |".format(
                    M=entry["M"],
                    cells=entry["cells"],
                    mi=cell(integrated["mmtv"]),
                    mo=optional(original, "mmtv"),
                    dm=optional(entry["paired"], "mmtv", digits=4),
                    gi=cell(integrated["gskl"], 2),
                    go=optional(original, "gskl", digits=2),
                    dg=optional(entry["paired"], "gskl", digits=3),
                    ei=cell(integrated["elbo_err_headline"], 2),
                    eo=optional(original, "elbo_err_headline", digits=2),
                    de=optional(
                        entry["paired"], "elbo_err_headline", digits=3
                    ),
                    dw=cell(entry["max_abs_dw"], 4),
                    ti=cell(integrated["optimize_seconds"], 1),
                    to=optional(original, "optimize_seconds", digits=1),
                    ratio=cell(entry["runtime_ratio"]),
                )
            )
        lines += [
            "",
            "Paired equivalence tests (exact signed-rank on the per-cell "
            "differences, one Holm family per metric over every condition "
            f"and `M` at alpha {summary['alpha']}); a flagged cell is one "
            "the family rejects with the integrated class the worse of the "
            "two.",
            "",
            "| M | pairs | dMMTV median | MMTV p | MMTV Holm | dgsKL median "
            "| gsKL p | gsKL Holm | flagged |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for entry in condition["M"]:
            tests = entry["tests"]
            if not tests:
                lines.append(
                    f"| {entry['M']} | 0 | - | - | - | - | - | - | - |"
                )
                continue
            mmtv, gskl = tests["mmtv"], tests["gskl"]
            flagged = [
                metric
                for metric, test in tests.items()
                if test["integrated_worse"]
            ]
            lines.append(
                "| {M} | {n} | {dm} | {pm:.4g} | {hm:.4g} | {dg} | "
                "{pg:.4g} | {hg:.4g} | {flag} |".format(
                    M=entry["M"],
                    n=mmtv["n_pairs"],
                    dm=number(mmtv["median_paired_difference"], 4),
                    pm=mmtv["pvalue"],
                    hm=mmtv["holm_adjusted_pvalue"],
                    dg=number(gskl["median_paired_difference"], 3),
                    pg=gskl["pvalue"],
                    hg=gskl["holm_adjusted_pvalue"],
                    flag=", ".join(flagged) if flagged else "no",
                )
            )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# The single-process run
# --------------------------------------------------------------------------


def integer_list(text):
    return [int(v) for v in str(text).split(",") if v.strip()]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--pool",
        action="append",
        type=Path,
        default=None,
        help="a pool directory written by svbmc_pool_run.py, repeatable",
    )
    parser.add_argument(
        "--fixtures",
        help="comma-separated fixture groups to compare instead of pools",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--conditions", help="comma-separated pool labels")
    parser.add_argument("--M", default="2,4,8,16")
    parser.add_argument("--repetitions", default="20,20,20,10")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--gpyreg-source",
        type=Path,
        help="the gpyreg checkout both arms import (default: "
        "PYVBMC_GPYREG_SOURCE, and for --pool runs without it the path "
        "the first pool's manifest names); for --pool runs it must be a "
        "clean checkout at every manifest's gpyreg commit",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="rebuild summary.json and summary.md from a finished "
        "results.json, running no cell",
    )
    parser.add_argument(
        "--from-results",
        type=Path,
        action="append",
        help="the results.json to summarize (default: <out>/results.json); "
        "repeatable, and the files' cells are then summarized together",
    )
    parser.add_argument(
        "--arms",
        default="both",
        help="which implementations stack the cells of each M: both "
        "(default), or integrated, the integrated class alone, for the "
        "larger-M regime where the original's cost is not worth paying; "
        "one name for every M, or one per M",
    )
    args = parser.parse_args(argv)
    args.M = integer_list(args.M)
    args.repetitions = integer_list(args.repetitions)
    if len(args.M) != len(args.repetitions):
        parser.error("--M and --repetitions must have the same length")
    if len(set(args.M)) != len(args.M):
        parser.error("--M names an M twice")
    if args.summarize_only:
        unusable = [
            flag
            for flag, value in (
                ("--pool", args.pool),
                ("--fixtures", args.fixtures),
                ("--conditions", args.conditions),
                ("--gpyreg-source", args.gpyreg_source),
                ("--arms", None if args.arms == "both" else args.arms),
            )
            if value
        ]
        if unusable:
            parser.error(
                f"--summarize-only runs no cell, so {', '.join(unusable)} "
                "cannot apply"
            )
        return args
    try:
        args.arms_by_M = arm_sets(args.arms, args.M)
    except ValueError as error:
        parser.error(str(error))
    if args.from_results:
        parser.error("--from-results needs --summarize-only")
    if bool(args.pool) == bool(args.fixtures):
        parser.error("give either --pool (repeatable) or --fixtures")
    if args.fixtures and args.conditions:
        parser.error(
            "--conditions selects labels of a pool; --fixtures already "
            "names the groups to compare"
        )
    return args


def load_results(paths):
    """The cells, single-run rows and settings of one or more results files.

    Several files are the runs of one comparison split by arm set or by
    ``M`` (both arms up to one ``M``, the integrated arm beyond it): their
    cells are concatenated, the settings are the first file's with ``M``,
    ``repetitions``, ``arms`` and ``arms_by_M`` widened to cover every
    file and a ``merged_from`` record of each file, and the single-run
    rows are the first file's, since every run scores the same pools.
    Files that differ in any setting they share other than those, that
    hold the same cell twice, or that hold cells of one condition and
    ``M`` with different arm sets, cannot be summarized together; the
    repetitions of an ``M`` are counted from the merged cells.
    """
    results = [
        (path, json.loads(path.read_text(encoding="utf-8"))) for path in paths
    ]
    for path, result in results:
        if result.get("settings") is None:
            raise RuntimeError(
                f"{path} holds no settings, which a summary needs"
            )
        require_reference_fields(result["cells"], path)
    first_path, first = results[0]
    settings = dict(first["settings"])
    if len(results) > 1:
        # Every setting the files share must agree, except the ones a
        # split by M or by arm set is allowed to differ in.
        free = {"M", "repetitions", "arms", "arms_by_M", "merged_from"}
        for path, result in results[1:]:
            for key in sorted(set(settings) & set(result["settings"]) - free):
                if result["settings"][key] != settings[key]:
                    raise RuntimeError(
                        f"{path} was run with {key}="
                        f"{result['settings'][key]!r} and {first_path} "
                        f"with {settings[key]!r}; their cells cannot be "
                        "summarized together"
                    )
        seen, arm_sets_held = {}, {}
        for path, result in results:
            for row in result["cells"]:
                key = (row["condition"], row["M"], row["repetition"])
                if key in seen:
                    raise RuntimeError(
                        f"{path} and {seen[key]} both hold the cell "
                        f"{key[0]} M={key[1]} r={key[2]}; a cell is "
                        "summarized once"
                    )
                seen[key] = path
                # A cell set is summarized with one arm set: its paired
                # quantities and tests would otherwise be computed on a
                # subset of the cells its medians describe.
                arms = tuple(sorted(row["arms"]))
                held = arm_sets_held.setdefault(key[:2], (arms, path))
                if held[0] != arms:
                    raise RuntimeError(
                        f"{path} and {held[1]} hold cells of {key[0]} "
                        f"M={key[1]} with different arm sets "
                        f"({', '.join(arms)} against {', '.join(held[0])}); "
                        "a cell set is summarized with one arm set"
                    )
        repetitions = {}
        for path, result in results:
            for M, R in zip(
                result["settings"]["M"], result["settings"]["repetitions"]
            ):
                repetitions[int(M)] = max(repetitions.get(int(M), 0), int(R))
            for row in result["cells"]:
                repetitions[int(row["M"])] = max(
                    repetitions.get(int(row["M"]), 0),
                    int(row["repetition"]) + 1,
                )
        settings["M"] = sorted(repetitions)
        settings["repetitions"] = [repetitions[M] for M in settings["M"]]
        settings["arms"] = [
            arm
            for arm in ARMS
            if any(arm in setting(r["settings"], "arms") for _, r in results)
        ]
        by_M = {}
        for _, result in results:
            for row in result["cells"]:
                by_M.setdefault(int(row["M"]), set()).add(
                    "both" if paired_row(row) else "integrated"
                )
        settings["arms_by_M"] = [
            next(iter(by_M[M])) if len(by_M.get(M, ())) == 1 else None
            for M in settings["M"]
        ]
        settings["merged_from"] = [
            {
                "path": str(path),
                "generated": result.get("generated"),
                "M": result["settings"]["M"],
                "repetitions": result["settings"]["repetitions"],
                "arms": setting(result["settings"], "arms"),
                "cells": len(result["cells"]),
            }
            for path, result in results
        ]
    cells = [row for _, result in results for row in result["cells"]]
    return cells, first["single_run"], settings


def write_summaries(out, rows, singles, settings):
    """``summary.json`` and ``summary.md`` of a set of cells; returns the
    markdown. The bootstrap is seeded from the comparison's seed and drawn
    in the order of ``rows``."""
    summary = build_summary(
        rows, singles, settings, np.random.default_rng(settings["seed"])
    )
    contract.write_json(out / "summary.json", summary)
    text = summary_markdown(summary)
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def summarize_only(args):
    """Rebuild the summaries of a finished comparison, running no cell."""
    out = args.out.resolve()
    paths = [
        path.resolve()
        for path in (args.from_results or [out / "results.json"])
    ]
    cells, single_run, settings = load_results(paths)
    out.mkdir(parents=True, exist_ok=True)
    text = write_summaries(out, cells, single_run, settings)
    print(text, flush=True)
    print(
        f"{len(cells)} cells from {', '.join(str(p) for p in paths)} -> {out}",
        flush=True,
    )
    return 0


def single_process_gpyreg(args, manifests):
    """The gpyreg checkout of a single-process run, and what named it.

    ``--gpyreg-source``, else ``PYVBMC_GPYREG_SOURCE``, else, for a run on
    pools, the path the first pool's manifest names. For a run on pools
    the checkout must be at every manifest's gpyreg commit
    (:func:`check_gpyreg_against_pools`).
    """
    candidates = [
        (args.gpyreg_source, "--gpyreg-source"),
        (os.environ.get("PYVBMC_GPYREG_SOURCE"), "PYVBMC_GPYREG_SOURCE"),
    ]
    if manifests:
        candidates.append(
            (manifests[0][1].get("gpyreg_source"), "the pool manifest")
        )
    for source, origin in candidates:
        if source:
            break
    else:
        raise RuntimeError(
            "name the gpyreg checkout both arms import, with "
            "--gpyreg-source or PYVBMC_GPYREG_SOURCE"
        )
    source = Path(source).resolve()
    if manifests:
        check_gpyreg_against_pools(source, manifests)
    elif not (source / "gpyreg").is_dir():
        raise RuntimeError(f"no gpyreg package under {source}")
    return source, origin


def run_comparison(args):
    """The whole comparison in this process, every cell in plan order."""
    out = args.out.resolve()
    if (out / "results.json").exists() and not args.overwrite:
        raise RuntimeError(f"{out} already holds a comparison; --overwrite")
    out.mkdir(parents=True, exist_ok=True)

    kind = "fixture" if args.fixtures else "run"
    only = [s.strip() for s in (args.conditions or "").split(",") if s.strip()]
    manifests = [
        (
            Path(pool).resolve() / "manifest.json",
            contract.read_json(Path(pool).resolve() / "manifest.json"),
        )
        for pool in (args.pool or [])
    ]
    source, gpyreg_origin = single_process_gpyreg(args, manifests)
    gpyreg_source = activate_gpyreg(source)
    arms_by_M = args.arms_by_M
    runs_original = any("original" in ARM_SETS[a] for a in arms_by_M)
    arms = [a for a in ARMS if any(a in ARM_SETS[s] for s in arms_by_M)]
    checkout = baseline_checkout() if runs_original else None
    if checkout is not None:
        refuse_upstream_on_path(checkout / "src")
    torch = import_torch()

    # This raises unless PyVBMC imported gpyreg from the campaign's frozen
    # worktree, so the controller's own environment is settled before the
    # baseline's is checked and long before a cell runs.
    integrated_identity = identity(gpyreg_source)
    verification = (
        verify_baseline(read_baseline_record(), checkout)
        if checkout is not None
        else None
    )

    if kind == "run":
        conditions, pool_identities, labels = pool_conditions(
            args.pool, only, gpyreg_source=gpyreg_source
        )
        fixtures = []
    else:
        groups = [s.strip() for s in args.fixtures.split(",") if s.strip()]
        conditions, pool_identities = fixture_conditions(groups), []
        labels = sorted(FIXTURE_PROBLEMS)
        fixtures = fixture_sources(conditions)
    indices = {label: labels.index(label) for label in conditions}
    origins = {
        label: origin
        for record in pool_identities
        for label, origin in record["selection"]["conditions"].items()
    }
    for label, entries in conditions.items():
        print(
            f"{label}: {len(entries)} filtered runs from "
            f"{origins.get(label, 'the fixture group')}",
            flush=True,
        )

    problems = Problems()
    singles = {
        condition: single_run_rows(condition, kind, entries, problems)
        for condition, entries in conditions.items()
    }
    plan, skipped = plan_cells(
        conditions, args.M, args.repetitions, args.seed, indices, arms_by_M
    )
    for entry in skipped:
        print(
            f"{entry['condition']} M={entry['M']}: {entry['drawn']} of "
            f"{entry['requested']} repetitions, the most that {entry['pool']} "
            "filtered runs give in disjoint subsets",
            flush=True,
        )
    if not plan:
        raise RuntimeError(
            "no cell to run: every requested M exceeds the filtered pools"
        )
    print(f"{len(plan)} cells: {describe_arms(args.M, arms_by_M)}", flush=True)

    started = time.time()
    with contextlib.ExitStack() as stack:
        server = (
            stack.enter_context(
                OriginalArm(gpyreg_source, checkout, out / "original_arm.log")
            )
            if runs_original
            else None
        )
        # Process-scoped request, as in the pool runner's sweep: permit
        # display sleep, prevent idle system sleep while the cells run,
        # since a full grid takes hours on a laptop.
        if sys.platform == "win32":
            import ctypes

            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
            stack.callback(
                ctypes.windll.kernel32.SetThreadExecutionState, 0x80000000
            )
        warm_up(plan[0], conditions, kind, server, problems, arms)
        rows = run_cells(
            plan, conditions, kind, server, problems, args.max_steps, out
        )
    elapsed = time.time() - started

    settings = comparison_settings(
        args.seed, args.M, args.repetitions, arms_by_M, args.max_steps, kind
    )
    contract.write_json(
        out / "results.json",
        {
            "campaign": "svbmc_pool",
            "kind": kind,
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": elapsed,
            "settings": settings,
            "skipped": skipped,
            "cells": rows,
            "single_run": singles,
        },
    )
    text = write_summaries(out, rows, singles, settings)
    contract.write_json(
        out / "sources.json",
        {
            "campaign": "svbmc_pool",
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "harness": {
                "path": str(Path(__file__).resolve()),
                "sha256": contract.sha256_file(Path(__file__).resolve()),
            },
            "settings": settings,
            "hostname": platform.node(),
            "arms": {
                "integrated": {
                    "environment": integrated_identity,
                    "torch_version": torch.__version__,
                    "torch_import": str(Path(torch.__file__).resolve()),
                    "torch_threads": torch.get_num_threads(),
                    "pythonpath": os.environ.get("PYTHONPATH"),
                },
                "original": None if server is None else server.info,
            },
            "baseline": (
                None
                if server is None
                else {
                    "BASELINE_DIR": os.environ.get("BASELINE_DIR"),
                    "checkout": str(checkout),
                    "pythonpath": server.pythonpath,
                }
            ),
            "baseline_environment": verification,
            "gpyreg_source": str(gpyreg_source),
            "gpyreg_source_origin": gpyreg_origin,
            "pools": pool_identities,
            "fixtures": fixtures,
            "threads": {key: os.environ.get(key) for key in THREAD_KEYS},
        },
    )
    print(text, flush=True)
    print(f"{len(rows)} cells in {elapsed / 60:.1f} min -> {out}", flush=True)
    return 0


# --------------------------------------------------------------------------
# The campaign
# --------------------------------------------------------------------------


def read_manifest(out):
    """The manifest of a campaign directory of this harness."""
    path = Path(out) / "manifest.json"
    manifest = contract.read_json(path)
    if manifest.get("campaign") != CAMPAIGN:
        raise SystemExit(f"{path} is not a manifest of {CAMPAIGN}")
    return manifest


def manifest_conditions(manifest):
    """``{label: entries}`` of a manifest, in its order."""
    return {
        entry["label"]: entry["entries"] for entry in manifest["conditions"]
    }


def cell_path(cell):
    """A cell's file, relative to the campaign directory."""
    return (
        f"{cell['condition']}/M{cell['M']}_r{cell['repetition']}{CELL_SUFFIX}"
    )


def campaign_tasks(manifest):
    """The tasks of a campaign, in the order of its case list.

    A task holds every planned repetition of one condition and ``M`` below
    the manifest's ``split_from``, and one cell from it on. Each task has
    its ``tag``, its case ``line``, its ``condition``, ``M`` and ``arms``,
    the numbers of its ``cells`` in the manifest's plan and their ``paths``.
    """
    split_from = int(manifest["split_from"])
    tasks, by_key = [], {}
    for number, cell in enumerate(manifest["plan"]):
        condition, M = cell["condition"], int(cell["M"])
        key = (condition, M, None if M < split_from else cell["repetition"])
        if key not in by_key:
            tag = f"{condition}/M{M}" + (
                "" if key[2] is None else f"_r{key[2]}"
            )
            by_key[key] = {
                "tag": tag,
                "condition": condition,
                "M": M,
                "arms": list(cell["arm_set"]),
                "cells": [],
                "paths": [],
            }
            tasks.append(by_key[key])
        by_key[key]["cells"].append(number)
        by_key[key]["paths"].append(cell_path(cell))
    for task in tasks:
        repetitions = contract.compress_indices(
            manifest["plan"][n]["repetition"] for n in task["cells"]
        ).replace(",", "+")
        task["line"] = (
            f"{task['tag']} M={task['M']} repetitions={repetitions} "
            f"arms={'+'.join(task['arms'])}"
        )
    return tasks


def task_identity(expected, with_baseline):
    """The identity a task compares: the manifest's, without the baseline
    tree for a task of the integrated arm alone."""
    if with_baseline:
        return expected
    trimmed = copy.deepcopy(expected)
    trimmed.get("source", {}).get("trees", {}).pop("baseline", None)
    return trimmed


def campaign_process(with_baseline):
    """What a campaign process sets up before its identity is taken.

    Pins gpyreg to ``PYVBMC_GPYREG_SOURCE``, imports Torch and, for a
    process that runs the original arm, finds the ``BASELINE_DIR``
    checkout, keeps its ``src`` off this process's path and verifies it
    by content (:func:`verify_baseline`). Every failure raises
    :class:`campaign_contract.IdentityError`, which the worker reports as
    an identity refusal. Returns ``{"gpyreg", "checkout", "baseline"}``,
    the last two None without the original arm.
    """
    source = os.environ.get("PYVBMC_GPYREG_SOURCE")
    if not source:
        raise contract.IdentityError(
            "PYVBMC_GPYREG_SOURCE is not set; it names the gpyreg checkout "
            "of the campaign"
        )
    try:
        gpyreg_source = Path(activate_gpyreg(source))
    except RuntimeError as error:
        raise contract.IdentityError(str(error)) from error
    import_torch()
    state = {"gpyreg": gpyreg_source, "checkout": None, "baseline": None}
    if with_baseline:
        checkout = baseline_checkout()
        refuse_upstream_on_path(checkout / "src")
        state["checkout"] = checkout
        state["baseline"] = verify_baseline(read_baseline_record(), checkout)
    return state


def parse_campaign_args(argv):
    parser = argparse.ArgumentParser(
        prog="svbmc_pool_stack.py",
        description="The stacking comparison as a campaign of the Slurm "
        "driver (dev/plans/slurm-benchmark-support.md, the campaign "
        "contract).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="write the campaign manifest")
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument(
        "--pool",
        action="append",
        type=Path,
        help="a pool directory written by svbmc_pool_run.py, repeatable",
    )
    prepare.add_argument(
        "--fixtures",
        help="comma-separated fixture groups to compare instead of pools",
    )
    prepare.add_argument("--conditions", help="comma-separated pool labels")
    prepare.add_argument("--M", default=",".join(str(M) for M in RELEASE_M))
    prepare.add_argument(
        "--repetitions",
        default=",".join(str(R) for R in RELEASE_REPETITIONS),
    )
    prepare.add_argument(
        "--arms",
        default=",".join(RELEASE_ARMS),
        help="both or integrated, once or once per M (default: the "
        "release gate's)",
    )
    prepare.add_argument("--seed", type=int, default=0)
    prepare.add_argument("--max-steps", type=int, default=500)
    prepare.add_argument(
        "--split-from",
        type=int,
        default=SPLIT_FROM,
        help="the M from which a task holds one cell rather than every "
        f"repetition of its condition and M (default {SPLIT_FROM})",
    )
    prepare.add_argument(
        "--allow-dirty",
        action="store_true",
        help="record that the comparison may be computed from a harness "
        "checkout with uncommitted or untracked changes",
    )
    cases = sub.add_parser("cases", help="print the case list")
    cases.add_argument("--out", type=Path, required=True)
    cases.add_argument(
        "--subset",
        help="M<M> (the tasks at one M), both (the tasks that run the "
        "original arm) or integrated (those that do not)",
    )
    worker = sub.add_parser("worker", help="run one task")
    worker.add_argument("--out", type=Path, required=True)
    worker.add_argument("--case", required=True)
    for name, text in (
        ("verify", "re-check every task and reconcile the allocation"),
        ("assemble", "the finishing step: results.json and the summaries"),
    ):
        sub.add_parser(name, help=text).add_argument(
            "--out", type=Path, required=True
        )
    args = parser.parse_args(argv)
    if args.command == "prepare":
        args.M = integer_list(args.M)
        args.repetitions = integer_list(args.repetitions)
        if len(args.M) != len(args.repetitions):
            parser.error("--M and --repetitions must have the same length")
        if len(set(args.M)) != len(args.M):
            parser.error("--M names an M twice")
        try:
            args.arms_by_M = arm_sets(args.arms, args.M)
        except ValueError as error:
            parser.error(str(error))
        if bool(args.pool) == bool(args.fixtures):
            parser.error("give either --pool (repeatable) or --fixtures")
        if args.fixtures and args.conditions:
            parser.error(
                "--conditions selects labels of a pool; --fixtures already "
                "names the groups to compare"
            )
    return args


#: The manifest fields a prepared campaign is bound to; ``prepare`` on a
#: prepared directory refuses a difference in any of them or in the source
#: identity.
STRUCTURAL_KEYS = (
    "kind",
    "settings",
    "split_from",
    "conditions",
    "plan",
    "skipped",
    "pools",
    "allow_dirty",
    "finishing_steps",
    "tracked_copies",
)
#: What a manifest keeps of a pool run's recorded metrics: the fields of
#: the single-run rows.
SINGLE_RUN_METRICS = ("elbo_err", "gskl", "mmtv")


def manifest_entry(entry):
    """A pool entry as the manifest holds it."""
    kept = {
        key: entry[key]
        for key in ("kind", "name", "path", "seed", "label", "sha256")
        if key in entry
    }
    metrics = entry.get("metrics")
    kept["metrics"] = (
        None
        if metrics is None
        else {k: metrics[k] for k in SINGLE_RUN_METRICS if k in metrics}
    )
    return kept


def cmd_prepare(args):
    """Write the campaign's manifest, or check that a prepared one is this.

    Refuses a dirty gpyreg checkout, and a dirty harness checkout without
    ``--allow-dirty``, which the manifest records; and a pool that has not
    verified, or whose selection was not made after its verification or
    disagrees with it (``svbmc_pool_run.stackable_selection``).
    """
    out = args.out.resolve()
    arms_by_M = args.arms_by_M
    with_baseline = any("original" in ARM_SETS[a] for a in arms_by_M)
    state = campaign_process(with_baseline)
    source_identity = campaign_identity(state["gpyreg"], state["checkout"])
    dirty = dirty_trees(source_identity, args.allow_dirty, "the comparison")
    kind = "fixture" if args.fixtures else "run"
    only = [s.strip() for s in (args.conditions or "").split(",") if s.strip()]
    if kind == "run":
        manifests = [
            (
                Path(pool).resolve() / "manifest.json",
                contract.read_json(Path(pool).resolve() / "manifest.json"),
            )
            for pool in args.pool
        ]
        check_gpyreg_against_pools(state["gpyreg"], manifests)
        verified = [stackable_selection(pool, only) for pool in args.pool]
        conditions, pools, labels = pool_conditions(
            args.pool, only, gpyreg_source=state["gpyreg"]
        )
        for pool, verification in zip(pools, verified):
            pool["verification"] = verification
        fixtures = []
    else:
        groups = [s.strip() for s in args.fixtures.split(",") if s.strip()]
        conditions, pools = fixture_conditions(groups), []
        labels = sorted(FIXTURE_PROBLEMS)
        fixtures = fixture_sources(conditions)
    reserved = sorted(set(conditions) & RESERVED_NAMES)
    if reserved:
        raise SystemExit(
            f"the conditions {reserved} would take a directory the "
            "campaign contract reserves"
        )
    indices = {label: labels.index(label) for label in conditions}
    plan, skipped = plan_cells(
        conditions, args.M, args.repetitions, args.seed, indices, arms_by_M
    )
    if not plan:
        raise SystemExit(
            "no cell to run: every requested M exceeds the filtered pools"
        )
    record = read_baseline_record()
    manifest = {
        "contract": contract.CONTRACT_VERSION,
        "campaign": CAMPAIGN,
        "kind": kind,
        "settings": comparison_settings(
            args.seed,
            args.M,
            args.repetitions,
            arms_by_M,
            args.max_steps,
            kind,
        ),
        "split_from": int(args.split_from),
        "conditions": [
            {
                "label": label,
                "index": indices[label],
                "origin": next(
                    (
                        pool["selection"]["conditions"][label]
                        for pool in pools
                        if label in pool["selection"]["conditions"]
                    ),
                    "the fixture group",
                ),
                "entries": [manifest_entry(entry) for entry in entries],
            }
            for label, entries in conditions.items()
        ],
        "plan": plan,
        "skipped": skipped,
        "pools": pools,
        "fixtures": fixtures,
        "baseline": {
            "record": BASELINE_RECORD.relative_to(ROOT).as_posix(),
            "commit": record["baseline"]["commit"],
            "version": record["baseline"]["version"],
            "files_sha256_committed": {
                name: entry["sha256"]
                for name, entry in record["baseline"][
                    "files_sha256_committed"
                ].items()
            },
            "torch_version": record["torch"]["version"],
            "verification": state["baseline"],
        },
        "gpyreg_source": str(state["gpyreg"]),
        "identity": source_identity,
        "allow_dirty": bool(args.allow_dirty),
        "site": contract.site_block(),
        "pip_freeze": contract.pip_freeze(),
        "finishing_steps": [["assemble"]],
        "tracked_copies": TRACKED_COPIES,
        "created": contract.now(),
    }
    contract.tracked_copies(manifest)
    path = out / "manifest.json"
    if path.exists():
        previous = contract.read_json(path)
        differing = [
            f"identity.{key}"
            for key in contract.source_differences(
                manifest["identity"], previous.get("identity")
            )
        ] + [
            key
            for key in STRUCTURAL_KEYS
            if manifest[key] != previous.get(key)
        ]
        if differing:
            raise SystemExit(
                f"{path} is prepared differently, in {differing}; prepare "
                "a new directory instead of changing a campaign in place"
            )
        print(f"{path}: prepared already, and unchanged", flush=True)
        return 0
    contract.write_json(path, manifest)
    tasks = campaign_tasks(manifest)
    print(
        f"{path}: {len(conditions)} conditions, {len(plan)} cells in "
        f"{len(tasks)} tasks; {describe_arms(args.M, arms_by_M)}"
        + (
            f"; computed from a dirty {', '.join(dirty)} tree" if dirty else ""
        ),
        flush=True,
    )
    for entry in skipped:
        print(
            f"  {entry['condition']} M={entry['M']}: {entry['drawn']} of "
            f"{entry['requested']} repetitions, the most that {entry['pool']} "
            "filtered runs give in disjoint subsets",
            flush=True,
        )
    return 0


def subset_tasks(name, tasks):
    """The case indices of a named subset of the tasks."""
    if name == "both":
        chosen = [i for i, t in enumerate(tasks, 1) if "original" in t["arms"]]
    elif name == "integrated":
        chosen = [
            i for i, t in enumerate(tasks, 1) if "original" not in t["arms"]
        ]
    elif name.startswith("M") and name[1:].isdigit():
        chosen = [i for i, t in enumerate(tasks, 1) if t["M"] == int(name[1:])]
    else:
        raise SystemExit(
            f"no subset {name!r}: the subsets are M<M>, both and integrated"
        )
    if not chosen:
        raise SystemExit(f"the subset {name!r} holds no task")
    return chosen


def cmd_cases(args):
    tasks = campaign_tasks(read_manifest(args.out.resolve()))
    if args.subset:
        for index in subset_tasks(args.subset, tasks):
            print(f"{index} {tasks[index - 1]['line']}")
    else:
        for task in tasks:
            print(task["line"])
    return 0


def write_cell(path, tag, row):
    """Write one cell's file, whole or not at all."""
    contract.write_json(path, {"campaign": CAMPAIGN, "task": tag, "cell": row})


def run_task(out, manifest, task, state, actual):
    """Run one task's cells; return its cell files and record fields."""
    kind = manifest["kind"]
    condition = task["condition"]
    conditions = {condition: manifest_conditions(manifest)[condition]}
    cells = [manifest["plan"][number] for number in task["cells"]]
    max_steps = int(manifest["settings"]["max_steps"])
    problems = Problems()
    written, original = [], None
    with contextlib.ExitStack() as stack:
        server = None
        if "original" in task["arms"]:
            server = stack.enter_context(
                OriginalArm(state["gpyreg"], state["checkout"])
            )
            differing = contract.source_differences(
                server.info["identity"], actual
            )
            if differing:
                raise RuntimeError(
                    "the original arm's subprocess runs with another source "
                    f"identity than this process, differing in {differing}"
                )
        warm = warm_up(
            cells[0], conditions, kind, server, problems, task["arms"]
        )
        for number, (cell, path) in enumerate(zip(cells, task["paths"])):
            print(
                f"START {condition} M={cell['M']} r={cell['repetition']} "
                f"({number + 1}/{len(cells)}, {cell['first_arm']} first)",
                flush=True,
            )
            row = run_cell(cell, conditions, kind, server, problems, max_steps)
            write_cell(out / path, task["tag"], row)
            written.append(out / path)
        if server is not None:
            server.close()
            original = server.summary()
    return written, {
        "cells": len(written),
        "warm_up_seconds": warm,
        "original_arm": original,
        "baseline": state["baseline"],
        "peak_rss_bytes": {
            "harness": peak_rss(),
            "original_arm": None
            if original is None
            else original["peak_rss_bytes"],
        },
    }


def task_of_line(manifest, line):
    """The task a line of ``cases`` names, or None."""
    for task in campaign_tasks(manifest):
        if task["line"] == line:
            return task
    return None


def cmd_worker(args):
    """One task under the contract's worker sequence; a line that is not a
    case of the campaign, or a directory that is not one of this harness's
    campaigns, is refused with exit 64 and touches nothing."""
    out = args.out.resolve()
    manifest = contract.read_json(out / "manifest.json")
    if manifest.get("campaign") != CAMPAIGN:
        print(
            f"{out / 'manifest.json'} is not a manifest of {CAMPAIGN}",
            file=sys.stderr,
            flush=True,
        )
        return EXIT_USAGE
    task = task_of_line(manifest, args.case)
    if task is None:
        print(
            f"{args.case!r} is not a case of {out}; the worker takes a line "
            "of `cases`",
            file=sys.stderr,
            flush=True,
        )
        return EXIT_USAGE
    with_baseline = "original" in task["arms"]
    state = {}

    def this_identity():
        state.update(campaign_process(with_baseline))
        return campaign_identity(state["gpyreg"], state["checkout"])

    return contract.run_worker(
        out,
        args.case,
        task_identity(manifest["identity"], with_baseline),
        this_identity,
        lambda actual: run_task(out, manifest, task, state, actual),
        lambda: [out / path for path in task["paths"]],
    )


def check_task(out, manifest, task, node_feature):
    """Re-check one completed task: its record, and every cell file
    against the manifest's plan. Raises
    :class:`campaign_contract.CompletionError`."""
    record = contract.check_completion(
        out,
        task["tag"],
        task_identity(manifest["identity"], "original" in task["arms"]),
        required=task["paths"],
        node_feature=node_feature,
    )
    problems = []
    unowned = sorted(set(record["artifacts"]) - set(task["paths"]))
    if unowned:
        problems.append(f"the record lists files of no cell: {unowned}")
    for number, path in zip(task["cells"], task["paths"]):
        try:
            stored = contract.read_json(out / path)
        except (OSError, ValueError) as error:
            problems.append(f"{path} cannot be read: {error}")
            continue
        if (
            stored.get("campaign") != CAMPAIGN
            or stored.get("task") != task["tag"]
        ):
            problems.append(f"{path} is not a cell file of this task")
            continue
        row, cell = stored["cell"], manifest["plan"][number]
        differing = [key for key in PLAN_KEYS if row.get(key) != cell[key]]
        if differing:
            problems.append(f"{path} is not the planned cell: {differing}")
        if sorted(row.get("arms", {})) != sorted(cell["arm_set"]):
            problems.append(f"{path} holds the arms {sorted(row['arms'])}")
        for arm, outcome in row.get("arms", {}).items():
            missing = [key for key in REFERENCE_FIELDS if key not in outcome]
            if missing:
                problems.append(f"{path}: the {arm} arm lacks {missing}")
    if problems:
        raise contract.CompletionError(task["tag"], problems)
    return {"cells": len(task["paths"])}


def stray_cells(out, manifest, tasks):
    """The cell files in the conditions' directories that no task owns."""
    owned = {path for task in tasks for path in task["paths"]}
    stray = []
    for condition in manifest_conditions(manifest):
        try:
            listing = sorted(os.scandir(out / condition), key=lambda e: e.name)
        except FileNotFoundError:
            continue
        for entry in listing:
            name = entry.name
            relative = f"{condition}/{name}"
            if (
                entry.is_file()
                and name.endswith(CELL_SUFFIX)
                and not name.startswith(".")
                and relative not in owned
            ):
                stray.append(relative)
    return stray


def campaign_report(out, manifest, tasks, query=None):
    """``verify``'s reconciliation of a campaign directory."""
    by_tag = {task["tag"]: task for task in tasks}
    node_feature = (manifest.get("site") or {}).get("NODE_FEATURE")

    def partial_files(tag):
        return [p for p in by_tag[tag]["paths"] if (out / p).exists()]

    return contract.reconcile(
        out,
        [task["line"] for task in tasks],
        lambda tag: check_task(out, manifest, by_tag[tag], node_feature),
        partial_files,
        stray=stray_cells(out, manifest, tasks),
        query=query,
    )


def cmd_verify(args):
    out = args.out.resolve()
    manifest = read_manifest(out)
    report = campaign_report(out, manifest, campaign_tasks(manifest))
    contract.write_json(
        out / "verification.json", {"campaign": CAMPAIGN, **report}
    )
    for case in report["cases"]:
        if case["status"] != "verified":
            print(json.dumps(case), flush=True)
    for name in report["stray"]:
        print(f"stray: {name}", flush=True)
    print(json.dumps(report["counts"]), flush=True)
    return report["exit_code"]


def cmd_assemble(args):
    """The finishing step: the comparison's outputs from its cell files.

    Refuses unless every task of the allocation verifies and no stray
    file lies in the directory, so that a partial campaign is never
    merged, and unless this process runs with the campaign's source
    identity (the baseline aside, which assembling does not import).
    """
    out = args.out.resolve()
    manifest = read_manifest(out)
    tasks = campaign_tasks(manifest)
    report = campaign_report(out, manifest, tasks)
    counts = report["counts"]
    if counts["verified"] != len(tasks) or report["stray"]:
        for case in report["cases"]:
            if case["status"] != "verified":
                print(json.dumps(case), flush=True)
        raise SystemExit(
            f"assemble merges a complete campaign only: {json.dumps(counts)}"
        )
    state = campaign_process(False)
    actual = campaign_identity(state["gpyreg"])
    differing = contract.source_differences(
        actual, task_identity(manifest["identity"], False)
    )
    if differing:
        raise SystemExit(
            "the source identity differs from the manifest's in "
            f"{differing}; the campaign is assembled by the code that ran it"
        )
    rows = [None] * len(manifest["plan"])
    records = {}
    for task in tasks:
        records[task["tag"]] = contract.read_json(
            contract.record_path(out, task["tag"])
        )
        for number, path in zip(task["cells"], task["paths"]):
            rows[number] = contract.read_json(out / path)["cell"]
    kind, settings = manifest["kind"], manifest["settings"]
    problems = Problems()
    singles = {
        condition: single_run_rows(condition, kind, entries, problems)
        for condition, entries in manifest_conditions(manifest).items()
    }
    elapsed = float(sum(r["elapsed_seconds"] for r in records.values()))
    contract.write_json(
        out / "results.json",
        {
            "campaign": "svbmc_pool",
            "kind": kind,
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": elapsed,
            "settings": settings,
            "skipped": manifest["skipped"],
            "cells": rows,
            "single_run": singles,
        },
    )
    write_summaries(out, rows, singles, settings)

    def host(record):
        found = (record.get("identity") or {}).get("host") or {}
        slurm = found.get("slurm") or {}
        return {
            "hostname": found.get("hostname"),
            "cpu_model": found.get("cpu_model"),
            "job": slurm.get("array_job_id") or slurm.get("job_id"),
            "array_task": slurm.get("array_task_id"),
        }

    contract.write_json(
        out / "sources.json",
        {
            "campaign": "svbmc_pool",
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "harness": {
                "path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
                "sha256": contract.sha256_file(Path(__file__).resolve()),
            },
            "settings": settings,
            "manifest_sha256": contract.sha256_file(out / "manifest.json"),
            "verification": counts,
            "identity": actual,
            "baseline": manifest["baseline"],
            "gpyreg_source": str(state["gpyreg"]),
            "pools": manifest["pools"],
            "fixtures": manifest["fixtures"],
            "tasks": {
                tag: {
                    "elapsed_seconds": record["elapsed_seconds"],
                    "host": host(record),
                    "warm_up_seconds": record.get("warm_up_seconds"),
                    "peak_rss_bytes": record.get("peak_rss_bytes"),
                    "original_arm": record.get("original_arm"),
                }
                for tag, record in records.items()
            },
        },
    )
    print(
        f"{len(rows)} cells of {len(tasks)} tasks assembled -> {out}",
        flush=True,
    )
    return 0


COMMANDS = {
    "prepare": cmd_prepare,
    "cases": cmd_cases,
    "worker": cmd_worker,
    "verify": cmd_verify,
    "assemble": cmd_assemble,
}


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "--serve-original":
        return serve_original(argv[1], argv[2])
    # The driver reads these outputs in bash, which takes a carriage return
    # for part of a word.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(newline="\n")
        except AttributeError:
            pass
    if argv and argv[0] in COMMANDS:
        args = parse_campaign_args(argv)
        return COMMANDS[args.command](args)
    args = parse_args(argv)
    if args.summarize_only:
        return summarize_only(args)
    return run_comparison(args)


if __name__ == "__main__":
    sys.exit(main())
