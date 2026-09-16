"""Matched comparison of the two S-VBMC implementations on pooled runs.

The integrated ``pyvbmc.svbmc.SVBMC`` and the original standalone ``svbmc``
0.1.1 stack the same subsets of the same finished VBMC runs, cell by cell,
and every cell records the optimized weights, every ELBO variant, the
entropy, the seconds spent, and the quality of 100 000 draws from the
stacked posterior against the target's truth. The design, the conditions,
the grid and the acceptance criteria are
``dev/plans/svbmc-benchmark-campaign.md``, sections "Stacking comparison",
"Metrics" and "Acceptance criteria for the comparison"; the pools come from
``svbmc_pool_run.py`` and are read through ``svbmc_pool_io.load_run``.

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

Usage. ``TORCH_PATH`` is the Torch overlay recorded in
``dev/experiments/svbmc_pool/baseline_environment.json``, the only Torch on
the campaign's machine, so every invocation that runs a cell needs it::

    PYTHONPATH="<TORCH_PATH>" python -u dev/scripts/svbmc_pool_stack.py \\
        --pool DIR --out DIR [--conditions L1,L2] [--M 2,4,8,16] \\
        [--repetitions 20,20,20,10] [--seed 0] [--max-steps 500] \\
        [--gpyreg-source DIR] [--arms both|integrated] [--overwrite]
    PYTHONPATH="<TORCH_PATH>" python -u dev/scripts/svbmc_pool_stack.py \\
        --fixtures upstream_Ring --out DIR --M 2,3 --repetitions 2,2 \\
        --max-steps 3
    python dev/scripts/svbmc_pool_stack.py --summarize-only --out DIR \\
        [--from-results R1.json --from-results R2.json]

A condition's filtered pool is the runs its pool directory's
``selection.json`` names, written by ``svbmc_pool_run.py select``, which
is the campaign's stopping rule applied to the generated runs; a directory
without one contributes every run whose completion record says it passed
the filters. The line printed for each condition and ``sources.json`` say
which of the two it was.

A cell is one subset of ``M`` runs drawn without replacement from a
condition's filtered pool by ``np.random.default_rng([seed,
condition_index, M, repetition])``, where ``condition_index`` is the
condition's place among every label the pools hold, so that the subsets do
not depend on which conditions an invocation selects.
``np.random.SeedSequence`` on the same key yields the cell's seed, and
spawning that sequence yields one seed per entry of the subset, with which
both arms rebuild the cell's posteriors: the original draws each run's
block of samples from that run's own generator, which one shared seed
would couple. Both arms receive the subset in the same order and never run
at the same time; which arm runs first alternates from cell to cell. The
integrated arm runs in this process, seeded through
``SVBMC(seed=cell_seed)``; the original arm runs in one long-lived worker
subprocess, seeded by ``np.random.seed(cell_seed)`` immediately before
construction, because it draws its entropy samples from NumPy's global
legacy stream and its posterior samples from the input posteriors' own
generators.

``--arms integrated`` runs the integrated arm alone, for the larger-``M``
regime where the original's cost, quadratic in ``M`` and two to four
times the integrated arm's, is not worth paying: every cell then carries
one arm and no paired quantity, and the summaries carry the integrated
arm's medians, biases and headline-bias growth. ``--summarize-only``
takes several ``--from-results`` files and summarizes their cells
together, so a run of both arms up to one ``M`` and a run of the
integrated arm beyond it give one summary, whose paired quantities and
equivalence tests cover the cell sets both arms ran and whose
headline-bias growth spans every ``M``.

The two arms need different import paths, both recorded in
``dev/experiments/svbmc_pool/baseline_environment.json``, which this script
re-verifies (commit, clean tree, file hashes, Torch version) before any
cell runs: the controller carries only the Torch overlay, so that no
``import svbmc`` here can reach the pinned upstream package, and the worker
carries the overlay and the upstream source. Both arms import gpyreg from
the campaign's frozen worktree through ``PYVBMC_GPYREG_SOURCE`` and record
where it resolved. A pool's manifest names that worktree as an absolute
path on the machine that generated the pool; for a pool copied from
another machine, ``--gpyreg-source`` names a local clean checkout at the
manifest's gpyreg commit (``svbmc_pool_run.pinned_gpyreg_source``), which
is checked before PyVBMC is imported, and ``sources.json`` records which
of the two named the checkout.

``--fixtures`` presents the shipped S-VBMC posterior fixtures
(``pyvbmc/testing/svbmc/fixtures/``) as pools, one condition per fixture
group, scored against the ported target the group's runs used. It needs no
pool directory and is what ``test_svbmc_pool_stack.py`` exercises.

Outputs under ``--out``: ``cells.jsonl`` (one line per finished cell,
written as the sweep goes), ``results.json`` (every cell, the single-run
rows and the settings), ``summary.json`` and ``summary.md`` (per condition
and ``M``: medians with 10 000-resample bootstrap 95 % intervals of the
biases, the KL gap, the metrics, the maximum weight difference and the
runtime ratio, the paired differences integrated minus original with
exact signed-rank tests on the metrics, and criterion 3's two gates,
``headline_bias_not_worse`` per ``M`` and ``headline_bias_growth`` per
condition),
``sources.json`` (both arms' commits, working-tree state, import paths,
versions and thread settings, the baseline re-verification, the pools' or
fixtures' identities and this file's SHA-256) and ``original_arm.log``
(everything the worker wrote to its standard streams).
``--summarize-only`` rebuilds ``summary.json`` and ``summary.md`` from a
finished ``results.json`` without running a cell. It describes that
comparison by the settings the file records — the draw counts, the
entropy reference, the bootstrap and the stacking call — and falls back
to this module's constants only for a field a results file written before
it existed does not carry, so a rebuilt summary never attributes today's
constants to an older run.
"""

import argparse
import inspect
import json
import logging
import os
import platform
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
# NumPy is imported, so this import stays above the one below.
from svbmc_pool_run import (  # noqa: E402
    DEFAULT_GPYREG,
    THREAD_KEYS,
    activate_gpyreg,
    git,
    identity,
    pinned_gpyreg_source,
    write_json,
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
#: The variant each arm reports as its ELBO; criterion 3 compares the
#: biases of these two.
HEADLINE_VARIANT = {"integrated": "headline", "original": "estimated"}


def sha256(path):
    # Imported here rather than at module level: svbmc_pool_io imports
    # PyVBMC and with it gpyreg, which must not happen before
    # `activate_gpyreg` has put the campaign's frozen worktree on the path.
    import svbmc_pool_io as pool_io

    return pool_io.sha256(path)


# --------------------------------------------------------------------------
# Baseline environment
# --------------------------------------------------------------------------


def verify_baseline(record, path):
    """Re-verify the recorded original-S-VBMC baseline; return the report.

    The checkout must sit at the recorded commit with a clean tree and the
    recorded SHA-256 for every file of the package, and the Torch overlay
    must still hold the recorded version. Raises ``RuntimeError`` naming
    every mismatch: the baseline is the comparison's other arm, so a
    campaign that cannot identify it must not start.
    """
    import torch

    baseline = record["baseline"]
    checkout = Path(baseline["checkout"])
    source = Path(baseline["source_dir"])
    failures = []
    if not source.is_dir():
        raise RuntimeError(f"the baseline checkout is missing: {source}")
    commit = git(checkout, "rev-parse", "HEAD")
    if commit != baseline["commit"]:
        failures.append(f"commit {commit}, recorded {baseline['commit']}")
    status = git(checkout, "status", "--porcelain")
    if status:
        failures.append(f"the checkout has uncommitted changes: {status}")
    files = {}
    for name, stored in baseline["files_sha256"].items():
        actual = sha256(source / name)
        files[name] = actual
        if actual != stored["sha256"]:
            failures.append(f"{name} differs from the recorded hash")
    if torch.__version__ != record["torch"]["version"]:
        failures.append(
            f"Torch {torch.__version__}, recorded {record['torch']['version']}"
        )
    if failures:
        raise RuntimeError(
            f"{path} does not re-verify: " + "; ".join(failures)
        )
    return {
        "record": str(path),
        "commit": commit,
        "clean": True,
        "files_sha256": files,
        "torch_version": torch.__version__,
        "torch_import": str(Path(torch.__file__).resolve()),
        "verified": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def path_sets(record):
    """``(TORCH_PATH, BASELINE_PATH)`` of the plan, from the record."""
    overlay = str(Path(record["torch"]["overlay_dir"]).resolve())
    upstream = str(Path(record["baseline"]["source_dir"]).resolve().parent)
    return overlay, os.pathsep.join([overlay, upstream])


def refuse_upstream_on_path(baseline_path):
    """The controller must not be able to ``import svbmc`` from upstream."""
    upstream = baseline_path.split(os.pathsep)[-1]
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
            f"{upstream} is on this process's import path; the controller "
            "must carry only the Torch overlay, so that the integrated "
            "class is the only S-VBMC it can import"
        )


# --------------------------------------------------------------------------
# Pools, entries and problems
# --------------------------------------------------------------------------


def pool_entry(directory, tag, origin="the pool directory"):
    """One filtered run of a pool, from its completion record.

    ``origin`` names the list the tag came from, so that a run whose
    record or artifact the directory does not hold is reported as the
    wrong entry of that list rather than as an absent file. A pool is
    generated once and read many times, and the case that leads here is a
    selection copied without the artifacts it names, or one whose run
    failed after the selection was written.
    """
    import svbmc_pool_io as pool_io

    record_file = pool_io.record_path(directory, tag)
    if not record_file.exists():
        raise RuntimeError(
            f"{origin} names {tag}, which has no completion record "
            f"({record_file}): the case failed or was never generated"
        )
    record = json.loads(record_file.read_text(encoding="utf-8"))
    missing = [
        path.name
        for path in pool_io.artifact_paths(directory, record["tag"])
        if not path.exists()
    ]
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
    }


def pool_conditions(pool_dirs, only=None, gpyreg_source=None):
    """The filtered runs of every condition, ordered by seed.

    One entry per run of the condition's filtered pool, carrying the
    artifact path (without suffix) and the metrics recorded for that single
    run, which are the comparison's ``M = 1`` rows.

    The pools must have been generated against one gpyreg source. With
    ``gpyreg_source`` given, a local checkout that :func:`main` has already
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
        manifest = json.loads(
            (directory / "manifest.json").read_text(encoding="utf-8")
        )
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
            "identity": manifest["identity"],
            "gpyreg_source": manifest["gpyreg_source"],
            "options": manifest["options"],
            "selection": {
                "path": str(selection_path) if selection else None,
                "generated": selection["generated"] if selection else None,
                "conditions": {},
            },
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
                for path in sorted(
                    (directory / "records").glob(
                        f"{label}_seed*.complete.json"
                    )
                ):
                    entry = pool_entry(
                        directory,
                        path.name[: -len(".complete.json")],
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
            f"{sorted(sources)}"
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
                {
                    "name": entry["name"],
                    "sha256": {
                        suffix: sha256(FIXTURES_DIR / (entry["name"] + suffix))
                        for suffix in (".npz", ".json")
                    },
                }
                for entry in entries
            ],
        }
        for condition, entries in conditions.items()
    ]


def load_entry(entry, rng):
    """Rebuild one pool entry's posterior with its own generator."""
    if entry["kind"] == "fixture":
        from pyvbmc.testing.svbmc._fixtures import load_vp

        return load_vp(entry["name"], rng=rng)[0]
    from svbmc_pool_io import load_run

    return load_run(entry["path"], rng=rng)["vp"]


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
# The original arm's worker
# --------------------------------------------------------------------------


def worker_main(gpyreg_source):
    """Serve stacking requests for the original implementation.

    Requests and replies are one JSON object per line. The original
    implementation prints progress and warnings, so file descriptor 1 is
    pointed at stderr (which the controller captures into a log) and the
    protocol keeps a private copy of the real stdout.
    """
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr

    activate_gpyreg(gpyreg_source)
    import torch

    torch.set_num_threads(1)
    import svbmc

    def send(payload):
        protocol.write(json.dumps(payload) + "\n")
        protocol.flush()

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


class Worker:
    """The original implementation, served by one long-lived subprocess."""

    def __init__(self, baseline_path, gpyreg_source, log_path):
        environment = dict(os.environ)
        environment["PYTHONPATH"] = baseline_path
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYVBMC_GPYREG_SOURCE"] = str(gpyreg_source)
        environment["MPLBACKEND"] = "Agg"
        environment.update({key: "1" for key in THREAD_KEYS})
        self.log = Path(log_path).open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "--worker",
                str(gpyreg_source),
            ],
            cwd=str(ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            env=environment,
            text=True,
            bufsize=1,
        )
        self.info = self._read()
        if not self.info.get("ready"):
            raise RuntimeError(f"the original arm did not start: {self.info}")

    def _read(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(
                "the original arm's worker exited; see its log for the "
                "traceback"
            )
        return json.loads(line)

    def request(self, payload):
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()
        reply = self._read()
        if "error" in reply:
            raise RuntimeError("the original arm failed:\n" + reply["error"])
        return reply

    def close(self):
        try:
            self.request({"op": "quit"})
        except (OSError, ValueError, RuntimeError):
            pass
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.log.close()


# --------------------------------------------------------------------------
# The sweep
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


def plan_cells(conditions, grid, repetitions, seed, indices):
    """Every cell of the comparison, in the order it will be run.

    ``indices`` gives each condition its place among all the labels the
    pools hold, so that the subsets drawn for a condition do not depend on
    which conditions this invocation selected.
    """
    plan, skipped = [], []
    for condition, entries in conditions.items():
        index = indices[condition]
        for M, R in zip(grid, repetitions):
            if M > len(entries):
                skipped.append(
                    {"condition": condition, "M": M, "pool": len(entries)}
                )
                continue
            for repetition in range(R):
                key = [seed, index, M, repetition]
                subset = sorted(
                    int(i)
                    for i in np.random.default_rng(key).choice(
                        len(entries), M, replace=False
                    )
                )
                cell_seed = int(
                    np.random.SeedSequence(key).generate_state(
                        1, dtype=np.uint32
                    )[0]
                )
                plan.append(
                    {
                        "condition": condition,
                        "condition_index": index,
                        "M": int(M),
                        "repetition": int(repetition),
                        "indices": subset,
                        "entries": [entries[i]["name"] for i in subset],
                        "seeds": [entries[i]["seed"] for i in subset],
                        "cell_seed": cell_seed,
                        "entry_seeds": entry_seeds(cell_seed, M),
                    }
                )
    return plan, skipped


def fit_cell(
    arm,
    worker,
    condition,
    kind,
    entries,
    seeds,
    cell_seed,
    max_steps,
    problem,
    ref,
):
    """One arm of one cell: in this process, or through the worker.

    Returns the arm's record and, for the integrated arm, the fitted
    object that :func:`add_reference` scores the cell with; the original
    arm's object lives in the worker process and never crosses back.
    """
    if arm == "integrated":
        return fit_integrated(
            entries, seeds, cell_seed, max_steps, problem, ref
        )
    return (
        worker.request(
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


def warm_up(plan, conditions, kind, worker, problems, arms=ARMS):
    """One short discarded fit per arm, so no cell pays the first-call cost.

    Both implementations build their Torch graph, import what they need
    lazily and touch the sampling and metric paths on their first fit, which
    would otherwise land on the first recorded cell and inflate its
    optimization seconds. The fit is two Adam steps on the first two runs of
    the first cell and nothing about it is recorded; every recorded cell
    rebuilds its posteriors and reseeds both arms, so it leaves no trace.
    """
    cell = plan[0]
    condition = cell["condition"]
    entries = [conditions[condition][i] for i in cell["indices"][:2]]
    for arm in arms:
        started = time.perf_counter()
        fit_cell(
            arm,
            worker,
            condition,
            kind,
            entries,
            cell["entry_seeds"][:2],
            cell["cell_seed"],
            2,
            problems.get(condition, kind),
            problems.reference(condition, kind),
        )
        print(
            f"warmed up {arm} in {time.perf_counter() - started:.1f} s",
            flush=True,
        )


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


def run_cells(
    plan, conditions, kind, worker, problems, max_steps, out, arms=ARMS
):
    """Run every planned cell, its arms one at a time; return the rows.

    With both arms, which arm goes first alternates from cell to cell and
    the cell records their pairing; with the integrated arm alone, the
    cell carries that arm's record and no paired quantity.
    """
    progress = (out / "cells.jsonl").open("w", encoding="utf-8")
    rows = []
    started = time.time()
    try:
        for number, cell in enumerate(plan):
            condition = cell["condition"]
            entries = [conditions[condition][i] for i in cell["indices"]]
            problem = problems.get(condition, kind)
            reference = problems.reference(condition, kind)
            first = arms[number % len(arms)]
            print(
                f"START {condition} M={cell['M']} r={cell['repetition']} "
                f"({number + 1}/{len(plan)}, {first} first, "
                f"{(time.time() - started) / 60:.1f} min elapsed)",
                flush=True,
            )
            outcomes, stacked = {}, None
            for arm in (first, *[a for a in arms if a != first]):
                outcomes[arm], fitted = fit_cell(
                    arm,
                    worker,
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
            row = dict(cell, first_arm=first, arms=outcomes)
            where = f"{condition} M={cell['M']} r={cell['repetition']}"
            row["max_abs_dw"] = (
                check_pairing(where, outcomes)
                if set(arms) == set(ARMS)
                else None
            )
            reference_started = time.perf_counter()
            add_reference(row, stacked, problem, cell["cell_seed"])
            row["reference_seconds"] = time.perf_counter() - reference_started
            rows.append(row)
            progress.write(json.dumps(row) + "\n")
            progress.flush()
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
        f"subsets drawn with seed {settings['seed']}. Medians over the "
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
# Main
# --------------------------------------------------------------------------


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
        help="the gpyreg checkout both arms import: for --fixtures runs, "
        "which have no pool manifest to name one (default: "
        f"{DEFAULT_GPYREG}); for --pool runs, a local checkout in place of "
        "the path the manifest names, for a pool copied from another "
        "machine, accepted only as a clean checkout at the manifest's "
        "gpyreg commit",
    )
    parser.add_argument(
        "--baseline-record", type=Path, default=BASELINE_RECORD
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
        choices=("both", "integrated"),
        default="both",
        help="which implementations stack every cell: both (default), or "
        "the integrated class alone, for the larger-M regime where the "
        "original's cost is not worth paying",
    )
    args = parser.parse_args(argv)
    args.M = [int(v) for v in args.M.split(",") if v.strip()]
    args.repetitions = [
        int(v) for v in args.repetitions.split(",") if v.strip()
    ]
    if len(args.M) != len(args.repetitions):
        parser.error("--M and --repetitions must have the same length")
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
    ``repetitions`` and ``arms`` widened to cover every file and a
    ``merged_from`` record of each file, and the single-run rows are the
    first file's, since every run scores the same pools. Files that
    differ in any setting they share other than ``M``, ``repetitions``
    and ``arms``, that hold the same cell twice, or that hold cells of
    one condition and ``M`` with different arm sets, cannot be
    summarized together; the repetitions of an ``M`` are counted from
    the merged cells.
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
        free = {"M", "repetitions", "arms", "merged_from"}
        for path, result in results[1:]:
            for key in sorted(set(settings) & set(result["settings"]) - free):
                if result["settings"][key] != settings[key]:
                    raise RuntimeError(
                        f"{path} was run with {key}="
                        f"{result['settings'][key]!r} and {first_path} "
                        f"with {settings[key]!r}; their cells cannot be "
                        "summarized together"
                    )
        seen, arm_sets = {}, {}
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
                held = arm_sets.setdefault(key[:2], (arms, path))
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


def summarize_only(args):
    """Rebuild the summaries of a finished comparison, running no cell."""
    out = args.out.resolve()
    paths = [
        path.resolve()
        for path in (args.from_results or [out / "results.json"])
    ]
    cells, single_run, settings = load_results(paths)
    out.mkdir(parents=True, exist_ok=True)
    summary = build_summary(
        cells, single_run, settings, np.random.default_rng(settings["seed"])
    )
    write_json(out / "summary.json", summary)
    text = summary_markdown(summary)
    (out / "summary.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(
        f"{len(cells)} cells from {', '.join(str(p) for p in paths)} -> {out}",
        flush=True,
    )
    return 0


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "--worker":
        return worker_main(argv[1])
    args = parse_args(argv)
    if args.summarize_only:
        return summarize_only(args)
    out = args.out.resolve()
    if (out / "results.json").exists() and not args.overwrite:
        raise RuntimeError(f"{out} already holds a comparison; --overwrite")
    out.mkdir(parents=True, exist_ok=True)

    kind = "fixture" if args.fixtures else "run"
    only = [s.strip() for s in (args.conditions or "").split(",") if s.strip()]
    if kind == "run":
        manifests = [
            json.loads(
                (Path(pool).resolve() / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            for pool in args.pool
        ]
        if args.gpyreg_source:
            # A pool copied from another machine: the local checkout must
            # be at every pool's gpyreg commit, the library the pools'
            # numerics were produced by.
            for manifest in manifests:
                gpyreg_source = pinned_gpyreg_source(
                    args.gpyreg_source, manifest
                )
            gpyreg_origin = "--gpyreg-source"
        else:
            gpyreg_source = manifests[0]["gpyreg_source"]
            gpyreg_origin = "the pool manifest"
        gpyreg_source = activate_gpyreg(gpyreg_source)
    else:
        gpyreg_source = activate_gpyreg(args.gpyreg_source or DEFAULT_GPYREG)
        gpyreg_origin = (
            "--gpyreg-source" if args.gpyreg_source else "the default"
        )

    baseline = json.loads(
        Path(args.baseline_record).read_text(encoding="utf-8")
    )
    torch_path, baseline_path = path_sets(baseline)
    refuse_upstream_on_path(baseline_path)
    try:
        import torch
    except ImportError as error:  # the overlay is the only Torch here
        raise RuntimeError(
            "the integrated class needs Torch: start this script with "
            f'PYTHONPATH="{torch_path}"'
        ) from error

    torch.set_num_threads(1)
    logging.getLogger("SVBMC").setLevel(logging.WARNING)
    import svbmc_pool_io as pool_io

    # This raises unless PyVBMC imported gpyreg from the campaign's frozen
    # worktree, so the controller's own environment is settled before the
    # baseline's is checked and long before a cell runs.
    integrated_identity = identity(gpyreg_source)
    verification = verify_baseline(baseline, args.baseline_record)

    if kind == "run":
        conditions, pool_identities, labels = pool_conditions(
            args.pool, only, gpyreg_source=args.gpyreg_source
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
        conditions, args.M, args.repetitions, args.seed, indices
    )
    for entry in skipped:
        print(
            f"skipping {entry['condition']} M={entry['M']}: the filtered "
            f"pool holds {entry['pool']} runs",
            flush=True,
        )
    if not plan:
        raise RuntimeError(
            "no cell to run: every requested M exceeds the filtered pools"
        )
    arms = ARMS if args.arms == "both" else ("integrated",)
    print(
        f"{len(plan)} cells, "
        + ("both arms" if len(arms) == 2 else "the integrated arm alone"),
        flush=True,
    )

    worker = (
        Worker(baseline_path, gpyreg_source, out / "original_arm.log")
        if "original" in arms
        else None
    )
    started = time.time()
    # Process-scoped request, as in the pool runner's sweep: permit display
    # sleep, prevent idle system sleep while the cells run, since a full
    # grid takes hours on a laptop.
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        warm_up(plan, conditions, kind, worker, problems, arms)
        rows = run_cells(
            plan, conditions, kind, worker, problems, args.max_steps, out, arms
        )
    finally:
        if worker is not None:
            worker.close()
        if sys.platform == "win32":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    elapsed = time.time() - started

    settings = {
        "seed": args.seed,
        "M": args.M,
        "repetitions": args.repetitions,
        "max_steps": args.max_steps,
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
        "arms": list(arms),
    }
    write_json(
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
    summary = build_summary(
        rows, singles, settings, np.random.default_rng(args.seed)
    )
    write_json(out / "summary.json", summary)
    text = summary_markdown(summary)
    (out / "summary.md").write_text(text, encoding="utf-8")
    write_json(
        out / "sources.json",
        {
            "campaign": "svbmc_pool",
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "harness": {
                "path": str(Path(__file__).resolve()),
                "sha256": pool_io.sha256(Path(__file__).resolve()),
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
                "original": None if worker is None else worker.info,
            },
            "path_sets": {
                "TORCH_PATH": torch_path,
                "BASELINE_PATH": baseline_path,
            },
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


if __name__ == "__main__":
    sys.exit(main())
