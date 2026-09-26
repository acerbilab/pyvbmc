"""Run a population of VBMC runs, keeping each run's final-boost decision.

A population is a set of cases, each one VBMC run of a configuration of the
targets module (``benchmark_targets.py``, a label of one of its suites) at
one seed, made by ``golden_trace.run_task`` with the campaign's options.
Every case runs in a fresh process. The production final boost of every run
is observed (:class:`BoostCapture`) without changing its calculation or its
random stream, so that ``analyze_population_run.py`` can check every boost
decision and compare other guards offline.

The harness meets the campaign contract of
``dev/plans/slurm-benchmark-support.md`` through ``campaign_contract.py``:
the driver under ``dev/scripts/hpc/`` runs a population as Slurm job
arrays, and ``run`` runs the same cases one after another on a
workstation::

    python dev/scripts/population_run.py prepare --out DIR \\
        [--suite production] [--seeds 0-99] [--labels L1,L2] \\
        [--options JSON] [--arm NAME] [--confirmatory FILE] [--pair DIR]
    python dev/scripts/population_run.py cases --out DIR [--subset NAME]
    python dev/scripts/population_run.py worker --out DIR --case LINE
    python dev/scripts/population_run.py verify --out DIR
    python dev/scripts/population_run.py summarize --out DIR
    python dev/scripts/population_run.py rescore --out DIR [--campaign DIR]
    python dev/scripts/population_run.py run --out DIR [--subset NAME] \\
        [--limit N]

**Source trees.** A campaign's code comes from three trees, and the source
identity that every worker compares holds the commit and the clean state of
each: the harness checkout (this file's; it supplies ``golden_trace.py``,
``profile_run.py``, the targets module and its data in every campaign), the
package tree and gpyreg. The package tree is the checkout that
``PYVBMC_SOURCE`` names (for code other than the harness checkout's, a
detached worktree at that commit), or the harness checkout itself when the
variable is unset; gpyreg is the checkout that ``PYVBMC_GPYREG_SOURCE``
names, which every process needs. This module puts the harness checkout,
then gpyreg, first on ``sys.path``; run as a script, it then puts the
package tree ahead of them and imports PyVBMC from it
(:func:`use_package_tree`) before it imports ``golden_trace.py`` and
``profile_run.py``, and when PyVBMC does not import from that tree it
exits 78, the contract's exit code for an identity that cannot be
established, having written nothing. No other script reads
``PYVBMC_SOURCE``, and a process
that imports this module (the analysis, the tests) takes PyVBMC from the
harness checkout. The identity refuses a process whose ``pyvbmc`` or
``gpyreg`` resolves outside its tree, or whose harness modules come from
anywhere but this directory. It also hashes the harness
modules, the targets module and ``dev/scripts/data``. The sidecar of every
case holds the ``source`` and ``imports`` parts of its process's identity
under ``provenance``: each tree's commit and path, the import paths, the
imported versions, and the versions that the installed distributions name,
labelled as such.

**Cases and files.** ``prepare`` fixes the allocation (a suite, its labels,
all of them by default, and one seed range for every label), the options
(:data:`DEFAULT_OPTIONS` and ``--options``) and the confirmatory family of
the comparison of two arms (:func:`confirmatory_family`), and names the
finishing steps and the tracked copies (:func:`tracked_copies`, which
``campaign_contract.py redact`` writes for the repository). Case ``i``
is line ``i`` of the list that ``cases`` prints,
``<label>/<label>_seed<seed> <label> <seed>``, the labels in suite order
and the seeds rising within each; the subsets are each label,
``noisy``, ``noiseless`` and ``canary`` (the first seed of every label).
A case writes, relative to the campaign directory::

    <label>/<label>_seed<seed>.npz          the trace (golden_trace.py)
    <label>/<label>_seed<seed>.json         its sidecar, with provenance
    <label>/<label>_seed<seed>.vp.npz       what the trace lacks to rebuild
                                            the returned posterior
    boost/<label>/<label>_seed<seed>.boost.pkl    the boost capture (dill)
    boost/<label>/<label>_seed<seed>.boost.json   its decision, scores and
                                                  metrics

with its completion record, claim and error file where the contract puts
them. The worker refuses with exit 64, touching nothing, a line that is
not a case of the allocation and a directory without a readable manifest
of this harness. Each configuration's directory is a population directory of
``golden_trace.py`` (``summary``, ``load_population``, the ``--baseline``
of ``golden_replay.py``); the boost files lie apart from it, out of reach of
its sidecar discovery (``*_seed*.json``).

**The returned posterior as plain arrays.** The trace holds the returned
posterior's weights, means, scales and axis lengths (``final_w``,
``final_mu``, ``final_sigma``, ``final_lambd``) and every iteration's
transformer centre, width, scale and rotation (``pt_mu``, ``pt_delta``,
``pt_scale``, ``pt_R``). The returned posterior is the one of the
sidecar's ``best_iter``, boosted or not, and the boost leaves its
transformer as it is. The trace lacks the family of the bounded transform
and whether the transformer rescales and rotates at all, since it stores
ones and the identity where it does not; ``<stem>.vp.npz`` holds those
(:func:`posterior_extras`). With the configuration's bounds they rebuild
the posterior in whatever package is imported (:func:`returned_posterior`).
The worker fails a case unless the rebuilt posterior equals the one the run
returned, in every array and transformer attribute, and scores exactly as
the run's own metrics say, and unless its artifacts pass
:func:`check_case_artifacts` (finite metrics, 750 evaluations for
``cigar_D15_exhaust``, the campaign's options requested and applied, a
boost report consistent with its capture): such a case gets an error file
and no completion record. ``verify`` repeats the checks.

**Verification.** ``verify`` runs in the campaign's own trees (a different
source identity is refused, exit 78), since the boost captures unpickle
only in the package that wrote them. It reconciles the allocation by the
contract's states, re-checks each completion record (its identity, the
SHA-256 of every artifact, the node feature and the CPU affinity where the
manifest's site block names a node feature) and each case's artifacts
(:func:`check_case_artifacts`), and writes ``verification.json``, which
holds the SHA-256 of each verified case's record. What reads the report
(``summarize``, ``rescore``, the comparison of two arms) takes it only as
a description of the directory as it stands (:func:`checked_verification`):
a report written while cases were missing or in flight, such as that of a
finish that looked at the canary, is refused once one of those cases has a
record, until the campaign is verified again. ``summarize`` tabulates the
verified cases alone.

**Rescoring.** The in-run metrics come from the package each run imported,
so two arms of different code are compared on metrics recomputed by one
code. ``rescore`` runs in a process of the harness checkout's own package
(the release code) and in the trees and environment of the campaign at
``--out`` (its source identity is the manifest's), and refuses any other,
and a process without ``PYVBMC_GPYREG_SOURCE``, with exit 78.
For each verified case of that campaign and of each ``--campaign`` it
rebuilds the returned posterior from its plain arrays and recomputes the
evidence error, gsKL, MMTV and the RMSE of the mean with
``benchmark_targets.metrics``, whose random draws come from generators of
fixed seeds, as they do in the run; the evidence error uses the run's own
ELBO. The files it reads must be the ones verification checked. A campaign
of the rescoring's own code must reproduce every case's in-run metrics
exactly, or the step fails. It writes ``<out>/rescored/<campaign directory
name>.json``, with the rescoring process's identity and, for each case, the
metrics, whether each equals the run's own and whether each is finite; and
then ``<out>/rescoring.json`` (:data:`RESCORING`), with that identity and
the SHA-256 of each file of rescored metrics, which the comparison checks.
Its work files, ``<out>/rescored/<name>.parts/<label>.json``, are written
after every case, and a later ``rescore`` takes from them every case whose
files are unchanged and that code of the same source identity rescored, so
that a finish run again rescores only what it lacks.

A campaign of the release code lists ``rescore`` among its finishing
steps, with ``--campaign`` for the campaign given to ``prepare --pair``
(the other arm); a campaign of other code lists only ``summarize``.
``prepare --pair`` refuses a campaign that is not the other arm of this
one: another allocation, options or confirmatory family, another harness
checkout, other harness files or environment versions, or a package tree
at the same commit. Since ``rescore`` reads the other arm's
``verification.json``, the other arm is finished completely before this
one's finish, and this one is finished again after any later finish of
the other.

**Workstation runs.** ``run`` works on a prepared directory: it writes
``cases.txt``, runs each case of the list (or of ``--subset``, the first
``--limit`` of them) that has no completion record as a fresh ``worker``
process with its log under ``logs/``, stops at the first case that does
not complete, and, once every case of the allocation has a record, runs
``verify`` and the finishing steps. It holds ``campaign.lock`` and, on
Windows, keeps the system from sleeping while it works.

**Records of earlier campaigns.** Campaigns that ran before this harness
met the contract (a launch manifest, ``launch.json``, one flat directory,
records of four hashes) are checked by :func:`validate_case`.
"""

import argparse
import contextlib
import copy
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
for key in THREAD_KEYS:
    os.environ[key] = "1"
os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, str(ROOT))
if os.environ.get("PYVBMC_GPYREG_SOURCE"):
    sys.path.insert(0, os.environ["PYVBMC_GPYREG_SOURCE"])


def use_package_tree(environ=None):
    """Import PyVBMC from the campaign's package tree, before anything else.

    The tree that ``PYVBMC_SOURCE`` names goes first on ``sys.path``, and
    PyVBMC is imported at once, from that tree or, when the variable is
    unset, from the harness checkout, so that the harness modules, which
    put the harness checkout first on ``sys.path`` when they are imported,
    find the package imported already. Only this module, run as a script,
    calls it; a process that imports this module takes PyVBMC from the
    harness checkout whatever ``PYVBMC_SOURCE`` says, and its identity
    (:func:`this_identity`) then refuses a tree that the variable names.
    """
    tree = (os.environ if environ is None else environ).get("PYVBMC_SOURCE")
    if tree:
        sys.path.insert(0, tree)
    package = importlib.import_module("pyvbmc")
    where = Path(package.__file__).resolve().parents[1]
    if tree and where != Path(tree).resolve():
        raise ImportError(
            f"PyVBMC is imported from {where}, not from PYVBMC_SOURCE={tree}"
        )


if __name__ == "__main__":
    try:
        use_package_tree()
    except Exception as error:  # noqa: BLE001
        # No identity can be established without the package tree; the
        # contract's exit code for that, and no error file (the case's
        # files are not touched).
        import traceback

        from campaign_contract import EXIT_IDENTITY

        traceback.print_exc()
        print(
            "population_run.py refused: PyVBMC cannot be imported from "
            f"the campaign's package tree ({type(error).__name__}: {error})",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(EXIT_IDENTITY)

import campaign_contract as contract
import dill
import golden_trace
import numpy as np
from benchmark_targets import (
    find_config,
    metrics,
    missing_truths,
    suite_configs,
)
from filelock import FileLock
from profile_run import jsonable

from pyvbmc import VBMC

VBMC_MODULE = importlib.import_module("pyvbmc.vbmc.vbmc")
#: The artifacts of a case of a campaign run before array mode, by suffix
#: (:func:`validate_case`).
SUFFIXES = (".npz", ".json", ".boost.pkl", ".boost.json")
#: The harness's own files, whose SHA-256 the source identity holds.
HARNESS_FILES = (
    "dev/scripts/population_run.py",
    "dev/scripts/golden_trace.py",
    "dev/scripts/profile_run.py",
    "dev/scripts/benchmark_targets.py",
    "dev/scripts/campaign_contract.py",
    "dev/scripts/data",
)
#: The harness modules a process must import from this directory.
HELPERS = ("golden_trace", "profile_run", "benchmark_targets")
#: The options every campaign sets, beside those of its configurations:
#: the targets are evaluated one point at a time, the historical chunk
#: settings apply (no machine calibration), the production guard of the
#: final boost, and no display.
DEFAULT_OPTIONS = {
    "vectorized_target": False,
    "performance_calibration": "off",
    "tol_elcbo_boost": 0.1,
    "display": "off",
    "plot": False,
    "print_iteration_header": False,
}
#: The metrics a paired test may take; usability has McNemar's test.
PAIRED_METRICS = ("elbo_err", "gskl", "mmtv", "func_count")
#: The metrics ``rescore`` recomputes.
RESCORED_METRICS = ("elbo_err", "gskl", "mmtv", "rmse")
#: The named subsets beside one per label.
SUBSETS = ("canary", "noisy", "noiseless")
#: The record ``rescore`` writes in the campaign it runs for: the rescoring
#: process's identity and the SHA-256 of every file of rescored metrics.
RESCORING = "rescoring.json"
#: What of a finished campaign enters the repository, beside its manifest
#: and verification report (``campaign_contract.tracked_copies``): the
#: summary, and for every verified case its completion record, its sidecar
#: and its boost report, which are what the comparison of two arms reads
#: (``analyze_population_run.py --arms``). A campaign that rescores adds
#: :data:`RESCORED_COPIES` (:func:`tracked_copies`).
TRACKED_COPIES = {
    "files": ["summary.md"],
    "cases": {"record": True, "artifacts": ["*.json"]},
}
#: The rescored metrics and the rescoring's record.
RESCORED_COPIES = ["rescored/*.json", RESCORING]
EXIT_USAGE = 64

sha256 = contract.sha256_file
write_json = contract.write_json


def case_path(out, tag, suffix):
    """A file of a case of a campaign run before array mode."""
    # Keep auxiliary JSON outside golden_trace's *_seed*.json discovery.
    folder = (
        Path(out) / "records"
        if suffix in (".boost.json", ".complete.json")
        else Path(out)
    )
    return folder / f"{tag}{suffix}"


def clone_vp(vp):
    result = copy.deepcopy(vp)
    result.rng = copy.deepcopy(vp.rng)
    return result


def scores(vp):
    if vp is None:
        return None
    return {k: float(vp.stats[k]) for k in ("elbo", "elbo_sd")}


class BoostCapture:
    """Observe the real final boost without rescoring or making random draws.

    The patched ``VBMC.final_boost`` keeps a copy of the pre-boost
    posterior, and the ``optimize_vp`` it calls (positionally, as
    ``optimize_vp(options, optim_state, vp, gp, n_fast, n_slow, K_new)``)
    keeps the options of the boost, its call and the raw candidate; both
    call the originals unchanged.
    """

    def __init__(self):
        self.state = None

    def __enter__(self):
        original_boost = VBMC.final_boost
        original_optimize = VBMC_MODULE.optimize_vp
        owner = self

        def boost(vbmc, vp, gp):
            if owner.state is not None:
                raise RuntimeError("More than one final boost in a run")
            state = {
                "pre": clone_vp(vp),
                "candidate": None,
                "rng_before": copy.deepcopy(vbmc.rng.bit_generator.state),
                "tolerance": vbmc.options.get("tol_elcbo_boost"),
                "boost_options": None,
                "boost_call": None,
            }
            owner.state = state

            def optimize(options, *args, **kwargs):
                if state["candidate"] is not None:
                    raise RuntimeError("More than one optimizer call in boost")
                state["boost_options"] = {
                    k: jsonable(options.get(k)) for k in options
                }
                state["boost_call"] = dict(
                    zip(("n_fast_opts", "n_slow_opts", "K_new"), args[3:])
                )
                result = original_optimize(options, *args, **kwargs)
                state["candidate"] = clone_vp(result[0])
                return result

            with patch.object(VBMC_MODULE, "optimize_vp", optimize):
                result = original_boost(vbmc, vp, gp)
            state["returned"] = clone_vp(result[0])
            state["attempted"] = state["candidate"] is not None
            state["accepted"] = bool(result[3])
            state["rng_after"] = copy.deepcopy(vbmc.rng.bit_generator.state)
            return result

        self.context = patch.object(VBMC, "final_boost", boost)
        self.context.__enter__()
        return self

    def __exit__(self, *exc):
        return self.context.__exit__(*exc)


# --------------------------------------------------------------------------
# Source trees and identity
# --------------------------------------------------------------------------


def source_trees(environ=None):
    """The campaign's trees: ``{"harness", "pyvbmc", "gpyreg"}``, resolved."""
    env = os.environ if environ is None else environ
    gpyreg = env.get("PYVBMC_GPYREG_SOURCE")
    if not gpyreg:
        raise contract.IdentityError(
            "PYVBMC_GPYREG_SOURCE is not set; it names the campaign's gpyreg "
            "checkout"
        )
    return {
        "harness": ROOT,
        "pyvbmc": Path(env.get("PYVBMC_SOURCE") or ROOT).resolve(),
        "gpyreg": Path(gpyreg).resolve(),
    }


def release_code(environ=None):
    """Whether the package tree is the harness checkout (the release code)."""
    return source_trees(environ)["pyvbmc"] == ROOT


def helper_origins():
    """Where the harness modules were imported from, which must be here."""
    found = {}
    for name in HELPERS:
        module = sys.modules.get(name) or importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if path != HERE / f"{name}.py":
            raise contract.IdentityError(
                f"{name} is imported from {path}, not from {HERE}"
            )
        found[name] = str(path)
    return found


def this_identity(host=True):
    """This process's identity (``campaign_contract.identity``)."""
    record = contract.identity(
        source_trees(),
        HARNESS_FILES,
        modules={"pyvbmc": "pyvbmc", "gpyreg": "gpyreg"},
        host=host,
    )
    record["imports"]["harness_modules"] = helper_origins()
    return record


# --------------------------------------------------------------------------
# Allocation and cases
# --------------------------------------------------------------------------


def allocation(suite, labels=None, seeds="0-99"):
    """The allocation ``{"suite", "labels", "seeds"}``, labels in suite order."""
    known = [config.label for config in suite_configs(suite)]
    if labels:
        unknown = sorted(set(labels) - set(known))
        if unknown:
            raise SystemExit(f"the suite {suite} has no label {unknown}")
        known = [label for label in known if label in set(labels)]
    chosen = golden_trace.parse_seeds(seeds)
    if not chosen or min(chosen) < 0:
        raise SystemExit(f"--seeds {seeds} names no seed, or a negative one")
    return {"suite": suite, "labels": known, "seeds": chosen}


def case_lines(manifest):
    """The campaign's case lines, in the order of their indices."""
    alloc = manifest["allocation"]
    return [
        f"{label}/{golden_trace._tag(label, seed)} {label} {seed}"
        for label in alloc["labels"]
        for seed in alloc["seeds"]
    ]


def parse_case(line):
    """``(tag, label, seed)`` of a case line, checked."""
    tag = contract.case_tag(line)
    fields = line.split(" ")
    if len(fields) != 3 or not fields[2].isdigit():
        raise contract.ContractError(f"{line!r} is not '<tag> <label> <seed>'")
    label, seed = fields[1], int(fields[2])
    if tag != f"{label}/{golden_trace._tag(label, seed)}":
        raise contract.ContractError(f"{line!r}: the tag names another case")
    return tag, label, seed


def case_files(label, seed):
    """The artifacts of one case, relative to the campaign directory."""
    stem = golden_trace._tag(label, seed)
    return {
        "trace": f"{label}/{stem}.npz",
        "sidecar": f"{label}/{stem}.json",
        "posterior": f"{label}/{stem}.vp.npz",
        "boost_state": f"boost/{label}/{stem}.boost.pkl",
        "boost_report": f"boost/{label}/{stem}.boost.json",
    }


def subset_indices(manifest, name):
    """The case indices of a named subset (module docstring)."""
    alloc = manifest["allocation"]
    if name not in SUBSETS and name not in alloc["labels"]:
        raise SystemExit(
            f"no subset {name!r}: the subsets are {', '.join(SUBSETS)} and "
            "each label"
        )
    first = min(alloc["seeds"])

    def member(label, seed):
        if name == "canary":
            return seed == first
        if name in ("noisy", "noiseless"):
            noisy = find_config(label).noise_sd is not None
            return noisy == (name == "noisy")
        return label == name

    indices = [
        index
        for index, line in enumerate(case_lines(manifest), start=1)
        if member(*parse_case(line)[1:])
    ]
    if not indices:
        raise SystemExit(f"the subset {name} holds no case of this campaign")
    return indices


def confirmatory_family(spec, labels):
    """The confirmatory family of the comparison of two arms.

    The default: the seeds paired across the arms, exact two-sided
    signed-rank tests of the evidence error, gsKL and MMTV and exact
    McNemar tests of usability for every configuration, under one Holm
    correction at alpha 0.05. ``spec`` (a mapping, from ``prepare
    --confirmatory``) overrides any of ``labels`` (a subset of the
    allocation's), ``signed_rank`` (metrics of :data:`PAIRED_METRICS`),
    ``mcnemar_usability``, ``alpha`` and ``statement``.
    """
    family = {
        "pairing": "seed",
        "labels": list(labels),
        "signed_rank": ["elbo_err", "gskl", "mmtv"],
        "mcnemar_usability": True,
        "correction": "holm",
        "alpha": 0.05,
        "statement": None,
    }
    # A family copied from a manifest holds its count of tests, which is
    # recomputed here.
    spec = {
        key: value for key, value in (spec or {}).items() if key != "tests"
    }
    unknown = sorted(set(spec) - set(family))
    if unknown:
        raise SystemExit(f"the confirmatory family has no field {unknown}")
    family.update(spec)
    if family["pairing"] != "seed" or family["correction"] != "holm":
        raise SystemExit("the confirmatory family pairs by seed under Holm")
    if not family["labels"] or set(family["labels"]) - set(labels):
        raise SystemExit(
            "the confirmatory family's labels must be labels of the "
            "allocation"
        )
    if set(family["signed_rank"]) - set(PAIRED_METRICS):
        raise SystemExit(
            f"signed-rank tests take the metrics {PAIRED_METRICS} alone"
        )
    if not 0 < float(family["alpha"]) < 1:
        raise SystemExit("the confirmatory alpha lies between 0 and 1")
    family["labels"] = [label for label in labels if label in family["labels"]]
    per_label = len(family["signed_rank"]) + bool(family["mcnemar_usability"])
    family["tests"] = len(family["labels"]) * per_label
    if not family["tests"]:
        raise SystemExit("the confirmatory family holds no test")
    if not family["statement"]:
        family["statement"] = (
            "Seeds paired across the two arms: exact two-sided signed-rank "
            f"tests of {', '.join(family['signed_rank']) or 'no metric'}"
            + (
                " and exact McNemar tests of usability"
                if family["mcnemar_usability"]
                else ""
            )
            + f" for each of {len(family['labels'])} configurations; "
            f"{family['tests']} tests under one Holm correction at alpha "
            f"{family['alpha']}, fixed before the runs."
        )
    return family


def tracked_copies(rescores):
    """The tracked copies of a campaign; ``rescores``: it runs ``rescore``."""
    spec = copy.deepcopy(TRACKED_COPIES)
    if rescores:
        spec["files"] += RESCORED_COPIES
    return spec


def read_manifest(out):
    path = Path(out) / "manifest.json"
    if not path.is_file():
        raise SystemExit(f"{path} does not exist; run prepare first")
    return contract.read_json(path)


def checked_verification(campaign, manifest):
    """A campaign's ``verification.json``, checked against the campaign.

    The report must reconcile the manifest's allocation, name the
    manifest's SHA-256 and have passed (exit code 0). It must also describe
    the directory as it stands: every case it places as verified keeps the
    completion record it checked (``record_sha256``), and no case it places
    in another state has a record. A case in another state that has one
    completed after the report was written, as the cases of a campaign do
    after a finish that looked at its canary; the report is then older
    than the directory, and is refused until the campaign is verified
    again. ``campaign`` may be the tracked copies of a campaign
    (``campaign_contract.source_sha256``), which hold the records of the
    verified cases alone.

    Returns
    -------
    dict
        The report.

    Raises
    ------
    campaign_contract.ContractError
        When a check fails.
    """
    campaign = Path(campaign)
    path = campaign / "verification.json"
    if not path.is_file():
        raise contract.ContractError(
            f"{campaign} has no verification.json; run verify first"
        )
    report = contract.read_json(path)
    if [case["case"] for case in report["cases"]] != case_lines(manifest):
        raise contract.ContractError(
            f"{path} does not reconcile the allocation of its manifest"
        )
    if report.get("exit_code") != 0:
        raise contract.ContractError(f"{campaign} failed its verification")
    if report.get("manifest_sha256") != contract.source_sha256(
        campaign, "manifest.json"
    ):
        raise contract.ContractError(
            f"{path} verified another manifest than {campaign}'s"
        )
    completed = []
    for case in report["cases"]:
        tag = case["tag"]
        rel = f"{contract.RECORDS}/{tag}{contract.RECORD_SUFFIX}"
        if case["status"] != "verified":
            if (campaign / rel).exists():
                completed.append(tag)
            continue
        if not (campaign / rel).is_file():
            raise contract.ContractError(
                f"{tag}: the completion record that {path} verified is gone"
            )
        if contract.source_sha256(campaign, rel) != case.get("record_sha256"):
            raise contract.ContractError(
                f"{tag}: {rel} is not the record that {path} verified"
            )
    if completed:
        raise contract.ContractError(
            f"{path} is older than {campaign}: {len(completed)} cases it "
            "places as not verified have completion records now (the first "
            f"{completed[0]}); verify the campaign again, by its finish"
        )
    return report


def write_case_list(out, lines):
    """Write ``cases.txt`` once, or refuse a list that differs from it."""
    path = Path(out) / "cases.txt"
    text = "".join(f"{line}\n" for line in lines).encode("utf-8")
    if path.exists():
        if path.read_bytes() != text:
            raise SystemExit(f"{path} lists other cases than the manifest")
        return
    path.write_bytes(text)


# --------------------------------------------------------------------------
# The returned posterior as plain arrays
# --------------------------------------------------------------------------


def posterior_extras(vp):
    """What a trace lacks to rebuild ``vp``: the ``<stem>.vp.npz`` arrays."""
    transformer = vp.parameter_transformer
    return {
        "bounded_types": np.asarray(transformer.bounded_types, dtype=np.int64),
        "scale_is_none": np.asarray(transformer.scale is None),
        "rotation_is_none": np.asarray(transformer.R_mat is None),
    }


def returned_posterior(trace, extras, problem, best_iter):
    """Rebuild a run's returned posterior from its plain arrays.

    Parameters
    ----------
    trace : mapping
        The run's trace (``<stem>.npz``).
    extras : mapping
        Its ``<stem>.vp.npz`` (:func:`posterior_extras`).
    problem : benchmark_targets.Problem
        The configuration's problem, whose bounds the transformer takes.
    best_iter : int
        The iteration the posterior was returned from (the sidecar's).

    Returns
    -------
    VariationalPosterior
        A posterior of the imported package with the returned posterior's
        weights, means, scales, axis lengths and transformer. Its own
        generator (seeded 0) is never the run's; ``eta``, ``stats`` and the
        optimization flags are the constructor's, since the metrics do not
        read them.
    """
    from pyvbmc.parameter_transformer import ParameterTransformer
    from pyvbmc.variational_posterior import VariationalPosterior

    types = np.asarray(extras["bounded_types"]).ravel()
    if types.size != 1:
        raise ValueError(f"one bounded transform expected, not {types}")
    D = problem.D
    transformer = ParameterTransformer(
        D, problem.lb, problem.ub, transform_type=int(types[0])
    )
    transformer.mu = np.array(trace["pt_mu"][best_iter])
    transformer.delta = np.array(trace["pt_delta"][best_iter])
    transformer.scale = (
        None
        if bool(extras["scale_is_none"])
        else np.array(trace["pt_scale"][best_iter])
    )
    transformer.R_mat = (
        None
        if bool(extras["rotation_is_none"])
        else np.array(trace["pt_R"][best_iter])
    )
    w = np.asarray(trace["final_w"])
    K = w.size
    vp = VariationalPosterior(
        D, K, parameter_transformer=transformer, rng=np.random.default_rng(0)
    )
    vp.w = w.reshape(1, K)
    vp.mu = np.asarray(trace["final_mu"])
    vp.sigma = np.asarray(trace["final_sigma"]).reshape(1, K)
    vp.lambd = np.asarray(trace["final_lambd"]).reshape(D, 1)
    return vp


def transformer_state(transformer):
    """A transformer's data, and the names of the transforms it builds."""
    state = {
        key: value
        for key, value in vars(transformer).items()
        if key != "_bounded_transforms"
    }
    state["_bounded_transforms"] = sorted(
        vars(transformer).get("_bounded_transforms", {})
    )
    return state


def _same(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return (
            a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
        )
    return a == b


def posterior_differences(rebuilt, returned):
    """The attributes in which a rebuilt posterior differs from the returned.

    Compared: the dimension, the number of components, ``w``, ``mu``,
    ``sigma``, ``lambd`` (shape, dtype and values) and every attribute of the
    transformer but the functions it builds from them.
    """
    found = [
        name
        for name in ("D", "K", "w", "mu", "sigma", "lambd")
        if not _same(getattr(rebuilt, name), getattr(returned, name))
    ]
    a = transformer_state(rebuilt.parameter_transformer)
    b = transformer_state(returned.parameter_transformer)
    missing = object()
    for key in sorted(set(a) | set(b)):
        x, y = a.get(key, missing), b.get(key, missing)
        if x is missing or y is missing or not _same(x, y):
            found.append(f"parameter_transformer.{key}")
    return found


def same_value(a, b):
    """Equality of two metric values, NaN equal to NaN."""
    if a is None or b is None:
        return a is None and b is None
    return bool(a == b or (np.isnan(a) and np.isnan(b)))


def rescore_metrics(problem, vp, elbo):
    """``benchmark_targets.metrics`` of ``vp``: the rescored fields."""
    found = metrics(problem, vp, elbo)
    result = {key: float(found[key]) for key in RESCORED_METRICS}
    result["moment_method"] = found["moment_method"]
    return result


# --------------------------------------------------------------------------
# Checks of a case's artifacts
# --------------------------------------------------------------------------


def check_case_artifacts(tag, paths, options):
    """Check the artifacts of one case; raise on any inconsistency.

    Parameters
    ----------
    tag : str
        ``<label>_seed<seed>``.
    paths : mapping of str to Path
        ``trace``, ``sidecar``, ``boost_state``, ``boost_report`` and, in a
        campaign of array mode, ``posterior``.
    options : mapping
        The campaign's options, which the run must have requested and
        applied.

    Returns
    -------
    dict
        The sidecar.
    """
    side = json.loads(Path(paths["sidecar"]).read_text())
    if f"{side['label']}_seed{side['seed']}" != tag:
        raise RuntimeError(f"{tag}: wrong configuration or seed")
    for metric in (*golden_trace.METRICS, "elbo", "elbo_sd"):
        if not np.isfinite(side["final"][metric]):
            raise RuntimeError(f"{tag}: nonfinite {metric}")
    if side["label"] == "cigar_D15_exhaust":
        if side["final"]["func_count"] != 750:
            raise RuntimeError(f"{tag}: incorrect exhaust budget")
    for key, value in options.items():
        if side["requested_options"].get(key) != value:
            raise RuntimeError(f"{tag}: incorrect requested {key}")
        if side["effective_options"].get(key) != value:
            raise RuntimeError(f"{tag}: incorrect effective {key}")
    with np.load(paths["trace"], allow_pickle=False) as data:
        for key in data.files:
            _ = data[key]
    with Path(paths["boost_state"]).open("rb") as stream:
        capture = dill.load(stream)
    for key in ("pre", "returned"):
        if capture[key].parameter_transformer is None:
            raise RuntimeError(f"{tag}: missing {key} transformer")
    if capture["attempted"]:
        if capture["candidate"] is None:
            raise RuntimeError(f"{tag}: missing candidate")
        if capture["boost_options"]["weight_penalty"] != 0:
            raise RuntimeError(f"{tag}: boost penalty was not disabled")
    report = json.loads(Path(paths["boost_report"]).read_text())
    for key in (
        "attempted",
        "accepted",
        "tolerance",
        "boost_options",
        "boost_call",
    ):
        if key not in report or report[key] != jsonable(capture[key]):
            raise RuntimeError(f"{tag}: inconsistent boost {key}")
    if (
        type(report["attempted"]) is not bool
        or type(report["accepted"]) is not bool
    ):
        raise RuntimeError(f"{tag}: invalid boost decision type")
    if report["tolerance"] != options["tol_elcbo_boost"]:
        raise RuntimeError(f"{tag}: incorrect boost tolerance")
    keys = (
        ("pre", "candidate", "returned")
        if capture["attempted"]
        else ("pre", "returned")
    )
    if set(report.get("metrics", {})) != set(keys):
        raise RuntimeError(f"{tag}: incomplete boost metrics")
    if set(report.get("scores", {})) != {"pre", "candidate", "returned"}:
        raise RuntimeError(f"{tag}: incomplete boost scores")
    for key in keys:
        vp = capture[key]
        if vp.parameter_transformer is None:
            raise RuntimeError(f"{tag}: missing {key} transformer")
        np.testing.assert_equal(report["scores"][key], scores(vp))
        met = report["metrics"][key]
        if key != "returned" and isinstance(met.get("error"), str):
            continue
        for metric in (
            "elbo_err",
            "gskl",
            "mmtv",
            "rmse",
            "post_mean",
            "post_cov",
            "moment_method",
        ):
            if metric not in met:
                raise RuntimeError(f"{tag}: missing {key}/{metric}")
        for metric in ("elbo_err", "gskl", "mmtv", "rmse"):
            if key == "returned" and not np.isfinite(met[metric]):
                raise RuntimeError(f"{tag}: nonfinite {key}/{metric}")
    if not capture["attempted"] and (
        capture["accepted"] or capture["candidate"] is not None
    ):
        raise RuntimeError(f"{tag}: inconsistent skipped boost")
    chosen = capture["candidate"] if capture["accepted"] else capture["pre"]
    for attr in ("w", "mu", "sigma", "lambd"):
        np.testing.assert_array_equal(
            getattr(chosen, attr), getattr(capture["returned"], attr)
        )
    # A transformer builds its bounded transforms afresh when it is copied
    # or unpickled, so two equal transformers hold different functions.
    np.testing.assert_equal(
        transformer_state(chosen.parameter_transformer),
        transformer_state(capture["returned"].parameter_transformer),
    )
    np.testing.assert_equal(
        scores(capture["returned"]),
        {k: side["final"][k] for k in ("elbo", "elbo_sd")},
    )
    if "posterior" in paths:
        problem = find_config(side["label"]).make(seed=side["seed"])
        with np.load(paths["trace"], allow_pickle=False) as trace, np.load(
            paths["posterior"], allow_pickle=False
        ) as extras:
            rebuilt = returned_posterior(
                trace, extras, problem, side["final"]["best_iter"]
            )
        differing = posterior_differences(rebuilt, capture["returned"])
        if differing:
            raise RuntimeError(
                f"{tag}: the posterior rebuilt from the plain arrays differs "
                f"from the returned one in {differing}"
            )
    return side


def validate_case(out, tag, expected):
    """Check a case of a campaign run before array mode.

    Such a campaign holds its cases in one directory, with the boost report
    and a completion record of the four artifacts' SHA-256 under
    ``records/``, and its launch record holds the identity every case must
    carry (``expected``). Raises on incomplete or changed artifacts and
    never skips one silently; returns the completion record.
    """
    out = Path(out)
    done = json.loads(case_path(out, tag, ".complete.json").read_text())
    if done["identity"] != expected:
        raise RuntimeError(f"{tag}: source/environment/options changed")
    if set(done["hashes"]) != set(SUFFIXES):
        raise RuntimeError(f"{tag}: incomplete file manifest")
    for suffix, digest in done["hashes"].items():
        if sha256(case_path(out, tag, suffix)) != digest:
            raise RuntimeError(f"{tag}: changed {suffix}")
    check_case_artifacts(
        tag,
        {
            "trace": out / f"{tag}.npz",
            "sidecar": out / f"{tag}.json",
            "boost_state": out / f"{tag}.boost.pkl",
            "boost_report": case_path(out, tag, ".boost.json"),
        },
        expected["options"],
    )
    return done


def diagnostic_metrics(problem, vp, required=False):
    try:
        return jsonable(metrics(problem, vp, vp.stats["elbo"]))
    except Exception as error:
        if required:
            raise
        return {"error": f"{type(error).__name__}: {error}"}


def boost_report(state, problem):
    """The decision, scores and metrics of a boost capture."""
    return {
        "attempted": state["attempted"],
        "accepted": state["accepted"],
        "tolerance": state["tolerance"],
        "boost_options": state["boost_options"],
        "boost_call": jsonable(state["boost_call"]),
        "scores": {
            k: scores(state[k]) for k in ("pre", "candidate", "returned")
        },
        "metrics": {
            k: diagnostic_metrics(problem, state[k], required=k == "returned")
            for k in ("pre", "candidate", "returned")
            if state[k] is not None
        },
    }


# --------------------------------------------------------------------------
# One case
# --------------------------------------------------------------------------


def run_case(out, label, seed, options, identity):
    """Run one case and write its artifacts; return ``(paths, extra)``.

    The contract's worker calls it (``campaign_contract.run_worker``) with
    the process's identity, whose ``source`` and ``imports`` parts become
    the sidecar's ``provenance``. Raises when the run fails, with the
    traceback that ``golden_trace.run_task`` wrote; when the plain arrays
    do not rebuild the returned posterior or its metrics exactly; and when
    the artifacts fail the checks that ``verify`` repeats
    (:func:`check_case_artifacts`: finite metrics, the exhaust budget, the
    options requested and applied, the boost's consistency). The worker
    then writes the case's error file and no completion record, so that
    ``verify`` places the case as failed.
    """
    files = {
        key: Path(out) / rel for key, rel in case_files(label, seed).items()
    }
    for path in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    stem = golden_trace._tag(label, seed)
    directory = files["trace"].parent
    with BoostCapture() as capture:
        result = golden_trace.run_task(label, seed, options, directory)
    if not result["ok"]:
        error = directory / f"{stem}.error.txt"
        detail = (
            error.read_text(encoding="utf-8", errors="replace")
            if error.exists()
            else "(it wrote no traceback)\n"
        )
        raise RuntimeError(f"golden_trace.run_task failed:\n{detail}")
    state = capture.state
    if state is None:
        raise RuntimeError("the run made no final boost to capture")
    with files["boost_state"].open("wb") as stream:
        dill.dump(state, stream)
    problem = find_config(label).make(seed=seed)
    write_json(files["boost_report"], boost_report(state, problem))
    extras = posterior_extras(state["returned"])
    np.savez(files["posterior"], **extras)
    side = json.loads(files["sidecar"].read_text())
    final = side["final"]
    with np.load(files["trace"], allow_pickle=False) as trace:
        rebuilt = returned_posterior(
            trace, extras, problem, final["best_iter"]
        )
    differing = posterior_differences(rebuilt, state["returned"])
    if differing:
        raise RuntimeError(
            "the posterior rebuilt from the plain arrays differs from the "
            f"returned one in {differing}"
        )
    again = rescore_metrics(problem, rebuilt, final["elbo"])
    unequal = {
        key: (again[key], final[key])
        for key in RESCORED_METRICS
        if not same_value(again[key], final[key])
    }
    if unequal:
        raise RuntimeError(
            "the rebuilt posterior scores otherwise than the returned one "
            f"(rebuilt, in run): {unequal}"
        )
    side["provenance"] = {
        "source": identity["source"],
        "imports": identity["imports"],
    }
    # The same form golden_trace.run_task writes.
    files["sidecar"].write_text(json.dumps(side, indent=1))
    check_case_artifacts(stem, files, options)
    return list(files.values()), {
        "harness": "population_run",
        "label": label,
        "seed": seed,
        "boost": {
            "attempted": state["attempted"],
            "accepted": state["accepted"],
        },
    }


# --------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------


def pair_differences(identity, other):
    """Why two identities cannot be the two arms of one comparison.

    The arms differ in their code alone: they share the harness checkout's
    commit and clean state, the SHA-256 of the harness files (the targets
    module and its data among them) and the imported versions of Python,
    NumPy, SciPy, cma and Torch, and their package trees are at different
    commits. Returns the reasons, an empty list when the two can be paired.
    """
    a, b = identity["source"], other["source"]
    found = [
        f"the arms have different {what}"
        for key, what in (
            ("files", "harness files"),
            ("versions", "environment versions"),
        )
        if a.get(key) != b.get(key)
    ]
    if a["trees"].get("harness") != b["trees"].get("harness"):
        found.insert(0, "the arms have different harness checkouts")
    commit = [s["trees"].get("pyvbmc", {}).get("commit") for s in (a, b)]
    if commit[0] == commit[1]:
        found.append(
            f"both arms' package trees are at {commit[0]}, so the comparison "
            "would compare the code with itself"
        )
    return found


def cmd_prepare(args):
    out = args.out.resolve()
    labels = [label for label in (args.labels or "").split(",") if label]
    alloc = allocation(args.suite, labels, args.seeds)
    missing = missing_truths([find_config(label) for label in alloc["labels"]])
    if missing:  # a population without a truth has NaN metrics: no gate
        raise SystemExit(
            "no ground truth for "
            + ", ".join(missing)
            + "; generate it with dev/scripts/make_benchmark_truths.py"
        )
    options = dict(DEFAULT_OPTIONS)
    options.update(json.loads(args.options) if args.options else {})
    spec = contract.read_json(args.confirmatory) if args.confirmatory else None
    family = confirmatory_family(spec, alloc["labels"])
    release = release_code()
    identity = this_identity()
    steps = [["summarize"]]
    pair = None
    if args.pair:
        if not release:
            raise SystemExit(
                "--pair names the campaign that this one rescores beside "
                "itself, and only a campaign of the harness checkout's own "
                "package (the release code) rescores; this one's PyVBMC is "
                "the tree PYVBMC_SOURCE names"
            )
        pair = args.pair.resolve()
        if pair == out:
            raise SystemExit("--pair names the campaign itself")
        other = read_manifest(pair)
        for key, value in (
            ("allocation", alloc),
            ("options", options),
            ("confirmatory", family),
        ):
            if other.get(key) != value:
                raise SystemExit(
                    f"the paired campaign {pair} has another {key}; the arms "
                    "pair seed by seed on one allocation and one family"
                )
        differing = pair_differences(identity, other["identity"])
        if differing:
            raise SystemExit(
                f"the paired campaign {pair} cannot be this one's other arm: "
                + "; ".join(differing)
            )
        if pair.name == out.name:
            raise SystemExit(
                "the paired campaigns' directories share the name "
                f"{out.name}, which names their rescored files"
            )
    if release:
        steps.append(
            ["rescore"] + (["--campaign", pair.as_posix()] if pair else [])
        )
    manifest = {
        "harness": "population_run",
        "contract": contract.CONTRACT_VERSION,
        "arm": args.arm,
        "allocation": alloc,
        "options": options,
        "confirmatory": family,
        "pair": None if pair is None else str(pair),
        "identity": identity,
        "site": contract.site_block(),
        "pip_freeze": contract.pip_freeze(),
        "finishing_steps": steps,
        "tracked_copies": tracked_copies(release),
        "created": contract.now(),
    }
    contract.finishing_steps(manifest)
    contract.tracked_copies(manifest)
    path = out / "manifest.json"
    if path.exists():
        previous = contract.read_json(path)
        differing = contract.source_differences(
            manifest["identity"], previous["identity"]
        )
        differing += [
            key
            for key in (
                "allocation",
                "options",
                "confirmatory",
                "arm",
                "pair",
                "finishing_steps",
                "tracked_copies",
            )
            if previous.get(key) != manifest[key]
        ]
        if differing:
            raise SystemExit(
                f"{path} is prepared otherwise: {differing}; a prepared "
                "campaign is not revised"
            )
        print(f"{path} is prepared already", flush=True)
        return 0
    write_json(path, manifest)
    print(
        f"{path}: {len(case_lines(manifest))} cases, "
        f"{len(alloc['labels'])} configurations of {alloc['suite']} at "
        f"{len(alloc['seeds'])} seeds; finishing steps {steps}",
        flush=True,
    )
    return 0


def cmd_cases(args):
    manifest = read_manifest(args.out.resolve())
    lines = case_lines(manifest)
    if args.subset:
        for index in subset_indices(manifest, args.subset):
            print(f"{index} {lines[index - 1]}")
    else:
        for line in lines:
            print(line)
    return 0


def cmd_worker(args):
    """One case under the contract's worker sequence.

    A directory without a readable manifest of this harness
    (``"harness": "population_run"``) and a line that is not a case of its
    allocation are refused with :data:`EXIT_USAGE`, touching nothing.
    """
    out = args.out.resolve()
    path = out / "manifest.json"
    try:
        manifest = contract.read_json(path)
    except (OSError, ValueError) as error:
        print(
            f"{out} holds no readable manifest.json ({error}); the worker "
            "runs cases in a directory that `prepare` wrote",
            flush=True,
        )
        return EXIT_USAGE
    if not isinstance(manifest, dict) or (
        manifest.get("harness") != "population_run"
    ):
        print(f"{path} is not a manifest of population_run", flush=True)
        return EXIT_USAGE
    if args.case not in case_lines(manifest):
        print(f"{args.case!r} is not a case of {out}", flush=True)
        return EXIT_USAGE
    _, label, seed = parse_case(args.case)
    paths = [out / rel for rel in case_files(label, seed).values()]
    return contract.run_worker(
        out,
        args.case,
        manifest["identity"],
        this_identity,
        lambda identity: run_case(
            out, label, seed, manifest["options"], identity
        ),
        lambda: paths,
    )


def _listing(directory):
    """The entries of one directory, sorted; none when it does not exist."""
    try:
        return sorted(os.scandir(directory), key=lambda entry: entry.name)
    except FileNotFoundError:
        return []


def stray_artifacts(out, manifest):
    """Files in the configurations' directories that no case owns.

    Only ``<label>/`` and ``boost/<label>/`` of the allocation's labels and
    the top of ``boost/`` are listed; dot-files are temporary files, and the
    error files are the contract's to reconcile.
    """
    out = Path(out)
    alloc = manifest["allocation"]
    expected = {
        rel
        for label in alloc["labels"]
        for seed in alloc["seeds"]
        for rel in case_files(label, seed).values()
    }
    stray = []
    for entry in _listing(out / "boost"):
        if entry.name.startswith("."):
            continue
        if not entry.is_dir():
            stray.append(f"boost/{entry.name}")
        elif entry.name not in alloc["labels"]:
            stray.append(f"boost/{entry.name}/")
    for label in alloc["labels"]:
        for folder in (label, f"boost/{label}"):
            for entry in _listing(out / folder):
                name, rel = entry.name, f"{folder}/{entry.name}"
                if name.startswith("."):
                    continue
                if entry.is_dir():
                    stray.append(f"{rel}/")
                elif folder == label and name.endswith(contract.ERROR_SUFFIX):
                    continue
                elif rel not in expected:
                    stray.append(rel)
    return stray


def cmd_verify(args):
    out = args.out.resolve()
    manifest = read_manifest(out)
    try:
        actual = this_identity(host=False)
    except Exception as error:
        print(f"verify refused: no identity ({error})", flush=True)
        return contract.EXIT_IDENTITY
    differing = contract.source_differences(actual, manifest["identity"])
    if differing:
        print(
            "verify refused: it runs in the campaign's own trees and "
            "environment, and this process's source identity differs from "
            f"the manifest's in {differing}",
            flush=True,
        )
        return contract.EXIT_IDENTITY
    lines = case_lines(manifest)
    parsed = {parse_case(line)[0]: parse_case(line) for line in lines}
    node_feature = (manifest.get("site") or {}).get("NODE_FEATURE")

    def check(tag):
        _, label, seed = parsed[tag]
        files = case_files(label, seed)
        record = contract.check_completion(
            out,
            tag,
            manifest["identity"],
            required=list(files.values()),
            node_feature=node_feature,
        )
        side = check_case_artifacts(
            golden_trace._tag(label, seed),
            {key: out / rel for key, rel in files.items()},
            manifest["options"],
        )
        identity = record["identity"]
        provenance = side.get("provenance") or {}
        if provenance != {
            "source": identity["source"],
            "imports": identity["imports"],
        }:
            raise RuntimeError(
                f"{tag}: the sidecar's provenance is not its record's identity"
            )
        return {
            "elapsed_seconds": record["elapsed_seconds"],
            "boost": record.get("boost"),
            # What the readers of the report check the record against
            # (checked_verification).
            "record_sha256": sha256(contract.record_path(out, tag)),
        }

    def partial(tag):
        _, label, seed = parsed[tag]
        return [
            rel
            for rel in case_files(label, seed).values()
            if (out / rel).exists()
        ]

    report = contract.reconcile(
        out, lines, check, partial, stray=stray_artifacts(out, manifest)
    )
    write_json(
        out / "verification.json",
        {
            "harness": "population_run",
            "manifest_sha256": sha256(out / "manifest.json"),
            **report,
        },
    )
    for case in report["cases"]:
        if case["status"] != "verified":
            print(json.dumps(case), flush=True)
    for name in report["stray"]:
        print(f"stray: {name}", flush=True)
    print(json.dumps(report["counts"]), flush=True)
    return report["exit_code"]


def cmd_summarize(args):
    """Write ``summary.md``: the population of the verified cases alone.

    The table is ``golden_trace.summary_text``'s, of the sidecars of the
    cases that ``verification.json`` places as verified (a case in flight
    may have written its sidecar already), with a configuration's failed
    cases counted as its failures; the counts of every state and the boost
    decisions follow.
    """
    out = args.out.resolve()
    manifest = read_manifest(out)
    report = checked_verification(out, manifest)
    population = {}
    for case in report["cases"]:
        _, label, seed = parse_case(case["case"])
        entry = population.setdefault(
            label, {"seeds": [], "rows": [], "fails": 0}
        )
        if case["status"] == "failed":
            entry["fails"] += 1
        elif case["status"] == "verified":
            side = contract.read_json(out / case_files(label, seed)["sidecar"])
            entry["seeds"].append(seed)
            entry["rows"].append(side["final"])
    population = golden_trace.merge_populations(
        [
            {
                label: entry
                for label, entry in population.items()
                if entry["rows"] or entry["fails"]
            }
        ]
    )
    lines = [golden_trace.summary_text(population, out.name), ""]
    counts = report["counts"]
    lines.append(
        f"Cases: {len(report['cases'])}; "
        + ", ".join(
            f"{state.replace('_', ' ')} {counts[state]}"
            for state in (*contract.STATUSES, "stray")
            if counts.get(state)
        )
        + "."
    )
    boosts = [
        case.get("boost") or {}
        for case in report["cases"]
        if case["status"] == "verified"
    ]
    attempted = sum(bool(b.get("attempted")) for b in boosts)
    accepted = sum(bool(b.get("accepted")) for b in boosts)
    lines.append(
        f"Final boost: attempted {attempted} (accepted {accepted}, "
        f"rejected {attempted - accepted}), skipped "
        f"{len(boosts) - attempted}."
    )
    text = "\n".join(lines)
    (out / "summary.md").write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)
    return 0


#: The method of the rescoring, recorded with its results.
RESCORING_METHOD = (
    "benchmark_targets.metrics of this process's package on the returned "
    "posterior rebuilt from its plain arrays "
    "(population_run.returned_posterior); the evidence error from the run's "
    "own ELBO; the random draws from the generators of fixed seeds that the "
    "in-run metrics use"
)


def rescore_case(campaign, label, seed, hashes):
    """The rescored metrics of one verified case (module docstring).

    ``hashes`` holds the SHA-256 of the case's trace, posterior arrays and
    sidecar, which the caller checked against the case's record; the entry
    returned carries them, and ``finite`` says which rescored metric is
    finite.
    """
    files = case_files(label, seed)
    side = json.loads((Path(campaign) / files["sidecar"]).read_text())
    final = side["final"]
    problem = find_config(label).make(seed=seed)
    with np.load(Path(campaign) / files["trace"], allow_pickle=False) as trace:
        with np.load(
            Path(campaign) / files["posterior"], allow_pickle=False
        ) as extras:
            vp = returned_posterior(trace, extras, problem, final["best_iter"])
    again = rescore_metrics(problem, vp, final["elbo"])
    return {
        "status": "rescored",
        "label": label,
        "seed": seed,
        "metrics": {key: again[key] for key in RESCORED_METRICS},
        "moment_method": again["moment_method"],
        "equal_to_in_run": {
            key: same_value(again[key], final[key]) for key in RESCORED_METRICS
        },
        "finite": {
            key: bool(np.isfinite(again[key])) for key in RESCORED_METRICS
        },
        "artifacts": dict(hashes),
    }


def _read_part(path, source):
    """The cases of a rescoring work file made by code of ``source``; none
    when the file does not exist or other code made it."""
    if not path.is_file():
        return {}
    part = contract.read_json(path)
    if part.get("rescoring_source") != source:
        print(
            f"[rescore] {path} was made by other code; its cases are "
            "rescored anew",
            flush=True,
        )
        return {}
    return part.get("cases") or {}


def rescore_campaign(campaign, identity, parts):
    """Rescore every verified case of one campaign (module docstring).

    Parameters
    ----------
    campaign : path
        The campaign directory, whose ``verification.json`` must pass
        :func:`checked_verification`.
    identity : dict
        This process's identity.
    parts : path
        The directory of the campaign's work files, one per configuration
        (``<label>.json``), written after every case. A case whose entry
        there names the SHA-256 its files have now, made by code of this
        process's source identity, is taken from it and not rescored, so
        that a rescoring that stopped, or a finish run again, rescores only
        the cases it lacks.

    Returns
    -------
    dict
        The rescored metrics of the campaign. A campaign of this process's
        source identity must reproduce the in-run metrics of every case
        exactly, and raises :class:`campaign_contract.ContractError` when
        one does not.
    """
    campaign, parts = Path(campaign), Path(parts)
    manifest = read_manifest(campaign)
    verification = checked_verification(campaign, manifest)
    source = identity["source"]
    own = not contract.source_differences(identity, manifest["identity"])
    started = time.time()
    cases = {}
    counts = {"rescored": 0, "reused": 0, "not_verified": 0}
    work = {}
    for entry in verification["cases"]:
        tag, label, seed = parse_case(entry["case"])
        stem = golden_trace._tag(label, seed)
        if entry["status"] != "verified":
            cases[stem] = {"status": entry["status"]}
            counts["not_verified"] += 1
            continue
        record = contract.read_json(contract.record_path(campaign, tag))
        files = case_files(label, seed)
        hashes = {}
        for key in ("trace", "posterior", "sidecar"):
            rel = files[key]
            hashes[rel] = sha256(campaign / rel)
            if hashes[rel] != record["artifacts"][rel]["sha256"]:
                raise contract.ContractError(
                    f"{campaign / rel} is not the file its verification "
                    "checked"
                )
        if label not in work:
            work[label] = _read_part(parts / f"{label}.json", source)
        done = work[label]
        if stem in done and done[stem]["artifacts"] == hashes:
            cases[stem] = done[stem]
            counts["reused"] += 1
        else:
            cases[stem] = done[stem] = rescore_case(
                campaign, label, seed, hashes
            )
            write_json(
                parts / f"{label}.json",
                {
                    "campaign": campaign.name,
                    "rescoring_source": source,
                    "cases": done,
                },
            )
        counts["rescored"] += 1
        if counts["rescored"] % 50 == 0:
            print(
                f"[rescore] {campaign.name}: {counts['rescored']} cases "
                f"({counts['reused']} from earlier work), "
                f"{(time.time() - started) / 60:.1f} min",
                flush=True,
            )
    rescored = [
        case for case in cases.values() if case["status"] == "rescored"
    ]
    counts["equal_to_in_run"] = {
        key: sum(bool(case["equal_to_in_run"][key]) for case in rescored)
        for key in RESCORED_METRICS
    }
    counts["nonfinite"] = {
        key: sum(not case["finite"][key] for case in rescored)
        for key in RESCORED_METRICS
    }
    if own:
        unequal = sorted(
            stem
            for stem, case in cases.items()
            if case["status"] == "rescored"
            and not all(case["equal_to_in_run"].values())
        )
        if unequal:
            raise contract.ContractError(
                f"{campaign} ran the code of this process, and the rescored "
                f"metrics of {len(unequal)} of its cases differ from their "
                f"in-run metrics (the first {unequal[0]}; the work files "
                f"under {parts} hold them)"
            )
    return {
        "harness": "population_run",
        "contract": contract.CONTRACT_VERSION,
        "campaign": {
            "name": campaign.name,
            "path": str(campaign),
            "arm": manifest.get("arm"),
            "manifest_sha256": sha256(campaign / "manifest.json"),
            "verification_sha256": sha256(campaign / "verification.json"),
            "source": manifest["identity"]["source"],
        },
        "method": RESCORING_METHOD,
        "rescoring": {
            "identity": identity,
            "started": time.strftime(
                "%Y-%m-%dT%H:%M:%S%z", time.localtime(started)
            ),
            "finished": contract.now(),
            "elapsed_seconds": time.time() - started,
        },
        "counts": counts,
        "cases": cases,
    }


def cmd_rescore(args):
    """Rescore the campaign at ``--out`` and each ``--campaign``.

    Writes ``rescored/<name>.json`` for each campaign, then
    :data:`RESCORING` with this process's identity and the SHA-256 of each
    of those files, which the comparison of two arms checks. The work
    files lie under ``rescored/<name>.parts/``.
    """
    out = args.out.resolve()
    try:
        trees = source_trees()
    except contract.IdentityError as error:
        print(f"rescore refused: no identity ({error})", flush=True)
        return contract.EXIT_IDENTITY
    if trees["pyvbmc"] != ROOT:
        print(
            "rescore refused: it runs in a process of the harness checkout's "
            "own package (the release code), and PYVBMC_SOURCE names "
            f"{trees['pyvbmc']}",
            flush=True,
        )
        return contract.EXIT_IDENTITY
    manifest = read_manifest(out)
    try:
        identity = this_identity()
    except Exception as error:
        print(f"rescore refused: no identity ({error})", flush=True)
        return contract.EXIT_IDENTITY
    differing = contract.source_differences(identity, manifest["identity"])
    if differing:
        print(
            "rescore refused: it runs in the trees and environment of the "
            f"campaign at {out}, and this process's source identity differs "
            f"from its manifest's in {differing}",
            flush=True,
        )
        return contract.EXIT_IDENTITY
    campaigns = [out] + [Path(path).resolve() for path in args.campaign or ()]
    names = [campaign.name for campaign in campaigns]
    if len(set(names)) != len(names):
        raise SystemExit(f"the campaigns' directory names repeat: {names}")
    rescored = {}
    for campaign in campaigns:
        rel = f"rescored/{campaign.name}.json"
        report = rescore_campaign(
            campaign, identity, out / "rescored" / f"{campaign.name}.parts"
        )
        write_json(out / rel, report)
        rescored[campaign.name] = {
            "file": rel,
            "sha256": sha256(out / rel),
            "path": str(campaign),
            "manifest_sha256": report["campaign"]["manifest_sha256"],
            "verification_sha256": report["campaign"]["verification_sha256"],
        }
        counts = report["counts"]
        print(
            f"[rescore] {out / rel}: {counts['rescored']} cases rescored "
            f"({counts['reused']} from earlier work), equal to their in-run "
            f"metrics {counts['equal_to_in_run']}, not finite "
            f"{counts['nonfinite']}",
            flush=True,
        )
    write_json(
        out / RESCORING,
        {
            "harness": "population_run",
            "contract": contract.CONTRACT_VERSION,
            "campaign": out.name,
            "identity": identity,
            "method": RESCORING_METHOD,
            "finished": contract.now(),
            "rescored": rescored,
        },
    )
    return 0


@contextlib.contextmanager
def _awake():
    """Keep Windows from sleeping while idle (the display may sleep)."""
    if sys.platform != "win32":
        yield
        return
    import ctypes

    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        yield
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def cmd_run(args):
    out = args.out.resolve()
    manifest = read_manifest(out)
    lines = case_lines(manifest)
    write_case_list(out, lines)
    indices = (
        subset_indices(manifest, args.subset)
        if args.subset
        else range(1, len(lines) + 1)
    )
    selected = [lines[index - 1] for index in indices]
    if args.limit:
        selected = selected[: args.limit]
    started = time.time()
    with _awake(), FileLock(str(out / "campaign.lock"), timeout=0):
        for number, line in enumerate(selected, start=1):
            tag = contract.case_tag(line)
            if contract.record_path(out, tag).exists():
                continue
            log = out / "logs" / f"{tag}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            print(f"START {tag} ({number}/{len(selected)})", flush=True)
            command = [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "worker",
                "--out",
                str(out),
                "--case",
                line,
            ]
            with log.open("w", encoding="utf-8") as stream:
                code = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                ).returncode
            if code:
                print(f"FAILED {tag} (exit {code}); see {log}", flush=True)
                return code
            print(
                f"DONE {tag}; {(time.time() - started) / 3600:.2f} h",
                flush=True,
            )
        remaining = sum(
            not contract.record_path(out, contract.case_tag(line)).exists()
            for line in lines
        )
        if remaining:
            print(
                f"{remaining} of {len(lines)} cases have no completion "
                "record yet; verify and the finishing steps wait for them",
                flush=True,
            )
            return 0
        code = cmd_verify(argparse.Namespace(out=out))
        if code:
            return code
        for step in contract.finishing_steps(manifest):
            code = main([*step, "--out", str(out)])
            if code:
                return code
    return 0


COMMANDS = {
    "prepare": cmd_prepare,
    "cases": cmd_cases,
    "worker": cmd_worker,
    "verify": cmd_verify,
    "summarize": cmd_summarize,
    "rescore": cmd_rescore,
    "run": cmd_run,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="write the manifest")
    prepare.add_argument("--suite", default="production")
    prepare.add_argument("--seeds", default="0-99")
    prepare.add_argument("--labels", help="comma-separated; default all")
    prepare.add_argument(
        "--options", help="JSON merged over the default options"
    )
    prepare.add_argument("--arm", help="a name for this arm, recorded only")
    prepare.add_argument(
        "--confirmatory",
        type=Path,
        help="JSON overriding fields of the default confirmatory family",
    )
    prepare.add_argument(
        "--pair",
        type=Path,
        help="the prepared campaign of the other arm, which this one's "
        "rescore step rescores too",
    )
    cases = sub.add_parser("cases", help="print the case list or a subset")
    cases.add_argument("--subset")
    worker = sub.add_parser("worker", help="run one case")
    worker.add_argument("--case", required=True)
    sub.add_parser("verify", help="reconcile and re-check every case")
    sub.add_parser("summarize", help="write summary.md")
    rescore = sub.add_parser(
        "rescore",
        help="rescore verified campaigns with the release code, resuming",
    )
    rescore.add_argument("--campaign", type=Path, action="append")
    run = sub.add_parser("run", help="run the cases one after another")
    run.add_argument("--subset")
    run.add_argument("--limit", type=int)
    for command in sub.choices.values():
        command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if getattr(args, "limit", None) is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def main(argv=None):
    # The driver reads the case list in bash, which takes a carriage
    # return for part of a word.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(newline="\n")
        except AttributeError:
            pass
    args = parse_args(argv)
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
