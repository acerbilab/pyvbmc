"""Per-run artifacts of the S-VBMC run-pool campaign: write, reload, verify.

One finished VBMC run is stored as a plain-array snapshot through the
oracle codec (``pyvbmc/testing/oracles/_state.py``): ``<tag>.npz`` with
every array and ``<tag>.json`` with the tree, the scalars and the run's
metadata. ``dev/plans/svbmc-benchmark-campaign.md``, section "Per-run
artifact", is the contract implemented here; its consumers are the ELBO
debiasing estimator and the stacking comparison, which need the returned
posterior with all of ``stats``, the GP that produced those statistics
(``vbmc.get_gp(results["best_iter"])``), the parameter transformer, every
evaluation of the run and the run's ``optim_state``.

:func:`save_run` writes the pair and verifies it against the live
objects; :func:`load_run` rebuilds every object through the public
constructors; :func:`verify_run` re-runs the contract's checks on a
stored artifact, with the live run or on its own; :func:`filter_verdict`
applies the pool's two filters (the stability flag and the stacking bound
on ``J_sjk``).

The checks a saved artifact must pass, in the numbering of the plan:

(a) the rebuilt posterior's arrays and all of ``stats`` equal the live
    ones, and the rebuilt logger's live rows equal the live logger's
    (with NaN equal to NaN, since non-finite floats round-trip through a
    tag);
(b) the rebuilt GP predicts the live GP's values at the run's live
    evaluations within ``TOL_GP``;
(c) the recomputation gate: ``_gp_log_joint`` on the rebuilt posterior
    and GP alone reproduces the stored ``I_sk`` and ``J_sjk``. They are
    weight-independent, so a pruned posterior recomputes exactly, and the
    gate also runs without the live run. On the machine that generated
    the run the recomputation is bit-identical; on another machine its
    BLAS rounds differently, and the rebuilt posterior factors carry
    that rounding amplified by the condition number of the GP's kernel
    matrix, which reaches 1e17 on runs whose training set holds far-tail
    evaluations (the noisy Rosenbrock pool condition). The gate
    therefore allows each artifact ``TOL_STATS`` plus
    ``ROUNDING_FACTOR`` times machine epsilon times that condition
    number, relative to the largest stored value, and reports the
    condition number, so that amplified rounding passes and stays
    visible while a corrupted or mismatched artifact still fails;
(d) the float64 canary on the rebuilt state.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _entry in (str(ROOT), str(HERE)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)
# The campaign pins gpyreg to a frozen worktree (the plan, "GP library
# pin"). Importing this module imports PyVBMC and with it gpyreg, so the
# pin is honoured here too: a sys.path entry wins over the editable
# install, whose finder is appended to sys.meta_path.
_GPYREG_SOURCE = os.environ.get("PYVBMC_GPYREG_SOURCE")
if _GPYREG_SOURCE and _GPYREG_SOURCE not in sys.path:
    sys.path.insert(1, _GPYREG_SOURCE)

import numpy as np  # noqa: E402
from profile_run import effective_options, jsonable  # noqa: E402

from pyvbmc.testing._dtype import (  # noqa: E402
    assert_float64,
    load_bearing_arrays,
)
from pyvbmc.testing.oracles._state import (  # noqa: E402
    build_state,
    encode,
    load_snapshot,
    save_snapshot,
    snapshot_from_objects,
)
from pyvbmc.vbmc.variational_optimization import _gp_log_joint  # noqa: E402

SUFFIXES = (".npz", ".json")
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
#: Agreement required of the rebuilt GP's predictions (check b).
TOL_GP = 1e-10
#: Absolute agreement required of the recomputed expected log joint
#: (check c), what a recomputation on the generating machine meets.
TOL_STATS = 1e-8
#: Multiples of ``eps * cond(K)`` the gate allows, relative to the largest
#: stored value, for the rounding another machine's BLAS introduces into
#: the rebuilt posterior factors; 0 makes the gate the absolute
#: ``TOL_STATS`` alone.
ROUNDING_FACTOR = 100.0
#: The paper's stacking filter: ``sqrt(max J_sjk) < sqrt(5)``.
S_MAX = float(np.sqrt(5))
#: The stats keys a pool posterior must carry.
STATS_KEYS = (
    "I_sk",
    "J_sjk",
    "e_log_joint",
    "e_log_joint_sd",
    "elbo",
    "elbo_sd",
    "entropy",
    "entropy_sd",
    "stable",
    "uncertainty_handling_level",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact_paths(out_dir, tag):
    """The two files of one run: ``(<tag>.npz, <tag>.json)``."""
    out_dir = Path(out_dir)
    return out_dir / f"{tag}.npz", out_dir / f"{tag}.json"


def record_path(out_dir, tag):
    """The completion record of one run, ``records/<tag>.complete.json``."""
    return Path(out_dir) / "records" / f"{tag}.complete.json"


def artifact_hashes(out_dir, tag):
    return {s: sha256(Path(out_dir) / f"{tag}{s}") for s in SUFFIXES}


def recorded_hashes(record, name):
    """``{suffix: SHA-256}`` of a run's two files, from its completion record.

    ``name`` is the artifact's file name without suffix. A record of the
    campaign contract lists every artifact file under ``artifacts`` by its
    path relative to the campaign directory; a record of the flat layout
    holds ``hashes`` by suffix. A file the record does not list maps to
    None.
    """
    if "hashes" in record:
        return {suffix: record["hashes"].get(suffix) for suffix in SUFFIXES}
    listed = {
        PurePosixPath(path).name: entry.get("sha256")
        for path, entry in (record.get("artifacts") or {}).items()
    }
    return {suffix: listed.get(f"{name}{suffix}") for suffix in SUFFIXES}


def _git(path, *args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def installed_version(name):
    """The version the installed distribution ``name`` names, or None.

    None where no distribution of that name is installed: a package
    imported from a source tree on ``sys.path`` has none.
    """
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def environment(gpyreg_source=None):
    """This process's identity, in two halves: ``source`` and ``host``.

    ``source`` is the code and library state a pool is generated by:
    commits and the versions of Python, NumPy and SciPy, to which
    ``svbmc_pool_run.identity`` adds the working-tree state and the hashes
    of the harness modules. Every process of one campaign must agree on
    it, so that a resumed or distributed campaign cannot mix versions.
    ``host`` is where the process ran: hostname, platform, interpreter,
    import paths, the thread environment and, under
    ``installed_metadata_versions``, the versions that the installed
    distributions of PyVBMC and gpyreg name. It is recorded and never
    compared, so that any node may run any case of a campaign.

    Every version read from an installed distribution's metadata
    (:func:`installed_version`) is None where no distribution of that name
    is installed, as in the campaign environment of the Slurm plan, which
    imports PyVBMC and gpyreg from their source trees on ``sys.path``.

    With ``gpyreg_source`` given, raises unless gpyreg was imported from
    ``<gpyreg_source>/gpyreg``: every process of a campaign must use the
    frozen worktree the manifest names.
    """
    import gpyreg

    import pyvbmc

    gp_dir = Path(gpyreg.__file__).resolve().parent
    if gpyreg_source is not None:
        expected = Path(gpyreg_source).resolve() / "gpyreg"
        if gp_dir != expected:
            raise RuntimeError(
                f"gpyreg imported from {gp_dir}, not the campaign's "
                f"frozen source {expected}"
            )
    return {
        "source": {
            "python": sys.version.split()[0],
            "pyvbmc_commit": _git(ROOT, "rev-parse", "HEAD"),
            "gpyreg_commit": _git(gp_dir.parent, "rev-parse", "HEAD"),
            "numpy": installed_version("numpy"),
            "scipy": installed_version("scipy"),
        },
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "executable": sys.executable,
            "pyvbmc_import": str(Path(pyvbmc.__file__).resolve().parent),
            "gpyreg_import": str(gp_dir),
            # What the installed distributions name, which an editable
            # install freezes at install time and which a tree imported by
            # path need not match: provenance of the node, not of the
            # source.
            "installed_metadata_versions": {
                name: installed_version(name) for name in ("pyvbmc", "gpyreg")
            },
            "threads": {k: os.environ.get(k) for k in THREAD_KEYS},
        },
    }


def filter_verdict(vp, s_max=S_MAX):
    """The pool filters on a finished posterior.

    Returns ``stable`` (the run's stability flag), ``max_J_sjk`` (the
    maximum over the whole ``(Ns, K, K)`` array) and ``passes``, true when
    the run is stable and ``sqrt(max_J_sjk) < s_max``. These are the two
    filters of the S-VBMC paper and the default of both implementations.
    Raises unless the posterior carries the two statistics they read.
    """
    stats = vp.stats
    if stats is None or not {"stable", "J_sjk"} <= set(stats):
        raise RuntimeError(
            "filter_verdict needs a finished posterior, whose stats carry "
            "'stable' and 'J_sjk'"
        )
    stable = bool(stats["stable"])
    max_J_sjk = float(np.max(stats["J_sjk"]))
    return {
        "stable": stable,
        "max_J_sjk": max_J_sjk,
        "s_max": float(s_max),
        "passes": bool(stable and np.sqrt(max_J_sjk) < s_max),
    }


def load_run(path, rng=None):
    """Rebuild one stored run through the public constructors.

    Returns a dict with keys ``vp``, ``gp``, ``pt``, ``logger``,
    ``optim_state``, ``options``, ``meta``.

    ``path`` is the artifact's base name, without the ``.npz`` / ``.json``
    suffix. ``rng`` (an integer seed or a generator) becomes the rebuilt
    posterior's generator; without it the codec's fixed seed is used, so
    nothing reads NumPy's global state. The logger is rebuilt without a
    target, since no consumer evaluates one.
    """
    snapshot = load_snapshot(path)
    state = build_state(
        snapshot,
        fun=None,
        rng=None if rng is None else np.random.default_rng(rng),
    )
    return {k: state[k] for k in load_run.KEYS}


load_run.KEYS = ("vp", "gp", "pt", "logger", "optim_state", "options", "meta")


def _same(a, b):
    """Array equality with NaN equal to NaN, over any dtype."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if np.issubdtype(a.dtype, np.floating) and np.issubdtype(
        b.dtype, np.floating
    ):
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def _max_abs(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def gp_condition_number(gp):
    """The largest condition number of the GP's kernel matrices.

    One per hyperparameter sample, from the stored posterior factor: in
    gpyreg's high-noise parametrization ``L`` is the Cholesky factor of
    the scaled kernel matrix, whose condition number is that of ``L``
    squared; otherwise ``L`` holds the negative inverse of the kernel
    matrix and shares its condition number.
    """
    conditions = []
    for posterior in gp.posteriors:
        L = np.asarray(posterior.L, dtype=float)
        condition = float(np.linalg.cond(L))
        conditions.append(condition**2 if posterior.L_chol else condition)
    return max(conditions)


def gate_tolerance(stored, condition, rounding_factor):
    """The recomputation gate's tolerance for one stored array."""
    scale = float(np.max(np.abs(stored))) if np.size(stored) else 0.0
    # A factor of 0 is the absolute gate whatever the condition number,
    # a singular factor included (its condition number is infinite).
    rounding = (
        rounding_factor * np.finfo(float).eps * condition
        if rounding_factor > 0
        else 0.0
    )
    return TOL_STATS + rounding * scale


def verify_run(
    path, vbmc=None, results=None, rounding_factor=None, record=None
):
    """Check one stored run against its contract; return a report.

    Always runs the recomputation gate (c) and the float64 canary (d) on
    the rebuilt state. With the live ``vbmc`` and its ``results`` it also
    runs (a) and (b) against the objects the run returned, and compares
    the rebuilt evaluations with the live ones. Checked on its own
    instead, and with a completion record, it compares both files against
    the hashes that record holds (:func:`recorded_hashes`); a record is
    written after the artifact it describes, so with live objects it
    belongs to an earlier attempt and is not read. ``record`` is the
    record's path; by default it is the flat layout's, ``records/`` beside
    the artifact, and the check is left out where no record exists. Raises
    ``RuntimeError`` naming every check that failed.

    The gate allows ``TOL_STATS`` plus ``rounding_factor`` (default
    ``ROUNDING_FACTOR``) times machine epsilon times the condition number
    of the GP's kernel matrix, relative to the largest stored value; the
    report carries the absolute and relative differences, the condition
    number and the tolerance each array was held to.
    """
    if rounding_factor is None:
        rounding_factor = ROUNDING_FACTOR
    path = Path(path)
    tag = path.name
    state = load_run(path)
    vp, gp, logger, pt = (
        state["vp"],
        state["gp"],
        state["logger"],
        state["pt"],
    )
    report = {"tag": tag, "checks": [], "differences": {}}
    failures = []

    missing = [k for k in STATS_KEYS if k not in (vp.stats or {})]
    if missing:
        failures.append(f"stats missing {missing}")
    report["checks"].append("stats_keys")

    # (c) the recomputation gate, on the rebuilt posterior and GP alone.
    out = _gp_log_joint(vp, gp, False, True, True, True, True)
    condition = gp_condition_number(gp)
    report["condition_number"] = condition
    report["rounding_factor"] = float(rounding_factor)
    report["relative"] = {}
    report["tolerance"] = {}
    for key, value in (("I_sk", out[5]), ("J_sjk", out[6])):
        stored = vp.stats[key]
        difference = _max_abs(value, stored)
        scale = float(np.max(np.abs(stored))) if np.size(stored) else 0.0
        tolerance = gate_tolerance(stored, condition, rounding_factor)
        report["differences"][key] = difference
        report["relative"][key] = difference / scale if scale else difference
        report["tolerance"][key] = tolerance
        if not difference <= tolerance:
            failures.append(
                f"recomputed {key} differs by {difference:.3e} "
                f"(tolerance {tolerance:.3e}: {TOL_STATS:g} plus "
                f"{rounding_factor:g} eps times the condition number "
                f"{condition:.3e}, relative to {scale:.3e})"
            )
    report["checks"].append("recomputation_gate")

    # (d) the float64 canary on the rebuilt state.
    report["dtype_leaves"] = assert_float64(
        load_bearing_arrays(vp=vp, gp=gp, logger=logger, pt=pt)
    )
    report["checks"].append("dtype_canary")

    record = record_path(path.parent, tag) if record is None else Path(record)
    if vbmc is None and record.exists():
        stored = recorded_hashes(
            json.loads(record.read_text(encoding="utf-8")), tag
        )
        actual = artifact_hashes(path.parent, tag)
        for suffix in SUFFIXES:
            if stored.get(suffix) != actual[suffix]:
                failures.append(f"{suffix} differs from the recorded hash")
        report["checks"].append("hashes")
        report["hashes"] = actual

    if vbmc is not None:
        if results is None:
            raise ValueError("verify_run needs `results` with a live `vbmc`")
        live_vp = vbmc.vp
        live_gp = vbmc.get_gp(results["best_iter"])
        # (a) posterior arrays and every statistic.
        for name in ("w", "eta", "mu", "sigma", "lambd"):
            if not _same(getattr(vp, name), getattr(live_vp, name)):
                failures.append(f"rebuilt vp.{name} differs")
        if set(vp.stats) != set(live_vp.stats):
            failures.append("rebuilt vp.stats has different keys")
        for key in set(vp.stats) & set(live_vp.stats):
            if not _same(vp.stats[key], live_vp.stats[key]):
                failures.append(f"rebuilt vp.stats[{key!r}] differs")
        report["checks"].append("posterior")
        # (a) the evaluations, live rows against live rows: the stored
        # arrays are trimmed to the logger's filled rows, so the two
        # objects carry the same evaluations in arrays of different
        # length.
        live_logger = vbmc.function_logger
        rows, live_rows = (
            np.ravel(logger.X_flag),
            np.ravel(live_logger.X_flag),
        )
        if int(np.sum(rows)) != int(np.sum(live_rows)):
            failures.append("rebuilt logger holds a different row count")
        else:
            for name in ("X_orig", "y_orig", "S", "n_evals"):
                live_value = getattr(live_logger, name, None)
                if live_value is None:
                    continue
                value = getattr(logger, name, None)
                if value is None or not _same(
                    np.asarray(value)[rows],
                    np.asarray(live_value)[live_rows],
                ):
                    failures.append(f"rebuilt logger.{name} differs")
        report["checks"].append("evaluations")
        # (b) GP predictions at the run's live evaluations.
        X = np.asarray(live_logger.X)[np.ravel(live_logger.X_flag)]
        difference = max(
            _max_abs(a, b)
            for a, b in zip(
                gp.predict(X, separate_samples=True),
                live_gp.predict(X, separate_samples=True),
            )
        )
        report["differences"]["gp_predict"] = difference
        if not difference <= TOL_GP:
            failures.append(
                f"rebuilt GP predictions differ by {difference:.3e} "
                f"(tolerance {TOL_GP:g})"
            )
        report["checks"].append("gp_prediction")

    if failures:
        raise RuntimeError(f"{tag}: " + "; ".join(failures))
    return report


def encode_meta(meta):
    """The metadata as strict JSON, with non-finite floats tagged.

    The snapshot's sidecar is written with ``allow_nan=False`` and the
    codec stores ``meta`` as it is given, so a metric that is NaN because
    a truth is unknown would make the file unwritable. Tagging it the way
    the codec tags every other non-finite float keeps it readable and
    round-trips it back to NaN. Arrays belong in the ``.npz``, so any
    left in the metadata are an error: pass them through
    ``profile_run.jsonable`` first.
    """
    arrays = {}
    tree = encode(meta, "meta", arrays)
    if arrays:
        raise TypeError(f"meta holds arrays: {sorted(arrays)}")
    return tree


def save_run(
    out_dir,
    tag,
    vbmc,
    results,
    problem,
    cfg,
    seed,
    timing,
    meta_extra=None,
):
    """Write one finished run's artifact and verify it against the run.

    ``timing`` is a mapping of named seconds (``wall_s`` at least) and
    ``meta_extra`` anything the caller adds to the metadata (the metrics
    against the truth and the filter verdict, for the two pool scripts).
    Returns the verification report with the two paths and their hashes.
    """
    out_dir = Path(out_dir)
    gp = vbmc.get_gp(results["best_iter"])
    logger = vbmc.function_logger
    requested = jsonable(dict(vbmc.options.user_options or {}))
    meta = {
        "campaign": "svbmc_pool",
        "label": cfg.label,
        "seed": int(seed),
        "target": problem.name,
        "D": int(problem.D),
        "noise_sd": (
            None if problem.noise_sd is None else float(problem.noise_sd)
        ),
        "requested_options": requested,
        "effective_options": effective_options(vbmc, requested.keys()),
        "results": jsonable(results),
        "K": int(vbmc.vp.K),
        "func_count": int(results["func_count"]),
        "target_eval_s": float(logger.total_fun_eval_time),
        "best_iter": int(results["best_iter"]),
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
        "identity": environment(),
    }
    meta.update({k: float(v) for k, v in dict(timing).items()})
    meta.update(meta_extra or {})
    arrays, tree = snapshot_from_objects(
        vbmc.vp,
        gp,
        logger,
        vbmc._optim_state_record(),
        vbmc.options,
        meta=encode_meta(meta),
    )
    path = out_dir / tag
    save_snapshot(path, arrays, tree)
    report = verify_run(path, vbmc=vbmc, results=results)
    npz, sidecar = artifact_paths(out_dir, tag)
    report["files"] = {".npz": str(npz), ".json": str(sidecar)}
    report["hashes"] = artifact_hashes(out_dir, tag)
    report["bytes"] = {
        s: (out_dir / f"{tag}{s}").stat().st_size for s in SUFFIXES
    }
    return report
