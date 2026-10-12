"""Record the six seeded gate runs twice, with the provenance they lack.

The gate script,
``dev/experiments/port_review_20260919/verification/scripts/wave2_fixpass_gate_runs.py``,
holds six short seeded runs at the default options: the four that gated the
fix passes of the port review, and two on a two-dimensional Rosenbrock
likelihood that pass a prior object with ``prior=``. It saves every number
that describes their trajectories in one ``.npz`` and compares two such
files bit for bit (``--compare``), but records nothing of the code, the
machine or the settings it ran with. This wrapper runs them for the replay
fingerprints of the release gate (``dev/plans/slurm-benchmark-support.md``,
Phase 9), whose promotion read what it writes
(``dev/golden/promotion_20261007/promote.py``)::

    python -u dev/scripts/seeded_gate_runs.py run --out DIR
    python dev/scripts/seeded_gate_runs.py check DIR

``run`` records the six runs twice, each time in a fresh process of this
script (``record``, which ``run`` starts), and compares the two recordings
with the gate script's own ``compare``. Each recording imports PyVBMC from
this checkout and gpyreg from the checkout that ``PYVBMC_GPYREG_SOURCE``
names, or else from the checkout the imported gpyreg lies in; runs with one
BLAS thread (``OMP_NUM_THREADS``, ``OPENBLAS_NUM_THREADS`` and
``MKL_NUM_THREADS`` set to 1); and adds ``performance_calibration="off"`` to
every run's options, so that no machine calibration enters the runs (the
option takes the historical chunk settings). ``DIR`` must be new or empty,
and receives, for each recording ``first`` and ``second``:

    <name>.npz     the gate script's arrays, as its own ``--out`` writes them
    <name>.log     the recording's output
    <name>.json    its record: the identity of its process
                   (``campaign_contract.identity``: the commit and clean
                   state of this checkout and of the gpyreg checkout, the
                   SHA-256 of this file, the imported versions and import
                   paths, and the host part: the hostname, the CPU, the
                   BLAS libraries with their thread counts and the thread
                   variables), the gate script's path and SHA-256, and
                   each run's digest (the SHA-256 the gate script prints),
                   counts, final ELBO and seconds

and ``gate_runs.json``, which holds both records, the SHA-256 of every file
above, the comparison, and whether the recordings are identical: they are
when the comparison finds no difference and both processes had one source
identity, from clean checkouts unless ``--allow-dirty``. ``DIR`` must be
new or empty and, inside this checkout, a path git ignores (under
``dev/scripts/runs/``, say), since the recordings' files would otherwise
make the checkout dirty between them. Without ``PYVBMC_GPYREG_SOURCE`` the
checkout that holds the imported gpyreg must track its ``__init__.py``, so
that a gpyreg installed into an environment inside another checkout is not
taken for that checkout. The exit code is 0 when the recordings are
identical, 1 when they differ or a recording fails, 64 for a ``DIR`` that is
a file, holds files or lies unignored in the checkout, and 78 when a
process has no identity (PyVBMC or gpyreg does not import from a checkout)
or, without ``--allow-dirty``, a checkout is not clean: a record of a dirty
tree names no commit that reproduces it.

``check`` re-reads a finished ``DIR``: every file must be the one
``gate_runs.json`` hashes and each run's arrays the ones its digest names,
the recordings must hold the runs of :data:`GATE_RUNS` with
:data:`OPTIONS_ADDED`, one source identity and, unless the record allowed
dirty checkouts, clean ones, and compare identical again, and the gate
script at its recorded path must be the one they ran (compared with LF
line endings, as git may check the script out with CRLF). It prints what
fails and exits 1 if anything does. ``--script`` names another gate
script, one that defines ``build_runs``, ``record`` and ``compare`` as the
gate script does; the tests use it, and ``check`` refuses its records for
their runs.
"""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
for key in THREAD_KEYS:
    os.environ[key] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")
# This checkout's package, whichever checkout is installed, and gpyreg from
# the checkout that PYVBMC_GPYREG_SOURCE names, where it names one.
sys.path.insert(0, str(ROOT))
if os.environ.get("PYVBMC_GPYREG_SOURCE"):
    sys.path.insert(0, os.environ["PYVBMC_GPYREG_SOURCE"])
sys.path.insert(0, str(HERE))

import campaign_contract as contract  # noqa: E402

#: The gate script, relative to the checkout.
GATE_SCRIPT = (
    "dev/experiments/port_review_20260919/verification/scripts/"
    "wave2_fixpass_gate_runs.py"
)
#: The runs the gate script holds, in its order.
GATE_RUNS = (
    "rosenbrock_D2_unbounded",
    "two_blobs_D3_bounded",
    "rosenbrock_D2_noisy",
    "two_blobs_D3_noisy_user_budget",
    "rosenbrock_D2_uniform_prior",
    "rosenbrock_D2_spline_trapezoidal_prior",
)
#: What this wrapper adds to every run's options.
OPTIONS_ADDED = {"performance_calibration": "off"}
#: The position of the options dict in the gate script's run specifications
#: (target factory, D, lb, ub, plb, pub, x0, options, seed[, prior factory]).
OPTIONS_INDEX = 7
RECORDINGS = ("first", "second")
RECORD = "gate_runs.json"
EXIT_FAILED = 1
EXIT_USAGE = 64
EXIT_IDENTITY = contract.EXIT_IDENTITY


class Refusal(Exception):
    """A run that cannot start; ``code`` is its exit code."""

    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def load_script(path):
    """The gate script at ``path``, imported as a module."""
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def script_path(value):
    """The gate script's path, relative to the checkout where it lies in it."""
    path = Path(value or ROOT / GATE_SCRIPT).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve(relative):
    path = Path(relative)
    return path if path.is_absolute() else ROOT / path


def gpyreg_tree():
    """The gpyreg checkout of the recordings.

    ``PYVBMC_GPYREG_SOURCE`` where it is set, and otherwise the checkout
    that holds the gpyreg this process imports.
    """
    named = os.environ.get("PYVBMC_GPYREG_SOURCE")
    if named:
        return Path(named).resolve()
    try:
        import gpyreg
    except ImportError as error:
        raise contract.IdentityError(
            f"gpyreg cannot be imported: {error}"
        ) from error
    init = Path(gpyreg.__file__).resolve()
    tree = contract.enclosing_checkout(init.parent)
    try:
        contract.git(
            tree, "ls-files", "--error-unmatch", init.relative_to(tree)
        )
    except (OSError, TypeError, ValueError, subprocess.CalledProcessError):
        raise contract.IdentityError(
            f"gpyreg is imported from {init.parent}, which no git checkout "
            f"tracks (the enclosing one is {tree}); name gpyreg's checkout "
            "with PYVBMC_GPYREG_SOURCE"
        ) from None
    return tree


def this_identity(host=True):
    """The identity of this process (``campaign_contract.identity``): this
    checkout, with this file's SHA-256, and the gpyreg checkout."""
    return contract.identity(
        {"harness": ROOT, "gpyreg": gpyreg_tree()},
        [Path(__file__).resolve().relative_to(ROOT).as_posix()],
        modules={"pyvbmc": "harness", "gpyreg": "gpyreg"},
        host=host,
    )


def text_sha256(path):
    """The SHA-256 of a text file with its line endings read as LF."""
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def script_record(script):
    """The gate script's path, as :func:`script_path` gives it, and its
    SHA-256 with LF line endings."""
    return {"path": script, "sha256": text_sha256(resolve(script))}


def dirty_trees(identity):
    return sorted(
        name
        for name, tree in identity["source"]["trees"].items()
        if not tree["clean"]
    )


def unignored_in_checkout(path):
    """Whether ``path`` lies in this checkout where git does not ignore it."""
    path = Path(path).resolve()
    if ROOT not in path.parents:
        return False
    result = subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "-q", str(path)],
        capture_output=True,
        stdin=subprocess.DEVNULL,
    )
    return result.returncode != 0


def verdict(record, allow_dirty):
    """Whether the recordings of ``record`` are identical, its exit code and
    the reasons for a verdict other than identical.

    A recording that exits 78 had no identity; any other failure is exit 1,
    whatever the child's own code (a crashed process's code on Windows is
    too large for an exit status). The recordings are identical when both
    completed, the comparison finds no difference, both processes had one
    source identity, and, unless ``allow_dirty``, every checkout was clean.
    """
    codes = record["exit_codes"]
    failed = [code for code in codes.values() if code != 0]
    if failed or sorted(codes) != sorted(RECORDINGS):
        code = EXIT_IDENTITY if EXIT_IDENTITY in failed else EXIT_FAILED
        return False, code, [f"a recording exited {failed}"]
    identities = [
        record["recordings"][name]["record"]["identity"] for name in RECORDINGS
    ]
    reasons = []
    if not record["comparison"]["identical"]:
        reasons.append("the comparison finds differences")
    differing = contract.source_differences(*identities)
    if differing:
        reasons.append(
            f"the recordings' source identities differ in {differing}"
        )
    dirty = sorted({tree for i in identities for tree in dirty_trees(i)})
    if dirty and not allow_dirty:
        reasons.append(f"the checkouts {dirty} were not clean")
    if not reasons:
        return True, 0, []
    code = EXIT_IDENTITY if dirty and not allow_dirty else EXIT_FAILED
    return False, code, reasons


def with_options_added(runs):
    """The gate script's runs with :data:`OPTIONS_ADDED` in their options."""
    changed = {}
    for name, spec in runs.items():
        spec = list(spec)
        options = spec[OPTIONS_INDEX]
        if not isinstance(options, dict):
            raise TypeError(
                f"{name}: the specification holds no options dict at "
                f"position {OPTIONS_INDEX}"
            )
        clash = {
            key: options[key]
            for key, value in OPTIONS_ADDED.items()
            if key in options and options[key] != value
        }
        if clash:
            raise ValueError(f"{name}: the gate script sets {clash}")
        spec[OPTIONS_INDEX] = {**options, **OPTIONS_ADDED}
        changed[name] = tuple(spec)
    return changed


def run_digest(arrays, name):
    """The SHA-256 the gate script prints for run ``name``."""
    import numpy as np

    digest = hashlib.sha256()
    prefix = f"{name}/"
    for key in sorted(k for k in arrays if k.startswith(prefix)):
        digest.update(np.ascontiguousarray(arrays[key]).tobytes())
    return digest.hexdigest()


def run_summary(arrays, name):
    """The counts and the final ELBO of run ``name``."""
    final_elbo = arrays[f"{name}/final_elbo"]
    return {
        "iterations": int(len(arrays[f"{name}/elbo"])),
        "func_count": int(arrays[f"{name}/func_count"][-1]),
        "elbo": float(final_elbo[0]),
        "elbo_sd": float(final_elbo[1]),
        "best_iter": int(arrays[f"{name}/best_iter"][0]),
        "n_warps": int(arrays[f"{name}/n_warps"][0]),
    }


# --------------------------------------------------------------------------
# One recording
# --------------------------------------------------------------------------


def cmd_record(args):
    """One recording of the six runs, in this process (module docstring)."""
    import numpy as np

    out = Path(args.out).resolve()
    script = script_path(args.script)
    identity = this_identity()
    import gpyreg

    import pyvbmc

    gate = load_script(resolve(script))
    runs = with_options_added(gate.build_runs())
    logging.disable(logging.CRITICAL)
    print("pyvbmc:", pyvbmc.__file__, flush=True)
    print("gpyreg:", gpyreg.__file__, flush=True)
    started = contract.now()
    t0 = time.perf_counter()
    arrays, seconds = {}, {}
    for name, spec in runs.items():
        t_run = time.perf_counter()
        arrays.update(gate.record(name, spec))
        seconds[name] = time.perf_counter() - t_run
    elapsed = time.perf_counter() - t0
    np.savez(out / f"{args.name}.npz", **arrays)
    contract.write_json(
        out / f"{args.name}.json",
        {
            "recording": args.name,
            "script": script_record(script),
            "options_added": OPTIONS_ADDED,
            "identity": identity,
            "runs": {
                name: {
                    "sha256": run_digest(arrays, name),
                    **run_summary(arrays, name),
                    "seconds": seconds[name],
                }
                for name in runs
            },
            "started": started,
            "finished": contract.now(),
            "elapsed_seconds": elapsed,
        },
    )
    print(f"saved {out / f'{args.name}.npz'}", flush=True)
    return 0


# --------------------------------------------------------------------------
# Two recordings and their comparison
# --------------------------------------------------------------------------


def compare_recordings(gate, directory):
    """The gate script's ``compare`` of the two recordings, captured."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = gate.compare(
            str(directory / f"{RECORDINGS[0]}.npz"),
            str(directory / f"{RECORDINGS[1]}.npz"),
        )
    return {
        "exit_code": code,
        "identical": code == 0,
        "output": buffer.getvalue(),
    }


def file_entry(path):
    return {
        "sha256": contract.sha256_file(path),
        "bytes": Path(path).stat().st_size,
    }


def start_recording(out, name, script, gpyreg):
    """Run ``record`` in a fresh process; stream its output to its log."""
    env = dict(os.environ)
    env.update({key: "1" for key in THREAD_KEYS})
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYVBMC_GPYREG_SOURCE"] = str(gpyreg)
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "record",
        "--out",
        str(out),
        "--name",
        name,
        "--script",
        str(resolve(script)),
    ]
    with open(out / f"{name}.log", "w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            cwd=ROOT,
        )
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(f"[{name}] {line}", end="", flush=True)
        return process.wait()


def cmd_run(args):
    out = Path(args.out).resolve()
    if out.exists() and not out.is_dir():
        raise Refusal(EXIT_USAGE, f"{out} is a file, not a directory")
    if out.exists() and any(out.iterdir()):
        raise Refusal(
            EXIT_USAGE, f"{out} is not empty; a record is not revised"
        )
    if unignored_in_checkout(out):
        raise Refusal(
            EXIT_USAGE,
            f"{out} lies in the checkout where git does not ignore it, so "
            "the first recording's files would make the checkout dirty for "
            "the second; write the record under dev/scripts/runs/ or outside "
            "the checkout",
        )
    script = script_path(args.script)
    try:
        identity = this_identity(host=False)
    except contract.ContractError as error:
        raise Refusal(EXIT_IDENTITY, f"no identity: {error}") from error
    dirty = dirty_trees(identity)
    if dirty and not args.allow_dirty:
        raise Refusal(
            EXIT_IDENTITY,
            f"the checkouts {dirty} are not clean, so the record would name "
            "no commit that reproduces it; commit, or pass --allow-dirty",
        )
    gpyreg = Path(identity["imports"]["trees"]["gpyreg"]["path"])
    out.mkdir(parents=True, exist_ok=True)
    started = contract.now()
    codes = {}
    for name in RECORDINGS:
        print(f"[run] recording {name} into {out}", flush=True)
        codes[name] = start_recording(out, name, script, gpyreg)
        if codes[name] != 0:
            print(f"[run] recording {name} exited {codes[name]}", flush=True)
            break
    record = {
        "tool": "seeded_gate_runs",
        "script": script_record(script),
        "options_added": OPTIONS_ADDED,
        "allow_dirty": bool(args.allow_dirty),
        "started": started,
        "finished": contract.now(),
        "exit_codes": codes,
        "recordings": {},
    }
    if all(codes.get(name) == 0 for name in RECORDINGS):
        gate = load_script(resolve(script))
        record["comparison"] = compare_recordings(gate, out)
        for name in RECORDINGS:
            record["recordings"][name] = {
                "files": {
                    suffix: {
                        "file": f"{name}.{suffix}",
                        **file_entry(out / f"{name}.{suffix}"),
                    }
                    for suffix in ("npz", "json", "log")
                },
                "record": contract.read_json(out / f"{name}.json"),
            }
        identities = [
            record["recordings"][name]["record"]["identity"]
            for name in RECORDINGS
        ]
        record["same_source_identity"] = not contract.source_differences(
            *identities
        )
    identical, code, reasons = verdict(record, args.allow_dirty)
    record.update(identical=identical, exit_code=code, reasons=reasons)
    contract.write_json(out / RECORD, record)
    if "comparison" in record:
        print(record["comparison"]["output"], end="", flush=True)
    print(
        f"[run] {'identical' if identical else 'NOT identical'}"
        + "".join(f"; {reason}" for reason in reasons)
        + f"; record {out / RECORD}",
        flush=True,
    )
    return code


# --------------------------------------------------------------------------
# The check of a finished record
# --------------------------------------------------------------------------


def check_record(directory):
    """Re-check a finished record (module docstring); return it and the
    problems found, an empty list when there are none."""
    import numpy as np

    directory = Path(directory)
    record = contract.read_json(directory / RECORD)
    problems = []
    if record.get("exit_code") != 0 or not record.get("identical"):
        problems.append("the record is not of two identical recordings")
    recordings = record.get("recordings") or {}
    if sorted(recordings) != sorted(RECORDINGS):
        problems.append(
            f"the record holds the recordings {sorted(recordings)}"
        )
        return record, problems
    for name in RECORDINGS:
        for entry in recordings[name]["files"].values():
            path = directory / entry["file"]
            if not path.is_file():
                problems.append(f"{entry['file']} is missing")
            elif contract.sha256_file(path) != entry["sha256"]:
                problems.append(f"{entry['file']} is not the file recorded")
        if (directory / f"{name}.json").is_file() and contract.read_json(
            directory / f"{name}.json"
        ) != recordings[name]["record"]:
            problems.append(
                f"{name}.json is not the record gate_runs.json holds"
            )
        runs = recordings[name]["record"]["runs"]
        if list(runs) != list(GATE_RUNS):
            problems.append(
                f"{name} holds the runs {list(runs)}, not those of the gate "
                f"script ({list(GATE_RUNS)})"
            )
        if recordings[name]["record"]["options_added"] != OPTIONS_ADDED:
            problems.append(f"{name} ran with other options")
        npz = directory / f"{name}.npz"
        if npz.is_file():
            with np.load(npz, allow_pickle=False) as data:
                arrays = {key: data[key] for key in data.files}
            held = {key.split("/", 1)[0] for key in arrays}
            if held != set(runs):
                problems.append(f"{name}.npz holds the runs {sorted(held)}")
            problems += [
                f"{name}.npz: the arrays of {run} are not the ones its "
                "digest names"
                for run in runs
                if run_digest(arrays, run) != runs[run]["sha256"]
            ]
    identities = [
        recordings[name]["record"]["identity"] for name in RECORDINGS
    ]
    differing = contract.source_differences(*identities)
    if differing:
        problems.append(
            f"the recordings' source identities differ in {differing}"
        )
    dirty = sorted({tree for i in identities for tree in dirty_trees(i)})
    if dirty and not record.get("allow_dirty"):
        problems.append(f"the checkouts {dirty} were not clean")
    if any(
        recordings[name]["record"]["script"] != record["script"]
        for name in RECORDINGS
    ):
        problems.append("the recordings ran another gate script")
    script = resolve(record["script"]["path"])
    if not script.is_file():
        problems.append(
            f"the gate script {record['script']['path']} is missing"
        )
    elif text_sha256(script) != record["script"]["sha256"]:
        problems.append(
            f"the gate script {record['script']['path']} is not the one the "
            "recordings ran"
        )
    elif not problems:
        again = compare_recordings(load_script(script), directory)
        if not again["identical"]:
            problems.append("the recordings no longer compare identical")
    return record, problems


def cmd_check(args):
    record, problems = check_record(args.directory)
    for problem in problems:
        print(f"check: {problem}", flush=True)
    if not problems:
        first = record["recordings"][RECORDINGS[0]]["record"]
        trees = first["identity"]["source"]["trees"]
        print(
            f"check: {len(first['runs'])} runs recorded twice, identical; "
            f"checkout {trees['harness']['commit']}, gpyreg "
            f"{trees['gpyreg']['commit']}, host "
            f"{first['identity']['host']['hostname']}",
            flush=True,
        )
    return EXIT_FAILED if problems else 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="record the runs twice and compare")
    run.add_argument("--out", required=True, type=Path)
    run.add_argument("--script", default=None)
    run.add_argument("--allow-dirty", action="store_true")
    run.set_defaults(func=cmd_run)
    record = sub.add_parser("record", help="one recording (run starts it)")
    record.add_argument("--out", required=True, type=Path)
    record.add_argument("--name", required=True, choices=RECORDINGS)
    record.add_argument("--script", default=None)
    record.set_defaults(func=cmd_record)
    check = sub.add_parser("check", help="re-check a finished record")
    check.add_argument("directory", type=Path)
    check.set_defaults(func=cmd_check)
    return parser.parse_args(argv)


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    try:
        return args.func(args)
    except Refusal as refusal:
        print(f"seeded_gate_runs.py refused: {refusal}", flush=True)
        return refusal.code
    except contract.ContractError as error:
        print(
            f"seeded_gate_runs.py refused: no identity ({error})", flush=True
        )
        return EXIT_IDENTITY


if __name__ == "__main__":
    sys.exit(main())
