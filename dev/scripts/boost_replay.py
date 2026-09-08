"""Run the seven approved final-boost trajectory checks.

Each selected trajectory is run once through :func:`golden_trace.run_task`
with the guarded boost objective and ``tol_elcbo_boost=0.1``.  The driver
captures the authentic pre-boost restart inputs and the unselected raw boost
candidate, then classifies that same candidate at tolerances 0.1 and 0.2.
It also compares the emitted 0.1 trace with the immutable 870-run reference.

Examples::

    python -u dev/scripts/boost_replay.py --out dev/scripts/runs/boost_phase2
    python -u dev/scripts/boost_replay.py --out dev/scripts/runs/boost_phase2 \
        --case logreg_D5:5
    python -u dev/scripts/boost_replay.py --out dev/scripts/runs/boost_stress \
        --case rosenbrock_D2_noise3:7 --case rosenbrock_D2_noise3:17

The output directory is required. Existing per-case result files are never
overwritten, so interrupted work can be continued by selecting only cases
whose files do not yet exist. Two noisy Rosenbrock counterexamples are also
available by explicit selection as targeted stress checks, not population
calibration; they are not part of the default seven.
"""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
REFERENCE = HERE / "runs" / "golden" / "reference_870_20260907"
SIDECARS = REPO_ROOT / "dev" / "golden" / "baseline"
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
PRIMARY_TOLERANCE = 0.1
SECONDARY_TOLERANCE = 0.2
EXPECTED_GPYREG_SHA = "a2f8ddc"
DEFAULT_CASES = (
    ("normal_D5", 0),
    ("banana_D2", 0),
    ("halfnormal_D2", 0),
    ("cigar_D4", 0),
    ("rosenbrock_D2_noise1", 0),
    ("logreg_D5", 5),
    ("student_D4", 19),
)
TARGETED_CASES = (
    ("rosenbrock_D2_noise3", 7),
    ("rosenbrock_D2_noise3", 17),
)


def _tag(label, seed):
    return f"{label}_seed{seed}"


def _parse_case(value):
    try:
        label, seed_text = value.rsplit(":", 1)
        case = (label, int(seed_text))
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            "case must have the form LABEL:SEED"
        ) from exc
    allowed_cases = DEFAULT_CASES + TARGETED_CASES
    if case not in allowed_cases:
        allowed = ", ".join(f"{label}:{seed}" for label, seed in allowed_cases)
        raise argparse.ArgumentTypeError(
            "case is outside the approved defaults and targeted stress "
            f"checks; choose one of: {allowed}"
        )
    return case


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="explicit output directory (created when absent)",
    )
    parser.add_argument(
        "--case",
        action="append",
        type=_parse_case,
        dest="cases",
        help=(
            "approved LABEL:SEED to run; repeatable (default: the original "
            "seven; noisy counterexample stress checks require selection)"
        ),
    )
    return parser.parse_args(argv)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_output(path, *args):
    return subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={Path(path).resolve()}",
            "-C",
            str(path),
            *args,
        ],
        text=True,
    ).strip()


def _git_state(path):
    return {
        "head": _git_output(path, "rev-parse", "HEAD"),
        "dirty": bool(
            _git_output(path, "status", "--porcelain", "--untracked-files=no")
        ),
    }


def _module_record(module):
    path = Path(module.__file__).resolve()
    return {"path": str(path), "sha256": _sha256(path)}


def _vp_arrays(vp, prefix):
    import numpy as np

    return {
        f"{prefix}_w": np.array(vp.w, copy=True),
        f"{prefix}_mu": np.array(vp.mu, copy=True),
        f"{prefix}_sigma": np.array(vp.sigma, copy=True),
        f"{prefix}_lambd": np.array(vp.lambd, copy=True),
    }


def _vp_matches_arrays(vp, arrays, prefix):
    import numpy as np

    return all(
        np.array_equal(
            np.asarray(getattr(vp, name)), arrays[f"{prefix}_{name}"]
        )
        for name in ("w", "mu", "sigma", "lambd")
    )


def _valid_score(elbo, elbo_sd):
    import numbers

    import numpy as np

    values = (elbo, elbo_sd)
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, numbers.Real)
        for value in values
    ):
        return False
    return bool(np.isfinite(elbo) and np.isfinite(elbo_sd) and elbo_sd >= 0)


def classify_candidate(
    pre_elbo, pre_elbo_sd, candidate_elbo, candidate_elbo_sd, tolerance
):
    """Classify one already-computed candidate without draws or evaluations."""
    import numpy as np

    pre_valid = _valid_score(pre_elbo, pre_elbo_sd)
    candidate_valid = _valid_score(candidate_elbo, candidate_elbo_sd)
    result = {
        "tolerance": float(tolerance),
        "pre_valid": pre_valid,
        "candidate_valid": candidate_valid,
        "delta_elbo": float("nan"),
        "delta_sd": float("nan"),
        "endpoint_b0": float("nan"),
        "endpoint_b5": float("nan"),
        "worst_delta": float("nan"),
        "endpoint_b0_passes": None,
        "endpoint_b5_passes": None,
    }
    if candidate_valid and not pre_valid:
        result.update(
            accepted=True, selected="candidate", reason="pre-invalid"
        )
        return result
    if not candidate_valid and not pre_valid:
        result.update(accepted=None, selected=None, reason="both-invalid")
        return result
    if not candidate_valid:
        result.update(
            accepted=False, selected="pre", reason="candidate-invalid"
        )
        return result

    delta_elbo = float(candidate_elbo - pre_elbo)
    delta_sd = float(candidate_elbo_sd - pre_elbo_sd)
    endpoint_b5 = delta_elbo - 5.0 * delta_sd
    worst_delta = min(delta_elbo, endpoint_b5)
    # Use the production comparison order so exact-boundary behavior is
    # identical despite possible floating-point reassociation in deltas.
    endpoint_b0_passes = bool(candidate_elbo > pre_elbo - tolerance)
    endpoint_b5_passes = bool(
        candidate_elbo - 5.0 * candidate_elbo_sd
        > pre_elbo - 5.0 * pre_elbo_sd - tolerance
    )
    accepted = endpoint_b0_passes and endpoint_b5_passes
    result.update(
        delta_elbo=delta_elbo,
        delta_sd=delta_sd,
        endpoint_b0=delta_elbo,
        endpoint_b5=endpoint_b5,
        worst_delta=worst_delta,
        endpoint_b0_passes=endpoint_b0_passes,
        endpoint_b5_passes=endpoint_b5_passes,
        accepted=accepted,
        selected="candidate" if accepted else "pre",
        reason="within-tolerance" if accepted else "beyond-tolerance",
    )
    assert np.isfinite(worst_delta)
    return result


def _case_paths(out_dir, label, seed):
    stem = out_dir / _tag(label, seed)
    capture_stem = out_dir / "captures" / _tag(label, seed)
    return {
        "trace": stem.with_suffix(".npz"),
        "sidecar": stem.with_suffix(".json"),
        "error": Path(f"{stem}.error.txt"),
        "restart": Path(f"{capture_stem}.boost_pre.dill"),
        "candidate": Path(f"{capture_stem}.boost_candidate.dill"),
        "arrays": Path(f"{capture_stem}.boost.npz"),
        "report": Path(f"{capture_stem}.boost.json"),
    }


def _refuse_existing(paths):
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"refusing to overwrite existing result files: {names}"
        )


def _dump_dill(path, value):
    import dill

    with path.open("xb") as stream:
        dill.dump(value, stream, recurse=True)


def _metrics_json(metrics, jsonable):
    return {
        key: jsonable(value)
        for key, value in metrics.items()
        if key not in ("post_mean", "post_cov")
    }


def _run_case(label, seed, out_dir, provenance, pop):
    import numpy as np
    from benchmark_targets import find_config, metrics
    from golden_replay import compare_run
    from golden_trace import run_task
    from profile_run import jsonable

    import pyvbmc.vbmc.vbmc as vbmc_module

    paths = _case_paths(out_dir, label, seed)
    _refuse_existing(paths)
    problem = find_config(label).make(seed=seed)
    original_final_boost = vbmc_module.VBMC.final_boost
    capture = {"calls": 0, "candidate": None}

    def instrumented_final_boost(self, vp, gp):
        # Serialize the exact restart boundary before final_boost mutates
        # optim_state, flags, options copies, or the shared generator.
        _dump_dill(
            paths["restart"],
            {
                "vp": vp,
                "gp": gp,
                "optim_state": self.optim_state,
                "options": self.options,
                "rng_state": copy.deepcopy(self.rng.bit_generator.state),
                "label": label,
                "seed": seed,
                "provenance": provenance,
            },
        )
        pre_arrays = _vp_arrays(vp, "pre")
        pre_stats = copy.deepcopy(vp.stats)
        pre_metrics = metrics(problem, vp, vp.stats["elbo"])
        original_optimize_vp = vbmc_module.optimize_vp

        def instrumented_optimize_vp(*args, **kwargs):
            capture["calls"] += 1
            if capture["calls"] != 1:
                raise RuntimeError(
                    "final boost called optimize_vp more than once"
                )
            candidate, var_ss, pruned = original_optimize_vp(*args, **kwargs)
            capture["candidate"] = candidate
            capture["candidate_stats"] = copy.deepcopy(candidate.stats)
            capture["candidate_arrays"] = _vp_arrays(candidate, "candidate")
            _dump_dill(
                paths["candidate"],
                {
                    "vp": candidate,
                    "stats": capture["candidate_stats"],
                    "rng_state": copy.deepcopy(
                        candidate.rng.bit_generator.state
                    ),
                    "capture_stage": (
                        "raw optimize_vp output before final_boost restores "
                        "the pre-boost stable flag"
                    ),
                    "label": label,
                    "seed": seed,
                    "provenance": provenance,
                },
            )
            return candidate, var_ss, pruned

        vbmc_module.optimize_vp = instrumented_optimize_vp
        try:
            returned_vp, elbo, elbo_sd, changed = original_final_boost(
                self, vp, gp
            )
        finally:
            vbmc_module.optimize_vp = original_optimize_vp

        if capture["calls"] != 1 or capture["candidate"] is None:
            raise RuntimeError(
                "approved trajectory did not produce one boost candidate"
            )
        candidate = capture["candidate"]
        candidate_stats = capture["candidate_stats"]
        candidate_metrics = metrics(
            problem, candidate, candidate_stats["elbo"]
        )
        decisions = {
            str(tol): classify_candidate(
                pre_stats["elbo"],
                pre_stats["elbo_sd"],
                candidate_stats["elbo"],
                candidate_stats["elbo_sd"],
                tol,
            )
            for tol in (PRIMARY_TOLERANCE, SECONDARY_TOLERANCE)
        }
        primary = decisions[str(PRIMARY_TOLERANCE)]
        if primary["accepted"] is None:
            raise RuntimeError(
                "both pre-boost and candidate scores are invalid"
            )
        if bool(changed) != bool(primary["accepted"]):
            raise RuntimeError(
                "production 0.1 decision disagrees with offline rule"
            )
        selected_arrays = (
            capture["candidate_arrays"] if primary["accepted"] else pre_arrays
        )
        selected_prefix = "candidate" if primary["accepted"] else "pre"
        if not _vp_matches_arrays(
            returned_vp, selected_arrays, selected_prefix
        ) or (elbo, elbo_sd) != (
            returned_vp.stats["elbo"],
            returned_vp.stats["elbo_sd"],
        ):
            raise RuntimeError(
                "production 0.1 return does not match the selected posterior"
            )

        selected_metrics = {
            str(tol): (
                candidate_metrics if decision["accepted"] else pre_metrics
            )
            for tol, decision in (
                (PRIMARY_TOLERANCE, decisions[str(PRIMARY_TOLERANCE)]),
                (SECONDARY_TOLERANCE, decisions[str(SECONDARY_TOLERANCE)]),
            )
        }
        arrays = dict(pre_arrays)
        arrays.update(capture["candidate_arrays"])
        arrays.update(
            pre_post_mean=np.asarray(pre_metrics["post_mean"]),
            pre_post_cov=np.asarray(pre_metrics["post_cov"]),
            candidate_post_mean=np.asarray(candidate_metrics["post_mean"]),
            candidate_post_cov=np.asarray(candidate_metrics["post_cov"]),
        )
        np.savez_compressed(paths["arrays"], **arrays)
        capture["boost_report"] = {
            "pre_stats": jsonable(pre_stats),
            "candidate_stats": jsonable(candidate_stats),
            "candidate_capture_stage": (
                "raw optimize_vp output before final_boost restores the "
                "pre-boost stable flag"
            ),
            "returned_stats": {"elbo": float(elbo), "elbo_sd": float(elbo_sd)},
            "optimize_vp_calls": capture["calls"],
            "pre_metrics": _metrics_json(pre_metrics, jsonable),
            "candidate_metrics": _metrics_json(candidate_metrics, jsonable),
            "decisions": decisions,
            "selected_metrics": {
                str(tol): _metrics_json(value, jsonable)
                for tol, value in selected_metrics.items()
            },
            "postboost_rng_note": (
                "The eventual return-time RNG state may differ between selected "
                "posteriors because optimize() performs downstream KL draws."
            ),
        }
        return returned_vp, elbo, elbo_sd, changed

    vbmc_module.VBMC.final_boost = instrumented_final_boost
    try:
        run_row = run_task(
            label,
            seed,
            {
                "vectorized_target": False,
                "tol_elcbo_boost": PRIMARY_TOLERANCE,
            },
            out_dir,
        )
    finally:
        vbmc_module.VBMC.final_boost = original_final_boost

    if not run_row["ok"]:
        raise RuntimeError(f"trajectory failed; see {paths['error']}")
    comparison = compare_run(label, seed, out_dir, REFERENCE, SIDECARS, pop)
    if comparison.get("loop_identical") is not True:
        raise RuntimeError(
            "pre-boost loop differs from the canonical reference"
        )
    if comparison.get("initial_design_ok") is not True:
        raise RuntimeError(
            "initial design is not exactly certified against the canonical "
            "reference"
        )

    report = {
        "label": label,
        "seed": seed,
        "primary_tolerance": PRIMARY_TOLERANCE,
        "secondary_tolerance": SECONDARY_TOLERANCE,
        "options": {
            "vectorized_target": False,
            "tol_elcbo_boost": PRIMARY_TOLERANCE,
        },
        "provenance": provenance,
        "boost": capture["boost_report"],
        "canonical_comparison": comparison,
        "artifacts": {
            key: str(path.relative_to(out_dir)) for key, path in paths.items()
        },
    }
    paths["report"].write_text(
        json.dumps(report, indent=1, default=str), encoding="utf-8"
    )
    return report


def _runtime_provenance():
    import benchmark_targets
    import golden_replay
    import golden_trace
    import gpyreg
    import numpy as np
    from profile_run import pkg_version

    import pyvbmc
    import pyvbmc.vbmc.vbmc as vbmc_module

    sibling = REPO_ROOT.parent / "gpyreg"
    repo_git = _git_state(REPO_ROOT)
    gpyreg_git = _git_state(sibling)
    if not gpyreg_git["head"].startswith(EXPECTED_GPYREG_SHA):
        raise RuntimeError(
            f"gpyreg HEAD is {gpyreg_git['head']}, expected "
            f"{EXPECTED_GPYREG_SHA}"
        )
    if gpyreg_git["dirty"]:
        raise RuntimeError("the pinned sibling gpyreg checkout is dirty")
    expected_roots = {
        "pyvbmc": REPO_ROOT,
        "benchmark_targets": HERE,
        "golden_trace": HERE,
        "golden_replay": HERE,
        "gpyreg": sibling,
    }
    modules = {
        "pyvbmc": pyvbmc,
        "vbmc": vbmc_module,
        "benchmark_targets": benchmark_targets,
        "golden_trace": golden_trace,
        "golden_replay": golden_replay,
        "gpyreg": gpyreg,
    }
    records = {
        name: _module_record(module) for name, module in modules.items()
    }
    driver_path = Path(__file__).resolve()
    records["boost_replay"] = {
        "path": str(driver_path),
        "sha256": _sha256(driver_path),
    }
    for name, expected_root in expected_roots.items():
        actual = Path(records[name]["path"])
        if expected_root.resolve() not in actual.parents:
            raise RuntimeError(
                f"{name} imported from {actual}, expected under {expected_root}"
            )
    return {
        "repo_git": repo_git,
        "gpyreg_git": gpyreg_git,
        "modules": records,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": pkg_version("scipy"),
        "pyvbmc": pkg_version("pyvbmc"),
        "gpyreg": pkg_version("gpyreg"),
        "cma": pkg_version("cma"),
        "threads": {key: os.environ.get(key) for key in THREAD_KEYS},
    }


def main(argv=None):
    args = parse_args(argv)
    cases = args.cases or list(DEFAULT_CASES)
    if len(cases) != len(set(cases)):
        raise SystemExit("duplicate --case selections are not allowed")
    if not REFERENCE.is_dir():
        raise SystemExit(
            f"canonical reference directory is absent: {REFERENCE}"
        )
    if not SIDECARS.is_dir():
        raise SystemExit(f"canonical sidecars directory is absent: {SIDECARS}")

    for key in THREAD_KEYS:
        os.environ[key] = "1"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "captures").mkdir(exist_ok=True)
    all_paths = [
        path
        for label, seed in cases
        for path in _case_paths(args.out, label, seed).values()
    ]
    existing = [path for path in all_paths if path.exists()]
    if existing:
        names = ", ".join(path.name for path in existing)
        raise SystemExit(
            f"refusing to overwrite existing result files: {names}"
        )

    # Imports happen only after the BLAS environment is fixed.
    from golden_trace import load_population

    provenance = _runtime_provenance()
    pop = load_population(SIDECARS)
    print(
        f"[boost-replay] {len(cases)} case(s), sequential, workspace "
        f"{provenance['repo_git']['head'][:8]}"
        f"{' (dirty)' if provenance['repo_git']['dirty'] else ''}, gpyreg "
        f"{provenance['gpyreg_git']['head'][:8]} "
        f"-> {args.out}",
        flush=True,
    )
    started = time.time()
    for index, (label, seed) in enumerate(cases, 1):
        tag = _tag(label, seed)
        print(f"[boost-replay] {index}/{len(cases)} {tag} ...", flush=True)
        report = _run_case(label, seed, args.out, provenance, pop)
        decision_01 = report["boost"]["decisions"][str(PRIMARY_TOLERANCE)]
        decision_02 = report["boost"]["decisions"][str(SECONDARY_TOLERANCE)]
        comparison = report["canonical_comparison"]
        print(
            f"[boost-replay] {tag}: loop exact; design "
            f"{comparison.get('design', 'not reported')}; "
            f"0.1={decision_01['selected']}, 0.2={decision_02['selected']}",
            flush=True,
        )
    print(
        f"[boost-replay] complete in {(time.time() - started) / 60:.1f} min",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
