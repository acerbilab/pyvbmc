"""Time three paired final boosts with and without the weight penalty."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from benchmark_targets import find_config, metrics
from boost_reconstruction import (
    _fresh_common_state,
    _sha256,
    reconstruct_state,
    runtime_provenance,
)

REFERENCE = HERE / "runs" / "golden" / "reference_870_20260907"
CASES = (
    ("student_D8_noise3", 0),
    ("lumpy_D10", 0),
    ("cigar_D15_exhaust", 0),
)
PENALTIES = (0.1, 0.0)
COMMON_SEED = 20260908
VP_ARRAYS = ("w", "eta", "mu", "sigma", "lambd")
METRIC_KEYS = ("elbo_err", "gskl", "mmtv", "rmse")


def _tag(label, seed):
    return f"{label}_seed{seed}"


def _seconds(started):
    return time.perf_counter() - started


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return repr(value)
    return value


def _provenance():
    provenance = runtime_provenance()
    source = Path(__file__).resolve()
    provenance["boost_penalty_pilot"] = {
        "path": str(source),
        "sha256": _sha256(source),
    }
    return provenance


def _numeric_hash(*values):
    """Hash all numeric leaves, including array dtype and shape."""
    digest = hashlib.sha256()

    def update(value):
        if isinstance(value, dict):
            for key in sorted(value, key=str):
                digest.update(str(key).encode())
                update(value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(str(len(value)).encode())
            for item in value:
                update(item)
        elif isinstance(value, np.ndarray):
            digest.update(value.dtype.str.encode())
            digest.update(str(value.shape).encode())
            if value.dtype.hasobject:
                update(value.tolist())
            else:
                digest.update(np.ascontiguousarray(value).tobytes())
        elif isinstance(value, (bool, int, float, complex)):
            digest.update(type(value).__name__.encode())
            digest.update(repr(value).encode())
        elif isinstance(value, np.generic):
            array = np.asarray(value)
            digest.update(array.dtype.str.encode())
            digest.update(array.tobytes())

    for value in values:
        update(value)
    return digest.hexdigest()


def _state_payload(vp, gp, optim_state, options, rng_state):
    return {
        "vp": {key: getattr(vp, key) for key in VP_ARRAYS},
        "vp_stats": vp.stats,
        "gp": {
            "X": gp.X,
            "y": gp.y,
            "s2": gp.s2,
            "hyp": gp.get_hyperparameters(as_array=True),
            "posteriors": [
                {
                    key: getattr(posterior, key)
                    for key in ("hyp", "alpha", "L", "L_chol", "sW")
                }
                for posterior in gp.posteriors
            ],
        },
        "optim_state": optim_state,
        "options": dict(options),
        "rng": rng_state,
    }


def _fork(rebuilt, rng_state):
    runner = copy.copy(rebuilt["runner"])
    vp = copy.deepcopy(rebuilt["vp"])
    gp = copy.deepcopy(rebuilt["gp"])
    runner.options = copy.deepcopy(rebuilt["options"])
    runner.optim_state = copy.deepcopy(rebuilt["optim_state"])
    rng = np.random.default_rng()
    rng.bit_generator.state = copy.deepcopy(rng_state)
    vp.rng = rng
    runner.rng, runner.vp, runner.gp = rng, vp, gp
    payload = _state_payload(
        vp, gp, runner.optim_state, runner.options, rng.bit_generator.state
    )
    return runner, vp, gp, _numeric_hash(payload)


def _run_arm(runner, vp, gp, penalty):
    import pyvbmc.vbmc.vbmc as vbmc_module

    runner.options.__setitem__("tol_elcbo_boost", None, force=True)
    runner.options.__setitem__("weight_penalty", penalty, force=True)
    capture = {"calls": 0}
    original = vbmc_module.optimize_vp

    def instrumented(options, *args, **kwargs):
        capture["calls"] += 1
        if options.get("weight_penalty") != penalty:
            raise RuntimeError("final_boost changed weight_penalty")
        if options.get("tol_weight") != 0:
            raise RuntimeError("final_boost did not disable pruning")
        capture["actual_options"] = copy.deepcopy(options)
        started = time.perf_counter()
        result = original(options, *args, **kwargs)
        capture["optimizer_s"] = _seconds(started)
        capture["candidate"] = copy.deepcopy(result[0])
        capture["optimizer_outputs"] = copy.deepcopy(result[1:])
        return result

    func_count = runner.function_logger.func_count
    started = time.perf_counter()
    vbmc_module.optimize_vp = instrumented
    try:
        selected, elbo, elbo_sd, changed = runner.final_boost(vp, gp)
    finally:
        vbmc_module.optimize_vp = original
    final_boost_s = _seconds(started)
    if capture["calls"] != 1:
        raise RuntimeError(
            f"expected one optimize_vp call, got {capture['calls']}"
        )
    if runner.function_logger.func_count != func_count:
        raise RuntimeError("final boost evaluated the target")
    if not changed:
        raise RuntimeError(
            "unguarded final boost did not select its candidate"
        )
    capture.update(
        selected=selected,
        returned_elbo=elbo,
        returned_elbo_sd=elbo_sd,
        changed=changed,
        rng_exit_state=copy.deepcopy(vp.rng.bit_generator.state),
        final_boost_s=final_boost_s,
    )
    return capture


def _diagnostics(problem, vp):
    values = metrics(problem, vp, float(vp.stats["elbo"]))
    return {key: values[key] for key in METRIC_KEYS}


def _run_case(label, seed, out_dir, provenance):
    import dill

    whole_started = time.perf_counter()
    tag = _tag(label, seed)
    side_path = REFERENCE / f"{tag}.json"
    sidecar = json.loads(side_path.read_text(encoding="utf-8"))
    reconstruction_started = time.perf_counter()
    with np.load(side_path.with_suffix(".npz"), allow_pickle=False) as trace:
        rebuilt = reconstruct_state(sidecar, trace, rng_seed=COMMON_SEED)
    reconstruction_s = _seconds(reconstruction_started)

    common_rng = _fresh_common_state(COMMON_SEED, label, seed)
    setup_started = time.perf_counter()
    forks = [_fork(rebuilt, common_rng) for _ in PENALTIES]
    copy_setup_s = _seconds(setup_started)
    if forks[0][3] != forks[1][3]:
        raise RuntimeError("paired arms do not have identical numeric inputs")

    arms = {}
    for penalty, (runner, vp, gp, entry_hash) in zip(PENALTIES, forks):
        name = f"weight_penalty_{penalty:g}"
        print(f"[boost-penalty-pilot] {tag} {name} ...", flush=True)
        arm_started = time.perf_counter()
        arms[name] = _run_arm(runner, vp, gp, penalty)
        print(
            f"[boost-penalty-pilot] {tag} {name}: "
            f"{_seconds(arm_started):.2f}s",
            flush=True,
        )
        arms[name]["entry_hash"] = entry_hash
        arms[name]["rng_entry_state"] = copy.deepcopy(common_rng)

    problem = find_config(label).make(seed=seed)
    diagnostics_started = time.perf_counter()
    diagnostic_values = {"pre": _diagnostics(problem, rebuilt["vp"])}
    for name, arm in arms.items():
        diagnostic_values[name] = _diagnostics(problem, arm["candidate"])
    diagnostics_s = _seconds(diagnostics_started)

    capture = {
        "provenance": provenance,
        "case": {"tag": tag, "label": label, "seed": seed},
        "pre_state": rebuilt,
        "pre_score_provenance": "historical trace ELBO/SD; not rescored",
        "common_rng_state": common_rng,
        "arms": arms,
        "diagnostics": diagnostic_values,
        "acceptance_criteria": "deferred; timing pilot only",
    }
    serialization_started = time.perf_counter()
    with (out_dir / f"{tag}.dill").open("wb") as stream:
        dill.dump(capture, stream)
    serialization_s = _seconds(serialization_started)

    optimizer_s = sum(arm["optimizer_s"] for arm in arms.values())
    final_boost_s = sum(arm["final_boost_s"] for arm in arms.values())
    timings = {
        "reconstruction_s": reconstruction_s,
        "copy_setup_s": copy_setup_s,
        "optimizer_s": optimizer_s,
        "final_boost_s": final_boost_s,
        "final_boost_overhead_s": final_boost_s - optimizer_s,
        "diagnostics_s": diagnostics_s,
        "serialization_s": serialization_s,
        "whole_wall_s": _seconds(whole_started),
    }
    report = {
        "case": capture["case"],
        "paired_entry_hash": forks[0][3],
        "paired_inputs_identical": True,
        "pre_score_provenance": capture["pre_score_provenance"],
        "acceptance_criteria": capture["acceptance_criteria"],
        "pre_scores": _jsonable(rebuilt["vp"].stats),
        "arms": {
            name: {
                "weight_penalty": arm["actual_options"]["weight_penalty"],
                "tol_weight": arm["actual_options"]["tol_weight"],
                "scores": {
                    key: _jsonable(arm["candidate"].stats[key])
                    for key in ("elbo", "elbo_sd", "stable")
                },
                "returned_elbo": arm["returned_elbo"],
                "returned_elbo_sd": arm["returned_elbo_sd"],
                "optimizer_s": arm["optimizer_s"],
                "final_boost_s": arm["final_boost_s"],
            }
            for name, arm in arms.items()
        },
        "diagnostics": _jsonable(diagnostic_values),
        "timings": timings,
        "artifacts": {"capture": f"{tag}.dill"},
    }
    (out_dir / f"{tag}.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"refusing nonempty output directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    provenance = _provenance()
    started = time.perf_counter()
    reports = []
    for index, (label, seed) in enumerate(CASES, 1):
        print(
            f"[boost-penalty-pilot] {index}/{len(CASES)} {_tag(label, seed)}",
            flush=True,
        )
        reports.append(_run_case(label, seed, args.out, provenance))
    summary = {
        "provenance": provenance,
        "cases": [report["case"]["tag"] for report in reports],
        "case_timings": {
            report["case"]["tag"]: report["timings"] for report in reports
        },
        "whole_wall_s": _seconds(started),
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"[boost-penalty-pilot] complete in {summary['whole_wall_s']:.2f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
