"""Run the resumable 870-case paired final-boost campaign."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
# The editable environment points at the main checkout.  Put this isolated
# checkout first before importing any PyVBMC module or helper that imports it.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(1, str(HERE))

import numpy as np
from benchmark_targets import find_config, metrics
from boost_penalty_pilot import (
    PENALTIES,
    _fork,
    _fresh_common_state,
    _jsonable,
    _run_arm,
    _seconds,
    _sha256,
    _tag,
)
from boost_reconstruction import (
    _capture_options,
    _capture_paths,
    _load_capture,
    audit_population,
    reconstruct_state,
    runtime_provenance,
)

import pyvbmc
from pyvbmc import VBMC
from pyvbmc.vbmc.variational_optimization import _neg_elcbo

EXPECTED_CASES = 870
APPROVED_PYVBMC_BASE = "764a17774f89f9e8345e4d2ee910caacf8b88852"
EXPECTED_GPYREG_HEAD = "a2f8ddce867f502e29717959cf0ff3529f598618"
OPTIMIZATION_SEED = 20260908
DIAGNOSTIC_SEED = 20260909
DIAGNOSTIC_TOTAL_ENTROPY_DRAWS = 100_000
PROTOCOL_VERSION = 1


def _atomic_text(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path, value):
    _atomic_text(
        path, json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n"
    )


def _atomic_dill(path, value):
    import dill

    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            dill.dump(value, stream, recurse=True)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_dill(path):
    import dill

    with Path(path).open("rb") as stream:
        return dill.load(stream)


def _git_head(path):
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_tracked_clean(path):
    result = subprocess.run(
        ["git", "-C", str(path), "diff", "--quiet", "HEAD", "--"],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"tracked source changes present in {path}")


def _require_approved_numerical_tree():
    result = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "diff",
            "--quiet",
            APPROVED_PYVBMC_BASE,
            "--",
            "pyvbmc",
        ],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"pyvbmc differs from approved base {APPROVED_PYVBMC_BASE}"
        )


def _source_records():
    paths = (
        Path(__file__),
        HERE / "boost_penalty_pilot.py",
        HERE / "boost_reconstruction.py",
        HERE / "benchmark_targets.py",
        REPO_ROOT / "pyvbmc" / "vbmc" / "vbmc.py",
        REPO_ROOT / "pyvbmc" / "vbmc" / "variational_optimization.py",
        REPO_ROOT / "pyvbmc" / "entropy" / "entmc_vbmc.py",
    )
    return [
        {"path": str(path.resolve()), "sha256": _sha256(path)}
        for path in paths
    ]


def _verify_imports():
    import gpyreg

    pyvbmc_path = Path(pyvbmc.__file__).resolve()
    if not pyvbmc_path.is_relative_to(REPO_ROOT):
        raise RuntimeError(
            f"pyvbmc imported from {pyvbmc_path}, expected {REPO_ROOT}"
        )
    gpyreg_path = Path(gpyreg.__file__).resolve()
    gpyreg_root = gpyreg_path.parent.parent
    _require_tracked_clean(REPO_ROOT)
    _require_approved_numerical_tree()
    _require_tracked_clean(gpyreg_root)
    head = _git_head(gpyreg_root)
    if head != EXPECTED_GPYREG_HEAD:
        raise RuntimeError(
            f"gpyreg HEAD is {head}, expected {EXPECTED_GPYREG_HEAD}"
        )
    return {
        "pyvbmc_path": str(pyvbmc_path),
        "gpyreg_path": str(gpyreg_path),
        "gpyreg_root": str(gpyreg_root),
        "gpyreg_head": head,
    }


def _build_manifest(reference, capture_root):
    reference = Path(reference).resolve()
    capture_root = Path(capture_root).resolve()
    json_paths = sorted(reference.glob("*.json"))
    npz_by_tag = {path.stem: path for path in reference.glob("*.npz")}
    json_tags = {path.stem for path in json_paths}
    if len(json_paths) != EXPECTED_CASES or set(npz_by_tag) != json_tags:
        raise RuntimeError(
            "reference must contain exactly 870 matching JSON/NPZ pairs"
        )

    captures = {}
    for path in _capture_paths(capture_root):
        tag = path.name.removesuffix(".boost_pre.dill")
        if tag not in json_tags:
            raise RuntimeError(f"capture has no reference pair: {path}")
        if tag in captures:
            raise RuntimeError(f"duplicate authentic capture for {tag}")
        capture = _load_capture(path)
        expected = _tag(capture["label"], int(capture["seed"]))
        if expected != tag:
            raise RuntimeError(f"capture identity mismatch: {path}")
        captures[tag] = path
    if len(captures) != 9:
        raise RuntimeError(
            f"expected 9 authentic captures, found {len(captures)}"
        )

    cases = []
    for side_path in json_paths:
        tag = side_path.stem
        sidecar = json.loads(side_path.read_text(encoding="utf-8"))
        if _tag(sidecar["label"], int(sidecar["seed"])) != tag:
            raise RuntimeError(f"reference identity mismatch: {side_path}")
        npz_path = npz_by_tag[tag]
        capture_path = captures.get(tag)
        cases.append(
            {
                "tag": tag,
                "label": sidecar["label"],
                "seed": int(sidecar["seed"]),
                "json": str(side_path),
                "json_sha256": _sha256(side_path),
                "npz": str(npz_path),
                "npz_sha256": _sha256(npz_path),
                "state_source": (
                    "authentic" if capture_path else "reconstructed"
                ),
                "capture": str(capture_path) if capture_path else None,
                "capture_sha256": (
                    _sha256(capture_path) if capture_path else None
                ),
            }
        )
    return cases


def _campaign_config(reference, capture_root, cases):
    value = {
        "protocol_version": PROTOCOL_VERSION,
        "reference": str(Path(reference).resolve()),
        "capture_root": str(Path(capture_root).resolve()),
        "optimization_seed": OPTIMIZATION_SEED,
        "diagnostic_seed": DIAGNOSTIC_SEED,
        "diagnostic_total_entropy_draws": DIAGNOSTIC_TOTAL_ENTROPY_DRAWS,
        "penalties": list(PENALTIES),
        "sources": _source_records(),
        "cases": cases,
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    value["config_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    return value


def _initialize_config(out_dir, config):
    path = out_dir / "campaign.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != config:
            raise RuntimeError(
                "campaign source or input config changed; use a new output directory"
            )
    else:
        _atomic_json(path, config)


def _runner_shell(sidecar, capture):
    problem = find_config(sidecar["label"]).make(seed=int(sidecar["seed"]))
    args, _ = problem.vbmc_args()
    runner = VBMC(
        *args,
        options=_capture_options(sidecar),
        seed=int(sidecar["seed"]),
    )
    runner.options = capture["options"]
    runner.optim_state = capture["optim_state"]
    runner.vp = capture["vp"]
    runner.gp = capture["gp"]
    runner.parameter_transformer = capture["vp"].parameter_transformer
    runner.function_logger.parameter_transformer = runner.parameter_transformer
    return runner


def _load_pre_state(case):
    sidecar = json.loads(Path(case["json"]).read_text(encoding="utf-8"))
    if case["state_source"] == "authentic":
        capture = _load_capture(case["capture"])
        runner = _runner_shell(sidecar, capture)
        rebuilt = {
            "runner": runner,
            "vp": capture["vp"],
            "gp": capture["gp"],
            "options": capture["options"],
            "optim_state": capture["optim_state"],
        }
        pre_rng_state = copy.deepcopy(capture["rng_state"])
        capture_provenance = copy.deepcopy(capture.get("provenance"))
    else:
        with np.load(case["npz"], allow_pickle=False) as trace:
            rebuilt = reconstruct_state(
                sidecar, trace, rng_seed=OPTIMIZATION_SEED
            )
        pre_rng_state = copy.deepcopy(rebuilt["vp"].rng.bit_generator.state)
        capture_provenance = None
    return sidecar, rebuilt, pre_rng_state, capture_provenance


def _rng_from_state(state):
    rng = np.random.default_rng()
    rng.bit_generator.state = copy.deepcopy(state)
    return rng


def _score(problem, vp, gp, rng_state):
    diagnostic_vp = copy.deepcopy(vp)
    diagnostic_vp.rng = _rng_from_state(rng_state)
    K = int(diagnostic_vp.K)
    Ns = int(np.ceil(DIAGNOSTIC_TOTAL_ENTROPY_DRAWS / K))
    Ns = int(np.ceil(Ns / 2)) * 2
    theta = diagnostic_vp.get_parameters()
    result = _neg_elcbo(
        theta,
        gp,
        diagnostic_vp,
        beta=0,
        Ns=Ns,
        compute_grad=False,
        compute_var=True,
        theta_bnd=None,
        separate_K=True,
    )
    F, _, G, H, varF, _, varG_ss, varG, varH, I_sk, J_sjk = result
    elbo = float(-F)
    varG = float(varG)
    if not np.isfinite(elbo) or not np.isfinite(varG) or varG < 0:
        raise RuntimeError(
            f"invalid diagnostic score: elbo={elbo!r}, varG={varG!r}"
        )
    gp_sd = float(np.sqrt(varG))
    metric_vp = copy.deepcopy(vp)
    metric_vp.rng = _rng_from_state(rng_state)
    metric_values = metrics(problem, metric_vp, elbo)
    return {
        "K": K,
        "elbo": elbo,
        "gp_sd": gp_sd,
        "elcbo_beta5": elbo - 5 * gp_sd,
        "G": G,
        "H": H,
        "varF": varF,
        "varG": varG,
        "varH": varH,
        "varG_ss": varG_ss,
        "I_sk": I_sk,
        "J_sjk": J_sjk,
        "entropy_draws_per_component": Ns,
        "entropy_draws_total": Ns * K,
        "variance_note": "GP variance only; entropy variance is zero",
        "draw_alignment_note": (
            "each posterior starts at the same diagnostic RNG state; Ns is "
            "rounded up to an even count per K for antithetic sampling"
        ),
        "metrics": metric_values,
    }


def _summary_score(score):
    return {
        key: _jsonable(score[key])
        for key in (
            "K",
            "elbo",
            "gp_sd",
            "elcbo_beta5",
            "entropy_draws_per_component",
            "entropy_draws_total",
            "variance_note",
            "draw_alignment_note",
            "metrics",
        )
    }


def _generate_pair(case, provenance, config_hash):
    started = time.perf_counter()
    sidecar, rebuilt, pre_rng_state, capture_provenance = _load_pre_state(case)
    common_rng = _fresh_common_state(
        OPTIMIZATION_SEED, case["label"], case["seed"]
    )
    forks = [_fork(rebuilt, common_rng) for _ in PENALTIES]
    if forks[0][3] != forks[1][3]:
        raise RuntimeError("paired arms do not have identical numeric inputs")

    arms = {}
    for penalty, (runner, vp, gp, entry_hash) in zip(PENALTIES, forks):
        name = f"weight_penalty_{penalty:g}"
        print(f"[boost-campaign] {case['tag']} {name} ...", flush=True)
        arm = _run_arm(runner, vp, gp, penalty)
        actual = arm["actual_options"]
        if (
            actual.get("tol_elcbo_boost") is not None
            or actual.get("tol_weight") != 0
            or actual.get("weight_penalty") != penalty
        ):
            raise RuntimeError(f"incorrect optimizer-entry options for {name}")
        arm["entry_hash"] = entry_hash
        arm["rng_entry_state"] = copy.deepcopy(common_rng)
        arms[name] = arm

    pre_state = {
        "state_source": case["state_source"],
        "vp": rebuilt["vp"],
        "gp": rebuilt["gp"],
        "options": rebuilt["options"],
        "optim_state": rebuilt["optim_state"],
        "parameter_transformer": rebuilt["vp"].parameter_transformer,
        "rng_state": pre_rng_state,
        "capture_provenance": capture_provenance,
    }
    return {
        "generated_config_sha256": config_hash,
        "provenance": provenance,
        "case": case,
        "sidecar": sidecar,
        "pre_state": pre_state,
        "historical_pre_scores": copy.deepcopy(rebuilt["vp"].stats),
        "historical_pre_score_note": (
            "stored restart/trace scores; kept distinct from campaign rescoring"
        ),
        "optimization_common_rng_state": common_rng,
        "paired_entry_hash": forks[0][3],
        "arms": arms,
        "acceptance_criteria": "deferred; candidates and scores retained",
        "generation_wall_s": _seconds(started),
    }


def _finish_pair(generated):
    case = generated["case"]
    pre_state = generated["pre_state"]
    problem = find_config(case["label"]).make(seed=case["seed"])
    diagnostic_rng = _fresh_common_state(
        DIAGNOSTIC_SEED, case["label"], case["seed"]
    )
    scores = {
        "pre": _score(
            problem, pre_state["vp"], pre_state["gp"], diagnostic_rng
        )
    }
    for name, arm in generated["arms"].items():
        scores[name] = _score(
            problem, arm["candidate"], pre_state["gp"], diagnostic_rng
        )
    payload = copy.copy(generated)
    payload["diagnostic_common_rng_state"] = diagnostic_rng
    payload["diagnostic_scores"] = scores
    payload["wall_s"] = generated["generation_wall_s"]
    report = {
        "case": {
            key: case[key] for key in ("tag", "label", "seed", "state_source")
        },
        "paired_entry_hash": generated["paired_entry_hash"],
        "historical_pre_scores": generated["historical_pre_scores"],
        "diagnostic_scores": {
            name: _summary_score(score) for name, score in scores.items()
        },
        "arms": {
            name: {
                "actual_weight_penalty": arm["actual_options"].get(
                    "weight_penalty"
                ),
                "actual_tol_weight": arm["actual_options"].get("tol_weight"),
                "actual_tol_elcbo_boost": arm["actual_options"].get(
                    "tol_elcbo_boost"
                ),
                "returned_elbo": arm["returned_elbo"],
                "returned_elbo_sd": arm["returned_elbo_sd"],
                "optimizer_outputs": arm["optimizer_outputs"],
                "optimizer_s": arm["optimizer_s"],
                "final_boost_s": arm["final_boost_s"],
            }
            for name, arm in generated["arms"].items()
        },
        "wall_s": payload["wall_s"],
    }
    return payload, report


def _verified_complete(out_dir, case, config_hash):
    report_path = out_dir / f"{case['tag']}.json"
    capture_path = out_dir / f"{case['tag']}.dill"
    if not report_path.is_file() or not capture_path.is_file():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        return bool(
            report["completion"]["config_sha256"] == config_hash
            and report["completion"]["capture_sha256"] == _sha256(capture_path)
            and report["case"]["tag"] == case["tag"]
        )
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False


def _lock(out_dir):
    path = out_dir / ".campaign.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"campaign is already locked: {path}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "host": platform.node()}, stream)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cases", nargs="*", metavar="TAG")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.limit is not None and args.cases is not None:
        parser.error("--limit and --cases are mutually exclusive")

    args.out.mkdir(parents=True, exist_ok=True)
    lock_path = _lock(args.out)
    started = time.perf_counter()
    try:
        imports = _verify_imports()
        cases = _build_manifest(args.reference, args.capture_root)
        config = _campaign_config(args.reference, args.capture_root, cases)
        _initialize_config(args.out, config)
        audit = audit_population(args.reference)
        if (
            audit["runs"] != EXPECTED_CASES
            or audit["eligible_count"] != EXPECTED_CASES
        ):
            raise RuntimeError(
                "reference structural audit did not pass 870/870"
            )
        preflight = {
            "zero_fit": True,
            "imports": imports,
            "runs": audit["runs"],
            "eligible_count": audit["eligible_count"],
            "authentic_captures": sum(
                case["state_source"] == "authentic" for case in cases
            ),
            "config_sha256": config["config_sha256"],
        }
        _atomic_json(args.out / "preflight.json", preflight)
        if args.preflight:
            print("[boost-campaign] zero-fit preflight passed", flush=True)
            return 0

        if args.cases is not None:
            by_tag = {case["tag"]: case for case in cases}
            missing = set(args.cases) - set(by_tag)
            if missing:
                raise RuntimeError(
                    f"unknown campaign cases: {sorted(missing)}"
                )
            selected = [by_tag[tag] for tag in args.cases]
        else:
            selected = cases[: args.limit] if args.limit is not None else cases
        provenance = runtime_provenance()
        provenance["campaign"] = {
            "config_sha256": config["config_sha256"],
            "imports": imports,
            "sources": config["sources"],
        }
        counts = {
            "selected": len(selected),
            "skipped": 0,
            "succeeded": 0,
            "errors": 0,
        }
        for index, case in enumerate(selected, 1):
            if _verified_complete(args.out, case, config["config_sha256"]):
                counts["skipped"] += 1
                print(
                    f"[boost-campaign] {index}/{len(selected)} {case['tag']} verified; skip",
                    flush=True,
                )
                continue
            print(
                f"[boost-campaign] {index}/{len(selected)} {case['tag']}",
                flush=True,
            )
            try:
                generated_path = args.out / f"{case['tag']}.generated.dill"
                if generated_path.exists():
                    generated = _load_dill(generated_path)
                    if (
                        generated.get("generated_config_sha256")
                        != config["config_sha256"]
                        or generated.get("case", {}).get("tag") != case["tag"]
                    ):
                        raise RuntimeError(
                            f"incompatible generated checkpoint: {generated_path}"
                        )
                    print(
                        f"[boost-campaign] {case['tag']} recover generated arms",
                        flush=True,
                    )
                else:
                    generated = _generate_pair(
                        case, provenance, config["config_sha256"]
                    )
                    _atomic_dill(generated_path, generated)
                payload, report = _finish_pair(generated)
                capture_path = args.out / f"{case['tag']}.dill"
                _atomic_dill(capture_path, payload)
                report["completion"] = {
                    "config_sha256": config["config_sha256"],
                    "capture_sha256": _sha256(capture_path),
                }
                _atomic_json(args.out / f"{case['tag']}.json", report)
                generated_path.unlink(missing_ok=True)
                error_path = args.out / f"{case['tag']}.error.json"
                if error_path.exists():
                    error_path.unlink()
                counts["succeeded"] += 1
            except Exception as error:
                counts["errors"] += 1
                _atomic_json(
                    args.out / f"{case['tag']}.error.json",
                    {
                        "tag": case["tag"],
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    },
                )
                print(
                    f"[boost-campaign] {case['tag']} ERROR: {error}",
                    flush=True,
                )
            progress = {
                **counts,
                "processed": index,
                "population": len(cases),
                "config_sha256": config["config_sha256"],
                "wall_s": _seconds(started),
            }
            _atomic_json(args.out / "progress.json", progress)

        completed = sum(
            _verified_complete(args.out, case, config["config_sha256"])
            for case in cases
        )
        summary = {
            **counts,
            "population": len(cases),
            "verified_complete_population": completed,
            "pending_population": len(cases) - completed,
            "config_sha256": config["config_sha256"],
            "wall_s": _seconds(started),
        }
        _atomic_json(args.out / "summary.json", summary)
        print(f"[boost-campaign] finished: {summary}", flush=True)
        return 1 if counts["errors"] else 0
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
