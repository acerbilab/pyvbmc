"""Run the bounded equal-iteration A/B eta experiment from saved inputs.

The runner reuses each saved C-sieve result, intercepts the variant's actual
Adam call, forces 400 iterations without early stopping, and exits before
endpoint selection or pruning.  It performs no sieve, GP fit, target call,
or full VBMC trajectory.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
import platform
import sys
import time
from pathlib import Path

import dill
import eta_bound_comparison as previous
import eta_bound_variants
import gpyreg
import numpy as np
import scipy
from eta_bound_variants import get_variant

import pyvbmc

REPO_ROOT = Path(__file__).resolve().parents[2]
STATES = (
    "rosenbrock_D2_noise1_seed0_pre",
    "rosenbrock_D2_noise3_seed17_pre",
)
ARMS = ("A", "B")
REPLICATES = (0, 1, 2)
CHECKPOINTS = (40, 100, 200, 400)
DIAGNOSTIC_REPLICATES = 10
DIAGNOSTIC_SAMPLES = 100_000
DIAGNOSTIC_SEED_BASE = 2026090900
ITERATIONS = 400
VERSION = 1


class _AdamComplete(Exception):
    """Private control flow used to skip endpoint selection and pruning."""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        default=Path(
            "dev/scripts/runs/latent_fixes/eta_bound_20260908/"
            "comparison/pairs"
        ),
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit-pairs", type=int)
    return parser.parse_args()


def resolve(path):
    return (
        path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
    )


def compact_json(path, value):
    data = (
        json.dumps(
            previous._jsonable(value),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    previous._atomic_bytes(path, data)


def configuration():
    import pyvbmc.vbmc.minimize_adam as adam_module

    modules = {
        name: previous._module_record(module)
        for name, module in {
            "runner": sys.modules[__name__],
            "previous_runner": previous,
            "eta_bound_variants": eta_bound_variants,
            "variational_optimization": previous.baseline_vo,
            "minimize_adam": adam_module,
            "pyvbmc": pyvbmc,
            "gpyreg": gpyreg,
            "numpy": np,
            "scipy": scipy,
            "dill": dill,
        }.items()
    }
    config = {
        "schema": "pyvbmc-eta-equal-budget-config-v1",
        "version": VERSION,
        "git_head": previous._git_head(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "states": list(STATES),
        "replicates": list(REPLICATES),
        "arms": list(ARMS),
        "iterations": ITERATIONS,
        "use_early_stopping": False,
        "checkpoints": list(CHECKPOINTS),
        "checkpoint_rule": "mean of latest 20 post-update theta iterates",
        "diagnostic_replicates": DIAGNOSTIC_REPLICATES,
        "diagnostic_samples": DIAGNOSTIC_SAMPLES,
        "diagnostic_seed_base": DIAGNOSTIC_SEED_BASE,
        "threads": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
        "modules": modules,
    }
    config["config_sha256"] = previous._sha256_bytes(
        previous._canonical_json(config)
    )
    return config


def prepare_output(out, config):
    out.mkdir(parents=True, exist_ok=True)
    path = out / "config.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != previous._jsonable(
            config
        ):
            raise ValueError(f"configuration differs from existing {path}")
    else:
        previous._atomic_json(path, config)


def load_input(inputs, state, replicate):
    path = inputs / state / f"replicate_{replicate}" / "input.dill"
    pair = previous._load_dill(path)
    if (pair.get("state_label"), pair.get("replicate")) != (state, replicate):
        raise ValueError(f"saved label/replicate mismatch in {path}")
    if pair.get("state_index") not in (5, 7):
        raise ValueError(f"unexpected saved state index in {path}")
    settings = pair.get("optimizer_settings", {})
    values = settings.get("values", {})
    if (
        settings.get("slow_opts_N") != 1
        or values.get("stochastic_optimizer") != "adam"
        or values.get("max_iter_stochastic") != ITERATIONS
    ):
        raise ValueError(f"saved optimizer settings differ in {path}")
    return path, pair


def run_arm(arm, pair):
    variant = get_variant(arm)
    rng = previous._clone_rng(pair["post_sieve_rng_state"])
    vp = previous._clone_vp(pair["post_sieve_vp"], rng)
    gp = copy.deepcopy(pair["gp"])
    optim_state = copy.deepcopy(pair["optim_state"])
    options = copy.deepcopy(pair["options"])
    if (
        previous._input_hashes(vp, gp, optim_state, options)
        != pair["post_sieve_hashes"]
    ):
        raise RuntimeError(f"arm {arm} input clone differs")

    captured = {}
    original_sieve = variant._sieve
    original_adam = variant.minimize_adam

    def supplied_sieve(*args, **kwargs):
        if captured.get("sieve_calls", 0):
            raise RuntimeError("saved sieve requested more than once")
        captured["sieve_calls"] = 1
        result = previous._clone_sieve_result(pair["sieve_result"], rng)
        if previous._sieve_hashes(result) != pair["sieve_hashes"]:
            raise RuntimeError("supplied sieve differs from saved sieve")
        return result

    def intercepted_adam(objective, x0, *args, **kwargs):
        if "result" in captured:
            raise RuntimeError("variant called Adam more than once")
        closure = inspect.getclosurevars(objective).nonlocals
        selected_vp = closure.get("vp0")
        if selected_vp is None:
            raise RuntimeError("could not recover selected VP from objective")
        original_kwargs = copy.deepcopy(kwargs)
        run_kwargs = dict(kwargs)
        run_kwargs.update(max_iter=ITERATIONS, use_early_stopping=False)
        x0_copy = np.asarray(x0).copy()
        selected_copy = copy.deepcopy(selected_vp)
        objective_calls = 0

        def counted_objective(theta):
            nonlocal objective_calls
            objective_calls += 1
            return objective(theta)

        started = time.perf_counter()
        result = original_adam(counted_objective, x0, *args, **run_kwargs)
        elapsed = time.perf_counter() - started
        theta, values, iterations = result[2], result[3], int(result[4])
        if (
            iterations != ITERATIONS
            or theta.shape[1] != ITERATIONS
            or objective_calls != ITERATIONS
        ):
            raise RuntimeError(f"arm {arm} did not run exactly 400 iterations")
        checkpoints = {}
        for checkpoint in CHECKPOINTS:
            mean_theta = np.mean(
                theta[:, checkpoint - 20 : checkpoint], axis=1
            )
            checkpoint_vp = copy.deepcopy(selected_copy)
            checkpoint_vp.set_parameters(mean_theta)
            checkpoints[checkpoint] = {
                "theta": mean_theta,
                "vp": checkpoint_vp,
            }
        captured["result"] = {
            "x0": x0_copy,
            "x0_sha256": previous._digest(x0_copy),
            "objective": f"{objective.__module__}.{objective.__qualname__}",
            "original_kwargs": original_kwargs,
            "run_kwargs": run_kwargs,
            "theta_trajectory": np.asarray(theta).copy(),
            "y_trajectory": np.asarray(values).copy(),
            "iterations": iterations,
            "objective_calls": objective_calls,
            "elapsed_seconds": elapsed,
            "selected_vp": selected_copy,
            "checkpoints": checkpoints,
        }
        raise _AdamComplete

    variant._sieve = supplied_sieve
    variant.minimize_adam = intercepted_adam
    rng_before = copy.deepcopy(rng.bit_generator.state)
    try:
        try:
            variant.optimize_vp(
                options,
                optim_state,
                vp,
                gp,
                int(pair["state_spec"]["fast_opts_N"]),
                int(pair["state_spec"]["slow_opts_N"]),
                vp.K,
            )
        except _AdamComplete:
            pass
        else:
            raise RuntimeError(
                "optimizer returned instead of interception exit"
            )
    finally:
        variant._sieve = original_sieve
        variant.minimize_adam = original_adam
    if captured.get("sieve_calls") != 1 or "result" not in captured:
        raise RuntimeError(f"arm {arm} interception was incomplete")
    captured["result"]["rng_before"] = rng_before
    captured["result"]["rng_after"] = copy.deepcopy(rng.bit_generator.state)
    return captured["result"]


def diagnostic_state(pair, checkpoint, diagnostic_replicate):
    sequence = np.random.SeedSequence(
        [
            DIAGNOSTIC_SEED_BASE,
            int(pair["state_index"]),
            int(pair["replicate"]),
            checkpoint,
            diagnostic_replicate,
        ]
    )
    return copy.deepcopy(np.random.default_rng(sequence).bit_generator.state)


def score_checkpoints(pair, arms):
    scores = {}
    for checkpoint in CHECKPOINTS:
        checkpoint_scores = []
        for diagnostic_replicate in range(DIAGNOSTIC_REPLICATES):
            state = diagnostic_state(pair, checkpoint, diagnostic_replicate)
            arm_scores = {
                arm: previous._score_vp(
                    arms[arm]["checkpoints"][checkpoint]["vp"],
                    pair["gp"],
                    state,
                    DIAGNOSTIC_SAMPLES,
                )
                for arm in ARMS
            }
            arm_scores["B_minus_A"] = {
                key: arm_scores["B"][key] - arm_scores["A"][key]
                for key in ("elbo", "elbo_sd", "elcbo_beta5")
            }
            checkpoint_scores.append(
                {"diagnostic_replicate": diagnostic_replicate, **arm_scores}
            )
        scores[checkpoint] = checkpoint_scores
    return scores


def compact_report(pair, input_path, arms, scores, config):
    return {
        "schema": "pyvbmc-eta-equal-budget-pair-v1",
        "config_sha256": config["config_sha256"],
        "state_label": pair["state_label"],
        "state_index": pair["state_index"],
        "replicate": pair["replicate"],
        "input": {
            "path": str(input_path),
            "sha256": previous._sha256_file(input_path),
            "source_sha256": pair["state_spec"].get("source_sha256"),
            "post_sieve_hashes": pair["post_sieve_hashes"],
            "sieve_hashes": pair["sieve_hashes"],
            "post_sieve_rng_sha256": previous._digest(
                pair["post_sieve_rng_state"]
            ),
        },
        "arms": {
            arm: {
                "x0_sha256": data["x0_sha256"],
                "objective": data["objective"],
                "original_kwargs": data["original_kwargs"],
                "run_kwargs": data["run_kwargs"],
                "iterations": data["iterations"],
                "objective_calls": data["objective_calls"],
                "elapsed_seconds": data["elapsed_seconds"],
                "theta_trajectory_sha256": previous._digest(
                    data["theta_trajectory"]
                ),
                "y_trajectory_sha256": previous._digest(data["y_trajectory"]),
                "rng_before_sha256": previous._digest(data["rng_before"]),
                "rng_after_sha256": previous._digest(data["rng_after"]),
                "checkpoint_theta_sha256": {
                    checkpoint: previous._digest(item["theta"])
                    for checkpoint, item in data["checkpoints"].items()
                },
            }
            for arm, data in arms.items()
        },
        "scores": scores,
        "notes": [
            "Each checkpoint is the latest-20 mean of post-update Adam theta iterates, before selection or pruning.",
            "y_trajectory records the objective before each corresponding Adam update.",
            "Scoring resets a common diagnostic RNG across A/B; GP SD excludes entropy Monte Carlo uncertainty.",
            "Selected saved states and optimizer replicates are conditional local evidence, not independent benchmark runs.",
        ],
    }


def run_pair(inputs, out, state, replicate, config):
    pair_dir = out / state / f"replicate_{replicate}"
    complete_path = pair_dir / "complete.json"
    input_path, pair = load_input(inputs, state, replicate)
    input_sha = previous._sha256_file(input_path)
    if complete_path.exists():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        expected = (config["config_sha256"], input_sha)
        observed = (
            complete.get("config_sha256"),
            complete.get("input_sha256"),
        )
        if observed != expected:
            raise ValueError(f"incompatible completion {complete_path}")
        for name, digest in complete["artifacts"].items():
            if previous._sha256_file(pair_dir / name) != digest:
                raise ValueError(
                    f"changed completion artifact {pair_dir / name}"
                )
        return "skipped"

    pair_dir.mkdir(parents=True, exist_ok=True)
    arms = {arm: run_arm(arm, pair) for arm in ARMS}
    if arms["A"]["x0_sha256"] != arms["B"]["x0_sha256"]:
        raise RuntimeError("A/B selected optimizer starts differ")
    if previous._digest(arms["A"]["rng_before"]) != previous._digest(
        arms["B"]["rng_before"]
    ):
        raise RuntimeError("A/B post-sieve RNG starts differ")
    if previous._digest(arms["A"]["rng_after"]) != previous._digest(
        arms["B"]["rng_after"]
    ):
        raise RuntimeError("A/B 400-step RNG endpoints differ")
    for key in ("max_iter", "master_min", "master_max", "master_decay"):
        if arms["A"]["run_kwargs"].get(key) != arms["B"]["run_kwargs"].get(
            key
        ):
            raise RuntimeError(f"A/B Adam schedule differs for {key}")
    scores = score_checkpoints(pair, arms)
    capture = {
        "version": VERSION,
        "config_sha256": config["config_sha256"],
        "input_sha256": input_sha,
        "state_label": state,
        "replicate": replicate,
        "arms": arms,
        "scores": scores,
    }
    dill_path = pair_dir / "result.dill"
    json_path = pair_dir / "result.json"
    previous._atomic_dill(dill_path, capture)
    compact_json(
        json_path, compact_report(pair, input_path, arms, scores, config)
    )
    complete = {
        "schema": "pyvbmc-eta-equal-budget-complete-v1",
        "config_sha256": config["config_sha256"],
        "input_sha256": input_sha,
        "artifacts": {
            path.name: previous._sha256_file(path)
            for path in (dill_path, json_path)
        },
    }
    previous._atomic_json(complete_path, complete)
    return "completed"


def main():
    args = parse_args()
    inputs, out = resolve(args.inputs), resolve(args.out)
    if args.limit_pairs is not None and args.limit_pairs < 1:
        raise ValueError("--limit-pairs must be positive")
    config = configuration()
    prepare_output(out, config)
    tasks = [
        (state, replicate) for state in STATES for replicate in REPLICATES
    ]
    if args.limit_pairs is not None:
        tasks = tasks[: args.limit_pairs]
    counts = {"completed": 0, "skipped": 0}
    for state, replicate in tasks:
        status = run_pair(inputs, out, state, replicate, config)
        counts[status] += 1
        print(f"{status}: {state} replicate {replicate}", flush=True)
    print(json.dumps(counts, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
