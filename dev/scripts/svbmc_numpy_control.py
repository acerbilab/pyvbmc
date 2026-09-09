"""Complete-fit NumPy versus optimized-Torch S-VBMC attribution control.

This bounded development experiment keeps the pinned upstream S-VBMC
initialization, objective wrapper, Adam loop, rounded-loss stopping, cache
behavior, and result conversion.  Its sole algorithm override replaces
``stacked_entropy`` with the optimized NumPy preparation from
``svbmc_numpy_core`` followed by a vectorized resident Torch reduction.

The frozen primary comparison must run first.  This control writes a separate
artifact and does not modify the primary runner, numerical core, upstream
checkout, or production package.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import svbmc_numpy_prototype as harness

VERSIONS = ("all-weights", "posterior-only")
SEEDS = (1701, 1702, 1703)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=harness.DEFAULT_SOURCE)
    parser.add_argument("--deps", type=Path, default=harness.DEFAULT_DEPS)
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=tuple(harness.GROUPS),
        default=list(harness.GROUPS),
    )
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.n_samples <= 0 or args.max_steps <= 0:
        parser.error("sample and step counts must be positive")
    # Frozen-runner helpers consume these standard configuration fields.
    args.seeds = list(SEEDS)
    args.versions = list(VERSIONS)
    args.smoke_max_steps = 2
    args.prep_repeats = 1
    args.kernel_repeats = 1
    args.diagnostic_seed = SEEDS[0]
    return args


def optimized_torch_class(base_class: type, core: Any, torch: Any) -> type:
    """Return an upstream subclass overriding only sampled entropy."""

    class OptimizedTorchSVBMC(base_class):
        def stacked_entropy(self, w: Any, n_samples: int = 20):
            component_count = int(sum(self.K))
            logq_matrix, corrections = core.prepare_entropy(self, n_samples)
            logq = torch.tensor(logq_matrix, dtype=w.dtype, device=w.device)
            normalized = w.reshape(1, component_count)
            normalized = normalized / normalized.sum()
            log_mixture = torch.logsumexp(
                logq + torch.log(normalized + 1e-40), dim=1
            )
            component_means = log_mixture.reshape(
                component_count, n_samples
            ).mean(dim=1)
            entropy = -(normalized @ component_means)
            return entropy[0], corrections

    OptimizedTorchSVBMC.__name__ = "OptimizedTorchSVBMC"
    return OptimizedTorchSVBMC


def archive_sources(output: Path) -> dict[str, Any]:
    archive = output.parent / f"{output.stem}.sources"
    archive.mkdir(parents=True, exist_ok=True)
    sources = (
        Path(__file__).resolve(),
        harness.REPO_ROOT / "dev/scripts/svbmc_numpy_core.py",
        harness.REPO_ROOT / "dev/scripts/svbmc_numpy_prototype.py",
    )
    records = {}
    for source in sources:
        destination = archive / f"{source.name}.txt"
        destination.write_bytes(source.read_bytes())
        records[source.name] = {
            "live_path": str(source),
            "archive_path": str(destination),
            "sha256": harness.sha256_file(destination),
        }
    return {"directory": str(archive), "files": records}


def make_control_stacker(
    args: argparse.Namespace, modules: dict[str, Any], group: str
) -> Any:
    with contextlib.redirect_stdout(harness._NullWriter()):
        return modules["ControlSVBMC"](
            harness.load_group(args.source, group), testing=False
        )


def entropy_gate(
    args: argparse.Namespace, modules: dict[str, Any], group: str
) -> dict[str, Any]:
    """Compare the override directly with unchanged upstream and its gradient."""
    np, torch = modules["np"], modules["torch"]
    actual = harness.make_stacker(args, modules, group)
    control = make_control_stacker(args, modules, group)
    weights_np = np.asarray(actual.w, dtype=np.float64).reshape(-1)

    harness.seed_all(SEEDS[0], np, torch)
    actual_weights = torch.tensor(
        weights_np, dtype=torch.float64, requires_grad=True
    )
    actual_entropy, actual_jacobian = actual.stacked_entropy(
        actual_weights, n_samples=args.n_samples
    )
    actual_entropy.backward()
    actual_rng = np.random.get_state()

    harness.seed_all(SEEDS[0], np, torch)
    control_weights = torch.tensor(
        weights_np, dtype=torch.float64, requires_grad=True
    )
    control_entropy, control_jacobian = control.stacked_entropy(
        control_weights, n_samples=args.n_samples
    )
    control_entropy.backward()
    control_rng = np.random.get_state()

    return {
        "group": group,
        "entropy": harness.compare_arrays(
            float(control_entropy.detach()),
            float(actual_entropy.detach()),
            harness.TOLERANCES["fixed_objective"],
            np,
        ),
        "weight_gradient": harness.compare_arrays(
            control_weights.grad.detach().cpu().numpy(),
            actual_weights.grad.detach().cpu().numpy(),
            harness.TOLERANCES["fixed_gradient"],
            np,
        ),
        "jacobian": harness.compare_arrays(
            control_jacobian,
            actual_jacobian,
            harness.TOLERANCES["jacobian_correction"],
            np,
        ),
        "rng_state_equal": harness.rng_states_equal(
            actual_rng, control_rng, np
        ),
    }


def run_control_fit(
    group: str,
    seed: int,
    version: str,
    args: argparse.Namespace,
    modules: dict[str, Any],
) -> dict[str, Any]:
    np, torch = modules["np"], modules["torch"]
    construct_wall_started = time.perf_counter()
    construct_cpu_started = time.process_time()
    stacker = make_control_stacker(args, modules, group)
    construct_cpu_seconds = time.process_time() - construct_cpu_started
    construct_wall_seconds = time.perf_counter() - construct_wall_started

    harness.seed_all(seed, np, torch)
    rng_before = np.random.get_state()
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    with contextlib.redirect_stdout(harness._NullWriter()):
        stacker.optimize(
            n_samples=args.n_samples,
            lr=args.lr,
            max_steps=args.max_steps,
            version=version,
        )
    cpu_seconds = time.process_time() - cpu_started
    wall_seconds = time.perf_counter() - wall_started
    rng_after = np.random.get_state()
    result = harness.upstream_result(stacker, np)
    result.update(
        {
            "backend": "optimized_torch_control",
            "wall_seconds": wall_seconds,
            "cpu_seconds": cpu_seconds,
            "load_and_construction_wall_seconds": construct_wall_seconds,
            "load_and_construction_cpu_seconds": construct_cpu_seconds,
            "rng_before_sha256": harness.rng_state_digest(rng_before, np),
            "rng_after_sha256": harness.rng_state_digest(rng_after, np),
        }
    )
    return result


def run_pair(
    group: str,
    seed: int,
    version: str,
    pair_index: int,
    args: argparse.Namespace,
    modules: dict[str, Any],
) -> dict[str, Any]:
    np = modules["np"]
    order = ["numpy", "optimized_torch_control"]
    if pair_index % 2:
        order.reverse()
    fits = {}
    for backend in order:
        if backend == "numpy":
            fits[backend] = harness.run_numpy_fit(
                group,
                seed,
                version,
                args.max_steps,
                args,
                modules,
                trace=False,
            )
        else:
            fits[backend] = run_control_fit(
                group, seed, version, args, modules
            )
    comparison = harness.compare_fit_results(
        fits["numpy"], fits["optimized_torch_control"], np
    )
    result = {
        "group": group,
        "seed": seed,
        "version": version,
        "execution_order": order,
        "fits": fits,
        "comparison": comparison,
    }
    if not comparison["all_numeric_passed"]:
        rescore = harness.rescore_weights_on_identical_draws(
            group,
            seed,
            fits["numpy"]["weights"],
            fits["optimized_torch_control"]["weights"],
            args,
            modules,
        )
        rescore["weights"]["optimized_torch_control_fit"] = rescore[
            "weights"
        ].pop("upstream_fit")
        result["independent_rescore"] = rescore
    return result


def warm_up(
    args: argparse.Namespace, modules: dict[str, Any]
) -> dict[str, Any]:
    started = time.perf_counter()
    group = args.groups[0]
    version = VERSIONS[0]
    original_steps = args.max_steps
    args.max_steps = 2
    try:
        harness.run_numpy_fit(
            group, SEEDS[0], version, 2, args, modules, trace=False
        )
        run_control_fit(group, SEEDS[0], version, args, modules)
    finally:
        args.max_steps = original_steps
    return {
        "wall_seconds": time.perf_counter() - started,
        "excluded_from_clean_timings": True,
        "fits": 2,
        "steps_per_fit": 2,
    }


def checkpoint(report: dict[str, Any], output: Path) -> None:
    report["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    harness.atomic_write_json(output, report)


def run(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"output already exists: {output}; pass --overwrite to replace it"
        )
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "initializing",
        "tolerances_declared_before_measurement": harness.TOLERANCES,
        "config": {
            "groups": args.groups,
            "seeds": args.seeds,
            "versions": args.versions,
            "n_samples": args.n_samples,
            "lr": args.lr,
            "max_steps": args.max_steps,
            "source": str(args.source.resolve()),
            "deps": str(args.deps.resolve()),
        },
        "design": {
            "control": (
                "unchanged upstream optimize/Adam/stopping with only "
                "stacked_entropy replaced by optimized NumPy preparation "
                "and vectorized resident float64 Torch reduction"
            ),
            "fit_order": "backend order alternates across paired cells",
            "timing": (
                "clean optimize-only wall/process times; load/construction, "
                "warmup, and gates are separate"
            ),
        },
        "results": [],
        "errors": [],
    }
    report["frozen_experiment_sources"] = archive_sources(output)
    checkpoint(report, output)

    modules = harness.import_backends(args)
    modules["ControlSVBMC"] = optimized_torch_class(
        modules["SVBMC"], modules["core"], modules["torch"]
    )
    report["environment"] = harness.verify_environment(args, modules)
    report["optimizer_parity"] = harness.optimizer_parity_check(modules)
    report["entropy_gates"] = [
        entropy_gate(args, modules, group) for group in args.groups
    ]
    report["warmup"] = warm_up(args, modules)
    report["status"] = "running"
    checkpoint(report, output)

    pair_index = 0
    for group in args.groups:
        for version in args.versions:
            for seed in args.seeds:
                report["results"].append(
                    run_pair(
                        group,
                        seed,
                        version,
                        pair_index,
                        args,
                        modules,
                    )
                )
                pair_index += 1
                checkpoint(report, output)

    failures = harness.collect_validation_failures(report["entropy_gates"])
    failures.extend(
        harness.collect_validation_failures(
            report["optimizer_parity"], "optimizer_parity"
        )
    )
    failures.extend(harness.collect_validation_failures(report["results"]))
    failures = sorted(set(failures))
    report["validation"] = {
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
    }
    report["status"] = "complete" if not failures else "complete_with_failures"
    report["completed_utc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )
    checkpoint(report, output)
    return 0 if not failures else 2


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        output = args.output.resolve()
        if not isinstance(exc, FileExistsError) and output.exists():
            try:
                report = json.loads(output.read_text(encoding="utf-8"))
                report["status"] = "failed_exception"
                report.setdefault("errors", []).append(
                    {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                checkpoint(report, output)
            except Exception as checkpoint_error:
                print(
                    f"failed to checkpoint exception: {checkpoint_error}",
                    file=sys.stderr,
                )
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
