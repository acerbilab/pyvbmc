"""Measure call-local prediction-kernel reuse on captured VIQR states.

This developer experiment creates isolated subclasses from the installed
source with checked textual substitutions. It edits no production module
and patches no live class. Prediction optionally returns its cross-kernel
matrices; the acquisition receives them as a call-local argument. Compare
transpose views with C-contiguous copies, including prediction and copying
in every timed public call. No target evaluations or sampling occur.

Example::

    python dev/scripts/probe_viqr_kernel_reuse.py --captures \
        dev/scripts/runs/viqr_sinh_20260913/replay --out DIR

Pass --captures more than once to include supplementary captured states.
The capture archives are local ignored artifacts from validate_viqr_sinh.py.
"""

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import textwrap
import time
import tracemalloc
from pathlib import Path

# Set thread limits before any numerical import, including transitive ones.
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import gpyreg
import numpy as np
import validate_viqr_sinh as validation
from threadpoolctl import threadpool_info

from pyvbmc.acquisition_functions import AcqFcnVIQR
from pyvbmc.acquisition_functions.abstract_acq_fcn import AbstractAcqFcn


def replace_once(source, old, new):
    assert source.count(old) == 1, (old, source.count(old))
    return source.replace(old, new)


def compile_method(source, original, name):
    namespace = original.__globals__.copy()
    exec(compile(source, f"<kernel-reuse-probe:{name}>", "exec"), namespace)
    return namespace[original.__name__]


def prototypes():
    """Return isolated GP/acquisition types and their exact generated source."""
    source = textwrap.dedent(inspect.getsource(gpyreg.GP.predict))
    source = replace_once(
        source,
        "    return_lpd: bool = False,",
        "    return_lpd: bool = False,\n    return_cross_covariance=False,",
    )
    source = replace_once(
        source,
        "    # Preallocate space",
        "    cross_covariance = [] if return_cross_covariance else None\n"
        "    # Preallocate space",
    )
    line = "            Ks = self.covariance.compute(hyp[0:cov_N], self.X, x_star)"
    source = replace_once(
        source,
        line,
        line + "\n            if return_cross_covariance:\n"
        "                cross_covariance.append(Ks)",
    )
    source = replace_once(
        source,
        "        return mu, s2\n",
        "        if return_cross_covariance:\n"
        "            return mu, s2, cross_covariance\n"
        "        return mu, s2\n",
    )
    gp_type = type(
        "ProbeGP",
        (gpyreg.GP,),
        {"predict": compile_method(source, gpyreg.GP.predict, "predict")},
    )
    sources = {"predict.py": source}

    wrapper = textwrap.dedent(inspect.getsource(AbstractAcqFcn.__call__))
    wrapper = replace_once(
        wrapper,
        "    f_mu, f_s2 = gp.predict(x_star=Xs, separate_samples=True)",
        "    f_mu, f_s2, cross_covariance = gp.predict(\n"
        "        x_star=Xs, separate_samples=True,\n"
        "        return_cross_covariance=True,\n    )",
    )
    wrapper = replace_once(
        wrapper,
        "        var_tot,\n    )",
        "        var_tot,\n" "        cross_covariance,\n    )",
    )
    sources["call.py"] = wrapper
    call = compile_method(wrapper, AbstractAcqFcn.__call__, "call")
    types = {}
    for layout in ("view", "contiguous"):
        compute = textwrap.dedent(
            inspect.getsource(AcqFcnVIQR._compute_acquisition_function)
        )
        compute = replace_once(
            compute,
            "    var_tot: None,",
            "    var_tot: None,\n" "    cross_covariance,",
        )
        compute = replace_once(
            compute,
            "super()._estimate_observation_noise(Xs, gp, optim_state)",
            "AbstractAcqFcn._estimate_observation_noise(\n"
            "        self, Xs, gp, optim_state\n    )",
        )
        kernel = "cross_covariance[s].T"
        if layout == "contiguous":
            kernel = f"np.ascontiguousarray({kernel})"
        compute = replace_once(
            compute,
            '            tmp = cdist(Xs_ell, gp.X / ell, "sqeuclidean")\n'
            "            K_Xs_X = sf2 * np.exp(-tmp / 2)",
            f"            K_Xs_X = {kernel}",
        )
        sources[f"compute_{layout}.py"] = compute
        types[layout] = type(
            f"ProbeVIQR_{layout}",
            (AcqFcnVIQR,),
            {
                "__call__": call,
                "_compute_acquisition_function": compile_method(
                    compute, AcqFcnVIQR._compute_acquisition_function, layout
                ),
            },
        )
    return gp_type, types, sources


def captured_states(directories):
    for directory in directories:
        for path in sorted((directory / "snapshots").glob("*.status.json")):
            status = json.loads(path.read_text())
            for filename, expected in status["file_hashes"].items():
                assert (
                    validation.file_hash(directory / "snapshots" / filename)
                    == expected
                )
            for checkpoint in status["captured"]:
                name = f"{status['label']}_{checkpoint}"
                state = validation.restore(directory / "snapshots" / name)
                assert state["meta"]["shared_source_hashes"] == (
                    validation.shared_source_hashes()
                )
                yield directory, name, status["file_hashes"], state


def state_digest(state):
    gp, vp = state["gp"], state["vp"]
    return validation.digest(
        (
            state["cand"],
            gp.X,
            gp.y,
            gp.posteriors,
            gp.temporary_data,
            vp.mu,
            vp.w,
            vp.sigma,
            vp.lambd,
            state["optim_state"],
            vp.rng.bit_generator.state,
            np.random.get_state(),
        )
    )


def compare(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    finite = np.isfinite(expected)
    return {
        "exact": bool(np.array_equal(actual, expected)),
        "max_abs": float(
            np.max(np.abs(actual[finite] - expected[finite]), initial=0)
        ),
    }


def run(args):
    pools = threadpool_info()
    assert all(p["num_threads"] == 1 for p in pools if p["user_api"] == "blas")
    gp_type, acq_types, sources = prototypes()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        (args.out / name).write_text(source, encoding="utf-8")
    report = {
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=validation.ROOT, text=True
        ).strip(),
        "probe_sha256": validation.file_hash(__file__),
        "threadpools": pools,
        "source_hashes": validation.replay_source_hashes(),
        "execution_inputs": validation.execution_inputs(),
        "generated_source_hashes": {
            name: hashlib.sha256(source.encode()).hexdigest()
            for name, source in sources.items()
        },
        "rounds": args.rounds,
        "small_pairs_per_round": args.small_pairs,
        "method": "Alternating complete calls; baseline vs each reuse layout",
        "limitations": [
            "Prototype interface, not a production implementation",
            "Captured-state coverage only; no end-to-end run speed claim",
            "Peak allocation is a single tracemalloc observation per method",
            "Cross kernels are retained for all hyperparameter samples",
        ],
        "rows": [],
    }
    for directory, name, hashes, state in captured_states(args.captures):
        gp = state["gp"]
        probe_gp = object.__new__(gp_type)
        probe_gp.__dict__ = gp.__dict__.copy()
        assert gp.y is not None
        candidates = state["cand"]["Xs"]
        tail = (state["vp"], state["logger"], state["optim_state"])
        before = state_digest(state)
        baseline = AcqFcnVIQR()
        live = baseline(candidates.copy(), gp, *tail)
        capture_comparison = compare(live, state["ref"]["acq"])
        baseline_repeats = [
            compare(baseline(candidates.copy(), gp, *tail), live)
            for _ in range(3)
        ]
        population = 4 + int(np.floor(3 * np.log(gp.D)))
        for count in (1, population, len(candidates)):
            xs = candidates[:count].copy()
            methods = {
                "baseline": lambda: baseline(xs, gp, *tail),
                **{
                    key: lambda acq=cls(): acq(xs, probe_gp, *tail)
                    for key, cls in acq_types.items()
                },
            }
            mu, s2 = gp.predict(xs, separate_samples=True)
            mu_default, s2_default = probe_gp.predict(
                xs, separate_samples=True
            )
            default_prediction = [
                compare(mu_default, mu),
                compare(s2_default, s2),
            ]
            mu_new, s2_new, kernels = probe_gp.predict(
                xs, separate_samples=True, return_cross_covariance=True
            )
            reused_prediction = [compare(mu_new, mu), compare(s2_new, s2)]
            from scipy.spatial.distance import cdist

            kernels_exact = True
            for posterior, matrix in zip(gp.posteriors, kernels):
                ell = np.exp(posterior.hyp[: gp.D])
                expected = np.exp(2 * posterior.hyp[gp.D]) * np.exp(
                    -cdist(xs / ell, gp.X / ell, "sqeuclidean") / 2
                )
                kernels_exact &= np.array_equal(expected, matrix.T)
            retained_bytes = sum(matrix.nbytes for matrix in kernels)
            del kernels, matrix, expected
            outputs = {key: call() for key, call in methods.items()}
            differences = {}
            for key in acq_types:
                np.testing.assert_allclose(
                    outputs[key], outputs["baseline"], rtol=1e-12, atol=1e-12
                )
                finite = np.isfinite(outputs["baseline"])
                differences[key] = {
                    "exact": bool(
                        np.array_equal(outputs[key], outputs["baseline"])
                    ),
                    "max_abs": float(
                        np.max(
                            np.abs(
                                outputs[key][finite]
                                - outputs["baseline"][finite]
                            ),
                            initial=0,
                        )
                    ),
                    "argmin_equal": bool(
                        np.argmin(outputs[key])
                        == np.argmin(outputs["baseline"])
                    ),
                }
            pairs = args.small_pairs if count < 100 else 1
            timings = {key: {"baseline": [], "reuse": []} for key in acq_types}
            for round_id in range(args.rounds):
                for key in list(acq_types)[:: 1 if round_id % 2 == 0 else -1]:
                    samples = {"baseline": [], "reuse": []}
                    for pair in range(pairs):
                        order = ("baseline", "reuse")
                        if (pair + round_id) % 2:
                            order = order[::-1]
                        for member in order:
                            call = methods[
                                "baseline" if member == "baseline" else key
                            ]
                            start = time.perf_counter_ns()
                            call()
                            samples[member].append(
                                (time.perf_counter_ns() - start) / 1e6
                            )
                    for member in samples:
                        timings[key][member].extend(samples[member])
            peaks = {}
            for key, call in methods.items():
                tracemalloc.start()
                call()
                peaks[key] = tracemalloc.get_traced_memory()[1]
                tracemalloc.stop()
            row = {
                "capture_directory": str(directory),
                "snapshot": name,
                "capture_file_hashes": hashes,
                "N": len(gp.X),
                "D": gp.D,
                "Ns": len(gp.posteriors),
                "Nc": count,
                "L_chol": [bool(p.L_chol) for p in gp.posteriors],
                "kernels_exact": bool(kernels_exact),
                "default_prediction": default_prediction,
                "reused_prediction": reused_prediction,
                "baseline_vs_capture": capture_comparison,
                "baseline_repeats": baseline_repeats,
                "differences": differences,
                "retained_kernel_bytes": retained_bytes,
                "peak_traced_bytes": peaks,
                "timings_ms": timings,
                "paired_median_speedups": {
                    key: float(
                        np.median(
                            np.array(t["baseline"]) / np.array(t["reuse"])
                        )
                    )
                    for key, t in timings.items()
                },
            }
            report["rows"].append(row)
            (args.out / "results.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            print(
                name,
                count,
                row["paired_median_speedups"],
                differences,
                flush=True,
            )
        after = state_digest(state)
        assert before == after, "Probe mutated state or consumed random draws"
    assert report["rows"], "No captured states found"
    report["all_state_and_rng_checks_passed"] = True
    (args.out / "results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--captures", type=Path, action="append", required=True
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--small-pairs", type=int, default=31)
    run(parser.parse_args())
