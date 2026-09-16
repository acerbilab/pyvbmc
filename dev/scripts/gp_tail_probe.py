"""Replay and diagnose the first tail acquisitions of the GP box study.

This probe is specific to the corrected Rosenbrock noise-3 runs, seeds
1000--1002, produced by gp_box_probe.py on 2026-09-16. It never calls a
fresh noisy target: replay supplies recorded observations and requires
exact evaluation coordinates, GP hyperparameters, VP parameters and RNG.

Run replay once per seed into a new output directory, then analyze and
refit. Precision runs once per seed after analyze and needs mpmath 1.3.0.
All paths are explicit; raw input locations are listed in runs/LOCAL.md.
The accompanying report is dev/results/2026-09-16-gp-tail-acquisition.md.
Only load trusted saved VBMC instances and diagnostic pickles.
"""

import argparse
import copy
import importlib
import inspect
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def replay(args, seed):
    """Capture one saved batch through read-only Python trace callbacks."""
    iteration = {1000: 14, 1001: 24, 1002: 17}[seed]
    out = args.out / str(seed)
    out.mkdir(parents=True, exist_ok=False)
    tag = f"rosenbrock_D2_noise3_svbmc_seed{seed}"
    base = args.runs / tag
    original = cloudpickle.loads(base.with_suffix(".vbmc.pkl").read_bytes())
    with np.load(base.with_suffix(".npz")) as archive:
        calls = {key: archive[key] for key in ("X", "y")}
    vbmc = VBMC.load(
        base.with_suffix(".vbmc.pkl"),
        iteration=iteration,
        set_random_state=True,
    )
    next_call = int(vbmc.function_logger.func_count)
    start_call = next_call

    def target(x):
        nonlocal next_call
        expected = calls["X"][next_call]
        actual = np.asarray(x).reshape(-1)
        if not np.array_equal(expected, actual):
            raise AssertionError(
                f"Call {next_call+1} differs: {actual} vs {expected}; delta {actual-expected}"
            )
        value = float(calls["y"][next_call])
        next_call += 1
        return value, 3.0

    vbmc.function_logger.fun = target
    active_module = importlib.import_module("pyvbmc.vbmc.active_sample")
    vbmc_module = importlib.import_module("pyvbmc.vbmc.vbmc")
    active = active_module.active_sample
    source, start = inspect.getsourcelines(active)
    line_sieve = start + next(
        i for i, s in enumerate(source) if "X_acq = X_search[[idx]]" in s
    )
    line_capture = start + next(
        i for i, s in enumerate(source) if 'timer.start_timer("fun_time")' in s
    )
    sieve = None
    proposal = None

    def trace_local(frame, event, arg):
        nonlocal sieve, proposal
        if frame.f_code is active_module._get_search_points.__code__:
            if event == "return":
                loc = frame.f_locals
                proposal = copy.deepcopy(
                    {
                        k: loc[k]
                        for k in [
                            "N_heavy",
                            "N_mvn",
                            "N_box",
                            "N_vp",
                            "N_hpd",
                            "N_search_cache",
                            "box_lb",
                            "box_ub",
                        ]
                        if k in loc
                    }
                )
            return trace_local
        if event == "line" and frame.f_lineno == line_sieve:
            loc = frame.f_locals
            sieve = copy.deepcopy(
                {
                    k: loc[k]
                    for k in ["X_search", "acq_fast", "idx", "n_train_cand"]
                }
            )
        if event == "line" and frame.f_lineno == line_capture:
            loc = frame.f_locals
            keys = [
                "gp",
                "vp",
                "optim_state",
                "function_logger",
                "options",
                "X_acq",
                "f_val_old",
                "f_val_optim",
                "xsearch_optim",
                "active_sample_full_update",
            ]
            snapshot = {k: loc[k] for k in keys if k in loc}
            snapshot["function_logger"] = copy.copy(loc["function_logger"])
            snapshot["function_logger"].fun = None
            snapshot.update(sieve=sieve, proposal=proposal, call=next_call + 1)
            (out / f"call{next_call+1}.pkl").write_bytes(
                cloudpickle.dumps(snapshot)
            )
        return trace_local

    def trace(frame, event, arg):
        if event == "call" and frame.f_code in (
            active.__code__,
            active_module._get_search_points.__code__,
        ):
            return trace_local

    class Finished(Exception):
        pass

    def wrapped(*args, **kwargs):
        if next_call > start_call:
            raise Finished
        return active(*args, **kwargs)

    vbmc_module.active_sample = wrapped
    previous_trace = sys.gettrace()
    try:
        sys.settrace(trace)
        vbmc.optimize()
    except Finished:
        pass
    finally:
        sys.settrace(previous_trace)
        vbmc_module.active_sample = active

    i = iteration + 1
    actual_gp = vbmc.get_gp(i)
    expected_gp = original.get_gp(i)
    checks = {
        "seed": seed,
        "iteration": i,
        "first_call": start_call + 1,
        "last_call": next_call,
        "batch_complete": next_call == start_call + 5,
        "calls_exact": True,
        "gp_hyp_exact": np.array_equal(
            [p.hyp for p in actual_gp.posteriors],
            [p.hyp for p in expected_gp.posteriors],
        ),
        "gp_y_exact": np.array_equal(actual_gp.y, expected_gp.y),
        "gp_X_exact": np.array_equal(actual_gp.X, expected_gp.X),
        "vp_exact": all(
            np.array_equal(
                getattr(vbmc.iteration_history["vp"][i], k),
                getattr(original.iteration_history["vp"][i], k),
            )
            for k in ["mu", "w", "sigma", "lambd"]
        ),
        "rng_exact": vbmc.iteration_history["random_state"][i]
        == original.iteration_history["random_state"][i],
    }
    (out / "replay.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks), flush=True)
    assert all(
        checks[k]
        for k in [
            "calls_exact",
            "batch_complete",
            "gp_hyp_exact",
            "gp_y_exact",
            "gp_X_exact",
            "vp_exact",
            "rng_exact",
        ]
    )


def analyze(args):
    """Compare each choice with the best high-density sieve candidate."""
    out = args.out
    target = find_config("rosenbrock_D2").make(seed=0).log_density_vec
    summary = []
    for seed, tail in [(1000, 80), (1001, 134), (1002, 98)]:
        expected = {
            1000: range(76, 81),
            1001: range(131, 136),
            1002: range(96, 101),
        }[seed]
        paths = list((out / str(seed)).glob("call*.pkl"))
        if {int(p.stem[4:]) for p in paths} != set(expected):
            raise ValueError(
                f"Incomplete or unexpected replay captures for {seed}"
            )
        for path in sorted(
            (out / str(seed)).glob("call*.pkl"), key=lambda p: int(p.stem[4:])
        ):
            d = cloudpickle.loads(path.read_bytes())
            gp = d["gp"]
            vp = d["vp"]
            state = d["optim_state"]
            logger = d["function_logger"]
            sieve = d["sieve"]
            acq = AcqFcnVIQR()
            xs = sieve["X_search"]
            scores = sieve["acq_fast"].reshape(-1)
            idx = int(sieve["idx"])
            orig = vp.parameter_transformer.inverse(xs)
            truths = target(orig).reshape(-1)
            central = truths >= truths.max() - 20
            center_idx = np.flatnonzero(central)[np.argmin(scores[central])]
            candidates = np.vstack([d["X_acq"], xs[idx], xs[center_idx]])
            if d["call"] == tail:
                np.savez(out / str(seed) / "candidates.npz", X=candidates)
            point_orig = vp.parameter_transformer.inverse(candidates)
            mu, var = gp.predict(candidates, separate_samples=True)
            covn = gp.covariance.hyperparameter_count(gp.D)
            noisn = gp.noise.hyperparameter_count()
            hyp = np.array([p.hyp for p in gp.posteriors])
            ai = state["active_importance_sampling"]
            tau_all = []
            sn2 = acq._estimate_observation_noise(candidates, gp, state)
            for s, p in enumerate(gp.posteriors):
                k = gp.covariance.compute(p.hyp[:covn], candidates, gp.X)
                ka = gp.covariance.compute(p.hyp[:covn], candidates, ai["X"])
                c = (
                    ka - k @ ai["C_tmp"][s]
                    if p.L_chol
                    else ka + k @ ai["C_tmp"][s]
                )
                tau_all.append(c * c / (var[:, s] + sn2)[:, None])
            tau = np.array(tau_all).transpose(1, 2, 0)
            s_after = np.sqrt(np.maximum(ai["f_s2"][None, :, :] - tau, 0))
            baseline = np.mean(
                np.sum(2 * np.sinh(acq.u * np.sqrt(ai["f_s2"])), axis=0)
            )
            after = np.mean(
                np.sum(2 * np.sinh(acq.u * s_after), axis=1), axis=1
            )
            boundaries = np.cumsum(
                [sieve["n_train_cand"]]
                + [
                    d["proposal"].get(k, 0)
                    for k in [
                        "N_search_cache",
                        "N_heavy",
                        "N_mvn",
                        "N_hpd",
                        "N_box",
                        "N_vp",
                    ]
                ]
            )
            group = ["training", "cache", "heavy", "mvn", "hpd", "box", "vp"][
                np.flatnonzero(idx < boundaries)[0]
            ]
            row = {
                "seed": seed,
                "call": d["call"],
                "tail": d["call"] == tail,
                "full_update": bool(d["active_sample_full_update"]),
                "condition": gp_condition_number(gp),
                "N": len(gp.y),
                "training_y_min": gp.y.min(),
                "sieve_origin": group,
                "sieve_index": idx,
                "candidate_names": ["selected", "sieve", "best_central_sieve"],
                "x_orig": point_orig,
                "truth": target(point_orig),
                "score": acq(candidates, gp, vp, logger, state),
                "stored_sieve_score": scores[idx],
                "stored_optimizer_score": d["f_val_optim"],
                "best_central_stored_score": scores[center_idx],
                "fraction_iqr_reduction": 1 - after / baseline,
                "mu_transformed": mu.mean(axis=1),
                "mu_orig": mu.mean(axis=1)
                - vp.parameter_transformer.log_abs_det_jacobian(
                    candidates
                ).reshape(-1),
                "latent_variance": var.mean(axis=1),
                "candidate_noise": sn2,
                "clipped_variances": (tau > ai["f_s2"][None, :, :]).sum(
                    axis=(1, 2)
                ),
                "ell_min": np.exp(hyp[:, : gp.D]).min(axis=0),
                "ell_max": np.exp(hyp[:, : gp.D]).max(axis=0),
                "sf2_min": np.exp(2 * hyp[:, gp.D]).min(),
                "sf2_max": np.exp(2 * hyp[:, gp.D]).max(),
                "training_noise_range": [
                    np.min(
                        [
                            gp.noise.compute(
                                p.hyp[covn : covn + noisn], gp.X, gp.y, gp.s2
                            )
                            for p in gp.posteriors
                        ]
                    ),
                    np.max(
                        [
                            gp.noise.compute(
                                p.hyp[covn : covn + noisn], gp.X, gp.y, gp.s2
                            )
                            for p in gp.posteriors
                        ]
                    ),
                ],
                "sn2_mult": [p.sn2_mult for p in gp.posteriors],
                "importance_truth_range": [
                    np.min(target(vp.parameter_transformer.inverse(ai["X"]))),
                    np.max(target(vp.parameter_transformer.inverse(ai["X"]))),
                ],
            }
            summary.append(jsonable(row))
            if row["tail"]:
                print(json.dumps(jsonable(row)), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def precision(args, seed):
    """Rebuild all GP covariances and the fixed-point VIQR at 50 digits."""
    import mpmath as mp

    out = args.out
    call = {1000: 80, 1001: 134, 1002: 98}[seed]
    mp.mp.dps = 50
    d = cloudpickle.loads((out / str(seed) / f"call{call}.pkl").read_bytes())
    gp = d["gp"]
    state = d["optim_state"]
    ai = state["active_importance_sampling"]
    acq = AcqFcnVIQR()
    with np.load(out / str(seed) / "candidates.npz") as archive:
        candidates = archive["X"]
    points = np.vstack([candidates, ai["X"]])
    n = len(gp.X)
    m = len(points)
    sn = acq._estimate_observation_noise(candidates, gp, state)
    double = acq(candidates, gp, d["vp"], d["function_logger"], state)
    all_after = []
    all_before = []
    var_errors = []
    clipped = 0
    started = time.perf_counter()
    for s, p in enumerate(gp.posteriors):
        ell = [mp.exp(float(x)) for x in p.hyp[: gp.D]]
        sf2 = mp.exp(2 * mp.mpf(float(p.hyp[gp.D])))
        train = [
            [mp.mpf(float(x)) / e for x, e in zip(row, ell)] for row in gp.X
        ]
        test = [
            [mp.mpf(float(x)) / e for x, e in zip(row, ell)] for row in points
        ]

        def kernel(x, y):
            return sf2 * mp.exp(
                -mp.fsum((a - b) ** 2 for a, b in zip(x, y)) / 2
            )

        cn = gp.covariance.hyperparameter_count(gp.D)
        nn = gp.noise.hyperparameter_count()
        noise = (
            np.broadcast_to(
                np.asarray(
                    gp.noise.compute(p.hyp[cn : cn + nn], gp.X, gp.y, gp.s2)
                ).reshape(-1),
                (n,),
            )
            * p.sn2_mult
        )
        A = mp.matrix(n)
        for i in range(n):
            for j in range(i + 1):
                A[i, j] = A[j, i] = kernel(train[i], train[j])
            A[i, i] += mp.mpf(float(noise[i]))
        L = mp.cholesky(A)
        # Solve L V = K(X, points); each column gives a posterior variance.
        V = []
        for x in test:
            b = [kernel(t, x) for t in train]
            v = []
            for i in range(n):
                v.append((b[i] - mp.fdot(L[i, :i], v)) / L[i, i])
            V.append(v)
        variance = [sf2 - mp.fdot(v, v) for v in V]
        var_errors.append(
            max(
                abs(float(v) - float(ref))
                for v, ref in zip(variance[3:], ai["f_s2"][:, s])
            )
        )
        before = mp.fsum(
            2 * mp.sinh(mp.mpf(float(acq.u)) * mp.sqrt(v))
            for v in variance[3:]
        )
        all_before.append(before)
        after = []
        for j in range(3):
            terms = []
            for a in range(3, m):
                cov = kernel(test[j], test[a]) - mp.fdot(V[j], V[a])
                residual = variance[a] - cov**2 / (
                    variance[j] + mp.mpf(float(sn[j]))
                )
                if residual < 0:
                    clipped += 1
                terms.append(
                    2
                    * mp.sinh(mp.mpf(float(acq.u)) * mp.sqrt(max(residual, 0)))
                )
            after.append(mp.fsum(terms))
        all_after.append(after)
        print(
            f"seed {seed} posterior {s+1}/{len(gp.posteriors)}: {time.perf_counter()-started:.1f}s",
            flush=True,
        )
    scores = [
        float(mp.log(mp.fsum(row[j] for row in all_after) / len(all_after)))
        for j in range(3)
    ]
    reductions = [
        float(1 - mp.fsum(row[j] for row in all_after) / mp.fsum(all_before))
        for j in range(3)
    ]
    result = {
        "seed": seed,
        "call": call,
        "dps": mp.mp.dps,
        "method": "SE kernel rebuilt from exact float64 coordinates and hyperparameters; training noise and candidate noise held at float64 values; fresh arbitrary-precision Cholesky and posterior covariance; same importance points",
        "candidate_names": ["selected", "sieve", "best_central_sieve"],
        "score_float64": double.tolist(),
        "score_mp": scores,
        "score_max_abs_error": max(abs(double - np.array(scores))),
        "fraction_iqr_reduction_mp": reductions,
        "importance_variance_max_abs_error": max(var_errors),
        "clipped_variances_mp": clipped,
        "wall_s": time.perf_counter() - started,
    }
    (out / str(seed) / "precision.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)


def refit(args):
    """Separate added-input conditioning from the following refit."""
    out = args.out

    def measure(gp):
        hyps = gp.get_hyperparameters(as_array=True)
        cn = gp.covariance.hyperparameter_count(gp.D)
        nn = gp.noise.hyperparameter_count()
        noise = np.array(
            [
                gp.noise.compute(h[cn : cn + nn], gp.X, gp.y, gp.s2)
                for h in hyps
            ]
        )
        return {
            "N": len(gp.y),
            "y_min": gp.y.min(),
            "y_max": gp.y.max(),
            "condition": gp_condition_number(gp),
            "sf2_range": [
                np.exp(2 * hyps[:, gp.D]).min(),
                np.exp(2 * hyps[:, gp.D]).max(),
            ],
            "ell_min": np.exp(hyps[:, : gp.D]).min(axis=0),
            "ell_max": np.exp(hyps[:, : gp.D]).max(axis=0),
            "noise_range": [noise.min(), noise.max()],
            "sn2_mult": [p.sn2_mult for p in gp.posteriors],
        }

    rows = []
    for seed, i in [(1000, 15), (1001, 25), (1002, 18)]:
        tag = f"rosenbrock_D2_noise3_svbmc_seed{seed}"
        vbmc = cloudpickle.loads((args.runs / f"{tag}.vbmc.pkl").read_bytes())
        pre = vbmc.get_gp(i - 1)
        post = vbmc.get_gp(i)
        fixed = copy.deepcopy(post)
        fixed.set_hyperparameters(pre.get_hyperparameters(as_array=True))
        row = {
            "seed": seed,
            "iteration": i,
            "previous_iteration": measure(pre),
            "new_data_old_hyperparameters": measure(fixed),
            "next_iteration": measure(post),
            "history": [
                {
                    "iteration": j,
                    "calls": int(vbmc.iteration_history["func_count"][j]),
                    **measure(vbmc.get_gp(j)),
                }
                for j in range(len(vbmc.iteration_history["gp"]))
            ],
        }
        rows.append(jsonable(row))
        print(
            json.dumps(
                jsonable({k: v for k, v in row.items() if k != "history"})
            ),
            flush=True,
        )
    (out / "refit.json").write_text(json.dumps(rows, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=["replay", "analyze", "refit", "precision"]
    )
    parser.add_argument(
        "--runs",
        type=Path,
        required=True,
        help="Corrected gp_box_probe output directory.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpyreg-source", type=Path, required=True)
    parser.add_argument(
        "--mpmath-source",
        type=Path,
        help="Optional directory containing the mpmath package.",
    )
    parser.add_argument("--seed", type=int, choices=[1000, 1001, 1002])
    args = parser.parse_args()
    if args.mode in ("replay", "precision") and args.seed is None:
        parser.error("--seed is required for replay and precision")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    sys.path[:0] = [str(ROOT), str(args.gpyreg_source.resolve())]
    if args.mpmath_source is not None:
        sys.path.insert(0, str(args.mpmath_source.resolve()))
    global cloudpickle, np, VBMC, AcqFcnVIQR, find_config
    global gp_condition_number, jsonable
    import cloudpickle
    import gpyreg
    import numpy as np
    from benchmark_targets import find_config
    from profile_run import jsonable
    from svbmc_pool_io import gp_condition_number

    import pyvbmc
    from pyvbmc import VBMC
    from pyvbmc.acquisition_functions import AcqFcnVIQR

    for module, expected in [
        (pyvbmc, ROOT / "pyvbmc"),
        (gpyreg, args.gpyreg_source.resolve() / "gpyreg"),
    ]:
        if Path(module.__file__).resolve().parent != expected:
            raise RuntimeError(f"Wrong import path for {module.__name__}")
    if args.mode in ("replay", "precision"):
        globals()[args.mode](args, args.seed)
    else:
        globals()[args.mode](args)


if __name__ == "__main__":
    main()
