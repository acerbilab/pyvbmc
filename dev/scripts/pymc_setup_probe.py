"""Preapproval experiments for the PyMC target adapter.

Part A compares a strictly capped L-BFGS-B search with a 2000-call reference
using the adapter-coordinate log density, joint value/gradient compilation,
and an exact Hessian. It records every trial (including rejected line-search
trials), accepted iterates, stopping rules, prior location flags, and boxes.
The historical ``pymc_feasibility.py`` supplies only models and coordinate
mapping; its find_MAP and finite-difference setup are never called.

One value/gradient call costs one setup unit; one exact Hessian costs D
units. Hessians do not supply reusable log-density observations. Nonfinite
trials count against the cap but cannot be handed to FunctionLogger.
Prior sampling, compilation, and reference calculations are timed separately
and do not count as setup density evaluations. All numerical jobs run serially.

Run in the PyMC environment listed in ``dev/scripts/runs/LOCAL.md``::

    python dev/scripts/pymc_setup_probe.py a --out DIR
    python dev/scripts/pymc_setup_probe.py references --out REF_DIR
    python dev/scripts/pymc_setup_probe.py b --setup DIR --references REF_DIR --out B_DIR
    python dev/scripts/pymc_setup_probe.py check --setup DIR --out CHECK_DIR
    python dev/scripts/pymc_setup_probe.py summarize --setup DIR --references REF_DIR --results B_DIR --out TRACKED_DIR
    python dev/scripts/pymc_setup_probe.py coverage --setup DIR --references REF_DIR --out COVERAGE_DIR
    python dev/scripts/pymc_setup_probe.py coverage_summary --results COVERAGE_DIR --baseline B_DIR --out TRACKED_COVERAGE_DIR

Part B compares discard, in-box f_vals with a shortened initial design,
and all finite unique setup points in the logger with the ordinary initial
design. Total budgets are 100 units (vector and ill-conditioned regression)
and 150 (non-centered eight-schools). Setup is identical across the five
paired seeds. A wrapper shortens the final acquisition batch and disables
early termination until the budget is spent; other inference options retain
their defaults. Logger seeding and batch limiting are local to this process.

The focused ``coverage`` follow-up keeps the logger interface and single
starting point for both filtered arms. Its default filter retains points
within 10D nats of the best, with a D+1 minimum, following the normal warmup
pruning rule. ``--filter box`` reproduces the initial geometric comparison.
It varies only the number of uniform initial draws (full or shortened), on
the two hard models. Comparing its filtered/full arm with Part B's
all-points/full arm measures point retention.
The existing f_vals arm also changes the initial variational component means
through its multiple starting rows, so it cannot isolate coverage by itself.

JSON summaries and source hashes are suitable for tracking; detailed search
paths and posterior draws live under the gitignored raw output directory.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
import traceback
from pathlib import Path

# Set before NumPy, PyMC, or BLAS is imported, including in child processes.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import pymc_feasibility as historical
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
MODELS = (
    "scalar",
    "positive",
    "vector",
    "bounded",
    "one_sided",
    "schools_centered",
    "schools_noncentered",
    "ill_conditioned",
    "logistic20",
    "varying20",
    "no_gradient",
)


def dump(path, value):
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(type(obj).__name__)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, default=convert) + "\n", encoding="utf-8"
    )


def sources():
    import importlib.metadata as metadata

    import gpyreg
    import pytensor

    def git(path, *args):
        return subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={Path(path).as_posix()}",
                "-C",
                str(path),
                *args,
            ],
            text=True,
        ).strip()

    paths = [
        Path(__file__),
        Path(historical.__file__),
        ROOT / "pyvbmc/vbmc/vbmc.py",
        ROOT / "pyvbmc/vbmc/active_sample.py",
        ROOT / "pyvbmc/function_logger/function_logger.py",
    ]
    gp_root = Path(gpyreg.__file__).resolve().parents[1]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            n: metadata.version(n)
            for n in ("pymc", "pytensor", "arviz", "numpy", "scipy", "gpyreg")
        },
        "pyvbmc_commit": git(ROOT, "rev-parse", "HEAD"),
        "gpyreg_commit": git(gp_root, "rev-parse", "HEAD"),
        "gpyreg_dirty": git(gp_root, "status", "--porcelain"),
        "linker": type(
            pytensor.compile.mode.get_default_mode().linker
        ).__name__,
        "floatX": pytensor.config.floatX,
        "blas_thread_environment": {
            n: os.environ[n]
            for n in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
        "sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in paths
        },
    }


def build_model(name):
    import pymc as pm

    index = MODELS.index(name)
    rng = np.random.default_rng([0, 0, index])
    if index < 5:
        return getattr(historical, "model_" + name)(rng)
    if name.startswith("schools_"):
        y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
        sd = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
        with pm.Model() as model:
            mu = pm.Normal("mu", 0.0, 5.0)
            tau = pm.HalfCauchy("tau", 5.0)
            if name == "schools_centered":
                theta = pm.Normal("theta", mu, tau, shape=8)
            else:
                z = pm.Normal("z", 0.0, 1.0, shape=8)
                theta = pm.Deterministic("theta", mu + tau * z)
            pm.Normal("y", theta, sd, observed=y)
        return {"name": name, "model": model}
    if name == "ill_conditioned":
        n = 80
        X = np.column_stack(
            [
                np.ones(n),
                rng.normal(2.0, 0.3, n),
                rng.normal(20.0, 3.0, n),
                rng.normal(200.0, 30.0, n),
                rng.normal(0.02, 0.003, n),
            ]
        )
        beta_true = np.array([0.5, -0.8, 0.05, 0.005, 2.0])
        y = X @ beta_true + rng.normal(0.0, 0.7, n)
        with pm.Model() as model:
            beta = pm.Normal("beta", 0.0, 5.0, shape=5)
            pm.Normal("y", X @ beta, 0.7, observed=y)
        precision = np.eye(5) / 25 + X.T @ X / 0.7**2
        cov = np.linalg.inv(precision)
        return {
            "name": name,
            "model": model,
            "ln_Z": historical._mvn_logpdf(
                y, 0.7**2 * np.eye(n) + 25 * X @ X.T
            ),
            "truth_mean": cov @ X.T @ y / 0.7**2,
            "truth_cov": cov,
            "X": X,
            "y": y,
        }
    if name == "logistic20":
        X = rng.normal(size=(200, 20))
        X = np.sqrt(0.85) * rng.normal(size=(200, 1)) + np.sqrt(0.15) * X
        beta_true = rng.normal(0, 0.3, 20)
        from scipy.special import expit

        y = rng.binomial(1, expit(X @ beta_true))
        with pm.Model() as model:
            beta = pm.Normal("beta", 0.0, 2.0, shape=20)
            pm.Bernoulli("y", logit_p=X @ beta, observed=y)
    elif name == "varying20":
        groups = np.repeat(np.arange(18), 10)
        y = 1 + rng.normal(0, 0.6, 18)[groups] + rng.normal(0, 1, len(groups))
        with pm.Model() as model:
            mu = pm.Normal("mu", 0.0, 5.0)
            tau = pm.HalfNormal("tau", 2.0)
            z = pm.Normal("z", 0.0, 1.0, shape=18)
            pm.Normal("y", mu + tau * z[groups], 1.0, observed=y)
    elif name == "no_gradient":
        import pytensor.tensor as pt
        from pytensor.compile.ops import as_op

        @as_op(itypes=[pt.dscalar], otypes=[pt.dscalar])
        def likelihood(x):
            return np.asarray(-0.5 * ((x - 1.2) / 0.3) ** 2)

        with pm.Model() as model:
            x = pm.Normal("x", 0.0, 3.0)
            pm.Potential("likelihood", likelihood(x))
    else:
        raise ValueError(name)
    return {"name": name, "model": model}


def search_bounds(target):
    bounds = []
    for lo, hi in zip(target.lb.ravel(), target.ub.ravel()):
        if np.isfinite(lo) and np.isfinite(hi):
            margin = 0.002 * (hi - lo)
            bounds.append((lo + margin, hi - margin))
        else:
            bounds.append((None, None))
    return bounds


def compile_derivatives(target):
    partial = target.partial
    rvs = [
        next(rv for rv in partial.free_RVs if rv.name == n)
        for n in target.names
    ]
    start = time.perf_counter()
    fg = partial.compile_fn(
        [partial.logp(jacobian=True), partial.dlogp(vars=rvs, jacobian=True)],
        inputs=partial.value_vars,
        on_unused_input="ignore",
    )
    gradient_time = time.perf_counter() - start
    start = time.perf_counter()
    hessian = partial.compile_d2logp(
        vars=rvs, jacobian=True, negate_output=False
    )
    return (
        fg,
        hessian,
        {
            "gradient_compile_s": gradient_time,
            "hessian_compile_s": time.perf_counter() - start,
        },
    )


class SearchStop(Exception):
    pass


def search(target, fg, cap=2000, ftol=1e-14, gtol=1e-8, atol=None):
    calls, accepted = [], []
    bounds = search_bounds(target)
    xstart = target.x0.ravel().copy()
    for i, (lo, hi) in enumerate(bounds):
        if lo is not None:
            xstart[i] = np.clip(xstart[i], lo, hi)

    def objective(x):
        if len(calls) >= cap:
            raise SearchStop("cap")
        start = time.perf_counter()
        value, gradient = fg(target.unflatten(x))
        value, gradient = float(value), np.asarray(gradient)
        calls.append(
            {
                "x": x.copy(),
                "value": value,
                "gradient": gradient.copy(),
                "seconds": time.perf_counter() - start,
            }
        )
        return -value, -gradient

    def callback(x):
        row = next(r for r in reversed(calls) if np.array_equal(r["x"], x))
        accepted.append(
            {"call": len(calls), "value": row["value"], "x": x.copy()}
        )
        if atol is not None and len(accepted) > 1:
            delta = accepted[-1]["value"] - accepted[-2]["value"]
            if 0 <= delta <= atol:
                raise SearchStop("absolute_improvement")

    start = time.perf_counter()
    try:
        result = minimize(
            objective,
            xstart,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            callback=callback,
            options={
                "ftol": ftol,
                "gtol": gtol,
                "maxfun": cap,
                "maxiter": cap,
                "maxls": 40,
            },
        )
        message, success = str(result.message), bool(result.success)
    except SearchStop as exc:
        message, success = str(exc), False
    finite = [r for r in calls if np.isfinite(r["value"])]
    best = max(finite, key=lambda r: r["value"])
    return {
        "x": best["x"],
        "value": best["value"],
        "calls": calls,
        "accepted": accepted,
        "message": message,
        "success": success,
        "seconds": time.perf_counter() - start,
        "n_calls": len(calls),
        "nonfinite": len(calls) - len(finite),
    }


def curvature(target, hessian, x):
    start = time.perf_counter()
    H = np.asarray(hessian(target.unflatten(x)))
    precision = -(H + H.T) / 2
    eig = np.linalg.eigvalsh(precision)
    # Positive diagonal entries of an indefinite inverse are not valid
    # marginal variances. Require the whole precision to be positive definite.
    try:
        np.linalg.cholesky(precision)
        cov = np.linalg.inv(precision)
        sd = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        sd = np.full(target.D, np.nan)
    return {
        "precision": precision,
        "sd": sd,
        "eigenvalues": eig,
        "positive_definite": bool(np.all(eig > 0)),
        "seconds": time.perf_counter() - start,
    }


def prior_draws(target, seed=0):
    start = time.perf_counter()
    draws = target.pm.draw(
        target.model.free_RVs,
        draws=4000,
        random_seed=np.random.default_rng([seed, 781]),
    )
    original = dict(zip(target.names, draws))
    values = target.to_values(original, draws_axis=True)
    flat = np.column_stack(
        [values[v].reshape(4000, -1) for v in target.value_names]
    )
    return flat, time.perf_counter() - start


def distance(x, ref, ref_curv):
    delta = np.asarray(x) - ref["x"]
    if not np.all(np.isfinite(ref_curv["sd"])):
        return {"mahalanobis": None, "max_marginal_shift": None}
    return {
        "mahalanobis": float(
            np.sqrt(max(0.0, delta @ ref_curv["precision"] @ delta))
        ),
        "max_marginal_shift": float(np.max(np.abs(delta) / ref_curv["sd"])),
    }


def make_setup(target, run, hessian, prior, q=0.005):
    x = run["x"].copy()
    curv = curvature(target, hessian, x)
    prior_lo, prior_hi = np.quantile(prior, [0.05, 0.95], axis=0)
    location_lo, location_hi = np.quantile(prior, [q, 1 - q], axis=0)
    relocated = (x < location_lo) | (x > location_hi)
    usable = np.isfinite(curv["sd"]) & (curv["sd"] > 0)
    sd = np.where(usable, curv["sd"], (prior_hi - prior_lo) / 6)
    sd[relocated] = ((prior_hi - prior_lo) / 6)[relocated]
    x[relocated] = ((prior_lo + prior_hi) / 2)[relocated]
    plb, pub, clipped = target._clip(x - 3 * sd, x + 3 * sd)
    plb, pub = plb.ravel(), pub.ravel()
    for i, (lo, hi) in enumerate(zip(target.lb.ravel(), target.ub.ravel())):
        if plb[i] >= x[i]:
            plb[i] = (lo + x[i]) / 2
        if pub[i] <= x[i]:
            pub[i] = (hi + x[i]) / 2
    rows = [r for r in run["calls"] if np.isfinite(r["value"])]
    extra = 0
    if np.any(relocated):
        value = target.log_joint(x)
        if not np.isfinite(value):
            raise ValueError("Relocated point has nonfinite density")
        rows = rows + [{"x": x.copy(), "value": value}]
        extra = 1
    return {
        "x0": x,
        "plb": plb,
        "pub": pub,
        "X": np.array([r["x"] for r in rows]),
        "y": np.array([r["value"] for r in rows]),
        "cost": run["n_calls"] + target.D + extra,
        "search_calls": run["n_calls"],
        "hessian_units": target.D,
        "finiteness_calls": extra,
        "cap_reached": run["message"] == "cap",
        "relocated": np.flatnonzero(relocated),
        "curvature_fallback": np.flatnonzero(~usable),
        "clipped": np.flatnonzero(clipped),
        "q": q,
    }


def part_a(args):
    report = {"sources": sources(), "models": []}
    for name in args.models.split(",") if args.models else MODELS:
        print("A START", name, flush=True)
        spec = build_model(name)
        target = historical.PyMCTarget(spec["model"])
        entry = {"name": name, "D": target.D}
        prior, prior_time = prior_draws(target)
        entry["prior_seconds"] = prior_time
        try:
            fg, hessian, timings = compile_derivatives(target)
        except (NotImplementedError, ValueError) as exc:
            if name != "no_gradient":
                raise
            entry.update(
                {
                    "gradient_error": f"{type(exc).__name__}: {exc}",
                    "finiteness_value": target.log_joint(target.x0),
                    "setup_cost": 1,
                    "prior_box": np.quantile(prior, [0.05, 0.95], axis=0),
                }
            )
            report["models"].append(entry)
            dump(args.out / "part_a.json", report)
            print("A DONE", name, "no gradient", flush=True)
            continue
        entry.update(timings)
        reference = search(target, fg)
        ref_curv = curvature(target, hessian, reference["x"])
        cap = 19 + 4 * target.D
        capped = search(target, fg, cap=cap)
        capped_curv = curvature(target, hessian, capped["x"])
        entry["reference"] = {
            k: v
            for k, v in reference.items()
            if k not in ("calls", "accepted")
        }
        entry["reference_curvature"] = ref_curv
        entry["capped"] = {
            k: v for k, v in capped.items() if k not in ("calls", "accepted")
        }
        entry["capped"].update(distance(capped["x"], reference, ref_curv))
        entry["capped"]["logp_gap"] = reference["value"] - capped["value"]
        entry["capped"]["sd_ratio"] = capped_curv["sd"] / ref_curv["sd"]
        best = -np.inf
        hits = {
            "gap_1": None,
            "gap_0.1": None,
            "mahal_1": None,
            "mahal_0.3": None,
        }
        for i, row in enumerate(reference["calls"], 1):
            if row["value"] > best:
                best, best_x = row["value"], row["x"]
            dist = distance(best_x, reference, ref_curv)["mahalanobis"]
            for key, passes in (
                ("gap_1", reference["value"] - best <= 1),
                ("gap_0.1", reference["value"] - best <= 0.1),
                ("mahal_1", dist is not None and dist <= 1),
                ("mahal_0.3", dist is not None and dist <= 0.3),
            ):
                if passes and hits[key] is None:
                    hits[key] = i
        entry["first_hit"] = hits
        entry["location"] = {}
        for q in (0.005, 0.01, 0.05):
            lo, hi = np.quantile(prior, [q, 1 - q], axis=0)
            entry["location"][str(q)] = {
                label: np.flatnonzero((run["x"] < lo) | (run["x"] > hi))
                for label, run in (
                    ("capped", capped),
                    ("reference", reference),
                )
            }
        entry["rules"] = []
        paths = {"reference": reference, "capped": capped}
        candidates = [("relative", v) for v in (1e-3, 1e-5, 1e-7, 1e-9)]
        candidates += [("absolute", v) for v in (0.1, 0.01, 0.001)]
        candidates += [("gradient", v) for v in (1.0, 0.1, 0.01)]
        for kind, tol in candidates:
            options = (
                {"ftol": tol}
                if kind == "relative"
                else ({"atol": tol} if kind == "absolute" else {"gtol": tol})
            )
            run = search(target, fg, **options)
            label = f"{kind}_{tol}"
            paths[label] = run
            entry["rules"].append(
                {
                    "rule": label,
                    "calls": run["n_calls"],
                    "logp_gap": reference["value"] - run["value"],
                    **distance(run["x"], reference, ref_curv),
                }
            )
        selected = search(target, fg, cap=cap, atol=0.001)
        paths["selected_capped_absolute_0.001"] = selected
        setup = make_setup(target, selected, hessian, prior)
        entry["selected"] = {
            "n_calls": selected["n_calls"],
            "message": selected["message"],
            "setup_cost": setup["cost"],
            "relocated": setup["relocated"],
            "logp_gap": reference["value"] - selected["value"],
            **distance(selected["x"], reference, ref_curv),
            "sd_ratio": curvature(target, hessian, selected["x"])["sd"]
            / ref_curv["sd"],
        }
        assert setup["cost"] <= 20 + 5 * target.D
        dump(args.out / "setups" / f"{name}.json", setup)
        dump(args.out / "paths" / f"{name}.json", paths)
        report["models"].append(entry)
        dump(args.out / "part_a.json", report)
        print(
            "A DONE",
            name,
            "calls",
            capped["n_calls"],
            "/",
            reference["n_calls"],
            "shift",
            entry["capped"]["max_marginal_shift"],
            flush=True,
        )
    return report


REUSE_MODELS = ("vector", "schools_noncentered", "ill_conditioned")
ARMS = ("discard", "f_vals", "logger")


def posterior_metrics(draws, reference):
    """Moment and marginal distances, in the adapter's flat coordinates."""
    from scipy.linalg import solve_triangular

    mean, ref_mean = draws.mean(0), reference.mean(0)
    cov, ref_cov = np.atleast_2d(np.cov(draws.T)), np.atleast_2d(
        np.cov(reference.T)
    )
    sd = np.sqrt(np.diag(ref_cov))
    L = np.linalg.cholesky(ref_cov)
    delta = solve_triangular(L, mean - ref_mean, lower=True)
    whitened = solve_triangular(L, cov, lower=True)
    whitened = solve_triangular(L, whitened.T, lower=True).T
    D = draws.shape[1]
    symkl = 0.25 * (
        np.trace(np.linalg.solve(ref_cov, cov))
        + np.trace(np.linalg.solve(cov, ref_cov))
        - 2 * D
        + (mean - ref_mean)
        @ (
            np.linalg.solve(cov, mean - ref_mean)
            + np.linalg.solve(ref_cov, mean - ref_mean)
        )
    )
    quantiles = np.linspace(0.01, 0.99, 99)
    w1 = (
        np.mean(
            np.abs(
                np.quantile(draws, quantiles, axis=0)
                - np.quantile(reference, quantiles, axis=0)
            ),
            axis=0,
        )
        / sd
    )
    return {
        "mean_mahalanobis": float(np.linalg.norm(delta)),
        "max_mean_sd": float(np.max(np.abs(mean - ref_mean) / sd)),
        "covariance_relative_frobenius": float(
            np.linalg.norm(whitened - np.eye(D)) / np.sqrt(D)
        ),
        "gaussian_symmetric_kl": float(symkl),
        "max_marginal_w1_sd": float(w1.max()),
        "mean_marginal_w1_sd": float(w1.mean()),
    }


def references(args):
    import pymc as pm
    from arviz_stats import ess, rhat

    manifest = {"sources": sources(), "models": []}
    for name in args.models.split(",") if args.models else REUSE_MODELS:
        print("NUTS START", name, flush=True)
        spec = build_model(name)
        target = historical.PyMCTarget(spec["model"])
        start = time.perf_counter()
        with spec["model"]:
            data = pm.sample(
                draws=args.draws,
                tune=args.tune,
                chains=4,
                cores=1,
                blas_cores=1,
                random_seed=7341,
                target_accept=0.95,
                init="jitter+adapt_full",
                progressbar=False,
            )
        elapsed = time.perf_counter() - start
        ds = data["posterior"].dataset
        original = {
            n: np.asarray(ds[n]).reshape((-1,) + s)
            for n, s in zip(target.names, target.shapes)
        }
        values = target.to_values(original, draws_axis=True)
        flat = np.column_stack(
            [values[v].reshape(4 * args.draws, -1) for v in target.value_names]
        )
        chains = flat.reshape(4, args.draws, target.D)
        import xarray as xr

        diagnostic = xr.Dataset(
            {"value": (("chain", "draw", "parameter"), chains)}
        )
        rh = np.asarray(rhat(diagnostic)["value"])
        bulk = np.asarray(ess(diagnostic, method="bulk")["value"])
        tail = np.asarray(ess(diagnostic, method="tail")["value"])
        stats = data["sample_stats"].dataset
        divergences = int(stats["diverging"].sum())
        entry = {
            "name": name,
            "draws_per_chain": args.draws,
            "tune_per_chain": args.tune,
            "chains": 4,
            "seconds": elapsed,
            "rhat_max": float(rh.max()),
            "ess_bulk_min": float(bulk.min()),
            "ess_tail_min": float(tail.min()),
            "divergences": divergences,
            "tree_depth_max": int(stats["tree_depth"].max()),
            "accepted": bool(
                rh.max() < 1.01
                and bulk.min() >= 1000
                and tail.min() >= 1000
                and divergences == 0
            ),
            "split_chain_distance": posterior_metrics(
                chains[:2].reshape(-1, target.D),
                chains[2:].reshape(-1, target.D),
            ),
        }
        if "truth_mean" in spec:
            entry["analytic_max_mean_sd"] = float(
                np.max(
                    np.abs(flat.mean(0) - spec["truth_mean"])
                    / np.sqrt(np.diag(spec["truth_cov"]))
                )
            )
            entry["analytic_sd_ratio"] = flat.std(0) / np.sqrt(
                np.diag(spec["truth_cov"])
            )
        args.out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.out / f"{name}.npz",
            draws=chains,
            rhat=rh,
            ess_bulk=bulk,
            ess_tail=tail,
        )
        dump(args.out / f"{name}.json", entry)
        manifest["models"].append(entry)
        dump(args.out / "references.json", manifest)
        print("NUTS DONE", name, entry, flush=True)


def reuse_case(
    name, seed, arm, setup, reference, budget, out, uniform_initial=None
):
    import importlib
    from unittest.mock import patch

    from pyvbmc import VBMC

    module = importlib.import_module("pyvbmc.vbmc.vbmc")
    spec = build_model(name)
    target = historical.PyMCTarget(spec["model"])
    X, y = np.asarray(setup["X"]), np.asarray(setup["y"])
    # Exact duplicate trials carry no new information for a deterministic target.
    _, unique = np.unique(X, axis=0, return_index=True)
    unique.sort()
    X, y = X[unique], y[unique]
    x0, plb, pub = [
        np.asarray(setup[k]).reshape(1, -1) for k in ("x0", "plb", "pub")
    ]
    remaining = budget - setup["cost"]
    options = {
        "display": "off",
        "max_fun_evals": remaining,
        "min_fun_evals": remaining,
        "min_iter": 0,
    }
    if uniform_initial is not None:
        if arm != "logger" or uniform_initial < 0:
            raise ValueError("Uniform design override requires logger seeding")
        options["fun_eval_start"] = 1 + uniform_initial
    selected = np.arange(len(X))
    if arm == "f_vals":
        selected = np.flatnonzero(np.all((X > plb) & (X < pub), axis=1))
        selected = selected[np.argsort(-y[selected], kind="stable")]
        if not len(selected):
            raise ValueError("No setup point strictly inside the box")
        initial = X[selected]
        options.update(
            f_vals=y[selected], fun_eval_start=max(target.D, 10, len(selected))
        )
    else:
        initial = x0
        if arm == "logger":
            best_row = np.flatnonzero(np.all(X == x0, axis=1))
            if len(best_row) != 1:
                raise ValueError(
                    "Starting point missing from setup observations"
                )
            options["f_vals"] = y[best_row]
        else:
            selected = np.array([], dtype=int)
    calls = []

    def density(x):
        value = target.log_joint(x)
        calls.append((np.asarray(x).copy(), value))
        return value

    vbmc = VBMC(
        density,
        initial,
        target.lb,
        target.ub,
        plb,
        pub,
        options=options,
        seed=seed,
    )
    assert np.array_equal(vbmc.plausible_lower_bounds, plb)
    assert np.array_equal(vbmc.plausible_upper_bounds, pub)
    if arm == "logger":
        for x, value in zip(X, y):
            # The initial-design cache inserts x0 once. Seeding it here as
            # well would increment n_evals twice for one stored observation.
            if np.array_equal(x, x0.ravel()):
                continue
            vbmc.function_logger.add(
                vbmc.parameter_transformer(x), float(value)
            )
    original_active = module.active_sample
    initial_audit = {}

    def limited_active(
        gp, sample_count, optim_state, logger, history, vp, opts
    ):
        # VBMC checks max_fun_evals between batches. Shorten only the last
        # acquisition batch so every successful arm spends exactly its budget.
        if gp is not None:
            sample_count = min(sample_count, remaining - logger.func_count)
        result = original_active(
            gp, sample_count, optim_state, logger, history, vp, opts
        )
        if gp is None:
            assert np.all(logger.n_evals[logger.X_flag] == 1)
            initial_audit.update(
                rows=int(logger.Xn + 1),
                new_calls=logger.func_count,
                cached_calls=logger.cache_count,
                value_range=[
                    float(np.min(logger.y_orig[logger.X_flag])),
                    float(np.max(logger.y_orig[logger.X_flag])),
                ],
            )
        return result

    entry = {
        "name": name,
        "seed": seed,
        "arm": arm,
        "total_budget": budget,
        "setup_cost": setup["cost"],
        "setup_unique_finite_rows": len(X),
        "reused_rows": len(selected),
        "setup_logp_range": [float(y.min()), float(y.max())],
        "x0": x0,
        "plb": plb,
        "pub": pub,
    }
    start = time.perf_counter()
    try:
        # Existing cached-evaluation runs omit the cache-count argument in
        # the final display row. Formatting executes even with display off.
        # Empty formatting suppresses only that presentation defect, equally
        # in all arms; inference, final boost, and RNG use are untouched.
        with (
            patch.object(module, "active_sample", limited_active),
            patch.object(
                VBMC, "_setup_logging_display_format", return_value=""
            ),
        ):
            vp, results = vbmc.optimize()
        assert len(calls) == remaining == vbmc.function_logger.func_count
        draws, _ = vp.sample(20000, orig_flag=True)
        entry.update(
            {
                "elbo": results["elbo"],
                "elbo_sd": results["elbo_sd"],
                "elbo_error": (
                    results["elbo"] - spec["ln_Z"] if "ln_Z" in spec else None
                ),
                "success_flag": results["success_flag"],
                "message": results["message"],
                "metrics": posterior_metrics(draws, reference),
            }
        )
        np.savez_compressed(out / "posterior.npz", draws=draws)
    except Exception as exc:
        entry["error"] = f"{type(exc).__name__}: {exc}"
        (out / "traceback.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
    entry.update(
        {
            "seconds": time.perf_counter() - start,
            "new_calls": len(calls),
            "total_spent": setup["cost"] + len(calls),
            "initial_design": initial_audit,
            "iterations": vbmc.iteration + 1,
        }
    )
    stable = vbmc.iteration_history.get("stable")
    entry["first_stable_iteration"] = (
        next((i + 1 for i, v in enumerate(stable) if v), None)
        if stable is not None
        else None
    )
    np.savez_compressed(
        out / "calls.npz",
        X=np.array([r[0] for r in calls]),
        y=np.array([r[1] for r in calls]),
    )
    dump(out / "result.json", entry)
    return entry


def part_b(args):
    report = {"sources": sources(), "cases": []}
    for name in args.models.split(",") if args.models else REUSE_MODELS:
        setup = json.loads(
            (args.setup / "setups" / f"{name}.json").read_text()
        )
        ref_info = json.loads((args.references / f"{name}.json").read_text())
        if not ref_info["accepted"]:
            raise ValueError(
                f"NUTS reference for {name} failed its diagnostics"
            )
        with np.load(args.references / f"{name}.npz") as arrays:
            reference = arrays["draws"].reshape(-1, len(setup["x0"]))
        budget = 150 if name == "schools_noncentered" else 100
        for seed in range(args.seeds):
            # Rotate arm order to reduce systematic timing-order effects.
            arms = ARMS[seed % 3 :] + ARMS[: seed % 3]
            for arm in arms:
                out = args.out / "cells" / f"{name}_{seed}_{arm}"
                out.mkdir(parents=True, exist_ok=True)
                print("B START", name, seed, arm, flush=True)
                if (out / "result.json").exists():
                    entry = json.loads((out / "result.json").read_text())
                else:
                    entry = reuse_case(
                        name, seed, arm, setup, reference, budget, out
                    )
                    entry["execution_source_sha256"] = hashlib.sha256(
                        (args.out / "probe_source.py").read_bytes()
                    ).hexdigest()
                    dump(out / "result.json", entry)
                report["cases"].append(entry)
                dump(args.out / "part_b.json", report)
                print(
                    "B DONE",
                    name,
                    seed,
                    arm,
                    entry.get("error", entry.get("metrics")),
                    flush=True,
                )
    return report


def coverage(args):
    """Separate observation filtering from uniform initial-design coverage."""
    report = {
        "sources": sources(),
        "filter": args.filter,
        "filter_audit": [],
        "cases": [],
    }
    names = (
        args.models.split(",")
        if args.models
        else ("schools_noncentered", "ill_conditioned")
    )
    for name in names:
        original = json.loads(
            (args.setup / "setups" / f"{name}.json").read_text()
        )
        X, y = np.asarray(original["X"]), np.asarray(original["y"])
        _, unique = np.unique(X, axis=0, return_index=True)
        unique.sort()
        X, y = X[unique], y[unique]
        inside = np.all((X > original["plb"]) & (X < original["pub"]), axis=1)
        D = len(original["x0"])
        gaps = np.max(y) - y
        density_keep = gaps < 10 * D
        if np.sum(density_keep) < min(D + 1, len(y)):
            density_keep[np.argsort(-y, kind="stable")[: D + 1]] = True
        keep = density_keep if args.filter == "log_density" else inside
        report["filter_audit"].append(
            {
                "name": name,
                "D": D,
                "all_rows": len(X),
                "density_threshold": 10 * D,
                "density_rows": int(density_keep.sum()),
                "box_rows": int(inside.sum()),
                "same_masks": bool(np.array_equal(inside, density_keep)),
                "relative_log_densities": gaps,
                "box_mask": inside,
                "density_mask": density_keep,
            }
        )
        selected = {**original, "X": X[keep], "y": y[keep]}
        assert np.any(np.all(selected["X"] == original["x0"], axis=1))
        ref_info = json.loads((args.references / f"{name}.json").read_text())
        assert ref_info["accepted"]
        with np.load(args.references / f"{name}.npz") as arrays:
            reference = arrays["draws"].reshape(-1, D)
        budget = 150 if name == "schools_noncentered" else 100
        for seed in range(args.seeds):
            arms = ("filtered_full", "filtered_short")
            if seed % 2:
                arms = arms[::-1]
            for arm in arms:
                uniform = (
                    max(D, 10) - 1
                    if arm == "filtered_full"
                    else max(0, max(D, 10) - len(selected["X"]))
                )
                out = args.out / "cells" / f"{name}_{seed}_{arm}"
                out.mkdir(parents=True, exist_ok=True)
                print("COVERAGE START", name, seed, arm, flush=True)
                if (out / "result.json").exists():
                    entry = json.loads((out / "result.json").read_text())
                    assert entry["arm"] == arm
                    assert entry.get("retention_rule", "box") == args.filter
                    assert entry["reused_rows"] == len(selected["X"])
                    assert entry["uniform_initial"] == uniform
                else:
                    entry = reuse_case(
                        name,
                        seed,
                        "logger",
                        selected,
                        reference,
                        budget,
                        out,
                        uniform_initial=uniform,
                    )
                    entry.update(
                        arm=arm,
                        retention_rule=args.filter,
                        uniform_initial=uniform,
                        setup_all_unique_finite_rows=len(X),
                        setup_all_logp_range=[float(y.min()), float(y.max())],
                        execution_source_sha256=hashlib.sha256(
                            (args.out / "probe_source.py").read_bytes()
                        ).hexdigest(),
                    )
                    if "error" not in entry:
                        audit = entry["initial_design"]
                        assert audit["new_calls"] == uniform
                        assert audit["cached_calls"] == len(selected["X"])
                        assert audit["rows"] == uniform + len(selected["X"])
                    dump(out / "result.json", entry)
                report["cases"].append(entry)
                dump(args.out / "coverage.json", report)
                print(
                    "COVERAGE DONE",
                    name,
                    seed,
                    arm,
                    entry.get("error", entry.get("metrics")),
                    flush=True,
                )


def coverage_summary(args):
    """Compare the two controlled arms with the saved all-points baseline."""
    import ast
    import copy

    from pyvbmc import VBMC

    def read(path):
        return json.loads(path.read_text())

    follow = read(args.results / "coverage.json")
    baseline = read(args.baseline / "part_b.json")
    names = ("schools_noncentered", "ill_conditioned")
    assert len(follow["cases"]) == 20
    cases = follow["cases"] + [
        c
        for c in baseline["cases"]
        if c["name"] in names and c["arm"] == "logger"
    ]
    assert len(cases) == 30
    assert len({(c["name"], c["seed"], c["arm"]) for c in cases}) == 30
    evidence = schools_evidence()["checks"][-1]["ln_Z"]
    for c in cases:
        if c["name"] == "schools_noncentered" and "elbo" in c:
            c["elbo_error"] = c["elbo"] - evidence

    # The baseline fit must differ only by the explicitly added optional
    # initial-design-count override, which defaults to its old behavior.
    def functions(path):
        return {
            node.name: node
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.FunctionDef)
        }

    old = functions(args.baseline / "probe_source.py")
    new = functions(args.results / "probe_source.py")
    normalized = copy.deepcopy(new["reuse_case"])
    normalized.args.args = normalized.args.args[:-1]
    normalized.args.defaults = []
    normalized.body = [
        n
        for n in normalized.body
        if not (
            isinstance(n, ast.If)
            and ast.unparse(n.test) == "uniform_initial is not None"
        )
    ]
    assert ast.dump(normalized) == ast.dump(old["reuse_case"])
    for name in ("build_model", "posterior_metrics"):
        assert ast.dump(new[name]) == ast.dump(old[name])

    report = {
        "sources": sources(),
        "filter": follow["filter"],
        "filter_audit": follow["filter_audit"],
        "groups": [],
        "paired": [],
        "checks": [],
    }
    arms = ("logger", "filtered_full", "filtered_short")
    lines = [
        "# Filtering and initial coverage: focused comparison",
        "",
        "Logger seeding and single-point initialization in every arm. "
        "Five seeds per model; setup charged in full. Lower distances are better.",
        "Medians summarize returned fits only; numerical failures are retained "
        "and listed separately. Every arm has the same budget cap.",
        f"Filter: `{follow['filter']}`. The density rule retains points within "
        "10D nats of the best, with a D+1 minimum. Initial uniform counts are "
        "nine (full) and zero (shortened) on both models.",
        "",
        "| Model | Arm | Completed | Median marginal distance [range] | Median absolute ELBO error | Reached stability | Median seconds |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        for arm in arms:
            group = [c for c in cases if c["name"] == name and c["arm"] == arm]
            assert {c["seed"] for c in group} == set(range(5))
            complete = [c for c in group if "error" not in c]
            for c in group:
                assert c["total_spent"] == c["setup_cost"] + c["new_calls"]
                if "error" not in c:
                    assert c["total_spent"] == c["total_budget"]
                    design = c["initial_design"]
                    assert (
                        design["rows"]
                        == design["new_calls"] + design["cached_calls"]
                    )
                    assert design["new_calls"] == (
                        0 if arm == "filtered_short" else 9
                    )
                else:
                    assert c["total_spent"] <= c["total_budget"]
            distances = [c["metrics"]["mean_marginal_w1_sd"] for c in complete]
            item = {
                "name": name,
                "arm": arm,
                "completed": len(complete),
                "failures": [
                    {"seed": c["seed"], "error": c["error"]}
                    for c in group
                    if "error" in c
                ],
                "median_w1": float(np.median(distances)),
                "w1_range": [min(distances), max(distances)],
                "median_abs_elbo_error": float(
                    np.median([abs(c["elbo_error"]) for c in complete])
                ),
                "elbo_error_range": [
                    min(c["elbo_error"] for c in complete),
                    max(c["elbo_error"] for c in complete),
                ],
                "median_gaussian_kl": float(
                    np.median(
                        [
                            c["metrics"]["gaussian_symmetric_kl"]
                            for c in complete
                        ]
                    )
                ),
                "stable_cases": sum(
                    c["first_stable_iteration"] is not None for c in complete
                ),
                "median_seconds": float(
                    np.median([c["seconds"] for c in complete])
                ),
            }
            report["groups"].append(item)
            lines.append(
                f"| {name} | {arm} | {len(complete)}/5 | {item['median_w1']:.4g} "
                f"[{min(distances):.4g}, {max(distances):.4g}] | "
                f"{item['median_abs_elbo_error']:.4g} | {item['stable_cases']}/5 | {item['median_seconds']:.4g} |"
            )
        for seed in range(5):
            rows = {
                c["arm"]: c
                for c in cases
                if c["name"] == name and c["seed"] == seed
            }
            assert len({c["total_budget"] for c in rows.values()}) == 1
            assert len({c["setup_cost"] for c in rows.values()}) == 1
            for c in rows.values():
                for key in ("x0", "plb", "pub"):
                    assert np.array_equal(c[key], rows["logger"][key])
            # Constructor-only control: the function is never evaluated.
            # Both named models have unbounded adapter coordinates.
            c = rows["filtered_full"]
            x0, plb, pub = [np.asarray(c[k]) for k in ("x0", "plb", "pub")]
            initial = [
                VBMC(
                    lambda x: float(-0.5 * np.sum(np.asarray(x) ** 2)),
                    x0,
                    np.full_like(x0, -np.inf),
                    np.full_like(x0, np.inf),
                    plb,
                    pub,
                    options={"display": "off", "fun_eval_start": n},
                    seed=seed,
                )
                for n in (10, 1)
            ]
            for field in ("mu", "sigma", "lambd", "w"):
                assert np.array_equal(
                    getattr(initial[0].vp, field),
                    getattr(initial[1].vp, field),
                )
            assert (
                initial[0].rng.bit_generator.state
                == initial[1].rng.bit_generator.state
            )
            paths = [
                args.baseline
                / "cells"
                / f"{name}_{seed}_logger"
                / "calls.npz",
                args.results
                / "cells"
                / f"{name}_{seed}_filtered_full"
                / "calls.npz",
            ]
            with np.load(paths[0]) as a, np.load(paths[1]) as b:
                assert np.array_equal(a["X"][:9], b["X"][:9])
                assert np.array_equal(a["y"][:9], b["y"][:9])
            report["checks"].append(
                {
                    "name": name,
                    "seed": seed,
                    "full_design_calls_identical": True,
                    "filtered_initial_vp_and_rng_identical": True,
                }
            )
            if all("metrics" in c for c in rows.values()):
                d = {
                    arm: c["metrics"]["mean_marginal_w1_sd"]
                    for arm, c in rows.items()
                }
                report["paired"].append(
                    {
                        "name": name,
                        "seed": seed,
                        "filtering_delta": d["filtered_full"] - d["logger"],
                        "full_coverage_delta": d["filtered_full"]
                        - d["filtered_short"],
                    }
                )
    lines += [
        "",
        "Negative paired deltas favor filtering or full coverage, respectively.",
        "",
        "| Model | Seed | Filtering delta | Full-coverage delta |",
        "|---|---:|---:|---:|",
    ]
    for p in report["paired"]:
        lines.append(
            f"| {p['name']} | {p['seed']} | {p['filtering_delta']:.6g} | {p['full_coverage_delta']:.6g} |"
        )
    lines += [
        "",
        "## Individual cases",
        "",
        "| Model | Seed | Arm | ELBO error | Marginal distance | Gaussian sym. KL | First stable iteration |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for c in sorted(
        cases, key=lambda c: (c["name"], c["seed"], arms.index(c["arm"]))
    ):
        if "error" not in c:
            lines.append(
                f"| {c['name']} | {c['seed']} | {c['arm']} | {c['elbo_error']:.6g} | {c['metrics']['mean_marginal_w1_sd']:.6g} | {c['metrics']['gaussian_symmetric_kl']:.6g} | {c['first_stable_iteration']} |"
            )
        else:
            lines.append(
                f"| {c['name']} | {c['seed']} | {c['arm']} | FAILED | — | — | — |"
            )
    lines += ["", "## Numerical failures", ""]
    for c in cases:
        if "error" in c:
            lines.append(
                f"- {c['name']}, seed {c['seed']}, {c['arm']}: "
                f"{c['error']}; spent {c['total_spent']}/{c['total_budget']} units."
            )
    report["input_sha256"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (
            args.results / "coverage.json",
            args.baseline / "part_b.json",
            args.results / "probe_source.py",
            args.baseline / "probe_source.py",
        )
    }
    dump(args.out / "comparison.json", {"cases": cases})
    dump(args.out / "summary.json", report)
    (args.out / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("Validated 20 new fits against 10 saved baseline cases", flush=True)


def schools_evidence():
    """Integrate the schools and population mean analytically, then tau."""
    from scipy.integrate import quad

    y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
    sd = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

    def marginal(tau):
        cov = np.diag(sd**2 + tau**2) + 25 * np.ones((8, 8))
        return historical._mvn_logpdf(y, cov)

    peak = marginal(1.0)

    def integrand(tau):
        return (
            np.exp(marginal(tau) - peak)
            * 2
            / (5 * np.pi * (1 + (tau / 5) ** 2))
        )

    values = []
    for tol in (1e-10, 1e-12):
        value, error = quad(
            integrand, 0.0, np.inf, epsabs=tol, epsrel=tol, limit=200
        )
        values.append(
            {
                "ln_Z": float(np.log(value) + peak),
                "relative_quadrature_error": float(error / value),
            }
        )
    assert abs(values[0]["ln_Z"] - values[1]["ln_Z"]) < 1e-9
    return {
        "method": "Gaussian integration over mu and school effects; quadrature over HalfCauchy(5) tau",
        "checks": values,
    }


def summarize(args):
    """Publish compact evidence and check paired budgets without fitting."""

    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    a = read(args.setup / "part_a.json")
    refs = read(args.references / "references.json")
    b = read(args.results / "part_b.json")
    evidence = schools_evidence()
    b["schools_evidence"] = evidence
    for case in b["cases"]:
        if case["name"] == "schools_noncentered" and "elbo" in case:
            case["elbo_error"] = case["elbo"] - evidence["checks"][-1]["ln_Z"]
    cases = b["cases"]
    assert len(cases) == 45
    assert len({(c["name"], c["seed"], c["arm"]) for c in cases}) == 45
    summary = {"groups": [], "paired": [], "validation": {}}
    lines = [
        "# PyMC setup reuse: measured results",
        "",
        "Medians over completed cases; failures are counted separately.",
        "",
        "| Model | Arm | Completed | Reused rows | Absolute ELBO error | Gaussian sym. KL | Mean marginal W1 / SD | Reached stability | Seconds |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    input_paths = [args.setup / "part_a.json", args.results / "part_b.json"]
    for path in sorted((args.setup / "setups").glob("*.json")):
        dump(args.out / "setups" / path.name, clean(read(path)))
    for name in REUSE_MODELS:
        setup = read(args.setup / "setups" / f"{name}.json")
        ref = read(args.references / f"{name}.json")
        assert ref["accepted"]
        input_paths.extend(
            [
                args.setup / "setups" / f"{name}.json",
                args.references / f"{name}.json",
                args.references / f"{name}.npz",
            ]
        )
        for arm in ARMS:
            group = [c for c in cases if c["name"] == name and c["arm"] == arm]
            assert {c["seed"] for c in group} == set(range(5))
            complete = [c for c in group if "error" not in c]
            for c in group:
                assert c["setup_cost"] == setup["cost"]
                design = c["initial_design"]
                if design:
                    assert (
                        design["rows"]
                        == design["new_calls"] + design["cached_calls"]
                    )
                assert c["total_spent"] == c["setup_cost"] + c["new_calls"]
                if "error" not in c:
                    assert c["total_spent"] == c["total_budget"]
                else:
                    assert c["total_spent"] <= c["total_budget"]

            def median(values):
                values = [v for v in values if v is not None]
                return float(np.median(values)) if values else None

            def span(values):
                values = [v for v in values if v is not None]
                return (
                    [float(min(values)), float(max(values))]
                    if values
                    else None
                )

            item = {
                "name": name,
                "arm": arm,
                "completed": len(complete),
                "failures": [
                    {"seed": c["seed"], "error": c["error"]}
                    for c in group
                    if "error" in c
                ],
                "reused_rows": group[0]["reused_rows"],
                "median_abs_elbo_error": median(
                    [
                        (
                            abs(c["elbo_error"])
                            if c["elbo_error"] is not None
                            else None
                        )
                        for c in complete
                    ]
                ),
                "median_seconds": median([c["seconds"] for c in complete]),
                "elbo_error_range": span([c["elbo_error"] for c in complete]),
                "marginal_w1_range": span(
                    [c["metrics"]["mean_marginal_w1_sd"] for c in complete]
                ),
                "gaussian_kl_range": span(
                    [c["metrics"]["gaussian_symmetric_kl"] for c in complete]
                ),
                "stable_cases": sum(
                    c["first_stable_iteration"] is not None for c in complete
                ),
                "median_first_stable_iteration": median(
                    [c["first_stable_iteration"] for c in complete]
                ),
                "median_iterations": median(
                    [c["iterations"] for c in complete]
                ),
                "metrics": {
                    k: median([c["metrics"][k] for c in complete])
                    for k in (
                        "gaussian_symmetric_kl",
                        "mean_marginal_w1_sd",
                        "max_mean_sd",
                        "covariance_relative_frobenius",
                    )
                },
            }
            summary["groups"].append(item)

            def fmt(v):
                return "—" if v is None else f"{v:.4g}"

            lines.append(
                f"| {name} | {arm} | {len(complete)}/5 | {item['reused_rows']} | "
                f"{fmt(item['median_abs_elbo_error'])} | {fmt(item['metrics']['gaussian_symmetric_kl'])} | "
                f"{fmt(item['metrics']['mean_marginal_w1_sd'])} | {item['stable_cases']}/5 | {fmt(item['median_seconds'])} |"
            )
        for seed in range(5):
            rows = {
                c["arm"]: c
                for c in cases
                if c["name"] == name and c["seed"] == seed
            }
            summary["paired"].append(
                {
                    "name": name,
                    "seed": seed,
                    "w1_delta_vs_discard": {
                        arm: rows[arm]["metrics"]["mean_marginal_w1_sd"]
                        - rows["discard"]["metrics"]["mean_marginal_w1_sd"]
                        for arm in ("f_vals", "logger")
                        if "metrics" in rows[arm]
                        and "metrics" in rows["discard"]
                    },
                }
            )
    summary["validation"] = {
        "cases": 45,
        "completed": sum("error" not in c for c in cases),
        "paired_budgets_checked": True,
        "references_accepted": True,
    }
    lines += [
        "",
        "## Individual cases",
        "",
        "Completed fits may still be unconverged; inspect the stability column and the full error range.",
        "",
        "| Model | Seed | Arm | ELBO error | Gaussian sym. KL | Mean marginal W1 / SD | First stable iteration | Seconds |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for case in sorted(
        cases,
        key=lambda c: (
            REUSE_MODELS.index(c["name"]),
            c["seed"],
            ARMS.index(c["arm"]),
        ),
    ):
        if "error" in case:
            lines.append(
                f"| {case['name']} | {case['seed']} | {case['arm']} | {case['error']} | — | — | — | {case['seconds']:.4g} |"
            )
        else:
            lines.append(
                f"| {case['name']} | {case['seed']} | {case['arm']} | {fmt(case['elbo_error'])} | "
                f"{fmt(case['metrics']['gaussian_symmetric_kl'])} | {fmt(case['metrics']['mean_marginal_w1_sd'])} | "
                f"{fmt(case['first_stable_iteration'])} | {case['seconds']:.4g} |"
            )
    for filename, content in (
        ("part_a.json", a),
        ("references.json", refs),
        ("part_b.json", b),
        ("summary.json", summary),
    ):
        dump(args.out / filename, clean(content))
    (args.out / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    dump(
        args.out / "sources.json",
        {
            "summary_runner": sources(),
            "input_sha256": {
                str(p.resolve().relative_to(ROOT)): hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
                for p in input_paths
            },
        },
    )
    print("Validated 45 paired cases; summaries under", args.out, flush=True)


def check(args):
    """Check derivative order, stored observations, and unchanged bounds."""
    import logging

    import pymc as pm

    from pyvbmc import VBMC

    report = {"sources": sources(), "models": []}
    for name in (
        "scalar",
        "positive",
        "vector",
        "bounded",
        "one_sided",
        "schools_noncentered",
        "ill_conditioned",
        "logistic20",
        "varying20",
        "schools_centered",
    ):
        print("CHECK", name, flush=True)
        target = historical.PyMCTarget(build_model(name)["model"])
        setup = json.loads(
            (args.setup / "setups" / f"{name}.json").read_text()
        )
        x = np.array(setup["x0"])
        fg, hessian, _ = compile_derivatives(target)
        value, gradient = fg(target.unflatten(x))
        H = np.asarray(hessian(target.unflatten(x)))
        assert np.asarray(gradient).dtype == np.float64
        assert H.dtype == np.float64
        steps = 1e-5 * np.maximum(1, np.abs(x))
        fd_gradient = np.zeros(target.D)
        fd_hessian = np.zeros_like(H)
        for i, h in enumerate(steps):
            dx = np.zeros(target.D)
            dx[i] = h
            fd_gradient[i] = (
                target.log_joint(x + dx) - target.log_joint(x - dx)
            ) / (2 * h)
            fd_hessian[:, i] = (
                fg(target.unflatten(x + dx))[1]
                - fg(target.unflatten(x - dx))[1]
            ) / (2 * h)
        gradient_error = float(
            np.max(
                np.abs(fd_gradient - gradient)
                / np.maximum(1, np.abs(gradient))
            )
        )
        hessian_error = float(
            np.max(np.abs(fd_hessian - H)) / max(1, np.max(np.abs(H)))
        )
        values = np.array([target.log_joint(row) for row in setup["X"]])
        value_error = float(np.max(np.abs(values - setup["y"])))
        value_scaled_error = float(
            np.max(np.abs(values - setup["y"]) / np.maximum(1, np.abs(values)))
        )
        messages = []

        class Capture(logging.Handler):
            def emit(self, record):
                messages.append(record.getMessage())

        capture = Capture()
        logger = logging.getLogger("VBMC_init")
        logger.addHandler(capture)
        try:
            vbmc = VBMC(
                target.log_joint,
                x[None],
                target.lb,
                target.ub,
                np.array(setup["plb"])[None],
                np.array(setup["pub"])[None],
                options={"display": "off"},
                seed=0,
            )
        finally:
            logger.removeHandler(capture)
        assert not messages, messages
        assert np.allclose(
            vbmc.parameter_transformer.inverse(vbmc.x0),
            x[None],
            rtol=0,
            atol=1e-12,
        )
        assert np.array_equal(
            vbmc.plausible_lower_bounds.ravel(), setup["plb"]
        )
        assert np.array_equal(
            vbmc.plausible_upper_bounds.ravel(), setup["pub"]
        )
        assert np.isfinite(value)
        assert gradient_error < 1e-5, (name, gradient_error)
        assert hessian_error < 1e-5, (name, hessian_error)
        # Joint value/gradient and density-only graphs may reduce sums in
        # different orders. Collapsing-scale trials can reach -3e18 nats.
        assert np.allclose(values, setup["y"], rtol=1e-12, atol=1e-8), (
            name,
            value_error,
            value_scaled_error,
        )
        # A one-off Hessian need not pay the default linker's JIT latency.
        # Compare an exact Python-linked Hessian as a supplementary timing
        # probe; it does not change the measured setup or any reuse fit.
        rvs = [
            next(rv for rv in target.partial.free_RVs if rv.name == n)
            for n in target.names
        ]
        start = time.perf_counter()
        python_hessian = target.partial.compile_d2logp(
            vars=rvs, jacobian=True, negate_output=False, mode="FAST_COMPILE"
        )
        python_compile_time = time.perf_counter() - start
        start = time.perf_counter()
        H_python = np.asarray(python_hessian(target.unflatten(x)))
        python_call_time = time.perf_counter() - start
        python_difference = float(
            np.max(np.abs(H_python - H)) / max(1, np.max(np.abs(H)))
        )
        assert python_difference < 1e-10, (name, python_difference)
        report["models"].append(
            {
                "name": name,
                "gradient_relative_error": gradient_error,
                "hessian_relative_error": hessian_error,
                "cached_value_max_absolute_error": value_error,
                "cached_value_max_scaled_error": value_scaled_error,
                "bounds_unchanged": True,
                "python_hessian_compile_seconds": python_compile_time,
                "python_hessian_first_call_seconds": python_call_time,
                "python_hessian_relative_difference": python_difference,
            }
        )
        dump(args.out / "checks.json", report)
    # A likelihood can put a valid mode far beyond any fixed prior quantile.
    # Measure the planned unconditional relocation without changing the
    # predeclared setup used in the reuse experiment.
    with pm.Model() as model:
        x = pm.Normal("x", 0.0, 1.0)
        pm.Normal("y", x, 0.1, observed=10.0)
    target = historical.PyMCTarget(model)
    fg, hessian, _ = compile_derivatives(target)
    prior, _ = prior_draws(target)
    run = search(target, fg, cap=23, atol=0.001)
    setup = make_setup(target, run, hessian, prior)
    H = curvature(target, hessian, run["x"])
    assert abs(run["x"][0] - 1000 / 101) < 1e-6
    assert H["positive_definite"]
    assert setup["relocated"].tolist() == [0]
    report["location_counterexample"] = {
        "model": "x ~ Normal(0,1); observed 10 ~ Normal(x,0.1)",
        "analytic_mode": 1000 / 101,
        "analytic_sd": 1 / np.sqrt(101),
        "search_mode": run["x"],
        "search_calls": run["n_calls"],
        "precision_positive_definite": H["positive_definite"],
        "prior_location_interval": np.quantile(prior, [0.005, 0.995], axis=0),
        "unconditional_x0": setup["x0"],
        "unconditional_plb": setup["plb"],
        "unconditional_pub": setup["pub"],
        "curvature_guard_would_relocate": False,
    }
    original = subprocess.check_output(
        ["git", "show", "HEAD:dev/scripts/pymc_feasibility.py"], cwd=ROOT
    )
    current = Path(historical.__file__).read_bytes().replace(b"\r\n", b"\n")
    assert current == original.replace(b"\r\n", b"\n")
    report["historical_prototype_unchanged"] = True
    dump(args.out / "checks.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "part",
        choices=[
            "a",
            "references",
            "b",
            "summarize",
            "check",
            "coverage",
            "coverage_summary",
        ],
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models")
    parser.add_argument("--setup", type=Path)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument(
        "--filter", choices=["log_density", "box"], default="log_density"
    )
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--tune", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    if (
        args.part in ("b", "summarize", "check", "coverage")
        and args.setup is None
    ):
        parser.error(f"{args.part} requires --setup")
    if args.part in ("b", "summarize", "coverage") and args.references is None:
        parser.error(f"{args.part} requires --references")
    if args.part == "summarize" and args.results is None:
        parser.error("summarize requires --results")
    if args.part == "coverage_summary" and (
        args.results is None or args.baseline is None
    ):
        parser.error("coverage_summary requires --results and --baseline")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.part not in ("summarize", "coverage_summary"):
        (args.out / "probe_source.py").write_bytes(Path(__file__).read_bytes())
    dump(args.out / "invocation.json", vars(args))
    try:
        {
            "a": part_a,
            "references": references,
            "b": part_b,
            "summarize": summarize,
            "check": check,
            "coverage": coverage,
            "coverage_summary": coverage_summary,
        }[args.part](args)
    except Exception:
        (args.out / "error.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    main()
