"""Check whether compact golden traces reproduce the final-boost boundary.

The reconstruction uses only a golden JSON/NPZ pair and the matching
``benchmark_targets`` configuration.  Dill restart captures are loaded only
as validation references.  By default the script performs a cheap structural
audit of the full trace population and compares every available restart
capture with its reconstructed VP, transformer, GP, and boost-relevant state.

Optional ``--replay`` cases run one final boost from the authentic and rebuilt
states with identical fresh RNG states.  This is deliberately opt-in because
variational optimization is much more expensive than reconstruction.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import benchmark_targets
from benchmark_targets import find_config, metrics
from boost_comparison import _build_transformer, _build_vp

from pyvbmc import VBMC
from pyvbmc.vbmc.gaussian_process_train import (
    _cov_identifier_to_covariance_function,
    _meanfun_name_to_mean_function,
)
from pyvbmc.vbmc.variational_optimization import _gp_log_joint

DEFAULT_TRACE_DIR = HERE / "runs" / "golden" / "reference_870_20260907"
DEFAULT_CAPTURE_ROOT = HERE / "runs" / "latent_fixes" / "boost_20260907"
BOOST_OPTION_KEYS = (
    "min_final_components",
    "ns_ent",
    "ns_ent_fast",
    "ns_ent_fine",
    "ns_ent_boost",
    "ns_ent_fast_boost",
    "ns_ent_fine_boost",
    "ns_elbo",
    "ns_elbo_incr",
    "variable_means",
    "variable_weights",
    "tol_weight",
    "weight_penalty",
    "tol_elcbo_boost",
    "hpd_frac",
    "tol_length",
    "tol_con_loss",
    "det_entropy_tol_opt",
    "stochastic_optimizer",
    "sgd_step_size",
    "tol_fun_stochastic",
    "elcbo_midpoint",
    "skip_elbo_variance",
    "max_iter_stochastic",
    "elcbo_impro_weight",
    "tol_improvement",
    "pruning_threshold_multiplier",
)
OPTIM_STATE_KEYS = (
    "N",
    "iter",
    "n_eff",
    "warmup",
    "entropy_switch",
    "entropy_alpha",
    "uncertainty_handling_level",
    "vp_K",
    "gp_mean_fun",
    "gp_cov_fun",
    "gp_noise_fun",
)
VP_ARRAY_KEYS = ("w", "eta", "mu", "sigma", "lambd")


def _seconds(start):
    return time.perf_counter() - start


def _json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return repr(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_scalar(item) for item in value]
    return repr(value)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_provenance():
    """Record the actual imported modules and reconstruction source."""
    import gpyreg
    import scipy

    import pyvbmc
    import pyvbmc.vbmc.variational_optimization as variational_optimization
    import pyvbmc.vbmc.vbmc as vbmc_module

    modules = {
        "pyvbmc": pyvbmc,
        "gpyreg": gpyreg,
        "numpy": np,
        "scipy": scipy,
        "benchmark_targets": benchmark_targets,
        "pyvbmc.vbmc.vbmc": vbmc_module,
        "pyvbmc.vbmc.variational_optimization": variational_optimization,
    }
    records = {}
    for name, module in modules.items():
        path = Path(module.__file__).resolve()
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = getattr(module, "__version__", None)
        records[name] = {
            "version": version,
            "path": str(path),
            "sha256": _sha256(path),
        }
    source = Path(__file__).resolve()
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
        "modules": records,
        "boost_reconstruction": {
            "path": str(source),
            "sha256": _sha256(source),
        },
    }


def _array_error(actual, expected):
    """Return shape, exactness, and finite absolute/relative errors."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    result = {
        "actual_shape": list(actual.shape),
        "expected_shape": list(expected.shape),
        "shape_equal": actual.shape == expected.shape,
    }
    if actual.shape != expected.shape:
        result.update(exact=False, max_abs=None, rms_abs=None, max_rel=None)
        return result
    result["exact"] = bool(np.array_equal(actual, expected, equal_nan=True))
    if actual.size == 0:
        result.update(max_abs=0.0, rms_abs=0.0, max_rel=0.0)
        return result
    try:
        a = actual.astype(float)
        e = expected.astype(float)
    except (TypeError, ValueError):
        result.update(max_abs=None, rms_abs=None, max_rel=None)
        return result
    finite = np.isfinite(a) & np.isfinite(e)
    if not np.any(finite):
        result.update(max_abs=0.0, rms_abs=0.0, max_rel=0.0)
        return result
    delta = np.abs(a[finite] - e[finite])
    scale = np.maximum(np.abs(e[finite]), np.finfo(float).tiny)
    result.update(
        max_abs=float(np.max(delta)),
        rms_abs=float(np.sqrt(np.mean(delta**2))),
        max_rel=float(np.max(delta / scale)),
    )
    return result


def _tag(label, seed):
    return f"{label}_seed{seed}"


def _capture_paths(root):
    return sorted(Path(root).glob("*/captures/*.boost_pre.dill"))


def _load_capture(path):
    import dill

    with Path(path).open("rb") as stream:
        return dill.load(stream)


def _capture_options(sidecar):
    """Return the options used by boost_replay's captured trajectories."""
    options = dict(sidecar["requested_options"])
    options.update(vectorized_target=False, tol_elcbo_boost=0.1)
    return options


def audit_population(trace_dir):
    """Audit the information-loss conditions needed by reconstruction."""
    started = time.perf_counter()
    rows = []
    for side_path in sorted(Path(trace_dir).glob("*.json")):
        npz_path = side_path.with_suffix(".npz")
        side = json.loads(side_path.read_text(encoding="utf-8"))
        if not npz_path.is_file():
            rows.append({"tag": side_path.stem, "missing_npz": True})
            continue
        with np.load(npz_path, allow_pickle=False) as trace:
            best = int(side["final"]["best_iter"])
            last = len(trace["iter"]) - 1
            live_n = len(trace["X_orig"])
            final_func_equals_N = bool(
                int(trace["func_count"][-1]) == int(trace["N"][-1])
            )
            final_neff_equals_live = bool(int(trace["n_eff"][-1]) == live_n)
            warmup_false_after_best = bool(
                np.all(np.asarray(trace["warmup"])[best:] == 0)
            )
            prefix_n = int(trace["n_eff"][best])
            rows.append(
                {
                    "tag": side_path.stem,
                    "best_iter": best,
                    "last_iter": last,
                    "earlier_best": best < last,
                    "best_n_eff": prefix_n,
                    "live_rows": live_n,
                    "final_func_count_equals_N": final_func_equals_N,
                    "final_n_eff_equals_live_rows": final_neff_equals_live,
                    "warmup_false_from_best": warmup_false_after_best,
                    "eligible": bool(
                        final_func_equals_N
                        and final_neff_equals_live
                        and warmup_false_after_best
                        and 0 < prefix_n <= live_n
                    ),
                }
            )
    present = [row for row in rows if not row.get("missing_npz")]
    failed = [row["tag"] for row in present if not row["eligible"]]
    return {
        "trace_dir": str(Path(trace_dir).resolve()),
        "runs": len(present),
        "missing_npz": sum(row.get("missing_npz", False) for row in rows),
        "earlier_best_count": sum(row["earlier_best"] for row in present),
        "final_func_count_equals_N_count": sum(
            row["final_func_count_equals_N"] for row in present
        ),
        "final_n_eff_equals_live_rows_count": sum(
            row["final_n_eff_equals_live_rows"] for row in present
        ),
        "warmup_false_from_best_count": sum(
            row["warmup_false_from_best"] for row in present
        ),
        "eligible_count": sum(row["eligible"] for row in present),
        "failed_tags": failed,
        "earlier_best_tags": [
            row["tag"] for row in present if row["earlier_best"]
        ],
        "interpretation": (
            "func_count[-1] == N[-1] rules out repeated evaluations; "
            "n_eff[-1] == len(X_orig) and warmup false from best_iter make "
            "X_orig[:n_eff[best_iter]] the best-iteration live set"
        ),
        "wall_s": _seconds(started),
    }


def reconstruct_state(sidecar, trace, *, rng_seed=0):
    """Rebuild the boost inputs without consulting a dill capture."""
    started = time.perf_counter()
    label = sidecar["label"]
    seed = int(sidecar["seed"])
    best = int(sidecar["final"]["best_iter"])
    problem = find_config(label).make(seed=seed)
    args, _ = problem.vbmc_args()
    runner = VBMC(*args, options=_capture_options(sidecar), seed=seed)

    transform_type = sidecar["requested_options"].get(
        "bounded_transform", "probit"
    )
    pt = _build_transformer(problem, trace, best, transform_type)
    rng = np.random.default_rng(rng_seed)
    vp = _build_vp(trace, pt, best)
    vp.rng = rng
    vp.stats = {
        "elbo": float(trace["elbo"][best]),
        "elbo_sd": float(trace["elbo_sd"][best]),
        "stable": bool(trace["stable"][best]),
    }

    live_n = len(trace["X_orig"])
    best_n = int(trace["n_eff"][best])
    eligibility = {
        "func_count_equals_N": bool(
            int(trace["func_count"][-1]) == int(trace["N"][-1])
        ),
        "final_n_eff_equals_live": bool(int(trace["n_eff"][-1]) == live_n),
        "warmup_false_from_best": bool(
            np.all(np.asarray(trace["warmup"])[best:] == 0)
        ),
        "valid_prefix": 0 < best_n <= live_n,
    }
    if not all(eligibility.values()):
        failed = [key for key, value in eligibility.items() if not value]
        raise ValueError(
            f"{_tag(label, seed)} is not reconstructable: {failed}"
        )

    X_orig = np.asarray(trace["X_orig"], dtype=float)[:best_n]
    y_orig = np.asarray(trace["y_orig"], dtype=float)[:best_n].reshape(-1, 1)
    X = pt(X_orig)
    y = y_orig + np.asarray(pt.log_abs_det_jacobian(X)).reshape(-1, 1)
    s2 = (
        None
        if problem.noise_sd is None
        else np.full((best_n, 1), float(problem.noise_sd) ** 2)
    )
    hyp_mask = np.asarray(trace["gp_hyp_iter"]) == best
    hyp = np.asarray(trace["gp_hyp"], dtype=float)[hyp_mask]

    import gpyreg as gpr

    state = runner.optim_state
    noise_switches = state["gp_noise_fun"]
    noise = gpr.noise_functions.GaussianNoise(
        constant_add=noise_switches[0] == 1,
        user_provided_add=noise_switches[1] == 1,
        scale_user_provided=noise_switches[1] == 2,
        rectified_linear_output_dependent_add=noise_switches[2] == 1,
    )
    gp = gpr.GP(
        D=problem.D,
        covariance=_cov_identifier_to_covariance_function(state["gp_cov_fun"]),
        mean=_meanfun_name_to_mean_function(state["gp_mean_fun"]),
        noise=noise,
    )
    gp.update(X_new=X, y_new=y, s2_new=s2, hyp=hyp, compute_posterior=True)

    last = len(trace["iter"]) - 1
    state.update(
        N=int(trace["N"][last]),
        iter=last,
        n_eff=int(trace["n_eff"][last]),
        warmup=False,
        entropy_switch=False,
        entropy_alpha=0,
        vp_K=int(trace["K"][last]),
    )
    runner.parameter_transformer = pt
    runner.vp = vp
    runner.gp = gp
    runner.rng = rng
    return {
        "runner": runner,
        "vp": vp,
        "gp": gp,
        "optim_state": state,
        "options": runner.options,
        "eligibility": eligibility,
        "best_iter": best,
        "best_n_eff": best_n,
        "wall_s": _seconds(started),
    }


def _posterior_errors(actual_gp, expected_gp):
    rows = []
    if len(actual_gp.posteriors) != len(expected_gp.posteriors):
        return {"count_equal": False, "samples": rows}
    for index, (actual, expected) in enumerate(
        zip(actual_gp.posteriors, expected_gp.posteriors)
    ):
        row = {"sample": index}
        for key in ("alpha", "L", "sW"):
            row[key] = _array_error(
                getattr(actual, key), getattr(expected, key)
            )
        row["L_chol_equal"] = bool(actual.L_chol == expected.L_chol)
        rows.append(row)
    return {"count_equal": True, "samples": rows}


def compare_capture(capture_path, trace_dir):
    """Compare one trace-only reconstruction with its authentic capture."""
    started = time.perf_counter()
    capture = _load_capture(capture_path)
    label, seed = capture["label"], int(capture["seed"])
    tag = _tag(label, seed)
    side_path = Path(trace_dir) / f"{tag}.json"
    npz_path = side_path.with_suffix(".npz")
    side = json.loads(side_path.read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=False) as trace:
        rebuilt = reconstruct_state(side, trace)
    vp, gp = rebuilt["vp"], rebuilt["gp"]
    true_vp, true_gp = capture["vp"], capture["gp"]

    vp_errors = {
        key: _array_error(getattr(vp, key), getattr(true_vp, key))
        for key in VP_ARRAY_KEYS
    }
    vp_errors["K_equal"] = vp.K == true_vp.K
    vp_errors["flags_equal"] = all(
        getattr(vp, key) == getattr(true_vp, key)
        for key in (
            "optimize_mu",
            "optimize_sigma",
            "optimize_lambd",
            "optimize_weights",
        )
    )
    vp_errors["stats"] = {
        key: _array_error(vp.stats[key], true_vp.stats[key])
        for key in ("elbo", "elbo_sd", "stable")
    }
    vp_errors["unavailable_stats"] = sorted(set(true_vp.stats) - set(vp.stats))
    vp_errors["eta_note"] = (
        "eta is a stale cached representation in some captures and is not "
        "read by get_parameters; optimization derives raw weights from w"
    )

    pt, true_pt = vp.parameter_transformer, true_vp.parameter_transformer
    pt_errors = {
        key: _array_error(getattr(pt, key), getattr(true_pt, key))
        for key in (
            "lb_orig",
            "ub_orig",
            "mu",
            "delta",
            "scale",
            "R_mat",
            "type",
        )
    }
    probe_orig = np.asarray(side["plb"] + side["pub"], dtype=float).reshape(
        2, -1
    )
    pt_errors["forward_probe"] = _array_error(
        pt(probe_orig), true_pt(probe_orig)
    )
    pt_errors["inverse_gp_probe"] = _array_error(
        pt.inverse(gp.X[: min(11, len(gp.X))]),
        true_pt.inverse(gp.X[: min(11, len(gp.X))]),
    )

    gp_errors = {
        "X": _array_error(gp.X, true_gp.X),
        "y": _array_error(gp.y, true_gp.y),
        "s2": _array_error(
            np.array([]) if gp.s2 is None else gp.s2,
            np.array([]) if true_gp.s2 is None else true_gp.s2,
        ),
        "hyp": _array_error(
            gp.get_hyperparameters(as_array=True),
            true_gp.get_hyperparameters(as_array=True),
        ),
        "model_types_equal": bool(
            type(gp.covariance) is type(true_gp.covariance)
            and type(gp.mean) is type(true_gp.mean)
            and type(gp.noise) is type(true_gp.noise)
        ),
        "noise_switches": _array_error(
            gp.noise.parameters, true_gp.noise.parameters
        ),
        "posterior": _posterior_errors(gp, true_gp),
    }
    probe_idx = np.unique(
        np.linspace(0, len(gp.X) - 1, min(17, len(gp.X))).astype(int)
    )
    probe = gp.X[probe_idx]
    pred_started = time.perf_counter()
    pred = gp.predict(probe, separate_samples=True)
    true_pred = true_gp.predict(probe, separate_samples=True)
    gp_errors["prediction"] = {
        "mean": _array_error(pred[0], true_pred[0]),
        "variance": _array_error(pred[1], true_pred[1]),
        "wall_s": _seconds(pred_started),
    }
    joint_started = time.perf_counter()
    joint = _gp_log_joint(vp, gp, False, compute_var=True)
    true_joint = _gp_log_joint(true_vp, true_gp, False, compute_var=True)
    gp_errors["gp_log_joint"] = {
        f"output_{index}": (
            None
            if actual is None and expected is None
            else _array_error(actual, expected)
        )
        for index, (actual, expected) in enumerate(zip(joint, true_joint))
    }
    gp_errors["gp_log_joint"]["wall_s"] = _seconds(joint_started)
    gradient_started = time.perf_counter()
    gradient = _gp_log_joint(vp, gp, True, compute_var=False)
    true_gradient = _gp_log_joint(true_vp, true_gp, True, compute_var=False)
    gp_errors["gp_log_joint_gradient_raw_state"] = {
        f"output_{index}": (
            None
            if actual is None and expected is None
            else _array_error(actual, expected)
        )
        for index, (actual, expected) in enumerate(
            zip(gradient, true_gradient)
        )
    }
    gp_errors["gp_log_joint_gradient_raw_state"]["wall_s"] = _seconds(
        gradient_started
    )
    canonical_started = time.perf_counter()
    canonical_vp = copy.deepcopy(vp)
    canonical_true_vp = copy.deepcopy(true_vp)
    canonical_theta = canonical_vp.get_parameters()
    canonical_true_theta = canonical_true_vp.get_parameters()
    canonical_vp.set_parameters(canonical_theta)
    canonical_true_vp.set_parameters(canonical_true_theta)
    canonical_vp.eta = (
        canonical_theta[-canonical_vp.K :]
        - np.max(canonical_theta[-canonical_vp.K :])
    ).reshape(1, -1)
    canonical_true_vp.eta = (
        canonical_true_theta[-canonical_true_vp.K :]
        - np.max(canonical_true_theta[-canonical_true_vp.K :])
    ).reshape(1, -1)
    canonical = _gp_log_joint(canonical_vp, gp, True, compute_var=False)
    canonical_true = _gp_log_joint(
        canonical_true_vp, true_gp, True, compute_var=False
    )
    gp_errors["gp_log_joint_gradient_optimizer_entry"] = {
        f"output_{index}": (
            None
            if actual is None and expected is None
            else _array_error(actual, expected)
        )
        for index, (actual, expected) in enumerate(
            zip(canonical, canonical_true)
        )
    }
    gp_errors["gp_log_joint_gradient_optimizer_entry"]["wall_s"] = _seconds(
        canonical_started
    )
    gp_errors["gradient_note"] = (
        "raw-state gradients retain harmless cached-parameter inconsistencies; "
        "optimizer-entry gradients compare copies after the same parameter "
        "canonicalization and max-shifted eta assignment used by _neg_elcbo"
    )

    state_errors = {
        key: {
            "rebuilt": _json_scalar(rebuilt["optim_state"].get(key)),
            "captured": _json_scalar(capture["optim_state"].get(key)),
            "equal": bool(
                np.array_equal(
                    rebuilt["optim_state"].get(key),
                    capture["optim_state"].get(key),
                )
            ),
        }
        for key in OPTIM_STATE_KEYS
    }
    K_new = max(vp.K, rebuilt["options"]["min_final_components"])
    option_errors = {}
    for key in BOOST_OPTION_KEYS:
        actual = rebuilt["options"].get(key)
        expected = capture["options"].get(key)
        row = {
            "rebuilt": _json_scalar(actual),
            "captured": _json_scalar(expected),
            "type_equal": type(actual) is type(expected),
        }
        if key.startswith("ns_") or key == "pruning_threshold_multiplier":
            row["evaluated_at_K_new"] = _array_error(
                rebuilt["options"].eval(key, {"K": K_new}),
                capture["options"].eval(key, {"K": K_new}),
            )
        else:
            try:
                row["equal"] = bool(actual == expected)
            except ValueError:
                row["equal"] = bool(np.array_equal(actual, expected))
        option_errors[key] = row

    return {
        "tag": tag,
        "capture": str(Path(capture_path).resolve()),
        "trace": str(npz_path.resolve()),
        "best_iter": rebuilt["best_iter"],
        "best_n_eff": rebuilt["best_n_eff"],
        "eligibility": rebuilt["eligibility"],
        "vp": vp_errors,
        "transformer": pt_errors,
        "gp": gp_errors,
        "optim_state": state_errors,
        "options": option_errors,
        "reconstruction_wall_s": rebuilt["wall_s"],
        "comparison_wall_s": _seconds(started),
    }


def _fresh_common_state(base_seed, label, seed):
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    label_word = int.from_bytes(digest[:4], "little")
    rng = np.random.default_rng(
        np.random.SeedSequence([base_seed, seed, label_word])
    )
    return copy.deepcopy(rng.bit_generator.state)


def _run_boost(runner, vp, gp, rng_state):
    import pyvbmc.vbmc.vbmc as vbmc_module

    rng = np.random.default_rng()
    rng.bit_generator.state = copy.deepcopy(rng_state)
    vp = copy.deepcopy(vp)
    vp.rng = rng
    runner.rng = rng
    captured = {}
    original = vbmc_module.optimize_vp

    def instrumented(*args, **kwargs):
        result = original(*args, **kwargs)
        captured["candidate"] = copy.deepcopy(result[0])
        return result

    vbmc_module.optimize_vp = instrumented
    started = time.perf_counter()
    try:
        selected, elbo, elbo_sd, changed = runner.final_boost(vp, gp)
    finally:
        vbmc_module.optimize_vp = original
    return {
        "candidate": captured.get("candidate"),
        "selected": selected,
        "elbo": elbo,
        "elbo_sd": elbo_sd,
        "changed": changed,
        "wall_s": _seconds(started),
    }


def _metric_summary(label, seed, vp):
    problem = find_config(label).make(seed=seed)
    values = metrics(problem, vp, float(vp.stats["elbo"]))
    return {
        key: _json_scalar(values[key])
        for key in ("elbo_err", "gskl", "mmtv", "rmse", "moment_method")
    }


def _boost_summary(result, label, seed):
    candidate = result["candidate"]
    selected = result["selected"]
    return {
        "candidate_stats": {
            key: _json_scalar(candidate.stats[key])
            for key in ("elbo", "elbo_sd")
        },
        "candidate_metrics": _metric_summary(label, seed, candidate),
        "selected_stats": {
            key: _json_scalar(selected.stats[key])
            for key in ("elbo", "elbo_sd")
        },
        "selected_metrics": _metric_summary(label, seed, selected),
        "returned_elbo": _json_scalar(result["elbo"]),
        "returned_elbo_sd": _json_scalar(result["elbo_sd"]),
        "changed": bool(result["changed"]),
        "wall_s": result["wall_s"],
    }


def replay_capture(capture_path, trace_dir, common_seed):
    """Run authentic and reconstructed boosts with a common fresh RNG."""
    capture = _load_capture(capture_path)
    label, seed = capture["label"], int(capture["seed"])
    tag = _tag(label, seed)
    side_path = Path(trace_dir) / f"{tag}.json"
    side = json.loads(side_path.read_text(encoding="utf-8"))
    with np.load(side_path.with_suffix(".npz"), allow_pickle=False) as trace:
        rebuilt = reconstruct_state(side, trace, rng_seed=common_seed)
    rng_state = _fresh_common_state(common_seed, label, seed)

    authentic_runner = rebuilt["runner"]
    authentic_runner.options = copy.deepcopy(capture["options"])
    authentic_runner.optim_state = copy.deepcopy(capture["optim_state"])
    authentic = _run_boost(
        authentic_runner,
        capture["vp"],
        capture["gp"],
        rng_state,
    )
    rebuilt_runner = rebuilt["runner"]
    rebuilt_runner.options = rebuilt["options"]
    rebuilt_runner.optim_state = rebuilt["optim_state"]
    reconstructed = _run_boost(
        rebuilt_runner, rebuilt["vp"], rebuilt["gp"], rng_state
    )

    comparisons = {}
    for name in ("candidate", "selected"):
        left, right = authentic[name], reconstructed[name]
        comparisons[name] = {
            key: _array_error(getattr(right, key), getattr(left, key))
            for key in VP_ARRAY_KEYS
        }
        comparisons[name]["stats"] = {
            key: _array_error(right.stats[key], left.stats[key])
            for key in ("elbo", "elbo_sd")
        }
    return {
        "tag": tag,
        "rng": {
            "kind": "fresh common RNG; no captured RNG field used",
            "base_seed": common_seed,
        },
        "authentic": _boost_summary(authentic, label, seed),
        "reconstructed": _boost_summary(reconstructed, label, seed),
        "comparison": comparisons,
    }


def population_numerics(trace_dir):
    """Recompute the pre-boost GP expected-log-joint SD for every trace."""
    started = time.perf_counter()
    rows = []
    for side_path in sorted(Path(trace_dir).glob("*.json")):
        row_started = time.perf_counter()
        side = json.loads(side_path.read_text(encoding="utf-8"))
        npz_path = side_path.with_suffix(".npz")
        with np.load(npz_path, allow_pickle=False) as trace:
            rebuilt = reconstruct_state(side, trace)
            G, _, varG, _, _ = _gp_log_joint(
                rebuilt["vp"], rebuilt["gp"], False, compute_var=True
            )
            best = rebuilt["best_iter"]
            stored_sd = float(trace["elbo_sd"][best])
        reconstructed_sd = float(np.sqrt(varG))
        delta = reconstructed_sd - stored_sd
        rows.append(
            {
                "tag": side_path.stem,
                "label": side["label"],
                "stored_sd": stored_sd,
                "reconstructed_sd": reconstructed_sd,
                "delta_sd": delta,
                "relative_delta_sd": (
                    delta / stored_sd if stored_sd != 0 else None
                ),
                "G": float(G),
                "wall_s": _seconds(row_started),
            }
        )
    absolute = np.abs([row["delta_sd"] for row in rows])
    return {
        "method": (
            "sqrt(varG) from _gp_log_joint(compute_var=True); the current "
            "ELCBO path has zero entropy variance, so this is the stored "
            "pre-boost elbo_sd quantity"
        ),
        "runs": len(rows),
        "abs_delta_gt_1e-4": int(np.sum(absolute > 1e-4)),
        "abs_delta_gt_1e-3": int(np.sum(absolute > 1e-3)),
        "abs_delta_gt_1e-2": int(np.sum(absolute > 1e-2)),
        "max_abs_delta": float(np.max(absolute)) if len(absolute) else None,
        "rows": rows,
        "wall_s": _seconds(started),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument(
        "--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--replay",
        nargs="*",
        metavar="TAG",
        help="run common-RNG boost checks for tags; no tags means all captures",
    )
    parser.add_argument("--common-seed", type=int, default=20260908)
    parser.add_argument(
        "--population-numerics",
        action="store_true",
        help="recompute GP log-joint SD for all traces (no optimization)",
    )
    args = parser.parse_args(argv)

    started = time.perf_counter()
    capture_paths = _capture_paths(args.capture_root)
    checks = [compare_capture(path, args.trace_dir) for path in capture_paths]
    report = {
        "provenance": runtime_provenance(),
        "trace_only_reconstruction": True,
        "captures_used_only_as_comparison_references": True,
        "population_audit": audit_population(args.trace_dir),
        "state_checks": checks,
    }
    if args.replay is not None:
        selected = set(args.replay)
        replay_paths = [
            path
            for path in capture_paths
            if not selected
            or path.name.removesuffix(".boost_pre.dill") in selected
        ]
        missing = selected - {
            path.name.removesuffix(".boost_pre.dill") for path in replay_paths
        }
        if missing:
            raise ValueError(f"unknown replay tags: {sorted(missing)}")
        report["boost_replays"] = [
            replay_capture(path, args.trace_dir, args.common_seed)
            for path in replay_paths
        ]
    if args.population_numerics:
        report["population_numerics"] = population_numerics(args.trace_dir)
    report["total_wall_s"] = _seconds(started)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
