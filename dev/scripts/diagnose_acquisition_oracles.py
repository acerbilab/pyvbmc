"""Report cross-platform numerical differences in acquisition oracles.

This diagnostic only rebuilds stored snapshots and evaluates deterministic
pointwise quantities.  It does not fit a GP, call a target, or run an
acquisition search.  Run it from the repository root::

    python dev/scripts/diagnose_acquisition_oracles.py

The concise table covers every stored snapshot.  A JSON block then records
the worst ``corr_D5_warped/AcqFcn`` candidates and enough intermediate values
to distinguish GP-prediction drift from the variance penalty and other
acquisition factors.
"""

import contextlib
import io
import json
import platform
import sys
from pathlib import Path

import gpyreg
import numpy as np
import scipy

import pyvbmc
from pyvbmc.acquisition_functions import AcqFcn
from pyvbmc.testing.oracles._oracles import (
    ORACLES,
    cast_outputs,
    compare,
    prepare_gp_for_acq,
)
from pyvbmc.testing.oracles._state import (
    build_state,
    load_snapshot,
    snapshot_names,
)

HERE = Path(__file__).resolve().parent
FIXTURES = HERE.parents[1] / "pyvbmc" / "testing" / "oracles" / "fixtures"
ACQUISITIONS = ("AcqFcn", "AcqFcnVanilla", "AcqFcnNoisy", "AcqFcnLog")
DETAIL_SNAPSHOT = "corr_D5_warped"
DETAIL_COUNT = 8


def _json_value(value):
    """Convert NumPy values to strict-JSON-compatible Python values."""
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else repr(value)
    return value


def _total_variance(f_mu, f_s2):
    """Reproduce ``AbstractAcqFcn.__call__`` summation and shapes."""
    ns = f_mu.shape[1]
    f_bar = np.sum(f_mu, axis=1, keepdims=True) / ns
    var_bar = np.sum(f_s2, axis=1, keepdims=True) / ns
    if ns > 1:
        var_f = np.sum((f_mu - f_bar) ** 2, axis=1, keepdims=True) / (ns - 1)
    else:
        var_f = 0
    return np.ravel(f_bar), np.ravel(var_f + var_bar)


def _penalty(var_tot, tol_var):
    penalty = np.ones_like(var_tot)
    positive = (var_tot < tol_var) & (var_tot > 0)
    penalty[positive] = np.exp(-(tol_var / var_tot[positive] - 1))
    penalty[(var_tot < tol_var) & (var_tot == 0)] = 0.0
    return penalty


def _environment():
    config = io.StringIO()
    with contextlib.redirect_stdout(config):
        np.show_config()
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "gpyreg": getattr(gpyreg, "__version__", "unknown"),
            "pyvbmc": getattr(pyvbmc, "__version__", "unknown"),
        },
        "import_paths": {
            name: str(Path(module.__file__).resolve())
            for name, module in {
                "numpy": np,
                "scipy": scipy,
                "gpyreg": gpyreg,
                "pyvbmc": pyvbmc,
            }.items()
        },
        "numpy_show_config": config.getvalue().strip(),
    }


def _evaluate_all():
    actuals = {}
    print(
        "snapshot                     oracle           max_abs      scaled   ok"
    )
    for name in snapshot_names(FIXTURES):
        snap = load_snapshot(FIXTURES / name)
        for acq_name in ACQUISITIONS:
            oracle_name = f"acq_{acq_name}"
            if oracle_name not in snap["ref"]:
                continue
            oracle = ORACLES[oracle_name]
            state = build_state(snap)
            actual = cast_outputs(
                oracle.fn(state, snap["meta"]["oracle_seed"])
            )
            row = compare(
                snap["ref"][oracle_name], actual, oracle.rtol, oracle.atol
            )[0]
            actuals[(name, oracle_name)] = actual["acq"]
            print(
                f"{name:28s} {acq_name:14s} {row[1]:10.3g}"
                f" {row[2]:11.3g} {str(row[3]):>4s}"
            )
    return actuals


def _corr_details(actuals):
    snap = load_snapshot(FIXTURES / DETAIL_SNAPSHOT)
    required = {
        "gp_predict": {"fmu_samples", "fs2_samples"},
        "vp_pdf": {"pdf", "logpdf"},
        "acq_AcqFcn": {"acq"},
        "acq_AcqFcnLog": {"acq"},
    }
    for oracle, keys in required.items():
        missing = keys - set(snap["ref"].get(oracle, {}))
        if missing:
            raise KeyError(
                f"{DETAIL_SNAPSHOT}/{oracle} lacks {sorted(missing)}"
            )

    state = build_state(snap)
    xs = state["cand"]["Xs"]
    prepare_gp_for_acq(state["gp"], state["logger"], state["optim_state"])
    cur_mu, cur_s2 = state["gp"].predict(xs, separate_samples=True)
    cur_fbar, cur_var = _total_variance(cur_mu, cur_s2)
    cur_pdf = np.ravel(state["vp"].pdf(xs, orig_flag=False))
    cur_logpdf = np.ravel(state["vp"].pdf(xs, orig_flag=False, log_flag=True))

    ref_gp = snap["ref"]["gp_predict"]
    ref_mu, ref_s2 = ref_gp["fmu_samples"], ref_gp["fs2_samples"]
    ref_fbar, ref_var = _total_variance(ref_mu, ref_s2)
    ref_pdf = np.ravel(snap["ref"]["vp_pdf"]["pdf"])
    ref_logpdf = np.ravel(snap["ref"]["vp_pdf"]["logpdf"])
    tol_var = state["optim_state"]["tol_gp_var"]
    cur_penalty, ref_penalty = _penalty(cur_var, tol_var), _penalty(
        ref_var, tol_var
    )

    raw_state = build_state(snap)
    raw_state["optim_state"]["variance_regularized_acq_fcn"] = False
    prepare_gp_for_acq(
        raw_state["gp"], raw_state["logger"], raw_state["optim_state"]
    )
    current_raw = AcqFcn()(
        np.array(raw_state["cand"]["Xs"]),
        raw_state["gp"],
        raw_state["vp"],
        raw_state["logger"],
        raw_state["optim_state"],
    )
    realmin = sys.float_info.min
    z = state["logger"].y_max
    reference_raw = (
        -ref_var * np.exp(ref_fbar - z) * np.maximum(ref_pdf, realmin)
    )
    actual = actuals[(DETAIL_SNAPSHOT, "acq_AcqFcn")]
    reference = np.ravel(snap["ref"]["acq_AcqFcn"]["acq"])
    current_log = actuals[(DETAIL_SNAPSHOT, "acq_AcqFcnLog")]
    reference_log = np.ravel(snap["ref"]["acq_AcqFcnLog"]["acq"])

    finite = np.isfinite(reference) & np.isfinite(actual)
    floor = max(
        float(np.quantile(np.abs(reference[finite]), 0.25)),
        np.finfo(float).tiny,
    )
    scale = np.maximum(np.abs(reference), floor)
    scaled = np.abs(actual - reference) / scale
    worst = np.argsort(scaled)[-DETAIL_COUNT:][::-1]

    variance_only = reference_raw * (cur_var / ref_var) * cur_penalty
    valid_identity = (
        (reference < 0)
        & (actual < 0)
        & (ref_pdf > realmin)
        & (cur_pdf > realmin)
        & (ref_logpdf > np.log(realmin))
        & (cur_logpdf > np.log(realmin))
        & np.isfinite(reference_log)
        & np.isfinite(current_log)
    )

    candidates = []
    for i in worst:
        identity = bool(valid_identity[i])
        candidates.append(
            {
                "index": int(i),
                "Xs": xs[i],
                "reference_acq": reference[i],
                "actual_acq": actual[i],
                "absolute_error": abs(actual[i] - reference[i]),
                "comparison_scale": scale[i],
                "scaled_error": scaled[i],
                "reference_fmu_samples": ref_mu[i],
                "current_fmu_samples": cur_mu[i],
                "reference_fs2_samples": ref_s2[i],
                "current_fs2_samples": cur_s2[i],
                "reference_total_variance": ref_var[i],
                "current_total_variance": cur_var[i],
                "reference_tol_over_variance": tol_var / ref_var[i],
                "current_tol_over_variance": tol_var / cur_var[i],
                "reference_penalty": ref_penalty[i],
                "current_penalty": cur_penalty[i],
                "reference_raw_disabled_acq": reference_raw[i],
                "current_raw_disabled_acq": current_raw[i],
                "reference_pdf": ref_pdf[i],
                "current_pdf": cur_pdf[i],
                "reference_logacq": reference_log[i],
                "current_logacq": current_log[i],
                "variance_only_acq": variance_only[i],
                "nonvariance_residual": actual[i] - variance_only[i],
                "exp_log_identity_valid": identity,
                "reference_exp_log_residual": (
                    reference[i] + np.exp(-reference_log[i])
                    if identity
                    else None
                ),
                "current_exp_log_residual": (
                    actual[i] + np.exp(-current_log[i]) if identity else None
                ),
            }
        )

    expected_current = current_raw * cur_penalty
    expected_reference = reference_raw * ref_penalty
    return {
        "snapshot": DETAIL_SNAPSHOT,
        "limitations": (
            "variance_only_acq holds reference mean/PDF fixed; its residual may "
            "contain mean, PDF, and operation-order differences. Exp/log identity "
            "is reported only where neither PDF path is floored and neither "
            "acquisition underflows."
        ),
        "summary": {
            "candidate_count": len(reference),
            "comparison_floor": floor,
            "regularized_reference_count": int(np.sum(ref_var < tol_var)),
            "production_penalty_exact": bool(
                np.array_equal(actual, expected_current)
            ),
            "production_penalty_max_abs_error": np.max(
                np.abs(actual - expected_current)
            ),
            "reference_formula_max_abs_error": np.max(
                np.abs(reference - expected_reference)
            ),
            "exp_log_identity_valid_count": int(np.sum(valid_identity)),
        },
        "worst_scaled_candidates": candidates,
    }


def main():
    print(
        json.dumps({"environment": _environment()}, indent=2, allow_nan=False)
    )
    actuals = _evaluate_all()
    details = _json_value(_corr_details(actuals))
    print(
        json.dumps(
            {"acquisition_diagnostic": details}, indent=2, allow_nan=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
