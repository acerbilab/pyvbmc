"""Snapshot codec and rebuilders for the stage-level oracles.

A *snapshot* is the algorithm state of one VBMC iteration stored as plain
arrays: a ``.npz`` file (``allow_pickle=False``) holding every array, and a
JSON sidecar holding scalars, strings, booleans, ``None``, lists and the
tree structure, with the string marker ``"@array:<key>"`` where an array
belongs. Nothing in a snapshot depends on the layout of any Python class,
so the fixtures survive refactors and can be consumed by another backend.

``snapshot_from_vbmc`` takes the state out of a live :class:`VBMC` (or one
of its recorded iterations); ``build_*`` rebuild the objects through the
public constructors; ``save_snapshot`` / ``load_snapshot`` do the files.
The generator (``dev/scripts/make_oracle_fixtures.py``) and the tests
(``test_oracles.py``) share this module so that the reference outputs and
the checked outputs are computed from identically rebuilt state.
"""

import copy
import json
import numbers
from pathlib import Path

import gpyreg as gpr
import numpy as np

from pyvbmc import acquisition_functions
from pyvbmc.acquisition_functions import AbstractAcqFcn
from pyvbmc.function_logger import FunctionLogger
from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.vbmc.gaussian_process_train import (
    _cov_identifier_to_covariance_function,
    _meanfun_name_to_mean_function,
)
from pyvbmc.vbmc.options import Options

ARRAY_MARKER = "@@npz:"
FLOAT_TAG = "@float"
BASIC_OPTIONS = "option_configs/basic_vbmc_options.ini"
ADVANCED_OPTIONS = "option_configs/advanced_vbmc_options.ini"
# Generator for a rebuilt VP when the caller does not supply one, so that
# rebuilding never touches NumPy's global legacy state (``get_rng(None)``
# would draw its seed from it). Oracles that need randomness reseed anyway.
STATE_SEED = 0


# --------------------------------------------------------------------------
# Typed codec: arbitrary trees of dict / list / scalar / ndarray / None
# --------------------------------------------------------------------------


def _encode_float(x):
    if np.isfinite(x):
        return float(x)
    # Strict JSON has no inf/nan; tag them so any parser can read the file.
    return {FLOAT_TAG: "nan" if np.isnan(x) else ("inf" if x > 0 else "-inf")}


def encode(obj, prefix, arrays):
    """Encode ``obj`` into a JSON-able tree; arrays go into ``arrays``.

    Raises ``TypeError`` on any type it does not know, so that new state
    types are added to the codec deliberately instead of being dropped.
    Non-finite floats become ``{"@float": "inf" | "-inf" | "nan"}`` so the
    sidecar is strict JSON.
    """
    if isinstance(obj, np.ndarray):
        arrays[prefix] = obj
        return ARRAY_MARKER + prefix
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, (bool, str)) or obj is None:
        return obj
    if isinstance(obj, numbers.Integral):
        return int(obj)
    if isinstance(obj, numbers.Real):
        return _encode_float(float(obj))
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(f"{prefix}: non-string dict key {k!r}")
            out[k] = encode(v, f"{prefix}/{k}", arrays)
        return out
    if isinstance(obj, (list, tuple)):
        return [encode(v, f"{prefix}/{i}", arrays) for i, v in enumerate(obj)]
    raise TypeError(f"cannot encode {prefix}: {type(obj).__name__}")


def decode(tree, arrays):
    """Inverse of :func:`encode`."""
    if isinstance(tree, str) and tree.startswith(ARRAY_MARKER):
        key = tree[len(ARRAY_MARKER) :]
        if key not in arrays:
            raise KeyError(f"array marker {tree!r} has no array in the .npz")
        return np.array(arrays[key])
    if isinstance(tree, dict):
        if set(tree) == {FLOAT_TAG}:
            return float(tree[FLOAT_TAG])
        return {k: decode(v, arrays) for k, v in tree.items()}
    if isinstance(tree, list):
        return [decode(v, arrays) for v in tree]
    return tree


# --------------------------------------------------------------------------
# Snapshot extraction
# --------------------------------------------------------------------------


def _live(a, n):
    return None if a is None else np.array(a[:n])


def _encode_user_options(user_options):
    """User options as JSON; acquisition objects become class names.

    Array-valued user options are refused (the sidecar holds no arrays),
    so that nothing is dropped silently.
    """
    out = {}
    for k, v in (user_options or {}).items():
        if isinstance(v, (list, tuple)) and any(
            isinstance(a, AbstractAcqFcn) for a in v
        ):
            out[k] = {"@acq": [type(a).__name__ for a in v]}
        elif isinstance(v, AbstractAcqFcn):
            out[k] = {"@acq": [type(v).__name__]}
        elif isinstance(v, np.ndarray):
            raise TypeError(f"user option {k!r} is an array; not supported")
        else:
            out[k] = encode(v, f"options/{k}", {})  # scalars only
    return out


def _decode_user_options(tree):
    out = {}
    for k, v in tree.items():
        if isinstance(v, dict) and "@acq" in v:
            out[k] = [getattr(acquisition_functions, n)() for n in v["@acq"]]
        else:
            out[k] = v
    return out


def snapshot_from_vbmc(vbmc, iteration=None, meta=None):
    """Extract a snapshot from a :class:`VBMC` instance.

    With ``iteration=None`` the live ``vbmc.vp``, ``vbmc.gp``,
    ``vbmc.function_logger`` and ``vbmc.optim_state`` are used; otherwise
    the deep copies recorded in ``vbmc.iteration_history`` at that
    iteration. Returns ``(arrays, tree)`` ready for :func:`save_snapshot`;
    ``tree["meta"]`` is ``meta`` plus what can be read off the instance.
    """
    if iteration is None:
        vp, gp, fl, os_ = (
            vbmc.vp,
            vbmc.gp,
            vbmc.function_logger,
            vbmc.optim_state,
        )
    else:
        h = vbmc.iteration_history
        vp, gp, fl, os_ = (
            h["vp"][iteration],
            h["gp"][iteration],
            h["function_logger"][iteration],
            h["optim_state"][iteration],
        )
    return snapshot_from_objects(
        vp, gp, fl, os_, vbmc.options, meta=meta, iteration=iteration
    )


def snapshot_from_objects(
    vp, gp, function_logger, optim_state, options, meta=None, iteration=None
):
    arrays = {}
    tree = {}
    pt = vp.parameter_transformer
    D = vp.D

    # --- GP -----------------------------------------------------------
    hyp = np.array([p.hyp for p in gp.posteriors])
    tree["gp"] = {
        "X": encode(np.array(gp.X), "gp/X", arrays),
        "y": encode(np.array(gp.y), "gp/y", arrays),
        "s2": None
        if gp.s2 is None
        else encode(np.array(gp.s2), "gp/s2", arrays),
        "hyp": encode(hyp, "gp/hyp", arrays),
        "Ns": int(hyp.shape[0]),
        "cov_fun": encode(optim_state["gp_cov_fun"], "gp/cov_fun", arrays),
        "mean_fun": optim_state["gp_mean_fun"],
        # The noise function's own switches (constant, user-provided or
        # scaled user-provided, output-dependent), not the option triple
        # they were derived from: the two can disagree.
        "noise_parameters": [int(v) for v in np.ravel(gp.noise.parameters)],
        "noise_fun": [int(v) for v in optim_state["gp_noise_fun"]],
    }

    # --- transformer ----------------------------------------------------
    tree["pt"] = {
        "D": int(D),
        "lb_orig": encode(np.array(pt.lb_orig), "pt/lb_orig", arrays),
        "ub_orig": encode(np.array(pt.ub_orig), "pt/ub_orig", arrays),
        "plb_orig": encode(
            np.array(optim_state["plb_orig"]), "pt/plb_orig", arrays
        ),
        "pub_orig": encode(
            np.array(optim_state["pub_orig"]), "pt/pub_orig", arrays
        ),
        "mu": encode(np.array(pt.mu), "pt/mu", arrays),
        "delta": encode(np.array(pt.delta), "pt/delta", arrays),
        "type": encode(np.array(pt.type), "pt/type", arrays),
        "R_mat": None
        if pt.R_mat is None
        else encode(np.array(pt.R_mat), "pt/R_mat", arrays),
        "scale": None
        if pt.scale is None
        else encode(np.array(pt.scale), "pt/scale", arrays),
        "transform_type": options["bounded_transform"],
    }

    # --- VP -------------------------------------------------------------
    tree["vp"] = {
        "D": int(D),
        "K": int(vp.K),
        "w": encode(np.array(vp.w), "vp/w", arrays),
        "eta": encode(np.array(vp.eta), "vp/eta", arrays),
        "mu": encode(np.array(vp.mu), "vp/mu", arrays),
        "sigma": encode(np.array(vp.sigma), "vp/sigma", arrays),
        "lambd": encode(np.array(vp.lambd), "vp/lambd", arrays),
        "optimize_mu": bool(vp.optimize_mu),
        "optimize_sigma": bool(vp.optimize_sigma),
        "optimize_lambd": bool(vp.optimize_lambd),
        "optimize_weights": bool(vp.optimize_weights),
        "stats": encode(vp.stats, "vp/stats", arrays),
        "bounds": encode(vp.bounds, "vp/bounds", arrays),
    }

    # --- function logger (rows up to the last filled one) -----------------
    n = function_logger.Xn + 1
    fl = function_logger
    tree["logger"] = {
        "D": int(D),
        "noise_flag": bool(fl.noise_flag),
        "uncertainty_handling_level": int(fl.uncertainty_handling_level),
        "cache_size": int(fl.X_flag.shape[0]),
        "Xn": int(fl.Xn),
        "func_count": int(fl.func_count),
        "cache_count": int(fl.cache_count),
        "y_max": float(fl.y_max),
        "total_fun_eval_time": float(fl.total_fun_eval_time),
        "X_orig": encode(_live(fl.X_orig, n), "logger/X_orig", arrays),
        "y_orig": encode(_live(fl.y_orig, n), "logger/y_orig", arrays),
        "X": encode(_live(fl.X, n), "logger/X", arrays),
        "y": encode(_live(fl.y, n), "logger/y", arrays),
        "S": encode(_live(fl.S, n), "logger/S", arrays)
        if fl.noise_flag
        else None,
        "n_evals": encode(_live(fl.n_evals, n), "logger/n_evals", arrays),
        "X_flag": encode(_live(fl.X_flag, n), "logger/X_flag", arrays),
        "fun_eval_time": encode(
            _live(fl.fun_eval_time, n), "logger/fun_eval_time", arrays
        ),
    }

    # --- optim_state, options, meta ---------------------------------------
    tree["optim_state"] = encode(optim_state, "optim_state", arrays)
    tree["options"] = _encode_user_options(options.user_options)
    tree["meta"] = dict(meta or {})
    tree["meta"]["iteration"] = None if iteration is None else int(iteration)
    tree["cand"] = {}
    tree["ref"] = {}
    return arrays, tree


# --------------------------------------------------------------------------
# Rebuilders (public constructors + attribute assignment)
# --------------------------------------------------------------------------


def build_transformer(d):
    pt = ParameterTransformer(
        d["D"],
        d["lb_orig"],
        d["ub_orig"],
        d["plb_orig"],
        d["pub_orig"],
        transform_type=d["transform_type"],
    )
    pt.mu = np.array(d["mu"])
    pt.delta = np.array(d["delta"])
    pt.type = np.array(d["type"])
    pt.R_mat = None if d["R_mat"] is None else np.array(d["R_mat"])
    pt.scale = None if d["scale"] is None else np.array(d["scale"])
    return pt


def build_vp(d, pt, rng=None):
    vp = VariationalPosterior(
        d["D"],
        d["K"],
        x0=np.zeros((1, d["D"])),
        parameter_transformer=pt,
        rng=np.random.default_rng(STATE_SEED) if rng is None else rng,
    )
    vp.w = np.array(d["w"])
    vp.eta = np.array(d["eta"])
    vp.mu = np.array(d["mu"])
    vp.sigma = np.array(d["sigma"])
    vp.lambd = np.array(d["lambd"])
    vp.optimize_mu = d["optimize_mu"]
    vp.optimize_sigma = d["optimize_sigma"]
    vp.optimize_lambd = d["optimize_lambd"]
    vp.optimize_weights = d["optimize_weights"]
    vp.stats = d["stats"]
    vp.bounds = d["bounds"]
    return vp


def build_gp(d):
    p = d["noise_parameters"]
    noise = gpr.noise_functions.GaussianNoise(
        constant_add=p[0] == 1,
        user_provided_add=p[1] >= 1,
        scale_user_provided=p[1] == 2,
        rectified_linear_output_dependent_add=p[2] == 1,
    )
    if [int(v) for v in np.ravel(noise.parameters)] != list(p):
        raise ValueError(
            f"noise switches {p} not reproducible by GaussianNoise"
        )
    gp = gpr.GP(
        D=d["X"].shape[1],
        covariance=_cov_identifier_to_covariance_function(d["cov_fun"]),
        mean=_meanfun_name_to_mean_function(d["mean_fun"]),
        noise=noise,
    )
    gp.update(
        X_new=np.array(d["X"]),
        y_new=np.array(d["y"]),
        s2_new=None if d["s2"] is None else np.array(d["s2"]),
        compute_posterior=False,
    )
    gp.set_hyperparameters(np.array(d["hyp"]), compute_posterior=True)
    return gp


def _missing_fun(x):
    raise RuntimeError(
        "this rebuilt FunctionLogger has no target; set `logger.fun`"
    )


def build_logger(d, pt, fun=None):
    fl = FunctionLogger(
        fun if fun is not None else _missing_fun,
        d["D"],
        d["noise_flag"],
        d["uncertainty_handling_level"],
        cache_size=d["cache_size"],
        parameter_transformer=pt,
    )
    n = d["Xn"] + 1
    for name in ("X_orig", "y_orig", "X", "y", "n_evals", "fun_eval_time"):
        getattr(fl, name)[:n] = d[name]
    if d["noise_flag"]:
        fl.S[:n] = d["S"]
    fl.X_flag[:n] = d["X_flag"]
    fl.Xn = d["Xn"]
    fl.func_count = d["func_count"]
    fl.cache_count = d["cache_count"]
    fl.y_max = d["y_max"]
    fl.total_fun_eval_time = d["total_fun_eval_time"]
    return fl


def build_options(user_options, D):
    """Mirror ``VBMC.__init__``: basic, advanced, defaults, validation."""
    options = Options(
        BASIC_OPTIONS,
        evaluation_parameters={"D": D},
        user_options=_decode_user_options(user_options),
    )
    options.load_options_file(ADVANCED_OPTIONS, evaluation_parameters={"D": D})
    options.update_defaults()
    options.validate_option_names([BASIC_OPTIONS, ADVANCED_OPTIONS])
    return options


def build_state(snap, fun=None, rng=None):
    """Rebuild every object of a decoded snapshot.

    Returns a dict with ``pt, vp, gp, logger, optim_state, options, cand,
    ref, meta``. ``fun`` is attached to the logger (needed only by the
    active-sampling oracle); ``rng`` seeds the VP's generator.
    """
    # Work on a private copy: oracles may mutate what they are handed
    # (`get_bounds` accumulates into `vp.bounds`, `prepare_gp_for_acq`
    # writes into `optim_state`), and the caller may reuse the snapshot.
    snap = copy.deepcopy(snap)
    pt = build_transformer(snap["pt"])
    vp = build_vp(snap["vp"], pt, rng=rng)
    gp = build_gp(snap["gp"])
    logger = build_logger(snap["logger"], pt, fun=fun)
    options = build_options(snap["options"], snap["pt"]["D"])
    return {
        "pt": pt,
        "vp": vp,
        "gp": gp,
        "logger": logger,
        "optim_state": snap["optim_state"],
        "options": options,
        "cand": snap["cand"],
        "ref": snap["ref"],
        "meta": snap["meta"],
    }


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------


def _files(path):
    # Not `with_suffix`: a name such as ``x_noise0.5`` would be truncated.
    path = Path(path)
    return path.parent / (path.name + ".npz"), path.parent / (
        path.name + ".json"
    )


def save_snapshot(path, arrays, tree):
    """Write ``<path>.npz`` and ``<path>.json`` (strict JSON, LF)."""
    npz, js = _files(path)
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, **arrays)
    with open(js, "w", encoding="utf-8", newline="\n") as f:
        json.dump(tree, f, indent=1, sort_keys=True, allow_nan=False)
        f.write("\n")


def load_snapshot(path):
    """Read a snapshot back into a decoded tree (arrays in place)."""
    npz, js = _files(path)
    tree = json.loads(js.read_text(encoding="utf-8"))
    with np.load(npz, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    return decode(tree, arrays)


def snapshot_names(fixtures_dir):
    """Names of the snapshots present in a directory (sorted)."""
    return sorted(
        p.name[: -len(".json")] for p in Path(fixtures_dir).glob("*.json")
    )
