"""Codec and replay helpers for authentic GP-training history captures."""

import copy
import importlib
import inspect
import platform
from unittest.mock import patch

import gpyreg as gpr
import numpy as np

from pyvbmc.testing.oracles._state import (
    build_logger,
    build_options,
    build_transformer,
)
from pyvbmc.vbmc.gaussian_process_train import (
    _get_gp_training_options,
    _get_hyp_cov,
    train_gp,
)
from pyvbmc.vbmc.iteration_history import IterationHistory

OPTION_KEYS = (
    "hyp_run_weight",
    "fun_evals_per_iter",
    "hpd_frac",
    "tol_gp_noise",
    "noise_size",
    "upper_gp_length_factor",
    "gp_quadratic_mean_bound",
    "tol_sd",
    "gp_length_prior_mean",
    "gp_length_prior_std",
    "ns_gp_max",
    "ns_gp_max_warmup",
    "ns_gp_max_main",
    "stable_gp_sampling",
    "stable_gp_vp_k",
    "stable_gp_samples",
    "gp_sample_thin",
    "gp_train_init_method",
    "gp_tol_opt",
    "gp_tol_opt_mcmc",
    "gp_hyp_sampler",
    "gp_sample_widths",
    "cov_sample_thresh",
    "gp_train_n_init",
    "gp_train_n_init_final",
    "fun_eval_start",
    "max_fun_evals",
    "gp_retrain_threshold",
    "weighted_hyp_cov",
    "tol_skl",
    "tol_cov_weight",
    "bounded_transform",
)


class _HistoryGP:
    def __init__(self, hyp):
        self.hyp = np.array(hyp, copy=True)

    def get_hyperparameters(self, as_array=False):
        assert as_array
        return self.hyp.copy()


def _logger_tree(logger):
    n = logger.Xn + 1
    value = lambda name: np.array(getattr(logger, name)[:n], copy=True)
    return {
        "D": logger.D,
        "noise_flag": logger.noise_flag,
        "uncertainty_handling_level": logger.uncertainty_handling_level,
        "cache_size": logger.X_flag.shape[0],
        "Xn": logger.Xn,
        "func_count": logger.func_count,
        "cache_count": logger.cache_count,
        "y_max": logger.y_max,
        "total_fun_eval_time": logger.total_fun_eval_time,
        "X_orig": value("X_orig"),
        "y_orig": value("y_orig"),
        "X": value("X"),
        "y": value("y"),
        "S": value("S") if logger.noise_flag else None,
        "n_evals": value("n_evals"),
        "X_flag": value("X_flag"),
        "fun_eval_time": value("fun_eval_time"),
    }


def _transformer_tree(pt, D, plb_tran, pub_tran, transform_type):
    return {
        "D": D,
        "lb_orig": np.array(pt.lb_orig, copy=True),
        "ub_orig": np.array(pt.ub_orig, copy=True),
        "plb_orig": np.array(pt.inverse(plb_tran), copy=True),
        "pub_orig": np.array(pt.inverse(pub_tran), copy=True),
        "mu": np.array(pt.mu, copy=True),
        "delta": np.array(pt.delta, copy=True),
        "type": np.array(pt.type, copy=True),
        "R_mat": None if pt.R_mat is None else np.array(pt.R_mat, copy=True),
        "scale": None if pt.scale is None else np.array(pt.scale, copy=True),
        "transform_type": transform_type,
    }


def capture_inputs(
    hyp_dict,
    optim_state,
    logger,
    history,
    options,
    plb_tran,
    pub_tran,
    rng,
):
    """Copy every input needed by ``train_gp`` before it can be mutated."""
    iteration = int(optim_state["iter"])
    h = {"gp": [], "gp_hyp_full": [], "sKL": [], "r_index": []}
    for i in range(iteration):
        h["gp"].append(
            np.array(history["gp"][i].get_hyperparameters(as_array=True))
        )
        h["gp_hyp_full"].append(np.array(history["gp_hyp_full"][i]))
        h["sKL"].append(float(history["sKL"][i]))
        h["r_index"].append(float(history["r_index"][i]))
    pt = logger.parameter_transformer
    return {
        "hyp_dict": copy.deepcopy(hyp_dict),
        "optim_state": copy.deepcopy(optim_state),
        "logger": _logger_tree(logger),
        "pt": _transformer_tree(
            pt, logger.D, plb_tran, pub_tran, options["bounded_transform"]
        ),
        "history": h,
        "options": {k: copy.deepcopy(options[k]) for k in OPTION_KEYS},
        "plb_tran": np.array(plb_tran, copy=True),
        "pub_tran": np.array(pub_tran, copy=True),
        "rng": {
            "bit_generator": type(rng.bit_generator).__name__,
            "state": copy.deepcopy(rng.bit_generator.state),
        },
    }


def build_inputs(pre):
    pt = build_transformer(pre["pt"])
    logger = build_logger(pre["logger"], pt)
    options = build_options({}, pre["pt"]["D"])
    for key, value in pre["options"].items():
        options.__setitem__(key, copy.deepcopy(value), force=True)
    history = IterationHistory(list(pre["history"]))
    for i, hyp in enumerate(pre["history"]["gp"]):
        history.record("gp", _HistoryGP(hyp), i)
        for key in ("gp_hyp_full", "sKL", "r_index"):
            history.record(key, pre["history"][key][i], i)
    bitgen = getattr(np.random, pre["rng"]["bit_generator"])()
    bitgen.state = copy.deepcopy(pre["rng"]["state"])
    return (
        copy.deepcopy(pre["hyp_dict"]),
        copy.deepcopy(pre["optim_state"]),
        logger,
        history,
        options,
        np.array(pre["plb_tran"], copy=True),
        np.array(pre["pub_tran"], copy=True),
        np.random.Generator(bitgen),
    )


def run_observed(train, args):
    """Observe proposed and effective widths during one sampled GP fit.

    ``gpyreg.GP.fit`` receives PyVBMC's proposed widths, then caps them with
    a locally computed ``widths_default`` before constructing SliceSampler.
    The constructor spy reads that named local from its caller frame. This is
    intentionally pinned-source fixture instrumentation, not production API.
    """
    fit_seen = []
    sampler_seen = []
    original_fit = gpr.GP.fit
    gp_module = importlib.import_module("gpyreg.gaussian_process")
    original_sampler = gp_module.SliceSampler

    def fit_spy(self, *fit_args, **fit_kwargs):
        fit_seen.append(copy.deepcopy(fit_kwargs["options"]))
        return original_fit(self, *fit_args, **fit_kwargs)

    def sampler_spy(*sampler_args, **sampler_kwargs):
        caller = inspect.currentframe().f_back
        widths_default = caller.f_locals.get("widths_default")
        if widths_default is None:
            raise RuntimeError(
                "gpyreg.GP.fit no longer exposes widths_default"
            )
        sampler_seen.append(
            {
                "effective_widths": np.array(sampler_args[2], copy=True),
                "widths_default": np.array(widths_default, copy=True),
            }
        )
        return original_sampler(*sampler_args, **sampler_kwargs)

    with (
        patch.object(gpr.GP, "fit", fit_spy),
        patch.object(gp_module, "SliceSampler", sampler_spy),
    ):
        result = train(*args[:-1], rng=args[-1])
    if len(fit_seen) != 1:
        raise RuntimeError(
            f"expected one GP.fit call, observed {len(fit_seen)}"
        )
    if result[1] > 0 and len(sampler_seen) != 1:
        raise RuntimeError(
            f"expected one SliceSampler call, observed {len(sampler_seen)}"
        )
    observation = {
        "gp_fit_widths": fit_seen[0]["widths"],
        "effective_widths": None,
        "widths_default": None,
    }
    if sampler_seen:
        observation.update(sampler_seen[0])
    return result, observation


def fit_outputs(result):
    gp, gp_s_N, sn2_hpd, hyp_dict = result
    mu, var = gp.predict(gp.X, gp.y, gp.s2, add_noise=False)
    out = {
        "gp_hyp": gp.get_hyperparameters(as_array=True),
        "gp_s_N": np.asarray(float(gp_s_N)),
        "sn2_hpd": np.asarray(float(sn2_hpd)),
        "prediction_mean": mu,
        "prediction_variance": var,
    }
    for key in ("hyp", "full", "logp", "run_cov"):
        if hyp_dict.get(key) is not None:
            out[f"hyp_dict_{key}"] = np.asarray(hyp_dict[key])
    return out


def portable_outputs(pre, hyp_n, gp_s_N):
    args = build_inputs(pre)
    hyp_dict, optim_state, _, history, options = args[:5]
    covariance = _get_hyp_cov(
        optim_state, history, options, hyp_dict, hyp_n=hyp_n
    )
    training = _get_gp_training_options(
        optim_state,
        history,
        options,
        hyp_dict,
        gp_s_N,
        hyp_n=hyp_n,
    )
    return {
        "covariance": covariance,
        "gp_fit_widths": training["widths"],
    }


def history_block_summary(pre, hyp_n):
    """Describe the compatible history blocks and their total weights."""
    options = pre["options"]
    iteration = int(pre["optim_state"]["iter"])
    weight = 1.0
    indices, counts, weights = [], [], []
    for offset in range(iteration):
        if offset:
            skl = pre["history"]["sKL"][iteration - offset]
            ratio = skl / (options["tol_skl"] * options["fun_evals_per_iter"])
            decay = max(1.0, np.log(ratio)) if ratio > 0 else 1.0
            weight *= options["hyp_run_weight"] ** (
                options["fun_evals_per_iter"] * decay
            )
        if weight < options["tol_cov_weight"]:
            break
        index = iteration - 1 - offset
        block = pre["history"]["gp_hyp_full"][index]
        if block.ndim == 2 and block.shape[1] == hyp_n:
            indices.append(index)
            counts.append(block.shape[0])
            weights.append(weight)
    return {"indices": indices, "sample_counts": counts, "weights": weights}


def replay(snapshot):
    result, observation = run_observed(train_gp, build_inputs(snapshot["pre"]))
    return fit_outputs(result), observation


def same_platform(snapshot):
    return platform.platform() == snapshot["meta"]["platform"]
