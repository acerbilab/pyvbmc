"""Frozen-state search experiments using the production sampling controller.

The controller executes through candidate selection and stops before the
function logger can evaluate or record a target value. Experimental searches
replace only selection after the ordinary coarse sieve has been evaluated.
All instrumentation is confined to this developer module.
"""

from __future__ import annotations

import copy
import importlib
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
from noisy_acq_quadrature import make_rule, prepare_rule
from scipy.optimize import minimize

from pyvbmc.acquisition_functions import AcqFcnVIQR
from pyvbmc.function_logger import FunctionLogger
from pyvbmc.timer import main_timer


@dataclass(frozen=True)
class SearchConfig:
    """Settings frozen before a search treatment reaches held-out states."""

    arm: str = "S1"
    sieve_size: int = 1024
    accurate_method: str = "mc"
    accurate_budget: int = 1600
    shortlist_size: int = 8
    max_iterations: int = 50
    max_candidate_rows: int = 1000
    finite_difference_step: float = 1e-4
    gradient_relative_tolerance: float = 0.25
    start_separation: float = 0.5
    ftol: float = 1e-9
    gtol: float = 1e-5
    # Importance nodes of the VIQR estimate: the production Monte Carlo
    # draw, or the package's quasi-Monte Carlo rule switched on through its
    # option, with the frozen node count and the component order of the
    # first Sobol' coordinate ("axis": sorted along the leading axis of the
    # means, the production rule; "index": the VP's own order).
    importance_qmc: bool = False
    importance_qmc_samples: int = 96
    importance_qmc_order: str = "axis"

    def validate(self):
        if self.arm not in {"S0", "S1", "S2", "S3"}:
            raise ValueError("arm must be S0, S1, S2, or S3")
        if self.importance_qmc_order not in {"axis", "index"}:
            raise ValueError("importance_qmc_order must be axis or index")
        if self.importance_qmc:
            if self.arm != "S0":
                raise ValueError(
                    "quasi-Monte Carlo importance nodes are evaluated on the"
                    " production search (S0) only"
                )
            if self.importance_qmc_samples != 96:
                raise ValueError(
                    "the frozen quasi-Monte Carlo node count is 96"
                )
        if self.sieve_size not in {1024, 2048}:
            raise ValueError("experimental sieve size must be 1024 or 2048")
        if not 1 <= self.shortlist_size <= 8:
            raise ValueError("shortlist size must be between one and eight")
        if not 1 <= self.max_iterations <= 50:
            raise ValueError(
                "at most 50 local iterations in total are allowed"
            )
        if not self.shortlist_size <= self.max_candidate_rows <= 1000:
            raise ValueError("candidate-row budget must cover the shortlist")
        if self.accurate_budget <= 0 or self.finite_difference_step <= 0:
            raise ValueError(
                "accurate budget and finite-difference step must be positive"
            )


class _Selected(Exception):
    def __init__(self, x, **record):
        self.x = np.array(x, dtype=np.float64, copy=True).reshape(1, -1)
        self.record = record


class _RefinementStopped(Exception):
    pass


def _copy_state(state, seed):
    private = copy.deepcopy(state)
    private["vp"].rng = np.random.default_rng(int(seed))
    private["pt"] = private["vp"].parameter_transformer
    private["logger"].parameter_transformer = private["pt"]
    return private


def _bounds(state, x):
    dimension = x.size
    lower = np.asarray(state["optim_state"]["lb_search"], dtype=float).reshape(
        -1
    )
    upper = np.asarray(state["optim_state"]["ub_search"], dtype=float).reshape(
        -1
    )
    if lower.shape != (dimension,) or upper.shape != (dimension,):
        raise ValueError(
            "search bounds must each have one entry per dimension"
        )
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise _RefinementStopped("nonfinite_search_bounds")
    lower, upper = np.minimum(lower, x), np.maximum(upper, x)
    if np.any(lower >= upper):
        raise _RefinementStopped("degenerate_search_bounds")
    return lower, upper


def _coordinate_scales(vp):
    _, covariance = vp.moments(orig_flag=False, cov_flag=True)
    scales = np.sqrt(np.diag(np.atleast_2d(covariance)))
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise _RefinementStopped("invalid_coordinate_scales")
    return scales


def _refine(state, evaluator, points, scores, repeat_flags, config):
    """Refine a shortlist with shared iteration and candidate-row budgets."""
    best = int(np.argmin(scores))
    selected = points[best].copy()
    best_score = float(scores[best])
    record = {
        "accurate_candidate_rows": len(points),
        "local_iterations": 0,
        "local_runs": [],
        "fallback_reason": None,
        "refinement_stop_reason": None,
        "coarse_fallback_score": best_score,
        "selected_accurate_score": best_score,
        "selected_shortlist_index": best,
    }
    if config.arm == "S1":
        return selected, record
    if np.any(state["optim_state"].get("integer_vars")):
        record["fallback_reason"] = "integer_variables"
        return selected, record
    if repeat_flags[best]:
        record["fallback_reason"] = "selected_repeat"
        return selected, record
    try:
        scales = _coordinate_scales(state["vp"])
        lower, upper = _bounds(state, selected)
    except _RefinementStopped as error:
        record["fallback_reason"] = str(error)
        return selected, record

    # The entire penalized log objective receives the same fixed positive
    # affine transform throughout every local start in this selection.
    offset = best_score
    finite_scores = scores[np.isfinite(scores)]
    objective_scale = max(float(np.ptp(finite_scores)), 1e-3)
    record["objective_offset"] = offset
    record["objective_scale"] = objective_scale
    record["coordinate_scales"] = scales.tolist()
    starts = [best]
    if config.arm == "S3":
        for index in np.argsort(scores, kind="stable"):
            if (
                index == best
                or repeat_flags[index]
                or not np.isfinite(scores[index])
            ):
                continue
            distances = np.linalg.norm(
                (points[starts] - points[index]) / scales, axis=1
            )
            if np.all(distances >= config.start_separation):
                starts.append(int(index))
            if len(starts) == 4:
                break

    def evaluate(z):
        z = np.atleast_2d(z)
        count = z.shape[0]
        if (
            record["accurate_candidate_rows"] + count
            > config.max_candidate_rows
        ):
            raise _RefinementStopped("candidate_row_budget")
        record["accurate_candidate_rows"] += count
        values = evaluator.score(z * scales, diagnostics=False)
        if not np.all(values["valid"]):
            raise _RefinementStopped("invalid_local_score")
        return (values["full_score"] - offset) / objective_scale

    lb, ub = lower / scales, upper / scales

    def value_gradient(z, step):
        plus = np.minimum(z + np.eye(z.size) * step, ub)
        minus = np.maximum(z - np.eye(z.size) * step, lb)
        widths = np.diag(plus - minus)
        if np.any(widths <= 0):
            raise _RefinementStopped("zero_difference_width")
        values = evaluate(np.vstack((z, plus, minus)))
        gradient = (values[1 : 1 + z.size] - values[1 + z.size :]) / widths
        if not np.all(np.isfinite(gradient)):
            raise _RefinementStopped("nonfinite_gradient")
        return float(values[0]), gradient

    for index in starts:
        remaining_iterations = (
            config.max_iterations - record["local_iterations"]
        )
        if remaining_iterations <= 0:
            record["refinement_stop_reason"] = "iteration_budget"
            break
        run = {
            "shortlist_index": index,
            "status": None,
            "iterations": 0,
            "iteration_limit": remaining_iterations,
        }
        record["local_runs"].append(run)

        def count_iteration(_):
            if record["local_iterations"] >= config.max_iterations:
                raise _RefinementStopped("iteration_budget")
            run["iterations"] += 1
            record["local_iterations"] += 1

        try:
            lower, upper = _bounds(state, points[index])
            lb, ub = lower / scales, upper / scales
            start = points[index] / scales
            _, gradient = value_gradient(start, config.finite_difference_step)
            _, half_gradient = value_gradient(
                start, config.finite_difference_step / 2
            )
            discrepancy = float(np.linalg.norm(gradient - half_gradient))
            reference = max(float(np.linalg.norm(half_gradient)), config.gtol)
            run["step_halving_relative_difference"] = discrepancy / reference
            if discrepancy > config.gradient_relative_tolerance * reference:
                raise _RefinementStopped("finite_difference_instability")
            result = minimize(
                lambda z: value_gradient(z, config.finite_difference_step),
                start,
                method="L-BFGS-B",
                jac=True,
                bounds=list(zip(lb, ub)),
                callback=count_iteration,
                options={
                    "maxiter": remaining_iterations,
                    "ftol": config.ftol,
                    "gtol": config.gtol,
                },
            )
            value = float(evaluate(result.x)[0] * objective_scale + offset)
            run.update(
                status="evaluated",
                solver_success=bool(result.success),
                message=str(result.message),
                solver_reported_iterations=int(result.nit),
                score=value,
            )
            if value < best_score:
                selected = result.x * scales
                best_score = value
                record["selected_shortlist_index"] = None
        except _RefinementStopped as error:
            run["status"] = str(error)
            record["fallback_reason"] = str(error)
            selected = points[best].copy()
            best_score = float(scores[best])
            record["selected_shortlist_index"] = best
            break
    record["selected_accurate_score"] = best_score
    return selected, record


@contextmanager
def prepare_selection(
    state, config, *, search_seed, accurate_seed, retain_panel=True
):
    """Prepare a private search and yield its zero-argument operation.

    S0 executes production ``active_sample`` until its first logger call.
    S1--S3 execute the same setup and candidate generator, then use an
    independent fixed integration rule for shortlist selection/refinement.
    The elapsed time includes setup, node generation, cache construction,
    sieve evaluation and refinement. State copying and patch setup are
    outside that interval. Judge evaluations are performed by the caller.
    """
    config.validate()
    private = _copy_state(state, search_seed)
    active = importlib.import_module("pyvbmc.vbmc.active_sample")
    options = private["options"]
    acquisition_functions = options["search_acq_fcn"]
    if len(acquisition_functions) != 1 or (
        type(acquisition_functions[0]) is not AcqFcnVIQR
        or acquisition_functions[0].loss != "iqr"
    ):
        raise ValueError(
            "search experiment requires the standard VIQR acquisition"
        )
    if active._selection_policy_callback is not None:
        raise ValueError(
            "a selection policy is installed; the frozen-state search runs"
            " the production controller unmodified"
        )
    if options.get("active_importance_sampling_qmc", False):
        raise ValueError(
            "captured options already request quasi-Monte Carlo importance"
            " nodes; the frozen-state search expects the production Monte"
            " Carlo draw as its baseline"
        )
    evaluation = {
        "K": private["vp"].K,
        "n_vars": private["vp"].D,
        "D": private["vp"].D,
    }
    if options.eval("active_importance_sampling_mcmc_samples", evaluation) != (
        100
    ):
        raise ValueError(
            "search experiment requires the production 100-node coarse rule"
        )
    expected_nodes = 100
    if config.importance_qmc:
        options.__setitem__("active_importance_sampling_qmc", True, force=True)
        options.__setitem__(
            "active_importance_sampling_qmc_samples",
            int(config.importance_qmc_samples),
            force=True,
        )
        expected_nodes = int(config.importance_qmc_samples)
        if (
            options.eval("active_importance_sampling_qmc_samples", evaluation)
            != expected_nodes
        ):
            raise ValueError(
                "quasi-Monte Carlo node count did not reach the private"
                " options"
            )
    if config.arm != "S0":
        options.__setitem__("ns_search", config.sieve_size, force=True)
    importance = importlib.import_module(
        "pyvbmc.vbmc.active_importance_sampling"
    )
    gp, vp, logger, optim = (
        private[key] for key in ("gp", "vp", "logger", "optim_state")
    )
    original_call = AcqFcnVIQR.__call__
    original_search = active._get_search_points
    trace = {"coarse_candidate_rows": 0, "production_candidate_rows": 0}
    saved_timer = copy.deepcopy(main_timer.__dict__)

    def search_points(*args, **kwargs):
        X, indices = original_search(*args, **kwargs)
        trace["generated_count"] = len(X)
        trace["cache_indices"] = np.array(indices, copy=True)
        return X, indices

    def stop_at_target(self, x, *args, **kwargs):
        if self is not logger:
            raise RuntimeError("unexpected logger during frozen-state search")
        cache_index = np.nan
        if args:
            row = trace["coarse_winner_index"]
            repeat_count = (
                trace["coarse_candidate_rows"] - trace["generated_count"]
            )
            if row >= repeat_count:
                cache_index = trace["cache_indices"][row - repeat_count]
        raise _Selected(x, target_called=False, cache_index=cache_index)

    def reject_fit(*args, **kwargs):
        raise RuntimeError("a frozen-state search attempted to refit the GP")

    def acquisition_call(acq, X, call_gp, call_vp, call_logger, call_optim):
        trace["production_candidate_rows"] += np.atleast_2d(X).shape[0]
        values = original_call(
            acq, X, call_gp, call_vp, call_logger, call_optim
        )
        if trace["coarse_candidate_rows"]:
            return values
        trace["coarse_candidate_rows"] = len(X)
        trace["coarse_winner_index"] = int(np.argmin(values))
        trace["coarse_winner"] = np.array(X[np.argmin(values)], copy=True)
        trace["coarse_minimum"] = float(np.min(values))
        trace["importance_node_count"] = int(
            np.asarray(call_optim["active_importance_sampling"]["X"]).shape[0]
        )
        if trace["importance_node_count"] != expected_nodes:
            raise RuntimeError(
                f"the selection used {trace['importance_node_count']}"
                f" importance nodes where its arm prescribes {expected_nodes}"
            )
        if retain_panel:
            trace["coarse_candidates"] = np.array(X, copy=True)
            trace["coarse_scores"] = np.array(values, copy=True)
            trace["coarse_nodes"] = np.array(
                call_optim["active_importance_sampling"]["X"], copy=True
            )
        if config.arm == "S0":
            return values
        if not np.any(np.isfinite(values)):
            raise RuntimeError("no finite coarse candidate")
        order = np.argsort(values, kind="stable")[: config.shortlist_size]
        points = np.array(X[order], copy=True)
        repeat_count = len(X) - trace["generated_count"]
        repeated = order < repeat_count
        if np.any(repeated):
            points[repeated] = logger.X[logger.X_flag][order[repeated]]
        rule = make_rule(
            vp, config.accurate_method, config.accurate_budget, accurate_seed
        )
        if not rule["available"]:
            raise RuntimeError(
                "accurate rule unavailable at its frozen budget"
            )
        evaluator = prepare_rule(private, rule)
        score = evaluator.score(points, diagnostics=False)
        if not np.any(score["valid"]):
            raise RuntimeError("no valid accurately rescored candidate")
        accurate = np.where(score["valid"], score["full_score"], np.inf)
        chosen, detail = _refine(
            private, evaluator, points, accurate, repeated, config
        )
        trace.update(detail)
        trace["accurate_rule_metadata"] = rule["metadata"]
        trace["evaluator_metadata"] = dict(evaluator.metadata)
        trace["shortlist_indices"] = order
        trace["shortlist_scores"] = accurate
        shortlist_index = detail["selected_shortlist_index"]
        if shortlist_index is None:
            cache_index = np.nan
        else:
            row = int(order[shortlist_index])
            cache_index = (
                np.nan
                if row < repeat_count
                else trace["cache_indices"][row - repeat_count]
            )
        raise _Selected(chosen, target_called=False, cache_index=cache_index)

    try:
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(active, "_get_search_points", search_points)
            )
            stack.enter_context(patch.object(active, "train_gp", reject_fit))
            stack.enter_context(
                patch.object(active, "reupdate_gp", reject_fit)
            )
            stack.enter_context(
                patch.object(FunctionLogger, "__call__", stop_at_target)
            )
            stack.enter_context(
                patch.object(FunctionLogger, "add", stop_at_target)
            )
            stack.enter_context(
                patch.object(AcqFcnVIQR, "__call__", acquisition_call)
            )
            if config.importance_qmc:
                stack.enter_context(
                    patch.object(
                        importance,
                        "_QMC_COMPONENT_ORDER",
                        config.importance_qmc_order,
                    )
                )
            executed = False

            def execute():
                nonlocal executed
                if executed:
                    raise RuntimeError(
                        "a prepared search can execute only once"
                    )
                executed = True
                started = time.perf_counter()
                try:
                    active.active_sample(
                        gp,
                        1,
                        optim,
                        logger,
                        {"r_index": [np.inf]},
                        vp,
                        options,
                    )
                except _Selected as selected:
                    elapsed = time.perf_counter() - started
                    return {
                        "arm": config.arm,
                        "selected": selected.x,
                        "elapsed_seconds": elapsed,
                        "search_seed": int(search_seed),
                        "accurate_seed": int(accurate_seed),
                        **trace,
                        **selected.record,
                    }
                raise RuntimeError(
                    "sampling controller returned without selecting a point"
                )

            yield execute
    finally:
        main_timer.__dict__.clear()
        main_timer.__dict__.update(saved_timer)


def select_candidate(
    state, config, *, search_seed, accurate_seed, retain_panel=True
):
    """Execute one prepared frozen-state search and return its selected point."""
    with prepare_selection(
        state,
        config,
        search_seed=search_seed,
        accurate_seed=accurate_seed,
        retain_panel=retain_panel,
    ) as operation:
        return operation()
