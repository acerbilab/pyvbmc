"""Bounded, developer-only calibration of PDF and entropy chunk budgets.

This script gathers machine-local evidence without changing production code,
defaults, caches, algorithmic sample counts, or solver random streams.  Its
source-derived callables deliberately fail closed when the production source
no longer has the exact seams studied here.
"""

from __future__ import annotations

import time

_IMPORT_STARTED = time.perf_counter()

import argparse
import ast
import copy
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import sys
import textwrap
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from types import FunctionType
from typing import Callable

import numpy as np

from pyvbmc.decorators import handle_0D_1D_input
from pyvbmc.entropy.entmc_vbmc import entmc_vbmc
from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.variational_posterior import VariationalPosterior

_IMPORT_SECONDS = time.perf_counter() - _IMPORT_STARTED

DEFAULT_BUDGET = 2**16
CANDIDATE_BUDGETS = tuple(2**power for power in range(14, 19))
SCHEMA_VERSION = 1
SCRIPT_PATH = Path(__file__).resolve()


class SourceGuardError(RuntimeError):
    """The production source is no longer the implementation calibrated."""


@dataclass(frozen=True)
class Workload:
    """One fixed-shape calibration workload."""

    kernel: str
    name: str
    D: int
    K: int
    count: int
    grad_flags: tuple[bool, bool, bool, bool] | None = None
    jacobian_flag: bool = True


def _checked_budget(budget: int) -> int:
    if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)):
        raise ValueError("chunk budget must be a positive integer")
    budget = int(budget)
    if budget <= 0:
        raise ValueError("chunk budget must be a positive integer")
    return budget


def _source(function: Callable) -> str:
    try:
        return textwrap.dedent(inspect.getsource(function))
    except (OSError, TypeError) as exc:
        raise SourceGuardError(f"cannot inspect {function!r}") from exc


def make_pdf_clone(
    budget: int, source_function: Callable | None = None
) -> Callable:
    """Clone the undecorated ``VariationalPosterior.pdf`` with one budget.

    The guard matches the complete step expression, replaces exactly its
    ``2**16`` node, and removes decorators from the compiled private clone.
    """
    budget = _checked_budget(budget)
    original = inspect.unwrap(
        VariationalPosterior.pdf
        if source_function is None
        else source_function
    )
    tree = ast.parse(_source(original))
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef)
    ]
    if len(functions) != 1:
        raise SourceGuardError("expected exactly one PDF function definition")
    function_node = functions[0]
    expected = ast.parse("step = max(1, 2**16 // max(1, K * D))").body[0].value
    matches = [
        node
        for node in ast.walk(function_node)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "step"
        and ast.dump(node.value, include_attributes=False)
        == ast.dump(expected, include_attributes=False)
    ]
    if len(matches) != 1:
        raise SourceGuardError(
            "PDF source must contain exactly the guarded 2**16 step assignment"
        )
    powers = [
        node
        for node in ast.walk(matches[0].value)
        if isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Pow)
        and isinstance(node.left, ast.Constant)
        and node.left.value == 2
        and isinstance(node.right, ast.Constant)
        and node.right.value == 16
    ]
    if len(powers) != 1:
        raise SourceGuardError("PDF step assignment has an unexpected budget")
    replacement = ast.copy_location(
        ast.Name(id="_CALIBRATION_PDF_BUDGET", ctx=ast.Load()), powers[0]
    )
    for field, value in ast.iter_fields(matches[0].value):
        if value is powers[0]:
            setattr(matches[0].value, field, replacement)
            break
        if isinstance(value, list):
            for index, item in enumerate(value):
                if item is powers[0]:
                    value[index] = replacement
                    break
    else:
        # The power is nested in the floor-division expression.
        assert isinstance(matches[0].value, ast.Call)
        division = matches[0].value.args[1]
        assert isinstance(division, ast.BinOp)
        division.left = replacement

    function_node.decorator_list = []
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    namespace["_CALIBRATION_PDF_BUDGET"] = budget
    filename = inspect.getsourcefile(original) or "<calibration-pdf-clone>"
    exec(compile(tree, filename, "exec"), namespace)
    clone = namespace[function_node.name]
    clone.__name__ = f"calibration_pdf_{budget}"
    clone.__qualname__ = clone.__name__
    return clone


def make_entropy_clone(
    budget: int, source_function: Callable | None = None
) -> Callable:
    """Clone ``entmc_vbmc`` with copied globals and a private budget."""
    budget = _checked_budget(budget)
    original = entmc_vbmc if source_function is None else source_function
    if (
        source_function is None
        and original.__globals__.get("_MAX_TENSOR_ELEMENTS") != DEFAULT_BUDGET
    ):
        raise SourceGuardError(
            "production entropy budget no longer equals the guarded default"
        )
    tree = ast.parse(_source(original))
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef)
    ]
    if len(functions) != 1:
        raise SourceGuardError(
            "expected exactly one entropy function definition"
        )
    budget_loads = [
        node
        for node in ast.walk(functions[0])
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == "_MAX_TENSOR_ELEMENTS"
    ]
    guarded_assignments = [
        node
        for node in ast.walk(functions[0])
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "budget"
        and isinstance(node.value, ast.Name)
        and node.value.id == "_MAX_TENSOR_ELEMENTS"
    ]
    if len(budget_loads) != 1 or len(guarded_assignments) != 1:
        raise SourceGuardError(
            "entropy source must read _MAX_TENSOR_ELEMENTS once into budget"
        )
    namespace = dict(original.__globals__)
    namespace["_MAX_TENSOR_ELEMENTS"] = budget
    clone = FunctionType(
        original.__code__,
        namespace,
        f"calibration_entmc_{budget}",
        original.__defaults__,
        original.__closure__,
    )
    clone.__kwdefaults__ = copy.copy(original.__kwdefaults__)
    clone.__annotations__ = dict(original.__annotations__)
    clone.__doc__ = original.__doc__
    clone.__module__ = original.__module__
    return clone


def _make_pdf_timing_clone(budget: int) -> Callable:
    """Reapply the public 0-D/1-D adapter around the guarded kernel clone."""
    kernel = make_pdf_clone(budget)
    return handle_0D_1D_input(patched_kwargs=["x"], patched_argpos=[0])(kernel)


def _source_fingerprints() -> dict:
    pdf = inspect.unwrap(VariationalPosterior.pdf)
    sources = {
        "variational_posterior_pdf": _source(pdf),
        "entmc_vbmc": _source(entmc_vbmc),
    }
    return {
        name: {
            "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "source_file": str(inspect.getsourcefile(function)),
        }
        for (name, source), function in zip(
            sources.items(), (pdf, entmc_vbmc), strict=True
        )
    }


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _make_vp(D: int, K: int, seed: int, adverse: bool = False):
    data_rng = _rng(seed)
    vp = VariationalPosterior(D, K, rng=_rng(seed + 10_000))
    span = 12.0 if adverse else 1.5
    scale_span = 2.5 if adverse else 0.35
    vp.mu = data_rng.uniform(-span, span, (D, K)).astype(np.float64)
    vp.sigma = np.exp(
        data_rng.uniform(-scale_span, scale_span, (1, K))
    ).astype(np.float64)
    vp.lambd = np.exp(
        data_rng.uniform(-scale_span, scale_span, (D, 1))
    ).astype(np.float64)
    vp.eta = data_rng.uniform(-6.0 if adverse else -1.0, 1.0, (1, K))
    eta_shifted = vp.eta - np.max(vp.eta)
    vp.w = (np.exp(eta_shifted) / np.exp(eta_shifted).sum()).astype(np.float64)
    return vp


def _problem(workload: Workload, seed: int = 1201):
    vp = _make_vp(workload.D, workload.K, seed)
    if workload.kernel == "pdf":
        x = _rng(seed + 1).normal(size=(workload.count, workload.D))
        return vp, x.astype(np.float64)
    return (vp,)


def effective_pdf_signature(budget: int, D: int, K: int, N: int) -> tuple:
    step = max(1, _checked_budget(budget) // max(1, int(K) * int(D)))
    return (min(int(N), step),)


def effective_entropy_signature(budget: int, D: int, K: int, Ns: int) -> tuple:
    budget = _checked_budget(budget)
    Ns = int(math.ceil(Ns / 2)) * 2
    per_component = Ns * int(D) * int(K)
    g = max(1, min(int(K), budget // max(1, per_component)))
    step = (
        Ns if per_component <= budget else max(1, budget // (int(D) * int(K)))
    )
    return (g, min(Ns, step), Ns)


def _signature(workload: Workload, budget: int) -> tuple:
    if workload.kernel == "pdf":
        return effective_pdf_signature(
            budget, workload.D, workload.K, workload.count
        )
    return effective_entropy_signature(
        budget, workload.D, workload.K, workload.count
    )


def _representatives(workload: Workload, budgets: tuple[int, ...]):
    groups: dict[tuple, list[int]] = {}
    for budget in budgets:
        groups.setdefault(_signature(workload, budget), []).append(budget)
    aliases = {}
    for members in groups.values():
        representative = (
            DEFAULT_BUDGET if DEFAULT_BUDGET in members else members[0]
        )
        aliases.update({budget: representative for budget in members})
    return aliases, tuple(dict.fromkeys(aliases.values()))


def _invoke(workload: Workload, function: Callable, problem, seed: int):
    if workload.kernel == "pdf":
        vp, x = problem
        return function(
            vp,
            x,
            orig_flag=False,
            log_flag=False,
            grad_flag=workload.grad_flags is not None,
        )
    (vp,) = problem
    call_rng = _rng(seed)  # Explicit reset is outside the timed region.
    return function(
        vp,
        workload.count,
        grad_flags=workload.grad_flags,
        jacobian_flag=workload.jacobian_flag,
        rng=call_rng,
    )


def _timed_call(workload, function, problem, seed):
    if workload.kernel == "entropy":
        (vp,) = problem
        call_rng = _rng(seed)  # Do not include seed reset in kernel timing.
        started = time.perf_counter()
        result = function(
            vp,
            workload.count,
            grad_flags=workload.grad_flags,
            jacobian_flag=workload.jacobian_flag,
            rng=call_rng,
        )
    else:
        vp, x = problem
        started = time.perf_counter()
        result = function(
            vp,
            x,
            orig_flag=False,
            log_flag=False,
            grad_flag=workload.grad_flags is not None,
        )
    elapsed = time.perf_counter() - started
    del result
    return elapsed


def _balanced_order(items: tuple, round_index: int) -> tuple:
    if not items:
        return ()
    cycle, shift = divmod(round_index, len(items))
    # Every full cycle puts each arm in every timing position exactly once.
    # Reverse only between complete cycles, never within one.
    if cycle % 2:
        items = tuple(reversed(items))
    order = items[shift:] + items[:shift]
    return order


def _estimated_workspace(workload: Workload, budget: int) -> dict:
    D, K = workload.D, workload.K
    if workload.kernel == "pdf":
        rows = _signature(workload, budget)[0]
        doubles = (
            3 * rows * K * D
            + 2 * rows * K
            + workload.count * D
            + workload.count * (1 + (D if workload.grad_flags else 0))
        )
        detail = (
            "partial live-array tally: input copy/output plus diff, quotient, "
            "square, d2, and component-density temporaries"
        )
    else:
        g, step, Ns = _signature(workload, budget)
        block = g * step
        # epsilon persists; concatenation briefly overlaps eps_half,
        # -eps_half and epsilon, totaling 2*K*Ns*D doubles.
        doubles = (
            2 * K * Ns * D + block * D * K + 2 * block * K + 3 * block * D
        )
        detail = (
            "estimated transient antithetic setup, delta, d2/E, and "
            "sample/gradient work arrays"
        )
    return {
        "bytes_estimate": int(doubles * np.dtype(np.float64).itemsize),
        "label": "estimated NumPy workspace; not a hard memory cap or RSS peak",
        "detail": detail,
    }


def _validate_workload(
    workload: Workload,
    functions: dict[int, Callable],
    budgets: tuple[int, ...],
    problem,
) -> dict:
    """Compare every timed candidate with production on fixed inputs."""
    vp = problem[0]
    vp_before = _vp_snapshot(vp)
    input_before = (
        np.array(problem[1], copy=True) if workload.kernel == "pdf" else None
    )
    vp_rng_before = copy.deepcopy(vp.rng.bit_generator.state)
    global_before = copy.deepcopy(np.random.get_state())
    reference_function = (
        VariationalPosterior.pdf if workload.kernel == "pdf" else entmc_vbmc
    )
    reference_rng = _rng(61_001)
    if workload.kernel == "pdf":
        reference = reference_function(
            vp,
            problem[1],
            orig_flag=False,
            log_flag=False,
            grad_flag=workload.grad_flags is not None,
        )
        reference_rng_after = None
    else:
        reference = reference_function(
            vp,
            workload.count,
            grad_flags=workload.grad_flags,
            jacobian_flag=workload.jacobian_flag,
            rng=reference_rng,
        )
        reference_rng_after = copy.deepcopy(reference_rng.bit_generator.state)
    checks = []
    for budget in budgets:
        call_rng = _rng(61_001)
        if workload.kernel == "pdf":
            actual = functions[budget](
                vp,
                problem[1],
                orig_flag=False,
                log_flag=False,
                grad_flag=workload.grad_flags is not None,
            )
            rng_equal = True
        else:
            actual = functions[budget](
                vp,
                workload.count,
                grad_flags=workload.grad_flags,
                jacobian_flag=workload.jacobian_flag,
                rng=call_rng,
            )
            rng_equal = _same_state(
                call_rng.bit_generator.state, reference_rng_after
            )
        comparison = _comparison(
            actual,
            reference,
            exact=workload.kernel == "pdf" or budget == DEFAULT_BUDGET,
        )
        comparison["rng_advancement_exact"] = rng_equal
        comparison["pass"] = comparison["pass"] and rng_equal
        checks.append({"budget": budget, **comparison})
    vp_and_input_unchanged = all(
        np.array_equal(before, getattr(vp, name))
        for name, before in vp_before.items()
    ) and _same_state(vp_rng_before, vp.rng.bit_generator.state)
    if input_before is not None:
        vp_and_input_unchanged = vp_and_input_unchanged and np.array_equal(
            input_before, problem[1]
        )
    global_unchanged = _legacy_random_state_equal(
        global_before, np.random.get_state()
    )
    return {
        "pass": (
            all(check["pass"] for check in checks)
            and vp_and_input_unchanged
            and global_unchanged
        ),
        "checks": checks,
        "vp_and_input_state_unchanged": vp_and_input_unchanged,
        "numpy_global_rng_unchanged": global_unchanged,
    }


def _measure_case(
    workload: Workload,
    functions: dict[int, Callable],
    budgets: tuple[int, ...],
    repeats: int,
    warmups: int,
    deadline: float,
    case_seconds: float,
):
    case_started = time.perf_counter()
    synthetic_started = time.perf_counter()
    problem = _problem(workload)
    synthetic_setup_seconds = time.perf_counter() - synthetic_started
    aliases, representatives = _representatives(workload, budgets)
    record = {
        "kernel": workload.kernel,
        "name": workload.name,
        "D": workload.D,
        "K": workload.K,
        "requested_count": workload.count,
        "effective_count": (
            int(math.ceil(workload.count / 2)) * 2
            if workload.kernel == "entropy"
            else workload.count
        ),
        "grad_flags": workload.grad_flags,
        "jacobian_flag": workload.jacobian_flag,
        "budget_aliases": {str(k): v for k, v in aliases.items()},
        "signatures": {
            str(budget): list(_signature(workload, budget))
            for budget in budgets
        },
        "first_use_seconds": {},
        "round_seconds": {str(budget): [] for budget in representatives},
        "workspace_estimates": {
            str(budget): _estimated_workspace(workload, budget)
            for budget in representatives
        },
        "complete": False,
        "diagnostic_seconds": 0.0,
        "measurement_seconds": 0.0,
        "synthetic_setup_seconds": synthetic_setup_seconds,
        "selection_wall_seconds": 0.0,
        "timing_scope": (
            "public PDF input adapter plus source-derived kernel"
            if workload.kernel == "pdf"
            else "source-derived entropy function"
        ),
    }
    selection_started = None

    def finish(complete=False):
        record["complete"] = complete
        record["measurement_seconds"] = max(
            0.0,
            time.perf_counter() - case_started - record["diagnostic_seconds"],
        )
        if selection_started is not None:
            record["selection_wall_seconds"] = (
                time.perf_counter() - selection_started
            )
        return record

    case_deadline = min(deadline, time.perf_counter() + case_seconds)
    for index, budget in enumerate(representatives):
        if time.perf_counter() >= case_deadline:
            return finish()
        record["first_use_seconds"][str(budget)] = _timed_call(
            workload, functions[budget], problem, 31_000
        )
    diagnostic_started = time.perf_counter()
    record["numerical_validation"] = _validate_workload(
        workload, functions, budgets, problem
    )
    record["diagnostic_seconds"] = time.perf_counter() - diagnostic_started
    if not record["numerical_validation"]["pass"]:
        return finish()
    selection_started = time.perf_counter()
    for warmup in range(warmups):
        for budget in _balanced_order(representatives, warmup):
            if time.perf_counter() >= case_deadline:
                return finish()
            _invoke(
                workload,
                functions[budget],
                problem,
                41_000 + 100 * warmup,
            )
    for round_index in range(repeats):
        for budget in _balanced_order(representatives, round_index):
            if time.perf_counter() >= case_deadline:
                return finish()
            elapsed = _timed_call(
                workload,
                functions[budget],
                problem,
                51_000 + 100 * round_index,
            )
            record["round_seconds"][str(budget)].append(elapsed)
    return finish(complete=True)


def select_budget(
    cases: list[dict],
    budgets: tuple[int, ...] = CANDIDATE_BUDGETS,
    default: int = DEFAULT_BUDGET,
    min_advantage: float = 0.10,
    min_win_fraction: float = 0.80,
    min_repeats: int = 3,
) -> dict:
    """Choose one global budget using paired, shape-balanced observations."""
    evidence = {}
    eligible = []
    if not cases or any(not case.get("complete") for case in cases):
        return {
            "budget": default,
            "accepted": False,
            "reason": "incomplete calibration",
            "candidates": evidence,
        }
    for candidate in budgets:
        if candidate == default:
            continue
        ratios = []
        case_medians = []
        complete = True
        for case in cases:
            aliases = case["budget_aliases"]
            candidate_rep = aliases[str(candidate)]
            default_rep = aliases[str(default)]
            if candidate_rep == default_rep:
                continue
            candidate_times = case["round_seconds"].get(str(candidate_rep), [])
            default_times = case["round_seconds"].get(str(default_rep), [])
            if min(len(candidate_times), len(default_times)) < min_repeats:
                complete = False
                break
            paired = [
                default_time / candidate_time
                for default_time, candidate_time in zip(
                    default_times, candidate_times, strict=True
                )
            ]
            ratios.extend(paired)
            case_medians.append(median(paired))
        if not complete or not ratios:
            evidence[str(candidate)] = {"eligible": False, "ratios": ratios}
            continue
        ratio_median = median(ratios)
        win_fraction = sum(ratio >= 1.0 for ratio in ratios) / len(ratios)
        no_case_loss = min(case_medians) >= 0.95
        accepted = (
            ratio_median >= 1.0 + min_advantage
            and win_fraction >= min_win_fraction
            and no_case_loss
        )
        evidence[str(candidate)] = {
            "eligible": accepted,
            "paired_speedups": ratios,
            "median_paired_speedup": ratio_median,
            "win_fraction": win_fraction,
            "case_median_speedups": case_medians,
            "no_case_median_below_0_95": no_case_loss,
        }
        if accepted:
            eligible.append((ratio_median, -candidate, candidate))
    if not eligible:
        return {
            "budget": default,
            "accepted": False,
            "reason": "no candidate cleared conservative selection gates",
            "candidates": evidence,
        }
    winner = max(eligible)[2]
    return {
        "budget": winner,
        "accepted": True,
        "reason": "candidate cleared calibration gates",
        "candidates": evidence,
    }


def _same_state(left, right) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _same_state(left[key], right[key]) for key in left
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.array_equal(left, right)
    return left == right


def _legacy_random_state_equal(left, right) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _vp_snapshot(vp) -> dict:
    return {
        name: np.array(getattr(vp, name), copy=True)
        for name in ("mu", "sigma", "lambd", "eta", "w")
    }


def _comparison(actual, expected, exact: bool) -> dict:
    actual_items = actual if isinstance(actual, tuple) else (actual,)
    expected_items = expected if isinstance(expected, tuple) else (expected,)
    arrays = [np.asarray(item) for item in actual_items]
    references = [np.asarray(item) for item in expected_items]
    finite = all(np.all(np.isfinite(item)) for item in arrays)
    dtype_ok = all(item.dtype == np.float64 for item in arrays)
    exact_ok = all(
        np.array_equal(item, reference)
        for item, reference in zip(arrays, references, strict=True)
    )
    close_ok = all(
        np.allclose(item, reference, rtol=1e-12, atol=1e-13)
        for item, reference in zip(arrays, references, strict=True)
    )
    max_abs = max(
        (float(np.max(np.abs(item - reference))) if item.size else 0.0)
        for item, reference in zip(arrays, references, strict=True)
    )
    per_output_max_abs = [
        float(np.max(np.abs(item - reference))) if item.size else 0.0
        for item, reference in zip(arrays, references, strict=True)
    ]
    return {
        "pass": finite and dtype_ok and (exact_ok if exact else close_ok),
        "finite": finite,
        "float64": dtype_ok,
        "exact": exact_ok,
        "close": close_ok,
        "max_abs_difference": max_abs,
        "per_output_max_abs_difference": per_output_max_abs,
    }


def numerical_validation() -> dict:
    """Exercise default equality, partial chunks, flags, state, and RNG."""
    global_before = copy.deepcopy(np.random.get_state())
    records = []

    pdf_vp = _make_vp(3, 7, 901, adverse=True)
    pdf_x = _rng(902).normal(size=(11, 3)).astype(np.float64)
    pdf_before = _vp_snapshot(pdf_vp)
    pdf_rng_before = copy.deepcopy(pdf_vp.rng.bit_generator.state)
    raw_pdf = inspect.unwrap(VariationalPosterior.pdf)
    default_pdf = make_pdf_clone(DEFAULT_BUDGET)
    partial_pdf = make_pdf_clone(100)  # row step 4: 4, 4, 3
    for name, kwargs in (
        ("value", {}),
        ("log_value", {"log_flag": True}),
        ("gradient", {"grad_flag": True}),
    ):
        reference = raw_pdf(pdf_vp, pdf_x, orig_flag=False, **kwargs)
        same = default_pdf(pdf_vp, pdf_x, orig_flag=False, **kwargs)
        partial = partial_pdf(pdf_vp, pdf_x, orig_flag=False, **kwargs)
        records.append(
            {
                "kernel": "pdf",
                "variant": name,
                **_comparison(same, reference, True),
            }
        )
        records.append(
            {
                "kernel": "pdf",
                "variant": f"{name}_partial_rows",
                **_comparison(partial, reference, True),
            }
        )
    bounded_vp = _make_vp(3, 7, 905, adverse=True)
    bounded_vp.parameter_transformer = ParameterTransformer(
        3,
        lb_orig=np.full((1, 3), -2.0),
        ub_orig=np.full((1, 3), 2.0),
        plb_orig=np.full((1, 3), -1.0),
        pub_orig=np.full((1, 3), 1.0),
    )
    bounded_x = np.linspace(-1.8, 1.8, 33, dtype=np.float64).reshape(11, 3)
    bounded_x_before = bounded_x.copy()
    for name, kwargs in (("value", {}), ("log_value", {"log_flag": True})):
        reference = raw_pdf(bounded_vp, bounded_x, orig_flag=True, **kwargs)
        for label, function in (
            ("default_clone", default_pdf),
            ("partial_rows", partial_pdf),
        ):
            actual = function(bounded_vp, bounded_x, orig_flag=True, **kwargs)
            records.append(
                {
                    "kernel": "pdf",
                    "variant": f"bounded_original_{name}_{label}",
                    **_comparison(actual, reference, True),
                }
            )
    bounded_input_unchanged = np.array_equal(bounded_x, bounded_x_before)
    pdf_unchanged = all(
        np.array_equal(before, getattr(pdf_vp, name))
        for name, before in pdf_before.items()
    ) and _same_state(pdf_rng_before, pdf_vp.rng.bit_generator.state)

    ent_vp = _make_vp(3, 7, 911, adverse=True)
    ent_before = _vp_snapshot(ent_vp)
    ent_vp_rng_before = copy.deepcopy(ent_vp.rng.bit_generator.state)
    default_entropy = make_entropy_clone(DEFAULT_BUDGET)
    partial_components = make_entropy_clone(2600)  # g=3: 3, 3, 1
    partial_samples = make_entropy_clone(500)  # g=1, samples 23 and 17
    variants = (
        ("all_grad_jacobian", (True, True, True, True), True),
        ("all_grad_raw", (True, True, True, True), False),
        ("sigma_only", (False, True, False, False), True),
        ("value_only", (False, False, False, False), True),
    )
    for variant, flags, jacobian in variants:
        reference_rng = _rng(77)
        reference = entmc_vbmc(
            ent_vp,
            40,
            grad_flags=flags,
            jacobian_flag=jacobian,
            rng=reference_rng,
        )
        reference_rng_after = copy.deepcopy(reference_rng.bit_generator.state)
        for label, function, exact in (
            ("default_clone", default_entropy, True),
            ("partial_components", partial_components, False),
            ("partial_samples", partial_samples, False),
        ):
            call_rng = _rng(77)
            actual = function(
                ent_vp,
                40,
                grad_flags=flags,
                jacobian_flag=jacobian,
                rng=call_rng,
            )
            comparison = _comparison(actual, reference, exact)
            comparison["rng_advancement_exact"] = _same_state(
                call_rng.bit_generator.state, reference_rng_after
            )
            comparison["pass"] = (
                comparison["pass"] and comparison["rng_advancement_exact"]
            )
            records.append(
                {
                    "kernel": "entropy",
                    "variant": f"{variant}_{label}",
                    **comparison,
                }
            )
    ent_unchanged = all(
        np.array_equal(before, getattr(ent_vp, name))
        for name, before in ent_before.items()
    ) and _same_state(ent_vp_rng_before, ent_vp.rng.bit_generator.state)
    global_unchanged = _legacy_random_state_equal(
        global_before, np.random.get_state()
    )
    return {
        "pass": (
            all(record["pass"] for record in records)
            and pdf_unchanged
            and bounded_input_unchanged
            and ent_unchanged
            and global_unchanged
        ),
        "checks": records,
        "pdf_state_unchanged": pdf_unchanged,
        "bounded_pdf_input_unchanged": bounded_input_unchanged,
        "entropy_state_unchanged": ent_unchanged,
        "numpy_global_rng_unchanged": global_unchanged,
    }


def _heldout(
    kernel: str,
    workloads: list[Workload],
    selected_by_regime: dict[str, int],
    repeats: int,
    deadline: float,
):
    factory = _make_pdf_timing_clone if kernel == "pdf" else make_entropy_clone
    records = []
    clone_setup_seconds = 0.0
    for case_index, workload in enumerate(workloads):
        selected = selected_by_regime[workload.name]
        clone_started = time.perf_counter()
        arms = {
            "selected": factory(selected),
            "default": factory(DEFAULT_BUDGET),
            "selected_same_budget_control": factory(selected),
            "default_same_budget_control": factory(DEFAULT_BUDGET),
        }
        clone_setup_seconds += time.perf_counter() - clone_started
        problem = _problem(workload, seed=7201 + case_index)
        row = {
            "name": workload.name,
            "selected_budget": selected,
            "round_seconds": {name: [] for name in arms},
        }
        names = tuple(arms)
        for name in _balanced_order(names, case_index):
            if time.perf_counter() >= deadline:
                row["complete"] = False
                records.append(row)
                return {
                    "complete": False,
                    "clone_setup_seconds": clone_setup_seconds,
                    "cases": records,
                }
            _invoke(
                workload,
                arms[name],
                problem,
                71_000 + 1000 * case_index,
            )
        for round_index in range(repeats):
            for name in _balanced_order(names, round_index):
                if time.perf_counter() >= deadline:
                    row["complete"] = False
                    records.append(row)
                    return {
                        "complete": False,
                        "clone_setup_seconds": clone_setup_seconds,
                        "cases": records,
                    }
                row["round_seconds"][name].append(
                    _timed_call(
                        workload,
                        arms[name],
                        problem,
                        81_000 + 1000 * case_index + 100 * round_index,
                    )
                )
        row["complete"] = True
        records.append(row)
    return {
        "complete": True,
        "clone_setup_seconds": clone_setup_seconds,
        "cases": records,
    }


def _heldout_accepts(case: dict, selected: int) -> dict:
    if not case.get("complete"):
        return {"pass": False, "reason": "incomplete held-out timing"}
    if selected == DEFAULT_BUDGET:
        return {"pass": True, "reason": "default selected"}
    ratios = [
        default / selected_time
        for default, selected_time in zip(
            case["round_seconds"]["default"],
            case["round_seconds"]["selected"],
            strict=True,
        )
    ]
    win_fraction = sum(ratio >= 1.0 for ratio in ratios) / len(ratios)
    passed = median(ratios) >= 1.10 and win_fraction >= 0.80
    return {
        "pass": passed,
        "reason": (
            "held-out gates passed" if passed else "held-out gates failed"
        ),
        "paired_speedups": ratios,
        "median_paired_speedup": median(ratios),
        "win_fraction": win_fraction,
    }


def _trace_peak(workload: Workload, budget: int, seed: int) -> dict:
    factory = (
        _make_pdf_timing_clone
        if workload.kernel == "pdf"
        else make_entropy_clone
    )
    function = factory(budget)
    problem = _problem(workload, seed)
    tracemalloc.start()
    try:
        _invoke(workload, function, problem, seed + 1)
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {
        "current_bytes": current,
        "peak_bytes": peak,
        "label": (
            "Python/NumPy allocator trace outside timings; excludes any "
            "native or process allocations tracemalloc does not observe"
        ),
    }


def _workloads(quick: bool):
    if quick:
        return {
            "pdf": [
                Workload("pdf", "small_batch", 3, 5, 8),
                Workload("pdf", "sieve_scaled_smoke", 4, 10, 512),
                Workload("pdf", "export_scaled_smoke", 8, 12, 2048),
            ],
            "entropy": [
                Workload("entropy", "adam", 3, 5, 59, (True,) * 4),
                Workload(
                    "entropy", "boost_scaled_smoke", 4, 12, 88, (True,) * 4
                ),
                Workload(
                    "entropy", "fine_scaled_smoke", 3, 5, 256, (False,) * 4
                ),
                Workload("entropy", "active", 3, 5, 40, (True,) * 4),
            ],
        }
    return {
        "pdf": [
            Workload("pdf", "small_batch", 4, 10, 8),
            Workload("pdf", "sieve_8192", 4, 20, 8192),
            Workload("pdf", "large_export", 15, 26, 100_000),
        ],
        "entropy": [
            Workload(
                "entropy",
                "adam",
                4,
                20,
                math.ceil(100 * 20 ** (-1 / 3)),
                (True,) * 4,
            ),
            Workload(
                "entropy",
                "boost",
                4,
                50,
                math.ceil(200 * 50 ** (-1 / 3)),
                (True,) * 4,
            ),
            Workload(
                "entropy",
                "boost_high_d",
                15,
                50,
                math.ceil(200 * 50 ** (-1 / 3)),
                (True,) * 4,
            ),
            Workload("entropy", "fine", 4, 20, 4096, (False,) * 4),
            Workload("entropy", "fine_high_d", 15, 26, 4096, (False,) * 4),
            Workload("entropy", "active", 4, 20, 200, (True,) * 4),
        ],
    }


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _environment() -> dict:
    try:
        from threadpoolctl import threadpool_info

        threadpools = {"available": True, "libraries": threadpool_info()}
    except ImportError:
        threadpools = {
            "available": False,
            "libraries": None,
            "reason": "threadpoolctl is not installed; effective threads unknown",
        }
    try:
        numpy_configuration = np.__config__.show(mode="dicts")
    except TypeError:
        numpy_configuration = "unavailable with this NumPy API"
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "python": sys.version,
        "versions": {
            "numpy": np.__version__,
            "scipy": _version("scipy"),
            "pyvbmc": _version("pyvbmc"),
            "threadpoolctl": _version("threadpoolctl"),
        },
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "threadpools": threadpools,
        "numpy_configuration": numpy_configuration,
    }


def _strict(value):
    if isinstance(value, dict):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_report(path: Path, report: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_strict(report), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_campaign(args) -> dict:
    campaign_global_rng_before = copy.deepcopy(np.random.get_state())
    started = time.perf_counter()
    total_deadline = started + args.deadline
    calibration_deadline = min(
        total_deadline, started + args.calibration_seconds
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "developer-only machine-local chunk calibration evidence",
        "status": "running",
        "mode": "quick" if args.quick else "full",
        "configuration": {
            "budgets": list(args.budgets),
            "default_budget": DEFAULT_BUDGET,
            "repeats": args.repeats,
            "heldout_repeats": args.heldout_repeats,
            "warmups": args.warmups,
            "deadline_seconds": args.deadline,
            "calibration_seconds": args.calibration_seconds,
            "case_seconds": args.case_seconds,
            "trace_allocations": args.trace_allocations,
            "selection_gates": {
                "minimum_median_advantage": 0.10,
                "minimum_win_fraction": 0.80,
                "minimum_case_median_speedup": 0.95,
            },
        },
        "environment": _environment(),
        "provenance": {},
        "timings_seconds": {
            "imports": _IMPORT_SECONDS,
            "setup": 0.0,
            "calibration": 0.0,
            "validation": 0.0,
            "total": 0.0,
        },
        "cases": {"pdf": [], "entropy": []},
        "selection": {
            kernel: {
                "scope": "per_regime_evidence_only",
                "by_regime": {},
            }
            for kernel in ("pdf", "entropy")
        },
    }
    _write_report(args.output, report)
    try:
        setup_started = time.perf_counter()
        report["provenance"]["sources"] = _source_fingerprints()
        report["provenance"]["script_sha256"] = hashlib.sha256(
            SCRIPT_PATH.read_bytes()
        ).hexdigest()
        functions = {
            "pdf": {
                budget: _make_pdf_timing_clone(budget)
                for budget in args.budgets
            },
            "entropy": {
                budget: make_entropy_clone(budget) for budget in args.budgets
            },
        }
        report["timings_seconds"]["setup"] = (
            time.perf_counter() - setup_started
        )

        validation_started = time.perf_counter()
        report["numerical_validation"] = numerical_validation()
        report["timings_seconds"]["validation"] += (
            time.perf_counter() - validation_started
        )
        if not report["numerical_validation"]["pass"]:
            raise SourceGuardError("numerical validation failed")

        workloads = _workloads(args.quick)
        all_complete = True
        for kernel in ("pdf", "entropy"):
            for workload in workloads[kernel]:
                if time.perf_counter() >= calibration_deadline:
                    all_complete = False
                    break
                case = _measure_case(
                    workload,
                    functions[kernel],
                    args.budgets,
                    args.repeats,
                    args.warmups,
                    calibration_deadline,
                    args.case_seconds,
                )
                report["cases"][kernel].append(case)
                all_complete = all_complete and case["complete"]
                report["timings_seconds"]["calibration"] += case[
                    "measurement_seconds"
                ]
                report["timings_seconds"]["validation"] += case[
                    "diagnostic_seconds"
                ]
                _write_report(args.output, report)
            if time.perf_counter() >= calibration_deadline:
                all_complete = False

        minimum_repeats = min(3, args.repeats)
        tentative = {
            kernel: {
                case["name"]: select_budget(
                    [case], args.budgets, min_repeats=minimum_repeats
                )
                for case in report["cases"][kernel]
            }
            for kernel in ("pdf", "entropy")
        }
        report["tentative_selection"] = tentative
        if all_complete and time.perf_counter() < total_deadline:
            validation_started = time.perf_counter()
            report["heldout"] = {}
            for kernel in ("pdf", "entropy"):
                heldout = _heldout(
                    kernel,
                    workloads[kernel],
                    {
                        name: decision["budget"]
                        for name, decision in tentative[kernel].items()
                    },
                    args.heldout_repeats,
                    total_deadline,
                )
                all_complete = all_complete and heldout["complete"]
                gates = {}
                decisions = {}
                heldout_cases = {
                    case["name"]: case for case in heldout["cases"]
                }
                for name, decision in tentative[kernel].items():
                    gate = _heldout_accepts(
                        heldout_cases.get(name, {"complete": False}),
                        decision["budget"],
                    )
                    gates[name] = gate
                    if decision["accepted"] and gate["pass"]:
                        decisions[name] = decision
                    elif decision["accepted"]:
                        decisions[name] = {
                            **decision,
                            "budget": DEFAULT_BUDGET,
                            "accepted": False,
                            "reason": gate["reason"],
                        }
                    else:
                        decisions[name] = decision
                report["heldout"][kernel] = {**heldout, "gates": gates}
                report["selection"][kernel]["by_regime"] = decisions
            report["timings_seconds"]["validation"] += (
                time.perf_counter() - validation_started
            )
        else:
            all_complete = False

        if args.trace_allocations and all_complete:
            trace_started = time.perf_counter()
            report["traced_allocations"] = {}
            for kernel in ("pdf", "entropy"):
                report["traced_allocations"][kernel] = [
                    {
                        "name": workload.name,
                        "budget": report["selection"][kernel]["by_regime"][
                            workload.name
                        ]["budget"],
                        **_trace_peak(
                            workload,
                            report["selection"][kernel]["by_regime"][
                                workload.name
                            ]["budget"],
                            91_000 + index,
                        ),
                    }
                    for index, workload in enumerate(workloads[kernel])
                    if time.perf_counter() < total_deadline
                ]
            report["timings_seconds"]["validation"] += (
                time.perf_counter() - trace_started
            )

        if all_complete and time.perf_counter() <= total_deadline:
            report["status"] = "complete"
        else:
            report["status"] = "partial_defaulted"
            for kernel in ("pdf", "entropy"):
                report["selection"][kernel] = {
                    "scope": "per_regime_evidence_only",
                    "by_regime": {
                        workload.name: {
                            "budget": DEFAULT_BUDGET,
                            "accepted": False,
                            "reason": "bounded campaign did not complete",
                        }
                        for workload in workloads[kernel]
                    },
                }
    except (SourceGuardError, AssertionError, ValueError) as exc:
        report["status"] = "failed_closed"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        report["timings_seconds"]["total"] = time.perf_counter() - started
        report["campaign_numpy_global_rng_unchanged"] = (
            _legacy_random_state_equal(
                campaign_global_rng_before, np.random.get_state()
            )
        )
        if not report["campaign_numpy_global_rng_unchanged"]:
            report["status"] = "failed_closed"
            report["failure"] = {
                "type": "GlobalRNGMutation",
                "message": "campaign changed NumPy's legacy global RNG state",
            }
        all_call_times = []
        for cases in report["cases"].values():
            for case in cases:
                all_call_times.extend(
                    case.get("first_use_seconds", {}).values()
                )
                for times in case.get("round_seconds", {}).values():
                    all_call_times.extend(times)
        for heldout in report.get("heldout", {}).values():
            for case in heldout.get("cases", []):
                for times in case.get("round_seconds", {}).values():
                    all_call_times.extend(times)
        report["soft_deadlines"] = {
            "maximum_observed_call_seconds": max(all_call_times, default=0.0),
            "total_deadline_overshoot_seconds": max(
                0.0, report["timings_seconds"]["total"] - args.deadline
            ),
            "note": "deadlines are checked between calls and cannot interrupt NumPy",
        }
        _write_report(args.output, report)
    return report


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dev/scripts/runs/machine_local_calibration.json"),
    )
    parser.add_argument(
        "--quick", action="store_true", help="small smoke campaign"
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=list(CANDIDATE_BUDGETS),
    )
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--heldout-repeats", type=int)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--calibration-seconds", type=float)
    parser.add_argument("--case-seconds", type=float)
    parser.add_argument("--trace-allocations", action="store_true")
    args = parser.parse_args(argv)
    args.budgets = tuple(dict.fromkeys(args.budgets))
    if DEFAULT_BUDGET not in args.budgets:
        parser.error(f"--budgets must include the default {DEFAULT_BUDGET}")
    try:
        for budget in args.budgets:
            _checked_budget(budget)
    except ValueError as exc:
        parser.error(str(exc))
    defaults = (
        {
            "repeats": 2,
            "heldout_repeats": 2,
            "deadline": 20.0,
            "calibration_seconds": 12.0,
            "case_seconds": 3.0,
        }
        if args.quick
        else {
            "repeats": 5,
            "heldout_repeats": 4,
            "deadline": 180.0,
            "calibration_seconds": 90.0,
            "case_seconds": 20.0,
        }
    )
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if (
        args.repeats <= 0
        or args.heldout_repeats <= 0
        or args.warmups < 0
        or args.deadline <= 0
        or args.calibration_seconds <= 0
        or args.case_seconds <= 0
    ):
        parser.error("repeats and time allowances must be positive")
    return args


def main(argv=None) -> int:
    args = _parse_args(argv)
    report = run_campaign(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(args.output),
                "selection": report["selection"],
                "total_seconds": report["timings_seconds"]["total"],
            },
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )
    return 0 if report["status"] in {"complete", "partial_defaulted"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
