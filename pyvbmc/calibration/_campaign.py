"""Machine-local calibration campaign for PyVBMC numerical chunk sizes.

The campaign is explicit, synchronous, CPU-only, and independent of user or
solver random streams.  It calls the production numerical helpers with an
explicit budget and never changes module globals or algorithmic sample counts.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from statistics import median
from typing import Callable, NamedTuple

import numpy as np

DEFAULT_BUDGET = 2**16
CANDIDATE_BUDGETS = tuple(2**power for power in range(14, 19))
DISCOVERY_ROUNDS = 5
HELDOUT_ROUNDS = 4
WATCHDOG_SECONDS = 300.0
ESTIMATED_SECONDS = 30.0
MAX_INCREMENTAL_BYTES = 64 * 2**20
_MIN_SPEEDUP = 1.10
_MIN_WIN_FRACTION = 0.80
_MAX_WORKLOAD_SLOWDOWN = 1.05
_CONTROL_LOW = 1 / 1.05
_CONTROL_HIGH = 1.05

SETTING_GROUPS = {
    "pdf": "pdf_chunk_elements",
    "entropy_grad": "entropy_grad_chunk_elements",
    "entropy_value": "entropy_value_chunk_elements",
}


class _WatchdogExpired(RuntimeError):
    pass


class _IncompleteCoverage(RuntimeError):
    pass


@dataclass(frozen=True)
class _Workload:
    group: str
    name: str
    D: int
    K: int
    count: int
    grad_flags: tuple[bool, bool, bool, bool] | None = None
    jacobian_flag: bool = True


class _KernelAPI(NamedTuple):
    make_vp: Callable
    pdf: Callable
    public_pdf: Callable
    entropy: Callable
    public_entropy: Callable
    make_bounded_transformer: Callable


class _Watchdog:
    def __init__(
        self, clock: Callable[[], float], deadline: float, *, timer=None
    ):
        self.clock = clock
        self.timer = clock if timer is None else timer
        self.deadline = float(deadline)
        self.maximum_call_seconds = 0.0
        self.last_checkpoint = "setup"

    def check(self, checkpoint: str):
        self.last_checkpoint = checkpoint
        if self.clock() >= self.deadline:
            raise _WatchdogExpired(
                f"calibration watchdog expired at {checkpoint}"
            )

    def call(self, checkpoint: str, function: Callable):
        self.check(f"before:{checkpoint}")
        started = self.timer()
        value = function()
        elapsed = self.timer() - started
        if elapsed <= 0 or not math.isfinite(elapsed):
            raise _IncompleteCoverage(
                f"timer did not resolve a positive finite duration at {checkpoint}"
            )
        self.maximum_call_seconds = max(self.maximum_call_seconds, elapsed)
        self.check(f"after:{checkpoint}")
        return value, elapsed


def _load_kernel_api() -> _KernelAPI:
    # Lazy imports keep importing pyvbmc.calibration free of VP/VBMC cycles.
    from pyvbmc.entropy.entmc_vbmc import _entmc_vbmc, entmc_vbmc
    from pyvbmc.parameter_transformer import ParameterTransformer
    from pyvbmc.variational_posterior import VariationalPosterior

    def make_vp(D, K, *, rng, transformer=None):
        kwargs = {"rng": rng, "calibration": "off"}
        if transformer is not None:
            kwargs["parameter_transformer"] = transformer
        return VariationalPosterior(D, K, **kwargs)

    def pdf(vp, x, *, orig_flag, log_flag, grad_flag, budget):
        return vp._pdf(
            x,
            orig_flag=orig_flag,
            log_flag=log_flag,
            grad_flag=grad_flag,
            chunk_elements=budget,
        )

    def public_pdf(vp, x, *, orig_flag, log_flag, grad_flag, rng=None):
        del rng
        return vp.pdf(
            x,
            orig_flag=orig_flag,
            log_flag=log_flag,
            grad_flag=grad_flag,
        )

    def entropy(vp, Ns, *, grad_flags, jacobian_flag, rng, budget):
        return _entmc_vbmc(
            vp,
            Ns,
            grad_flags,
            jacobian_flag,
            budget=budget,
            rng=rng,
        )

    def public_entropy(vp, Ns, *, grad_flags, jacobian_flag, rng, budget=None):
        del budget
        return entmc_vbmc(
            vp,
            Ns,
            grad_flags,
            jacobian_flag,
            rng=rng,
        )

    def make_bounded_transformer(D):
        return ParameterTransformer(
            D,
            lb_orig=np.full((1, D), -2.0),
            ub_orig=np.full((1, D), 2.0),
            plb_orig=np.full((1, D), -1.0),
            pub_orig=np.full((1, D), 1.0),
        )

    return _KernelAPI(
        make_vp,
        pdf,
        public_pdf,
        entropy,
        public_entropy,
        make_bounded_transformer,
    )


def _workloads() -> tuple[_Workload, ...]:
    return (
        _Workload("pdf", "small_value", 4, 20, 8),
        _Workload("pdf", "sieve_value", 4, 20, 8192),
        _Workload(
            "pdf",
            "sieve_gradient",
            4,
            20,
            8192,
            (True, True, True, True),
        ),
        _Workload("pdf", "large_density", 15, 26, 100_000),
        _Workload(
            "entropy_grad",
            "adam_d4_k20",
            4,
            20,
            math.ceil(100 * 20 ** (-1 / 3)),
            (True, True, True, True),
        ),
        _Workload(
            "entropy_grad",
            "boost_d4_k50",
            4,
            50,
            math.ceil(200 * 50 ** (-1 / 3)),
            (True, True, True, True),
        ),
        _Workload(
            "entropy_grad",
            "boost_d15_k50",
            15,
            50,
            math.ceil(200 * 50 ** (-1 / 3)),
            (True, True, True, True),
        ),
        _Workload(
            "entropy_grad",
            "active_d4_k20",
            4,
            20,
            200,
            (True, True, True, True),
        ),
        _Workload(
            "entropy_value",
            "fine_d4_k20",
            4,
            20,
            4096,
            (False, False, False, False),
        ),
        _Workload(
            "entropy_value",
            "fine_d15_k26",
            15,
            26,
            4096,
            (False, False, False, False),
        ),
    )


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _make_problem(
    workload: _Workload,
    kernel_api: _KernelAPI,
    seed: int,
    *,
    adverse: bool = False,
    bounded: bool = False,
):
    data_rng = _rng(seed)
    transformer = (
        kernel_api.make_bounded_transformer(workload.D) if bounded else None
    )
    vp = kernel_api.make_vp(
        workload.D,
        workload.K,
        rng=_rng(seed + 100_000),
        transformer=transformer,
    )
    location_span = 8.0 if adverse else 1.25
    scale_span = 1.6 if adverse else 0.3
    vp.mu = data_rng.uniform(
        -location_span, location_span, (workload.D, workload.K)
    ).astype(np.float64)
    vp.sigma = np.exp(
        data_rng.uniform(-scale_span, scale_span, (1, workload.K))
    ).astype(np.float64)
    vp.lambd = np.exp(
        data_rng.uniform(-scale_span, scale_span, (workload.D, 1))
    ).astype(np.float64)
    vp.eta = data_rng.uniform(-4.0 if adverse else -1.0, 1.0, (1, workload.K))
    eta = vp.eta - np.max(vp.eta)
    vp.w = (np.exp(eta) / np.exp(eta).sum()).astype(np.float64)
    if workload.group == "pdf":
        if bounded:
            x = data_rng.uniform(-1.75, 1.75, (workload.count, workload.D))
        else:
            x = data_rng.normal(size=(workload.count, workload.D))
        return vp, x.astype(np.float64)
    return (vp,)


def _effective_count(workload: _Workload) -> int:
    if workload.group.startswith("entropy"):
        return int(math.ceil(workload.count / 2)) * 2
    return workload.count


def _layout_signature(workload: _Workload, budget: int) -> tuple[int, ...]:
    D, K = int(workload.D), int(workload.K)
    budget = int(budget)
    if workload.group == "pdf":
        step = max(1, budget // max(1, D * K))
        step = min(workload.count, step)
        return (step, workload.count // step, workload.count % step)
    Ns = _effective_count(workload)
    per_component = Ns * D * K
    g = max(1, min(K, budget // max(1, per_component)))
    step = Ns if per_component <= budget else max(1, budget // (D * K))
    step = min(Ns, step)
    return (g, K // g, K % g, step, Ns // step, Ns % step)


def _layout_aliases(
    workload: _Workload, budgets: tuple[int, ...] = CANDIDATE_BUDGETS
) -> dict[int, int]:
    groups: dict[tuple[int, ...], list[int]] = {}
    for budget in budgets:
        groups.setdefault(_layout_signature(workload, budget), []).append(
            budget
        )
    aliases = {}
    for members in groups.values():
        representative = (
            DEFAULT_BUDGET if DEFAULT_BUDGET in members else min(members)
        )
        aliases.update({budget: representative for budget in members})
    return aliases


def _balanced_orders(
    items: tuple[int, ...], rounds: int
) -> tuple[tuple[int, ...], ...]:
    """Pure rotations; reverse direction only between complete cycles."""
    if not items:
        return tuple()
    orders = []
    size = len(items)
    for round_index in range(rounds):
        cycle = round_index // size
        offset = round_index % size
        base = items if cycle % 2 == 0 else tuple(reversed(items))
        orders.append(base[offset:] + base[:offset])
    return tuple(orders)


def _aligned_timing_orders(
    aliases: dict[int, int], round_index: int
) -> tuple[tuple[int, ...], ...]:
    """Return position-balanced orders for one aligned discovery round.

    Five distinct layouts use one arm per aligned round and complete their
    rotation over the five rounds.  A collapsed layout set uses a complete
    rotation cycle inside every aligned round; its per-arm median is the
    aligned observation.
    """
    representatives = tuple(dict.fromkeys(aliases.values()))
    if len(representatives) == len(CANDIDATE_BUDGETS):
        return (
            _balanced_orders(CANDIDATE_BUDGETS, DISCOVERY_ROUNDS)[round_index],
        )
    base = (
        representatives
        if round_index % 2 == 0
        else tuple(reversed(representatives))
    )
    return _balanced_orders(base, len(base))


def _deduplicate_order(order: tuple[int, ...], aliases: dict[int, int]):
    seen = set()
    result = []
    for budget in order:
        representative = aliases[budget]
        if representative not in seen:
            seen.add(representative)
            result.append(representative)
    return tuple(result)


def _legacy_rng_equal(left, right) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _state_equal(left, right) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _state_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.array_equal(left, right)
    return left == right


def _vp_snapshot(vp) -> dict:
    return {
        name: np.array(getattr(vp, name), copy=True)
        for name in ("mu", "sigma", "lambd", "eta", "w")
    }


def _output_comparison(actual, expected, *, exact: bool) -> dict:
    actual_items = actual if isinstance(actual, tuple) else (actual,)
    expected_items = expected if isinstance(expected, tuple) else (expected,)
    arrays = [np.asarray(item) for item in actual_items]
    references = [np.asarray(item) for item in expected_items]
    if len(arrays) != len(references):
        return {"pass": False, "reason": "output arity differs"}
    shapes_equal = all(
        item.shape == reference.shape
        for item, reference in zip(arrays, references, strict=True)
    )
    if not shapes_equal:
        return {
            "pass": False,
            "finite": bool(all(np.all(np.isfinite(item)) for item in arrays)),
            "float64": bool(all(item.dtype == np.float64 for item in arrays)),
            "shapes_equal": False,
            "actual_shapes": [list(item.shape) for item in arrays],
            "expected_shapes": [list(item.shape) for item in references],
        }
    finite = all(np.all(np.isfinite(item)) for item in arrays)
    float64 = all(item.dtype == np.float64 for item in arrays)
    exact_equal = all(
        np.array_equal(item, reference)
        for item, reference in zip(arrays, references, strict=True)
    )
    close = all(
        np.allclose(item, reference, rtol=1e-12, atol=1e-12)
        for item, reference in zip(arrays, references, strict=True)
    )
    differences = []
    for item, reference in zip(arrays, references, strict=True):
        difference = (
            float(np.max(np.abs(item - reference))) if item.size else 0.0
        )
        differences.append(difference if math.isfinite(difference) else None)
    return {
        "pass": finite and float64 and (exact_equal if exact else close),
        "finite": bool(finite),
        "float64": bool(float64),
        "shapes_equal": True,
        "exact": bool(exact_equal),
        "within_tolerance": bool(close),
        "per_output_max_abs_difference": differences,
    }


def _invoke(
    workload: _Workload,
    problem,
    kernel_api: _KernelAPI,
    budget: int,
    seed: int,
    *,
    public: bool = False,
    log_flag: bool = False,
    orig_flag: bool = False,
):
    if workload.group == "pdf":
        vp, x = problem
        function = kernel_api.public_pdf if public else kernel_api.pdf
        kwargs = {
            "orig_flag": orig_flag,
            "log_flag": log_flag,
            "grad_flag": workload.grad_flags is not None,
        }
        if public:
            return function(vp, x, **kwargs)
        return function(vp, x, budget=budget, **kwargs)
    (vp,) = problem
    function = kernel_api.public_entropy if public else kernel_api.entropy
    call_rng = _rng(seed)
    kwargs = {
        "grad_flags": workload.grad_flags,
        "jacobian_flag": workload.jacobian_flag,
        "rng": call_rng,
    }
    if public:
        output = function(vp, workload.count, **kwargs)
    else:
        output = function(vp, workload.count, budget=budget, **kwargs)
    return output, copy.deepcopy(call_rng.bit_generator.state)


def _timed_invoke(
    watchdog: _Watchdog,
    checkpoint: str,
    workload: _Workload,
    problem,
    kernel_api: _KernelAPI,
    budget: int,
    seed: int,
):
    if workload.group == "pdf":
        return watchdog.call(
            checkpoint,
            lambda: _invoke(
                workload, problem, kernel_api, budget, seed, public=False
            ),
        )[1]
    call_rng = _rng(seed)  # Seed reset stays outside the timed interval.
    (vp,) = problem
    return watchdog.call(
        checkpoint,
        lambda: kernel_api.entropy(
            vp,
            workload.count,
            grad_flags=workload.grad_flags,
            jacobian_flag=workload.jacobian_flag,
            rng=call_rng,
            budget=budget,
        ),
    )[1]


def _workspace_estimate(workload: _Workload, budget: int) -> dict:
    D, K, count = workload.D, workload.K, _effective_count(workload)
    if workload.group == "pdf":
        step = _layout_signature(workload, budget)[0]
        # Input copy/output and fixed scales, plus the largest row block.
        # The block allows diff and two chained ufunc results to overlap,
        # alongside reduced distances, component densities and a reduction
        # output.  NumPy allocation metadata and reduction workspace receive
        # a further 25% headroom below.
        raw_doubles = (
            workload.count * D
            + workload.count * (1 + (D if workload.grad_flags else 0))
            + 3 * step * K * D
            + 2 * step * K
            + step * D
            + (1 + (1 if workload.grad_flags else 0)) * K * D
            + K
        )
    else:
        g, _, _, step, _, _ = _layout_signature(workload, budget)
        block = g * step
        # Antithetic construction briefly overlaps eps_half, -eps_half and
        # epsilon (2*K*Ns*D doubles).  The kernel later retains epsilon while
        # a block can require four distance-tensor-sized buffers: the stored
        # delta, chained ufunc input/output, and conservative einsum workspace.
        # Include reduced densities, sample/gradient intermediates, fixed
        # mixture arrays and the softmax Jacobian.  Summing these phase peaks
        # is intentionally conservative; 25% additional allocator/reduction
        # headroom is applied below.
        grad_mu, grad_sigma, grad_lambd, grad_w = workload.grad_flags
        raw_doubles = (
            2 * K * count * D
            + 4 * block * D * K
            + 3 * block * K
            + block
            + 5 * block * D
            + (3 + int(grad_mu)) * K * D
            + (6 + int(grad_sigma) + int(grad_w)) * K
            + (1 + int(grad_lambd)) * D
            + (4 * K * K if grad_w and workload.jacobian_flag else 0)
        )
    # Small calls also allocate Python objects and ndarray metadata, which
    # a proportional allowance alone cannot bound usefully.
    doubles = (5 * raw_doubles + 3) // 4 + (64 * 2**10 // 8)
    return {
        "bytes": int(doubles * np.dtype(np.float64).itemsize),
        "limit_bytes": MAX_INCREMENTAL_BYTES,
        "within_limit": bool(doubles * 8 <= MAX_INCREMENTAL_BYTES),
        "label": (
            "conservative incremental live NumPy storage estimate with "
            "25% allocator/reduction headroom plus 64 KiB metadata allowance; "
            "not RSS and not the chunk "
            "tensor alone"
        ),
    }


def _validate_workload(
    workload: _Workload,
    problem,
    kernel_api: _KernelAPI,
    watchdog: _Watchdog,
) -> dict:
    vp = problem[0]
    vp_before = _vp_snapshot(vp)
    vp_rng_before = copy.deepcopy(vp.rng.bit_generator.state)
    x_before = (
        np.array(problem[1], copy=True) if workload.group == "pdf" else None
    )
    seed = 610_000
    public_result, public_seconds = watchdog.call(
        f"numerical:{workload.name}:public",
        lambda: _invoke(
            workload,
            problem,
            kernel_api,
            DEFAULT_BUDGET,
            seed,
            public=True,
        ),
    )
    if workload.group == "pdf":
        reference = public_result
        reference_rng_state = None
    else:
        reference, reference_rng_state = public_result
    checks = []
    valid_budgets = {}
    for budget in CANDIDATE_BUDGETS:
        result, diagnostic_seconds = watchdog.call(
            f"numerical:{workload.name}:{budget}",
            lambda budget=budget: _invoke(
                workload, problem, kernel_api, budget, seed
            ),
        )
        if workload.group == "pdf":
            output = result
            rng_equal = True
        else:
            output, rng_state = result
            rng_equal = _state_equal(rng_state, reference_rng_state)
        comparison = _output_comparison(
            output,
            reference,
            exact=workload.group == "pdf" or budget == DEFAULT_BUDGET,
        )
        comparison["rng_advancement_exact"] = bool(rng_equal)
        comparison["pass"] = bool(comparison["pass"] and rng_equal)
        valid_budgets[budget] = comparison["pass"]
        checks.append(
            {
                "budget": budget,
                "diagnostic_call_seconds": diagnostic_seconds,
                **comparison,
            }
        )
    unchanged = all(
        np.array_equal(before, getattr(vp, name))
        for name, before in vp_before.items()
    ) and _state_equal(vp_rng_before, vp.rng.bit_generator.state)
    if x_before is not None:
        unchanged = unchanged and np.array_equal(x_before, problem[1])
    return {
        "pass": bool(all(valid_budgets.values()) and unchanged),
        "checks": checks,
        "first_public_call_seconds": public_seconds,
        "valid_budgets": {str(k): v for k, v in valid_budgets.items()},
        "inputs_vp_and_vp_rng_unchanged": bool(unchanged),
    }


def _dedicated_numerics(kernel_api: _KernelAPI, watchdog: _Watchdog) -> dict:
    records = []
    # Partial row blocks, transformed coordinates, log values and gradients.
    pdf = _Workload(
        "pdf",
        "dedicated_pdf",
        3,
        7,
        11,
        (True, True, True, True),
    )
    pdf_problem = _make_problem(pdf, kernel_api, 901, adverse=True)
    pdf_vp_before = _vp_snapshot(pdf_problem[0])
    pdf_rng_before = copy.deepcopy(pdf_problem[0].rng.bit_generator.state)
    pdf_x_before = np.array(pdf_problem[1], copy=True)
    pdf_reference, _ = watchdog.call(
        "dedicated:pdf:reference",
        lambda: _invoke(pdf, pdf_problem, kernel_api, DEFAULT_BUDGET, 77),
    )
    for label, kwargs in (
        ("gradient_transformed_partial_rows", {}),
        ("log_transformed_partial_rows", {"log_flag": True}),
    ):
        candidate_workload = pdf
        if kwargs:
            candidate_workload = _Workload("pdf", label, 3, 7, 11)
            pdf_reference, _ = watchdog.call(
                f"dedicated:pdf:{label}:reference",
                lambda: _invoke(
                    candidate_workload,
                    pdf_problem,
                    kernel_api,
                    DEFAULT_BUDGET,
                    77,
                    **kwargs,
                ),
            )
        actual, _ = watchdog.call(
            f"dedicated:pdf:{label}:partial",
            lambda: _invoke(
                candidate_workload,
                pdf_problem,
                kernel_api,
                100,
                77,
                **kwargs,
            ),
        )
        records.append(
            {
                "name": label,
                **_output_comparison(actual, pdf_reference, exact=True),
            }
        )

    bounded = _Workload("pdf", "bounded_original", 3, 7, 11)
    bounded_problem = _make_problem(
        bounded, kernel_api, 905, adverse=False, bounded=True
    )
    bounded_vp_before = _vp_snapshot(bounded_problem[0])
    bounded_rng_before = copy.deepcopy(
        bounded_problem[0].rng.bit_generator.state
    )
    bounded_x_before = np.array(bounded_problem[1], copy=True)
    for log_flag in (False, True):
        reference, _ = watchdog.call(
            f"dedicated:pdf:bounded:{log_flag}:reference",
            lambda log_flag=log_flag: _invoke(
                bounded,
                bounded_problem,
                kernel_api,
                DEFAULT_BUDGET,
                78,
                orig_flag=True,
                log_flag=log_flag,
            ),
        )
        actual, _ = watchdog.call(
            f"dedicated:pdf:bounded:{log_flag}:partial",
            lambda log_flag=log_flag: _invoke(
                bounded,
                bounded_problem,
                kernel_api,
                100,
                78,
                orig_flag=True,
                log_flag=log_flag,
            ),
        )
        records.append(
            {
                "name": f"bounded_original_log_{log_flag}",
                **_output_comparison(actual, reference, exact=True),
            }
        )

    entropy = _Workload(
        "entropy_grad",
        "dedicated_entropy",
        3,
        7,
        40,
        (True, True, True, True),
    )
    entropy_problem = _make_problem(entropy, kernel_api, 911, adverse=True)
    entropy_vp_before = _vp_snapshot(entropy_problem[0])
    entropy_rng_before = copy.deepcopy(
        entropy_problem[0].rng.bit_generator.state
    )
    for label, flags, jacobian, budget in (
        ("all_grad_partial_components", (True, True, True, True), True, 2600),
        ("all_grad_partial_samples", (True, True, True, True), False, 500),
        ("mu_partial_components", (True, False, False, False), True, 2600),
        ("sigma_partial_samples", (False, True, False, False), True, 500),
        ("lambda_partial_samples", (False, False, True, False), False, 500),
        (
            "weights_partial_components",
            (False, False, False, True),
            False,
            2600,
        ),
        ("value_partial_samples", (False, False, False, False), True, 500),
    ):
        variant = _Workload("entropy_grad", label, 3, 7, 40, flags, jacobian)
        reference_result, _ = watchdog.call(
            f"dedicated:entropy:{label}:reference",
            lambda variant=variant: _invoke(
                variant, entropy_problem, kernel_api, DEFAULT_BUDGET, 79
            ),
        )
        actual_result, _ = watchdog.call(
            f"dedicated:entropy:{label}:partial",
            lambda variant=variant, budget=budget: _invoke(
                variant, entropy_problem, kernel_api, budget, 79
            ),
        )
        reference, reference_state = reference_result
        actual, actual_state = actual_result
        comparison = _output_comparison(actual, reference, exact=False)
        comparison["rng_advancement_exact"] = _state_equal(
            actual_state, reference_state
        )
        comparison["pass"] = bool(
            comparison["pass"] and comparison["rng_advancement_exact"]
        )
        records.append({"name": label, **comparison})
    state_unchanged = (
        all(
            np.array_equal(value, getattr(pdf_problem[0], name))
            for name, value in pdf_vp_before.items()
        )
        and _state_equal(
            pdf_rng_before, pdf_problem[0].rng.bit_generator.state
        )
        and np.array_equal(pdf_x_before, pdf_problem[1])
        and all(
            np.array_equal(value, getattr(bounded_problem[0], name))
            for name, value in bounded_vp_before.items()
        )
        and _state_equal(
            bounded_rng_before, bounded_problem[0].rng.bit_generator.state
        )
        and np.array_equal(bounded_x_before, bounded_problem[1])
        and all(
            np.array_equal(value, getattr(entropy_problem[0], name))
            for name, value in entropy_vp_before.items()
        )
        and _state_equal(
            entropy_rng_before, entropy_problem[0].rng.bit_generator.state
        )
    )
    return {
        "pass": bool(
            all(item["pass"] for item in records) and state_unchanged
        ),
        "checks": records,
        "inputs_vp_and_vp_rng_unchanged": bool(state_unchanged),
    }


def _geometric(values: list[float]) -> float:
    if not values or any(
        value <= 0 or not math.isfinite(value) for value in values
    ):
        raise _IncompleteCoverage("timing ratios must be finite and positive")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _group_ratios(
    workloads: list[dict],
    candidate: int,
    rounds: int,
    *,
    control: str | None = None,
) -> dict:
    affected = [
        item
        for item in workloads
        if item["aliases"][str(candidate)]
        != item["aliases"][str(DEFAULT_BUDGET)]
    ]
    if not affected:
        return {"complete": True, "affected_workloads": [], "rounds": []}
    group_rounds = []
    per_workload = {}
    for item in affected:
        default_rep = item["aliases"][str(DEFAULT_BUDGET)]
        candidate_rep = item["aliases"][str(candidate)]
        default_times = item["round_seconds"].get(str(default_rep), [])
        if control is None:
            candidate_times = item["round_seconds"].get(str(candidate_rep), [])
        else:
            candidate_times = item[control]
        if len(default_times) != rounds or len(candidate_times) != rounds:
            return {
                "complete": False,
                "affected_workloads": [entry["name"] for entry in affected],
                "rounds": [],
            }
        ratios = [
            default / comparison
            for default, comparison in zip(
                default_times, candidate_times, strict=True
            )
        ]
        per_workload[item["name"]] = ratios
    for round_index in range(rounds):
        group_rounds.append(
            _geometric(
                [per_workload[item["name"]][round_index] for item in affected]
            )
        )
    return {
        "complete": True,
        "affected_workloads": [item["name"] for item in affected],
        "per_workload": per_workload,
        "rounds": group_rounds,
    }


def _gate_ratios(ratios: dict, rounds: int) -> dict:
    if not ratios["affected_workloads"]:
        return {"pass": False, "reason": "layout equivalent to default"}
    if not ratios.get("complete") or len(ratios.get("rounds", [])) != rounds:
        return {"pass": False, "reason": "incomplete aligned timing coverage"}
    group_rounds = ratios["rounds"]
    group_median = median(group_rounds)
    wins = sum(value > 1.0 for value in group_rounds)
    needed_wins = math.ceil(_MIN_WIN_FRACTION * rounds)
    workload_medians = {
        name: median(values) for name, values in ratios["per_workload"].items()
    }
    regression_ok = all(
        value >= _CONTROL_LOW for value in workload_medians.values()
    )
    passed = (
        group_median >= _MIN_SPEEDUP and wins >= needed_wins and regression_ok
    )
    return {
        "pass": bool(passed),
        "median_group_speedup": group_median,
        "winning_rounds": wins,
        "required_winning_rounds": needed_wins,
        "workload_median_speedups": workload_medians,
        "regression_veto_pass": bool(regression_ok),
    }


def _control_gate(ratios: dict, rounds: int) -> dict:
    if not ratios["affected_workloads"]:
        return {"pass": True, "reason": "no affected workloads"}
    if not ratios.get("complete") or len(ratios.get("rounds", [])) != rounds:
        return {"pass": False, "reason": "incomplete control coverage"}
    value = median(ratios["rounds"])
    return {
        "pass": bool(_CONTROL_LOW <= value <= _CONTROL_HIGH),
        "median_group_ratio": value,
        "accepted_interval": [_CONTROL_LOW, _CONTROL_HIGH],
    }


def _select_group(
    workloads: list[dict], valid_budgets: dict[int, bool], rounds: int
) -> dict:
    candidates = {}
    passing = []
    for budget in CANDIDATE_BUDGETS:
        if budget == DEFAULT_BUDGET:
            continue
        if not valid_budgets.get(budget, False):
            candidates[str(budget)] = {
                "pass": False,
                "reason": "candidate failed numerical validation",
            }
            continue
        ratios = _group_ratios(workloads, budget, rounds)
        gate = _gate_ratios(ratios, rounds)
        controls = _group_ratios(
            workloads, budget, rounds, control="default_control_seconds"
        )
        control_gate = _control_gate(controls, rounds)
        accepted = gate["pass"] and control_gate["pass"]
        candidates[str(budget)] = {
            "pass": bool(accepted),
            "affected_workloads": ratios.get("affected_workloads", []),
            "group_speedups": ratios.get("rounds", []),
            "median_group_speedup": gate.get("median_group_speedup"),
            "winning_rounds": gate.get("winning_rounds"),
            "required_winning_rounds": gate.get("required_winning_rounds"),
            "workload_median_speedups": gate.get(
                "workload_median_speedups", {}
            ),
            "regression_veto_pass": gate.get("regression_veto_pass"),
            "default_control_group_ratios": controls.get("rounds", []),
            "default_control_median_group_ratio": control_gate.get(
                "median_group_ratio"
            ),
            "default_control_pass": control_gate["pass"],
        }
        if accepted:
            passing.append((gate["median_group_speedup"], -budget, budget))
    if not passing:
        return {
            "budget": DEFAULT_BUDGET,
            "accepted": False,
            "reason": "no candidate passed discovery and control gates",
            "candidates": candidates,
        }
    winner = max(passing)[2]
    return {
        "budget": winner,
        "accepted": True,
        "reason": "candidate passed discovery and control gates",
        "candidates": candidates,
    }


def _measure_discovery_group(
    group: str,
    workloads: tuple[_Workload, ...],
    problems: dict[str, tuple],
    kernel_api: _KernelAPI,
    watchdog: _Watchdog,
    rounds: int = DISCOVERY_ROUNDS,
) -> list[dict]:
    records = []
    for workload in workloads:
        aliases = _layout_aliases(workload)
        representatives = tuple(dict.fromkeys(aliases.values()))
        record = {
            "name": workload.name,
            "D": workload.D,
            "K": workload.K,
            "requested_count": workload.count,
            "effective_count": _effective_count(workload),
            "grad_flags": (
                list(workload.grad_flags)
                if workload.grad_flags is not None
                else None
            ),
            "layouts": {
                str(budget): list(_layout_signature(workload, budget))
                for budget in CANDIDATE_BUDGETS
            },
            "aliases": {str(key): value for key, value in aliases.items()},
            "first_timing_call_seconds": {},
            "round_seconds": {str(rep): [] for rep in representatives},
            "default_control_seconds": [],
            "workspace_estimates": {
                str(budget): _workspace_estimate(workload, budget)
                for budget in CANDIDATE_BUDGETS
            },
        }
        if not all(
            estimate["within_limit"]
            for estimate in record["workspace_estimates"].values()
        ):
            raise _IncompleteCoverage(
                f"{workload.name} exceeds the 64 MiB storage estimate"
            )
        problem = problems[workload.name]
        for representative in representatives:
            record["first_timing_call_seconds"][
                str(representative)
            ] = _timed_invoke(
                watchdog,
                f"first:{workload.name}:{representative}",
                workload,
                problem,
                kernel_api,
                representative,
                710_000,
            )
        # One complete unmeasured warmup pass, including the control arm.
        for representative in representatives:
            _timed_invoke(
                watchdog,
                f"warmup:{workload.name}:{representative}",
                workload,
                problem,
                kernel_api,
                representative,
                720_000,
            )
        _timed_invoke(
            watchdog,
            f"warmup:{workload.name}:default_control",
            workload,
            problem,
            kernel_api,
            DEFAULT_BUDGET,
            720_000,
        )
        for round_index in range(rounds):
            observations = {rep: [] for rep in representatives}
            control_observations = []
            timing_orders = _aligned_timing_orders(aliases, round_index)
            for subround, timing_order in enumerate(timing_orders):
                if (round_index + subround) % 2 == 0:
                    control_observations.append(
                        _timed_invoke(
                            watchdog,
                            "discovery:"
                            f"{workload.name}:{round_index}:{subround}:control",
                            workload,
                            problem,
                            kernel_api,
                            DEFAULT_BUDGET,
                            730_000 + round_index,
                        )
                    )
                for representative in timing_order:
                    observations[representative].append(
                        _timed_invoke(
                            watchdog,
                            "discovery:"
                            f"{workload.name}:{round_index}:{subround}:"
                            f"{representative}",
                            workload,
                            problem,
                            kernel_api,
                            representative,
                            730_000 + round_index,
                        )
                    )
                if (round_index + subround) % 2 == 1:
                    control_observations.append(
                        _timed_invoke(
                            watchdog,
                            "discovery:"
                            f"{workload.name}:{round_index}:{subround}:control",
                            workload,
                            problem,
                            kernel_api,
                            DEFAULT_BUDGET,
                            730_000 + round_index,
                        )
                    )
            for representative, values in observations.items():
                record["round_seconds"][str(representative)].append(
                    median(values)
                )
            record["default_control_seconds"].append(
                median(control_observations)
            )
        records.append(record)
    return records


def _heldout_orders(rounds: int) -> tuple[tuple[int, ...], ...]:
    """Balance every arm's position and every pair's direction in four rounds."""
    cycle = (
        (0, 1, 2, 3),
        (1, 0, 3, 2),
        (2, 3, 0, 1),
        (3, 2, 1, 0),
    )
    return tuple(cycle[index % 4] for index in range(rounds))


def _measure_heldout_group(
    group: str,
    workloads: tuple[_Workload, ...],
    problems: dict[str, tuple],
    selected: int,
    kernel_api: _KernelAPI,
    watchdog: _Watchdog,
    rounds: int = HELDOUT_ROUNDS,
) -> list[dict]:
    del group
    records = []
    for workload in workloads:
        record = {
            "name": workload.name,
            "selected_budget": selected,
            "default_seconds": [],
            "selected_seconds": [],
            "default_control_seconds": [],
            "selected_control_seconds": [],
        }
        problem = problems[workload.name]
        arms = (
            ("selected_seconds", selected),
            ("default_seconds", DEFAULT_BUDGET),
            ("selected_control_seconds", selected),
            ("default_control_seconds", DEFAULT_BUDGET),
        )
        for name, budget in arms:
            _timed_invoke(
                watchdog,
                f"heldout_warmup:{workload.name}:{name}",
                workload,
                problem,
                kernel_api,
                budget,
                810_000,
            )
        orders = _heldout_orders(rounds)
        for round_index, order in enumerate(orders):
            for arm_index in order:
                name, budget = arms[arm_index]
                record[name].append(
                    _timed_invoke(
                        watchdog,
                        f"heldout:{workload.name}:{round_index}:{name}",
                        workload,
                        problem,
                        kernel_api,
                        budget,
                        820_000 + round_index,
                    )
                )
        records.append(record)
    return records


def _heldout_ratios(
    discovery_workloads: list[dict],
    heldout: list[dict],
    selected: int,
    rounds: int,
) -> tuple[dict, dict, dict]:
    affected = {
        item["name"]
        for item in discovery_workloads
        if item["aliases"][str(selected)]
        != item["aliases"][str(DEFAULT_BUDGET)]
    }

    def make_ratio(numerator: str, denominator: str):
        per_workload = {}
        for item in heldout:
            if item["name"] not in affected:
                continue
            if (
                len(item[numerator]) != rounds
                or len(item[denominator]) != rounds
            ):
                return {
                    "complete": False,
                    "affected_workloads": sorted(affected),
                }
            per_workload[item["name"]] = [
                left / right
                for left, right in zip(
                    item[numerator], item[denominator], strict=True
                )
            ]
        group_rounds = (
            [
                _geometric([values[index] for values in per_workload.values()])
                for index in range(rounds)
            ]
            if per_workload
            else []
        )
        return {
            "complete": len(per_workload) == len(affected),
            "affected_workloads": sorted(affected),
            "per_workload": per_workload,
            "rounds": group_rounds,
        }

    return (
        make_ratio("default_seconds", "selected_seconds"),
        make_ratio("default_seconds", "default_control_seconds"),
        make_ratio("selected_seconds", "selected_control_seconds"),
    )


def _validate_heldout(
    discovery: list[dict], heldout: list[dict], selected: int
) -> dict:
    if selected == DEFAULT_BUDGET:
        return {
            "pass": True,
            "accepted_budget": DEFAULT_BUDGET,
            "reason": "discovery retained the default",
        }
    ratios, default_control, selected_control = _heldout_ratios(
        discovery, heldout, selected, HELDOUT_ROUNDS
    )
    gate = _gate_ratios(ratios, HELDOUT_ROUNDS)
    default_gate = _control_gate(default_control, HELDOUT_ROUNDS)
    selected_gate = _control_gate(selected_control, HELDOUT_ROUNDS)
    passed = gate["pass"] and default_gate["pass"] and selected_gate["pass"]
    return {
        "pass": bool(passed),
        "accepted_budget": selected if passed else DEFAULT_BUDGET,
        "reason": (
            "held-out gates passed" if passed else "held-out gates failed"
        ),
        "affected_workloads": ratios.get("affected_workloads", []),
        "group_speedups": ratios.get("rounds", []),
        "median_group_speedup": gate.get("median_group_speedup"),
        "winning_rounds": gate.get("winning_rounds"),
        "required_winning_rounds": gate.get("required_winning_rounds"),
        "workload_median_speedups": gate.get("workload_median_speedups", {}),
        "regression_veto_pass": gate.get("regression_veto_pass"),
        "default_control_group_ratios": default_control.get("rounds", []),
        "default_control_median_group_ratio": default_gate.get(
            "median_group_ratio"
        ),
        "default_control_pass": default_gate["pass"],
        "selected_control_group_ratios": selected_control.get("rounds", []),
        "selected_control_median_group_ratio": selected_gate.get(
            "median_group_ratio"
        ),
        "selected_control_pass": selected_gate["pass"],
    }


def _emit(
    progress: Callable | None,
    stage: str,
    completed: int,
    total: int,
    message: str,
):
    if progress is not None:
        progress(
            {
                "stage": stage,
                "completed": completed,
                "total": total,
                "message": message,
            }
        )


def _defaults() -> dict[str, int]:
    return {setting: DEFAULT_BUDGET for setting in SETTING_GROUPS.values()}


def _summary(report: dict, settings: dict, status: str) -> dict:
    summary = {}
    for group, setting in SETTING_GROUPS.items():
        group_report = report.get("groups", {}).get(group, {})
        discovery = group_report.get("discovery_selection", {})
        heldout = group_report.get("heldout_validation", {})
        speedup = None
        heldout_rounds = heldout.get("group_speedups", [])
        if heldout_rounds:
            speedup = median(heldout_rounds)
        if status != "complete":
            reason = report.get("reason", "campaign incomplete")
        elif heldout:
            reason = heldout.get("reason", discovery.get("reason"))
        else:
            reason = discovery.get("reason", "historical default retained")
        summary[setting] = {
            "selected": settings[setting],
            "reason": reason,
            "heldout_speedup": speedup,
        }
    return summary


def run_campaign(
    progress: Callable | None = None,
    deadline: float | None = None,
    *,
    _clock: Callable[[], float] | None = None,
    _kernel_api: _KernelAPI | None = None,
    _recipe: tuple[_Workload, ...] | None = None,
) -> dict:
    """Run a fresh calibration and return settings, report, and status.

    ``deadline`` is an absolute monotonic watchdog supplied by the API layer.
    The private injection arguments support deterministic scheduler tests.
    """
    clock = time.monotonic if _clock is None else _clock
    started = clock()
    deadline = (
        started + WATCHDOG_SECONDS if deadline is None else float(deadline)
    )
    # On supported older Windows Pythons, monotonic() can have millisecond
    # resolution while perf_counter() resolves short numerical calls. Keep
    # absolute watchdog comparisons and high-resolution durations separate.
    watchdog = _Watchdog(
        clock,
        deadline,
        timer=time.perf_counter if _clock is None else _clock,
    )
    global_rng_before = copy.deepcopy(np.random.get_state())
    report = {
        "schema_version": 1,
        "recipe_version": 1,
        "status": "running",
        "estimated_seconds": ESTIMATED_SECONDS,
        "watchdog_seconds": WATCHDOG_SECONDS,
        "candidate_budgets": list(CANDIDATE_BUDGETS),
        "discovery_rounds": DISCOVERY_ROUNDS,
        "heldout_rounds": HELDOUT_ROUNDS,
        "memory_limit_bytes": MAX_INCREMENTAL_BYTES,
        "groups": {},
        "numerical": {},
        "timings_seconds": {
            "setup": 0.0,
            "numerical": 0.0,
            "discovery": 0.0,
            "heldout": 0.0,
            "total": 0.0,
        },
    }
    settings = _defaults()
    status = "incomplete"
    try:
        _emit(progress, "setup", 0, 4, "Preparing representative workloads")
        setup_started = clock()
        kernel_api = _load_kernel_api() if _kernel_api is None else _kernel_api
        recipe = _workloads() if _recipe is None else _recipe
        expected_groups = set(SETTING_GROUPS)
        if {workload.group for workload in recipe} != expected_groups:
            raise _IncompleteCoverage(
                "recipe does not cover all setting groups"
            )
        memory_estimates = {
            workload.name: {
                str(budget): _workspace_estimate(workload, budget)
                for budget in CANDIDATE_BUDGETS
            }
            for workload in recipe
        }
        report["memory_estimates"] = memory_estimates
        oversized = [
            workload_name
            for workload_name, estimates in memory_estimates.items()
            if not all(
                estimate["within_limit"] for estimate in estimates.values()
            )
        ]
        if oversized:
            raise _IncompleteCoverage(
                "required workloads exceed the 64 MiB storage estimate: "
                + ", ".join(oversized)
            )
        discovery_problems = {
            workload.name: _make_problem(workload, kernel_api, 10_000 + index)
            for index, workload in enumerate(recipe)
        }
        heldout_problems = {
            workload.name: _make_problem(workload, kernel_api, 20_000 + index)
            for index, workload in enumerate(recipe)
        }
        report["timings_seconds"]["setup"] = clock() - setup_started
        watchdog.check("setup_complete")

        _emit(progress, "numerical", 1, 4, "Checking numerical consistency")
        numerical_started = clock()
        dedicated = _dedicated_numerics(kernel_api, watchdog)
        report["numerical"]["dedicated"] = dedicated
        workload_numerics = {}
        valid_by_group = {
            group: {budget: True for budget in CANDIDATE_BUDGETS}
            for group in SETTING_GROUPS
        }
        baseline_valid = dedicated["pass"]
        for workload in recipe:
            result = _validate_workload(
                workload,
                discovery_problems[workload.name],
                kernel_api,
                watchdog,
            )
            workload_numerics[workload.name] = result
            baseline_valid = (
                baseline_valid
                and result["valid_budgets"][str(DEFAULT_BUDGET)]
                and result["inputs_vp_and_vp_rng_unchanged"]
            )
            for budget in CANDIDATE_BUDGETS:
                valid_by_group[workload.group][budget] = bool(
                    valid_by_group[workload.group][budget]
                    and result["valid_budgets"][str(budget)]
                )
        report["numerical"]["workloads"] = workload_numerics
        report["numerical"]["valid_by_group"] = {
            group: {str(key): value for key, value in values.items()}
            for group, values in valid_by_group.items()
        }
        report["timings_seconds"]["numerical"] = clock() - numerical_started
        if not baseline_valid:
            status = "invalid"
            report["reason"] = "baseline numerical validation failed"
            raise _IncompleteCoverage(report["reason"])

        _emit(progress, "discovery", 2, 4, "Comparing candidate settings")
        discovery_started = clock()
        selections = {}
        for group in SETTING_GROUPS:
            group_recipe = tuple(
                workload for workload in recipe if workload.group == group
            )
            label = {
                "pdf": "PDF settings",
                "entropy_grad": "entropy gradient settings",
                "entropy_value": "entropy value settings",
            }[group]
            _emit(progress, group, 2, 4, f"Comparing {label}")
            discovery = _measure_discovery_group(
                group,
                group_recipe,
                discovery_problems,
                kernel_api,
                watchdog,
            )
            selection = _select_group(
                discovery, valid_by_group[group], DISCOVERY_ROUNDS
            )
            selections[group] = selection
            report["groups"][group] = {
                "setting": SETTING_GROUPS[group],
                "discovery": discovery,
                "discovery_selection": selection,
            }
        report["timings_seconds"]["discovery"] = clock() - discovery_started

        _emit(progress, "heldout", 3, 4, "Validating selected settings")
        heldout_started = clock()
        final_budgets = {}
        for group in SETTING_GROUPS:
            group_recipe = tuple(
                workload for workload in recipe if workload.group == group
            )
            selected = selections[group]["budget"]
            label = {
                "pdf": "PDF setting",
                "entropy_grad": "entropy gradient setting",
                "entropy_value": "entropy value setting",
            }[group]
            _emit(progress, f"heldout_{group}", 3, 4, f"Validating {label}")
            heldout = _measure_heldout_group(
                group,
                group_recipe,
                heldout_problems,
                selected,
                kernel_api,
                watchdog,
            )
            validation = _validate_heldout(
                report["groups"][group]["discovery"], heldout, selected
            )
            report["groups"][group]["heldout"] = heldout
            report["groups"][group]["heldout_validation"] = validation
            final_budgets[group] = validation["accepted_budget"]
        report["timings_seconds"]["heldout"] = clock() - heldout_started
        watchdog.check("campaign_complete")
        settings = {
            SETTING_GROUPS[group]: final_budgets[group]
            for group in SETTING_GROUPS
        }
        status = "complete"
        _emit(progress, "complete", 4, 4, "Measurements complete")
    except _WatchdogExpired as exc:
        status = "incomplete"
        report["reason"] = str(exc)
    except (MemoryError, _IncompleteCoverage) as exc:
        if status != "invalid":
            status = "incomplete"
        report.setdefault("reason", str(exc))
    finally:
        report["timings_seconds"]["total"] = clock() - started
        report["watchdog"] = {
            "deadline": deadline,
            "last_checkpoint": watchdog.last_checkpoint,
            "maximum_kernel_call_seconds": watchdog.maximum_call_seconds,
            "stopped": status == "incomplete" and clock() >= deadline,
        }
        report["numpy_global_rng_unchanged"] = _legacy_rng_equal(
            global_rng_before, np.random.get_state()
        )
        if not report["numpy_global_rng_unchanged"]:
            status = "invalid"
            settings = _defaults()
            report["reason"] = "campaign changed NumPy's global RNG state"
        if status != "complete":
            settings = _defaults()
        report["status"] = status
        report["settings"] = dict(settings)
        report["summary"] = _summary(report, settings, status)
    return {"settings": settings, "report": report, "status": status}


__all__ = ["run_campaign"]
