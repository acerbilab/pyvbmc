"""Complete variational-fit adapter for the Stage 4 Torch experiment.

This module is experiment scaffolding, not a production backend.  It executes
the current :func:`optimize_vp` and :meth:`VBMC.final_boost` code objects in an
isolated globals dictionary.  Only their numerical objective, Adam optimizer,
and SciPy entry point are replaced.  The production modules are never patched.
Source hashes make copied orchestration fail closed when its assumptions drift.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import inspect
import math
import textwrap
import time
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

_EXPECTED_SOURCE_HASHES = {
    "optimize_vp": "63956ae1afdfc116e9b809a8c2fc2ab49e2e6d211aa2136ace986d1ec782ca14",
    "_eval_full_elcbo": "a1a614a2fb2ed222e70d11736101a5cfab3d366cdd1da7dc1679db8ae7b24094",
    "_sieve": "51d8c000986fa4dcd8752c3921551c7979f1c03294b9c5bedfefe7021d7921cf",
    "_vb_init": "8886d106f3bfcb713fa6af5b8e2fa0e9fc5dd3cbc39b4ddb27bc109a3524c1a5",
    "minimize_adam": "122c432424ae562d736b1ccc55a53088a779e2d6a26bf8104aa5974477f9a293",
    "final_boost": "58ae501e08de6fbc012a9f225cd36144dd6fd11207bc3be19b242b9f620e7c2c",
}


def _source_hash(function: Any) -> str:
    path = Path(inspect.getsourcefile(function))
    module_source = path.read_text(encoding="utf-8")
    tree = ast.parse(module_source)
    line = function.__code__.co_firstlineno
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function.__name__
        and node.lineno <= line <= node.end_lineno
    ]
    if len(nodes) != 1:
        raise RuntimeError(
            f"cannot identify source definition for {function.__qualname__}"
        )
    source = textwrap.dedent(
        ast.get_source_segment(module_source, nodes[0])
    ).rstrip()
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def source_hashes(check: bool = True) -> dict[str, str]:
    """Return hashes of every production function whose control flow is reused."""
    from pyvbmc.vbmc import variational_optimization as vo
    from pyvbmc.vbmc.minimize_adam import minimize_adam
    from pyvbmc.vbmc.vbmc import VBMC

    functions = {
        "optimize_vp": vo.optimize_vp,
        "_eval_full_elcbo": vo._eval_full_elcbo,
        "_sieve": vo._sieve,
        "_vb_init": vo._vb_init,
        "minimize_adam": minimize_adam,
        "final_boost": VBMC.final_boost,
    }
    observed = {name: _source_hash(fun) for name, fun in functions.items()}
    if check:
        mismatches = {
            name: (expected, observed[name])
            for name, expected in _EXPECTED_SOURCE_HASHES.items()
            if observed[name] != expected
        }
        if mismatches:
            details = ", ".join(
                f"{name}: expected {old}, observed {new}"
                for name, (old, new) in mismatches.items()
            )
            raise RuntimeError(
                "Torch VI adapter source guard failed; review the isolated "
                f"dispatch against current production control flow ({details})."
            )
    return observed


def _clone_function(function: Any, replacements: Mapping[str, Any]) -> Any:
    namespace = dict(function.__globals__)
    namespace.update(replacements)
    cloned = types.FunctionType(
        function.__code__,
        namespace,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    cloned.__kwdefaults__ = copy.copy(function.__kwdefaults__)
    cloned.__annotations__ = copy.copy(function.__annotations__)
    return cloned


def _array_hash(array: Any) -> str:
    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(repr(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _theta_without_mutation(vp: Any) -> np.ndarray:
    clone = copy.deepcopy(vp)
    return np.asarray(clone.get_parameters(), dtype=np.float64).copy()


def _input_digest(vp: Any, gp: Any, options: Any, optim_state: Any) -> str:
    digest = hashlib.sha256()
    arrays = [vp.w, vp.eta, vp.mu, vp.sigma, vp.lambd, gp.X, gp.y]
    for value in arrays:
        digest.update(_array_hash(value).encode())
    digest.update(_array_hash(_theta_without_mutation(vp)).encode())
    digest.update(
        repr(sorted((str(k), repr(options[k])) for k in options)).encode()
    )
    digest.update(
        repr(
            sorted((str(k), repr(v)) for k, v in optim_state.items())
        ).encode()
    )
    for posterior in gp.posteriors:
        for name in ("hyp", "alpha", "L", "sW"):
            digest.update(_array_hash(getattr(posterior, name)).encode())
        digest.update(str(bool(posterior.L_chol)).encode())
    return digest.hexdigest()


def _as_numpy(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    return np.array(array, copy=True)


def _candidate_record(vp: Any, kind: str, candidate_id: str) -> dict[str, Any]:
    theta = _theta_without_mutation(vp)
    mean, covariance = vp.moments(orig_flag=False, cov_flag=True)
    scales = np.asarray(vp.lambd) * np.asarray(vp.sigma)
    weights = np.asarray(vp.w)
    return {
        "candidate_id": candidate_id,
        "kind": kind,
        "K": int(vp.K),
        "theta": theta,
        "theta_hash": _array_hash(theta),
        "mu": np.array(vp.mu, copy=True),
        "sigma": np.array(vp.sigma, copy=True),
        "lambd": np.array(vp.lambd, copy=True),
        "w": np.array(vp.w, copy=True),
        "eta": np.array(vp.eta, copy=True),
        "mean_internal": np.asarray(mean),
        "covariance_internal": np.asarray(covariance),
        "valid_weights": bool(
            np.all(np.isfinite(weights))
            and np.all(weights >= 0)
            and np.isclose(np.sum(weights), 1.0)
        ),
        "valid_scales": bool(
            np.all(np.isfinite(scales)) and np.all(scales > 0)
        ),
        "transformer_id": id(vp.parameter_transformer),
    }


class _Trace:
    def __init__(self, backend: str, device: str):
        self.backend = backend
        self.device = device
        self.events: list[dict[str, Any]] = []
        self.candidates: list[dict[str, Any]] = []
        self.candidate_vps: dict[str, Any] = {}
        self.phase = "setup"
        self.start_id = -1
        self.evaluation_id = 0
        self.dispatch_counts = {
            "sieve": 0,
            "adam": 0,
            "scipy": 0,
            "fine": 0,
            "prune": 0,
        }
        self.timings = {
            "setup": 0.0,
            "gp_extract_transfer": 0.0,
            "vp_bounds_transfer": 0.0,
            "candidate_sieve": 0.0,
            "optimizer": 0.0,
            "fine_scoring_pruning": 0.0,
            "output_transfer": 0.0,
            "rng_generation_transfer": 0.0,
            "rng_staging_transfer": 0.0,
            "stopping_sync": 0.0,
            "scipy_bridge": 0.0,
            "objective_resident": 0.0,
            "objective_enqueue": 0.0,
        }
        self.crossings = {"host_to_device": 0, "device_to_host": 0}
        self.optimizer_runs: list[dict[str, Any]] = []
        self.scipy_runs: list[dict[str, Any]] = []

    @contextlib.contextmanager
    def timed(self, name: str):
        started = time.perf_counter()
        yield
        self.timings[name] += time.perf_counter() - started

    def event(self, kind: str, **values: Any) -> dict[str, Any]:
        record = {
            "event_id": len(self.events),
            "kind": kind,
            "phase": self.phase,
            "start_id": self.start_id,
            "evaluation_id": self.evaluation_id,
        }
        record.update(values)
        self.events.append(record)
        return record

    def retain(self, vp: Any, kind: str, suffix: str | int) -> dict[str, Any]:
        candidate_id = f"{kind}:{suffix}"
        record = _candidate_record(vp, kind, candidate_id)
        self.candidates.append(record)
        clone = copy.deepcopy(vp)
        clone._rng = np.random.default_rng(0)
        self.candidate_vps[candidate_id] = clone
        return record


class _RecordingRNG:
    """Transparent Generator proxy which records candidate and entropy draws."""

    def __init__(self, generator: np.random.Generator, trace: _Trace):
        self._generator = generator
        self._trace = trace
        self._adam_draw_start = None

    @property
    def bit_generator(self):
        return self._generator.bit_generator

    def standard_normal(self, size=None, dtype=np.float64, out=None):
        state_before = copy.deepcopy(self._generator.bit_generator.state)
        started = time.perf_counter()
        value = self._generator.standard_normal(
            size=size, dtype=dtype, out=out
        )
        self._trace.timings["rng_generation_transfer"] += (
            time.perf_counter() - started
        )
        retain_actual = np.size(value) <= 2**16 and (
            self._trace.phase != "adam"
            or self._adam_draw_start != self._trace.start_id
        )
        if self._trace.phase == "adam":
            self._adam_draw_start = self._trace.start_id
        self._trace.event(
            "rng_standard_normal",
            shape=tuple(np.shape(value)),
            dtype=np.asarray(value).dtype.str,
            value_hash=_array_hash(value),
            generator_state_before=state_before,
            storage="actual" if retain_actual else "reconstructable",
            epsilon=np.array(value, copy=True) if retain_actual else None,
        )
        return value

    def integers(self, *args, **kwargs):
        value = self._generator.integers(*args, **kwargs)
        self._trace.event("rng_integers", value=np.asarray(value).copy())
        return value

    def permutation(self, *args, **kwargs):
        value = self._generator.permutation(*args, **kwargs)
        self._trace.event("rng_permutation", value=np.asarray(value).copy())
        return value

    def random(self, *args, **kwargs):
        value = self._generator.random(*args, **kwargs)
        self._trace.event("rng_random", value=np.asarray(value).copy())
        return value

    def __getattr__(self, name: str) -> Any:
        return getattr(self._generator, name)


def _sync(device: str) -> None:
    if str(device).startswith("cuda"):
        import torch

        torch.cuda.synchronize(device)


def minimize_adam_tensor(
    f,
    x0,
    lb=None,
    ub=None,
    tol_fun=0.001,
    max_iter=10000,
    master_min=0.001,
    master_max=0.1,
    master_decay=200,
    use_early_stopping=True,
    *,
    device="cpu",
    trace: _Trace | None = None,
    fixed_updates: int | None = None,
    timing: dict[str, Any] | None = None,
):
    """Tensor-resident transcription of production's custom Adam routine."""
    import torch

    if fixed_updates is not None:
        if isinstance(fixed_updates, bool) or int(fixed_updates) <= 0:
            raise ValueError(
                "fixed_updates must be a positive integer or None"
            )
        max_iter = int(fixed_updates)
        use_early_stopping = False
    max_iter = int(max_iter)
    if max_iter < 20:
        raise ValueError(
            "custom Adam needs at least 20 updates for final averaging"
        )

    x = torch.as_tensor(x0, dtype=torch.float64, device=device).clone()
    initial_x_numpy = (
        np.asarray(_as_numpy(x0), dtype=np.float64).copy()
        if trace is not None
        else None
    )
    n_vars = x.numel()
    lower = (
        torch.full_like(x, -torch.inf)
        if lb is None
        else torch.as_tensor(lb, dtype=torch.float64, device=device)
    )
    upper = (
        torch.full_like(x, torch.inf)
        if ub is None
        else torch.as_tensor(ub, dtype=torch.float64, device=device)
    )
    m = torch.zeros_like(x)
    v = torch.zeros_like(x)
    x_tab = torch.zeros((n_vars, max_iter), dtype=torch.float64, device=device)
    y_tab = torch.full(
        (max_iter,), torch.nan, dtype=torch.float64, device=device
    )
    grad_tab = (
        torch.zeros((n_vars, max_iter), dtype=torch.float64, device=device)
        if fixed_updates is not None and trace is not None
        else None
    )
    fudge_factor = math.sqrt(np.spacing(1))
    beta_1, beta_2 = 0.9, 0.999
    batch_size, min_iter = 20, 40
    tol_x, tol_x_max = 0.001, 0.1
    tol_fun_max = tol_fun * 100
    stopping_reason = "max_iter"
    sync_count = 0

    for i in range(max_iter):
        y, grad = f(x)
        y = torch.as_tensor(y, dtype=torch.float64, device=device).reshape(())
        grad = torch.as_tensor(
            grad, dtype=torch.float64, device=device
        ).reshape(x.shape)
        if grad_tab is not None:
            grad_tab[:, i] = grad.detach()
        y_tab[i] = y.detach()
        m = beta_1 * m + (1 - beta_1) * grad
        v = beta_2 * v + (1 - beta_2) * grad**2
        m_hat = m / (1 - beta_1 ** (i + 1))
        v_hat = v / (1 - beta_2 ** (i + 1))
        step_size = master_min + (master_max - master_min) * math.exp(
            -(i + 1) / master_decay
        )
        x = x - step_size * m_hat / (torch.sqrt(v_hat) + fudge_factor)
        x = torch.minimum(upper, torch.maximum(lower, x))
        x_tab[:, i] = x.detach()

        minibatch_end = math.remainder(i + 1, batch_size) == 0
        if use_early_stopping and minibatch_end and i + 1 >= min_iter:
            started = time.perf_counter()
            _sync(device)
            y_window = y_tab[i - batch_size + 1 : i + 1].detach().cpu().numpy()
            x_window = (
                x_tab[:, i - 2 * batch_size + 1 : i + 1].detach().cpu().numpy()
            )
            sync_count += 1
            if trace is not None:
                trace.crossings["device_to_host"] += 2
            xxp = np.linspace(
                -(batch_size - 1) / 2, (batch_size - 1) / 2, batch_size
            )
            p, covariance = np.polyfit(xxp, y_window, 1, cov=True)
            slope = p[0]
            slope_err = np.sqrt(covariance[0, 0] + tol_fun**2)
            slope_err_max = np.sqrt(covariance[0, 0] + tol_fun_max**2)
            dx = np.sqrt(
                np.sum(
                    (
                        np.mean(x_window[:, batch_size:], axis=1)
                        - np.mean(x_window[:, :batch_size], axis=1)
                    )
                    ** 2
                    / batch_size
                )
            )
            if trace is not None:
                trace.timings["stopping_sync"] += time.perf_counter() - started
                trace.event(
                    "adam_stopping_check",
                    iteration=i + 1,
                    slope=float(slope),
                    slope_error=float(slope_err),
                    slope_error_max=float(slope_err_max),
                    displacement=float(dx),
                )
            if timing is not None:
                timing["stopping_sync"] += time.perf_counter() - started
                timing["sync_count"] += 1
            if (dx < tol_x and abs(slope) < slope_err_max) or (
                abs(slope) < slope_err and dx < tol_x_max
            ):
                stopping_reason = "early_stopping"
                break

    iterations = i + 1
    start = i - batch_size + 1
    x_mean = torch.mean(x_tab[:, start : i + 1], dim=1)
    y_mean = torch.mean(y_tab[start : i + 1])
    transfer_started = time.perf_counter()
    _sync(device)
    y_numpy = y_tab[:iterations].detach().cpu().numpy().copy()
    x_numpy = None
    grad_numpy = None
    if trace is not None:
        x_numpy = x_tab[:, :iterations].detach().cpu().numpy().copy()
        grad_numpy = (
            None
            if grad_tab is None
            else grad_tab[:, :iterations].detach().cpu().numpy().copy()
        )
    if timing is not None:
        timing["required_transfer"] += time.perf_counter() - transfer_started
        timing["sync_count"] += 1
    if trace is not None:
        trace.crossings["device_to_host"] += 2 + int(grad_numpy is not None)
        objective_events = [
            event
            for event in trace.events
            if event["kind"] == "objective"
            and event["phase"] == "adam"
            and event["start_id"] == trace.start_id
        ]
        pre_update = np.column_stack((initial_x_numpy, x_numpy[:, :-1]))
        if objective_events and len(objective_events) != iterations:
            raise RuntimeError(
                "Adam objective trace lost its one-call-per-update association"
            )
        if objective_events:
            for index, event in enumerate(objective_events):
                event["value"] = float(y_numpy[index])
                event["theta_hash"] = _array_hash(pre_update[:, index])
        trace.optimizer_runs.append(
            {
                "start_id": trace.start_id,
                "iterations": iterations,
                "stopping_reason": stopping_reason,
                "stopping_syncs": sync_count,
                "fixed_updates": fixed_updates,
                "pre_update_objectives": y_numpy,
                "gradients": grad_numpy,
                "post_update_theta_hashes": [
                    _array_hash(x_numpy[:, j]) for j in range(iterations)
                ],
            }
        )
    return x_mean, y_mean, x_tab[:, :iterations], y_numpy, iterations


def _materialize_candidate(vp: Any, theta: Any) -> Any:
    candidate = copy.deepcopy(vp)
    candidate.set_parameters(np.asarray(_as_numpy(theta), dtype=np.float64))
    return candidate


def _matching_candidate(trace: _Trace, theta: Any) -> Any:
    theta_hash = _array_hash(_as_numpy(theta))
    for candidate_id, candidate in trace.candidate_vps.items():
        if candidate_id.startswith("sieve_ranked:"):
            if _array_hash(_theta_without_mutation(candidate)) == theta_hash:
                return candidate
    return trace._base_vp


def _import_core():
    try:
        import torch_vi_core as core
    except ModuleNotFoundError:
        from dev.scripts import torch_vi_core as core
    return core


def _build_isolated_adapter(
    backend: str,
    device: str,
    torch_gp: Any,
    trace: _Trace,
    fixed_updates: int | None,
):
    import scipy as sp

    from pyvbmc.vbmc import variational_optimization as vo

    objective_counter = 0
    core = _import_core() if backend == "torch" else None
    vp_templates: dict[int, tuple[Any, Any]] = {}
    bound_templates: dict[int, tuple[Any, Any]] = {}

    def dispatched_neg_elcbo(
        theta,
        gp,
        vp,
        beta,
        Ns,
        compute_grad=True,
        compute_var=False,
        theta_bnd=None,
        entropy_alpha=0.0,
        separate_K=False,
    ):
        nonlocal objective_counter
        if entropy_alpha != 0:
            raise ValueError(
                "Stage 4 adapter supports current entropy_alpha=0 only"
            )
        trace.evaluation_id = objective_counter
        objective_counter += 1
        phase = trace.phase
        if phase not in trace.dispatch_counts:
            phase = "fine" if separate_K else "sieve"
        trace.dispatch_counts[phase] += 1
        epsilon = None
        epsilon_hash = None

        if backend == "numpy":
            started = time.perf_counter()
            result = vo._neg_elcbo(
                np.asarray(theta),
                gp,
                vp,
                beta,
                Ns,
                compute_grad,
                compute_var,
                theta_bnd,
                entropy_alpha,
                separate_K,
            )
        else:
            import torch

            transfer_started = time.perf_counter()
            cached_vp = vp_templates.get(id(vp))
            if cached_vp is None or cached_vp[0] is not vp:
                template = core.TorchVPTemplate.from_vp(vp, device)
                vp_templates[id(vp)] = (vp, template)
                trace.crossings["host_to_device"] += 1
            else:
                template = cached_vp[1]
            bounds = None
            if theta_bnd is not None:
                cached_bounds = bound_templates.get(id(theta_bnd))
                if cached_bounds is None or cached_bounds[0] is not theta_bnd:
                    bounds = core.TorchBounds.from_dict(theta_bnd, device)
                    bound_templates[id(theta_bnd)] = (theta_bnd, bounds)
                    trace.crossings["host_to_device"] += 1
                else:
                    bounds = cached_bounds[1]
            trace.timings["vp_bounds_transfer"] += (
                time.perf_counter() - transfer_started
            )

            theta_is_tensor = torch.is_tensor(theta)
            if theta_is_tensor:
                theta_tensor = theta.to(dtype=torch.float64, device=device)
            else:
                theta_tensor = torch.as_tensor(
                    theta, dtype=torch.float64, device=device
                )
                trace.crossings["host_to_device"] += 1
            if Ns > 0:
                half = int(math.ceil(Ns / 2))
                epsilon_numpy = vp.rng.standard_normal((vp.K, half, vp.D))
                rng_started = time.perf_counter()
                epsilon_hash = _array_hash(epsilon_numpy)
                epsilon = torch.as_tensor(
                    epsilon_numpy, dtype=torch.float64, device=device
                )
                trace.timings["rng_staging_transfer"] += (
                    time.perf_counter() - rng_started
                )
                trace.crossings["host_to_device"] += 1
            started = time.perf_counter()
            result = core.neg_elcbo(
                theta_tensor,
                torch_gp,
                template,
                beta,
                Ns,
                compute_grad,
                compute_var,
                bounds,
                separate_K,
                epsilon=epsilon,
            )
            if not theta_is_tensor or separate_K:
                result = tuple(_as_numpy(item) for item in result)
                trace.crossings["device_to_host"] += sum(
                    item is not None for item in result
                )

        objective_elapsed = time.perf_counter() - started
        timing_key = (
            "objective_enqueue"
            if backend == "torch" and str(device).startswith("cuda")
            else "objective_resident"
        )
        trace.timings[timing_key] += objective_elapsed
        resident_adam = (
            backend == "torch" and phase == "adam" and theta_is_tensor
        )
        value = None if resident_adam else _as_numpy(result[0])
        gradient = (
            None if resident_adam or not compute_grad else _as_numpy(result[1])
        )
        trace.event(
            "objective",
            backend=backend,
            Ns=int(Ns),
            compute_grad=bool(compute_grad),
            compute_var=bool(compute_var),
            separate_K=bool(separate_K),
            theta_hash=(
                None if resident_adam else _array_hash(_as_numpy(theta))
            ),
            value=None if value is None else float(value),
            gradient_hash=None if gradient is None else _array_hash(gradient),
            gradient_norm=(
                None if gradient is None else float(np.linalg.norm(gradient))
            ),
            epsilon_shape=None if epsilon is None else tuple(epsilon.shape),
            epsilon_hash=epsilon_hash,
        )
        return result

    cloned_eval = None

    def eval_full(*args, **kwargs):
        nonlocal cloned_eval
        vp = args[2] if len(args) > 2 else kwargs["vp"]
        theta = args[1] if len(args) > 1 else kwargs["theta"]
        if hasattr(theta, "detach"):
            theta = np.asarray(_as_numpy(theta), dtype=np.float64)
            if len(args) > 1:
                args = list(args)
                args[1] = theta
                args = tuple(args)
            else:
                kwargs = dict(kwargs)
                kwargs["theta"] = theta
            trace.crossings["device_to_host"] += 1
        kind = (
            "prune"
            if trace.phase == "prune" or vp.K < trace._target_K
            else "fine"
        )
        old_phase = trace.phase
        trace.phase = kind
        started = time.perf_counter()
        try:
            result = cloned_eval(*args, **kwargs)
        finally:
            trace.timings["fine_scoring_pruning"] += (
                time.perf_counter() - started
            )
            trace.phase = old_phase
        candidate = _materialize_candidate(vp, theta)
        index = len(
            [
                c
                for c in trace.candidates
                if c["kind"] in {"midpoint", "endpoint", "prune_attempt"}
            ]
        )
        label = (
            "prune_attempt"
            if kind == "prune"
            else (
                "midpoint"
                if (args[0] if args else kwargs["idx"]) % 2 == 0
                else "endpoint"
            )
        )
        record = trace.retain(candidate, label, index)
        stats = result
        idx = args[0] if args else kwargs["idx"]
        record["fine_nelbo"] = float(stats["nelbo"][idx])
        record["fine_nelcbo"] = float(stats["nelcbo"][idx])
        record["I_sk_shape"] = tuple(
            stats["I_sk"][idx, :, : candidate.K].shape
        )
        record["J_sjk_shape"] = tuple(
            stats["J_sjk"][idx, :, : candidate.K, : candidate.K].shape
        )
        return result

    cloned_eval = _clone_function(
        vo._eval_full_elcbo, {"_neg_elcbo": dispatched_neg_elcbo}
    )

    cloned_sieve = None

    def sieve(*args, **kwargs):
        old_phase = trace.phase
        trace.phase = "sieve"
        started = time.perf_counter()
        try:
            result = cloned_sieve(*args, **kwargs)
        finally:
            trace.timings["candidate_sieve"] += time.perf_counter() - started
            trace.phase = old_phase
        for index, (candidate, candidate_type) in enumerate(
            zip(result[0], result[1])
        ):
            record = trace.retain(candidate, "sieve_ranked", index)
            record["candidate_type"] = int(candidate_type)
            record["rank"] = index
        return result

    cloned_sieve = _clone_function(
        vo._sieve, {"_neg_elcbo": dispatched_neg_elcbo}
    )

    def tensor_adam(fun, x0, **kwargs):
        trace.start_id += 1
        old_phase = trace.phase
        trace.phase = "adam"
        selected_base = _matching_candidate(trace, x0)
        trace.retain(
            _materialize_candidate(selected_base, x0),
            "selected_start",
            trace.start_id,
        )
        started = time.perf_counter()
        try:
            return minimize_adam_tensor(
                fun,
                x0,
                device=device,
                trace=trace,
                fixed_updates=fixed_updates,
                **kwargs,
            )
        finally:
            trace.timings["optimizer"] += time.perf_counter() - started
            trace.phase = old_phase

    original_minimize = sp.optimize.minimize

    def scipy_minimize(fun, x0, *args, **kwargs):
        trace.start_id += 1
        old_phase = trace.phase
        trace.phase = "scipy"
        selected_base = _matching_candidate(trace, x0)
        trace.retain(
            _materialize_candidate(selected_base, x0),
            "selected_start",
            trace.start_id,
        )
        started = time.perf_counter()
        try:
            result = original_minimize(fun, x0, *args, **kwargs)
        finally:
            elapsed = time.perf_counter() - started
            trace.timings["optimizer"] += elapsed
            trace.timings["scipy_bridge"] += elapsed
            trace.phase = old_phase
        trace.scipy_runs.append(
            {
                "start_id": trace.start_id,
                "method": kwargs.get("method") or "BFGS (SciPy default)",
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "iterations": int(getattr(result, "nit", -1)),
                "function_evaluations": int(getattr(result, "nfev", -1)),
                "gradient_evaluations": int(getattr(result, "njev", -1)),
            }
        )
        return result

    scipy_proxy = types.SimpleNamespace(
        optimize=types.SimpleNamespace(minimize=scipy_minimize),
        linalg=sp.linalg,
    )

    def numpy_adam(fun, x0, **kwargs):
        trace.start_id += 1
        old_phase = trace.phase
        trace.phase = "adam"
        selected_base = _matching_candidate(trace, x0)
        trace.retain(
            _materialize_candidate(selected_base, x0),
            "selected_start",
            trace.start_id,
        )
        if fixed_updates is not None:
            kwargs["max_iter"] = int(fixed_updates)
            kwargs["use_early_stopping"] = False
        gradients = []

        def captured_fun(theta):
            value, gradient = fun(theta)
            if fixed_updates is not None:
                gradients.append(np.array(gradient, copy=True))
            return value, gradient

        started = time.perf_counter()
        try:
            result = vo.minimize_adam(captured_fun, x0, **kwargs)
        finally:
            trace.timings["optimizer"] += time.perf_counter() - started
            trace.phase = old_phase
        iterations = int(result[4])
        reason = (
            "fixed_updates"
            if fixed_updates is not None
            else (
                "max_iter"
                if iterations == int(kwargs.get("max_iter", 10000))
                else "early_stopping"
            )
        )
        trace.optimizer_runs.append(
            {
                "start_id": trace.start_id,
                "iterations": iterations,
                "stopping_reason": reason,
                "stopping_syncs": (
                    iterations // 20 if reason != "fixed_updates" else 0
                ),
                "fixed_updates": fixed_updates,
                "pre_update_objectives": np.array(result[3], copy=True),
                "gradients": (
                    None
                    if fixed_updates is None
                    else np.column_stack(gradients)
                ),
                "post_update_theta_hashes": [
                    _array_hash(result[2][:, j]) for j in range(iterations)
                ],
            }
        )
        return result

    optimize = _clone_function(
        vo.optimize_vp,
        {
            "_sieve": sieve,
            "_eval_full_elcbo": eval_full,
            "_neg_elcbo": dispatched_neg_elcbo,
            "minimize_adam": tensor_adam if backend == "torch" else numpy_adam,
            "sp": scipy_proxy,
        },
    )
    optimize._stage4_neg_elcbo = dispatched_neg_elcbo
    return optimize


def _normal_budgets(
    state: Mapping[str, Any], options: Any, optim_state: dict, vp: Any
):
    fast = state.get("fast_opts_N")
    slow = state.get("slow_opts_N")
    if fast is None:
        fast = math.ceil(options.eval("ns_elbo", {"K": vp.K}))
        if not optim_state.get("recompute_var_post") and not options.get(
            "always_refit_vp"
        ):
            fast = math.ceil(fast * options.get("ns_elbo_incr"))
    if slow is None:
        slow = (
            options.get("elbo_starts")
            if (
                optim_state.get("recompute_var_post")
                or options.get("always_refit_vp")
            )
            else 1
        )
    if optim_state.get("recompute_var_post") or options.get("always_refit_vp"):
        optim_state["recompute_var_post"] = False
    return int(fast), int(slow)


def _boost_adapter(optimize, trace: _Trace):
    from pyvbmc.vbmc.vbmc import VBMC

    return _clone_function(VBMC.final_boost, {"optimize_vp": optimize})


def _populate_preboost_stats(
    vp: Any, gp: Any, options: Any, objective: Any
) -> None:
    """Supply the score final_boost expects for synthetic pre-boost states."""
    per_component = int(
        math.ceil(options.eval("ns_ent_fine", {"K": vp.K}) / vp.K)
    )
    result = objective(
        vp.get_parameters(),
        gp,
        vp,
        0.0,
        per_component,
        False,
        True,
        None,
        0.0,
        True,
    )
    variance = float(result[4])
    vp.stats = {
        "elbo": -float(result[0]),
        "elbo_sd": math.sqrt(variance),
        "e_log_joint": float(result[2]),
        "e_log_joint_sd": math.sqrt(float(result[7])),
        "entropy": float(result[3]),
        "entropy_sd": math.sqrt(float(result[8])),
        "stable": False,
        "I_sk": np.array(result[9], copy=True),
        "J_sjk": np.array(result[10], copy=True),
    }


def _build_clean_adapter(
    backend: str,
    device: str,
    torch_gp: Any,
    fixed_updates: int | None,
    timing: dict[str, Any],
):
    """Build source-guarded orchestration without diagnostic observation."""
    import scipy as sp

    from pyvbmc.vbmc import variational_optimization as vo

    core = _import_core() if backend == "torch" else None
    vp_templates: dict[int, tuple[Any, Any]] = {}
    bound_templates: dict[int, tuple[Any, Any]] = {}
    optimizer_runs: list[dict[str, Any]] = []
    scipy_runs: list[dict[str, Any]] = []
    phase = "sieve"

    def transferred(callable_, synchronization_points=0):
        started = time.perf_counter()
        value = callable_()
        timing["required_transfer"] += time.perf_counter() - started
        if str(device).startswith("cuda"):
            timing["sync_count"] += synchronization_points
        return value

    def dispatched_neg_elcbo(
        theta,
        gp,
        vp,
        beta,
        Ns,
        compute_grad=True,
        compute_var=False,
        theta_bnd=None,
        entropy_alpha=0.0,
        separate_K=False,
    ):
        if entropy_alpha != 0:
            raise ValueError(
                "Stage 4 adapter supports current entropy_alpha=0 only"
            )
        if backend == "numpy":
            return vo._neg_elcbo(
                np.asarray(theta),
                gp,
                vp,
                beta,
                Ns,
                compute_grad,
                compute_var,
                theta_bnd,
                entropy_alpha,
                separate_K,
            )

        import torch

        cached_vp = vp_templates.get(id(vp))
        if cached_vp is None or cached_vp[0] is not vp:
            template = transferred(
                lambda: core.TorchVPTemplate.from_vp(vp, device)
            )
            vp_templates[id(vp)] = (vp, template)
        else:
            template = cached_vp[1]
        bounds = None
        if theta_bnd is not None:
            cached_bounds = bound_templates.get(id(theta_bnd))
            if cached_bounds is None or cached_bounds[0] is not theta_bnd:
                bounds = transferred(
                    lambda: core.TorchBounds.from_dict(theta_bnd, device)
                )
                bound_templates[id(theta_bnd)] = (theta_bnd, bounds)
            else:
                bounds = cached_bounds[1]

        theta_is_tensor = torch.is_tensor(theta)
        if theta_is_tensor:
            theta_tensor = theta.to(dtype=torch.float64, device=device)
        else:
            theta_tensor = transferred(
                lambda: torch.as_tensor(
                    theta, dtype=torch.float64, device=device
                )
            )
        epsilon = None
        if Ns > 0:
            started = time.perf_counter()
            epsilon_numpy = vp.rng.standard_normal(
                (vp.K, int(math.ceil(Ns / 2)), vp.D)
            )
            timing["rng_generation"] += time.perf_counter() - started
            epsilon = transferred(
                lambda: torch.as_tensor(
                    epsilon_numpy, dtype=torch.float64, device=device
                )
            )
        result = core.neg_elcbo(
            theta_tensor,
            torch_gp,
            template,
            beta,
            Ns,
            compute_grad,
            compute_var,
            bounds,
            separate_K,
            epsilon=epsilon,
        )
        if not theta_is_tensor or separate_K:
            result = transferred(
                lambda: tuple(_as_numpy(item) for item in result),
                sum(item is not None for item in result),
            )
        return result

    cloned_eval = None

    def eval_full(*args, **kwargs):
        nonlocal phase
        theta = args[1] if len(args) > 1 else kwargs["theta"]
        if hasattr(theta, "detach"):
            theta = transferred(
                lambda: np.asarray(_as_numpy(theta), dtype=np.float64), 1
            )
            if len(args) > 1:
                args = list(args)
                args[1] = theta
                args = tuple(args)
            else:
                kwargs = dict(kwargs)
                kwargs["theta"] = theta
        old_phase = phase
        phase = "fine"
        try:
            return cloned_eval(*args, **kwargs)
        finally:
            phase = old_phase

    cloned_eval = _clone_function(
        vo._eval_full_elcbo, {"_neg_elcbo": dispatched_neg_elcbo}
    )
    cloned_sieve = _clone_function(
        vo._sieve, {"_neg_elcbo": dispatched_neg_elcbo}
    )

    def tensor_adam(fun, x0, **kwargs):
        nonlocal phase
        old_phase = phase
        phase = "adam"
        started = time.perf_counter()
        try:
            result = minimize_adam_tensor(
                fun,
                x0,
                device=device,
                trace=None,
                fixed_updates=fixed_updates,
                timing=timing,
                **kwargs,
            )
        finally:
            timing["optimizer"] += time.perf_counter() - started
            phase = old_phase
        iterations = int(result[4])
        reason = (
            "fixed_updates"
            if fixed_updates is not None
            else (
                "max_iter"
                if iterations == int(kwargs.get("max_iter", 10000))
                else "early_stopping"
            )
        )
        optimizer_runs.append(
            {"iterations": iterations, "stopping_reason": reason}
        )
        return result

    def numpy_adam(fun, x0, **kwargs):
        nonlocal phase
        old_phase = phase
        phase = "adam"
        if fixed_updates is not None:
            kwargs["max_iter"] = int(fixed_updates)
            kwargs["use_early_stopping"] = False
        started = time.perf_counter()
        try:
            result = vo.minimize_adam(fun, x0, **kwargs)
        finally:
            timing["optimizer"] += time.perf_counter() - started
            phase = old_phase
        iterations = int(result[4])
        reason = (
            "fixed_updates"
            if fixed_updates is not None
            else (
                "max_iter"
                if iterations == int(kwargs.get("max_iter", 10000))
                else "early_stopping"
            )
        )
        optimizer_runs.append(
            {"iterations": iterations, "stopping_reason": reason}
        )
        return result

    original_minimize = sp.optimize.minimize

    def scipy_minimize(fun, x0, *args, **kwargs):
        nonlocal phase
        old_phase = phase
        phase = "scipy"
        started = time.perf_counter()
        try:
            result = original_minimize(fun, x0, *args, **kwargs)
        finally:
            timing["optimizer"] += time.perf_counter() - started
            phase = old_phase
        scipy_runs.append(
            {
                "method": kwargs.get("method") or "BFGS (SciPy default)",
                "success": bool(result.success),
                "iterations": int(getattr(result, "nit", -1)),
            }
        )
        return result

    scipy_proxy = types.SimpleNamespace(
        optimize=types.SimpleNamespace(minimize=scipy_minimize),
        linalg=sp.linalg,
    )
    optimize = _clone_function(
        vo.optimize_vp,
        {
            "_sieve": cloned_sieve,
            "_eval_full_elcbo": eval_full,
            "_neg_elcbo": dispatched_neg_elcbo,
            "minimize_adam": tensor_adam if backend == "torch" else numpy_adam,
            "sp": scipy_proxy,
        },
    )
    optimize._stage4_neg_elcbo = dispatched_neg_elcbo
    return optimize, {
        "optimizer_runs": optimizer_runs,
        "scipy_runs": scipy_runs,
        "dispatch": {
            "objective_backend": backend,
            "sieve": True,
            "optimizer": True,
            "fine_scoring": True,
            "pruning": True,
        },
    }


def _run_step_clean(
    state: Mapping[str, Any],
    backend: str,
    device: str,
    seed: int,
    boost: bool,
    fixed_updates: int | None,
    fast_opts_N: int | None,
    slow_opts_N: int | None,
    K: int | None,
) -> dict[str, Any]:
    """Execute the same fit without diagnostic observation overhead."""
    if backend not in {"numpy", "torch"}:
        raise ValueError("backend must be 'numpy' or 'torch'")
    if backend == "numpy" and device != "cpu":
        raise ValueError("NumPy backend only supports device='cpu'")
    required = {"vp", "gp", "options", "optim_state"}
    missing = required.difference(state)
    if missing:
        raise ValueError(f"state lacks required entries: {sorted(missing)}")

    source_guard_started = time.perf_counter()
    hashes = source_hashes(check=True)
    source_guard = time.perf_counter() - source_guard_started
    core_module = None
    pre_sync = 0.0
    if backend == "torch":
        core_module = _import_core()
        started = time.perf_counter()
        _sync(device)
        pre_sync = time.perf_counter() - started
    timing = {
        "setup": 0.0,
        "gp_extract_transfer": 0.0,
        "required_transfer": 0.0,
        # NumPy draws occur inside the unwrapped production objective and are
        # therefore retained only in the fair whole-fit total.
        "rng_generation": 0.0 if backend == "torch" else None,
        "optimizer": 0.0,
        "stopping_sync": 0.0,
        "output_sync": 0.0,
        "sync_count": int(str(device).startswith("cuda")),
        "sync_count_semantics": (
            "blocking transfer and explicit synchronization points; "
            "not a literal cuda.synchronize call count"
        ),
        "pre_sync_outside_total": pre_sync,
        "source_guard_outside_total": source_guard,
    }
    total_started = time.perf_counter()
    setup_started = time.perf_counter()
    vp = copy.deepcopy(state["vp"])
    gp = copy.deepcopy(state["gp"])
    options = copy.deepcopy(state["options"])
    optim_state = copy.deepcopy(state["optim_state"])
    vp._rng = np.random.default_rng(seed)
    input_transformer = vp.parameter_transformer
    target_k = int(K if K is not None else state.get("K", vp.K))
    timing["setup"] = time.perf_counter() - setup_started

    torch_gp = None
    if backend == "torch":
        TorchGP = core_module.TorchGP
        started = time.perf_counter()
        _sync(device)
        torch_gp = TorchGP.from_gp(gp, device=device)
        _sync(device)
        timing["gp_extract_transfer"] = time.perf_counter() - started
        if str(device).startswith("cuda"):
            timing["sync_count"] += 2

    optimize, metadata = _build_clean_adapter(
        backend, device, torch_gp, fixed_updates, timing
    )
    effective_fast = None
    effective_slow = None
    effective_k = target_k
    var_ss = None
    pruned = 0
    boost_result = None
    if boost:
        from pyvbmc.vbmc.vbmc import VBMC

        if vp.stats is None:
            _populate_preboost_stats(
                vp, gp, options, optimize._stage4_neg_elcbo
            )
        effective_k = max(vp.K, options.get("min_final_components"))
        effective_fast = int(
            math.ceil(
                math.ceil(options.eval("ns_elbo", {"K": effective_k}))
                * options.get("ns_elbo_incr")
            )
        )
        effective_slow = 1
        dummy = types.SimpleNamespace(
            options=options,
            optim_state=optim_state,
            logger=types.SimpleNamespace(warning=lambda *args, **kwargs: None),
            _validate_final_boost_tolerance=VBMC._validate_final_boost_tolerance,
            _is_valid_final_boost_score=VBMC._is_valid_final_boost_score,
            _accept_final_boost_candidate=VBMC._accept_final_boost_candidate,
        )
        output_vp, elbo, elbo_sd, changed = _boost_adapter(optimize, None)(
            dummy, vp, gp
        )
        boost_result = {
            "changed": bool(changed),
            "elbo": float(elbo),
            "elbo_sd": float(elbo_sd),
        }
    else:
        if fast_opts_N is None or slow_opts_N is None:
            derived_fast, derived_slow = _normal_budgets(
                state, options, optim_state, vp
            )
            fast_opts_N = derived_fast if fast_opts_N is None else fast_opts_N
            slow_opts_N = derived_slow if slow_opts_N is None else slow_opts_N
        effective_fast = int(fast_opts_N)
        effective_slow = int(slow_opts_N)
        output_vp, var_ss, pruned = optimize(
            options,
            optim_state,
            vp,
            gp,
            effective_fast,
            effective_slow,
            target_k,
        )

    started = time.perf_counter()
    _sync(device)
    timing["output_sync"] = time.perf_counter() - started
    if str(device).startswith("cuda"):
        timing["sync_count"] += 1
    optimizer_runs = metadata["optimizer_runs"]
    result = {
        "complete": output_vp.stats is not None,
        "mode": (
            "fixed_replay" if fixed_updates is not None else "complete_fit"
        ),
        "production_policy_complete": fixed_updates is None,
        "K": int(output_vp.K),
        "pruned": int(pruned),
        "iterations_by_start": [run["iterations"] for run in optimizer_runs],
        "stopping_reasons": [run["stopping_reason"] for run in optimizer_runs],
        "scipy_runs": metadata["scipy_runs"],
        "dispatch": metadata["dispatch"],
        "transformer_identity": (
            output_vp.parameter_transformer is input_transformer
        ),
        "budgets": {
            "K": int(effective_k),
            "fast_opts_N": effective_fast,
            "slow_opts_N": effective_slow,
            "fixed_updates": fixed_updates,
        },
        "boost": boost_result,
    }
    timing["total"] = time.perf_counter() - total_started
    return {
        "vp": output_vp,
        "var_ss": var_ss,
        "pruned": int(pruned),
        "backend": backend,
        "device": device,
        "boost": bool(boost),
        "fixed_updates": fixed_updates,
        "trace_enabled": False,
        "result": result,
        "timing": timing,
        "source_hashes": hashes,
    }


def run_step(
    state: Mapping[str, Any],
    backend: str = "numpy",
    device: str = "cpu",
    seed: int = 1701,
    boost: bool = False,
    fixed_updates: int | None = None,
    trace_enabled: bool = True,
    *,
    fast_opts_N: int | None = None,
    slow_opts_N: int | None = None,
    K: int | None = None,
) -> dict[str, Any]:
    """Run one complete main-loop or final-boost variational fit.

    ``state`` must contain ``vp``, ``gp``, ``options`` and ``optim_state``.
    Every object is copied and the copied posterior receives a private seeded
    NumPy stream.  ``fixed_updates`` caps every Adam start and disables its
    early stopping; deterministic SciPy fits report that mode as inapplicable.
    """
    if not isinstance(trace_enabled, (bool, np.bool_)):
        raise TypeError("trace_enabled must be boolean")
    if not trace_enabled:
        return _run_step_clean(
            state,
            backend,
            device,
            seed,
            boost,
            fixed_updates,
            fast_opts_N,
            slow_opts_N,
            K,
        )
    if backend not in {"numpy", "torch"}:
        raise ValueError("backend must be 'numpy' or 'torch'")
    if backend == "numpy" and device != "cpu":
        raise ValueError("NumPy backend only supports device='cpu'")
    required = {"vp", "gp", "options", "optim_state"}
    missing = required.difference(state)
    if missing:
        raise ValueError(f"state lacks required entries: {sorted(missing)}")

    hashes = source_hashes(check=True)
    trace = _Trace(backend, device)
    total_started = time.perf_counter()
    setup_started = time.perf_counter()
    before_digest = _input_digest(
        state["vp"], state["gp"], state["options"], state["optim_state"]
    )
    vp = copy.deepcopy(state["vp"])
    gp = copy.deepcopy(state["gp"])
    options = copy.deepcopy(state["options"])
    optim_state = copy.deepcopy(state["optim_state"])
    private_rng = np.random.default_rng(seed)
    recording_rng = _RecordingRNG(private_rng, trace)
    vp._rng = recording_rng
    input_transformer = vp.parameter_transformer
    trace._base_vp = vp
    target_K = int(K if K is not None else state.get("K", vp.K))
    trace._target_K = target_K
    trace.timings["setup"] += time.perf_counter() - setup_started

    torch_gp = None
    if backend == "torch":
        TorchGP = _import_core().TorchGP
        started = time.perf_counter()
        _sync(device)
        torch_gp = TorchGP.from_gp(gp, device=device)
        _sync(device)
        trace.timings["gp_extract_transfer"] += time.perf_counter() - started
        trace.crossings["host_to_device"] += 1

    optimize = _build_isolated_adapter(
        backend, device, torch_gp, trace, fixed_updates
    )
    var_ss = None
    pruned = 0
    boost_record = None
    effective_fast_opts = None
    effective_slow_opts = None
    effective_k = target_K
    if boost:
        from pyvbmc.vbmc.vbmc import VBMC

        if vp.stats is None:
            old_phase = trace.phase
            trace.phase = "fine"
            _populate_preboost_stats(
                vp, gp, options, optimize._stage4_neg_elcbo
            )
            trace.phase = old_phase
        if (
            fixed_updates is not None
            and vp.K == 1
            and optim_state.get("entropy_switch")
        ):
            trace.event(
                "fixed_updates_inapplicable", reason="deterministic_scipy"
            )
        dummy = types.SimpleNamespace(
            options=options,
            optim_state=optim_state,
            logger=types.SimpleNamespace(warning=lambda *args, **kwargs: None),
            _validate_final_boost_tolerance=VBMC._validate_final_boost_tolerance,
            _is_valid_final_boost_score=VBMC._is_valid_final_boost_score,
            _accept_final_boost_candidate=VBMC._accept_final_boost_candidate,
        )
        pre = trace.retain(vp, "boost_pre", 0)
        effective_k = max(vp.K, options.get("min_final_components"))
        effective_fast_opts = int(
            math.ceil(
                math.ceil(options.eval("ns_elbo", {"K": effective_k}))
                * options.get("ns_elbo_incr")
            )
        )
        effective_slow_opts = 1
        boosted, elbo, elbo_sd, changed = _boost_adapter(optimize, trace)(
            dummy, vp, gp
        )
        candidate = trace.retain(boosted, "boost_returned", 0)
        boost_record = {
            "pre": pre,
            "candidate_or_returned": candidate,
            "changed": bool(changed),
            "elbo": float(elbo),
            "elbo_sd": float(elbo_sd),
            "guard_tolerance": options.get("tol_elcbo_boost"),
        }
        output_vp = boosted
        pruned = 0
    else:
        if fast_opts_N is None or slow_opts_N is None:
            derived_fast, derived_slow = _normal_budgets(
                state, options, optim_state, vp
            )
            fast_opts_N = derived_fast if fast_opts_N is None else fast_opts_N
            slow_opts_N = derived_slow if slow_opts_N is None else slow_opts_N
        output_vp, var_ss, pruned = optimize(
            options,
            optim_state,
            vp,
            gp,
            int(fast_opts_N),
            int(slow_opts_N),
            target_K,
        )
        effective_fast_opts = int(fast_opts_N)
        effective_slow_opts = int(slow_opts_N)

    final = trace.retain(output_vp, "final", 0)
    final["stats"] = copy.deepcopy(output_vp.stats)
    final["transformer_identity"] = (
        output_vp.parameter_transformer is input_transformer
    )
    fine_candidates = [
        c for c in trace.candidates if c["kind"] in {"midpoint", "endpoint"}
    ]
    selected_fine = (
        min(fine_candidates, key=lambda c: c["fine_nelcbo"])
        if fine_candidates
        else None
    )
    if selected_fine is not None:
        selected_fine["selected"] = True
    prune_attempts = [
        candidate
        for candidate in trace.candidates
        if candidate["kind"] == "prune_attempt"
    ]
    accepted_k = selected_fine["K"] if selected_fine is not None else target_K
    for index, attempt in enumerate(prune_attempts):
        next_k = (
            prune_attempts[index + 1]["K"]
            if index + 1 < len(prune_attempts)
            else output_vp.K
        )
        accepted = next_k < attempt["K"] or (
            index + 1 == len(prune_attempts) and output_vp.K == attempt["K"]
        )
        attempt["pruning_decision"] = "accepted" if accepted else "rejected"
        attempt["K_before"] = int(accepted_k)
        if accepted:
            accepted_k = attempt["K"]
    if boost_record is not None:
        boost_record["candidate"] = selected_fine
        boost_record["returned_candidate_id"] = final["candidate_id"]

    output_started = time.perf_counter()
    _sync(device)
    trace.timings["output_transfer"] += time.perf_counter() - output_started
    after_digest = _input_digest(
        state["vp"], state["gp"], state["options"], state["optim_state"]
    )
    total = time.perf_counter() - total_started
    trace.timings["total"] = total
    trace.timings["unattributed_orchestration"] = max(
        0.0,
        total
        - sum(
            trace.timings[key]
            for key in (
                "setup",
                "gp_extract_transfer",
                "candidate_sieve",
                "optimizer",
                "fine_scoring_pruning",
                "output_transfer",
            )
        ),
    )
    trace.timings["subregions_overlap"] = True
    iterations = [record["iterations"] for record in trace.optimizer_runs]
    stopping = [record["stopping_reason"] for record in trace.optimizer_runs]
    complete = output_vp.stats is not None and bool(
        fine_candidates or boost_record
    )
    result = {
        "complete": bool(complete),
        "mode": (
            "fixed_replay" if fixed_updates is not None else "complete_fit"
        ),
        "production_policy_complete": fixed_updates is None,
        "K": int(output_vp.K),
        "budgets": {
            "K": int(effective_k),
            "fast_opts_N": effective_fast_opts,
            "slow_opts_N": effective_slow_opts,
            "fixed_updates": fixed_updates,
        },
        "var_ss": None if var_ss is None else _as_numpy(var_ss),
        "pruned": int(pruned),
        "stats": copy.deepcopy(output_vp.stats),
        "iterations_by_start": iterations,
        "stopping_reasons": stopping,
        "scipy_runs": copy.deepcopy(trace.scipy_runs),
        "optimizer_runs": copy.deepcopy(trace.optimizer_runs),
        "dispatch_counts": copy.deepcopy(trace.dispatch_counts),
        "transformer_identity": output_vp.parameter_transformer
        is input_transformer,
        "transformer_identity_note": (
            "final_boost preserves production's independent pre-boost copy; "
            "the outer VBMC caller owns any live-state identity restoration"
            if boost
            and output_vp.parameter_transformer is not input_transformer
            else None
        ),
        "selected_candidate_id": (
            None if selected_fine is None else selected_fine["candidate_id"]
        ),
        "boost": boost_record,
        "fixed_updates_applicable": bool(trace.optimizer_runs),
    }
    return {
        "vp": output_vp,
        "var_ss": var_ss,
        "pruned": int(pruned),
        "backend": backend,
        "device": device,
        "boost": bool(boost),
        "fixed_updates": fixed_updates,
        "result": result,
        "trace": trace.events,
        "candidates": trace.candidates,
        "timing": {
            **trace.timings,
            "crossings": trace.crossings,
            "dispatch_counts": trace.dispatch_counts,
        },
        "source_hashes": hashes,
        "input_unchanged": before_digest == after_digest,
        "_context": {
            "gp": gp,
            "options": options,
            "torch_gp": torch_gp,
            "candidate_vps": trace.candidate_vps,
        },
    }


def run_numpy_control(
    state: Mapping[str, Any],
    seed: int = 1701,
    boost: bool = False,
    *,
    fast_opts_N: int | None = None,
    slow_opts_N: int | None = None,
    K: int | None = None,
) -> dict[str, Any]:
    """Run uninstrumented production orchestration for adapter-overhead gates."""
    from pyvbmc.vbmc.variational_optimization import optimize_vp
    from pyvbmc.vbmc.vbmc import VBMC

    required = {"vp", "gp", "options", "optim_state"}
    missing = required.difference(state)
    if missing:
        raise ValueError(f"state lacks required entries: {sorted(missing)}")
    hashes = source_hashes(check=True)
    setup_started = time.perf_counter()
    before_digest = _input_digest(
        state["vp"], state["gp"], state["options"], state["optim_state"]
    )
    vp = copy.deepcopy(state["vp"])
    gp = copy.deepcopy(state["gp"])
    options = copy.deepcopy(state["options"])
    optim_state = copy.deepcopy(state["optim_state"])
    vp._rng = np.random.default_rng(seed)
    setup = time.perf_counter() - setup_started
    fit_started = time.perf_counter()
    if boost:
        if vp.stats is None:
            from pyvbmc.vbmc.variational_optimization import _neg_elcbo

            _populate_preboost_stats(vp, gp, options, _neg_elcbo)
        dummy = types.SimpleNamespace(
            options=options,
            optim_state=optim_state,
            logger=types.SimpleNamespace(warning=lambda *args, **kwargs: None),
            _validate_final_boost_tolerance=VBMC._validate_final_boost_tolerance,
            _is_valid_final_boost_score=VBMC._is_valid_final_boost_score,
            _accept_final_boost_candidate=VBMC._accept_final_boost_candidate,
        )
        output_vp, elbo, elbo_sd, changed = VBMC.final_boost(dummy, vp, gp)
        var_ss = None
        pruned = 0
        boost_result = {
            "elbo": float(elbo),
            "elbo_sd": float(elbo_sd),
            "changed": bool(changed),
        }
    else:
        target_k = int(K if K is not None else state.get("K", vp.K))
        if fast_opts_N is None or slow_opts_N is None:
            derived_fast, derived_slow = _normal_budgets(
                state, options, optim_state, vp
            )
            fast_opts_N = derived_fast if fast_opts_N is None else fast_opts_N
            slow_opts_N = derived_slow if slow_opts_N is None else slow_opts_N
        output_vp, var_ss, pruned = optimize_vp(
            options,
            optim_state,
            vp,
            gp,
            int(fast_opts_N),
            int(slow_opts_N),
            target_k,
        )
        boost_result = None
    fit = time.perf_counter() - fit_started
    after_digest = _input_digest(
        state["vp"], state["gp"], state["options"], state["optim_state"]
    )
    return {
        "vp": output_vp,
        "var_ss": var_ss,
        "pruned": int(pruned),
        "boost": boost_result,
        "timing": {"setup": setup, "fit": fit, "total": setup + fit},
        "source_hashes": hashes,
        "input_unchanged": before_digest == after_digest,
    }


def rescore_candidates(
    run_or_candidates: Mapping[str, Any] | Sequence[Any],
    gp: Any = None,
    options: Any = None,
    seeds: Sequence[int] = (2701, 2702, 2703, 2704, 2705),
    samples: int | None = None,
) -> dict[str, Any]:
    """Re-score retained candidates with independent common NumPy streams."""
    from pyvbmc.vbmc.variational_optimization import _neg_elcbo

    if (
        isinstance(run_or_candidates, Mapping)
        and "_context" in run_or_candidates
    ):
        context = run_or_candidates["_context"]
        retained_prefixes = (
            "selected_start:",
            "midpoint:",
            "endpoint:",
            "prune_attempt:",
            "final:",
            "boost_pre:",
            "boost_returned:",
        )
        retained = {
            candidate_id: candidate
            for candidate_id, candidate in context["candidate_vps"].items()
            if candidate_id.startswith(retained_prefixes)
        }
        candidate_vps = {}
        aliases = {}
        hashes = {}
        for candidate_id, candidate in retained.items():
            key = (
                int(candidate.K),
                _array_hash(_theta_without_mutation(candidate)),
            )
            canonical = hashes.get(key)
            if canonical is None:
                hashes[key] = candidate_id
                candidate_vps[candidate_id] = candidate
            else:
                aliases[candidate_id] = canonical
        gp = context["gp"] if gp is None else gp
        options = context["options"] if options is None else options
    else:
        if gp is None or options is None:
            raise ValueError(
                "gp and options are required for an external candidate list"
            )
        aliases = {}
        candidate_vps = {
            str(i): candidate for i, candidate in enumerate(run_or_candidates)
        }
    if len(seeds) != 5:
        raise ValueError(
            "the Phase 3 diagnostic requires exactly five scoring streams"
        )

    scores: dict[str, list[dict[str, Any]]] = {}
    for candidate_id, candidate in candidate_vps.items():
        candidate_scores = []
        for seed in seeds:
            vp = copy.deepcopy(candidate)
            vp._rng = np.random.default_rng(int(seed))
            per_component = (
                int(math.ceil(options.eval("ns_ent_fine", {"K": vp.K}) / vp.K))
                if samples is None
                else int(math.ceil(samples / vp.K))
            )
            theta = vp.get_parameters().copy()
            result = _neg_elcbo(
                theta, gp, vp, 0.0, per_component, False, True, None, 0.0, True
            )
            variance = float(result[4])
            candidate_scores.append(
                {
                    "seed": int(seed),
                    "samples_per_component": per_component,
                    "samples_total": per_component * vp.K,
                    "elbo": -float(result[0]),
                    "elbo_sd": math.sqrt(variance),
                    "elcbo_beta5": -float(result[0]) - 5 * math.sqrt(variance),
                    "G": float(result[2]),
                    "H": float(result[3]),
                    "I_sk_shape": tuple(np.shape(result[9])),
                    "J_sjk_shape": tuple(np.shape(result[10])),
                }
            )
        scores[candidate_id] = candidate_scores
    return {
        "seeds": [int(seed) for seed in seeds],
        "scores": scores,
        "aliases": aliases,
        "common_streams": True,
        "note": "Draws are identical only for candidates with the same K and sample shape.",
    }


def compare_fixed_replays(
    numpy_run: Mapping[str, Any], torch_run: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare fixed-work Adam traces after their single batched transfers."""
    if (
        numpy_run.get("fixed_updates") is None
        or torch_run.get("fixed_updates") is None
    ):
        raise ValueError("both runs must be fixed-update replays")
    if numpy_run["fixed_updates"] != torch_run["fixed_updates"]:
        raise ValueError("fixed-update replay lengths differ")
    numpy_starts = numpy_run["result"]["optimizer_runs"]
    torch_starts = torch_run["result"]["optimizer_runs"]
    if len(numpy_starts) != len(torch_starts):
        raise ValueError(
            "fixed-update runs selected different numbers of starts"
        )

    comparisons = []
    for numpy_start, torch_start in zip(numpy_starts, torch_starts):
        objective_difference = np.asarray(
            torch_start["pre_update_objectives"]
        ) - np.asarray(numpy_start["pre_update_objectives"])
        gradient_difference = np.asarray(
            torch_start["gradients"]
        ) - np.asarray(numpy_start["gradients"])
        denominator = np.maximum(
            np.abs(numpy_start["gradients"]), np.spacing(1)
        )
        comparisons.append(
            {
                "start_id": int(numpy_start["start_id"]),
                "updates": int(numpy_start["iterations"]),
                "objective_max_abs": float(
                    np.max(np.abs(objective_difference))
                ),
                "objective_rms": float(
                    np.sqrt(np.mean(objective_difference**2))
                ),
                "gradient_max_abs": float(np.max(np.abs(gradient_difference))),
                "gradient_max_rel": float(
                    np.max(np.abs(gradient_difference) / denominator)
                ),
                "gradient_rms": float(
                    np.sqrt(np.mean(gradient_difference**2))
                ),
                "post_update_theta_hash_matches": [
                    left == right
                    for left, right in zip(
                        numpy_start["post_update_theta_hashes"],
                        torch_start["post_update_theta_hashes"],
                    )
                ],
            }
        )
    return {
        "fixed_updates": int(numpy_run["fixed_updates"]),
        "starts": comparisons,
        "branch_diverged": any(
            not all(comparison["post_update_theta_hash_matches"])
            for comparison in comparisons
        ),
    }


__all__ = [
    "compare_fixed_replays",
    "minimize_adam_tensor",
    "rescore_candidates",
    "run_numpy_control",
    "run_step",
    "source_hashes",
]
