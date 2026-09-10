"""Scoped ``torch.compile`` adapter for the Stage 4 VI prototype.

This module is experiment scaffolding, not a production backend.  It compiles
only :func:`torch_vi_core._neg_elcbo_forward` and temporarily installs the
compiled callable at the module-global lookup used by ``neg_elcbo``.  The
existing wrapper therefore continues to own validation, the detached float64
parameter leaf, ``torch.autograd.grad``, result detachment, and tuple layout.

Compilation is lazy: constructing :class:`CompiledForward` creates the
``torch.compile`` wrapper, while the first installed call pays tracing and
code-generation costs.  Compiler failures are never converted to eager
execution by this adapter.  Dynamo and Inductor diagnostics use private,
version-sensitive APIs defensively and report unavailable fields explicitly.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_INDUCTOR_METRIC_NAMES = (
    "generated_kernel_count",
    "generated_cpp_vec_kernel_count",
    "num_bytes_accessed",
    "ir_nodes_pre_fusion",
    "num_comprehensive_padding",
    "num_matches_for_scatter_upon_const_tensor",
    "cpp_to_dtype_count",
    "cpp_outer_loop_fused_inner_counts",
    "nodes_num_elem",
    "node_runtimes",
)


def _torch(torch_module: Any = None) -> Any:
    if torch_module is not None:
        return torch_module
    return importlib.import_module("torch")


def _qualified_name(value: Any) -> str:
    module = getattr(value, "__module__", type(value).__module__)
    name = getattr(value, "__qualname__", type(value).__qualname__)
    return f"{module}.{name}"


def _json_safe(value: Any) -> Any:
    """Copy common compiler diagnostics into JSON-safe Python containers."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_safe(item) for item in value]
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return repr(value)


def compiler_capabilities(torch_module: Any = None) -> dict[str, Any]:
    """Describe the compiler hooks available in the imported Torch build."""
    torch = _torch(torch_module)
    compiler = getattr(torch, "compiler", None)
    cuda = getattr(torch, "cuda", None)
    version = getattr(torch, "version", None)
    report = {
        "torch_version": str(getattr(torch, "__version__", "unknown")),
        "torch_path": str(getattr(torch, "__file__", "unknown")),
        "torch_compile": callable(getattr(torch, "compile", None)),
        "torch_compiler_reset": callable(getattr(compiler, "reset", None)),
        "cuda_build": _json_safe(getattr(version, "cuda", None)),
        "cuda_available": bool(
            cuda is not None
            and callable(getattr(cuda, "is_available", None))
            and cuda.is_available()
        ),
        "dynamo": False,
        "dynamo_counters": False,
        "dynamo_reset": False,
        "inductor_metrics": False,
        "errors": {},
    }
    try:
        dynamo = importlib.import_module("torch._dynamo")
        report["dynamo"] = True
        report["dynamo_reset"] = callable(getattr(dynamo, "reset", None))
        utils = importlib.import_module("torch._dynamo.utils")
        report["dynamo_counters"] = hasattr(utils, "counters")
    except Exception as error:  # private diagnostics differ across releases
        report["errors"]["dynamo"] = repr(error)
    try:
        metrics = importlib.import_module("torch._inductor.metrics")
        report["inductor_metrics"] = callable(getattr(metrics, "reset", None))
    except Exception as error:  # private diagnostics differ across releases
        report["errors"]["inductor_metrics"] = repr(error)
    return report


def reset_compiler_state(torch_module: Any = None) -> dict[str, Any]:
    """Reset compile caches and diagnostic counters for one cold session.

    Call this before constructing a :class:`CompiledForward`.  Calling it
    between repetitions invalidates the hot-cache measurement.
    """
    torch = _torch(torch_module)
    if not callable(getattr(torch, "compile", None)):
        raise RuntimeError("this Torch build does not provide torch.compile")

    actions: list[str] = []
    compiler = getattr(torch, "compiler", None)
    reset = getattr(compiler, "reset", None)
    if callable(reset):
        reset()
        actions.append("torch.compiler.reset")
    else:
        dynamo = importlib.import_module("torch._dynamo")
        reset = getattr(dynamo, "reset", None)
        if not callable(reset):
            raise RuntimeError(
                "this Torch build exposes no compiler reset API"
            )
        reset()
        actions.append("torch._dynamo.reset")

    unavailable: dict[str, str] = {}
    try:
        utils = importlib.import_module("torch._dynamo.utils")
        counters = getattr(utils, "counters")
        counters.clear()
        actions.append("torch._dynamo.utils.counters.clear")
    except Exception as error:
        unavailable["dynamo_counters"] = repr(error)
    try:
        metrics = importlib.import_module("torch._inductor.metrics")
        metrics.reset()
        actions.append("torch._inductor.metrics.reset")
    except Exception as error:
        unavailable["inductor_metrics"] = repr(error)
    return {"actions": actions, "unavailable": unavailable}


def snapshot_compile_evidence(torch_module: Any = None) -> dict[str, Any]:
    """Return a JSON-safe snapshot of Dynamo counters and Inductor metrics."""
    torch = _torch(torch_module)
    report: dict[str, Any] = {
        "torch_version": str(getattr(torch, "__version__", "unknown")),
        "dynamo_counters": {},
        "inductor_metrics": {},
        "errors": {},
    }
    try:
        utils = importlib.import_module("torch._dynamo.utils")
        report["dynamo_counters"] = _json_safe(getattr(utils, "counters"))
    except Exception as error:
        report["errors"]["dynamo_counters"] = repr(error)
    try:
        metrics = importlib.import_module("torch._inductor.metrics")
        report["inductor_metrics"] = {
            name: _json_safe(getattr(metrics, name))
            for name in _INDUCTOR_METRIC_NAMES
            if hasattr(metrics, name)
        }
    except Exception as error:
        report["errors"]["inductor_metrics"] = repr(error)
    return report


def evidence_delta(before: Any, after: Any) -> Any:
    """Subtract numeric leaves in two evidence trees, retaining new values."""
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        return {
            str(key): evidence_delta(before.get(key), after[key])
            for key in after
        }
    if (
        isinstance(before, (int, float))
        and not isinstance(before, bool)
        and isinstance(after, (int, float))
        and not isinstance(after, bool)
    ):
        return after - before
    return _json_safe(after)


def source_metadata() -> dict[str, str]:
    """Return the adapter source path and byte-level SHA-256 for archiving."""
    path = Path(__file__).resolve()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


class CompiledForward:
    """Reusable compiled forward and scoped ``torch_vi_core`` interception.

    Full-variance/per-component scoring is value-only in ``neg_elcbo`` and can
    contain thousands of Python entropy chunks at the largest follow-up shape.
    It stays on the exact eager forward by default, avoiding a huge unrolled
    graph while compiling the ordinary objective and its backward pass.
    """

    def __init__(
        self,
        core: Any,
        *,
        backend: Any = "inductor",
        fullgraph: bool = True,
        dynamic: bool | None = False,
        mode: str | None = None,
        options: Mapping[str, Any] | None = None,
        eager_full_variance: bool = True,
        torch_module: Any = None,
    ) -> None:
        torch = _torch(torch_module)
        compile_function = getattr(torch, "compile", None)
        if not callable(compile_function):
            raise RuntimeError(
                "this Torch build does not provide torch.compile"
            )
        if mode is not None and options is not None:
            raise ValueError("torch.compile accepts mode or options, not both")
        if not isinstance(fullgraph, bool):
            raise TypeError("fullgraph must be boolean")
        if dynamic is not None and not isinstance(dynamic, bool):
            raise TypeError("dynamic must be boolean or None")
        if not isinstance(eager_full_variance, bool):
            raise TypeError("eager_full_variance must be boolean")
        if not hasattr(core, "_neg_elcbo_forward"):
            raise AttributeError("core has no _neg_elcbo_forward callable")
        original = core._neg_elcbo_forward
        if not callable(original):
            raise TypeError("core._neg_elcbo_forward must be callable")

        compile_kwargs: dict[str, Any] = {
            "backend": backend,
            "fullgraph": fullgraph,
            "dynamic": dynamic,
        }
        if mode is not None:
            compile_kwargs["mode"] = mode
        if options is not None:
            compile_kwargs["options"] = dict(options)

        self.core = core
        self.torch = torch
        self.original = original
        self.compiled = compile_function(original, **compile_kwargs)
        self.config = {
            "backend": backend if isinstance(backend, str) else repr(backend),
            "fullgraph": fullgraph,
            "dynamic": dynamic,
            "mode": mode,
            "options": None if options is None else _json_safe(dict(options)),
            "eager_full_variance": eager_full_variance,
        }
        self.call_count = 0
        self.compiled_call_count = 0
        self.eager_full_variance_call_count = 0
        self._installed = False
        self._timeline: list[dict[str, Any]] = []

        def dispatch(*args: Any, **kwargs: Any) -> Any:
            self.call_count += 1
            compute_var = kwargs.get(
                "compute_var", args[5] if len(args) > 5 else False
            )
            separate_K = kwargs.get(
                "separate_K", args[7] if len(args) > 7 else False
            )
            if eager_full_variance and (compute_var or separate_K):
                self.eager_full_variance_call_count += 1
                return self.original(*args, **kwargs)
            self.compiled_call_count += 1
            return self.compiled(*args, **kwargs)

        self.installed_callable = dispatch
        self._created = snapshot_compile_evidence(torch)
        self.mark("created")

    @contextlib.contextmanager
    def install(self):
        """Install this wrapper for a scope and restore the prior callable."""
        if self._installed:
            raise RuntimeError("this compiled forward is already installed")
        prior = self.core._neg_elcbo_forward
        if prior is not self.original:
            raise RuntimeError(
                "core._neg_elcbo_forward changed after session construction"
            )
        self._installed = True
        self.core._neg_elcbo_forward = self.installed_callable
        try:
            yield self
        finally:
            self.core._neg_elcbo_forward = prior
            self._installed = False

    def mark(self, label: str) -> dict[str, Any]:
        """Append and return a labeled compiler-evidence snapshot."""
        if not isinstance(label, str) or not label:
            raise ValueError("evidence label must be a non-empty string")
        record = {
            "label": label,
            "call_count": int(self.call_count),
            "compiled_call_count": int(self.compiled_call_count),
            "eager_full_variance_call_count": int(
                self.eager_full_variance_call_count
            ),
            "snapshot": snapshot_compile_evidence(self.torch),
        }
        self._timeline.append(record)
        return record

    def evidence(self) -> dict[str, Any]:
        """Return session configuration, timeline, and current counter delta."""
        current = snapshot_compile_evidence(self.torch)
        return {
            "config": _json_safe(self.config),
            "forward": _qualified_name(self.original),
            "compiled_type": _qualified_name(self.compiled),
            "call_count": int(self.call_count),
            "compiled_call_count": int(self.compiled_call_count),
            "eager_full_variance_call_count": int(
                self.eager_full_variance_call_count
            ),
            "installed": bool(self._installed),
            "timeline": _json_safe(self._timeline),
            "current": current,
            "delta_since_creation": evidence_delta(self._created, current),
        }


@contextlib.contextmanager
def installed_compiled_forward(core: Any, **compile_kwargs: Any):
    """Construct, install, yield, and restore one :class:`CompiledForward`."""
    session = CompiledForward(core, **compile_kwargs)
    with session.install():
        yield session


__all__ = [
    "CompiledForward",
    "compiler_capabilities",
    "evidence_delta",
    "installed_compiled_forward",
    "reset_compiler_state",
    "snapshot_compile_evidence",
    "source_metadata",
]
