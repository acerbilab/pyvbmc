"""Scoped live-search policies for the bounded E5 inference experiment.

The S0 arm observes the production search without changing its decisions.
The S2 arm replaces selection after the ordinary 1024-row coarse sieve with
the frozen MC1600 shortlist/refinement policy.  Both arms leave target calls,
cache consumption, repeat accounting, and subsequent GP/VP updates in the
production active-sampling controller.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import time
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from noisy_acq_quadrature import make_rule, prepare_rule
from noisy_acq_search import SearchConfig, _refine

from pyvbmc.acquisition_functions import AcqFcnVIQR


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _digest_json(value) -> str:
    payload = json.dumps(
        value,
        default=_json_default,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"cannot make {type(value).__name__} JSON-safe")


def _row_digest(row) -> str:
    data = np.ascontiguousarray(np.asarray(row, dtype="<f8").reshape(-1))
    return hashlib.sha256(data.tobytes()).hexdigest()


def _array_digest(array, dtype) -> str:
    data = np.ascontiguousarray(np.asarray(array, dtype=dtype))
    return hashlib.sha256(data.tobytes()).hexdigest()


def _compact_rule_record(rule) -> dict[str, Any]:
    metadata = dict(rule.get("metadata", {}))
    assignment = metadata.pop("component_assignment", [])
    return {
        "method": str(rule["method"]),
        "budget": int(rule["budget"]),
        "seed": int(rule["seed"]),
        "available": bool(rule["available"]),
        "metadata": _json_safe(metadata),
        "component_assignment_count": len(assignment),
        "component_assignment_sha256": _array_digest(assignment, "<i8"),
        "nodes_sha256": _array_digest(rule["nodes"], "<f8"),
        "weights_sha256": _array_digest(rule["weights"], "<f8"),
    }


def _derive_accurate_seed(seed: int, label: str, selection_index: int) -> int:
    payload = json.dumps(
        ["pyvbmc-e5-s2-accurate-rule-v1", seed, label, selection_index],
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


class InferencePolicy:
    """Install one bounded live-search policy for a single inference fit.

    Parameters
    ----------
    arm
        ``"S0"`` observes the unmodified production selection. ``"S2"``
        installs the frozen 1024-row coarse sieve, MC1600 accurate shortlist,
        and single-start refinement.
    seed
        Fit seed used only as provenance for private accurate-rule seeds.
        The policy never seeds or replaces the variational posterior's RNG.
    label
        Stable fit/cell label included in seed derivation and records.
    accurate_seed_factory
        Optional test hook called as ``factory(seed, label, index)``. It can
        reproduce an earlier frozen-state accurate-rule seed without changing
        production allocation.

    Notes
    -----
    ``search_seconds`` starts immediately before ``_get_search_points`` and
    stops immediately before the function logger target/cache call. It covers
    candidate generation, coarse setup and evaluation, accurate rescoring,
    and local refinement or the production local optimizer. It excludes the
    target call and later GP/VP updates.
    """

    def __init__(
        self,
        arm: str,
        seed: int,
        label: str,
        *,
        accurate_seed_factory: Callable[[int, str, int], int] | None = None,
        testing_seeds: Sequence[int] | None = None,
    ):
        if arm not in {"S0", "S2"}:
            raise ValueError("arm must be S0 or S2")
        if isinstance(seed, (bool, np.bool_)) or not isinstance(
            seed, (int, np.integer)
        ):
            raise TypeError("seed must be a nonnegative integer")
        if int(seed) < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not isinstance(label, str) or not label:
            raise ValueError("label must be a nonempty string")
        if accurate_seed_factory is not None and testing_seeds is not None:
            raise ValueError(
                "accurate_seed_factory and testing_seeds are alternatives"
            )
        if testing_seeds is not None:
            testing_seeds = tuple(int(value) for value in testing_seeds)
            if any(value < 0 for value in testing_seeds):
                raise ValueError("testing seeds must be nonnegative")

            def factory(_seed, _label, index):
                try:
                    return testing_seeds[index]
                except IndexError as error:
                    raise RuntimeError(
                        "no testing seed for selection"
                    ) from error

            accurate_seed_factory = factory
        self.arm = arm
        self.seed = int(seed)
        self.label = label
        self._accurate_seed_factory = accurate_seed_factory
        self.records: list[dict[str, Any]] = []
        self._active = None
        self._entered = False
        self._active_module = None
        self._option_originals: dict[int, tuple[Any, int]] = {}
        self._config = SearchConfig(arm="S2")
        self._config.validate()

    def __enter__(self):
        if self._entered:
            raise RuntimeError("an inference policy context cannot be reused")
        active = importlib.import_module("pyvbmc.vbmc.active_sample")
        if active._selection_policy_callback is not None:
            raise RuntimeError(
                "nested active inference policies are not allowed"
            )
        self._entered = True
        self._active_module = active
        active._selection_policy_callback = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        incomplete = self._active is not None
        try:
            if incomplete:
                self._finish_failed(exc_type, exc_value)
            for options, original in reversed(
                list(self._option_originals.values())
            ):
                options.__setitem__("ns_search", original, force=True)
        finally:
            if self._active_module is not None:
                self._active_module._selection_policy_callback = None
            self._active_module = None
        if incomplete and exc_type is None:
            raise RuntimeError("inference policy selection did not finish")
        return False

    def _accurate_seed(self, index: int) -> int:
        if self._accurate_seed_factory is None:
            value = _derive_accurate_seed(self.seed, self.label, index)
        else:
            value = self._accurate_seed_factory(self.seed, self.label, index)
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError("accurate seed factory must return an integer")
        value = int(value)
        if value < 0:
            raise ValueError("accurate seed must be nonnegative")
        return value

    def start(self, *, gp, vp, function_logger, optim_state, options):
        """Begin one selection immediately before candidate generation."""
        if self._active is not None:
            raise RuntimeError("nested active-sampling selections are invalid")
        if self.arm == "S2":
            key = id(options)
            if key not in self._option_originals:
                self._option_originals[key] = (
                    options,
                    int(options["ns_search"]),
                )
            options.__setitem__(
                "ns_search", self._config.sieve_size, force=True
            )
        rng_before = _digest_json(vp.rng.bit_generator.state)
        self._active = {
            "selection_index": len(self.records),
            "started": time.perf_counter(),
            "vp": vp,
            "vp_rng_before_sha256": rng_before,
            "detail": {
                "gp_training_rows": int(len(gp.X)),
                "logger_live_rows": int(
                    np.count_nonzero(function_logger.X_flag)
                ),
            },
        }

    def select(
        self,
        *,
        candidates,
        coarse_scores,
        cache_indices,
        n_train,
        gp,
        vp,
        function_logger,
        optim_state,
        options,
    ):
        """Observe S0 or return the frozen S2 selection override."""
        if self._active is None:
            raise RuntimeError("selection callback was not started")
        points = np.asarray(candidates, dtype=np.float64)
        scores = np.asarray(coarse_scores, dtype=np.float64).reshape(-1)
        cache_indices = np.asarray(cache_indices).reshape(-1)
        if points.ndim != 2 or points.shape != (scores.size, gp.D):
            raise RuntimeError("coarse candidate and score shapes differ")
        if cache_indices.shape != scores.shape:
            raise RuntimeError("coarse cache indices and scores differ")
        n_train = int(n_train)
        if not 0 <= n_train <= len(points):
            raise RuntimeError("invalid training-repeat row count")
        generated_rows = len(points) - n_train
        detail = self._active["detail"]
        detail.update(
            coarse_candidate_rows=int(len(points)),
            generated_candidate_rows=int(generated_rows),
            training_repeat_rows=n_train,
            coarse_winner_index=int(np.argmin(scores)),
            coarse_winner_score=float(np.min(scores)),
        )
        if self.arm == "S0":
            detail.update(
                accurate_seed=None,
                shortlist_rows=0,
                accurate_candidate_rows=0,
                local_iterations=0,
                fallback_reason=None,
                refinement_stop_reason=None,
            )
            return None

        if not np.any(np.isfinite(scores)):
            raise RuntimeError("no finite coarse candidate")

        acquisition_functions = options["search_acq_fcn"]
        if len(acquisition_functions) != 1 or (
            type(acquisition_functions[0]) is not AcqFcnVIQR
            or acquisition_functions[0].loss != "iqr"
        ):
            raise RuntimeError("S2 requires the standard VIQR acquisition")
        coarse_budget = options.eval(
            "active_importance_sampling_mcmc_samples",
            {"K": vp.K, "n_vars": vp.D, "D": vp.D},
        )
        if coarse_budget != 100:
            raise RuntimeError(
                "S2 requires the production 100-node coarse rule"
            )

        order = np.argsort(scores, kind="stable")[
            : self._config.shortlist_size
        ]
        shortlist = np.array(points[order], dtype=np.float64, copy=True)
        repeated = order < n_train
        live_rows = function_logger.X[function_logger.X_flag]
        if np.any(repeated):
            shortlist[repeated] = live_rows[order[repeated]]

        selection_index = self._active["selection_index"]
        accurate_seed = self._accurate_seed(selection_index)
        state = {
            "gp": gp,
            "vp": vp,
            "logger": function_logger,
            "optim_state": optim_state,
            "options": options,
        }
        rule = make_rule(
            vp,
            self._config.accurate_method,
            self._config.accurate_budget,
            accurate_seed,
        )
        if not rule["available"]:
            raise RuntimeError("S2 accurate rule is unavailable")
        evaluator = prepare_rule(state, rule)
        result = evaluator.score(shortlist, diagnostics=False)
        if not np.any(result["valid"]):
            raise RuntimeError("no valid accurately rescored candidate")
        accurate_scores = np.where(
            result["valid"], result["full_score"], np.inf
        )
        selected, refinement = _refine(
            state,
            evaluator,
            shortlist,
            accurate_scores,
            repeated,
            self._config,
        )
        shortlist_index = refinement["selected_shortlist_index"]
        if shortlist_index is None:
            cache_index = np.nan
            repeat = False
            selected_source = "refined"
        else:
            row = int(order[int(shortlist_index)])
            repeat = row < n_train
            if repeat:
                cache_index = np.nan
                selected_source = "repeat"
                if not np.array_equal(selected, live_rows[row]):
                    raise RuntimeError(
                        "S2 repeat did not preserve the exact row"
                    )
            else:
                cache_index = cache_indices[row]
                selected_source = (
                    "cache" if np.isfinite(cache_index) else "generated"
                )
        selected = np.asarray(selected, dtype=np.float64).reshape(1, gp.D)
        if not np.all(np.isfinite(selected)):
            raise RuntimeError("S2 selected nonfinite coordinates")
        if np.isfinite(cache_index):
            numeric_index = float(cache_index)
            if not numeric_index.is_integer():
                raise RuntimeError("S2 selected a nonintegral cache index")
            integer_index = int(numeric_index)
            cache_rows = len(optim_state["cache"]["x_orig"])
            if not 0 <= integer_index < cache_rows:
                raise RuntimeError("S2 selected an out-of-range cache index")
            cache_index = integer_index

        detail.update(
            accurate_seed=accurate_seed,
            shortlist_rows=int(len(shortlist)),
            accurate_candidate_rows=int(refinement["accurate_candidate_rows"]),
            local_iterations=int(refinement["local_iterations"]),
            fallback_reason=refinement["fallback_reason"],
            refinement_stop_reason=refinement["refinement_stop_reason"],
            selected_source=selected_source,
            selected_accurate_score=float(
                refinement["selected_accurate_score"]
            ),
            cache_index=(
                int(cache_index) if np.isfinite(cache_index) else None
            ),
            repeat=bool(repeat),
            shortlist_indices=[int(value) for value in order],
            shortlist_scores=[
                float(value) if np.isfinite(value) else None
                for value in accurate_scores
            ],
            accurate_rule=_compact_rule_record(rule),
            evaluator_metadata=_json_safe(getattr(evaluator, "metadata", {})),
            refinement_diagnostics=_json_safe(refinement),
        )
        return selected, cache_index, repeat

    def finish(self, *, selected, cache_index, repeat):
        """Finish one selection immediately before the target/cache call."""
        finished = time.perf_counter()
        if self._active is None:
            raise RuntimeError("selection callback was not started")
        active = self._active
        selected = np.asarray(selected, dtype=np.float64).reshape(-1)
        selected_all_finite = bool(np.all(np.isfinite(selected)))
        if self.arm == "S2" and not selected_all_finite:
            raise RuntimeError("selection produced nonfinite coordinates")
        cache_value = None
        if np.isfinite(cache_index):
            numeric_index = float(cache_index)
            if self.arm == "S2" and not numeric_index.is_integer():
                raise RuntimeError(
                    "selection produced a nonintegral cache index"
                )
            cache_value = int(numeric_index)
        self._active = None
        record = {
            "status": "complete",
            "arm": self.arm,
            "seed": self.seed,
            "label": self.label,
            "selection_index": int(active["selection_index"]),
            "search_seconds": float(finished - active["started"]),
            "vp_rng_before_sha256": active["vp_rng_before_sha256"],
            "vp_rng_after_sha256": _digest_json(
                active["vp"].rng.bit_generator.state
            ),
            **active["detail"],
            "selected_source": active["detail"].get(
                "selected_source",
                "repeat"
                if repeat
                else ("cache" if cache_value is not None else "production"),
            ),
            "selected": _json_safe(selected),
            "selected_all_finite": selected_all_finite,
            "selected_sha256": _row_digest(selected),
            "cache_index": cache_value,
            "repeat": bool(repeat),
        }
        self.records.append(record)

    def _finish_failed(self, exc_type, exc_value):
        active = self._active
        self._active = None
        record = {
            "status": "failed",
            "arm": self.arm,
            "seed": self.seed,
            "label": self.label,
            "selection_index": int(active["selection_index"]),
            "search_seconds": float(time.perf_counter() - active["started"]),
            "vp_rng_before_sha256": active["vp_rng_before_sha256"],
            "vp_rng_after_sha256": _digest_json(
                active["vp"].rng.bit_generator.state
            ),
            **active["detail"],
            "exception_type": None if exc_type is None else exc_type.__name__,
            "exception_message": None if exc_value is None else str(exc_value),
        }
        self.records.append(record)

    def summary(self) -> dict[str, Any]:
        """Return compact JSON-safe timing and selection diagnostics."""
        complete = [
            item for item in self.records if item["status"] == "complete"
        ]
        failures = [
            item for item in self.records if item["status"] != "complete"
        ]
        fallback_counts = Counter(
            item["fallback_reason"]
            for item in complete
            if item.get("fallback_reason") is not None
        )
        return {
            "arm": self.arm,
            "seed": self.seed,
            "label": self.label,
            "selection_count": len(self.records),
            "complete_selection_count": len(complete),
            "failed_selection_count": len(failures),
            "total_search_seconds": float(
                sum(item["search_seconds"] for item in self.records)
            ),
            "accurate_seeds": [
                item["accurate_seed"]
                for item in complete
                if item.get("accurate_seed") is not None
            ],
            "fallback_counts": dict(sorted(fallback_counts.items())),
            "repeat_count": sum(bool(item.get("repeat")) for item in complete),
            "cache_count": sum(
                item.get("cache_index") is not None for item in complete
            ),
        }


__all__ = ["InferencePolicy"]
