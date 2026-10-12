"""Fixed-node quadrature tools for noisy-acquisition experiments.

The functions in this module are developer instrumentation.  They construct
independent integration rules and evaluate the VIQR acquisition on those
fixed rules without changing the production acquisition implementation or
the state from which an experiment was captured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from gpyreg.covariance_functions import SquaredExponential
from scipy.linalg import solve_triangular
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from scipy.stats import norm, qmc, t

from pyvbmc.acquisition_functions.acq_fcn_viqr import AcqFcnVIQR, _log_viqr_sum

_METHOD_ALIASES = {
    "mc": "mc",
    "ordinary_mc": "mc",
    "stratified_mc": "stratified_mc",
    "component_mc": "stratified_mc",
    "rqmc": "stratified_rqmc",
    "stratified_rqmc": "stratified_rqmc",
    "component_rqmc": "stratified_rqmc",
}


def _as_seed(seed: Any) -> int:
    """Return a nonnegative integer seed suitable for ``SeedSequence``."""
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise TypeError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return seed


def _seed_record(seed_sequence: np.random.SeedSequence) -> dict[str, Any]:
    """Return JSON-compatible provenance for a seed sequence."""
    entropy = seed_sequence.entropy
    if isinstance(entropy, np.ndarray):
        entropy = entropy.tolist()
    elif isinstance(entropy, np.integer):
        entropy = int(entropy)
    return {
        "entropy": entropy,
        "spawn_key": [int(value) for value in seed_sequence.spawn_key],
        "pool_size": int(seed_sequence.pool_size),
        "n_children_spawned": int(seed_sequence.n_children_spawned),
    }


def _normal_from_unit(unit: np.ndarray) -> np.ndarray:
    """Map unit-cube nodes to finite standard-normal coordinates."""
    unit = np.asarray(unit, dtype=np.float64)
    lower = np.nextafter(np.float64(0.0), np.float64(1.0))
    upper = np.nextafter(np.float64(1.0), np.float64(0.0))
    return norm.ppf(np.clip(unit, lower, upper)).astype(np.float64, copy=False)


def _component_counts(weights: np.ndarray, budget: int) -> np.ndarray | None:
    """Allocate power-of-two counts by the largest ``weight / count``."""
    positive = np.flatnonzero(weights > 0)
    if budget < positive.size:
        return None

    counts = np.zeros(weights.size, dtype=np.int64)
    counts[positive] = 1
    used = int(positive.size)
    while True:
        candidates = positive[used + counts[positive] <= budget]
        if candidates.size == 0:
            break
        ratios = weights[candidates] / counts[candidates]
        # ``argmax`` supplies the fixed, lowest-component-index tie break.
        component = int(candidates[int(np.argmax(ratios))])
        increment = int(counts[component])
        counts[component] *= 2
        used += increment
    return counts


def _vp_arrays(
    vp: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read and validate the Gaussian-mixture arrays used by the rules."""
    weights = np.asarray(vp.w, dtype=np.float64).reshape(-1)
    means = np.asarray(vp.mu, dtype=np.float64)
    sigma = np.asarray(vp.sigma, dtype=np.float64).reshape(-1)
    lambd = np.asarray(vp.lambd, dtype=np.float64).reshape(-1)
    if means.ndim != 2 or means.shape[1] != weights.size:
        raise ValueError("vp.mu and vp.w have incompatible shapes")
    if sigma.size != weights.size or lambd.size != means.shape[0]:
        raise ValueError("vp scales have incompatible shapes")
    if not (
        np.all(np.isfinite(weights))
        and np.all(np.isfinite(means))
        and np.all(np.isfinite(sigma))
        and np.all(np.isfinite(lambd))
    ):
        raise ValueError("variational-posterior parameters must be finite")
    if np.any(weights < 0) or not np.sum(weights) > 0:
        raise ValueError("variational-posterior weights must be nonnegative")
    if np.any(sigma <= 0) or np.any(lambd <= 0):
        raise ValueError("variational-posterior scales must be positive")
    weights = weights / np.sum(weights)
    return weights, means, sigma, lambd


def make_rule(vp: Any, method: str, budget: int, seed: int) -> dict[str, Any]:
    """Construct an independent integration rule for a variational mixture.

    Parameters
    ----------
    vp
        Variational posterior.  Its random-number generator is never read or
        advanced.
    method
        ``"mc"``, ``"stratified_mc"``, or ``"stratified_rqmc"``.
    budget
        Maximum number of integration nodes.  Component-stratified rules can
        leave part of this budget unused because every component count is a
        power of two.
    seed
        Seed for a private :class:`numpy.random.SeedSequence`.

    Returns
    -------
    dict
        Rule record with float64 ``nodes`` and ``weights``, its requested and
        actual sizes, component assignments, and complete seed provenance.
        If a stratified rule cannot give every positive-weight component one
        node, ``available`` is false and the arrays are empty.
    """
    if method not in _METHOD_ALIASES:
        raise ValueError(f"unknown integration method: {method!r}")
    method = _METHOD_ALIASES[method]
    if isinstance(budget, (bool, np.bool_)) or not isinstance(
        budget, (int, np.integer)
    ):
        raise TypeError("budget must be a positive integer")
    budget = int(budget)
    if budget <= 0:
        raise ValueError("budget must be a positive integer")
    seed = _as_seed(seed)
    weights, means, sigma, lambd = _vp_arrays(vp)
    dimension, components = means.shape
    root_seed = np.random.SeedSequence(seed)

    if method == "mc":
        generator = np.random.default_rng(root_seed)
        assignment = generator.choice(components, size=budget, p=weights)
        standard = generator.standard_normal((budget, dimension))
        nodes = means[:, assignment].T + standard * (
            sigma[assignment, None] * lambd[None, :]
        )
        node_weights = np.full(budget, 1.0 / budget, dtype=np.float64)
        counts = np.bincount(assignment, minlength=components).astype(np.int64)
        child_records: list[dict[str, Any]] = []
        scramble = False
    else:
        counts = _component_counts(weights, budget)
        if counts is None:
            metadata = {
                "requested_budget": budget,
                "actual_nodes": 0,
                "unused_budget": budget,
                "component_counts": [0] * components,
                "component_assignment": [],
                "scramble": method == "stratified_rqmc",
                "seed_sequence": _seed_record(root_seed),
                "component_seeds": [],
                "reason": "budget_below_positive_component_count",
            }
            return {
                "method": method,
                "budget": budget,
                "seed": seed,
                "available": False,
                "nodes": np.empty((0, dimension), dtype=np.float64),
                "weights": np.empty(0, dtype=np.float64),
                "metadata": metadata,
            }

        positive = np.flatnonzero(counts > 0)
        child_sequences = root_seed.spawn(int(positive.size))
        node_blocks = []
        weight_blocks = []
        assignment_blocks = []
        child_records = []
        scramble = method == "stratified_rqmc"
        for component, child_seed in zip(positive, child_sequences):
            count = int(counts[component])
            if method == "stratified_mc":
                component_generator = np.random.default_rng(child_seed)
                standard = component_generator.standard_normal(
                    (count, dimension)
                )
                engine_seed = None
            else:
                engine_seed = int(
                    child_seed.generate_state(1, dtype=np.uint32)[0]
                )
                sobol = qmc.Sobol(d=dimension, scramble=True, seed=engine_seed)
                standard = _normal_from_unit(
                    sobol.random_base2(int(np.log2(count)))
                )
            node_blocks.append(
                means[:, component][None, :]
                + standard * (sigma[component] * lambd)[None, :]
            )
            weight_blocks.append(
                np.full(count, weights[component] / count, dtype=np.float64)
            )
            assignment_blocks.append(np.full(count, component, dtype=np.int64))
            record = _seed_record(child_seed)
            record["component"] = int(component)
            record["engine_seed"] = engine_seed
            child_records.append(record)
        nodes = np.concatenate(node_blocks, axis=0)
        node_weights = np.concatenate(weight_blocks)
        assignment = np.concatenate(assignment_blocks)

    nodes = np.asarray(nodes, dtype=np.float64)
    node_weights = np.asarray(node_weights, dtype=np.float64)
    if not np.all(np.isfinite(nodes)):
        raise FloatingPointError("integration rule produced nonfinite nodes")
    if not (
        np.all(np.isfinite(node_weights))
        and np.all(node_weights > 0)
        and np.sum(node_weights) > 0
    ):
        raise FloatingPointError("integration rule produced invalid weights")

    actual = int(nodes.shape[0])
    metadata = {
        "requested_budget": budget,
        "actual_nodes": actual,
        "unused_budget": budget - actual,
        "component_counts": counts.astype(int).tolist(),
        "component_assignment": assignment.astype(int).tolist(),
        "scramble": scramble,
        "seed_sequence": _seed_record(root_seed),
        "component_seeds": child_records,
        "weight_sum": float(np.sum(node_weights)),
    }
    return {
        "method": method,
        "budget": budget,
        "seed": seed,
        "available": True,
        "nodes": nodes,
        "weights": node_weights,
        "metadata": metadata,
    }


def _log_two_sinh(value: np.ndarray) -> np.ndarray:
    """Return ``log(2 sinh(value))`` without overflowing at large values."""
    value = np.asarray(value, dtype=np.float64)
    result = np.full(value.shape, np.nan, dtype=np.float64)
    zero = value == 0
    result[zero] = -np.inf
    finite_positive = np.isfinite(value) & (value > 0)
    direct_limit = np.log(np.finfo(np.float64).max) - np.log(2.0)
    direct = finite_positive & (value <= direct_limit)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        result[direct] = np.log(2.0) + np.log(np.sinh(value[direct]))
        large = finite_positive & ~direct
        result[large] = value[large] + np.log1p(-np.exp(-2.0 * value[large]))
    return result


def _log_weighted_residual(
    scaled_sd: np.ndarray,
    weights: np.ndarray,
    log_weights: np.ndarray,
) -> np.ndarray:
    """Return rowwise logs of weighted ``2 sinh`` sums."""
    scaled_sd = np.asarray(scaled_sd, dtype=np.float64)
    was_vector = scaled_sd.ndim == 1
    if was_vector:
        scaled_sd = scaled_sd[None, :]
    if scaled_sd.ndim != 2 or scaled_sd.shape[1] != weights.size:
        raise ValueError("scaled_sd and weights have incompatible shapes")
    # Retain the exact production arithmetic for equal-weight parity.
    if np.all(weights == weights[0]):
        result = _log_viqr_sum(scaled_sd) + np.log(weights[0])
    else:
        direct_limit = np.log(np.finfo(np.float64).max) - np.log(2.0)
        row_min = np.min(scaled_sd, axis=1)
        row_max = np.max(scaled_sd, axis=1)
        safe = (
            np.isfinite(row_min)
            & np.isfinite(row_max)
            & (row_min >= 0.0)
            & (row_max <= direct_limit)
        )
        result = np.empty(scaled_sd.shape[0], dtype=np.float64)
        if np.any(safe):
            with np.errstate(divide="ignore"):
                result[safe] = np.log(
                    2.0
                    * np.sum(
                        np.sinh(scaled_sd[safe]) * weights[None, :],
                        axis=1,
                    )
                )
        if np.any(~safe):
            terms = _log_two_sinh(scaled_sd[~safe]) + log_weights[None, :]
            result[~safe] = logsumexp(terms, axis=1)
    return result[0] if was_vector else result


def _positive_exp(log_value: np.ndarray) -> np.ndarray:
    """Exponentiate a log quantity while keeping the log value authoritative."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return np.exp(log_value)


def _stable_log_reduction(
    acquisition: AcqFcnVIQR,
    tau2: np.ndarray,
    source_variance: np.ndarray,
    log_weights: np.ndarray,
) -> np.ndarray:
    """Return ``log(R0 - R)`` consistently with clipped residual variance."""
    clipped_tau2 = np.minimum(tau2, source_variance[None, :])
    return acquisition._log_iqr_reduction(
        clipped_tau2, source_variance, log_weights
    ) + np.log(2.0)


@dataclass
class _QuadratureCache:
    f_s2: np.ndarray
    K_Xa_X: np.ndarray | None
    C_tmp: np.ndarray | None
    log_reference_by_hyper: np.ndarray
    log_reference: float
    factor_representations: tuple[str, ...]
    streamed: bool
    node_block_size: int


class FixedVIQREvaluator:
    """VIQR evaluator whose integration nodes and weights remain fixed."""

    def __init__(self, state: Mapping[str, Any], rule: Mapping[str, Any]):
        self._gp = state["gp"]
        self._vp = state["vp"]
        self._optim_state = state["optim_state"]
        self._pt = state.get("pt", self._vp.parameter_transformer)

        self.nodes = np.array(rule["nodes"], dtype=np.float64, copy=True)
        raw_weights = np.array(rule["weights"], dtype=np.float64, copy=True)
        if self.nodes.ndim != 2 or raw_weights.shape != (self.nodes.shape[0],):
            raise ValueError("rule nodes and weights have incompatible shapes")
        if self.nodes.shape[0] == 0:
            raise ValueError("rule must contain at least one node")
        if not np.all(np.isfinite(self.nodes)):
            raise ValueError("rule nodes must be finite")
        if not (
            np.all(np.isfinite(raw_weights))
            and np.all(raw_weights > 0)
            and np.sum(raw_weights) > 0
        ):
            raise ValueError(
                "rule weights must be finite and strictly positive"
            )
        if np.all(raw_weights == raw_weights[0]):
            self.weights = np.full(
                raw_weights.size, 1.0 / raw_weights.size, dtype=np.float64
            )
        else:
            self.weights = raw_weights / np.sum(raw_weights)
        self.log_weights = np.log(self.weights)
        metadata = dict(rule.get("metadata", {}))
        quantile = float(metadata.get("quantile", 0.75))
        self._u = float(norm.ppf(quantile))
        if not np.isfinite(self._u) or self._u <= 0:
            raise ValueError("the VIQR quantile must lie between 0.5 and 1")
        self._acquisition = AcqFcnVIQR(quantile=quantile)
        requested_chunk_size = int(metadata.get("candidate_chunk_size", 256))
        if requested_chunk_size <= 0:
            raise ValueError("candidate_chunk_size must be positive")
        self._cache_max_bytes = int(
            metadata.get("cache_max_bytes", 256 * 1024**2)
        )
        self._work_max_bytes = int(
            metadata.get("work_max_bytes", 64 * 1024**2)
        )
        if self._cache_max_bytes <= 0 or self._work_max_bytes <= 0:
            raise ValueError("cache and work memory caps must be positive")
        training_count = int(self._gp.X.shape[0])
        max_nodes_by_work = max(
            1, self._work_max_bytes // max(1, 4 * training_count * 8)
        )
        self._node_block_size = min(
            self.nodes.shape[0],
            int(metadata.get("node_block_size", 8192)),
            max_nodes_by_work,
        )
        if self._node_block_size <= 0:
            raise ValueError("node_block_size must be positive")
        max_candidate_node_pairs = max(1, self._work_max_bytes // (8 * 5))
        self._chunk_size = min(
            requested_chunk_size,
            max(1, max_candidate_node_pairs // self._node_block_size),
        )
        self.rule = {
            "method": str(rule.get("method", "external")),
            "budget": int(rule.get("budget", self.nodes.shape[0])),
            "seed": rule.get("seed"),
            "available": True,
            "nodes": self.nodes.copy(),
            "weights": self.weights.copy(),
            "metadata": metadata,
        }
        self.cache = self._build_cache()
        self.metadata = {
            "normalized_weights": True,
            "input_weight_sum": float(np.sum(raw_weights)),
            "node_count": int(self.nodes.shape[0]),
            "dimension": int(self.nodes.shape[1]),
            "hyperparameter_samples": int(len(self._gp.posteriors)),
            "factor_representations": list(self.cache.factor_representations),
            "kernel_reuse_eligible": bool(
                self._acquisition._can_reuse_prediction_kernel(
                    self.nodes, self._gp
                )
            ),
            "candidate_chunk_size": self._chunk_size,
            "requested_candidate_chunk_size": requested_chunk_size,
            "node_block_size": self.cache.node_block_size,
            "streamed_node_cache": self.cache.streamed,
            "cache_max_bytes": self._cache_max_bytes,
            "work_max_bytes": self._work_max_bytes,
            "log_reference_residual": self.cache.log_reference,
            "reference_residual": float(
                _positive_exp(np.array(self.cache.log_reference))
            ),
        }

    def _build_cache(self) -> _QuadratureCache:
        gp = self._gp
        if not isinstance(gp.covariance, SquaredExponential):
            raise TypeError(
                "fixed VIQR evaluation requires a squared-exponential GP"
            )
        samples = len(gp.posteriors)
        node_count = self.nodes.shape[0]
        training_count = gp.X.shape[0]
        dense_bytes = 2 * samples * node_count * training_count * 8
        streamed = dense_bytes > self._cache_max_bytes
        if streamed:
            f_s2 = np.empty((node_count, samples), dtype=np.float64)
            for start in range(0, node_count, self._node_block_size):
                stop = min(start + self._node_block_size, node_count)
                _, block_f_s2 = gp.predict(
                    self.nodes[start:stop], separate_samples=True
                )
                f_s2[start:stop] = block_f_s2
        else:
            _, f_s2 = gp.predict(self.nodes, separate_samples=True)
            f_s2 = np.asarray(f_s2, dtype=np.float64)
        if streamed:
            K_Xa_X = None
            C_tmp = None
        else:
            K_Xa_X = np.empty(
                (samples, node_count, training_count), dtype=np.float64
            )
            C_tmp = np.empty(
                (samples, training_count, node_count), dtype=np.float64
            )
        representations = []
        for sample, posterior in enumerate(gp.posteriors):
            if posterior.L_chol:
                representations.append("cholesky")
            else:
                representations.append("inverse")
            if not streamed:
                covariance_count = gp.covariance.hyperparameter_count(gp.D)
                kernel = gp.covariance.compute(
                    posterior.hyp[:covariance_count], self.nodes, gp.X
                )
                K_Xa_X[sample] = kernel
                C_tmp[sample] = self._factor_product(posterior, kernel)

        log_reference_by_hyper = np.empty(samples, dtype=np.float64)
        for sample in range(samples):
            with np.errstate(invalid="ignore"):
                scaled_sd = self._u * np.sqrt(f_s2[:, sample])
            log_reference_by_hyper[sample] = _log_weighted_residual(
                scaled_sd, self.weights, self.log_weights
            )
        log_reference = float(
            logsumexp(log_reference_by_hyper) - np.log(samples)
        )
        return _QuadratureCache(
            f_s2=f_s2,
            K_Xa_X=K_Xa_X,
            C_tmp=C_tmp,
            log_reference_by_hyper=log_reference_by_hyper,
            log_reference=log_reference,
            factor_representations=tuple(representations),
            streamed=streamed,
            node_block_size=self._node_block_size,
        )

    @staticmethod
    def _factor_product(posterior: Any, kernel: np.ndarray) -> np.ndarray:
        """Apply either supported GP posterior factor representation."""
        if posterior.L_chol:
            sn2_eff = 1.0 / posterior.sW[0] ** 2
            return (
                solve_triangular(
                    posterior.L,
                    solve_triangular(
                        posterior.L,
                        kernel.T,
                        trans=True,
                        check_finite=False,
                    ),
                    check_finite=False,
                )
                / sn2_eff
            )
        return posterior.L @ kernel.T

    def _node_factor_block(
        self, sample: int, start: int, stop: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return node-to-training kernels and their posterior product."""
        if self.cache.K_Xa_X is not None:
            return (
                self.cache.K_Xa_X[sample, start:stop],
                self.cache.C_tmp[sample, :, start:stop],
            )
        posterior = self._gp.posteriors[sample]
        covariance_count = self._gp.covariance.hyperparameter_count(self._gp.D)
        kernel = self._gp.covariance.compute(
            posterior.hyp[:covariance_count],
            self.nodes[start:stop],
            self._gp.X,
        )
        return kernel, self._factor_product(posterior, kernel)

    def _raw_score_chunk(
        self, X: np.ndarray, diagnostics: bool
    ) -> tuple[
        np.ndarray,
        np.ndarray | None,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        gp = self._gp
        optim_state = self._optim_state
        (
            f_mu,
            f_s2,
            prediction_context,
        ) = self._acquisition._predict_with_context(X, gp)
        f_mu = np.asarray(f_mu, dtype=np.float64)
        f_s2 = np.asarray(f_s2, dtype=np.float64)
        prediction_samples = f_mu.shape[1]
        f_bar_column = np.sum(f_mu, axis=1, keepdims=True) / prediction_samples
        variance_bar = np.sum(f_s2, axis=1, keepdims=True) / prediction_samples
        if prediction_samples > 1:
            variance_mean = np.sum(
                (f_mu - f_bar_column) ** 2, axis=1, keepdims=True
            ) / (prediction_samples - 1)
        else:
            variance_mean = 0
        f_bar = np.ravel(f_bar_column)
        var_tot = np.ravel(variance_mean + variance_bar)
        sn2 = self._acquisition._estimate_observation_noise(X, gp, optim_state)
        y_s2 = f_s2 + sn2[:, None]

        samples = len(gp.posteriors)
        candidates = X.shape[0]
        log_residual_by_hyper = np.full(
            (candidates, samples), -np.inf, dtype=np.float64
        )
        if diagnostics:
            log_reduction_by_hyper = np.full(
                (candidates, samples), -np.inf, dtype=np.float64
            )
        else:
            log_reduction_by_hyper = None
        for sample, posterior in enumerate(gp.posteriors):
            hyp = posterior.hyp
            ell = np.exp(hyp[: gp.D])
            sf2 = np.exp(2 * hyp[gp.D])
            if prediction_context is None:
                distance = cdist(X / ell, gp.X / ell, "sqeuclidean")
                K_Xs_X = sf2 * np.exp(-0.5 * distance)
            else:
                K_Xs_X = np.asarray(
                    prediction_context[sample], dtype=np.float64
                ).T
            valid_observation_variance = np.isfinite(y_s2[:, sample]) & (
                y_s2[:, sample] > 0
            )
            for start in range(0, self.nodes.shape[0], self._node_block_size):
                stop = min(start + self._node_block_size, self.nodes.shape[0])
                _, factor_product = self._node_factor_block(
                    sample, start, stop
                )
                distance = cdist(
                    X / ell,
                    self.nodes[start:stop] / ell,
                    "sqeuclidean",
                )
                K_Xs_Xa = sf2 * np.exp(-0.5 * distance)
                if posterior.L_chol:
                    covariance = K_Xs_Xa - K_Xs_X @ factor_product
                else:
                    covariance = K_Xs_Xa + K_Xs_X @ factor_product
                with np.errstate(
                    divide="ignore", invalid="ignore", over="ignore"
                ):
                    tau2 = covariance**2 / y_s2[:, sample, None]
                    posterior_variance = np.maximum(
                        self.cache.f_s2[start:stop, sample][None, :] - tau2,
                        0.0,
                    )
                    scaled_sd = self._u * np.sqrt(posterior_variance)
                valid_candidate_numerics = valid_observation_variance & np.all(
                    np.isfinite(tau2) & (tau2 >= 0), axis=1
                )
                block_log_residual = _log_weighted_residual(
                    scaled_sd,
                    self.weights[start:stop],
                    self.log_weights[start:stop],
                )
                block_log_residual[~valid_candidate_numerics] = np.nan
                log_residual_by_hyper[:, sample] = np.logaddexp(
                    log_residual_by_hyper[:, sample], block_log_residual
                )
                if diagnostics:
                    source_variance = self.cache.f_s2[start:stop, sample]
                    valid_source = np.all(
                        np.isfinite(source_variance) & (source_variance >= 0)
                    )
                    safe_source_variance = np.where(
                        np.isfinite(source_variance) & (source_variance >= 0),
                        source_variance,
                        0.0,
                    )
                    safe_tau2 = np.where(
                        np.isfinite(tau2) & (tau2 >= 0), tau2, 0.0
                    )
                    block_log_reduction = _stable_log_reduction(
                        self._acquisition,
                        safe_tau2,
                        safe_source_variance,
                        self.log_weights[start:stop],
                    )
                    block_log_reduction[
                        ~valid_candidate_numerics | ~valid_source
                    ] = np.nan
                    log_reduction_by_hyper[:, sample] = np.logaddexp(
                        log_reduction_by_hyper[:, sample], block_log_reduction
                    )

        log_residual = logsumexp(log_residual_by_hyper, axis=1) - np.log(
            samples
        )
        if diagnostics:
            log_reduction = logsumexp(log_reduction_by_hyper, axis=1) - np.log(
                samples
            )
        else:
            log_reduction = None
        return log_residual, log_reduction, f_bar, var_tot, f_s2

    def score(
        self, X: np.ndarray, full: bool = True, diagnostics: bool = True
    ) -> dict[str, Any]:
        """Evaluate normalized VIQR values at candidate points.

        The result uses the minimization convention ``F = log(R) + P``.
        ``log_residual`` is the normalized weighted log residual before the
        production variance penalty and hard-bound mask.  ``log_reduction``
        is the stable log of the reduction from the cached reference
        residual; its exponentiated counterpart can underflow to zero.  Set
        ``diagnostics=False`` during timed VIQR searches to omit that separate
        reduction calculation; both reduction fields are then ``None``.
        """
        input_X = np.asarray(X, dtype=np.float64)
        if input_X.ndim == 1:
            input_X = input_X.reshape(1, -1)
        if input_X.ndim != 2 or input_X.shape[1] != self._gp.D:
            raise ValueError("X must have shape (N, D)")
        X_used = np.array(input_X, dtype=np.float64, copy=True)
        finite_input = np.all(np.isfinite(X_used), axis=1)
        finite_indices = np.flatnonzero(finite_input)
        integer_vars = self._optim_state.get("integer_vars")
        if finite_indices.size:
            snapped = self._acquisition._real2int(
                X_used[finite_indices].copy(), self._pt, integer_vars
            )
            X_used[finite_indices] = snapped

        size = X_used.shape[0]
        log_residual = np.full(size, np.nan, dtype=np.float64)
        if diagnostics:
            log_reduction = np.full(size, np.nan, dtype=np.float64)
        else:
            log_reduction = None
        f_bar = np.full(size, np.nan, dtype=np.float64)
        var_tot = np.full(size, np.nan, dtype=np.float64)
        for start in range(0, finite_indices.size, self._chunk_size):
            index = finite_indices[start : start + self._chunk_size]
            raw = self._raw_score_chunk(X_used[index], diagnostics)
            log_residual[index] = raw[0]
            if diagnostics:
                log_reduction[index] = raw[1]
            f_bar[index], var_tot[index] = raw[2:4]

        numeric_valid = (
            finite_input & ~np.isnan(log_residual) & (log_residual < np.inf)
        )
        penalty = np.zeros(size, dtype=np.float64)
        if full and self._optim_state.get(
            "variance_regularized_acq_fcn",
            self._optim_state.get("variance_regularized_acqfcn", False),
        ):
            tolerance = self._optim_state["tol_gp_var"]
            regularized = numeric_valid & (var_tot < tolerance)
            positive = regularized & (var_tot > 0)
            penalty[positive] = tolerance / var_tot[positive] - 1.0
            penalty[regularized & ~positive] = np.inf

        with np.errstate(invalid="ignore", over="ignore"):
            full_score = log_residual + penalty
        full_score = np.maximum(full_score, -np.finfo(np.float64).max)
        inside_bounds = np.ones(size, dtype=bool)
        if full and finite_indices.size:
            original = self._pt.inverse(X_used[finite_indices])
            outside = np.any(
                (original < self._optim_state["lb_eps_orig"])
                | (original > self._optim_state["ub_eps_orig"]),
                axis=1,
            )
            inside_bounds[finite_indices[outside]] = False
            full_score[finite_indices[outside]] = np.inf
        full_score[~finite_input] = np.inf
        valid = numeric_valid & inside_bounds & np.isfinite(full_score)

        reasons = np.full(size, None, dtype=object)
        reasons[~finite_input] = "nonfinite_candidate"
        reasons[finite_input & ~numeric_valid] = "invalid_weighted_residual"
        reasons[numeric_valid & ~inside_bounds] = "outside_hard_bounds"
        reasons[
            numeric_valid & inside_bounds & ~np.isfinite(full_score)
        ] = "nonfinite_penalty"
        residual = _positive_exp(log_residual)
        reduction = _positive_exp(log_reduction) if diagnostics else None
        score_metadata = dict(self.metadata)
        score_metadata["diagnostics"] = bool(diagnostics)
        return {
            "full_score": full_score,
            "log_residual": log_residual,
            "residual": residual,
            "log_reduction": log_reduction,
            "reduction": reduction,
            "penalty": penalty,
            "valid": valid,
            "failure_reason": reasons,
            "X_used": X_used,
            "f_bar": f_bar,
            "variance": var_tot,
            "reference_log_residual": self.cache.log_reference,
            "reference_residual": float(
                _positive_exp(np.array(self.cache.log_reference))
            ),
            "metadata": score_metadata,
        }


def prepare_rule(
    state: Mapping[str, Any], rule: Mapping[str, Any]
) -> FixedVIQREvaluator:
    """Prepare a fixed-node VIQR evaluator from a captured state and rule."""
    if not isinstance(state, Mapping):
        raise TypeError("state must be a mapping")
    missing = {"gp", "vp", "optim_state"} - set(state)
    if missing:
        raise KeyError(f"state is missing required entries: {sorted(missing)}")
    if not isinstance(rule, Mapping):
        raise TypeError("rule must be a mapping")
    if not rule.get("available", True):
        reason = rule.get("metadata", {}).get("reason", "unspecified")
        raise ValueError(f"integration rule is unavailable: {reason}")
    return FixedVIQREvaluator(state, rule)


def practical_band(
    log_reference: float,
    log_baseline: float,
    *,
    baseline_gain_resolved: bool = True,
) -> dict[str, Any]:
    """Compute the practical score band used by the paired judge."""
    floor = float(np.log1p(1e-6))
    scaled = 0.01 * abs(float(log_reference) - float(log_baseline))
    eps_f = max(floor, scaled) if baseline_gain_resolved else floor
    return {
        "eps_F": float(eps_f),
        "floor": floor,
        "scaled_baseline_gain": float(scaled),
        "baseline_gain_resolved": bool(baseline_gain_resolved),
    }


def _paired_student_summary(
    values: np.ndarray,
) -> dict[str, float | list[float]]:
    """Summarize a finite paired quantity with a two-sided 95% interval."""
    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / np.sqrt(values.size))
    critical = float(t.ppf(0.975, values.size - 1))
    half_width = critical * standard_error
    return {
        "mean": mean,
        "standard_error": standard_error,
        "t_critical": critical,
        "half_width": float(half_width),
        "ci": [float(mean - half_width), float(mean + half_width)],
    }


def baseline_gain_resolution(
    log_reductions: np.ndarray, log_references: np.ndarray
) -> dict[str, Any]:
    """Assess whether eight paired reductions resolve a positive raw gain.

    Reductions are scaled by their paired current residuals before the
    Student interval is formed.  The baseline gain is resolved only when the
    interval's lower endpoint exceeds the fixed relative numerical floor.
    """
    log_reductions = np.asarray(log_reductions, dtype=np.float64).reshape(-1)
    log_references = np.asarray(log_references, dtype=np.float64).reshape(-1)
    if log_reductions.shape != log_references.shape:
        raise ValueError(
            "reduction and reference logs must have the same shape"
        )
    floor = 1e-12
    record: dict[str, Any] = {
        "log_reductions": log_reductions.tolist(),
        "log_references": log_references.tolist(),
        "n": int(log_reductions.size),
        "relative_gain_floor": floor,
    }
    if log_reductions.size != 8:
        record.update(
            {
                "relative_gains": None,
                "mean_relative_gain": None,
                "standard_error": None,
                "t_critical": None,
                "half_width": None,
                "ci": None,
                "resolved": False,
                "classification": "unresolved",
                "reason": "requires_eight_replicates",
            }
        )
        return record
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        relative_gains = np.exp(log_reductions - log_references)
    record["relative_gains"] = relative_gains.tolist()
    if not np.all(np.isfinite(relative_gains)):
        record.update(
            {
                "mean_relative_gain": None,
                "standard_error": None,
                "t_critical": None,
                "half_width": None,
                "ci": None,
                "resolved": False,
                "classification": "unresolved",
                "reason": "nonfinite_relative_gain",
            }
        )
        return record
    summary = _paired_student_summary(relative_gains)
    resolved = summary["ci"][0] > floor
    record.update(
        {
            "mean_relative_gain": summary["mean"],
            "standard_error": summary["standard_error"],
            "t_critical": summary["t_critical"],
            "half_width": summary["half_width"],
            "ci": summary["ci"],
            "resolved": bool(resolved),
            "classification": (
                "resolved_positive" if resolved else "unresolved"
            ),
            "reason": None if resolved else "gain_at_numerical_floor",
        }
    )
    return record


def raw_loss_assessment(
    log_arm_reductions: np.ndarray,
    log_baseline_reductions: np.ndarray,
    log_references: np.ndarray,
    previous: Mapping[str, Any] | str | None = None,
) -> dict[str, Any]:
    """Assess a paired loss of more than 10% of baseline raw reduction.

    A common log scale keeps ``Delta_arm - 0.9 Delta_baseline`` finite.  A
    verdict requires a resolved baseline gain, a precise directional
    interval, and the same interval classification at consecutive budgets.
    """
    arm = np.asarray(log_arm_reductions, dtype=np.float64).reshape(-1)
    baseline = np.asarray(log_baseline_reductions, dtype=np.float64).reshape(
        -1
    )
    references = np.asarray(log_references, dtype=np.float64).reshape(-1)
    if arm.shape != baseline.shape or arm.shape != references.shape:
        raise ValueError(
            "arm, baseline, and reference logs must have one shape"
        )
    baseline_gain = baseline_gain_resolution(baseline, references)
    record: dict[str, Any] = {
        "log_arm_reductions": arm.tolist(),
        "log_baseline_reductions": baseline.tolist(),
        "log_references": references.tolist(),
        "n": int(arm.size),
        "loss_fraction": 0.1,
        "retained_fraction": 0.9,
        "relative_gain_floor": baseline_gain["relative_gain_floor"],
        "baseline_gain": baseline_gain,
        "common_log_scale": None,
        "scaled_arm_reductions": None,
        "scaled_baseline_reductions": None,
        "scaled_paired_differences": None,
        "arm_to_baseline_ratio": None,
        "mean_difference": None,
        "standard_error": None,
        "t_critical": None,
        "half_width": None,
        "ci": None,
        "interval_classification": "unresolved",
        "budget_classification": "unresolved",
        "previous_budget_classification": None,
        "classification": "unresolved",
        "reason": None,
    }
    if not baseline_gain["resolved"]:
        record["reason"] = "baseline_gain_unresolved"
        return record
    if arm.size != 8:
        record["reason"] = "requires_eight_replicates"
        return record
    finite_logs = np.concatenate(
        (arm[np.isfinite(arm)], baseline[np.isfinite(baseline)])
    )
    if (
        finite_logs.size == 0
        or np.any(np.isnan(arm))
        or np.any(np.isnan(baseline))
    ):
        record["reason"] = "nonfinite_reduction_logs"
        return record
    common_scale = float(np.max(finite_logs))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        scaled_arm = np.exp(arm - common_scale)
        scaled_baseline = np.exp(baseline - common_scale)
    paired = scaled_arm - 0.9 * scaled_baseline
    if not (
        np.all(np.isfinite(scaled_arm))
        and np.all(np.isfinite(scaled_baseline))
        and np.all(np.isfinite(paired))
    ):
        record["reason"] = "nonfinite_scaled_reductions"
        return record
    baseline_mean = float(np.mean(scaled_baseline))
    ratio = (
        float(np.mean(scaled_arm) / baseline_mean)
        if baseline_mean > 0
        else None
    )
    summary = _paired_student_summary(paired)
    lower, upper = summary["ci"]
    if upper < 0:
        interval_classification = "material_loss"
    elif lower >= 0:
        interval_classification = "no_material_loss"
    else:
        interval_classification = "unresolved"

    budget_classification = interval_classification
    budget_reason = None
    if interval_classification == "unresolved":
        budget_reason = "interval_crosses_loss_boundary"
    elif summary["half_width"] > 0.25 * abs(summary["mean"]):
        budget_classification = "unresolved"
        budget_reason = "directional_precision"

    if isinstance(previous, Mapping):
        previous_classification = str(
            previous.get(
                "budget_classification",
                previous.get("classification", "unresolved"),
            )
        )
    elif previous is None:
        previous_classification = None
    else:
        previous_classification = str(previous)

    classification = budget_classification
    reason = budget_reason
    if budget_classification == "unresolved":
        pass
    elif previous_classification is None:
        classification = "unresolved"
        reason = "needs_consecutive_budget"
    elif previous_classification != budget_classification:
        classification = "unresolved"
        reason = "budget_stability"

    record.update(
        {
            "common_log_scale": common_scale,
            "scaled_arm_reductions": scaled_arm.tolist(),
            "scaled_baseline_reductions": scaled_baseline.tolist(),
            "scaled_paired_differences": paired.tolist(),
            "arm_to_baseline_ratio": ratio,
            "mean_difference": summary["mean"],
            "standard_error": summary["standard_error"],
            "t_critical": summary["t_critical"],
            "half_width": summary["half_width"],
            "ci": summary["ci"],
            "interval_classification": interval_classification,
            "budget_classification": budget_classification,
            "previous_budget_classification": previous_classification,
            "classification": classification,
            "reason": reason,
        }
    )
    return record


def _point_classification(value: float, eps_f: float) -> str:
    if value < -eps_f:
        return "beneficial"
    if value > eps_f:
        return "harmful"
    return "practical_tie"


def judge_paired(
    arm_scores: np.ndarray,
    reference_scores: np.ndarray,
    eps_f: float,
    *,
    previous: Mapping[str, Any] | str | None = None,
    replicate_seeds: list[Any] | None = None,
    pooled_difference: float | None = None,
) -> dict[str, Any]:
    """Classify paired score differences using a two-sided Student interval.

    A resolved result requires the same interval classification at two
    consecutive budgets.  Directional conclusions additionally require the
    interval half-width to be at most one quarter of the mean magnitude.
    When a pooled residual classification is supplied, disagreement with the
    mean-log classification makes the result unresolved.
    """
    arm = np.asarray(arm_scores, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference_scores, dtype=np.float64).reshape(-1)
    eps_f = float(eps_f)
    if eps_f < 0 or not np.isfinite(eps_f):
        raise ValueError("eps_f must be finite and nonnegative")
    if arm.shape != reference.shape:
        raise ValueError("paired score arrays must have the same shape")
    difference = arm - reference
    record: dict[str, Any] = {
        "arm_scores": arm.tolist(),
        "reference_scores": reference.tolist(),
        "paired_differences": difference.tolist(),
        "replicate_seeds": list(replicate_seeds or []),
        "n": int(difference.size),
        "eps_F": eps_f,
        "pooled_difference": (
            None if pooled_difference is None else float(pooled_difference)
        ),
    }
    if difference.size < 2 or not np.all(np.isfinite(difference)):
        record.update(
            {
                "mean_difference": np.nan,
                "standard_error": np.nan,
                "t_critical": np.nan,
                "ci": [np.nan, np.nan],
                "half_width": np.nan,
                "interval_classification": "unresolved",
                "budget_classification": "unresolved",
                "pooled_classification": None,
                "classification": "unresolved",
                "reason": "insufficient_or_nonfinite_pairs",
            }
        )
        return record

    mean = float(np.mean(difference))
    standard_error = float(
        np.std(difference, ddof=1) / np.sqrt(difference.size)
    )
    critical = float(t.ppf(0.975, difference.size - 1))
    half_width = critical * standard_error
    lower, upper = mean - half_width, mean + half_width
    if upper < -eps_f:
        interval_classification = "beneficial"
    elif lower > eps_f:
        interval_classification = "harmful"
    elif lower >= -eps_f and upper <= eps_f:
        interval_classification = "practical_tie"
    else:
        interval_classification = "unresolved"

    pooled_classification = None
    if pooled_difference is not None:
        if np.isfinite(pooled_difference):
            pooled_classification = _point_classification(
                float(pooled_difference), eps_f
            )
        else:
            pooled_classification = "unresolved"

    budget_classification = interval_classification
    budget_reason = None
    if interval_classification == "unresolved":
        budget_reason = "interval_crosses_decision_boundary"
    elif interval_classification in {"beneficial", "harmful"} and (
        half_width > 0.25 * abs(mean)
    ):
        budget_classification = "unresolved"
        budget_reason = "directional_precision"
    elif (
        pooled_classification is not None
        and pooled_classification != interval_classification
    ):
        budget_classification = "unresolved"
        budget_reason = "pooled_mean_log_disagreement"

    if isinstance(previous, Mapping):
        previous_classification = str(
            previous.get(
                "budget_classification",
                previous.get("classification", "unresolved"),
            )
        )
    elif previous is None:
        previous_classification = None
    else:
        previous_classification = str(previous)

    classification = budget_classification
    reason = budget_reason
    if budget_classification == "unresolved":
        pass
    elif previous_classification is None:
        classification = "unresolved"
        reason = "needs_consecutive_budget"
    elif previous_classification != budget_classification:
        classification = "unresolved"
        reason = "budget_stability"

    record.update(
        {
            "mean_difference": mean,
            "standard_error": standard_error,
            "t_critical": critical,
            "ci": [float(lower), float(upper)],
            "half_width": float(half_width),
            "interval_classification": interval_classification,
            "budget_classification": budget_classification,
            "pooled_classification": pooled_classification,
            "previous_budget_classification": previous_classification,
            "classification": classification,
            "reason": reason,
        }
    )
    return record


__all__ = [
    "FixedVIQREvaluator",
    "baseline_gain_resolution",
    "judge_paired",
    "make_rule",
    "practical_band",
    "prepare_rule",
    "raw_loss_assessment",
]
