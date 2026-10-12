"""Bounded NumPy prototype of S-VBMC's weight optimization.

This development-only module adapts the numerical procedure in
``acerbilab/svbmc`` commit ``13a78f6c4a3ffe9c4557fee4f4b8f67c98b29e01``.
It deliberately preserves the estimator, initialization, Adam defaults,
rounded-loss stopping, and best-iterate policy of that implementation while
using explicit float64 NumPy/SciPy arithmetic.  It is not a public PyVBMC API.

The adapted source is distributed under the following BSD 3-Clause terms:

Copyright (c) 2025, S-VBMC Developers and their Assignees
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy import stats

_WEIGHT_EPS = 1e-40
_ADAM_BETA1 = 0.9
_ADAM_BETA2 = 0.999
_ADAM_EPS = 1e-8
_DEFAULT_CHUNK_BYTES = 32 * 1024**2
_NORMAL_LOG_CONSTANT = 0.5 * np.log(2.0 * np.pi)


def _component_data(stacker):
    """Return canonical float64 component arrays in posterior order."""
    counts = np.asarray(stacker.K, dtype=np.int64)
    means = []
    scales = []
    for vp in stacker.vp_list:
        mu = np.asarray(vp.mu, dtype=np.float64)
        sigma = np.asarray(vp.lambd * vp.sigma, dtype=np.float64)
        means.append(mu.T)
        scales.append(sigma.T)
    return (
        counts,
        np.concatenate(means, axis=0),
        np.concatenate(scales, axis=0),
    )


def _draw_standard_normal(stacker, n_samples, rng):
    """Make one upstream-compatible SciPy draw and canonicalize its shape."""
    kwargs = {}
    if rng is not None:
        kwargs["random_state"] = rng
    draw = stats.multivariate_normal.rvs(
        mean=np.zeros(stacker.D, dtype=np.float64),
        cov=np.eye(stacker.D, dtype=np.float64),
        size=n_samples,
        **kwargs,
    )
    # SciPy squeezes D=1 and/or size=1.  Upstream expects (n_samples, D).
    return np.asarray(draw, dtype=np.float64).reshape(n_samples, stacker.D)


def _sampling_rng(stacker):
    """Match upstream's deterministic testing branch and global stream."""
    if getattr(stacker, "testing", False):
        seed = getattr(stacker, "_svbmc_random_seed", 0)
        return np.random.RandomState(seed)
    return None


def prepare_entropy(stacker, n_samples, *, chunk_rows=None):
    """Draw stratified samples and construct their component log densities.

    Draw order, SciPy sampling, coordinate transforms, and density formula
    match the pinned upstream implementation.  The only shape correction is
    the canonical ``reshape(n_samples, D)`` required when either dimension is
    one.  Work shared by components belonging to one posterior is batched:
    its inverse transform is applied once to all its component draw blocks,
    and its forward transform/Jacobian is evaluated once per row chunk.

    Parameters
    ----------
    stacker : object
        An initialized upstream-compatible ``SVBMC`` instance.
    n_samples : int
        Number of samples drawn from every Gaussian component.
    chunk_rows : int or None, optional
        Maximum sample rows used in a density block.  The default targets at
        most 32 MiB for the largest per-posterior ``rows * K_m * D`` work
        array.  Chunking does not alter draw or row order.

    Returns
    -------
    logq_matrix : ndarray, shape (sum(K) * n_samples, sum(K))
        Original-space component log densities, without mixture log weights.
    J_corrections : ndarray, shape (sum(K),)
        Mean inverse-transform log Jacobian for each source stratum.
    """
    n_samples = int(n_samples)
    counts, means, scales = _component_data(stacker)
    k_total = int(counts.sum())
    sample_count = k_total * n_samples
    rng = _sampling_rng(stacker)

    transformed_draws = np.empty((sample_count, stacker.D), dtype=np.float64)
    for component in range(k_total):
        rows = slice(component * n_samples, (component + 1) * n_samples)
        standard = _draw_standard_normal(stacker, n_samples, rng)
        transformed_draws[rows] = (
            standard * scales[component] + means[component]
        )

    x_original = np.empty_like(transformed_draws)
    corrections = np.empty(k_total, dtype=np.float64)
    component_start = 0
    for m, vp in enumerate(stacker.vp_list):
        component_stop = component_start + int(counts[m])
        rows = slice(component_start * n_samples, component_stop * n_samples)
        transformed = transformed_draws[rows]
        jacobian = np.asarray(
            vp.parameter_transformer.log_abs_det_jacobian(transformed),
            dtype=np.float64,
        ).reshape(int(counts[m]), n_samples)
        corrections[component_start:component_stop] = np.mean(jacobian, axis=1)
        x_original[rows] = vp.parameter_transformer.inverse(transformed)
        component_start = component_stop

    if chunk_rows is None:
        largest_k = max(int(np.max(counts)), 1)
        bytes_per_row = 8 * largest_k * max(int(stacker.D), 1)
        chunk_rows = max(1, _DEFAULT_CHUNK_BYTES // bytes_per_row)
    chunk_rows = int(chunk_rows)
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")

    logq_matrix = np.empty((sample_count, k_total), dtype=np.float64)
    component_start = 0
    for m, vp in enumerate(stacker.vp_list):
        component_stop = component_start + int(counts[m])
        mu = means[component_start:component_stop]
        sigma = scales[component_start:component_stop]
        log_sigma = np.log(sigma)
        workspace = np.empty(
            (
                min(chunk_rows, sample_count),
                int(counts[m]),
                stacker.D,
            ),
            dtype=np.float64,
        )
        for row_start in range(0, sample_count, chunk_rows):
            row_stop = min(row_start + chunk_rows, sample_count)
            rows_in_chunk = row_stop - row_start
            transformed = np.asarray(
                vp.parameter_transformer(x_original[row_start:row_stop]),
                dtype=np.float64,
            ).reshape(rows_in_chunk, stacker.D)
            jacobian = np.asarray(
                vp.parameter_transformer.log_abs_det_jacobian(transformed),
                dtype=np.float64,
            ).reshape(rows_in_chunk)
            terms = workspace[:rows_in_chunk]
            np.subtract(
                transformed[:, np.newaxis, :],
                mu[np.newaxis, :, :],
                out=terms,
            )
            terms /= sigma[np.newaxis, :, :]
            np.square(terms, out=terms)
            terms *= -0.5
            terms -= _NORMAL_LOG_CONSTANT
            terms -= log_sigma[np.newaxis, :, :]
            log_density = np.sum(terms, axis=2, dtype=np.float64)
            logq_matrix[row_start:row_stop, component_start:component_stop] = (
                log_density - jacobian[:, np.newaxis]
            )
        component_start = component_stop

    return logq_matrix, corrections


def prepare_entropy_reference(stacker, n_samples):
    """Literal preparation reference for parity checks, with the D=1 fix."""
    n_samples = int(n_samples)
    counts, means, scales = _component_data(stacker)
    k_total = int(counts.sum())
    sample_count = k_total * n_samples
    rng = _sampling_rng(stacker)
    x_original = np.zeros((sample_count, stacker.D), dtype=np.float64)
    corrections = np.zeros(k_total, dtype=np.float64)

    component = 0
    for vp, count in zip(stacker.vp_list, counts):
        for _ in range(int(count)):
            standard = _draw_standard_normal(stacker, n_samples, rng)
            transformed = standard * scales[component] + means[component]
            corrections[component] = np.mean(
                vp.parameter_transformer.log_abs_det_jacobian(transformed)
            )
            rows = slice(component * n_samples, (component + 1) * n_samples)
            x_original[rows] = vp.parameter_transformer.inverse(transformed)
            component += 1

    logq_matrix = np.zeros((sample_count, k_total), dtype=np.float64)
    component = 0
    for vp, count in zip(stacker.vp_list, counts):
        for _ in range(int(count)):
            transformed = vp.parameter_transformer(x_original)
            jacobian = vp.parameter_transformer.log_abs_det_jacobian(
                transformed
            )
            transformed_logq = np.sum(
                stats.norm.logpdf(
                    transformed, means[component], scales[component]
                ),
                axis=1,
            )
            logq_matrix[:, component] = transformed_logq - jacobian
            component += 1
    return logq_matrix, corrections


def objective_weights(w, logq_matrix, corrected_integrals, n_samples):
    """Evaluate the sampled ELBO and its derivative with respect to weights.

    This intentionally retains upstream's unusual direct-weight semantics:
    entropy uses ``w / sum(w)`` internally, whereas the expected log joint is
    linear in the raw, possibly unnormalized ``w``.  Its derivative therefore
    includes the entropy normalization Jacobian.  The entropy derivative has
    both the outer stratum-weight term and the sample-density term.
    """
    weights = np.asarray(w, dtype=np.float64).reshape(-1)
    logq = np.asarray(logq_matrix, dtype=np.float64)
    integrals = np.asarray(corrected_integrals, dtype=np.float64).reshape(-1)
    n_samples = int(n_samples)
    k_total = weights.size
    if logq.shape != (k_total * n_samples, k_total):
        raise ValueError(
            "logq_matrix must have shape (len(w) * n_samples, len(w))"
        )
    if integrals.size != k_total:
        raise ValueError(
            "corrected_integrals must contain one value per weight"
        )

    weight_sum = np.sum(weights, dtype=np.float64)
    entropy_weights = weights / weight_sum
    weights_with_epsilon = entropy_weights + _WEIGHT_EPS
    # Keep one S-by-K workspace.  After stable log-sum-exp it is reused first
    # for responsibilities and then for q_j(x) / q_mix(x), which avoids a
    # second full exponential and another full-size temporary.
    density_ratio = logq.copy()
    density_ratio += np.log(weights_with_epsilon)[np.newaxis, :]
    row_max = np.max(density_ratio, axis=1)
    density_ratio -= row_max[:, np.newaxis]
    np.exp(density_ratio, out=density_ratio)
    row_sum = np.sum(density_ratio, axis=1, dtype=np.float64)
    log_mixture = row_max + np.log(row_sum)
    density_ratio /= row_sum[:, np.newaxis]
    density_ratio /= weights_with_epsilon[np.newaxis, :]
    stratum_logq = log_mixture.reshape(k_total, n_samples).mean(axis=1)
    entropy = -np.dot(entropy_weights, stratum_logq)

    # d log(sum_j (p_j + eps) q_j) / d p_l = q_l / mixture.
    sample_outer_weights = np.repeat(entropy_weights / n_samples, n_samples)
    gradient_p = -stratum_logq - sample_outer_weights @ density_ratio
    gradient_entropy = (
        gradient_p - np.dot(entropy_weights, gradient_p)
    ) / weight_sum

    expected_log_joint = np.dot(weights, integrals)
    elbo = expected_log_joint + entropy
    gradient = integrals + gradient_entropy
    return (
        np.float64(elbo),
        np.float64(entropy),
        np.asarray(gradient, dtype=np.float64),
    )


def _softmax(logits):
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    shifted = logits - np.max(logits)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities, dtype=np.float64)
    return probabilities


def objective_logits(
    logits,
    base_weights,
    counts,
    version,
    logq_matrix,
    corrected_integrals,
    n_samples,
):
    """Evaluate an S-VBMC objective and analytic gradient in logit space."""
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    base_weights = np.asarray(base_weights, dtype=np.float64).reshape(-1)
    counts = np.asarray(counts, dtype=np.int64).reshape(-1)

    if version == "all-weights":
        if logits.size != base_weights.size:
            raise ValueError("all-weights requires one logit per component")
        component_logits = logits
    elif version == "posterior-only":
        if logits.size != counts.size or counts.sum() != base_weights.size:
            raise ValueError(
                "posterior-only requires one logit per posterior and "
                "counts summing to the number of components"
            )
        component_logits = np.repeat(logits, counts) + np.log(base_weights)
    else:
        raise AttributeError(
            "S-VBMC version not recognized. Check the spelling!"
        )

    weights = _softmax(component_logits)
    elbo, entropy, gradient_w = objective_weights(
        weights, logq_matrix, corrected_integrals, n_samples
    )
    gradient_component_logits = weights * (
        gradient_w - np.dot(weights, gradient_w)
    )
    if version == "posterior-only":
        starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
        gradient_logits = np.add.reduceat(gradient_component_logits, starts)
    else:
        gradient_logits = gradient_component_logits
    return elbo, entropy, gradient_logits, weights


def initial_logits(stacker, version):
    """Return the exact upstream initialization, flattened to float64."""
    base_weights = np.asarray(stacker.w, dtype=np.float64).reshape(-1)
    individual_elbos = np.asarray(
        stacker.individual_elbos, dtype=np.float64
    ).reshape(-1)
    if version == "all-weights":
        logits = np.log(base_weights) + np.repeat(
            individual_elbos, np.asarray(stacker.K, dtype=np.int64)
        )
    elif version == "posterior-only":
        logits = individual_elbos.copy()
    else:
        raise AttributeError(
            "S-VBMC version not recognized. Check the spelling!"
        )
    return np.asarray(logits - np.max(logits), dtype=np.float64)


def _adam_step(
    parameters,
    gradient,
    first_moment,
    second_moment,
    step,
    lr,
):
    """Apply one minimizing PyTorch-default Adam update in float64."""
    first_moment = _ADAM_BETA1 * first_moment + (1.0 - _ADAM_BETA1) * gradient
    second_moment = _ADAM_BETA2 * second_moment + (
        1.0 - _ADAM_BETA2
    ) * np.square(gradient)
    first_unbiased = first_moment / (1.0 - _ADAM_BETA1**step)
    second_unbiased = second_moment / (1.0 - _ADAM_BETA2**step)
    parameters = parameters - lr * first_unbiased / (
        np.sqrt(second_unbiased) + _ADAM_EPS
    )
    return parameters, first_moment, second_moment


def _cache_first_corrections(stacker, corrected_integrals):
    if len(stacker.I_corrected) != 0:
        return
    corrected = np.asarray(corrected_integrals, dtype=np.float64).reshape(
        1, -1
    )
    stacker.I_corrected = corrected.copy()
    stacker.E_corrected = np.zeros(stacker.M, dtype=np.float64)
    start = 0
    for m, vp in enumerate(stacker.vp_list):
        stop = start + int(stacker.K[m])
        stacker.E_corrected[m] = np.sum(
            corrected[0, start:stop]
            * np.asarray(vp.w, dtype=np.float64).reshape(-1),
            dtype=np.float64,
        )
        start = stop


def optimize_numpy(
    stacker,
    n_samples=20,
    lr=0.1,
    max_steps=500,
    version="all-weights",
    trace=False,
):
    """Run the bounded NumPy S-VBMC prototype with fresh iteration draws.

    The stopping iteration performs neither a gradient update nor another
    draw.  If ``stacker.testing`` is true, upstream behavior deliberately
    restarts its private fixed-seed stream on every preparation; otherwise
    each iteration advances NumPy's global random stream.
    """
    logits = initial_logits(stacker, version)
    base_weights = np.asarray(stacker.w, dtype=np.float64).reshape(-1)
    counts = np.asarray(stacker.K, dtype=np.int64)
    first_moment = np.zeros_like(logits)
    second_moment = np.zeros_like(logits)
    best_logits = logits.copy()
    best_elbo = None
    best_entropy = None
    best_iteration = None
    loss_old = np.float64(1e8)
    convergence_counter = 0
    stopped = False

    preparation_time = 0.0
    objective_time = 0.0
    optimizer_time = 0.0
    trace_data = {
        "iteration": [],
        "elbo": [],
        "entropy": [],
        "rounded_loss": [],
        "convergence_counter": [],
        "is_best": [],
        "weights": [],
    }

    steps = 0
    for iteration in range(int(max_steps)):
        started = perf_counter()
        logq_matrix, corrections = prepare_entropy(stacker, n_samples)
        preparation_time += perf_counter() - started
        corrected_integrals = (
            np.asarray(stacker.I, dtype=np.float64).reshape(-1) - corrections
        )
        _cache_first_corrections(stacker, corrected_integrals)

        started = perf_counter()
        elbo, entropy, gradient_elbo, weights = objective_logits(
            logits,
            base_weights,
            counts,
            version,
            logq_matrix,
            corrected_integrals,
            n_samples,
        )
        objective_time += perf_counter() - started
        steps = iteration + 1
        loss = -elbo
        rounded_loss = np.round(loss * 1e5) / 1e5

        is_best = bool(rounded_loss < loss_old)
        if is_best:
            convergence_counter = 0
            best_logits = (
                np.repeat(logits, counts) + np.log(base_weights)
                if version == "posterior-only"
                else logits.copy()
            )
            best_elbo = np.float64(elbo)
            best_entropy = np.float64(entropy)
            best_iteration = iteration
            loss_old = rounded_loss
        else:
            convergence_counter += 1

        if trace:
            trace_data["iteration"].append(iteration)
            trace_data["elbo"].append(float(elbo))
            trace_data["entropy"].append(float(entropy))
            trace_data["rounded_loss"].append(float(rounded_loss))
            trace_data["convergence_counter"].append(convergence_counter)
            trace_data["is_best"].append(is_best)
            trace_data["weights"].append(weights.tolist())

        if convergence_counter >= 5:
            stopped = True
            break

        started = perf_counter()
        # Adam minimizes loss = -ELBO.
        logits, first_moment, second_moment = _adam_step(
            logits,
            -gradient_elbo,
            first_moment,
            second_moment,
            iteration + 1,
            np.float64(lr),
        )
        optimizer_time += perf_counter() - started

    final_weights = _softmax(best_logits).astype(np.float64, copy=False)
    final_weights /= np.sum(final_weights, dtype=np.float64)
    estimated = float(best_elbo)
    final_entropy = float(best_entropy)
    expected_log_joint = estimated - final_entropy
    median_i = np.median(np.asarray(stacker.I_corrected, dtype=np.float64))
    median_e = np.median(np.asarray(stacker.E_corrected, dtype=np.float64))

    result = {
        "weights": final_weights,
        "elbo": estimated,
        "entropy": final_entropy,
        "elbo_debiased_I_median": float(
            min(expected_log_joint, median_i) + final_entropy
        ),
        "elbo_debiased_E_median": float(
            min(expected_log_joint, median_e) + final_entropy
        ),
        "steps": steps,
        "stopped": stopped,
        "best_iteration": best_iteration,
        "version": version,
        "timings": {
            "preparation": preparation_time,
            "objective": objective_time,
            "optimizer": optimizer_time,
        },
    }
    if trace:
        result["trace"] = trace_data
    return result
