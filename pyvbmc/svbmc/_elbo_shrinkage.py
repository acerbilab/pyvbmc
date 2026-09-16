# S-VBMC: Stacking Variational Bayesian Monte Carlo.
#
# Copyright (c) 2025, S-VBMC Developers and their Assignees.
# All rights reserved. Distributed under the BSD 3-Clause License; the
# full text is in LICENSE.txt next to this file.
"""Private empirical-Bayes estimates for S-VBMC ELBO reporting."""

from __future__ import annotations

import warnings

import numpy as np


def _unavailable(cause, run_index=None):
    """Warn that the estimate is undefined and return its nullable fields."""
    where = "" if run_index is None else f" for retained run {run_index}"
    warnings.warn(
        "S-VBMC two-level shrinkage is unavailable: " + cause + where + ".",
        RuntimeWarning,
        stacklevel=3,
    )
    return None, None


def _moments(values, covariance):
    """Population mean, excess variance and noise in the observed spread."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    covariance = np.asarray(covariance, dtype=np.float64)
    n = values.size
    with np.errstate(over="ignore", invalid="ignore"):
        mean = float(np.mean(values))
        spread = float(np.var(values, ddof=1)) if n > 1 else 0.0
        noise = (
            float((np.trace(covariance) - np.sum(covariance) / n) / (n - 1))
            if n > 1
            else 0.0
        )
        excess = max(spread - noise, 0.0)
    return mean, excess, spread, noise


def _run_covariance(I_sk, J_sjk):
    """Full covariance of one run's component expected log joints."""
    I_sk = np.asarray(I_sk, dtype=np.float64)
    J_sjk = np.asarray(J_sjk, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        covariance = np.mean(J_sjk, axis=0, dtype=np.float64)
        if I_sk.shape[0] > 1:
            covariance = covariance + np.asarray(
                np.cov(I_sk, rowvar=False, ddof=1), dtype=np.float64
            ).reshape(I_sk.shape[1], I_sk.shape[1])
        covariance = 0.5 * (covariance + covariance.T)
    # Preserve this shape explicitly, including K = 1 where np.cov is scalar.
    return np.asarray(covariance, dtype=np.float64).reshape(
        I_sk.shape[1], I_sk.shape[1]
    )


def _two_level_shrinkage(
    I_sk_runs,
    J_sjk_runs,
    corrected_means,
    original_weights,
    selected_weights,
    entropy,
):
    """Return the full-covariance two-level ELBO and noise-share diagnostic.

    Parameters are plain arrays grouped by retained run. ``corrected_means``
    contains the component expected log joints in original coordinates;
    ``original_weights`` contains each run's normalized component weights;
    and ``selected_weights`` contains the concatenated weights selected by
    stacking. The calculation is deterministic and does not mutate its inputs.

    Undefined numerical cases return ``(None, None)`` after a
    :class:`RuntimeWarning`. A missing noise-share diagnostic alone does not
    invalidate a finite ELBO estimate.
    """
    selected = np.asarray(selected_weights, dtype=np.float64).reshape(-1)
    entropy = float(entropy)
    runs = []
    levels = []
    level_variances = []
    offset = 0

    for m, (I_sk, J_sjk, I, own) in enumerate(
        zip(I_sk_runs, J_sjk_runs, corrected_means, original_weights)
    ):
        I = np.asarray(I, dtype=np.float64).reshape(-1)
        own = np.asarray(own, dtype=np.float64).reshape(-1)
        covariance = _run_covariance(I_sk, J_sjk)
        if not np.all(np.isfinite(covariance)):
            return _unavailable("nonfinite component covariance", m)

        mean, excess, spread, noise = _moments(I, covariance)
        if not np.all(np.isfinite([mean, excess, spread, noise])):
            return _unavailable("nonfinite within-run moments", m)

        if not np.any(covariance):
            within = I.copy()
        elif excess > 0.0:
            with np.errstate(over="ignore", invalid="ignore"):
                system = excess * np.eye(I.size, dtype=np.float64) + covariance
            if not np.all(np.isfinite(system)):
                return _unavailable("nonfinite within-run linear system", m)
            try:
                with np.errstate(over="ignore", invalid="ignore"):
                    within = mean + excess * np.linalg.solve(system, I - mean)
            except np.linalg.LinAlgError:
                return _unavailable("singular within-run linear solve", m)
        else:
            within = np.full(I.size, mean, dtype=np.float64)
        if not np.all(np.isfinite(within)):
            return _unavailable("nonfinite within-run shrinkage result", m)

        with np.errstate(over="ignore", invalid="ignore"):
            level = float(own @ I)
            level_variance = float(own @ covariance @ own)
        if not np.isfinite(level) or not np.isfinite(level_variance):
            return _unavailable("nonfinite run-level moments", m)
        if level_variance < 0.0:
            return _unavailable("negative run-level variance", m)

        stop = offset + I.size
        mass = float(np.sum(selected[offset:stop], dtype=np.float64))
        runs.append((within, spread, noise, mass))
        levels.append(level)
        level_variances.append(level_variance)
        offset = stop

    levels = np.asarray(levels, dtype=np.float64)
    level_variances = np.asarray(level_variances, dtype=np.float64)
    run_mean, run_excess, run_spread, run_noise = _moments(
        levels, np.diag(level_variances)
    )
    if not np.all(np.isfinite([run_mean, run_excess, run_spread, run_noise])):
        return _unavailable("nonfinite between-run moments")

    # A level known with zero variance stays at its observed value, including
    # the all-zero and zero-excess-variance cases.
    shifted = np.empty_like(levels)
    for m, (level, variance) in enumerate(zip(levels, level_variances)):
        if variance == 0.0:
            shifted[m] = level
        elif run_excess <= 0.0:
            shifted[m] = run_mean
        else:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                denominator = run_excess + variance
                factor = run_excess / denominator
                shifted[m] = run_mean + factor * (level - run_mean)
            if not np.isfinite(denominator):
                return _unavailable("nonfinite run-level denominator", m)
        if not np.isfinite(shifted[m]):
            return _unavailable("nonfinite run-level shrinkage result", m)

    with np.errstate(over="ignore", invalid="ignore"):
        corrected = np.concatenate(
            [
                within + (shifted[m] - levels[m])
                for m, (within, _, _, _) in enumerate(runs)
            ]
        )
    if not np.all(np.isfinite(corrected)):
        return _unavailable("nonfinite corrected component estimates")
    with np.errstate(over="ignore", invalid="ignore"):
        estimate = float(selected @ corrected + entropy)
    if not np.isfinite(estimate):
        return _unavailable("nonfinite final shrinkage result")

    share_weight = 0.0
    weighted_share = 0.0
    for m, (_, spread, noise, mass) in enumerate(runs):
        if spread > 0.0 and mass > 0.0:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                share = noise / spread
                contribution = mass * share
            if not np.isfinite(share) or not np.isfinite(contribution):
                return _unavailable("nonfinite shrinkage noise share", m)
            with np.errstate(over="ignore", invalid="ignore"):
                share_weight += mass
                weighted_share += contribution
            if not np.isfinite(share_weight) or not np.isfinite(
                weighted_share
            ):
                return _unavailable("nonfinite accumulated noise share", m)
    if share_weight > 0.0:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            noise_share = float(weighted_share / share_weight)
        if not np.isfinite(noise_share):
            return _unavailable("nonfinite shrinkage noise share")
    else:
        noise_share = None
    return estimate, noise_share
