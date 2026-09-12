# S-VBMC: Stacking Variational Bayesian Monte Carlo.
#
# Copyright (c) 2025, S-VBMC Developers and their Assignees.
# All rights reserved. Distributed under the BSD 3-Clause License; the
# full text is in LICENSE.txt next to this file.
#
# This module is the implementation of the standalone ``svbmc`` package
# (acerbilab/svbmc, version 0.1.1, commit 13a78f6) moved into PyVBMC.
"""Stacking Variational Bayesian Monte Carlo (S-VBMC)."""

from __future__ import annotations

__all__ = ["SVBMC"]

import copy
import logging
import sys
import warnings

import numpy as np
import scipy as sp

from pyvbmc.rng import get_rng

from ._jacobian import expected_log_jacobian
from ._runtime_tips import consider_runtime_tip

_REQUIRED_STATS = ("stable", "elbo", "I_sk", "J_sjk")
_VERSIONS = ("all-weights", "posterior-only", "ns")


def _validate_optimization(n_samples, max_steps, version):
    """Validate optimization arguments before draws or user guidance."""
    if version not in _VERSIONS:
        raise ValueError(
            f"Unknown S-VBMC version {version!r}; choose one of "
            f"{', '.join(repr(v) for v in _VERSIONS)}."
        )
    if int(max_steps) < 1:
        raise ValueError("`max_steps` should be at least 1.")
    if int(n_samples) < 1:
        raise ValueError("`n_samples` should be at least 1.")


def _import_torch():
    """Return the torch module, or raise a message naming the extra."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "S-VBMC requires torch; install pyvbmc[torch]."
        ) from exc
    return torch


def _validate_posteriors(vp_list):
    """Check that ``vp_list`` holds fitted posteriors of one dimension.

    Returns the common ``D``. The checks cover what stacking reads: the
    component parameters, the transformer, and the ``stable``, ``elbo``,
    ``I_sk`` and ``J_sjk`` statistics a completed VBMC run stores.
    """
    if len(vp_list) == 0:
        raise ValueError("`vp_list` is empty; pass at least one posterior.")
    D = None
    for i, vp in enumerate(vp_list):
        for attr in (
            "mu",
            "w",
            "sigma",
            "lambd",
            "stats",
            "parameter_transformer",
        ):
            if not hasattr(vp, attr):
                raise TypeError(
                    f"`vp_list[{i}]` is not a fitted VariationalPosterior "
                    f"(missing `{attr}`)."
                )
        mu = np.asarray(vp.mu)
        if mu.ndim != 2 or mu.shape[1] < 1:
            raise ValueError(
                f"`vp_list[{i}].mu` should have shape (D, K), got {mu.shape}."
            )
        d, k = mu.shape
        if D is None:
            D = d
        elif d != D:
            raise ValueError(
                f"`vp_list[{i}]` has D = {d}, but `vp_list[0]` has D = {D}; "
                "stack only posteriors of the same problem."
            )
        if np.asarray(vp.w).size != k or np.asarray(vp.sigma).size != k:
            raise ValueError(
                f"`vp_list[{i}]` has {k} components in `mu` but `w` and "
                "`sigma` do not match."
            )
        if np.asarray(vp.lambd).size != d:
            raise ValueError(
                f"`vp_list[{i}].lambd` should hold D = {d} scales, got "
                f"{np.asarray(vp.lambd).size}."
            )
        # A posterior VBMC has not finished (or an unfitted one) has no
        # statistics at all.
        if vp.stats is None or not hasattr(vp.stats, "__contains__"):
            raise ValueError(
                f"`vp_list[{i}]` has no statistics; stack only posteriors "
                "returned by a completed VBMC run."
            )
        for key in _REQUIRED_STATS:
            if key not in vp.stats:
                raise ValueError(
                    f"`vp_list[{i}].stats` lacks {key!r}; stack only "
                    "posteriors returned by a completed VBMC run."
                )
        I_sk = np.asarray(vp.stats["I_sk"], dtype=float)
        J_sjk = np.asarray(vp.stats["J_sjk"], dtype=float)
        if I_sk.ndim != 2 or I_sk.shape[1] != k:
            raise ValueError(
                f"`vp_list[{i}].stats['I_sk']` should have shape (Ns, K) "
                f"with K = {k}, got {I_sk.shape}."
            )
        if J_sjk.ndim != 3 or J_sjk.shape[1:] != (k, k):
            raise ValueError(
                f"`vp_list[{i}].stats['J_sjk']` should have shape (Ns, K, K) "
                f"with K = {k}, got {J_sjk.shape}."
            )
        if I_sk.shape[0] < 1 or I_sk.shape[0] != J_sjk.shape[0]:
            raise ValueError(
                f"`vp_list[{i}]` needs matching, nonempty GP sample axes "
                "in `I_sk` and `J_sjk`."
            )
        # The filters discard a run VBMC did not mark as converged, so only
        # a run that can be retained needs finite statistics.
        if bool(vp.stats["stable"]) and not (
            np.all(np.isfinite(I_sk))
            and np.all(np.isfinite(J_sjk))
            and np.isfinite(float(vp.stats["elbo"]))
        ):
            raise ValueError(
                f"`vp_list[{i}]` is marked stable but has nonfinite `elbo`, "
                "`I_sk` or `J_sjk` statistics."
            )
    return D


def _balanced_counts(n, omega, rng):
    """Split ``n`` draws across runs in proportion to ``omega`` exactly.

    Every run gets the integer part of its quota ``n * omega``; the
    remaining draws go to distinct runs chosen with probability
    proportional to the fractional parts of their quotas, so every count is
    within one draw of its quota. (The balanced mode of
    :meth:`VariationalPosterior.sample` allocates its remainder across
    components with replacement instead.)
    """
    quota = n * omega
    counts = np.floor(quota).astype(int)
    remainder = int(n - counts.sum())
    if remainder > 0:
        # The fractional parts sum to the remainder and are each below one,
        # so at least ``remainder`` of them are nonzero.
        frac = quota - counts
        extra = rng.choice(
            omega.size, size=remainder, replace=False, p=frac / frac.sum()
        )
        counts[extra] += 1
    return counts


class SVBMC:
    """
    Stacking Variational Bayesian Monte Carlo (S-VBMC).

    Combines the variational posteriors of several completed VBMC runs on
    the same problem into one stacked mixture by optimizing the *stacked
    ELBO* with respect to the mixture weights only. Component locations
    and scales, and the expected log-joint statistics VBMC stored with each
    posterior, stay fixed, so no further evaluations of the model are
    needed.

    Construct the object with the fitted posteriors, then call
    :meth:`optimize`. Construction discards runs that VBMC did not mark as
    converged (``stats["stable"]``) and runs whose expected log-joint
    carries excessive uncertainty (``s_max``), and raises when fewer than
    ``M_min`` runs remain.

    The stacked posterior is a composite over the input posteriors: each
    VBMC run has its own parameter transform, so components of different
    runs live in different transformed spaces and are only comparable in
    the original space. Use :meth:`sample` for estimates and plots; do not
    combine the component means and covariances of different runs
    directly.

    Requires the optional ``torch`` extra (``pip install pyvbmc[torch]``);
    construction raises ``ImportError`` without it.

    Parameters
    ----------
    vp_list : list of VariationalPosterior
        Posteriors returned by completed VBMC runs on the same model, data
        and original parameter space. The object structure cannot verify
        that the runs describe the same target.
    s_max : float, optional
        Tolerance on the standard deviation of the expected log-joint of
        any single component, ``sqrt(max(stats["J_sjk"]))``. A run at or
        above it is discarded as poorly converged. Default ``sqrt(5)``.
    M_min : int or float, optional
        Minimum number of runs that must survive the filters: a count when
        greater than 1, a proportion of ``len(vp_list)`` when at most 1
        (so exactly 1 means every run). Default ``2/3``.
    seed : None, int, array_like[int], SeedSequence, BitGenerator or \
Generator, optional
        Seed of the object's random generator, which drives the Monte
        Carlo entropy estimate and :meth:`sample`. ``None`` derives the
        generator from NumPy's global state, so ``np.random.seed`` before
        construction still fixes a run. Default ``None``.
    show_tips : bool, optional
        Show occasional guidance when INFO logging is enabled. Set to
        ``False`` to suppress tips while keeping progress and cap messages.
        Default ``True``.
    noisy : bool or None, optional
        Override the noise status of the whole stack. With ``None`` (the
        default), read each retained run's ``uncertainty_handling_level``
        statistic; when absent, infer noise from ``elbo_sd > 0.1``. This
        inference can misclassify old posteriors. Any noisy retained run
        selects the capped headline ELBO.

    Attributes
    ----------
    vp_list : list of VariationalPosterior
        The retained posteriors, in input order. They are read, never
        modified: sampling works on copies that use this object's
        generator.
    D : int
        Dimension of the problem.
    M : int
        Number of retained runs.
    K : list of int
        Number of components of each retained run.
    w : np.ndarray, shape (1, K_total)
        Weights of all components, concatenated in run order and
        normalized. :meth:`optimize` replaces them with the optimized ones.
    elbo : float or None
        After :meth:`optimize`: the component-median capped ELBO for noisy
        runs, or the raw ELBO for noiseless runs, evaluated with fresh draws
        at the returned weights. Capping is a heuristic correction for
        optimistic expected log-joint estimates.
    elbo_sd : float or None
        Estimated uncertainty of the uncapped evaluation at the returned
        weights: entropy Monte Carlo error and GP quadrature uncertainty,
        treating runs as independent. It excludes selection bias and
        uncertainty of the cap; it does not define a calibrated confidence
        interval for the capped headline.
    elbo_details : dict or None
        After :meth:`optimize`: ``raw``, ``capped_I_median``,
        ``capped_E_median`` and ``naive`` ELBO estimates; ``headline_method``
        (``raw`` or ``capped_I_median``); ``cap_amount`` (the reduction
        applied to the headline); ``entropy_sd``, ``gp_sd`` and ``raw_sd``
        (equal to :attr:`elbo_sd`); ``noisy`` and ``noise_status_source``.
        Naive stacking equally weights retained runs with their original
        internal weights. Its estimate inherits the runs' errors and is
        a diagnostic baseline, not a bound on the optimized ELBO.
    noisy : bool
        Whether the stack is treated as noisy.
    noise_status_source : tuple of str
        How each retained run's noise status was obtained: ``recorded``,
        ``inferred`` or ``override``, in retained-run order.
    entropy : float or None
        After :meth:`optimize`: the entropy estimate of the stacked
        posterior at the optimized weights.
    I : np.ndarray, shape (1, K_total)
        Expected log-joint of every component (the mean over the run's GP
        hyperparameter samples of ``stats["I_sk"]``), in the run's own
        transformed space.
    individual_elbos : list of float
        The ELBO of each retained run, used to initialize the weights.
    I_corrected, E_corrected : np.ndarray
        The expected log-joints corrected to the original space, per
        component (shape ``(1, K_total)``) and per run (shape ``(M,)``),
        computed deterministically at construction. The capped ELBO
        variants limit the expected log-joint to their medians.
    rng : np.random.Generator
        The object's random generator.
    logger : logging.Logger
        Progress goes to the ``"SVBMC"`` logger at ``INFO`` level; raise its
        level (to ``WARNING``, say) to quieten a run.
    """

    def __init__(
        self,
        vp_list: list,
        s_max: float = np.sqrt(5),
        M_min: float | int = 2 / 3,
        seed=None,
        *,
        show_tips: bool = True,
        noisy: bool | None = None,
    ):
        # Fail before any work if the optional dependency is missing.
        _import_torch()
        if not isinstance(show_tips, (bool, np.bool_)):
            raise TypeError("`show_tips` should be a boolean.")
        if noisy is not None and not isinstance(noisy, (bool, np.bool_)):
            raise TypeError("`noisy` should be a boolean or None.")
        self.show_tips = bool(show_tips)
        # Root logger as in VBMC (a no-op if one is configured already).
        logging.basicConfig(stream=sys.stdout, format="%(message)s")
        self.logger = logging.getLogger("SVBMC")
        if self.logger.level == logging.NOTSET:
            # Progress is shown by default; a level set by the user stays.
            self.logger.setLevel(logging.INFO)

        self.rng = get_rng(seed)

        vp_list = list(vp_list)
        self.D = int(_validate_posteriors(vp_list))
        n_input = len(vp_list)

        # Keep the converged runs whose expected log-joint is precise enough.
        self.vp_list = [
            vp
            for vp in vp_list
            if bool(vp.stats["stable"])
            and np.sqrt(np.max(vp.stats["J_sjk"])) < s_max
        ]

        # Minimum number of runs: a proportion when M_min <= 1, else a count.
        if M_min <= 0:
            raise ValueError(
                f"`M_min` should be a positive number, but got {M_min}."
            )
        elif M_min <= 1:
            M_min = n_input * M_min
        elif M_min > n_input:
            warnings.warn(
                f"`M_min` should be at most `len(vp_list)`, but got {M_min} "
                f"and {n_input}, respectively. Setting `M_min` to {n_input}.",
                stacklevel=2,
            )
            M_min = n_input
        elif np.round(M_min) != M_min:
            warnings.warn(
                "A minimum number of runs should be an integer; rounding "
                f"`M_min` = {M_min} to {int(np.round(M_min))}.",
                stacklevel=2,
            )
            M_min = int(np.round(M_min))
        else:
            M_min = int(M_min)
        if len(self.vp_list) < M_min:
            raise ValueError(
                f"Expected at least {int(np.ceil(M_min))} well-converged "
                f"VBMC runs, but got {len(self.vp_list)}. Check your VBMC "
                "runs or change the values of `s_max` and `M_min`."
            )
        self.logger.info(
            "Got %d well-converged runs after filters.", len(self.vp_list)
        )

        noise_flags, noise_sources = [], []
        for vp in self.vp_list:
            if noisy is not None:
                noise_flags.append(bool(noisy))
                noise_sources.append("override")
            elif "uncertainty_handling_level" in vp.stats:
                level = vp.stats["uncertainty_handling_level"]
                if level not in (0, 1, 2):
                    raise ValueError(
                        "`uncertainty_handling_level` should be 0, 1 or 2."
                    )
                noise_flags.append(level > 0)
                noise_sources.append("recorded")
            else:
                sd = vp.stats.get("elbo_sd")
                if sd is None or not np.isfinite(sd) or sd < 0:
                    raise ValueError(
                        "A retained posterior without noise metadata needs "
                        "a finite, nonnegative `elbo_sd`; set `noisy` "
                        "explicitly when its noise status is known."
                    )
                noise_flags.append(sd > 0.1)
                noise_sources.append("inferred")
        self.noisy = bool(any(noise_flags))
        self.noise_status_source = tuple(noise_sources)

        # Components per run, their weights (normalized over all runs) and
        # the posterior-mean expected log-joint of every component.
        self.K = [int(vp.mu.shape[1]) for vp in self.vp_list]
        self.M = len(self.vp_list)
        self.w = np.concatenate(
            [
                np.reshape(np.asarray(vp.w, dtype=np.float64), (1, k))
                for vp, k in zip(self.vp_list, self.K)
            ],
            axis=1,
        )
        self.w = self.w / np.sum(self.w)
        self._naive_weights = np.concatenate(
            [np.ravel(vp.w) / np.sum(vp.w) / self.M for vp in self.vp_list]
        ).astype(np.float64)
        self.I = np.concatenate(
            [
                np.mean(
                    np.asarray(vp.stats["I_sk"], dtype=np.float64),
                    axis=0,
                    keepdims=True,
                )
                for vp in self.vp_list
            ],
            axis=1,
        )
        self.individual_elbos = [
            float(vp.stats["elbo"]) for vp in self.vp_list
        ]
        self._jacobian_corrections = np.concatenate(
            [expected_log_jacobian(vp) for vp in self.vp_list]
        )
        self.I_corrected = self.I - self._jacobian_corrections
        offsets = np.concatenate([[0], np.cumsum(self.K)])
        self.E_corrected = np.array(
            [
                np.dot(
                    self.I_corrected[0, offsets[m] : offsets[m + 1]],
                    np.ravel(vp.w) / np.sum(vp.w),
                )
                for m, vp in enumerate(self.vp_list)
            ]
        )
        self.elbo = None
        self.elbo_sd = None
        self.elbo_details = None
        self.entropy = None

    def stacked_entropy(self, w, n_samples: int = 20):
        """
        Monte Carlo estimate of the entropy of the stacked posterior.

        Differentiable with respect to the mixture weights. Every component
        contributes ``n_samples`` draws, taken in its own run's transformed
        space and mapped to the original space; the log density of the
        stacked mixture at every draw is then evaluated by mapping the draws
        into each component's space with the matching Jacobian correction.
        The draws come from this object's generator.

        Parameters
        ----------
        w : torch.Tensor
            Weights of the stacked posterior (any shape holding ``K_total``
            values; normalized internally). Converted to float64.
        n_samples : int, optional
            Draws per component. Default 20.

        Returns
        -------
        H : torch.Tensor
            The entropy estimate (a float64 scalar).
        J_corrections : np.ndarray, shape (K_total,)
            Per-component expected log-Jacobian, computed deterministically
            at construction, which :meth:`stacked_ELBO` subtracts from the
            stored expected log-joints to express them in original space.
        """
        H, corrections, _ = self._stacked_entropy(w, n_samples)
        return H, corrections

    def _stacked_entropy(self, w, n_samples, *, compute_variance=False):
        """Evaluate entropy and optionally its stratified sampling variance."""
        torch = _import_torch()
        n_samples = int(n_samples)
        if n_samples < 1:
            raise ValueError("`n_samples` should be at least 1.")

        K_total = int(np.sum(self.K))
        dtype = torch.float64
        device = w.device

        w = w.to(dtype).reshape(1, K_total)
        w = w / w.sum()
        log_w = torch.log(w + 1e-40)

        # One entry per component: its run's transform, location and scale.
        subcomps = []
        for vp in self.vp_list:
            sigma = vp.lambd * vp.sigma
            for k in range(vp.mu.shape[1]):
                subcomps.append(
                    (vp.parameter_transformer, vp.mu[:, k], sigma[:, k])
                )

        # Step 1: draws from every component, mapped to the original space.
        S = K_total * n_samples
        X_orig = np.zeros((S, self.D))
        comp_index = np.repeat(np.arange(K_total), n_samples)
        for mk, (transform, mu, sigma) in enumerate(subcomps):
            z = self.rng.standard_normal((n_samples, self.D))
            x_mk_transform = z * sigma + mu  # (n_samples, D)
            rows = slice(mk * n_samples, (mk + 1) * n_samples)
            X_orig[rows, :] = transform.inverse(x_mk_transform)

        # Step 2: log q_mk(x) in the original space for every draw and
        # component (diagonal normal in the component's space, Jacobian
        # corrected).
        logq_matrix = np.zeros((S, K_total))
        for mk, (transform, mu, sigma) in enumerate(subcomps):
            X_transform_mk = transform(X_orig)
            jac_corr = transform.log_abs_det_jacobian(X_transform_mk)
            logq_mk_transform = np.sum(
                sp.stats.norm.logpdf(X_transform_mk, mu, sigma), axis=1
            )
            logq_matrix[:, mk] = logq_mk_transform - jac_corr

        # Step 3: the weights enter here, so switch to torch for the
        # gradient.
        logq_matrix = torch.as_tensor(logq_matrix, dtype=dtype, device=device)
        logq_orig = torch.logsumexp(logq_matrix + log_w, dim=1)  # (S,)

        # E_{q_mk}[log q(x)] for every component: mean over its draws.
        sum_logq = torch.zeros(K_total, dtype=dtype, device=device)
        count_logq = torch.zeros(K_total, dtype=dtype, device=device)
        var_logq = torch.zeros(K_total, dtype=dtype, device=device)
        for mk in range(K_total):
            mask = comp_index == mk
            count_mk = mask.sum()
            if count_mk > 0:
                sum_logq[mk] = logq_orig[mask].sum()
                count_logq[mk] = count_mk
                if compute_variance:
                    var_logq[mk] = logq_orig[mask].var(unbiased=True)
        E_mk_logq = sum_logq / (count_logq + 1e-40)

        H = -w @ E_mk_logq
        varH = None
        if compute_variance:
            varH = float((w.square() @ (var_logq / n_samples)).item())
        return H[0], self._jacobian_corrections.copy(), varH

    def stacked_ELBO(self, w, n_samples: int = 20):
        """
        Estimate the stacked ELBO at given weights.

        The expected log-joint is the weighted sum of the components' stored
        expected log-joints (from the runs' GP surrogates), corrected to the
        original space with the Jacobian terms of :meth:`stacked_entropy`;
        the entropy is estimated by Monte Carlo. The expected log-joints
        are corrected deterministically during construction.

        Parameters
        ----------
        w : torch.Tensor, np.ndarray or list
            Weights of the stacked posterior (``K_total`` values; normalized
            internally). NumPy or list input is converted to a float64
            tensor.
        n_samples : int, optional
            Draws per component for the entropy estimate. Default 20.

        Returns
        -------
        ELBO : torch.Tensor
            The estimated stacked ELBO (float64 scalar).
        H : torch.Tensor
            The estimated entropy (float64 scalar).
        """
        torch = _import_torch()
        if isinstance(w, torch.Tensor):
            w = w.to(torch.float64)
            w = w / w.sum()
        elif isinstance(w, (np.ndarray, list, tuple)):
            w_np = np.asarray(w, dtype=np.float64)
            w = torch.as_tensor(w_np / np.sum(w_np), dtype=torch.float64)
        else:
            raise TypeError(
                "`w` should be a torch.Tensor, a NumPy array or a list, "
                f"got {type(w).__name__}."
            )

        H, _ = self.stacked_entropy(w, n_samples)

        I_corr_t = torch.as_tensor(
            self.I_corrected, dtype=torch.float64, device=w.device
        )
        G = (w.reshape(1, -1) @ I_corr_t.T).squeeze()
        return G + H, H

    def maximize_ELBO(
        self,
        n_samples: int = 20,
        lr: float = 0.1,
        max_steps: int = 500,
        version: str = "all-weights",
    ):
        """
        Maximize the stacked ELBO over the weights with Adam.

        The weights are the softmax of unconstrained logits, initialized
        from the runs' own weights and ELBOs so that better runs start
        heavier. Every step draws fresh entropy samples. The optimization
        stops when the ELBO, rounded to five decimals, has not improved for
        five consecutive steps, or after ``max_steps``; the best iterate is
        returned.

        Parameters
        ----------
        n_samples : int, optional
            Draws per component for the entropy estimate. Default 20.
        lr : float, optional
            Adam learning rate. Default 0.1.
        max_steps : int, optional
            Maximum number of steps. Default 500.
        version : {"all-weights", "posterior-only", "ns"}, optional
            ``"all-weights"`` optimizes the weight of every component;
            ``"posterior-only"`` optimizes one weight per run, keeping the
            relative weights within a run; ``"ns"`` (naive stacking)
            performs no optimization and returns the normalized input
            weights. Default ``"all-weights"``.

        Returns
        -------
        w_final : torch.Tensor, shape (K_total,)
            The optimized weights (float64).
        elbo_best : torch.Tensor
            Optimization-time Monte Carlo ELBO from the selected iteration
            (float64 scalar). :meth:`optimize` re-evaluates the returned
            weights with fresh draws for the reported headline.
        entropy_best : torch.Tensor
            The entropy estimate at the returned weights (float64 scalar).
        """
        torch = _import_torch()
        _validate_optimization(n_samples, max_steps, version)
        w_init = torch.as_tensor(self.w, dtype=torch.float64)
        log_w = torch.log(w_init)  # (1, K_total); optimize in log space
        repeats = torch.as_tensor(self.K)

        if version == "all-weights":
            self.logger.info("Optimizing the stacked ELBO w.r.t. all weights.")
            # Logits start at log w + ELBO of the run, favoring better runs.
            broadcasted_elbos = torch.as_tensor(
                np.repeat(self.individual_elbos, self.K), dtype=torch.float64
            )
            w_logits_init = log_w + broadcasted_elbos
            w_logits_init = w_logits_init - torch.max(w_logits_init)
            w_logits = w_logits_init.detach().clone()
            w_logits.requires_grad_(True)
            optimizer = torch.optim.Adam([w_logits], lr=lr)
            w_best = w_logits.detach().clone()

        elif version == "posterior-only":
            self.logger.info(
                "Optimizing the stacked ELBO w.r.t. the weights of "
                "individual VBMC posteriors."
            )
            omega_init = torch.as_tensor(
                self.individual_elbos, dtype=torch.float64
            )
            omega_init = omega_init - torch.max(omega_init)
            omega_logits = omega_init.detach().clone()
            omega_logits.requires_grad_(True)
            optimizer = torch.optim.Adam([omega_logits], lr=lr)
            w_best = (
                torch.repeat_interleave(
                    omega_logits.detach().clone(), repeats=repeats, dim=0
                )
                + log_w
            )

        else:  # "ns"
            self.logger.info("Naive stacking: averaging the VBMC posteriors.")
            w_final = torch.as_tensor(
                self._naive_weights.copy(), dtype=torch.float64
            )
            elbo_best, entropy_best = self.stacked_ELBO(
                w_final, n_samples=n_samples
            )
            return w_final, elbo_best, entropy_best

        # Converged when the rounded ELBO has not improved for 5 steps.
        convergence_counter = 0
        loss_old = np.inf
        elbo_best = None
        entropy_best = None

        for step in range(int(max_steps)):
            optimizer.zero_grad()
            if version == "posterior-only":
                w_logits = (
                    torch.repeat_interleave(
                        omega_logits, repeats=repeats, dim=0
                    )
                    + log_w
                )
            w = torch.softmax(w_logits, dim=-1)
            ELBO, H = self.stacked_ELBO(w, n_samples=n_samples)

            if step == 0:
                self.logger.info("Initial elbo = %s", ELBO.detach().item())
            if (step + 1) % 5 == 0:
                self.logger.info(
                    "iter %d: elbo = %s", step + 1, ELBO.detach().item()
                )

            loss = -ELBO
            loss_new = torch.round(loss * 1e5) / 1e5
            if loss_new >= loss_old:
                convergence_counter += 1
            else:
                convergence_counter = 0
                w_best = w_logits.detach().clone()
                elbo_best = -loss
                entropy_best = H
                loss_old = loss_new
            if convergence_counter >= 5:
                break

            loss.backward()
            optimizer.step()

        w_final = torch.softmax(w_best, dim=-1).flatten()
        return w_final, elbo_best, entropy_best

    def optimize(
        self,
        n_samples: int = 20,
        lr: float = 0.1,
        max_steps: int = 500,
        version: str = "all-weights",
        *,
        n_samples_final: int = 100,
    ):
        """
        Optimize the stacked posterior in place.

        Runs :meth:`maximize_ELBO`, stores the optimized weights in
        :attr:`w` (NumPy float64), the entropy estimate in :attr:`entropy`
        and a fresh final evaluation in :attr:`elbo`, :attr:`elbo_sd` and
        :attr:`elbo_details`.
        Returns ``None``.

        Parameters
        ----------
        n_samples : int, optional
            Draws per component for the entropy estimate. Default 20.
        lr : float, optional
            Adam learning rate. Default 0.1.
        max_steps : int, optional
            Maximum number of steps. Default 500.
        version : {"all-weights", "posterior-only", "ns"}, optional
            Optimization mode; see :meth:`maximize_ELBO`. Default
            ``"all-weights"``.
        n_samples_final : int, optional
            Fresh entropy draws per component at the returned weights and
            at the naive-stacking weights. At least 2, to estimate sampling
            variance. Default 100. Naive mode reuses one final evaluation.
        """
        if (
            isinstance(n_samples_final, (bool, np.bool_))
            or not isinstance(n_samples_final, (int, np.integer))
            or n_samples_final < 2
        ):
            raise ValueError("`n_samples_final` should be an integer >= 2.")
        _validate_optimization(n_samples, max_steps, version)
        torch = _import_torch()
        consider_runtime_tip(
            enabled=self.show_tips,
            display=self.logger.isEnabledFor(logging.INFO),
            noisy=self.noisy,
        )
        w, _, _ = self.maximize_ELBO(
            n_samples=n_samples, lr=lr, max_steps=max_steps, version=version
        )
        self.w = w.detach().cpu().numpy().astype(np.float64).reshape(1, -1)
        self.w /= self.w.sum(dtype=np.float64)
        with torch.no_grad():
            final_w = torch.as_tensor(self.w, dtype=torch.float64)
            H, _, varH = self._stacked_entropy(
                final_w, n_samples_final, compute_variance=True
            )
            self.entropy = float(H.item())
            G = float(np.dot(self.w.ravel(), self.I_corrected.ravel()))
            raw = G + self.entropy
            if version == "ns":
                naive = raw
            else:
                naive_H, _, _ = self._stacked_entropy(
                    torch.as_tensor(self._naive_weights, dtype=torch.float64),
                    n_samples_final,
                )
                naive = float(
                    np.dot(self._naive_weights, self.I_corrected.ravel())
                    + naive_H.item()
                )

        # Cap the expected log-joint at the median component / run value to
        # counter the optimistic bias of maximizing a noisy estimate.
        I_median = float(np.median(self.I_corrected))
        E_median = float(np.median(self.E_corrected))
        method = "capped_I_median" if self.noisy else "raw"
        varG = self._expected_log_joint_variance(self.w.ravel())
        self.elbo_sd = float(np.sqrt(varG + varH))
        capped_I = min(G, I_median) + self.entropy
        self.elbo = float(capped_I if self.noisy else raw)
        self.elbo_details = {
            "raw": raw,
            "capped_I_median": capped_I,
            "capped_E_median": min(G, E_median) + self.entropy,
            "naive": naive,
            "headline_method": method,
            "cap_amount": float(max(0.0, raw - self.elbo)),
            "entropy_sd": float(np.sqrt(varH)),
            "gp_sd": float(np.sqrt(varG)),
            "raw_sd": self.elbo_sd,
            "noisy": self.noisy,
            "noise_status_source": self.noise_status_source,
        }
        if self.elbo_details["cap_amount"] > 0:
            self.logger.info(
                "Expected log-joint capped by %.3g nats.",
                self.elbo_details["cap_amount"],
            )

    def _expected_log_joint_variance(self, w):
        """GP uncertainty at fixed weights, summed over independent runs."""
        variance = 0.0
        offset = 0
        for vp, k in zip(self.vp_list, self.K):
            weights = w[offset : offset + k]
            I = np.asarray(vp.stats["I_sk"], dtype=np.float64)
            J = np.array(vp.stats["J_sjk"], dtype=np.float64, copy=True)
            diag = np.arange(k)
            J[:, diag, diag] = np.maximum(np.spacing(1), J[:, diag, diag])
            conditional = np.einsum("sjk,j,k->s", J, weights, weights)
            if np.any(weights):
                variance += float(
                    np.mean(np.maximum(conditional, np.spacing(1)))
                )
                if I.shape[0] > 1:
                    variance += float(np.var(I @ weights, ddof=1))
            offset += k
        return variance

    def sample(self, n_samples: int, balance_flag: bool = False):
        """
        Draw samples from the stacked posterior in the original space.

        The stacked posterior is a mixture over the retained runs, so a
        draw picks a run with probability equal to that run's total weight
        and then a point from that run's reweighted posterior through its
        own transform. By default the run counts are multinomial, the
        component choice within a run is random (see
        :meth:`VariationalPosterior.sample`) and the rows are shuffled, so
        the result is an independent sample of exactly ``n_samples`` rows.
        With ``balance_flag=True`` the draws are split across runs, and
        across components within a run, in proportion to the weights, every
        count within one draw of its exact share (a stratified sample with
        lower variance for expectations, not an independent one). All randomness comes from this object's
        generator; the input posteriors are not touched.

        Parameters
        ----------
        n_samples : int
            Number of draws.
        balance_flag : bool, optional
            Stratify the draws by run and component. Default ``False``.

        Returns
        -------
        X : np.ndarray, shape (n_samples, D)
            Draws in the original parameter space.
        """
        n = int(n_samples)
        if n < 0:
            raise ValueError("`n_samples` should be non-negative.")
        w_flat = np.ravel(self.w)
        offsets = np.concatenate([[0], np.cumsum(self.K)])
        omega = np.array(
            [w_flat[offsets[m] : offsets[m + 1]].sum() for m in range(self.M)]
        )
        omega = omega / omega.sum()

        if balance_flag:
            counts = _balanced_counts(n, omega, self.rng)
        else:
            counts = self.rng.multinomial(n, omega)

        Xs = []
        for m, vp in enumerate(self.vp_list):
            if counts[m] == 0:
                continue
            # A copy carries the optimized weights of this run's components
            # and draws from this object's generator, not the input's.
            vp_copy = copy.deepcopy(vp)
            vp_copy.rng = self.rng
            comp_w = w_flat[offsets[m] : offsets[m + 1]]
            vp_copy.w = (comp_w / comp_w.sum()).reshape(1, -1)
            X_m, _ = vp_copy.sample(int(counts[m]), balance_flag=balance_flag)
            Xs.append(np.asarray(X_m, dtype=np.float64).reshape(-1, self.D))

        if not Xs:
            return np.zeros((0, self.D))
        X = np.concatenate(Xs, axis=0)
        return X[self.rng.permutation(X.shape[0])]

    def plot(
        self,
        n_samples: int = 10000,
        color: str = "black",
        figsize: tuple | None = None,
        smooth: float | None = None,
        **corner_kwargs,
    ):
        """
        Draw a corner plot of the stacked posterior from fresh samples.

        Parameters
        ----------
        n_samples : int, optional
            Draws taken with :meth:`sample`. Default 10000.
        color : str, optional
            Matplotlib color of the contours and histograms. Default
            ``"black"``.
        figsize : (float, float) or None, optional
            Figure size in inches; ``None`` uses 2.5 inches per dimension.
        smooth : float or None, optional
            Gaussian kernel smoothing applied by ``corner``.
        **corner_kwargs
            Passed to :func:`corner.corner`.

        Returns
        -------
        matplotlib.figure.Figure
            The figure.
        """
        import corner
        from matplotlib import pyplot as plt

        samples = self.sample(n_samples)
        if figsize is None:
            base = 2.5
            figsize = (base * self.D, base * self.D)
        labels = [f"$x_{{{i+1}}}$" for i in range(self.D)]

        fig = plt.figure(figsize=figsize)
        corner.corner(
            samples,
            labels=labels,
            color=color,
            smooth=smooth,
            show_titles=True,
            fig=fig,
            **corner_kwargs,
        )
        fig.tight_layout()
        if not plt.get_backend().lower().endswith("agg"):
            plt.show()
        else:
            fig.canvas.draw()
        return fig
