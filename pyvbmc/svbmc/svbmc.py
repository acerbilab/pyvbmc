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

_REQUIRED_STATS = ("stable", "elbo", "I_sk", "J_sjk")
_VERSIONS = ("all-weights", "posterior-only", "ns")


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
        if not (
            np.all(np.isfinite(I_sk))
            and np.all(np.isfinite(J_sjk))
            and np.isfinite(float(vp.stats["elbo"]))
        ):
            raise ValueError(
                f"`vp_list[{i}]` has nonfinite `elbo`, `I_sk` or `J_sjk` "
                "statistics."
            )
    return D


def _balanced_counts(n, omega, rng):
    """Split ``n`` draws across runs in proportion to ``omega`` exactly.

    Every run gets the integer part of its quota ``n * omega``; the
    remaining draws go to runs chosen without replacement with probability
    proportional to the fractional parts, the scheme
    :meth:`VariationalPosterior.sample` uses across components.
    """
    quota = n * omega
    counts = np.floor(quota).astype(int)
    remainder = int(n - counts.sum())
    if remainder > 0:
        frac = quota - counts
        if np.count_nonzero(frac) >= remainder:
            p = frac / frac.sum()
            extra = rng.choice(omega.size, size=remainder, replace=False, p=p)
        else:  # rounding left too few fractional parts: uniform fallback
            extra = rng.choice(omega.size, size=remainder, replace=False)
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
    elbo : dict or None
        After :meth:`optimize`: ``"estimated"`` (the optimized stacked
        ELBO), ``"debiased_I_median"`` and ``"debiased_E_median"`` (the
        same with the expected log-joint capped at the median component
        and median run values, which counters the optimistic bias of
        maximizing a noisy estimate).
    entropy : float or None
        After :meth:`optimize`: the entropy estimate of the stacked
        posterior at the optimized weights.
    rng : np.random.Generator
        The object's random generator.
    logger : logging.Logger
        Progress goes to the ``"SVBMC"`` logger at ``INFO`` level; lower its
        level to quieten a run.
    """

    def __init__(
        self,
        vp_list: list,
        s_max: float = np.sqrt(5),
        M_min: float | int = 2 / 3,
        seed=None,
    ):
        # Fail before any work if the optional dependency is missing.
        _import_torch()
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
                f"Expected at least {M_min} well-converged VBMC runs, but "
                f"got {len(self.vp_list)}. Check your VBMC runs or change "
                "the values of `s_max` and `M_min`."
            )
        self.logger.info(
            "Got %d well-converged runs after filters.", len(self.vp_list)
        )

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
        # Jacobian-corrected expected log-joints, per component and per run,
        # fixed at the first ELBO evaluation and used to debias the result.
        self.I_corrected = None
        self.E_corrected = np.zeros(self.M)
        self.elbo = None
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
            Per-component mean log-Jacobian of its run's transform at the
            draws, which :meth:`stacked_ELBO` subtracts from the stored
            expected log-joints so that both terms refer to the original
            space.
        """
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
        J_corrections = np.zeros(K_total)
        for mk, (transform, mu, sigma) in enumerate(subcomps):
            z = self.rng.standard_normal((n_samples, self.D))
            x_mk_transform = z * sigma + mu  # (n_samples, D)
            J_corrections[mk] = np.mean(
                transform.log_abs_det_jacobian(x_mk_transform)
            )
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
        for mk in range(K_total):
            mask = comp_index == mk
            count_mk = mask.sum()
            if count_mk > 0:
                sum_logq[mk] = logq_orig[mask].sum()
                count_logq[mk] = count_mk
        E_mk_logq = sum_logq / (count_logq + 1e-40)

        H = -w @ E_mk_logq
        return H[0], J_corrections

    def stacked_ELBO(self, w, n_samples: int = 20):
        """
        Estimate the stacked ELBO at given weights.

        The expected log-joint is the weighted sum of the components' stored
        expected log-joints (from the runs' GP surrogates), corrected to the
        original space with the Jacobian terms of :meth:`stacked_entropy`;
        the entropy is estimated by Monte Carlo. The first call fixes the
        corrected expected log-joints used by :meth:`optimize` to debias
        the result.

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

        H, J_corrections = self.stacked_entropy(w, n_samples)
        I_corrected = self.I - J_corrections  # (1, K_total)

        if self.I_corrected is None:
            self.I_corrected = I_corrected
            idx = 0
            for m, (vp, k) in enumerate(zip(self.vp_list, self.K)):
                self.E_corrected[m] = np.sum(
                    I_corrected[0, idx : idx + k] * np.ravel(vp.w)
                )
                idx += k

        I_corr_t = torch.as_tensor(
            I_corrected, dtype=torch.float64, device=w.device
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
            The stacked ELBO at the returned weights (float64 scalar).
        entropy_best : torch.Tensor
            The entropy estimate at the returned weights (float64 scalar).
        """
        torch = _import_torch()
        if version not in _VERSIONS:
            raise ValueError(
                f"Unknown S-VBMC version {version!r}; choose one of "
                f"{', '.join(repr(v) for v in _VERSIONS)}."
            )
        if int(max_steps) < 1:
            raise ValueError("`max_steps` should be at least 1.")
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
            w_final = (w_init / w_init.sum()).flatten()
            elbo_best, entropy_best = self.stacked_ELBO(
                w_final, n_samples=n_samples
            )
            return w_final, elbo_best, entropy_best

        # Converged when the rounded ELBO has not improved for 5 steps.
        convergence_counter = 0
        loss_old = 1e8
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
    ):
        """
        Optimize the stacked posterior in place.

        Runs :meth:`maximize_ELBO`, stores the optimized weights in
        :attr:`w` (NumPy float64), the entropy estimate in :attr:`entropy`
        and the estimated and debiased stacked ELBO values in :attr:`elbo`.
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
        """
        w, ELBO, H = self.maximize_ELBO(
            n_samples=n_samples, lr=lr, max_steps=max_steps, version=version
        )
        self.w = w.detach().cpu().numpy().astype(np.float64).reshape(1, -1)
        self.w /= self.w.sum(dtype=np.float64)
        self.entropy = float(H.detach().cpu().numpy())
        ELBO = float(ELBO.detach().cpu().numpy())

        # Cap the expected log-joint at the median component / run value to
        # counter the optimistic bias of maximizing a noisy estimate.
        I_median = float(np.median(self.I_corrected))
        E_median = float(np.median(self.E_corrected))
        G = ELBO - self.entropy
        self.elbo = {
            "estimated": ELBO,
            "debiased_I_median": min(G, I_median) + self.entropy,
            "debiased_E_median": min(G, E_median) + self.entropy,
        }

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
        across components within a run, in exact proportion to the weights
        (a stratified sample with lower variance for expectations, not an
        independent one). All randomness comes from this object's
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
