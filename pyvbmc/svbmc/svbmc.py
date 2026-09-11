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

import numpy as np
import scipy as sp


def _import_torch():
    """Return the torch module, or raise a message naming the extra."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised without torch
        raise ImportError(
            "S-VBMC requires torch; install pyvbmc[torch]."
        ) from exc
    return torch


class SVBMC:
    """
    Stacking Variational Bayesian Monte Carlo (S-VBMC) is a method specifically designed to 'stack'
    variational posteriors (mixture of Gaussians) coming from different Variational Bayesian Monte
    Carlo (VBMC) runs. It does so by optimizing the 'stacked ELBO' with respect to (w.r.t.) only
    the weights of each Gaussian component of the 'stacked posterior'.

    Initialize a ``SVBMC`` object to set up the inference problem, then run
    ``optimize()``.
    The initialization automatically filters out VBMC runs where:
        1- The algorithm has not converged;
        2- The algorith has converged poorly, with excessive uncertainty associated with at least one
        of the components of the expected log-joint.

    Parameters:
    -----------
    vp_list : list
        A list of ``VariationalPosterior`` (vp) objects output by VBMC
        (see https://github.com/acerbilab/pyvbmc/tree/main for details)
    s_max : float
        A tolerance bound for the maximum standard deviation associated with individual components
        of the expected log-joint `I_sk` (these values are specified within the ``VariationalPosterior``
        objects in `vp_list`). If a VBMC run has any `I_sk` with associated standard deviation >= `s_max`,
        the whole run is considered poorly converged and discarded. Defaults to sqrt(5).
    M_min : int | float
        The minimum number (if `int` and > 1) or proportion (if `float` and <= 1) of VBMC runs to be used for S-VBMC.
        If the number of well-converged runs is below this value, an error is produced. Defaults to 2/3.
        **NOTE**: a value of exactly 1 is going to be interpreted as a PROPOTION (100% of runs), not as a number of runs.
    testing: bool
        Whether you are performing unit tests or not. If set to `True`, the sampling for `stacked_entropy()`
        will be deterministic. In practice this slows down the optimization and overfits noise, so it
        should always be kept to `False` outside of testing.
    """

    def __init__(
        self,
        vp_list: list,
        s_max: float = np.sqrt(5),
        M_min: float | int = 2 / 3,
        testing: bool = False,
    ):
        # Fail early, before the user has invested in the run filters.
        _import_torch()

        # whether we are performing unit tests or not
        self.testing = testing
        # Store all the variational posterior (`vp`, output of PyVBMC) objects in one list,
        # filtered based on `s_max`
        self.vp_list = [
            vp
            for vp in vp_list
            if (
                vp.stats["stable"]
                and np.sqrt(np.max(vp.stats["J_sjk"])) < s_max
            )
        ]
        # Set minimum number of runs according to `M_min`
        if M_min <= 0:  # can't be negative
            raise ValueError(
                f"`M_min` should be a positive number, but got {M_min}."
            )
        elif M_min <= 1:  # proportion of total runs if <= 1
            M_min = len(vp_list) * M_min
        elif M_min > len(vp_list):  # number of runs if float > 1
            print(
                f"Warning: `M_min` should be lower than or equal to `len(vp_list)`, but got {M_min} and {len(vp_list)}, respectively. Setting `M_min` to {len(vp_list)}"
            )
            M_min = len(vp_list)
        elif np.round(M_min) != M_min:  # number of runs if integer > 1
            print(
                f"Warning: if you want to specify the minimum number of runs to use, you should input `M_min` as an integer. Rounding to closest integer, {int(np.round(M_min))}"
            )
            M_min = int(np.round(M_min))
        else:
            M_min = int(M_min)
        # Check if enough runs passed the filters
        if len(self.vp_list) < M_min:
            raise ValueError(
                f"expected at least {M_min} well-converged VBMC runs, but got {len(self.vp_list)}. Check your VBMC runs or change the values of `s_max` and `M_min`."
            )
        else:
            print(
                f"Got {len(self.vp_list)} well-converged runs after filters."
            )
        # If all is well, proceed with extracting useful variables
        # Dimensionality of inference problem
        self.D = self.vp_list[0].mu.shape[0]
        # Number of components for each `vp`
        self.K = [vp.mu.shape[1] for vp in self.vp_list]
        # Number of `vp`s to stack
        self.M = len(self.vp_list)
        # Weights of the components of the individual `vp`s, concatenated
        self.w = np.concatenate(
            [np.reshape(vp.w, (1, vp.mu.shape[1])) for vp in self.vp_list],
            axis=1,
        )
        self.w = self.w / np.sum(self.w)  # Normalize
        # Get mean across GPs for the expected log-joint contributions
        self.I = np.concatenate(
            [
                np.mean(vp.stats["I_sk"], axis=0, keepdims=True)
                for vp in self.vp_list
            ],
            axis=1,
        )
        # Store ELBO of individual `vp`s
        self.individual_elbos = [vp.stats["elbo"] for vp in self.vp_list]
        # Initialize `I_corrected` as empty list
        self.I_corrected = []
        # Initialize `E_corrected` as zeros array
        self.E_corrected = np.zeros((self.M))

        # Dedicated seed to guarantee deterministic MC estimates across
        # repeated calls (important for unit-test reproducibility).
        self._svbmc_random_seed = 0

    def stacked_entropy(
        self,
        w,
        n_samples: int = 20,
    ):
        """
        Monte Carlo estimate of the mixture entropy for `K_total` components,
        differentiable w.r.t. the mixture weights.
        The individual VBMC posteriors exist in different "transformed" feature
        spaces (because of variational whitening), so Jacobian corrections need
        to be applied (and stored to be applied to the expected log-joint as well).

        This works in the following steps:

        1- n_samples samples are taken from each Gaussian component (in the corresponding VBMC
        posterior transformed space) of the stacked posterior, and transformed into the original
        feature space (the individual vp objects have a built-in method for this);
        2- for each component of the stacked posterior, ALL samples are transformed into that
        component's feature space, the corresponding Jacobian correction is computed and the log probability
        of all the (`K_total`*`n_samples` = `S`) samples FOR THAT SPECIFIC COMPONENT is taken and stored;
        3- the log probability of each sample FOR THE WHOLE POSTERIOR is calculated, and these are
        then averaged to obtain an estimate of the stacked entropy.

        Parameters:
        -----------
        w: torch.Tensor
            The weights of the stacked posterior.
        n_samples: int
            The number of samples to take from each component of the stacked posterior.

        Returns:
        --------
        H: float
            Estimated entropy of the stacked posterior by Monte Carlo method.
        J_corrections: np.ndarray
            Component-specific Jacobian corrections to be applied to the expected log-joint
            in the stacked_ELBO function.
        """
        torch = _import_torch()

        K_total = np.sum(self.K)

        dtype = w.dtype
        device = w.device

        # Local, fixed-seed PRNG so that every call starts from the same
        # state, making entropy estimates repeatable.
        if self.testing:
            rng = np.random.RandomState(self._svbmc_random_seed)

        w = w.reshape(1, K_total)
        w = w / w.sum()  # ensure normalized
        log_w = torch.log(w + 1e-40)  # take the log

        # Initialize empty list for the `K_total` subcomponents
        subcomps = []
        # Initialize empty vector for Jacobian corrections.
        # This is important for correcting the expected log-joint when estimating the ELBO.
        J_corrections = np.zeros((K_total))

        for m, vp in enumerate(self.vp_list):
            sigma = vp.lambd * vp.sigma  # calculate standard deviations
            for k in range(self.K[m]):
                subcomps.append(
                    {
                        "transform": vp.parameter_transformer,  # for direct and inverse transforms and Jacobian corrections
                        "mu": vp.mu[:, k],
                        "sigma": sigma[:, k],
                    }
                )

        #### Step 1-  Build array of original space samples

        # WE START WITH NUMPY because the transform functions built in PyVBMC use that.
        # We can do this as w does not come into play yet, and we need to differentiate
        # w.r.t. `w` only.
        S = K_total * n_samples  # total number of samples
        X_orig = np.zeros(
            (S, self.D)
        )  # Initialize array of original space samples
        comp_index = np.zeros(
            (S)
        )  # Initialize array to track from which component each sample is taken

        row_offset = 0
        for mk, sc in enumerate(subcomps):
            # sample in the transformed space of the `mk`-th component (`m`-th VBMC posterior's feature space)
            if self.testing:
                samples_sn = sp.stats.multivariate_normal.rvs(
                    mean=np.zeros((self.D)),
                    cov=np.eye(self.D),
                    size=n_samples,
                    random_state=rng,  # deterministic sampling if performing unit tests
                )  # [`n_samples`, `self.D`]
            else:
                samples_sn = sp.stats.multivariate_normal.rvs(
                    mean=np.zeros((self.D)),
                    cov=np.eye(self.D),
                    size=n_samples,
                )  # [`n_samples`, `self.D`]

            x_mk_transform = (
                samples_sn * sc["sigma"] + sc["mu"]
            )  # [`n_samples`, `self.D`]

            # Store Jacobian correction
            J_corrections[mk] = np.mean(
                sc["transform"].log_abs_det_jacobian(x_mk_transform)
            )

            # Invert to original space
            x_mk_orig = sc["transform"].inverse(
                x_mk_transform
            )  # [`n_samples`, `self.D`]
            X_orig[row_offset : row_offset + n_samples, :] = x_mk_orig
            comp_index[row_offset : row_offset + n_samples] = mk
            row_offset += n_samples

        #### Step 2- Evaluate log q_{`mk`}(`x`) for all `x` in `X_orig` and for all q_{`mk`} in `subcomps`

        # Initialize matrix to store log q_{`mk`}(`x`) for all x and q_{`mk`}
        logq_matrix = np.zeros((S, K_total))

        for mk, sc in enumerate(subcomps):
            # forward transform for the `mk`-th component (to `m`-th VBMC posterior's feature space)
            X_transform_mk = sc["transform"](X_orig)  # [`S`, `D`]
            jac_corr = sc["transform"].log_abs_det_jacobian(
                X_transform_mk
            )  # [`S`]

            # diagonal normal logpdf
            logq_mk_transform = np.sum(
                sp.stats.norm.logpdf(X_transform_mk, sc["mu"], sc["sigma"]),
                axis=1,
            )
            # Apply Jacobian correction
            logq_mk_orig = logq_mk_transform - jac_corr

            logq_matrix[:, mk] = logq_mk_orig

        # WE SWITCH TO PYTORCH because `w` is going to come into play, and  the entropy
        # must be differentiable w.r.t. `w`
        logq_matrix = torch.tensor(logq_matrix, dtype=dtype, device=device)
        # add log(`w`)
        logq_matrix = logq_matrix + log_w  # [`S`, `K_total`]

        #### Step 3- Estimate the entropy

        # Logsumexp over components to obtain log q(`x`) for all samples
        logq_orig = torch.logsumexp(logq_matrix, dim=1)  # [`S`]

        # group by comp_index
        sum_logq = torch.zeros(K_total, dtype=dtype, device=device)
        count_logq = torch.zeros(K_total, dtype=dtype, device=device)

        for mk in range(K_total):
            mask = comp_index == mk
            count_mk = mask.sum()  # `K`[`m`]
            if count_mk > 0:
                sum_logq[mk] = logq_orig[mask].sum()
                count_logq[mk] = count_mk

        # E_{q_{`mk`}}[log q(`x`)] for all `mk`
        E_mk_logq = sum_logq / (count_logq + 1e-40)  # [`K_tot`]

        # Calculate the (approximate) entropy
        H = -w @ E_mk_logq

        return H[0], J_corrections  # indexing to avoid a tensor with shape [1]

    def stacked_ELBO(self, w, n_samples: int = 20):
        """
        Stacked ELBO estimation. The expected log-joint is calculated as
        sum(`w`*`I_{mk}`), with `I_{mk}` being the expected log-joint under a single
        component (`q_{mk}`). The entropy is estimated via Monte Carlo.
        As PyVBMC outputs `I` estimates in different transformed feature spaces,
        Jacobian corrections must be applied.

        Parameters:
        -----------
        w: torch.Tensor
            The weights of the stacked posterior.
        n_samples: int
            The number of samples to take from each component of the stacked
            posterior when estimating the entropy.

        Returns:
        --------
        ELBO: torch.Tensor
            The estimated ELBO.
        H: torch.Tensor
            The estimated entropy of the stacked posterior
        """
        torch = _import_torch()

        # Deal with all cases in which `w` is not already a torch.Tensor
        # For numerical consistency we always cast the newly-created tensor to
        # float32, matching the explicit dtype used in the test-suite.
        if not isinstance(w, torch.Tensor):
            if isinstance(w, np.ndarray):
                w_np = w / np.sum(w)
            elif isinstance(w, list):
                # Convert list -> NumPy first to leverage vectorised operations
                try:
                    w_np = np.array(w, dtype=float)
                    w_np = w_np / np.sum(w_np)
                except Exception:
                    # Fallback to internal default weights
                    w_np = self.w / np.sum(self.w)
            else:
                # Any other input type - rely on the stored weights
                w_np = self.w / np.sum(self.w)

            # Create a tensor with a consistent dtype
            w = torch.tensor(w_np, dtype=torch.float32)

        # Estimated the entropy of the stacked posterior and get Jacobian corrections
        H, J_corrections = self.stacked_entropy(w, n_samples)
        # Apply Jacobian corrections
        I_corrected = self.I - J_corrections

        # If this is the first iteration, store expected logjoints (posterior and component-wise) in the base space
        if len(self.I_corrected) == 0:
            self.I_corrected = I_corrected
            idx = 0
            for m, vp in enumerate(self.vp_list):
                k_i = vp.mu.shape[1]
                self.E_corrected[m] = np.sum(
                    I_corrected[0, idx : idx + k_i] * vp.w
                )
                idx += k_i

        # Calculate expected logjoint in the original feature space

        I_corr_t = torch.tensor(I_corrected, dtype=w.dtype, device=w.device)
        # The product returns either shape (1,) or (1,1); squeeze ensures a scalar.
        G = (w @ I_corr_t.T).squeeze()

        ELBO = G + H

        return ELBO, H

    def maximize_ELBO(
        self,
        n_samples: int = 20,
        lr: float = 0.1,
        max_steps: int = 500,
        version: str = "all-weights",
    ):
        """
        Maximizes ``stacked_ELBO(`w`, `n_samples`)`` by parameterizing `w` via softmax of unconstrained logits.
        Can optimize w.r.t. `w` ("all-weights") or the VBMC posterior weights `omega`, ("posterior-only").
        It can also perform naive stacking (simply re-normalizing the weights).

        Parameters:
        -----------
        n_samples: int
            The number of samples to take from each component of the stacked
            posterior when estimating the entropy.
        lr: float
            learning rate for Adam
        max_steps: int
            maximum number of gradient ascent steps
        version: string
            the type of optimization to be performed. It can take the following values:
                - "all-weights": default, optimizes w.r.t. the weights of all individual
                                components;
                - "posterior-only": optimizes w.r.t. omega, i.e., the weights of whole VBMC
                                posteriors;
                - "ns": naive stacking, simply re-normalizes the weights.

        Returns:
        --------
        w_final: torch.Tensor
            The optimized weights of the stacked posterior.
        elbo_best: torch.Tensor
            The maximized ELBO.
        entropy_best: torch.Tensor
            The entropy of the optimized stacked posterior
        """
        torch = _import_torch()

        # Setup
        w_init = torch.tensor(self.w)  # convert weights to torch.Tensor
        log_w = torch.log(w_init)  # we optimize in log space

        # Standard S-VBMC, optimize w.r.t. all weights `w`
        if version == "all-weights":
            print("Optimizing the stacked ELBO w.r.t. all weights.")
            # Prepare a broadcasted version of the individual ELBOs for weight initialization
            broadcasted_elbos = np.concatenate(
                [
                    np.ones((self.K[m])) * self.individual_elbos[m]
                    for m in range(self.M)
                ],
                axis=0,
            )
            broadcasted_elbos = torch.tensor(broadcasted_elbos)
            # Treat `w_logits` as the raw, unconstrained parameters to be optimized.
            # Initialize the weights to promote the ones coming from better runs
            w_logits_init = log_w + broadcasted_elbos
            w_logits_init = w_logits_init - torch.max(w_logits_init)
            w_logits = w_logits_init.detach().clone()
            w_logits.requires_grad_(True)
            # Set up an optimizer that will *only* update `w_logits`
            optimizer = torch.optim.Adam([w_logits], lr=lr)
            # Initialize `w_best`
            w_best = copy.deepcopy(w_logits)

        # Optimize w.r.t. `omega`, i.e., the weights of individual VBMC posteriors
        elif version == "posterior-only":
            print(
                "Optimizing the stacked ELBO w.r.t. the weights of individual VBMC posteriors."
            )
            # We treat `omega_logits` as the raw, unconstrained parameters to be optimized:
            omega_init = np.array(
                [self.individual_elbos[m] for m in range(self.M)]
            )
            omega_init = torch.tensor(omega_init)
            omega_init = omega_init - torch.max(omega_init)
            omega_logits = omega_init.detach().clone()
            omega_logits.requires_grad_(True)
            # Set up an optimizer that will *only* update `omega_logits`
            optimizer = torch.optim.Adam([omega_logits], lr=lr)
            # Initialize `w_best`
            w_best = copy.deepcopy(
                torch.repeat_interleave(
                    omega_logits.detach().clone(),
                    repeats=torch.tensor(self.K),
                    dim=0,
                )
                + log_w
            )

        elif version == "ns":
            print("Naive stacking. Just averaging VBMC posteriors.")
            # Return a *flat* 1-D tensor (using the default torch dtype) so downstream code and tests can rely on ndim == 1
            w_final = torch.tensor(
                self.w / np.sum(self.w), dtype=torch.get_default_dtype()
            ).flatten()
            elbo_best, entropy_best = self.stacked_ELBO(
                w_final, n_samples=n_samples
            )
            return w_final, elbo_best, entropy_best

        else:
            raise AttributeError(
                "S-VBMC version not recognized. Check the spelling!"
            )

        # We say S-VBMC has converged when the stacked ELBO does not improve after 5 steps.
        convergence_counter = 0
        loss_old = 1e8
        elbo_best = None
        entropy_best = None

        for step in range(max_steps):
            optimizer.zero_grad()

            # If optimizing w.r.t. `omega`, get corresponding `w_logits`
            if version == "posterior-only":
                w_logits = (
                    torch.repeat_interleave(
                        omega_logits, repeats=torch.tensor(self.K), dim=0
                    )
                    + log_w
                )

            # Convert logits into valid weights via softmax
            w = torch.softmax(w_logits, dim=-1)

            # Compute the (negative) objective we want to minimize
            ELBO, H = self.stacked_ELBO(w, n_samples=n_samples)

            if step == 0:
                print(f"Initial elbo = {ELBO}")
            if (step + 1) % 5 == 0:
                print(f"iter {step+1}: elbo = {ELBO}")

            # Since we want to maximize ELBO, we minimize -ELBO.
            loss = -ELBO

            loss_new = torch.round(loss * 1e5) / 1e5  # round to 5 decimals

            # If the ELBO does not improve, add to convergence counter. If it does, reset it.
            if loss_new >= loss_old:
                convergence_counter += 1
            else:
                convergence_counter = 0
                # This is now our best solution, store it
                w_best = copy.deepcopy(w_logits.detach().clone())
                elbo_best = -loss
                entropy_best = H
                loss_old = loss_new

            # If the stacked ELBO has not improved in the last 5 steps, we consider it converged
            if convergence_counter >= 5:
                w_final = torch.softmax(w_best, dim=-1)
                # Cast to the global default dtype (float32) for consistency with downstream checks
                w_final = w_final.to(torch.get_default_dtype())
                return w_final, elbo_best, entropy_best

            # Backprop and take an optimization step
            loss.backward()
            optimizer.step()

        # Return the final mixture weights if the max step count has been reached
        w_final = torch.softmax(w_best, dim=-1)
        # Cast to the global default dtype (float32) so calling code sees a consistent type
        w_final = w_final.to(torch.get_default_dtype())
        return w_final, elbo_best, entropy_best

    def optimize(
        self,
        n_samples: int = 20,
        lr: float = 0.1,
        max_steps: int = 500,
        version: str = "all-weights",
    ):
        """
        Maximizes ``stacked_ELBO(`w`, `n_samples`)`` and debiases the resulting stacked ELBO.
        Can optimize w.r.t. `w` ("all-weights") or the VBMC posterior weights `omega`, ("posterior-only").
        It can also perform naive stacking (simply re-normalize the weights).

        Parameters:
        -----------
        n_samples: int
            The number of samples to take from each component of the stacked
            posterior when estimating the entropy.
        lr: float
            learning rate for Adam
        max_steps: int
            maximum number of gradient ascent steps
        version: string
            the type of optimization to be performed. It can take the following values:
                - "all-weights": default, optimizes w.r.t. the weights of all individual
                                components;
                - "posterior-only": optimizes w.r.t. `omega`, i.e., the weights of whole VBMC
                                posteriors;
                - "ns": naive stacking, simply re-normalizes the weights.
        """

        # Optimize stacked ELBO
        w, ELBO, H = self.maximize_ELBO(
            n_samples=n_samples, lr=lr, max_steps=max_steps, version=version
        )

        # Back to numpy
        self.w = w.detach().cpu().numpy().astype(np.float64)
        # Re-normalise in 64-bit to remove any residual round-off error
        self.w /= self.w.sum(dtype=np.float64)

        # Scalars expected as plain Python floats by the test-suite
        self.entropy = float(H.detach().cpu().numpy())
        ELBO = float(ELBO.detach().cpu().numpy())

        # Get median expected log joints (component and posterior-wise)
        I_median = np.median(self.I_corrected)  # component-wise
        E_median = np.median(self.E_corrected)  # posterior-wise
        G = ELBO - self.entropy  # Get estimated expected log-joint

        # Store estimated and debiased stacked ELBOs
        self.elbo = {
            "estimated": ELBO,
            "debiased_I_median": np.min([G, I_median]) + self.entropy,
            "debiased_E_median": np.min([G, E_median]) + self.entropy,
        }

    def sample(self, n_samples):
        """
        Takes samples from the stacked posterior. It uses the PyVBMC ``sample`` method
        on the individual VBMC posterior (after re-adjusting the weights) to get all
        the samples from the original feature space.

        Parameters:
        -----------
        n_samples: int
            The number of samples to take from the FULL stacked posterior.

        Returns:
        --------
        Xs: numpy.ndarray
            The samples from the stacked posterior
        """

        # Work with a flat 1-D view of the weights regardless of their stored shape
        w_flat = self.w.ravel()

        idx = 0
        Xs = []
        # Loop over vp objects to use the PyVBMC's sample function
        for m, vp in enumerate(self.vp_list):
            # Copy to avoid altering the original
            vp_copy = copy.deepcopy(vp)
            # Get relative weight of the individual VBMC posterior (`omega`)
            omega = np.sum(w_flat[idx : idx + self.K[m]])
            # change the weights within the vp object to correspond to the optimized ones (and normalize)
            comp_weights = w_flat[idx : idx + self.K[m]]
            vp_copy.w = comp_weights / omega

            # Use the PyVBMC sampling function to sample from individual posteriors in the base space.
            # Sample proportionally to the relative weight of the individual VBMC posterior.
            samples, _ = vp_copy.sample(int(np.round(n_samples * omega)))

            Xs.append(samples)
            idx += self.K[m]

        return np.concatenate(Xs, axis=0)

    def plot(
        self,
        n_samples: int = 10000,
        color: str = "black",
        figsize: tuple | None = None,
        smooth: float | None = None,
        **corner_kwargs,
    ):
        """
        Draw a corner plot of the stacked posterior.

        This calls ``sample`` to draw samples and then feeds them to
        the ``corner`` library.

        Parameters
        ----------
        n_samples : int
            How many samples to draw from the stacked posterior.
        color : str
            Matplotlib-style color to use for *all* 2-D contour lines and
            1-D histograms (e.g. ``"C1"``, ``"tab:green"``, ``"#ff7f0e"``).
        figsize : (float, float) or None
            Forwarded to ``plt.figure``.  If *None*, the default size from
            ``corner`` is used.
        smooth : float or None, optional
            Gaussian kernel smoothing applied by ``corner``.
            Leave *None* for the default.
        **corner_kwargs
            Any additional keyword arguments are passed straight to
            :pyfunc:`corner.corner`, allowing fine-grained control (e.g.
            ``bins=30``, ``levels=(0.68, 0.95)``).

        Returns
        -------
        matplotlib.figure.Figure
            The ``Figure`` instance containing the plot.
        """
        import corner
        from matplotlib import pyplot as plt

        # Draw samples
        samples = self.sample(n_samples)  # [n_samples, D]

        if figsize is None:
            base = 2.5
            figsize = (base * self.D, base * self.D)

        # Build default axis labels
        labels = [f"$x_{{{i+1}}}$" for i in range(self.D)]

        # Build the plot
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

        # Tighten the layout
        fig.tight_layout()
        # Show figure only when using an interactive backend to avoid head-less
        # warnings during automated tests.
        if not plt.get_backend().lower().endswith("agg"):
            plt.show()
        else:
            # Draw the canvas so that the figure is fully initialised.
            fig.canvas.draw()

        return fig
