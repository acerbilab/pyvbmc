"""Variational optimization / training of variational posterior"""

import math
import numpy as np

def update_K(optim_state, iteration_history, options):
    """
    Update number of variational mixture components.
    """
    K_new = optim_state["vpK"]

    # Compute maximum number of components
    K_max = math.ceil(
        options.eval(
            "kfunmax", {"N": optim_state["n_eff"]}
        )
    )

    # Evaluate bonus for stable solution.
    K_bonus = round(options.eval("adaptivek", {"unkn" : K_new}))

    if not optim_state["warmup"] and optim_state["iter"] > 0:
        recent_iters = math.ceil(
            0.5
            * options["tolstablecount"]
            / options["funevalsperiter"]
        )

        # Check if ELCBO has improved wrt recent iterations
        lower_end = max(0, optim_state["iter"] - recent_iters)
        elbos = iteration_history["elbo"][lower_end:]
        elboSDs = iteration_history["elbo_sd"][lower_end:]
        elcbos = elbos - options["elcboimproweight"] * elboSDs
        warmups = iteration_history["warmup"][lower_end:]
        elcbos_after = elcbos[~warmups]
        # Ignore two iterations right after warmup.
        elcbos_after[0 : min(2, optim_state["iter"])] = -np.inf
        elcbo_max = np.max(elcbos_after)
        improving_flag = elcbos_after[-1] >= elcbo_max and np.isfinite(
            elcbos_after[-1]
        )

        # Add one component if ELCBO is improving and no pruning in last iteration
        if iteration_history["pruned"][-1] == 0 and improving_flag:
            K_new += 1

        # Bonus components for stable solution (speed up exploration)
        if (
            iteration_history["rindex"][-1] < 1
            and not optim_state["recompute_var_post"]
            and improving_flag
        ):
            # No bonus if any component was very recently pruned.
            new_lower_end = max(
                0, optim_state["iter"] - math.ceil(0.5 * recent_iters)
            )
            if np.all(
                iteration_history["pruned"][new_lower_end:] == 0
            ):
                K_new += K_bonus

        K_new = max(optim_state["vpK"], min(K_new, K_max))

    return K_new

def optimize_vp(vp, gp, Nfastopts, Nslowopts, K=None):
    """
    Optimize variational posterior.
    """
    # use interface for vp optimzation?
    if K is None:
        K = vp.K
    varss = []
    pruned = 0
    return varss, pruned
    
def _vp_bound_loss(
    vp, theta, theta_bnd, tol_con=1e-3, compute_grad=True
):
    """
    Variational paramtere loss function for soft optimization bounds.
    """

    if vp.optimize_mu:
        mu = theta[: vp.D * vp.K]
        start_idx = vp.D * vp.K
    else:
        mu = vp.mu.flatten()
        start_idx = 0

    if vp.optimize_sigma:
        ln_sigma = theta[start_idx : start_idx + vp.K]
        start_idx += vp.K
    else:
        ln_sigma = np.log(vp.sigma.flatten())

    if vp.optimize_lambd:
        ln_lambd = theta[start_idx : start_idx + vp.D].T
    else:
        ln_lambd = np.log(vp.lambd.flatten())

    if vp.optimize_weights:
        eta = theta[-vp.K :]
    else:
        eta = None

    ln_scale = np.reshape(ln_lambd, (-1, 1)) + np.reshape(
        ln_sigma, (1, -1)
    )
    theta_ext = []
    if vp.optimize_mu:
        theta_ext.append(mu.flatten())
    if vp.optimize_sigma or vp.optimize_lambda:
        theta_ext.append(ln_scale.flatten())
    if vp.optimize_weights:
        theta_ext.append(eta.flatten())
    theta_ext = np.concatenate(theta_ext)

    if compute_grad:
        L, dL = _soft_bound_loss(
            theta_ext,
            theta_bnd["lb"].flatten(),
            theta_bnd["ub"].flatten(),
            tol_con,
            compute_grad=True,
        )

        dL_new = np.array([])
        if vp.optimize_mu:
            dL_new = np.concatenate(
                (dL_new, dL[0 : vp.D * vp.K].flatten())
            )
            start_idx = vp.D * vp.K
        else:
            start_idx = 0

        if vp.optimize_sigma or vp.optimize_lambda:
            dlnscale = np.reshape(
                dL[start_idx : start_idx + vp.D * vp.K], (vp.D, vp.K)
            )

            if vp.optimize_sigma:
                dL_new = np.concatenate((dL_new, np.sum(dlnscale, axis=0)))

            if vp.optimize_lambda:
                dL_new = np.concatenate((dL_new, np.sum(dlnscale, axis=1)))

        if vp.optimize_weights:
            dL_new = np.concatenate((dL_new, dL[-vp.K :].flatten()))

        return L, dL_new

    L = _soft_bound_loss(
        theta_ext,
        theta_bnd["lb"].flatten(),
        theta_bnd["ub"].flatten(),
        tol_con,
    )
    
    return L

def _soft_bound_loss(x, slb, sub, tol_con=1e-3, compute_grad=False):
    """
    Loss function for soft bounds for function minimization.
    """
    ell = (sub - slb) * tol_con
    y = 0
    dy = np.zeros(x.shape)

    idx = x < slb
    if np.any(idx):
        y += 0.5 * np.sum((slb[idx] - x[idx]) / ell[idx] ** 2)
        if compute_grad:
            dy[idx] = (x[idx] - slb[idx]) / ell[idx] ** 2

    idx = x > sub
    if np.any(idx):
        y += 0.5 * np.sum((x[idx] - sub[idx]) / ell[idx] ** 2)
        if compute_grad:
            dy[idx] = (x[idx] - sub[idx]) / ell[idx] ** 2

    if compute_grad:
        return y, dy
    return y
