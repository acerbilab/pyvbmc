"""Variational optimization / training of variational posterior"""

import math
import copy
import numpy as np
import scipy as sp
import cma

import gpyreg as gpr

from pyvbmc.entropy import entlb_vbmc, entmc_vbmc

from .gaussian_process_train import _get_hpd
from .minimize_adam import minimize_adam


def update_K(optim_state, iteration_history, options):
    """
    Update number of variational mixture components.
    """
    K_new = optim_state["vpK"]

    # Compute maximum number of components
    K_max = math.ceil(options.eval("kfunmax", {"N": optim_state["n_eff"]}))

    # Evaluate bonus for stable solution.
    K_bonus = round(options.eval("adaptivek", {"unkn": K_new}))

    if not optim_state["warmup"] and optim_state["iter"] > 0:
        recent_iters = math.ceil(
            0.5 * options["tolstablecount"] / options["funevalsperiter"]
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
            if np.all(iteration_history["pruned"][new_lower_end:] == 0):
                K_new += K_bonus

        K_new = max(optim_state["vpK"], min(K_new, K_max))

    return K_new


def optimize_vp(
    options, optim_state, K_orig, vp, gp, fast_opts_N, slow_opts_N, K=None, prnt=0
):
    """
    Optimize variational posterior.
    """

    if K is None:
        K = vp.K

    # Quick sieve optimization to determine starting point(s)
    vp0_vec, vp0_type, elcbo_beta, compute_var, nsent_K, _ = _sieve(
        options,
        optim_state,
        K_orig,
        vp,
        gp,
        K=K,
        init_N=fast_opts_N,
        best_N=slow_opts_N,
    )

    # Compute soft bounds for variational parameter optimization.
    theta_bnd = vp.get_bounds(gp.X, options, K)

    ## Perform optimization starting from one or few selected points.
    
    theta_N = np.size(vp0_vec[0].get_parameters())
    Ns = np.size(gp.posteriors)
    elbo_stats = _initialize_full_elcbo(slow_opts_N * 2, theta_N, K, Ns)

    # For the moment no gradient available for variance
    gradient_available = compute_var == 0

    vbtrain_options = {}
    if gradient_available:
        # Set basic options for deterministic (?) optimizer
        # TODO: options for optimizer here
        compute_grad = True
    else:
        # TODO: options for optimizer here
        options["stochasticoptimizer"] = "cmaes"
        compute_grad = False

    vp0_fine = {}
    for i in range(0, slow_opts_N):
        # Be careful with off-by-one errors here, Python is zero-based.
        i_mid = (i + 1) * 2 - 2
        i_end = (i + 1) * 2 - 1

        # Select points from best ones depending on subset
        if slow_opts_N == 1:
            idx = 0
        elif slow_opts_N == 2:
            if i == 0:
                idx = np.where(vp0_type == 1)[0][0]
            else:
                idx = np.where((vp0_type == 2) | (vp0_type == 3))[0][0]
        else:
            idx = np.where(vp0_type == math.remainder(i, 3) + 1)[0][0]

        vp0 = vp0_vec[idx]
        vp0_vec = np.delete(vp0_vec, idx)
        vp0_type = np.delete(vp0_type, idx)
        theta0 = vp0.get_parameters()

        # Objective function, should only return value and gradient.
        def vbtrain_mc_fun(theta_):
            res = _negelcbo(
                theta_,
                gp,
                vp0,
                K_orig,
                elcbo_beta,
                nsent_K,
                compute_grad=True,
                compute_var=compute_var,
                theta_bnd=theta_bnd,
            )
            return res[0], res[1]

        if nsent_K == 0:
            # Fast optimization via deterministic entropy approximation

            # Objective function
            def vbtrain_fun(theta_):
                res = _negelcbo(
                    theta_,
                    gp,
                    vp0,
                    K_orig,
                    elcbo_beta,
                    0,
                    compute_grad=compute_grad,
                    compute_var=compute_var,
                    theta_bnd=theta_bnd,
                )
                if compute_grad:
                    return res[0], res[1]
                return res[0]
            
            res = sp.optimize.minimize(vbtrain_fun, theta0, jac=compute_grad, tol=options["detenttolopt"])
            if not res.success:
                # SciPy minimize failed, try with CMA-ES
                if prnt >= 0:
                    print("Cannot optimize variational parameters with scipy.optimize.minimize. Trying with CMA-ES (slower).")

                # Objective function with no gradient computation.
                def vbtrain_fun(theta_):
                    res = _negelcbo(
                        theta_,
                        gp,
                        vp0,
                        K_orig,
                        elcbo_beta,
                        0,
                        compute_grad=False,
                        compute_var=compute_var,
                        theta_bnd=theta_bnd,
                    )
                    return res[0]
                    
                insigma_list = []
                if vp.optimize_mu:
                    insigma_list.append(np.tile(vp.bounds["mu_ub"] - vp.bounds["mu_lb"], (K,)))
                if vp.optimize_sigma:
                    insigma_list.append(np.ones((K,)))
                if vp.optimize_lambd:
                    insigma_list.append(np.ones((vp.D,)))
                if vp.optimize_weights:
                    insigma_list.append(np.ones((K,)))
                insigma = np.concatenate(insigma_list)
                
                cma_options = {
                    "verbose": -9,
                    "tolx": 1e-8 * np.max(insigma),
                    "tolfun": 1e-6,
                    "tolfunhist": 1e-7,
                    "maxfevals": 200*(vp.D+2),
                    "CMA_active": True,
                }
                res = cma.fmin(vbtrain_fun, theta0, np.max(insigma), options=cma_options)
                theta_opt = res[0]
            else:
                theta_opt = res.x
        else:
            theta_opt = theta0

            if options["stochasticoptimizer"] == "adam":
                master_min = min(options["sgdstepsize"], 0.001)
                if optim_state["warmup"] or not vp.optimize_weights:
                    scaling_factor = min(0.1, options["sgdstepsize"] * 10)
                else:
                    scaling_factor = min(0.1, options["sgdstepsize"])

                # Fixed master stepsize
                master_max = scaling_factor

                # Note: we tried to adapt the stepsizes guided by the GP
                # hyperparameters, but this did not seem to help (the former
                # experimental option was "GPStochasticStepsize").
                master_max = max(master_min, master_max)
                master_decay = 200
                max_iter = min(10000, options["maxiterstochastic"])
                theta_opt, _, theta_lst, f_val_lst, n_iters = minimize_adam(
                    vbtrain_mc_fun,
                    theta_opt,
                    tol_fun=options["tolfunstochastic"],
                    max_iter=max_iter,
                    master_min=master_min,
                    master_max=master_max,
                    master_decay=master_decay,
                )

                if options["elcbomidpoint"]:
                    # Recompute ELCBO at best midpoint with full variance and more precision.
                    idx_mid = np.argmin(f_val_lst)
                    elbo_stats = _eval_full_elcbo(
                        i_mid,
                        theta_lst[:, idx_mid],
                        vp0,
                        gp,
                        elbo_stats,
                        elcbo_beta,
                        options,
                        K_orig,
                    )
            elif options["stochasticoptimizer"] == "cmaes":
                # Objective function with no gradient computation.
                def vbtrain_fun(theta_):
                    res = _negelcbo(
                        theta_,
                        gp,
                        vp0,
                        K_orig,
                        elcbo_beta,
                        Ns=nsent_K,
                        compute_grad=False,
                        compute_var=compute_var,
                        theta_bnd=theta_bnd,
                    )
                    return res[0]
                    
                insigma_list = []
                if vp.optimize_mu:
                    insigma_list.append(np.tile(vp.bounds["mu_ub"] - vp.bounds["mu_lb"], (K,)))
                if vp.optimize_sigma:
                    insigma_list.append(np.ones((K,)))
                if vp.optimize_lambd:
                    insigma_list.append(np.ones((vp.D,)))
                if vp.optimize_weights:
                    insigma_list.append(np.ones((K,)))
                insigma = np.concatenate(insigma_list)
                
                cma_options = {
                    "verbose": -9,
                    "tolx": 1e-6 * np.max(insigma),
                    "tolfun": 1e-4,
                    "tolfunhist": 1e-5,
                    "CMA_active": True,
                }
                res = cma.fmin(vbtrain_fun, theta0, np.max(insigma), options=cma_options, noise_handler=cma.NoiseHandler(np.size(theta0)))
                theta_opt = res[0]
            else:
                raise Exception("Unknown stochastic optimizer!")

        # Recompute ELCBO at endpoint with full variance and more precision
        elbo_stats = _eval_full_elcbo(
            i_end, theta_opt, vp0, gp, elbo_stats, elcbo_beta, options, K_orig
        )
        
        vp0_fine[i_mid] = copy.deepcopy(vp0)
        vp0_fine[i_end] = copy.deepcopy(vp0)  # Parameters get assigned later

    ## Finalize optimization by taking variational parameters with best ELCBO
    
    idx = np.argmin(elbo_stats["nelcbo"])
    elbo = -elbo_stats["nelbo"][idx]
    elbo_sd = np.sqrt(elbo_stats["varF"][idx])
    G = elbo_stats["G"][idx]
    H = elbo_stats["H"][idx]
    varss = elbo_stats["varss"][idx]
    varG = elbo_stats["varG"][idx]
    varH = elbo_stats["varH"][idx]
    I_sk = np.zeros((Ns, K))
    J_sjk = np.zeros((Ns, K, K))
    I_sk[:, :] = elbo_stats["I_sk"][idx, :, :]
    J_sjk[:, :, :] = elbo_stats["J_sjk"][idx, :, :, :]
    vp = vp0_fine[idx]
    # vp = rescale_params(vp,elbostats.theta(idx,:))

    ## Potentionally prune mixture components
    pruned = 0
    if vp.optimize_weights:
        already_checked = np.full((vp.K,), False)

        while np.any((vp.w < options["tolweight"]) & ~already_checked):
            vp_pruned = copy.deepcopy(vp)

            # Choose a random component below threshold
            idx = np.argwhere(
                (vp.w < options["tolweight"]) & ~already_checked
            ).flatten()
            idx = idx[np.random.randint(0, np.size(idx))]
            vp_pruned.w = np.delete(vp_pruned.w, idx)
            vp_pruned.eta = np.delete(vp_pruned.eta, idx)
            vp_pruned.sigma = np.delete(vp_pruned.sigma, idx)
            vp_pruned.mu = np.delete(vp_pruned.mu, idx, axis=1)
            vp_pruned.K -= 1
            theta_pruned = vp_pruned.get_parameters()

            # Recompute ELCBO
            elbo_stats = _eval_full_elcbo(
                0,
                theta_pruned,
                vp_pruned,
                gp,
                elbo_stats,
                elcbo_beta,
                options,
                K_orig,
            )
            elbo_pruned = -elbo_stats["nelbo"][0]
            elbo_pruned_sd = np.sqrt(elbo_stats["varF"][0])

            # Difference in ELCBO (before and after pruning)
            delta_elcbo = np.abs(
                (elbo_pruned - options["elcboimproweight"] * elbo_pruned_sd)
                - (elbo - options["elcboimproweight"] * elbo_sd)
            )

            # Prune component if it has neglible influence on ELCBO
            pruning_threshold = options["tolimprovement"] * options.eval(
                "pruningthresholdmultiplier", {"K": K}
            )

            if delta_elcbo < pruning_threshold:
                vp = vp_pruned
                elbo = elbo_pruned
                elbo_sd = elbo_pruned_sd
                G = elbo_stats["G"][0]
                H = elbo_stats["H"][0]
                varss = elbo_stats["varss"][0]
                varG = elbo_stats["varG"][0]
                varH = elbo_stats["varH"][0]
                pruned += 1
                already_checked = np.delete(already_checked, idx)
                I_sk = np.delete(I_sk, idx, axis=1)
                J_sjk = np.delete(J_sjk, idx, axis=2)
            else:
                already_checked[idx] = True

    vp.stats = {}
    vp.stats["elbo"] = elbo  # ELBO
    vp.stats["elbo_sd"] = elbo_sd  # Error on the ELBO
    vp.stats["elogjoint"] = G  # Expected log joint
    vp.stats["elogjoint_sd"] = np.sqrt(varG)  # Error on expected log joint
    vp.stats["entropy"] = H  # Entropy
    vp.stats["entropy_sd"] = np.sqrt(varH)  # Error on the entropy
    vp.stats["stable"] = False  # Unstable until proven otherwise
    vp.stats["I_sk"] = I_sk  # Expected log joint per component
    vp.stats["J_sjk"] = J_sjk  # Covariance of expected log joint

    return vp, varss, pruned


def _initialize_full_elcbo(idx, D, K, Ns):
    elbo_stats = {}
    elbo_stats["nelbo"] = np.full((idx,), np.inf)
    elbo_stats["G"] = np.full((idx,), np.nan)
    elbo_stats["H"] = np.full((idx,), np.nan)
    elbo_stats["varF"] = np.full((idx,), np.nan)
    elbo_stats["varG"] = np.full((idx,), np.nan)
    elbo_stats["varH"] = np.full((idx,), np.nan)
    elbo_stats["varss"] = np.full((idx,), np.nan)
    elbo_stats["nelcbo"] = np.full((idx,), np.inf)
    elbo_stats["theta"] = np.full((idx, D), np.nan)
    elbo_stats["I_sk"] = np.full((idx, Ns, K), np.nan)
    elbo_stats["J_sjk"] = np.full((idx, Ns, K, K), np.nan)
    return elbo_stats


def _eval_full_elcbo(
    idx, theta, vp, gp, elbo_stats, beta, options, K_orig, entropy_alpha=0
):
    # Number of samples per component for MC approximation of the entropy.
    K = vp.K
    nsent_fine_K = math.ceil(options.eval("nsentfine", {"K": K}) / K)

    if "skipelbovariance" in options and options["skipelbovariance"]:
        compute_var = False
    else:
        compute_var = True

    nelbo, _, G, H, varF, _, varss, varG, varH, I_sk, J_sjk = _negelcbo(
        theta,
        gp,
        vp,
        K_orig,
        0,
        nsent_fine_K,
        False,
        compute_var,
        None,
        entropy_alpha,
        True,
    )
    nelcbo = nelbo + beta * np.sqrt(varF)

    elbo_stats["nelbo"][idx] = nelbo
    elbo_stats["G"][idx] = G
    elbo_stats["H"][idx] = H
    elbo_stats["varF"][idx] = varF
    elbo_stats["varG"][idx] = varG
    elbo_stats["varH"][idx] = varH
    elbo_stats["varss"][idx] = varss
    elbo_stats["nelcbo"][idx] = nelcbo
    elbo_stats["theta"][idx, 0 : np.size(theta)] = theta
    elbo_stats["I_sk"][idx, :, 0:K] = I_sk
    elbo_stats["J_sjk"][idx, :, 0:K, 0:K] = J_sjk

    return elbo_stats


def _vp_bound_loss(vp, theta, theta_bnd, tol_con=1e-3, compute_grad=True):
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

    ln_scale = np.reshape(ln_lambd, (-1, 1)) + np.reshape(ln_sigma, (1, -1))
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
            dL_new = np.concatenate((dL_new, dL[0 : vp.D * vp.K].flatten()))
            start_idx = vp.D * vp.K
        else:
            start_idx = 0

        if vp.optimize_sigma or vp.optimize_lambda:
            dlnscale = np.reshape(
                dL[start_idx : start_idx + vp.D * vp.K], (vp.D, vp.K)
            )

            if vp.optimize_sigma:
                dL_new = np.concatenate((dL_new, np.sum(dlnscale, axis=0)))

            if vp.optimize_lambd:
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


def _sieve(
    options, optim_state, K_orig, vp, gp, init_N=None, best_N=1, K=None
):
    """
    Preliminary 'sieve' method for fitting variational posterior.
    """
    if K is None:
        K = vp.K

    # Missing port: assign default values to optim_state (since
    #               this doesn't seem to be necessary)

    ## Set up optimization variables and options.

    # TODO: delta is wrong size, since plb and pub are wrong size
    #       which are the wrong size since plausible upper and lower
    #       bounds are wrong size which are the wrong size since?
    #       everything runs but could maybe be simpler?
    vp.delta = optim_state["delta"]

    # Number of initial starting points
    if init_N is None:
        init_N = math.ceil(options.eval("nselbo", {"K": K}))
    nelcbo_fill = np.zeros((init_N,))

    # Number of samples per component for MC approximation of the entropy.
    nsent_K = math.ceil(options.eval("nsent", {"K": K}) / K)

    # Number of samples per component for preliminary MC approximation of the entropy.
    nsent_K_fast = math.ceil(options.eval("nsentfast", {"unknown": K}) / K)

    # Deterministic entropy if entropy switch is on or only one component
    if optim_state["entropy_switch"] or K == 1:
        nsent_K = 0
        nsent_K_fast = 0

    # Confidence weight
    # Missing port: elcboweight does not exist
    # elcbo_beta = self._eval_option(self.options["elcboweight"], self.optim_state["n_eff"])
    elcbo_beta = 0
    compute_var = elcbo_beta != 0

    # Compute soft bounds for variational parameter optimization
    theta_bnd = vp.get_bounds(gp.X, options, K)

    ## Perform quick shotgun evaluation of many candidate parameters

    if init_N > 0:
        # Get high-posterior density points
        X_star, y_star, _ = _get_hpd(gp.X, gp.y, options["hpdfrac"])

        # Generate a bunch of random candidate variational parameters.
        if best_N == 1:
            vp0_vec, vp0_type = _vbinit(vp, 1, init_N, K, X_star, y_star)
        else:
            vp0_vec1, vp0_type1 = _vbinit(
                vp, 1, math.ceil(init_N / 3), K, X_star, y_star
            )
            vp0_vec2, vp0_type2 = _vbinit(
                vp, 2, math.ceil(init_N / 3), K, X_star, y_star
            )
            vp0_vec3, vp0_type3 = _vbinit(
                vp, 3, init_N - 2 * math.ceil(init_N / 3), K, X_star, y_star
            )
            vp0_vec = np.concatenate([vp0_vec1, vp0_vec2, vp0_vec3])
            vp0_type = np.concatenate([vp0_type1, vp0_type2, vp0_type3])

        if len(optim_state["vp_repo"]) > 0 and options["variationalinitrepo"]:
            theta_N = np.size(vp0_vec[0].get_parameters())
            idx = np.where(
                [np.size(item) for item in optim_state["vp_repo"]] == theta_N
            )[0]
            if np.size(idx) > 0:
                idx = [np.size(item) for item in optim_state["vp_repo"]].index(
                    theta_N
                )
                vp0_list4 = []
                for idx_i in idx:
                    vp_new = copy.deepcopy(vp0_list[0])
                    vp_new.set_parameters(optim_state["vp_repo"][idx_i])
                    vp0_list4.append(vp_new)
                vp0_vec4 = np.array(vp0_list4)
                vp0_vec = np.concatenate([vp0_vec, vp0_vec4])
                vp0_type = np.concatenate(
                    [vp0_type, np.ones((len(vp0_list4),))]
                )

        # Quickly estimate ELCBO at each candidate variational posterior.
        for i, vp in enumerate(vp0_vec):
            theta = vp.get_parameters()
            nelbo_tmp, _, _, _, varF_tmp = _negelcbo(
                theta,
                gp,
                vp,
                K_orig,
                0,
                nsent_K_fast,
                0,
                compute_var,
                theta_bnd,
            )
            nelcbo_fill[i] = nelbo_tmp + elcbo_beta * np.sqrt(varF_tmp)

        # Sort by negative ELCBO
        order = np.argsort(nelcbo_fill)
        vp0_vec = vp0_vec[order]
        vp0_type = vp0_type[order]

        return (
            vp0_vec,
            vp0_type,
            elcbo_beta,
            compute_var,
            nsent_K,
            nsent_K_fast,
        )

    return copy.deepcopy(vp), 1, elcbo_beta, compute_var, nsent_K, nsent_K_fast


def _vbinit(vp, vbtype, opts_N, K_new, X_star, y_star):
    """
    Generate array of random starting parameters for variational posterior.
    X_star and y_star are usually HPD regions.
    """

    D = vp.D
    K = vp.K
    N_star = X_star.shape[0]
    add_jitter = True
    type_vec = vbtype * np.ones((opts_N))
    lambd0 = vp.lambd.copy()
    mu0 = vp.mu.copy()
    w0 = vp.w.copy()

    if vbtype == 1:
        # Start from old variational parameters
        sigma0 = vp.sigma
    elif vbtype == 2:
        # Start from highest-posterior density training points
        if vp.optimize_mu:
            order = np.argsort(y_star, axis=None)[::-1]
            idx_order = np.tile(
                range(0, min(K_new, N_star)), (math.ceil(K_new / N_star),)
            )
            mu0 = X_star[order[idx_order[0:K_new]], :].T
        if K > 1:
            V = np.var(mu0, axis=1, ddof=1)
        else:
            V = np.var(X_star, axis=0, ddof=1)
        sigma0 = np.sqrt(np.mean(V / lambd0 ** 2) / K_new) * np.exp(
            0.2 * np.random.randn(1, K_new)
        )
    else:
        # Start from random provided training points.
        if vp.optimize_mu:
            mu0 = np.zeros((D, K))
        sigma0 = np.zeros((1, K))

    vp0_list = []
    for i in range(0, opts_N):
        mu = mu0.copy()
        sigma = sigma0.copy()
        lambd = lambd0.copy()
        if vp.optimize_weights:
            w = w0.copy()

        if vbtype == 1:
            # Start from old variation parameters

            # Copy previous parameters verbatim.
            if i == 0:
                add_jitter = False

            if K_new > vp.K:
                # Spawn a new component near an existing one
                for i_new in range(K, K_new):
                    idx = np.random.randint(0, K)
                    mu[:, i_new] = mu[:, idx]
                    sigma[i_new] = sigma[idx]
                    mu[:, i_new] += (
                        0.5 * sigma[i_new] * lambd * np.random.randn(D, 1)
                    )

                    if vp.optimize_sigma:
                        sigma[i_new] *= np.exp(0.2 * np.random.randn())

                        if vp.optimize_weights:
                            xi = 0.25 + 0.25 * np.random.rand()
                            w[i_new] = xi * w[idx]
                            w[idx] *= 1 - xi
        elif vbtype == 2:
            if i == 0:
                add_jitter = False
            if vp.optimize_lambd:
                lambd = np.reshape(np.std(X_star, axis=0, ddof=1), (-1, 1))
                lambd *= np.sqrt(D / np.sum(lambd ** 2))
            if vp.optimize_weights:
                w = np.ones((1, K_new)) / K_new
        elif vbtype == 3:
            if vp.optimize_mu:
                order = np.random.permutation(N_star)
                idx_order = np.tile(
                    range(0, min(K_new, N_star)),
                    (math.ceil(K_new / N_star),),
                )
                mu = X_star[order[idx_order[0:K_new]], :].T
            else:
                mu = mu0.copy()

            if vp.optimize_sigma:
                if K > 1:
                    V = np.var(mu, axis=1, ddof=1)
                else:
                    V = np.var(X_star, axis=0, ddof=1)
                sigma = np.sqrt(np.mean(V) / K_new) * np.exp(
                    0.2 * np.random.randn(1, K_new)
                )

            if vp.optimize_lambd:
                lambd = np.reshape(np.std(X_star, axis=0, ddof=1), (-1, 1))
                lambd *= np.sqrt(D / np.sum(lambd ** 2))

            if vp.optimize_weights:
                w = np.ones((1, K_new)) / K_new
        else:
            raise Exception("Unsupported type!")

        if add_jitter:
            if vp.optimize_mu:
                mu += sigma * lambd * np.random.standard_normal(mu.shape)
            if vp.optimize_sigma:
                sigma *= np.exp(0.2 * np.random.randn(1, K_new))
            if vp.optimize_lambd:
                lambd *= np.exp(0.2 * np.random.randn(D, 1))
            if vp.optimize_weights:
                w *= np.exp(0.2 * np.random.randn(1, K_new))
                w /= np.sum(w)

        new_vp = copy.deepcopy(vp)
        new_vp.K = K_new

        if vp.optimize_weights:
            new_vp.w = w
        else:
            new_vp.w = np.ones((1, K_new)) / K_new
        if vp.optimize_mu:
            new_vp.mu = mu
        else:
            new_vp.mu = mu0.copy()
        new_vp.sigma = sigma
        new_vp.lambd = lambd
        # TODO: are these right?
        new_vp.eta = np.ones((1, K_new)) / K_new
        new_vp.bounds = None
        new_vp.stats = None

        vp0_list.append(new_vp)

    return np.array(vp0_list), type_vec


def _negelcbo(
    theta,
    gp,
    vp,
    K_orig,
    beta=0,
    Ns=0,
    compute_grad=True,
    compute_var=None,
    theta_bnd=None,
    entropy_alpha=0,
    separate_K=False,
):
    """
    Expected variational log joint probability via GP approximation.
    """
    if not np.isfinite(beta):
        beta = 0
    if compute_var is None:
        compute_var = beta != 0

    if compute_grad and beta != 0 and compute_var != 2:
        raise Exception(
            "Computation of the gradient of ELBO with full variance not supported"
        )

    D = vp.D
    K = vp.K

    # Average over multiple GP hyperparameters if provided
    avg_flag = 1
    # Variational parameters are transformed
    jacobian_flag = 1

    # Reformat variational parameters from theta.
    if vp.optimize_mu:
        vp.mu = np.reshape(theta[: D * K], (D, K))
        start_idx = D * K
    else:
        start_idx = 0

    if vp.optimize_sigma:
        vp.sigma = np.exp(theta[start_idx : start_idx + K])
        start_idx += K_orig

    if vp.optimize_lambd:
        vp.lambd = np.exp(theta[start_idx : start_idx + D]).T

    if vp.optimize_weights:
        eta = theta[-K_orig:]
        eta = eta - np.amax(eta)
        vp.w = np.exp(eta.T)[np.newaxis, :]

    # Which gradients should be computed, if any?
    if compute_grad:
        grad_flags = (
            vp.optimize_mu,
            vp.optimize_sigma,
            vp.optimize_lambd,
            vp.optimize_weights,
        )
    else:
        grad_flags = (False, False, False, False)

    # Only weight optimization?
    onlyweights_flag = (
        vp.optimize_weights
        and not vp.optimize_mu
        and not vp.optimize_sigma
        and not vp.optimize_lambd
    )

    if separate_K:
        if compute_grad:
            raise Exception(
                "Computing the gradient of variational parameters and requesting per-component results at the same time."
            )

        if onlyweights_flag:
            if compute_var:
                assert False
            else:
                assert False
            varGss = np.nan
        else:
            if compute_var:
                G, _, varG, _, varGss, I_sk, J_sjk = _gplogjoint(
                    vp,
                    gp,
                    grad_flags,
                    avg_flag,
                    jacobian_flag,
                    compute_var,
                    True,
                )
            else:
                G, dG, _, _, _, I_sk, _ = _gplogjoint(
                    vp, gp, grad_flags, avg_flag, jacobian_flag, 0, True
                )
                varG = varGss = 0
                J_jsk = None
    else:
        if onlyweights_flag:
            if compute_var:
                if compute_grad:
                    assert False
                else:
                    assert False
            else:
                assert False
            varGss = np.nan
        else:
            if compute_var:
                if compute_grad:
                    G, dG, varG, dvarG, varGss = _gplogjoint(
                        vp,
                        gp,
                        grad_flags,
                        avg_flag,
                        jacobian_flag,
                        compute_var,
                    )
                else:
                    G, _, varG, _, varGss = _gplogjoint(
                        vp,
                        gp,
                        grad_flags,
                        avg_flag,
                        jacobian_flag,
                        compute_var,
                    )
            else:
                G, dG, _, _, _ = _gplogjoint(
                    vp, gp, grad_flags, avg_flag, jacobian_flag, 0
                )
                varG = varGss = 0

    # Entropy term
    if Ns > 0:
        # Monte carlo approximation
        H, dH = entmc_vbmc(vp, Ns, grad_flags, jacobian_flag)
    else:
        # Deterministic approximation via lower bound on the entropy
        H, dH = entlb_vbmc(vp, grad_flags, jacobian_flag)

    # Negative ELBO and its gradient
    F = -G - H
    if compute_grad:
        dF = -dG - dH
    else:
        dF = None

    # For the moment use zero variance for entropy
    varH = 0
    if compute_var:
        varF = varG + varH
    else:
        varF = 0

    # Negative ELCBO (add confidence bound)
    if beta != 0:
        F += beta * np.sqrt(varF)
        if compute_grad:
            dF += 0.5 * beta * dvarG / np.sqrt(varF)

    # Additional loss for variational parameter bound violation (soft obunds)
    # and for weight size (if optimizing mixture weights)
    # Only done when optimizing the variational parameters, but not when
    # computing the EL(C)BO at each iteration.
    if theta_bnd is not None:
        if compute_grad:
            L, dL = _vp_bound_loss(
                vp, theta, theta_bnd, tol_con=theta_bnd["tol_con"]
            )
            dF += dL
        else:
            L = _vp_bound_loss(
                vp,
                theta,
                theta_bnd,
                tol_con=theta_bnd["tol_con"],
                compute_grad=False,
            )
        F += L

        #  Penalty to reduce weight size.
        if vp.optimize_weights:
            thresh = theta_bnd["weight_threshold"]
            L = (
                np.sum(vp.w * (vp.w < thresh) + thresh * (vp.w >= thresh))
                * theta_bnd["weight_penalty"]
            )
            F += L
            if compute_grad:
                w_grad = theta_bnd["weight_penalty"] * (
                    vp.w.flatten() < thresh
                )
                eta_sum = np.sum(np.exp(vp.eta))
                J_w = (
                    -np.exp(vp.eta).T * np.exp(vp.eta) / eta_sum ** 2
                ) + np.diag(np.exp(vp.eta.flatten()) / eta_sum)
                w_grad = np.dot(J_w, w_grad)
                dL = np.zeros(dF.shape)
                dL[-vp.K :] = w_grad
                dF += dL

    if separate_K:
        return F, dF, G, H, varF, dH, varGss, varG, varH, I_sk, J_sjk
    else:
        return F, dF, G, H, varF


def _gplogjoint(
    vp,
    gp,
    grad_flags,
    avg_flag=True,
    jacobian_flag=True,
    compute_var=None,
    separate_K=False,
):
    if np.isscalar(grad_flags):
        if grad_flags:
            grad_flags = (True, True, True, True)
        else:
            grad_flags = (False, False, False, False)

    compute_vargrad = compute_var and np.any(grad_flags)
    if compute_vargrad and compute_var != 2:
        raise Exception(
            "Computation of gradient of log joint variance is currently available only for diagonal approximation of the variance."
        )

    D = vp.D
    K = vp.K
    N = gp.X.shape[0]
    mu = vp.mu.copy()
    sigma = vp.sigma.copy()
    lambd = vp.lambd.copy().reshape(-1, 1)
    w = vp.w.copy()[0, :]
    Ns = len(gp.posteriors)

    # TODO: once we get more mean function add a check here
    # if all(gp.meanfun ~= [0,1,4,6,8,10,12,14,16,18,20,22])
    # error('gplogjoint:UnsupportedMeanFun', ...
    # 'Log joint computation currently only supports zero, constant, negative quadratic, negative quadratic (fixed/isotropic), negative quadratic-only, or squared exponential mean functions.');
    # end

    # Which mean function is being used?
    quadratic_meanfun = isinstance(
        gp.mean, gpr.mean_functions.NegativeQuadratic
    )

    F = np.zeros((Ns,))
    # Check which gradients are computed
    if grad_flags[0]:
        mu_grad = np.zeros((D, K, Ns))
    if grad_flags[1]:
        sigma_grad = np.zeros((K, Ns))
    if grad_flags[2]:
        lambd_grad = np.zeros((D, Ns))
    if grad_flags[3]:
        w_grad = np.zeros((K, Ns))
    if compute_var:
        varF = np.zeros((Ns,))
    # Compute gradient of variance?
    if compute_vargrad:
        if grad_flags[0]:
            mu_vargrad = np.zeros((D, K, Ns))
        if grad_flags[1]:
            sigma_vargrad = np.zeros((K, Ns))
        if grad_flags[2]:
            lambd_vargrad = np.zeros((D, Ns))
        if grad_flags[3]:
            w_vargrad = np.zeros((K, Ns))

    # Store contribution to the jog joint separately for each component?
    if separate_K:
        I_sk = np.zeros((Ns, K))
        if compute_var:
            J_sjk = np.zeros((Ns, K, K))

    if vp.delta is None:
        delta = 0
    else:
        delta = vp.delta.copy().T

    Xt = np.zeros((D, N, K))
    for k in range(0, K):
        Xt[:, :, k] = np.reshape(mu[:, k], (-1, 1)) - gp.X.T

    cov_N = gp.covariance.hyperparameter_count(D)
    mean_N = gp.mean.hyperparameter_count(D)
    noise_N = gp.noise.hyperparameter_count()

    # Loop over hyperparameter samples.
    
    for s in range(0, Ns):
        hyp = gp.posteriors[s].hyp

        # Extract GP hyperparameters from hyperparameter array.
        ell = np.exp(hyp[0:D]).reshape(-1, 1)
        ln_sf2 = 2 * hyp[D]
        sum_lnell = np.sum(hyp[0:D])

        # GP mean function hyperparameters
        if isinstance(gp.mean, gpr.mean_functions.ZeroMean):
            m0 = 0
        else:
            m0 = hyp[cov_N + noise_N]

        if quadratic_meanfun:
            xm = hyp[cov_N + noise_N + 1 : cov_N + noise_N + D + 1].reshape(
                -1, 1
            )
            omega = np.exp(hyp[cov_N + noise_N + D + 1 :]).reshape(-1, 1)

        # GP posterior parameters
        alpha = gp.posteriors[s].alpha
        L = gp.posteriors[s].L
        L_chol = gp.posteriors[s].L_chol
        sn2_eff = gp.posteriors[s].sW[0] ** 2

        for k in range(0, K):
            tau_k = np.sqrt(sigma[k] ** 2 * lambd ** 2 + ell ** 2 + delta ** 2)
            lnnf_k = (
                ln_sf2 + sum_lnell - np.sum(np.log(tau_k), axis=0)
            )  # Covariance normalization factor
            delta_k = Xt[:, :, k] / tau_k
            z_k = np.exp(lnnf_k - 0.5 * np.sum(delta_k ** 2, axis=0))
            I_k = np.dot(z_k, alpha) + m0

            if quadratic_meanfun:
                nu_k = -0.5 * np.sum(
                    1
                    / omega ** 2
                    * (
                        mu[:, k] ** 2
                        + sigma[k] ** 2
                        - 2 * mu[:, k] * xm
                        + xm ** 2
                    ),
                )
                I_k += nu_k
            F[s] += w[k] * I_k

            if separate_K:
                I_sk[s, k] = I_k

            if grad_flags[0]:
                dz_dmu = -(delta_k / tau_k) * z_k
                mu_grad[:, k, s : s + 1] = w[k] * np.dot(dz_dmu, alpha)
                if quadratic_meanfun:
                    mu_grad[:, k, s : s + 1] -= (
                        w[k] / omega ** 2 * (mu[:, k : k + 1] - xm)
                    )

            if grad_flags[1]:
                dz_dsigma = np.sum(
                    (lambd / tau_k) ** 2 * (delta_k ** 2 - 1), axis=0
                )
                sigma_grad[k, s] = w[k] * np.dot(dz_dsigma, alpha)
                if quadratic_meanfun:
                    tmp = 1 / omega ** 2 * lambd ** 2
                    sigma_grad[k, s] -= (
                        w[k]
                        / sigma[k]
                        * np.sum(1 / omega ** 2 * lambd ** 2, axis=0)
                    )

            if grad_flags[2]:
                dz_dlambd = (
                    (sigma[k] / tau_k) ** 2
                    * (delta_k ** 2 - 1)
                    * (lambd * z_k)
                )
                lambd_grad[:, s : s + 1] += w[k] * np.dot(dz_dlambd, alpha)
                if quadratic_meanfun:
                    lambd_grad[:, s : s + 1] -= (
                        w[k] * sigma[k] ** 2 / omega ** 2 * lambd
                    )

            if grad_flags[3]:
                w_grad[k, s] = I_k

            if compute_var == 2:
                # Compute only self-variance
                assert False
            elif compute_var:
                for j in range(0, k + 1):
                    tau_j = np.sqrt(
                        sigma[j] ** 2 * lambd ** 2 + ell ** 2 + delta ** 2
                    )
                    lnnf_j = ln_sf2 + sum_lnell - np.sum(np.log(tau_j), axis=0)
                    delta_j = (mu[:, j : j + 1] - gp.X.T) / tau_j
                    z_j = np.exp(lnnf_j - 0.5 * np.sum(delta_j ** 2, axis=0))

                    tau_jk = np.sqrt(
                        (sigma[j] ** 2 + sigma[k] ** 2) * lambd ** 2
                        + ell ** 2
                        + 2 * delta ** 2
                    )
                    lnnf_jk = ln_sf2 + sum_lnell - np.sum(np.log(tau_jk))
                    delta_jk = (mu[:, j : j + 1] - mu[:, k : k + 1]) / tau_jk

                    J_jk = np.exp(
                        lnnf_jk - 0.5 * np.sum(delta_jk ** 2, axis=0)
                    )
                    if L_chol:
                        J_jk = np.exp(
                            lnnf_jk - 0.5 * np.sum(delta_jk ** 2, axis=0)
                        )
                        J_jk -= np.dot(
                            z_k,
                            sp.linalg.solve_triangular(
                                L,
                                sp.linalg.solve_triangular(
                                    L, z_j, trans=1, check_finite=False
                                ),
                                trans=0,
                                check_finite=False,
                            )
                            / sn2_eff,
                        )
                    else:
                        J_jk += np.dot(z_k, np.dot(L, z_j.T))

                    # Off-diagonal elements are symmetric (count twice)
                    if j == k:
                        varF[s] += w[k] ** 2 * np.maximum(np.spacing(1), J_jk)
                        if separate_K:
                            J_sjk[s, k, k] = J_jk
                    else:
                        varF[s] += 2 * w[j] * w[k] * J_jk
                        if separate_K:
                            J_sjk[s, j, k] = J_jk
                            J_sjk[s, k, j] = J_jk

    # Correct for numerical error
    if compute_var:
        varF = np.maximum(varF, np.spacing(1))
    else:
        varF = None

    if np.any(grad_flags):
        grad_list = []
        if grad_flags[0]:
            mu_grad = np.reshape(mu_grad, (D * K, Ns))
            grad_list.append(mu_grad)

        # Correct for standard log reparametrization of sigma
        if jacobian_flag and grad_flags[1]:
            sigma_grad *= np.reshape(sigma, (-1, 1))
            grad_list.append(sigma_grad)

        # Correct for standard log reparametrization of lambd
        if jacobian_flag and grad_flags[2]:
            lambd_grad *= lambd
            grad_list.append(lambd_grad)

        # Correct for standard softmax reparametrization of w
        if jacobian_flag and grad_flags[3]:
            eta_sum = np.sum(np.exp(vp.eta))
            J_w = (
                -np.exp(vp.eta).T * np.exp(vp.eta) / eta_sum ** 2
                + np.diag(np.exp(vp.eta.flatten())) / eta_sum
            )
            w_grad = np.dot(J_w, w_grad)
            grad_list.append(w_grad)

        dF = np.concatenate(grad_list, axis=0)
    else:
        dF = None

    if compute_vargrad:
        if grad_flags[0]:
            assert False

        # Correct for standard log reparametrization of sigma
        if jacobian_flag and grad_flags[1]:
            assert False

        # Correct for standard log reparametrization of lambd
        if jacobian_flag and grad_flags[2]:
            assert False

        # Correct for standard softmax reparametrization of w
        if jacobian_flag and grad_flags[3]:
            assert False
    else:
        dvarF = None

    # Average multiple hyperparameter samples
    varss = 0
    if Ns > 1 and avg_flag:
        F_bar = np.sum(F) / Ns
        if compute_var:
            # Estimated variance of the samples
            varFss = np.sum((F - F_bar) ** 2)
            # Variability due to sampling
            varss = varFss + np.std(varF, ddof=1)
            varF = np.sum(varF, axis=0) / Ns + varFss
        if compute_vargrad:
            dvv = 2 * np.sum(F * dF, axis=1) / (Ns - 1) - 2 * F_bar * np.sum(
                dF, axis=1
            ) / (Ns - 1)
            dvarF = np.sum(dvarF, axis=1) / Ns + dvv
        F = F_bar
        if np.any(grad_flags):
            dF = np.sum(dF, axis=1) / Ns

    # In case of separate samples but only one sample simplify
    # expressions slightly.
    # TODO: what other parts need to be fixed like this?
    if Ns == 1:
        F = F[0]
        if np.any(grad_flags):
            dF = dF[:, 0]

    if separate_K:
        return F, dF, varF, dvarF, varss, I_sk, J_sjk
    return F, dF, varF, dvarF, varss
