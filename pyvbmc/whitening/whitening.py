import numpy as np
import gpyreg as gpr
import copy


def unscent_warp(fun, x, sigma):
    r"""Compute the unscented transform of the warping function `fun`.

    Parameters
    ----------
    fun : function
        A single-argument function which warps input points.
    x : (n,D) or (D,) np.ndarray
        The input mean for which to approximate the unscented transform.
    sigma : (n,D) or (D,) np.ndarray
        The input standard deviation matrix for which to approximate the
        unscented transform.

    Returns
    -------
    x_warped_mean : (n,D) or (D,) np.ndarray
        The unscented estimate of the mean.
    x_warped_sigma : (n,D) np.ndarray
        The unscented estimate of the std.
    x_warped : (U,n,D) np.ndarray
        The warped mean points at `x_warped[0, :, :]`, and the warped std simplex
        points, at `[1:, :, :]`. Here `U=2*D+1`.

    Raises
    ------
    AssertionError
        If the rows of `x` and `sigma` cannot be coerced to match.
    """
    x_shape_orig = x.shape
    x = np.atleast_2d(x).copy()
    sigma = np.atleast_2d(sigma).copy()
    (N1, D) = x.shape
    (N2, D2) = sigma.shape

    N = np.max([N1, N2])

    assert N1 in (1, N), "Mismatch between rows of X and SIGMA."
    assert N2 in (1, N), "Mismatch between rows of X and SIGMA."
    assert (D == D2), "Mismatch between columns of X and SIGMA."

    if (N1 == 1) and (N > 1):
        x = np.tile(x, [N, 1])
    if (N2 == 1) and (N > 1):
        sigma = np.tile(sigma, [N, 1])

    U = 2*D+1

    # For each dimension, collect the points at +- one std. deviation
    # along that dimension, and the original points x
    xx = np.tile(x, [U, 1, 1])
    for d in range(D):
        sigma_slice = np.sqrt(D) * sigma[:, d]
        # xx already contains:
        # xx[0, :, d] = x[d]
        xx[2*d+1, :, d] = xx[2*d+1, :, d] + sigma_slice
        xx[2*d+2, :, d] = xx[2*d+2, :, d] - sigma_slice

    # Drop points into column to apply warping, then reshape back
    x_warped = np.reshape(xx, [N*U, D])
    x_warped = fun(x_warped)
    x_warped = np.reshape(x_warped, [U, N, D])

    # Estimate the mean and standard deviation of the warped points
    # by the mean and std of these sigma-points
    x_warped_mean = np.reshape(np.mean(x_warped, axis=0), x_shape_orig)
    x_warped_sigma = np.std(x_warped, axis=0, ddof=1)

    return x_warped_mean, x_warped_sigma, x_warped


def warp_input_vbmc(vp, optim_state, function_logger, options):
    r"""Compute the whitening transformation and update the cached points in function_logger.

    The whitening transformation is a rotation and rescaling (implemented) or nonlinear transformation (not implemented) which is applied to the inference space such that the variational posterior acheives unit diagonal covariance.

    Parameters
    ----------
    vp : VariationalPosterior
        The current VP object for which to compute the transformation.
    optim_state : dict
        The dictionary recording the current optimization state.
    function_logger : FunctionLogger
        The record including cached function values.

    Returns
    -------
    parameter_transformer : ParameterTransformer
        A ParameterTransformer object representing the new transformation between original coordinates and inference space coordinates, with the input warping applied.
    optim_state : dict
        An updated copy of the original optimization state dict.
    function_logger : FunctionLogger
        An updated copy of the original function logger.
    warp_action : str
        The type of warping performed ("rotoscaling" or "warp")

    Raises
    ------
    NotImplementedError
        If vbmc.options["warpnonlinear"] is set other than False.
    """
    parameter_transformer = copy.deepcopy(vp.parameter_transformer)
    optim_state = copy.deepcopy(optim_state)
    function_logger = copy.deepcopy(function_logger)

    if options.get("warprotoscaling"):
        if options.get("warpnonlinear"):
            raise NotImplementedError
        else:
            # Get covariance matrix analytically
            __, vp_Sigma = vp.moments(origflag=False, covflag=True)
            delta = parameter_transformer.delta
            R_mat = parameter_transformer.R_mat
            scale = parameter_transformer.scale
            if R_mat is None:
                R_mat = np.eye(vp.D)
            if scale is None:
                scale = np.ones(vp.D)
            vp_Sigma = R_mat @ np.diag(scale) @ vp_Sigma @ np.diag(scale) @ R_mat.T
            vp_Sigma = np.diag(delta) @ vp_Sigma @ np.diag(delta)

    # Remove low-correlation entries
    if options["warprotocorrthresh"] > 0:
        vp_corr = vp_Sigma / np.sqrt(np.outer(np.diag(vp_Sigma), np.diag(vp_Sigma)))
        mask_idx = (np.abs(vp_corr) <= options["warprotocorrthresh"])
        vp_Sigma[mask_idx] = 0

    # Regularization of covariance matrix towards diagonal
    if type(options["warpcovreg"]) == float or type(options["warpcovreg"]) == int:
        w_reg = options["warpcovreg"]
    else:
        w_reg = options.warpcovreg[optim_state["N"]]
    w_reg = np.max([0, np.min([1, w_reg])])
    vp_Sigma = (1 - w_reg) * vp_Sigma + w_reg * np.diag(np.diag(vp_Sigma))

    # Compute whitening transform (rotoscaling)
    U, s, __ = np.linalg.svd(vp_Sigma)
    if np.linalg.det(U) < 0:
        U[:, 0] = -U[:, 0]
    scale = np.sqrt(s+np.finfo(np.float64).eps)
    parameter_transformer.R_mat = U
    parameter_transformer.scale = scale

    # Update Plausible Bounds:
    parameter_transformer.mu = np.zeros(vp.D)
    parameter_transformer.delta = np.ones(vp.D)
    Nrnd = 100000
    xx = np.random.rand(Nrnd, vp.D) * \
        (optim_state["pub_orig"]-optim_state["plb_orig"])\
        + optim_state["plb_orig"]
    yy = parameter_transformer(xx)

    # Quantile-based estimate of plausible bounds
    [plb, pub] = np.quantile(yy, [0.05, 0.95], axis=0)
    delta_temp = pub-plb
    plb = plb - delta_temp/9
    pub = pub + delta_temp/9
    plb = np.atleast_2d(plb)
    pub = np.atleast_2d(pub)

    optim_state["plb"] = plb
    optim_state["pub"] = pub

    # Temperature scaling
    if optim_state.get("temperature"):
        T = optim_state["temperature"]
    else:
        T = 1

    # Adjust stored points after warping
    X_flag = function_logger.X_flag
    X_orig = function_logger.X_orig[X_flag, :]
    y_orig = function_logger.y_orig[X_flag].T
    X = parameter_transformer(X_orig)
    dy = parameter_transformer.log_abs_det_jacobian(X)
    y = y_orig + dy/T
    function_logger.X[X_flag, :] = X
    function_logger.y[X_flag] = y.T
    function_logger.parameter_transformer = parameter_transformer

    # Update search bounds:
    # Invert points to original space with old transform,
    # then map to new space with new transform
    def warpfun(x):
        return parameter_transformer(
                   vp.parameter_transformer.inverse(x)
               )
    Nrnd = 1000
    xx = np.random.rand(Nrnd, vp.D) * \
        (optim_state["ub_search"] - optim_state["lb_search"])\
        + optim_state["lb_search"]
    yy = warpfun(xx)
    yyMin = np.min(yy, axis=0)
    yyMax = np.max(yy, axis=0)
    delta = yyMax - yyMin
    optim_state["lb_search"] = np.atleast_2d(yyMin - delta/Nrnd)
    optim_state["ub_search"] = np.atleast_2d(yyMax + delta/Nrnd)

    # If search cache is not empty, update it
    if optim_state.get("search_cache"):
        optim_state["search_cache"] = warpfun(
            optim_state["search_cache"]
        )

    # Update other state fields
    optim_state["recompute_var_post"] = True
    optim_state["skipactivesampling"] = True
    optim_state["warping_count"] += 1
    optim_state["last_warping"] = optim_state["iter"]
    optim_state["last_successful_warping"] = optim_state["iter"]

    # Reset GP Hyperparameters
    optim_state["run_mean"] = []
    optim_state["run_cov"] = []
    optim_state["last_run_avg"] = np.nan

    # Warp action for output display
    if options.get("warpnonlinear"):
        warp_action = "warp"
    else:
        warp_action = "rotoscale"

    return parameter_transformer, optim_state, function_logger, warp_action


def warp_gpandvp_vbmc(parameter_transformer, vp_old, vbmc):
    r"""Update the GP and VP with a given warp transformation.

    Applies an updated ParameterTransformer object (with new warping transformation) to the GP and VP parameters.

    Parameters
    ----------
    parameter_transformer : ParameterTransformer
        The new (warped) transformation between input coordinates and inference coordinates.
    vp_old : VariationalPosterior
        The current variational posterior.
    vbmc : VBMC
        The current VBMC object.

    Returns
    -------
    vp : VariationalPosterior
        An updated copy of the original variational posterior.
    hyp_warped : dict
        An updated copy of the dictionary of original GP hyperparameters, with the warping transformation apploed.
    """
    vp_old = copy.deepcopy(vp_old)

    # Invert points from the old inference space to the input space,
    # then push them back to the new inference space
    def warpfun(x):
        return parameter_transformer(
            vp_old.parameter_transformer.inverse(x)
        )

    # Temperature scaling
    if vbmc.optim_state.get("temperature"):
        T = vbmc.optim_state["temperature"]
    else:
        T = 1

    # Get the number of GP hyperparameters, for indexing:
    Ncov = vp_old.gp.covariance.hyperparameter_count(vbmc.D)
    Nnoise = vp_old.gp.noise.hyperparameter_count()
    Nmean = vp_old.gp.mean.hyperparameter_count(vbmc.D)
    # MATLAB: if ~isempty(gp_old.outwarpfun); Noutwarp = gp_old.Noutwarp; else; Noutwarp = 0; end
    # (Not used, see gaussian_process.py)

    Ns_gp = len(vp_old.gp.posteriors)
    hyp_warped = np.zeros([Ncov + Nnoise + Nmean, Ns_gp])
    hyps = vp_old.gp.get_hyperparameters(as_array=True)

    for s in range(Ns_gp):
        hyp = hyps[s]
        hyp_warped[:, s] = hyp.copy()

        # UpdateGP input length scales
        ell = np.exp(hyp[0:vbmc.D]).T
        (__, ell_new, __) = unscent_warp(warpfun, vp_old.gp.X, ell)
        hyp_warped[0:vbmc.D, s] = np.mean(np.log(ell_new), axis=0)

        # We assume relatively no change to GP output and noise scales
        if isinstance(vp_old.gp.mean, gpr.mean_functions.ConstantMean):
            # Warp constant mean
            m0 = hyp[Ncov+Nnoise]
            dy_old = vp_old.parameter_transformer.log_abs_det_jacobian(vp_old.gp.X)
            dy = parameter_transformer.log_abs_det_jacobian(warpfun(vp_old.gp.X))
            m0w = m0 + (np.mean(dy, axis=0) - np.mean(dy_old, axis=0))/T

            hyp_warped[Ncov+Nnoise, s] = m0w

        elif isinstance(vp_old.gp.mean, gpr.mean_functions.NegativeQuadratic):
            # Warp quadratic mean
            m0 = hyp[Ncov + Nnoise]
            xm = hyp[Ncov + Nnoise + 1 : Ncov + Nnoise + vbmc.D + 1].T
            omega = np.exp(hyp[Ncov + Nnoise + vbmc.D + 1:]).T

            # Warp location and scale
            (xmw, omegaw, __) = unscent_warp(warpfun, xm, omega)

            # Warp maximum
            dy_old = vp_old.parameter_transformer.log_abs_det_jacobian(xm).T
            dy = parameter_transformer.log_abs_det_jacobian(xmw).T
            m0w = m0 + (dy - dy_old)/T

            hyp_warped[Ncov + Nnoise, s] = m0w
            hyp_warped[Ncov + Nnoise + 1 : Ncov + Nnoise + vbmc.D + 1, s] = xmw.T
            hyp_warped[Ncov + Nnoise + vbmc.D + 1 :, s] = np.log(omegaw).reshape(-1)
        else:
            raise ValueError("Unsupported GP mean function for input warping.")
    hyp_warped = hyp_warped.T

    # Update variational posterior

    vp = copy.deepcopy(vp_old)
    vp.parameter_transformer = copy.deepcopy(parameter_transformer)

    mu = vp.mu.T
    sigmalambda = (vp_old.lambd * vp_old.sigma).T

    (muw, sigmalambdaw, __) = unscent_warp(warpfun, mu, sigmalambda)

    vp.mu = muw.T
    lambdaw = np.sqrt(vbmc.D*np.mean(
        (sigmalambdaw**2).T / np.sum(sigmalambdaw**2, axis=1),
        axis=1)).T
    vp.lambd[:, 0] = lambdaw

    sigmaw = np.exp(np.mean(np.log(sigmalambdaw / lambdaw), axis=1))
    vp.sigma[0, :] = sigmaw

    # Approximate change in weight:
    dy_old = vp_old.parameter_transformer.log_abs_det_jacobian(mu)
    dy = parameter_transformer.log_abs_det_jacobian(muw)

    ww = vp_old.w * np.exp((dy - dy_old)/T)
    vp.w = ww / np.sum(ww)

    return vp, hyp_warped
