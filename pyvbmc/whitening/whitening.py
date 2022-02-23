import numpy as np
from pyvbmc.decorators import handle_0D_1D_input
import gpyreg as gpr
import copy


# @handle_0D_1D_input(patched_kwargs=["x", "sigma"], patched_argpos=[1, 2])
def unscent_warp(fun, x, sigma):
    x_shape = x.shape
    # sigma_shape = sigma.shape
    # print(x.shape)
    # print(sigma.shape)
    x = np.atleast_2d(x).copy()
    sigma = np.atleast_2d(sigma).copy()
    [N1, D] = x.shape
    [N2, D2] = sigma.shape

    N = np.max([N1, N2])

    assert (N1 == N) or (N1 == 1), "Mismatch between rows of X and SIGMA."
    assert (N2 == N) or (N2 == 1), "Mismatch between rows of X and SIGMA."
    assert (D == D2), "Mismatch between columns of X and SIGMA."

    if (N1 == 1) and (N > 1):
        x = np.tile(x, [N, 1])
    if (N2 == 1) and (N > 1):
        sigma = np.tile(sigma, [N, 1])

    U = 2*D+1

    x3 = np.zeros([1, N1, D])
    x3[0, :, :] = x
    xx = np.tile(x3, [U, 1, 1])

    for d in range(1,D+1):
        # sigma3 = np.zeros([1, N, 1])
        # sigma3[0, :, 0] = np.sqrt(D)*sigma[:, d]
        sigma3 = np.sqrt(D)*sigma[:, d-1]
        xx[2*d-1, :, d-1] = xx[2*d-1, :, d-1] + sigma3
        xx[2*d, :, d-1] = xx[2*d, :, d-1] - sigma3

    xu = np.reshape(xx, [N*U, D])
    xu = fun(xu)
    xu = np.reshape(xu, [U, N, D])
    # xu = np.reshape(fun(np.reshape(xx, [N*U, D])), [U, N, D])

    xw = np.reshape(np.mean(xu, axis=0), x_shape)
    sigmaw = np.std(xu, axis=0, ddof=1)
    assert np.all(~np.isinf(sigmaw))
    return (xw, sigmaw, xu)


def warp_input_vbmc(vp, optim_state, function_logger, options):
    parameter_transformer = copy.deepcopy(vp.parameter_transformer)
    optim_state = copy.deepcopy(optim_state)
    function_logger = copy.deepcopy(function_logger)

    # Calculate rescaling and rotation from moments:
    __, vp_Sigma = vp.moments(origflag=False, covflag=True)
    R_mat = parameter_transformer.R_mat
    scale = parameter_transformer.scale
    if R_mat is None:
        R_mat = np.eye(vp.D)
    if scale is None:
        scale = np.ones(vp.D)
    delta = parameter_transformer.delta
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
    U, s, Vh = np.linalg.svd(vp_Sigma)
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
    [plb, pub] = np.quantile(yy, [0.05, 0.95], axis=0)
    delta_temp = pub-plb
    plb = plb - delta_temp/9
    pub = pub + delta_temp/9
    plb = np.reshape(plb, (-1, len(plb)))
    pub = np.reshape(pub, (-1, len(pub)))

    optim_state["plb"] = plb
    optim_state["pub"] = pub

    # TODO: Add temperature scaling?
    T = 1
    # Adjust stored points after warping:
    X_flag = function_logger.X_flag
    # x_orig = self.optim_state["cache"]["x_orig"][X_flag, :]
    # y_orig = self.optim_state["cache"]["y_orig"][X_flag]
    X_orig = function_logger.X_orig[X_flag, :]
    y_orig = function_logger.y_orig[X_flag].T
    X = parameter_transformer(X_orig)
    dy = parameter_transformer.log_abs_det_jacobian(X)
    y = y_orig + dy/T
    function_logger.X[X_flag, :] = X
    function_logger.y[X_flag] = y.T

    # Update search bounds:
    assert not (parameter_transformer is vp.parameter_transformer)
    # vp.parameter_transformer = parameter_transformer
    # self.function_logger.parameter_transformer = parameter_transformer

    def warpfun(x):
        # Copy probably unneccesary:
        x = np.copy(x)
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
    optim_state["lb_search"] = np.reshape(yyMin - delta/Nrnd,
                                                (-1, len(yyMin - delta/Nrnd)
                                                ))
    optim_state["ub_search"] = np.reshape(yyMax + delta/Nrnd,
                                                (-1, len(yyMax + delta/Nrnd)
                                                ))

    # If search cache is not empty, update it:
    if optim_state.get("search_cache"):
        optim_state["search_cache"] = warpfun(
            optim_state["search_cache"]
        )

    # Update other state fields:
    optim_state["recompute_var_post"] = True
    optim_state["skipactivesampling"] = True
    optim_state["warping_count"] += 1
    optim_state["last_warping"] = optim_state["iter"]
    optim_state["last_successful_warping"] = optim_state["iter"]

    # Reset GP Hyperparameters:
    optim_state["run_mean"] = []
    optim_state["run_cov"] = []
    optim_state["last_run_avg"] = np.nan

    # Warp action for output display:
    if options.get("warpnonlinear"):
        warp_action = "warp"
    else:
        warp_action = "rotoscale"

    return parameter_transformer, optim_state, function_logger, warp_action


def warp_gpandvp_vbmc(parameter_transformer, vp_old, vbmc):
    vp_old = copy.deepcopy(vp_old)
    def warpfun(x):
        return parameter_transformer(
            vp_old.parameter_transformer.inverse(x)
        )
    # TODO: Add temperature scaling?
    T = 1
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
            omega = np.exp(hyp[Ncov + Nnoise + vbmc.D + 1 : ]).T

            # Warp location and scale
            (xmw, omegaw, __) = unscent_warp(warpfun, xm, omega)

            # Warp maximum
            dy_old = vp_old.parameter_transformer.log_abs_det_jacobian(xm).T
            dy = parameter_transformer.log_abs_det_jacobian(xmw).T
            m0w = m0 + (dy - dy_old)/T

            hyp_warped[Ncov + Nnoise, s] = m0w
            hyp_warped[Ncov + Nnoise + 1 : Ncov + Nnoise + vbmc.D + 1, s] = xmw.T
            hyp_warped[Ncov + Nnoise + vbmc.D + 1 : , s] = np.log(omegaw).reshape(-1)
        else:
            raise ValueError("Unsupported GP mean function for input warping.")
    hyp_warped = hyp_warped.T

    # Update VP:

    vp = copy.deepcopy(vp_old)
    vp.parameter_transformer = copy.deepcopy(parameter_transformer)

    mu = vp.mu.T
    sigmalambda = (vp_old.lambd * vp_old.sigma).T

    (muw, sigmalambdaw, __) = unscent_warp(warpfun, mu, sigmalambda)

    vp.mu = muw.T
    lambdaw = np.sqrt(vbmc.D*np.mean(
        sigmalambdaw**2 / (sigmalambdaw**2+2),
        axis=0))
    vp.lambd[:, 0] = lambdaw

    sigmaw = np.exp(np.mean(np.log(sigmalambdaw / lambdaw), axis=1))
    vp.sigma[0, :] = sigmaw

    # Approximate change in weight:
    dy_old = vp_old.parameter_transformer.log_abs_det_jacobian(mu)
    dy = parameter_transformer.log_abs_det_jacobian(muw)

    ww = vp_old.w * np.exp((dy - dy_old)/T)
    vp.w = ww / np.sum(ww)

    return vp, hyp_warped
