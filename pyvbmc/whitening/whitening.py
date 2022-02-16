import numpy as np
from pyvbmc.decorators import handle_0D_1D_input


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
    assert np.all(np.isfinite(xu))
    xu = fun(xu)
    assert np.all(~np.isnan(xu))
    assert np.all(np.isfinite(xu))
    xu = np.reshape(xu, [U, N, D])
    # xu = np.reshape(fun(np.reshape(xx, [N*U, D])), [U, N, D])

    xw = np.reshape(np.mean(xu, axis=0), x_shape)
    sigmaw = np.std(xu, axis=0, ddof=1)
    return (xw, sigmaw, xu)


def warp_input_vbmc(vbmc):

        # Calculate rescaling and rotation from moments:
        __, vp_Sigma = vbmc.vp.moments(origflag=False, covflag=True)
        R_mat = vbmc.parameter_transformer.R_mat
        scale = vbmc.parameter_transformer.scale
        delta = vbmc.parameter_transformer.delta
        vp_Sigma = R_mat @ np.diag(scale) @ vp_Sigma @ np.diag(scale) @ R_mat.T
        vp_Sigma = np.diag(delta) @ vp_Sigma @ np.diag(delta)

        # Remove low-correlation entries
        if vbmc.options["warprotocorrthresh"] > 0:
            vp_corr = vp_Sigma / np.sqrt(np.outer(np.diag(vp_Sigma), np.diag(vp_Sigma)))
            mask_idx = (np.abs(vp_corr) <= vbmc.options["warprotocorrthresh"])
            vp_Sigma[mask_idx] = 0

        # Regularization of covariance matrix towards diagonal
        if type(vbmc.options["warpcovreg"]) == float or type(vbmc.options["warpcovreg"]) == int:
            w_reg = vbmc.options["warpcovreg"]
        else:
            w_reg = vbmc.options.warpcovreg[vbmc.optim_state["N"]]
        w_reg = np.max([0, np.min([1, w_reg])])
        vp_Sigma = (1 - w_reg) * vp_Sigma + w_reg * np.diag(np.diag(vp_Sigma))

        # Compute whitening transform (rotoscaling)
        U, s, Vh = np.linalg.svd(vp_Sigma)
        if np.linalg.det(U) < 0:
            U[:, 0] = -U[:, 0]
        scale = np.sqrt(s)

        return U, scale
