import numpy as np

from pyvbmc.decorators import handle_0D_1D_input


@handle_0D_1D_input(
    patched_kwargs=["mu1", "sigma1", "mu2", "sigma2"],
    patched_argpos=[0, 1, 2, 3],
)
def kl_div_mvn(mu1, sigma1, mu2, sigma2):
    """
    Compute the analytical Kullback-Leibler divergence between two multivariate
    normal pdfs.

    Parameters
    ----------
    mu1 : np.ndarray
        The k-dimensional mean vector of the first multivariate normal pdf.
    sigma1 : np.ndarray
        The covariance matrix of the first multivariate normal pdf.
    mu2 : np.ndarray
        The k-dimensional mean vector of the second multivariate normal pdf.
    sigma2 : np.ndarray
        The covariance matrix of the second multivariate normal pdf.

    Returns
    -------
    kl_div : np.array
        The computed Kullback-Leibler divergence.
    """
    D = mu1.size
    mu1 = mu1.reshape(-1, 1)
    mu2 = mu2.reshape(-1, 1)
    dmu = mu2 - mu1
    detq1 = np.linalg.det(sigma1)
    detq2 = np.linalg.det(sigma2)
    lndet = np.log(detq2 / detq1)
    a, _, _, _ = np.linalg.lstsq(sigma2, sigma1, rcond=None)
    b, _, _, _ = np.linalg.lstsq(sigma2, dmu, rcond=None)
    kl1 = 0.5 * (np.trace(a) + dmu.T @ b - D + lndet)
    a, _, _, _ = np.linalg.lstsq(sigma1, sigma2, rcond=None)
    b, _, _, _ = np.linalg.lstsq(sigma1, dmu, rcond=None)
    kl2 = 0.5 * (np.trace(a) + dmu.T @ b - D - lndet)
    return np.concatenate((kl1, kl2), axis=None)
