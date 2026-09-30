import numpy as np


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
        The forward and reverse Kullback-Leibler divergence. Both are
        infinite if either covariance matrix is singular or has a negative
        determinant, neither being the covariance of a distribution with a
        density.
    """
    mu1 = np.asarray(mu1)
    sigma1 = np.atleast_2d(sigma1)
    mu2 = np.asarray(mu2)
    sigma2 = np.atleast_2d(sigma2)

    D = mu1.size
    mu1 = mu1.reshape(-1, 1)
    mu2 = mu2.reshape(-1, 1)
    dmu = mu2 - mu1
    # The determinants themselves leave the range of a double at moderate
    # dimension: of a covariance of scale s per coordinate the determinant
    # is s**(2 * D). Their logarithms stay in range, and the divergence
    # needs no more than the difference of the two.
    sign1, lndetq1 = np.linalg.slogdet(sigma1)
    sign2, lndetq2 = np.linalg.slogdet(sigma2)
    if sign1 <= 0 or sign2 <= 0:
        # A zero sign is a singular matrix and a negative one a negative
        # determinant: KL divergence is infinite
        return np.concatenate((np.inf, np.inf), axis=None)
    lndet = lndetq2 - lndetq1
    a, _, _, _ = np.linalg.lstsq(sigma2, sigma1, rcond=None)
    b, _, _, _ = np.linalg.lstsq(sigma2, dmu, rcond=None)
    kl1 = 0.5 * (np.trace(a) + dmu.T @ b - D + lndet)
    a, _, _, _ = np.linalg.lstsq(sigma1, sigma2, rcond=None)
    b, _, _, _ = np.linalg.lstsq(sigma1, dmu, rcond=None)
    kl2 = 0.5 * (np.trace(a) + dmu.T @ b - D - lndet)
    return np.concatenate((kl1, kl2), axis=None)
