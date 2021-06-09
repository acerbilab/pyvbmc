import numpy as np
from pyvbmc.decorators import handle_0D_1D_input


@handle_0D_1D_input(
    patched_kwargs=["mu1", "sigma1", "mu2", "sigma2"],
    patched_argpos=[0, 1, 2, 3],
)
def kldiv_mvn(mu1, sigma1, mu2, sigma2):
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