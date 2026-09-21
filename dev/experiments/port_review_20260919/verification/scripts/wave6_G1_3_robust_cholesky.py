"""Settles the eigenvector sign normalization of ``GP.__robust_cholesky``.

``gpyreg/gaussian_process.py:2620-2622`` computes ``maxidx`` per column and
then writes ``negidx = U[maxidx] < 0``, which is *row* fancy indexing, so
``U[negidx] *= -1`` flips individual entries.  ``gplite_rnd.m:97-99``
(read at vbmc 396d649) uses the linear index ``maxidx + (0:n:(m-1)*n)``,
which picks one element per column, and flips whole columns.
(G1 internal F3 / G1 comparison F2.)

The script compares ``T.T @ T`` with the input for gpyreg's function, for a
per-column transcription of ``robustchol``, and for the intermediate
``U diag(D) U.T``, on three inputs whose plain Cholesky fails.
"""

import gpyreg as gpr
import numpy as np
import scipy as sp
from gpyreg.gaussian_process import GP

print("gpyreg:", gpr.__file__)

robust = GP._GP__robust_cholesky


def robustchol_matlab(sigma):
    """Transcription of gplite_rnd.m:88-112 (per-column sign flip)."""
    n, m = sigma.shape
    try:
        return sp.linalg.cholesky(sigma, check_finite=False), 0
    except sp.linalg.LinAlgError:
        pass
    D, U = np.linalg.eigh((sigma + sigma.T) / 2)
    maxidx = np.argmax(np.abs(U), axis=0)
    negidx = U[maxidx, np.arange(m)] < 0
    U[:, negidx] *= -1
    tol = abs(np.spacing(np.max(D))) * D.shape[0]
    t = np.abs(D) > tol
    Dk = D[t]
    p = int(np.sum(Dk < 0))
    if p == 0:
        return np.diag(np.sqrt(Dk)) @ U[:, t].T, 0
    return np.zeros((0, m)), p


def show(label, C):
    C = (C + C.T) / 2
    try:
        sp.linalg.cholesky(C, check_finite=False)
        print(f"\n{label}: plain Cholesky SUCCEEDS, fallback not taken")
        return
    except sp.linalg.LinAlgError:
        pass
    print(f"\n{label}: plain Cholesky fails, fallback taken")
    T = robust(C)
    err_py = np.max(np.abs(T.T @ T - C)) if T.size else float("nan")
    Tm, p = robustchol_matlab(C.copy())
    err_ml = np.max(np.abs(Tm.T @ Tm - C)) if Tm.size else float("nan")
    print(f"  ||C||_max                        = {np.max(np.abs(C)):.6g}")
    print(
        f"  gpyreg   max|T^T T - C|          = {err_py:.6g}  shape {T.shape}"
    )
    print(
        f"  MATLAB   max|T^T T - C|          = {err_ml:.6g}  shape {Tm.shape}, p={p}"
    )
    # step by step on the eigendecomposition
    D, U = sp.linalg.eig((C + C.T) / 2)
    U0 = U.copy()
    rec0 = np.max(np.abs(np.real(U0 @ np.diag(D) @ U0.T) - C))
    maxidx = np.argmax(np.abs(U), axis=0)
    negidx = U[maxidx] < 0
    U1 = U.copy()
    U1[negidx] *= -1
    rec1 = np.max(np.abs(np.real(U1 @ np.diag(D) @ U1.T) - C))
    U2 = U.copy()
    col = U2[maxidx, np.arange(U2.shape[1])] < 0
    U2[:, col] *= -1
    rec2 = np.max(np.abs(np.real(U2 @ np.diag(D) @ U2.T) - C))
    print(f"  max|U D U^T - C| before flip     = {rec0:.6g}")
    print(f"  ... after the flip as written    = {rec1:.6g}")
    print(f"  ... after a per-column flip      = {rec2:.6g}")
    print(f"  negidx shape as written {negidx.shape}, per-column {col.shape}")
    # sp.linalg.eig (general solver) against MATLAB's eig on a symmetric
    # input (the symmetric solver, i.e. eigh): with a repeated eigenvalue
    # the general solver need not return an orthogonal basis of the
    # eigenspace, and then U D U^T = C fails before any sign flip.
    De, Ue = np.linalg.eigh((C + C.T) / 2)
    rec_eigh = np.max(np.abs(Ue @ np.diag(De) @ Ue.T - C))
    orth_eig = np.max(np.abs(np.real(U0.T @ U0) - np.eye(U0.shape[0])))
    orth_eigh = np.max(np.abs(Ue.T @ Ue - np.eye(Ue.shape[0])))
    print(f"  max|U D U^T - C| with eigh       = {rec_eigh:.6g}")
    print(f"  max|U^T U - I|: eig {orth_eig:.6g}  eigh {orth_eigh:.6g}")


# (1) a rank-deficient PSD matrix
rng = np.random.default_rng(1)
A = rng.standard_normal((5, 3))
show("rank-deficient A@A.T (5x5, rank 3)", A @ A.T)

# (2) a squared-exponential kernel on duplicated points
xs = np.array([[0.0], [0.0], [0.5], [0.5], [1.0]])
d2 = (xs - xs.T) ** 2
show("SE kernel on duplicated test points", np.exp(-d2 / 2))

# (3) one genuinely negative eigenvalue
M = np.diag([1.0, 1.0, 1.0, 1.0, -0.5])
Qr, _ = np.linalg.qr(rng.standard_normal((5, 5)))
show("one negative eigenvalue", Qr @ M @ Qr.T)

# (4) a diagonal input: eigenvectors are unit vectors, flip is harmless
show("diagonal, rank-deficient", np.diag([1.0, 2.0, 0.0, 3.0, 0.0]))

# (5) a repeated POSITIVE eigenvalue with a rank deficiency: here
# sp.linalg.eig's non-orthogonal basis of the degenerate eigenspace breaks
# U D U^T = C on its own, before the sign flip.  MATLAB's eig on a
# symmetric input uses the symmetric solver.
Qr2, _ = np.linalg.qr(rng.standard_normal((5, 5)))
show(
    "repeated positive eigenvalue, rank 3",
    Qr2 @ np.diag([1.0, 1.0, 1.0, 0.0, 0.0]) @ Qr2.T,
)
