"""F1: log density and its gradient where the mixture density underflows.

Compares vp.pdf(orig_flag=False, log_flag=True, grad_flag=True) with an
independent log-sum-exp evaluation along a ray, locates where -inf/NaN
start, checks the linear density there, checks for the numpy warning, and
compares with a line-by-line Python transcription of vbmc_pdf.m:49-66 and
:107-110 (MATLAB semantics: log(0) = -Inf, 0/0 = NaN).
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from wave7_O2_common import banner, build_vp, ref_logq_and_grad

banner()
np.set_printoptions(precision=6, linewidth=150)


def matlab_vbmc_pdf_log_grad(mu, sigma, lambd, w, X):
    """Transcription of vbmc_pdf.m:41-66 and :107-110, origflag=0."""
    X = np.atleast_2d(X)
    N, D = X.shape
    K = mu.shape[1]
    lam = np.ravel(lambd)
    mu_t = mu.T
    s = np.ravel(sigma)
    w = np.ravel(w)
    y = np.zeros((N, 1))
    dy = np.zeros((N, D))
    nf = 1 / (2 * np.pi) ** (D / 2) / np.prod(lam)
    with np.errstate(all="ignore"):
        for k in range(K):
            d2 = np.sum(((X - mu_t[k]) / (s[k] * lam)) ** 2, axis=1)[:, None]
            nn = nf * w[k] / s[k] ** D * np.exp(-0.5 * d2)
            y = y + nn
            dy = dy - nn * ((X - mu_t[k]) / (lam**2 * s[k] ** 2))
        dy = dy / y
        y = np.log(y)
    return y, dy


# --- 1-D unit Gaussian: where the log density turns to -inf ---------------
vp1 = build_vp([[0.0]], [1.0], [1.0], [1.0])
ts = np.arange(37.0, 40.01, 0.05)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ly = vp1.pdf(ts[:, None], orig_flag=False, log_flag=True).ravel()
first_inf = ts[np.argmax(~np.isfinite(ly))]
print(
    f"D=1 unit Gaussian: log q first -inf at x = {first_inf:.2f}; "
    f"exact log q there = {-0.5*np.log(2*np.pi) - 0.5*first_inf**2:.2f}; "
    f"last finite value {ly[np.isfinite(ly)][-1]:.2f}",
    flush=True,
)

# --- D=2, K=2 mixture along a ray ----------------------------------------
mu = np.array([[0.0, 1.5], [0.0, -0.5]])
sigma = np.array([1.0, 0.7])
lambd = np.array([1.0, 0.6])
w = np.array([0.6, 0.4])
vp = build_vp(mu, sigma, lambd, w)
direction = np.array([1.0, 0.3]) / np.linalg.norm([1.0, 0.3])
print(
    "\nD=2, K=2 mixture; t = distance along a ray from the origin, in "
    "units of component 1's SD in x1",
    flush=True,
)
print(
    " t      code log q       ref log q     |dlogq|/|ref|  "
    "max rel err grad   code grad",
    flush=True,
)
for t in [5, 20, 30, 34, 35, 36, 37, 38, 40, 45, 60, 100]:
    X = (t * direction)[None, :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ly, dly = vp.pdf(X, orig_flag=False, log_flag=True, grad_flag=True)
        lin = vp.pdf(X, orig_flag=False)
    lq, g = ref_logq_and_grad(X, mu, sigma, lambd, w)
    ev = (
        abs(ly[0, 0] - lq[0]) / abs(lq[0]) if np.isfinite(ly[0, 0]) else np.inf
    )
    with np.errstate(invalid="ignore"):
        eg = np.max(np.abs(dly - g) / np.abs(g))
    print(
        f"{t:5.1f} {ly[0,0]:14.4f} {lq[0]:14.4f} {ev:12.2e} {eg:14.2e}   "
        f"{dly.ravel()}  (ref {g.ravel()}; linear pdf {lin[0,0]:.3e})",
        flush=True,
    )

# --- warnings emitted by the call ----------------------------------------
X = (45 * direction)[None, :]
with warnings.catch_warnings(record=True) as rec:
    warnings.simplefilter("always")
    vp.pdf(X, orig_flag=False, log_flag=True, grad_flag=True)
print(
    "\nwarnings from log_pdf+grad at t=45:",
    [f"{r.category.__name__}: {r.message}" for r in rec],
    flush=True,
)
with warnings.catch_warnings(record=True) as rec:
    warnings.simplefilter("always")
    vp.pdf(X, orig_flag=False, log_flag=True)
print(
    "warnings from log_pdf alone at t=45:",
    [f"{r.category.__name__}: {r.message}" for r in rec],
    flush=True,
)

# --- the MATLAB transcription gives the same numbers ----------------------
Xs = np.array([t * direction for t in (5, 30, 37, 45, 100)])
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ly, dly = vp.pdf(Xs, orig_flag=False, log_flag=True, grad_flag=True)
my, mdy = matlab_vbmc_pdf_log_grad(mu, sigma, lambd, w, Xs)
same = np.array_equal(ly, my, equal_nan=True) and np.array_equal(
    dly, mdy, equal_nan=True
)
print(
    "\nMATLAB transcription vs code at t = 5, 30, 37, 45, 100:",
    "identical (NaN-aware)" if same else "DIFFER",
    flush=True,
)
print("  transcription log q:", my.ravel(), flush=True)
fin = np.isfinite(dly)
print(
    "  same non-finite pattern (value, grad):",
    np.array_equal(np.isfinite(ly), np.isfinite(my)),
    np.array_equal(fin, np.isfinite(mdy)),
    flush=True,
)
print(
    "  finite log q identical:",
    np.array_equal(ly[np.isfinite(ly)], my[np.isfinite(my)]),
    "; max rel diff of finite gradient entries:",
    np.max(np.abs(dly[fin] - mdy[fin]) / np.abs(mdy[fin])),
    flush=True,
)

# --- orig_flag=True on a bounded transform: the same underflow ------------
from pyvbmc.parameter_transformer import ParameterTransformer

pt = ParameterTransformer(
    1,
    np.array([[0.0]]),
    np.array([[1.0]]),
    np.array([[0.1]]),
    np.array([[0.9]]),
)
vpb = build_vp([[0.0]], [1.0], [0.01], [1.0], transformer=pt)
xs = np.array([[0.5], [0.52], [0.6], [0.7], [0.9]])
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    lyb = vpb.pdf(xs, orig_flag=True, log_flag=True).ravel()
u = pt(xs).ravel()
print("\nbounded [0,1], sigma*lambda = 0.01 in transformed space:", flush=True)
for xi, ui, li in zip(xs.ravel(), u, lyb):
    print(
        f"  x = {xi:.2f}: u = {ui:+.4f} ({abs(ui)/0.01:.1f} sd), "
        f"log p(x) = {li}",
        flush=True,
    )
