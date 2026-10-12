"""P7-6: pyvbmc.stats.kde_1d against a transcription of shared/kde1d.m.

Settles four claims about kde_1d:
  (i)   the returned density sums to n/(n-1) over its grid, times the
        spacing -- and whether MATLAB's does too;
  (ii)  the binning: nearest grid point (Python) against histc's
        left-edge binning (MATLAB, kde1d.m:46-48), i.e. a half-bin shift;
  (iii) the fallback when the root solve fails: Scott's rule (Python)
        against fminbnd on |f| (MATLAB, kde1d.m:136-138);
  (iv)  negative round-off set to 0 (Python) against eps (MATLAB, :62).
Also measures the effect on mtv, the one in-package consumer, which
renormalizes the density on both sides.
"""

import numpy as np
from scipy import fftpack
from scipy.optimize import brentq, minimize_scalar

import pyvbmc
from pyvbmc.stats import kde_1d

print("pyvbmc.__file__ =", pyvbmc.__file__)


# ---------------------------------------------------------------- MATLAB side
def m_histc(data, edges):
    """histc(data, edges): bin k counts [edges(k), edges(k+1)); the last bin
    counts data == edges(end).  Values outside are dropped."""
    counts = np.zeros(len(edges))
    idx = np.searchsorted(edges, data, side="right") - 1
    for j, x in zip(idx, data):
        if x == edges[-1]:
            counts[len(edges) - 1] += 1
        elif 0 <= j < len(edges) - 1:
            counts[j] += 1
    return counts


def m_dct1d(data):
    """kde1d.m dct1d."""
    n = len(data)
    weight = np.concatenate(
        ([1.0], 2.0 * np.exp(-1j * np.arange(1, n) * np.pi / (2 * n)))
    )
    reordered = np.concatenate((data[0::2], data[::-1][0::2]))
    return np.real(weight * np.fft.fft(reordered))


def m_idct1d(data):
    """kde1d.m idct1d."""
    n = len(data)
    weights = n * np.exp(1j * np.arange(n) * np.pi / (2 * n))
    d = np.real(np.fft.ifft(weights * data))
    out = np.zeros(n)
    out[0::2] = d[: n // 2]
    out[1::2] = d[n - 1 : n // 2 - 1 : -1]
    return out


def m_fixed_point(t, N, I, a2):
    ell = 7
    f = (
        2
        * np.pi ** (2 * ell)
        * np.sum(I**ell * a2 * np.exp(-I * np.pi**2 * t))
    )
    for s in range(ell - 1, 1, -1):
        K0 = np.prod(np.arange(1, 2 * s, 2)) / np.sqrt(2 * np.pi)
        const = (1 + (1 / 2) ** (s + 1 / 2)) / 3
        time = (2 * const * K0 / N / f) ** (2 / (3 + 2 * s))
        f = (
            2
            * np.pi ** (2 * s)
            * np.sum(I**s * a2 * np.exp(-I * np.pi**2 * time))
        )
    return t - (2 * N * np.sqrt(np.pi) * f) ** (-2 / 5)


def m_root(f, N):
    """kde1d.m root: fzero on [0, tol], doubling; fminbnd on |f| as the last
    resort."""
    N = 50 * (N <= 50) + 1050 * (N >= 1050) + N * ((N < 1050) & (N > 50))
    tol = 1e-12 + 0.01 * (N - 50) / 1000
    used_fminbnd = False
    while True:
        try:
            t = brentq(f, 0, tol)
            return t, used_fminbnd
        except Exception:
            tol = min(tol * 2, 0.1)
        if tol == 0.1:
            r = minimize_scalar(
                lambda x: abs(f(x)), bounds=(0, 0.1), method="bounded"
            )
            return r.x, True


def matlab_kde1d(data, n, MIN, MAX):
    n = int(2 ** np.ceil(np.log2(n)))
    R = MAX - MIN
    dx = R / (n - 1)
    xmesh = MIN + np.arange(n) * dx
    N = len(np.unique(data))
    initial = m_histc(data, xmesh) / N
    initial = initial / np.sum(initial)
    a = m_dct1d(initial)
    I = np.arange(1, n) ** 2.0
    a2 = (a[1:] / 2) ** 2
    t_star, used_fminbnd = m_root(lambda t: m_fixed_point(t, N, I, a2), N)
    a_t = a * np.exp(-np.arange(n) ** 2 * np.pi**2 * t_star / 2)
    density = m_idct1d(a_t) / R
    bandwidth = np.sqrt(t_star) * R
    density[density < 0] = np.finfo(float).eps
    return bandwidth, density, xmesh, t_star, used_fminbnd


# ---------------------------------------------------------------- comparison
rng = np.random.default_rng(20260921)
data = rng.standard_normal(3000)
lo, hi = data.min() - 0.5, data.max() + 0.5

for n in [2**10, 2**12, 2**14]:
    py_d, py_x, py_bw = kde_1d(data, n, np.array([lo]), np.array([hi]))
    m_bw, m_d, m_x, m_t, _ = matlab_kde1d(data, n, lo, hi)
    dx = py_x[1] - py_x[0]
    print(f"\n== n = {n} ==")
    print("  grids identical:", np.allclose(py_x, m_x, rtol=0, atol=1e-12))
    print(
        "  sum(density)*dx  python:",
        np.sum(py_d) * dx,
        "  matlab:",
        np.sum(m_d) * dx,
        "  n/(n-1) =",
        n / (n - 1),
    )
    print(
        "  bandwidth python:",
        float(np.ravel(py_bw)[0]),
        " matlab:",
        float(np.ravel(m_bw)[0]),
        " rel diff:",
        abs(float(np.ravel(py_bw)[0]) - float(np.ravel(m_bw)[0]))
        / float(np.ravel(m_bw)[0]),
    )
    # half-bin shift: compare python(x) with matlab(x) and matlab shifted
    resid_plain = np.max(np.abs(py_d - m_d))
    m_shift_r = np.interp(py_x, m_x + dx / 2, m_d)
    m_shift_l = np.interp(py_x, m_x - dx / 2, m_d)
    print("  max|py - matlab|          :", resid_plain)
    print("  max|py - matlab(+dx/2)|   :", np.max(np.abs(py_d - m_shift_r)))
    print("  max|py - matlab(-dx/2)|   :", np.max(np.abs(py_d - m_shift_l)))
    # mean of the estimate, a direct measure of the shift
    print(
        "  grid mean python:",
        np.sum(py_x * py_d) / np.sum(py_d),
        " matlab:",
        np.sum(m_x * m_d) / np.sum(m_d),
        " sample mean:",
        data.mean(),
        " dx/2 =",
        dx / 2,
    )

print("\n== (ii) the binning rule, sample by sample ==")
edges = np.linspace(0.0, 1.0, 5)  # dx = 0.25
dxx = edges[1] - edges[0]
for s in [0.20, 0.26, 0.30, 0.40, 0.49, 0.51]:
    sample = np.array([s])
    m_idx = int(np.argmax(m_histc(sample, edges)))
    py_idx = int(np.floor((s - (edges[0] - 0.5 * dxx)) / dxx))
    print(
        f"  x={s:.2f}: histc -> grid {edges[m_idx]:.2f} (index {m_idx}); "
        f"nearest -> grid {edges[py_idx]:.2f} (index {py_idx}); "
        f"same={m_idx == py_idx}"
    )

print("\n== (iii) the fallback ==")
from pyvbmc.stats.kde_1d import _root, _scott_rule_1d

print("  PyVBMC: _root returns None when tol >= 1, kde_1d then takes")
print("          Scott's rule (kde_1d.py:247-250) and t_star = (bw/delta)^2")
print("  MATLAB: root() calls fminbnd(@(x)abs(f(x)),0,.1) once tol == .1")
no_root = _root(lambda t, c: 1.0, 500, args=(0,))
print("  _root on a function with no root in [0, tol]:", no_root)
m_t, m_used = m_root(lambda t: 1.0, 500)
print(
    "  the MATLAB transcription on the same function: t =",
    m_t,
    " via fminbnd:",
    m_used,
)
print("  Scott's rule on the sample:", _scott_rule_1d(data))

print("\n== (iv) negative round-off ==")
wide = np.concatenate(
    (
        rng.standard_normal(100),
        rng.standard_normal(100) * 2 + 35,
        rng.standard_normal(100) + 55,
    )
)
w_lo, w_hi = wide.min() - 5, wide.max() + 5
py_w, py_wx, _ = kde_1d(wide, 2**14, np.array([w_lo]), np.array([w_hi]))
_, m_w, _, _, _ = matlab_kde1d(wide, 2**14, w_lo, w_hi)
print("  widely separated modes, n = 2^14:")
print(
    "   #(python density == 0):",
    int(np.sum(py_w == 0.0)),
    "  #(matlab density == eps):",
    int(np.sum(m_w == np.finfo(float).eps)),
    "  eps =",
    np.finfo(float).eps,
)
print("   the floor differs by eps = 2.2e-16 on those grid points alone")

print("\n== the effect on mtv: both sides renormalize ==")
from scipy.integrate import trapezoid

n = 2**13
py_d, py_x, _ = kde_1d(data, n, np.array([lo]), np.array([hi]))
m_bw, m_d, m_x, _, _ = matlab_kde1d(data, n, lo, hi)
dx = py_x[1] - py_x[0]
py_norm = py_d / (trapezoid(py_d) * dx)
m_norm = m_d / ((np.sum(m_d) - 0.5 * (m_d[0] + m_d[-1])) * dx)
print(
    "  after renormalization, max|py - matlab| =",
    np.max(np.abs(py_norm - m_norm)),
)
print(
    "  total variation between the two normalized estimates =",
    0.5 * trapezoid(np.abs(py_norm - m_norm)) * dx,
)
true = np.exp(-0.5 * py_x**2) / np.sqrt(2 * np.pi)
print(
    "  TV(python KDE, true N(0,1))  =",
    0.5 * trapezoid(np.abs(py_norm - true)) * dx,
)
print(
    "  TV(matlab KDE, true N(0,1))  =",
    0.5 * trapezoid(np.abs(m_norm - true)) * dx,
)
