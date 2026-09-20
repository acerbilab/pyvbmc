"""Settles P8c F1, F2 and F3 by my own transcription of warpvars_vbmc.m.

F1 - ``_to_unit_interval``'s nudge of an interior point whose unit-interval
     image rounds to 0 or 1; MATLAB feeds the rounded value straight to the
     bounded map and returns -/+Inf.
F2 - the inverse clamp: PyVBMC uses ``np.nextafter``, MATLAB
     ``a + eps(a)`` / ``b - eps(b)`` (``shared/warpvars_vbmc.m:457-459``),
     which is one ulp further inside where the clamp moves toward zero
     across a power-of-two boundary.
F3 - the student4 inverse groups ``((3/8)*x)/sqrt(...)`` in MATLAB
     (``:451``) and ``(3/8)*(x/sqrt(...))`` in PyVBMC (``:606``).

The transcription below was written from the MATLAB source read in this
session, not from the reviewer's script.
"""

import numpy as np
from scipy.special import erfc, erfcinv

import pyvbmc
from pyvbmc.parameter_transformer import ParameterTransformer

print("pyvbmc.__file__ =", pyvbmc.__file__)
print()


# --- transcription of shared/warpvars_vbmc.m (types 3, 12, 13) -----------
def m_direct(x, a, b, mu, delta, t):
    z = (x - a) / (b - a)
    if t == 3:
        y = np.log(z / (1 - z))
    elif t == 12:
        y = -np.sqrt(2) * erfcinv(2 * z)
    elif t == 13:
        aa = np.sqrt(4 * z * (1 - z))
        q = np.cos(np.arccos(aa) / 3) / aa
        y = np.sign(z - 0.5) * (2 * np.sqrt(q - 1))
    return (y - mu) / delta


def m_inverse(y, a, b, mu, delta, t):
    x = y * delta + mu
    if t == 3:
        f = 1.0 / (1 + np.exp(-x))
        out = a + (b - a) * f
    elif t == 12:
        out = a + (b - a) * (0.5 * erfc(-x / np.sqrt(2)))
    elif t == 13:
        t2 = x**2
        f = 0.5 + 3 / 8 * x / np.sqrt(1 + t2 / 4) * (
            1 - t2 / (1 + t2 / 4) / 12
        )
        out = a + (b - a) * f
    # Force to stay within bounds: a + eps(a), b - eps(b)
    a2 = a + abs(np.spacing(a)) if np.isfinite(a) else a
    b2 = b - abs(np.spacing(b)) if np.isfinite(b) else b
    return np.minimum(np.maximum(out, a2), b2)


def m_logp(y, a, b, mu, delta, t):
    yy = y * delta + mu
    if t == 3:
        z = -np.log1p(np.exp(-yy))
        p = np.log(b - a) + (-yy + 2 * z)
    elif t == 12:
        p = np.log(b - a) + (-0.5 * np.log(2 * np.pi) - 0.5 * yy**2)
    elif t == 13:
        p = np.log(b - a) + (np.log(3 / 8) - 5 / 2 * np.log1p(yy**2 / 4))
    return p + np.log(delta)


TYPES = {"logit": 3, "probit": 12, "student4": 13}

# --- bulk agreement on interior points -----------------------------------
print("=== bulk check: direct / inverse / log-Jacobian, 20000 points ===")
rng = np.random.default_rng(20260920)
for a, b, pa, pb in [
    (-5.0, 5.0, -1.0, 2.0),
    (0.0, 1.0, 0.2, 0.8),
    (1.0, 3.0, 1.5, 2.5),
]:
    for name, t in TYPES.items():
        pt = ParameterTransformer(
            1,
            np.array([[a]]),
            np.array([[b]]),
            np.array([[pa]]),
            np.array([[pb]]),
            transform_type=name,
        )
        mu, delta = pt.mu[0], pt.delta[0]
        # reference mu/delta from the transcription
        m_mu = 0.5 * (
            m_direct(pa, a, b, 0.0, 1.0, t) + m_direct(pb, a, b, 0.0, 1.0, t)
        )
        m_de = m_direct(pb, a, b, 0.0, 1.0, t) - m_direct(
            pa, a, b, 0.0, 1.0, t
        )
        x = rng.uniform(a, b, size=(20000, 1))
        py = pt(x)
        my = m_direct(x, a, b, m_mu, m_de, t)
        u = rng.uniform(-6, 6, size=(20000, 1))
        pxi = pt.inverse(u)
        mxi = m_inverse(u, a, b, m_mu, m_de, t)
        plj = pt.log_abs_det_jacobian(u)
        mlj = m_logp(u, a, b, m_mu, m_de, t).ravel()
        print(
            f"  [{a},{b}] {name:9s} mu/delta match:"
            f" {np.isclose(mu, m_mu) and np.isclose(delta, m_de)}"
            f" | direct differing: {np.mean(py != my):.4%}"
            f" | inverse differing: {np.mean(pxi != mxi):.4%}"
            f" | max rel inv err: {np.max(np.abs(pxi - mxi) / (abs(b - a))):.2e}"
            f" | logJ differing: {np.mean(plj != mlj):.4%}"
        )
print()

# --- F1: the nudge -------------------------------------------------------
print("=== F1: interior point whose unit-interval image rounds to 0 or 1 ===")
cases = [
    (-5.0, 5.0, np.nextafter(5.0, -np.inf), "x = nextafter(ub, -inf)"),
    (0.0, 3.0, 5e-324, "x = 5e-324, lb = 0"),
    (0.0, 1.0, 5e-324, "x = 5e-324, lb = 0, ub = 1"),
    (1.0, 3.0, np.nextafter(1.0, np.inf), "x = nextafter(lb, +inf)"),
]
for a, b, x, label in cases:
    z = (x - a) / (b - a)
    print(
        f"  {label}: a={a}, b={b}, raw z = {z!r}, x==a: {x == a},"
        f" x==b: {x == b}"
    )
    for name, t in TYPES.items():
        pt = ParameterTransformer(
            1, np.array([[a]]), np.array([[b]]), transform_type=name
        )
        with np.errstate(all="ignore"):
            py = pt(np.array([[x]]))[0, 0]
            my = m_direct(np.array([[x]]), a, b, 0.0, 1.0, t)[0, 0]
        print(f"      {name:9s} PyVBMC {py!r:24s} MATLAB {my!r}")
print()

# --- F2: the clamp -------------------------------------------------------
print("=== F2: inverse clamp, eps(bound) vs nextafter ===")
for bound in [1.0, 2.0, 4.0, 3.0, 5.0, 11.0, 0.0, -1.0, -2.0, -5.0]:
    m_lo = bound + abs(np.spacing(bound))
    p_lo = np.nextafter(bound, np.inf)
    m_hi = bound - abs(np.spacing(bound))
    p_hi = np.nextafter(bound, -np.inf)
    print(
        f"  bound {bound:6}: lower clamp MATLAB {m_lo!r:24s}"
        f" PyVBMC {p_lo!r:24s} same={m_lo == p_lo}"
        f" | upper clamp MATLAB {m_hi!r:24s} PyVBMC {p_hi!r:24s}"
        f" same={m_hi == p_hi}"
    )
print()

# --- F3: the student4 grouping -------------------------------------------
print("=== F3: student4 inverse grouping ===")
u = np.concatenate([np.linspace(-40, 40, 200001), rng.normal(0, 5, 200000)])
t2 = u**2
matlab = 0.5 + 3 / 8 * u / np.sqrt(1 + t2 / 4) * (1 - t2 / (1 + t2 / 4) / 12)
python = 0.5 + (3 / 8) * (u / np.sqrt(1 + t2 / 4)) * (
    1 - t2 / (1 + t2 / 4) / 12
)
diff = matlab != python
print(f"  differing values: {np.mean(diff):.2%}")
rel = np.abs(matlab - python) / np.maximum(np.abs(matlab), 1e-300)
k = int(np.argmax(rel))
print(
    f"  max relative difference {rel[k]:.3e} at u = {u[k]:.4f}:"
    f" MATLAB {matlab[k]!r} PyVBMC {python[k]!r}"
)
print(
    f"  max absolute difference {np.max(np.abs(matlab - python)):.3e}"
    " (of a unit-interval value; multiply by ub-lb)"
)
