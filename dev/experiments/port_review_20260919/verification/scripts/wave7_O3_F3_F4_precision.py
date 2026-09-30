"""O3 F3 and F4: precision of the bounded transforms near the middle
(student4 forward) and near the bounds (all three), against references
that compute the distance to each bound separately.

References (independent of the code under test):
- logit: u = log(x-a) - log(b-x), inverse distance to the upper bound
  (b-a) * expit(-y) = (b-a) / (1 + exp(y));
- probit: scipy.special.ndtri of the distance to the nearer bound, with the
  sign; inverse distance (b-a) * ndtr(-y);
- student4: the tail quantile from p = min(z, 1-z) (Shaw's closed form is
  accurate for small p), the centre from r = 2 sin(asin(2z-1)/3),
  t = 2r/sqrt(1-r^2); the lower-tail CDF as (1+r)^2 (2-r)/4 with
  1 + r = 4 / (s (s + |t|)), s = sqrt(t^2 + 4), for t < 0.
The transformer is built with plausible bounds equal to the hard bounds, so
that mu = 0 and delta = 1 and u is the raw quantile.
"""
import gpyreg as gpr
import numpy as np
from scipy.special import expit, ndtr, ndtri

import pyvbmc
from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.parameter_transformer.parameter_transformer import (
    _inverse_student4,
    _student4,
)

print("pyvbmc:", pyvbmc.__file__)
print("gpyreg:", gpr.__file__)


def pt_of(a, b, kind):
    lb, ub = np.array([[a]]), np.array([[b]])
    return ParameterTransformer(1, lb, ub, lb, ub, transform_type=kind)


def s4_quantile_ref(p_lower, p_upper):
    """Student-t(4) quantile from the two tail masses, each exact."""
    p = min(p_lower, p_upper)
    if p > 0.1:
        r = 2 * np.sin(np.arcsin(p_lower - p_upper) / 3)  # 2z-1 = pl - pu
        return 2 * r / np.sqrt((1 - r) * (1 + r))
    al = 2 * np.sqrt(p) * np.sqrt(1 - p)
    q = np.cos(np.arccos(al) / 3) / al
    t = 2 * np.sqrt(q - 1)
    return -t if p_lower < p_upper else t


def s4_lower_tail_ref(t):
    s = np.sqrt(t * t + 4)
    one_plus_r = 4 / (s * (s + abs(t)))
    r = -1 + one_plus_r
    return one_plus_r**2 * (2 - r) / 4


print("\n== F4, upper bound, |b| << b-a ==")
for a, b, frac in (
    (-1000.0, 1.0, 1e-8),
    (-1.0, 0.0, 1e-12),
    (-1.0, 0.0, 1e-20),
    (10.0, 11.0, 1e-12),
):
    for kind in ("logit", "probit"):
        pt = pt_of(a, b, kind)
        d = frac * (b - a)
        x = b - d
        d_exact = b - x  # exact where x is within a factor 2 of b, or b = 0
        pl, pu = (x - a) / (b - a), d_exact / (b - a)
        if kind == "logit":
            u_ref = np.log(x - a) - np.log(d_exact)
        else:
            u_ref = -ndtri(pu)
        u = pt(np.array([[x]]))[0, 0]
        x_back = pt.inverse(np.array([[u_ref]]))[0, 0]
        if kind == "logit":
            d_ref = (b - a) * expit(-u_ref)
        else:
            d_ref = (b - a) * ndtr(-u_ref)
        print(
            f"  [{a:g},{b:g}] {kind:6s} (b-x)/(b-a)={frac:.0e}: u={u:.10g} "
            f"u_ref={u_ref:.10g} rel err {abs(u / u_ref - 1):.1e}; "
            f"inverse(u_ref): b-x={b - x_back:.4g} vs {d_ref:.4g} "
            f"(rel err {abs((b - x_back) / d_ref - 1):.1e})"
        )

print("\n== F4, lower bound, zero and nonzero ==")
worst = {}
for a, b in ((0.0, 1.0), (0.0, 7.0), (3.0, 5.0), (-5.0, 2.0)):
    for kind in ("logit", "probit"):
        pt = pt_of(a, b, kind)
        ef, ei = 0.0, 0.0
        # With a nonzero bound, x - a cannot be smaller than the spacing of
        # the numbers near a; the check stays above it (x - a is then exact,
        # and the inverse is compared with the representable neighbour).
        fracs = (
            (1e-3, 1e-8, 1e-12, 1e-50, 1e-200)
            if a == 0
            else (1e-3, 1e-6, 1e-9, 1e-12)
        )
        for frac in fracs:
            d = frac * (b - a)
            x = a + d
            dx = x - a
            if kind == "logit":
                u_ref = np.log(dx) - np.log(b - x)
                d_back_ref = (b - a) * expit(u_ref)
            else:
                u_ref = ndtri(dx / (b - a))
                d_back_ref = (b - a) * ndtr(u_ref)
            u = pt(np.array([[x]]))[0, 0]
            x_back = pt.inverse(np.array([[u_ref]]))[0, 0]
            ef = max(ef, abs(u / u_ref - 1))
            if a == 0:
                ei = max(ei, abs((x_back - a) / d_back_ref - 1))
            else:  # error in units of the spacing of the numbers near x
                ei = max(ei, abs((x_back - a) - d_back_ref) / np.spacing(x))
        unit = "rel err of x-a" if a == 0 else "error in ulps of x"
        print(
            f"  [{a:g},{b:g}] {kind:6s}: max forward rel err {ef:.1e}, max "
            f"inverse {unit} {ei:.1e} (x-a from {fracs[0]:.0e} to "
            f"{fracs[-1]:.0e} of b-a)"
        )

print("\n== F4, student4 inverse at a zero lower bound (rel err of x-a) ==")
pt = pt_of(0.0, 1.0, "student4")
for t in (-10.0, -100.0, -1e3, -1e4):
    x = pt.inverse(np.array([[t]]))[0, 0]
    ref = s4_lower_tail_ref(t)
    print(
        f"  t={t:>8g}: x={x:.6e} ref={ref:.6e} rel err {abs(x / ref - 1):.1e}"
    )
x0 = 1e-18
u = pt(np.array([[x0]]))[0, 0]
print(
    f"  round trip of x=1e-18: u={u:.6g} (ref {s4_quantile_ref(x0, 1 - x0):.6g}),"
    f" back to {pt.inverse(np.array([[u]]))[0, 0]:.3e}"
)

print("\n== F3, student4 forward near z = 1/2 (absolute error of t) ==")
for dz in (1e-3, 1e-6, 1e-7, 1e-8, 1e-9):
    z = 0.5 + dz
    t = _student4(np.array([z]))[0]
    tr = s4_quantile_ref(z, 1 - z)
    print(
        f"  z-1/2={dz:.0e}: code {t:.7e} ref {tr:.7e} abs err {abs(t - tr):.1e}"
    )
zz = 0.5 + np.linspace(-1e-3, 1e-3, 200001)
tt = _student4(zz)
tr = np.array([s4_quantile_ref(z, 1 - z) for z in zz])
i = np.argmax(np.abs(tt - tr))
print(
    f"  max abs error over 1/2 +- 1e-3: {np.abs(tt - tr).max():.2e} at z-1/2={zz[i] - 0.5:.2e}"
)
# round trip through the transformer on [0, 1]
back = pt.inverse(pt(zz[:, None]))[:, 0]
print(
    f"  round trip x->u->x near the midpoint, max |dx|/(b-a): {np.abs(back - zz).max():.2e}"
)
# first-order relation: error in u of ~1.6e-8 at the density 3/8 of t
print(f"  Student-t(4) density at 0: 3/8; so dx ~ (3/8) * du")

print(
    "\n== Q2: log|J| as a function of u, against an independent evaluation =="
)
for kind in ("logit", "probit", "student4"):
    pt = pt_of(-1.0, 0.0, kind)
    uu = np.linspace(-40, 40, 8001)[:, None]
    lj = pt.log_abs_det_jacobian(uu)
    y = uu[:, 0]
    if kind == "logit":
        ref = -np.logaddexp(0, y) - np.logaddexp(0, -y)
    elif kind == "probit":
        ref = -0.5 * np.log(2 * np.pi) - 0.5 * y**2
    else:
        ref = np.log(0.375) - 2.5 * np.log1p(y**2 / 4)
    print(
        f"  {kind:8s}: max |log|J| - ref| over u in [-40,40]: {np.abs(lj - ref).max():.1e}"
    )
