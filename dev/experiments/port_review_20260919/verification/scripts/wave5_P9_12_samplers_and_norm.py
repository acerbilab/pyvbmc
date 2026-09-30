"""P9-12. The first-question answers of both reports: does every sampler draw
from the density of its class, on both sides?

Settles: (a) that the rejection samplers' bounding height `pdf(0.5*(u+v))`
is the maximum of the density for every valid argument; (b) the smooth-box
sampler's mixture weights against the exact masses; (c) the normalizers of
the four families by quadrature with parameters that differ across
dimensions, and their separability; (d) Kolmogorov-Smirnov tests of every
sampler, PyVBMC's and my MATLAB transcription's, against closed-form CDFs
that I derive here and check against quadrature; (e) that two of the four
PyVBMC classes reproduce my transcription of their MATLAB counterparts to
rounding; (f) that `_init_log_joint` combines prior and likelihood as
`lpostfun.m` does, noisy branch included, in the caller's original
coordinates.
"""

import matlab_transcription as M
import numpy as np
from scipy.integrate import quad
from scipy.stats import kstest
from scipy.stats import norm as spnorm

import pyvbmc
from pyvbmc import VBMC
from pyvbmc.priors import SmoothBox, SplineTrapezoidal, Trapezoidal, UniformBox

print("pyvbmc.__file__ =", pyvbmc.__file__)
print()

SQ2PI = np.sqrt(2 * np.pi)


# ------------------------------------------------- closed-form CDFs (mine)
def cdf_unif(x, a, b):
    return np.clip((np.asarray(x, float) - a) / (b - a), 0.0, 1.0)


def cdf_trapez(x, a, u, v, b):
    x = np.asarray(x, float)
    h = 2.0 / (b - a + v - u)
    F = np.zeros_like(x)
    m = (x >= a) & (x < u)
    F[m] = h * (x[m] - a) ** 2 / (2 * (u - a))
    m = (x >= u) & (x < v)
    F[m] = h * (u - a) / 2 + h * (x[m] - u)
    m = (x >= v) & (x < b)
    F[m] = 1 - h * (b - x[m]) ** 2 / (2 * (b - v))
    F[x >= b] = 1.0
    return F


def cdf_spline(x, a, u, v, b):
    x = np.asarray(x, float)
    h = 2.0 / (b - a + v - u)
    F = np.zeros_like(x)
    m = (x >= a) & (x < u)
    z = (x[m] - a) / (u - a)
    F[m] = h * (u - a) * (z**3 - z**4 / 2)
    m = (x >= u) & (x < v)
    F[m] = h * (u - a) / 2 + h * (x[m] - u)
    m = (x >= v) & (x < b)
    z = (b - x[m]) / (b - v)
    F[m] = 1 - h * (b - v) * (z**3 - z**4 / 2)
    F[x >= b] = 1.0
    return F


def cdf_smooth(x, a, b, s):
    x = np.asarray(x, float)
    C = 1.0 / ((b - a) + SQ2PI * s)
    F = np.empty_like(x)
    m = x < a
    F[m] = C * s * SQ2PI * spnorm.cdf((x[m] - a) / s)
    m = (x >= a) & (x <= b)
    F[m] = C * s * SQ2PI * 0.5 + C * (x[m] - a)
    m = x > b
    F[m] = 1 - C * s * SQ2PI * (1 - spnorm.cdf((x[m] - b) / s))
    return F


print(
    "--- my closed-form CDFs checked against quadrature of the code's "
    "own pdf ---"
)
checks = [
    (
        "UniformBox(-3,2)",
        UniformBox(-3.0, 2.0),
        cdf_unif,
        (-3.0, 2.0),
        -3.0,
        [-2.0, 0.0, 1.5],
    ),
    (
        "Trapezoidal(0,.25,.75,1)",
        Trapezoidal(0.0, 0.25, 0.75, 1.0),
        cdf_trapez,
        (0.0, 0.25, 0.75, 1.0),
        0.0,
        [0.1, 0.25, 0.5, 0.9],
    ),
    (
        "SplineTrapezoidal(0,.25,.75,1)",
        SplineTrapezoidal(0.0, 0.25, 0.75, 1.0),
        cdf_spline,
        (0.0, 0.25, 0.75, 1.0),
        0.0,
        [0.1, 0.25, 0.5, 0.9],
    ),
    (
        "SmoothBox(0,1,1)",
        SmoothBox(0.0, 1.0, 1.0),
        cdf_smooth,
        (0.0, 1.0, 1.0),
        -40.0,
        [-2.0, 0.0, 0.5, 1.0, 3.0],
    ),
]
for name, p, F, args, lo, pts in checks:
    worst = 0.0
    for t in pts:
        num = quad(
            lambda z: float(p.pdf(np.array([[z]]), keepdims=False)[0]),
            lo,
            t,
            limit=400,
        )[0]
        worst = max(worst, abs(num - float(F(np.array([t]), *args)[0])))
    print(f"{name:32s} max |F_quad - F_closed| = {worst:.3e}")
print()

# ---------------------------------------------------------------- (a)
print("--- (a) is pdf(0.5*(u+v)) the maximum of the density? ---")
rng = np.random.default_rng(11)
for cls in (Trapezoidal, SplineTrapezoidal):
    worst = 0.0
    for _ in range(300):
        pts = np.sort(rng.uniform(-20, 20, 4))
        if len(set(pts)) < 4:
            continue
        a, u, v, b = pts
        p = cls(a, u, v, b)
        x0 = 0.5 * (u + v)
        ymax = float(p.pdf(np.array([[x0]]), keepdims=False)[0])
        grid = np.linspace(a, b, 20001).reshape(-1, 1)
        with np.errstate(all="ignore"):
            dens = p.pdf(grid, keepdims=False)
        worst = max(worst, (float(np.max(dens)) - ymax) / max(ymax, 1e-300))
    print(
        f"{cls.__name__:20s} worst relative excess of max(density) over "
        f"the envelope, 300 random argument sets: {worst:.3e}"
    )
for cls in (Trapezoidal, SplineTrapezoidal):
    for args in (
        (0.0, 1e-8, 2e-8, 10.0),
        (0.0, 9.999, 9.9999, 10.0),
        (-1e6, -1.0, 1.0, 1e6),
    ):
        p = cls(*args)
        a, u, v, b = args
        ymax = float(p.pdf(np.array([[0.5 * (u + v)]]), keepdims=False)[0])
        grid = np.linspace(a, b, 200001).reshape(-1, 1)
        with np.errstate(all="ignore"):
            dens = p.pdf(grid, keepdims=False)
        print(
            f"  {cls.__name__:20s} {args}: envelope {ymax:.6g}, "
            f"grid max {float(np.max(dens)):.6g}"
        )
print()

# ---------------------------------------------------------------- (b)
print("--- (b) smooth-box mixture weights against the exact masses ---")
for a, b, s in (
    (0.0, 1.0, 1.0),
    (-5.0, 5.0, 0.1),
    (0.0, 1e-3, 5.0),
    (0.0, 100.0, 1.0),
):
    nf = 1 + (b - a) / (SQ2PI * s)
    C = 1.0 / ((b - a) + SQ2PI * s)
    n = 400000
    draws = SmoothBox(a, b, s).sample(n, rng=np.random.default_rng(5)).ravel()
    print(
        f"a={a:>6}, b={b:>6}, s={s:>5}: code tail {0.5 / nf:.6f} / true "
        f"{C * s * SQ2PI / 2:.6f}; code plateau {(nf - 1) / nf:.6f} / "
        f"true {C * (b - a):.6f}; empirical plateau "
        f"{float(np.mean((draws >= a) & (draws <= b))):.6f}, left tail "
        f"{float(np.mean(draws < a)):.6f}"
    )
print()

# ---------------------------------------------------------------- (c)
print("--- (c) normalizers by quadrature, parameters differing per dim ---")
a3 = np.array([-3.0, 0.0, 10.0])
u3 = np.array([-2.0, 0.5, 11.0])
v3 = np.array([1.0, 0.75, 18.0])
b3 = np.array([2.0, 4.0, 20.0])
s3 = np.array([0.2, 1.0, 5.0])


def one_d(name, d):
    if name == "UniformBox":
        return UniformBox(a3[d], b3[d])
    if name == "Trapezoidal":
        return Trapezoidal(a3[d], u3[d], v3[d], b3[d])
    if name == "SplineTrapezoidal":
        return SplineTrapezoidal(a3[d], u3[d], v3[d], b3[d])
    return SmoothBox(a3[d], b3[d], s3[d])


fams = [
    ("UniformBox", UniformBox(a3, b3), a3, b3),
    ("Trapezoidal", Trapezoidal(a3, u3, v3, b3), a3, b3),
    ("SplineTrapezoidal", SplineTrapezoidal(a3, u3, v3, b3), a3, b3),
    ("SmoothBox", SmoothBox(a3, b3, s3), a3 - 40 * s3, b3 + 40 * s3),
]
for name, p, lo, hi in fams:
    prod = 1.0
    for d in range(3):
        f1 = one_d(name, d)
        val, _ = quad(
            lambda t: float(f1.pdf(np.array([[t]]), keepdims=False)[0]),
            lo[d],
            hi[d],
            limit=400,
        )
        prod *= val
    xs = [np.linspace(lo[d], hi[d], 9) for d in range(3)]
    pts = np.stack(np.meshgrid(*xs, indexing="ij"), -1).reshape(-1, 3)
    with np.errstate(all="ignore"):
        joint = p.pdf(pts, keepdims=False)
    sep = np.ones(len(pts))
    for d in range(3):
        with np.errstate(all="ignore"):
            sep *= one_d(name, d).pdf(pts[:, [d]], keepdims=False)
    print(
        f"{name:20s} product of the three marginal integrals = {prod!r}; "
        f"max |joint - product| = {float(np.nanmax(np.abs(joint - sep))):.3e}"
    )
print()

# ---------------------------------------------------------------- (d)
print("--- (d) KS of every sampler against the closed-form CDF, n=200000 ---")
n = 200000
settings = [
    (
        "UniformBox(-3,2)",
        UniformBox(-3.0, 2.0),
        cdf_unif,
        (-3.0, 2.0),
        lambda r: M.munifboxrnd(np.array([[-3.0]]), np.array([[2.0]]), n, r),
    ),
    (
        "Trapezoidal(0,.25,.75,1)",
        Trapezoidal(0.0, 0.25, 0.75, 1.0),
        cdf_trapez,
        (0.0, 0.25, 0.75, 1.0),
        lambda r: M.mtrapezrnd(
            np.array([[0.0]]),
            np.array([[0.25]]),
            np.array([[0.75]]),
            np.array([[1.0]]),
            n,
            r,
        ),
    ),
    (
        "Trapezoidal(0,.01,.02,10)",
        Trapezoidal(0.0, 0.01, 0.02, 10.0),
        cdf_trapez,
        (0.0, 0.01, 0.02, 10.0),
        lambda r: M.mtrapezrnd(
            np.array([[0.0]]),
            np.array([[0.01]]),
            np.array([[0.02]]),
            np.array([[10.0]]),
            n,
            r,
        ),
    ),
    (
        "SplineTrapezoidal(0,.25,.75,1)",
        SplineTrapezoidal(0.0, 0.25, 0.75, 1.0),
        cdf_spline,
        (0.0, 0.25, 0.75, 1.0),
        lambda r: M.msplinetrapezrnd(
            np.array([[0.0]]),
            np.array([[0.25]]),
            np.array([[0.75]]),
            np.array([[1.0]]),
            n,
            r,
        ),
    ),
    (
        "SplineTrapezoidal(0,.01,.02,10)",
        SplineTrapezoidal(0.0, 0.01, 0.02, 10.0),
        cdf_spline,
        (0.0, 0.01, 0.02, 10.0),
        lambda r: M.msplinetrapezrnd(
            np.array([[0.0]]),
            np.array([[0.01]]),
            np.array([[0.02]]),
            np.array([[10.0]]),
            n,
            r,
        ),
    ),
    (
        "SmoothBox(0,1,1)",
        SmoothBox(0.0, 1.0, 1.0),
        cdf_smooth,
        (0.0, 1.0, 1.0),
        lambda r: M.msmoothboxrnd(
            np.array([[0.0]]), np.array([[1.0]]), np.array([[1.0]]), n, r
        ),
    ),
    (
        "SmoothBox(-5,5,.1)",
        SmoothBox(-5.0, 5.0, 0.1),
        cdf_smooth,
        (-5.0, 5.0, 0.1),
        lambda r: M.msmoothboxrnd(
            np.array([[-5.0]]), np.array([[5.0]]), np.array([[0.1]]), n, r
        ),
    ),
]
for name, p, F, args, mrnd in settings:
    py = p.sample(n, rng=np.random.default_rng(13)).ravel()
    rpy = kstest(py, lambda t: F(t, *args))
    ml = mrnd(np.random.default_rng(17)).ravel()
    rml = kstest(ml, lambda t: F(t, *args))
    print(
        f"{name:32s} PyVBMC D={rpy.statistic:.5f} p={rpy.pvalue:.3f}   "
        f"MATLAB D={rml.statistic:.5f} p={rml.pvalue:.3f}"
    )
print()

print("--- per-column KS for a D=3 prior with different parameters ---")
for name, p, F, argsets in (
    (
        "Trapezoidal",
        Trapezoidal(a3, u3, v3, b3),
        cdf_trapez,
        [(a3[d], u3[d], v3[d], b3[d]) for d in range(3)],
    ),
    (
        "SmoothBox",
        SmoothBox(a3, b3, s3),
        cdf_smooth,
        [(a3[d], b3[d], s3[d]) for d in range(3)],
    ),
):
    draws = p.sample(100000, rng=np.random.default_rng(19))
    out = []
    for d in range(3):
        r = kstest(draws[:, d], lambda t, ar=argsets[d]: F(t, *ar))
        out.append(f"d{d}: D={r.statistic:.5f} p={r.pvalue:.3f}")
    print(f"{name:20s} " + "   ".join(out))
print()

# ---------------------------------------------------------------- (e)
print("--- (e) PyVBMC classes against my MATLAB transcription ---")
rng = np.random.default_rng(23)
for name, pycls, mfun, order in (
    ("Trapezoidal", Trapezoidal, M.mtrapezlogpdf, "auvb"),
    ("SmoothBox", SmoothBox, M.msmoothboxlogpdf, "abs"),
    ("UniformBox", UniformBox, M.munifboxlogpdf, "ab"),
    ("SplineTrapezoidal", SplineTrapezoidal, M.msplinetrapezlogpdf, "auvb"),
):
    worst = 0.0
    n_inf = 0
    for _ in range(60):
        D = int(rng.integers(1, 4))
        if order == "auvb":
            pts = np.sort(rng.uniform(-10, 10, (4, D)), axis=0)
            args = tuple(pts)
        elif order == "ab":
            pts = np.sort(rng.uniform(-10, 10, (2, D)), axis=0)
            args = tuple(pts)
        else:
            pts = np.sort(rng.uniform(-10, 10, (2, D)), axis=0)
            args = (pts[0], pts[1], rng.uniform(0.05, 3.0, D))
        p = pycls(*args)
        xs = rng.uniform(-12, 12, (300, D))
        piv = np.vstack([np.atleast_2d(q) for q in args])
        xs = np.vstack(
            [xs, piv, np.nextafter(piv, -np.inf), np.nextafter(piv, np.inf)]
        )
        with np.errstate(all="ignore"):
            ypy = p.log_pdf(xs, keepdims=False)
            ym = mfun(xs, *[np.atleast_2d(q) for q in args])
        assert np.array_equal(np.isneginf(ypy), np.isneginf(ym)), name
        both_inf = np.isneginf(ypy)
        n_inf += int(both_inf.sum())
        fin = ~both_inf
        if fin.any():
            worst = max(worst, float(np.max(np.abs(ypy[fin] - ym[fin]))))
    print(
        f"{name:20s} max |PyVBMC - MATLAB| on finite entries = "
        f"{worst:.3e};  -inf sets coincide exactly ({n_inf} entries)"
    )
print()

# ---------------------------------------------------------------- (f)
print("--- (f) _init_log_joint against lpostfun.m ---")
prior = UniformBox(np.array([-4.0, -4.0]), np.array([4.0, 4.0]))
lb, ub = np.array([[-10.0, -10.0]]), np.array([[10.0, 10.0]])
plb, pub = np.array([[-2.0, -2.0]]), np.array([[2.0, 2.0]])
x0 = np.array([[0.3, -0.7]])
ll = lambda t: -0.5 * float(np.sum(np.asarray(t) ** 2))
v = VBMC(ll, x0, lb, ub, plb, pub, prior=prior, options={"display": "off"})
pt = np.array([0.3, -0.7])
print(
    "noiseless: log_joint(x) =",
    v.log_joint(pt),
    " ll(x) + log_prior(x) =",
    ll(pt) + prior.log_pdf(pt).item(),
)
print(
    "  prior term -2*log 8 =",
    -2 * np.log(8.0),
    " measured",
    prior.log_pdf(pt).item(),
)

lln = lambda t: (-0.5 * float(np.sum(np.asarray(t) ** 2)), 0.37)
vn = VBMC(
    lln,
    x0,
    lb,
    ub,
    plb,
    pub,
    prior=prior,
    options={"display": "off", "specify_target_noise": True},
)
val, sd = vn.log_joint(pt)
print(
    "noisy: log_joint(x) =",
    val,
    " sd =",
    sd,
    "| equals ll + log_prior:",
    bool(np.isclose(val, lln(pt)[0] + prior.log_pdf(pt).item())),
    "| sd untouched:",
    sd == 0.37,
)
print()

print("--- which coordinates does the prior see? ---")
seen = []
base = UniformBox(np.array([-4.0, -4.0]), np.array([4.0, 4.0]))


def probe_log_prior(x):
    seen.append(np.array(x).ravel().copy())
    return base.log_pdf(x).item()


vp = VBMC(
    ll,
    x0,
    lb,
    ub,
    plb,
    pub,
    log_prior=probe_log_prior,
    options={"display": "off"},
)
vp.function_logger(vp.x0[0])
print("x0 original   :", vp.x0_orig.ravel())
print("x0 transformed:", vp.x0.ravel())
print("prior saw     :", seen[-1])
print(
    "=> the prior is evaluated in the caller's original coordinates:",
    bool(np.allclose(seen[-1], vp.x0_orig.ravel())),
)
