"""Settles question 1 of both G1 briefs: which mixture ``uuinv`` inverts.

``gpyreg/f_min_fill.py:193-214`` documents
``w U(B1,B2) + (1-w)/2 (U(B0,B1) + U(B2,B3))``, i.e. equal weight on the
two tails.  The body (``:218-250``) builds ``L = (B1-B0) + (B3-B2)`` and
splits ``1-w`` over the union of the tails in proportion to their lengths.
``fminfill.m:133-136`` (read at vbmc 396d649) carries the same sentence as
a comment above a body with the same length weighting, so the two sides
agree with each other and both disagree with the wording.
(G1 internal F12 + Q1 / G1 comparison Q1.)

The script
  1. inverts each candidate CDF, written from its definition, and measures
     ``F(uuinv(p)) - p`` over a grid of ``p``;
  2. runs a line-for-line transcription of MATLAB's ``uuinv`` beside
     gpyreg's on the same inputs;
  3. measures the per-coordinate and joint probability of the plausible
     box in a PyVBMC-shaped design (G1 internal M-A);
  4. checks the out-of-range guard on all three return paths
     (G1 internal M-C).

Run from anywhere.
"""

import gpyreg as gpr
import numpy as np
from gpyreg.f_min_fill import f_min_fill, uuinv

print("gpyreg:", gpr.__file__)


def cdf_docstring(x, B, w):
    """CDF of w*U(B1,B2) + ((1-w)/2)*(U(B0,B1) + U(B2,B3))."""
    B0, B1, B2, B3 = B
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    lo, hi = B1 - B0, B3 - B2
    # lower tail
    m = x < B1
    if lo > 0:
        out[m] = (1 - w) / 2 * np.clip((x[m] - B0) / lo, 0, 1)
    else:
        out[m] = 0.0
    # plateau
    m = (x >= B1) & (x < B2)
    mid = B2 - B1
    out[m] = (1 - w) / 2 + (w * (x[m] - B1) / mid if mid > 0 else 0.0)
    # upper tail
    m = x >= B2
    if hi > 0:
        out[m] = (
            (1 - w) / 2 + w + (1 - w) / 2 * np.clip((x[m] - B2) / hi, 0, 1)
        )
    else:
        out[m] = (1 - w) / 2 + w
    return out


def cdf_body(x, B, w):
    """CDF of the length-weighted mixture the body implements."""
    B0, B1, B2, B3 = B
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    lo, hi = B1 - B0, B3 - B2
    L = lo + hi
    mid = B2 - B1
    m = x < B1
    out[m] = (1 - w) * (x[m] - B0) / L if L > 0 else 0.0
    m = (x >= B1) & (x < B2)
    out[m] = (1 - w) * lo / L if L > 0 else 0.0
    out[m] = out[m] + (w * (x[m] - B1) / mid if mid > 0 else 0.0)
    m = x >= B2
    base = ((1 - w) * lo / L if L > 0 else 0.0) + w
    out[m] = base + ((1 - w) * (x[m] - B2) / L if L > 0 else 0.0)
    return np.clip(out, 0.0, 1.0)


def uuinv_matlab(p, B, w):
    """Transcription of fminfill.m:132-171 (uuinv, post-1d1f20d)."""
    p = np.asarray(p, dtype=float)
    x = np.zeros(p.shape)
    L = B[3] - B[0] + B[1] - B[2]
    if w == 1:
        return p * (B[2] - B[1]) + B[1]
    if L == 0:
        i1 = p <= (1 - w) / 2
        x[i1] = B[0]
        if w != 0:
            i2 = (p <= (1 - w) / 2 + w) & ~i1
            x[i2] = (p[i2] - (1 - w) / 2) * (B[2] - B[1]) / w + B[1]
        i3 = p > (1 - w) / 2 + w
        x[i3] = B[3]
        return x
    i1 = p <= (1 - w) * (B[1] - B[0]) / L
    x[i1] = B[0] + p[i1] * L / (1 - w)
    i2 = (p <= (1 - w) * (B[1] - B[0]) / L + w) & ~i1
    if w != 0:
        x[i2] = (p[i2] - (1 - w) * (B[1] - B[0]) / L) * (B[2] - B[1]) / w + B[
            1
        ]
    i3 = p > (1 - w) * (B[1] - B[0]) / L + w
    x[i3] = (p[i3] - w - (1 - w) * (B[1] - B[0]) / L) * L / (1 - w) + B[2]
    x[(p < 0) | (p > 1)] = np.nan
    return x


print("\n=== 1. which mixture uuinv inverts ===")
grid = np.linspace(1e-9, 1 - 1e-9, 4001)
bound_sets = [
    ("[-3,-1,0.5,4] asymmetric", [-3.0, -1.0, 0.5, 4.0]),
    ("[-3,-1,1,3] symmetric", [-3.0, -1.0, 1.0, 3.0]),
    ("[-1,-1,0.5,4] PLB == LB", [-1.0, -1.0, 0.5, 4.0]),
    ("[-3,-1,4,4] PUB == UB", [-3.0, -1.0, 4.0, 4.0]),
    ("[-3,0.5,0.5,4] zero plateau", [-3.0, 0.5, 0.5, 4.0]),
    ("[-1,-1,2,2] both tails empty", [-1.0, -1.0, 2.0, 2.0]),
]
for label, B in bound_sets:
    line = [f"  {label:30s}"]
    for w in (0.0, 0.25, 0.5, 0.9, 1.0):
        x = uuinv(grid.copy(), B, w)
        e_doc = np.max(np.abs(cdf_docstring(x, B, w) - grid))
        e_body = np.max(np.abs(cdf_body(x, B, w) - grid))
        line.append(f"w={w:.2f}: doc {e_doc:.2e} body {e_body:.2e}")
    print("\n     ".join(line))

print("\n=== 2. gpyreg's uuinv against a transcription of MATLAB's ===")
worst = 0.0
for label, B in bound_sets:
    for w in (0.0, 0.1, 0.25, 0.5, 0.9, 1.0):
        a = uuinv(grid.copy(), B, w)
        b = uuinv_matlab(grid.copy(), B, w)
        d = np.max(np.abs(a - b))
        worst = max(worst, d)
print(
    f"  max |gpyreg - MATLAB transcription| over all bound sets and w:"
    f" {worst:.3e}"
)
# and the out-of-range inputs
for w in (0.0, 0.5, 1.0):
    for B in ([-3.0, -1.0, 0.5, 4.0], [-1.0, -1.0, 2.0, 2.0]):
        p = np.array([-0.1, 0.5, 1.1])
        print(
            f"  w={w} B={B}: gpyreg {uuinv(p.copy(), B, w)}"
            f"  MATLAB {uuinv_matlab(p.copy(), B, w)}"
        )

print("\n=== 3. tail masses at w = 0.25, [-1,-1,0.5,4] ===")
B = [-1.0, -1.0, 0.5, 4.0]
p = np.linspace(1e-9, 1 - 1e-9, 200001)
x = uuinv(p.copy(), B, 0.25)
below = float(np.mean(x < B[1]))
above = float(np.mean(x > B[2]))
print(f"  body:      below PLB {below:.4f}  above PUB {above:.4f}")
print(
    f"  docstring: below PLB {(1 - 0.25) / 2:.4f}  above PUB "
    f"{(1 - 0.25) / 2:.4f}"
)

print("\n=== 4. a PyVBMC-shaped design (internal M-A) ===")
D = 2
rng = np.random.default_rng(5)
X = rng.uniform(-3.0, 3.0, size=(40, D))
y = np.sum(1.0 + np.sin(X), axis=1, keepdims=True)
gp = gpr.GP(
    D=D,
    covariance=gpr.covariance_functions.SquaredExponential(),
    mean=gpr.mean_functions.NegativeQuadratic(),
    noise=gpr.noise_functions.GaussianNoise(constant_add=True),
)
gp.X, gp.y = X, y
gp.set_bounds(None)
gp.set_priors(
    {
        "covariance_log_lengthscale": (
            "student_t",
            (0.0, 0.5 * np.log(1e3), 3),
        ),
        "covariance_log_outputscale": None,
        "noise_log_scale": (
            "student_t",
            (np.log(np.sqrt(1e-5)), np.log(10), 3),
        ),
        "mean_const": None,
        "mean_location": None,
        "mean_log_scale": None,
    }
)
gp.set_bounds(gp.get_recommended_bounds())
LB, UB = gp.lower_bounds, gp.upper_bounds
cbi = gp.covariance.get_bounds_info(X, y)
nbi = gp.noise.get_bounds_info(X, y)
mbi = gp.mean.get_bounds_info(X, y)
PLB = np.concatenate([cbi["PLB"], nbi["PLB"], mbi["PLB"]])
PUB = np.concatenate([cbi["PUB"], nbi["PUB"], mbi["PUB"]])
PLB = np.minimum(np.maximum(PLB, LB), UB)
PUB = np.maximum(np.minimum(PUB, UB), LB)
n_vars = LB.size
design, _ = f_min_fill(
    lambda h: 0.0,
    np.reshape(np.minimum(np.maximum((PLB + PUB) / 2, LB), UB), (1, -1)),
    LB,
    UB,
    PLB,
    PUB,
    gp.hyper_priors,
    1024,
    "sobol",
    rng=np.random.default_rng(9),
)
w = 0.5 ** (1 / n_vars)
inside = (design >= PLB) & (design <= PUB)
names = [
    "ell_0",
    "ell_1",
    "sf",
    "sn",
    "m0",
    "loc_0",
    "loc_1",
    "logsc_0",
    "logsc_1",
]
print(f"  n_vars = {n_vars}, w = {w:.6f}")
for i, nm in enumerate(names[:n_vars]):
    has_prior = np.isfinite(gp.hyper_priors["mu"][i]) and np.isfinite(
        gp.hyper_priors["sigma"][i]
    )
    branch = "student_t" if has_prior else "uuinv"
    print(
        f"    {nm:8s} {branch:10s} inside = {inside[:, i].mean():.4f}"
        f"  tails: lo {float(np.mean(design[:, i] < PLB[i])):.4f}"
        f" hi {float(np.mean(design[:, i] > PUB[i])):.4f}"
        f"  tail lengths: lo {PLB[i] - LB[i]:.3f} hi {UB[i] - PUB[i]:.3f}"
    )
print(
    f"  fraction of design points inside the plausible box in EVERY"
    f" coordinate: {np.all(inside, axis=1).mean():.4f} (intent: 0.5)"
)
