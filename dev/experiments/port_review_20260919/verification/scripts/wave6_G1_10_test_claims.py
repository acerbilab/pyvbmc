"""Checks two claims the reports make about gpyreg's own tests.

  (a) G1 internal, test-adequacy note 1: of the four bound sets of
      ``test_uuinv`` (``gpyreg/testing/test_smoothbox.py:95``), only
      ``[0,10,30,66]`` is said to distinguish the length-weighted mixture
      the body implements from the equal-weight mixture the docstring
      names.  The script evaluates the test's three assertions under both
      conventions for every bound set.
  (b) G1 internal, F11 and test-adequacy note 2: substituting 0.3 for the
      constant of ``test_constant_trace_in_a_free_parameter_fails_the_
      diagnostics`` (``test_slice_sample.py:118``) and of
      ``test_fixed_parameter_is_left_out_of_the_diagnostics`` (``:133``)
      makes their assertions fail.  The script re-runs the bodies of both
      tests with 0.3 in place of 0 and of 1.

Run from anywhere.
"""

import gpyreg as gpr
import numpy as np
from gpyreg.f_min_fill import uuinv
from gpyreg.slice_sample import SliceSampler

print("gpyreg:", gpr.__file__)

print("\n=== (a) test_uuinv's bound sets under the two conventions ===")
p = np.linspace(0, 1, 1000)
Bs = [[0, 0, 30, 66], [0, 10, 30, 66], [0, 10, 66, 66], [0, 10, 10, 66]]
ws = [0, 0.6, 1]


def docstring_sample(p, B, w):
    """Draw from w*U(B1,B2) + ((1-w)/2)*(U(B0,B1) + U(B2,B3)) by inverting
    that CDF; an empty tail becomes an atom at its hard bound."""
    B0, B1, B2, B3 = [float(v) for v in B]
    x = np.zeros_like(p)
    half = (1 - w) / 2
    lo, mid, hi = B1 - B0, B2 - B1, B3 - B2
    m = p <= half
    x[m] = B0 + (p[m] / half * lo if half > 0 else 0.0)
    m2 = (p > half) & (p <= half + w)
    x[m2] = B1 + ((p[m2] - half) / w * mid if w > 0 else 0.0)
    m3 = p > half + w
    x[m3] = B2 + ((p[m3] - half - w) / half * hi if half > 0 else 0.0)
    return x


for B, w in [(B, w) for B in Bs for w in ws]:
    for label, x in (
        ("body", uuinv(p.copy(), B, w)),
        ("docstring", docstring_sample(p.copy(), B, w)),
    ):
        a1 = np.sum((x <= B[2]) & (B[1] <= x)) / np.size(x)
        a2 = np.sum((x < B[1]) & (B[0] <= x)) / np.size(x)
        a3 = np.sum((x <= B[3]) & (B[2] < x)) / np.size(x)
        denom = (B[1] - B[0]) + (B[3] - B[2])
        e1 = w
        e2 = (1 - w) * (B[1] - B[0]) / denom if denom else np.nan
        e3 = (1 - w) * (B[3] - B[2]) / denom if denom else np.nan
        # exactly the test's comparison: np.isclose(..., atol=1e-3)
        ok = [
            bool(np.isclose(a, e, atol=1e-3))
            for a, e in ((a1, e1), (a2, e2), (a3, e3))
        ]
        if label == "body":
            print(f"  B={B} w={w}")
        print(
            f"    {label:10s} masses ({a1:.4f}, {a2:.4f}, {a3:.4f})"
            f"  expected ({e1:.4f}, {e2:.4f}, {e3:.4f})"
            f"  asserts pass: {ok}"
        )

print("\n=== (b) the diagnostics tests with 0.3 in place of 0 and 1 ===")
options = {"display": "off"}
s = SliceSampler(
    lambda x: -0.5 * float(np.sum(np.asarray(x) ** 2)),
    np.array([0.0]),
    options=options,
    rng=np.random.default_rng(0),
)
diagnose = s._SliceSampler__diagnose
for c in (0.0, 0.3):
    samples = np.full((20, 1), c)
    with np.errstate(all="ignore"):
        flag, R, eff = diagnose(samples)
    print(
        f"  constant trace of {c}: exit_flag={flag}  R={R}  eff_N={eff}"
        f"  -> test's asserts (flag==-3, isnan(R), isnan(eff_N)): "
        f"{[flag == -3, bool(np.all(np.isnan(R))), bool(np.all(np.isnan(eff)))]}"
    )

from scipy.stats import multivariate_normal

for c in (1.0, 0.3):
    rv = multivariate_normal(np.zeros(2), np.eye(2))
    slicer = SliceSampler(
        rv.logpdf,
        np.array([0.0, c]),
        LB=np.array([-np.inf, c]),
        UB=np.array([np.inf, c]),
        options=options,
        rng=np.random.default_rng(2),
    )
    with np.errstate(all="ignore"):
        res = slicer.sample(200)
    print(
        f"  fixed parameter at {c}: R={np.round(res['R'], 6)}"
        f"  eff_N={np.round(res['eff_N'], 4)}  exit_flag={res['exit_flag']}"
    )
    print(
        f"    test's asserts (isnan(R[1]), isnan(eff_N[1]), exit_flag==1): "
        f"{[bool(np.isnan(res['R'][1])), bool(np.isnan(res['eff_N'][1])), res['exit_flag'] == 1]}"
    )

print("\n=== (c) the zero-width freeze at a small burn-in ===")
# Bears on the two rows the orchestrator verifies itself (comparison F1 and
# M6): with floor(burn/2) == 1 gpyreg accumulates exactly one iteration, so
# the variance estimate is exactly 0 and np.maximum(..., 0) gives a width of
# 0 for every coordinate; the chain then cannot move again.
for burn in (2, 3, 4, 5, 15):
    slicer = SliceSampler(
        lambda x: -0.5 * float(np.sum(np.asarray(x) ** 2)),
        np.array([0.0, 0.0]),
        widths=[1.0, 1.0],
        options={"display": "off", "diagnostics": False},
        rng=np.random.default_rng(0),
    )
    res = slicer.sample(6, burn=burn)
    uniq = len(np.unique(res["samples"], axis=0))
    print(
        f"  burn={burn:3d}: widths after burn-in = {np.round(slicer.widths, 6)}"
        f"  distinct recorded samples = {uniq}/6"
    )
