"""P9-13. The four MATLAB-side defects reported at the end of the comparison
report (M1 to M4).

Settles: M1 -- `msmoothboxrnd.m:59,66` index the pivot matrices with a
logical (n,1) vector, which MATLAB resolves as column-major linear
indexing, so every dimension's tails are anchored at dimension 1's pivots;
demonstrated with my transcription, against the repaired `a(idx,d)` and
against PyVBMC.  M2 to M4 are settled by reading and by the arithmetic of
M2.
"""

import matlab_transcription as M
import numpy as np

import pyvbmc
from pyvbmc.priors import SmoothBox

print("pyvbmc.__file__ =", pyvbmc.__file__)
print()

print("--- M1: how MATLAB resolves `a(idx)` for a logical (n,1) index ---")
print("`a` is n-by-D (msmoothboxrnd.m:42 repmats it to n rows).")
print("`idx` is the logical (n,1) vector `u < 0.5`.")
print("A logical index whose shape is not that of the array is treated as")
print("linear indexing in column-major order, so `a(idx)` selects the")
print("elements at linear positions 1..n, i.e. COLUMN 1 of `a`, at the rows")
print("where idx is true.  The result is sum(idx)-by-1, which matches the")
print("shape of `z1`, so no size error is raised.  Line 72 of the plateau")
print("branch uses the correct `a(idx,d)`.")
print()

a = np.array([[-5.0, 0.0, 10.0]])
b = np.array([[-4.0, 1.0, 12.0]])
sigma = np.array([[0.2, 0.3, 0.4]])
n = 200000

r_bad = M.msmoothboxrnd(
    a, b, sigma, n, np.random.default_rng(31), repaired=False
)
r_fix = M.msmoothboxrnd(
    a, b, sigma, n, np.random.default_rng(31), repaired=True
)
r_py = SmoothBox(a.ravel(), b.ravel(), sigma.ravel()).sample(
    n, rng=np.random.default_rng(32)
)

exact = []
for d in range(3):
    aa, bb, ss = a[0, d], b[0, d], sigma[0, d]
    C = 1.0 / ((bb - aa) + np.sqrt(2 * np.pi) * ss)
    # plateau contribution + the two tails (each a half normal of sd ss)
    mean = C * (bb**2 - aa**2) / 2
    mean += C * ss * np.sqrt(2 * np.pi) / 2 * (aa - ss * np.sqrt(2 / np.pi))
    mean += C * ss * np.sqrt(2 * np.pi) / 2 * (bb + ss * np.sqrt(2 / np.pi))
    exact.append(mean)
print(
    "exact marginal means      :",
    np.array2string(np.array(exact), precision=4),
)
print(
    "msmoothboxrnd as written  :",
    np.array2string(r_bad.mean(axis=0), precision=4),
)
print(
    "msmoothboxrnd with a(idx,d):",
    np.array2string(r_fix.mean(axis=0), precision=4),
)
print(
    "PyVBMC SmoothBox.sample   :",
    np.array2string(r_py.mean(axis=0), precision=4),
)
print()
for d in range(3):
    left = r_bad[:, d][r_bad[:, d] < a[0, d]]
    print(
        f"  dim {d}: as written, the left tail sits below "
        f"{left.max():.4f} (its pivot should be {a[0, d]}; "
        f"column 1's pivot is {a[0, 0]})"
    )
print()
print("PyVBMC's smooth_box.py:132,138 use self.a[d] / self.b[d]:")
print(
    "  max |PyVBMC mean - exact| =",
    float(np.max(np.abs(r_py.mean(axis=0) - np.array(exact)))),
)
print(
    "  max |repaired MATLAB mean - exact| =",
    float(np.max(np.abs(r_fix.mean(axis=0) - np.array(exact)))),
)
print()
print("test/test_pdfs_vbmc.m:101 histograms r(:,1) only, so the MATLAB")
print("test cannot see it (column 1 is the one column that is right).")
print()

print("--- M2: mtrapezlogpdf with u == a ---")
with np.errstate(all="ignore"):
    lnf = np.log(0.5) + np.log(1.0 - 0.0 + 0.75 - 0.0) + np.log(0.0 - 0.0)
    print(f"  lnf = log(0.5) + log(b-a+v-u) + log(u-a) = {lnf}")
    print(f"  plateau branch  log(u-a) - lnf = {np.log(0.0) - lnf}")
    print(
        "  mtrapezlogpdf(0.5, 0, 0, 0.75, 1) =",
        M.mtrapezlogpdf([[0.5]], 0.0, 0.0, 0.75, 1.0),
    )
    print(
        "  mtrapezlogpdf(0.9, 0, 0, 0.75, 1) =",
        M.mtrapezlogpdf([[0.9]], 0.0, 0.0, 0.75, 1.0),
        "(the right ramp is NaN too)",
    )
    print(
        "  msplinetrapezlogpdf(0.5, 0, 0, 0.75, 1) =",
        M.msplinetrapezlogpdf([[0.5]], 0.0, 0.0, 0.75, 1.0),
        "(the spline has no such factor)",
    )
print()

print("--- M3: the nargin test of mtrapezrnd / msplinetrapezrnd ---")
print("Both declare five inputs (a,u,v,b,n) and line 22 is")
print("  if nargin < 3 || isempty(n)")
print("With the documented four-argument call, nargin = 4, so `nargin < 3`")
print("is false and MATLAB must evaluate `isempty(n)`, where `n` was never")
print("supplied: the call errors at line 22 itself, before the else branch")
print("and before line 38's repmat(a,[n,1]).  munifboxrnd.m:19 tests")
print("`nargin < 3` for three declared inputs and msmoothboxrnd.m:26 tests")
print("`nargin < 4` for four; both are right.")
print()

print("--- M4: error identifiers that name another function ---")
print("  mtrapezlogpdf.m:40   error('mtrapezpdf:SizeError', ...)")
print("  msmoothboxlogpdf.m:27 error('msmoothboxpdf:NonPositiveSigma', ...)")
print("  msmoothboxlogpdf.m:38 error('msmoothboxpdf:SizeError', ...)")
print("  msmoothboxlogpdf.m:47 error('msmoothboxpdf:OrderError', ...)")
print("  munifboxrnd.m:33     error('munifboxpdf:OrderError', ...)")
print("  (munifboxlogpdf.m and msplinetrapezlogpdf.m name themselves.)")
