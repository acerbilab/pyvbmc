"""Settles the claims about ``gpyreg.slice_sample.SliceSampler``.

  * ``base_widths`` is copied before the infinities are replaced
    (``slice_sample.py:186`` against ``:188``), the validity check runs
    after (``:213-220``), and the geometric-mean recombination at
    ``:590-592`` puts the infinity back      -- internal F10.
    ``slicesamplebnd.m:168`` / ``:176`` / ``:198`` / ``:377`` (read at
    vbmc 396d649) have exactly the same order, so the claim is checked
    against MATLAB by reading, not only in Python.
  * the Metropolis option key is spelled ``metopolis_rnd`` (``:238``)
                                                        -- internal M-B;
    MATLAB's ``metropolis_flag`` is an OR of the two options
    (``slicesamplebnd.m:189``) where gpyreg uses an AND (``:239-241``).
  * ``thin`` and ``burn`` are not floored, where
    ``slicesamplebnd.m:184-185`` applies ``floor``     -- comparison M7.
  * a free parameter whose chain is constant is flagged only when the
    constant's mean is exact (``:683-685``)             -- internal F11.
  * the split of an odd number of samples drops the last sample
    (``:667-672``) and ``__effective_n`` pairs from lag 0 (``:926-930``)
                                                        -- internal M-J.

Run from anywhere.
"""

import gpyreg as gpr
import numpy as np
from gpyreg.slice_sample import SliceSampler

print("gpyreg:", gpr.__file__)

target = lambda x: -0.5 * float(np.sum(np.asarray(x) ** 2))

print("\n=== 1. an infinite user width (internal F10) ===")
s = SliceSampler(
    target,
    np.array([0.0, 0.0]),
    widths=[np.inf, 1.0],
    LB=[-np.inf, -5.0],
    UB=[np.inf, 5.0],
    options={"display": "off", "diagnostics": False},
    rng=np.random.default_rng(0),
)
print("  constructed without complaint")
print("  widths      =", s.widths)
print("  base_widths =", s.base_widths)
res = s.sample(20, burn=10)
print("  widths after burn-in =", s.widths)
print(
    "  non-finite recorded samples:",
    int(np.sum(~np.isfinite(res["samples"]))),
    "of",
    res["samples"].size,
)
print("  MATLAB slicesamplebnd.m:168 copies basewidths before :176 replaces")
print(
    "  the infinities and :198 asserts -- the same order, so MATLAB shares this."
)

print("\n=== 2. the Metropolis option key (internal M-B) ===")
for key in ("metropolis_rnd", "metopolis_rnd"):
    s = SliceSampler(
        target,
        np.array([0.0]),
        widths=[1.0],
        options={
            "display": "off",
            "diagnostics": False,
            "metropolis_pdf": lambda x: 1.0,
            key: lambda: np.array([0.0]),
        },
        rng=np.random.default_rng(0),
    )
    print(
        f"  options key {key!r:18s} -> metropolis_flag = {s.metropolis_flag}"
    )
s = SliceSampler(
    target,
    np.array([0.0]),
    widths=[1.0],
    options={
        "display": "off",
        "diagnostics": False,
        "metropolis_pdf": lambda x: 1.0,
    },
    rng=np.random.default_rng(0),
)
print(
    "  only metropolis_pdf given -> metropolis_flag =",
    s.metropolis_flag,
    "(MATLAB's OR at slicesamplebnd.m:189 would set it and then the assert"
    " at :208 would error)",
)

print("\n=== 3. a non-integral burn / thin (comparison M7) ===")
s = SliceSampler(
    target,
    np.array([0.0]),
    widths=[1.0],
    options={"display": "off", "diagnostics": False},
    rng=np.random.default_rng(0),
)
try:
    s.sample(4, burn=2.5)
    print("  burn=2.5 -> completed")
except Exception as exc:  # noqa: BLE001
    print(f"  burn=2.5 -> {type(exc).__name__}: {exc}")
s2 = SliceSampler(
    target,
    np.array([0.0]),
    widths=[1.0],
    options={"display": "off", "diagnostics": False},
    rng=np.random.default_rng(0),
)
try:
    r = s2.sample(4, thin=1.5, burn=2)
    print("  thin=1.5 -> completed, samples shape", r["samples"].shape)
except Exception as exc:  # noqa: BLE001
    print(f"  thin=1.5 -> {type(exc).__name__}: {exc}")

print("\n=== 4. a constant chain in a free parameter (internal F11) ===")
s = SliceSampler(
    target,
    np.array([0.0, 0.0]),
    widths=[1.0, 1.0],
    LB=[-5.0, -5.0],
    UB=[5.0, 5.0],
    options={"display": "off", "diagnostics": True},
    rng=np.random.default_rng(0),
)
diagnose = s._SliceSampler__diagnose
rng = np.random.default_rng(2)
moving = rng.standard_normal((20, 1))
for c in (0.0, 1.0, 0.5, 0.25, 2.0, -7.0, 0.3, 1e-3, float(np.log(10))):
    trace = np.hstack([np.full((20, 1), c), moving])
    with np.errstate(all="ignore"):
        flag, R, eff = diagnose(trace)
    print(
        f"  constant {c!r:20s} exit_flag={flag:3d} R[0]={R[0]!r:22s} "
        f"eff_N[0]={eff[0]!r}"
    )

print("\n=== 5. the split of an odd number of samples (internal M-J) ===")
N = 21
half = N // 2
print(
    f"  N={N}: first half rows 0..{half - 1}, second half rows "
    f"{half}..{2 * half - 1}; row {2 * half} (the last) is dropped"
)
print(
    "  BDA3 / Stan drop the middle sample, i.e. the second half is rows"
    f" {N - half}..{N - 1}"
)

print(
    "\n=== 6. the autocorrelation pairing of __effective_n (internal M-J) ==="
)
src = (
    open(
        gpr.slice_sample.__file__
        if hasattr(gpr, "slice_sample")
        else SliceSampler.__module__
        and __import__("gpyreg.slice_sample", fromlist=["x"]).__file__,
        encoding="utf-8",
    )
    .read()
    .splitlines()
)
for ln in range(921, 932):
    print(f"  {ln + 1:3d}: {src[ln]}")
print(
    "  rho[0] = 1 by construction (line 914), so the first pair"
    " (rho[0], rho[1]) is accepted unless rho[1] <= -1"
)
