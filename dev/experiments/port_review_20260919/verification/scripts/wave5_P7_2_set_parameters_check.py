"""P7-2: the positivity check of set_parameters(raw_flag=False) slices theta[-check_idx:].

Settles which entries of theta the check at variational_posterior.py:1086
actually inspects, for general (D, K) and for each of the sixteen
combinations of the four optimize_* flags, and checks the two reports'
examples (D=3,K=4; D=2,K=2; D=3,K=2; D=1,K=4; D=5,K=5).
"""

import itertools

import numpy as np

import pyvbmc
from pyvbmc.variational_posterior import VariationalPosterior

print("pyvbmc.__file__ =", pyvbmc.__file__)


def layout(D, K, om, os_, ol, ow):
    """Blocks of theta in order, as set_parameters/get_parameters lay them out."""
    blocks = []
    i = 0
    if om:
        blocks.append(("mu", i, i + D * K))
        i += D * K
    if os_:
        blocks.append(("sigma", i, i + K))
        i += K
    if ol:
        blocks.append(("lambd", i, i + D))
        i += D
    if ow:
        blocks.append(("w", i, i + K))
        i += K
    return blocks, i


def coded_slice(D, K, om, os_, ol, ow, n):
    check_idx = 0
    if ow:
        check_idx -= K
    if ol:
        check_idx -= D
    if os_:
        check_idx -= K
    start = -check_idx  # what the code writes
    return (
        check_idx,
        start,
        list(range(start, n)) if start != 0 else list(range(n)),
    )


def which(blocks, idx):
    for name, a, b in blocks:
        if a <= idx < b:
            return name
    return "?"


print("\n== every (D, K) of the two reports, all four flags True ==")
for D, K in [(3, 4), (2, 2), (3, 2), (1, 4), (5, 5)]:
    blocks, n = layout(D, K, True, True, True, True)
    check_idx, start, inspected = coded_slice(D, K, True, True, True, True, n)
    intended = list(range(n + check_idx, n))
    names = sorted({which(blocks, i) for i in inspected})
    print(
        f"D={D} K={K}: len(theta)={n} blocks={blocks} check_idx={check_idx} "
        f"coded theta[{start}:] -> indices {inspected[0] if inspected else '-'}"
        f"..{inspected[-1] if inspected else '-'} ({names}); "
        f"intended {intended[0]}..{intended[-1]} "
        f"({sorted({which(blocks, i) for i in intended})})"
    )

print("\n== all sixteen flag combinations at D=3, K=4 ==")
D, K = 3, 4
for om, os_, ol, ow in itertools.product([True, False], repeat=4):
    blocks, n = layout(D, K, om, os_, ol, ow)
    if n == 0:
        print(f"mu={om} sigma={os_} lambd={ol} w={ow}: empty theta")
        continue
    check_idx, start, inspected = coded_slice(D, K, om, os_, ol, ow, n)
    intended = (
        list(range(n + check_idx, n)) if check_idx != 0 else list(range(n))
    )
    cov_coded = sorted({which(blocks, i) for i in inspected})
    cov_int = sorted({which(blocks, i) for i in intended})
    print(
        f"mu={int(om)} sigma={int(os_)} lambd={int(ol)} w={int(ow)}: n={n} "
        f"check_idx={check_idx} coded=[{start}:] -> {cov_coded}  "
        f"intended -> {cov_int}  same={inspected == intended}"
    )

print("\n== live behavior: spurious rejection, D=3 K=4, negative means ==")
vp = VariationalPosterior(3, 4, rng=np.random.default_rng(0))
theta_ok = np.concatenate(
    [
        np.array([-1.0, 2.0, -3.0] * 4),  # mu, some negative
        np.full(4, 0.5),  # sigma > 0
        np.full(3, 1.0),  # lambd > 0
        np.full(4, 0.25),  # w > 0
    ]
)
try:
    vp.set_parameters(theta_ok, raw_flag=False)
    print("D=3,K=4 valid theta with negative mu: accepted")
except ValueError as e:
    print(
        "D=3,K=4 valid theta with negative mu: REJECTED ->",
        " ".join(str(e).split()),
    )

print("\n== live behavior: missed rejection, D=2 K=2, sigma = -0.5 ==")
vp2 = VariationalPosterior(2, 2, rng=np.random.default_rng(0))
theta_bad = np.concatenate(
    [
        np.array([1.0, 1.0, 2.0, 2.0]),  # mu
        np.array([-0.5, 0.9]),  # sigma, one negative
        np.array([1.0, 1.0]),  # lambd
        np.array([0.5, 0.5]),  # w
    ]
)
try:
    vp2.set_parameters(theta_bad, raw_flag=False)
    print("D=2,K=2 negative sigma: ACCEPTED, vp.sigma =", vp2.sigma)
except ValueError as e:
    print("D=2,K=2 negative sigma: rejected")

print(
    "\n== no scale or weight optimized: check_idx == 0 -> theta[-0:] is all =="
)
vp3 = VariationalPosterior(2, 2, rng=np.random.default_rng(0))
vp3.optimize_sigma = vp3.optimize_lambd = vp3.optimize_weights = False
try:
    vp3.set_parameters(np.array([-1.0, 1.0, 1.0, 1.0]), raw_flag=False)
    print("only mu optimized, negative mu: accepted")
except ValueError:
    print("only mu optimized, negative mu: REJECTED (whole vector inspected)")
