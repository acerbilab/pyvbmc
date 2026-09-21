"""P7-5: get_parameters rescales lambd, sigma and w in place.

Settles what the in-place rescaling changes on the caller's object, whether
the density is preserved (scales and weights separately), whether the
rescaling is idempotent, and whether it interacts with the _mode cache.
Compared with misc/get_vptheta.m (which returns a rescaled struct as a
second output) and misc/rescale_params.m (which also removes vp.mode).
"""

import numpy as np

import pyvbmc
from pyvbmc.variational_posterior import VariationalPosterior

print("pyvbmc.__file__ =", pyvbmc.__file__)

D, K = 2, 3
xq = np.array([[0.3, -0.2], [1.0, 1.0]])


def make_vp(lambd, sigma, w, opt_w=True):
    vp = VariationalPosterior(
        D, K, x0=np.zeros((1, D)), rng=np.random.default_rng(5)
    )
    vp.mu = np.array([[0.0, 1.0, -1.0], [0.0, -1.0, 1.0]])
    vp.lambd = np.array(lambd, dtype=float).reshape(D, 1)
    vp.sigma = np.array(sigma, dtype=float).reshape(1, K)
    vp.w = np.array(w, dtype=float).reshape(1, K)
    vp.optimize_weights = opt_w
    return vp


print("\n== scale rescaling: density preserved? ==")
vp = make_vp([2.0, 4.0], [0.3, 0.5, 0.7], [1 / 3, 1 / 3, 1 / 3])
before = vp.pdf(xq, orig_flag=False).ravel()
l0, s0 = vp.lambd.copy(), vp.sigma.copy()
theta = vp.get_parameters()
after = vp.pdf(xq, orig_flag=False).ravel()
print("  lambd", l0.ravel(), "->", vp.lambd.ravel())
print("  sigma", s0.ravel(), "->", vp.sigma.ravel())
print(
    "  sigma_k * lambd_d invariant:",
    np.allclose(
        s0.reshape(1, -1) * l0.reshape(-1, 1),
        vp.sigma.reshape(1, -1) * vp.lambd.reshape(-1, 1),
    ),
)
print(
    "  density before",
    before,
    "after",
    after,
    "equal:",
    np.allclose(before, after, rtol=0, atol=1e-15),
)

print("\n== weight rescaling: density preserved? ==")
vp = make_vp([1.0, 1.0], [0.5, 0.5, 0.5], [1.0, 2.0, 3.0])
before = vp.pdf(xq, orig_flag=False).ravel()
w0 = vp.w.copy()
_ = vp.get_parameters()
after = vp.pdf(xq, orig_flag=False).ravel()
print("  w", w0.ravel(), "->", vp.w.ravel())
print("  density before", before, "after", after, "  ratio", (before / after))

print("\n== unnormalized weights with optimize_weights = False ==")
vp = make_vp([1.0, 1.0], [0.5, 0.5, 0.5], [1.0, 2.0, 3.0], opt_w=False)
w0 = vp.w.copy()
_ = vp.get_parameters()
print("  w", w0.ravel(), "->", vp.w.ravel(), "(untouched)")

print("\n== idempotence ==")
vp = make_vp([2.0, 4.0], [0.3, 0.5, 0.7], [0.2, 0.3, 0.5])
t1 = vp.get_parameters()
l1, s1, w1 = vp.lambd.copy(), vp.sigma.copy(), vp.w.copy()
t2 = vp.get_parameters()
print(
    "  second call changes nothing:",
    np.array_equal(l1, vp.lambd)
    and np.array_equal(s1, vp.sigma)
    and np.array_equal(w1, vp.w),
    " theta equal:",
    np.array_equal(t1, t2),
)

print("\n== interaction with the mode cache ==")
vp = make_vp([2.0, 4.0], [0.3, 0.5, 0.7], [1.0, 2.0, 3.0])
vp._mode = np.array([0.123, 0.456])
_ = vp.get_parameters()
print(
    "  _mode after get_parameters():",
    vp._mode,
    "(MATLAB rescale_params.m:40 removes vp.mode)",
)

print(
    "\n== the production gauge: is the rescaling a no-op after set_parameters? =="
)
vp = make_vp([2.0, 4.0], [0.3, 0.5, 0.7], [0.2, 0.3, 0.5])
vp.set_parameters(vp.get_parameters())
nl = np.sqrt(np.sum(vp.lambd**2) / vp.D)
l_before, s_before, w_before = (vp.lambd.copy(), vp.sigma.copy(), vp.w.copy())
_ = vp.get_parameters()
print("  nl after set_parameters =", repr(nl), " nl - 1 =", nl - 1.0)
print(
    "  lambd bit-identical after a further get_parameters:",
    np.array_equal(l_before, vp.lambd),
    " sigma:",
    np.array_equal(s_before, vp.sigma),
    " w:",
    np.array_equal(w_before, vp.w),
)
print(
    "  max |dlambd| =",
    np.max(np.abs(l_before - vp.lambd)),
    " max |dsigma| =",
    np.max(np.abs(s_before - vp.sigma)),
)

print("\n== does the density of the cached-mode point change? ==")
vp = make_vp([1.0, 1.0], [0.5, 0.5, 0.5], [1.0, 2.0, 3.0])
p_before = vp.pdf(np.array([[0.0, 0.0]]), orig_flag=False).ravel()
_ = vp.get_parameters()
p_after = vp.pdf(np.array([[0.0, 0.0]]), orig_flag=False).ravel()
print(
    "  pdf at the mixture centre before/after:",
    p_before,
    p_after,
    " argmax location unchanged (a scale factor on every component):",
    np.allclose(p_before / p_after, (p_before / p_after)[0]),
)
