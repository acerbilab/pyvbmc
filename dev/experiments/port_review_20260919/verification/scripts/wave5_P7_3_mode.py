"""P7-3: the four mode() findings (variational_posterior.py:1227-1298).

(a) mode(orig_flag=True) on a one-dimensional posterior: does the
    np.stack of the squeezed bounds raise?
(b) the optimizer's box offset by an absolute sqrt(eps), and x0 clamped to
    the raw bounds while the optimizer gets the shrunken pair.
(c) the _mode cache: does it ignore n_opts, does a direct assignment to
    mu/sigma/lambd/w leave a stale mode, does get_parameters clear it?
(d) the search: how many optimizations, from where, and how many draws of
    self.rng a call consumes; does a call of mode() change what a later
    draw on the same generator gives?
"""

import numpy as np

import pyvbmc
from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.variational_posterior import VariationalPosterior

print("pyvbmc.__file__ =", pyvbmc.__file__)


def make_vp(D, K, seed=7, lb=None, ub=None):
    pt = (
        ParameterTransformer(D)
        if lb is None
        else ParameterTransformer(
            D,
            np.array(lb, dtype=float).reshape(1, D),
            np.array(ub, dtype=float).reshape(1, D),
        )
    )
    vp = VariationalPosterior(
        D,
        K,
        x0=np.zeros((1, D)),
        parameter_transformer=pt,
        rng=np.random.default_rng(seed),
    )
    rng = np.random.default_rng(seed + 1)
    vp.mu = rng.standard_normal((D, K))
    vp.sigma = np.full((1, K), 0.5)
    vp.lambd = np.ones((D, 1))
    vp.w = np.ones((1, K)) / K
    return vp


print("\n== (a) D = 1, mode(orig_flag=True) ==")
for lb, ub, tag in [(None, None, "unbounded"), ([-3.0], [3.0], "bounded")]:
    vp = make_vp(1, 2, lb=lb, ub=ub)
    print(
        f"  D=1 {tag}: lb_orig.shape =",
        vp.parameter_transformer.lb_orig.shape,
        " .squeeze().ndim =",
        vp.parameter_transformer.lb_orig.squeeze().ndim,
    )
    try:
        m = vp.mode(orig_flag=True)
        print("    mode(orig_flag=True) ->", m)
    except Exception as e:
        print(f"    mode(orig_flag=True) raises {type(e).__name__}: {e}")
    vp2 = make_vp(1, 2, lb=lb, ub=ub)
    print("    mode(orig_flag=False) ->", vp2.mode(orig_flag=False))

print("\n== (a2) D = 2 for contrast ==")
vp = make_vp(2, 3, lb=[-3.0, -3.0], ub=[3.0, 3.0])
print("  D=2 bounded mode(orig_flag=True) ->", vp.mode(orig_flag=True))

print("\n== (b) the absolute sqrt(eps) box offset ==")
sq = np.sqrt(np.finfo(float).eps)
print("  sqrt(eps) =", sq)
for width in [1e-4, 1e-6, 1e-8, 1e-9]:
    vp = make_vp(2, 2, lb=[0.0, 0.0], ub=[width, width])
    vp.mu = np.full((2, 2), width / 2)
    vp.sigma = np.full((1, 2), width / 10)
    frac = sq / width
    try:
        m = vp.mode(orig_flag=True)
        print(
            f"  width={width:g}: offset is {frac:.4%} of the range; mode={m}"
        )
    except Exception as e:
        print(
            f"  width={width:g}: offset is {frac:.4%} of the range; "
            f"{type(e).__name__}: {e}"
        )

print(
    "\n== (b2) x0 clamped to the raw bounds, optimizer gets the shrunken pair =="
)
vp = make_vp(2, 2, lb=[0.0, 0.0], ub=[1.0, 1.0])
lb_o = vp.parameter_transformer.lb_orig
ub_o = vp.parameter_transformer.ub_orig
x0_on_bound = np.array([0.0, 0.5])
clamped = np.minimum(ub_o, np.maximum(x0_on_bound, lb_o)).squeeze()
box = np.stack((lb_o.squeeze() + sq, ub_o.squeeze() - sq), axis=1)
print("  x0 after the code's clamp:", clamped)
print("  optimizer's box:", box.tolist())
print("  x0[0] < box lower bound?", clamped[0] < box[0, 0])
from scipy.optimize import minimize as _minimize

res = _minimize(lambda z: float(np.sum(z**2)), x0=clamped, bounds=box)
print("  scipy accepts the infeasible x0:", res.success, "x =", res.x)

print("\n== (c) the _mode cache ==")
vp = make_vp(2, 4, lb=[-5.0, -5.0], ub=[5.0, 5.0])
m1 = vp.mode()  # default n_opts = ceil(sqrt(4)) = 2
m50 = vp.mode(n_opts=50)
print("  mode() then mode(n_opts=50) identical:", np.array_equal(m1, m50))
print("  n_opts default for K=4:", int(np.ceil(np.sqrt(4))))
vp.mu = vp.mu * 100.0
m_after = vp.mode()
print("  after vp.mu *= 100, mode() unchanged:", np.array_equal(m1, m_after))
print("  _mode is not None after direct mu assignment:", vp._mode is not None)
vp.w = np.array([[0.7, 0.1, 0.1, 0.1]])
_ = vp.get_parameters()
print("  after get_parameters(), _mode still cached:", vp._mode is not None)
vp.set_parameters(vp.get_parameters())
print("  after set_parameters(), _mode is None:", vp._mode is None)
vp_f = make_vp(2, 4, lb=[-5.0, -5.0], ub=[5.0, 5.0])
mf = vp_f.mode(orig_flag=False)
print(
    "  mode(orig_flag=False) caches?",
    vp_f._mode is None,
    "(True = not cached)",
)

print(
    "\n== (d) the search: draws consumed, and the effect on a shared stream =="
)
K = 9
vp = make_vp(2, K, lb=[-5.0, -5.0], ub=[5.0, 5.0], seed=11)
print(
    "  K =",
    K,
    "n_opts = ceil(sqrt(K)) =",
    int(np.ceil(np.sqrt(K))),
    "; MATLAB would run min(nmax=20, K) =",
    min(20, K),
    "optimizations",
)
g1 = np.random.default_rng(123)
vp.rng = g1 if hasattr(vp, "rng") else None
vpA = make_vp(2, K, lb=[-5.0, -5.0], ub=[5.0, 5.0], seed=11)
vpA._rng = np.random.default_rng(123)
vpB = make_vp(2, K, lb=[-5.0, -5.0], ub=[5.0, 5.0], seed=11)
vpB._rng = np.random.default_rng(123)
a_before = vpA.sample(3, orig_flag=False)[0]
_ = vpB.mode()
b_after = vpB.sample(3, orig_flag=False)[0]
print("  draw of 3 without a prior mode() call:\n", a_before)
print("  draw of 3 after one mode() call on the same generator:\n", b_after)
print("  the two draws differ:", not np.allclose(a_before, b_after))

vpC = make_vp(2, K, lb=[-5.0, -5.0], ub=[5.0, 5.0], seed=11)
vpC._rng = np.random.default_rng(123)
m_first = vpC.mode()
vpC._mode = None  # force a recomputation on the advanced stream
m_second = vpC.mode()
print(
    "  mode() recomputed on the advanced stream equals the first:",
    np.allclose(m_first, m_second),
    m_first,
    m_second,
)
