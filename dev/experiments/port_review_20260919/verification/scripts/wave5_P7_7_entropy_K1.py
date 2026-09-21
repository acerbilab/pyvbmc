"""P7-7: the weight gradient of the two entropies for K = 1.

Settles what entlb_vbmc and entmc_vbmc return as the weight gradient of a
one-component posterior with jacobian_flag=False (0 and H - 1
respectively), what the MATLAB files return in the same case, and that the
softmax Jacobian of a one-component posterior sends both to 0, which is
what every production caller sees (jacobian_flag = 1).
"""

import numpy as np

import pyvbmc
from pyvbmc.entropy import entlb_vbmc, entmc_vbmc
from pyvbmc.variational_posterior import VariationalPosterior

print("pyvbmc.__file__ =", pyvbmc.__file__)

D = 3
vp = VariationalPosterior(
    D, 1, x0=np.zeros((1, D)), rng=np.random.default_rng(17)
)
vp.mu = np.array([[0.4], [-0.2], [1.1]])
vp.sigma = np.array([[0.8]])
vp.lambd = np.array([[1.2], [0.9], [0.95]])
vp.w = np.ones((1, 1))
vp.eta = np.zeros((1, 1))

for jac in [False, True]:
    H_lb, dH_lb = entlb_vbmc(vp, tuple([True] * 4), jac)
    H_mc, dH_mc = entmc_vbmc(
        vp, 200, tuple([True] * 4), jac, rng=np.random.default_rng(1)
    )
    print(f"\n  jacobian_flag = {jac}")
    print("   entlb H =", H_lb, "  w-gradient =", dH_lb[-1:])
    print(
        "   entmc H =",
        H_mc,
        "  w-gradient =",
        dH_mc[-1:],
        "  H - 1 =",
        H_mc - 1,
    )

print("\n  MATLAB, by reading:")
print("   ent/entlb_vbmc.m:45-47  -> K == 1 branch sets w_grad = 0")
print(
    "   ent/entmc_vbmc.m:96-101 -> w_grad(1) = -sum(log q)/Ns"
    " - w(1)*sum(norm/q)/Ns; for K = 1, w = 1 and norm/q == 1, so the"
)
print(
    "                              second term is exactly 1 and w_grad = H - 1"
)
print("   both files apply the same softmax Jacobian at :140-144 / :120-124")

print("\n  the softmax Jacobian of a one-component posterior:")
eta = vp.eta.ravel()
eta_exp = np.exp(eta)
eta_sum = eta_exp.sum()
J_w = -np.outer(eta_exp, eta_exp) / eta_sum**2 + np.diag(eta_exp) / eta_sum
print("   J_w =", J_w, " -> J_w @ anything = ", J_w @ np.array([3.14]))

print("\n  every production call site of the entropies:")
print(
    "   variational_optimization.py:1191 sets jacobian_flag = 1 in "
    "_neg_elcbo; _neg_elcbo is the only package caller of either entropy"
)
