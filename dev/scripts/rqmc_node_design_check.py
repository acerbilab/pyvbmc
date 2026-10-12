"""Developer check behind the F2 node rule of the noisy-acquisition
efficiency plan (``dev/plans/noisy-acquisition-efficiency.md``, "Approved
F2 amendment").

VIQR estimates its integral over the variational posterior on a fixed
number of nodes drawn from the mixture. This script asks how much a
randomized quasi-Monte Carlo node set of the same size gains over plain
Monte Carlo once the nodes pass through the component selection and the
inverse-normal map into each component, and how the candidate designs
behave when the mixture has many components with small weights. The
integrand is a Gaussian bump, the shape of the squared-exponential kernel
factor that localizes the VIQR variance reduction around a candidate; it
stands in for the real integrand, which the frozen-state stage measures.

Designs at n = 96 nodes:

* ``coord``: one scrambled Sobol sequence in D + 1 dimensions; the first
  coordinate selects the component through the cumulative weights, the
  other D pass through the inverse normal CDF into that component. Equal
  node weights. This is the F2 rule.
* ``coord sorted``: the same, with the components ordered along one axis
  before the cumulative weights are formed (here the first coordinate of
  the means; the F2 rule uses the leading principal axis of the means).
* ``blocks prop``: one scrambled Sobol block per component with
  proportional allocation by largest remainder and at least one node per
  component; node weights ``w_k / n_k``.
* ``blocks pow2``: the E2 rule, one node per component and then doubling
  the component with the largest ``w_k / n_k`` that fits; weights
  ``w_k / n_k``.

Part A: K = 3 and K = 12 components, D = 2, 5, 10, 20. Part B: K = 40
with skewed weights (many below 0.01), D = 2, 5, 10. Each row reports the
root-mean-square error over 400 independent replicates relative to plain
Monte Carlo at 96 nodes (1.00 = no gain), against a reference from 10,000
Monte Carlo estimates. The last column, the Monte Carlo RMSE divided by
the integral, flags rows where no 96-node rule measures the integral at
all (values above 1): those ratios are noise. Runs in about two minutes;
exact values depend on the NumPy and SciPy versions (recorded numbers:
NumPy 2.5, SciPy 1.18).
"""

import warnings

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

warnings.simplefilter("ignore")  # Sobol balance warning for n = 96

N_NODES = 96
N_REPLICATES = 400
N_REFERENCE = 10_000
EPS = 1e-12


def bump(x, c, ell):
    return np.exp(-np.sum((x - c) ** 2, axis=1) / (2 * ell**2))


def est_mc(rng, w, mu, sig, f, n=N_NODES):
    k = rng.choice(len(w), size=n, p=w)
    x = mu[k] + sig[k, None] * rng.standard_normal((n, mu.shape[1]))
    return f(x).mean()


def est_coord(rng, w, mu, sig, f, order=None, n=N_NODES):
    D = mu.shape[1]
    if order is not None:
        w, mu, sig = w[order], mu[order], sig[order]
    u = qmc.Sobol(D + 1, scramble=True, seed=rng).random(n)
    k = np.searchsorted(np.cumsum(w), u[:, 0], side="right")
    k = np.minimum(k, len(w) - 1)
    z = ndtri(np.clip(u[:, 1:], EPS, 1 - EPS))
    return f(mu[k] + sig[k, None] * z).mean()


def _blocks(rng, w, mu, sig, f, nk):
    D = mu.shape[1]
    total = 0.0
    for k in range(len(w)):
        u = qmc.Sobol(D, scramble=True, seed=rng).random(int(nk[k]))
        x = mu[k] + sig[k] * ndtri(np.clip(u, EPS, 1 - EPS))
        total += w[k] * f(x).mean()
    return total


def est_blocks_prop(rng, w, mu, sig, f, n=N_NODES):
    raw = n * w
    nk = np.maximum(np.floor(raw).astype(int), 1)
    while nk.sum() > n:
        nk[np.argmax(nk)] -= 1
    for i in np.argsort(-(raw - np.floor(raw))):
        if nk.sum() >= n:
            break
        nk[i] += 1
    return _blocks(rng, w, mu, sig, f, nk)


def est_blocks_pow2(rng, w, mu, sig, f, n=N_NODES):
    nk = np.ones(len(w), dtype=int)
    while True:
        for i in np.argsort(-(w / nk)):
            if nk.sum() + nk[i] <= n:
                nk[i] *= 2
                break
        else:
            break
    return _blocks(rng, w, mu, sig, f, nk)


def rmse_ratios(rng, w, mu, sig, f, estimators):
    ref = np.mean([est_mc(rng, w, mu, sig, f) for _ in range(N_REFERENCE)])
    errors = {}
    for name, est in estimators:
        v = np.array([est(rng, w, mu, sig, f) for _ in range(N_REPLICATES)])
        errors[name] = np.sqrt(np.mean((v - ref) ** 2))
    ratios = {k: errors[k] / errors["mc"] for k in errors if k != "mc"}
    return ratios, errors["mc"] / ref


def part_a():
    rng = np.random.default_rng(1)
    print(
        f"Part A: n = {N_NODES} nodes, {N_REPLICATES} replicates;"
        " RMSE relative to plain MC (1.00 = no gain)"
    )
    print(f"{'D':>3} {'K':>3} {'bump':>5} | {'coord':>6} {'blocks prop':>11}")
    for D in (2, 5, 10, 20):
        for K in (3, 12):
            w = rng.dirichlet(np.ones(K) * 2.0)
            mu = rng.uniform(-2, 2, size=(K, D))
            sig = rng.uniform(0.5, 1.5, size=K)
            for ell in (1.0, 0.5):
                c = mu[np.argmax(w)] + 0.5 * sig[np.argmax(w)]
                f = lambda x: bump(x, c, ell)
                r, noise = rmse_ratios(
                    rng,
                    w,
                    mu,
                    sig,
                    f,
                    (
                        ("mc", est_mc),
                        ("coord", est_coord),
                        ("prop", est_blocks_prop),
                    ),
                )
                print(
                    f"{D:3d} {K:3d} {ell:5.1f} | {r['coord']:6.2f}"
                    f" {r['prop']:11.2f}   (MC rmse/ref = {noise:.3f})"
                )


def part_b():
    rng = np.random.default_rng(7)
    K = 40
    print(
        f"\nPart B: n = {N_NODES} nodes, {N_REPLICATES} replicates, K = {K};"
        " RMSE relative to plain MC"
    )
    print(
        f"{'D':>3} {'bump':>5} | {'coord':>6} {'coord sorted':>12}"
        f" {'blocks prop':>11} {'blocks pow2':>11} | weights: min / #<0.01 / max"
    )
    for D in (2, 5, 10):
        w = rng.dirichlet(np.ones(K) * 0.4)
        mu = rng.uniform(-2, 2, size=(K, D))
        sig = rng.uniform(0.5, 1.5, size=K)
        order = np.argsort(mu[:, 0])
        for ell in (1.0, 0.5):
            c = mu[np.argmax(w)] + 0.5 * sig[np.argmax(w)]
            f = lambda x: bump(x, c, ell)
            r, noise = rmse_ratios(
                rng,
                w,
                mu,
                sig,
                f,
                (
                    ("mc", est_mc),
                    ("coord", est_coord),
                    ("sorted", lambda *a: est_coord(*a, order=order)),
                    ("prop", est_blocks_prop),
                    ("pow2", est_blocks_pow2),
                ),
            )
            print(
                f"{D:3d} {ell:5.1f} | {r['coord']:6.2f} {r['sorted']:12.2f}"
                f" {r['prop']:11.2f} {r['pow2']:11.2f} | "
                f"{w.min():.4f} / {(w < 0.01).sum():2d} / {w.max():.3f}"
                f"   (MC rmse/ref = {noise:.3f})"
            )


if __name__ == "__main__":
    part_a()
    part_b()
