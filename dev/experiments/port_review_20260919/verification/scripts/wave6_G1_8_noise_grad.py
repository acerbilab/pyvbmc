"""Settles the scalar-``sn2`` branch of the noise gradient.

``gaussian_process.py:2812-2819`` takes the ``np.isscalar(sn2)`` branch and
writes ``np.dot(dsn2[i], tr_Q).item()``, where ``dsn2[i]`` is *row* ``i``.
``gplite_core.m:242-244`` (read at vbmc 396d649) writes
``dnlZ(Ncov+i) = 0.5*sn2_mult*dsn2(i)*trQ``, a linear index into a
column-major array.  The two shapes are decided independently:
``noise_functions.py:243-246`` / ``gplite_noisefun.m:167-172`` size
``dsn2`` by ``any(parameters[1:] > 0)`` while the branch is chosen by
whether ``sn2`` came out a scalar.  (G1 comparison F6.)

The script builds the configuration where they disagree
(``noisefun = [1 2 0]`` with no ``s2``), calls the gradient, and evaluates
both index conventions against a finite difference.  It also confirms that
every configuration with a vector ``sn2``, and the one-hyperparameter
scalar cases, agree with finite differences.

Run from anywhere.
"""

import gpyreg as gpr
import numpy as np

print("gpyreg:", gpr.__file__)


def build(noise, D=2, n=14, s2=None, seed=23):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2.0, 2.0, size=(n, D))
    y = np.sum(1.0 + np.sin(X), axis=1, keepdims=True)
    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.ConstantMean(),
        noise=noise,
    )
    gp.X, gp.y, gp.s2 = X, y, s2
    gp.set_bounds(None)
    gp.set_priors({k: None for k in gp.get_priors()})
    return gp


def fd_grad(f, h0, eps=1e-6):
    g = np.zeros_like(h0)
    for i in range(h0.size):
        hp, hm = h0.copy(), h0.copy()
        hp[i] += eps
        hm[i] -= eps
        g[i] = (f(hp) - f(hm)) / (2 * eps)
    return g


rng = np.random.default_rng(3)
S2 = rng.uniform(0.01, 0.3, size=(14, 1))

configs = [
    (
        "[1 0 0] constant only, no s2",
        gpr.noise_functions.GaussianNoise(constant_add=True),
        None,
    ),
    (
        "[1 1 0] constant + s2 as is, s2 given",
        gpr.noise_functions.GaussianNoise(
            constant_add=True, user_provided_add=1
        ),
        S2,
    ),
    (
        "[1 2 0] constant + scaled s2, s2 given",
        gpr.noise_functions.GaussianNoise(
            constant_add=True, user_provided_add=True, scale_user_provided=True
        ),
        S2,
    ),
    (
        "[1 2 0] constant + scaled s2, s2 OMITTED",
        gpr.noise_functions.GaussianNoise(
            constant_add=True, user_provided_add=True, scale_user_provided=True
        ),
        None,
    ),
    (
        "[0 2 0] scaled s2 only, s2 OMITTED",
        gpr.noise_functions.GaussianNoise(
            constant_add=False,
            user_provided_add=True,
            scale_user_provided=True,
        ),
        None,
    ),
    (
        "[1 0 1] constant + rectified output-dependent",
        gpr.noise_functions.GaussianNoise(
            constant_add=True, rectified_linear_output_dependent_add=True
        ),
        None,
    ),
]

for label, noise, s2 in configs:
    gp = build(noise, s2=s2)
    nN = noise.hyperparameter_count()
    hyp = np.concatenate([np.zeros(3), np.full(nN, -1.0), [0.0]])
    if noise.parameters[2] == 1:
        hyp[3 + nN - 2] = 0.5  # the threshold, inside the y range
    sn2 = noise.compute(hyp[3 : 3 + nN], gp.X, gp.y, gp.s2)
    _, dsn2 = noise.compute(
        hyp[3 : 3 + nN], gp.X, gp.y, gp.s2, compute_grad=True
    )
    print(f"\n-- {label}")
    print(
        f"   noise_N={nN}  np.isscalar(sn2)={np.isscalar(sn2)}"
        f"  sn2 type={type(sn2).__name__}  dsn2.shape={np.shape(dsn2)}"
    )
    try:
        val, grad = gp.log_likelihood(hyp, compute_grad=True)
        fd = fd_grad(lambda h: gp.log_likelihood(h), hyp)
        err = np.max(np.abs(grad - fd))
        print(f"   analytic vs finite difference: max abs err {err:.3g}")
        print(f"   noise entries: analytic {grad[3:3+nN]}  fd {fd[3:3+nN]}")
    except Exception as exc:  # noqa: BLE001
        print(
            f"   log_likelihood(compute_grad=True) RAISED "
            f"{type(exc).__name__}: {exc}"
        )
        # what would each index convention give?
        fd = fd_grad(lambda h: gp.log_likelihood(h), hyp)
        dsn2 = np.atleast_2d(dsn2)
        print(f"   dsn2 =\n{dsn2[:3]} ... (first rows)")
        print(f"   finite-difference noise entries: {fd[3:3+nN]}")
        print(
            "   gpyreg wants dsn2[i] (a row); MATLAB's linear index"
            " dsn2(i) picks element (i-1 mod N, floor((i-1)/N)):"
        )
        N = dsn2.shape[0]
        for i in range(nN):
            lin = np.asfortranarray(dsn2).ravel(order="F")[i]
            print(
                f"     i={i}: correct dsn2[0,{i}]={dsn2[0, i]:.6g}"
                f"   MATLAB dsn2({i + 1}) = {lin:.6g}"
            )
