"""P7-1: kl_div(samples=..., gauss_flag=True) uses np.mean(samples), one scalar.

Settles whether variational_posterior.py:1482 collapses the (N, D) sample
matrix to a single scalar mean, where vbmc_kldiv.m:63 takes mean(vp2, 1), a
1-by-D row, and what the consequence is when the coordinates have different
means.  Also prints the MATLAB-equivalent answer (per-coordinate mean) for
the same inputs.
"""

import numpy as np

import pyvbmc
from pyvbmc.stats import kl_div_mvn
from pyvbmc.variational_posterior import VariationalPosterior

print("pyvbmc.__file__ =", pyvbmc.__file__)

rng = np.random.default_rng(20260921)

# A D = 3 posterior whose coordinates have clearly different means.
D, K = 3, 2
vp = VariationalPosterior(
    D, K, x0=np.zeros((1, D)), rng=np.random.default_rng(1)
)
vp.mu = np.tile(np.array([[10.0], [-5.0], [2.0]]), (1, K))
vp.sigma = np.ones((1, K))
vp.lambd = np.ones((D, 1))
vp.w = np.ones((1, K)) / K

N = 20000
samples, _ = vp.sample(N, orig_flag=True, balance_flag=True)
print("np.mean(samples)            =", np.mean(samples))
print("np.mean(samples, axis=0)    =", np.mean(samples, axis=0))

kls_coded = vp.kl_div(samples=samples, N=N, gauss_flag=True)
print("vp.kl_div(samples=...)      =", kls_coded)

# MATLAB equivalent: q2mu = mean(vp2,1); q2sigma = cov(vp2)
q1mu, q1sigma = vp.moments(N, True, True)
q2mu = np.mean(samples, axis=0)
q2sigma = np.cov(samples.T)
kls_matlab = np.maximum(0, kl_div_mvn(q1mu, q1sigma, q2mu, q2sigma))
print("MATLAB-equivalent kl_div    =", kls_matlab)

# The shape that kl_div_mvn actually sees on the coded path
print(
    "np.atleast_2d(np.mean(samples)).shape =",
    np.atleast_2d(np.mean(samples)).shape,
)

# Sanity: with equal coordinate means the two agree (the tests' configuration)
vp2 = VariationalPosterior(
    D, K, x0=np.full((1, D), 5.0), rng=np.random.default_rng(2)
)
vp2.sigma = np.ones((1, K))
s2, _ = vp2.sample(N, orig_flag=True, balance_flag=True)
print(
    "equal-mean case, coded      =",
    vp2.kl_div(samples=s2, N=N, gauss_flag=True),
)
m1, S1 = vp2.moments(N, True, True)
print(
    "equal-mean case, MATLAB-eq  =",
    np.maximum(0, kl_div_mvn(m1, S1, np.mean(s2, axis=0), np.cov(s2.T))),
)
