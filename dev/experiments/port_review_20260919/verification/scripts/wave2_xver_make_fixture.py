"""Write the small bounded posterior of
`pyvbmc/testing/variational_posterior/test_vp_save_bounded_py31*.pkl`.

Usage: python wave2_xver_make_fixture.py <output file>

The two fixtures were written with this script under Python 3.11.9 and
3.12.6 by PyVBMC at `e3ccd0e`, the last commit whose `ParameterTransformer`
pickled its bounded transforms; with later code the file holds no function
and is of no use as that fixture.
"""

import sys

import numpy as np

from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.variational_posterior import VariationalPosterior

out = sys.argv[1]
D, K = 2, 3
pt = ParameterTransformer(
    D,
    np.array([[0.0, -np.inf]]),
    np.array([[10.0, np.inf]]),
    np.array([[2.0, -1.0]]),
    np.array([[6.0, 1.0]]),
)
vp = VariationalPosterior(
    D=D, K=K, x0=np.zeros((1, D)), parameter_transformer=pt, rng=123
)
vp.mu = np.array([[-0.8, 0.1, 0.9], [-0.5, 0.0, 0.6]])
vp.sigma = np.array([[0.3, 0.2, 0.4]])
vp.lambd = np.array([[1.1], [0.9]])
vp.w = np.array([[0.2, 0.5, 0.3]])
vp.save(out, overwrite=True)
print("written under Python", sys.version.split()[0], "->", out)
