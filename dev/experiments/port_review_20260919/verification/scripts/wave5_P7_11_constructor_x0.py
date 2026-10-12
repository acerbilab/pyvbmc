"""P7-11: VariationalPosterior(D, K, x0) with a (D, 1) column x0.

Settles whether the constructor refuses a column vector because the
x0.reshape(-1, 1) at variational_posterior.py:136 discards its result, and
what the constructor's docstring promises.
"""

import numpy as np

import pyvbmc
from pyvbmc.variational_posterior import VariationalPosterior

print("pyvbmc.__file__ =", pyvbmc.__file__)

D, K = 3, 4
forms = {
    "(1, D) row": np.array([[1.0, 2.0, 3.0]]),
    "(D,) flat": np.array([1.0, 2.0, 3.0]),
    "(D, 1) column": np.array([[1.0], [2.0], [3.0]]),
    "(2, D) two rows": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
}
for name, x0 in forms.items():
    try:
        vp = VariationalPosterior(D, K, x0=x0, rng=np.random.default_rng(0))
        print(
            f"  {name:>16} {x0.shape}: mu.shape={vp.mu.shape}  "
            f"mu[:,0]={np.round(vp.mu[:, 0], 6)}"
        )
    except Exception as e:
        print(f"  {name:>16} {x0.shape}: {type(e).__name__}: {e}")

print("\n  the branch taken: x0.size == D is True for both (D,) and (D,1)")
for name, x0 in forms.items():
    print(f"   {name:>16}: x0.size == D -> {x0.size == D}")

print("\n  what the discarded reshape would have produced:")
col = forms["(D, 1) column"]
print("   x0.reshape(-1,1).shape =", col.reshape(-1, 1).shape)
print(
    "   np.tile(col, (K,1)).T.shape =",
    np.tile(col, (K, 1)).T.shape,
    " (the shape the code then builds from the unreshaped column)",
)
print(
    "   np.tile(col.reshape(-1,1).T, (K,1)).T.shape =",
    np.tile(col.reshape(-1, 1).T, (K, 1)).T.shape,
)

print("\n  constructor docstring (variational_posterior.py:43-46):")
print("   x0 : np.ndarray, optional -- 'The starting vector for the mixture")
print("   components means. It can be a single array or multiple rows (up to")
print("   K); missing rows are duplicated by making copies of x0'")
