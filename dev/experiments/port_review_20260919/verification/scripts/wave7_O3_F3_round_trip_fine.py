"""O3 F3: the x -> u -> x round trip of student4 on [0, 1] near z = 1/2 on
a fine grid (spacing 1e-11 over 1/2 +- 3e-8), where the closed-form
quantile rounds alpha to 1 and returns t = 0."""
import gpyreg as gpr
import numpy as np

import pyvbmc
from pyvbmc.parameter_transformer import ParameterTransformer

print("pyvbmc:", pyvbmc.__file__)
print("gpyreg:", gpr.__file__)
pt = ParameterTransformer(
    1, np.array([[0.0]]), np.array([[1.0]]), transform_type="student4"
)
z = 0.5 + np.linspace(-3e-8, 3e-8, 6001)
back = pt.inverse(pt(z[:, None]))[:, 0]
err = np.abs(back - z)
i = np.argmax(err)
print(
    f"max |dx| = {err.max():.2e} at z-1/2 = {z[i]-0.5:.3e}; u there = {pt(z[i:i+1, None])[0,0]:.3e}"
)
zero = pt(z[:, None])[:, 0] == 0
print(f"t = 0 returned for |z-1/2| up to {np.abs(z[zero]-0.5).max():.3e}")
