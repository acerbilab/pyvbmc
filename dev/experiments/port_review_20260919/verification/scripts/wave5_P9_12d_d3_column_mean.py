"""P9-12d. Is the third column of a D=3 `Trapezoidal` draw unbiased?

Settles: whether the borderline KS p-value and the 2.2-standard-error mean
of P9_12c's single pooled run are systematic, by repeating the pooled test
over five independent seeds and comparing every column's mean with its
exact mean.
"""
import numpy as np
from scipy.integrate import quad
from scipy.stats import kstest

import pyvbmc
from pyvbmc.priors import Trapezoidal

print("pyvbmc.__file__ =", pyvbmc.__file__)
a3 = np.array([-3.0, 0.0, 10.0])
u3 = np.array([-2.0, 0.5, 11.0])
v3 = np.array([1.0, 0.75, 18.0])
b3 = np.array([2.0, 4.0, 20.0])
p3 = Trapezoidal(a3, u3, v3, b3)
exact = []
for d in range(3):
    p1 = Trapezoidal(a3[d], u3[d], v3[d], b3[d])
    exact.append(
        quad(
            lambda t: t * float(p1.pdf(np.array([[t]]), keepdims=False)[0]),
            a3[d],
            b3[d],
        )[0]
    )
print("exact means:", np.array(exact))
n = 1000000
for s in range(5):
    dr = p3.sample(n, rng=np.random.default_rng(500 + s))
    m = dr.mean(axis=0)
    se = dr.std(axis=0, ddof=1) / np.sqrt(n)
    z = (m - np.array(exact)) / se
    print(
        f"seed {s}: mean {np.array2string(m, precision=5)}  z = {np.array2string(z, precision=2)}"
    )
