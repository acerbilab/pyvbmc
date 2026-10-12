"""P9-10. Is a frozen univariate with array-valued parameters accepted as a
one-dimensional `SciPy` prior, and do its sampler and density disagree?

Settles: the accepted `D = 1`; what `sample` returns; what `log_pdf`
raises; and what `VBMC` and `Product` do with such an object.
"""

import numpy as np
from scipy.stats import norm

import pyvbmc
from pyvbmc import VBMC
from pyvbmc.priors import Product, SciPy, UniformBox

print("pyvbmc.__file__ =", pyvbmc.__file__)
print()

d = norm(loc=np.array([0.0, 10.0, 100.0]))
p = SciPy(d)
print("SciPy(norm(loc=[0,10,100])) -> D =", p.D, " a =", p.a, " b =", p.b)
print("sample(3) =")
print(p.sample(3, rng=np.random.default_rng(0)))
try:
    print("log_pdf(zeros((3,1))) =", p.log_pdf(np.zeros((3, 1))))
except Exception as e:
    print(f"log_pdf raised {type(e).__name__}: {e}")
try:
    print("log_pdf(array([[0.0]])) =", p.log_pdf(np.array([[0.0]])))
except Exception as e:
    print(f"log_pdf(1 point) raised {type(e).__name__}: {e}")
print()

print("--- as a Product marginal ---")
try:
    pr = Product([d, UniformBox(0.0, 1.0)])
    print("Product built, D =", pr.D, " a =", pr.a, " b =", pr.b)
    print("sample(2) =", pr.sample(2, rng=np.random.default_rng(1)))
    try:
        print("log_pdf =", pr.log_pdf(np.array([[0.0, 0.5]])))
    except Exception as e:
        print(f"Product.log_pdf raised {type(e).__name__}: {e}")
except Exception as e:
    print(f"Product construction raised {type(e).__name__}: {str(e)[:200]}")
print()

print("--- as a VBMC prior ---")
lb, ub = np.array([[-5.0, -5.0]]), np.array([[5.0, 5.0]])
plb, pub = np.array([[-1.0, -1.0]]), np.array([[1.0, 1.0]])
x0 = np.array([[0.0, 0.0]])
ll = lambda t: -0.5 * float(np.sum(np.asarray(t) ** 2))
try:
    VBMC(ll, x0, lb, ub, plb, pub, prior=d, options={"display": "off"})
    print("VBMC(prior=norm(loc=[0,10,100])) constructed (D=2 model)")
except Exception as e:
    print(f"VBMC raised {type(e).__name__}: {str(e)[:160]}")

# and for a one-dimensional model, where prior.D == self.D
lb1, ub1 = np.array([[-5.0]]), np.array([[5.0]])
plb1, pub1 = np.array([[-1.0]]), np.array([[1.0]])
try:
    v = VBMC(
        ll,
        np.array([[0.0]]),
        lb1,
        ub1,
        plb1,
        pub1,
        prior=d,
        options={"display": "off"},
    )
    print(
        "VBMC(D=1, prior=norm(loc=[0,10,100])) constructed; " "prior =",
        type(v.prior).__name__,
    )
    try:
        print("  log_joint([0.0]) =", v.log_joint(np.array([0.0])))
    except Exception as e:
        print(f"  log_joint raised {type(e).__name__}: {str(e)[:120]}")
except Exception as e:
    print(f"VBMC(D=1) raised {type(e).__name__}: {str(e)[:160]}")
