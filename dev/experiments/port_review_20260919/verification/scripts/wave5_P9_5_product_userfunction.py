"""P9-5. Can a `Product` whose marginal is a `UserFunction` evaluate its
density, and what reaches that call from `VBMC(..., prior=[...])`?

Settles: the `TypeError` from `Product.log_pdf`; the behavior of a callable
that tolerates `**kwargs` (a column of n points, a scalar return broadcast
to all rows); that `Product.sample` routes around the same marginal; and
the path `VBMC(prior=[...])` -> `convert_to_prior` -> `Product` ->
first log-joint evaluation.
"""

import numpy as np

import pyvbmc
from pyvbmc import VBMC
from pyvbmc.priors import Product, UniformBox, convert_to_prior
from pyvbmc.priors.user_function import UserFunction

print("pyvbmc.__file__ =", pyvbmc.__file__)
print()

log_p = lambda x: float(np.sum(np.log(np.asarray(x) + 1.0)))
sampler = lambda n: np.random.default_rng(0).uniform(0.0, 1.0, size=(n, 1))

uf = UserFunction(log_p, sampler, D=1)
prod = Product([uf, UniformBox(0.0, 1.0)])
print(
    "Product built:",
    type(prod).__name__,
    "D =",
    prod.D,
    " marginals:",
    [type(m).__name__ for m in prod.marginals],
)
print("Product.a, .b =", prod.a, prod.b)
print()

print("--- sample works (the special case at product.py:112-116) ---")
print(prod.sample(3))
print()

print("--- log_pdf ---")
try:
    print(prod.log_pdf(np.array([[0.2, 0.5]])))
except Exception as e:
    print(f"raised {type(e).__name__}: {e}")
print()

print("--- a callable that tolerates the keyword ---")
calls = []


def log_p_kw(x, **kwargs):
    calls.append((np.shape(x), dict(kwargs)))
    return float(np.sum(np.log(np.asarray(x) + 1.0)))


prod2 = Product([UserFunction(log_p_kw, sampler, D=1), UniformBox(0.0, 1.0)])
x = np.array([[0.1, 0.5], [0.2, 0.5], [0.3, 0.5]])
y = prod2.log_pdf(x)
print("calls to the user callable:", calls)
print("Product.log_pdf(x) rows =", y.ravel())
print(
    "per-row truth (log(x0+1) + log(1)) =",
    [float(np.log(v + 1.0)) for v in x[:, 0]],
)
print(
    "the scalar return was broadcast to every row:",
    bool(np.allclose(y.ravel(), y.ravel()[0])),
)
print()

print("--- the path from VBMC(prior=[...]) ---")
p = convert_to_prior([uf, UniformBox(0.0, 1.0)], None, None, 2)
print(
    "convert_to_prior([UserFunction, UniformBox]) ->",
    type(p).__name__,
    "D =",
    p.D,
)

lb = np.array([[-1.0, -1.0]])
ub = np.array([[2.0, 2.0]])
plb = np.array([[0.0, 0.0]])
pub = np.array([[1.0, 1.0]])
x0 = np.array([[0.5, 0.5]])
try:
    v = VBMC(
        lambda t: -0.5 * float(np.sum(np.asarray(t) ** 2)),
        x0,
        lb,
        ub,
        plb,
        pub,
        prior=[UserFunction(log_p, sampler, D=1), UniformBox(0.0, 1.0)],
        options={"display": "off"},
    )
    print("VBMC constructed; prior =", type(v.prior).__name__)
    try:
        print("log_joint(x0) =", v.log_joint(x0.ravel()))
    except Exception as e:
        print(f"log_joint raised {type(e).__name__}: {e}")
    try:
        print("function_logger(x0_transformed) ...")
        print(v.function_logger(v.x0[0]))
    except Exception as e:
        print(f"function_logger raised {type(e).__name__}: " f"{str(e)[:200]}")
except Exception as e:
    print(f"VBMC construction raised {type(e).__name__}: {e}")
print()

print("--- UserFunction(D=None) as a marginal ---")
try:
    Product([UserFunction(log_p, sampler, D=None), UniformBox(0.0, 1.0)])
except Exception as e:
    print(f"raised {type(e).__name__}: {str(e)[:160]}")
