"""Initial variational posterior means against the transformed x0.

Wave-2 finding P1b internal F1: VBMC.__init__ hands the untransformed x0 to
VariationalPosterior, which stores it in vp.mu, a transformed-space quantity.
MATLAB (misc/setupvars_vbmc.m:65, :82) transforms x0 first.
"""

import numpy as np

import pyvbmc
from pyvbmc import VBMC

print("pyvbmc:", pyvbmc.__file__, flush=True)


def f(x):
    return -0.5 * np.sum(x**2)


cases = {
    "bounded 0..10, PLB 4, PUB 6, x0 5 (probit)": dict(
        x0=[[5.0]], lb=[[0.0]], ub=[[10.0]], plb=[[4.0]], pub=[[6.0]], opt={}
    ),
    "bounded 0..10, PLB 4, PUB 6, x0 5 (logit)": dict(
        x0=[[5.0]],
        lb=[[0.0]],
        ub=[[10.0]],
        plb=[[4.0]],
        pub=[[6.0]],
        opt={"bounded_transform": "logit"},
    ),
    "unbounded, PLB 2, PUB 4, x0 3": dict(
        x0=[[3.0]],
        lb=[[-np.inf]],
        ub=[[np.inf]],
        plb=[[2.0]],
        pub=[[4.0]],
        opt={},
    ),
    "unbounded, PLB -1, PUB 1, x0 0 (symmetric: invisible)": dict(
        x0=[[0.0]],
        lb=[[-np.inf]],
        ub=[[np.inf]],
        plb=[[-1.0]],
        pub=[[1.0]],
        opt={},
    ),
    "D=2 unbounded, PLB (2,-3), PUB (4,5), x0 (3.5, 0)": dict(
        x0=[[3.5, 0.0]],
        lb=[[-np.inf, -np.inf]],
        ub=[[np.inf, np.inf]],
        plb=[[2.0, -3.0]],
        pub=[[4.0, 5.0]],
        opt={},
    ),
}
for name, c in cases.items():
    opt = dict(display="off", **c["opt"])
    v = VBMC(
        f,
        np.array(c["x0"]),
        np.array(c["lb"]),
        np.array(c["ub"]),
        np.array(c["plb"]),
        np.array(c["pub"]),
        options=opt,
        seed=1,
    )
    x0_tran = v.x0  # transformed by __init__ after the VP was built
    mu = v.vp.mu  # (D, K)
    back = v.parameter_transformer.inverse(mu.T)  # original coordinates
    print("\n" + name, flush=True)
    print("  x0 original      :", np.ravel(c["x0"]))
    print("  x0 transformed   :", np.ravel(x0_tran))
    print("  vp.mu (col 0)    :", mu[:, 0])
    print("  vp.mu in original:", back[0])
    print(
        "  |mu - x0_tran| in plausible-box widths (transformed box is 1):",
        np.abs(mu[:, 0] - np.ravel(x0_tran)),
    )
