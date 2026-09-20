import sys

import numpy as np

sys.path.insert(0, r"C:/Users/luigi/Documents/GitHub/pyvbmc")
import torch
from mpmath import erfinv, mp, mpf
from mpmath import sqrt as msqrt

mp.dps = 60
from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.variational_posterior._torch import _InverseParameterTransform


def tensor(v):
    return torch.tensor(np.asarray(v).copy(), dtype=torch.float64)


lb, ub = -2.0, 3.0
pt = ParameterTransformer(
    1,
    np.array([[lb]]),
    np.array([[ub]]),
    np.array([[lb + 1.25]]),
    np.array([[ub - 1.25]]),
    transform_type="probit",
)
tr = _InverseParameterTransform(pt, 1, tensor)
print("mu", pt.mu, "delta", pt.delta, flush=True)
for gap in (1e-3, 1e-6, 1e-9, 1e-12, 1e-14, 1e-15, 4.44e-16, 8.88e-16):
    x = ub - gap
    if x >= ub:
        continue
    X = np.array([[x]])
    u_np = float(pt(X)[0, 0])
    u_t = float(tr.inv(torch.as_tensor(X)).numpy()[0, 0])
    z = (mpf(ub) - mpf(x)) / (mpf(ub) - mpf(lb))
    probit = -msqrt(2) * erfinv(2 * z - 1)  # probit(1-z)
    u_ref = float((probit - mpf(float(pt.mu[0]))) / mpf(float(pt.delta[0])))
    ulps = gap / np.spacing(ub)
    print(
        f"gap={gap:g} ({ulps:.1f} ulp): numpy={u_np:.12f} torch={u_t:.12f} exact={u_ref:.12f} "
        f"|np-ex|={abs(u_np-u_ref):.2e} |t-ex|={abs(u_t-u_ref):.2e}",
        flush=True,
    )
