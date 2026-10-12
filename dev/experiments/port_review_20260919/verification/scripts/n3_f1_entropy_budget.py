import numpy as np

from pyvbmc.entropy.entmc_vbmc import _entmc_vbmc
from pyvbmc.variational_posterior import VariationalPosterior

D, K, Ns = 4, 20, 38
rng = np.random.default_rng(0)
vp = VariationalPosterior(
    D, K, rng=np.random.default_rng(1), calibration="off"
)
vp.mu = rng.uniform(-1.25, 1.25, (D, K))
vp.sigma = np.exp(rng.uniform(-0.3, 0.3, (1, K)))
vp.lambd = np.exp(rng.uniform(-0.3, 0.3, (D, 1)))
vp.eta = rng.uniform(-1.0, 1.0, (1, K))
e = vp.eta - vp.eta.max()
vp.w = np.exp(e) / np.exp(e).sum()

ref = None
for b in (2**14, 2**15, 2**16, 2**17, 2**18):
    H, dH = _entmc_vbmc(
        vp, Ns, (True,) * 4, True, np.random.default_rng(7), budget=b
    )
    if ref is None:
        ref = (H, dH)
        print(f"budget {b:7d}  H={H!r}  (reference)")
    else:
        dh = np.max(np.abs(dH - ref[1]))
        print(
            f"budget {b:7d}  H exact={H == ref[0]}  |dH-ref|max={dh:.3e}  "
            f"dH exact={np.array_equal(dH, ref[1])}"
        )
print()
# value-only
ref = None
for b in (2**14, 2**16, 2**18):
    H, _ = _entmc_vbmc(
        vp, 4096, (False,) * 4, True, np.random.default_rng(7), budget=b
    )
    if ref is None:
        ref = H
        print(f"value-only budget {b:7d} H={H!r} (reference)")
    else:
        print(
            f"value-only budget {b:7d} H={H!r} exact={H == ref} diff={H-ref:.3e}"
        )
