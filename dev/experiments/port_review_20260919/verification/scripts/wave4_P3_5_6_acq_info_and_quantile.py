"""P3-5 and P3-6: the `acq_info` contract and an unvalidated `quantile`.

P3-5 settles which keys each shipped acquisition's `acq_info` holds and
whether `get_info()[...]` can raise. P3-6 settles what an out-of-range
`quantile` produces, what survives `AbstractAcqFcn.__call__`'s
`np.maximum(acq, -realmax)` and therefore what reaches the selection in
`active_sample` (`np.argmin` / `np.argsort`).
"""
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from wave4_P3_common import banner, load  # noqa: E402

banner()

from pyvbmc.acquisition_functions import (  # noqa: E402
    AcqFcn,
    AcqFcnIMIQR,
    AcqFcnLog,
    AcqFcnNoisy,
    AcqFcnVanilla,
    AcqFcnVIQR,
)
from pyvbmc.testing.oracles._oracles import (  # noqa: E402
    prepare_gp_for_acq,
    prepare_importance_sampling,
)

print("--- P3-5: acq_info keys ---")
for cls in (
    AcqFcn,
    AcqFcnLog,
    AcqFcnVanilla,
    AcqFcnNoisy,
    AcqFcnVIQR,
    AcqFcnIMIQR,
):
    a = cls()
    info = a.get_info()
    try:
        v = info["compute_var_log_joint"]
        status = f"present ({v})"
    except KeyError:
        status = "KeyError"
    print(
        f"  {cls.__name__:14s} keys={sorted(info)}  "
        f"get_info()['compute_var_log_joint'] -> {status}"
    )

print("\n--- P3-6: quantile validation ---")
SEED = 20260904
for q in (0.0, 0.25, 0.5, 0.75, 1.0):
    for cls in (AcqFcnVIQR, AcqFcnIMIQR):
        st = load("rosenbrock_D2_noise1_viqr", seed=SEED)
        gp, vp, logger, os_ = (
            st["gp"],
            st["vp"],
            st["logger"],
            st["optim_state"],
        )
        Xs = np.array(st["cand"]["Xs"], dtype=float)
        prepare_gp_for_acq(gp, logger, os_)
        with np.errstate(all="ignore"):
            acq = cls(quantile=q)
            print(f"  q={q}  {cls.__name__:12s} u={acq.u!r}")
            try:
                prepare_importance_sampling(st, acq, SEED)
                a = acq(Xs.copy(), gp, vp, logger, os_)
            except Exception as exc:
                print(f"        raised: {type(exc).__name__}: {exc}")
                continue
            n_nan = int(np.sum(np.isnan(a)))
            n_neg_realmax = int(np.sum(a == -sys.float_info.max))
            n_inf = int(np.sum(np.isposinf(a)))
            print(
                f"        out: nan={n_nan} ==-realmax={n_neg_realmax} "
                f"+inf={n_inf} finite_other="
                f"{int(np.sum(np.isfinite(a) & (a != -sys.float_info.max)))} "
                f"argmin={np.argmin(a)} argsort[0]={np.argsort(a)[0]}"
            )
