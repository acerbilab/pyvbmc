"""W6-1: where the GP hyperparameter samples of a PyVBMC run lie against
their bounds, with gpyreg's pooled bound recommendations and with gplite's
per-column ones.

gpyreg took the statistics of the training inputs over all entries of `X`
where gplite takes them per column (row W6-1 of `verification/wave6.md`).
The hard bounds that change are those of the negative quadratic mean's
location `xm_d` and log scale `log omega_d`, which `_gp_hyp` leaves to
gpyreg's recommendation at every fit:

    xm_d:        [min_d - w_d / 2, max_d + w_d / 2]
    log omega_d: [log w_d + log 1e-6, log w_d + 3]

with `min_d`, `max_d` and the width `w_d = max_d - min_d` of column `d`
(per column) or of all entries (pooled); the pooled box contains the
per-column one. Every other hard bound is the same in both forms: the
kernel's were per column already (only the starting value of the length
scales, which enters the first fit alone, changes), and those of the output
scale, the noise and the mean's constant come from the targets. The other
hyperparameters can still move with the change, compensating for a mean
whose location is confined, and reach their own bounds.

The script runs benchmark targets as `wave3_gate_benchmark_sweep.py` does,
with `gpyreg.GP.fit` wrapped: after each fit it reads the hyperparameter
vectors the fit returns, the training inputs and the hard bounds in effect,
and draws nothing. Which gpyreg runs is decided by `PYTHONPATH`; the script
reads which form it has from the source of the mean's helper. For each run
it prints, per hyperparameter, the median of the returned vectors and the
fraction that lie within 1% of the box's width of each hard bound; and,
with the pooled bounds, the fraction of the mean's location and log scale
that lie outside gplite's per-column box. Each run prints the sweep's
RESULT line, which must equal the sweep's for the same seed and code, so
that the wrapper is seen to change nothing. The records of every fit are
saved to an `.npz` for `wave6_A1d_read.py`.

Usage, from the root of the main checkout:
    python -u wave6_A1d_mean_bounds_probe.py OUT.npz LABEL SEEDS [LABEL SEEDS ...]
with SEEDS as in the sweep (a count, or a range such as 2:4).
Run with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1.
"""

import inspect
import sys
import time

import numpy as np

sys.path.insert(0, "dev/scripts")

import gpyreg as gpr  # noqa: E402
import gpyreg.mean_functions as mean_module  # noqa: E402
from benchmark_targets import find_config, metrics  # noqa: E402

from pyvbmc import VBMC  # noqa: E402

PER_COLUMN = "np.max(X, axis=0) - np.min(X, axis=0)" in inspect.getsource(
    mean_module._bounds_info_helper
)
FORM = "per_column" if PER_COLUMN else "pooled"
print("gpyreg:", gpr.__file__, "| form of the bounds:", FORM, flush=True)

TOL = 1e-6
BIG_LOG = 3.0
NEAR = 0.01  # a vector within this fraction of the box's width is at a bound

RECORDS = []
_CURRENT = {}


def _names(gp):
    names = []
    for component, info in (
        (gp.covariance, gp.covariance.hyperparameter_info(gp.D)),
        (gp.noise, gp.noise.hyperparameter_info()),
        (gp.mean, gp.mean.hyperparameter_info(gp.D)),
    ):
        for name, count in info:
            if count == 1:
                names.append(name)
            else:
                names.extend(f"{name}[{i}]" for i in range(count))
    return names


def _per_column_mean_box(X):
    low, high = np.min(X, axis=0), np.max(X, axis=0)
    w = high - low
    return (
        (low - 0.5 * w, high + 0.5 * w),
        (np.log(w) + np.log(TOL), np.log(w) + BIG_LOG),
    )


_original_fit = gpr.GP.fit


def _recording_fit(self, *args, **kwargs):
    out = _original_fit(self, *args, **kwargs)
    hyp = np.atleast_2d(np.asarray(out[0], dtype=float))
    record = dict(
        label=_CURRENT["label"],
        seed=_CURRENT["seed"],
        fit=_CURRENT["fits"],
        N=self.X.shape[0],
        names=np.array(_names(self)),
        hyp=hyp,
        lb=np.asarray(self.lower_bounds, dtype=float).copy(),
        ub=np.asarray(self.upper_bounds, dtype=float).copy(),
    )
    if isinstance(self.mean, gpr.mean_functions.NegativeQuadratic):
        D = self.D
        start = (
            self.covariance.hyperparameter_count(D)
            + self.noise.hyperparameter_count()
        )
        (xm_lb, xm_ub), (lw_lb, lw_ub) = _per_column_mean_box(self.X)
        record.update(
            xm_index=np.arange(start + 1, start + 1 + D),
            lw_index=np.arange(start + 1 + D, start + 1 + 2 * D),
            col_xm_lb=xm_lb,
            col_xm_ub=xm_ub,
            col_lw_lb=lw_lb,
            col_lw_ub=lw_ub,
        )
    RECORDS.append(record)
    _CURRENT["fits"] += 1
    return out


gpr.GP.fit = _recording_fit


def _summary(records):
    lines = []
    names = records[0]["names"]
    hyp = np.concatenate([r["hyp"] for r in records])
    lb = np.concatenate(
        [np.broadcast_to(r["lb"], r["hyp"].shape) for r in records]
    )
    ub = np.concatenate(
        [np.broadcast_to(r["ub"], r["hyp"].shape) for r in records]
    )
    near = NEAR * (ub - lb)
    at_lower = hyp <= lb + near
    at_upper = hyp >= ub - near
    n = hyp.shape[0]
    lines.append(f"    per hyperparameter, over {n} vectors:")
    for j, name in enumerate(names):
        lines.append(
            f"    {name:32s} median {np.median(hyp[:, j]):9.3f}  at lower "
            f"bound {at_lower[:, j].mean():.3f}  at upper bound "
            f"{at_upper[:, j].mean():.3f}"
        )
    if not PER_COLUMN and "xm_index" in records[0]:
        lines.append("    outside gplite's per-column box:")
        for group in ("xm", "lw"):
            index = records[0][f"{group}_index"]
            vals = np.concatenate(
                [r["hyp"][:, r[f"{group}_index"]] for r in records]
            )
            clb = np.concatenate(
                [
                    np.broadcast_to(
                        r[f"col_{group}_lb"], (r["hyp"].shape[0], len(index))
                    )
                    for r in records
                ]
            )
            cub = np.concatenate(
                [
                    np.broadcast_to(
                        r[f"col_{group}_ub"], (r["hyp"].shape[0], len(index))
                    )
                    for r in records
                ]
            )
            for d in range(len(index)):
                lines.append(
                    f"    {names[index[d]]:32s} below "
                    f"{(vals[:, d] < clb[:, d]).mean():.3f}  above "
                    f"{(vals[:, d] > cub[:, d]).mean():.3f}"
                )
    return lines


out_path = sys.argv[1]
pairs = sys.argv[2:]
for label, n_seeds in zip(pairs[0::2], pairs[1::2]):
    first, last = (
        (int(v) for v in n_seeds.split(":"))
        if ":" in n_seeds
        else (1, int(n_seeds))
    )
    for seed in range(first, last + 1):
        _CURRENT.update(label=label, seed=seed, fits=0)
        n_before = len(RECORDS)
        cfg = find_config(label)
        prob = cfg.make(seed=seed)
        args, options = prob.vbmc_args()
        options.update(
            display="off",
            plot=False,
            print_iteration_header=False,
            performance_calibration="off",
        )
        t0 = time.time()
        vbmc = VBMC(*args, options=options, seed=seed)
        vp, results = vbmc.optimize()
        m = metrics(prob, vp, results["elbo"])
        print(
            f"RESULT {label} seed {seed}: elbo - lnZ "
            f"{results['elbo'] - prob.ln_Z:+.4f} "
            f"(elbo_sd {results['elbo_sd']:.4f})  gsKL {m['gskl']:.4f}  "
            f"MMTV {m['mmtv']:.4f}  evaluations {results['func_count']}  "
            f"iterations {results['iterations']}  stable "
            f"{results['convergence_status']}  ({time.time() - t0:.0f} s)",
            flush=True,
        )
        run = RECORDS[n_before:]
        print(f"  {len(run)} fits recorded ({FORM})", flush=True)
        for line in _summary(run):
            print(line, flush=True)

arrays = {}
for i, r in enumerate(RECORDS):
    for key, value in r.items():
        arrays[f"{i:04d}/{key}"] = np.asarray(value)
np.savez(out_path, form=np.array(FORM), **arrays)
print(f"saved {len(RECORDS)} fits to {out_path}", flush=True)
