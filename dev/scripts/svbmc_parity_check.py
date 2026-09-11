"""Parity check of ``pyvbmc.svbmc.SVBMC`` against the pinned upstream package.

Loads the thirty posteriors shipped with S-VBMC 0.1.1 (three groups of ten),
runs the moved class and the upstream class side by side with matched random
draws, and reports the largest differences in the optimized weights, the
three ELBO values, the entropy, one direct ``stacked_ELBO`` evaluation at the
initial weights, and ``sample`` under a seeded global NumPy state.

Matched draws need both implementations to draw their entropy samples the
same way. Upstream does so only in its ``testing=True`` mode (a fixed legacy
``RandomState`` per call), so this check applies to the moved code while it
still carries that flag, before the ``seed`` argument replaces it. Once the
moved class draws from its own generator the two implementations cannot be
matched draw for draw, and the shipped regression references take over as
the numerical gate.

Run from the repository root with the compatibility campaign's Torch overlay
on the path::

    PYTHONPATH=dev/scripts/runs/svbmc_compat_20260908/deps \\
        python dev/scripts/svbmc_parity_check.py

``--source`` and ``--deps`` point at the pinned checkout and the overlay
(defaults: the ignored campaign directories). ``--max-steps`` (default 25)
bounds the optimization: with the fixed per-call draws of ``testing=True``
the objective is deterministic, the five-non-improvement stop rarely
triggers, and a fit otherwise runs to upstream's 500-step cap. A fixed step
count compares the same matched trajectory in both implementations. The
summary is written as JSON under ``dev/scripts/runs/``. Exit status is 1
when any difference exceeds ``--tol``.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import io
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO / "dev/scripts/runs/svbmc_compat_20260908/source"
DEFAULT_DEPS = REPO / "dev/scripts/runs/svbmc_compat_20260908/deps"
GROUPS = ("GMM", "GMM_noisy", "Ring")
MODES = ("all-weights", "posterior-only", "ns")


def load_group(source, group):
    files = sorted(glob.glob(str(source / "vbmc_runs" / group / "*.pkl")))
    if len(files) != 10:
        raise SystemExit(
            f"{group}: expected 10 posteriors, found {len(files)}"
        )
    vps = []
    for f in files:
        with open(f, "rb") as fh:
            vps.append(pickle.load(fh))
    return vps


def fit(cls, vps, mode, max_steps):
    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet):
        stacked = cls(vps, testing=True)
        t0 = time.perf_counter()
        stacked.optimize(max_steps=max_steps, version=mode)
        elapsed = time.perf_counter() - t0
    return stacked, elapsed


def direct_elbo(stacked, w0):
    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet):
        elbo, entropy = stacked.stacked_ELBO(np.array(w0))
    return float(elbo), float(entropy)


def seeded_sample(stacked, n, seed):
    np.random.seed(seed)
    return np.asarray(stacked.sample(n))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--deps", type=Path, default=DEFAULT_DEPS)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--n-sample", type=int, default=256)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(args.deps))
    sys.path.insert(0, str(args.source / "src"))
    import torch

    torch.set_num_threads(1)
    import svbmc as upstream_pkg
    from svbmc.svbmc import SVBMC as Upstream

    from pyvbmc.svbmc import SVBMC as Moved

    up_file = Path(upstream_pkg.__file__).resolve()
    if args.source.resolve() not in up_file.parents:
        raise SystemExit(
            f"upstream svbmc resolved outside --source: {up_file}"
        )
    print(f"upstream: {up_file}  (version {upstream_pkg.__version__})")
    print(
        f"moved:    {Path(Moved.__module__.replace('.', '/')).with_suffix('.py')}"
    )
    print(f"torch {torch.__version__}, numpy {np.__version__}")

    rows = []
    worst = 0.0
    for group in GROUPS:
        vps = load_group(args.source, group)
        for mode in MODES:
            up, t_up = fit(Upstream, vps, mode, args.max_steps)
            mv, t_mv = fit(Moved, vps, mode, args.max_steps)
            d_w = float(np.max(np.abs(up.w - mv.w)))
            d_elbo = {
                k: abs(float(up.elbo[k]) - float(mv.elbo[k])) for k in up.elbo
            }
            d_h = abs(up.entropy - mv.entropy)
            # Direct evaluation at the initial (normalized concatenated) weights.
            w0 = np.concatenate(
                [np.reshape(vp.w, (1, -1)) for vp in up.vp_list], axis=1
            )
            w0 = w0 / w0.sum()
            e_up, h_up = direct_elbo(up, w0)
            e_mv, h_mv = direct_elbo(mv, w0)
            d_direct = max(abs(e_up - e_mv), abs(h_up - h_mv))
            # Sampling under a seeded global state (the pickled posteriors
            # derive their generators from it on first use).
            s_up = seeded_sample(up, args.n_sample, 1701)
            s_mv = seeded_sample(mv, args.n_sample, 1701)
            same_shape = s_up.shape == s_mv.shape
            d_sample = (
                float(np.max(np.abs(s_up - s_mv))) if same_shape else np.inf
            )
            row = {
                "group": group,
                "mode": mode,
                "steps_upstream_s": round(t_up, 2),
                "steps_moved_s": round(t_mv, 2),
                "max_abs_dw": d_w,
                "d_elbo": d_elbo,
                "d_entropy": d_h,
                "d_direct_elbo_entropy": d_direct,
                "sample_shape": list(s_up.shape),
                "sample_rows_returned": int(s_up.shape[0]),
                "max_abs_dsample": d_sample,
                "w_shape": list(mv.w.shape),
            }
            rows.append(row)
            worst = max(
                worst, d_w, max(d_elbo.values()), d_h, d_direct, d_sample
            )
            print(
                f"{group:10s} {mode:15s} dw={d_w:.1e} "
                f"dELBO={max(d_elbo.values()):.1e} dH={d_h:.1e} "
                f"ddirect={d_direct:.1e} dsample={d_sample:.1e} "
                f"rows={s_up.shape[0]}/{args.n_sample} "
                f"t={t_up:.1f}s/{t_mv:.1f}s"
            )

    out_dir = REPO / "dev/scripts/runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"svbmc_parity_{stamp}.json"
    summary = {
        "upstream_file": str(up_file),
        "upstream_version": upstream_pkg.__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "max_steps": args.max_steps,
        "tol": args.tol,
        "worst_difference": worst,
        "rows": rows,
    }
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"worst difference {worst:.3e} (tol {args.tol:.0e}); wrote {out}")
    return 0 if worst <= args.tol else 1


if __name__ == "__main__":
    sys.exit(main())
