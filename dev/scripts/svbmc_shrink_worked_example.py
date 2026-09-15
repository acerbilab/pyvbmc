"""The worked example of the shrinkage tutorial, recomputed from the pool.

Rebuilds one recorded stack per condition (``M = 4``, the first
repetition, of the stage D comparison) exactly as
``svbmc_shrink_elbo.py`` rebuilds a cell, and prints what section 7 of
``dev/2026-09-15-svbmc-shrinkage-explained.md`` quotes: per run, the
spread of the components' estimates, the mean estimation standard
deviation, the mean off-diagonal error correlation, the noise share, the
diagonal and full-covariance shrinkage factors and the run's level; the
top components by stack weight before and after shrinkage; the run-level
moments and shifts; and the raw, capped and two-level full biases of the
stack. Needs Torch on ``PYTHONPATH`` and the pool's gpyreg through
``--gpyreg-source``::

    PYTHONPATH="<TORCH_PATH>" python -u dev/scripts/svbmc_shrink_worked_example.py \\
        --pool DIR --cells RESULTS.json --gpyreg-source PATH \\
        [--conditions L1,L2] [--M 4] [--repetition 0]
"""

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from svbmc_pool_run import (  # noqa: E402
    DEFAULT_GPYREG,
    THREAD_KEYS,
    activate_gpyreg,
    pinned_gpyreg_source,
)

# isort: split
import numpy as np  # noqa: E402
from svbmc_shrink_elbo import (  # noqa: E402
    moments,
    run_estimates,
    shrink_diagonal,
    shrink_full,
)

DEFAULT_CONDITIONS = (
    "multisensory_s1_D6_noise3_svbmc",
    "student_D8_noise3_svbmc",
    "gmm_D2_svbmc",
)


def describe(cell, pool):
    from svbmc_pool_io import load_run

    from pyvbmc.svbmc import SVBMC

    arm = cell["arms"]["integrated"]
    vps = [
        load_run(pool / name, rng=seed)["vp"]
        for name, seed in zip(cell["entries"], cell["entry_seeds"])
    ]
    stacked = SVBMC(vps, seed=cell["cell_seed"], show_tips=False)
    w = np.asarray(arm["w"], dtype=float)
    H = float(arm["entropy"])
    G = float(arm["elbos"]["raw"]) - H
    elbo_mc = float(arm["elbo_mc"])
    offsets = np.concatenate([[0], np.cumsum(stacked.K)])
    jacobian = np.ravel(stacked._jacobian_corrections)
    print(
        f"\n##### {cell['condition']} M={cell['M']} r={cell['repetition']}: "
        f"K per run {list(stacked.K)}, G = {G:.3f}, H = {H:.3f}; raw bias "
        f"{G + H - elbo_mc:+.3f}, cap bias "
        f"{float(arm['elbos']['capped_I_median']) - elbo_mc:+.3f}"
    )
    runs = [
        run_estimates(vp, jacobian[offsets[m] : offsets[m + 1]])
        for m, vp in enumerate(stacked.vp_list)
    ]
    own = [np.ravel(vp.w) / np.sum(vp.w) for vp in stacked.vp_list]
    levels = np.array([float(np.dot(o, I)) for o, (I, _, _) in zip(own, runs)])
    level_var = np.array([float(o @ C @ o) for o, (_, _, C) in zip(own, runs)])
    mu_r, tau2_r, spread_r, noise_r = moments(levels, np.diag(level_var))
    shr_levels, lf = shrink_diagonal(levels, level_var, mu_r, tau2_r)
    shifts = shr_levels - levels
    G_two = 0.0
    for m, (I, V, C) in enumerate(runs):
        sl = slice(offsets[m], offsets[m + 1])
        wm = w[sl]
        mu, tau2, spread, noise = moments(I, C)
        s_diag, f_diag = shrink_diagonal(I, V, mu, tau2)
        s_full, f_full = shrink_full(I, C, mu, tau2)
        sd = np.sqrt(np.diag(C))
        corr = C / np.outer(sd, sd)
        off = corr[~np.eye(I.size, dtype=bool)]
        share = noise / spread if spread else float("nan")
        print(
            f"  run {m}: K={I.size}, stack mass {wm.sum():.2f}, level "
            f"{levels[m]:.3f} (sd {np.sqrt(level_var[m]):.3f}); spread sd "
            f"{np.sqrt(spread):.3f}, mean sd_k {sd.mean():.3f}, mean "
            f"off-diagonal correlation {off.mean():.2f}, noise share "
            f"{share:.2f}; diagonal factor {f_diag.mean():.2f}, full row-sum "
            f"factor {f_full.mean():.2f}; run-level shift {shifts[m]:+.3f} "
            f"(factor {lf[m]:.2f})"
        )
        for k in np.argsort(-wm)[:3]:
            print(
                f"    weight {wm[k]:.3f}: I - mu {I[k] - mu:+.3f}, sd "
                f"{sd[k]:.3f}, diagonal {s_diag[k] - mu:+.3f}, full "
                f"{s_full[k] - mu:+.3f}"
            )
        G_two += float(np.dot(wm, s_full + shifts[m]))
    print(
        f"  run level: mu {mu_r:.3f}, spread {spread_r:.3f}, noise {noise_r:.3f}, "
        f"tau2 {tau2_r:.3f}; two-level full bias {G_two + H - elbo_mc:+.3f}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--gpyreg-source", type=Path, default=DEFAULT_GPYREG)
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--M", type=int, default=4)
    parser.add_argument("--repetition", type=int, default=0)
    args = parser.parse_args(argv)
    for key in THREAD_KEYS:
        os.environ.setdefault(key, "1")
    os.environ.setdefault("MPLBACKEND", "Agg")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    pool = args.pool.resolve()
    manifest = json.loads((pool / "manifest.json").read_text(encoding="utf-8"))
    activate_gpyreg(pinned_gpyreg_source(args.gpyreg_source, manifest))
    import torch

    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    cells = json.loads(args.cells.read_text(encoding="utf-8"))["cells"]
    wanted = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for condition in wanted:
        cell = next(
            c
            for c in cells
            if c["condition"] == condition
            and int(c["M"]) == args.M
            and int(c["repetition"]) == args.repetition
        )
        describe(cell, pool)
    return 0


if __name__ == "__main__":
    sys.exit(main())
