"""Same offline count as stale_transformer_frequency.py, under both selection
rules: the rank criterion (the default since the 2026-09-19 correction) and
the look-back rule that the stored runs actually used (rank criterion not
passed, so off; the window counted from the last index)."""

import glob
import sys

import dill
import numpy as np

sys.path.insert(0, "dev/scripts")

paths = sorted(
    glob.glob("dev/scripts/runs/gp_box_20260916/*/*.vbmc.pkl")
    + glob.glob("dev/scripts/runs/svbmc_pool_20260913/pool/*.vbmc.pkl")
)
count = {True: 0, False: 0}
later = 0
for p in paths:
    with open(p, "rb") as fh:
        v = dill.load(fh)
    h = v.iteration_history
    n = len(h["iter"])
    acts = [
        (
            [str(a) for a in np.atleast_1d(h["logging_action"][t])]
            if h["logging_action"][t] is not None
            else []
        )
        for t in range(n)
    ]
    warps = [t for t in range(n) if any("rotoscale" in a for a in acts[t])]
    undone = [t for t in range(n) if any("undo" in a.lower() for a in acts[t])]
    for t in warps:
        if not [w for w in warps if w < t and w not in undone]:
            continue
        later += 1
        for flag in (True, False):
            _, _, _, idx = v.determine_best_vp(
                max_idx=t - 1,
                safe_sd=5,
                frac_back=0.25,
                rank_criterion_flag=flag,
            )
            same = (
                h["vp"][idx].parameter_transformer
                == h["vp"][t - 1].parameter_transformer
            )
            if not same:
                count[flag] += 1
                name = p.replace("\\", "/").split("/")[-1]
                print(
                    f"rank={flag}: {name} warp at {t} -> chose iter {idx}",
                    flush=True,
                )
print(
    f"warps after an earlier kept warp: {later}; stale under rank rule: "
    f"{count[True]}; stale under look-back rule: {count[False]}"
)
