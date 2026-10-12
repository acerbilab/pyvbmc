"""Count the kept rotoscalings of the golden reference and what follows them.

Reads the traces of a golden reference (a directory of ``<case>.json`` and
``<case>.npz`` pairs written by ``dev/scripts/golden_trace.py``) and writes
``golden_scan.json`` beside this script. A trace flags an iteration whose
actions include ``rotoscale``, whether the warp was kept or undone; the warp
was kept where the transformer's rotation or scale differs from the
iteration before. For each kept warp the scan records the ELBO before it,
in its iteration and in the three that follow, and flags a warp whose
iteration reports an ELBO more than 0.5 above the true log evidence or
that is followed by a drop of more than 5 nats below the ELBO before it,
and counts the kept warps whose iteration reports an sKL above 10 between
the posteriors before and after it::

    python dev/experiments/example2_rotoscale_20260926/scan_golden_warps.py \
        dev/scripts/runs/golden/reference_990_20260913
"""

import collections
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OVERSHOOT = 0.5
DROP = 5.0
LARGE_SKL = 10.0


def kept_warps(z):
    proposed = z["warped"].astype(bool)
    R = np.nan_to_num(z["pt_R"])
    scale = np.nan_to_num(z["pt_scale"])
    changed = np.r_[
        False,
        np.any(np.abs(np.diff(R, axis=0)) > 1e-12, axis=(1, 2))
        | np.any(np.abs(np.diff(scale, axis=0)) > 1e-12, axis=1),
    ]
    return proposed, proposed & changed


def main(reference):
    reference = Path(reference)
    per_label = collections.defaultdict(
        lambda: {
            "runs": 0,
            "proposed": 0,
            "kept": 0,
            "overshoot": 0,
            "drop": 0,
            "large_skl": 0,
        }
    )
    flagged = []
    for js in sorted(reference.glob("*.json")):
        npz = js.with_suffix(".npz")
        if not npz.exists():
            continue
        meta = json.loads(js.read_text(encoding="utf-8"))
        z = np.load(npz, allow_pickle=True)
        elbo = z["elbo"]
        lnZ = meta["ln_Z"]
        proposed, kept = kept_warps(z)
        counts = per_label[meta["label"]]
        counts["runs"] += 1
        counts["proposed"] += int(proposed.sum())
        counts["kept"] += int(kept.sum())
        for t in np.flatnonzero(kept):
            after = elbo[t + 1 : t + 4]
            drop = float(elbo[t - 1] - after.min()) if len(after) else None
            overshoot = float(elbo[t] - lnZ)
            is_drop = drop is not None and drop > DROP
            is_overshoot = overshoot > OVERSHOOT
            counts["drop"] += int(is_drop)
            counts["overshoot"] += int(is_overshoot)
            counts["large_skl"] += int(z["sKL"][t] > LARGE_SKL)
            if is_drop or is_overshoot:
                flagged.append(
                    {
                        "label": meta["label"],
                        "seed": meta["seed"],
                        "iter": int(t),
                        "elbo_before": float(elbo[t - 1]),
                        "elbo": float(elbo[t]),
                        "elbo_sd": float(z["elbo_sd"][t]),
                        "sKL": float(z["sKL"][t]),
                        "elbo_after": [float(e) for e in after],
                        "drop": drop,
                        "overshoot": overshoot,
                        "final_error": float(meta["final"]["elbo"] - lnZ),
                    }
                )
    totals = {
        key: sum(c[key] for c in per_label.values())
        for key in (
            "runs",
            "proposed",
            "kept",
            "overshoot",
            "drop",
            "large_skl",
        )
    }
    out = {
        "reference": reference.name,
        "overshoot_threshold": OVERSHOOT,
        "drop_threshold": DROP,
        "large_skl_threshold": LARGE_SKL,
        "totals": totals,
        "per_label": dict(sorted(per_label.items())),
        "flagged": flagged,
    }
    path = HERE / "golden_scan.json"
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(totals))
    for row in flagged:
        if row["drop"] is not None and row["drop"] > DROP:
            print("drop:", json.dumps(row))
    print("wrote", path)


if __name__ == "__main__":
    main(sys.argv[1])
