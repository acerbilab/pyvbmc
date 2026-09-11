"""Export the data of the real-data benchmark targets from a benchflow checkout.

Reads the MATLAB files that benchflow's ``BayesianTiming`` and
``Multisensory_6D`` tasks load and writes plain NumPy archives under
``dev/scripts/data/`` (float64 arrays and int64 indices, readable with
``np.load(path, allow_pickle=False)``), so that the benchmark targets in
``benchmark_targets.py`` never touch a ``.mat`` file. ``data/README.md``
documents the layout and the provenance; ``provenance.json`` records the
benchflow commit and the SHA-256 of every source file.

Usage::

    python dev/scripts/export_benchflow_data.py [--benchflow ../benchflow]

``timing.npz`` (Bayesian time-interval reproduction, Acerbi, Wolpert &
Vijayakumar 2012; subject 2 of Experiment 3, 1512 trials):

``stim_index``     int64 (1512,)  0-based index of the trial's interval
``response``       float64 (1512,)  reproduced interval (s), binned at
                   ``bin_size``
``stimuli``        float64 (6,)  the six intervals (s)
``bin_size``       float64 ()  response discretization (s)
``lb, ub, plb, pub``  float64 (5,)  the paper's bounds (Table S2 of the
                   2020 paper), parameters ``(w_s, w_m, mu_p, sigma_p,
                   lambda)``
``paper_*``        the 2020 paper's ground truth for the uniform prior:
                   ``paper_ln_z`` (), ``paper_post_mean`` (5,),
                   ``paper_post_cov`` (5, 5), ``paper_post_mode`` (5,),
                   ``paper_post_mode_val`` (), ``paper_marginal_bounds``
                   (2, 5), ``paper_marginal_pdf`` (5, 8192), ``paper_mle``
                   (5,), ``paper_mle_val`` (). Kept for comparison; the
                   suite's truth is regenerated under its own prior.

``multisensory.npz`` (visuo-vestibular unity judgments, Acerbi, Dokka,
Angelaki & Ma 2018; subjects 1 and 2 of the 2020 paper, benchflow indices
0 and 1), for ``s`` in ``1, 2`` and coherence level ``c`` in ``1, 2, 3``:

``s{s}_c{c}_stim``  float64 (n, 2)  vestibular and visual directions (deg)
``s{s}_c{c}_resp``  int64 (n,)  response code as benchflow's likelihood
                    reads it: 2 when the subject judged the cues to come
                    from different sources, 1 otherwise
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "data"
TIMING_REL = Path("benchflow/tasks/task_info/timing/timing.mat")
MULTISENSORY_REL = Path(
    "benchflow/tasks/task_info/multisensory_6D/acerbidokka2018_data.mat"
)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit(repo):
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def export_timing(mat_path):
    y = loadmat(mat_path)["y"]
    data = y["Data"][0, 0]
    post = y["Post"][0, 0]
    X = np.asarray(data["X"][0, 0], dtype=float)
    stim_index = X[:, 2].astype(np.int64) - 1  # MATLAB 1-based
    stimuli = np.asarray(data["S"][0, 0], dtype=float).ravel()
    if stim_index.min() < 0 or stim_index.max() >= stimuli.size:
        raise ValueError("stimulus index out of range")
    out = {
        "stim_index": stim_index,
        "response": np.asarray(data["R"][0, 0], dtype=float).ravel(),
        "stimuli": stimuli,
        "bin_size": np.float64(data["binsize"][0, 0].item()),
        "lb": np.asarray(y["LB"][0, 0], dtype=float).ravel(),
        "ub": np.asarray(y["UB"][0, 0], dtype=float).ravel(),
        "plb": np.asarray(y["PLB"][0, 0], dtype=float).ravel(),
        "pub": np.asarray(y["PUB"][0, 0], dtype=float).ravel(),
        "paper_ln_z": np.float64(post["lnZ"][0, 0].item()),
        "paper_post_mean": np.asarray(post["Mean"][0, 0], dtype=float).ravel(),
        "paper_post_cov": np.asarray(post["Cov"][0, 0], dtype=float),
        "paper_post_mode": np.asarray(post["Mode"][0, 0], dtype=float).ravel(),
        "paper_post_mode_val": np.float64(post["ModeFval"][0, 0].item()),
        "paper_marginal_bounds": np.asarray(
            post["MarginalBounds"][0, 0], dtype=float
        ),
        "paper_marginal_pdf": np.asarray(
            post["MarginalPdf"][0, 0], dtype=float
        ),
        "paper_mle": np.asarray(y["Mode"][0, 0], dtype=float).ravel(),
        "paper_mle_val": np.float64(y["ModeFval"][0, 0].item()),
    }
    if out["response"].shape != (X.shape[0],):
        raise ValueError("response count differs from trial count")
    return out


def export_multisensory(mat_path):
    unity = loadmat(mat_path, squeeze_me=True)["unity_data"]
    out = {}
    for s in (1, 2):
        subject = unity[s - 1]
        if len(subject) != 3:
            raise ValueError(f"subject {s}: expected 3 coherence levels")
        for c in (1, 2, 3):
            a = np.atleast_2d(np.asarray(subject[c - 1], dtype=float))
            if a.shape[1] != 5:
                raise ValueError(f"subject {s} level {c}: expected 5 columns")
            resp = a[:, 4]
            if not np.all(np.isin(resp, (1.0, 2.0))):
                raise ValueError(
                    f"subject {s} level {c}: response not in 1, 2"
                )
            out[f"s{s}_c{c}_stim"] = np.ascontiguousarray(a[:, 2:4])
            out[f"s{s}_c{c}_resp"] = resp.astype(np.int64)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--benchflow",
        type=Path,
        default=REPO_ROOT.parent / "benchflow",
        help="benchflow checkout (default: sibling of this repository)",
    )
    args = ap.parse_args(argv)
    sources = {
        "timing.npz": args.benchflow / TIMING_REL,
        "multisensory.npz": args.benchflow / MULTISENSORY_REL,
    }
    for path in sources.values():
        if not path.is_file():
            sys.exit(f"missing source file: {path}")
    OUT_DIR.mkdir(exist_ok=True)
    timing = export_timing(sources["timing.npz"])
    multisensory = export_multisensory(sources["multisensory.npz"])
    np.savez(OUT_DIR / "timing.npz", **timing)
    np.savez(OUT_DIR / "multisensory.npz", **multisensory)
    provenance = {
        "benchflow_commit": _git_commit(args.benchflow),
        "sources": {
            name: {
                "path": str(path.relative_to(args.benchflow)).replace(
                    "\\", "/"
                ),
                "sha256": _sha256(path),
            }
            for name, path in sources.items()
        },
        "trials": {
            "timing": int(timing["response"].size),
            **{
                f"multisensory_s{s}": int(
                    sum(
                        multisensory[f"s{s}_c{c}_resp"].size for c in (1, 2, 3)
                    )
                )
                for s in (1, 2)
            },
        },
    }
    with open(OUT_DIR / "provenance.json", "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
        f.write("\n")
    for name in sources:
        with np.load(OUT_DIR / name, allow_pickle=False) as z:
            print(f"{name}: {len(z.files)} arrays")
    print(json.dumps(provenance["trials"]))


if __name__ == "__main__":
    main()
