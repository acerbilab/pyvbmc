"""Write the S-VBMC test fixtures under ``pyvbmc/testing/svbmc/fixtures/``.

Three modes, each rewriting only its own files:

``convert``
    The thirty fitted posteriors shipped with S-VBMC 0.1.1 (three groups of
    ten pickles in the pinned checkout) become plain-array snapshots, one
    per posterior, with the source commit and the SHA-256 of every pickle
    recorded in the JSON sidecar. Needs the pinned checkout (``--source``)
    but not Torch.

``generate``
    Supplementary posteriors from short seeded VBMC runs on inline targets
    that the upstream corpus does not cover: a one-dimensional unbounded
    normal, a bounded two-dimensional normal fitted with a different
    plausible box in every run (so the runs carry different transforms),
    and a correlated three-dimensional normal that triggers the rotoscale
    warp. Runs that do not end ``stable`` are replaced by the next seed.
    Heavy: about ten VBMC runs.

``references``
    Regression references for the integrated class: for every group and
    every optimization mode, the weights, ELBO values and entropy after a
    seeded three-step optimization. Needs Torch (the compatibility
    campaign's overlay on ``PYTHONPATH`` works).

Run from the repository root, for example::

    python dev/scripts/make_svbmc_fixtures.py convert
    python dev/scripts/make_svbmc_fixtures.py generate
    PYTHONPATH=dev/scripts/runs/svbmc_compat_20260908/deps \\
        python dev/scripts/make_svbmc_fixtures.py references

``FIXTURES.md`` next to the fixtures documents the format and every file.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import hashlib
import io
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pyvbmc.testing.oracles._state import save_snapshot  # noqa: E402
from pyvbmc.testing.svbmc._fixtures import (  # noqa: E402
    FIXTURES_DIR,
    REFERENCES,
    fixture_names,
    group_names,
    load_group,
    save_vp,
)

DEFAULT_SOURCE = REPO / "dev/scripts/runs/svbmc_compat_20260908/source"
UPSTREAM_GROUPS = ("GMM", "GMM_noisy", "Ring")
MODES = ("all-weights", "posterior-only", "ns")


def _git_head(path):
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# convert
# --------------------------------------------------------------------------


def convert(args):
    source = args.source
    commit = _git_head(source)
    written = []
    for group in UPSTREAM_GROUPS:
        files = sorted(glob.glob(str(source / "vbmc_runs" / group / "*.pkl")))
        if len(files) != 10:
            raise SystemExit(
                f"{group}: expected 10 pickles, found {len(files)}"
            )
        for i, f in enumerate(files):
            with open(f, "rb") as fh:
                vp = pickle.load(fh)
            name = f"upstream_{group}_{i:02d}"
            meta = {
                "group": f"upstream_{group}",
                "origin": "svbmc",
                "svbmc_commit": commit,
                "svbmc_version": "0.1.1",
                "source_file": f"vbmc_runs/{group}/{Path(f).name}",
                "source_sha256": _sha256(f),
                "D": int(vp.D),
                "K": int(vp.K),
                "stable": bool(vp.stats["stable"]),
                "elbo": float(vp.stats["elbo"]),
                "elbo_sd": float(vp.stats["elbo_sd"]),
                "converted": time.strftime("%Y-%m-%d"),
            }
            save_vp(name, vp, meta)
            written.append(name)
    print(f"converted {len(written)} posteriors from {source} @ {commit[:7]}")


# --------------------------------------------------------------------------
# generate
# --------------------------------------------------------------------------


def _normal_d1():
    def log_density(x):
        x = np.atleast_2d(x)
        return -0.5 * ((x[:, 0] - 0.5) / 1.2) ** 2

    return dict(
        fun=log_density,
        D=1,
        lb=None,
        ub=None,
        boxes=[(-3.0, 4.0)],
        description="log N(x; 0.5, 1.2^2) up to a constant, unbounded",
    )


def _bounded_d2():
    mean = np.array([0.5, -0.5])
    sd = np.array([0.8, 0.6])

    def log_density(x):
        x = np.atleast_2d(x)
        return -0.5 * np.sum(((x - mean) / sd) ** 2, axis=1)

    return dict(
        fun=log_density,
        D=2,
        lb=np.array([[-3.0, -3.0]]),
        ub=np.array([[3.0, 3.0]]),
        # One plausible box per run: the transforms differ across runs.
        boxes=[
            (np.array([[-2.0, -2.0]]), np.array([[2.0, 2.0]])),
            (np.array([[-1.5, -1.5]]), np.array([[1.5, 1.5]])),
            (np.array([[-1.0, -2.5]]), np.array([[2.5, 1.0]])),
            (np.array([[-2.5, -1.0]]), np.array([[1.0, 2.5]])),
        ],
        description=(
            "log N(x; [0.5, -0.5], diag(0.8^2, 0.6^2)) on the box "
            "[-3, 3]^2 (probit transform); a different plausible box "
            "in every run"
        ),
    )


def _corr_d3():
    sd = np.array([1.0, 2.0, 0.5])
    corr = np.full((3, 3), 0.85)
    np.fill_diagonal(corr, 1.0)
    cov = corr * np.outer(sd, sd)
    chol = np.linalg.cholesky(cov)

    def log_density(x):
        x = np.atleast_2d(x)
        z = np.linalg.solve(chol, x.T)
        return -0.5 * np.sum(z**2, axis=0)

    return dict(
        fun=log_density,
        D=3,
        lb=None,
        ub=None,
        boxes=[(np.full((1, 3), -4.0), np.full((1, 3), 4.0))],
        description=(
            "log N(x; 0, S R S) with sd (1, 2, 0.5) and all correlations "
            "0.85, unbounded; strong correlation triggers the rotoscale warp"
        ),
    )


TARGETS = {
    "normal_D1": _normal_d1,
    "bounded_D2": _bounded_d2,
    "corr_D3": _corr_d3,
}


def _run_vbmc(spec, box, seed, options):
    from pyvbmc import VBMC

    plb, pub = np.atleast_2d(box[0]), np.atleast_2d(box[1])
    rng = np.random.default_rng(seed)
    x0 = plb + (pub - plb) * rng.random((1, spec["D"]))
    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet):
        vbmc = VBMC(
            spec["fun"],
            x0,
            spec["lb"],
            spec["ub"],
            plb,
            pub,
            options=options,
            seed=seed,
        )
        vp, results = vbmc.optimize()
    return vp, results


def generate(args):
    commit = _git_head(REPO)
    options = {"display": "off"}
    for name in args.targets:
        spec = TARGETS[name]()
        stable = []
        seed = args.base_seed
        attempts = 0
        while len(stable) < args.runs and attempts < args.max_attempts:
            box = spec["boxes"][attempts % len(spec["boxes"])]
            t0 = time.perf_counter()
            vp, results = _run_vbmc(spec, box, seed, options)
            elapsed = time.perf_counter() - t0
            warped = vp.parameter_transformer.R_mat is not None
            ok = bool(vp.stats["stable"]) and np.all(
                np.isfinite(vp.stats["I_sk"])
            )
            print(
                f"{name} seed {seed}: stable={vp.stats['stable']} "
                f"warped={warped} K={vp.K} elbo={vp.stats['elbo']:.3f} "
                f"sd={vp.stats['elbo_sd']:.3f} ({elapsed:.0f}s)"
                + ("" if ok else "  -> discarded")
            )
            if ok:
                idx = len(stable)
                meta = {
                    "group": name,
                    "origin": "pyvbmc",
                    "pyvbmc_commit": commit,
                    "target": spec["description"],
                    "seed": int(seed),
                    "plausible_lower": np.atleast_2d(box[0]).ravel().tolist(),
                    "plausible_upper": np.atleast_2d(box[1]).ravel().tolist(),
                    "options": options,
                    "D": int(vp.D),
                    "K": int(vp.K),
                    "stable": bool(vp.stats["stable"]),
                    "warped": bool(warped),
                    "elbo": float(vp.stats["elbo"]),
                    "elbo_sd": float(vp.stats["elbo_sd"]),
                    "fun_evals": int(results["func_count"]),
                    "generated": time.strftime("%Y-%m-%d"),
                }
                save_vp(f"{name}_{idx:02d}", vp, meta)
                stable.append(vp)
            seed += 1
            attempts += 1
        if len(stable) < args.runs:
            raise SystemExit(
                f"{name}: only {len(stable)} stable runs in {attempts} attempts"
            )
        if name == "corr_D3" and not any(
            vp.parameter_transformer.R_mat is not None for vp in stable
        ):
            raise SystemExit("corr_D3: no run ended with a warp in place")


# --------------------------------------------------------------------------
# references
# --------------------------------------------------------------------------


def references(args):
    import torch

    from pyvbmc.svbmc import SVBMC

    torch.set_num_threads(1)
    arrays = {}
    tree = {
        "meta": {
            "pyvbmc_commit": _git_head(REPO),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "seed": args.seed,
            "max_steps": args.steps,
            "generated": time.strftime("%Y-%m-%d"),
            "description": (
                "SVBMC(vps, seed=seed).optimize(max_steps=steps, "
                "version=mode) on every fixture group; the fixtures were "
                "loaded with load_group(group, rng=0)"
            ),
        },
        "groups": {},
    }
    for group in group_names():
        vps, _ = load_group(group, rng=0)
        tree["groups"][group] = {}
        for mode in MODES:
            quiet = io.StringIO()
            with contextlib.redirect_stdout(quiet):
                stacked = SVBMC(vps, seed=args.seed)
                stacked.optimize(max_steps=args.steps, version=mode)
            key = f"{group}/{mode}"
            arrays[f"{key}/w"] = np.array(stacked.w)
            tree["groups"][group][mode] = {
                "w": f"@@npz:{key}/w",
                "elbo": {k: float(v) for k, v in stacked.elbo.items()},
                "entropy": float(stacked.entropy),
                "M": int(stacked.M),
                "K": [int(k) for k in stacked.K],
            }
            print(
                f"{group:20s} {mode:15s} elbo={stacked.elbo['estimated']:.6f} "
                f"H={stacked.entropy:.6f}"
            )
    save_snapshot(FIXTURES_DIR / REFERENCES, arrays, tree)
    print(f"wrote {FIXTURES_DIR / REFERENCES}.npz/.json")


# --------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("convert")
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.set_defaults(func=convert)

    p = sub.add_parser("generate")
    p.add_argument("--targets", nargs="+", default=list(TARGETS))
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--base-seed", type=int, default=100)
    p.add_argument("--max-attempts", type=int, default=8)
    p.set_defaults(func=generate)

    p = sub.add_parser("references")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=3)
    p.set_defaults(func=references)

    args = parser.parse_args(argv)
    args.func(args)
    print(f"fixtures now: {len(fixture_names())} posteriors")


if __name__ == "__main__":
    main()
