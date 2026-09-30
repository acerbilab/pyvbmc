"""Export the points of the video's opening: an MCMC run and a PyBADS run.

    python scripts/export_intro.py [--evals 27828] [--walkers 32] [--seed 1]

Both explore the target of ``trace_wordmark.js`` (the banana alone) with the
plausible box of its PyVBMC run. Writes ``trace_intro.js`` next to the pages
(``window.VBMC_INTRO = {...}``):

``chain``: the states of an emcee ensemble, one per evaluation of the
target, as little-endian int16 pairs in base64, each coordinate scaled from
``chain_range``. The ensemble uses the stretch move with ``--walkers``
walkers drawn uniformly in the plausible box, the setting and the stopping
point of the headline of ``dev/results/2026-09-30-matched-mcmc-budget.md``
on ``dev-next``: emcee needs a median of 27,828 evaluations to match the
MMTV of a typical PyVBMC run on this target. The states are the walkers'
starting points, then the walkers' positions after each step, cut at
``--evals``. Within a step they follow the walkers' order, which is not the
order of their evaluations (emcee evaluates the walkers in two halves drawn
at random), but a state and its evaluation fall in the same step.
``bads``: every point that PyBADS evaluated, in order, ``[x, y]``;
``bads_best``: the point it returned.
``meta``: the target, the seeds, the ensemble's walkers, evaluations and
acceptance fraction, and the versions of emcee and PyBADS.

Needs numpy, emcee and pybads (none of them a dependency of PyVBMC). The target is the
banana of ``dev/scripts/export_animation_trace.py`` with no lobe, written
out again here so that PyVBMC need not be installed; the script checks the
copy against the log densities that the committed trace recorded.
"""

import argparse
import base64
import json
import re
from importlib.metadata import version
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
X0 = np.array([-3.0, 5.5])  # the starting point of the PyVBMC run
PLB, PUB = np.array([-5.0, -3.5]), np.array([5.0, 7.5])
LB, UB = np.array([-7.0, -5.0]), np.array([7.0, 9.0])  # the display window
CHAIN_RANGE = np.array([[-12.0, 12.0], [-8.0, 24.0]])


def log_density(X):
    """The banana: x ~ N(0, 2^2) and y - (x^2 - 4) / 2 ~ N(0, 1)."""
    X = np.atleast_2d(X)
    z2 = X[:, 1] - 0.5 * (X[:, 0] ** 2 - 4.0)
    return (
        -0.5 * (X[:, 0] / 2.0) ** 2
        - 0.5 * z2**2
        - np.log(2.0)
        - np.log(2 * np.pi)
    )


def check_target():
    """Compare the target with the evaluations stored in the trace."""
    text = (HERE / "trace_wordmark.js").read_text(encoding="utf-8")
    points = np.array(
        json.loads(re.search(r"\"points\":(\[\[.*?\]\])", text)[1])
    )
    err = np.abs(log_density(points[:, :2]) - points[:, 2]).max()
    if err > 1e-3:
        raise SystemExit(f"the target differs from the trace's by {err:.2e}")
    return len(points)


def ensemble(n_evals, walkers, seed):
    """Run emcee until ``n_evals`` evaluations; the states in their order."""
    import emcee

    calls = [0]

    def log_prob(x):
        calls[0] += 1
        return log_density(x)[0]

    np.random.seed(seed)  # emcee copies NumPy's global state when built
    p0 = np.random.default_rng(seed).uniform(PLB, PUB, (walkers, 2))
    steps = -(-(n_evals - walkers) // walkers)
    sampler = emcee.EnsembleSampler(walkers, 2, log_prob)
    sampler.run_mcmc(p0, steps)
    if calls[0] != walkers * (steps + 1):
        raise SystemExit(
            f"emcee made {calls[0]} evaluations, not {walkers * (steps + 1)}"
        )
    states = np.concatenate([p0, sampler.get_chain().reshape(-1, 2)])
    return states[:n_evals], float(np.mean(sampler.acceptance_fraction))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--evals", type=int, default=27828)
    ap.add_argument("--walkers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    print(f"target checked on {check_target()} evaluations", flush=True)

    chain, rate = ensemble(args.evals, args.walkers, args.seed)
    print(
        f"emcee: {len(chain)} evaluations with {args.walkers} walkers, "
        f"acceptance {rate:.3f}",
        flush=True,
    )

    from pybads import BADS

    evaluated = []

    def neg_log_density(x):
        evaluated.append([round(float(v), 4) for v in x])
        return -log_density(x)[0]

    bads = BADS(
        neg_log_density,
        X0,
        LB,
        UB,
        PLB,
        PUB,
        options={"display": "off", "random_seed": args.seed},
    )
    res = bads.optimize()
    best = [round(float(v), 4) for v in res["x"]]
    print(
        f"PyBADS: {len(evaluated)} evaluations, best {best}, "
        f"log density {-res['fval']:.4f} (the maximum is "
        f"{log_density([0.0, -2.0])[0]:.4f} at [0, -2])",
        flush=True,
    )

    lo, hi = CHAIN_RANGE[:, 0], CHAIN_RANGE[:, 1]
    q = np.round((np.clip(chain, lo, hi) - lo) / (hi - lo) * 65535 - 32768)
    trace = {
        "meta": {
            "target": "twisted Gaussian",
            "x0": X0.tolist(),
            "seed": args.seed,
            "sampler": "emcee",
            "walkers": args.walkers,
            "evals": len(chain),
            "acceptance": round(rate, 4),
            "emcee": version("emcee"),
            "pybads": version("pybads"),
            "bads_evals": len(evaluated),
        },
        "chain_range": CHAIN_RANGE.tolist(),
        "chain": base64.b64encode(q.astype("<i2").tobytes()).decode(),
        "bads": evaluated,
        "bads_best": best,
    }
    out = HERE / "trace_intro.js"
    payload = json.dumps(trace, separators=(",", ":"))
    out.write_text(f"window.VBMC_INTRO={payload};\n", encoding="utf-8")
    print(f"wrote {out} ({len(payload) / 1e3:.0f} kB)", flush=True)


if __name__ == "__main__":
    main()
