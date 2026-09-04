"""Generate (or check) the stage-level oracle fixtures.

Runs a handful of short, seeded VBMC runs on the benchmark targets with
regime-forcing options, snapshots the algorithm state at chosen iterations
as plain arrays (``pyvbmc/testing/oracles/_state.py``), computes every
applicable oracle (``_oracles.py``) on the *rebuilt* state and stores the
outputs as the reference next to the state. Fixtures land in
``pyvbmc/testing/oracles/fixtures/`` and are read by ``test_oracles.py``.

    python dev/scripts/make_oracle_fixtures.py --list
    python dev/scripts/make_oracle_fixtures.py            # regenerate all
    python dev/scripts/make_oracle_fixtures.py --only cigar_D4_largeK
    python dev/scripts/make_oracle_fixtures.py --check    # recompute, compare

Regenerating **replaces the references**: do it only when the current code
is the one the references should pin (a fresh baseline), never to make a
failing oracle pass. One process, single BLAS thread, about six minutes.
Plan and worklog: ``dev/plans/fixture-generator-and-oracles.md``.
"""

import argparse
import copy
import os
import platform
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))

from benchmark_targets import find_config  # noqa: E402
from profile_run import git_info, pkg_version  # noqa: E402

from pyvbmc import VBMC  # noqa: E402
from pyvbmc.testing.oracles._oracles import (  # noqa: E402
    DEFAULT_SEED,
    ORACLES,
    applicable,
    compare,
    format_rows,
)
from pyvbmc.testing.oracles._state import (  # noqa: E402
    build_state,
    encode,
    load_snapshot,
    save_snapshot,
    snapshot_from_objects,
    snapshot_names,
)
from pyvbmc.variational_posterior import VariationalPosterior  # noqa: E402
from pyvbmc.vbmc.active_sample import _get_search_points  # noqa: E402

FIXTURES = REPO_ROOT / "pyvbmc" / "testing" / "oracles" / "fixtures"
PROBLEM_SEED = 0
N_CAND = 512
SIEVE = 2**13


# --------------------------------------------------------------------------
# recipes
# --------------------------------------------------------------------------


class Recipe:
    def __init__(self, name, config, options, pick, note, check=None):
        self.name = name
        self.config = config
        self.options = dict(options)
        self.pick = pick  # int | "last" | "last_warped" | "final_vp" | "k1"
        self.note = note
        self.check = check  # callable(snapshot_tree) -> None (asserts)


def _check_singlesample(tree):
    assert tree["gp"]["Ns"] == 1, f"expected Ns == 1, got {tree['gp']['Ns']}"


def _check_warped(tree):
    assert tree["pt"]["R_mat"] is not None, "expected a rotoscale warp"


def _check_noisy(tree):
    assert tree["logger"]["noise_flag"] and tree["gp"]["s2"] is not None
    assert tree["logger"]["S"] is not None


def _check_boosted(tree):
    assert tree["vp"]["K"] == 50, f"expected K == 50, got {tree['vp']['K']}"


def _check_k1(tree):
    assert tree["vp"]["K"] == 1


RECIPES = [
    Recipe(
        "normal_D2_warmup",
        "normal_D2",
        {},
        2,
        "warm-up iteration: K = 2, Ns = 8, no warp; the baseline state",
    ),
    Recipe(
        "normal_D2_K1",
        "normal_D2",
        {},
        "k1",
        "the warm-up GP of normal_D2 with a synthetic single-component VP "
        "(K = 1 branches of the entropy and density code)",
        _check_k1,
    ),
    Recipe(
        "normal_D2_singlesample",
        "normal_D2",
        {"stable_gp_sampling": 1, "min_iter": 0},
        "last",
        "GP hyperparameter sampling switched off from the first fit: a "
        "single posterior (the Ns = 1 squeeze regime)",
        _check_singlesample,
    ),
    Recipe(
        "corr_D5_warped",
        "corr_D5",
        {},
        "last_warped",
        "last iteration with a rotoscale warp in place (R_mat, scale set) "
        "on a correlated 5-D Gaussian",
        _check_warped,
    ),
    Recipe(
        "cigar_D4_largeK",
        "cigar_D4",
        {},
        "last",
        "last iteration of a correlated, ill-conditioned target: large K, "
        "warped transformer",
        _check_warped,
    ),
    Recipe(
        "cigar_D4_boosted",
        "cigar_D4",
        {},
        "final_vp",
        "the returned (final-boost, K = 50) VP with the best iteration's GP",
        _check_boosted,
    ),
    Recipe(
        "halfnormal_D2_bounded",
        "halfnormal_D2",
        {},
        "last",
        "finite bounds: probit transform and a non-constant log-Jacobian",
    ),
    Recipe(
        "rosenbrock_D2_noise1_viqr",
        "rosenbrock_D2_noise1",
        {"max_iter": 4, "min_iter": 0, "min_fun_evals": 0},
        "last",
        "noisy target on the VIQR path: per-point noise, pre-drawn "
        "importance samples for the VIQR/IMIQR acquisitions",
        _check_noisy,
    ),
]


# --------------------------------------------------------------------------
# runs and snapshots
# --------------------------------------------------------------------------

_RUNS = {}


def run_config(config, options):
    key = (config, tuple(sorted(options.items())))
    if key in _RUNS:
        return _RUNS[key]
    cfg = find_config(config)
    prob = cfg.make(seed=PROBLEM_SEED)
    args, opts = prob.vbmc_args()
    opts.update(display="off", plot=False, print_iteration_header=False)
    opts.update(options)
    vbmc = VBMC(*args, options=opts, seed=PROBLEM_SEED)
    t0 = time.perf_counter()
    vp, results = vbmc.optimize()
    print(
        f"[fixtures] ran {config} {options or ''}: {results['iterations']} "
        f"iterations, {results['func_count']} evaluations, "
        f"{time.perf_counter() - t0:.0f}s",
        flush=True,
    )
    _RUNS[key] = (vbmc, prob, vp, results)
    return _RUNS[key]


def pick_iteration(vbmc, pick):
    h = vbmc.iteration_history
    n = len(h["elbo"])
    if pick == "last":
        return n - 1
    if pick == "last_warped":
        for i in range(n - 1, -1, -1):
            if h["vp"][i].parameter_transformer.R_mat is not None:
                return i
        raise RuntimeError("no warped iteration in this run")
    if isinstance(pick, int):
        assert 0 <= pick < n, f"iteration {pick} not in 0..{n - 1}"
        return pick
    raise ValueError(pick)


def make_snapshot(recipe):
    vbmc, prob, vp_final, results = run_config(recipe.config, recipe.options)
    h = vbmc.iteration_history
    if recipe.pick == "final_vp":
        # `final_boost` re-optimizes against the *best* iteration's GP.
        i = int(results["best_iter"])
    elif recipe.pick == "k1":
        i = pick_iteration(vbmc, 2)
    else:
        i = pick_iteration(vbmc, recipe.pick)
    gp = h["gp"][i]
    fl = h["function_logger"][i]
    os_ = copy.deepcopy(h["optim_state"][i])
    vp = h["vp"][i]
    # The recorded importance samples belong to the GP before the last
    # evaluation of the iteration (they are stale for the recorded GP and
    # large); the oracles redraw them from the rebuilt state.
    if "active_importance_sampling" in os_:
        os_["active_importance_sampling"] = None
    # `vp`, `gp` and the logger are recorded after the variational fit, the
    # `optim_state` at the end of the iteration (after any warm-up trimming
    # of the logger): make sure both describe the same data.
    n_eff_logger = int(np.sum(fl.n_evals[fl.X_flag]))
    assert os_["N"] == fl.Xn + 1 and int(os_["n_eff"]) == n_eff_logger, (
        f"{recipe.name}: optim_state (N={os_['N']}, n_eff={os_['n_eff']}) and "
        f"logger (Xn+1={fl.Xn + 1}, n_eff={n_eff_logger}) disagree at "
        f"iteration {i}; pick another iteration"
    )
    # The VP's transformer must be the one the logger's rows and the GP's
    # inputs were produced with (a warp after this iteration would break
    # that for a VP taken from elsewhere, e.g. the returned one).
    pt = vp.parameter_transformer
    live = fl.X_flag
    assert np.allclose(
        pt(fl.X_orig[live]), fl.X[live], atol=1e-10
    ), f"{recipe.name}: transformer does not reproduce the logger rows"
    assert np.array_equal(gp.X, fl.X[live]) and np.array_equal(
        np.ravel(gp.y), np.ravel(fl.y[live])
    ), f"{recipe.name}: GP data differ from the logger's live rows"
    if recipe.pick == "final_vp":
        vp = vp_final
    elif recipe.pick == "k1":
        src = vp
        vp = VariationalPosterior(
            src.D,
            1,
            x0=np.zeros((1, src.D)),
            parameter_transformer=src.parameter_transformer,
        )
        w = np.ravel(src.w)
        vp.mu = (src.mu @ w).reshape(-1, 1)
        vp.sigma = np.array([[float(np.sum(w * np.ravel(src.sigma)))]])
        vp.lambd = np.array(src.lambd)
        vp.w = np.ones((1, 1))
        vp.eta = np.ones((1, 1))
        vp.stats = None
    meta = {
        "recipe": recipe.name,
        "config": recipe.config,
        "problem": prob.name,
        "D": int(prob.D),
        "noise_sd": prob.noise_sd,
        "problem_seed": PROBLEM_SEED,
        "recipe_options": recipe.options,
        "pick": recipe.pick,
        "note": recipe.note,
        "r_index": float(h["r_index"][i]),
        "best_iter": int(results["best_iter"]),
        "n_iterations": int(results["iterations"]),
        "K": int(vp.K),
        "Ns": len(gp.posteriors),
        "N": int(gp.X.shape[0]),
        "oracle_seed": DEFAULT_SEED,
        "git": git_info(),
        "versions": {
            p: pkg_version(p) for p in ("pyvbmc", "gpyreg", "numpy", "scipy")
        },
        "python": platform.python_version(),
        # The `active_sample_step` oracle reproduces only on the platform
        # (BLAS build) that generated the fixture; the test gates on this.
        "platform": platform.platform(),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    arrays, tree = snapshot_from_objects(
        vp, gp, fl, os_, vbmc.options, meta=meta, iteration=i
    )
    # Candidate set: a fixed subsample of the seeded sieve.
    vp_c = copy.deepcopy(vp)
    vp_c.rng = np.random.default_rng(DEFAULT_SEED)
    Xs, _ = _get_search_points(
        SIEVE, copy.deepcopy(os_), copy.deepcopy(fl), vp_c, vbmc.options
    )
    Xs = np.array(Xs[:: SIEVE // N_CAND][:N_CAND])
    tree["cand"] = {"Xs": encode(Xs, "cand/Xs", arrays)}
    if recipe.check is not None:
        recipe.check(tree)
    return arrays, tree, prob


def compute_references(path, fun):
    """Rebuild the stored state and run every applicable oracle."""
    snap = load_snapshot(path)
    refs = {}
    for name in applicable(build_state(snap, fun=fun)):
        state = build_state(snap, fun=fun)
        t0 = time.perf_counter()
        refs[name] = ORACLES[name](state, DEFAULT_SEED)
        print(f"    {name:22s} {time.perf_counter() - t0:6.2f}s", flush=True)
    return snap, refs


def generate(recipes):
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for recipe in recipes:
        print(f"[fixtures] {recipe.name}", flush=True)
        arrays, tree, prob = make_snapshot(recipe)
        path = FIXTURES / recipe.name
        save_snapshot(path, arrays, tree)  # state only, no references yet
        snap, refs = compute_references(path, prob.fun)
        for name, out in refs.items():
            tree["ref"][name] = encode(out, f"ref/{name}", arrays)
        save_snapshot(path, arrays, tree)
        # Round trip: reload, rebuild, recompute; must be identical.
        bad = check_one(path, prob.fun, exact=True)
        if bad:
            raise RuntimeError(f"round trip failed for {recipe.name}: {bad}")
        size = os.path.getsize(path.with_suffix(".npz")) + os.path.getsize(
            path.with_suffix(".json")
        )
        print(
            f"[fixtures] wrote {recipe.name} ({size / 1024:.0f} KB)",
            flush=True,
        )


def check_one(path, fun, exact=False, verbose=False):
    snap = load_snapshot(path)
    bad = []
    for name, ref in snap["ref"].items():
        orc = ORACLES[name]
        state = build_state(snap, fun=fun)
        if not orc.applies(state):
            bad.append((name, "not applicable on rebuilt state"))
            continue
        out = orc(state, snap["meta"]["oracle_seed"])
        rtol, atol = (0.0, 0.0) if exact else (orc.rtol, orc.atol)
        rows = compare(ref, out, rtol, atol)
        if verbose or not all(r[3] for r in rows):
            print(f"  [{path.stem}] {name}\n{format_rows(rows)}", flush=True)
        if not all(r[3] for r in rows):
            bad.append((name, [r[0] for r in rows if not r[3]]))
    return bad


def target_for(meta):
    prob = find_config(meta["config"]).make(seed=meta["problem_seed"])
    return prob.fun


def check(names, verbose):
    failures = {}
    for name in names:
        path = FIXTURES / name
        fun = target_for(load_snapshot(path)["meta"])
        bad = check_one(path, fun, verbose=verbose)
        print(
            f"[check] {name:28s} {'ok' if not bad else 'FAIL ' + str(bad)}",
            flush=True,
        )
        if bad:
            failures[name] = bad
    return failures


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument(
        "--only", default=None, help="comma-separated recipe names"
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    if args.list:
        for r in RECIPES:
            print(
                f"{r.name:28s} {r.config:24s} pick={r.pick!s:12s} {r.options or ''}"
            )
            print(f"{'':28s} {r.note}")
        print("oracles:", ", ".join(ORACLES))
        return 0
    wanted = set(args.only.split(",")) if args.only else None
    if wanted:
        unknown = wanted - {r.name for r in RECIPES}
        if unknown:
            sys.exit(f"unknown recipe(s): {sorted(unknown)}")
    if args.check:
        names = [
            n for n in snapshot_names(FIXTURES) if not wanted or n in wanted
        ]
        failures = check(names, args.verbose)
        print(
            f"[check] {len(names) - len(failures)} of {len(names)} fixtures ok"
        )
        return 1 if failures else 0
    recipes = [r for r in RECIPES if not wanted or r.name in wanted]
    generate(recipes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
