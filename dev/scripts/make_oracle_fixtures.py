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
    python dev/scripts/make_oracle_fixtures.py --check --exact   # bit for bit
    python dev/scripts/make_oracle_fixtures.py --rebaseline active_sample_step \
        --reason "..."                     # one oracle, from the stored state
    python dev/scripts/make_oracle_fixtures.py --add-oracle gp_fit \
        --reason "..."                     # a new oracle, from the stored state
    python dev/scripts/make_oracle_fixtures.py --dump-outputs DIR   # current outputs
    python dev/scripts/make_oracle_fixtures.py --check --exact --against DIR

Regenerating **replaces the references**: do it only when the current code
is the one the references should pin (a fresh baseline), never to make a
failing oracle pass. One process, single BLAS thread, about six minutes.
``--rebaseline`` replaces a single oracle's references without rerunning
the source run (whose trajectory would move and take every other reference
with it); it is for the one oracle that is *expected* to change under an
arithmetic-preserving refactor, the CMA-ES search of ``active_sample_step``,
after the ``acq_*`` oracles have confirmed the acquisition itself.
``--add-oracle`` computes a newly registered oracle from the stored state
and adds its references, leaving every existing array bit-identical (the
recipes are not rerun, so the snapshots keep pinning what they pin).
``--check --exact`` compares bit for bit instead of at the tolerances. The
committed references pin the numerics of the day they were made (several
outputs have since moved within tolerance, Stage 2 items 1–3), so the gate
for an identity-preserving refactor is ``--dump-outputs DIR`` on the code
just before it and ``--check --exact --against DIR`` after. Plan and
worklog: ``dev/plans/fixture-generator-and-oracles.md``.
"""

import argparse
import copy
import json
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
    PLATFORM_BOUND,
    applicable,
    compare,
    format_rows,
)
from pyvbmc.testing.oracles._state import (  # noqa: E402
    _files,
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


def check_one(path, fun, exact=False, verbose=False, reference=None, skip=()):
    """Recompute every stored oracle of one snapshot and compare with the
    stored references, or with ``reference`` (``{oracle: {key: array}}``,
    e.g. a dump of an earlier code state, see :func:`dump_outputs`);
    oracles named in ``skip`` are not checked."""
    snap = load_snapshot(path)
    refs = snap["ref"] if reference is None else reference
    bad = []
    for name, ref in refs.items():
        if name in skip:
            continue
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


def _dump_files(out_dir, name):
    out_dir = Path(out_dir)
    return out_dir / (name + ".npz"), out_dir / (name + ".json")


def dump_outputs(names, out_dir):
    """Write the *current* code's outputs of every stored oracle on every
    snapshot to ``out_dir`` (one ``.npz`` per snapshot, keys
    ``<oracle>/<output>``, plus a ``.json`` with the provenance).

    The committed references pin the numerics of the day they were made,
    at tolerance; a refactor that must not change any output at all is
    checked against a dump of the code just before it (``--check --exact
    --against <dir>``), which the dump makes reproducible without a second
    checkout.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = FIXTURES / name
        snap = load_snapshot(path)
        fun = target_for(snap["meta"])
        arrays = {}
        for oracle_name in snap["ref"]:
            state = build_state(snap, fun=fun)
            out = ORACLES[oracle_name](state, snap["meta"]["oracle_seed"])
            for key, val in out.items():
                arrays[f"{oracle_name}/{key}"] = np.asarray(val, dtype=float)
        npz, js = _dump_files(out_dir, name)
        np.savez_compressed(npz, **arrays)
        with open(js, "w", encoding="utf-8", newline="\n") as f:
            json.dump(
                {
                    "snapshot": name,
                    "oracles": sorted(snap["ref"]),
                    "git": git_info(),
                    "platform": platform.platform(),
                    "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "threads": {
                        k: os.environ.get(k)
                        for k in (
                            "OMP_NUM_THREADS",
                            "OPENBLAS_NUM_THREADS",
                            "MKL_NUM_THREADS",
                        )
                    },
                },
                f,
                indent=1,
                sort_keys=True,
            )
            f.write("\n")
        print(f"[dump] {name:28s} {len(arrays)} arrays", flush=True)


def load_dump(out_dir, name):
    npz, _ = _dump_files(out_dir, name)
    reference = {}
    with np.load(npz, allow_pickle=False) as z:
        for key in z.files:
            oracle_name, output = key.split("/", 1)
            reference.setdefault(oracle_name, {})[output] = z[key]
    return reference


def check(names, verbose, exact=False, against=None):
    failures = {}
    for name in names:
        path = FIXTURES / name
        fun = target_for(load_snapshot(path)["meta"])
        reference = None if against is None else load_dump(against, name)
        bad = check_one(
            path, fun, exact=exact, verbose=verbose, reference=reference
        )
        print(
            f"[check] {name:28s} {'ok' if not bad else 'FAIL ' + str(bad)}",
            flush=True,
        )
        if bad:
            failures[name] = bad
    return failures


def _refuse_off_platform(name, oracle_name, meta):
    if (
        oracle_name in PLATFORM_BOUND
        and meta.get("platform") != platform.platform()
    ):
        sys.exit(
            f"{name}: {oracle_name} is platform-bound and this is not the"
            f" generating platform ({meta.get('platform')})"
        )


def _write_fixture(path, arrays, tree):
    """Write both files through temporary files outside the fixtures
    directory (which the tests glob), renamed into place."""
    npz, js = _files(path)
    tmp = FIXTURES.parent / (path.name + ".rewrite-tmp")
    tmp_npz, tmp_js = _files(tmp)
    try:
        save_snapshot(tmp, arrays, tree)
        os.replace(tmp_npz, npz)  # the pair is not atomic together;
        os.replace(tmp_js, js)  # the json holds only markers + meta
    finally:
        for p in (tmp_npz, tmp_js):
            if p.exists():
                p.unlink()


def _verify_rewrite(path, fun, oracle_name, before, extra_keys=(), skip=()):
    """After a targeted rewrite: every array outside ``ref/<oracle>/`` is
    bit-identical to ``before`` (and only ``extra_keys`` were added), the
    rewritten oracle reproduces exactly from the stored state, and the
    other oracles still pass at their own tolerances (not bit-exactly: a
    refactor that these modes exist for may have moved them by ulps),
    except those named in ``skip`` (oracles known to move as well, to be
    re-baselined next)."""
    prefix = f"ref/{oracle_name}/"
    npz, _ = _files(path)
    with np.load(npz, allow_pickle=False) as z:
        after = {k: z[k] for k in z.files}
    assert set(after) == set(before) | set(extra_keys), path.name
    for k in before:
        if not k.startswith(prefix):
            assert np.array_equal(before[k], after[k], equal_nan=True), (
                path.name,
                k,
            )
    snap = load_snapshot(path)
    state = build_state(snap, fun=fun)
    again = ORACLES[oracle_name](state, snap["meta"]["oracle_seed"])
    exact = compare(snap["ref"][oracle_name], again, 0.0, 0.0)
    if not all(r[3] for r in exact):
        raise RuntimeError(
            f"{path.name}: {oracle_name} does not reproduce from the stored"
            f" state\n{format_rows(exact)}"
        )
    bad = check_one(path, fun, exact=False, skip=skip)
    if bad:
        raise RuntimeError(f"{path.name}: other oracles fail after: {bad}")


def add_oracle(names, oracle_name, reason, expect_moving=()):
    """Compute a newly registered oracle from the *stored* state of each
    snapshot and add its references, leaving everything else bit-identical
    (``expect_moving`` as in :func:`rebaseline`).

    The counterpart of :func:`rebaseline` for an oracle the fixtures do not
    hold yet: rerunning the recipes would move every snapshot (the runs'
    trajectories change with every Stage 2 item), so a new oracle is pinned
    on the states as they are. Records the event under
    ``meta["oracles_added"]``.
    """
    orc = ORACLES[oracle_name]
    prefix = f"ref/{oracle_name}/"
    pending = []
    for name in names:
        path = FIXTURES / name
        npz, js = _files(path)
        tree = json.loads(js.read_text(encoding="utf-8"))
        with np.load(npz, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        snap = load_snapshot(path)
        meta = snap["meta"]
        if oracle_name in snap["ref"]:
            sys.exit(
                f"{name}: {oracle_name} already has a reference"
                " (use --rebaseline to replace it)"
            )
        _refuse_off_platform(name, oracle_name, meta)
        fun = target_for(meta)
        state = build_state(snap, fun=fun)
        if not orc.applies(state):
            print(
                f"[add-oracle] {name:28s} {oracle_name} not applicable; skip"
            )
            continue
        out = orc(state, meta["oracle_seed"])
        shapes = {k: list(np.shape(v)) for k, v in out.items()}
        print(f"  [{name}] {oracle_name} outputs {shapes}")
        pending.append((name, path, fun, arrays, tree, out, shapes))

    for name, path, fun, arrays, tree, out, shapes in pending:
        before = {k: v.copy() for k, v in arrays.items()}
        tree["ref"][oracle_name] = encode(out, f"ref/{oracle_name}", arrays)
        tree["meta"].setdefault("oracles_added", []).append(
            {
                "oracle": oracle_name,
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "git": git_info(),
                "reason": reason,
                "outputs": shapes,
            }
        )
        _write_fixture(path, arrays, tree)
        _verify_rewrite(
            path, fun, oracle_name, before, [prefix + k for k in out]
        )
        print(f"[add-oracle] {name:28s} {oracle_name} added", flush=True)
    print(f"[add-oracle] {len(pending)} of {len(names)} fixtures rewritten")
    return len(pending)


def rebaseline(names, oracle_name, reason, expect_moving=()):
    """Recompute one oracle from the *stored* state of each snapshot and
    replace only its references.

    Regenerating a recipe would rerun its VBMC run, whose trajectory moves
    with any change to the search, so every reference of the snapshot
    would move and stop pinning the pre-change numerics. This mode keeps
    the state and every other oracle's reference bit-identical (asserted
    after the write) and records the event under ``meta["rebaselined"]``.
    When a change moves more than one oracle (a random-stream change moves
    every oracle that draws), name the others in ``expect_moving`` so the
    post-write check does not fail on them; re-baseline them next.
    """
    orc = ORACLES[oracle_name]
    prefix = f"ref/{oracle_name}/"
    # Phase 1: recompute for every snapshot and check, writing nothing, so
    # that a refusal or a mismatch leaves every fixture as it was.
    pending = []
    for name in names:
        path = FIXTURES / name
        npz, js = _files(path)
        tree = json.loads(js.read_text(encoding="utf-8"))
        with np.load(npz, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        snap = load_snapshot(path)
        meta = snap["meta"]
        if oracle_name not in snap["ref"]:
            print(f"[rebaseline] {name:28s} no {oracle_name} reference; skip")
            continue
        _refuse_off_platform(name, oracle_name, meta)
        fun = target_for(meta)
        state = build_state(snap, fun=fun)
        if not orc.applies(state):
            sys.exit(f"{name}: {oracle_name} not applicable on rebuilt state")
        out = orc(state, meta["oracle_seed"])
        old = snap["ref"][oracle_name]
        if set(out) != set(old):
            sys.exit(
                f"{name}: {oracle_name} outputs {sorted(out)} but the"
                f" reference holds {sorted(old)}"
            )
        for key, val in out.items():
            if np.shape(val) != np.shape(old[key]):
                sys.exit(
                    f"{name}: {oracle_name}/{key} changed shape"
                    f" {np.shape(old[key])} -> {np.shape(val)}"
                )
        rows = compare(old, out, orc.rtol, orc.atol)
        print(f"  [{name}] {oracle_name} old vs new\n{format_rows(rows)}")
        pending.append((name, path, fun, arrays, tree, out, rows))

    # Phase 2: write each fixture through temporary files (outside the
    # fixtures directory, which the tests glob) renamed into place, then
    # verify that every other array is bit-identical, that the new
    # reference reproduces exactly from the stored state, and that the
    # other oracles still pass at their own tolerances (not bit-exactly:
    # a refactor that this mode exists for may have moved them by ulps).
    def finite(v):
        return float(v) if np.isfinite(v) else repr(float(v))

    for name, path, fun, arrays, tree, out, rows in pending:
        before = {k: v.copy() for k, v in arrays.items()}
        for key, val in out.items():
            arrays[prefix + key] = np.asarray(val, dtype=float)
        tree["meta"].setdefault("rebaselined", []).append(
            {
                "oracle": oracle_name,
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "git": git_info(),
                "reason": reason,
                "max_abs_change": {r[0]: finite(r[1]) for r in rows},
            }
        )
        _write_fixture(path, arrays, tree)
        _verify_rewrite(path, fun, oracle_name, before, skip=expect_moving)
        print(f"[rebaseline] {name:28s} {oracle_name} replaced", flush=True)
    print(f"[rebaseline] {len(pending)} of {len(names)} fixtures rewritten")
    return len(pending)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument(
        "--only", default=None, help="comma-separated recipe names"
    )
    ap.add_argument(
        "--rebaseline",
        default=None,
        metavar="ORACLE",
        help="replace one oracle's references from the stored state"
        " (needs --reason); every other reference stays bit-identical",
    )
    ap.add_argument(
        "--add-oracle",
        default=None,
        metavar="ORACLE",
        help="add a newly registered oracle's references from the stored"
        " state (needs --reason); every existing array stays bit-identical",
    )
    ap.add_argument("--reason", default=None)
    ap.add_argument(
        "--expect-moving",
        default=None,
        metavar="ORACLES",
        help="with --rebaseline / --add-oracle: comma-separated oracles that"
        " the same change moves too (re-baselined next); excluded from the"
        " post-write check",
    )
    ap.add_argument(
        "--exact",
        action="store_true",
        help="with --check: compare bit for bit instead of at the tolerances",
    )
    ap.add_argument(
        "--against",
        default=None,
        metavar="DIR",
        help="with --check: compare with a dump of an earlier code state"
        " (see --dump-outputs) instead of the stored references",
    )
    ap.add_argument(
        "--dump-outputs",
        default=None,
        metavar="DIR",
        help="write the current code's outputs of every stored oracle to DIR",
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    targeted = [m for m in (args.rebaseline, args.add_oracle) if m]
    modes = len(targeted) + args.list + args.check + bool(args.dump_outputs)
    if modes > 1:
        sys.exit(
            "--list, --check, --rebaseline, --add-oracle and --dump-outputs"
            " are mutually exclusive"
        )
    if (args.exact or args.against) and not args.check:
        sys.exit("--exact and --against only apply to --check")
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
        failures = check(
            names, args.verbose, exact=args.exact, against=args.against
        )
        print(
            f"[check{' --exact' if args.exact else ''}"
            f"{' --against ' + args.against if args.against else ''}]"
            f" {len(names) - len(failures)} of {len(names)} fixtures ok"
        )
        return 1 if failures else 0
    if args.dump_outputs:
        names = [
            n for n in snapshot_names(FIXTURES) if not wanted or n in wanted
        ]
        dump_outputs(names, args.dump_outputs)
        return 0
    if targeted:
        oracle_name = targeted[0]
        if oracle_name not in ORACLES:
            sys.exit(f"unknown oracle {oracle_name!r}")
        if not args.reason:
            sys.exit("--rebaseline / --add-oracle need --reason")
        names = [
            n for n in snapshot_names(FIXTURES) if not wanted or n in wanted
        ]
        moving = (
            set(args.expect_moving.split(",")) if args.expect_moving else set()
        )
        if moving - set(ORACLES):
            sys.exit(
                f"unknown oracle(s) in --expect-moving: {sorted(moving - set(ORACLES))}"
            )
        mode = rebaseline if args.rebaseline else add_oracle
        return 0 if mode(names, oracle_name, args.reason, moving) else 1
    recipes = [r for r in RECIPES if not wanted or r.name in wanted]
    generate(recipes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
