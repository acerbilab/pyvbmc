"""Isolate the CMA-ES local search of `active_sample` on the stored oracle
states and run it under five variants of the three settings under review.

Nothing in the PyVBMC repository is modified: the only intervention is a
temporary monkeypatch of `pyvbmc.vbmc.active_sample.cma.fmin`, in this
process, that records the arguments of the one call a real `active_sample`
makes and then raises to abort before any target evaluation.
"""

import copy
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(r"C:\Users\luigi\Documents\GitHub\pyvbmc")
FIX = REPO / "pyvbmc" / "testing" / "oracles" / "fixtures"
sys.path.insert(0, str(REPO / "dev" / "scripts"))

import cma  # noqa: E402

import pyvbmc.vbmc.active_sample  # noqa: E402,F401

# `pyvbmc.vbmc` re-exports the *function* `active_sample`, which shadows
# the submodule attribute, so reach the module through `sys.modules`.
asmod = sys.modules["pyvbmc.vbmc.active_sample"]
from pyvbmc.stats import get_hpd  # noqa: E402
from pyvbmc.testing.oracles._oracles import DEFAULT_SEED, legacy_seed  # noqa
from pyvbmc.testing.oracles._state import (  # noqa: E402
    build_state,
    load_snapshot,
    snapshot_names,
)

VARIANTS = ("current", "S", "N", "B", "SNB")
# (use_CMA_stds, drop_noise_handler) of the optimizer run each variant reads,
# and which point it takes out of that run.
RUN_OF = {
    "current": ((False, False), "bestever"),
    "S": ((True, False), "bestever"),
    "N": ((False, True), "bestever"),
    "B": ((False, False), "matlab"),
    "SNB": ((True, True), "matlab"),
}


class _Captured(Exception):
    pass


def _target(meta):
    from benchmark_targets import find_config

    return find_config(meta["config"]).make(seed=meta["problem_seed"]).fun


def _acq_name(acq_fun):
    """The acquisition object out of the closure of `active_sample`'s
    `acq_fun` (its free variable ``acq_eval``)."""
    names = acq_fun.__code__.co_freevars
    cells = acq_fun.__closure__
    obj = cells[names.index("acq_eval")].cell_contents
    return type(obj).__name__


def capture(name, seed=DEFAULT_SEED):
    """Run one real `active_sample` step and record the `cma.fmin` call."""
    snap = load_snapshot(FIX / name)
    state = build_state(snap, fun=_target(snap["meta"]))
    vp = copy.deepcopy(state["vp"])
    vp.rng = np.random.default_rng(seed)
    gp = copy.deepcopy(state["gp"])
    optim_state = copy.deepcopy(state["optim_state"])
    fl = copy.deepcopy(state["logger"])
    fl.parameter_transformer = vp.parameter_transformer
    options = state["options"]
    history = {"r_index": np.array([state["meta"].get("r_index", np.inf)])}

    box = {}

    def fake_fmin(objective_function, x0, sigma0, **kw):
        box["acq_fun"] = objective_function
        box["x0"] = np.array(x0, dtype=float)
        box["sigma0"] = float(sigma0)
        box["cma_options"] = dict(kw["options"])
        box["parallel_objective"] = kw.get("parallel_objective")
        box["noise_handler"] = kw.get("noise_handler")
        raise _Captured

    real_fmin = asmod.cma.fmin
    asmod.cma.fmin = fake_fmin
    try:
        with legacy_seed(seed):
            asmod.active_sample(gp, 1, optim_state, fl, history, vp, options)
    except _Captured:
        pass
    finally:
        asmod.cma.fmin = real_fmin
    if "x0" not in box:
        raise RuntimeError(f"{name}: no cmaes search happened")

    # `insigma` is not passed to `cma.fmin`; recompute it exactly as
    # `active_sample.py:510-516` does and check it against the scalar the
    # captured call received.
    if options["search_cmaes_vp_init"]:
        _, Sigma = vp.moments(orig_flag=False, cov_flag=True)
    else:
        X_hpd = get_hpd(gp.X, gp.y, options["hpd_frac"])[0]
        Sigma = np.cov(X_hpd, rowvar=False, bias=True)
    insigma = np.sqrt(np.diag(Sigma))
    assert float(np.max(insigma)) == box["sigma0"], name

    box.update(
        name=name,
        insigma=insigma,
        D=int(gp.D),
        K=int(vp.K),
        n_train=int(gp.X.shape[0]),
        Ns=len(gp.posteriors),
        noisy=bool(fl.noise_flag),
        acq=_acq_name(box["acq_fun"]),
        integer_vars=np.asarray(optim_state.get("integer_vars")),
        state=(gp, vp, optim_state, fl, options),
    )
    box["f_x0"] = float(box["acq_fun"](box["x0"]))
    return box


def run_once(cap, seed, use_stds, drop_noise):
    """One `cma.fmin` search; returns both candidate return points."""
    rng = np.random.default_rng(seed)
    acq = cap["acq_fun"]
    opts = dict(cap["cma_options"])
    opts["randn"] = lambda *shape: rng.standard_normal(shape)
    if use_stds:
        opts["CMA_stds"] = cap["insigma"] / cap["sigma0"]
    nh = (
        None
        if drop_noise
        else asmod._BatchedNoiseHandler(cap["x0"].size, acq, rng)
    )

    pending = []
    last = {}

    def obj(X):
        f = acq(X)
        if not (isinstance(X, np.ndarray) and X.ndim == 1):
            pending.append((np.array(X, dtype=float), np.asarray(f, float)))
        return f

    def cb(es):
        if pending:
            X, f = pending[0]
            del pending[:]
            last["X"], last["f"] = X, f
            last["idx"] = int(es.fit.idx[0])

    t0 = time.perf_counter()
    res = cma.fmin(
        obj,
        cap["x0"],
        cap["sigma0"],
        options=opts,
        parallel_objective=obj,
        noise_handler=nh,
        callback=cb,
    )
    wall = time.perf_counter() - t0

    x_best = np.asarray(res[0], dtype=float)
    n_evals = int(res[3])
    n_iters = int(res[4])
    x_mean = np.asarray(res[5], dtype=float)

    # MATLAB's rule (`cmaes_modded.m:1708-1721`): the best point of the last
    # generation, replaced by the (bounds-repaired) final mean if that is
    # better.
    Xg, fg, ig = last["X"], last["f"], last["idx"]
    aligned = bool(np.argmin(fg) == ig or fg[ig] == np.min(fg))
    x_lastgen = Xg[ig]
    f_lastgen = float(acq(x_lastgen))
    f_mean = float(acq(x_mean))
    x_matlab = x_mean if f_mean < f_lastgen else x_lastgen

    return {
        "bestever": x_best,
        "matlab": np.asarray(x_matlab, dtype=float),
        "n_evals": n_evals,
        "n_iters": n_iters,
        "wall": wall,
        "aligned": aligned,
        "mean_used": bool(f_mean < f_lastgen),
        "stop": ";".join(sorted(res[7].keys())),
    }
