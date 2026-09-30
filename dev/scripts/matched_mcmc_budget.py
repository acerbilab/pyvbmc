"""Evaluations that a black-box MCMC sampler needs to match PyVBMC.

For one eligible benchmark target, the number of target evaluations at
which the median posterior error of a gradient-free MCMC sampler falls to
the median error of PyVBMC, ``N*``, with a bootstrap interval. The plan,
with the definitions this script implements, is
``dev/plans/matched-mcmc-budget.md``; the target, the truth and the
metrics are those of ``benchmark_targets.py``.

A configuration is either ``--config exporter``, the two-dimensional
banana run that ``export_animation_trace.py --target banana`` makes (the
exporter's start point and plausible box, PyVBMC through that script's own
``run``, the truth of the benchmark's ``banana`` at ``D = 2``), or
``--target NAME --D D`` for an eligible target of ``benchmark_targets.py``
(``check`` lists them), set up as the benchmark sets it up.

Subcommands, from the repository root::

    python -u dev/scripts/matched_mcmc_budget.py check --config exporter
    python -u dev/scripts/matched_mcmc_budget.py pyvbmc --config exporter \\
        --seeds 0-99 --out RUN/pyvbmc --jobs 4
    python -u dev/scripts/matched_mcmc_budget.py pilot --config exporter \\
        --sampler slice --out RUN/pilot --jobs 4
    python -u dev/scripts/matched_mcmc_budget.py mcmc --config exporter \\
        --sampler slice --setting c=1,step_out=0,adaptive=1 \\
        --seeds 0-99 --out RUN/mcmc --jobs 4
    python -u dev/scripts/matched_mcmc_budget.py report RUN

``check`` runs the checks that come before the runs and lists the eligible
targets; it exits with status 1 when one fails. ``pyvbmc`` writes
``seed_NNNN.json`` per seed: ``func_count`` (every evaluation, including
those trimmed at the end of warm-up), ``gskl`` and ``mmtv`` from
``metrics``, ``elbo``, ``elbo_err`` and ``raised``, plus the exporter's
own ``gskl`` (Monte Carlo moments) for the exporter configuration.
``pilot`` runs the plan's grid of settings of one sampler on seeds 1000 to
1009 into ``OUT/SAMPLER/SETTING/`` and, once the reference exists
(``--reference``, by default the ``pyvbmc`` directory beside ``OUT``),
prints each setting's pilot ``N*`` and the ``--setting`` that ``mcmc`` is
to use for each metric. ``mcmc`` runs one setting on the replicates into
``OUT/SAMPLER/SETTING/seed_NNNN.json``: the budgets, and for each burn-in
fraction the gsKL and MMTV at every budget and the number of states they
used, the replicate's evaluation count, and the sampler's own count. The
sampler ``exact`` stands for exact independent draws, ``(1 - f) N`` of
them at budget ``N`` for burn-in fraction ``f``. ``report RUN`` reads
``RUN/pyvbmc``, ``RUN/pilot`` and ``RUN/mcmc`` and writes ``report.md``,
``summary.json`` and ``figures/`` into ``RUN``. Every subcommand that runs
resumes: a seed whose file exists is skipped (for ``mcmc`` and ``pilot``,
one whose file holds at least the ``--n-max`` requested).

Each seed of ``pyvbmc``, ``pilot`` and ``mcmc`` runs in a process of its
own, forked for it (``--jobs`` of them at a time). A sampler's process
first calls ``np.random.seed(seed)`` and ``random.seed(seed)``: zeus draws
from both global streams and emcee copies NumPy's when it is built. Run
with BLAS single-threaded (``OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1``) when ``--jobs`` is above one. Each invocation appends
its command line, commit, versions, platform and wall time to
``invocations.jsonl`` in its output directory.

An error that cannot be computed is ``+inf``, written as ``null`` in the
JSON files: fewer than ``D + 1`` states, a singular covariance (gsKL), a
marginal with a single value (MMTV), and a run that raised. The MMTV of a
set of states is ``sample_metrics``'s, computed with the kernel density
estimate of the exact reference draws made once; ``check`` confirms that
the two agree.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import multiprocessing
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
import traceback
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import benchmark_targets as bt
import numpy as np

REPO = Path(__file__).resolve().parents[2]

# The exporter's setup (export_animation_trace.run); check confirms it.
EXPORTER_X0 = np.array([[-3.0, 5.5]])
EXPORTER_PLB = np.array([[-5.0, -3.5]])
EXPORTER_PUB = np.array([[5.0, 7.5]])
EXPORTER_MAX_FUN_EVALS = 200

SAMPLERS = ("slice", "emcee", "zeus", "rwm")
BLACK_BOX = ("slice", "emcee", "zeus")
# The headline's metric. gsKL compares only the first two moments, so it
# misses what a Gaussian with the right moments misses; it is reported
# beside MMTV but does not enter the headline.
HEADLINE_METRIC = "mmtv"
METRICS = ("gskl", "mmtv")
METRIC_NAMES = {"gskl": "gsKL", "mmtv": "MMTV"}
SAMPLER_NAMES = {
    "slice": "slice sampling",
    "emcee": "emcee",
    "zeus": "zeus",
    "rwm": "random-walk Metropolis",
    "exact": "exact draws",
}

N_MAX = 300_000
N_MIN = 100
N_BUDGETS = 30
BURN_FRACTIONS = (0.5, 0.1, 0.25)  # the first is the rule; the others are kept
SLICE_ADAPT_SWEEPS = 20
PILOT_SEEDS = tuple(range(1000, 1010))
PILOT_GRID = {
    "slice": [
        {"c": c, "step_out": s, "adaptive": a}
        for c in (0.1, 0.3, 1.0)
        for s in (0, 1)
        for a in (0, 1)
    ],
    "emcee": [{"W": w} for w in (4, 8, 16, 32)],
    "zeus": [{"W": w} for w in (4, 8, 16, 32)],
    "rwm": [{"c": c} for c in (0.05, 0.1, 0.2, 0.4)],
}
N_BOOT = 1000
BOOT_SEED = 20260930
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclasses.dataclass
class Setup:
    """What PyVBMC and the samplers are given, and the truth.

    ``problem`` supplies the log density the samplers receive, the truth
    and the metrics; ``lb``, ``ub``, ``plb`` and ``pub`` are ``(1, D)``.
    """

    label: str
    problem: bt.Problem
    lb: np.ndarray
    ub: np.ndarray
    plb: np.ndarray
    pub: np.ndarray
    exporter: bool

    @property
    def D(self):
        return self.problem.D

    @property
    def paired(self):
        """Whether PyVBMC's seed ``r`` and replicate ``r`` share ``x0``."""
        return not self.exporter

    def x0(self, seed):
        if self.exporter:
            return EXPORTER_X0.copy()
        return bt.make_problem(self.problem.name, self.D, seed=seed).x0


# Targets whose sampler resamples a stored importance-sampling population
# without setting ``sampler_n_eff``.
RESAMPLING = ("logreg",)


def eligible(problem):
    """Noiseless, with an exact sampler and known moments."""
    return (
        problem.noise_sd is None
        and problem.sampler is not None
        and problem.sampler_n_eff is None
        and problem.name not in RESAMPLING
        and problem.true_mean is not None
        and problem.true_cov is not None
    )


def make_setup(args):
    if args.config == "exporter":
        problem = bt.make_problem("banana", 2)
        return Setup(
            label="exporter",
            problem=problem,
            lb=np.full((1, 2), -np.inf),
            ub=np.full((1, 2), np.inf),
            plb=EXPORTER_PLB.copy(),
            pub=EXPORTER_PUB.copy(),
            exporter=True,
        )
    if args.config is not None:
        raise SystemExit(f"unknown --config {args.config!r}")
    if args.target is None or args.D is None:
        raise SystemExit("give --config exporter, or --target and --D")
    problem = bt.make_problem(args.target, args.D)
    if not eligible(problem):
        raise SystemExit(
            f"{args.target} at D = {args.D} is not eligible: it needs to be "
            "noiseless with an exact sampler and known moments"
        )
    return Setup(
        label=f"{args.target}_D{args.D}",
        problem=problem,
        lb=problem.lb.copy(),
        ub=problem.ub.copy(),
        plb=problem.plb.copy(),
        pub=problem.pub.copy(),
        exporter=False,
    )


def budgets(n_max):
    """30 budgets evenly spaced in log N from 100 to ``N_MAX``, continued at
    about the same spacing up to ``n_max`` when it is larger."""
    grid = np.geomspace(N_MIN, N_MAX, N_BUDGETS)
    if n_max > N_MAX:
        step = math.log(N_MAX / N_MIN) / (N_BUDGETS - 1)
        k = max(1, round(math.log(n_max / N_MAX) / step))
        grid = np.concatenate([grid, np.geomspace(N_MAX, n_max, k + 1)[1:]])
    elif n_max < N_MAX:
        grid = grid[grid <= n_max]
    return np.unique(np.round(grid).astype(np.int64))


# --------------------------------------------------------------------------
# Errors of a set of states
# --------------------------------------------------------------------------


class Scorer:
    """gsKL and MMTV of a set of states, as ``sample_metrics`` computes them
    against ``reference``, with the reference's kernel density estimates
    made once."""

    def __init__(self, problem, reference, nkde=2**13, n_grid=100_000):
        from scipy.interpolate import interp1d

        self.problem = problem
        self.reference = np.atleast_2d(np.asarray(reference, dtype=float))
        self.nkde = nkde
        self.n_grid = n_grid
        D = problem.D
        low = np.amin(self.reference, axis=0)
        high = np.amax(self.reference, axis=0)
        span = high - low
        # The reference's range is widened by a tenth and not clipped, as
        # sample_metrics passes it unbounded limits.
        lb2 = np.maximum(low - span / 10, np.full(D, -np.inf))
        ub2 = np.minimum(high + span / 10, np.full(D, np.inf))
        self.ref_curves = []
        for d in range(D):
            yy, mesh = bt._normalized_kde(
                self.reference[:, d], nkde, lb2[d], ub2[d]
            )
            curve = interp1d(
                mesh,
                yy,
                kind="cubic",
                fill_value=np.array([0]),
                bounds_error=False,
            )
            self.ref_curves.append((mesh, curve))

    def gskl(self, samples):
        from pyvbmc.stats import kl_div_mvn

        D = self.problem.D
        if samples.shape[0] < D + 1:
            return math.inf
        mean = np.mean(samples, axis=0, keepdims=True)
        cov = np.cov(samples, rowvar=False).reshape(D, D)
        try:
            kl = kl_div_mvn(
                mean, cov, self.problem.true_mean, self.problem.true_cov
            )
        except np.linalg.LinAlgError:
            return math.inf
        value = float(0.5 * np.sum(kl))
        return value if np.isfinite(value) else math.inf

    def mmtv(self, samples):
        from scipy.interpolate import interp1d

        D = self.problem.D
        if samples.shape[0] < D + 1 or np.any(np.ptp(samples, axis=0) == 0):
            return math.inf
        low = np.amin(samples, axis=0)
        high = np.amax(samples, axis=0)
        span = high - low
        lb1 = np.maximum(low - span / 10, np.ravel(self.problem.lb))
        ub1 = np.minimum(high + span / 10, np.ravel(self.problem.ub))
        tv = np.zeros(D)
        try:
            for d in range(D):
                yy1, mesh1 = bt._normalized_kde(
                    samples[:, d], self.nkde, lb1[d], ub1[d]
                )
                curve1 = interp1d(
                    mesh1,
                    yy1,
                    kind="cubic",
                    fill_value=np.array([0]),
                    bounds_error=False,
                )
                mesh2, curve2 = self.ref_curves[d]
                edges = np.sort([mesh1[0], mesh1[-1], mesh2[0], mesh2[-1]])
                for j in range(3):
                    grid = np.linspace(
                        edges[j], edges[j + 1], num=int(self.n_grid)
                    )
                    difference = np.abs(curve1(grid) - curve2(grid))
                    tv[d] += (
                        0.5 * np.trapezoid(difference) * (grid[1] - grid[0])
                    )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            return math.inf
        value = float(np.mean(tv))
        return value if np.isfinite(value) else math.inf

    def __call__(self, samples):
        samples = np.atleast_2d(np.asarray(samples, dtype=float))
        return {"gskl": self.gskl(samples), "mmtv": self.mmtv(samples)}


def reference_draws(problem):
    """The exact draws that ``metrics`` compares PyVBMC against."""
    return problem.sampler(
        bt.DIAG_TV_SAMPLES, np.random.default_rng(bt.DIAG_SEED + 1)
    )


# --------------------------------------------------------------------------
# The log density that the samplers receive
# --------------------------------------------------------------------------


class Logged:
    """The target's log density at one point (``-inf`` outside the bounds),
    logging every call; ``n`` counts them and ``X[:n]`` holds the points in
    order."""

    def __init__(self, setup):
        self.logp = setup.problem.log_density_vec
        self.lb = np.ravel(setup.lb)
        self.ub = np.ravel(setup.ub)
        self.bounded = bool(
            np.any(np.isfinite(self.lb)) or np.any(np.isfinite(self.ub))
        )
        self.X = np.empty((4096, setup.D))
        self.n = 0

    def __call__(self, x):
        x = np.asarray(x, dtype=float).ravel()
        if self.n == self.X.shape[0]:
            self.X = np.concatenate([self.X, np.empty_like(self.X)])
        self.X[self.n] = x
        self.n += 1
        if self.bounded and (np.any(x < self.lb) or np.any(x > self.ub)):
            return -math.inf
        return float(self.logp(x[None, :])[0])

    @property
    def log(self):
        return self.X[: self.n]


def last_index(log, states):
    """1-based index of the last row of ``log`` equal to each state (bit for
    bit), 0 for a state that is not in it."""
    D = log.shape[1]
    kind = np.dtype((np.void, log.dtype.itemsize * D))
    keys = np.ascontiguousarray(log).view(kind).ravel()
    wanted = np.ascontiguousarray(states, dtype=log.dtype).view(kind).ravel()
    if keys.size == 0:
        return np.zeros(wanted.size, dtype=np.int64)
    unique, first_reversed = np.unique(keys[::-1], return_index=True)
    last = keys.size - first_reversed
    pos = np.minimum(np.searchsorted(unique, wanted), unique.size - 1)
    return np.where(unique[pos] == wanted, last[pos], 0).astype(np.int64)


# --------------------------------------------------------------------------
# Samplers
# --------------------------------------------------------------------------


@dataclasses.dataclass
class Chain:
    """States in order with the evaluation index (1-based) at which each
    was reached, the evaluations made and the sampler's own count."""

    states: np.ndarray
    index: np.ndarray
    n_evals: int
    own_count: int
    own_count_name: str
    extra: dict = dataclasses.field(default_factory=dict)


def walkers(setup, W, seed):
    """Initial walkers of replicate ``seed``, uniform in the plausible box,
    from a generator of their own."""
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    return setup.plb + rng.random((W, setup.D)) * (setup.pub - setup.plb)


def run_slice(setup, setting, seed, n_max, f, full_log=False):
    from gpyreg.slice_sample import SliceSampler

    D = setup.D
    c = float(setting["c"])
    step_out = bool(setting["step_out"])
    adaptive = bool(setting["adaptive"])
    B = SLICE_ADAPT_SWEEPS if adaptive else 0
    # A sweep evaluates at least one point per coordinate, and at least
    # three with stepping out on an unbounded target (both ends and the
    # proposal), so these sweeps reach n_max evaluations in one call.
    per_sweep = D * (3 if step_out and not f.bounded else 1)
    n = max(1, math.ceil((n_max - 1) / per_sweep) - B)
    sampler = SliceSampler(
        f,
        setup.x0(seed).ravel(),
        widths=c * (setup.pub - setup.plb).ravel(),
        LB=setup.lb.ravel(),
        UB=setup.ub.ravel(),
        options={
            "display": "off",
            "diagnostics": False,
            "step_out": step_out,
            "adaptive": adaptive,
        },
        rng=np.random.default_rng(seed),
    )
    states = sampler.sample(n, burn=B)["samples"]
    log = f.log if full_log else f.log[:n_max]
    index = last_index(log, states)
    found = index > 0
    n_found = int(found.sum())
    if not full_log:
        # The states past the budget are not in the truncated log; those
        # inside it must come first.
        if np.any(found[n_found:]) or not np.all(found[:n_found]):
            raise RuntimeError("the states inside the budget are not a prefix")
        states, index = states[:n_found], index[:n_found]
    return Chain(
        states=states,
        index=index,
        n_evals=f.n,
        own_count=int(sampler.func_count),
        own_count_name="func_count",
        extra={"sweeps": n + B, "found": n_found, "returned": len(found)},
    )


def run_emcee(setup, setting, seed, n_max, f):
    import emcee

    W, D = int(setting["W"]), setup.D
    p0 = walkers(setup, W, seed)
    sampler = emcee.EnsembleSampler(W, D, f)  # the stretch move
    steps = max(1, math.ceil(n_max / W) - 1)
    states = np.empty((steps, W, D))
    for t, state in enumerate(
        sampler.sample(p0, iterations=steps, store=False)
    ):
        states[t] = state.coords
    # W initial evaluations, then W per step.
    step_index = W * (np.arange(steps) + 2)
    return Chain(
        states=states.reshape(-1, D),
        index=np.repeat(step_index, W),
        n_evals=f.n,
        own_count=W * (steps + 1),
        own_count_name="W * (steps + 1)",
        extra={"steps": steps},
    )


def run_zeus(setup, setting, seed, n_max, f):
    import zeus

    W, D = int(setting["W"]), setup.D
    p0 = walkers(setup, W, seed)
    sampler = zeus.EnsembleSampler(W, D, f, verbose=False)
    # A step evaluates at least W points, so this many steps reach n_max.
    max_steps = math.ceil(n_max / W)
    states, step_index = [], []
    for X, _, _ in sampler.sample(p0, iterations=max_steps, progress=False):
        states.append(np.array(X, dtype=float))
        step_index.append(f.n)
        if f.n >= n_max:
            break
    states = np.array(states)
    # zeus's ncall leaves out the W evaluations of the initial walkers.
    return Chain(
        states=states.reshape(-1, D),
        index=np.repeat(np.array(step_index, dtype=np.int64), W),
        n_evals=f.n,
        own_count=int(sampler.ncall) + W,
        own_count_name="ncall + W",
        extra={"steps": len(step_index), "ncall": int(sampler.ncall)},
    )


def run_rwm(setup, setting, seed, n_max, f):
    D = setup.D
    rng = np.random.default_rng(seed)
    sd = float(setting["c"]) * np.ravel(setup.pub - setup.plb)
    x = setup.x0(seed).ravel().astype(float)
    lp = f(x)
    steps = max(1, n_max - 1)
    states = np.empty((steps, D))
    accepted = 0
    for t in range(steps):
        y = x + sd * rng.standard_normal(D)
        lq = f(y)
        if math.log(rng.random()) < lq - lp:
            x, lp = y, lq
            accepted += 1
        states[t] = x
    return Chain(
        states=states,
        index=np.arange(steps, dtype=np.int64) + 2,
        n_evals=f.n,
        own_count=steps + 1,
        own_count_name="steps + 1",
        extra={"acceptance": accepted / steps},
    )


RUNNERS = {
    "slice": run_slice,
    "emcee": run_emcee,
    "zeus": run_zeus,
    "rwm": run_rwm,
}


def parse_setting(sampler, text):
    if sampler == "exact":
        return {}
    if not text:
        raise SystemExit(f"--setting is needed for {sampler}")
    setting = {}
    for part in text.split(","):
        key, _, value = part.partition("=")
        key = key.strip()
        if key in ("step_out", "adaptive", "W"):
            setting[key] = int(value)
        elif key == "c":
            setting[key] = float(value)
        else:
            raise SystemExit(f"unknown setting {key!r} for {sampler}")
    expected = set(PILOT_GRID[sampler][0])
    if set(setting) != expected:
        raise SystemExit(f"{sampler} takes the settings {sorted(expected)}")
    return setting


def setting_text(sampler, setting):
    """The ``--setting`` argument that gives ``setting``."""
    if sampler == "exact":
        return ""
    return ",".join(f"{k}={setting[k]:g}" for k in PILOT_GRID[sampler][0])


def setting_label(sampler, setting):
    """The directory name of a setting."""
    if sampler == "exact":
        return "draws"
    return "_".join(f"{k}{setting[k]:g}" for k in PILOT_GRID[sampler][0])


# --------------------------------------------------------------------------
# One replicate
# --------------------------------------------------------------------------


def curve_errors(scorer, budget_grid, select):
    """Errors at every budget for each burn-in fraction; ``select(N, f)``
    returns the states of budget ``N`` with burn-in fraction ``f``."""
    out = {}
    for frac in BURN_FRACTIONS:
        entry = {"gskl": [], "mmtv": [], "n_states": []}
        for N in budget_grid:
            states = select(int(N), frac)
            errors = scorer(states)
            entry["gskl"].append(errors["gskl"])
            entry["mmtv"].append(errors["mmtv"])
            entry["n_states"].append(int(states.shape[0]))
        out[f"{frac:g}"] = entry
    return out


def replicate(setup, sampler, setting, seed, n_max, scorer):
    """Run one replicate and return its record."""
    t0 = time.time()
    grid = budgets(n_max)
    record = {
        "config": setup.label,
        "sampler": sampler,
        "setting": setting,
        "seed": seed,
        "n_max": int(n_max),
        "budgets": grid.tolist(),
        "burn_fractions": list(BURN_FRACTIONS),
    }
    if sampler == "exact":
        rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[1])
        draws = setup.problem.sampler(int(n_max), rng)

        def select(N, frac):
            return draws[: int(math.floor((1 - frac) * N))]

        record["n_evals"] = int(n_max)
        record["errors"] = curve_errors(scorer, grid, select)
        record["raised"] = False
        record["wall_s"] = round(time.time() - t0, 2)
        return record

    f = Logged(setup)
    try:
        chain = RUNNERS[sampler](setup, setting, seed, n_max, f)
    except Exception as exc:  # a run that raised counts with error +inf
        record["raised"] = True
        record["error"] = "".join(
            traceback.format_exception_only(type(exc), exc)
        ).strip()
        record["n_evals"] = int(f.n)
        inf = [math.inf] * len(grid)
        record["errors"] = {
            f"{frac:g}": {
                "gskl": inf,
                "mmtv": inf,
                "n_states": [0] * len(grid),
            }
            for frac in BURN_FRACTIONS
        }
        record["wall_s"] = round(time.time() - t0, 2)
        return record
    t_run = time.time() - t0
    index = chain.index

    def select(N, frac):
        keep = (index <= N) & (index > frac * N)
        return chain.states[keep]

    record["raised"] = False
    record["n_evals"] = int(chain.n_evals)
    record["own_count"] = {chain.own_count_name: int(chain.own_count)}
    record["extra"] = chain.extra
    record["errors"] = curve_errors(scorer, grid, select)
    record["run_s"] = round(t_run, 2)
    record["wall_s"] = round(time.time() - t0, 2)
    return record


# --------------------------------------------------------------------------
# PyVBMC
# --------------------------------------------------------------------------


def exporter_module():
    import export_animation_trace

    # The module sets the lobe's weight to 0.22 when imported; the banana
    # run of the video has none.
    export_animation_trace.LOBE_W = 0.0
    return export_animation_trace


def pyvbmc_run(setup, seed):
    """One PyVBMC run and its record."""
    t0 = time.time()
    record = {"config": setup.label, "seed": seed}
    vbmc = None
    try:
        if setup.exporter:
            eat = exporter_module()
            vbmc = eat.run(seed, EXPORTER_MAX_FUN_EVALS)
        else:
            from pyvbmc import VBMC

            problem = bt.make_problem(setup.problem.name, setup.D, seed=seed)
            args, options = problem.vbmc_args()
            options.setdefault("display", "off")
            vbmc = VBMC(*args, options=options, seed=seed)
        vp, results = vbmc.optimize()
        if setup.exporter:
            # Evaluated right after the run, as the exporter's sweep does.
            record["exporter_gskl"] = eat.gskl(vp)
        m = bt.metrics(setup.problem, vp, results["elbo"])
        record.update(
            raised=False,
            func_count=int(results["func_count"]),
            iterations=int(results["iterations"]),
            gskl=m["gskl"],
            mmtv=m["mmtv"],
            elbo=float(results["elbo"]),
            elbo_sd=float(results["elbo_sd"]),
            elbo_err=m["elbo_err"],
            moment_method=m["moment_method"],
            success_flag=bool(results["success_flag"]),
            message=str(results["message"]),
        )
    except Exception as exc:
        count = None
        if vbmc is not None and getattr(vbmc, "function_logger", None):
            count = int(vbmc.function_logger.func_count)
        record.update(
            raised=True,
            error="".join(
                traceback.format_exception_only(type(exc), exc)
            ).strip(),
            func_count=count,
            gskl=math.inf,
            mmtv=math.inf,
            elbo=None,
            elbo_err=math.inf,
        )
    record["wall_s"] = round(time.time() - t0, 2)
    return record


# --------------------------------------------------------------------------
# Files and processes
# --------------------------------------------------------------------------


def _encode(value):
    """JSON-ready copy: +inf as null, floats to 7 significant digits."""
    if isinstance(value, dict):
        return {k: _encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isinf(value) and value > 0:
            return None
        if not math.isfinite(value):
            raise ValueError(f"cannot write {value}")
        return float(f"{value:.7g}")
    if isinstance(value, np.ndarray):
        return _encode(value.tolist())
    return value


def _decode_errors(values):
    return np.array(
        [math.inf if v is None else v for v in values], dtype=float
    )


def write_json(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(_encode(record), separators=(",", ":")) + "\n")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def seed_path(directory, seed):
    return Path(directory) / f"seed_{seed:04d}.json"


def parse_seeds(text):
    seeds = []
    for part in text.split(","):
        if "-" in part:
            first, last = (int(v) for v in part.split("-"))
            seeds.extend(range(first, last + 1))
        else:
            seeds.append(int(part))
    return seeds


def done(path, n_max=None):
    if not path.exists():
        return False
    if n_max is None:
        return True
    return read_json(path).get("n_max", 0) >= n_max


def _package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def git_describe():
    try:
        out = subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def provenance():
    import pyvbmc

    return {
        "commit": git_describe(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpus": os.cpu_count(),
        "versions": {
            p: _package_version(p)
            for p in (
                "pyvbmc",
                "gpyreg",
                "numpy",
                "scipy",
                "emcee",
                "zeus-mcmc",
            )
        },
        "pyvbmc_file": pyvbmc.__file__,
        "threads": {v: os.environ.get(v) for v in THREAD_VARS},
    }


def log_invocation(directory, started, prov, seeds_run, extra=None):
    """Append an invocation to ``invocations.jsonl``; ``prov`` is the
    provenance taken when it started."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    entry = {
        "argv": sys.argv[1:],
        "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
        "wall_s": round(time.time() - started, 1),
        "seeds_run": seeds_run,
        **prov,
    }
    if extra:
        entry.update(extra)
    with open(directory / "invocations.jsonl", "a") as fh:
        fh.write(json.dumps(entry) + "\n")


_WORKER = {}


def _init_worker(setup, scorer):
    _WORKER["setup"] = setup
    _WORKER["scorer"] = scorer


def _mcmc_task(task):
    sampler, setting, seed, n_max, path = task
    np.random.seed(seed)
    random.seed(seed)
    record = replicate(
        _WORKER["setup"], sampler, setting, seed, n_max, _WORKER["scorer"]
    )
    write_json(Path(path), record)
    return path, record["wall_s"], record["raised"]


def _pyvbmc_task(task):
    seed, path = task
    record = pyvbmc_run(_WORKER["setup"], seed)
    write_json(Path(path), record)
    return path, record["wall_s"], record["raised"]


def run_tasks(function, tasks, jobs, setup, scorer=None):
    """Run each task in a process of its own, ``jobs`` at a time."""
    if not tasks:
        return
    if jobs > 1 and any(os.environ.get(v) != "1" for v in THREAD_VARS):
        print(
            "warning: BLAS is not single-threaded "
            f"({', '.join(f'{v}={os.environ.get(v)}' for v in THREAD_VARS)})",
            flush=True,
        )
    ctx = multiprocessing.get_context("fork")
    t0 = time.time()
    with ctx.Pool(
        processes=jobs,
        initializer=_init_worker,
        initargs=(setup, scorer),
        maxtasksperchild=1,
    ) as pool:
        for k, (path, wall, raised) in enumerate(
            pool.imap_unordered(function, tasks), start=1
        ):
            flag = "  RAISED" if raised else ""
            print(
                f"[{k}/{len(tasks)} {time.time() - t0:7.0f} s] {path} "
                f"({wall:.0f} s){flag}",
                flush=True,
            )


# --------------------------------------------------------------------------
# Matched budget
# --------------------------------------------------------------------------


def matched_budget(grid, median, reference):
    """``N*``: the smallest budget at which ``median`` falls to
    ``reference``, interpolated linearly in log N and log error between the
    grid points around the crossing. Returns ``(value, kind)`` with kind
    ``"below"`` (already at or below at the first budget; the value is that
    budget), ``"cross"`` or ``"none"`` (value ``inf``)."""
    grid = np.asarray(grid, dtype=float)
    median = np.asarray(median, dtype=float)
    at_or_below = median <= reference
    if not np.any(at_or_below):
        return math.inf, "none"
    k = int(np.argmax(at_or_below))
    if k == 0:
        return float(grid[0]), "below"
    m0, m1 = median[k - 1], median[k]
    if not np.isfinite(m0) or m1 <= 0 or reference <= 0 or m0 == m1:
        return float(grid[k]), "cross"
    t = (math.log(reference) - math.log(m0)) / (math.log(m1) - math.log(m0))
    log_n = math.log(grid[k - 1]) + t * (
        math.log(grid[k]) - math.log(grid[k - 1])
    )
    return float(math.exp(log_n)), "cross"


def load_pyvbmc(directory):
    files = sorted(Path(directory).glob("seed_*.json"))
    if not files:
        raise SystemExit(f"no PyVBMC runs in {directory}")
    records = [read_json(p) for p in files]
    seeds = np.array([r["seed"] for r in records])
    out = {"seeds": seeds, "records": records}
    for key in ("gskl", "mmtv", "elbo_err"):
        out[key] = _decode_errors([r.get(key) for r in records])
    out["func_count"] = np.array(
        [
            np.nan if r.get("func_count") is None else r["func_count"]
            for r in records
        ],
        dtype=float,
    )
    out["raised"] = np.array([bool(r["raised"]) for r in records])
    return out


def load_curves(directory, frac=0.5):
    """Replicates of one setting: seeds, budgets and errors ``(n, budgets)``
    per metric."""
    files = sorted(Path(directory).glob("seed_*.json"))
    if not files:
        return None
    records = [read_json(p) for p in files]
    n_max = {r["n_max"] for r in records}
    if len(n_max) != 1:
        raise SystemExit(
            f"{directory}: replicates of different lengths {n_max}"
        )
    grid = np.array(records[0]["budgets"], dtype=float)
    key = f"{frac:g}"
    out = {
        "seeds": np.array([r["seed"] for r in records]),
        "budgets": grid,
        "n_max": n_max.pop(),
        "records": records,
        "raised": int(sum(bool(r["raised"]) for r in records)),
    }
    for metric in METRICS:
        out[metric] = np.array(
            [_decode_errors(r["errors"][key][metric]) for r in records]
        )
    return out


def median_curve(errors):
    return np.median(errors, axis=0)


def pilot_table(pilot_dir, sampler, ref):
    """Pilot ``N*`` of every setting of ``sampler`` against the reference
    medians ``ref`` (metric -> value), and the chosen setting per metric."""
    rows = []
    for setting in PILOT_GRID[sampler]:
        directory = Path(pilot_dir) / sampler / setting_label(sampler, setting)
        curves = load_curves(directory)
        row = {"setting": setting, "n": 0}
        if curves is not None:
            row["n"] = len(curves["seeds"])
            row["n_max"] = curves["n_max"]
            for metric in METRICS:
                med = median_curve(curves[metric])
                value, kind = matched_budget(
                    curves["budgets"], med, ref[metric]
                )
                row[metric] = {
                    "n_star": value,
                    "kind": kind,
                    "median_at_n_max": float(med[-1]),
                    # Tie-break: the mean log median error over the budgets.
                    "mean_log_median": float(
                        np.mean(np.log(np.maximum(med, 1e-300)))
                    ),
                }
        rows.append(row)
    # A choice needs every setting run on the same seeds.
    counts = {r["n"] for r in rows}
    choice = {}
    if len(counts) == 1 and counts.pop() > 0:
        complete = rows
        for metric in METRICS:
            best = min(
                complete,
                key=lambda r: (
                    r[metric]["n_star"],
                    r[metric]["mean_log_median"],
                ),
            )
            choice[metric] = best["setting"]
    return rows, choice


def fmt_n(value, kind=None, n_max=None):
    if kind == "below":
        return f"< {N_MIN}"
    if value is None or not math.isfinite(value):
        return f"> {n_max:,}" if n_max else "none"
    return f"{value:,.0f}"


def fmt_interval(lo, hi, n_max):
    """A bootstrap interval; a bound at the first budget reads "< 100"."""

    def one(value):
        return fmt_n(value, "below" if value <= N_MIN else None, n_max)

    return f"{one(lo)} to {one(hi)}"


def join_names(names):
    names = list(names)
    return (
        ", ".join(names[:-1]) + " and " + names[-1]
        if len(names) > 1
        else "".join(names)
    )


def fmt_err(value):
    if value is None or not math.isfinite(value):
        return "inf"
    return f"{value:.3g}"


# --------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------


def cmd_check(args):
    setup = make_setup(args)
    failures = []

    def report(ok, text):
        print(f"{'ok  ' if ok else 'FAIL'} {text}", flush=True)
        if not ok:
            failures.append(text)

    import pyvbmc

    print(
        f"pyvbmc from {pyvbmc.__file__}, {json.dumps(provenance()['versions'])}"
    )
    print(
        f"commit {git_describe()}, threads {provenance()['threads']}",
        flush=True,
    )

    # Eligible targets.
    print("\neligible targets (noiseless, exact sampler, known moments):")
    for name in bt.TARGET_NAMES:
        dims = []
        for D in (2, 3, 5, 6, 10):
            try:
                problem = bt.make_problem(name, D)
            except (ValueError, FileNotFoundError, KeyError):
                continue
            if eligible(problem):
                dims.append(D)
        if dims:
            print(f"  {name:12s} D in {dims}")
    print("", flush=True)

    D = setup.D
    ref = reference_draws(setup.problem)
    scorer = Scorer(setup.problem, ref)

    if setup.exporter:
        eat = exporter_module()
        # The exporter's density with no lobe against the benchmark banana.
        rng = np.random.default_rng(1)
        X = EXPORTER_PLB + rng.random((10_000, 2)) * (
            EXPORTER_PUB - EXPORTER_PLB
        )
        a = eat.log_density_vec(X)
        b = setup.problem.log_density_vec(X)
        rel = float(np.max(np.abs(a - b) / np.abs(b)))
        report(
            rel <= 1e-12,
            f"exporter and benchmark banana agree: max rel {rel:.2e}",
        )
        # The exporter's setup is what the samplers receive.
        vbmc = eat.run(0, EXPORTER_MAX_FUN_EVALS)
        same = (
            np.array_equal(vbmc.x0_orig, EXPORTER_X0)
            and np.array_equal(vbmc.plausible_lower_bounds, EXPORTER_PLB)
            and np.array_equal(vbmc.plausible_upper_bounds, EXPORTER_PUB)
            and np.all(np.isinf(vbmc.lower_bounds))
            and np.all(np.isinf(vbmc.upper_bounds))
            and vbmc.options["max_fun_evals"] == EXPORTER_MAX_FUN_EVALS
            and vbmc.options["display"] == "off"
        )
        report(
            same,
            "exporter's x0, plausible box, bounds and options as used here",
        )

    # The fast MMTV and gsKL against sample_metrics.
    worst = 0.0
    for n in (50, 2_000, 150_000):
        s = setup.problem.sampler(n, np.random.default_rng(n))
        mine = scorer(s)
        theirs = bt.sample_metrics(setup.problem, s, {}, reference=ref)
        for metric in METRICS:
            worst = max(
                worst, abs(mine[metric] - theirs[metric]) / theirs[metric]
            )
    report(
        worst <= 1e-12, f"scorer equals sample_metrics: max rel {worst:.1e}"
    )

    # Counts, prefix property and the slice sampler's states in its log.
    n_short = 3000
    for sampler in SAMPLERS:
        for setting in PILOT_GRID[sampler]:
            label = setting_label(sampler, setting)
            np.random.seed(7)
            random.seed(7)
            f = Logged(setup)
            if sampler == "slice":
                chain = run_slice(setup, setting, 7, n_short, f, full_log=True)
                all_found = bool(np.all(chain.index > 0))
                increasing = bool(np.all(np.diff(chain.index) > 0))
                report(
                    all_found and increasing,
                    f"slice {label}: all {len(chain.index)} states in the log, "
                    "indices increasing",
                )
            else:
                chain = RUNNERS[sampler](setup, setting, 7, n_short, f)
            report(
                chain.n_evals == chain.own_count,
                f"{sampler} {label}: wrapper count {chain.n_evals} == "
                f"{chain.own_count_name} {chain.own_count}",
            )
            # A run of half the length is a prefix of the longer one.
            np.random.seed(7)
            random.seed(7)
            g = Logged(setup)
            short = RUNNERS[sampler](setup, setting, 7, n_short // 2, g)
            keep = chain.index <= n_short // 2
            keep_short = short.index <= n_short // 2
            prefix = (
                np.array_equal(chain.states[keep], short.states[keep_short])
                and np.array_equal(chain.index[keep], short.index[keep_short])
                and np.array_equal(f.log[: g.n], g.log)
            )
            report(prefix, f"{sampler} {label}: a shorter run is a prefix")

    # Exact draws: the errors fall with the number of draws.
    print("\nexact draws (median over 5 draw sets):")
    counts = (100, 1_000, 10_000, 100_000)
    medians = {m: [] for m in METRICS}
    for n in counts:
        values = {m: [] for m in METRICS}
        for k in range(5):
            s = setup.problem.sampler(n, np.random.default_rng([n, k]))
            e = scorer(s)
            for m in METRICS:
                values[m].append(e[m])
        for m in METRICS:
            medians[m].append(float(np.median(values[m])))
        print(
            f"  {n:7d} draws: gsKL {medians['gskl'][-1]:.3g}, "
            f"MMTV {medians['mmtv'][-1]:.3g}",
            flush=True,
        )
    for m in METRICS:
        report(
            bool(np.all(np.diff(medians[m]) < 0)),
            f"{METRIC_NAMES[m]} of exact draws falls with their number; "
            f"{medians[m][-1]:.3g} at 10^5 draws (the floor)",
        )

    if setup.exporter:
        # The tool's PyVBMC runs against the exporter's own sweep, both on
        # the executing machine.
        print("\nPyVBMC seeds 0 to 4, the tool against the exporter's sweep:")
        sweep = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(REPO / "dev" / "scripts" / "export_animation_trace.py"),
                "--target",
                "banana",
                "--sweep",
                "0:5",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=REPO,
        )
        with tempfile.TemporaryDirectory() as tmp:
            tasks = [(s, str(seed_path(tmp, s))) for s in range(5)]
            run_tasks(_pyvbmc_task, tasks, max(1, args.jobs - 1), setup)
            mine = {s: read_json(seed_path(tmp, s)) for s in range(5)}
        output, _ = sweep.communicate()
        print(output, end="", flush=True)
        rows = {}
        for line in output.splitlines():
            parts = line.split()
            if len(parts) == 8 and parts[0].isdigit():
                rows[int(parts[0])] = parts
        for s in range(5):
            r = mine[s]
            if s not in rows or r["raised"]:
                report(False, f"seed {s}: no sweep line or the run raised")
                continue
            evals, gskl_text = int(rows[s][2]), rows[s][5]
            ours = f"{r['exporter_gskl']:7.4f}".strip()
            report(
                r["func_count"] == evals and ours == gskl_text,
                f"seed {s}: func_count {r['func_count']} vs {evals}, "
                f"exporter gsKL {ours} vs {gskl_text} "
                f"(metrics gsKL {r['gskl']:.4f}, MMTV {r['mmtv']:.4f})",
            )

    print(
        f"\n{'all checks passed' if not failures else f'{len(failures)} checks failed'}",
        flush=True,
    )
    return 1 if failures else 0


def cmd_pyvbmc(args):
    setup = make_setup(args)
    started = time.time()
    prov = provenance()
    seeds = parse_seeds(args.seeds)
    tasks = [
        (s, str(seed_path(args.out, s)))
        for s in seeds
        if not done(seed_path(args.out, s))
    ]
    print(
        f"PyVBMC on {setup.label}: {len(tasks)} of {len(seeds)} seeds to run",
        flush=True,
    )
    run_tasks(_pyvbmc_task, tasks, args.jobs, setup)
    log_invocation(args.out, started, prov, [t[0] for t in tasks])
    return 0


def _mcmc_tasks(out, sampler, settings, seeds, n_max):
    tasks = []
    for setting in settings:
        directory = Path(out) / sampler / setting_label(sampler, setting)
        for s in seeds:
            path = seed_path(directory, s)
            if not done(path, n_max):
                tasks.append((sampler, setting, s, n_max, str(path)))
    return tasks


def cmd_mcmc(args):
    setup = make_setup(args)
    started = time.time()
    prov = provenance()
    setting = parse_setting(args.sampler, args.setting)
    seeds = parse_seeds(args.seeds)
    tasks = _mcmc_tasks(args.out, args.sampler, [setting], seeds, args.n_max)
    print(
        f"{args.sampler} {setting_label(args.sampler, setting)} on "
        f"{setup.label}: {len(tasks)} of {len(seeds)} replicates to run, "
        f"N_max {args.n_max:,}",
        flush=True,
    )
    scorer = Scorer(setup.problem, reference_draws(setup.problem))
    run_tasks(_mcmc_task, tasks, args.jobs, setup, scorer)
    log_invocation(
        Path(args.out) / args.sampler,
        started,
        prov,
        [t[2] for t in tasks],
        {"setting": setting, "n_max": args.n_max},
    )
    return 0


def cmd_pilot(args):
    setup = make_setup(args)
    started = time.time()
    prov = provenance()
    settings = PILOT_GRID[args.sampler]
    seeds = parse_seeds(args.seeds)
    tasks = _mcmc_tasks(args.out, args.sampler, settings, seeds, args.n_max)
    print(
        f"pilot of {args.sampler} on {setup.label}: {len(tasks)} of "
        f"{len(settings) * len(seeds)} runs to do, N_max {args.n_max:,}",
        flush=True,
    )
    scorer = Scorer(setup.problem, reference_draws(setup.problem))
    run_tasks(_mcmc_task, tasks, args.jobs, setup, scorer)
    log_invocation(
        Path(args.out) / args.sampler,
        started,
        prov,
        [[t[1], t[2]] for t in tasks],
        {"n_max": args.n_max},
    )
    reference = Path(args.reference or Path(args.out).parent / "pyvbmc")
    if not reference.exists():
        print(f"no reference at {reference}; the choice waits for it")
        return 0
    ref = load_pyvbmc(reference)
    ref_med = {m: float(np.median(ref[m])) for m in METRICS}
    rows, choice = pilot_table(args.out, args.sampler, ref_med)
    print(
        f"\nreference medians: gsKL {ref_med['gskl']:.4g}, "
        f"MMTV {ref_med['mmtv']:.4g} ({len(ref['seeds'])} runs)"
    )
    for row in rows:
        text = setting_text(args.sampler, row["setting"])
        cells = [
            f"{METRIC_NAMES[m]} N* {fmt_n(row[m]['n_star'], row[m]['kind'], row.get('n_max'))}"
            for m in METRICS
            if m in row
        ]
        print(f"  {text:32s} n={row['n']:2d}  " + "  ".join(cells))
    for metric, setting in choice.items():
        print(
            f"chosen for {METRIC_NAMES[metric]}: --setting "
            f"{setting_text(args.sampler, setting)}"
        )
    return 0


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def percentile(values, q, axis=None):
    """``np.nanpercentile``, with ``+inf`` where it interpolates between two
    infinite values (which gives NaN)."""
    values = np.asarray(values, dtype=float)
    p = np.nanpercentile(values, q, axis=axis)
    return np.where(
        np.isnan(p) & np.any(np.isinf(values), axis=axis), np.inf, p
    )


def _percentiles(values):
    return float(percentile(values, 5)), float(percentile(values, 95))


def bootstrap(ref, runs, paired, n_boot=N_BOOT, seed=BOOT_SEED):
    """Bootstrap ``N*`` of every (sampler, metric) in ``runs`` and the
    headline: the minimum of ``HEADLINE_METRIC``'s over the black-box
    samplers.

    ``ref`` holds PyVBMC's errors by metric and its seeds; ``runs`` maps
    ``(sampler, metric)`` to a replicate set (``load_curves``). A resample
    draws PyVBMC's seeds and each sampler's replicates with replacement:
    independently, or with one draw of seeds for all when ``paired``. A
    resample without a crossing counts as ``inf``, and one at or below the
    reference at the first budget as that budget.
    """
    rng = np.random.default_rng(seed)
    n_ref = len(ref["seeds"])
    samplers = sorted({s for s, _ in runs})
    out = {key: np.empty(n_boot) for key in runs}
    headline = np.empty(n_boot)
    for b in range(n_boot):
        ref_idx = rng.integers(0, n_ref, n_ref)
        ref_med = {m: float(np.median(ref[m][ref_idx])) for m in METRICS}
        sampler_idx = {}
        for s in samplers:
            if paired:
                sampler_idx[s] = ref_idx
            else:
                n_rep = len(
                    next(v for (t, _), v in runs.items() if t == s)["seeds"]
                )
                sampler_idx[s] = rng.integers(0, n_rep, n_rep)
        best = math.inf
        for (s, m), curves in runs.items():
            idx = sampler_idx[s]
            if paired:
                # Seed r of PyVBMC with replicate r.
                pos = {seed: k for k, seed in enumerate(curves["seeds"])}
                idx = np.array([pos[ref["seeds"][i]] for i in idx])
            med = median_curve(curves[m][idx])
            value, _ = matched_budget(curves["budgets"], med, ref_med[m])
            out[(s, m)][b] = value
            if s in BLACK_BOX and m == HEADLINE_METRIC:
                best = min(best, value)
        headline[b] = best
    return out, headline


def _figure(path, ref, runs, exact, metric, n_star, frac_key="0.5"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "slice": "#1f77b4",
        "emcee": "#2ca02c",
        "zeus": "#9467bd",
        "rwm": "#8c564b",
        "exact": "#555555",
    }
    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=120)
    for s in list(SAMPLERS) + ["exact"]:
        curves = exact if s == "exact" else runs.get((s, metric))
        if curves is None:
            continue
        E = curves[metric]
        med = median_curve(E)
        lo, hi = percentile(E, 25, axis=0), percentile(E, 75, axis=0)
        x = curves["budgets"]
        style = "--" if s == "exact" else "-"
        label = SAMPLER_NAMES[s]
        if s != "exact":
            label += f" ({setting_text(s, curves['setting'])})"
        ax.plot(x, med, style, color=colors[s], label=label, lw=1.8)
        ax.fill_between(x, lo, hi, color=colors[s], alpha=0.12, lw=0)
        value = n_star.get((s, metric))
        if value is not None and math.isfinite(value):
            ax.plot(
                [value], [np.median(ref[metric])], "o", color=colors[s], ms=6
            )
    ref_med = float(np.median(ref[metric]))
    q1, q3 = percentile(ref[metric], 25), percentile(ref[metric], 75)
    ax.axhline(ref_med, color="#d62728", lw=1.5, label="PyVBMC median")
    ax.axhspan(q1, q3, color="#d62728", alpha=0.08, lw=0)
    ax.axvline(
        float(np.nanmedian(ref["func_count"])),
        color="#d62728",
        ls=":",
        lw=1.2,
        label="PyVBMC median evaluations",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("target evaluations N")
    ax.set_ylabel(f"{METRIC_NAMES[metric]} (median, band: quartiles)")
    ax.set_title(f"{METRIC_NAMES[metric]} against evaluations, burn-in N/2")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def cmd_report(args):
    run = Path(args.run)
    ref = load_pyvbmc(run / "pyvbmc")
    ref_med = {m: float(np.median(ref[m])) for m in METRICS}
    lines = []
    summary = {"reference": {}, "pilot": {}, "runs": {}, "headline": None}

    invocations = []
    for path in sorted(run.rglob("invocations.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                entry["directory"] = str(path.parent.relative_to(run))
                invocations.append(entry)
    config = ref["records"][0]["config"]
    paired = config != "exporter"

    # Reference.
    def quart(values):
        # A run that raised before its logger existed has no evaluation
        # count (NaN); errors that cannot be computed are +inf.
        return [float(percentile(values, q)) for q in (25, 50, 75)]

    lines.append(f"# Matched MCMC budget: {config}\n")
    lines.append("## The reference (PyVBMC)\n")
    lines.append(
        f"{len(ref['seeds'])} runs (seeds {ref['seeds'].min()} to "
        f"{ref['seeds'].max()}), {int(ref['raised'].sum())} raised.\n"
    )
    lines.append("| quantity | 25% | median | 75% |")
    lines.append("|---|---|---|---|")
    for key, name in (
        ("func_count", "evaluations"),
        ("gskl", "gsKL"),
        ("mmtv", "MMTV"),
        ("elbo_err", "evidence error `elbo_err`"),
    ):
        q = quart(ref[key])
        summary["reference"][key] = q
        fmt = (lambda v: f"{v:.0f}") if key == "func_count" else fmt_err
        lines.append(f"| {name} | {fmt(q[0])} | {fmt(q[1])} | {fmt(q[2])} |")
    lines.append("")
    summary["reference"]["n"] = int(len(ref["seeds"]))
    summary["reference"]["raised"] = int(ref["raised"].sum())

    # Pilot.
    lines.append("## The pilot\n")
    lines.append(
        f"Seeds {PILOT_SEEDS[0]} to {PILOT_SEEDS[-1]}; pilot N* against the "
        "reference medians, burn-in N/2. The setting chosen for a metric is "
        "the one with the smallest pilot N* (ties broken by the mean log "
        "median error over the budgets).\n"
    )
    choices = {}
    for sampler in SAMPLERS:
        if not (run / "pilot" / sampler).exists():
            continue
        rows, choice = pilot_table(run / "pilot", sampler, ref_med)
        choices[sampler] = choice
        summary["pilot"][sampler] = {
            "rows": rows,
            "choice": {m: setting_text(sampler, s) for m, s in choice.items()},
        }
        lines.append(f"**{SAMPLER_NAMES[sampler]}**\n")
        lines.append(
            "| setting | runs | gsKL N* | gsKL median at N_max | MMTV N* | "
            "MMTV median at N_max | chosen for |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for row in rows:
            if row["n"] == 0:
                continue
            chosen = [
                METRIC_NAMES[m]
                for m, s in choice.items()
                if s == row["setting"]
            ]
            cells = []
            for m in METRICS:
                cells.append(
                    fmt_n(row[m]["n_star"], row[m]["kind"], row["n_max"])
                )
                cells.append(fmt_err(row[m]["median_at_n_max"]))
            lines.append(
                f"| `{setting_text(sampler, row['setting'])}` | {row['n']} | "
                + " | ".join(cells)
                + f" | {', '.join(chosen)} |"
            )
        lines.append("")

    # Runs of the chosen settings.
    runs = {}
    runs_by_frac = {}
    for sampler, choice in choices.items():
        for metric, setting in choice.items():
            directory = (
                run / "mcmc" / sampler / setting_label(sampler, setting)
            )
            curves = load_curves(directory)
            if curves is None:
                print(f"missing runs: {directory}")
                continue
            curves["setting"] = setting
            runs[(sampler, metric)] = curves
            for frac in BURN_FRACTIONS:
                c = load_curves(directory, frac)
                c["setting"] = setting
                runs_by_frac[(sampler, metric, frac)] = c
    exact = load_curves(run / "mcmc" / "exact" / "draws")
    if exact is not None:
        exact["setting"] = {}

    n_star = {}
    lines.append("## The matched budget\n")
    lines.append(
        f"N* is the smallest budget at which the median error over the "
        f"replicates falls to PyVBMC's median error (gsKL "
        f"{fmt_err(ref_med['gskl'])}, MMTV {fmt_err(ref_med['mmtv'])}), with "
        f"burn-in N/2; interval: 5th to 95th percentile over {N_BOOT} "
        f"bootstrap resamples ({'paired' if paired else 'unpaired'}).\n"
    )
    boot, headline_boot = (
        bootstrap(ref, runs, paired) if runs else ({}, np.array([]))
    )
    lines.append(
        "| sampler | setting | metric | replicates | raised | N* | 90% interval | "
        "median error at largest budget | largest budget |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for (sampler, metric), curves in sorted(
        runs.items(), key=lambda kv: (SAMPLERS.index(kv[0][0]), kv[0][1])
    ):
        med = median_curve(curves[metric])
        value, kind = matched_budget(curves["budgets"], med, ref_med[metric])
        n_star[(sampler, metric)] = value if kind != "below" else float(N_MIN)
        lo, hi = _percentiles(boot[(sampler, metric)])
        n_max = int(curves["budgets"][-1])
        interval = fmt_interval(lo, hi, n_max)
        summary["runs"][f"{sampler}/{metric}"] = {
            "setting": setting_text(sampler, curves["setting"]),
            "replicates": int(len(curves["seeds"])),
            "raised": curves["raised"],
            "n_star": value,
            "kind": kind,
            "interval": [lo, hi],
            "median_at_largest_budget": float(med[-1]),
            "largest_budget": n_max,
            "median_curve": med.tolist(),
            "budgets": curves["budgets"].tolist(),
        }
        lines.append(
            f"| {SAMPLER_NAMES[sampler]} | `{setting_text(sampler, curves['setting'])}` "
            f"| {METRIC_NAMES[metric]} | {len(curves['seeds'])} | {curves['raised']} | "
            f"{fmt_n(value, kind, n_max)} | {interval} | {fmt_err(med[-1])} | "
            f"{n_max:,} |"
        )
    lines.append("")

    # Headline.
    candidates = [
        (n_star[key], key)
        for key in n_star
        if key[0] in BLACK_BOX and key[1] == HEADLINE_METRIC
    ]
    if candidates:
        value, (sampler, metric) = min(candidates)
        curves = runs[(sampler, metric)]
        lo, hi = _percentiles(headline_boot)
        kind = matched_budget(
            curves["budgets"], median_curve(curves[metric]), ref_med[metric]
        )[1]
        n_max = int(curves["budgets"][-1])
        used = [s for s in BLACK_BOX if (s, metric) in n_star]
        text = (
            f"**Headline:** {SAMPLER_NAMES[sampler]} "
            f"(`{setting_text(sampler, curves['setting'])}`), {METRIC_NAMES[metric]}: "
            f"median N* {'' if kind == 'below' or not math.isfinite(value) else '= '}"
            f"{fmt_n(value, kind, n_max)} evaluations, 90% "
            f"interval {fmt_interval(lo, hi, n_max)} "
            f"({len(curves['seeds'])} replicates against {len(ref['seeds'])} "
            f"PyVBMC runs, whose median budget is "
            f"{np.nanmedian(ref['func_count']):.0f} evaluations). The smallest "
            f"{METRIC_NAMES[metric]} N* over "
            f"{join_names(SAMPLER_NAMES[s] for s in used)}; the interval takes "
            "that minimum inside each resample. MCMC gives no estimate of the "
            "model evidence, which PyVBMC returns from the same evaluations."
        )
        summary["headline"] = {
            "sampler": sampler,
            "setting": setting_text(sampler, curves["setting"]),
            "metric": metric,
            "n_star": value,
            "kind": kind,
            "interval": [lo, hi],
            "replicates": int(len(curves["seeds"])),
            "reference_runs": int(len(ref["seeds"])),
            "reference_median_evaluations": float(
                np.nanmedian(ref["func_count"])
            ),
        }
        lines.insert(1, text + "\n")

    # After the runs.
    lines.append("## After the runs\n")
    for (sampler, metric), curves in sorted(runs.items()):
        med = median_curve(curves[metric])
        ok = med[-1] <= ref_med[metric]
        lines.append(
            f"- {SAMPLER_NAMES[sampler]}, {METRIC_NAMES[metric]}: median "
            f"{fmt_err(med[-1])} at {int(curves['budgets'][-1]):,} "
            f"{'is below' if ok else 'is still above'} the reference "
            f"{fmt_err(ref_med[metric])}"
            + ("" if ok else f"; N* > {int(curves['budgets'][-1]):,}")
            + "."
        )
    lines.append("")

    # Burn-in sensitivity.
    lines.append("## N* with other burn-in fractions\n")
    lines.append(
        "| sampler | metric | "
        + " | ".join(f"burn-in {f:g} N" for f in BURN_FRACTIONS)
        + " |"
    )
    lines.append("|---|---|" + "---|" * len(BURN_FRACTIONS))
    for (sampler, metric), _ in sorted(runs.items()):
        cells = []
        for frac in BURN_FRACTIONS:
            c = runs_by_frac[(sampler, metric, frac)]
            v, k = matched_budget(
                c["budgets"], median_curve(c[metric]), ref_med[metric]
            )
            cells.append(fmt_n(v, k, int(c["budgets"][-1])))
        lines.append(
            f"| {SAMPLER_NAMES[sampler]} | {METRIC_NAMES[metric]} | "
            + " | ".join(cells)
            + " |"
        )
    lines.append("")

    # Curves.
    lines.append("## Median error curves (burn-in N/2)\n")
    grid = None
    columns = []
    for (sampler, metric), curves in sorted(
        runs.items(), key=lambda kv: (kv[0][1], SAMPLERS.index(kv[0][0]))
    ):
        columns.append(
            (
                f"{SAMPLER_NAMES[sampler]} {METRIC_NAMES[metric]}",
                curves,
                metric,
            )
        )
    if exact is not None:
        for metric in METRICS:
            columns.append(
                (f"exact draws {METRIC_NAMES[metric]}", exact, metric)
            )
    if columns:
        grid = max((c[1]["budgets"] for c in columns), key=len)
        lines.append("| N | " + " | ".join(c[0] for c in columns) + " |")
        lines.append("|---|" + "---|" * len(columns))
        for i, N in enumerate(grid):
            cells = []
            for _, curves, metric in columns:
                if i < len(curves["budgets"]) and curves["budgets"][i] == N:
                    cells.append(fmt_err(median_curve(curves[metric])[i]))
                else:
                    cells.append("")
            lines.append(f"| {int(N):,} | " + " | ".join(cells) + " |")
        lines.append("")
    if exact is not None:
        summary["exact"] = {}
        for metric in METRICS:
            med = median_curve(exact[metric])
            value, kind = matched_budget(
                exact["budgets"], med, ref_med[metric]
            )
            summary["exact"][metric] = {
                "n_star": value,
                "kind": kind,
                "median_curve": med.tolist(),
            }
            lines.append(
                f"- Exact draws reach the reference {METRIC_NAMES[metric]} at "
                f"N = {fmt_n(value, kind, int(exact['budgets'][-1]))} "
                f"({fmt_n(value / 2 if math.isfinite(value) else value, kind)} draws), "
                "a bound on any sampler under this burn-in rule."
            )
        lines.append("")

    # Figures.
    figures = []
    for metric in METRICS:
        path = run / "figures" / f"curves_{metric}.png"
        _figure(path, ref, runs, exact, metric, n_star)
        figures.append(path)
        lines.append(f"![{METRIC_NAMES[metric]}](figures/{path.name})\n")

    # Provenance.
    lines.append("## Provenance\n")
    lines.append("| directory | commit | wall | seeds run | argv |")
    lines.append("|---|---|---|---|---|")
    for entry in invocations:
        lines.append(
            f"| {entry['directory']} | {entry['commit']} | "
            f"{entry['wall_s'] / 60:.1f} min | {len(entry['seeds_run'])} | "
            f"`{' '.join(entry['argv'])}` |"
        )
    if invocations:
        last = invocations[-1]
        lines.append("")
        lines.append(
            f"Python {last['python']}, {last['platform']}, {last['cpus']} CPUs; "
            + ", ".join(f"{k} {v}" for k, v in last["versions"].items())
            + f"; threads {last['threads']}."
        )
    summary["invocations"] = invocations

    (run / "report.md").write_text("\n".join(lines) + "\n")
    (run / "summary.json").write_text(
        json.dumps(_encode(summary), indent=1) + "\n"
    )
    print("\n".join(lines))
    print(
        f"\nwrote {run / 'report.md'}, {run / 'summary.json'}, "
        + ", ".join(str(p) for p in figures)
    )
    return 0


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def add_config(p):
        p.add_argument("--config", choices=["exporter"])
        p.add_argument("--target")
        p.add_argument("--D", type=int)
        p.add_argument("--jobs", type=int, default=1)

    p = sub.add_parser("check", help="the checks before the runs")
    add_config(p)
    p.set_defaults(func=cmd_check, jobs=4)

    p = sub.add_parser("pyvbmc", help="the reference runs")
    add_config(p)
    p.add_argument("--seeds", default="0-99")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_pyvbmc)

    p = sub.add_parser("pilot", help="the grid of settings on the pilot seeds")
    add_config(p)
    p.add_argument("--sampler", required=True, choices=SAMPLERS)
    p.add_argument("--seeds", default=f"{PILOT_SEEDS[0]}-{PILOT_SEEDS[-1]}")
    p.add_argument("--n-max", type=int, default=N_MAX)
    p.add_argument("--out", required=True)
    p.add_argument("--reference", help="PyVBMC runs (default OUT/../pyvbmc)")
    p.set_defaults(func=cmd_pilot)

    p = sub.add_parser("mcmc", help="the replicates of one setting")
    add_config(p)
    p.add_argument("--sampler", required=True, choices=SAMPLERS + ("exact",))
    p.add_argument("--setting", default="")
    p.add_argument("--seeds", default="0-99")
    p.add_argument("--n-max", type=int, default=N_MAX)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_mcmc)

    p = sub.add_parser("report", help="the curves, N*, intervals and figures")
    p.add_argument("run")
    p.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
