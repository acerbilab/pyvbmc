"""W5, the remainder of the balanced draw of `VariationalPosterior.sample`.

`variational_posterior.py:641-644` computes the weights of the remainder of a
balanced draw as `w_extra += self.w * (repeats_extra - sum(w_extra))`, with
the builtin `sum` on a `(1, K)` array, where `vbmc_rnd.m:67-71` adds
`w*delta_extra` with the scalar `delta_extra = N_extra - sum(w_extra)`. This
script settles:

1. what the builtin returns for that array;
2. the probabilities of the remainder draw as coded and as MATLAB has them,
   and the expected number of draws per component under each, on toy weights
   and on the `K = 50` weights of the MATLAB-derived fixture `vp-test.npz`;
3. the component frequencies that `vp.sample` produces, by repetition;
4. what the corrected line would move: with the same seed, how often the
   index vector and the samples of a balanced draw differ between the method
   as coded and the method with `np.sum`, at the numbers of draws the search
   asks for, and whether one active-sampling step of the stored oracle
   states acquires another point.

Pass `--steps` for part 4's active-sampling steps (a CMA-ES search each).
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import inspect
import sys
import textwrap
from pathlib import Path

import numpy as np

import pyvbmc
import pyvbmc.variational_posterior.variational_posterior as vp_module
from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.variational_posterior import VariationalPosterior

print("pyvbmc:", pyvbmc.__file__, flush=True)

ROOT = Path(pyvbmc.__file__).resolve().parents[1]
FIXTURE = ROOT / "pyvbmc" / "testing" / "variational_posterior" / "vp-test.npz"


def remainder_probabilities(w, N):
    """(floor counts, coded p, MATLAB p) of the remainder of a balanced draw."""
    w = np.asarray(w, dtype=float).reshape(1, -1)
    floors = np.floor(w * N)
    e = w * N - floors
    R = np.ceil(np.sum(e))
    coded = e + w * (R - sum(e))  # the builtin: the row, not its total
    coded = coded / np.sum(coded)
    matlab = e + w * (R - np.sum(e))  # vbmc_rnd.m:67-71
    matlab = matlab / np.sum(matlab)
    return floors.ravel(), R, coded.ravel(), matlab.ravel()


print("\n--- 1. the builtin sum of a (1, K) array", flush=True)
a = np.array([[0.1, 0.6, 0.3]])
print("sum(a)    =", sum(a), "  shape", np.shape(sum(a)))
print("np.sum(a) =", np.sum(a))

print(
    "\n--- 2. probabilities of the remainder and expected counts", flush=True
)
for w, N in [([0.7, 0.2, 0.1], 13), ([1 / 3, 2 / 3], 10), ([0.5, 0.5], 11)]:
    floors, R, coded, matlab = remainder_probabilities(w, N)
    print(f"w = {np.round(w, 4)}, N = {N}: {int(R)} remainder draw(s)")
    print("   coded  p:", np.round(coded, 4))
    print("   MATLAB p:", np.round(matlab, 4))
    print("   expected counts, coded :", np.round(floors + R * coded, 4))
    print("   expected counts, MATLAB:", np.round(floors + R * matlab, 4))
    print("   w * N                  :", np.round(np.asarray(w) * N, 4))

with np.load(FIXTURE, allow_pickle=False) as fixture:
    w50 = fixture["w"].astype(float)
w50 = w50 / np.sum(w50)
print(
    f"\nfixture weights: K = {w50.size}, min {w50.min():.2e}, max {w50.max():.3f}"
)
for N in (100, 1000, 2048, 100000):
    floors, R, coded, matlab = remainder_probabilities(w50, N)
    tv = 0.5 * np.sum(np.abs(coded - matlab))
    bias = floors + R * coded - w50.ravel() * N
    print(
        f"   N = {N:6d}: {int(R):3d} remainder draws ({R / N:6.2%}); total "
        f"variation of the two remainder laws {tv:.3f}; expected count minus "
        f"w*N as coded: max |.| = {np.max(np.abs(bias)):.3f} draws, "
        f"MATLAB: {np.max(np.abs(floors + R * matlab - w50.ravel() * N)):.1e}"
    )

print("\n--- 3. frequencies of vp.sample by repetition", flush=True)
vp = VariationalPosterior(1, 3, np.array([[0.0]]))
vp.parameter_transformer = ParameterTransformer(1)
vp.w = np.array([[0.7, 0.2, 0.1]])
vp.mu = np.array([[-10.0, 0.0, 10.0]])
vp.sigma = np.ones((1, 3))
vp.lambd = np.ones((1, 1))
vp.rng = np.random.default_rng(1)
counts = np.zeros(3)
reps = 20000
for _ in range(reps):
    _, idx = vp.sample(13, orig_flag=False, balance_flag=True)
    counts += np.bincount(idx.astype(int), minlength=3)
floors, R, coded, matlab = remainder_probabilities([0.7, 0.2, 0.1], 13)
print("   observed frequencies:", np.round(counts / (13 * reps), 4))
print("   coded expectation   :", np.round((floors + R * coded) / 13, 4))
print("   MATLAB expectation  :", np.round((floors + R * matlab) / 13, 4))

print("\n--- 4. what the corrected line moves", flush=True)
source = textwrap.dedent(inspect.getsource(VariationalPosterior.sample))
old = "repeats_extra - sum(w_extra)"
assert source.count(old) == 1, "the line is not as the finding describes it"
namespace = dict(vars(vp_module))
exec(source.replace(old, "repeats_extra - np.sum(w_extra)"), namespace)
sample_coded = VariationalPosterior.sample
sample_fixed = namespace["sample"]


def fixture_vp():
    out = VariationalPosterior(2, 2, np.array([[5]]))
    with np.load(FIXTURE, allow_pickle=False) as f:
        out.D = int(f["D"][0, 0])
        out.K = int(f["K"][0, 0])
        out.w = f["w"] / np.sum(f["w"])
        out.mu = f["mu"]
        out.sigma = f["sigma"]
        out.lambd = f["lambd"]
    out.parameter_transformer = ParameterTransformer(out.D)
    return out


vp50 = fixture_vp()
for N in (100, 2048):
    differ, entries = 0, []
    for seed in range(200):
        vp50.rng = np.random.default_rng(seed)
        x_c, i_c = sample_coded(vp50, N, orig_flag=False, balance_flag=True)
        vp50.rng = np.random.default_rng(seed)
        x_f, i_f = sample_fixed(vp50, N, orig_flag=False, balance_flag=True)
        n_diff = int(np.sum(np.any(x_c != x_f, axis=1)))
        differ += n_diff > 0
        entries.append(n_diff)
    print(
        f"   K = 50, N = {N}: the samples differ in {differ} of 200 seeds; "
        f"rows that differ: median {int(np.median(entries))}, max {max(entries)}"
    )

if "--steps" in sys.argv:
    sys.path.insert(0, str(ROOT))
    from pyvbmc.testing.oracles._oracles import ORACLES
    from pyvbmc.testing.oracles._state import (
        build_state,
        load_snapshot,
        snapshot_names,
    )
    from pyvbmc.testing.oracles.test_oracles import FIXTURES, _target

    orc = ORACLES["active_sample_step"]
    for name in snapshot_names(FIXTURES):
        snap = load_snapshot(FIXTURES / name)
        if "active_sample_step" not in snap["ref"]:
            continue
        fun = _target(snap["meta"])
        seed = snap["meta"]["oracle_seed"]
        out = {}
        for label, method in (
            ("coded", sample_coded),
            ("fixed", sample_fixed),
        ):
            VariationalPosterior.sample = method
            state = build_state(snap, fun=fun)
            out[label] = orc.fn(state, seed)["X_new"]
        VariationalPosterior.sample = sample_coded
        K = build_state(snap, fun=fun)["vp"].K
        same = np.array_equal(out["coded"], out["fixed"])
        gap = np.max(np.abs(out["coded"] - out["fixed"]))
        print(
            f"   {name} (K = {K}): acquired points "
            f"{'identical' if same else 'differ'}, max |difference| {gap:.3g}",
            flush=True,
        )
