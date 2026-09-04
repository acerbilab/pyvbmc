"""Stage-level oracles: named computations on a rebuilt snapshot state.

Each oracle is a function ``(state, seed) -> dict[str, ndarray]`` over the
dict returned by :func:`_state.build_state`, with an ``applies`` predicate
and a tolerance. The generator stores what the oracles return as the
reference; the tests recompute and compare with :func:`compare`.

Randomness: functions that draw internally (``entmc_vbmc``, the
variational objective with Monte Carlo entropy, ``active_sample``) are
given a generator seeded with ``seed`` (and the legacy global state is
seeded for the pieces that still use it), so a change in the *order* of
draws re-baselines those oracles deliberately, while the pointwise oracles
(prediction, density, acquisition, analytic expectations, transformer) are
deterministic and must agree to rounding.

Tolerances are per element with a robust floor (see :func:`compare`),
may differ per output of one oracle, and were set from the floor measured
on 2026-09-04 by comparing single-threaded references against a
default-threaded recomputation of the same fixtures (a change of BLAS
summation order): the predictive mean, the densities, the expected log
joint, the ELCBO, both entropies and the transformer were bit-identical;
the predictive variance moved by up to 2e-5 of its per-element scale
(cancellation in ``k** - v'v`` near training points, where the variance
is tiny); the log acquisition by 3e-7; the exponential-form acquisitions
(``AcqFcn``, ``AcqFcnNoisy``, ``AcqFcnVanilla``) by 1e-5, since they are
exponentials of the log form.

A different BLAS build is a larger perturbation than a thread count: the
first Ubuntu CI run (fixtures generated on Windows, both NumPy's bundled
OpenBLAS) moved every quantity that passes through the GP solve on the
ill-conditioned cigar snapshot: expected-log-joint gradients by 3e-8 per
element, the per-sample expectations and the I_sk integrals by 3e-10, the
pairwise J_sjk terms by 2e-11 absolute (the Cholesky's conditioning,
about 1e8, times machine epsilon). Hence three classes. GP-solve outputs
(expected log joint and its gradients, the ELCBO and its gradient) are
held to 1e-6 relative plus 1e-10 absolute per element; the predictive
mean at the candidate points, which reach far into the probit tails on
the bounded snapshot, to 1e-4 (Ubuntu floor 1.2e-6 there).
Variance-type outputs, which are differences of nearly equal terms
(predictive variance, ``varG``, ``var_ss``, the pairwise ``J_sjk``
integrals, the ELCBO variances), are held to 1e-3 relative plus 1e-8
absolute: on the corr snapshot Ubuntu moved ``J_sjk`` by 2.3e-10 absolute
on entries of order 1e-6. GP-free outputs (densities, entropies, theta,
the transformer) stay at 1e-10, and the bit-identical ones are effectively
exact. Each tolerance leaves at least 30x over its measured floor; if a
platform exceeds one, re-measure there (``make_oracle_fixtures --check
--verbose``) rather than guess.

Two combinations here are not what production runs, on purpose: the
``entmc`` oracle and the first ``neg_elcbo`` call use ``ceil(ns_ent(K)/K)``
Monte Carlo samples even at ``K = 1``, where the sieve would switch to the
deterministic entropy (the ``*_detent`` outputs cover that path), and the
``K = 1`` snapshot is synthetic. Both are extra coverage, not behaviour.
"""

import contextlib
import copy
import math

import numpy as np

from pyvbmc import acquisition_functions as acqs
from pyvbmc.entropy import entlb_vbmc, entmc_vbmc
from pyvbmc.vbmc.active_importance_sampling import active_importance_sampling
from pyvbmc.vbmc.active_sample import active_sample
from pyvbmc.vbmc.variational_optimization import _gp_log_joint, _neg_elcbo

from ._state import _missing_fun

DEFAULT_SEED = 20260904


class Oracle:
    """``rtol`` / ``atol`` are floats, or dicts ``{output_key: value}`` with
    a ``"default"`` entry, so that outputs of one oracle with different
    conditioning (a mean and a variance) get different tolerances."""

    def __init__(self, name, fn, applies, rtol, atol):
        self.name = name
        self.fn = fn
        self.applies = applies
        self.rtol = rtol
        self.atol = atol

    def tolerance(self, key):
        def pick(v):
            return v.get(key, v["default"]) if isinstance(v, dict) else v

        return pick(self.rtol), pick(self.atol)

    def __call__(self, state, seed=DEFAULT_SEED):
        out = self.fn(state, seed)
        return {k: np.asarray(v, dtype=float) for k, v in out.items()}


ORACLES = {}


def oracle(name, rtol=1e-10, atol=1e-12, applies=lambda state: True):
    def wrap(fn):
        ORACLES[name] = Oracle(name, fn, applies, rtol, atol)
        return fn

    return wrap


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def prepare_gp_for_acq(gp, function_logger, optim_state):
    """What ``active_sample`` stores on the GP before evaluating an
    acquisition (``active_sample.py``, the ``sn2_new`` / ``X_rescaled``
    block): the mean noise variance per training point and the inputs
    rescaled by the geometric-mean length scale."""
    Ns = len(gp.posteriors)
    cov_N = gp.covariance.hyperparameter_count(gp.D)
    noise_N = gp.noise.hyperparameter_count()
    sn2new = np.zeros((gp.X.shape[0], Ns))
    for s in range(Ns):
        hyp_noise = gp.posteriors[s].hyp[cov_N : cov_N + noise_N]
        if hasattr(function_logger, "S"):
            s2 = (
                function_logger.S[function_logger.X_flag] ** 2
            ) * function_logger.n_evals[function_logger.X_flag]
        else:
            s2 = None
        sn2new[:, s] = gp.noise.compute(hyp_noise, gp.X, gp.y, s2).reshape(-1)
    gp.temporary_data["sn2_new"] = sn2new.mean(1)
    ln_ell = np.stack([gp.posteriors[s].hyp[: gp.D] for s in range(Ns)], 1)
    optim_state["gp_length_scale"] = np.exp(ln_ell.mean(1))
    gp.temporary_data["X_rescaled"] = gp.X / optim_state["gp_length_scale"]


@contextlib.contextmanager
def legacy_seed(seed):
    """Seed NumPy's global legacy state for the block and restore it after,
    so an oracle never leaves the process on a fixed stream (later
    unseeded tests must stay independent of it)."""
    saved = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(saved)


def _ns_ent_K(state, K):
    """Monte Carlo entropy samples *per component*, as ``optimize_vp``
    passes them (``ceil(ns_ent(K) / K)``)."""
    return int(math.ceil(state["options"].eval("ns_ent", {"K": K}) / K))


def _ns_ent_fine_K(state, K):
    """The `_eval_full_elcbo` count, ``ceil(ns_ent_fine(K) / K)``."""
    return int(math.ceil(state["options"].eval("ns_ent_fine", {"K": K}) / K))


def prepare_importance_sampling(state, acq, seed):
    """Draw the importance samples the VIQR/IMIQR acquisitions read from
    ``optim_state["active_importance_sampling"]``, from the rebuilt state
    with a seeded generator. The recorded ones belong to the GP *before*
    the iteration's last evaluation and do not match the recorded GP."""
    vp = state["vp"]
    vp.rng = np.random.default_rng(seed)
    with legacy_seed(seed):  # the MCMC branch draws from the legacy state
        state["optim_state"][
            "active_importance_sampling"
        ] = active_importance_sampling(vp, state["gp"], acq, state["options"])


def _is_noisy(state):
    return bool(state["logger"].noise_flag)


def _has_target(state):
    return getattr(state["logger"], "fun", None) not in (None, _missing_fun)


def _no_full_update(state):
    o = state["options"]
    return not (o["active_sample_gp_update"] or o["active_sample_vp_update"])


# --------------------------------------------------------------------------
# oracles
# --------------------------------------------------------------------------


# Mean: bit-identical across thread counts. Variance: floor 2e-5 per element
# (cancellation near the training points).
# Mean at arbitrary candidate points: Ubuntu moved the per-sample mean on
# the bounded (probit) snapshot by 1.2e-6 per element, worse than the
# analytic expectations under the VP, which weight the data region.
@oracle(
    "gp_predict",
    rtol={"default": 1e-4, "fs2": 1e-3, "fs2_samples": 1e-3},
    atol={"default": 1e-10, "fs2": 1e-8, "fs2_samples": 1e-8},
)
def gp_predict(state, seed):
    Xs = state["cand"]["Xs"]
    fmu_s, fs2_s = state["gp"].predict(Xs, separate_samples=True)
    fmu, fs2 = state["gp"].predict(Xs)
    return {
        "fmu_samples": fmu_s,
        "fs2_samples": fs2_s,
        "fmu": fmu,
        "fs2": fs2,
    }


@oracle("vp_pdf")
def vp_pdf(state, seed):
    vp = state["vp"]
    Xs = state["cand"]["Xs"]
    y, dy = vp.pdf(Xs, orig_flag=False, log_flag=False, grad_flag=True)
    ly, dly = vp.pdf(Xs, orig_flag=False, log_flag=True, grad_flag=True)
    y_orig = vp.pdf(vp.parameter_transformer.inverse(Xs), orig_flag=True)
    return {
        "pdf": y,
        "dpdf": dy,
        "logpdf": ly,
        "dlogpdf": dly,
        "pdf_orig": y_orig,
    }


def _make_acq_oracle(cls_name, applies, rtol):
    def fn(state, seed):
        gp = state["gp"]
        optim_state = state["optim_state"]
        prepare_gp_for_acq(gp, state["logger"], optim_state)
        acq = getattr(acqs, cls_name)()
        if acq.acq_info.get("importance_sampling"):
            prepare_importance_sampling(state, acq, seed)
        # `_real2int` inside `__call__` mutates its input: pass a copy.
        return {
            "acq": acq(
                np.array(state["cand"]["Xs"]),
                gp,
                state["vp"],
                state["logger"],
                optim_state,
            )
        }

    oracle(f"acq_{cls_name}", rtol=rtol, atol=0.0, applies=applies)(fn)


# Log-form acquisitions: floor 3e-7 per element (they carry the variance).
# Exponential-form ones (exp of the log form): floor 1e-5.
_make_acq_oracle("AcqFcnLog", lambda state: True, 1e-5)
for _name in ("AcqFcn", "AcqFcnVanilla", "AcqFcnNoisy"):
    _make_acq_oracle(_name, lambda state: True, 1e-3)
for _name in ("AcqFcnVIQR", "AcqFcnIMIQR"):
    _make_acq_oracle(_name, _is_noisy, 1e-5)


# GP-solve class (see the module docstring): Ubuntu floors on the
# ill-conditioned cigar snapshot were 3e-8 (gradients), 3e-10 (I_sk),
# 2e-11 absolute (J_sjk).
@oracle(
    "gp_log_joint",
    rtol={"default": 1e-6, "varG": 1e-3, "var_ss": 1e-3, "J_sjk": 1e-3},
    atol={"default": 1e-10, "varG": 1e-8, "var_ss": 1e-8, "J_sjk": 1e-8},
)
def gp_log_joint(state, seed):
    vp = copy.deepcopy(state["vp"])
    gp = state["gp"]
    # Gradients (no variance: the variance gradient is unimplemented for
    # the full variance), averaged over hyperparameter samples ...
    G, dG, _, _, _ = _gp_log_joint(vp, gp, True, True, True, False)
    out = {"G": G, "dG": dG}
    # ... and per sample.
    G_s, dG_s, _, _, _ = _gp_log_joint(vp, gp, True, False, True, False)
    out.update({"G_samples": G_s, "dG_samples": dG_s})
    # Variance of the expected log joint, with the per-component pieces.
    G_v, _, varG, _, var_ss, I_sk, J_sjk = _gp_log_joint(
        vp, gp, False, True, True, True, True
    )
    out.update(
        {
            "G_var_call": G_v,
            "varG": varG,
            "var_ss": var_ss,
            "I_sk": I_sk,
            "J_sjk": J_sjk,
        }
    )
    return out


# The entropies (H, H_detent) and theta do not touch the GP: exact class.
_VARIANCE_KEYS = ("varF", "varG", "varG_ss", "varH")


@oracle(
    "neg_elcbo",
    rtol={
        "default": 1e-6,
        "H": 1e-10,
        "H_detent": 1e-10,
        "theta": 1e-12,
        **{k: 1e-3 for k in _VARIANCE_KEYS},
    },
    atol={"default": 1e-10, **{k: 1e-8 for k in _VARIANCE_KEYS}},
)
def neg_elcbo(state, seed):
    gp = state["gp"]
    vp = copy.deepcopy(state["vp"])
    vp.rng = np.random.default_rng(seed)
    theta = vp.get_parameters(raw_flag=True)
    theta_bnd = vp.get_bounds(gp.X, state["options"], vp.K)
    ns_ent_K = _ns_ent_K(state, vp.K)
    # `_neg_elcbo` shifts the eta block of `theta` in place: every call
    # gets its own copy, and the stored theta is the unshifted one.
    # As Adam sees it: gradient, no variance (the variance gradient is
    # unimplemented for the full variance), Monte Carlo entropy.
    F, dF, G, H, _ = _neg_elcbo(
        theta.copy(), gp, vp, 0.0, ns_ent_K, True, False, theta_bnd
    )
    # Deterministic (Jensen lower bound) entropy, gradient.
    F0, dF0, _, H0, _ = _neg_elcbo(
        theta.copy(), gp, vp, 0.0, 0, True, False, theta_bnd
    )
    # As `_eval_full_elcbo` calls it: no gradient, full variance, per-K
    # pieces, `ns_ent_fine` samples per component.
    vp.rng = np.random.default_rng(seed)
    Ff, _, Gf, Hf, varF, _, varG_ss, varG, varH, _, _ = _neg_elcbo(
        theta.copy(),
        gp,
        vp,
        0.0,
        _ns_ent_fine_K(state, vp.K),
        False,
        True,
        None,
        0.0,
        True,
    )
    return {
        "theta": theta,
        "F": F,
        "dF": dF,
        "G": G,
        "H": H,
        "F_detent": F0,
        "dF_detent": dF0,
        "H_detent": H0,
        "F_full": Ff,
        "G_full": Gf,
        "H_full": Hf,
        "varF": varF,
        "varG_ss": varG_ss,
        "varG": varG,
        "varH": varH,
    }


@oracle("entlb")
def entlb(state, seed):
    H, dH = entlb_vbmc(copy.deepcopy(state["vp"]))
    return {"H": H, "dH": dH}


@oracle("entmc")
def entmc(state, seed):
    vp = copy.deepcopy(state["vp"])
    H, dH = entmc_vbmc(
        vp, _ns_ent_K(state, vp.K), rng=np.random.default_rng(seed)
    )
    return {"H": H, "dH": dH}


@oracle("transform", rtol=1e-12, atol=1e-14)
def transform(state, seed):
    pt = state["pt"]
    fl = state["logger"]
    X_orig = fl.X_orig[fl.X_flag]
    U = pt(X_orig)
    return {
        "U": U,
        "X_back": pt.inverse(U),
        "log_abs_det_jacobian": pt.log_abs_det_jacobian(U),
    }


@oracle(
    "active_sample_step",
    rtol=0.0,
    atol=1e-8,
    applies=lambda state: _has_target(state) and _no_full_update(state),
)
def active_sample_step(state, seed):
    vp = copy.deepcopy(state["vp"])
    vp.rng = np.random.default_rng(seed)
    gp = copy.deepcopy(state["gp"])
    optim_state = copy.deepcopy(state["optim_state"])
    fl = copy.deepcopy(state["logger"])
    # The deep copies each carry their own transformer; restore the shared
    # object the package assumes (numerically identical either way).
    fl.parameter_transformer = vp.parameter_transformer
    Xn0 = fl.Xn
    # Read only when the per-sample full update is on, which `applies`
    # excludes; kept so the call matches production's signature.
    history = {"r_index": np.array([state["meta"].get("r_index", np.inf)])}
    n = state["options"]["fun_evals_per_iter"]
    with legacy_seed(seed):  # cma.NoiseHandler draws from the legacy state
        fl, optim_state, vp, gp = active_sample(
            gp, n, optim_state, fl, history, vp, state["options"]
        )
    return {
        "X_new": fl.X_orig[Xn0 + 1 : fl.Xn + 1],
        "y_new": fl.y_orig[Xn0 + 1 : fl.Xn + 1],
    }


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------


def compare(reference, output, rtol, atol):
    """Compare two oracle output dicts; returns a list of
    ``(key, max_abs_err, max_scaled_err, ok)`` rows, one per key.
    ``rtol`` / ``atol`` may be floats or per-key dicts (see :class:`Oracle`).

    Per element, ``|out - ref| <= rtol * denom + atol`` with ``denom =
    max(|ref|, floor)`` and ``floor`` the lower quartile of ``|ref|`` over
    the finite entries; ``nan`` / ``inf`` patterns must match exactly. The
    floor keeps the criterion meaningful for quantities that are
    differences of nearly equal numbers (the GP predictive variance near
    training points, the acquisitions built on it), whose rounding noise is
    set by the scale of the terms rather than by the small result, while a
    per-element denominator keeps every entry of the array load-bearing
    (a global scale would let the tail entries, ten orders of magnitude
    below the peak, change freely).
    """

    def pick(v, key):
        return v.get(key, v["default"]) if isinstance(v, dict) else v

    rows = []
    for key, ref in reference.items():
        rtol_k, atol_k = pick(rtol, key), pick(atol, key)
        ref = np.asarray(ref, dtype=float)
        if key not in output:
            rows.append((key, np.inf, np.inf, False))
            continue
        out = np.asarray(output[key], dtype=float)
        if out.shape != ref.shape:
            rows.append((key, np.inf, np.inf, False))
            continue
        finite = np.isfinite(ref) & np.isfinite(out)
        pattern_ok = bool(
            np.array_equal(np.isnan(ref), np.isnan(out))
            and np.array_equal(np.isposinf(ref), np.isposinf(out))
            and np.array_equal(np.isneginf(ref), np.isneginf(out))
        )
        if finite.any():
            a = np.abs(ref[finite])
            diff = np.abs(out[finite] - ref[finite])
            floor = max(float(np.quantile(a, 0.25)), np.finfo(float).tiny)
            denom = np.maximum(a, floor)
            abs_err = float(np.max(diff))
            scaled_err = float(np.max(diff / denom))
            within = bool(np.all(diff <= rtol_k * denom + atol_k))
        else:
            abs_err, scaled_err, within = 0.0, 0.0, True
        rows.append((key, abs_err, scaled_err, pattern_ok and within))
    for key in output:
        if key not in reference:
            rows.append((key, np.inf, np.inf, False))
    return rows


def applicable(state):
    """Names of the oracles whose ``applies`` predicate holds on ``state``."""
    return [name for name, orc in ORACLES.items() if orc.applies(state)]


def format_rows(rows):
    return "\n".join(
        f"  {'ok ' if ok else 'BAD'} {k:24s} max|d| {a:.2e}  scaled {r:.2e}"
        for k, a, r, ok in rows
    )
