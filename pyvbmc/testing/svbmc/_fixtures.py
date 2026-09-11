"""Plain-array fixtures of fitted posteriors for the S-VBMC tests.

Every fixture under ``fixtures/`` is one fitted
:class:`~pyvbmc.VariationalPosterior` stored with the oracle snapshot codec
(:mod:`pyvbmc.testing.oracles._state`): a ``.npz`` holding the arrays,
loaded with ``allow_pickle=False``, and a JSON sidecar holding the tree,
the scalars and the provenance. :func:`load_vp` rebuilds the posterior and
its transformer through the public constructors, so the files do not
depend on the layout of any class. ``FIXTURES.md`` lists what every file
holds and where it came from; ``dev/scripts/make_svbmc_fixtures.py``
writes them.
"""

from pathlib import Path

import numpy as np

from pyvbmc.testing.oracles._state import (
    build_transformer,
    build_vp,
    encode,
    load_snapshot,
    save_snapshot,
    snapshot_names,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
REFERENCES = "references"


def _placeholder_plausible(lb, ub):
    """Finite plausible bounds strictly inside ``lb < ub`` for the rebuild.

    The transformer constructor derives ``mu`` and ``delta`` from them, and
    the rebuild then overwrites both with the stored values, so these never
    reach the rebuilt state; they only have to be valid.
    """
    plb = np.empty_like(lb)
    pub = np.empty_like(ub)
    for d in range(lb.shape[1]):
        lo, hi = lb[0, d], ub[0, d]
        if np.isfinite(lo) and np.isfinite(hi):
            plb[0, d], pub[0, d] = lo + 0.25 * (hi - lo), lo + 0.75 * (hi - lo)
        elif np.isfinite(lo):
            plb[0, d], pub[0, d] = lo + 1.0, lo + 2.0
        elif np.isfinite(hi):
            plb[0, d], pub[0, d] = hi - 2.0, hi - 1.0
        else:
            plb[0, d], pub[0, d] = -1.0, 1.0
    return plb, pub


def vp_snapshot(vp, meta):
    """Encode a fitted posterior and its transformer as ``(arrays, tree)``."""
    arrays = {}
    pt = vp.parameter_transformer
    D = int(vp.D)
    lb = np.array(pt.lb_orig, dtype=float).reshape(1, D)
    ub = np.array(pt.ub_orig, dtype=float).reshape(1, D)
    plb, pub = _placeholder_plausible(lb, ub)
    tree = {
        "pt": {
            "D": D,
            "lb_orig": encode(lb, "pt/lb_orig", arrays),
            "ub_orig": encode(ub, "pt/ub_orig", arrays),
            "plb_orig": encode(plb, "pt/plb_orig", arrays),
            "pub_orig": encode(pub, "pt/pub_orig", arrays),
            "mu": encode(np.array(pt.mu), "pt/mu", arrays),
            "delta": encode(np.array(pt.delta), "pt/delta", arrays),
            "type": encode(np.array(pt.type), "pt/type", arrays),
            "R_mat": None
            if pt.R_mat is None
            else encode(np.array(pt.R_mat), "pt/R_mat", arrays),
            "scale": None
            if pt.scale is None
            else encode(np.array(pt.scale), "pt/scale", arrays),
            "transform_type": "probit",
        },
        "vp": {
            "D": D,
            "K": int(vp.K),
            "w": encode(np.array(vp.w), "vp/w", arrays),
            "eta": encode(np.array(vp.eta), "vp/eta", arrays),
            "mu": encode(np.array(vp.mu), "vp/mu", arrays),
            "sigma": encode(np.array(vp.sigma), "vp/sigma", arrays),
            "lambd": encode(np.array(vp.lambd), "vp/lambd", arrays),
            "optimize_mu": bool(vp.optimize_mu),
            "optimize_sigma": bool(vp.optimize_sigma),
            "optimize_lambd": bool(vp.optimize_lambd),
            "optimize_weights": bool(vp.optimize_weights),
            "stats": encode(vp.stats, "vp/stats", arrays),
            "bounds": encode(vp.bounds, "vp/bounds", arrays),
        },
        "meta": encode(meta, "meta", arrays),
    }
    return arrays, tree


def save_vp(name, vp, meta, fixtures_dir=FIXTURES_DIR):
    """Write ``<name>.npz`` and ``<name>.json`` under ``fixtures_dir``."""
    arrays, tree = vp_snapshot(vp, meta)
    save_snapshot(Path(fixtures_dir) / name, arrays, tree)


def load_vp(name, rng=None, fixtures_dir=FIXTURES_DIR):
    """Rebuild one fixture; returns ``(vp, meta)``.

    ``rng`` seeds the rebuilt posterior's generator (a fixed seed when
    ``None``, so loading never touches NumPy's global state).
    """
    tree = load_snapshot(Path(fixtures_dir) / name)
    pt = build_transformer(tree["pt"])
    vp = build_vp(tree["vp"], pt, rng=rng)
    return vp, tree["meta"]


def fixture_names(fixtures_dir=FIXTURES_DIR):
    """Names of the posterior fixtures (the references file excluded)."""
    return [n for n in snapshot_names(fixtures_dir) if n != REFERENCES]


def group_names(fixtures_dir=FIXTURES_DIR):
    """Distinct ``meta["group"]`` values, sorted."""
    groups = set()
    for name in fixture_names(fixtures_dir):
        groups.add(load_snapshot(Path(fixtures_dir) / name)["meta"]["group"])
    return sorted(groups)


def load_group(group, rng=None, fixtures_dir=FIXTURES_DIR):
    """All posteriors of one group in file order; returns ``(vps, metas)``.

    Every posterior gets its own generator derived from ``rng`` (an
    integer seed, ``0`` when ``None``), so the group is reproducible
    without the posteriors sharing one stream. A ``Generator`` passed as
    ``rng`` is shared by all of them instead.
    """
    vps, metas = [], []
    seeds = None
    if not isinstance(rng, np.random.Generator):
        base = 0 if rng is None else rng
        seeds = iter(np.random.SeedSequence(base).spawn(10_000))
    for name in fixture_names(fixtures_dir):
        tree = load_snapshot(Path(fixtures_dir) / name)
        if tree["meta"]["group"] != group:
            continue
        pt = build_transformer(tree["pt"])
        this_rng = rng if seeds is None else np.random.default_rng(next(seeds))
        vps.append(build_vp(tree["vp"], pt, rng=this_rng))
        metas.append(tree["meta"])
    if not vps:
        raise KeyError(f"no fixtures in group {group!r} under {fixtures_dir}")
    return vps, metas


def load_references(fixtures_dir=FIXTURES_DIR):
    """The regression references tree (see ``FIXTURES.md``)."""
    return load_snapshot(Path(fixtures_dir) / REFERENCES)
