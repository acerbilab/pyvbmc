"""Dtype checks: the float64 canary.

PyVBMC computes in float64 because NumPy's defaults give it; no class
declares the dtype of its state. The oracle tests compare values after
casting every output to float, so they see a float32 regression only as
a value mismatch, and only where the tolerance is tight enough (the
densities, the entropies, the transformer); the looser classes (GP
prediction, the acquisitions, every variance) let it through, and an
array whose precision was lost before a widening cast (the transformer's
``astype(float)``, the ``dtype=float`` copies of the VP blocks in the
expected log joint and the Monte Carlo entropy) is invisible to any check
on outputs. The functions here make the property explicit.

:func:`iter_dtype_leaves` walks an object graph and yields every leaf
that carries a dtype. :func:`assert_float64` fails on a floating-point
leaf that is not float64, on a complex one, on a leaf whose dtype is not
NumPy's at all (a tensor of another backend is an offender, not something
the walk steps over), and on a walk that found fewer leaves than expected
(a broken descent cannot pass green). The walk enters containers, object
arrays and instances of ``pyvbmc`` / ``gpyreg`` classes; everything else
(loggers, callables, generators, SciPy distributions, plotting objects)
is a boundary, so a third-party object holding a float32 buffer for its
own purposes cannot fail the canary, and a new attribute of a PyVBMC
class is covered without registration.

:func:`load_bearing_arrays` names the arrays the numerics ride on and
:func:`assert_manifest_float64` requires each to be present and float64,
which catches what the walk cannot: an array that turned into an
integer, a scalar, or nothing.
"""

from collections import deque

import numpy as np

__all__ = [
    "iter_dtype_leaves",
    "non_float64_leaves",
    "assert_float64",
    "load_bearing_arrays",
    "assert_manifest_float64",
]

_ATOMS = (str, bytes, int, float, bool, complex, type(None), type, np.dtype)
_PACKAGES = ("pyvbmc", "gpyreg")


def _entered(obj):
    """Whether ``obj`` is an instance of a ``pyvbmc`` or ``gpyreg`` class."""
    module = type(obj).__module__ or ""
    return module.split(".", 1)[0] in _PACKAGES


def iter_dtype_leaves(obj, path="root"):
    """Yield ``(path, dtype)`` for every dtype-carrying leaf under ``obj``.

    Breadth first; each object is visited once (its ``id`` is remembered
    and a reference to it kept, so an id cannot be recycled during the
    walk). Leaves: a NumPy array (its dtype), a NumPy scalar (its dtype;
    ``np.float64`` and ``np.complex128`` are Python ``float`` and
    ``complex`` subclasses, so scalars are tested before the atoms), and
    any other object exposing a ``dtype`` attribute, yielded as is and
    not entered even if its class is a ``pyvbmc`` one. Entered: object
    arrays, ``dict`` values, ``list``, ``tuple`` and ``set`` elements, and
    the ``__dict__`` of instances of ``pyvbmc`` / ``gpyreg`` classes.
    Python scalars, strings, ``None``, classes and everything else end a
    branch.
    """
    seen = set()
    keep = []
    queue = deque([(obj, path)])
    while queue:
        o, p = queue.popleft()
        if isinstance(o, np.generic):
            yield p, o.dtype
            continue
        if isinstance(o, _ATOMS):
            continue
        if id(o) in seen:
            continue
        seen.add(id(o))
        keep.append(o)
        if isinstance(o, np.ndarray):
            if o.dtype == object:
                queue.extend((v, f"{p}[{i}]") for i, v in enumerate(o.ravel()))
            else:
                yield p, o.dtype
        elif isinstance(o, dict):
            queue.extend((v, f"{p}[{k!r}]") for k, v in o.items())
        elif isinstance(o, (list, tuple, set, frozenset)):
            queue.extend((v, f"{p}[{i}]") for i, v in enumerate(o))
        elif hasattr(o, "dtype"):
            yield p, o.dtype
        elif _entered(o):
            attrs = getattr(o, "__dict__", None)
            if attrs:
                queue.extend((v, f"{p}.{k}") for k, v in attrs.items())


def non_float64_leaves(obj, path="root"):
    """Return ``(offenders, n_leaves)`` for the walk from ``obj``.

    An offender is a leaf whose dtype is not a NumPy dtype, or is complex,
    or is a floating-point dtype other than float64; it is reported as
    ``(path, dtype name)``. Integer and boolean leaves pass.
    """
    offenders = []
    n_leaves = 0
    for p, dt in iter_dtype_leaves(obj, path):
        n_leaves += 1
        if not isinstance(dt, np.dtype):
            offenders.append((p, repr(dt)))
        elif dt.kind == "c" or (dt.kind == "f" and dt != np.float64):
            offenders.append((p, dt.name))
    return offenders, n_leaves


def assert_float64(obj, path="root", min_leaves=1):
    """Fail if the walk from ``obj`` finds fewer than ``min_leaves`` leaves,
    or any leaf that :func:`non_float64_leaves` reports. Returns the
    number of leaves found."""
    offenders, n_leaves = non_float64_leaves(obj, path)
    assert n_leaves >= min_leaves, (
        f"{path}: {n_leaves} dtype leaves found, fewer than the "
        f"{min_leaves} expected; the descent is broken or state moved"
    )
    assert (
        not offenders
    ), f"{path}: {len(offenders)} leaves are not float64:\n" + "\n".join(
        f"  {p}  <{d}>" for p, d in offenders
    )
    return n_leaves


def load_bearing_arrays(vp=None, gp=None, logger=None, pt=None):
    """The arrays the numerics ride on, as ``{name: array}``.

    VP: ``w, eta, mu, sigma, lambd``. GP: ``X, y``, ``s2`` when present,
    and ``hyp, alpha, sW, L`` of every posterior. Function logger:
    ``X_orig, y_orig, X, y``, and ``S`` when the target is noisy.
    Parameter transformer: ``lb_orig, ub_orig, mu, delta``, and ``R_mat,
    scale`` when a warp is installed. A renamed attribute raises
    ``AttributeError``, which is the intended failure. ``gp`` must be a
    complete GP: one without posteriors, or a lean history record (whose
    factors are ``None``; ``VBMC.get_gp`` restores them), is reported as
    missing arrays.
    """
    arrays = {}
    if vp is not None:
        for k in ("w", "eta", "mu", "sigma", "lambd"):
            arrays[f"vp.{k}"] = getattr(vp, k)
    if gp is not None:
        arrays["gp.X"] = gp.X
        arrays["gp.y"] = gp.y
        if gp.s2 is not None:
            arrays["gp.s2"] = gp.s2
        posteriors = gp.posteriors
        if posteriors is None:
            arrays["gp.posteriors"] = None
            posteriors = ()
        for s, post in enumerate(posteriors):
            for k in ("hyp", "alpha", "sW", "L"):
                arrays[f"gp.posteriors[{s}].{k}"] = getattr(post, k)
    if logger is not None:
        for k in ("X_orig", "y_orig", "X", "y"):
            arrays[f"logger.{k}"] = getattr(logger, k)
        if logger.noise_flag:
            arrays["logger.S"] = logger.S
    if pt is not None:
        for k in ("lb_orig", "ub_orig", "mu", "delta"):
            arrays[f"pt.{k}"] = getattr(pt, k)
        for k in ("R_mat", "scale"):
            v = getattr(pt, k)
            if v is not None:
                arrays[f"pt.{k}"] = v
    return arrays


def assert_manifest_float64(arrays):
    """Fail unless every value of ``arrays`` is a float64 NumPy array,
    naming the ones that are not and what they are instead."""
    bad = []
    for name, a in arrays.items():
        if not isinstance(a, np.ndarray):
            bad.append((name, type(a).__name__))
        elif a.dtype != np.float64:
            bad.append((name, a.dtype.name))
    assert not bad, (
        f"{len(bad)} load-bearing arrays are not float64 arrays:\n"
        + "\n".join(f"  {n}  <{d}>" for n, d in bad)
    )
