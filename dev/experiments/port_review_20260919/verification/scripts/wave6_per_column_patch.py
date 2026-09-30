"""W6, gplite's per-column bound recommendations, in this process alone.

gpyreg's two `_bounds_info_helper` functions take the statistics of the
training inputs over all entries of `X` where `gplite_covfun.m:126` and
`gplite_meanfun.m:142`, `:220-230` take them per column: the starting value
of every length scale (`covariance_functions.py:451`), and the bounds, the
plausible bounds and the starting values of the location and the scale of
the negative quadratic mean (`mean_functions.py:488`, `:511-524`).

`install()` replaces the two helpers by the same source with `axis=0` in
those statistics and returns the originals; `restore()` puts them back. Each
replaced line is asserted to be there once, so the module fails on a gpyreg
whose helpers are not as the findings describe them. Nothing is written to
the gpyreg checkout.
"""

import inspect
import textwrap

import gpyreg.covariance_functions as cov_module
import gpyreg.mean_functions as mean_module

COV_EDITS = [
    (
        "plausible_x0[0:D] = np.log(np.std(X, ddof=1))",
        "plausible_x0[0:D] = np.log(np.std(X, axis=0, ddof=1))",
    ),
]

MEAN_EDITS = [
    (
        "w = np.max(X) - np.min(X)\n",
        "w = np.max(X, axis=0) - np.min(X, axis=0)\n",
    ),
    (
        "LB[1 : 1 + D] = np.min(X) - 0.5 * w\n",
        "LB[1 : 1 + D] = np.min(X, axis=0) - 0.5 * w\n",
    ),
    (
        "UB[1 : 1 + D] = np.max(X) + 0.5 * w\n",
        "UB[1 : 1 + D] = np.max(X, axis=0) + 0.5 * w\n",
    ),
    (
        "PLB[1 : 1 + D] = np.min(X)\n",
        "PLB[1 : 1 + D] = np.min(X, axis=0)\n",
    ),
    (
        "PUB[1 : 1 + D] = np.max(X)\n",
        "PUB[1 : 1 + D] = np.max(X, axis=0)\n",
    ),
    (
        "x0[1 : 1 + D] = np.median(X)\n",
        "x0[1 : 1 + D] = np.median(X, axis=0)\n",
    ),
    (
        "x0[1 + D : mean_N] = np.log(np.std(X, ddof=1))\n",
        "x0[1 + D : mean_N] = np.log(np.std(X, axis=0, ddof=1))\n",
    ),
]


def _rebuild(module, edits):
    source = textwrap.dedent(inspect.getsource(module._bounds_info_helper))
    for old, new in edits:
        assert source.count(old) == 1, f"not there once: {old!r}"
        source = source.replace(old, new)
    namespace = dict(vars(module))
    exec(source, namespace)
    return namespace["_bounds_info_helper"]


_PER_COLUMN = {
    cov_module: _rebuild(cov_module, COV_EDITS),
    mean_module: _rebuild(mean_module, MEAN_EDITS),
}
_POOLED = {
    cov_module: cov_module._bounds_info_helper,
    mean_module: mean_module._bounds_info_helper,
}


def install():
    """Install the per-column helpers."""
    for module, helper in _PER_COLUMN.items():
        module._bounds_info_helper = helper


def restore():
    """Put gpyreg's own helpers back."""
    for module, helper in _POOLED.items():
        module._bounds_info_helper = helper
