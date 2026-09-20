"""P5-15 (P5c F17): tie order of the descending sorts.

`_estimate_noise` (`gaussian_process_train.py:849`) and
`pyvbmc/stats/get_hpd.py:34` both do `np.argsort(y, axis=None)[::-1]`.
MATLAB's `sort(y,'descend')` is a stable sort, so equal values keep their
input order. This script shows what the Python expression actually does
with ties (numpy's default `argsort` kind is `quicksort`, which is not
stable), how it compares with a reversed *stable* ascending sort, and
with MATLAB's stable descending order, and how often a tie at the cut
changes the selected subset.
"""

import numpy as np

import pyvbmc
from pyvbmc.stats import get_hpd

print("pyvbmc.__file__ =", pyvbmc.__file__)
print("numpy version   =", np.__version__)


def orders(y):
    py = np.argsort(y, axis=None)[::-1]  # the code
    rev_stable = np.argsort(y, axis=None, kind="stable")[
        ::-1
    ]  # reviewer's model
    matlab = np.argsort(
        -np.asarray(y).ravel(), kind="stable"
    )  # sort(y,'descend')
    return py, rev_stable, matlab


cases = {
    "5,3,3,1,0": np.array([[5.0], [3.0], [3.0], [1.0], [0.0]]),
    "three-way tie": np.array([[2.0], [2.0], [2.0], [1.0], [0.0]]),
    "tie at the 0.8 cut (N=10)": np.array(
        [[9.0], [8.0], [7.0], [6.0], [5.0], [4.0], [3.0], [3.0], [1.0], [0.0]]
    ),
    "tie at the 0.2 cut (N=10)": np.array(
        [[9.0], [9.0], [9.0], [6.0], [5.0], [4.0], [3.0], [2.0], [1.0], [0.0]]
    ),
}
for name, y in cases.items():
    py, rev_stable, matlab = orders(y)
    print(f"\n{name}: y = {y.ravel()}")
    print("  np.argsort(y)[::-1]        =", py)
    print("  reversed *stable* ascending=", rev_stable)
    print("  MATLAB stable 'descend'    =", matlab)
    print("  python == matlab ?", np.array_equal(py, matlab))
    n_hpd = round(0.8 * len(y))
    n_est = int(np.ceil(0.2 * len(y)))
    print(
        f"  get_hpd(0.8) keeps {n_hpd}: python {sorted(py[:n_hpd])}"
        f" vs matlab {sorted(matlab[:n_hpd])} -> same set:"
        f" {set(py[:n_hpd]) == set(matlab[:n_hpd])}"
    )
    print(
        f"  _estimate_noise keeps {n_est}: python {sorted(py[:n_est])}"
        f" vs matlab {sorted(matlab[:n_est])} -> same set:"
        f" {set(py[:n_est]) == set(matlab[:n_est])}"
    )

# Random search for a disagreement at the 0.8 cut.
rng = np.random.default_rng(0)
diff_hpd = diff_est = 0
trials = 2000
for _ in range(trials):
    y = rng.integers(0, 5, size=(10, 1)).astype(float)
    py, _, matlab = orders(y)
    if set(py[:8]) != set(matlab[:8]):
        diff_hpd += 1
    if set(py[:2]) != set(matlab[:2]):
        diff_est += 1
print(f"\n{trials} random 10-point integer-valued y vectors (many ties):")
print(f"  get_hpd(0.8) subset differs in {diff_hpd} of them")
print(f"  _estimate_noise subset differs in {diff_est} of them")

X = np.arange(5).reshape(-1, 1).astype(float)
hpd_X, hpd_y, hpd_range, idx = get_hpd(X, cases["5,3,3,1,0"], 0.6)
print("\nget_hpd(X, [5,3,3,1,0], 0.6) returns indices", idx)
