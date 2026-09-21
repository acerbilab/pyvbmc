"""P9-9. Does `tile_inputs` check its non-scalar arguments against `size`?

Settles: the docstring's promise against the code; the report's (2, 2)
example, where `UniformBox(np.zeros((2,2)), np.ones((2,2)), D=4)` builds a
four-dimensional prior; and when the check does fire.
"""

import numpy as np

import pyvbmc
from pyvbmc.priors import UniformBox, tile_inputs

print("pyvbmc.__file__ =", pyvbmc.__file__)
print()

print("--- the docstring's Raises block ---")
doc = tile_inputs.__doc__
i = doc.find("Raises")
print(" ".join(doc[i:].split()))
print()

print("--- the (2, 2) example ---")
p = UniformBox(np.zeros((2, 2)), np.ones((2, 2)), D=4)
print(
    "UniformBox(zeros((2,2)), ones((2,2)), D=4) -> D =",
    p.D,
    "a =",
    p.a,
    "b =",
    p.b,
)
print()

print("--- tile_inputs directly ---")
cases = [
    ("shapes agree with size", (np.zeros(4), np.ones(4)), dict(size=4)),
    (
        "shapes disagree, same count",
        (np.zeros((2, 2)), np.ones((2, 2))),
        dict(size=4),
    ),
    (
        "shapes disagree, same count, squeeze",
        (np.zeros((2, 2)), np.ones((2, 2))),
        dict(size=4, squeeze=True),
    ),
    ("counts differ", (np.zeros(3), np.ones(3)), dict(size=4)),
    ("mixed shapes between args", (np.zeros(3), np.ones(4)), dict(size=None)),
    ("scalar + array", (0.0, np.ones(4)), dict(size=4)),
    ("scalar + array, size disagrees", (0.0, np.ones(3)), dict(size=4)),
]
for name, args, kw in cases:
    try:
        out = tile_inputs(*args, **kw)
        print(f"{name:38s} -> shapes {[o.shape for o in out]}")
    except Exception as e:
        print(f"{name:38s} -> {type(e).__name__}: {str(e)[:90]}")
print()

print("--- what the existing test uses ---")
import pyvbmc.testing.priors.test_tile_inputs as t

src = __import__("inspect").getsource(t)
j = src.find("def test_tile_inputs_wrong_size")
print(src[j : j + 500] if j >= 0 else "(no such test)")
