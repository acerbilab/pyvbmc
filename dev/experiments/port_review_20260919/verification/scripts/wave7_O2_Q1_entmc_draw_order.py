"""First question: one standard_normal((K, Ns/2, D)) draw equals K
successive (Ns/2, D) draws and leaves the generator in the same state."""
import gpyreg
import numpy as np

import pyvbmc

print("pyvbmc:", pyvbmc.__file__)
print("gpyreg:", gpyreg.__file__)
for K, n, D in [(1, 5, 1), (3, 20, 4), (50, 679, 10)]:
    a = np.random.default_rng(8)
    b = np.random.default_rng(8)
    one = a.standard_normal((K, n, D))
    many = np.stack([b.standard_normal((n, D)) for _ in range(K)])
    print(
        K,
        n,
        D,
        "equal:",
        np.array_equal(one, many),
        "same state:",
        a.bit_generator.state == b.bit_generator.state,
    )
