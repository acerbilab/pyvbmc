"""Random number generation helpers shared across PyVBMC."""

import numpy as np


def get_rng(seed=None) -> np.random.Generator:
    """Return a NumPy random ``Generator``.

    Parameters
    ----------
    seed : None, int, array_like[int], SeedSequence, BitGenerator or \
Generator, optional
        Anything accepted by ``numpy.random.default_rng``. A ``Generator`` is
        returned unchanged, so a generator can be shared between objects. If
        ``None`` (default), a new ``Generator`` is seeded from NumPy's global
        legacy random state, so that a preceding ``np.random.seed(...)`` still
        makes the result reproducible.

    Returns
    -------
    rng : np.random.Generator
    """
    if seed is None:
        seed = np.random.randint(0, 2**32, size=4, dtype=np.uint32)
    return np.random.default_rng(seed)
