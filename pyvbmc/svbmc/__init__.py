"""Stacking VBMC (S-VBMC): combine the posteriors of several VBMC runs.

``SVBMC`` takes finished :class:`~pyvbmc.VariationalPosterior` objects
from independent VBMC runs on the same problem and optimizes the weights
of their pooled mixture components against the stacked ELBO. It needs the
optional ``pyvbmc[torch]`` extra; importing this package does not import
torch.
"""

from pyvbmc.svbmc import utils
from pyvbmc.svbmc.svbmc import SVBMC

__all__ = ["SVBMC", "utils"]
