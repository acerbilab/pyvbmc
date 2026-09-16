"""Optional PyMC integration for PyVBMC.

Install ``pyvbmc[pymc]`` to construct :class:`PyMCTarget` objects. Importing
this package does not import PyMC or PyTensor; the optional dependencies are
loaded only when a target is constructed.
"""

__all__ = ["PyMCTarget", "UnsupportedModel"]


def __getattr__(name):
    if name == "PyMCTarget":
        from ._target import PyMCTarget

        return PyMCTarget
    if name == "UnsupportedModel":
        from ._compat import UnsupportedModel

        return UnsupportedModel
    raise AttributeError(f"module 'pyvbmc.pymc' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
