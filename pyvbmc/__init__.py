# __init__.py
import pyvbmc.acquisition_functions
import pyvbmc.decorators
import pyvbmc.entropy
import pyvbmc.function_logger
import pyvbmc.parameter_transformer
import pyvbmc.stats
import pyvbmc.timer
import pyvbmc.variational_posterior
import pyvbmc.vbmc
from pyvbmc.calibration import CalibrationProfile, calibrate
from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.vbmc import VBMC


def __getattr__(name):
    # Optional integrations resolve only on request so importing pyvbmc does
    # not import their third-party dependencies.
    if name == "SVBMC":
        from pyvbmc.svbmc import SVBMC

        return SVBMC
    if name == "PyMCTarget":
        from pyvbmc.pymc import PyMCTarget

        return PyMCTarget
    raise AttributeError(f"module 'pyvbmc' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | {"PyMCTarget", "SVBMC"})
