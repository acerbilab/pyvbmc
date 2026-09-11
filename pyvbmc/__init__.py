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
    # S-VBMC needs the optional torch extra; resolve it only on request so
    # that importing pyvbmc never imports torch.
    if name == "SVBMC":
        from pyvbmc.svbmc import SVBMC

        return SVBMC
    raise AttributeError(f"module 'pyvbmc' has no attribute {name!r}")
