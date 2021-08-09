from pyvbmc.acquisition_functions import AbstractAcquisitionFunction


def test_acq_info():
    class BasicAcqClass(AbstractAcquisitionFunction):
        def _compute_acquisition_function(
            self, Xs, vp, gp, optimState, fmu, fs2, fbar, vtot
        ):
            pass

    acq_fcn = BasicAcqClass()
    assert isinstance(acq_fcn.acq_info, dict)
    assert isinstance(acq_fcn.get_info(), dict)
