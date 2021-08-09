from abc import ABC


class AbstractAcquisitionFunction(ABC):
    def __init__(self):
        self.acq_info = dict()

    def get_info(self):
        """
        Return a dict with information about the acquisition function.

        Returns
        -------
        acq_info : dict
            A dict containing information about the acquisition function.
        """
        return self.acq_info
