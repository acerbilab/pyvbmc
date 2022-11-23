import logging
import time

from pyvbmc.formatting import full_repr


class Timer:
    """
    A small Timer class used in the context of VBMC.
    """

    def __init__(self):
        """
        Initialize a new timer.
        """
        self._start_times = {}
        self._durations = {}

    def start_timer(self, name: str):
        """
        Start the specified timer.

        Parameters
        ----------
        name : str
            The name of the timer that should be started.
        """
        if name not in self._start_times:
            self._start_times[name] = time.time()

    def stop_timer(self, name: str):
        """
        Stop the specified timer.

        If this is the first call to ``stop_timer(name)`` for string ``name``,
        record the duration. Otherwise, add to the duration.

        Parameters
        ----------
        name : str
            The name of the timer that should be started.
        """

        if name in self._start_times:
            end_time = time.time()
            if self._durations.get(name) is not None:
                self._durations[name] += end_time - self._start_times[name]
            else:
                self._durations[name] = end_time - self._start_times[name]
            self._start_times.pop(name)
        else:
            logging.getLogger("timer").warning(
                "Timer start not found for key '%s'.", name
            )

    def get_duration(self, name: str):
        """
        Return the duration of the specified timer.

        Parameters
        ----------
        name : str
            The name of the timer which time should be returned.

        Returns
        -------
        duration : float
            The duration of the timer or None when the timer is not existing.
        """
        time_ = self._durations.get(name)
        if time_ is None:
            logging.getLogger("timer").warning(
                "Timer duration not found for key '%s'.", name
            )
        return time_

    def reset(self):
        """
        Reset the timer be emptying the durations and start times.
        """
        self._durations = {}
        self._start_times = {}

    def __repr__(self, arr_size_thresh=10, expand=True):
        """Construct a detailed string summary.

        Parameters
        ----------
        arr_size_thresh : float, optional
            If ``obj`` is an array whose product of dimensions is less than
            ``arr_size_thresh``, print the full array. Otherwise print only the
            shape. Default `10`.
        expand : bool, optional
            If ``expand`` is `False`, then describe any complex child
            attributes of the object by their name and memory location.
            Otherwise, recursively expand the child attributes into their own
            representations. Default `False`.

        Returns
        -------
        string : str
            The string representation of ``self``.
        """
        return full_repr(
            self, "Timer", expand=expand, arr_size_thresh=arr_size_thresh
        )

    def _short_repr(self):
        """Returns abbreviated string representation with memory location.

        Returns
        -------
        string : str
            The abbreviated string representation of the VBMC object.
        """
        return object.__repr__(self)
