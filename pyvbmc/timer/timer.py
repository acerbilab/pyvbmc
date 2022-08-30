import logging
import time


class Timer:
    """
    A small Timer class used in the context of VBMC.
    """

    def __init__(self):
        """
        Initialize a new timer.
        """
        self._start_times = dict()
        self._durations = dict()

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
                f"Timer not found for key '{name}'."
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
        return self._durations.get(
            name,
            logging.getLogger("timer").warning(
                f"Timer not found for key '{name}'."
            ),
        )

    def reset(self):
        """
        Reset the timer be emptying the durations and start times.
        """
        self._durations = dict()
        self._start_times = dict()
