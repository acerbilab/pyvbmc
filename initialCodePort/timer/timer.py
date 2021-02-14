import time


class Timer(object):
    """
    Timer class
    """

    def __init__(self):
        """
        __init__
        """
        self._start_times = dict()
        self._durations = dict()

    def start_timer(self, name: str):
        """
        start_timer stop the specified timer

        Parameters
        ----------
        name : str
            the name of the timer
        """
        if name not in self._start_times:
            self._start_times[name] = time.time()

    def stop_timer(self, name: str):
        """
        stop_timer stop the specified timer

        Parameters
        ----------
        name : str
            the name of the timer
        """

        if name in self._start_times:
            end_time = time.time()
            if name in self._durations:
                self._durations[name] = (
                    self._durations[name] + end_time - self._start_times[name]
                )
            else:
                self._durations[name] = end_time - self._start_times[name]
            self._start_times.pop(name)

    def get_duration(self, name: str):
        """
        get_duration return the duration of the timer

        Parameters
        ----------
        name : str
            the name of the timer

        Returns
        -------
        float
            the duration of the timer
            or None when not existing
        """
        return self._durations.get(name)
