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
        self._start_times[name] = 0

    def stop_timer(self, name: str):
        """
        stop_timer stop the specified timer

        Parameters
        ----------
        name : str
            the name of the timer
        """

        if name in self._start_times:
            if name in self._durations:
                self._durations[name] = (
                    self._durations[name] + 0 - self._start_times[name]
                )
            else:
                self._durations[name] = 0 - self._start_times[name]

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
