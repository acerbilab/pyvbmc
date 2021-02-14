class Timer(object):
    """
    Timer class
    """

    def __init__(self):
        self.start_times = dict()
        self.durations = dict()

    def start_timer(self, name):
        self.start_times[name] = 0

    def stop_timer(self, name):
        if name in start_times:
            self.durations[name] = (
                self.durations[name] + 0 - self.start_times[name]
            )

    def get_duration(self, name):
        return self.durations.get(name)
