class Timer(object):
    """
    Timer of VBMC algorithm
    """

    def __init__(self):
        self.finalize = 0
        self.variationalFit = 0
        self.gpTrain = 0

    
    @property
    def totalruntime(self):
        return self.finalize + self.variationalFit + self.gpTrain
