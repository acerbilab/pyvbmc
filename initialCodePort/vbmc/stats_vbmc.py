class Stats(object):
    """
    Stats of VBMC algorithm
    """

    def __init__(self):
        self.cachecount = list()
        self.elbo = list()
        self.elbo_sd = list()
        self.funccount = list()
        self.gp = list()
        self.gpHypFull = list()
        self.gpNoise_hpd = list()
        self.gpNsamples = list()
        self.gpSampleVar = list()
        self.iter = list()
        self.lcbmax = list()
        self.N = list()
        self.Neff = list()
        self.outwarp_threshold = list()
        self.pruned = list()
        self.rindex = list()
        self.sKL = list()
        self.sKL_true = list()
        self.t = list()
        self.timer = list()
        self.vp = list()
        self.vpK = list()
        self.warmup = list()

    def save_iteration(self, iteration):
        """
        Save the stats for the given iteration
        """
        self.iter = iteration