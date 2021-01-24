class Stats(object):
    """
    Stats of VBMC algorithm
    """

    def __init__(self):
        self.cachecount = None
        self.elbo = None
        self.elbo_sd = None
        self.funccount = None
        self.gp = None
        self.gpHypFull = None
        self.gpNoise_hpd = None
        self.gpNsamples = None
        self.gpSampleVar = None
        self.iter = None
        self.lcbmax = None
        self.N = None
        self.Neff = None
        self.outwarp_threshold = None
        self.pruned = None
        self.rindex = None
        self.sKL = None
        self.sKL_true = None
        self.t = None
        self.timer = None
        self.vp = None
        self.vpK = None
        self.warmup = None

    def save_iteration(self, iteration):
        """
        Save the stats for the given iteration
        """
        self.iter = iteration