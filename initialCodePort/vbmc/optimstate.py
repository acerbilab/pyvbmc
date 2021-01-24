class OptimState(object):
    """
    Contains the current state of the VMBC algorithm
    """

    def __init__(self):
        self.acqInfo = None

        self.Cache.active = None
        self.cachecount = None

        self.entropy_alpha = None
        self.funccount = None
        self.gpMeanfun = None
        self.gpOutwarpfun = None
        self.H = None
        self.hedge = None
        self.hypstruct = None
        self.iter = None
        self.LastRunAvg = None

        self.lcbmax = None
        self.lcbmax_vec = None

        self.N = None
        self.Neff = None
        self.nevals = None
        self.OutwarpDelta = None
        self.R = None
        self.RecomputeVarPost = None
        self.RunCov = None
        self.RunMean = None
        self.sn2hpd = None
        self.StopSampling = None

        self.totalfunevaltime = None
        self.totaltime = None

        self.UncertaintyHandlingLevel = None
        self.vpK = None
        self.vp_repo = None
        self.Warmup = None
        self.Xn = None
        self.X_flag = None
        self.ymax = None
        self.y_orig = None
