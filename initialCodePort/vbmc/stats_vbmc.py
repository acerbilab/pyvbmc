from .optimstate_vbmc import OptimState
from variational_posterior import VP


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
        self.sKL_true = None
        self.t = list()
        self.timer = list()
        self.vp = list()
        self.vpK = list()
        self.warmup = list()

    def save_iteration(
        self,
        optimState: OptimState,
        vp: VP,
        elbo,
        elbo_sd,
        varss,
        sKL,
        sKL_true,
        gp,
        hypstructfull,
        Ns_gp,
        pruned,
        timer,
        optionsDiagnostics,
    ):
        """
        Save the stats for the given iteration
        """
        iteration = optimState.iter
        self.iter[iteration] = iteration
        self.N[iteration] = optimState.N
        self.Neff[iteration] = optimState.Neff
        self.funccount[iteration] = optimState.funccount
        self.cachecount[iteration] = optimState.cachecount

        self.vpK[iteration] = vp.K
        self.warmup[iteration] = optimState.Warmup
        self.pruned[iteration] = pruned
        self.elbo[iteration] = elbo
        self.elbo_sd[iteration] = elbo_sd
        self.sKL[iteration] = sKL
        if sKL_true is not None:
            self.sKL_true = sKL_true
        self.gpNoise_hpd[iteration] = sqrt(optimState.sn2hpd)
        self.gpSampleVar[iteration] = varss
        self.gpNsamples[iteration] = Ns_gp
        self.gpHypFull[iteration] = hyp_full
        self.timer[iteration] = timer
        self.vp[iteration] = vp
        self.gp[iteration] = gplite_clean(gp)

        if optimState.gpOutwarpfun is not None:
            self.outwarp_threshold[iteration] = optimState.OutwarpDelta
        else:
            self.outwarp_threshold[iteration] = NaN
        self.lcbmax[iteration] = optimState.lcbmax
        self.t[iteration] = NaN    #Fill it at the end of the iteration
