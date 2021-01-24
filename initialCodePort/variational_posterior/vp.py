class VP(object):
    """
    Variational Posterior class
    """

    def __init__(self):
        self.K = None
        self.D = None
        self.optimize_mu = None
        self.optimize_weights = None
        self.mu = None
        self.mode = None
        self.gp = None
        self.w = None
        self.sigma = None
        self.Lambda = None
        # vp.stats.entropy, vp.stats.stable
        # vp.stats.elbo,vp.stats.elbo_sd
        # vp_fields = {"elbo", "elbo_sd", "G", "H", "varG", "varH"}
        self.stats = dict()
        # vp.trinfo.lb_orig
        # vp.trinfo.ub_orig
        self.trinfo = None
        pass

    def vbmc_moments(self, origflag, Ns) -> (mu, cov):
        """
        VBMC_MOMENTS(VP) computes the mean MU and covariance
        """
        pass

    def vbmc_rnd(self, N, origflag, balanceflag, df):
        """
        Random samples from VBMC posterior approximation.
        """
        pass

    def vbmc_kldiv(self, vp2, Ns, gaussflag):
        """
        compute Kullback-Leibler divergence
        """
        pass

    def vbmc_pdf(self, X, origflag, logflag, transflag, df):
        """
        Probability density function of VBMC posterior approximation.
        """
        pass

    def vbmc_mode(self, nmax, origflag):
        """
        Find mode of VBMC posterior approximation.
        """
        pass

    def vbmc_mtv(self, vp2, Ns):
        """
        Marginal Total Variation distances between two variational posteriors.
        """
        pass

    def vbmc_power(self, n, cutoff):
        """
        Compute power posterior of variational approximation.
        """
        pass

    def vbmc_plot(self, vp_array, stats):
        """
        docstring
        """
        pass

    # private methods of vp class

    def __get_vptheta(
        self, optimize_mu, optimize_sigma, optimize_lambda, optimize_weights
    ):
        """
        Get vector of variational parameters from variational posterior.
        """
        pass

    def __robustSampleFromVP(self, Ns, Xrnd, quantile_thresh):
        """
        Robust sample from variational posterior
        """
        pass

    def __vbmc_output(self, parameter_list):
        """
        create output struct -> essentially a print?
        """
        pass

    def __vbmc_iterplot(self, parameter_list):
        """
        plot vp
        """
        pass