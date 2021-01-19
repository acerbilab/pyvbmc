from typing import Any

class vp(object):
    """
    Variational Posterior class
    """
    
    

    def __init__(self):
        pass

    def vbmc_moments(self, origflag, Ns) -> (mu, cov):
        """
        VBMC_MOMENTS(VP) computes the mean MU and covariance
        """
        pass

    def vbmc_rnd(self, N, origflag, balanceflag,df):
        """
        Random samples from VBMC posterior approximation.
        """
        pass

    def vbmc_kldiv(self,vp2,Ns,gaussflag):
        """
        compute Kullback-Leibler divergence
        """
        pass

    def vbmc_pdf(self, X,origflag,logflag,transflag,df):
        """
        Probability density function of VBMC posterior approximation.
        """
        pass

    def vbmc_mode(self,nmax,origflag):
        """
        Find mode of VBMC posterior approximation.
        """
        pass

    def vbmc_mtv(self,vp2,Ns):
        """
        Marginal Total Variation distances between two variational posteriors.
        """
        pass

    def vbmc_power(self, n,cutoff):
        """
        Compute power posterior of variational approximation.
        """
        pass

    def vbmc_plot(self, vp_array,stats):
        """
        docstring
        """
        pass

    #VP entropy methods

    def entlb_vbmc(self, grad_flags,jacobian_flag):
        """
        ENTLB_VBMC Entropy lower bound for variational posterior
        """
        pass

    def entub_vbmc(self, grad_flags,jacobian_flag):
        """
        ENTUB_VBMC Entropy upper bound for variational posterior
        """
        pass

    def entmc_vbmc(self, Ns,grad_flags,jacobian_flag):
        """
        ENTMC_VBMC Monte Carlo estimate of entropy of variational posterior
        """
        pass

    #private methods of vp class

    def __get_vptheta(self, optimize_mu,optimize_sigma,optimize_lambda,optimize_weights):
        """
        Get vector of variational parameters from variational posterior.
        """
        pass

    def __robustSampleFromVP(self, Ns,Xrnd,quantile_thresh):
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