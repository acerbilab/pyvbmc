# An Exploration of Acquisition and Mean Functions in Variational Bayesian Monte Carlo - Appendix

---

#### Page 10

## Appendix A. Expected Log Joint via Bayesian Quadrature

An interesting question is which covariance and mean functions afford an analytical computation of the expected log joint in Equation (1). For a given variational posterior $q_{\boldsymbol{\phi}}$ represented by a Gaussian mixture model, as per Section 2, the expected log joint is

$$
\mathbb{E}_{\boldsymbol{\phi}}[f(\boldsymbol{x})]=\sum_{k=1}^{K} w_{k} \int \mathcal{N}\left(\boldsymbol{x} ; \boldsymbol{\mu}_{k}, \sigma_{k}^{2} \boldsymbol{\Sigma}\right) f(\boldsymbol{x}) d \boldsymbol{x} \equiv \sum_{k=1}^{K} w_{k} \mathcal{I}_{k}
$$

We recall that the posterior predictive mean of a GP $f$, given training data $\boldsymbol{\Xi}=\{\mathbf{X}, \boldsymbol{y}\}$, where $\mathbf{X}$ are $n$ training inputs with observed values $\boldsymbol{y}$, is (Rasmussen and Williams, 2006)

$$
\bar{f}(\boldsymbol{x})=\kappa(\boldsymbol{x}, \mathbf{X})\left[\kappa(\mathbf{X}, \mathbf{X})+\sigma_{\mathrm{obs}}^{2} \mathbf{I}_{n}\right]^{-1}(\boldsymbol{y}-m(\mathbf{X}))+m(\boldsymbol{x})
$$

where $\kappa(\cdot, \cdot)$ and $m(\cdot)$ are, respectively, the GP covariance and mean functions. Thus, for each integral in Eq. S1, we have in expectation over the GP posterior (Acerbi, 2018)

$$
\begin{aligned}
\mathbb{E}_{f \mid \boldsymbol{\Xi}}\left[\mathcal{I}_{k}\right]= & \int \mathcal{N}\left(\boldsymbol{x} ; \boldsymbol{\mu}_{k}, \sigma_{k}^{2} \boldsymbol{\Sigma}\right) \bar{f}(\boldsymbol{x}) d \boldsymbol{x} \\
= & {\left[\sigma_{f}^{2} \int \mathcal{N}\left(\boldsymbol{x} ; \boldsymbol{\mu}_{k}, \sigma_{k}^{2} \boldsymbol{\Sigma}\right) \kappa(\boldsymbol{x}, \mathbf{X}) d \boldsymbol{x}\right]\left[\kappa(\mathbf{X}, \mathbf{X})+\sigma_{\mathrm{obs}}^{2} \mathbf{I}\right]^{-1}(\boldsymbol{y}-m(\mathbf{X})) } \\
& +\sigma_{f}^{2} \int \mathcal{N}\left(\boldsymbol{x} ; \boldsymbol{\mu}_{k}, \sigma_{k}^{2} \boldsymbol{\Sigma}\right) m(\boldsymbol{x}) d \boldsymbol{x}
\end{aligned}
$$

From Equation (S3), we see that functional forms for the covariance and mean that would afford an analytical calculation of the integrals are Gaussian, exponential, polynomial, and products and linear combinations of such elementary forms (Ghahramani and Rasmussen, 2002). We are not aware of other general functional forms that could be meaningfully used in this context.

## Appendix B. Reduced Formulation of Generalized Acquisition Function

We show here that the generalized acquisition function described by Equation (2) can be reduced from three to two parameters with virtually no loss of generality.

First, the location of the optimum of a function is invariant to monotonic ${ }^{3}$ transformations of the output, and moreover in VBMC we optimize the acquisition function using CMA-ES (Hansen et al., 2003), which only uses the ranking of the objective function — making it invariant to monotonic transformation of the objective. Thus, we can apply a monotonic transformation to the acquisition function with absolutely no change to the entire optimization process. Second, we assume that for any "uncertainty sampling" acquisition function we want to keep dependence on the GP posterior predictive variance, that is $\alpha>0$.

With these considerations, we can rewrite Equation (2) as

$$
\log a_{\mathrm{gus}}(\boldsymbol{x}) \propto \log V_{\boldsymbol{\Xi}}(\boldsymbol{x})+\widetilde{\beta} \log q_{\boldsymbol{\phi}}(\boldsymbol{x})+\widetilde{\gamma} \bar{f}_{\boldsymbol{\Xi}}(\boldsymbol{x}), \quad \text { with } \widetilde{\beta}=\frac{\beta}{\alpha}, \widetilde{\gamma}=\frac{\gamma}{\alpha}
$$

which only depends on two parameters, and the logarithmic form is numerically convenient to avoid overflows.

[^3]
[^3]:    3. In all this paragraph, we mean monotonic with positive derivative.