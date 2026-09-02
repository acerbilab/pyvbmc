# Variational Bayesian Monte Carlo - Appendix

---

#### Page 12

# Supplementary Material

In this Supplement we include a number of derivations, implementation details, and additional results omitted from the main text.

Code used to generate the results in the paper is available at https://github.com/lacerbi/infbench. The VBMC algorithm is available at https://github.com/acerbilab/vbmc.

## Contents

A Computing and optimizing the ELBO ..... 13
A.1 Stochastic approximation of the entropy ..... 13
A.1.1 Gradient of the entropy ..... 13
A.2 Expected log joint ..... 14
A.2.1 Posterior mean of the integral and its gradient ..... 15
A.2.2 Posterior variance of the integral ..... 15
A.2.3 Negative quadratic mean function ..... 16
A.3 Optimization of the approximate ELBO ..... 16
A.3.1 Reparameterization ..... 16
A.3.2 Choice of starting points ..... 16
A.3.3 Stochastic gradient descent ..... 16
B Algorithmic details ..... 17
B.1 Regularization of acquisition functions ..... 17
B.2 GP hyperparameters and priors ..... 17
B.3 Transformation of variables ..... 17
B.4 Termination criteria ..... 18
B.4.1 Reliability index ..... 18
B.4.2 Long-term stability termination condition ..... 18
B.4.3 Validation of VBMC solutions ..... 19
C Experimental details and additional results ..... 19
C.1 Synthetic likelihoods ..... 19
C.2 Neuronal model ..... 20
C.2.1 Model parameters ..... 20
C.2.2 True and approximate posteriors ..... 20
D Analysis of VBMC ..... 23
D.1 Variability between VBMC runs ..... 23
D.2 Computational cost ..... 23
D.3 Analysis of the samples produced by VBMC ..... 24

---

#### Page 13

## A Computing and optimizing the ELBO

For ease of reference, we recall the expression for the ELBO, for $\boldsymbol{x} \in \mathbb{R}^{D}$,

$$
\mathcal{F}\left[q_{\boldsymbol{\phi}}\right]=\mathbb{E}_{\boldsymbol{\phi}}\left[\log \frac{p(\mathcal{D} \mid \boldsymbol{x}) p(\boldsymbol{x})}{q_{\boldsymbol{\phi}}(\boldsymbol{x})}\right]=\mathbb{E}_{\boldsymbol{\phi}}[f(\boldsymbol{x})]+\mathcal{H}\left[q_{\boldsymbol{\phi}}(\boldsymbol{x})\right]
$$

with $\mathbb{E}_{\boldsymbol{\phi}} \equiv \mathbb{E}_{q_{\boldsymbol{\phi}}}$, and of the variational posterior,

$$
q(\boldsymbol{x}) \equiv q_{\boldsymbol{\phi}}(\boldsymbol{x})=\sum_{k=1}^{K} w_{k} \mathcal{N}\left(\boldsymbol{x} ; \boldsymbol{\mu}_{k}, \sigma_{k}^{2} \boldsymbol{\Sigma}\right)
$$

where $w_{k}, \boldsymbol{\mu}_{k}$, and $\sigma_{k}$ are, respectively, the mixture weight, mean, and scale of the $k$-th component, and $\boldsymbol{\Sigma} \equiv \operatorname{diag}\left[{\lambda^{(1)}}^{2}, \ldots, {\lambda^{(D)}}^{2}\right]$ is a diagonal covariance matrix common to all elements of the mixture. The variational posterior for a given number of mixture components $K$ is parameterized by $\boldsymbol{\phi} \equiv\left(w_{1}, \ldots, w_{K}, \boldsymbol{\mu}_{1}, \ldots, \boldsymbol{\mu}_{K}, \sigma_{1}, \ldots, \sigma_{K}, \boldsymbol{\lambda}\right)$.
In the following paragraphs we derive expressions for the ELBO and for its gradient. Then, we explain how we optimize it with respect to the variational parameters.

### A.1 Stochastic approximation of the entropy

We approximate the entropy of the variational distribution via simple Monte Carlo sampling as follows. Let $\mathbf{R}=\operatorname{diag}[\boldsymbol{\lambda}]$ and $N_{\mathrm{s}}$ be the number of samples per mixture component. We have

$$
\begin{aligned}
\mathcal{H}[q(\boldsymbol{x})] & =-\int q(\boldsymbol{x}) \log q(\boldsymbol{x}) d \boldsymbol{x} \\
& \approx-\frac{1}{N_{\mathrm{s}}} \sum_{s=1}^{N_{\mathrm{s}}} \sum_{k=1}^{K} w_{k} \log q\left(\sigma_{k} \mathbf{R} \boldsymbol{\varepsilon}_{s, k}+\boldsymbol{\mu}_{k}\right) \quad \text { with } \quad \boldsymbol{\varepsilon}_{s, k} \sim \mathcal{N}\left(\mathbf{0}, \mathbb{I}_{D}\right) \\
& =-\frac{1}{N_{\mathrm{s}}} \sum_{s=1}^{N_{\mathrm{s}}} \sum_{k=1}^{K} w_{k} \log q\left(\boldsymbol{\xi}_{s, k}\right) \quad \text { with } \quad \boldsymbol{\xi}_{s, k} \equiv \sigma_{k} \mathbf{R} \boldsymbol{\varepsilon}_{s, k}+\boldsymbol{\mu}_{k}
\end{aligned}
$$

where we used the reparameterization trick separately for each component [18, 19]. For VBMC, we set $N_{\mathrm{s}}=100$ during the variational optimization, and $N_{\mathrm{s}}=2^{15}$ for evaluating the ELBO with high precision at the end of each iteration.

#### A.1.1 Gradient of the entropy

The derivative of the entropy with respect to a variational parameter $\phi \in\{\mu, \sigma, \lambda\}$ (that is, not a mixture weight) is

$$
\begin{aligned}
\frac{d}{d \phi} \mathcal{H}[q(\boldsymbol{x})] & \approx-\frac{1}{N_{\mathrm{s}}} \sum_{s=1}^{N_{\mathrm{s}}} \sum_{k=1}^{K} w_{k} \frac{d}{d \phi} \log q\left(\boldsymbol{\xi}_{s, k}\right) \\
& =-\frac{1}{N_{\mathrm{s}}} \sum_{s=1}^{N_{\mathrm{s}}} \sum_{k=1}^{K} w_{k}\left(\frac{\partial}{\partial \phi}+\sum_{i=1}^{D} \frac{d \xi_{s, k}^{(i)}}{d \phi} \frac{\partial}{\partial \xi_{s, k}^{(i)}}\right) \log q\left(\boldsymbol{\xi}_{s, k}\right) \\
& =-\frac{1}{N_{\mathrm{s}}} \sum_{s=1}^{N_{\mathrm{s}}} \sum_{k=1}^{K} \frac{w_{k}}{q\left(\boldsymbol{\xi}_{s, k}\right)} \sum_{i=1}^{D} \frac{d \xi_{s, k}^{(i)}}{d \phi} \frac{\partial}{\partial \xi_{s, k}^{(i)}} \sum_{l=1}^{K} w_{l} \mathcal{N}\left(\boldsymbol{\xi}_{s, k} ; \boldsymbol{\mu}_{l}, \sigma_{l}^{2} \boldsymbol{\Sigma}\right) \\
& =\frac{1}{N_{\mathrm{s}}} \sum_{s=1}^{N_{\mathrm{s}}} \sum_{k=1}^{K} \frac{w_{k}}{q\left(\boldsymbol{\xi}_{s, k}\right)} \sum_{i=1}^{D} \frac{d \xi_{s, k}^{(i)}}{d \phi} \sum_{l=1}^{K} w_{l} \frac{\xi_{s, k}^{(i)}-\mu_{l}^{(i)}}{\left(\sigma_{k} \lambda^{(i)}\right)^{2}} \mathcal{N}\left(\boldsymbol{\xi}_{s, k} ; \boldsymbol{\mu}_{l}, \sigma_{l}^{2} \boldsymbol{\Sigma}\right)
\end{aligned}
$$

where from the second to the third row we used the fact that the expected value of the score is zero, $\mathbb{E}_{q(\boldsymbol{\xi})}\left[\frac{\partial}{\partial \phi} \log q(\boldsymbol{\xi})\right]=0$.

---

#### Page 14

In particular, for $\phi=\mu_{j}^{(m)}$, with $1 \leq m \leq D$ and $1 \leq j \leq K$,

$$
\begin{aligned}
\frac{d}{d \mu_{j}^{(m)}} \mathcal{H}[q(\boldsymbol{x})] & \approx-\frac{1}{N_{\mathrm{s}}} \sum_{s=1}^{N_{\mathrm{s}}} \sum_{k=1}^{K} \frac{w_{k}}{q\left(\boldsymbol{\xi}_{s, k}\right)} \sum_{i=1}^{D} \frac{d \xi_{s, k}^{(i)}}{d \mu_{j}^{(m)}} \frac{\partial}{\partial \xi_{s, k}^{(i)}} \sum_{l=1}^{K} w_{l} \mathcal{N}\left(\boldsymbol{\xi}_{s, k} ; \boldsymbol{\mu}_{l}, \sigma_{l}^{2} \boldsymbol{\Sigma}\right) \\
& =\frac{w_{j}}{N_{\mathrm{s}}} \sum_{s=1}^{N_{\mathrm{s}}} \frac{1}{q\left(\boldsymbol{\xi}_{s, j}\right)} \sum_{l=1}^{K} w_{l} \frac{\xi_{s, j}^{(m)}-\mu_{l}^{(m)}}{\left(\sigma_{l} \lambda^{(m)}\right)^{2}} \mathcal{N}\left(\boldsymbol{\xi}_{s, j} ; \boldsymbol{\mu}_{l}, \sigma_{l}^{2} \boldsymbol{\Sigma}\right)
\end{aligned}
$$

where we used that fact that $\frac{d \xi_{s, k}^{(i)}}{d \mu_{j}^{(m)}}=\delta_{i m} \delta_{j k}$.
For $\phi=\sigma_{j}$, with $1 \leq j \leq K$,

$$
\begin{aligned}
\frac{d}{d \sigma_{j}} \mathcal{H}[q(\boldsymbol{x})] & \approx-\frac{1}{N_{\mathrm{s}}} \sum_{s=1}^{N_{\mathrm{s}}} \sum_{k=1}^{K} \frac{w_{k}}{q\left(\boldsymbol{\xi}_{s, k}\right)} \sum_{i=1}^{D} \frac{d \xi_{s, k}^{(i)}}{d \sigma_{j}} \frac{\partial}{\partial \xi_{s, k}^{(i)}} \sum_{l=1}^{K} w_{l} \mathcal{N}\left(\boldsymbol{\xi}_{s, k} ; \boldsymbol{\mu}_{l}, \sigma_{l}^{2} \boldsymbol{\Sigma}\right) \\
& =\frac{w_{j}}{K^{2} N_{\mathrm{s}}} \sum_{s=1}^{N_{\mathrm{s}}} \frac{1}{q\left(\boldsymbol{\xi}_{s, j}\right)} \sum_{i=1}^{D} \lambda^{(i)} \varepsilon_{s, j}^{(i)} \sum_{l=1}^{K} w_{l} \frac{\xi_{s, j}^{(i)}-\mu_{l}^{(i)}}{\left(\sigma_{l} \lambda^{(i)}\right)^{2}} \mathcal{N}\left(\boldsymbol{\xi}_{s, j} ; \boldsymbol{\mu}_{l}, \sigma_{l}^{2} \boldsymbol{\Sigma}\right)
\end{aligned}
$$

where we used that fact that $\frac{d \xi_{s, k}^{(i)}}{d \sigma_{j}}=\lambda^{(i)} \varepsilon_{s, j}^{(i)} \delta_{j k}$.
For $\phi=\lambda^{(m)}$, with $1 \leq m \leq D$,

$$
\begin{aligned}
\frac{d}{d \lambda^{(m)}} \mathcal{H}[q(\boldsymbol{x})] & \approx-\frac{1}{N_{\mathrm{s}}} \sum_{s=1}^{N_{\mathrm{s}}} \sum_{k=1}^{K} \frac{w_{k}}{q\left(\boldsymbol{\xi}_{s, k}\right)} \sum_{i=1}^{D} \frac{d \xi_{s, k}^{(i)}}{d \lambda^{(m)}} \frac{\partial}{\partial \xi_{s, k}^{(i)}} \sum_{l=1}^{K} w_{l} \mathcal{N}\left(\boldsymbol{\xi}_{s, k} ; \boldsymbol{\mu}_{l}, \sigma_{l}^{2} \boldsymbol{\Sigma}\right) \\
& =\frac{1}{N_{\mathrm{s}}} \sum_{s=1}^{N_{\mathrm{s}}} \sum_{k=1}^{K} \frac{w_{k} \sigma_{k} \varepsilon_{s, k}^{(m)}}{q\left(\boldsymbol{\xi}_{s, k}\right)} \sum_{l=1}^{K} w_{l} \frac{\xi_{s, k}^{(m)}-\mu_{l}^{(m)}}{\left(\sigma_{l} \lambda^{(m)}\right)^{2}} \mathcal{N}\left(\boldsymbol{\xi}_{s, k} ; \boldsymbol{\mu}_{l}, \sigma_{l}^{2} \boldsymbol{\Sigma}\right)
\end{aligned}
$$

where we used that fact that $\frac{d \xi_{s, k}^{(i)}}{d \lambda^{(m)}}=\sigma_{k} \varepsilon_{s, k}^{(i)} \delta_{i m}$.
Finally, the derivative with respect to variational mixture weight $w_{j}$, for $1 \leq j \leq K$, is

$$
\frac{\partial}{\partial w_{j}} \mathcal{H}[q(\boldsymbol{x})] \approx-\frac{1}{N_{\mathrm{s}}} \sum_{s=1}^{N_{\mathrm{s}}}\left[\log q\left(\boldsymbol{\xi}_{s, j}\right)+\sum_{k=1}^{K} \frac{w_{k}}{q\left(\boldsymbol{\xi}_{s, k}\right)} q_{j}\left(\boldsymbol{\xi}_{s, k}\right)\right]
$$

### A.2 Expected log joint

For the expected log joint we have

$$
\begin{aligned}
\mathcal{G}[q(\boldsymbol{x})]=\mathbb{E}_{\boldsymbol{\phi}}[f(\boldsymbol{x})] & =\sum_{k=1}^{K} w_{k} \int \mathcal{N}\left(\boldsymbol{x} ; \boldsymbol{\mu}_{k}, \sigma_{k}^{2} \boldsymbol{\Sigma}\right) f(\boldsymbol{x}) d \boldsymbol{x} \\
& =\sum_{k=1}^{K} w_{k} \mathcal{I}_{k}
\end{aligned}
$$

To solve the integrals in Eq. S9 we approximate $f(\boldsymbol{x})$ with a Gaussian process (GP) with a squared exponential (that is, rescaled Gaussian) covariance function,

$$
\mathbf{K}_{p q}=\kappa\left(\boldsymbol{x}_{p}, \boldsymbol{x}_{q}\right)=\sigma_{f}^{2} \Lambda \mathcal{N}\left(\boldsymbol{x}_{p} ; \boldsymbol{x}_{q}, \boldsymbol{\Sigma}_{\ell}\right) \quad \text { with } \boldsymbol{\Sigma}_{\ell}=\operatorname{diag}\left[{\ell^{(1)}}^{2}, \ldots, {\ell^{(D)}}^{2}\right]
$$

where $\Lambda \equiv(2 \pi)^{\frac{D}{2}} \prod_{i=1}^{D} \ell^{(i)}$ is equal to the normalization factor of the Gaussian. ${ }^{1}$ For the GP we also assume a Gaussian likelihood with observation noise variance $\sigma_{\text {obs }}^{2}$ and, for the sake of exposition, a constant mean function $m \in \mathbb{R}$. We will later consider the case of a negative quadratic mean function, as per the main text.

[^0]
[^0]: ${ }^{1}$ This choice of notation makes it easy to apply Gaussian identities used in Bayesian quadrature.

---

#### Page 15

#### A.2.1 Posterior mean of the integral and its gradient

The posterior predictive mean of the GP, given training data $\boldsymbol{\Xi}=\{\mathbf{X}, \boldsymbol{y}\}$, where $\mathbf{X}$ are $n$ training inputs with associated observed values $\boldsymbol{y}$, is

$$
\bar{f}(\boldsymbol{x})=\kappa(\boldsymbol{x}, \mathbf{X})\left[\kappa(\mathbf{X}, \mathbf{X})+\sigma_{\text {obs }}^{2} \mathbf{I}_{n}\right]^{-1}(\boldsymbol{y}-m)+m
$$

Thus, for each integral in Eq. S9 we have in expectation over the GP posterior

$$
\begin{aligned}
\mathbb{E}_{f \mid \boldsymbol{\Xi}}\left[\mathcal{I}_{k}\right] & =\int \mathcal{N}\left(\boldsymbol{x} ; \boldsymbol{\mu}_{k}, \sigma_{k}^{2} \boldsymbol{\Sigma}\right) \bar{f}(\boldsymbol{x}) d \boldsymbol{x} \\
& =\left[\sigma_{f}^{2} \int \mathcal{N}\left(\boldsymbol{x} ; \boldsymbol{\mu}_{k}, \sigma_{k}^{2} \boldsymbol{\Sigma}\right) \mathcal{N}\left(\boldsymbol{x} ; \mathbf{X}, \boldsymbol{\Sigma}_{\ell}\right) d \boldsymbol{x}\right]\left[\kappa(\mathbf{X}, \mathbf{X})+\sigma_{\text {obs }}^{2} \mathbf{I}\right]^{-1}(\boldsymbol{y}-m)+m \\
& =\boldsymbol{z}_{k}^{\top}\left[\kappa(\mathbf{X}, \mathbf{X})+\sigma_{\text {obs }}^{2} \mathbf{I}\right]^{-1}(\boldsymbol{y}-m)+m
\end{aligned}
$$

where $\boldsymbol{z}_{k}$ is a $n$-dimensional vector with entries $z_{k}^{(p)}=\sigma_{f}^{2} \mathcal{N}\left(\boldsymbol{\mu}_{k} ; \boldsymbol{x}_{p}, \sigma_{k}^{2} \boldsymbol{\Sigma}+\boldsymbol{\Sigma}_{\ell}\right)$ for $1 \leq p \leq n$. In particular, defining $\tau_{k}^{(i)} \equiv \sqrt{\sigma_{k}^{2} {\lambda^{(i)}}^{2}+{\ell^{(i)}}^{2}}$ for $1 \leq i \leq D$,

$$
z_{k}^{(p)}=\frac{\sigma_{f}^{2}}{(2 \pi)^{\frac{D}{2}} \prod_{i=1}^{D} \tau_{k}^{(i)}} \exp \left\{-\frac{1}{2} \sum_{i=1}^{D} \frac{\left(\mu_{k}^{(i)}-\boldsymbol{x}_{p}^{(i)}\right)^{2}}{{\tau_{k}^{(i)}}^{2}}\right\}
$$

We can compute derivatives with respect to the variational parameters $\phi \in(\mu, \sigma, \lambda)$ as

$$
\begin{aligned}
\frac{\partial}{\partial \mu_{j}^{(l)}} z_{k}^{(p)} & =\delta_{j k} \frac{\boldsymbol{x}_{p}^{(l)}-\mu_{k}^{(l)}}{{\tau_{k}^{(l)}}^{2}} z_{k}^{(p)} \\
\frac{\partial}{\partial \sigma_{j}} z_{k}^{(p)} & =\delta_{j k} \sum_{i=1}^{D} \frac{{\lambda^{(i)}}^{2}}{{\tau_{k}^{(i)}}^{2}}\left[\frac{\left(\mu_{k}^{(i)}-\boldsymbol{x}_{p}^{(i)}\right)^{2}}{{\tau_{k}^{(i)}}^{2}}-1\right] \sigma_{k} z_{k}^{(p)} \\
\frac{\partial}{\partial \lambda^{(l)}} z_{k}^{(p)} & =\frac{\sigma_{k}^{2}}{{\tau_{k}^{(l)}}^{2}}\left[\frac{\left(\mu_{k}^{(l)}-\boldsymbol{x}_{p}^{(l)}\right)^{2}}{{\tau_{k}^{(l)}}^{2}}-1\right] \lambda^{(l)} z_{k}^{(p)}
\end{aligned}
$$

The derivative of Eq. S9 with respect to mixture weight $w_{k}$ is simply $\mathcal{I}_{k}$.

#### A.2.2 Posterior variance of the integral

We compute the variance of Eq. S9 under the GP approximation as [8]

$$
\begin{aligned}
\operatorname{Var}_{f \mid \mathcal{X}}[\mathcal{G}] & =\int \int q(\boldsymbol{x}) q\left(\boldsymbol{x}^{\prime}\right) C_{\boldsymbol{\Xi}}\left(f(\boldsymbol{x}), f\left(\boldsymbol{x}^{\prime}\right)\right) d \boldsymbol{x} d \boldsymbol{x}^{\prime} \\
& =\sum_{j=1}^{K} \sum_{k=1}^{K} w_{j} w_{k} \int \int \mathcal{N}\left(\boldsymbol{x} ; \boldsymbol{\mu}_{j}, \sigma_{j}^{2} \boldsymbol{\Sigma}\right) \mathcal{N}\left(\boldsymbol{x}^{\prime} ; \boldsymbol{\mu}_{k}, \sigma_{k}^{2} \boldsymbol{\Sigma}\right) C_{\boldsymbol{\Xi}}\left(f(\boldsymbol{x}), f\left(\boldsymbol{x}^{\prime}\right)\right) d \boldsymbol{x} d \boldsymbol{x}^{\prime} \\
& =\sum_{j=1}^{K} \sum_{k=1}^{K} w_{j} w_{k} \mathcal{J}_{j k}
\end{aligned}
$$

where $C_{\boldsymbol{\Xi}}$ is the GP posterior predictive covariance,

$$
C_{\boldsymbol{\Xi}}\left(f(\boldsymbol{x}), f\left(\boldsymbol{x}^{\prime}\right)\right)=\kappa\left(\boldsymbol{x}, \boldsymbol{x}^{\prime}\right)-\kappa(\boldsymbol{x}, \mathbf{X})\left[\kappa(\mathbf{X}, \mathbf{X})+\sigma_{\text {obs }}^{2} \mathbf{I}_{n}\right]^{-1} \kappa\left(\mathbf{X}, \boldsymbol{x}^{\prime}\right)
$$

Thus, each term in Eq. S15 can be written as

$$
\begin{aligned}
\mathcal{J}_{j k}= & \int \int \mathcal{N}\left(\boldsymbol{x} ; \boldsymbol{\mu}_{j}, \sigma_{j}^{2} \boldsymbol{\Sigma}\right)\left[\sigma_{f}^{2} \mathcal{N}\left(\boldsymbol{x} ; \boldsymbol{x}^{\prime}, \boldsymbol{\Sigma}_{\ell}\right)-\sigma_{f}^{2} \mathcal{N}\left(\boldsymbol{x} ; \mathbf{X}, \boldsymbol{\Sigma}_{\ell}\right)\left[\kappa(\mathbf{X}, \mathbf{X})+\sigma_{\text {obs }}^{2} \mathbf{I}_{n}\right]^{-1} \sigma_{f}^{2} \mathcal{N}\left(\mathbf{X} ; \boldsymbol{x}^{\prime}, \boldsymbol{\Sigma}_{\ell}\right)\right] \times \\
& \times \mathcal{N}\left(\boldsymbol{x}^{\prime} ; \boldsymbol{\mu}_{k}, \sigma_{k}^{2} \boldsymbol{\Sigma}\right) d \boldsymbol{x} d \boldsymbol{x}^{\prime} \\
= & \sigma_{f}^{2} \mathcal{N}\left(\boldsymbol{\mu}_{j} ; \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{\ell}+\left(\sigma_{j}^{2}+\sigma_{k}^{2}\right) \boldsymbol{\Sigma}\right)-\boldsymbol{z}_{j}^{\top}\left[\kappa(\mathbf{X}, \mathbf{X})+\sigma_{\text {obs }}^{2} \mathbf{I}_{n}\right]^{-1} \boldsymbol{z}_{k}
\end{aligned}
$$

---

#### Page 16

#### A.2.3 Negative quadratic mean function

We consider now a GP with a negative quadratic mean function,

$$
m(\boldsymbol{x}) \equiv m_{\mathrm{NQ}}(\boldsymbol{x})=m_{0}-\frac{1}{2} \sum_{i=1}^{D} \frac{\left(x^{(i)}-x_{\mathrm{m}}^{(i)}\right)^{2}}{{\omega^{(i)}}^{2}}
$$

With this mean function, for each integral in Eq. S9 we have in expectation over the GP posterior,

$$
\begin{aligned}
\mathbb{E}_{f \mid \boldsymbol{\Xi}}\left[\mathcal{I}_{k}\right] & =\int \mathcal{N}\left(\boldsymbol{x} ; \boldsymbol{\mu}_{k}, \sigma_{k}^{2} \boldsymbol{\Sigma}\right)\left[\sigma_{f}^{2} \mathcal{N}\left(\boldsymbol{x} ; \mathbf{X}, \boldsymbol{\Sigma}_{\ell}\right)\left[\kappa(\mathbf{X}, \mathbf{X})+\sigma_{\text {obs }}^{2} \mathbf{I}\right]^{-1}(\boldsymbol{y}-m(\mathbf{X}))+m(\boldsymbol{x})\right] d \boldsymbol{x} \\
& =\boldsymbol{z}_{k}^{\top}\left[\kappa(\mathbf{X}, \mathbf{X})+\sigma_{\text {obs }}^{2} \mathbf{I}\right]^{-1}(\boldsymbol{y}-m(\mathbf{X}))+m_{0}+\nu_{k}
\end{aligned}
$$

where we defined

$$
\nu_{k}=-\frac{1}{2} \sum_{i=1}^{D} \frac{1}{{\omega^{(i)}}^{2}}\left({\mu_{k}^{(i)}}^{2}+\sigma_{k}^{2} {\lambda^{(i)}}^{2}-2 \mu_{k}^{(i)} x_{\mathrm{m}}^{(i)}+{x_{\mathrm{m}}^{(i)}}^{2}\right)
$$

### A.3 Optimization of the approximate ELBO

In the following paragraphs we describe how we optimize the ELBO in each iteration of VBMC, so as to find the variational posterior that best approximates the current GP model of the posterior.

#### A.3.1 Reparameterization

For the purpose of the optimization, we reparameterize the variational parameters such that they are defined in a potentially unbounded space. The mixture means, $\boldsymbol{\mu}_{k}$, remain the same. We switch from mixture scale parameters $\sigma_{k}$ to their logarithms, $\log \sigma_{k}$, and similarly from coordinate length scales, $\lambda^{(i)}$, to $\log \lambda^{(i)}$. Finally, we parameterize mixture weights as unbounded variables, $\eta_{k} \in \mathbb{R}$, such that $w_{k} \equiv e^{\eta_{k}} / \sum_{l} e^{\eta_{l}}$ (softmax function). We compute the appropriate Jacobian for the change of variables and apply it to the gradients calculated in Sections A.1 and A.2.

#### A.3.2 Choice of starting points

In each iteration, we first perform a quick exploration of the ELBO landscape in the vicinity of the current variational posterior by generating $n_{\text {fast }} \cdot K$ candidate starting points, obtained by randomly jittering, rescaling, and reweighting components of the current variational posterior. In this phase we also add new mixture components, if so requested by the algorithm, by randomly splitting and jittering existing components. We evaluate the ELBO at each candidate starting point, and pick the point with the best ELBO as starting point for the subsequent optimization.

For most iterations we use $n_{\text {fast }}=5$, except for the first iteration and the first iteration after the end of warm-up, for which we set $n_{\text {fast }}=50$.

#### A.3.3 Stochastic gradient descent

We optimize the (negative) ELBO via stochastic gradient descent, using a customized version of Adam [21]. Our modified version of Adam includes a time-decaying learning rate, defined as

$$
\alpha_{t}=\alpha_{\min }+\left(\alpha_{\max }-\alpha_{\min }\right) \exp \left[-\frac{t}{\tau}\right]
$$

where $t$ is the current iteration of the optimizer, $\alpha_{\min }$ and $\alpha_{\max }$ are, respectively, the minimum and maximum learning rate, and $\tau$ is the decay constant. We stop the optimization when the estimated change in function value or in the parameter vector across the past $n_{\text {batch }}$ iterations of the optimization goes below a given threshold.
We set as hyperparameters of the optimizer $\beta_{1}=0.9, \beta_{2}=0.99, \epsilon \approx 1.49 \cdot 10^{-8}$ (square root of double precision), $\alpha_{\min }=0.001, \tau=200, n_{\text {batch }}=20$. We set $\alpha_{\max }=0.1$ during warm-up, and $\alpha_{\max }=0.01$ thereafter.

---

#### Page 17

## B Algorithmic details

We report here several implementation details of the VBMC algorithm omitted from the main text.

### B.1 Regularization of acquisition functions

Active sampling in VBMC is performed by maximizing an acquisition function $a: \mathcal{X} \subseteq \mathbb{R}^{D} \rightarrow$ $[0, \infty)$, where $\mathcal{X}$ is the support of the target density. In the main text we describe two such functions, uncertainty sampling ( $a_{\mathrm{us}}$ ) and prospective uncertainty sampling ( $a_{\mathrm{pro}}$ ).
A well-known problem with GPs, in particular when using smooth kernels such as the squared exponential, is that they become numerically unstable when the training set contains points which are too close to each other, producing a ill-conditioned Gram matrix. Here we reduce the chance of this happening by introducing a correction factor as follows. For any acquisition function $a$, its regularized version $a^{\text {reg }}$ is defined as

$$
a^{\mathrm{reg}}(\boldsymbol{x})=a(\boldsymbol{x}) \exp \left\{-\left(\frac{V^{\mathrm{reg}}}{V_{\boldsymbol{\Xi}}(\boldsymbol{x})}-1\right)\left|\left[V_{\boldsymbol{\Xi}}(\boldsymbol{x})<V^{\mathrm{reg}}\right]\right|\right\}
$$

where $V_{\boldsymbol{\Xi}}(\boldsymbol{x})$ is the total posterior predictive variance of the GP at $\boldsymbol{x}$ for the given training set $\boldsymbol{\Xi}$, $V^{\text {reg }}$ a regularization parameter, and we denote with $|[\cdot]|$ Iverson's bracket [28], which takes value 1 if the expression inside the bracket is true, 0 otherwise. Eq. S22 enforces that the regularized acquisition function does not pick points too close to points in $\boldsymbol{\Xi}$. For VBMC, we set $V^{\text {reg }}=10^{-4}$.

### B.2 GP hyperparameters and priors

The GP model in VBMC has $3 D+3$ hyperparameters, $\boldsymbol{\psi}=\left(\boldsymbol{\ell}, \sigma_{f}, \sigma_{\mathrm{obs}}, m_{0}, \boldsymbol{x}_{\mathrm{m}}, \boldsymbol{\omega}\right)$. We define all scale hyperparameters, that is $\left\{\boldsymbol{\ell}, \sigma_{f}, \sigma_{\mathrm{obs}}, \boldsymbol{\omega}\right\}$, in log space.
We assume independent priors on each hyperparameter. For some hyperparameters, we impose as prior a broad Student's $t$ distribution with a given mean $\mu$, scale $\sigma$, and $\nu=3$ degrees of freedom. Following an empirical Bayes approach, mean and scale of the prior might depend on the current training set. For all other hyperparameters we assume a uniform flat prior. GP hyperparameters and their priors are reported in Table S1.

|        Hyperparameter        | Description                            |                           Prior mean $\mu$                           |                                                                        Prior scale $\sigma$                                                                         |
| :--------------------------: | :------------------------------------- | :------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|      $\log \ell^{(i)}$       | Input length scale ($i$-th dimension)    | $\log \operatorname{SD}\left[\mathbf{X}_{\text {hpd }}^{(i)}\right]$ | $\max \left\{2, \log \frac{\operatorname{diam}\left[\mathbf{X}_{\text {hpd }}^{(i)}\right]}{\operatorname{SD}\left[\mathbf{X}_{\text {hpd }}^{(i)}\right]}\right\}$ |
|      $\log \sigma_{f}$       | Output scale                           |                               Uniform                                |                                                                                  —                                                                                  |
| $\log \sigma_{\text {obs }}$ | Observation noise                      |                             $\log 0.001$                             |                                                                                 0.5                                                                                 |
|           $m_{0}$            | Mean function maximum                  |                 $\max \boldsymbol{y}_{\text {hpd }}$                 |                                                   $\operatorname{diam}\left[\boldsymbol{y}_{\text {hpd }}\right]$                                                   |
|    $x_{\mathrm{m}}^{(i)}$    | Mean function location ($i$-th dim.)     |                               Uniform                                |                                                                                  —                                                                                  |
|     $\log \omega^{(i)}$      | Mean function length scale ($i$-th dim.) |                               Uniform                                |                                                                                  —                                                                                  |

Table S1: GP hyperparameters and their priors. See text for more information.

In Table S1, $\operatorname{SD}[\cdot]$ denotes the sample standard deviation and $\operatorname{diam}[\cdot]$ the diameter of a set, that is the maximum element minus the minimum. We define the high posterior density training set, $\boldsymbol{\Xi}_{\text {hpd }}=\left\{\mathbf{X}_{\text {hpd }}, \boldsymbol{y}_{\text {hpd }}\right\}$, constructed by keeping a fraction $f_{\text {hpd }}$ of the training points with highest target density values. For VBMC, we use $f_{\text {hpd }}=0.8$ (that is, we only ignore a small fraction of the points in the training set).

### B.3 Transformation of variables

In VBMC, the problem coordinates are defined in an unbounded internal working space, $\boldsymbol{x} \in \mathbb{R}^{D}$. All original problem coordinates $x_{\text {orig }}^{(i)}$ for $1 \leq i \leq D$ are independently transformed by a mapping $g_{i}: \mathcal{X}_{\text {orig }}^{(i)} \rightarrow \mathbb{R}$ defined as follows.

---

#### Page 18

Unbounded coordinates are 'standardized' with respect to the plausible box, $g_{\text {unb }}\left(x_{\text {orig }}\right)=$ $\frac{x_{\text {orig }}-(\text { PLB }+ \text { PUB }) / 2}{\text { PUB-PLB }}$, where PLB and PUB are here, respectively, the plausible lower bound and plausible upper bound of the coordinate under consideration.
Bounded coordinates are first mapped to an unbounded space via a logit transform, $g_{\text {bnd }}\left(x_{\text {orig }}\right)=$ $\log \left(\frac{z}{1-z}\right)$ with $z=\frac{x_{\text {orig }}-\mathrm{LB}}{\mathrm{UB}-\mathrm{LB}}$, where LB and UB are here, respectively, the lower and upper bound of the coordinate under consideration. The remapped variables are then 'standardized' as above, using the remapped PLB and PUB values after the logit transform.

Note that probability densities are transformed under a change of coordinates by a multiplicative factor equal to the inverse of the determinant of the Jacobian of the transformation. Thus, the value of the observed log joint $y$ used by VBMC relates to the value $y_{\text {orig }}$ of the log joint density, observed in the original (untransformed) coordinates, as follows,

$$
y(\boldsymbol{x})=y^{\text {orig }}\left(\boldsymbol{x}_{\text {orig }}\right)-\sum_{i=1}^{D} \log g_{i}^{\prime}\left(\boldsymbol{x}_{\text {orig }}\right)
$$

where $g_{i}^{\prime}$ is the derivative of the transformation for the $i$-th coordinate, and $\boldsymbol{x}=g\left(\boldsymbol{x}_{\text {orig }}\right)$. See for example [24] for more information on transformations of variables.

### B.4 Termination criteria

The VBMC algorithm terminates when reaching a maximum number of target density evaluations, or when achieving long-term stability of the variational solution, as described below.

#### B.4.1 Reliability index

At the end of each iteration $t$ of the VBMC algorithm, we compute a set of reliability features of the current variational solution.

1. The absolute change in mean ELBO from the previous iteration:

$$
\rho_{1}(t)=\frac{|\mathbb{E}[\operatorname{ELBO}(t)]-\mathbb{E}[\operatorname{ELBO}(t-1)]|}{\Delta_{\mathrm{SD}}}
$$

where $\Delta_{\mathrm{SD}}>0$ is a tolerance parameter on the error of the ELBO.

2. The uncertainty of the current ELBO:

$$
\rho_{2}(t)=\frac{\sqrt{\mathbb{V}[\operatorname{ELBO}(t)]}}{\Delta_{\mathrm{SD}}}
$$

3. The change in symmetrized KL divergence between the current variational posterior $q_{t} \equiv$ $q_{\phi_{t}}(\boldsymbol{x})$ and the one from the previous iteration:

$$
\rho_{3}(t)=\frac{\mathrm{KL}\left(q_{t} \| q_{t-1}\right)+\mathrm{KL}\left(q_{t-1} \| q_{t}\right)}{2 \Delta_{\mathrm{KL}}}
$$

where for Eq. S26 we use the Gaussianized KL divergence (that is, we compare solutions only based on their mean and covariance), and $\Delta_{\mathrm{KL}}>0$ is a tolerance parameter for differences in variational posterior.

The parameters $\Delta_{\mathrm{SD}}$ and $\Delta_{\mathrm{KL}}$ are chosen such that $\rho_{j} \lesssim 1$, with $j=1,2,3$, for features that are deemed indicative of a good solution. For VBMC, we set $\Delta_{\mathrm{SD}}=0.1$ and $\Delta_{\mathrm{KL}}=0.01 \cdot \sqrt{D}$.
The reliability index $\rho(t)$ at iteration $t$ is obtained by averaging the individual reliability features $\rho_{j}(t)$.

#### B.4.2 Long-term stability termination condition

The long-term stability termination condition is reached at iteration $t$ when:

1. all reliability features $\rho_{j}(t)$ are below 1 ;

---

#### Page 19

2. the reliability index $\rho$ has remained below 1 for the past $n_{\text {stable }}$ iterations (with the exception of at most one iteration, excluding the current one);
3. the slope of the ELCBO computed across the past $n_{\text {stable }}$ iterations is below a given threshold $\Delta_{\text {IMPRO }}>0$, suggesting that the ELCBO is stationary.

For VBMC, we set by default $n_{\text {stable }}=8$ and $\Delta_{\text {IMPRO }}=0.01$. For computing the ELCBO we use $\beta_{\text {LCB }}=3$ (see Eq. 8 in the main text).

#### B.4.3 Validation of VBMC solutions

Long-term stability of the variational solution is suggestive of convergence of the algorithm to a (local) optimum, but it should not be taken as a conclusive result without further validation. In fact, without additional information, there is no way to know whether the algorithm has converged to a good solution, let alone to the global optimum. For this reason, we recommend to run the algorithm multiple times and compare the solutions, and to perform posterior predictive checks [29]. See also [30] for a discussion of methods to validate the results of variational inference.

## C Experimental details and additional results

### C.1 Synthetic likelihoods

We plot in Fig. S1 synthetic target densities belonging to the test families described in the main text (lumpy, Student, cigar), for the $D=2$ case. We also plot examples of solutions returned by VBMC after reaching long-term stability, and indicate the number of iterations.

> **Image description.** The image consists of six contour plots arranged in a 2x3 grid. Each column represents a different synthetic target density: "Lumpy", "Student", and "Cigar". The top row shows the "True" target densities, while the bottom row shows the corresponding variational posteriors returned by VBMC (Variational Bayesian Monte Carlo).
>
> Each plot has x1 and x2 axes. The plots are contained within square frames.
>
> - **Column 1 (Lumpy):**
>
>   - Top: A contour plot of the "True" lumpy density, showing an irregular shape with multiple peaks. The contours are nested and colored in shades of blue and yellow.
>   - Bottom: A contour plot of the VBMC solution for the lumpy density. The shape is similar to the "True" density but slightly smoother. The text "Iteration 11" is below the plot.
>
> - **Column 2 (Student):**
>
>   - Top: A contour plot of the "True" Student density, showing a roughly square shape with rounded corners. The contours are nested and colored in shades of blue.
>   - Bottom: A contour plot of the VBMC solution for the Student density. The shape is nearly circular. The text "Iteration 9" is below the plot.
>
> - **Column 3 (Cigar):**
>   - Top: A contour plot of the "True" cigar density, showing a highly elongated shape along a diagonal. The contours are nested and colored in shades of blue and green.
>   - Bottom: A scatter plot representing the VBMC solution for the cigar density. It shows a cluster of blue circles aligned along a diagonal, similar to the "True" density. The text "Iteration 22" is below the plot.
>
> The text labels "Lumpy", "Student", and "Cigar" are above their respective columns. The text "True" is above each plot in the top row, and "VBMC" is above each plot in the bottom row. The axes are labeled "x1" and "x2".

Figure S1: Synthetic target densities and example solutions. Top: Contour plots of two-dimensional synthetic target densities. Bottom: Contour plots of example variational posteriors returned by VBMC, and iterations until convergence.

Note that VBMC, despite being overall the best-performing algorithm on the cigar family in higher dimensions, still underestimates the variance along the major axis of the distribution. This is because the variational mixture components have axis-aligned (diagonal) covariances, and thus many mixture components are needed to approximate non-axis aligned densities. Future work should investigate alternative representations of the variational posterior to increase the expressive power of VBMC, while keeping its computational efficiency and stability.
We plot in Fig. S2 the performance of selected algorithms on the synthetic test functions, for $D \in\{2,4,6,8,10\}$. These results are the same as those reported in Fig. 2 in the main text, but with higher resolution. To avoid clutter, we exclude algorithms with particularly poor performance

---

#### Page 20

or whose plots are redundant with others. In particular, the performance of VBMC-U is virtually identical to VBMC-P here, so we only report the latter. Analogously, with a few minor exceptions, WSABI-M performs similarly or worse than WSABI-L across all problems. AIS suffers from the lack of problem-specific tuning, performing no better than SMC here, and the AGP algorithm diverges on most problems. Finally, we did not manage to get BAPE to run on the cigar family, for $D \leq 6$, without systematically incurring in numerical issues with the GP approximation (with and without regularization of the BAPE acquisition function, as per Section B.1), so these plots are missing.

### C.2 Neuronal model

As a real model-fitting problem, we considered in the main text a neuronal model that combines effects of filtering, suppression, and response nonlinearity, applied to two real data sets (one V1 and one V2 neurons) [14]. The purpose of the original study was to explore the origins of diversity of neuronal orientation selectivity in visual cortex via a combination of novel stimuli (orientation mixtures) and modeling [14]. This model was also previously considered as a case study for a benchmark of Bayesian optimization and other black-box optimization algorithms [6].

#### C.2.1 Model parameters

In total, the original model has 12 free parameters: 5 parameters specifying properties of a linear filtering mechanism, 2 parameters specifying nonlinear transformation of the filter output, and 5 parameters controlling response range and amplitude. For the analysis in the main text, we considered a subset of $D=7$ parameters deemed 'most interesting' by the authors of the original study [14], while fixing the others to their MAP values found by our previous optimization benchmark [6].
The seven model parameters of interest from the original model, their ranges, and the chosen plausible bounds are reported in Table S2.

| Parameter | Description                                  |    LB |  UB |  PLB | PUB |
| :-------: | :------------------------------------------- | ----: | --: | ---: | --: |
|  $x_{1}$  | Preferred direction of motion (deg)          |     0 | 360 |   90 | 270 |
|  $x_{2}$  | Preferred spatial frequency (cycles per deg) |  0.05 |  15 |  0.5 |  10 |
|  $x_{3}$  | Aspect ratio of 2-D Gaussian                 |   0.1 | 3.5 |  0.3 | 3.2 |
|  $x_{4}$  | Derivative order in space                    |   0.1 | 3.5 |  0.3 | 3.2 |
|  $x_{5}$  | Gain inhibitory channel                      |    -1 |   1 | -0.3 | 0.3 |
|  $x_{6}$  | Response exponent                            |     1 | 6.5 |    2 |   5 |
|  $x_{7}$  | Variance of response gain                    | 0.001 |  10 | 0.01 |   1 |

Table S2: Parameters and bounds of the neuronal model (before remapping).

Since all original parameters are bounded, for the purpose of our analysis we remapped them to an unbounded space via a shifted and rescaled logit transform, correcting the value of the log posterior with the log Jacobian (see Section B.3). For each parameter, we set independent Gaussian priors in the transformed space with mean equal to the average of the transformed values of PLB and PUB (see Table S2), and with standard deviation equal to half the plausible range in the transformed space.

#### C.2.2 True and approximate posteriors

We plot in Fig. S3 the 'true' posterior obtained via extensive MCMC sampling for one of the two datasets (V2 neuron), and we compare it with an example variational solution returned by VBMC after reaching long-term stability (here in 52 iterations, which correspond to 260 target density evaluations).
We note that VBMC obtains a good approximation of the true posterior, which captures several features of potential interest, such as the correlation between the inhibition gain $\left(x_{5}\right)$ and response exponent $\left(x_{6}\right)$, and the skew in the preferred spatial frequency $\left(x_{2}\right)$. The variational posterior, however, misses some details, such as the long tail of the aspect ratio $\left(x_{3}\right)$, which is considerably thinner in the approximation than in the true posterior.

---

#### Page 21

> **Image description.** This image contains two panels, A and B, each displaying a series of line graphs. Each panel contains five subplots arranged in a row, and each row represents a different problem ("Lumpy", "Student", "Cigar").
>
> Panel A:
>
> - Each subplot in panel A displays "Median LML error" on the y-axis (log scale) versus "Function evaluations" on the x-axis (linear scale).
> - The x-axis ranges vary between subplots, with maximum values of 200, 300, 400, 400, and 600.
> - The y-axis ranges from 10^-4 to 10 for the "Lumpy" and "Student" problems, and from 0.1 to 10^4 for the "Cigar" problem.
> - Each subplot contains multiple lines, each representing a different algorithm: "smc" (dotted gray), "bmc" (solid gray), "wsabi-L" (solid pink), "bbq" (dashed green), "bape" (solid green), and "vbmc-P" (solid black). Shaded regions around the lines represent confidence intervals.
> - A horizontal dashed line is present at y=1 in each subplot.
> - The columns are labeled "2D", "4D", "6D", "8D", and "10D".
>
> Panel B:
>
> - Each subplot in panel B displays "Median gsKL" on the y-axis (log scale) versus "Function evaluations" on the x-axis (linear scale).
> - The x-axis ranges are the same as in Panel A.
> - The y-axis ranges from 10^-4 to 10 for the "Lumpy" and "Student" problems, and from 10^-2 to 10^6 for the "Cigar" problem.
> - The same algorithms are represented with the same line styles and colors as in Panel A. Shaded regions around the lines represent confidence intervals.
> - A horizontal dashed line is present at y=1 in each subplot.
> - The columns are labeled "2D", "4D", "6D", "8D", and "10D".
>
> Overall:
> The image presents a comparison of different algorithms on synthetic likelihood problems, evaluating their performance based on "Median LML error" and "Median gsKL" metrics. The performance is shown as a function of the number of function evaluations for different problem dimensionalities (2D, 4D, 6D, 8D, 10D).

Figure S2: Full results on synthetic likelihoods. A. Median absolute error of the LML estimate with respect to ground truth, as a function of number of likelihood evaluations, on the lumpy (top), Student (middle), and cigar (bottom) problems, for $D \in\{2,4,6,8,10\}$ (columns). B. Median "Gaussianized" symmetrized KL divergence between the algorithm's posterior and ground truth. For both metrics, shaded areas are $95 \%$ CI of the median, and we consider a desirable threshold to be below one (dashed line). This figure reproduces Fig. 2 in the main text with more details. Note that panels here may have different vertical axes.

---

#### Page 22

> **Image description.** This image contains two triangle plots, one labeled "True" at the top and the other labeled "VBMC (iteration 52)" at the bottom. Each triangle plot displays a matrix of plots representing the posterior distribution of parameters in a model.
>
> Each row and column corresponds to a parameter, labeled x1 through x7. The diagonal elements of each matrix are histograms, representing the 1-D marginal distribution of the posterior for each parameter. The elements below the diagonal are contour plots, representing the 2-D marginal distribution for each pair of parameters.
>
> The "True" plot shows smoother, more defined contours and histograms, while the "VBMC (iteration 52)" plot shows similar shapes but with some differences in the contours and histograms. The axes are labeled with numerical values corresponding to the range of each parameter.

Figure S3: True and approximate posterior of neuronal model (V2 neuron). Top: Triangle plot of the 'true' posterior (obtained via MCMC) for the neuronal model applied to the V2 neuron dataset. Each panel below the diagonal is the contour plot of the 2-D marginal distribution for a given parameter pair. Panels on the diagonal are histograms of the 1-D marginal distribution of the posterior for each parameter. Bottom: Triangle plot of a typical variational solution returned by VBMC.

---

#### Page 23

## D Analysis of VBMC

In this section we report additional analyses of the VBMC algorithm.

### D.1 Variability between VBMC runs

In the main text we have shown the median performance of VBMC, but a crucial question for a practical application of the algorithm is the amount of variability between runs, due to stochasticity in the algorithm and choice of starting point (in this work, drawn uniformly randomly inside the plausible box). We plot in Fig. S4 the performance of one hundred runs of VBMC on the neuronal model datasets, together with the 50th (the median), 75th, and 90th percentiles. The performance of VBMC on this real problem is fairly robust, in that some runs take longer but the majority of them converges to quantitatively similar solutions.

> **Image description.** This image contains two panels, A and B, each displaying two line graphs. All four graphs share a similar structure, plotting the performance of an algorithm across multiple runs.
>
> **Panel A:**
>
> - **Title:** A is in the top left corner.
> - **Y-axis:** Labeled "Median LML error" on a logarithmic scale from 10^-2 to 10^4. The label "Neuronal model" is placed vertically to the left of the y-axis label.
> - **X-axis:** Labeled "Function evaluations" ranging from 0 to 400.
> - **Graphs:** Two graphs, labeled "V1" and "V2" at the top, showing the error as a function of function evaluations. Each graph contains multiple thin grey lines representing individual runs of the algorithm. Thicker lines represent the 50th (solid), 75th (dashed), and 90th (dotted) percentiles across runs, as indicated by a legend. A horizontal dashed line is present at y=1.
>
> **Panel B:**
>
> - **Title:** B is in the top left corner.
> - **Y-axis:** Labeled "Median gsKL" on a logarithmic scale from 10^-2 to 10^6.
> - **X-axis:** Labeled "Function evaluations" ranging from 0 to 400.
> - **Graphs:** Two graphs, labeled "V1" and "V2" at the top, showing the error as a function of function evaluations. Each graph contains multiple thin grey lines representing individual runs of the algorithm. Thicker lines represent the 50th (solid), 75th (dashed), and 90th (dotted) percentiles across runs, as indicated by a legend in Panel A. A horizontal dashed line is present at y=1.
>
> In both panels, the graphs show a general trend of decreasing error/divergence as the number of function evaluations increases, indicating convergence of the algorithm. The percentile lines provide insight into the variability of performance across different runs.

Figure S4: Variability of VBMC performance. A. Absolute error of the LML estimate, as a function of number of likelihood evaluations, for the two neuronal datasets. Each grey line is one of 100 distinct runs of VBMC. Thicker lines correspond to the 50th (median), 75th, and 90th percentile across runs (the median is the same as in Fig. 3 in the main text). B. "Gaussianized" symmetrized KL divergence between the algorithm's posterior and ground truth, for 100 distinct runs of VBMC. See also Fig. 3 in the main text.

### D.2 Computational cost

The computational cost of VBMC stems in each iteration of the algorithm primarily from three sources: active sampling, GP training, and variational optimization. Active sampling requires repeated computation of the acquisition function (for its optimization), whose cost is dominated by calculation of the posterior predictive variance of the GP, which scales as $O\left(n^{2}\right)$, where $n$ is the number of training points. GP training scales as $O\left(n^{3}\right)$, due to inversion of the Gram matrix. Finally, variational optimization scales as $O(K n)$, where $K$ is the number of mixture components. In practice, we found in many cases that in early iterations the costs are equally divided between the three phases, but later on both GP training and variational optimization dominate the algorithmic cost. In particular, the number of components $K$ has a large impact on the effective cost.
As an example, we plot in Fig. S5 the algorithmic cost per function evaluation of different inference algorithms that have been run on the V1 neuronal dataset (algorithmic costs are similar for the V2 dataset). We consider only methods which use active sampling with a reasonable performance on at least some of the problems. We define as algorithmic cost the time spent inside the algorithm, ignoring the time used to evaluate the log likelihood function. For comparison, evaluation of the log likelihood of this problem takes about 1 s on the reference laptop computer we used. Note that for the WSABI and BBQ algoritms, the algorithmic cost reported here does not include the additional computational cost of sampling an approximate distrbution from the GP posterior (WSABI and BBQ, per se, only compute an approximation of the marginal likelihood).
VBMC on this problem exhibits a moderate cost of 2-3 s per function evaluation, when averaged across the entire run. Moreover, many runs would converge within 250-300 function evaluations, as shown in Figure S4, further lowering the effective cost per function evaluation. For the considered budget of function evaluations, WSABI (in particular, WSABI-L) is up to one order of magnitude faster than VBMC. This speed is remarkable, although it does not offset the limited performance of

---

#### Page 24

> **Image description.** This is a line graph comparing the algorithmic cost per function evaluation for different algorithms performing inference on a V1 neuronal dataset.
>
> The graph has the following characteristics:
>
> - **Title:** "Neuronal model (V1)" is at the top of the graph.
>
> - **Axes:**
>
>   - The x-axis is labeled "Function evaluations" and ranges from approximately 0 to 400.
>   - The y-axis is labeled "Median algorithmic cost per function evaluation (s)" and uses a logarithmic scale, ranging from 0.01 to 100.
>
> - **Data Series:** The graph displays several data series, each representing a different algorithm:
>
>   - **wsabi-L:** A red line that starts low and gradually increases.
>   - **wsabi-M:** A blue line that starts low and gradually increases, staying below the vbmc-P line.
>   - **bbq:** A purple line that is initially high and very jagged, with sharp peaks, but gradually smooths out.
>   - **bape:** A dashed green line that fluctuates around a value of approximately 2.
>   - **vbmc-P:** A solid black line that starts high, dips, and then gradually increases, staying around a value of 1.
>
> - **Confidence Intervals:** Shaded areas around each line represent the 95% confidence interval of the median.
>
> - **Horizontal Line:** A dashed horizontal line is present at y = 1.
>
> - **Legend:** A legend on the right side of the graph identifies each data series with its corresponding algorithm name and color.

Figure S5: Algorithmic cost per function evaluation. Median algorithmic cost per function evaluation, as a function of number of likelihood function evaluations, for different algorithms performing inference over the V1 neuronal dataset. Shaded areas are $95 \%$ CI of the median.

the algorithm on more complex problems. WSABI-M is generally more expensive than WSABI-L (even though still quite fast), with a similar or slightly worse performance. Here our implementation of BAPE results to be slightly more expensive than VBMC. Perhaps it is possible to obtain faster implementations of BAPE, but, even so, the quality of solutions would still not match that of VBMC (also, note the general instability of the algorithm). Finally, we see that BBQ incurs in a massive algorithmic cost due to the complex GP approximation and expensive acquisition function used. Notably, the solutions obtained by BBQ in our problem sets are relatively good compared to the other algorithms, but still substantially worse than VBMC on all but the easiest problems, despite the much larger computational overhead.
The dip in cost that we observe in VBMC at around 275 function evaluations is due to the switch from GP hyperparameter sampling to optimization. The cost of BAPE oscillates because of the cost of retraining the GP model and MCMC sampling from the approximate posterior every 10 function evaluations. Similarly, by default BBQ retrains the GP model ten times, logarithmically spaced across its run, which appears here as logarithmically-spaced spikes in the cost.

### D.3 Analysis of the samples produced by VBMC

We report the results of two control experiments to better understand the performance of VBMC.
For the first control experiment, shown in Fig. S6A, we estimate the log marginal likelihood (LML) using the WSABI-L approximation trained on samples obtained by VBMC (with the $a_{\text {pro }}$ acquisition function). The LML error of WSABI-L trained on VBMC samples is lower than WSABI-L alone, showing that VBMC produces higher-quality samples and, given the same samples, a better approximation of the marginal likelihood. The fact that the LML error is still substantially higher in the control than with VBMC alone demonstrates that the error induced by the WSABI-L approximation can be quite large.
For the second control experiment, shown in Fig. S6B, we produce $2 \cdot 10^{4}$ posterior samples from a GP directly trained on the log joint distribution at the samples produced by VBMC. The quality of this posterior approximation is better than the posterior obtained by other methods, although generally not as good as the variational approximation (in particular, it is much more variable). While it is possible that the posterior approximation via direct GP fit could be improved, for example by using ad-hoc methods to increase the stability of the GP training procedure, this experiment shows that VBMC is able to reliably produce a high-quality variational posterior.

---

#### Page 25

> **Image description.** The image contains two panels, labeled A and B, each containing two line graphs. All four graphs share a similar structure.
>
> **Panel A:**
>
> - **Title:** "Neuronal model" is written vertically on the left side of the panel.
> - **Graphs:** Two line graphs, labeled "V1" and "V2" above each graph.
>   - X-axis: "Function evaluations", ranging from 0 to 400.
>   - Y-axis: "Median LML error", with a logarithmic scale ranging from 10^-2 to 10^4.
>   - Data: Each graph displays three lines representing different algorithms:
>     - "vbmc-P" (black line)
>     - "vbmc-control" (olive green line with shaded area around the line)
>     - "wsabi-L" (light red line with shaded area around the line)
>   - A dashed horizontal line is present at y=1.
>
> **Panel B:**
>
> - **Graphs:** Two line graphs, labeled "V1" and "V2" above each graph.
>   - X-axis: "Function evaluations", ranging from 0 to 400.
>   - Y-axis: "Median gsKL", with a logarithmic scale ranging from 10^-2 to 10^6.
>   - Data: Each graph displays three lines representing different algorithms:
>     - "vbmc-P" (black line)
>     - "vbmc-control" (olive green line with shaded area around the line)
>     - "wsabi-L" (light red line with shaded area around the line)
>   - A dashed horizontal line is present at y=1.
>
> **Legend:**
>
> - A legend is present between the two panels, mapping line colors to algorithm names:
>   - Black line: "vbmc-P"
>   - Olive green line: "vbmc-control"
>   - Light red line: "wsabi-L"
>
> In summary, the image presents four line graphs comparing the performance of three different algorithms ("vbmc-P", "vbmc-control", and "wsabi-L") across two different metrics ("Median LML error" and "Median gsKL") for two distinct neurons (V1 and V2) as a function of function evaluations. The shaded areas around the lines represent the 95% confidence interval of the median.

Figure S6: Control experiments on neuronal model likelihoods. A. Median absolute error of the LML estimate, as a function of number of likelihood evaluations, for two distinct neurons $(D=7)$. For the control experiment, here we computed the LML with WSABI-L trained on VBMC samples. B. Median "Gaussianized" symmetrized KL divergence between the algorithm's posterior and ground truth. For this control experiment, we produced posterior samples from a GP directly trained on the log joint at the samples produced by VBMC. For both metrics, shaded areas are $95 \%$ CI of the median, and we consider a desirable threshold to be below one (dashed line). See text for more details, and see also Fig. 3 in the main text.
