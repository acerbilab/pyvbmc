```
@article{silvestrin2025stacking,
  title={Stacking Variational Bayesian Monte Carlo},
  author={Francesco Silvestrin and Chengkun Li and Luigi Acerbi},
  year={2025},
  journal={TMLR},
  doi={10.48550/arXiv.2504.05004},
  url={https://www.semanticscholar.org/paper/e236094d9b9f6a20a5e103968138345cde97e845}
}
```

---

#### Page 1

# Stacking Variational Bayesian Monte Carlo

Francesco Silvestrin<br>francesco.silvestrin@helsinki.fi<br>Department of Computer Science<br>University of Helsinki<br>Chengkun Li<br>chengkun.li@helsinki.fi<br>Department of Computer Science<br>University of Helsinki

Luigi Acerbi
luigi.acerbi@helsinki.fi
Department of Computer Science
University of Helsinki

## Abstract

Approximate Bayesian inference for models with computationally expensive, black-box likelihoods poses a significant challenge, especially when the posterior distribution is complex. Many inference methods struggle to explore the parameter space efficiently under a limited budget of likelihood evaluations. Variational Bayesian Monte Carlo (VBMC) is a sample-efficient method that addresses this by building a local surrogate model of the log-posterior. However, its conservative exploration strategy, while promoting stability, can cause it to miss important regions of the posterior, such as distinct modes or long tails. In this work, we introduce Stacking Variational Bayesian Monte Carlo (S-VBMC), a method that overcomes this limitation by constructing a robust, global posterior approximation from multiple independent VBMC runs. Our approach merges these local approximations through a principled and inexpensive post-processing step that leverages VBMC's mixture posterior representation and per-component evidence estimates. Crucially, S-VBMC requires no additional likelihood evaluations and is naturally parallelisable, fitting seamlessly into existing inference workflows. We demonstrate its effectiveness on two synthetic problems designed to challenge VBMC's exploration and two real-world applications from computational neuroscience, showing substantial improvements in posterior approximation quality across all cases. Our code is available as a Python package at https://github.com/acerbilab/svbmc.

## 1 Introduction

Bayesian inference provides a powerful framework for parameter estimation and uncertainty quantification, but it is usually intractable, requiring approximate inference techniques (Brooks et al., 2011; Blei et al., 2017). Many scientific and engineering problems involve black-box models (Sacks et al., 1989; Kennedy & O'Hagan, 2001), where likelihood evaluation is time-consuming and gradients cannot be easily obtained, making traditional approximate inference approaches computationally prohibitive.

A promising approach to tackle expensive likelihoods is to construct a statistical surrogate model that approximates the target distribution, similar in spirit to surrogate approaches to global optimisation using Gaussian processes, generally known as Bayesian optimisation (Rasmussen & Williams, 2006; Garnett, 2023). However, unlike Bayesian optimisation, where the goal is to find a single point estimate (the global optimum), here the aim is to reconstruct the shape of the entire posterior distribution. In this setting, attempting to build a single global surrogate model may lead to numerical instabilities and poor approximations when the target distribution is complex or multi-modal, without ad hoc solutions (Wang & Li, 2018; Järvenpää et al., 2021; Li et al., 2025). Local or constrained surrogate models, while more limited in scope, tend to be more stable and reliable in practice (El Gammal et al., 2023; Järvenpää & Corander, 2024).

---

#### Page 2

Variational Bayesian Monte Carlo (VBMC; Acerbi, 2018) exemplifies this local approach, using active sampling to train a Gaussian process surrogate for the unnormalised log-posterior on which it performs variational inference. VBMC adopts a conservative exploration strategy that yields stable, local approximations (Acerbi, 2019). Compared to other surrogate-based approaches, the method offers a versatile set of features: it returns the approximate posterior as a tractable distribution (a mixture of Gaussians); it provides a lower bound for the model evidence (ELBO) via Bayesian quadrature (Ghahramani & Rasmussen, 2002), useful for model selection; and it can handle noisy log-likelihood evaluations (Acerbi, 2020), which arise in simulation-based models through estimation techniques such as inverse binomial sampling (van Opheusden et al., 2020) and synthetic likelihood (Wood, 2010; Price et al., 2018). However, VBMC's limited likelihood evaluation budget combined with its local exploration strategy can leave it vulnerable to potentially missing regions of the target posterior – particularly for distributions with distinct modes or long tails.

In this work, we propose a practical, yet effective approach to constructing global surrogate models while overcoming the limitations of standard VBMC by combining multiple local approximations. We introduce Stacking Variational Bayesian Monte Carlo (S-VBMC), a method for merging independent VBMC inference runs into a coherent global posterior approximation. S-VBMC inherits its parent algorithm's operating regime and is intended for low- to moderate-dimensional problems (D ≤ 10). Our approach leverages VBMC's unique properties – its mixture posterior representation and per-component Bayesian quadrature estimates of the ELBO – to combine and reweigh each component through a simple post-processing step. Figure 1 shows an example of two separate posteriors and the combined ("stacked") result obtained with S-VBMC.

> **Image description.** The image displays two side-by-side contour plots, illustrating the concept of combining separate posterior distributions into a single, "stacked" posterior. Both plots share identical axis ranges and labels.
>
> The left panel, titled "Separate VBMC posteriors", shows two distinct sets of contour lines. The x-axis is labeled "$\theta_2$" with numerical ticks at 1e-5, 2e-5, and 3e-5. The y-axis is labeled "$\theta_4$" with numerical ticks at -0.01, 0.00, and 0.01. One set of contours is colored blue, forming an elongated, roughly elliptical shape concentrated in the upper-left portion of the plot, peaking around (1e-5, 0.01). The second set of contours is colored orange, forming a similar elongated shape concentrated in the lower-right portion, peaking around (2e-5 to 3e-5, 0.00 to -0.01). These two distributions overlap significantly in the central region of the plot, creating a criss-cross pattern. The inner contours of each set represent higher probability density.
>
> The right panel, titled "Stacked posterior", presents a single set of magenta-colored contour lines. The x-axis and y-axis are identical to the left panel, labeled "$\theta_2$" and "$\theta_4$" respectively, with the same tick marks. This single distribution appears as an elongated, banana-shaped region that broadly encompasses the areas covered by both the blue and orange contours from the left panel. It stretches from the upper-left corner of the plot to the lower-right corner, indicating a strong negative correlation between $\theta_2$ and $\theta_4$. The contours are densest in the central part of this elongated shape, gradually spreading out towards the edges, signifying varying levels of probability density within the combined distribution.
>
> In essence, the image visually demonstrates how two distinct, overlapping posterior distributions (blue and orange) can be merged to form a single, broader, and more comprehensive "stacked" posterior distribution (magenta).

Figure 1: Two separate VBMC posteriors (left, shown as blue and orange contours) and the resulting stacked posterior via S-VBMC (right, red contour) for a neuronal model with real data (see Section 4.5); showing the marginal distribution of two out of the five model parameters.

Crucially, our method requires no additional evaluations of either the original model or the surrogate. This approach is easily parallelisable and naturally fits existing VBMC pipelines that already employ multiple independent runs (Huggins et al., 2023). While our method could theoretically extend to other variational approaches based on mixture posteriors, VBMC is uniquely suitable for it as re-estimation of the ELBO would otherwise become impractical with expensive likelihoods (see Section 3).

**Related work.** Our work addresses the challenge of building global posterior approximations by combining local solutions from the VBMC framework (Acerbi, 2018; 2019; 2020). While the idea of combining posterior distributions has been explored before, previous approaches differ substantially in their goals and methodology.

"Stacking" was first introduced in the context of supervised learning as a method for model averaging (Wolpert, 1992). Given a set of predictive models, the idea was to use a weighted average of their outputs, with the weights optimised to minimise the leave-one-out squared error. This approach has then been

---

#### Page 3

adapted to average Bayesian predictive distributions (Yao et al., 2018; 2022), which the authors named Bayesian stacking. This still relies on a leave-one-out strategy to optimise predictive performance, which requires access to the likelihood per data point, while S-VBMC optimises the ELBO on the full dataset, allowing treatment of the log-joint as a black box.

Another relevant approach is variational boosting (Guo et al., 2016; Miller et al., 2017; Campbell & Li, 2019). This method builds on black-box variational inference (Ranganath et al., 2014) and entails iteratively running a series of variational optimisations on the whole dataset, with each iteration increasing the complexity (number of components) of a mixture posterior distribution. This allows practitioners to obtain arbitrarily complex (and accurate) posteriors, trading compute time for inference accuracy. However, the process is inherently sequential, whilst S-VBMC can be implemented as a simple post-processing step, allowing individual (VBMC) inference runs to happen in parallel, offering significant computational advantages, further substantiated by VBMC's surrogate-based approach and sample efficiency, which make it particularly suitable for problems with expensive likelihoods.

Parallel computations have, on the other hand, been leveraged in a number of "divide-and-conquer" or embarrassingly parallel approximate inference techniques, starting from embarrassingly parallel Markov Chain Monte Carlo (MCMC) (Neiswanger et al., 2014). All these methods are based on dividing the data into subsets which are processed separately to obtain a set of "sub-posteriors", which are then merged in various ways to recreate the full (approximate) posterior (Wang & Dunson, 2013; Wang et al., 2015; Nemeth & Sherlock, 2018; Srivastava et al., 2018; Scott et al., 2022; De Souza et al., 2022; Chan et al., 2023). These methods are mostly motivated by the need to process very large datasets (Scott et al., 2022), or by privacy concerns requiring federated learning (Liang et al., 2025). In contrast, S-VBMC is motivated by a need to capture complex posteriors that elude single inference runs. Therefore, individual runs all use the complete dataset. This allows our method not to rely on the quality and representativeness of the sub-datasets, and to remain robust to individual run failures.

**Outline.** We first introduce variational inference and VBMC (Section 2), then present our algorithm for stacking VBMC posteriors (Section 3). We demonstrate the effectiveness of our approach through experiments on two synthetic problems and two real-world applications that are challenging for VBMC (Section 4). We then discuss an observed bias buildup in the ELBO estimation and propose a practical heuristic to counteract it (Section 5). We finally discuss our results (Section 6) and conclude with closing remarks (Section 7). Appendix A contains supplementary materials.

## 2 Background

### 2.1 Bayesian Inference

Given a dataset $\mathcal{D}$, and a model parametrised by the vector $\boldsymbol{\theta} \in \mathbb{R}^{D}$ describing how $\mathcal{D}$ was generated, Bayesian inference represents a principled framework to infer the probability distributions over $\boldsymbol{\theta}$. This is achieved through Bayes' rule:

$$
p(\boldsymbol{\theta} \mid \mathcal{D})=\frac{p(\boldsymbol{\theta}) p(\mathcal{D} \mid \boldsymbol{\theta})}{p(\mathcal{D})}
$$

where $p(\boldsymbol{\theta})$ represents the prior over model parameters, $p(\mathcal{D} \mid \boldsymbol{\theta})$ the likelihood and $p(\mathcal{D})=\int p(\boldsymbol{\theta}) p(\mathcal{D} \mid$ $\boldsymbol{\theta}) d \boldsymbol{\theta}$ the normalising constant, also called marginal likelihood or model evidence. This latter quantity is particularly useful for Bayesian model selection (MacKay, 2003), but the integral is often intractable, requiring some approximation technique (Brooks et al., 2011; Blei et al., 2017). In the following section, we discuss one of such techniques.

### 2.2 Variational Inference

Considering the setup outlined in Eq. 1, variational inference (Blei et al., 2017) is a technique that approximates the true posterior $p(\boldsymbol{\theta} \mid \mathcal{D})$ with a parametric distribution $q_{\boldsymbol{\phi}}(\boldsymbol{\theta})$, where $\boldsymbol{\phi}$ denotes the optimisable

---

#### Page 4

variational parameters. This is achieved by maximising the evidence lower bound (ELBO):

$$
\operatorname{ELBO}(\boldsymbol{\phi})=\mathbb{E}_{q_{\phi}}[\log p(\mathcal{D} \mid \boldsymbol{\theta}) p(\boldsymbol{\theta})]+\mathcal{H}\left[q_{\phi}(\boldsymbol{\theta})\right]
$$

where the first term is the expected log-joint distribution (the joint being likelihood times prior) and the second term is the entropy of the variational posterior. Maximising Eq. 2 is equivalent to minimising the Kullback-Leibler divergence between $q_{\phi}(\boldsymbol{\theta})$ and the true posterior, as

$$
\begin{aligned}
D_{\mathrm{KL}}\left[q_{\phi}(\boldsymbol{\theta}) \| p(\boldsymbol{\theta} \mid \mathcal{D})\right] & =\mathbb{E}_{q_{\phi}}\left[\log \frac{q_{\phi}(\boldsymbol{\theta})}{p(\boldsymbol{\theta} \mid \mathcal{D})}\right] \\
& =-\mathbb{E}_{q_{\phi}}[\log p(\boldsymbol{\theta}, \mathcal{D})]+\mathbb{E}_{q_{\phi}}[\log p(\mathcal{D})]+\mathbb{E}_{q_{\phi}}\left[\log q_{\phi}(\boldsymbol{\theta})\right]
\end{aligned}
$$

and, since the model evidence $p(\mathcal{D})$ is already a constant,

$$
\begin{aligned}
\log p(\mathcal{D})-D_{\mathrm{KL}}\left[q_{\phi}(\boldsymbol{\theta}) \| p(\boldsymbol{\theta} \mid \mathcal{D})\right] & =\mathbb{E}_{q_{\phi}}[\log p(\boldsymbol{\theta}, \mathcal{D})]+\mathcal{H}\left[q_{\phi}(\boldsymbol{\theta})\right] \\
& =\operatorname{ELBO}(\boldsymbol{\phi})
\end{aligned}
$$

Crucially, since $D_{\mathrm{KL}}[q \| p] \geq 0$, the ELBO provides a lower bound on the log model evidence $\log p(\mathcal{D})$ (hence the name), with equality when the approximation matches the true posterior (i.e., when $D_{\mathrm{KL}}[q \| p]=0$ ), thus constituting a useful metric for model selection.

### 2.3 Variational Bayesian Monte Carlo (VBMC)

VBMC (Acerbi, 2018; 2020) is a sample-efficient technique to obtain a variational approximation of a target density with only a small number of likelihood evaluations, often of the order of a few hundred. VBMC uses a Gaussian process (GP) as a surrogate of the log-joint, Bayesian quadrature to calculate the expected log-joint, and active sampling to decide which parameters to evaluate next. As an in-depth knowledge of the inner workings of VBMC is not necessary to understand our core contribution, here we only describe its relevant aspects. An interested reader should refer to the original papers (Acerbi, 2018; 2020) or Appendix A.1, where we provide a more detailed description of the algorithm.

As mentioned above, VBMC uses a GP as a surrogate of the target, which is often an unnormalised log-posterior (i.e., the log-joint). Crucially, VBMC performs variational inference on the surrogate

$$
f(\boldsymbol{\theta}) \approx \log p(\mathcal{D} \mid \boldsymbol{\theta}) p(\boldsymbol{\theta})
$$

instead of the true, expensive model. The efficacy of VBMC hinges on the quality of such approximation.
The variational posterior in VBMC is defined as a flexible mixture of $K$ components,

$$
q_{\phi}(\boldsymbol{\theta})=\sum_{k=1}^{K} w_{k} q_{k, \phi}(\boldsymbol{\theta})
$$

where $q_{k}$ is the $k$-th component (a multivariate normal) and $w_{k}$ its mixture weight, with $\sum_{k=1}^{K} w_{k}=1$ and $w_{k} \geq 0$. Plugging in the mixture posterior, the ELBO (Eq. 2) becomes:

$$
\operatorname{ELBO}(\boldsymbol{\phi})=\sum_{k=1}^{K} w_{k} \mathbb{E}_{q_{k, \phi}}[\log p(\mathcal{D} \mid \boldsymbol{\theta}) p(\boldsymbol{\theta})]+\mathcal{H}\left[q_{\phi}(\boldsymbol{\theta})\right]=\sum_{k=1}^{K} w_{k} I_{k}+\mathcal{H}\left[q_{\phi}(\boldsymbol{\theta})\right]
$$

where we defined the $k$-th component of the expected log-joint as:

$$
I_{k}=\mathbb{E}_{q_{k, \phi}}[\log p(\mathcal{D} \mid \boldsymbol{\theta}) p(\boldsymbol{\theta})] \approx \mathbb{E}_{q_{k, \phi}}[f(\boldsymbol{\theta})]=\hat{I}_{k}
$$

The efficacy of VBMC stems from the fact that Eq. 8 has a closed-form Gaussian expression via Bayesian quadrature (O'Hagan, 1991; Ghahramani & Rasmussen, 2002), which yields posterior mean $\hat{I}_{k}$ and covariance matrix $J_{k k^{\prime}}$ (Acerbi, 2018). The entropy of a mixture of Gaussians does not have an analytical solution, but

---

#### Page 5

gradients can be estimated via Monte Carlo. Thus, using the posterior mean of Eq. 8 as a plug-in estimator for the expected log-joint of each component, Eq. 7 can be efficiently optimised via stochastic gradient ascent (Kingma & Ba, 2014).

VBMC differs from other approaches that directly try to apply Bayesian quadrature to solve Bayes' rule (e.g., Ghahramani & Rasmussen, 2002; Osborne et al., 2012; Gunter et al., 2014; Adachi et al., 2022), in that instead it leverages Bayesian quadrature to estimate the ELBO. This is a simpler problem, since instead of attempting to solve the global integral of Eq. 1, namely the integral of prior times likelihood, where the prior is often diffuse, it mainly deals with the local integrals in Eq. 7, namely the expected log-joint, i.e., the integral of the approximate posterior times log-joint, where the approximate posterior is under our control and often more localised, and the entropy, which can often be estimated or approximated for known distributions.

## 3 Stacking VBMC

In this work, we introduce Stacking VBMC (S-VBMC), a novel approach to merge different variational posteriors obtained from different runs on the same model and dataset. Crucially, these runs can happen in parallel, with no information exchange required. The core idea is that a single global surrogate of the log-joint $f(\boldsymbol{\theta})$ might be inaccurate in some regions of the posterior. Therefore, it would be beneficial to leverage the combination of $M$ local surrogates $\left\{f_{m}(\boldsymbol{\theta})\right\}_{m=1}^{M}$ instead, each yielding a good approximation of the target in a different parameter region.

### 3.1 Optimisation objective

Given $M$ independent VBMC runs, one obtains $M$ variational posteriors

$$
q_{\boldsymbol{\phi}_{m}}(\boldsymbol{\theta})=\sum_{k=1}^{K_{m}} w_{m, k} q_{k, \boldsymbol{\phi}_{m}}(\boldsymbol{\theta})
$$

each with $K_{m}$ Gaussian components, as well as $M$ different $\hat{\mathbf{I}}_{m}$ vectors, as per Eq. 8, with $\hat{\mathbf{I}}_{m}=$ $\left(\hat{I}_{m, 1}, \ldots, \hat{I}_{m, K_{m}}\right)$. Our approach consists of "stacking" the Gaussian components of all posteriors $q_{\boldsymbol{\phi}_{m}}(\boldsymbol{\theta})$, leaving all individual components' parameters (means and covariances) unchanged, and reoptimising all the weights. The full set of new mixture weights is denoted by the vector $\tilde{\mathbf{w}}=\left\{\tilde{w}_{m, k}\right\}_{m=1, k=1}^{M, K_{m}}$. The complete set of parameters for the final stacked posterior, $q_{\tilde{\boldsymbol{\phi}}}$, is denoted by $\tilde{\boldsymbol{\phi}}$. This set consists of the frozen parameters (all means and covariances) from the original components $\left\{q_{k, \boldsymbol{\phi}_{m}}\right\}$ and the newly optimised weights $\tilde{\mathbf{w}}$.

Thus, given the stacked posterior

$$
q_{\tilde{\boldsymbol{\phi}}}(\boldsymbol{\theta})=\sum_{m=1}^{M} \sum_{k=1}^{K_{m}} \tilde{w}_{m, k} q_{k, \boldsymbol{\phi}_{m}}(\boldsymbol{\theta})
$$

we optimise the global evidence lower bound with respect to the weights $\tilde{\mathbf{w}}$,

$$
\operatorname{ELBO}_{\text {stacked }}(\tilde{\mathbf{w}})=\sum_{m=1}^{M} \sum_{k=1}^{K_{m}} \tilde{w}_{m, k} \hat{I}_{m, k}+\mathcal{H}\left[q_{\tilde{\boldsymbol{\phi}}}(\boldsymbol{\theta})\right]
$$

It should be noted that each VBMC run applies its own parameter transformation $g_{m}(\cdot)$ during inference (Acerbi, 2020), so variational posteriors are returned in transformed coordinates $q_{\boldsymbol{\phi}_{m}}\left(g_{m}(\boldsymbol{\theta})\right)$. For clarity, in this section, we present all quantities in a common parameter space of $\boldsymbol{\theta}$ and VBMC and S-VBMC posteriors expressed accordingly as $q_{\boldsymbol{\phi}_{m}}(\boldsymbol{\theta})$, and $q_{\tilde{\boldsymbol{\phi}}}(\boldsymbol{\theta})$, respectively. In practice, when applying our stacking approach, we account for these per-run transformations via change-of-variables (Jacobian) corrections; see Appendix A.2 for details.

---

#### Page 6

### 3.2 Algorithm

As the entropy term in Eq. 11 does not have a closed-form solution, it needs to be estimated via Monte Carlo sampling. To do this, we take $S$ samples $\left\{\mathbf{x}_{m, k}^{(s)} \sim q_{k, \boldsymbol{\phi}_{m}}\right\}_{s=1}^{S}$ from each component of the stacked posterior, and estimate the entropy as

$$
\mathcal{H}\left[q_{\tilde{\boldsymbol{\phi}}}(\boldsymbol{\theta})\right] \approx-\sum_{m=1}^{M} \sum_{k=1}^{K_{m}} \tilde{w}_{m, k}\left(\frac{1}{S} \sum_{s=1}^{S} \log q_{\tilde{\boldsymbol{\phi}}}\left(\mathbf{x}_{m, k}^{(s)}\right)\right)
$$

This works because

$$
\begin{aligned}
\mathcal{H}\left[q_{\tilde{\boldsymbol{\phi}}}(\boldsymbol{\theta})\right] & =-\mathbb{E}_{q_{\tilde{\boldsymbol{\phi}}}}\left[\log q_{\tilde{\boldsymbol{\phi}}}(\boldsymbol{\theta})\right] \\
& =-\int q_{\tilde{\boldsymbol{\phi}}}(\boldsymbol{\theta}) \log q_{\tilde{\boldsymbol{\phi}}}(\boldsymbol{\theta}) d \boldsymbol{\theta} \\
& =-\int \sum_{m=1}^{M} \sum_{k=1}^{K_{m}} \tilde{w}_{m, k} q_{k, \boldsymbol{\phi}_{m}}(\boldsymbol{\theta}) \log q_{\tilde{\boldsymbol{\phi}}}(\boldsymbol{\theta}) d \boldsymbol{\theta} \\
& =-\sum_{m=1}^{M} \sum_{k=1}^{K_{m}} \tilde{w}_{m, k} \int q_{k, \boldsymbol{\phi}_{m}}(\boldsymbol{\theta}) \log q_{\tilde{\boldsymbol{\phi}}}(\boldsymbol{\theta}) d \boldsymbol{\theta} \\
& =-\sum_{m=1}^{M} \sum_{k=1}^{K_{m}} \tilde{w}_{m, k} \mathbb{E}_{q_{k, \boldsymbol{\phi}_{m}}}\left[\log q_{\tilde{\boldsymbol{\phi}}}(\boldsymbol{\theta})\right]
\end{aligned}
$$

where we approximate the expectation via Monte Carlo,

$$
\mathbb{E}_{q_{k, \boldsymbol{\phi}_{m}}}\left[\log q_{\tilde{\boldsymbol{\phi}}}(\boldsymbol{\theta})\right] \approx \frac{1}{S} \sum_{s=1}^{S} \log q_{\tilde{\boldsymbol{\phi}}}\left(\mathbf{x}_{m, k}^{(s)}\right)
$$

Importantly, with this formulation, the entropy is differentiable with respect to the weights $\tilde{\mathbf{w}}$.
Since weights must be non-negative and sum to one (i.e., they live on the probability simplex), for the purpose of optimisation we follow standard practices in computational statistics (e.g., Carpenter et al., 2017): we reparameterise the weights with unconstrained logits $\mathbf{a}$ and compute the objective with softmax weights,

$$
\tilde{w}_{m, k}=\frac{\exp \left(a_{m, k}\right)}{\sum_{m^{\prime}=1}^{M} \sum_{k^{\prime}=1}^{K_{m^{\prime}}} \exp \left(a_{m^{\prime}, k^{\prime}}\right)}
$$

We initialise the logits as

$$
a_{m, k}=\log \left(w_{m, k}\right)+\operatorname{ELBO}\left(\boldsymbol{\phi}_{m}\right)-\max _{\substack{1 \leq m^{\prime} \leq M \\ 1 \leq k^{\prime} \leq K_{m^{\prime}}}} \left(\log \left(w_{m^{\prime}, k^{\prime}}\right)+\operatorname{ELBO}\left(\boldsymbol{\phi}_{m^{\prime}}\right)\right)
$$

where the last term ensures that $\max (\mathbf{a})=0$, preventing underflow in the exponentials. This is equivalent to initialising the weights as

$$
\tilde{w}_{m, k} \propto w_{m, k} \exp \left(\operatorname{ELBO}\left(\boldsymbol{\phi}_{m}\right)\right)
$$

effectively weighting each individual posterior by its approximate evidence (i.e., the exponential of the corresponding ELBO). This assigns a higher initial weight to the components coming from the best runs, providing a good starting point for the optimisation process.
We can then use the Adam optimiser (Kingma & Ba, 2014) to optimise the stacked ELBO. An overview of the procedure is outlined in Algorithm 1.

Notably, the optimisation can be performed as a pure post-processing step, requiring neither evaluations of the original likelihood $p(\mathcal{D} \mid \boldsymbol{\theta})$ nor of the surrogate models $f_{m}$, only that the estimates $\hat{I}_{m, k}$ are stored, as in current implementations (Huggins et al., 2023).

---

#### Page 7

#### Algorithm 1 S-VBMC

Input: outputs of $M$ VBMC runs $\left\{\boldsymbol{\phi}_{m}, \hat{\mathbf{I}}_{m}, \operatorname{ELBO}\left(\boldsymbol{\phi}_{m}\right)\right\}_{m=1}^{M}$

1. Stack the posteriors to obtain a mixture of $\sum_{m=1}^{M} K_{m}$ Gaussians
2. Reparametrise the weights with unconstrained logits $\mathbf{a}$
3. Initialise the logits as in Eq. 16
4. repeat
   (a) Compute the normalised weights $\tilde{\mathbf{w}}$ as in Eq. 15
   (b) Estimate $\mathrm{ELBO}_{\text {stacked }}(\tilde{\mathbf{w}})$ as in Eq. 11 with the entropy approximated as in Eq. 12
   (c) Compute $\frac{d \mathrm{ELBO}_{\text {stacked }}(\mathbf{a})}{d \mathbf{a}}$
   (d) Update $\mathbf{a}$ with a step of Adam (Kingma & Ba, 2014)
   until $\mathrm{ELBO}_{\text {stacked }}(\tilde{\mathbf{w}})$ converges
   return $\tilde{\boldsymbol{\phi}}, \mathrm{ELBO}_{\text {stacked }}$

Our stacking method hinges on the key feature of VBMC of providing accurate estimates $\hat{I}_{m, k}$. While in principle Eq. 11 could apply to any collection of variational posterior mixtures, without an efficient way of calculating each $I_{k}$ (Eq. 8), optimisation of the stacked ELBO would require many likelihood evaluations, which would be prohibitive for problems with expensive, black-box likelihoods.

In the following, we demonstrate the efficacy of this approach.

## 4 Experiments

In this section, we describe our experimental procedure (Section 4.1), baselines (Section 4.2), evaluation metrics (Section 4.3), and results for synthetic (Section 4.4) and real-world problems (Section 4.5). Finally, we briefly discuss computational costs in Section 4.6. Additional results are reported in the appendix, including additional experiments (Appendix A.3), more extensive tables of results (Appendix A.4), and example visualisations of posterior approximations (Appendix A.5).

### 4.1 Procedure

We first tested our method on two synthetic problems, designed to be particularly challenging for VBMC, and then on two real-world datasets and models (see below for full descriptions). We considered both noiseless problems (exact estimation) and noisy problems where Gaussian noise with $\sigma=3$ is applied to each log-likelihood measurement, emulating what practitioners might find when estimating the likelihood via simulation (van Opheusden et al., 2020). For each benchmark, we considered 100 VBMC runs obtained with the pyvbmc Python package (Huggins et al., 2023) that satisfied the following conditions:

1. the algorithm had converged (as assessed by the pyvbmc software);
2. $\max _{1 \leq k \leq K}\left(J_{k, k}\right)<5$, so all the estimates $\hat{I}_{k}$ had an associated variance lower than 5.

The reason for the latter is that S-VBMC's efficacy is strongly dependent on accurate VBMC estimates of the real $I_{k}$ from Bayesian quadrature (see Sections 2.3 and 3.2), and we sought to filter out "poorly converged" runs where this might not be the case. The results of this filtering procedure can be found in Appendix A.4.1. We performed all VBMC runs using the default settings (which always resulted in posteriors with 50 components, $K_{m}=50 \forall m \in\{1, \ldots, M\}$ ) and random uniform initialisation within plausible parameter bounds (Acerbi, 2018). To investigate the effect of combining a different number of posteriors, we adopted a bootstrapping approach: from these 100 runs, we randomly sampled and stacked

---

#### Page 8

with S-VBMC a varying number of runs (between 2 and 40) twenty times each, and computed the median with corresponding $95 \%$ confidence interval (computed from 10000 bootstrap resamples) for all metrics, described below. For all benchmark problems, the entropy is approximated as in Eq. 12 with $S=20$ during optimisation, and a final estimation (reported in all tables and figures) is performed with $S=100$ after convergence. The ELBO is optimised using Adam (Kingma & Ba, 2014) with learning rate set to 0.1.
All the experiments presented in this work were run on an AMD EPYC 7452 Processor with 16GB of RAM.

### 4.2 Baseline methods

**Black-box variational inference.** We used black-box variational inference (BBVI; Ranganath et al., 2014) as a baseline for all our benchmark problems.

Our implementation follows Li et al. (2025). For gradient-free black-box models, we cannot use the reparameterisation trick (Kingma & Welling, 2013) to estimate ELBO gradients. Instead, we employ the score function estimator (REINFORCE; Ranganath et al., 2014) with control variates to reduce gradient variance.
The variational posterior is parameterised as a mixture of Gaussians (MoG) with either $K=50$ or $K=500$ components, matching the form used in VBMC. We initialise the component means near the origin by adding Gaussian noise $(\sigma=0.1)$ and set all component variances to 0.01. We optimise the ELBO using Adam (Kingma & Ba, 2014) with stochastic gradients, performing a grid search over Monte Carlo sample sizes $\{1,10,100\}$ and learning rates $\{0.01,0.001\}$. We select the best hyperparameters based on the estimated ELBO.

For a fair comparison with S-VBMC, we set the target evaluation budget for a BBVI run to $2000(D+2)$ and $3000(D+2)$ evaluations for noiseless and noisy problems, respectively, matching the maximum evaluations used by 40 VBMC runs in total.

**Naive Stacking.** As a further baseline, we implemented a "naive" version of our stacking approach, consisting of a simple averaging of the individual VBMC posteriors. For this, we rewrite the stacked posterior as

$$
q_{\tilde{\boldsymbol{\phi}}}(\boldsymbol{\theta})=\sum_{m=1}^{M} \tilde{\omega}_{m} q_{\boldsymbol{\phi}_{m}}(\boldsymbol{\theta})
$$

and simply set

$$
\tilde{\omega}_{1}=\tilde{\omega}_{2}=\ldots=\tilde{\omega}_{m}=1 / M
$$

We call this approach "Naive Stacking" (NS).
As for S-VBMC, we ran NS by randomly sampling a varying number of runs (between 2 and 40) twenty times each, and then computed the median with corresponding $95 \%$ confidence intervals for all metrics.

### 4.3 Metrics

Following Acerbi (2020); Li et al. (2025), we evaluate our method using three metrics:

1. The absolute difference between true and estimated log marginal likelihood ( $\Delta \mathrm{LML}$ ), where values $<1$ are considered negligible for model selection (Burnham & Anderson, 2003).
2. The mean marginal total variation distance (MMTV), which measures the average (lack of) overlap between true and approximate posterior marginals across dimensions:

$$
\operatorname{MMTV}(p, q)=\frac{1}{2 D} \sum_{d=1}^{D} \int_{-\infty}^{\infty}\left|p_{d}\left(x_{d}\right)-q_{d}\left(x_{d}\right)\right| d x_{d}
$$

where $p_{d}$ and $q_{d}$ denote the marginal distributions along the $d$-th dimension.

---

#### Page 9

3. The "Gaussianised" symmetrised KL divergence (GsKL), which evaluates differences in means and covariances between the approximate and true posterior:

$$
\operatorname{GsKL}(p, q)=\frac{1}{2 D}\left[D_{\mathrm{KL}}(\mathcal{N}[p] \| \mathcal{N}[q])+D_{\mathrm{KL}}(\mathcal{N}[q] \| \mathcal{N}[p])\right]
$$

where $\mathcal{N}[p]$ denotes a Gaussian with the same mean and covariance as $p$.
We consider MMTV $<0.2$ and $\mathrm{GsKL}<\frac{1}{8}$ as target thresholds for reasonable posterior approximation (Li et al., 2025). Ground-truth estimates of the log marginal likelihood and posterior distributions are obtained through numerical integration, extensive MCMC sampling, or analytical methods as appropriate for each problem.

### 4.4 Synthetic problems

**GMM target.** Our synthetic GMM target consists of a mixture of 20 bivariate Gaussian components arranged in four distinct clusters. The cluster centroids were positioned at $(-8,-8),(-7,7),(6,-6)$ and $(5,5)$. Around each centroid, we placed five Gaussian components with means drawn from $\mathcal{N}\left(\boldsymbol{\mu}_{c}, \mathbf{I}\right)$, where $\boldsymbol{\mu}_{c}$ is the respective cluster centroid and $\mathbf{I}$ is the $2 \times 2$ identity matrix. Each component was assigned unit marginal variances and a correlation coefficient of $\pm 0.5$ (randomly selected with equal probability). This configuration produces an irregular mixture structure that requires a substantial number of components to approximate accurately. All components were assigned equal mixing weights. The resulting distribution is illustrated in Figure 2 (top panels).

> **Image description.** This image displays a grid of six 2D scatter plots, arranged in two rows and three columns. Each plot shares a dark purple background and common axis labels and ranges. The x-axis is labeled "$\theta_1$" and the y-axis is labeled "$\theta_2$", both ranging from -10 to 10 with major ticks at -10, 0, and 10. The plots illustrate the overlap between ground truth densities (depicted by color gradients and contours) and samples from posterior approximations (depicted by red points) for two different synthetic benchmarks.
>
> The top row of panels pertains to a "GMM target" distribution:
>
> - **Top-left panel (VBMC)**: The background shows four distinct, irregularly shaped density regions, colored with gradients from blue to green to yellow, with white contour lines. These regions are located roughly in the four quadrants of the plot: top-left, top-right, bottom-left, and bottom-right. The red points, representing the posterior approximation, are densely clustered in a single, irregular shape primarily within the bottom-right quadrant, showing limited overlap with the ground truth density.
> - **Top-middle panel (S-VBMC (5 posteriors))**: The background density regions are similar to the top-left panel. The red points are now distributed across all four quadrants, forming four distinct clusters. Each cluster of red points largely overlaps with one of the background density regions, indicating improved approximation compared to the VBMC method.
> - **Top-right panel (S-VBMC (20 posteriors))**: The background density regions remain consistent. The red points are again distributed across all four quadrants, forming four distinct clusters. The overlap between the red points and the background density regions appears even more complete and precise than in the S-VBMC (5 posteriors) panel, suggesting further improved approximation with more posteriors.
>
> The bottom row of panels pertains to a "ring target" distribution:
>
> - **Bottom-left panel (VBMC)**: The background features a single, thin, light grey-blue circular contour centred near (1, -2), with a radius of approximately 7-8 units. The red points, representing the posterior approximation, form a dense arc along the bottom-left portion of this circle, covering roughly one-quarter to one-third of the circumference, indicating a partial approximation of the ring.
> - **Bottom-middle panel (S-VBMC (5 posteriors))**: The background shows the same light grey-blue circular contour. The red points now form a more extensive arc along the circle, covering more than half of its circumference, but still leaving a noticeable gap in the approximation.
> - **Bottom-right panel (S-VBMC (20 posteriors))**: The background again shows the same light grey-blue circular contour. The red points now form a nearly complete circle, almost entirely overlapping the background contour, with only minimal or no visible gaps, demonstrating a highly accurate approximation of the ring target.

Figure 2: Examples of overlap between the ground truth and the posterior when combining different numbers of VBMC runs on the GMM (top panels) and ring (bottom panels) synthetic benchmarks. The red points indicate samples from the posterior approximation, with the target density depicted with colour gradients in the background.

In our experiments, we used both a noiseless version of this target (i.e., exact target evaluation) and a noisy one, where we applied i.i.d. Gaussian noise $(\sigma=3)$ to each log-likelihood evaluation.

**Ring target.** Our second synthetic target is a ring-shaped distribution defined by the probability density function

$$
p_{\text {ring }}\left(\theta_{1}, \theta_{2}\right) \propto \exp \left(-\frac{(r-R)^{2}}{2 \sigma^{2}}\right)
$$

---

#### Page 10

where $r=\sqrt{\left(\theta_{1}-c_{1}\right)^{2}+\left(\theta_{2}-c_{2}\right)^{2}}$ represents the radial distance from centre $\left(c_{1}, c_{2}\right), R$ is the ring radius, and $\sigma$ controls the width of the annulus. We set $R=8, \sigma=0.1$, and centred the ring at $\left(c_{1}, c_{2}\right)=(1,-2)$. The small value of $\sigma$ produces a narrow annular distribution that challenges VBMC's exploration capabilities. The resulting distribution is shown in Figure 2 (bottom panels).
As with the GMM target, we used both a noiseless version of this benchmark and a noisy one $(\sigma=3)$ in our experiments.

> **Image description.** This image is a multi-panel figure composed of 16 line graphs arranged in a 4x4 grid. Each row represents a different synthetic problem, and each column displays a different performance metric. The graphs plot metrics as a function of the "N. of runs" (number of VBMC runs stacked), with median values and 95% confidence intervals indicated by error bars.
>
> The four rows are labeled:
>
> - **(a) GMM (D = 2, noiseless)**
> - **(b) GMM (D = 2, σ = 3)**
> - **(c) Ring (D = 2, noiseless)**
> - **(d) Ring (D = 2, σ = 3)**
>
> The four columns represent the following metrics:
>
> - **ELBO** (evidence lower bound)
> - **Δ LML** (Difference in Log Marginal Likelihood)
> - **MMTV** (mean marginal total variation distance)
> - **GsKL** ("Gaussianised" symmetrised Kullback-Leibler divergence)
>
> All x-axes are labeled "N. of runs" and show tick marks at 1, 4, 8, 16, 24, 32, and 40. The y-axes for Δ LML and GsKL are on a logarithmic scale.
>
> Four different methods are plotted:
>
> - **S-VBMC** (blue line with circular markers)
> - **VBMC** (red line with upward-pointing triangular markers)
> - **NS** (grey line with downward-pointing triangular markers)
> - **BBVI** (a single green data point with a square marker, consistently plotted at N. of runs = 40)
>
> In the ELBO panels, a solid black horizontal line represents the ground-truth LML. In the Δ LML, MMTV, and GsKL panels, a dashed black horizontal line indicates a desirable threshold, with good performance being below this line.
>
> **Detailed observations for each row:**
>
> - **Row (a) GMM (D = 2, noiseless):**
>
>   - **ELBO:** S-VBMC (blue) and NS (grey) rapidly increase and converge towards the solid black ground-truth LML line, with S-VBMC slightly outperforming NS at higher run counts. VBMC (red) starts lower and shows slower convergence. BBVI (green) is very close to the ground-truth.
>   - **Δ LML:** All methods start with high values. S-VBMC (blue) shows a steep decrease, consistently dropping below the dashed threshold line for N. of runs greater than 8. NS (grey) also decreases but remains above S-VBMC. VBMC (red) stays high. BBVI (green) is very low, well below the threshold.
>   - **MMTV:** Similar to Δ LML, S-VBMC (blue) decreases sharply and goes below the dashed threshold line. NS (grey) decreases but stays higher. VBMC (red) remains high. BBVI (green) is very low.
>   - **GsKL:** Similar to Δ LML, S-VBMC (blue) decreases significantly, crossing the dashed threshold line. NS (grey) decreases but is higher than S-VBMC. VBMC (red) remains high. BBVI (green) is very low.
>
> - **Row (b) GMM (D = 2, σ = 3) (noisy):**
>
>   - **ELBO:** S-VBMC (blue) increases and plateaus slightly above the solid black ground-truth LML line. NS (grey) increases and converges closer to the ground-truth. VBMC (red) shows slower convergence. BBVI (green) is close to the ground-truth.
>   - **Δ LML:** S-VBMC (blue) decreases significantly, crossing the dashed threshold, but its values are generally higher than in the noiseless case. NS (grey) decreases but remains above S-VBMC. VBMC (red) stays high. BBVI (green) is low.
>   - **MMTV:** S-VBMC (blue) decreases, crossing the dashed threshold, with values generally higher than in the noiseless case. NS (grey) decreases but stays higher. VBMC (red) remains high. BBVI (green) is low.
>   - **GsKL:** S-VBMC (blue) decreases, crossing the dashed threshold, with values generally higher than in the noiseless case. NS (grey) decreases but is higher than S-VBMC. VBMC (red) remains high. BBVI (green) is low.
>
> - **Row (c) Ring (D = 2, noiseless):**
>
>   - **ELBO:** S-VBMC (blue) and NS (grey) rapidly increase and converge towards the solid black ground-truth LML line, with S-VBMC slightly outperforming NS at higher run counts. VBMC (red) starts lower and shows slower convergence. BBVI (green) is significantly lower than the ground-truth.
>   - **Δ LML:** S-VBMC (blue) decreases significantly, going below the dashed threshold. NS (grey) also decreases but stays above S-VBMC. VBMC (red) stays high. BBVI (green) is very high, well above the threshold.
>   - **MMTV:** S-VBMC (blue) decreases sharply and goes below the dashed threshold. NS (grey) decreases but stays higher. VBMC (red) remains high. BBVI (green) is high.
>   - **GsKL:** S-VBMC (blue) decreases significantly, crossing the dashed threshold. NS (grey) decreases but is higher than S-VBMC. VBMC (red) remains high. BBVI (green) is high.
>
> - **Row (d) Ring (D = 2, σ = 3) (noisy):**
>   - **ELBO:** S-VBMC (blue) increases and plateaus slightly above the solid black ground-truth LML line. NS (grey) increases and converges closer to the ground-truth. VBMC (red) shows slower convergence. BBVI (green) is significantly lower than the ground-truth.
>   - **Δ LML:** S-VBMC (blue) decreases, crossing the dashed threshold, with values generally higher than in the noiseless case. NS (grey) decreases but remains above S-VBMC. VBMC (red) stays high. BBVI (green) is very high.
>   - **MMTV:** S-VBMC (blue) decreases, crossing the dashed threshold, with values generally higher than in the noiseless case. NS (grey) decreases but stays higher. VBMC (red) remains high. BBVI (green) is high.
>   - **GsKL:** S-VBMC (blue) decreases, crossing the dashed threshold, with values generally higher than in the noiseless case. NS (grey) decreases but is higher than S-VBMC. VBMC (red) remains high. BBVI (green) is high.
>
> In summary, S-VBMC (blue) generally shows the best performance among the iterative methods, consistently reaching or surpassing the desirable thresholds for Δ LML, MMTV, and GsKL, and converging to the ground-truth LML for ELBO. BBVI (green) performs very well for GMM problems but struggles with Ring problems. VBMC (red) consistently shows the highest values for Δ LML, MMTV, and GsKL, indicating poorer performance compared to S-VBMC and NS. NS (grey) generally performs better than VBMC but not as well as S-VBMC. Error bars are visible for all data points, indicating the variability of the metrics.

Figure 3: Synthetic problems. Metrics plotted as a function of the number of VBMC runs stacked (median and $95 \%$ confidence interval). Metrics are plotted for S-VBMC (blue), VBMC (red), and NS (grey). The best BBVI results are shown in green. The black horizontal line in the ELBO panels represents the ground-truth LML, while the dashed lines on $\Delta \mathrm{LML}$, MMTV, and GsKL denote desirable thresholds for each metric (good performance is below the threshold; see Section 4.3)

**Results.** Results in Figure 3 and Table A.3 show that merging more posteriors leads to a steady improvement in the GsKL and MMTV metrics, which measure the quality of the posterior approximation. Remarkably, S-VBMC proves to be robust to noisy targets, with minor differences between noiseless and noisy settings. In all our synthetic benchmarks (both in noiseless and noisy settings), we observe that the values of the GsKL and MMTV metrics, which directly compare to the ground-truth target densities, reach good values - and start to plateau, or at least to improve at a much slower pace - by the time 10 VBMC

---

#### Page 11

posteriors are stacked. This suggests that a value of $M=10$ (or even $M \approx 6$ for noiseless problems, where metrics tend to converge faster) would be sufficient to obtain accurate stacked posteriors.

S-VBMC outperforms the BBVI baseline on the ring-shaped synthetic target. The BBVI baseline performs well and is only marginally worse compared to S-VBMC only on the GMM problem, where it effectively managed to capture the four clusters (see Figure 2 for a visualisation). While NS exhibits a similar improvement pattern with increasing values of $M$, S-VBMC consistently outperforms it, with differences being larger in noiseless settings.

As expected by design, individual VBMC runs tended to explore the two synthetic target distributions only partially, leading to poor performance. Still, the random initialisations allowed different runs to discover different portions of the posterior, allowing the merging process to cover the whole target (see Figure 2).

Finally, we observe that, in noisy settings, while the ELBO keeps increasing, the $\Delta$ LML error (difference between ELBO and true log marginal likelihood) initially decreases but then increases again as further components are added, a point which we will discuss later.

### 4.5 Real-world problems.

**Neuronal model.** Our first real-world problem involved fitting five biophysical parameters of a detailed compartmental model of a hippocampal CA1 pyramidal neuron. The model was constructed based on experimental data comprising a three-dimensional morphological reconstruction and electrophysiological recordings of neuronal responses to current injections. The deterministic neuronal responses were simulated using the NEURON simulation environment (Hines & Carnevale, 1997; Hines et al., 2009), applying current step inputs that matched the experimental protocol. The model's parameters characterise key biophysical properties: intracellular axial resistivity $\left(\theta_{1}\right)$, leak current reversal potential $\left(\theta_{2}\right)$, somatic leak conductance $\left(\theta_{3}\right)$, dendritic conductance gradient $\left(\theta_{4}\right.$, per $\mu \mathrm{m}$ ), and a dendritic surface scaling factor $\left(\theta_{5}\right)$. Based on independent measurements of membrane potential fluctuations, observation noise was modelled as a stationary Gaussian process with zero mean and a covariance function estimated from the data. The covariance structure was captured by the product of a cosine and an exponentially decaying function. For a similar approach applied to cerebellar Golgi cells, see Szoboszlay et al. (2016).

This model allowed exact log-likelihood evaluations, and no noise was added.
**Multisensory causal inference model.** Perceptual causal inference involves determining whether multiple sensory stimuli originate from a common source, a problem of particular interest in computational cognitive neuroscience (Körding et al., 2007). Our second real-world problem involved fitting a visuo-vestibular causal inference model to empirical data from a representative participant (S1 from Acerbi et al., 2018). In each trial of the modelled experiment, participants seated in a moving chair reported whether they perceived their movement direction $\left(s_{\text {vest }}\right)$ as congruent with an experimentally-manipulated looming visual field $\left(s_{\text {vis }}\right)$. The model assumes participants receive noisy sensory measurements, with vestibular information $z_{\text {vest }} \sim \mathcal{N}\left(s_{\text {vest }}, \sigma_{\text {vest }}^{2}\right)$ and visual information $z_{\text {vis }} \sim \mathcal{N}\left(s_{\text {vis }}, \sigma_{\text {vis }}^{2}(c)\right)$, where $\sigma_{\text {vest }}^{2}$ and $\sigma_{\text {vis }}^{2}$ represent sensory noise variances. The visual coherence level $c$ was experimentally manipulated across three levels $\left(c_{\text {low }}, c_{\text {med }}, c_{\text {high }}\right)$. The model assumes participants judge the stimuli as having a common cause when the absolute difference between sensory measurements falls below a threshold $\kappa$, with a lapse rate $\lambda$ accounting for random responses. The model parameters $\boldsymbol{\theta}$ comprise the visual noise parameters $\sigma_{\text {vis }}\left(c_{\text {low }}\right), \sigma_{\text {vis }}\left(c_{\text {med }}\right)$, $\sigma_{\text {vis }}\left(c_{\text {high }}\right)$, vestibular noise $\sigma_{\text {vest }}$, lapse rate $\lambda$, and decision threshold $\kappa$ (Acerbi et al., 2018).

We fitted this model assuming log-likelihood measurement noise ( $\sigma=3$, which we applied to each log-likelihood evaluation).

**Results.** The results in Figure 4 and Table A.4 confirm our earlier findings of improvements across the posterior metrics. We also find that S-VBMC is robust to noisy targets for real data, with performance that improves with the increasing number of stacked runs in the multisensory model problem.

Similar to what we observed for the synthetic problems, we find that stacking $\approx 10$ VBMC posteriors seems to be sufficient to vastly improve posterior quality (as indexed by the GsKL and MMTV metrics), compared to

---

#### Page 12

a single VBMC run $(M=1)$, with relatively small additional improvements for $M>10$. The only exception to this is the GsKL metric in the neuronal model, where the lower confidence intervals start to go below the desirable threshold (see Section 4.3) only if at least 18 posteriors are merged, and the full confidence interval never fully stabilises below such a threshold. However, this was our most challenging benchmark problem, as evidenced by the very poor performance obtained by both single-run VBMC and BBVI. Thus, while S-VBMC here does not achieve a posterior approximation fully on par with gold-standard MCMC (used to obtain the ground-truth posterior in this problem), the stacking process still vastly improves posterior quality at a much lower computational cost.

Finally, similarly to the synthetic problems, we see that S-VBMC performs consistently better than standard VBMC and BBVI across all metrics in both scenarios. Despite similar improvement patterns with increasing values of $M$, as observed above, S-VBMC consistently outperforms NS, with starker differences in the noisy setting (i.e., the multisensory model).

> **Image description.** A multi-panel figure composed of eight line graphs arranged in two rows and four columns, displaying various metrics as a function of the number of runs. A common legend is positioned at the bottom center of the figure.
>
> The figure is divided into two main sections, labeled (a) and (b), each representing a different model:
>
> **Panel (a): Neuronal model ($D=5$, noiseless)**
> This top row contains four sub-plots, all sharing the same x-axis labeled "N. of runs" with tick values 1, 4, 8, 16, 24, 32, and 40.
>
> - **ELBO (leftmost plot):**
>
>   - Y-axis: "ELBO" ranging from -7500 to -7450.
>   - A solid black horizontal line is present at approximately -7450, representing the ground-truth LML.
>   - S-VBMC (blue line with circular markers and error bars) starts around -7455 at N=1, rapidly increases, and stabilizes just below the black line, indicating convergence towards the ground-truth.
>   - VBMC (single red circular marker with error bar) is shown at N=1, around -7455.
>   - NS (grey line with square markers and error bars) generally follows the S-VBMC trend but slightly below it, also stabilizing near the ground-truth.
>   - BBVI (single green square marker with a large error bar) is plotted at N=40, significantly lower around -7495, with its error bar truncated at the bottom of the plot.
>
> - **Δ LML (second from left):**
>
>   - Y-axis: "Δ LML" on a logarithmic scale from 0.1 to 100.
>   - A dashed black horizontal line is present at 1, indicating a desirable threshold.
>   - S-VBMC (blue) starts around 3 at N=1, decreases sharply, and stabilizes below the dashed threshold, reaching values around 0.5 for higher N.
>   - VBMC (red) is at N=1, around 3.
>   - NS (grey) starts around 2.5 at N=1, decreases, and stabilizes just above or around the threshold of 1.
>   - BBVI (green) is at N=40, with a high value around 50, well above the threshold.
>
> - **MMTV (third from left):**
>
>   - Y-axis: "MMTV" on a linear scale from 0.0 to 0.6.
>   - A dashed black horizontal line is present at 0.2, indicating a desirable threshold.
>   - S-VBMC (blue) starts around 0.25 at N=1, decreases rapidly, and stabilizes significantly below the threshold, around 0.08.
>   - VBMC (red) is at N=1, around 0.3.
>   - NS (grey) starts around 0.2 at N=1, decreases, and stabilizes around 0.15, still below the threshold but higher than S-VBMC.
>   - BBVI (green) is at N=40, with a high value around 0.55, well above the threshold.
>
> - **GsKL (rightmost plot):**
>   - Y-axis: "GsKL" on a logarithmic scale from 0.1 to 100.
>   - A dashed black horizontal line is present at 0.1, indicating a desirable threshold.
>   - S-VBMC (blue) starts around 20 at N=1, decreases sharply, and stabilizes at or slightly below the threshold, around 0.1.
>   - VBMC (red) is at N=1, around 100.
>   - NS (grey) starts around 10 at N=1, decreases, and stabilizes around 0.5, above the threshold.
>   - BBVI (green) is at N=40, with a high value around 20, well above the threshold.
>
> **Panel (b): Multisensory ($D=6$, $\sigma=3$)**
> This bottom row also contains four sub-plots, sharing the same x-axis labeled "N. of runs" with identical tick values as panel (a).
>
> - **ELBO (leftmost plot):**
>
>   - Y-axis: "ELBO" ranging from -450 to -444.
>   - A solid black horizontal line is present at approximately -444, representing the ground-truth LML.
>   - S-VBMC (blue) starts around -445 at N=1, increases, and stabilizes just below the black line, indicating convergence.
>   - VBMC (red) is at N=1, around -445.
>   - NS (grey) generally follows the S-VBMC trend but stabilizes slightly lower, around -444.5.
>   - BBVI (green) is at N=40, significantly lower around -448.5, with a large error bar.
>
> - **Δ LML (second from left):**
>
>   - Y-axis: "Δ LML" on a logarithmic scale from 0.1 to 100.
>   - A dashed black horizontal line is present at 1, indicating a desirable threshold.
>   - S-VBMC (blue) starts around 0.5 at N=1, increases, and stabilizes above the threshold, around 1.5.
>   - VBMC (red) is at N=1, around 0.5.
>   - NS (grey) starts around 0.7 at N=1, increases, and stabilizes below the threshold, around 0.8.
>   - BBVI (green) is at N=40, with a value around 2, above the threshold.
>
> - **MMTV (third from left):**
>
>   - Y-axis: "MMTV" on a linear scale from 0.00 to 0.20.
>   - A dashed black horizontal line is present at 0.2, indicating a desirable threshold.
>   - S-VBMC (blue) starts around 0.15 at N=1, decreases rapidly, and stabilizes significantly below the threshold, around 0.08.
>   - VBMC (red) is at N=1, around 0.18.
>   - NS (grey) starts around 0.12 at N=1, decreases, and stabilizes around 0.1, below the threshold but higher than S-VBMC.
>   - BBVI (green) is at N=40, with a value around 0.11, below the threshold.
>
> - **GsKL (rightmost plot):**
>   - Y-axis: "GsKL" on a logarithmic scale from 0.01 to 1.
>   - A dashed black horizontal line is present at 0.1, indicating a desirable threshold.
>   - S-VBMC (blue) starts around 0.15 at N=1, decreases sharply, and stabilizes significantly below the threshold, around 0.03.
>   - VBMC (red) is at N=1, around 0.2.
>   - NS (grey) starts around 0.1 at N=1, decreases, and stabilizes around 0.06, below the threshold but higher than S-VBMC.
>   - BBVI (green) is at N=40, with a value around 0.15, above the threshold.
>
> **Legend (bottom center):**
> The legend defines the visual representations for four different methods, each with an associated colored square and error bar icon:
>
> - BBVI: Green square with error bar
> - NS: Grey square with error bar
> - VBMC: Red square with error bar
> - S-VBMC: Blue square with error bar
>
> All data points for S-VBMC and NS are connected by lines, indicating their performance across varying numbers of runs. VBMC and BBVI are shown as single points with error bars, representing their performance at specific run counts (N=1 for VBMC, N=40 for BBVI).

Figure 4: Real-world problems. Metrics plotted as a function of the number of VBMC runs stacked (median and $95 \%$ confidence interval). Metrics are plotted for S-VBMC (blue), VBMC (red), and NS (grey). The best BBVI results are shown in green. The black horizontal line in the ELBO panels represents the ground-truth LML, while the dashed lines on $\Delta$ LML, MMTV, and GsKL denote desirable thresholds for each metric (good performance is below the threshold; see Section 4.3). The BBVI error bar in the plot displaying the ELBO in the neuronal model (top left) is truncated for clarity.

### 4.6 Computational overhead

Here we present details about the additional computational cost (quantified as compute time) introduced by S-VBMC on top of VBMC. Additional runtime analyses can be found in Appendix A.3.2.

Figure 5 illustrates how S-VBMC introduces a relatively small computational overhead, even when comparing the post-process cost of S-VBMC with the average cost of one VBMC run, under the idealised condition where the $M$ VBMC runs happen all in parallel.[^0] In particular, running our algorithm with $M \approx 10$ - which vastly improves the resulting posterior, as shown above and in Appendices A.4.2 and A.5 - adds a small amount of post-processing time to VBMC for all our benchmark problems ( $\approx 5$-15\% overhead).

Put together, our results confirm that S-VBMC yields high returns in terms of inference performance at a very marginal cost in terms of compute time.

[^0]: In practice, completing $M$ VBMC runs will be more expensive due to additional parallelisation costs, making S-VBMC's relative overhead even smaller than what we report here.

---

#### Page 13

> **Image description.** A multi-panel plot, arranged in two rows and three columns, displays six line graphs showing "Compute time (s)" on the y-axis against "N. of runs" on the x-axis. Each subplot represents a different benchmark problem, indicated by its title. All subplots share common axis labels and a consistent visual style, including a light gray grid.
>
> **Common Elements Across Subplots:**
>
> - **X-axis**: Labeled "N. of runs", ranging from 0 to 40. Major tick marks are present at 1, 4, 8, 16, 24, 32, and 40.
> - **Y-axis**: Labeled "Compute time (s)". The range and major tick marks vary for each subplot.
> - **Data Representation**: Each subplot contains two sets of data points:
>   - A single red circular data point with a vertical error bar, representing "VBMC". This point is consistently plotted at N. of runs = 1.
>   - A series of blue circular data points, connected by implied lines, each with a vertical error bar, representing "S-VBMC (post-process)". These points show an increasing trend as "N. of runs" increases.
>
> **Legend:**
> A legend is positioned below the bottom-center subplot:
>
> - <span style="color:red">**&#x25CF;**</span> <span style="color:red">**I**</span> VBMC
> - <span style="color:blue">**&#x25CF;**</span> <span style="color:blue">**I**</span> S-VBMC (post-process)
>
> **Individual Subplot Descriptions:**
>
> 1.  **Top-Left Panel:**
>
>     - **Title**: GMM ($D=2$, noiseless)
>     - **Y-axis**: Ranges from 0 to 150, with major ticks at 0, 50, 100, 150.
>     - **VBMC (red)**: Approximately (1, 85) with an error bar spanning roughly 70 to 125.
>     - **S-VBMC (blue)**: Starts near (1, 5) and increases in a curvilinear fashion to approximately (40, 150). Error bars generally widen with increasing N. of runs.
>
> 2.  **Top-Middle Panel:**
>
>     - **Title**: Ring ($D=2$, noiseless)
>     - **VBMC (red)**: Approximately (1, 190) with an error bar spanning roughly 140 to 240.
>     - **S-VBMC (blue)**: Starts near (1, 5) and increases in a curvilinear fashion to approximately (40, 200).
>
> 3.  **Top-Right Panel:**
>
>     - **Title**: Neuronal model ($D=5$, noiseless)
>     - **VBMC (red)**: Approximately (1, 650) with an error bar spanning roughly 500 to 800.
>     - **S-VBMC (blue)**: Starts near (1, 10) and increases in a curvilinear fashion to approximately (40, 800).
>
> 4.  **Bottom-Left Panel:**
>
>     - **Title**: GMM ($D=2$, $\sigma=3$)
>     - **VBMC (red)**: Approximately (1, 240) with an error bar spanning roughly 170 to 330.
>     - **S-VBMC (blue)**: Starts near (1, 10) and increases in a curvilinear fashion to approximately (40, 450).
>
> 5.  **Bottom-Middle Panel:**
>
>     - **Title**: Ring ($D=2$, $\sigma=3$)
>     - **Y-axis**: Ranges from 0 to 800, with major ticks at 0, 200, 400, 600, 800.
>     - **VBMC (red)**: Approximately (1, 540) with an error bar spanning roughly 400 to 780.
>     - **S-VBMC (blue)**: Starts near (1, 20) and increases in a curvilinear fashion to approximately (40, 500).
>
> 6.  **Bottom-Right Panel:**
>     - **Title**: Multisensory ($D=6$, $\sigma=3$)
>     - **VBMC (red)**: Approximately (1, 850) with an error bar spanning roughly 650 to 1100.
>     - **S-VBMC (blue)**: Starts near (1, 20) and increases in a curvilinear fashion to approximately (40, 2300).
>
> The plots visually compare the compute time of VBMC (a single run) against the post-processing computational overhead of S-VBMC as more runs are stacked, across different models and conditions.

Figure 5: Compute time of a single VBMC run (red) and post-processing time only (i.e., computational overhead) of S-VBMC (blue) plotted as a function of the number of VBMC runs stacked (median and $95 \%$ confidence interval, computed from 10000 bootstrap resamples). Each subplot represents a different benchmark problem. The values plotted here correspond to the actual computation times of the experiments described in Sections 4.4 and 4.5.

## 5 ELBO estimation bias

As briefly mentioned in Section 4.4, in our results we observe that, in noisy settings, while the ELBO keeps increasing as more VBMC runs are stacked, the $\Delta \mathrm{LML}$ error also increases, after an initial decrease (see the second column of Figures 3 and 4). This apparently odd result can be explained by the presence of a positive bias build-up in the estimated ELBO. This bias is visible in Figure 3 (b) and (d), first column, and Figure 4 (b), first column, in that the estimated ELBO from S-VBMC on these problems slightly "overshoots" the ground truth LML (the S-VBMC estimates, blue dots, end above the black horizontal line). As discussed in Section 2.2, the ELBO is always lower than the LML, with equality when the approximate posterior perfectly corresponds to the true posterior. However, in these results, the estimated ELBO grows larger than the ground-truth LML. As this cannot be true, there must be a positive bias in the ELBO estimate. As one can see in the aforementioned figures (first column), this bias builds up as more and more VBMC runs are stacked. What characterises these problems is the fact that they use a stochastic estimator for the likelihood (as opposed to exact likelihood evaluations used in the other problems, where the estimate does not overshoot).

While this bias surprisingly does not affect other posterior quality metrics, which keep improving (or plateau) with increasing $M$, it might constitute an issue when using $\mathrm{ELBO}_{\text {stacked }}$ for model comparison. In this section, we provide a simplified model for how the bias might statistically arise in terms of the "winner's curse" and then provide an effective heuristic to counteract the bias.

### 5.1 Origin of bias

Here we analyse the ELBO overestimation observed in our results through a simplified example that illustrates one potential mechanism for this bias. In short, we suggest this occurs because all $I_{m, k}$ are noisy estimates of the true expected log-joint contributions, causing S-VBMC to overweigh the most overestimated

---

#### Page 14

mixture components - an effect that increases with the number of components $M$. While other factors may contribute, this analysis provides insight into why the bias tends to increase with the number of merged VBMC runs.

For the sake of argument, consider $M$ VBMC runs that return identical posteriors $q_{\boldsymbol{\phi}_{1}}(\boldsymbol{\theta})=\ldots=q_{\boldsymbol{\phi}_{M}}(\boldsymbol{\theta})$, each with a single component. The stacked posterior takes the form:

$$
q_{\tilde{\boldsymbol{\phi}}}(\boldsymbol{\theta})=\sum_{m=1}^{M} \tilde{w}_{m} q_{\boldsymbol{\phi}_{m}}(\boldsymbol{\theta})
$$

For each single-component posterior, the expected log-joint is approximated as

$$
I_{m}=\mathbb{E}_{q_{\boldsymbol{\phi}_{m}}}\left[\log p(\mathcal{D} \mid \boldsymbol{\theta}) p(\boldsymbol{\theta})\right] \approx \mathbb{E}_{q_{\boldsymbol{\phi}_{m}}}\left[f_{m}(\boldsymbol{\theta})\right]
$$

where $f_{m}(\boldsymbol{\theta})$ is the surrogate log-joint from the $m$-th VBMC run. Since all posteriors share identical parameters, their entropies are equal:

$$
\mathcal{H}\left[q_{\boldsymbol{\phi}_{1}}(\boldsymbol{\theta})\right]=\mathcal{H}\left[q_{\boldsymbol{\phi}_{2}}(\boldsymbol{\theta})\right]=\ldots=\mathcal{H}\left[q_{\boldsymbol{\phi}_{M}}(\boldsymbol{\theta})\right]
$$

The stacked posterior is thus a mixture of identical components with different associated values $I_{m}$. The optimal mixture weights $\tilde{\mathbf{w}}$ depend solely on the noisy estimates of $I_{m}$ :

$$
\hat{I}_{m}=\mathbb{E}_{q_{\boldsymbol{\phi}_{m}}}\left[f_{m}(\boldsymbol{\theta})\right]=\mathbb{E}_{q_{\boldsymbol{\phi}_{m}}}\left[\log p(\mathcal{D} \mid \boldsymbol{\theta}) p(\boldsymbol{\theta})\right]+\epsilon_{m}
$$

where $\epsilon_{m} \sim \mathcal{N}\left(0, J_{m}\right)$ represents estimation noise with variance $J_{m}$. Since all posteriors are identical and derived from the same data and model, differences in expected log-joint estimates arise purely from noise deriving from the Gaussian process surrogates $f_{m}$.

Given that entropy remains constant under merging, in this scenario optimising $\mathrm{ELBO}_{\text {stacked }}$ reduces to selecting the posterior with the highest expected log-joint estimate. If we denote $\hat{I}_{\max }=\max _{m} \hat{I}_{m}$, the optimal ELBO becomes

$$
\mathrm{ELBO}_{\text {stacked }}^{*}=\hat{I}_{\max }+\mathcal{H}\left[q_{\tilde{\boldsymbol{\phi}}}(\boldsymbol{\theta})\right]
$$

Since the true expected log-joint is identical across posteriors, the optimisation selects the most overestimated value. The magnitude of this overestimation increases with both $M$ and the observation noise for $f_{m}$, introducing a positive bias in $\mathrm{ELBO}_{\text {stacked }}^{*}$ that grows with the number of stacked runs and is more substantial for surrogates obtained from noisy log-likelihood observations.

While this simplified scenario does not capture the complexity of practical applications - where posteriors have multiple, non-overlapping components - it illustrates a fundamental issue: if we model each $\hat{I}_{m, k}$ as the sum of the true $I_{m, k}$ and noise, the merging process will favour overestimated components, biasing the final $\mathrm{ELBO}_{\text {stacked }}$ estimate upward.

This hypothesis is substantiated by our results, as we only observe a noticeable bias in problems with noisy targets, where levels of noise in the VBMC estimation of $I_{m, k}$ are non-negligible (note that VBMC outputs an estimate of such noise, see Section 2).

### 5.2 Debiasing heuristic

We propose here a heuristic approach to counteract the bias in the ELBO estimate.
First, as a baseline we can estimate the per-component ground-truth expected log-joint $I_{m, k}$, and the full expected log-joint summed over components, using Monte Carlo sampling on the true log-joint instead of the VBMC estimates (see Eqs. 8 and 11). This estimate is unbiased and does not increase with $M$, as opposed to the one estimated by S-VBMC, as shown for all noisy benchmarks in Figure 6. However, this requires numerous additional evaluations of the model's log-likelihood, which is assumed to be expensive, so it is not in general a viable solution.

---

#### Page 15

One potentially effective heuristic to counteract the bias - or at least prevent it from increasing with $M$ would be to cap its value post-hoc to some reasonable quantity. This heuristic has the advantage of being straightforward and inexpensive to implement, and, crucially, of not involving any tweaks to the ELBO optimisation process, thus not requiring any additional hyperparameters.

> **Image description.** A multi-panel line graph titled "Expected log-joint (S-VBMC)" displays three individual plots arranged horizontally, each comparing two data series with error bars across varying numbers of runs.
>
> Each of the three panels shares a common x-axis labeled "N. of runs", with tick marks at 2, 4, 8, 16, 24, 32, and 40. The y-axis for all panels is labeled "Expected log-joint", but the numerical range differs for each plot. Two data series are plotted in each panel: one represented by blue circular markers with blue error bars, and the other by black circular markers with black error bars.
>
> The panels are as follows:
>
> 1.  **Left Panel: "GMM, σ = 3"**
>
>     - The y-axis ranges from -1.6 to -0.8, with major ticks at -1.6, -1.4, -1.2, -1.0, and -0.8.
>     - The blue data series starts around -1.5 and shows a clear increasing trend, rising to approximately -0.85 by 40 runs. The error bars are visibly larger at lower run counts.
>     - The black data series remains relatively flat, hovering around -1.6, with smaller error bars.
>
> 2.  **Middle Panel: "Ring, σ = 3"**
>
>     - The y-axis ranges from -0.8 to 0.0, with major ticks at -0.8, -0.6, -0.4, -0.2, and 0.0.
>     - The blue data series starts around -0.5 and exhibits a consistent increasing trend, reaching approximately -0.05 by 40 runs. Error bars are present and appear to shrink slightly as the number of runs increases.
>     - The black data series maintains a relatively stable value around -0.8, with consistently small error bars.
>
> 3.  **Right Panel: "Multisensory, σ = 3"**
>     - The y-axis ranges from -452.5 to -451.0, with major ticks at -452.5, -452.0, -451.5, and -451.0.
>     - The blue data series begins around -452.0 and shows an upward trend, stabilizing near -451.05 at 40 runs. Error bars are visible.
>     - The black data series stays relatively constant around -452.5, displaying small error bars.
>
> Below the three panels, a legend clarifies the data series:
>
> - A blue circular marker with a horizontal error bar represents "E_stacked".
> - A black circular marker with a horizontal error bar represents "E_MC".
>
> Across all three plots, the blue data series consistently shows an increasing trend as the "N. of runs" increases, while the black data series remains relatively stable, suggesting a baseline or ground truth value. The blue series generally has larger error bars, particularly at lower run counts, compared to the black series.

Figure 6: Expected log-joint as estimated by S-VBMC ( $E_{\text {stacked }}$, blue) compared to that estimated via Monte Carlo sampling with numerous additional evaluations of the true log-joint ( $E_{\mathrm{MC}}$, black, which we use as the ground truth) for all our experiments with noisy targets. Dots represent the median value (of the 20 S-VBMC runs) and error bars $95 \%$ confidence intervals (computed from 10000 bootstrap resamples).

If we define the expected log-joint as estimated by S-VBMC as

$$
E_{\text {stacked }}=\sum_{m=1}^{M} \sum_{k=1}^{K_{m}} \tilde{w}_{m, k} \hat{I}_{m, k}
$$

the 'capped' ELBO will be

$$
\operatorname{ELBO}_{\text {stacked }}^{(\text {capped })}=\widehat{E}_{\text {stacked }}+\mathcal{H}\left[q_{\tilde{\boldsymbol{\phi}}}(\boldsymbol{\theta})\right]
$$

where

$$
\widehat{E}_{\text {stacked }}=\min \left(E_{\text {stacked }}, E_{\text {cap }}\right)
$$

Note that, as mentioned in Section 3.1, each VBMC run performs its own parameter transformation $g_{m}(\cdot)$ during inference (Acerbi, 2020), which prevents meaningful comparisons of the expected log-joint and its components across runs. Throughout this section, we refer to and use the corrected VBMC estimates $\hat{I}_{m, k}$ as expressed in the common parameter space of $\boldsymbol{\theta}$. More details on this correction can be found in Appendix A.2. In what follows, we discuss two candidates for $E_{\text {cap }}$.

**Capping with median expected log-joint (run-wise).** As a possible candidate for $E_{\text {cap }}$, we considered the median of the VBMC estimates of the expected log-joint from individual runs. So, if

$$
E_{m}=\sum_{k=1}^{K_{m}} w_{m, k} \hat{I}_{m, k}
$$

is the VBMC estimate of the expected log-joint from the $m$-th individual run, then

$$
E_{\text {median }}=\operatorname{median}_{1 \leq m \leq M}\left(E_{m}\right)
$$

---

#### Page 16

**Capping with median expected log-joint (component-wise).** As the value of $E_{\text {median }}$ might be unstable and heavily dependent on individual VBMC runs (especially for low values of $M$ ), we also consider the median of the expected log-joints with respect to all individual components,

$$
I_{\text {median }}=\underset{\substack{1 \leq m \leq M \\ 1 \leq k \leq K_{m}}}{\operatorname{median}}\left(\hat{I}_{m, k}\right)
$$

As each VBMC run typically has $K_{m}=50$, even with $M=2$ we would have a stacked posterior with 100 components, which should ensure more stability in the estimate.

**Debiasing results.** We applied both these corrections to all our experiments with noisy targets (described in Sections 4.4 and 4.5), with results shown in Figure 7. We observe that both solutions are effective in preventing the ELBO bias buildup for all three benchmark problems. The bias itself is still present for the ring and multisensory targets, but it remains roughly constant with increasing values of $M$, and in both cases is very limited. We also observe that using $I_{\text {median }}$ as $E_{\text {cap }}$ yields the most stable solution, with debiased ELBO values fluctuating less, and confidence intervals being considerably less wide. While this is true for all three scenarios, it is particularly evident in the multisensory benchmark, where the ELBO capped with $E_{\text {median }}$ tends to fluctuate wildly, sometimes leading to ELBO overestimation, and sometimes to ELBO underestimation. In contrast, capping with $I_{\text {median }}$ ensures a reliably contained bias $(<0.5)$. These results suggest that the latter method is more reliable across benchmarks and should therefore be preferred by practitioners.

> **Image description.** A multi-panel line graph titled "ELBO debiasing (S-VBMC)" displays the effects of different debiasing heuristics on noisy targets across varying numbers of runs. The figure consists of three subplots arranged horizontally, each representing a different benchmark problem: "GMM, $\sigma = 3$", "Ring, $\sigma = 3$", and "Multisensory, $\sigma = 3$". All three subplots share a common x-axis labeled "N. of runs" and a common y-axis labeled "ELBO". A legend at the bottom of the figure clarifies the different data series.
>
> Each subplot shows four distinct data series, represented by different colored markers and associated vertical error bars, plotted against the "N. of runs" (ranging from 2 to 40, with tick marks at 2, 4, 8, 16, 24, 32, 40). A horizontal black dotted line is present in each panel, representing a reference ELBO value, identified as ELBO$_{MC}$ from the context. The error bars represent 95% confidence intervals.
>
> **Panel 1: GMM, $\sigma = 3$**
>
> - The "No capping" series (blue circles) starts around 2.4 at 2 runs, increases steeply, and then plateaus around 3.7, significantly above the reference dotted line (at ELBO = 3.0). Its error bars are relatively small.
> - The "$E_{\text{cap}} = E_{\text{median}}$" series (orange circles) starts around 2.5, shows some fluctuation, and then stabilizes around 2.9-3.0, close to the reference line. This series exhibits larger error bars, particularly at lower run counts.
> - The "$E_{\text{cap}} = I_{\text{median}}$" series (purple triangles) starts around 2.7, increases, and then stabilizes consistently around 2.9, also close to the reference line. Its error bars are moderate, generally smaller than the orange series.
> - The "ELBO$_{MC}$" series (black circles) starts around 2.3, increases, and then stabilizes around 2.9, closely tracking the reference dotted line. Its error bars are the smallest among all series.
>
> **Panel 2: Ring, $\sigma = 3$**
>
> - The "No capping" series (blue circles) starts around 1.6 at 2 runs, increases sharply, and then plateaus around 3.1, well above the reference dotted line (at ELBO = 2.25). Its error bars are small.
> - The "$E_{\text{cap}} = E_{\text{median}}$" series (orange circles) starts around 1.8, shows considerable fluctuation, and then stabilizes around 2.4-2.5, slightly above the reference line. This series has very large error bars, especially at lower run counts.
> - The "$E_{\text{cap}} = I_{\text{median}}$" series (purple triangles) starts around 1.5, increases, and then stabilizes around 2.4, also slightly above the reference line. Its error bars are moderate, larger than the black and blue series but smaller than the orange.
> - The "ELBO$_{MC}$" series (black circles) starts around 1.3, increases, and then stabilizes around 2.2, closely tracking the reference dotted line. Its error bars are small.
>
> **Panel 3: Multisensory, $\sigma = 3$**
>
> - The "No capping" series (blue circles) starts around -445.0 at 2 runs, increases rapidly, and then plateaus around -443.7, significantly above the reference dotted line (at ELBO = -444.7). Its error bars are small.
> - The "$E_{\text{cap}} = E_{\text{median}}$" series (orange circles) starts around -444.8, exhibits extreme fluctuations, and then stabilizes around -444.4, noticeably above the reference line. This series displays exceptionally large error bars across all run counts.
> - The "$E_{\text{cap}} = I_{\text{median}}$" series (purple triangles) starts around -444.7, increases, and then stabilizes around -444.4, also above the reference line. Its error bars are moderate, significantly smaller than the orange series but larger than the blue and black series.
> - The "ELBO$_{MC}$" series (black circles) starts around -445.2, increases, and then stabilizes around -444.7, closely tracking the reference dotted line. Its error bars are small.
>
> **Legend:**
> The legend, positioned below the three panels, uses horizontal error bar symbols to represent the data series:
>
> - A blue circle with a horizontal error bar is labeled "No capping".
> - An orange circle with a horizontal error bar is labeled "$E_{\text{cap}} = E_{\text{median}}$".
> - A purple triangle with a horizontal error bar is labeled "$E_{\text{cap}} = I_{\text{median}}$".
> - A black circle with a horizontal error bar is labeled "ELBO$_{\text{MC}}$".

Figure 7: Effects of debiasing heuristics on noisy targets, with dots representing the median value and error bars $95 \%$ confidence intervals (computed from 10000 bootstrap resamples) across 20 S-VBMC runs. Each panel displays the uncorrected ELBO as output by S-VBMC (blue), the capped ELBO using $E_{\text {median }}$ (orange), the capped ELBO using $I_{\text {median }}$ (purple), and the ground-truth ELBO obtained via Monte Carlo $\mathrm{ELBO}_{\mathrm{MC}}$ (black), for distinct benchmark problems. In each panel, the black dotted line is the ground-truth log marginal likelihood.

## 6 Discussion

The core problem we tackled in this work is that certain target posteriors might have properties (e.g., multimodality, long tails, long and narrow shapes) that pose significant challenges to global surrogate-based approaches to Bayesian inference like VBMC. VBMC in particular was developed to tackle problems with expensive likelihood functions, and thus relies on a limited target evaluation budget and an active sampling strategy (Acerbi, 2019), which, while effective in many cases (Acerbi, 2018; 2020), makes it vulnerable to leaving high-density probability regions unexplored (e.g., missing a mode). Therefore, attempting to build

---

#### Page 17

a single, global surrogate approximation of complex targets might be counter-productive. Our proposed solution relies on the idea that, as an alternative, we can rely on a collection of local surrogates to build a better, global posterior.

In this work, we introduced S-VBMC, a simple approach for merging independent VBMC runs in a principled way to yield a global posterior approximation. We tested its effectiveness on synthetic problems that we expected VBMC to have trouble with, such as targets with multiple modes and long, narrow shapes, as well as on two real-world problems from computational neuroscience. We also probed S-VBMC's robustness to noise, by adding Gaussian noise to log-likelihood evaluations for some of our benchmark targets. This approach mimics realistic scenarios where the log-likelihood itself is not available, but can be estimated via simulation (Wood, 2010; van Opheusden et al., 2020; Järvenpää et al., 2021; Acerbi, 2020). As expected, our results show that individual VBMC runs fail to yield good posterior approximations in our challenging problems. Conversely, S-VBMC is remarkably effective across benchmarks, with all metrics steadily improving as more VBMC runs are merged, with minor differences between noisy and noiseless settings. Importantly, we built S-VBMC to be robust to variability in the quality of individual VBMC posteriors. We initialise the weights $\tilde{\mathbf{w}}$ using each run's ELBO (Eq. 16), which immediately downweights low-ELBO runs, and the subsequent optimisation further reduces their influence. To verify this robustness, in all benchmarks we deliberately retained low-ELBO (but converged) runs and only excluded those flagged as non-converged by pyvbmc (Huggins et al., 2023) and poorly converged ones, with large uncertainty associated with the estimates $\hat{I}_{k}$ (see Section 4.1). Despite this, the stacked posterior still improved as more runs were combined, suggesting that low-ELBO VBMC posteriors are naturally assigned negligible weight.

S-VBMC inherits the limitations of VBMC and other surrogate-based inference methods, such as applicability to relatively low-dimensional target posteriors (up to $\approx 10$ dimensions; see Acerbi, 2018; 2020; Järvenpää et al., 2021). While this fundamental constraint of the Gaussian process surrogate remains, it is possible that, in moderately high dimensions $(D \approx 10-15)$, diverse initialisations might allow different runs to capture complementary regions of the posterior, yielding incremental gains even when each surrogate is only locally accurate. However, we do not expect S-VBMC to overcome the core scaling issues of the surrogate itself, and a careful empirical study of higher-dimensional cases is left for future work. With that being said, S-VBMC effectively addresses some of VBMC's limitations, such as dealing with more complex posterior shapes and multimodality.

Our results show that S-VBMC represents a practical and effective approach for performing approximate Bayesian inference when the target log-likelihood is expensive or the posterior distribution exhibits specific features. In scenarios where the likelihood is fast to evaluate and noiseless, and the posterior largely unimodal, traditional inference methods such as Markov Chain Monte Carlo (MCMC) are likely to outperform S-VBMC and should remain the default choice. While methods like MCMC represent a gold standard in most cases, it is worth noting that standard MCMC algorithms can struggle to efficiently explore complex posteriors with multiple, well-separated modes, for which there is no off-the-shelf, easy solution. In contrast, S-VBMC's strategy of combining multiple independent local approximations, when paired with a diverse set of starting points, makes it better suited to explore challenging global structures. This suggests S-VBMC may be viable even in scenarios where the likelihood is not prohibitively expensive, but the posterior geometry is difficult for sequential samplers to navigate due to isolated modes.

Moreover, S-VBMC integrates seamlessly into the established best practices for its parent algorithm. For robustness and convergence diagnostics, performing several independent VBMC runs from different starting points is already recommended (Huggins et al., 2023). S-VBMC leverages this existing diagnostic workflow, providing a principled method to not only assess the inference landscape but to combine these runs into an improved, global posterior approximation at a small added computational cost. We intentionally designed S-VBMC as a simple post-processing step that requires no modification to the core VBMC algorithm. This simplicity is a key strength, as it preserves the embarrassingly parallel nature of running multiple independent inferences. While one could devise more sophisticated methods involving, for example, communication between runs to promote diversity and exploration, such approaches would sacrifice the ease of implementation and seamless integration that make S-VBMC a practical tool for practitioners. As our results show, this approach is most impactful for problems with features like multiple modes (GMM target), long and narrow shapes (ring target), or heavy tails (neuronal model, see Appendix A.5), where individual runs explore different facets of the posterior.

---

#### Page 18

To investigate the practical convenience of our approach, we measured and reported the computational overhead of S-VBMC. Our results, combined with our previous analyses, suggest that our approach yields considerable improvements in terms of posterior quality with a small added cost in terms of compute time, assuming VBMC runs can be executed in parallel. In light of this, we recommend using $M=10$ as a default choice, as it offers a strong accuracy-cost trade-off: most of the gains materialise by $M=10$, which in our experiments added a modest $\approx 5-15 \%$ overhead cost, assuming the VBMC runs are executed in parallel. As mentioned earlier, our method is, in principle, applicable for stacking mixture posteriors produced by any inference scheme, not strictly limited to VBMC. However, without a closed-form solution for the expected log-joints of the individual components of each run $\left(\left\{\mathbf{I}_{m}\right\}_{m=1}^{M}\right.$, estimates of which are available in VBMC), Monte Carlo estimates are required, greatly inflating the number of necessary likelihood evaluations, and thus the computational cost. This wouldn't be the case for inexpensive likelihoods, but, as discussed above, VBMC would not necessarily be the primary recommended approach.

Finally, we discussed the main noticeable downside of S-VBMC, namely an observed bias in the ELBO estimation building up with increasing numbers of merged VBMC runs in problems with noisy targets, and proposed a simple heuristic (applicable in an inexpensive post-processing step) that mitigates this. Even though this bias buildup didn't seem to affect posterior quality in our experiments, it would constitute a problem if one were to use the (inflated) ELBO estimates for model comparison. Our debiasing approach reduces this bias to a smaller magnitude and prevents it from building up, so we encourage practitioners to adopt it (or some other debiasing technique) when using S-VBMC estimates of the ELBO for model comparison in problems with noisy likelihoods.

We should note that, although we discussed a plausible candidate source for the ELBO bias build-up in Section 5.1, this work does not contain a thorough investigation of this phenomenon and its causes. Some of the bias could come from the internals of the VBMC algorithm itself, which could explain the presence of a residual bias (see Section 5.2). Furthermore, our debiasing method simply consists of a post-processing heuristic, which is attractive for its simplicity and speed, but preserves the bias build-up mechanism during optimisation and inference. Precise identification of bias sources could allow their neutralisation at inference time, possibly leading the algorithm to shift its focus from overestimating the expected log-joint to optimising an unbiased stacked ELBO. Nonetheless, an interesting finding is that in our benchmark problems the bias phenomenon did not affect posterior approximation quality as measured by our metrics, which kept improving steadily and approaching ground-truth posteriors while stacking more VBMC runs.

## 7 Conclusion

In this work, we introduced S-VBMC, a simple, novel approach for stacking variational posteriors generated by separate, possibly parallel, VBMC runs, and tested it on a set of challenging targets. We further suggested an effective and inexpensive method to address one main drawback of our approach (i.e., the ELBO bias build-up). Our results, both in terms of performance and compute time, show its practical convenience for VBMC users, especially when tackling particularly challenging inference problems.
