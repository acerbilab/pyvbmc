# Stacking Variational Bayesian Monte Carlo - Appendix

---

#### Page 22

# A Appendix

This appendix provides additional details and analyses to complement the main text, included in the following sections:

- An overview of Variational Bayesian Monte Carlo, A.1
- A description of how S-VBMC handles VBMC's parameter transformations, A.2
- Additional experiments, A.3
- Full experimental results, A.4
- Example posterior visualisations, A.5

## A.1 An overview of Variational Bayesian Monte Carlo

In this appendix we briefly describe Variational Bayesian Monte Carlo (VBMC). This is a simple overview of the various components of the algorithm, and a full in-depth description of these is beyond the scope of this appendix. For further details, see Acerbi (2018; 2020).

As mentioned in Section 2.3, VBMC addresses the problem of expensive likelihoods with black-box properties by using a surrogate for the log-joint (see Eq. 5). Like many other surrogate-based approaches (Garnett, 2023), VBMC uses a Gaussian process (GP) to approximate its expensive target (Rasmussen & Williams, 2006). GPs are stochastic processes such that any finite collection $f\left(\mathbf{x}_{1}\right), \ldots, f\left(\mathbf{x}_{n}\right)$ follows a multivariate normal distribution. A GP is fully specified by a mean function

$$
m(\mathbf{x})=\mathbb{E}[f(\mathbf{x})]
$$

a covariance function (also called a kernel)

$$
\kappa\left(\mathbf{x}, \mathbf{x}^{\prime}\right)=\operatorname{cov}\left[f(\mathbf{x}), f\left(\mathbf{x}^{\prime}\right)\right]
$$

and an observation noise model or likelihood. In VBMC, as in most surrogate modelling approaches, the likelihood is assumed to be Gaussian, which affords closed-form GP posterior computations. Specifically, given a training set $(\mathbf{X}, \mathbf{y}, \mathbf{S})$ with $\mathbf{X}$ being the observed input locations, $\mathbf{y}$ the corresponding observed function values, and $\mathbf{S}$ a diagonal covariance matrix representing observation noise, the posterior mean and covariance functions for a test input location $\tilde{\mathbf{x}}$ are, respectively,

$$
\mu_{p}(\tilde{\mathbf{x}})=\kappa(\tilde{\mathbf{x}}, \mathbf{X})(\kappa(\mathbf{X}, \mathbf{X})+\mathbf{S})^{-1}(\mathbf{y}-m(\mathbf{X}))+m(\tilde{\mathbf{x}})
$$

and

$$
\kappa_{p}(\tilde{\mathbf{x}}, \tilde{\mathbf{x}})=\kappa(\tilde{\mathbf{x}}, \tilde{\mathbf{x}})-\kappa(\tilde{\mathbf{x}}, \mathbf{X})(\kappa(\mathbf{X}, \mathbf{X})+\mathbf{S})^{-1} \kappa(\mathbf{X}, \tilde{\mathbf{x}})
$$

For further details on GPs and their use in machine learning, see Rasmussen & Williams (2006).
Crucially, a GP surrogate does not yield a usable posterior approximation, as the integral of Eq. 1 (Bayes' rule) remains intractable. Even if the integral can be solved, it does not yield a usable approximation of the posterior, such as the ability to draw samples from it. To address this point, VBMC makes use of Bayesian quadrature, a method for obtaining Bayesian estimates of intractable integrals (O'Hagan, 1991; Ghahramani & Rasmussen, 2002). Given an integral

$$
\mathcal{J}=\int f(\mathbf{x}) \pi(\mathbf{x}) d \mathbf{x}
$$

where $f$ is the target function and $\pi$ is a known probability distribution, if a GP prior is specified for $f$, the integral $\mathcal{J}$ is a Gaussian random variable with posterior mean

$$
\mathbb{E}_{f}[\mathcal{J}]=\int \mu_{p}(\mathbf{x}) \pi(\mathbf{x}) d \mathbf{x}
$$

---

#### Page 23

and variance

$$
\mathbb{V}_{f}[\mathcal{J}]=\int \int \kappa_{p}\left(\mathbf{x}, \mathbf{x}^{\prime}\right) \pi(\mathbf{x}) \pi\left(\mathbf{x}^{\prime}\right) d \mathbf{x} d \mathbf{x}^{\prime}
$$

If $f$ has a Gaussian kernel and $\pi(\mathbf{x})$ is a mixture of Gaussians - which is the case for the VBMC approximate posterior -, both integrals have closed-form solutions.
As mentioned in Section 2, VBMC leverages Bayesian quadrature to estimate the ELBO by building a surrogate model of the log-joint $f(\boldsymbol{\theta}) \approx \log p(\boldsymbol{\theta}) p(\mathcal{D} \mid \boldsymbol{\theta})$. In particular, the GP model $f$ uses an exponentiated quadratic (more commonly known, if incorrectly, as squared exponential) kernel and a negative quadratic mean function (Acerbi, 2018; 2019). The latter ensures integrability of the function and is equivalent to an inductive bias towards Gaussian posteriors, but note that it does not limit the modelled target to be a Gaussian, as the GP can model arbitrary deviations from the mean function.
Therefore, putting everything together, the posterior mean of the surrogate ELBO can be calculated as

$$
\mathbb{E}_{f}[\operatorname{ELBO}(\boldsymbol{\phi})]=\mathbb{E}_{f}\left[\mathbb{E}_{\boldsymbol{\phi}}[f(\boldsymbol{\theta})]\right]+\mathcal{H}\left[q_{\boldsymbol{\phi}}(\boldsymbol{\theta})\right]
$$

where $\mathbb{E}_{f}\left[\mathbb{E}_{\boldsymbol{\phi}}[f(\boldsymbol{\theta})]\right]$ is the posterior mean of the GP surrogate of the log-joint (i.e., the expected value of the expected log-joint). Here the expected log-joint takes the form

$$
\mathbb{E}_{\boldsymbol{\phi}}[f(\boldsymbol{\theta})]=\int f(\boldsymbol{\theta}) q_{\boldsymbol{\phi}}(\boldsymbol{\theta}) d \boldsymbol{\theta}
$$

which, since $q_{\boldsymbol{\phi}}(\boldsymbol{\theta})$ is a mixture of Gaussians, affords closed-form solutions for its posterior mean and variance, as well as its gradients.
To build a good surrogate approximation of the log-joint, a number of likelihood evaluations are needed to train the GP. Assuming the likelihood is expensive to compute, it is desirable to keep this number relatively contained. VBMC tackles this through an iterative process. In each iteration, VBMC selects the next points to evaluate by optimising an acquisition function that trades off exploration of new posterior regions and exploitation of high-density regions (see Acerbi, 2019; 2020). It then uses the resulting log-joint values to refine the GP surrogate, which is in turn used to refine the variational posterior via Bayesian quadrature. The VBMC algorithm begins with only two mixture components in a warm-up stage used to construct an initial surrogate model (see Acerbi, 2018). After the initial warm-up iterations are concluded, VBMC starts adding new components to the mixture to refine the posterior approximation. This process of acquiring new points, using them to improve the surrogate model, and then refining the variational approximation (possibly increasing the number of components), is repeated until a convergence criterion is met or until the likelihood evaluation budget is exceeded.

---

#### Page 24

## A.2 Change-of-variables corrections in S-VBMC

**Problem setting.** Due to the variational whitening feature introduced in Acerbi (2020), each VBMC run operates in its own transformed parameter space, whereas in Acerbi (2018) all VBMC runs shared the same transformed parameter space (a fixed transform for bounded variables). An interested reader should refer to Acerbi (2020) for more information about variational whitening, and an in-depth discussion of this is beyond the scope of this appendix.
In the context of the $m$-th VBMC run, we call $g_{m}(\cdot)$ the function determining the parameter transformation, $g_{m}(\boldsymbol{\theta})$ the transformed parameters, and $\boldsymbol{\psi}_{m}$ the parameters of the variational posterior expressed in the transformed space, $q_{\boldsymbol{\psi}_{m}}\left(g_{m}(\boldsymbol{\theta})\right)$.
Crucially, VBMC returns approximate posteriors in the transformed space, and, since each run is executed independently, and has its own transformation $g_{m}(\cdot)$, the densities obtained with the different approximate posterior parameters $q_{\boldsymbol{\psi}_{m}}\left(g_{m}(\boldsymbol{\theta})\right)$ are not directly comparable. To calculate the stacked ELBO, we need to operate in the common (original) parameter space, in which the parameters $\boldsymbol{\theta}$ are expressed.
Concretely, this means applying appropriate corrections to the densities used to evaluate the stacked ELBO. In the following, we provide a brief introduction to such corrections and discuss how they can be applied to compute the entropy and the expected log-joint (i.e., the two terms of the stacked ELBO, see Eq. 11) in the common parameter space. Importantly, these corrections do not depend on the mixture weights of the stacked posterior, preserving the differentiability of our objective function with respect to $\tilde{\mathbf{w}}$.

**The Jacobian correction.** Parameter transformations can be handled via the Jacobian correction. Given a random variable $\boldsymbol{x}$, its probability $p_{\boldsymbol{x}}(\boldsymbol{x})$ and a transformation $\boldsymbol{y}=T(\boldsymbol{x})$, we have

$$
p_{\boldsymbol{x}}(\boldsymbol{x})=p_{\boldsymbol{y}}(T(\boldsymbol{x}))\left|\operatorname{det} \frac{\partial T(\boldsymbol{x})}{\partial \boldsymbol{x}}\right|
$$

where the correction is applied with the absolute value of the determinant of the Jacobian of $T(\boldsymbol{x})$. Conveniently, the pyvbmc software (Huggins et al., 2023) provides a function to evaluate the (log) absolute value of the determinant of the Jacobian of the inverse transformation (evaluated at $g_{m}(\boldsymbol{\theta})$ ), which, in this toy example, would mean

$$
p_{\boldsymbol{x}}(\boldsymbol{x})=\frac{p_{\boldsymbol{y}}(T(\boldsymbol{x}))}{\left|\operatorname{det} \frac{\partial T^{-1}(T(\boldsymbol{x}))}{\partial T(\boldsymbol{x})}\right|}
$$

To tidy our notation, we will call our corrections term for the $m$-th transformation

$$
J_{m}\left(g_{m}(\boldsymbol{\theta})\right) \equiv\left|\operatorname{det} \frac{\partial g_{m}^{-1}\left(g_{m}(\boldsymbol{\theta})\right)}{\partial g_{m}(\boldsymbol{\theta})}\right|
$$

Bringing this back to our case, the variational posterior of the $m$-th VBMC run can be written as

$$
q_{\boldsymbol{\phi}_{m}}(\boldsymbol{\theta})=\sum_{k=1}^{K_{m}} w_{m, k} q_{k, \boldsymbol{\psi}_{m}}\left(g_{m}(\boldsymbol{\theta})\right) J_{m}^{-1}\left(g_{m}(\boldsymbol{\theta})\right)
$$

and the stacked posterior as

$$
q_{\tilde{\boldsymbol{\phi}}}(\boldsymbol{\theta})=\sum_{m=1}^{M} \sum_{k=1}^{K_{m}} \tilde{w}_{m, k} q_{k, \boldsymbol{\psi}_{m}}\left(g_{m}(\boldsymbol{\theta})\right) J_{m}^{-1}\left(g_{m}(\boldsymbol{\theta})\right)
$$

In line with the notation used in the main text, $\boldsymbol{\phi}_{m}$ and $\tilde{\boldsymbol{\phi}}$ are the parameters of the $m$-th VBMC posterior and of the stacked posterior, respectively, expressed in the common parameter space. With the exception of $\mathbf{w}_{m}$ and $\tilde{\mathbf{w}}$ (which are not affected by the transformation), these parameters are unknown, but, as we will show in the following paragraphs, they are not needed to estimate the stacked ELBO.

---

#### Page 25

**The corrected entropy.** One term of the stacked ELBO is the entropy

$$
\mathcal{H}\left[q_{\tilde{\boldsymbol{\phi}}}\right]=-\mathbb{E}_{q_{\tilde{\boldsymbol{\phi}}}}\left[\log q_{\tilde{\boldsymbol{\phi}}}(\boldsymbol{\theta})\right]
$$

for which no closed-form solution is available. We estimate it via Monte Carlo as in Eq. 12 of the main text, but crucially evaluate all component densities in the original space using the correction shown in Eqs. A.13 and A.14. Concretely, for each component $q_{k, \boldsymbol{\psi}_{m}}$ we draw $S$ samples in the $m$-th transformed space, $\left\{\mathbf{z}_{m, k}^{(s)} \sim q_{k, \boldsymbol{\psi}_{m}}\right\}_{s=1}^{S}$, and map them to the original space, $\mathbf{x}_{m, k}^{(s)}=g_{m}^{-1}\left(\mathbf{z}_{m, k}^{(s)}\right)$. Then, for every sample $\mathbf{x}_{m, k}^{(s)}$ and for every component $q_{k^{\prime}, \boldsymbol{\phi}_{m^{\prime}}}$, we compute the per-component log-density in the original space via

$$
\log q_{k^{\prime}, \boldsymbol{\phi}_{m^{\prime}}}\left(\mathbf{x}_{m, k}^{(s)}\right)=\log q_{k^{\prime}, \boldsymbol{\psi}_{m^{\prime}}}\left(g_{m^{\prime}}\left(\mathbf{x}_{m, k}^{(s)}\right)\right)-\log J_{m^{\prime}}\left(g_{m^{\prime}}\left(\mathbf{x}_{m, k}^{(s)}\right)\right)
$$

and then aggregate

$$
\log q_{\tilde{\boldsymbol{\phi}}}\left(\mathbf{x}_{m, k}^{(s)}\right)=\log \sum_{m^{\prime}=1}^{M} \sum_{k^{\prime}=1}^{K_{m^{\prime}}} \tilde{w}_{m^{\prime}, k^{\prime}} q_{k^{\prime}, \boldsymbol{\phi}_{m^{\prime}}}\left(\mathbf{x}_{m, k}^{(s)}\right)
$$

via log-sum-exp. Then these values can be plugged into Eq. 12 of the main text to estimate the entropy. Importantly, the transformations do not depend on the mixture weights, so the entropy remains differentiable with respect to $\tilde{\mathbf{w}}$.

**The corrected expected log-joint.** Let $\hat{L}_{m, k}$ be the VBMC estimate (computed in the run's transformed coordinates) of the component-wise expected log-joint,

$$
\hat{L}_{m, k} \approx \mathbb{E}_{q_{k, \boldsymbol{\psi}_{m}}}\left[\log p_{m}\left(\mathcal{D}, g_{m}(\boldsymbol{\theta})\right)\right]
$$

where $p_{m}\left(\mathcal{D}, g_{m}(\boldsymbol{\theta})\right)$ is the log-joint reparametrised with the $m$-th transform. To express this expectation in the common parameter space we apply the correction determined by $g_{m}(\boldsymbol{\theta})$

$$
\hat{I}_{m, k}=\hat{L}_{m, k}-\mathbb{E}_{q_{k, \boldsymbol{\psi}_{m}}}\left[\log J_{m}\left(g_{m}(\boldsymbol{\theta})\right)\right] \approx \mathbb{E}_{q_{k, \boldsymbol{\phi}_{m}}}\left[\log p(\mathcal{D}, \boldsymbol{\theta})\right]
$$

In practice we estimate the (per-component) Jacobian term in Eq. A.19 using the same samples $\left\{\mathbf{z}_{m, k}^{(s)} \sim q_{k, \boldsymbol{\psi}_{m}}\right\}_{s=1}^{S}$ employed for the entropy and setting

$$
\mathbb{E}_{q_{k, \boldsymbol{\psi}_{m}}}\left[\log J_{m}\left(g_{m}(\boldsymbol{\theta})\right)\right] \approx \frac{1}{S} \sum_{s=1}^{S} \log J_{m}\left(\mathbf{z}_{m, k}^{(s)}\right)
$$

After applying these corrections, and those to the entropy described above, we can calculate the corrected stacked ELBO as in Eq. 11 of the main text.

Importantly, the $\hat{I}_{m, k}$ described here (i.e., the corrected ones) are the values we use in Section 5.2 for debiasing the stacked ELBO in noisy settings.

---

#### Page 26

## A.3 Additional experiments

### A.3.1 S-VBMC variant

To probe the benefits of optimising the ELBO with respect to the weights of the individual components, we performed additional experiments comparing the version of S-VBMC presented in the main text ("all-weights") with a S-VBMC variant where we only reweigh the weights of each individual VBMC posterior ("posterior-only"). Specifically, we considered a stacked posterior written as:

$$
q_{\tilde{\boldsymbol{\phi}}}(\boldsymbol{\theta})=\sum_{m=1}^{M} \tilde{\omega}_{m} q_{\boldsymbol{\phi}_{m}}(\boldsymbol{\theta})
$$

For this "posterior-only" variant, we optimised the global ELBO with respect to the weights $\tilde{\omega}_{m}$ assigned to each posterior. This is similar to the naive stacking approach seen in the main paper (Eq. 18), with the difference that the posterior weights are now optimised. We ran this method for all the benchmark problems described in Sections 4.4 and 4.5, using the same bootstrapping procedure described in Section 4.1.

The results of this comparison are reported in Figures A.1 and A.2, and in further detail in Appendix A.4.2. We observe that optimising with respect to $\tilde{\boldsymbol{\omega}}$ ("posterior-only") performs well, with both MMTV and GsKL metrics steadily improving with increased numbers of stacked posteriors. In fact, for most problems, "posterior-only" S-VBMC performs comparably to the "all-weights" variant presented in the main paper, which optimises all components weights $\tilde{\mathbf{w}}$. Still, the "all-weights" variant performs slightly better in the GsKL metric and in some challenging scenarios (e.g., the multisensory model), so it remains our base recommendation, paired with the debiasing approach described in Section 5.

### A.3.2 Additional runtime analyses

Here we report the total runtime cost (in seconds) of BBVI for all our benchmark problems compared to that of running VBMC 40 times and stacking the resulting posteriors with S-VBMC. We consider this particular number of runs because the BBVI target evaluation budget was set to match that of S-VBMC with $M=40$. For both, we report the median and $95 \%$ confidence interval, computed from 10000 bootstrap resamples. For S-VBMC, each resample consisted of 40 VBMC runs and one S-VBMC run, then the S-VBMC runtime was added to that of the VBMC run with the highest runtime. This follows from the assumption that VBMC is run 40 times in parallel, and S-VBMC can be launched the moment the last VBMC run has converged.

It is important to note that, as is common in the surrogate-based literature (Acerbi, 2018; Wang & Li, 2018; Acerbi, 2020; Järvenpää et al., 2021; El Gammal et al., 2023; Järvenpää & Corander, 2024), in this work, we demonstrated the efficacy of our method on several problems where function evaluations are not computationally expensive, as a full benchmark with multiple expensive models is highly impractical. Therefore, wall-clock time needs to be interpreted carefully as a metric when comparing methods with different likelihood evaluation costs.

We can directly compare VBMC and S-VBMC, as we did in Section 4.6, because by construction they use the same backbone method and have the same evaluation costs (S-VBMC adds a small post-processing cost, which, crucially, does not depend on the cost of likelihood evaluation). Conversely, comparisons to non-VBMC methods become highly problem-dependent. The typical solution would consist of matching the number of function evaluations (as we did, see Section 4.2), for which non-surrogate-based baselines would be at a significant disadvantage, as demonstrated in previous work (Acerbi, 2018; 2020).

These considerations are crucial to interpret these results, displayed in Table A.1. As expected, BBVI is much faster than S-VBMC on problems with fast likelihood evaluation, but as soon as the likelihood becomes more expensive ( $\approx 0.7$ seconds per evaluation for the neuronal model) the cost of non-VBMC methods increases dramatically, illustrating the kind of scenarios VBMC was developed to solve in the first place.

Finally, it is worth noting that, as shown in Figures 3 and 4 and Tables A.3 and A.4, BBVI performs substantially worse than S-VBMC across our examples, particularly in our real-world problems. Therefore, even where there may be runtime advantages (with the caveats discussed above), these come at a considerable cost in terms of posterior quality.

---

#### Page 27

> **Image description.** A complex figure composed of a 4x4 grid of line graphs, totaling 16 individual plots, designed to compare the performance of two versions of S-VBMC.
> 
> The figure is organized into four rows, each representing a different problem type, and four columns, each representing a different performance metric.
> 
> **Rows (Problem Types):**
> *   **Row (a):** Titled "GMM (D = 2, noiseless)".
> *   **Row (b):** Titled "GMM (D = 2, σ = 3)".
> *   **Row (c):** Titled "Ring (D = 2, noiseless)".
> *   **Row (d):** Titled "Ring (D = 2, σ = 3)".
> 
> **Columns (Performance Metrics):**
> *   **Column 1:** Labeled "ELBO" on the y-axis.
> *   **Column 2:** Labeled "Δ LML" on the y-axis.
> *   **Column 3:** Labeled "MMTV" on the y-axis.
> *   **Column 4:** Labeled "GsKL" on the y-axis.
> 
> **Common Elements Across All Graphs:**
> *   **X-axis:** All graphs share the same x-axis, labeled "N. of runs", with tick marks at 4, 8, 16, 24, 32, and 40.
> *   **Data Representation:** Two distinct datasets are plotted in each graph:
>     *   A blue line with circular markers, accompanied by vertical error bars.
>     *   A yellow line with upward-pointing triangular markers, also accompanied by vertical error bars.
> *   **Legend:** Located at the bottom of the entire figure:
>     *   A yellow triangle and line segment denotes ""posterior-only" S-VBMC".
>     *   A blue circle and line segment denotes ""all-weights" S-VBMC".
> 
> **Specific Details for Each Column:**
> 
> *   **ELBO (Column 1):**
>     *   The y-axis has a linear scale, with ranges varying per row (e.g., 2.4 to 3.0 in row (a), 1.5 to 3.0 in row (d)).
>     *   A solid horizontal black line is present in each ELBO plot, representing the ground-truth LML.
>     *   In general, both blue and yellow lines show an increasing trend with "N. of runs". In rows (a) and (c) (noiseless problems), both lines typically converge towards the black ground-truth LML line. In rows (b) and (d) (σ = 3 problems), the blue line consistently achieves higher ELBO values than the yellow line, though neither fully reaches the ground-truth LML within the plotted range.
> 
> *   **Δ LML (Column 2), MMTV (Column 3), and GsKL (Column 4):**
>     *   **Y-axis Scales:**
>         *   Δ LML and GsKL plots utilize a logarithmic y-axis scale (e.g., 0.001 to 1 for Δ LML, 0.0001 to 1 for GsKL).
>         *   MMTV plots use a linear y-axis scale, with ranges varying per row (e.g., 0.0 to 0.2 in row (a), 0.0 to 0.6 in row (c)).
>     *   **Threshold Line:** A dashed horizontal black line is present in all plots in these three columns, indicating a desirable performance threshold.
>     *   **Visual Patterns:** For all three metrics, both blue and yellow lines generally show a decreasing trend with "N. of runs", indicating improving performance. The blue line ("all-weights" S-VBMC) consistently shows lower values (better performance) and often falls below the dashed threshold more reliably or quickly than the yellow line ("posterior-only" S-VBMC), particularly in rows (b) and (d). The error bars indicate the variability of the measurements.
> 
> In summary, the figure visually compares the convergence and performance metrics of two S-VBMC weighting schemes across different problem complexities, with "all-weights" S-VBMC (blue) generally demonstrating superior or faster convergence, especially in the more challenging noisy scenarios.

Figure A.1: Performance comparison between the two versions of S-VBMC ("all-weights" and "posterior-only") on synthetic problems. Metrics are plotted as a function of the number of VBMC runs stacked (median and $95 \%$ confidence interval, computed from 10000 bootstrap resamples) for S-VBMC when the ELBO is optimised with respect to "all-weights" (blue) and "posterior-only" weights (yellow). The black horizontal line in the ELBO panels represents the ground-truth LML, while the dashed lines on $\Delta$ LML, MMTV, and GsKL denote desirable thresholds for each metric (good performance is below the threshold; see Section 4.3)

---

#### Page 28

> **Image description.** A multi-panel figure presenting eight line graphs arranged in two rows and four columns, comparing the performance of two versions of S-VBMC on two different models. Each graph plots a specific metric against the number of VBMC runs.
> 
> The figure is divided into two main sections, labeled (a) and (b), each occupying a row.
> 
> **Section (a): Neuronal model ($D=5$, noiseless)**
> This top row consists of four line graphs, all sharing the same x-axis representing "N. of runs" from 4 to 40, with major ticks at 4, 8, 16, 24, 32, and 40. Each graph displays two data series: one in blue with circular markers and one in yellow with downward-pointing triangular markers. Both series include vertical error bars representing confidence intervals.
> 
> *   **Panel 1 (ELBO):** The y-axis is labeled "ELBO" and ranges from approximately -7456 to -7450. Both blue and yellow data series show an increasing trend, starting around -7454 and gradually rising to approximately -7452. The blue series generally shows slightly higher ELBO values than the yellow series. A thick black horizontal line is present at approximately -7451.5, representing the ground-truth LML.
> *   **Panel 2 ($\Delta$ LML):** The y-axis is labeled "$\Delta$ LML" and ranges from 0 to 1. Both series show a decreasing trend, starting above 1 and dropping towards 0. The blue series generally decreases faster and reaches lower values than the yellow series, especially as the number of runs increases. A thin black dashed horizontal line is present at y=1, indicating a desirable threshold.
> *   **Panel 3 (MMTV):** The y-axis is labeled "MMTV" and ranges from 0.0 to 0.4. Both series show a decreasing trend, starting around 0.25 and dropping towards 0.1 or below. The blue series generally achieves slightly lower MMTV values than the yellow series after about 16 runs. A thin black dashed horizontal line is present at y=0.2, indicating a desirable threshold.
> *   **Panel 4 (GsKL):** The y-axis is labeled "GsKL" and uses a logarithmic scale, ranging from 0.1 to 10. Both series show a decreasing trend, starting around 10 and dropping towards 0.1 or below. The blue series generally shows lower GsKL values and tighter error bars than the yellow series, particularly at higher numbers of runs. A thin black dashed horizontal line is present at y=0.1, indicating a desirable threshold.
> 
> **Section (b): Multisensory model ($D=6$, $\sigma=3$)**
> This bottom row also consists of four line graphs, mirroring the structure of the top row. The x-axis for the leftmost panel is labeled "N. of runs" (4 to 40), and this applies to all panels in this row. The two data series (blue circles and yellow downward triangles) with error bars are consistent.
> 
> *   **Panel 1 (ELBO):** The y-axis is labeled "ELBO" and ranges from approximately -444.5 to -443.5. Both series show an increasing trend, starting around -444.5 and rising. The blue series consistently shows higher ELBO values than the yellow series across all runs. A thick black horizontal line is present at approximately -444.6, representing the ground-truth LML.
> *   **Panel 2 ($\Delta$ LML):** The y-axis is labeled "$\Delta$ LML" and ranges from 0 to 1. Both series show a decreasing trend, starting above 1 and dropping. The blue series generally decreases faster and reaches lower values than the yellow series, with both converging towards 0. A thin black dashed horizontal line is present at y=1, indicating a desirable threshold.
> *   **Panel 3 (MMTV):** The y-axis is labeled "MMTV" and ranges from 0.00 to 0.20. Both series show a decreasing trend, starting around 0.15 and dropping towards 0.1 or below. The blue series generally achieves slightly lower MMTV values than the yellow series, especially after about 16 runs. A thin black dashed horizontal line is present at y=0.20, indicating a desirable threshold.
> *   **Panel 4 (GsKL):** The y-axis is labeled "GsKL" and uses a logarithmic scale, ranging from 0.1 to 10. Both series show a decreasing trend, starting around 10 and dropping towards 0.1 or below. The blue series generally shows lower GsKL values and tighter error bars than the yellow series. A thin black dashed horizontal line is present at y=0.1, indicating a desirable threshold.
> 
> A common legend is located below the bottom row of graphs. It identifies the two data series:
> *   An icon showing a yellow downward-pointing triangle with vertical error bars is labeled: "posterior-only" S-VBMC
> *   An icon showing a blue circle with vertical error bars is labeled: "all-weights" S-VBMC

Figure A.2: Performance comparison between the two versions of S-VBMC ("all-weights" and "posterior-only") on real-world problems. Metrics are plotted as a function of the number of VBMC runs stacked (median and $95 \%$ confidence interval, computed from 10000 bootstrap resamples) for S-VBMC when the ELBO is optimised with respect to "all-weights" (blue) and "posterior-only" weights (yellow). The black horizontal line in the ELBO panels represents the ground-truth LML, while the dashed lines on $\Delta$ LML, MMTV, and GsKL denote desirable thresholds for each metric (good performance is below the threshold; see Section 4.3)

Table A.1: BBVI runtime (in seconds) compared to that of 40 (parallel) VBMC runs and their subsequent stacking with S-VBMC. Values show median with $95 \%$ confidence interval in brackets. Bold entries indicate the best median performance (i.e., lowest compute time).

|  | Algorithm |  |
| :-- | :--: | :--: |
| Benchmark | BBVI runtime (s) | VBMC + S-VBMC runtime (s) |
| GMM (noiseless) | $\mathbf{9.9}[9.4,10.3]$ | $458.3[428.3,510.8]$ |
| GMM $(\sigma=3)$ | $\mathbf{12.8}[12.2,13.7]$ | $857.8[759.0,954.8]$ |
| Ring (noiseless) | $\mathbf{12.2}[11.7,12.6]$ | $559.3[516.9,794.2]$ |
| Ring $(\sigma=3)$ | $\mathbf{14.0}[13.4,15.0]$ | $1269.9[1206.0,1557.4]$ |
| Neuronal model (noiseless) | $8497.0[8411.2,8617.8]$ | $\mathbf{1561.8}[1445.1,1900.1]$ |
| Multisensory model $(\sigma=3)$ | $\mathbf{24.5}[21.2,27.3]$ | $3149.5[2616.3,3525.1]$ |

---

#### Page 29

## A.4 Full experimental results

### A.4.1 Filtering procedure

Here we briefly present the results of our filtering procedure, described in Section 4.1. As shown in Table A.2, VBMC had considerable trouble when run on the neuronal model, with over half the runs failing to converge (as assessed by the pyvbmc software, Huggins et al., 2023), suggesting a complex, non-trivial posterior structure which is reflected in the poor performance of other inference methods (see Figure 4 and Table A.4). Convergence issues were also found with the noisy Ring target, although to a lesser extent. Once non-converged runs were discarded, our second filtering criterion (i.e., excluding poorly converged runs with excessive uncertainty associated with the $\hat{I}_{k}$ estimates) led to considerably fewer exclusions overall, with only the Ring target being somewhat affected ( $8 \%$ and $13 \%$ of runs discarded in noiseless and noisy settings, respectively).

All our VBMC runs were indexed, and, for our experiments, we used the 100 filtered runs with the lowest indices.

Table A.2: Result of our filtering procedures. This table shows the total number of VBMC runs we performed ("Total"), those that did not converge ("Non-converged") and converged poorly ("Poorly converged") out of the total, and the ones that passed both filtering criteria ("Remaining").

|  | VBMC runs |  |  |  |
| :-- | :--: | :--: | :--: | :--: |
| Benchmark | Total | Non-converged | Poorly converged | Remaining |
| GMM (noiseless) | 120 | 2 | 3 | 115 |
| GMM $(\sigma=3)$ | 150 | 0 | 5 | 145 |
| Ring (noiseless) | 120 | 3 | 9 | 108 |
| Ring $(\sigma=3)$ | 149 | 34 | 15 | 100 |
| Neuronal model (noiseless) | 300 | 159 | 1 | 140 |
| Multisensory $(\sigma=3)$ | 150 | 0 | 1 | 149 |

### A.4.2 Posterior metrics

We present a comprehensive comparison of S-VBMC against VBMC, NS and BBVI in Tables A.3 and A.4, complementing the visualisations in Figures 3, 4, A.1 and A.2. We consider both the version of S-VBMC described in the main text (where the ELBO is optimised with respect to the component weights $\tilde{\mathbf{w}}$, "all-weights"), and the one described in Appendix A.3.1 (where the ELBO is optimised with respect to the posterior weights $\tilde{\boldsymbol{\omega}}$, "posterior-only").

For both synthetic problems (Table A.3) and real-world problems (Table A.4), S-VBMC generally demonstrates consistently improved posterior approximation metrics compared to the baselines. However, we observe an increase in $\Delta$ LML error with larger numbers of stacked runs in problems with noisy targets. This increase likely stems from the accumulation of ELBO estimation bias, a phenomenon analysed in detail in Section 5.

---

#### Page 30

Table A.3: Comparison of S-VBMC, VBMC, and BBVI performance on synthetic benchmark problems. Values show median with $95 \%$ confidence intervals (computed from 10000 bootstrap resamples) in brackets. Bold entries indicate best median performance; multiple entries are bolded when confidence intervals overlap with the best median. For compactness, we label the S-VBMC version described in the main text "w.r.t. $\tilde{\mathbf{w}}$ ", (indicating that the ELBO is optimised with respect to "all-weights" $\tilde{\mathbf{w}}$ ), and the version described in Appendix A.3.1 "w.r.t. $\tilde{\boldsymbol{\omega}}$ ", (indicating that the ELBO is optimised with respect to $\tilde{\boldsymbol{\omega}}$, or "posterior-only").

| Algorithm | Benchmarks |  |  |  |  |  |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |
|  | GMM |  |  | Ring |  |  |
|  | $\Delta\mathbf{LML}$ | MMTV | GsKL | $\Delta\mathbf{LML}$ | MMTV | GsKL |
| **Noiseless** |  |  |  |  |  |  |
| BBVI, MoG $(K=50)$ | $0.059[0.028,0.075]$ | $\mathbf{0.059}[0.035,0.08]$ | $\mathbf{0.0083}[0.0011,0.019]$ | $8[6.8,9.6]$ | $0.51[0.48,0.53]$ | $0.72[0.66,1.2]$ |
| BBVI, MoG $(K=500)$ | $0.053[0.029,0.11]$ | $0.052[0.043,0.07]$ | $0.0087[0.0025,0.013]$ | $8.3[6.9,12]$ | $0.47[0.45,0.49]$ | $0.67[0.55,0.81]$ |
| VBMC | $1.4[0.7,1.4]$ | $0.54[0.39,0.55]$ | $13[7.6,14]$ | $1.2[1.2,1.3]$ | $0.53[0.51,0.56]$ | $9.4[7.2,14]$ |
| NS (10 runs) | $0.091[0.07,0.12]$ | $0.15[0.12,0.17]$ | $0.054[0.034,0.075]$ | $0.16[0.09,0.24]$ | $0.19[0.18,0.22]$ | $0.04[0.021,0.091]$ |
| NS (20 runs) | $0.047[0.037,0.062]$ | $0.11[0.089,0.12]$ | $0.027[0.018,0.032]$ | $0.11[0.073,0.13]$ | $0.18[0.16,0.18]$ | $0.028[0.018,0.049]$ |
| S-VBMC (w.r.t. $\tilde{\boldsymbol{\omega}}, 10$ runs) | $\mathbf{0.0042}[0.0037,0.0087]$ | $\mathbf{0.032}[0.031,0.04]$ | $\mathbf{0.0017}[0.00081,0.003]$ | $0.08[0.057,0.22]$ | $0.16[0.15,0.2]$ | $0.0065[0.0029,0.043]$ |
| S-VBMC (w.r.t. $\tilde{\boldsymbol{\omega}}, 20$ runs) | $\mathbf{0.0059}[0.0035,0.0074]$ | $\mathbf{0.031}[0.024,0.035]$ | $\mathbf{0.0011}[0.00064,0.0016]$ | $0.04[0.037,0.048]$ | $\mathbf{0.15}[0.14,0.15]$ | $\mathbf{0.002}[0.0013,0.0028]$ |
| S-VBMC (w.r.t. $\tilde{\mathbf{w}}, 10$ runs) | $\mathbf{0.0089}[0.0043,0.015]$ | $\mathbf{0.036}[0.028,0.05]$ | $\mathbf{0.0015}[0.0011,0.004]$ | $0.034[0.027,0.047]$ | $\mathbf{0.14}[0.14,0.14]$ | $\mathbf{0.0013}[0.00081,0.0023]$ |
| S-VBMC (w.r.t. $\tilde{\mathbf{w}}, 20$ runs) | $\mathbf{0.0046}[0.0028,0.0072]$ | $\mathbf{0.031}[0.026,0.036]$ | $\mathbf{0.0013}[0.00047,0.0019]$ | $\mathbf{0.022}[0.019,0.026]$ | $\mathbf{0.14}[0.13,0.14]$ | $\mathbf{0.0011}[0.00096,0.0014]$ |
| **Noisy $(\sigma=3)$** |  |  |  |  |  |  |
| BBVI, MoG $(K=50)$ | $0.23[0.11,0.43]$ | $\mathbf{0.13}[0.092,0.18]$ | $0.03[0.01,0.12]$ | $4.3[3.3,4.7]$ | $0.51[0.47,0.54]$ | $1.1[0.65,1.7]$ |
| BBVI, MoG $(K=500)$ | $0.27[0.076,0.45]$ | $\mathbf{0.1}[0.094,0.13]$ | $0.019[0.011,0.034]$ | $4.7[4,5.5]$ | $0.93[0.91,0.94]$ | $48[28,49]$ |
| VBMC | $0.98[0.78,1.1]$ | $0.44[0.43,0.47]$ | $9.7[8.5,11]$ | $1.3[1.1,1.5]$ | $0.62[0.57,0.65]$ | $38[24,95]$ |
| NS (10 runs) | $\mathbf{0.11}[0.066,0.19]$ | $0.17[0.14,0.18]$ | $0.066[0.029,0.12]$ | $\mathbf{0.082}[0.066,0.12]$ | $0.23[0.21,0.26]$ | $0.056[0.033,0.091]$ |
| NS (20 runs) | $\mathbf{0.056}[0.046,0.082]$ | $\mathbf{0.1}[0.09,0.12]$ | $0.017[0.011,0.026]$ | $0.19[0.16,0.22]$ | $\mathbf{0.18}[0.18,0.2]$ | $0.023[0.017,0.03]$ |
| S-VBMC (w.r.t. $\tilde{\boldsymbol{\omega}}, 10$ runs) | $0.19[0.16,0.24]$ | $0.13[0.11,0.14]$ | $\mathbf{0.012}[0.0069,0.031]$ | $\mathbf{0.23}[0.12,0.28]$ | $0.23[0.21,0.24]$ | $0.02[0.01,0.026]$ |
| S-VBMC (w.r.t. $\tilde{\boldsymbol{\omega}}, 20$ runs) | $0.32[0.28,0.34]$ | $\mathbf{0.089}[0.078,0.098]$ | $\mathbf{0.0082}[0.004,0.013]$ | $0.37[0.34,0.41]$ | $\mathbf{0.18}[0.17,0.19]$ | $\mathbf{0.0054}[0.004,0.011]$ |
| S-VBMC (w.r.t. $\tilde{\mathbf{w}}, 10$ runs) | $0.32[0.27,0.45]$ | $\mathbf{0.11}[0.092,0.13]$ | $\mathbf{0.016}[0.0058,0.034]$ | $0.39[0.34,0.45]$ | $0.2[0.19,0.21]$ | $0.013[0.0089,0.025]$ |
| S-VBMC (w.r.t. $\tilde{\mathbf{w}}, 20$ runs) | $0.53[0.51,0.61]$ | $\mathbf{0.09}[0.084,0.097]$ | $\mathbf{0.0049}[0.0036,0.0072]$ | $0.68[0.63,0.71]$ | $\mathbf{0.17}[0.17,0.18]$ | $\mathbf{0.0045}[0.0025,0.0071]$ |

Table A.4: Comparison of S-VBMC, VBMC, and BBVI performance on neuronal and multisensory causal inference models. Bold entries indicate best median performance; multiple entries are bolded when confidence intervals overlap with the best median. See the caption of Table A.3 for further details.

| Algorithm | Benchmarks |  |  |  |  |  |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |
|  | Multisensory model $(\sigma=3)$ |  |  | Neuronal model |  |  |
|  | $\Delta\mathbf{LML}$ | MMTV | GsKL | $\Delta\mathbf{LML}$ | MMTV | GsKL |
| BBVI, MoG $(K=50)$ | $1.7[1.5,4.9]$ | $0.11[0.097,0.13]$ | $0.17[0.16,0.2]$ | $44[33,120]$ | $0.6[0.56,0.64]$ | $20[17,23]$ |
| BBVI, MoG $(K=500)$ | $1.8[1.6,2.5]$ | $0.31[0.28,0.33]$ | $0.53[0.48,0.55]$ | $170[140,260]$ | $0.67[0.64,0.7]$ | $21[18,26]$ |
| VBMC | $\mathbf{0.32}[0.23,0.37]$ | $0.18[0.17,0.19]$ | $0.21[0.17,0.23]$ | $3[3,3.1]$ | $0.32[0.31,0.32]$ | $140[97,190]$ |
| NS (10 runs) | $0.46[0.41,0.52]$ | $0.12[0.11,0.12]$ | $0.072[0.056,0.078]$ | $1.8[1.8,1.9]$ | $0.17[0.16,0.18]$ | $\mathbf{1.2}[0.27,1.4]$ |
| NS (20 runs) | $0.55[0.5,0.59]$ | $0.1[0.096,0.11]$ | $0.06[0.052,0.068]$ | $1.8[1.7,1.9]$ | $0.17[0.15,0.18]$ | $\mathbf{1}[0.35,1.6]$ |
| S-VBMC (w.r.t. $\tilde{\boldsymbol{\omega}}, 10$ runs) | $0.73[0.63,0.86]$ | $0.11[0.097,0.12]$ | $0.062[0.052,0.074]$ | $1.8[1.7,1.8]$ | $\mathbf{0.14}[0.12,0.15]$ | $\mathbf{0.67}[0.36,1.1]$ |
| S-VBMC (w.r.t. $\tilde{\boldsymbol{\omega}}, 20$ runs) | $0.88[0.86,0.95]$ | $0.1[0.095,0.11]$ | $\mathbf{0.047}[0.044,0.057]$ | $\mathbf{1.5}[1.5,1.6]$ | $\mathbf{0.11}[0.087,0.13]$ | $\mathbf{0.3}[0.037,0.57]$ |
| S-VBMC (w.r.t. $\tilde{\mathbf{w}}, 10$ runs) | $0.93[0.89,1]$ | $\mathbf{0.091}[0.086,0.094]$ | $\mathbf{0.042}[0.038,0.05]$ | $\mathbf{1.7}[1.6,1.7]$ | $\mathbf{0.14}[0.11,0.15]$ | $\mathbf{0.47}[0.17,0.79]$ |
| S-VBMC (w.r.t. $\tilde{\mathbf{w}}, 20$ runs) | $1.2[1.2,1.3]$ | $\mathbf{0.079}[0.076,0.091]$ | $\mathbf{0.039}[0.03,0.044]$ | $\mathbf{1.5}[1.5,1.6]$ | $\mathbf{0.12}[0.092,0.13]$ | $\mathbf{0.48}[0.059,0.54]$ |

---

#### Page 31

## A.5 Example posterior visualisations

We use corner plots (Foreman-Mackey, 2016) to visualise exemplar posterior approximations from different algorithms, including S-VBMC, VBMC and BBVI. These plots depict one-dimensional marginal distributions and all pairwise two-dimensional marginals of the posterior samples. Example results (chosen at random among the runs reported in Section 4 and Appendix A.4) are shown in Figures A.3, A.4, A.5, and A.6. S-VBMC consistently improves the posterior approximations over standard VBMC and generally outperforms BBVI, showing a closer alignment with the target posterior.

> **Image description.** A multi-panel figure displays four "corner plots," each visualizing posterior distributions for two variables, $\theta_1$ and $\theta_2$, under different conditions. The panels are arranged in a 2x2 grid, labeled (a) through (d).
> 
> Each corner plot consists of three sub-plots:
> *   A central square plot showing the two-dimensional joint density of $\theta_1$ (x-axis) and $\theta_2$ (y-axis) using contour lines and shading.
> *   An upper rectangular plot showing the one-dimensional marginal distribution of $\theta_1$ as a histogram.
> *   A right rectangular plot showing the one-dimensional marginal distribution of $\theta_2$ as a histogram.
> 
> Across all panels, the x-axis for $\theta_1$ and the y-axis for $\theta_2$ range from approximately -10 to 10. The histograms' y-axes represent density or frequency. In each sub-plot, two distributions are shown: a "target" distribution outlined in black (and gray contours/shading for 2D plots) and an "approximation" distribution outlined in orange (and orange contours/shading for 2D plots). The black target distribution consistently shows a multimodal structure with four distinct clusters in the 2D joint plot (at approximately (-8, 8), (8, 8), (-8, -8), and (8, -8)), and two peaks in each 1D marginal histogram (around -8 and 8).
> 
> The specific content of each panel is as follows:
> 
> *   **Panel (a) "VBMC"**:
>     *   **2D Plot**: The gray contours show four distinct clusters. The orange contours and shading show only one cluster, located in the bottom-right quadrant (around (8, -8)).
>     *   **$\theta_1$ Histogram**: The black histogram shows two peaks (around -8 and 8). The orange histogram shows a single peak, aligning with the rightmost peak of the black distribution (around 8).
>     *   **$\theta_2$ Histogram**: The black histogram shows two peaks (around -8 and 8). The orange histogram shows a single peak, aligning with the leftmost peak of the black distribution (around -8).
>     This indicates that the VBMC approximation only captures one of the four modes of the target posterior.
> 
> *   **Panel (b) "S-VBMC (20 runs)"**:
>     *   **2D Plot**: Both the gray and orange contours and shading are very similar, clearly showing all four distinct clusters. The orange contours are slightly more prominent.
>     *   **$\theta_1$ Histogram**: The black and orange histograms are nearly identical, both showing two distinct peaks (around -8 and 8).
>     *   **$\theta_2$ Histogram**: The black and orange histograms are nearly identical, both showing two distinct peaks (around -8 and 8).
>     This indicates that the S-VBMC approximation closely matches the target posterior, capturing all four modes.
> 
> *   **Panel (c) "VBMC (noisy)"**:
>     *   **2D Plot**: The gray contours show four distinct clusters. The orange contours and shading show only one cluster, located in the top-right quadrant (around (8, 8)).
>     *   **$\theta_1$ Histogram**: The black histogram shows two peaks (around -8 and 8). The orange histogram shows a single peak, aligning with the rightmost peak of the black distribution (around 8).
>     *   **$\theta_2$ Histogram**: The black histogram shows two peaks (around -8 and 8). The orange histogram shows a single peak, aligning with the rightmost peak of the black distribution (around 8).
>     Similar to panel (a), the VBMC approximation under noisy conditions also only captures one of the four modes of the target posterior, but a different one.
> 
> *   **Panel (d) "S-VBMC (20 runs, noisy)"**:
>     *   **2D Plot**: Both the gray and orange contours and shading are very similar, clearly showing all four distinct clusters. The orange contours are slightly more prominent.
>     *   **$\theta_1$ Histogram**: The black and orange histograms are nearly identical, both showing two distinct peaks (around -8 and 8).
>     *   **$\theta_2$ Histogram**: The black and orange histograms are nearly identical, both showing two distinct peaks (around -8 and 8).
>     Similar to panel (b), the S-VBMC approximation under noisy conditions also closely matches the target posterior, capturing all four modes.
> 
> In summary, panels (a) and (c) show that VBMC struggles to approximate the multimodal target posterior, capturing only one mode, while panels (b) and (d) demonstrate that S-VBMC successfully approximates all modes of the target posterior, even under noisy conditions. The black distributions representing the target posterior appear consistent across all four panels.

---

#### Page 32

> **Image description.** A 2x2 grid of four multi-panel plots, labeled (e), (f), (g), and (h), each visualizing posterior and ground truth distributions for two variables, $\theta_1$ and $\theta_2$. Each of these four larger panels is a "corner plot" displaying both 1D marginal histograms and a 2D joint contour plot.
> 
> **Overall Layout and Common Elements:**
> *   The image is structured as a grid of four distinct sub-figures.
> *   Each sub-figure consists of three individual plots:
>     *   A central square plot showing the 2D joint distribution of $\theta_1$ (horizontal axis) and $\theta_2$ (vertical axis).
>     *   A rectangular plot positioned directly above the central plot, illustrating the 1D marginal distribution of $\theta_1$.
>     *   A rectangular plot positioned directly to the right of the central plot, illustrating the 1D marginal distribution of $\theta_2$.
> *   **Axis Labels and Ranges**:
>     *   The horizontal axis for the central and top plots is consistently labeled "$\theta_1$", with values ranging from -10 to 10. Major tick marks are present at -8, 0, and 8, with minor ticks at -10 and 10.
>     *   The vertical axis for the central and right plots is consistently labeled "$\theta_2$", with values ranging from -10 to 10. Major tick marks are present at -10, 0, and 10.
> *   **Data Representation**:
>     *   In the 1D marginal plots (histograms), distributions are depicted as stepped outlines. Black outlines represent ground truth samples, while orange outlines represent posterior samples. Both distributions are visibly bimodal, with prominent peaks centered around -8 and 8.
>     *   In the 2D joint plots (contour plots), distributions are represented by concentric contours. Black contours represent ground truth samples, and orange contours represent posterior samples. These plots consistently show four distinct modes (clusters) arranged roughly in a square pattern, located approximately at (-8, -8), (-8, 8), (8, -8), and (8, 8).
> 
> **Specific Sub-figure Details:**
> 
> *   **(e) BBVI, MoG (K = 50)**:
>     *   The black (ground truth) and orange (posterior) distributions show a very close visual alignment across all three plots. The stepped histogram shapes and the positions and densities of the concentric contours are nearly identical, indicating a strong agreement between the posterior and ground truth.
> 
> *   **(f) BBVI, MoG (K = 500)**:
>     *   Similar to panel (e), the black and orange distributions are in very close agreement. The orange contours in the 2D plot appear slightly smoother and perhaps marginally more defined than in panel (e), suggesting a potentially better or higher-resolution fit.
> 
> *   **(g) BBVI, MoG (K = 50), noisy**:
>     *   In this panel, the orange posterior distributions show noticeable deviations from the black ground truth.
>     *   In the top $\theta_1$ marginal histogram, the orange distribution's peaks are slightly lower, and the distribution is slightly higher in the middle region compared to the black ground truth.
>     *   In the right $\theta_2$ marginal histogram, the orange distribution exhibits a more pronounced peak around 0, and its peaks at -8 and 8 are slightly lower and broader than the black ground truth.
>     *   In the 2D contour plot, the orange contours are visibly less sharp and appear more diffuse or spread out compared to the tightly clustered black contours, especially around the four modes.
> 
> *   **(h) BBVI, MoG (K = 500), noisy**:
>     *   The orange posterior distributions in this panel show improved alignment with the black ground truth compared to panel (g), although some differences persist.
>     *   The 1D histograms (top for $\theta_1$ and right for $\theta_2$) show the orange distributions more closely matching the black ones than in panel (g), with less pronounced discrepancies in peak heights and widths.
>     *   In the 2D contour plot, the orange contours are more defined and closer to the black ground truth contours than in panel (g), indicating a better approximation of the four modes, though they still appear slightly less sharp than the black contours.

Figure A.3: GMM $(D=2)$ example posterior visualisation. Orange contours and points represent posterior samples obtained from different algorithms, while the black contours and points represent ground truth samples.

---

#### Page 33

> **Image description.** This image presents a 2x2 grid of four multi-panel plots, labeled (a), (b), (c), and (d), each illustrating distributions of parameters $\theta_1$ and $\theta_2$ under different conditions. Each of the four main panels is a corner plot, containing three sub-plots arranged in an L-shape: a 1D marginal distribution for $\theta_1$ at the top, a 2D joint distribution for $\theta_1$ and $\theta_2$ at the bottom-left, and a 1D marginal distribution for $\theta_2$ at the bottom-right. The top-right position in each corner plot is empty.
> 
> Common elements across all panels:
> - The horizontal axis for the top-middle and bottom-left plots is labeled "$\theta_1$", with tick marks at -6, 0, and 6.
> - The vertical axis for the bottom-left and bottom-right plots is labeled "$\theta_2$", with tick marks at -6 and 0.
> - In each sub-plot, two distributions are shown: one with a dark grey/black outline and another with an orange outline. The dark grey/black lines generally appear smoother, while the orange lines are typically step-like, resembling histograms.
> - The 2D joint distribution plots consistently feature a large white circular region in the center, surrounded by a ring-like distribution.
> 
> Detailed description of each panel:
> 
> **Panel (a) VBMC:**
> - **Top-middle plot ($\theta_1$):** Shows a U-shaped distribution. The dark grey/black outline forms a smooth U-shape, high at the ends (around $\theta_1 = -6$ and $6$) and low in the middle (around $\theta_1 = 0$). The orange distribution closely follows this U-shape but appears as a step-like histogram, slightly lower in magnitude than the dark grey/black distribution.
> - **Bottom-left plot ($\theta_1$ vs $\theta_2$):** A 2D contour plot. A prominent dark grey/black ring encircles the central white area, representing the joint distribution. Inside this dark grey/black ring, a fainter, step-like orange contour forms a similar ring, indicating the estimated distribution. The orange contours are more granular.
> - **Bottom-right plot ($\theta_2$):** Similar to the $\theta_1$ plot, it displays a U-shaped distribution. The dark grey/black outline is smooth and U-shaped, while the orange distribution is a step-like histogram, closely mirroring the dark grey/black shape but slightly lower.
> - **Label:** (a) VBMC
> 
> **Panel (b) S-VBMC (20 runs):**
> - **Top-middle plot ($\theta_1$):** Shows a U-shaped distribution. The dark grey/black outline is a smooth U-shape. The orange distribution is a step-like histogram that also forms a U-shape, but its peaks at the ends appear slightly higher and more pronounced compared to panel (a), suggesting a tighter distribution at the extremes.
> - **Bottom-left plot ($\theta_1$ vs $\theta_2$):** A 2D contour plot. The dark grey/black ring is present, similar to panel (a). However, the orange contours within this ring are more distinct and appear spiky or "starburst-like", suggesting multiple modes or a more complex, less smooth estimation of the circular distribution compared to panel (a).
> - **Bottom-right plot ($\theta_2$):** Displays a U-shaped distribution. The dark grey/black outline is smooth. The orange distribution is a step-like histogram, with peaks at the ends that are slightly higher and more pronounced than in panel (a).
> - **Label:** (b) S-VBMC (20 runs)
> 
> **Panel (c) VBMC (noisy):**
> - **Top-middle plot ($\theta_1$):** The dark grey/black outline maintains its U-shape. In contrast to panels (a) and (b), the orange distribution here shows a distinct M-shape, with two prominent peaks around $\theta_1 = -3$ and $\theta_1 = 3$, and a clear dip in the middle (around $\theta_1 = 0$). This indicates a bimodal distribution for $\theta_1$.
> - **Bottom-left plot ($\theta_1$ vs $\theta_2$):** A 2D contour plot. The dark grey/black ring is visible. The orange contours form a ring, but they appear less uniform and more "patchy" or "noisy" compared to panel (a), with some areas of higher density.
> - **Bottom-right plot ($\theta_2$):** The dark grey/black outline is U-shaped. The orange distribution is a step-like histogram, forming a U-shape similar to panel (a), but possibly with slightly more jaggedness.
> - **Label:** (c) VBMC (noisy)
> 
> **Panel (d) S-VBMC (20 runs, noisy):**
> - **Top-middle plot ($\theta_1$):** The dark grey/black outline is U-shaped. The orange distribution is a step-like histogram forming a U-shape, similar to panel (b), but potentially with more jaggedness or slightly higher peaks, reflecting the "noisy" context. It does not show the M-shape seen in panel (c).
> - **Bottom-left plot ($\theta_1$ vs $\theta_2$):** A 2D contour plot. The dark grey/black ring is present. The orange contours within this ring are very distinct and exhibit a pronounced "starburst" or "spiky" pattern, similar to panel (b) but perhaps even more accentuated, suggesting a highly variable or multimodal estimation of the circular distribution under noisy conditions.
> - **Bottom-right plot ($\theta_2$):** The dark grey/black outline is U-shaped. The orange distribution is a step-like histogram, forming a U-shape similar to panel (b), but potentially with more jaggedness or slightly higher peaks.
> - **Label:** (d) S-VBMC (20 runs, noisy)


---

#### Page 34

> **Image description.** The image displays a 2x2 grid of four multi-panel plots, labeled (e), (f), (g), and (h), each illustrating a "Ring (D=2) example posterior visualisation." Each of the four main panels contains a 2x2 sub-grid of plots, where the top-right sub-plot is intentionally left blank. The remaining three sub-plots within each panel are: a 2D contour plot in the bottom-left, a 1D marginal distribution plot (step plot) for $\theta_1$ in the top-left, and a 1D marginal distribution plot (step plot) for $\theta_2$ in the bottom-right.
> 
> Common elements across all panels:
> *   The x-axis for the 2D contour plot and the 1D $\theta_1$ plot is labeled $\theta_1$, with tick marks at -6, 0, and 6.
> *   The y-axis for the 2D contour plot and the 1D $\theta_2$ plot is labeled $\theta_2$, with tick marks at -6, 0, and 6.
> *   In the 2D contour plots, a broad ring-shaped distribution is indicated by grey contour lines. Overlaid on this, a second distribution is shown with orange/brown contour lines.
> *   In the 1D marginal distribution plots, a black step plot represents one distribution, and an orange step plot represents another. The black step plots consistently show a bimodal distribution with peaks around -6 and 6 for both $\theta_1$ and $\theta_2$.
> 
> Detailed description of each panel:
> 
> **Panel (e): BBVI, MoG ($K=50$)**
> *   **2D Contour Plot (bottom-left):** The grey contours form a distinct ring centred near (1, -2). The orange/brown contours are concentrated within the left half of this ring, showing a somewhat diffuse distribution with higher density around $\theta_1 \approx -3$ and $\theta_2 \approx 0$.
> *   **1D $\theta_1$ Plot (top-left):** The black step plot shows two peaks at approximately -6 and 6. The orange step plot shows a single, broader peak centered around -3, gradually decreasing towards 6.
> *   **1D $\theta_2$ Plot (bottom-right):** The black step plot shows two peaks at approximately -6 and 6. The orange step plot shows a single, relatively broad peak centered around 0.
> 
> **Panel (f): BBVI, MoG ($K=500$)**
> *   **2D Contour Plot (bottom-left):** The grey contours again form a clear ring. The orange/brown contours are much more tightly concentrated than in (e), forming a sharp, distinct peak within the top-left quadrant of the ring, specifically around $\theta_1 \approx -3$ and $\theta_2 \approx 3$.
> *   **1D $\theta_1$ Plot (top-left):** The black step plot has peaks at -6 and 6. The orange step plot shows a very sharp and tall peak centered precisely at -3, with very low values elsewhere.
> *   **1D $\theta_2$ Plot (bottom-right):** The black step plot has peaks at -6 and 6. The orange step plot shows a very sharp and tall peak centered precisely at 3, with very low values elsewhere.
> 
> **Panel (g): BBVI, MoG ($K=50$), noisy**
> *   **2D Contour Plot (bottom-left):** The grey contours form the characteristic ring. The orange/brown contours are more spread out and irregular compared to (e), still largely concentrated on the left side of the ring, but with a less defined central peak and more scattered smaller concentrations.
> *   **1D $\theta_1$ Plot (top-left):** The black step plot has peaks at -6 and 6. The orange step plot is irregular, showing a primary peak around -3 but also significant values and smaller peaks across the range, including around 3.
> *   **1D $\theta_2$ Plot (bottom-right):** The black step plot has peaks at -6 and 6. The orange step plot is also irregular, showing a primary peak around 0 but with noticeable activity and smaller peaks around -3 and 3.
> 
> **Panel (h): BBVI, MoG ($K=500$), noisy**
> *   **2D Contour Plot (bottom-left):** The grey contours form the ring. The orange/brown contours are extremely concentrated, forming a very sharp and tall peak near the top center of the ring, specifically around $\theta_1 \approx 0$ and $\theta_2 \approx 6$.
> *   **1D $\theta_1$ Plot (top-left):** The black step plot has peaks at -6 and 6. The orange step plot shows an exceptionally sharp and tall peak precisely at 0, with minimal values elsewhere.
> *   **1D $\theta_2$ Plot (bottom-right):** The black step plot has peaks at -6 and 6. The orange step plot shows an exceptionally sharp and tall peak precisely at 6, with minimal values elsewhere.

Figure A.4: Ring $(D=2)$ example posterior visualisation. See the caption of Figure A.3 for further details.

---

#### Page 35

> **Image description.** A two-panel figure displaying two corner plots, each illustrating the marginal and joint posterior distributions for five parameters, labeled theta_1 through theta_5. Both panels are structured identically, featuring a lower triangular matrix of subplots with 1D histograms on the diagonal and 2D density plots on the off-diagonal.
> 
> **Panel (a): VBMC**
> This panel is labeled "(a) VBMC" at the bottom center.
> *   **Diagonal Plots (1D Histograms):** These plots show the marginal distributions for each parameter. Each histogram displays two distributions: one outlined in black and another filled in orange. The orange and black distributions are nearly identical, showing strong overlap.
>     *   The histogram for theta_1 (top-left) is unimodal and bell-shaped, centered approximately between 35 and 40.
>     *   The histogram for theta_2 is highly skewed to the right, peaking at a very small value (around 1e-5 to 2e-5) and extending towards 8e-5.
>     *   The histogram for theta_3 is unimodal and bell-shaped, centered around 2.8 to 3.0.
>     *   The histogram for theta_4 is highly skewed to the right, peaking near 0.00 and extending towards 0.08.
>     *   The histogram for theta_5 (bottom-right) is unimodal and bell-shaped, centered around -64.5.
> *   **Off-Diagonal Plots (2D Density Plots):** These plots show the joint distributions between pairs of parameters. Each plot contains a scatter of faint gray points, overlaid with black contour lines and orange-filled contours, indicating regions of higher density.
>     *   For example, the plot of theta_2 vs theta_1 shows an elongated, slightly curved distribution, with the highest density around theta_1 values of 35-40 and theta_2 values around 2e-5.
>     *   The plot of theta_3 vs theta_1 shows an elliptical distribution, centered around (theta_1 ~ 35-40, theta_3 ~ 2.8-3.0).
>     *   The plot of theta_4 vs theta_2 shows a distribution heavily concentrated near the origin (0, 0), with a tail extending along the theta_2 axis.
>     *   The plot of theta_5 vs theta_1 shows a circular distribution, centered around (theta_1 ~ 35-40, theta_5 ~ -64.5).
> *   **Axes Labels:** The y-axis labels for the rows are theta_2, theta_3, theta_4, and theta_5, from top to bottom. The x-axis labels for the columns are theta_1, theta_2, theta_3, and theta_4, from left to right. The x-axis for the bottom-right diagonal plot is labeled theta_5.
> 
> **Panel (b): S-VBMC (20 runs)**
> This panel is labeled "(b) S-VBMC (20 runs)" at the bottom center.
> *   The visual content of this panel is almost identical to Panel (a). The shapes, peak locations, and spread of all 1D histograms and 2D density plots appear to be consistent with those in Panel (a). The color scheme (black outlines, orange fills, gray scatter points) and contour lines are also the same.
> *   The distributions for each parameter and parameter pair show the same patterns as described for Panel (a), indicating a high degree of similarity in the inferred distributions between "VBMC" and "S-VBMC (20 runs)".
> *   **Axes Labels:** The axes are labeled identically to Panel (a).
(b) S-VBMC (20 runs)

---

#### Page 36

> **Image description.** A two-panel figure, labeled (c) and (d), each displaying a 5x5 lower-triangular matrix of plots, commonly known as a corner plot or matrix plot. These plots visualize the marginal and joint posterior distributions of five parameters, denoted as θ1, θ2, θ3, θ4, and θ5. Each panel compares two distributions: one outlined in black (with gray shading for 2D plots) and another outlined in orange (with orange shading and dashed contours for 2D plots).
> 
> **Panel (c):**
> This panel is labeled "(c) BBVI, MoG (K = 50)" below the plot matrix.
> The plot matrix is arranged with parameters θ1, θ2, θ3, θ4, and θ5 labeling the horizontal axes of the columns (from left to right) and the vertical axes of the rows (from bottom to top).
> 
> *   **Diagonal Plots (1D Marginal Distributions):** The plots along the diagonal display one-dimensional histograms for each parameter. Each histogram shows two distributions: one with a solid black outline and another with a solid orange outline.
>     *   For θ1 (top-left), the black distribution is a tall, narrow, unimodal histogram centered around 35-40. The orange distribution is wider and flatter, shifted slightly to the right.
>     *   For θ2, the black distribution is a very narrow, tall peak near the left edge, while the orange is a broader, shorter peak further to the right.
>     *   For θ3, both black and orange distributions are unimodal, with the black being taller and narrower, centered around 3.0, and the orange being wider and slightly shifted.
>     *   For θ4, the black distribution is a very narrow, tall peak at the left edge (near 0.00), and the orange is a broader, shorter peak slightly to the right.
>     *   For θ5 (bottom-right), the black distribution is a tall, narrow, unimodal histogram centered around -64.5, and the orange is a wider, flatter distribution shifted to the right.
> *   **Off-Diagonal Plots (2D Joint Distributions):** The plots in the lower triangle of the matrix display two-dimensional joint distributions for pairs of parameters.
>     *   One distribution is represented by solid black contour lines and gray shading, indicating regions of higher density.
>     *   The second distribution is represented by dashed orange contour lines and orange shading.
>     *   Faint gray dots are visible in the background of these 2D plots, likely representing underlying samples.
>     *   The shapes of these 2D contours vary, showing different correlations and dependencies between parameter pairs. For example, the plot for (θ2, θ1) shows a distinct, elongated black contour with a separate, smaller orange cluster. The plot for (θ3, θ1) shows concentric elliptical black contours with an overlapping, slightly shifted orange distribution.
> *   **Axis Labels and Ticks:**
>     *   θ1: horizontal axis ticks at approximately 30, 45.
>     *   θ2: horizontal axis ticks at approximately 1e-5, 4, 8.
>     *   θ3: horizontal axis ticks at approximately 2.5, 3.0, 3.5.
>     *   θ4: horizontal axis ticks at approximately 0.00, 0.04, 0.08.
>     *   θ5: horizontal axis ticks at approximately -64.8, -64.5, -64.2.
>     *   Vertical axis ticks for each row (e.g., θ2, θ3, θ4, θ5) show corresponding numerical values, generally increasing upwards.
> 
> **Panel (d):**
> This panel is labeled "(d) BBVI, MoG (K = 500)" below the plot matrix.
> The visual structure and content of this panel are strikingly similar to panel (c). It also consists of a 5x5 lower-triangular matrix of 1D histograms on the diagonal and 2D contour plots off-diagonal, comparing the same two distributions (black/gray and orange/dashed orange).
> 
> *   **Comparison with Panel (c):** While the overall patterns are consistent, subtle differences can be observed. The orange distributions in panel (d) (K=500) generally appear to align more closely with the black distributions compared to panel (c) (K=50). For instance, the orange 1D histogram for θ1 in panel (d) is slightly taller and better centered, more closely resembling the black distribution than its counterpart in panel (c). Similar subtle improvements in alignment or concentration of the orange distributions relative to the black ones can be observed across several plots, suggesting a potentially better approximation with K=500.
> *   **Axis Labels and Ticks:** The axis labels and tick values are identical to those in panel (c).
> 
> In both panels, the black distributions generally appear more concentrated and unimodal, while the orange distributions are often broader, sometimes shifted, or show slightly different modes, particularly in panel (c).

Figure A.5: Neuronal model $(D=5)$ example posterior visualisation. See the caption of Figure A.3 for further details.

---

#### Page 37

> **Image description.** This image displays two multi-panel corner plots, labeled (a) and (b), arranged vertically. Each corner plot is a square matrix showing the marginal and joint posterior distributions for six parameters, $\theta_1$ through $\theta_6$. The plots are designed to compare two different distributions, represented by black and orange outlines/fills.
> 
> **Panel (a): VBMC**
> This panel presents a 6x6 matrix of plots, where the diagonal elements show one-dimensional marginal distributions (histograms), and the off-diagonal elements in the lower triangle show two-dimensional joint distributions (scatter plots with contour lines). The upper triangle of the matrix is empty.
> 
> *   **Diagonal Plots (1D Marginal Distributions)**: These are histograms for each parameter ($\theta_1$ to $\theta_6$). Each histogram displays two distributions: one outlined in black and one outlined in orange. The orange distribution often appears slightly filled or more prominent.
>     *   For $\theta_1$, the distribution is bimodal, with the orange distribution slightly shifted to the right compared to the black one.
>     *   For $\theta_2$, $\theta_3$, $\theta_5$, and $\theta_6$, the distributions are unimodal and bell-shaped, with the black and orange histograms largely overlapping, indicating similar distributions.
>     *   For $\theta_4$, the distribution is skewed right, and the orange distribution is slightly shifted to the right of the black one.
>     *   The x-axis for these plots corresponds to the parameter value, and the y-axis represents density.
> *   **Off-Diagonal Plots (2D Joint Distributions)**: These plots show the relationship between pairs of parameters. Each plot contains a dense cloud of light gray data points. Overlaid on these points are contour lines: solid black lines, solid orange-filled contours, and a dashed orange line. The orange-filled contours typically represent the central tendency or higher density regions of one distribution, while the black lines represent another.
>     *   The contours generally show elliptical or elongated shapes, indicating various degrees of correlation between the parameters. For instance, the plot for $\theta_2$ vs $\theta_1$ shows a positive correlation, while $\theta_3$ vs $\theta_1$ shows a negative correlation.
>     *   The x-axis for each column of off-diagonal plots is labeled at the bottom of the matrix (from $\theta_1$ to $\theta_5$), and the y-axis for each row is labeled on the left side of the matrix (from $\theta_2$ to $\theta_6$).
> *   **Axis Labels and Ticks**: Numerical tick marks and labels are present on the outer edges of the matrix, indicating the ranges for each parameter (e.g., $\theta_1$ from approximately 5 to 10, $\theta_2$ from 12 to 18, $\theta_3$ from 16 to 32, $\theta_4$ from 5 to 10, $\theta_5$ from 0.1 to 0.2, $\theta_6$ from 18 to 24).
> *   **Panel Label**: Below this matrix, the text "(a) VBMC" is displayed.
> 
> **Panel (b): S-VBMC (20 runs)**
> This panel has an identical structure and layout to panel (a), also presenting a 6x6 corner plot for the same six parameters, $\theta_1$ through $\theta_6$.
> 
> *   **Diagonal Plots (1D Marginal Distributions)**: Similar to panel (a), these histograms compare black and orange distributions. The visual characteristics of the distributions (e.g., bimodality for $\theta_1$, unimodality for others, skewness for $\theta_4$) and the relative positions of the black and orange distributions are very similar to those observed in panel (a).
> *   **Off-Diagonal Plots (2D Joint Distributions)**: These plots also show light gray scatter points with overlaid black contour lines, orange-filled contours, and dashed orange lines. The shapes, orientations, and densities of the contours, as well as the implied correlations between parameters, closely resemble those in panel (a).
> *   **Axis Labels and Ticks**: The axis labels ($\theta_1$ to $\theta_6$) and numerical tick marks are consistent with panel (a).
> *   **Panel Label**: Below this matrix, the text "(b) S-VBMC (20 runs)" is displayed.
> 
> **Overall Comparison**:
> Both panels visually represent very similar sets of distributions. The black distributions (both 1D histograms and 2D contours) generally align well with the orange distributions, suggesting that the methods represented by "VBMC" and "S-VBMC (20 runs)" produce comparable results in estimating the underlying distributions. The orange dashed lines in the 2D plots consistently follow the shape of the orange-filled contours.
(b) S-VBMC (20 runs)

---

#### Page 38

> **Image description.** This image displays two separate corner plots, labeled (c) and (d), each presenting a matrix of one-dimensional histograms and two-dimensional kernel density estimates for six parameters, $\theta_1$ through $\theta_6$. Both plots share a similar structure and color scheme, comparing two different distributions.
> 
> **Panel (c): BBVI, MoG (K = 50)**
> This panel is a 6x6 grid of subplots, with the upper right triangle empty.
> *   **Diagonal Subplots (1D Histograms):** These plots show the marginal distributions for each parameter. Each histogram displays two distributions: one represented by a solid orange stepped line and another by a solid black stepped line. The orange distribution generally appears narrower and taller, indicating a more concentrated probability mass, while the black distribution is broader. For example, the histogram for $\theta_1$ (top-left) shows both distributions peaking around 4-5, with the orange distribution being notably higher and narrower. The x-axis labels for these diagonal plots are $\theta_1, \theta_2, \theta_3, \theta_4, \theta_5, \theta_6$ from left to right.
> *   **Off-Diagonal Subplots (2D Density Plots):** These plots show the joint distributions between pairs of parameters. Each plot features two sets of contours and shading:
>     *   One distribution is represented by solid black contour lines and light grey shading, indicating broader, less concentrated areas.
>     *   The second distribution is represented by dashed orange contour lines and solid orange shading, typically appearing more concentrated and often nested within the black contours. The orange shading is generally darker and more centrally located, suggesting higher density.
>     *   The y-axes for the rows are labeled $\theta_2, \theta_3, \theta_4, \theta_5, \theta_6$ from top to bottom. The x-axes for the columns are labeled $\theta_1, \theta_2, \theta_3, \theta_4, \theta_5$ from left to right.
>     *   The shapes of these 2D densities vary, ranging from somewhat circular to elongated ellipses, indicating different correlations between the parameters. For instance, the plot of $\theta_2$ vs $\theta_1$ shows an elongated, positively correlated distribution.
> 
> **Panel (d): BBVI, MoG (K = 500)**
> This panel has the identical structure and parameter labels as panel (c).
> *   **Diagonal Subplots (1D Histograms):** Similar to panel (c), these plots show two distributions in orange and black stepped lines. However, in panel (d), the orange distributions generally appear even narrower and taller, and they show a closer alignment or overlap with the black distributions compared to panel (c). This suggests a more precise or converged estimate for the orange distribution.
> *   **Off-Diagonal Subplots (2D Density Plots):** These plots also show two distributions with black solid contours/grey shading and orange dashed contours/orange shading. A key visual difference from panel (c) is that the orange contours and shading in panel (d) are noticeably more concentrated and tightly nested within the black contours. The orange regions appear smaller and more intense, indicating a tighter joint distribution that aligns more closely with the center of the broader black/grey distribution. The shapes and orientations of the densities are similar to panel (c) but with the orange distributions showing less spread.
> 
> In summary, both panels are corner plots comparing two distributions across six parameters. Panel (d) visually suggests a more concentrated and potentially better-aligned approximation (represented in orange) compared to panel (c), where the orange distributions are somewhat broader and less perfectly aligned with the black/grey distributions.

Figure A.6: Multisensory model $(D=6, \sigma=3)$ example posterior visualisation. See the caption of Figure A.3 for further details.