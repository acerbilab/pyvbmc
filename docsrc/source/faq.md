# PyVBMC: Frequently Asked Questions

This FAQ is curated by [Luigi Acerbi](https://luigiacerbi.com/), and in constant expansion.
It is adapted from the [MATLAB VBMC FAQ](https://github.com/acerbilab/vbmc/wiki)
for PyVBMC 1.5.

For a tutorial with detailed examples, see the [Jupyter notebook examples](examples.rst).

If you have questions not covered here, please feel free to ask in the lab
[Discussions forum](https://github.com/orgs/acerbilab/discussions).

The snippets below use `np` for NumPy and `VBMC` for the inference class:

```python
import numpy as np
from pyvbmc import VBMC
```

Supply your model's target `fun`, starting point `x0`, and bounds `LB`,
`UB`, `PLB`, `PUB` where they appear in a recipe. Use one-dimensional
NumPy arrays for the starting point and bounds.

## Table of contents

- [General](#faq-general)
  - [Which kind of problems is VBMC suited for?](#faq-which-kind-of-problems-is-vbmc-suited-for)
  - [What do I do if VBMC is not suited for my problem?](#faq-what-do-i-do-if-vbmc-is-not-suited-for-my-problem)
- [Installing PyVBMC](#faq-installation)
  - [Where can I download PyVBMC?](#faq-where-can-i-download-vbmc)
  - [Which external packages does PyVBMC require?](#faq-which-external-packages-does-pyvbmc-require)
  - [Which version of Python do I need?](#faq-which-version-of-python-do-i-need)
  - [I am having trouble installing PyVBMC. Can you help?](#faq-i-am-having-trouble-installing-vbmc-can-you-help)
- [Input arguments (target function: `fun`)](#faq-input-arguments-target-function-fun)
  - [What is the target function?](#faq-what-is-the-target-function)
  - [I do not have a likelihood function, but another loss function. Can I still use VBMC?](#faq-i-do-not-have-a-likelihood-function-but-another-loss-function-can-i-still-use-vbmc)
  - [Wait, do I need a prior?](#faq-wait-do-i-need-a-prior)
  - [How do I specify the prior to VBMC?](#faq-how-do-i-specify-the-prior-to-vbmc)
  - [My target function requires additional data/inputs. How do I pass them to VBMC?](#faq-my-target-function-requires-additional-datainputs-how-do-i-pass-them-to-vbmc)
  - [Does VBMC support inference with *noisy* (that is, *stochastic*) target functions?](#faq-does-vbmc-support-inference-with-noisy-that-is-stochastic-target-functions)
- [Input arguments (domain: `x0`,`LB`,`UB`,`PLB`,`PUB`)](#faq-input-arguments-domain-x0lbubplbpub)
  - [How do I choose the starting point `x0`?](#faq-how-do-i-choose-the-starting-point-x0)
  - [How do I choose `LB` and `UB`?](#faq-how-do-i-choose-lb-and-ub)
  - [Can I have parameters bounded only on one side?](#faq-can-i-have-parameters-bounded-only-on-one-side)
  - [Can I set `LB = UB` for some variable to fix it to a given value?](#faq-can-i-set-lb-ub-for-some-variable-to-fix-it-to-a-given-value)
  - [How do I choose `PLB` and `PUB`?](#faq-how-do-i-choose-plb-and-pub)
  - [How do I prevent VBMC from evaluating certain inputs or regions of input space?](#faq-how-do-i-prevent-vbmc-from-evaluating-certain-inputs-or-regions-of-input-space)
  - [Does VBMC support inference with integer parameters?](#faq-does-vbmc-support-inference-with-integer-parameters)
- [Output arguments](#faq-output-arguments)
  - [What is `vp` and what do I do with it?](#faq-what-is-vp-and-what-do-i-do-with-it)
  - [What are `elbo` and `elbo_sd`?](#faq-what-are-elbo-and-elbo-sd)
  - [How is `overhead` in `results` defined?](#faq-how-is-overhead-in-results-defined)
  - [Where can I find the internal state and iteration history?](#faq-where-can-i-find-the-internal-state-and-iteration-history)
- [Noisy target function](#faq-noisy-target-function)
  - [What's a noisy target function?](#faq-whats-a-noisy-target-function)
  - [Does VBMC automatically detect that the target function is noisy?](#faq-does-vbmc-automatically-detect-that-the-target-function-is-noisy)
  - [Does VBMC automatically infer the amount of noise in the target function?](#faq-does-vbmc-automatically-infer-the-amount-of-noise-in-the-target-function)
  - [Can I use *any* technique to estimate a noisy log-likelihood?](#faq-can-i-use-any-technique-to-estimate-a-noisy-log-likelihood)
  - [How do I estimate the standard deviation of the noisy log-likelihood?](#faq-how-do-i-estimate-the-standard-deviation-of-the-noisy-log-likelihood)
  - [How large can the noise be to perform successful inferences with VBMC?](#faq-how-large-can-the-noise-be-to-perform-successful-inferences-with-vbmc)
  - [Wait, can I just remove stochasticity by fixing the random seed at each function call?](#faq-wait-can-i-just-remove-stochasticity-by-fixing-the-random-seed-at-each-function-call)
- [Display](#faq-display)
  - [What are the quantities displayed by VBMC during the variational optimization?](#faq-what-are-the-quantities-displayed-by-vbmc-during-the-variational-optimization)
  - [I noticed that the series of displayed `Mean[ELBO]` values is *not* monotonically increasing. Shouldn't the ELBO keep increasing during the variational optimization?](#faq-i-noticed-that-the-series-of-displayed-meanelbo-values-is-not-monotonically-increasing-shouldnt-the-elbo-keep-increasing-during-the-variational-optimization)
- [Troubleshooting](#faq-troubleshooting)
  - [How can VBMC fail, how do I find out, and how do I fix it?](#faq-how-can-vbmc-fail-how-do-i-find-out-and-how-do-i-fix-it)
  - [VBMC warned me that the `Returned variational solution may have not converged.` What should I do?](#faq-vbmc-warned-me-that-the-returned-variational-solution-may-have-not-converged-what-should-i-do)
  - [VBMC crashes saying that `The returned function value must be a finite real-valued scalar`. What do I do?](#faq-vbmc-crashes-saying-that-the-returned-function-value-must-be-a-finite-real-valued-scalar-what-do-i-do)
  - [I have been running VBMC from the same starting point, but I get different results each time. Is something wrong?](#faq-i-have-been-running-vbmc-from-the-same-starting-point-but-i-get-different-results-each-time-is-something-wrong)
  - [How do I save and continue a run?](#faq-how-do-i-save-and-continue-a-run)
  - [Can I combine the posteriors of several runs?](#faq-can-i-combine-the-posteriors-of-several-runs)
- [Miscellanea](#faq-miscellanea)
  - [I already run PyBADS on my problem. How do I run VBMC?](#faq-i-already-run-pybads-on-my-problem-how-do-i-run-vbmc)
  - [This is interesting, but why is it better than just using point estimates? (e.g., maximum-likelihood or maximum-a-posteriori)](#faq-this-is-interesting-but-why-is-it-better-than-just-using-point-estimates-eg-maximum-likelihood-or-maximum-a-posteriori)
  - [Thanks, but I think I will stick to point estimates for now.](#faq-thanks-but-i-think-i-will-stick-to-point-estimates-for-now)
  - [Are you planning to port VBMC to other languages?](#faq-are-you-planning-to-port-vbmc-to-other-languages)

(faq-general)=
## General

(faq-which-kind-of-problems-is-vbmc-suited-for)=
### Which kind of problems is VBMC suited for?

I would recommend VBMC for problems in which:

- you can calculate an estimate of the target (log) likelihood function, which can be deterministic or noisy (see [below](#faq-noisy-target-function));
- the target likelihood function is at least moderately expensive to compute (at least ~0.1 s or more per function evaluation);
- the gradient may be unavailable;
- the number of input parameters is up to about `D = 10` (*maybe* up to `20`, but not more);
- the target posterior is continuous and reasonably smooth, such that it can be approximated by a Gaussian process (GP) with a smooth kernel. This is generally difficult to know *a priori*, but you can look for [telltale signs](#faq-vbmc-warned-me-that-the-returned-variational-solution-may-have-not-converged-what-should-i-do) that the VBMC approximation is failing.

If your likelihood function is fully analytical — or, more generally, fast to evaluate —, VBMC is most likely not the best tool for your problem (see [below](#faq-what-do-i-do-if-vbmc-is-not-suited-for-my-problem)).

(faq-what-do-i-do-if-vbmc-is-not-suited-for-my-problem)=
### What do I do if VBMC is not suited for my problem?

If the likelihood function is smooth and analytical (and fast to compute), you should consider estimating the posterior via Markov Chain Monte Carlo, for example using probabilistic programming languages such as [Stan](https://mc-stan.org/) or [PyMC](https://www.pymc.io/).

If the likelihood function is computationally expensive *and* non-smooth (or it produces a pathological posterior), then, well, tough luck. In this case, you may see if revising your model, for example by changing the parameterization, produces a posterior landscape which is more compatible with the approximations used by VBMC.

Alternatively, you may give up on full (approximate) Bayesian inference, and resort instead to maximum-likelihood (or maximum-a-posteriori) estimation, using an efficient optimization algorithm such as [Bayesian Adaptive Direct Search in Python (PyBADS)](https://acerbilab.github.io/pybads/).

(faq-installation)=
## Installing PyVBMC

(faq-where-can-i-download-vbmc)=
### Where can I download PyVBMC?

Install PyVBMC with:

```console
python -m pip install pyvbmc
```

or with Conda:

```console
conda install --channel=conda-forge pyvbmc
```

See the [installation instructions](installation.rst) for more details, and
the [GitHub repository](https://github.com/acerbilab/pyvbmc) for the source code.

(faq-which-external-packages-does-pyvbmc-require)=
### Which external packages does PyVBMC require?

PyVBMC uses NumPy, SciPy and several other Python packages, which are
installed automatically with PyVBMC. It does not require MATLAB.

The Torch posterior export and S-VBMC need the optional `torch` extra;
the ArviZ export needs the `arviz` extra. See
[optional integrations](installation.rst) for installation
instructions.

(faq-which-version-of-python-do-i-need)=
### Which version of Python do I need?

PyVBMC requires Python 3.10 or newer. The optional ArviZ export requires
Python 3.12 or newer.

(faq-i-am-having-trouble-installing-vbmc-can-you-help)=
### I am having trouble installing PyVBMC. Can you help?

Sure. The PyVBMC installation should be pretty straightforward, so write me in detail which problem you are having in the [Discussions forum](https://github.com/orgs/acerbilab/discussions). Include your operating system, Python version, installation command and the full error message.

(faq-input-arguments-target-function-fun)=
## Input arguments (target function: `fun`)

(faq-what-is-the-target-function)=
### What is the target function?

The target function in VBMC is the (unnormalized) log posterior or log joint probability density, specified via a Python callable `fun`, the first input argument to `VBMC`. You can also pass the log likelihood and a separate prior, as explained [below](#faq-how-do-i-specify-the-prior-to-vbmc).

For a typical model-fitting problem, `fun` is a function that computes the (unnormalized) log posterior of an input parameter vector `x`, for a given dataset and model. The log posterior is computed as log likelihood plus log prior.

If you are unsure, see the following questions for more information about the likelihood function and the prior.

By default, `fun` takes a one-dimensional NumPy parameter vector with shape
`(D,)` and returns one finite real scalar. Use natural logarithms. For example:

```python
from pyvbmc import VBMC

vbmc = VBMC(fun, x0, LB, UB, PLB, PUB)
vp, results = vbmc.optimize()
```

Opt-in [vectorized targets](quickstart.rst) receive a batch
of points; the default interface evaluates one point at a time.

(faq-i-do-not-have-a-likelihood-function-but-another-loss-function-can-i-still-use-vbmc)=
### I do not have a likelihood function, but another loss function. Can I still use VBMC?

For the results of VBMC to make sense, you really need a likelihood (and a prior; see [below](#faq-wait-do-i-need-a-prior)).

However, sometimes you might already almost have a likelihood, and you just do not know about it. For example, suppose you are fitting a model via *least squares*, such that your target is a quadratic loss function defined as:

```python
import numpy as np

def loss(x, data_x, data_y):
    return np.sum((f(data_x, x) - data_y) ** 2)
```

where `f` returns an array of model predictions for model parameters `x` and data `data_x`, and `data_y` is an array of observed responses. You can transform the above equation in a log likelihood as:

```python
def log_likelihood(x, data_x, data_y):
    sigma2 = np.exp(2 * x[-1])  # Extract sigma^2 from log sigma
    N = np.size(data_y)         # Number of scalar observations
    x_orig = x[:-1]             # Parameters without log sigma
    residuals = f(data_x, x_orig) - data_y
    return -0.5 * np.sum(residuals ** 2) / sigma2 - 0.5 * N * np.log(
        2 * np.pi * sigma2
    )
```

where we have added `log(sigma)` as an additional model parameter at the end of the parameter vector `x_orig` (the logarithmic representation is convenient to ensure that `sigma` is positive). `sigma` is the standard deviation of the observed values `data_y` with respect to the model predictions `f(data_x,x)` (that is, the standard deviation of the residuals). The last line defines the log likelihood of a series of observations corrupted by Gaussian noise. For a fixed `sigma`, maximizing `log_likelihood` with respect to the original model parameters is equivalent to minimizing the quadratic loss defined above. Keep the normalization term when estimating `sigma` or comparing model evidence.

(faq-wait-do-i-need-a-prior)=
### Wait, do I need a prior?

Yes, you do. For the marginal likelihood (model evidence) to make sense as a model comparison metric, you should use a *proper* prior over the parameters (that is, a prior that integrates to 1). Horrible things happen if you compute the marginal likelihood with an improper or unnormalized prior. Since VBMC computes the ELBO, a lower bound to the (log) marginal likelihood, you need a prior.

Also, a prior will come in handy to set plausible bounds for the parameters (see [below](#faq-how-do-i-choose-plb-and-pub)).

A uniform bounded prior over parameters is okay (although not ideal), as long as you correctly normalize it.

(faq-how-do-i-specify-the-prior-to-vbmc)=
### How do I specify the prior to VBMC?

You can pass to `VBMC` the target function `fun`, a Python callable that
returns the log joint distribution, that is log likelihood plus log prior.
For example, for a uniform prior over finite bounds you would have:

```python
def fun(x):
    return log_likelihood(x) - np.sum(np.log(UB - LB))

vbmc = VBMC(fun, x0, LB, UB, PLB, PUB)
```

where `log_likelihood` is the log likelihood function, and the second term
represents the log prior (uniform over the range). PyVBMC does *not*
automatically normalize a log prior that you write — that is up to you.

Alternatively, pass a separate prior with `prior=`. For independent uniform
priors, the same problem can be written as:

```python
from scipy.stats import uniform

prior = [uniform(loc=low, scale=high - low) for low, high in zip(LB, UB)]
vbmc = VBMC(log_likelihood, x0, LB, UB, PLB, PUB, prior=prior)
```

In this case PyVBMC adds the log prior to the log likelihood, so do not
include the prior in `log_likelihood` as well. Supported priors include
PyVBMC prior objects and appropriate `scipy.stats` distributions. You can
also supply a custom `log_prior=` callable. See the
[prior documentation](api/classes/priors.rst) for details.

The hard bounds constrain where PyVBMC evaluates the target; they do not
by themselves add or normalize a prior.

(faq-my-target-function-requires-additional-datainputs-how-do-i-pass-them-to-vbmc)=
### My target function requires additional data/inputs. How do I pass them to VBMC?

Suppose that your function takes two inputs, `fun(x, data)`.

The first solution consists of defining a new function:

```python
def funwdata(x):
    return fun(x, data)
```

where `data` has been defined before in the code. Now you can pass to
`VBMC` `funwdata`, which takes a single input. The wrapper reads `data`
when it is called; it does not embed a copy. Keep the data fixed throughout
the inference.

Alternatively, you can use `functools.partial`:

```python
from functools import partial

funwdata = partial(fun, data=data)
vbmc = VBMC(funwdata, x0, LB, UB, PLB, PUB)
```

This binds the `data` argument to the supplied object, also without copying
it.

(faq-does-vbmc-support-inference-with-noisy-that-is-stochastic-target-functions)=
### Does VBMC support inference with *noisy* (that is, *stochastic*) target functions?

Yes. See [this section](#faq-noisy-target-function) for more information.

(faq-input-arguments-domain-x0lbubplbpub)=
## Input arguments (domain: `x0`,`LB`,`UB`,`PLB`,`PUB`)

The examples use the following abbreviations for the bounds:

| Variable | `VBMC` keyword argument |
| --- | --- |
| `LB` | `lower_bounds` |
| `UB` | `upper_bounds` |
| `PLB` | `plausible_lower_bounds` |
| `PUB` | `plausible_upper_bounds` |

(faq-how-do-i-choose-the-starting-point-x0)=
### How do I choose the starting point `x0`?

First of all, keep in mind that you should run VBMC from different starting points, as a [sanity check](#faq-how-can-vbmc-fail-how-do-i-find-out-and-how-do-i-fix-it) for your solutions.

A good starting point belongs to regions of high posterior probability density (even better, to *typical* regions).
Generally, a point in the vicinity of the mode of the posterior would represent a reasonable starting point (this is [not necessarily true in high dimension](https://www.inference.vc/high-dimensional-gaussian-distributions-are-soap-bubble/), but VBMC only operates in relatively low dimension).
You can find such points by running a few iterations of an optimizer first.

Note that a starting point should not be *on* or too close to the bounds `LB` or `UB`. PyVBMC moves such points slightly inside and warns. You can avoid this issue by constraining the preliminary optimization to be inside the plausible box, that is within `PLB` and `PUB` (see also [below](#faq-how-do-i-choose-plb-and-pub)).

For example, you can use [Bayesian Adaptive Direct Search in Python (PyBADS)](https://acerbilab.github.io/pybads/) to perform the preliminary optimization inside the plausible box, as follows:

```python
import numpy as np
from pybads import BADS

# Pick a starting point via a preliminary optimization with PyBADS.
D = np.size(PLB)
rng = np.random.default_rng()
x0 = rng.uniform(PLB, PUB)
opt_options = {"max_fun_evals": 50 * D}
bads = BADS(lambda x: -fun(x), x0, PLB, PUB, options=opt_options)
x0 = bads.optimize()["x"]
```

This example assumes a deterministic `fun` returning the log joint. Install
PyBADS separately; its [documentation](https://acerbilab.github.io/pybads/)
also covers noisy optimization.

(faq-how-do-i-choose-lb-and-ub)=
### How do I choose `LB` and `UB`?

`LB` and `UB` are the *hard bounds* of the parameter space of the model of interest. These are *strict* inequalities, in that for any coordinate dimension `i`, `LB[i] < x[i] < UB[i]`.

You can set the hard bounds to the *support* of the posterior probability density (that is, the region with nonzero probability density). For each coordinate dimension, the choice is between:

- unconstrained coordinates, specified with `LB[i] = -np.inf` and `UB[i] = np.inf`; or
- bounded coordinates, in which both `LB[i]` and `UB[i]` are finite (with `LB[i] < UB[i]`).

(faq-can-i-have-parameters-bounded-only-on-one-side)=
### Can I have parameters bounded only on one side?

Half-bounded parameters are coordinate dimensions for which one bound is finite and the other is infinite. At the moment, VBMC does not support half-bounded parameters.

If you need to set a half-bound constraint, fix the 'unbounded' half to some large but meaningful value (e.g., the 0.01th or 99.9th percentile of the prior, for, respectively, an 'unbounded' lower/upper bound). Do **not** set the unbounded half to some impossibly large value, with respect to the scale of your problem, otherwise VBMC is likely to fail.

A finite cutoff truncates the target. Check that the omitted posterior mass
is negligible; if the cutoff defines a truncated prior for model comparison,
include its normalization in the log prior. Another approach is to
reparameterize, for example using the logarithm of a positive parameter.
Include the change-of-variables Jacobian in the log density when you do so.

(faq-can-i-set-lb-ub-for-some-variable-to-fix-it-to-a-given-value)=
### Can I set `LB = UB` for some variable to fix it to a given value?

No — at the moment, `LB` and `UB` need to be distinct. To fix a parameter, remove it from the vector passed to `VBMC` and insert its fixed value inside your target function.

(faq-how-do-i-choose-plb-and-pub)=
### How do I choose `PLB` and `PUB`?

`PLB` and `PUB` are the *plausible* (or *reasonable*) bounds, which denote a region of high posterior probability mass.
Plausible bounds are used by VBMC to draw training points to construct the starting approximation of the posterior, and to initialize various hyperparameters of the algorithm, such as typical scales for the parameters.

The plausible bounds need to be finite, and distinct from the hard bounds `LB` and `UB`.

In the absence of other information, I would recommend to set the plausible range as the ~68.26% percentile interval of the prior (that is, set `PLB` to the 15.87% percentile and `PUB` to the 84.13% percentile).

For example, for a Gaussian prior with a given mean `mu` and standard deviation `SD`, this would amount to

```python
from scipy.stats import norm

PLB = norm.ppf(0.1587, loc=mu, scale=SD)
PUB = norm.ppf(0.8413, loc=mu, scale=SD)
```

which is about equivalent to

```python
PLB = mu - SD
PUB = mu + SD
```

Somewhat counter-intuitively, if in doubt about the location of a high posterior probability region, it is better to choose *narrower* plausible bounds than wide ones. The rationale is that the Gaussian process approximation built by VBMC might fail if the algorithm ends up sampling from a region with extremely low posterior probability, and this is more likely to happen if the plausible bounds are arbitrarily large.

(faq-how-do-i-prevent-vbmc-from-evaluating-certain-inputs-or-regions-of-input-space)=
### How do I prevent VBMC from evaluating certain inputs or regions of input space?

If these regions can be identified by coordinate-wise ranges, use `LB` and `UB`.
If those regions remain possible under your model, use the [prior over parameters](#faq-wait-do-i-need-a-prior) to *gradually* and continuously reduce the probability for those regions (but never to zero).

Do **not** have `fun(x)` return `np.inf`, `-np.inf` or `np.nan` for invalid inputs.
VBMC would simply crash (see [this question](#faq-vbmc-crashes-saying-that-the-returned-function-value-must-be-a-finite-real-valued-scalar-what-do-i-do)).

Returning `0` is valid for a log density, but it does not mean zero probability and will not exclude a point.

Absolutely do **NOT** have `fun(x)` return an arbitrarily large negative number (such as `-1e100`) to enforce VBMC to avoid certain regions.
This strategy may seem innocent enough, but in fact it may cripple VBMC by making its model of the target function nonsensical.

For regions that are genuinely impossible, look for a parameterization whose
box constraints describe the allowed domain. Arbitrarily changing the prior
changes the inference problem.

(faq-does-vbmc-support-inference-with-integer-parameters)=
### Does VBMC support inference with integer parameters?

No, VBMC does not support integer parameters (that is, variables forced to be integers — or, more in general, constrained to discrete values).
Be aware that simple workarounds might break down the VBMC approximation, in particular the assumption that the underlying target function (the log posterior) is continuous and reasonably smooth.

(faq-output-arguments)=
## Output arguments

(faq-what-is-vp-and-what-do-i-do-with-it)=
### What is `vp` and what do I do with it?

`vp` is a `VariationalPosterior` object that represents the variational
posterior — an approximation of the true posterior — computed by the VBMC
inference algorithm.

You can query and manipulate the variational posterior with the following methods:

- `vp.sample(N)`: generate random samples from the variational posterior;
  returns `(samples, component_indices)`;

- `vp.pdf(x)`: evaluate the probability density of the variational posterior
  at a given point;

- `vp.moments(cov_flag=True)`: compute mean and covariance matrix of the
  variational posterior;

- `vp.mode()`: compute the mode of the variational posterior;

- `vp.kl_div(vp2=other_vp)`: compute Kullback-Leibler divergences between two
  variational posteriors;

- `vp.mtv(other_vp)`: compute marginal total variation distances between two
  variational posteriors;

- `vp.plot()`: visualize the variational posterior.

The mode of the variational posterior can be an unreliable estimate of the
target's maximum-a-posteriori (MAP) point, especially near the bounds.

For example:

```python
samples, component_indices = vp.sample(10000)
vp.plot()
```

These methods use original parameter coordinates by default. To check the
object's type, use `isinstance(vp, VariationalPosterior)` after importing
`VariationalPosterior` from `pyvbmc`.

You can see several usage examples in the [tutorials](examples.rst), and
the complete interface in the
[`VariationalPosterior` documentation](api/classes/variational_posterior.rst).
Optional [Torch](api/methods/variational_posterior_to_torch.rst)
and [ArviZ](api/methods/variational_posterior_to_arviz.rst) exports
let you use the posterior in other Python workflows. If you have several
posteriors, [compare their solutions](#faq-how-can-vbmc-fail-how-do-i-find-out-and-how-do-i-fix-it)
before [combining them with S-VBMC](#faq-can-i-combine-the-posteriors-of-several-runs).

(faq-what-are-elbo-and-elbo-sd)=
### What are `elbo` and `elbo_sd`?

The returned `results` dictionary contains `results["elbo"]` and
`results["elbo_sd"]`.

`elbo` is the variational Evidence Lower BOund estimated by VBMC, that is an approximation of a lower bound on the log marginal likelihood (or log model evidence). Note that there are two levels of approximation:

- variational inference computes the ELBO, which is a lower bound on the log marginal likelihood;
- VBMC estimates the ELBO using a GP surrogate of the true posterior and Bayesian quadrature.

`elbo_sd` is the uncertainty (standard deviation) on the *second part* of the above approximation. In variational inference, there is no direct way to know how far the ELBO is from the true log marginal likelihood (which would be the error of the first part).

You can use `elbo` as a metric for model selection, with `elbo_sd` providing one of several diagnostics of convergence. For noiseless targets, you should not trust solutions with a fairly large uncertainty (say, `elbo_sd` >> 0.1). Larger uncertainties are common with [noisy targets](#faq-noisy-target-function); interpret them alongside the evaluation noise and agreement across runs. A small `elbo_sd` does not guarantee that the ELBO is close to the true log evidence, and the estimated ELBO can exceed the true evidence because of surrogate error.

(faq-how-is-overhead-in-results-defined)=
### How is `overhead` in `results` defined?

PyVBMC currently returns `np.nan` for `results["overhead"]`; this field is
not implemented.

The *fractional overhead* is defined as (*total running time* / *total
function time* - 1). You can measure it by timing the complete optimization
and the time spent evaluating your target. Typically, you would expect
overhead to be of order 1 or smaller for normal runs of VBMC. If the
overhead is *much* larger than 1 (e.g., 10 or more), your problem affords
fast evaluations and it is possible that it would benefit from other
algorithms than VBMC (see [above](#faq-what-do-i-do-if-vbmc-is-not-suited-for-my-problem)).

For VBMC test problems and examples, the overhead can be astronomical,
which is expected since for demonstration purposes we are using simple
analytical functions which are extremely fast to evaluate.

(faq-where-can-i-find-the-internal-state-and-iteration-history)=
### Where can I find the internal state and iteration history?

`VBMC.optimize()` returns the variational posterior `vp` and a `results`
dictionary. The `VBMC` object also retains `vbmc.optim_state`, which
contains fields related to the internal state of the algorithm, and
`vbmc.iteration_history`, which contains detailed information and
statistics about each iteration.

These attributes are useful for development and debugging, but their
internal structure may change in future versions. Use the documented
[`VBMC` interface](api/classes/vbmc.rst) where possible; for example,
`vbmc.get_gp(iteration)` retrieves a recorded iteration's GP surrogate,
ready for prediction.

(faq-noisy-target-function)=
## Noisy target function

(faq-whats-a-noisy-target-function)=
### What's a noisy target function?

A target function is *noisy* (or *stochastic*) if it returns *noisy* observations of the true underlying log-joint (unnormalized log posterior). Conversely, a non-noisy target function is *deterministic*. Log likelihoods can be noisy when evaluated through simulation (e.g., via Monte Carlo methods).

PyVBMC supports noisy target functions as input, as explained below.

(faq-does-vbmc-automatically-detect-that-the-target-function-is-noisy)=
### Does VBMC automatically detect that the target function is noisy?

No. In order to perform inference with a noisy target function, you need to:

- manually set `options={"specify_target_noise": True}` when constructing `VBMC`;
- pass to VBMC a function `fun` that returns a pair `(log_density, noise_sd)`, where `noise_sd` is a finite, positive estimate of the standard deviation (SD) of the log-density evaluation at `x`. See [below](#faq-how-do-i-estimate-the-standard-deviation-of-the-noisy-log-likelihood) and [Example 6](_examples/pyvbmc_example_6_noisy_likelihoods.ipynb) for further information.

With a separate `prior=` or `log_prior=`, `log_density` is the noisy log
likelihood; otherwise it is the noisy log joint. The SD describes the
uncertainty of that returned estimate, not variability across observations
or across parameter values.

(faq-does-vbmc-automatically-infer-the-amount-of-noise-in-the-target-function)=
### Does VBMC automatically infer the amount of noise in the target function?

No. See [above](#faq-does-vbmc-automatically-detect-that-the-target-function-is-noisy).

(faq-can-i-use-any-technique-to-estimate-a-noisy-log-likelihood)=
### Can I use *any* technique to estimate a noisy log-likelihood?

Kind of. Whatever estimation technique you use, VBMC expects the estimates of the log-likelihood (or log-joint) to be approximately unbiased and normally-distributed, with an available estimate of the standard deviation. [*Inverse binomial sampling* (IBS)](https://github.com/acerbilab/ibs) is a technique that checks all these boxes. The *synthetic likelihood* (SL) method could also work.

(faq-how-do-i-estimate-the-standard-deviation-of-the-noisy-log-likelihood)=
### How do I estimate the standard deviation of the noisy log-likelihood?

If you use [*inverse binomial sampling* (IBS)](https://github.com/acerbilab/ibs), the algorithm returns the variability of the estimate as second output. Just ensure that the variability is returned as *standard deviation* (SD) and not as the variance (depending on the implementation, you may have to take the square root of the reported variance).

Otherwise, you can also estimate the SD via bootstrap or similar approaches.

(faq-how-large-can-the-noise-be-to-perform-successful-inferences-with-vbmc)=
### How large can the noise be to perform successful inferences with VBMC?

While VBMC has been shown to handle correctly up to a fairly large amount of observation noise (SD ~ 7 points in the modal region), ideally you want the SD of the noise in the log-likelihood to be around 1, and probably not larger than ~3, at least in the area covered by a meaningful fraction of posterior probability mass.

(faq-wait-can-i-just-remove-stochasticity-by-fixing-the-random-seed-at-each-function-call)=
### Wait, can I just remove stochasticity by fixing the random seed at each function call?

No, that is a silly idea. This approach might *freeze* the noise, but would not remove it (see [here](https://github.com/acerbilab/bads/wiki#can-i-make-a-noisy-objective-function-deterministic-by-fixing-the-noise-process)).

(faq-display)=
## Display

(faq-what-are-the-quantities-displayed-by-vbmc-during-the-variational-optimization)=
### What are the quantities displayed by VBMC during the variational optimization?

If `options={"display": "iter"}` is passed to `VBMC` (the default), VBMC displays the traces of several quantities:
  - the `Iteration` number;
  - the number of target function evaluations `f-count`;
  - the current estimate of the mean of the ELBO, `Mean[ELBO]`;
  - the uncertainty (standard deviation) on the current estimate of the ELBO, `Std[ELBO]`;
  - the change in variational posterior from the previous iteration, measured in terms of symmetrized Kullback-Leibler (KL) divergence between the two variational posteriors, `sKL-iter[q]`;
  - the number of components of the current variational mixture `K[q]`;
  - a metric of `Convergence` of the current solution (values less than 1 are *necessary* for convergence, but not sufficient);
  - additional actions (such as starting and ending of the warm-up stage).

(faq-i-noticed-that-the-series-of-displayed-meanelbo-values-is-not-monotonically-increasing-shouldnt-the-elbo-keep-increasing-during-the-variational-optimization)=
### I noticed that the series of displayed `Mean[ELBO]` values is *not* monotonically increasing. Shouldn't the ELBO keep increasing during the variational optimization?

Well spotted, but nothing to worry about. VBMC keeps updating its *estimate* of the ELBO, which means that this value may *decrease* across iterations, and sometimes (especially at the beginning) it will oscillate.
This is part of the normal functioning of VBMC. However, if the ELBO keeps oscillating wildly even at later stages in the inference, you may be in trouble (see [below](#faq-troubleshooting)).

(faq-troubleshooting)=
## Troubleshooting

(faq-how-can-vbmc-fail-how-do-i-find-out-and-how-do-i-fix-it)=
### How can VBMC fail, how do I find out, and how do I fix it?

So many questions. VBMC can fail in *many* ways, some of which blatant, some which harder to detect — similarly to other inference methods such as Markov Chain Monte Carlo. For this reason, you should always check and validate the solutions found by VBMC.

We list here the major failure modes, diagnostics, and possible solutions:

- In the simplest case, the VBMC algorithm fails to converge within the allotted budget of target function evaluations.
  - This is easy to detect (look at `results["success_flag"]` and `results["message"]`). There may be several distinct reasons for failure of convergence (see below).
- VBMC converges, but the variational optimization has only found a *local* optimum.
  - You cannot detect this issue by looking at a *single* VBMC run. For this reason, I recommend to run several VBMC runs from different starting points (at least 3-4) and compare the solutions, for example via visual inspection of the posteriors and comparing their ELBOs and posterior distances (see also [Example 4](_examples/pyvbmc_example_4_validation.ipynb)).
- Multiple runs of VBMC converge to pretty much the same variational solution, which fails to capture important aspects of the true posterior.
  - This problem has no obvious solution, in that it is intrinsic to the fact that we are using an approximation that it may deviate from the true posterior. For example, variational posteriors, for how the variational objective is defined, tend to underestimate the true uncertainty of the posterior. You can use [*posterior predictive checks*](https://stats.stackexchange.com/questions/115157/what-are-posterior-predictive-checks-and-what-makes-them-useful) to gain confidence that the found solution makes sensible predictions, also in terms of calibration.

(faq-vbmc-warned-me-that-the-returned-variational-solution-may-have-not-converged-what-should-i-do)=
### VBMC warned me that the `Returned variational solution may have not converged.` What should I do?

This warning means that the returned variational solution was not marked stable. A common reason is that VBMC reaches the maximum number of function evaluations before the solution has stabilized according to the reliability index. To understand better what's going on, and how you could solve this issue, have a look at the VBMC trace across iterations:

- On average, `Mean[ELBO]` has been increasing while `Std[ELBO]` has been decreasing over the last iterations, with some fluctuations. This suggests that VBMC is converging, it probably just needs more iterations. [Continue the saved run](#faq-how-do-i-save-and-continue-a-run) with a larger total evaluation budget.

- `Mean[ELBO]` and `Std[ELBO]` have been fluctuating wildly across iterations. This suggests that VBMC cannot build a reasonable Gaussian process (GP) approximation of the target posterior. There could be a number of reasons for that, but in many cases, the GP approximation is failing because VBMC has accidentally sampled one or a few points with *extremely* low posterior probability density. Because of these points, the GP approximation ends up having a very high output scale parameter (that is, the GP models a wildly fluctuating function). Sometimes, starting with a *narrower* plausible range, which only includes a region of (relatively) high posterior probability mass, is enough to prevent this kind of divergence.

If you change the target, prior or bounds, start a new `VBMC` instance.
PyVBMC does not implement MATLAB VBMC's `RetryMaxFunEvals` automated retry
option.

(faq-vbmc-crashes-saying-that-the-returned-function-value-must-be-a-finite-real-valued-scalar-what-do-i-do)=
### VBMC crashes saying that `The returned function value must be a finite real-valued scalar`. What do I do?

This error means that your target function has returned `Inf`, `NaN`, or a complex number. You should check your code and understand why it returned a non-finite or complex value.

`Inf`s and `NaN`s often arise because there are outcomes in your dataset (e.g., responses in a trial) to which
the tested model assigns a probability of `0` (usually due to numerical truncation). `log(0)` yields `-Inf`,
which is then propagated. If the zero is due to numerical underflow, compute the likelihood directly in log space, using log-density functions and stable operations such as `scipy.special.logsumexp`. If the model truly assigns zero probability to an observed outcome, check the model assumptions. A non-zero *lapse rate* or contamination component may be appropriate if it describes the data-generating process; an arbitrary probability floor changes the likelihood and its model evidence.

Complex numbers usually arise because you are taking either a `log` or a `sqrt` of a negative number (of a quantity that should not be negative). You might be setting wrong bounds for your variables, or maybe there are indexing issues.

Note that some optimizers are robust to `Inf`s and `NaN`s and just keep going, avoiding the problematic region in parameter space. For this reason, you might have been able to fit the same models before with an optimizer, without encountering errors. However, ignoring errors in the likelihood can be dangerous as it might hide deeper issues with the model implementation.

(faq-i-have-been-running-vbmc-from-the-same-starting-point-but-i-get-different-results-each-time-is-something-wrong)=
### I have been running VBMC from the same starting point, but I get different results each time. Is something wrong?

Nothing is wrong per se. VBMC uses stochastic optimization, so results will differ between different runs, even with the same initial condition `x0`. The question is whether the returned `elbo` or variational posterior `vp` vary *substantially* across runs from the same starting point. If so, it might be a sign that your posterior landscape is particularly difficult.

If you want to have reproducible results (and this advice applies beyond VBMC), pass a known seed when constructing the instance (e.g., setting it to the run number):

```python
vbmc = VBMC(fun, x0, LB, UB, PLB, PUB, seed=42)
```

Use the same seed, inputs, options and numerical environment to reproduce
a run. `seed=` also accepts a NumPy `Generator`; reusing a generator
continues its stream. The posterior shares the run's generator, so drawing
samples before continuing inference advances that stream.

If your target function itself is stochastic, manage its random generator
separately: `VBMC(seed=...)` does not seed the target. Keep drawing fresh
noise at each target call; fixing the algorithm's seed is different from
[freezing the target's noise](#faq-wait-can-i-just-remove-stochasticity-by-fixing-the-random-seed-at-each-function-call).

(faq-how-do-i-save-and-continue-a-run)=
### How do I save and continue a run?

Save the `VBMC` instance so you can continue the inference without starting
over:

```python
vbmc.save("fit.pkl")
```

To give it more evaluations, load it with a larger total evaluation budget
and call `optimize()`:

```python
continued = VBMC.load("fit.pkl", new_options={"max_fun_evals": 1000})
vp, results = continued.optimize()
```

Here `1000` is the *total* budget, including evaluations already made. If
the run reached its iteration limit, increase `max_iter` as well. Continue
with the same model, data, prior and bounds, preferably in the same Python
environment. See [`VBMC.save` and `VBMC.load`](api/classes/vbmc.rst).

If you only need to use the fitted posterior later, save it with
`vp.save("posterior.pkl")` and load it with
`VariationalPosterior.load("posterior.pkl")`. A saved posterior alone does
not contain the full state needed to resume a run.

(faq-can-i-combine-the-posteriors-of-several-runs)=
### Can I combine the posteriors of several runs?

Yes. S-VBMC (Stacking Variational Bayesian Monte Carlo) combines the
posteriors of several VBMC runs on the same model and data, without new
model evaluations. It can be useful when individual runs capture different
parts of the posterior. First check the individual runs and investigate
large disagreements; stacking cannot recover a region missed by every run.

With the optional `torch` extra installed, collect the returned posteriors
in a list `vps` and run:

```python
from pyvbmc import SVBMC

stacked = SVBMC(vps)
stacked.optimize()
samples = stacked.sample(10000)
```

The headline estimate is `stacked.elbo`; `stacked.elbo_details` contains
the detailed estimates. On noisy targets the headline uses a cap to reduce
optimism. `stacked.elbo_sd` describes uncertainty in the raw estimate,
not a confidence interval for that capped headline. See the
[S-VBMC documentation](api/classes/svbmc.rst) and
[Example 7](_examples/pyvbmc_example_7_stacking.ipynb)
for the filtering of input runs, reporting and examples.

(faq-miscellanea)=
## Miscellanea

(faq-i-already-run-pybads-on-my-problem-how-do-i-run-vbmc)=
### I already run PyBADS on my problem. How do I run VBMC?

The interface of VBMC is very similar to the one of PyBADS, and you may only need minor or even no changes to run VBMC on your problem. Note that:

- Unlike PyBADS, *plausible bounds* (`PLB` and `PUB`) in VBMC need to be distinct from the hard bounds `LB` and `UB` (see [here](#faq-how-do-i-choose-plb-and-pub)).
- Also, hard bounds in VBMC have slightly [different meaning and constraints](#faq-how-do-i-choose-lb-and-ub).
- The target function `fun` of VBMC is a [log posterior density](#faq-what-is-the-target-function) — that is, [you have to specify a prior](#faq-wait-do-i-need-a-prior);
- Beware of the sign! The target function in VBMC is the log-joint, that is log-likelihood plus log-prior — this differs from PyBADS, which *minimizes* the objective (for example, maximum-likelihood estimation in PyBADS would take as input the *negative* log-likelihood);
- You can use as starting point `x0` for VBMC the result of a [preliminary optimization](#faq-how-do-i-choose-the-starting-point-x0) with PyBADS, to ensure you are starting from a region of high posterior density;
- VBMC supports inference with noisy target functions, but it has more requirements than in PyBADS (see [this section](#faq-noisy-target-function));
- Finally, VBMC does not offer a special support for circular or periodic coordinates, such as angles — just treat them as normal bounded coordinates, setting hard bounds on the length of one period (e.g, `LB = 0` and `UB = 2 * np.pi`).

(faq-this-is-interesting-but-why-is-it-better-than-just-using-point-estimates-eg-maximum-likelihood-or-maximum-a-posteriori)=
### This is interesting, but why is it better than just using point estimates? (e.g., maximum-likelihood or maximum-a-posteriori)

There are several reasons for which having access to the full Bayesian posterior is desirable:

- Knowing the uncertainty in the parameter estimates;
- Making overall better model predictions (that is, more accurate and more calibrated) — for example, if the posterior is asymmetric or multi-modal, it is badly characterized by a single point estimate and model predictions can be unreasonably confident (in other words, you may overfit);
- Understanding features of the model, such as trade-offs between parameters, which may not be obvious.

Also, the log model evidence (as approximated by its lower bound, the ELBO) is a principled metric for Bayesian model comparison that takes into account goodness of fit and actual model complexity. With point estimates, you can only compute simple metrics such as Akaike's Information Criterion (AIC) or the "Bayesian" Information Criterion (BIC), which know very little about the structure of your model.

(faq-thanks-but-i-think-i-will-stick-to-point-estimates-for-now)=
### Thanks, but I think I will stick to point estimates for now.

Fair enough — just be aware that you may be missing important features of your model and data, and your estimates and predictions might be way overconfident.

If you only care about finding point estimates, you might want to have a look at our method for efficient Bayesian optimization, [Bayesian Adaptive Direct Search in Python (PyBADS)](https://acerbilab.github.io/pybads/).

(faq-are-you-planning-to-port-vbmc-to-other-languages)=
### Are you planning to port VBMC to other languages?

VBMC is available in MATLAB [here](https://github.com/acerbilab/vbmc) and in Python as the [PyVBMC package](https://github.com/acerbilab/pyvbmc). Currently, there is no plan to port the algorithm to other languages, since there are ways to run the Python version from several other languages (for example, our users have been able to run PyVBMC from [Julia](https://cjdoris.github.io/PythonCall.jl/stable/)).
