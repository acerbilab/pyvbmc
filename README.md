![PyVBMC](https://user-images.githubusercontent.com/70731267/225583733-6cc22c33-a198-4868-90be-3818206d1398.svg)

# PyVBMC: Variational Bayesian Monte Carlo in Python
![Version](https://img.shields.io/badge/dynamic/json?label=python&query=info.requires_python&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fpyvbmc%2Fjson)
[![Conda](https://img.shields.io/conda/v/conda-forge/pyvbmc)](https://anaconda.org/conda-forge/pyvbmc)
[![PyPI](https://img.shields.io/pypi/v/pyvbmc)](https://pypi.org/project/pyvbmc/)
<br />
[![Discussion](https://img.shields.io/badge/-discussion-blue?logo=github)](https://github.com/orgs/acerbilab/discussions)
[![tests](https://img.shields.io/github/actions/workflow/status/acerbilab/pyvbmc/tests.yml?branch=main&label=tests)](https://github.com/acerbilab/pyvbmc/actions/workflows/tests.yml)
[![docs](https://img.shields.io/github/actions/workflow/status/acerbilab/pyvbmc/docs.yml?branch=main&label=docs)](https://github.com/acerbilab/pyvbmc/actions/workflows/docs.yml)
[![build](https://img.shields.io/github/actions/workflow/status/acerbilab/pyvbmc/build.yml?branch=main&label=build)](https://github.com/acerbilab/pyvbmc/actions/workflows/build.yml)
### What is it?

PyVBMC is a Python implementation of the Variational Bayesian Monte Carlo (VBMC) algorithm for posterior and model inference, previously implemented [in MATLAB](https://github.com/acerbilab/vbmc). VBMC is an approximate inference method designed to fit and evaluate Bayesian models with a limited budget of potentially noisy likelihood evaluations (e.g., for computationally expensive models). Specifically, VBMC simultaneously computes:
- an approximate posterior distribution of the model parameters;
- an approximation — technically, an approximate lower bound — of the log model evidence (also known as log marginal likelihood), a metric used for [Bayesian model selection](https://en.wikipedia.org/wiki/Bayes_factor).

Extensive benchmarks on both artificial test problems and a large number of real model-fitting problems from computational and cognitive neuroscience show that VBMC generally — and often vastly — outperforms alternative methods for sample-efficient Bayesian inference [[2,3](#references-and-citation)].

### What's new in PyVBMC 1.5

PyVBMC 1.5 is faster and more efficient, integrates better with the modern scientific computing ecosystem and our other tools, and provides more guidance for using it effectively. Highlights include:

- **Faster inference and lower memory use**, with numerical improvements and more compact run histories. Optional [performance calibration](#optional-performance-calibration) tunes PyVBMC for your machine.
- **Stacking Variational Bayesian Monte Carlo (S-VBMC)** is included in PyVBMC to combine posteriors from independent runs ([Silvestrin et al., 2025](https://arxiv.org/abs/2504.05004); [usage below](#combine-runs-and-use-the-posterior-downstream)).
- **Explicit random seed control** for reproducing individual runs; see the [reproducibility guide](https://acerbilab.github.io/pyvbmc/quickstart.html#reproducible-runs).
- **Torch and JAX model integration** through small user-written wrappers, with optional batch evaluation of the initial points; see the [model integration guide](https://acerbilab.github.io/pyvbmc/quickstart.html#bring-a-torch-or-jax-model-into-pyvbmc).
- **Direct PyMC model support**, including model-aware coordinates, automatic setup, and structured posterior export; see the [PyMC integration guide](https://acerbilab.github.io/pyvbmc/quickstart.html#bring-a-pymc-model-into-pyvbmc).
- **Posterior exports to Torch and ArviZ** for further analysis; see the [export guide](https://acerbilab.github.io/pyvbmc/quickstart.html#use-a-fitted-posterior-downstream).
- **More practical guidance**, with tips during runs, a [PyVBMC FAQ](https://acerbilab.github.io/pyvbmc/faq.html), and a [coding-agent skill](skills/pyvbmc/SKILL.md) that points agents to the relevant documentation.

The [changelog](CHANGELOG.md) lists what changed since PyVBMC 1.0.4, including the corrections that make results differ from earlier versions and what to check in an existing script.

### Documentation

The full documentation is available at: https://acerbilab.github.io/pyvbmc/

For coding agents, the [PyVBMC skill](skills/pyvbmc/SKILL.md) points to the
documentation relevant to each task. Give your agent that file, or copy the
`skills/pyvbmc` folder into its skill directory. To update a copied skill,
copy the folder again from the PyVBMC version you use.

### When should I use PyVBMC?

PyVBMC is effective when:

- the model log-likelihood function is a black-box (e.g., the gradient is unavailable);
- the likelihood is at least moderately expensive to compute (say, half a second or more per evaluation);
- the model has up to `D = 10` continuous parameters (maybe a few more, but no more than `D = 20`);
- the target posterior distribution is continuous and reasonably smooth (see [here](https://acerbilab.github.io/pyvbmc/faq.html#faq-general));
- optionally, log-likelihood evaluations may be noisy (e.g., estimated [via simulation](https://github.com/acerbilab/ibs)).

Conversely, if your model can be written analytically and is fast to evaluate, you should exploit the powerful machinery of probabilistic programming frameworks such as [Stan](https://mc-stan.org/) or [PyMC](https://www.pymc.io/); PyMC users with an expensive supported model can pass it to PyVBMC through [`PyMCTarget`](https://acerbilab.github.io/pyvbmc/api/classes/pymc_target.html).

Note: If you are interested in point estimates or in finding better starting points for PyVBMC, check out [Bayesian Adaptive Direct Search in Python (PyBADS)](https://github.com/acerbilab/pybads), our companion method for fast Bayesian optimization.

## Installation

PyVBMC is available via `pip` and `conda-forge`.

1. Install with:
    ```console
    python -m pip install pyvbmc
    ```
    or:
    ```console
    conda install --channel=conda-forge pyvbmc
    ```
    PyVBMC requires Python version 3.10 or newer.

2. (Optional): Install [Jupyter Notebook](https://jupyter.org/install) to run the examples. You can skip this step if your environment already has Jupyter Notebook, but be aware that if the wrong `jupyter` executable is found on your path then import errors may arise.
   ```console
   python -m pip install notebook
   ```
   or, with Conda:
   ```console
   conda install --channel=conda-forge jupyter
   ```
   The example notebooks can then be accessed by running
   ```console
   python -m pyvbmc
   ```

Optional integrations are installed separately. S-VBMC and the Torch posterior
export ([see below](#combine-runs-and-use-the-posterior-downstream)) use the
`torch` extra. For a CPU-only torch setup,
install the official CPU wheel first, followed by the PyVBMC extra:
```console
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install "pyvbmc[torch]"
```
The torch integration requires torch 2.7 or newer. The current ArviZ export
requires Python 3.12 or newer and is installed with:
```console
python -m pip install "pyvbmc[arviz]"
```
The PyMC adapter also requires Python 3.12 or newer and is installed with:
```console
python -m pip install "pyvbmc[pymc]"
```
Conda users can install the corresponding `pytorch`, `arviz`, `arviz-base`,
and `pymc` packages from `conda-forge`; PyVBMC's `torch`, `arviz`, and `pymc`
extras are pip dependency groups.

If you wish to install directly from latest source code, please see the [instructions for developers and contributors](https://acerbilab.github.io/pyvbmc/development.html#installation-instructions-for-developers).

### Optional performance calibration

Optionally, we recommend running calibration once to tune performance for
your machine (memory, processor, etc.):

```python
from pyvbmc import calibrate
calibrate()
```

Run this when your machine is otherwise idle. It takes tens of seconds and
can make PyVBMC run faster. The settings are saved and reused automatically
by future runs with a compatible numerical environment. See the
[calibration guide](https://acerbilab.github.io/pyvbmc/api/functions/calibrate.html)
for details.

## Quick start
A typical PyVBMC workflow follows four steps:

1. Define the model, which defines a target log density (i.e., an unnormalized log posterior density);
2. Set up the parameters (parameter bounds, starting point);
3. Initialize and run the inference;
4. Examine and visualize the results.

PyVBMC is not concerned with how you define your model in step 1, as long as you can provide an (unnormalized) target log density. Running the inference in step 3 only involves a couple of lines of code:
```python
from pyvbmc import VBMC
# ... define your model/target density here
vbmc = VBMC(target, x0, LB, UB, PLB, PUB)
vp, results = vbmc.optimize()
```
with input arguments:
- `target`: the target (unnormalized) log density — often an unnormalized log posterior. `target` is a callable that should take as input a parameter vector and return the log density at the point. The returned log density must be a *finite* real value, i.e. neither `NaN` nor `+/-inf`. See the [PyVBMC FAQ](https://acerbilab.github.io/pyvbmc/faq.html#faq-how-do-i-prevent-vbmc-from-evaluating-certain-inputs-or-regions-of-input-space) for more details;
- `x0`: an array representing the starting point of the inference in parameter space;
- `LB` and `UB`: arrays of hard lower (resp. upper) bounds constraining the parameters (possibly `-/+np.inf` for unbounded parameters);
- `PLB` and `PUB`: arrays of plausible lower (resp. upper) bounds: that is, a box that ideally brackets a high posterior density region of the target.

The optional `seed` argument controls PyVBMC's internal random generator for reproducible runs.
For independent runs, leave `seed` unset or use different seeds. See the [reproducibility guide](https://acerbilab.github.io/pyvbmc/quickstart.html#reproducible-runs) for more details.

The outputs are:
- `vp`: a `VariationalPosterior` object which approximates the true target density;
- `results`: a `dict` containing additional information. Important keys are:
  - `"elbo"`: the estimated lower bound on the log model evidence (log normalization constant);
  - `"elbo_sd"`: the standard deviation of the estimate of the ELBO (*not* the error between the ELBO and the true log model evidence, which is generally unknown).

The `vp` object can be manipulated in various ways. For example, we can draw samples from `vp` with the `vp.sample()` method, or evaluate its density at a point with `vp.pdf()` (or log-density with `vp.log_pdf()`). See the [`VariationalPosterior` class documentation](https://acerbilab.github.io/pyvbmc/api/classes/variational_posterior.html) for details.

### PyVBMC with noisy targets

The quick start example above works for deterministic (noiseless) evaluations of the target log-density. PyVBMC also supports *noisy* evaluations of the target.
Noisy evaluations often arise from simulation-based models, for which a direct expression of the (log) likelihood is not available.

For information on how to run PyVBMC on a noisy target, see [this example notebook](examples/pyvbmc_example_6_noisy_likelihoods.ipynb) and the [PyVBMC FAQ](https://acerbilab.github.io/pyvbmc/faq.html#faq-noisy-target-function).

Note that `VBMC(seed=...)` only controls the internal random stream of PyVBMC and not
that of your simulations. Reproducing the analysis also requires setting the
simulator's seed before the run. See the [reproducibility guide](https://acerbilab.github.io/pyvbmc/quickstart.html#reproducible-runs) for how to do this.

### Combine runs and use the posterior downstream

Stacking Variational Bayesian Monte Carlo (S-VBMC; [Silvestrin et al., 2025](https://arxiv.org/abs/2504.05004)) is a technique to combine the
posteriors of several completed PyVBMC runs on the same model and data into a stacked
posterior. Stacking the posteriors almost always provides a better approximation
of the true posterior (sometimes much better), and it works as a post-processing step that
does not require further model evaluations. See [Example 7](examples/pyvbmc_example_7_stacking.ipynb) and the
[`SVBMC` documentation](https://acerbilab.github.io/pyvbmc/api/classes/svbmc.html).

A fitted `VariationalPosterior` can also be exported as a Torch distribution
with `vp.to_torch()`, or as samples in an ArviZ DataTree with `vp.to_arviz()`.
See the [posterior export guide](https://acerbilab.github.io/pyvbmc/quickstart.html#use-a-fitted-posterior-downstream)
and [PyVBMC Example 9: Torch and JAX models and posterior exports](examples/pyvbmc_example_9_torch_jax.ipynb)
for worked examples, including posterior predictions and density gradients in
Torch and summaries in ArviZ.

## Next steps

Once installed, example Jupyter notebooks can be found in the `pyvbmc/examples` directory. They can also be [viewed statically](https://acerbilab.github.io/pyvbmc/examples.html) on the [main documentation pages](https://acerbilab.github.io/pyvbmc/index.html). These examples will walk you through the basic usage of PyVBMC as well as some of its more advanced features.

For practical recommendations, such as how to set `LB` and `UB` and the plausible bounds, check out the [PyVBMC FAQ](https://acerbilab.github.io/pyvbmc/faq.html).

## How does it work?

VBMC/PyVBMC combine two machine learning techniques in a novel way:
- [variational inference](https://en.wikipedia.org/wiki/Variational_Bayesian_methods), a method to perform approximate Bayesian inference;
- [Bayesian quadrature](https://en.wikipedia.org/wiki/Bayesian_quadrature), a technique to estimate the value of expensive integrals.

PyVBMC iteratively builds an approximation of the true, expensive target posterior via a [Gaussian process](https://distill.pub/2019/visual-exploration-gaussian-processes/) (GP), and it matches a variational distribution — an expressive mixture of Gaussians — to the GP.

This matching process entails optimization of the *evidence lower bound* (ELBO), that is a lower bound on the log marginal likelihood (LML), also known as log model evidence. Crucially, we estimate the ELBO via Bayesian quadrature, which is fast and does not require further evaluation of the true target posterior.

In each iteration, PyVBMC uses *active sampling* to select which points to evaluate next in order to explore the posterior landscape and reduce uncertainty in the approximation.

![VBMC Demo](https://user-images.githubusercontent.com/70731267/225584285-5b90f78f-2ed9-4844-b0fa-ad5d4ab29923.gif)

In the figure above, we show an example PyVBMC run on a [Rosenbrock "banana" function](https://en.wikipedia.org/wiki/Rosenbrock_function). The bottom-left panel shows PyVBMC at work: in grayscale are samples from the variational posterior (drawn as small points) and the corresponding estimated density (drawn as contours). The solid orange circles are the active sampling points chosen at each iteration, and the hollow blue circles are the previously sampled points. The topmost and rightmost panels show histograms of the marginal densities along the $x_1$ and $x_2$ dimensions, respectively. PyVBMC converges to an excellent approximation of the true posterior with a few dozen evaluations of the target density.

See the VBMC papers [[1-3](#references-and-citation)] for more details.

## Troubleshooting and contact

PyVBMC is under active development. The VBMC algorithm has been extensively tested in several benchmarks and published papers, and the benchmarks have been replicated using PyVBMC. But as with any approximate inference technique, you should double-check your results. See the [examples](examples) for descriptions of the convergence diagnostics and suggestions on validating PyVBMC's results with multiple runs.

If you have trouble doing something with PyVBMC, spot bugs or strange behavior, or you simply have some questions, please feel free to:
- Post in the lab's [Discussions forum](https://github.com/orgs/acerbilab/discussions) with questions or comments about PyVBMC, your problems & applications;
- [Open an issue](https://github.com/acerbilab/pyvbmc/issues/new) on GitHub;
- Contact the project lead at <luigi.acerbi@helsinki.fi>, putting 'PyVBMC' in the subject of the email.

## References and citation

1. Huggins, B., Li, C., Tobaben, M., Aarnos, M., & Acerbi, L. (2023). [PyVBMC: Efficient Bayesian inference in Python](https://joss.theoj.org/papers/10.21105/joss.05428). *Journal of Open Source Software* 8(86), 5428, https://doi.org/10.21105/joss.05428.
2. Acerbi, L. (2018). Variational Bayesian Monte Carlo. In *Advances in Neural Information Processing Systems 31*: 8222-8232. ([paper + supplement on arXiv](https://arxiv.org/abs/1810.05558), [NeurIPS Proceedings](https://papers.nips.cc/paper/8043-variational-bayesian-monte-carlo))
3. Acerbi, L. (2020). Variational Bayesian Monte Carlo with Noisy Likelihoods. In *Advances in Neural Information Processing Systems 33*: 8211-8222 ([paper + supplement on arXiv](https://arxiv.org/abs/2006.08655), [NeurIPS Proceedings](https://papers.nips.cc/paper/2020/hash/5d40954183d62a82257835477ccad3d2-Abstract.html)).

Please cite all three references if you use PyVBMC in your work (the 2018 paper introduced the framework, and the 2020 paper includes a number of major improvements, including but not limited to support for noisy likelihoods). You can cite PyVBMC in your work with something along the lines of

> We estimated approximate posterior distributions and approximate lower bounds to the model evidence of our models using Variational Bayesian Monte Carlo (PyVBMC; Acerbi, 2018, 2020) via the PyVBMC software (Huggins et al., 2023). PyVBMC combines variational inference and active-sampling Bayesian quadrature to perform approximate Bayesian inference in a sample-efficient manner.

If you use S-VBMC (see [Additional references](#additional-references)), please also add a sentence such as:

> Posteriors from multiple PyVBMC runs on the same model and dataset were combined using S-VBMC (Silvestrin et al., 2025), which often improves the approximation to the true posterior by leveraging information from independent runs.

Besides formal citations, you can demonstrate your appreciation for PyVBMC in the following ways:

- *Star :star:* the VBMC repository on GitHub;
- [Subscribe](http://eepurl.com/idcvc9) to the lab's newsletter for news and updates (new features, bug fixes, new releases, etc.);
- [Follow Luigi Acerbi on Twitter](https://twitter.com/AcerbiLuigi) for updates about VBMC/PyVBMC and other projects;
- Tell us about your model-fitting problem and your experience with PyVBMC (positive or negative) in the lab's [Discussions forum](https://github.com/orgs/acerbilab/discussions).

You may also want to check out [Bayesian Adaptive Direct Search in Python (PyBADS)](https://github.com/acerbilab/pybads), our companion method for fast Bayesian optimization.

### Additional references

4. Acerbi, L. (2019). An Exploration of Acquisition and Mean Functions in Variational Bayesian Monte Carlo. In *Proc. Machine Learning Research* 96: 1-10. 1st Symposium on Advances in Approximate Bayesian Inference, Montréal, Canada. ([paper in PMLR](http://proceedings.mlr.press/v96/acerbi19a.html))

5. Silvestrin, F., Li, C., & Acerbi, L. (2025). Stacking Variational Bayesian Monte Carlo. *Transactions on Machine Learning Research*. ([paper + supplement on arXiv](https://arxiv.org/abs/2504.05004), [TMLR](https://openreview.net/forum?id=M2ilYAJdPe))

### BibTeX

```BibTeX
@article{huggins2023pyvbmc,
    title = {PyVBMC: Efficient Bayesian inference in Python},
    author = {Bobby Huggins and Chengkun Li and Marlon Tobaben and Mikko J. Aarnos and Luigi Acerbi},
    publisher = {The Open Journal},
    journal = {Journal of Open Source Software},
    url = {https://doi.org/10.21105/joss.05428},
    doi = {10.21105/joss.05428},
    year = {2023},
    volume = {8},
    number = {86},
    pages = {5428}
  }

@article{acerbi2018variational,
  title={{V}ariational {B}ayesian {M}onte {C}arlo},
  author={Acerbi, Luigi},
  journal={Advances in Neural Information Processing Systems},
  volume={31},
  pages={8222--8232},
  year={2018}
}

@article{acerbi2020variational,
  title={{V}ariational {B}ayesian {M}onte {C}arlo with noisy likelihoods},
  author={Acerbi, Luigi},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={8211--8222},
  year={2020}
}

@article{silvestrin2025stacking,
  title={Stacking {V}ariational {B}ayesian {M}onte {C}arlo},
  author={Silvestrin, Francesco and Li, Chengkun and Acerbi, Luigi},
  journal={Transactions on Machine Learning Research},
  year={2025},
  url={https://openreview.net/forum?id=M2ilYAJdPe}
}

@article{acerbi2019exploration,
  title={An Exploration of Acquisition and Mean Functions in {V}ariational {B}ayesian {M}onte {C}arlo},
  author={Acerbi, Luigi},
  journal={PMLR},
  volume={96},
  pages={1--10},
  year={2019}
}
```

### License

PyVBMC is released under the terms of the [BSD 3-Clause License](LICENSE).

### Acknowledgments

PyVBMC is developed by [members](https://www.helsinki.fi/en/researchgroups/machine-and-human-intelligence/people) (past and current) of the [Machine and Human Intelligence Lab](https://www.helsinki.fi/en/researchgroups/machine-and-human-intelligence/) at the University of Helsinki and [ELLIS Institute Finland](https://www.ellisinstitute.fi/). Development of PyVBMC 1.5 was assisted by coding agents, including Anthropic's [Claude Fable 5.1](https://www.anthropic.com/claude-fable-and-mythos-5-1) and OpenAI's [GPT-6 Astra](https://developers.openai.com/api/docs/models/gpt-6-astra).
Work on the PyVBMC package is supported by the Research Council of Finland (grants 356498 and 358980 to Luigi Acerbi) and its Flagship programme: [Finnish Center for Artificial Intelligence FCAI](https://fcai.fi/).
