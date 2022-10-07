******
PyVBMC
******
PyVBMC is a numerical Python 3.x implementation of Variational Bayesian Monte Carlo (VBMC).



What is it?
########################

  Can we perform Bayesian inference with expensive, black-box models?

VBMC is an approximate Bayesian inference method designed to fit computational models with a limited budget of potentially noisy likelihood evaluations, useful for computationally expensive models or for quick inference and model evaluation `(Acerbi, 2018; 2020) <#references>`_.
PyVBMC works with *black-box* models in that it only needs to evaluate an unnormalized target log density (e.g., an unnormalized target log posterior).

PyVBMC simultaneously computes:

- an approximate posterior distribution of the model parameters;

- an approximation — technically, an approximate lower bound — of the log model evidence (also known as log marginal likelihood or log Bayes factor), a metric used for Bayesian model selection.


Example run
-----------

The figure below shows an example PyVBMC run on a "banana" target density.
The corner plot shows the approximate posterior across iterations (contour plot and histograms of the marginals).
The dots represent evaluations of the target density (*blue*: previously sampled points, *green*: points sampled in the current iteration).
PyVBMC converges to an excellent approximation of the true posterior with a few dozens evaluations of the target density.

.. image:: _static/vbmc_animation.gif
    :width: 400px
    :align: center
    :height: 400px
    :alt: Animation of PyVBMC

Extensive benchmarks on both artificial test problems and a large number of real model-fitting problems from computational and cognitive neuroscience show that VBMC generally — and often vastly — outperforms alternative methods for sample-efficient Bayesian inference. VBMC runs with virtually no tuning and it is very easy to set up for your problem.


Should I use PyVBMC?
------------------------

PyVBMC is effective when:

- the model log-likelihood function is a black-box (e.g., the gradient is unavailable);
- the likelihood is at least moderately expensive to compute (say, half a second or more per evaluation);
- the model has up to ``D = 10`` continuous parameters (maybe a few more, but no more than ``D = 20``);
- the target posterior density is continuous and reasonably smooth;
- the log-likelihood can be evaluated *exactly* (e.g., analytically or numerically, but without noise in the evaluation itself).

Conversely, if your model can be written in closed form and is fast to evaluate, you should exploit the powerful machinery of probabilistic programming frameworks such as `Stan <https://mc-stan.org/>`_ or `PyMC3 <https://docs.pymc.io/>`_.


.. note::
    This project is work in progress. The current Python port implements VBMC with *exact* evaluations of the likelihood (Acerbi, 2018). We are currently planning to extend PyVBMC to support *noisy* likelihood evaluations, such as those arising from simulator-based models (Acerbi, 2020).

Documentation
#############
.. toctree::
   :maxdepth: 1
   :titlesonly:

   installation
   quickstart
   api/classes/vbmc
   api/classes/variational_posterior
   api/options/vbmc_options
   api/advanced_docs

Examples
########
.. toctree::
   :maxdepth: 1
   :titlesonly:
   :glob:

   _examples/*

References
###############

1. Acerbi, L. (2018). Variational Bayesian Monte Carlo. In *Advances in Neural Information Processing Systems 31*: 8222-8232. (`paper + supplement on arXiv <https://arxiv.org/abs/1810.05558>`__, `NeurIPS Proceedings <https://papers.nips.cc/paper/2018/hash/747c1bcceb6109a4ef936bc70cfe67de-Abstract.html>`__)
2. Acerbi, L. (2020). Variational Bayesian Monte Carlo with Noisy Likelihoods. In *Advances in Neural Information Processing Systems 33*: 8211-8222 (`paper + supplement on arXiv <https://arxiv.org/abs/2006.08655>`__, `NeurIPS Proceedings <https://papers.nips.cc/paper/2020/hash/5d40954183d62a82257835477ccad3d2-Abstract.html>`__).

You can cite VBMC in your work with something along the lines of

    We estimated approximate posterior distibutions and approximate lower bounds to the model evidence of our models using Variational Bayesian Monte Carlo (VBMC; Acerbi, 2018, 2020) via the PyVBMC software. VBMC combines variational inference and active-sampling Bayesian quadrature to perform approximate Bayesian inference in a sample-efficient manner.

License and source
------------------

PyVBMC is licensed under `BSD3 <https://opensource.org/licenses/BSD-3-Clause>`_.
The Python source code is on `GitHub <https://github.com/lacerbi/pyvbmc>`_.
You may also want to check out the original `MATLAB toolbox <https://github.com/lacerbi/vbmc>`_.


Acknowledgments
###############
Work on the PyVBMC package was funded by the `Finnish Center for Artificial Intelligence FCAI <https://fcai.fi/>`_.

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :hidden:

   about_us
