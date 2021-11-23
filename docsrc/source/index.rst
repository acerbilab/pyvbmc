******
PyVBMC
******
*What if there was a model-fitting method similar to Bayesian optimization, e.g.,* `BADS <https://github.com/lacerbi/bads>`_, *which, instead of returning just the optimal parameter vector, would also return its uncertainty (even better, the full posterior distribution of the parameters), and maybe even a metric than can be used for Bayesian model comparison?*

.. image:: _static/vbmc_animation.gif
    :width: 400px
    :align: center
    :height: 400px
    :alt: Animation of PyVBMC

VBMC is an approximate inference method designed to fit and evaluate computational models with a limited budget of potentially noisy likelihood evaluations (e.g., for computationally expensive models). Specifically, VBMC simultaneously computes:

- an approximate posterior distribution of the model parameters;

- an approximation — technically, an approximate lower bound — of the log model evidence (also known as log marginal likelihood or log Bayes factor), a metric used for Bayesian model selection.

Extensive benchmarks on both artificial test problems and a large number of real model-fitting problems from computational and cognitive neuroscience show that VBMC generally — and often vastly — outperforms alternative methods for sample-efficient Bayesian inference.  

VBMC runs with virtually no tuning and it is very easy to set up for your problem.

This repository contains the port of the VBMC algorithm to Python 3.x, called ``pyvbmc``. 
The original source is the `MATLAB toolbox <https://github.com/lacerbi/vbmc>`_.

.. warning::
    This project is work in progress!

Documentation
#############
.. toctree::
   :maxdepth: 1
   :titlesonly:

   installation
   api/classes/variational_posterior
   api/classes/vbmc
   api/advanced_docs

Examples
########
.. toctree::
   :maxdepth: 1
   :titlesonly:
   :glob:

   _examples/*

Attribution
###########

tbd

Authors & License
#################

tbd

Acknowledgments
###############
Work on the pyvbmc package was funded by the `Finnish Center for Artificial Intelligence FCAI <https://fcai.fi/>`_.
