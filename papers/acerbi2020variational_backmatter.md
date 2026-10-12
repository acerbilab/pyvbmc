# Variational Bayesian Monte Carlo with Noisy Likelihoods - Backmatter

---

#### Page 10

## Broader Impact

We believe this work has the potential to lead to net-positive improvements in the research community and more broadly in society at large. First, this paper makes Bayesian inference accessible to non-cheap models with noisy log-likelihoods, allowing more researchers to express uncertainty about their models and model parameters of interest in a principled way; with all the advantages of proper uncertainty quantification [2]. Second, with the energy consumption of computing facilities growing incessantly every hour, it is our duty towards the environment to look for ways to reduce the carbon footprint of our algorithms [52]. In particular, traditional methods for approximate Bayesian inference can be extremely sample-inefficient. The 'smart' sample-efficiency of VBMC can save a considerable amount of resources when model evaluations are computationally expensive.
Failures of VBMC can return largely incorrect posteriors and values of the model evidence, which if taken at face value could lead to wrong conclusions. This failure mode is not unique to VBMC, but a common problem of all approximate inference techniques (e.g., MCMC or variational inference [2,53]). VBMC returns uncertainty on its estimate and comes with a set of diagnostic functions which can help identify issues. Still, we recommend the user to follow standard good practices for validation of results, such as posterior predictive checks, or comparing results from different runs.
Finally, in terms of ethical aspects, our method - like any general, black-box inference technique - will reflect (or amplify) the explicit and implicit biases present in the models and in the data, especially with insufficient data [54]. Thus, we encourage researchers in potentially sensitive domains to explicitly think about ethical issues and consequences of the models and data they are using.

## Acknowledgments and Disclosure of Funding

We thank Ian Krajbich for sharing data for the aDDM model; Robbe Goris for sharing data and code for the neuronal model; Marko Järvenpää and Alexandra Gessner for useful discussions about their respective work; Nisheet Patel for helpful comments on an earlier version of this manuscript; and the anonymous reviewers for constructive remarks. This work has utilized the NYU IT High Performance Computing resources and services. This work was partially supported by the Academy of Finland Flagship programme: Finnish Center for Artificial Intelligence (FCAI).

## References

[1] MacKay, D. J. (2003) Information theory, inference and learning algorithms. (Cambridge University Press).
[2] Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013) Bayesian Data Analysis (3rd edition). (CRC Press).
[3] Wood, S. N. (2010) Statistical inference for noisy nonlinear ecological dynamic systems. Nature 466, $1102-1104$.
[4] Price, L. F., Drovandi, C. C., Lee, A., & Nott, D. J. (2018) Bayesian synthetic likelihood. Journal of Computational and Graphical Statistics 27, 1-11.
[5] Acerbi, L. (2018) Variational Bayesian Monte Carlo. Advances in Neural Information Processing Systems 31, 8222-8232.
[6] Acerbi, L. (2019) An exploration of acquisition and mean functions in Variational Bayesian Monte Carlo. Proceedings of The 1st Symposium on Advances in Approximate Bayesian Inference (PMLR) 96, 1-10.
[7] Rasmussen, C. & Williams, C. K. I. (2006) Gaussian Processes for Machine Learning. (MIT Press).
[8] O’Hagan, A. (1991) Bayes-Hermite quadrature. Journal of Statistical Planning and Inference 29, 245-260.
[9] Ghahramani, Z. & Rasmussen, C. E. (2002) Bayesian Monte Carlo. Advances in Neural Information Processing Systems 15, 505-512.
[10] Järvenpää, M., Gutmann, M. U., Vehtari, A., Marttinen, P., et al. (2020) Parallel Gaussian process surrogate Bayesian inference with noisy likelihood evaluations. Bayesian Analysis.
[11] Gessner, A., Gonzalez, J., & Mahsereci, M. (2019) Active multi-information source Bayesian quadrature. Proceedings of the Thirty-Fifth Conference on Uncertainty in Artificial Intelligence (UAI 2019) p. 245.

---

#### Page 11

[12] Järvenpää, M., Gutmann, M. U., Vehtari, A., Marttinen, P., et al. (2018) Gaussian process modelling in approximate Bayesian computation to estimate horizontal gene transfer in bacteria. The Annals of Applied Statistics 12, 2228-2251.
[13] Järvenpää, M., Gutmann, M. U., Pleska, A., Vehtari, A., Marttinen, P., et al. (2019) Efficient acquisition rules for model-based approximate Bayesian computation. Bayesian Analysis 14, 595-622.
[14] Rasmussen, C. E. (2003) Gaussian processes to speed up hybrid Monte Carlo for expensive Bayesian integrals. Bayesian Statistics 7, 651-659.
[15] Kandasamy, K., Schneider, J., & Póczos, B. (2015) Bayesian active learning for posterior estimation. Twenty-Fourth International Joint Conference on Artificial Intelligence.
[16] Wang, H. & Li, J. (2018) Adaptive Gaussian process approximation for Bayesian inference with expensive likelihood functions. Neural Computation pp. 1-23.
[17] Osborne, M., Duvenaud, D. K., Garnett, R., Rasmussen, C. E., Roberts, S. J., & Ghahramani, Z. (2012) Active learning of model evidence using Bayesian quadrature. Advances in Neural Information Processing Systems 25, 46-54.
[18] Gunter, T., Osborne, M. A., Garnett, R., Hennig, P., & Roberts, S. J. (2014) Sampling for inference in probabilistic models with fast Bayesian quadrature. Advances in Neural Information Processing Systems 27, 2789-2797.
[19] Briol, F.-X., Oates, C., Girolami, M., & Osborne, M. A. (2015) Frank-Wolfe Bayesian quadrature: Probabilistic integration with theoretical guarantees. Advances in Neural Information Processing Systems 28, 1162-1170.
[20] Chai, H., Ton, J.-F., Garnett, R., & Osborne, M. A. (2019) Automated model selection with Bayesian quadrature. Proceedings of the 36th International Conference on Machine Learning 97, 931-940.
[21] Jones, D. R., Schonlau, M., & Welch, W. J. (1998) Efficient global optimization of expensive black-box functions. Journal of Global Optimization 13, 455-492.
[22] Brochu, E., Cora, V. M., & De Freitas, N. (2010) A tutorial on Bayesian optimization of expensive cost functions, with application to active user modeling and hierarchical reinforcement learning. arXiv preprint arXiv:1012.2599.
[23] Snoek, J., Larochelle, H., & Adams, R. P. (2012) Practical Bayesian optimization of machine learning algorithms. Advances in Neural Information Processing Systems 25, 2951-2959.
[24] Picheny, V., Ginsbourger, D., Richet, Y., & Caplin, G. (2013) Quantile-based optimization of noisy computer experiments with tunable precision. Technometrics 55, 2-13.
[25] Gutmann, M. U. & Corander, J. (2016) Bayesian optimization for likelihood-free inference of simulator-based statistical models. The Journal of Machine Learning Research 17, 4256-4302.
[26] Acerbi, L. & Ma, W. J. (2017) Practical Bayesian optimization for model fitting with Bayesian adaptive direct search. Advances in Neural Information Processing Systems 30, 1834-1844.
[27] Letham, B., Karrer, B., Ottoni, G., Bakshy, E., et al. (2019) Constrained Bayesian optimization with noisy experiments. Bayesian Analysis 14, 495-519.
[28] Papamakarios, G. & Murray, I. (2016) Fast $\varepsilon$-free inference of simulation models with Bayesian conditional density estimation. Advances in Neural Information Processing Systems 29, 1028-1036.
[29] Lueckmann, J.-M., Goncalves, P. J., Bassetto, G., Öcal, K., Nonnenmacher, M., & Macke, J. H. (2017) Flexible statistical inference for mechanistic models of neural dynamics. Advances in Neural Information Processing Systems 30, 1289-1299.
[30] Greenberg, D. S., Nonnenmacher, M., & Macke, J. H. (2019) Automatic posterior transformation for likelihood-free inference. International Conference on Machine Learning pp. 2404-2414.
[31] Gonçalves, P. J., Lueckmann, J.-M., Deistler, M., Nonnenmacher, M., Öcal, K., Bassetto, G., Chintaluri, C., Podlaski, W. F., Haddad, S. A., Vogels, T. P., et al. (2019) Training deep neural density estimators to identify mechanistic models of neural dynamics. bioRxiv p. 838383.
[32] Gramacy, R. B. & Lee, H. K. (2012) Cases for the nugget in modeling computer experiments. Statistics and Computing 22, 713-722.
[33] Neal, R. M. (2003) Slice sampling. Annals of Statistics 31, 705-741.
[34] Kingma, D. P. & Welling, M. (2013) Auto-encoding variational Bayes. Proceedings of the 2nd International Conference on Learning Representations.
[35] Miller, A. C., Foti, N., & Adams, R. P. (2017) Variational boosting: Iteratively refining posterior approximations. Proceedings of the 34th International Conference on Machine Learning 70, 2420-2429.
[36] Kingma, D. P. & Ba, J. (2014) Adam: A method for stochastic optimization. Proceedings of the 3rd International Conference on Learning Representations.

---

#### Page 12

[37] Kanagawa, M. & Hennig, P. (2019) Convergence guarantees for adaptive Bayesian quadrature methods. Advances in Neural Information Processing Systems 32, 6234-6245.
[38] Carpenter, B., Gelman, A., Hoffman, M., Lee, D., Goodrich, B., Betancourt, M., Brubaker, M. A., Guo, J., Li, P., & Riddell, A. (2016) Stan: A probabilistic programming language. Journal of Statistical Software 20.
[39] Ankenman, B., Nelson, B. L., & Staum, J. (2010) Stochastic kriging for simulation metamodeling. Operations Research 58, 371-382.
[40] Haldane, J. (1945) On a method of estimating frequencies. Biometrika 33, 222-225.
[41] van Opheusden, B., Acerbi, L., & Ma, W. J. (2020) Unbiased and efficient log-likelihood estimation with inverse binomial sampling. arXiv preprint arXiv:2001.03985.
[42] Lyu, X., Binois, M., & Ludkovski, M. (2018) Evaluating Gaussian process metamodels and sequential designs for noisy level set estimation. arXiv preprint arXiv:1807.06712.
[43] Krajbich, I., Armel, C., & Rangel, A. (2010) Visual fixations and the computation and comparison of value in simple choice. Nature Neuroscience 13, 1292.
[44] Jazayeri, M. & Shadlen, M. N. (2010) Temporal context calibrates interval timing. Nature Neuroscience 13, 1020-1026.
[45] Acerbi, L., Wolpert, D. M., & Vijayakumar, S. (2012) Internal representations of temporal statistics and feedback calibrate motor-sensory interval timing. PLoS Computational Biology 8, e1002771.
[46] Körding, K. P., Beierholm, U., Ma, W. J., Quartz, S., Tenenbaum, J. B., & Shams, L. (2007) Causal inference in multisensory perception. PLoS One 2, e943.
[47] Acerbi, L., Dokka, K., Angelaki, D. E., & Ma, W. J. (2018) Bayesian comparison of explicit and implicit causal inference strategies in multisensory heading perception. PLoS Computational Biology 14, e1006110.
[48] Goris, R. L., Simoncelli, E. P., & Movshon, J. A. (2015) Origin and function of tuning diversity in macaque visual cortex. Neuron 88, 819-831.
[49] Akrami, A., Kopec, C. D., Diamond, M. E., & Brody, C. D. (2018) Posterior parietal cortex represents sensory history and mediates its effects on behaviour. Nature 554, 368-372.
[50] Roy, N. A., Bak, J. H., Akrami, A., Brody, C., & Pillow, J. W. (2018) Efficient inference for time-varying behavior during learning. Advances in Neural Information Processing Systems 31, 5695-5705.
[51] Kass, R. E. & Raftery, A. E. (1995) Bayes factors. Journal of the American Statistical Association 90, $773-795$.
[52] Strubell, E., Ganesh, A., & McCallum, A. (2019) Energy and policy considerations for deep learning in NLP. Annual Meeting of the Association for Computational Linguistics.
[53] Yao, Y., Vehtari, A., Simpson, D., & Gelman, A. (2018) Yes, but did it work?: Evaluating variational inference. Proceedings of the 35th International Conference on Machine Learning 80, 5581-5590.
[54] Chen, I., Johansson, F. D., & Sontag, D. (2018) Why is my classifier discriminatory? Advances in Neural Information Processing Systems 31, 3539-3550.
[55] Jordan, M. I., Ghahramani, Z., Jaakkola, T. S., & Saul, L. K. (1999) An introduction to variational methods for graphical models. Machine Learning 37, 183-233.
[56] Bishop, C. M. (2006) Pattern Recognition and Machine Learning. (Springer).
[57] Knuth, D. E. (1992) Two notes on notation. The American Mathematical Monthly 99, 403-422.
[58] Carpenter, B., Gelman, A., Hoffman, M. D., Lee, D., Goodrich, B., Betancourt, M., Brubaker, M., Guo, J., Li, P., & Riddell, A. (2017) Stan: A probabilistic programming language. Journal of Statistical Software 76.
[59] Haario, H., Laine, M., Mira, A., & Saksman, E. (2006) Dram: Efficient adaptive MCMC. Statistics and Computing 16, 339-354.
[60] Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017) Variational inference: A review for statisticians. Journal of the American Statistical Association 112, 859-877.
[61] Robert, C. P., Cornuet, J.-M., Marin, J.-M., & Pillai, N. S. (2011) Lack of confidence in approximate Bayesian computation model choice. Proceedings of the National Academy of Sciences 108, 15112-15117.