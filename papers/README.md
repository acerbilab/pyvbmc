# Papers

LLM-friendly Markdown versions of the papers that define the VBMC algorithm,
for use by maintainers and coding agents working on this repository. Copied
from the lab publication archive
[acerbilab/pubs-llms](https://github.com/acerbilab/pubs-llms); regenerate from
there rather than editing here. Each paper is split as in the archive:
`_main` (core text), `_appendix` (supplementary material), `_backmatter`
(references and acknowledgments, rarely worth feeding to a model when doing
technical work). The combined `_full` variants are omitted as redundant.

Text, equations, tables and captions were verified against the original
sources (2026-09-02). Figure descriptions are AI-generated and unverified.

| Key | Paper | What it defines |
|---|---|---|
| `acerbi2018variational` | Acerbi L (2018). *Variational Bayesian Monte Carlo.* NeurIPS 2018. | The original algorithm: GP surrogate with Bayesian quadrature, mixture-of-Gaussians variational posterior, ELBO with analytic expected log joint, entropy estimators, active sampling, warmup, and the K-adaptation and pruning rules. |
| `acerbi2019exploration` | Acerbi L (2019). *An Exploration of Acquisition and Mean Functions in Variational Bayesian Monte Carlo.* AABI 2019 (PMLR 96). | Comparison of acquisition functions and GP mean functions; motivates the negative-quadratic mean used by default. |
| `acerbi2020variational` | Acerbi L (2020). *Variational Bayesian Monte Carlo with Noisy Likelihoods.* NeurIPS 2020. | Noisy VBMC: the VIQR and IMIQR acquisition functions, active importance sampling, variational whitening, and the noise-handling changes to GP training and warmup. |
| `silvestrin2025stacking` | Silvestrin F, Li C & Acerbi L (2025). *Stacking Variational Bayesian Monte Carlo.* TMLR. arXiv:2504.05004. | Stacking of multiple VBMC runs into a single posterior; the most recent algorithmic extension. |
| `huggins2023pyvbmc` | Huggins B, Li C, Tobaben M, Aarnos MJ & Acerbi L (2023). *PyVBMC: Efficient Bayesian inference in Python.* Journal of Open Source Software 8(86):5428. | The software paper for this package: scope, features, and the differences from the MATLAB implementation. No appendix. |

