"""Data-only catalog of optional runtime tips."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TipFrequency = Literal["normal", "low_frequency"]


@dataclass(frozen=True)
class Tip:
    """One runtime tip and its reader-facing links."""

    id: str
    text: str
    frequency: TipFrequency
    urls: tuple[str, ...] = ()


# For wording or link edits, retain an entry's ID. Use a new ID when replacing
# its topic. Add, remove, or recategorize entries here; the selector derives its
# catalog size and category membership from these records.
TIPS = (
    Tip(
        id="multiple_runs",
        text=(
            "Run VBMC 3–4 times with different starting points. Compare the "
            "posterior plots across runs; large differences can indicate that "
            "a run missed part of the posterior."
        ),
        frequency="normal",
        urls=(
            "https://acerbilab.github.io/pyvbmc/_examples/"
            "pyvbmc_example_4_validation.html",
        ),
    ),
    Tip(
        id="evidence_uncertainty",
        text=(
            "When comparing models, report results['elbo'] together with "
            "results['elbo_sd']. The ELBO is a lower bound on the true log "
            "model evidence; its SD measures uncertainty in estimating that "
            "bound. A small SD does not guarantee that the bound is close to "
            "the true evidence!"
        ),
        frequency="normal",
        urls=("https://acerbilab.github.io/pyvbmc/quickstart.html",),
    ),
    Tip(
        id="plausible_bounds",
        text=(
            "Use PLB and PUB to guide PyVBMC's initial search toward likely "
            "parameter values. If you have no better information, use each "
            "prior's 16th and 84th percentiles: roughly mean minus and plus "
            "one SD for a Gaussian prior. These guide the search without "
            "restricting the posterior."
        ),
        frequency="normal",
        urls=(
            "https://acerbilab.github.io/pyvbmc/faq.html"
            "#faq-how-do-i-choose-plb-and-pub",
        ),
    ),
    Tip(
        id="pybads",
        text=(
            "A maximum-likelihood (MLE) or maximum a posteriori (MAP) estimate "
            "can provide a good starting point for PyVBMC. You can use PyBADS "
            "to find one, then pass it as x0."
        ),
        frequency="low_frequency",
        urls=("https://acerbilab.github.io/pybads/",),
    ),
    Tip(
        id="posterior_plot",
        text=(
            "Use vp.plot() to inspect parameter uncertainty and trade-offs "
            "after fitting. Broad marginal distributions indicate uncertainty; "
            "tilted or curved joint contours can reveal parameter trade-offs."
        ),
        frequency="normal",
        urls=(
            "https://acerbilab.github.io/pyvbmc/api/classes/"
            "variational_posterior.html",
        ),
    ),
    Tip(
        id="save_resume",
        text=(
            "Save your run with vbmc.save('fit.pkl') so you can continue later "
            "without starting over. If it needs more evaluations, load it with "
            "VBMC.load('fit.pkl', new_options={'max_fun_evals': N}), then call "
            "optimize() on the loaded object. Set N to the larger total "
            "evaluation budget."
        ),
        frequency="normal",
        urls=("https://acerbilab.github.io/pyvbmc/api/classes/vbmc.html",),
    ),
    Tip(
        id="noisy_target",
        text=(
            "PyVBMC can account for noise in simulated log-likelihood estimates "
            "if you supply its size. Set options={'specify_target_noise': True} "
            "and return (log_density, noise_sd) from your target, where noise_sd "
            "estimates the standard deviation of the noise in that log-density "
            "evaluation."
        ),
        frequency="normal",
        urls=(
            "https://acerbilab.github.io/pyvbmc/_examples/"
            "pyvbmc_example_6_noisy_likelihoods.html",
        ),
    ),
    Tip(
        id="svbmc",
        text=(
            "Using S-VBMC, you can improve your posterior estimates by "
            "combining multiple independent PyVBMC runs on the same model and "
            "data. S-VBMC re-optimizes mixture weights to form a combined "
            "posterior without new model evaluations. See the tutorial to "
            "combine your runs and draw samples."
        ),
        frequency="low_frequency",
        urls=("https://github.com/acerbilab/svbmc#how-to-use-s-vbmc",),
    ),
)
