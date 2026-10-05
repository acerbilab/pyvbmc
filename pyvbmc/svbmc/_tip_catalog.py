"""Optional guidance for users combining completed VBMC posteriors."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Tip:
    """A guidance message, its links and when it is relevant."""

    id: str
    text: str
    noisy_only: bool = False
    urls: tuple[str, ...] = ()


TIPS = (
    Tip(
        "noisy_elbo",
        "On noisy targets, the raw stacked ELBO can be optimistic, with "
        "optimism growing as more runs are stacked. Where it is defined, "
        "the headline elbo is a two-level shrinkage estimate, which targets "
        "the optimism that stacking adds and can also remove part of the "
        "runs' own bias; some bias can remain. Use elbo for model "
        "comparison and report elbo_sd alongside it. The SD describes "
        "uncertainty of the raw evaluation; it does not include selection "
        "bias or uncertainty of the shrinkage. The correction affects the "
        "reported evidence estimate only; the stacked posterior is "
        "unchanged.",
        noisy_only=True,
    ),
    Tip(
        "run_count",
        "Stacking about ten well-converged VBMC runs often captures most "
        "of the improvement in posterior quality seen in the S-VBMC paper.",
        urls=(
            "https://acerbilab.github.io/pyvbmc/_examples/"
            "pyvbmc_example_7_stacking.html",
        ),
    ),
    Tip(
        "starting_points",
        "Start the individual VBMC runs from different points to help "
        "them explore different regions of the posterior.",
        urls=(
            "https://acerbilab.github.io/pyvbmc/api/classes/svbmc.html"
            "#svbmc-helpers",
        ),
    ),
)
