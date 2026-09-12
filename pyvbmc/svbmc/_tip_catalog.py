"""Optional guidance for users combining completed VBMC posteriors."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Tip:
    """A guidance message and the condition under which it is relevant."""

    id: str
    text: str
    noisy_only: bool = False


TIPS = (
    Tip(
        "noisy_elbo",
        "On noisy targets, the raw stacked ELBO can be optimistic, with "
        "optimism growing as more runs are stacked. Use the capped elbo "
        "for model comparison and report elbo_sd alongside it. The SD "
        "describes uncertainty of the raw evaluation; it does not include "
        "selection bias or uncertainty of the cap. The cap affects the "
        "reported evidence estimate only; the stacked posterior is unchanged.",
        noisy_only=True,
    ),
    Tip(
        "run_count",
        "Stacking about ten well-converged VBMC runs often captures most "
        "of the improvement in posterior quality seen in the S-VBMC paper.",
    ),
    Tip(
        "starting_points",
        "Start the individual VBMC runs from different points to help "
        "them explore different regions of the posterior.",
    ),
)
