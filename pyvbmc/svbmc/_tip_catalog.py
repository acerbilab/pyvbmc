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
        "With noisy targets, the raw elbo values printed below during "
        "optimization are optimistic, more so the more runs are stacked. "
        "stacked.elbo corrects much of this bias with two-level shrinkage, "
        "a statistical debiasing technique. The last line of the output "
        "reports the amount of the correction.",
        noisy_only=True,
        urls=(
            "https://acerbilab.org/pyvbmc/api/classes/svbmc.html"
            "#svbmc-elbo-reporting",
        ),
    ),
    Tip(
        "run_count",
        "Stacking 3–5 well-converged VBMC runs is usually enough. On hard "
        "problems, such as multimodal posteriors, stacking more runs can "
        "still improve the posterior.",
        urls=(
            "https://acerbilab.org/pyvbmc/_examples/"
            "pyvbmc_example_7_stacking.html",
        ),
    ),
)
