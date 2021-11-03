from .vbmc import VBMC
import numpy as np


def create_vbmc_plot(vbmc: VBMC, path: str):
    figs = list()

    for i in range(0, len(vbmc.iteration_history["iter"]) + 1):

        if i >= len(vbmc.iteration_history["iter"]):
            vp = vbmc.vp
        else:
            vp = vbmc.iteration_history["vp"][i]

        if i > 0 and i < len(vbmc.iteration_history["vp"]) - 2:
            previous_gp = vbmc.iteration_history["vp"][i - 1].gp
            # find points that are new in this iteration
            # (hacky cause numpy only has 1D set diff)
            highlight_data = np.array(
                [
                    i
                    for i, x in enumerate(vp.gp.X)
                    if tuple(x) not in set(map(tuple, previous_gp.X))
                ]
            )
        else:
            highlight_data = None

        fig = vp.plot(highlight_data=highlight_data, plot_data=True)

        fig.suptitle("VBMC (iteration {})".format(i))
        if i == 18:
            fig.suptitle("VBMC (iteration {})".format(i - 1))

        figs.append(fig)

    # save as gif
