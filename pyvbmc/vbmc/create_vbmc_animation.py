import io
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np

from .vbmc import VBMC


def create_vbmc_animation(
    vbmc: VBMC,
    path: str,
    as_frames: bool = False,
    suptitle="full",
    **kwargs: dict,
):
    """
    Create and save a gif animation of a VBMC optimization run.

    Parameters
    ----------
    vbmc : VBMC
        The optimized VBMC.
    path : str
        The path where the gif should be saved to.
    as_frames: bool, optional
        If `True`, saves the animation as individual frames. The filename will
        be appended with the frame number. Default `False`.
    suptitle: str, optional
        What kind of supertitle to print. "full" (the default) means include
        the logging action. "iteration" means print only the iteration. "none"
        means do not supertitle the figures.
    **kwargs : dict, optional
        Keyword arguments, passed to ``vp.plot()``.

    Raises
    ------
    ValueError
        If the ``suptitle`` option is not one of the three supported values.
    """
    path = Path(path)
    # plot last figure to figure out x_lim and y_lim later
    last_figure_axes = np.array(
        vbmc.vp.plot(gp=vbmc.gp, **kwargs).axes
    ).reshape((vbmc.vp.D, vbmc.vp.D))

    images = []
    gp = None
    for i in range(0, len(vbmc.iteration_history["iter"]) + 1):

        if i >= len(vbmc.iteration_history["iter"]):
            vp = vbmc.vp
        else:
            vp = vbmc.iteration_history["vp"][i]

        if 0 < i < len(vbmc.iteration_history["vp"]) - 2:
            previous_gp = vbmc.iteration_history["gp"][i - 1]
            gp = vbmc.iteration_history["gp"][i]
            # find points that are new in this iteration
            # (hacky cause numpy only has 1D set diff)
            highlight_data = np.array(
                [
                    i
                    for i, x in enumerate(gp.X)
                    if tuple(x) not in set(map(tuple, previous_gp.X))
                ]
            )
        else:
            highlight_data = None

        fig = vp.plot(
            highlight_data=highlight_data, plot_data=True, gp=gp, **kwargs
        )

        # set title of plot accordingly
        if suptitle in ("iteration", "full"):
            fig.suptitle("PyVBMC iteration {}".format(i))
        elif (
            suptitle == "full"
            and i < len(vbmc.iteration_history["iter"])
            and len(vbmc.iteration_history["logging_action"][i]) > 0
        ):
            fig.suptitle(
                "PyVBMC iteration {} ({})".format(
                    i, "".join(vbmc.iteration_history["logging_action"][i])
                )
            )
        elif suptitle != "none":
            raise ValueError(f"Unsupported suptitle option {suptitle}.")

        if i == len(vbmc.iteration_history["iter"]):
            fig.suptitle("PyVBMC final ({} iterations)".format(i - 1))

        # make axis limits the same for all figures and subplots
        axes = np.array(fig.axes).reshape((vp.D, vp.D))
        for r in range(vp.D):
            for c in range(vp.D):
                axes[r, c].set_xlim(last_figure_axes[r, c].get_xlim())
                if r > c:
                    axes[r, c].set_ylim(last_figure_axes[r, c].get_ylim())

        plt.tight_layout()
        images.append(_fig_to_img(fig))

        # append final iteration multiple times to increase showing length
        if i == len(vbmc.iteration_history["iter"]):
            for _ in range(4):
                images.append(_fig_to_img(fig))

    if as_frames:
        stem = path.stem
        for (i, img) in enumerate(images):
            imageio.imsave(path.with_stem(f"{stem}-{i:03}"), img)
    else:
        imageio.mimsave(path, images, duration=0.5)

    return fig


def _fig_to_img(fig):
    """
    A private helper function to save a figure as an image array with correct
    dimensions.
    """
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format="raw", dpi=fig.dpi)
    io_buf.seek(0)
    img_arr = np.reshape(
        np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1),
    )
    io_buf.close()
    return img_arr
