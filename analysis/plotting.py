from analysis.utils import plotting as mcmc_plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
import numpy as np
from typing import Literal, Optional
import warnings

import sys
sys.path.append("../MCMC-sampling-for-MHN")


mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.family"] = "STIXGeneral"
mpl.rcParams["font.size"] = 8


def oncoplot(data: pd.DataFrame, ax: Axes = None) -> None:
    """Create an oncoplot from a dataframe

    Args:
        data (pd.DataFrame): Dataframe of binary cancer data.
        ax (Axes, optional): matplotlib Axes object. Defaults to None.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 4))

    ax.imshow(data.sort_values(data.columns.to_list()).T,
              cmap="Greys", interpolation="none")
    ax.set_yticks(np.arange(data.shape[1]), data.columns)
    ax.set_aspect(data.shape[0] / data.shape[1])


def plot_theta(
        log_theta: np.ndarray,
        model: Literal["omhn", "cmhn", "metmhn"] = "omhn",
        ax: Axes = None,
        events: Optional[list] = None,
        abs_max_ir: Optional[float] = None,
        min_br: Optional[float] = None,
        max_br: Optional[float] = None,
        border_len: float = 0.1,
        br_ir_sep: float = 0.2,
        linewidth: float = 0.3,
        threshold: int = 0.05,
):
    """Plot a theta matrix

    Args:
        theta (np.ndarray):  Theta matrix
        ax (Axes, optional): matplotlib Axes object. Defaults to None.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(1.5, 1.1))

    n_events = log_theta.shape[1] - (1 if model == "metmhn" else 0)

    if model == "metmhn":
        log_theta = np.roll(log_theta, shift=-2, axis=0)

    _max_br = np.diag(log_theta).max()
    _min_br = np.diag(log_theta).min()
    _log_theta = log_theta.copy()
    _log_theta[np.diag_indices(n_events)] = 0
    _abs_max_ir = np.abs(_log_theta).max()

    if abs_max_ir is None:
        abs_max_ir = _abs_max_ir
    elif abs_max_ir < _abs_max_ir:
        warnings.warn(
            f"Provided abs_max_ir {abs_max_ir} is smaller than "
            f"calculated maximum {_abs_max_ir}. "
        )
    if min_br is None:
        min_br = _min_br
    elif min_br > _min_br:
        warnings.warn(
            f"Provided min_br {min_br} is larger than "
            f"calculated minimum {_min_br}. "
        )
    if max_br is None:
        max_br = _max_br
    elif max_br < _max_br:
        warnings.warn(
            f"Provided max_br {max_br} is smaller than "
            f"calculated maximum {_max_br}. "
        )

    if events is None:
        events = np.arange(n_events).tolist()

    if ax is None:
        _, ax = plt.subplots()

    xs = [0.5] + (np.arange(
        1.5,
        n_events + 1 + n_events * border_len,
        1 + border_len) + br_ir_sep).tolist()
    ys = np.arange(0.5, n_events + n_events *
                   border_len, 1 + border_len).tolist()
    x_labels = [""] + events[:n_events]
    y_labels = events[:n_events]
    if model == "metmhn":
        xs.append(0.5 + br_ir_sep + (1 + border_len) * n_events + 1)
        x_labels.append("Seeding")
        ys.append(0.5 + (1 + border_len) * n_events)
        y_labels.append("Seeding")
        ys += [0.5 + (1 + border_len) * (n_events + i) + 1 +
               br_ir_sep for i in (0, 1)]
        y_labels += ["Observation " + entity for entity in ("PT", "Met")]
    if model == "omhn":
        ys.append(0.5 + (1 + border_len) * (n_events - 1) + 1 + br_ir_sep)
        y_labels.append("Observation")

    for i in range(log_theta.shape[0]):
        for j in range(log_theta.shape[1]):
            if i == j:
                ax.add_patch(mpl.patches.Rectangle(
                    (0,
                        ys[i]-0.5),
                    1, 1, linewidth=linewidth,
                    edgecolor="black",
                    facecolor=mcmc_plotting.OI_Greens(
                        (log_theta[i, i] - min_br)/(max_br - min_br))
                ))
                ax.text(
                    0.5,
                    ys[i],
                    f"{log_theta[i, i]:.1f}",
                    ha="center", va="center",
                    # fontsize=6,
                    color="white"
                    if abs(log_theta[i, j] - min_br) > (max_br - min_br) / 2
                    else "black"
                )
            else:
                ax.add_patch(mpl.patches.Rectangle(
                    (xs[j + 1] - 0.5, ys[i] - 0.5),
                    1, 1, linewidth=linewidth,
                    edgecolor="black",
                    facecolor=mcmc_plotting.OI_RdBu(
                        (log_theta[i, j]/abs_max_ir + 1) / 2))
                )
                if np.abs(log_theta[i, j]) <= threshold:
                    continue
                ax.text(
                    xs[j+1], ys[i],
                    f"{log_theta[i, j]:.1f}",
                    ha="center", va="center",
                    # fontsize=6,
                    color="white" if abs(
                        log_theta[i, j]) > abs_max_ir / 2 else "black"
                )

    ax.set_xlim(0 - border_len, xs[-1] + 0.5 + border_len)
    ax.set_ylim(ys[-1] + 0.5 + border_len, 0 - border_len)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.set_xticks(xs)
    ax.set_yticks(ys)
    ax.set_xticklabels(x_labels, rotation=90)
    ax.set_yticklabels(y_labels)
    ax.tick_params(length=0, pad=2)
    ax.set_aspect("equal")
