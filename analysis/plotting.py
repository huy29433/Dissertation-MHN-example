from analysis.utils import plotting as mcmc_plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
import numpy as np
from typing import Literal, Optional

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
        events: Optional[list] = None):
    """Plot a theta matrix

    Args:
        theta (np.ndarray):  Theta matrix
        ax (Axes, optional): matplotlib Axes object. Defaults to None.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(1.5, 1.1))

    n_events = log_theta.shape[1]
    border_len = 0.1
    br_ir_sep = 0.2
    linewidth = 0.3

    max_br = np.diag(log_theta).max()
    min_br = np.diag(log_theta).min()
    _log_theta = log_theta.copy()
    _log_theta[np.diag_indices(n_events)] = 0
    m_ir = np.abs(_log_theta).max()

    if events is None:
        events = np.arange(n_events).tolist()

    if ax is None:
        _, ax = plt.subplots()

    for i in range(n_events):
        for j in range(n_events):
            if i == j:
                ax.add_patch(mpl.patches.Rectangle(
                    (0,
                        j + j * border_len),
                    1, 1, linewidth=linewidth,
                    edgecolor="black",
                    facecolor=mcmc_plotting.OI_Greens(
                        (log_theta[i, j] - min_br)/(max_br - min_br))
                ))
            else:
                ax.add_patch(mpl.patches.Rectangle(
                    (j+1 + br_ir_sep + (j+1) * border_len,
                        i + i * border_len),
                    1, 1, linewidth=linewidth,
                    edgecolor="black",
                    facecolor=mcmc_plotting.OI_RdBu(
                        (log_theta[i, j]/m_ir + 1) / 2))
                )

    if model in ["omhn", "metmhn"]:

        for j in range(n_events):
            ax.add_patch(mpl.patches.Rectangle(
                (j+1 + br_ir_sep + (j+1) * border_len,
                    n_events + n_events * border_len),
                1, 1, linewidth=linewidth,
                edgecolor="black",
                facecolor=mcmc_plotting.OI_RdBu(
                    (log_theta[-1, j]/m_ir + 1) / 2),
            )
            )

    for i in range(n_events):
        for j in range(n_events):
            if i != j:
                if log_theta[i, j] == 0:
                    continue
                ax.text(
                    (j+1 + br_ir_sep + (j+1) * border_len)
                    + 0.5,
                    i + i * border_len + 0.5,
                    f"{log_theta[i, j]:.1f}",
                    ha="center", va="center",
                    fontsize=6,
                    color="white" if abs(
                        log_theta[i, j]) > m_ir / 2 else "black"
                )
            else:
                ax.text(
                    0 + 0.5,
                    j + j * border_len + 0.5,
                    f"{log_theta[i, j]:.1f}",
                    ha="center", va="center",
                    fontsize=6,
                    color="white"
                    if abs(log_theta[i, j] - min_br) > (max_br - min_br) / 2
                    else "black"
                )
    if model in ["omhn", "metmhn"]:
        for j in range(n_events):
            if log_theta[-1, j] == 0:
                continue
            ax.text(
                (j+1 + br_ir_sep + (j+1) * border_len)
                + 0.5,
                n_events + n_events * border_len + 0.5,
                f"{log_theta[-1, j]:.1f}",
                ha="center", va="center",
                fontsize=6,
                color="white" if abs(log_theta[-1, j]) > m_ir / 2 else "black",
            )

    ax.set_xlim(0 - border_len, (border_len + 1) * (n_events + 1) + br_ir_sep)
    ax.set_ylim((n_events + 1) * (border_len + 1) + border_len, 0 - border_len)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks(
        [0.5] + (np.arange(
            1.5,
            n_events + 1 + n_events * border_len,
            1 + border_len) + br_ir_sep).tolist())
    ax.set_yticks(np.arange(0.5, n_events + (1 if model == "omhn" else 0) +
                            n_events * border_len, 1 + border_len))
    ax.set_xticklabels([""] + events, rotation=90)
    ax.set_yticklabels(events + ["Observation"] if model == "omhn" else events)
    ax.tick_params(length=0, pad=2)
    ax.set_aspect("equal")
