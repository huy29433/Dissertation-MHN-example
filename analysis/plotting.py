import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
import numpy as np

mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.family"] = "STIXGeneral"
mpl.rcParams["font.size"] = 9


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
