from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_frequency_columns(columns):
    """
    Convert frequency column labels such as '0Hz', '20Hz', and '100Hz'
    into numeric values.

    Parameters
    ----------
    columns : iterable
        DataFrame column labels.

    Returns
    -------
    numpy.ndarray
        Numeric stimulation frequencies.
    """
    return np.asarray(
        [float(str(column).replace("Hz", "")) for column in columns]
    )


def style_candidate_stimulation_axis(
    ax,
    xlim=(0, 200),
    ylim=(0, 100),
    xticks=None,
    yticks=None,
    spine_offset=5,
    spine_width=1.2,
):
    """
    Apply common axis styling for candidate stimulation plots.
    """
    if xticks is None:
        xticks = np.arange(xlim[0], xlim[1] + 1, 40)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xticks(xticks)

    if yticks is not None:
        ax.set_yticks(yticks)

    ax.set_xlabel("Activation Firing Rate (Hz)")
    ax.set_ylabel("MN9 Firing Rate (Hz)")

    ax.spines["left"].set_position(("outward", spine_offset))
    ax.spines["bottom"].set_position(("outward", spine_offset))

    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_linewidth(spine_width)

    ax.tick_params(
        axis="both",
        width=spine_width,
    )


def plot_candidate_stimulation(
    ax,
    mean_df,
    sem_df,
    colors,
    labels=None,
    x_values=None,
    alphas=None,
    capsize=1,
    error_color="black",
    line_width=1.5,
    error_line_width=0.5,
    cap_thickness=0.5,
):
    """
    Plot MN9 firing rate across stimulation frequencies.

    Each row of mean_df is plotted as one line.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to draw the plot.
    mean_df : pandas.DataFrame
        Mean firing rates. Rows are conditions and columns are frequencies.
    sem_df : pandas.DataFrame
        SEM values with the same shape and labels as mean_df.
    colors : sequence or dict
        Either a list of colors matching mean_df rows, or a mapping from
        row labels to colors.
    labels : sequence, optional
        Legend labels. Defaults to mean_df.index.
    x_values : array-like, optional
        Numeric x values. If omitted, frequencies are parsed from column names.
    alphas : sequence or dict, optional
        Line transparency. Defaults to 1 for all rows.
    """
    if not mean_df.index.equals(sem_df.index):
        raise ValueError("mean_df and sem_df must have identical row indices.")

    if not mean_df.columns.equals(sem_df.columns):
        raise ValueError("mean_df and sem_df must have identical columns.")

    if x_values is None:
        x_values = parse_frequency_columns(mean_df.columns)

    if labels is None:
        labels = list(mean_df.index)

    if isinstance(colors, dict):
        row_colors = [colors[row] for row in mean_df.index]
    else:
        row_colors = list(colors)

    if len(row_colors) != len(mean_df.index):
        raise ValueError(
            "The number of colors must match the number of DataFrame rows."
        )

    # alpha (default = 1)
    if alphas is None:
        row_alphas = [1.0] * len(mean_df.index)
    elif isinstance(alphas, dict):
        row_alphas = [alphas[row] for row in mean_df.index]
    else:
        row_alphas = list(alphas)

    if len(row_alphas) != len(mean_df.index):
        raise ValueError(
            "The number of alphas must match the number of DataFrame rows."
        )

    for row_name, label, color, alpha in zip(
        mean_df.index,
        labels,
        row_colors,
        row_alphas,
    ):
        y = mean_df.loc[row_name].to_numpy(dtype=float)
        yerr = sem_df.loc[row_name].to_numpy(dtype=float)

        ax.errorbar(
            x_values,
            y,
            yerr=yerr,
            color=color,
            alpha=alpha,
            ecolor=error_color,
            label=label,
            capsize=capsize,
            linewidth=line_width,
            elinewidth=error_line_width,
            capthick=cap_thickness,
        )

    return ax


def create_candidate_stimulation_plot(
    mean_df,
    sem_df,
    colors,
    alphas=None,
    labels=None,
    figsize=(2, 2),
    xlim=(0, 200),
    ylim=(0, 100),
    xticks=None,
    yticks=None,
    legend=True,
    legend_kwargs=None,
    spine_width=1.2,
):
    """
    Create a complete candidate stimulation figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    plot_candidate_stimulation(
        ax=ax,
        mean_df=mean_df,
        sem_df=sem_df,
        colors=colors,
        alphas=alphas,
        labels=labels,
    )

    style_candidate_stimulation_axis(
        ax=ax,
        xlim=xlim,
        ylim=ylim,
        xticks=xticks,
        yticks=yticks,
        spine_width=spine_width,
    )

    if legend:
        if legend_kwargs is None:
            legend_kwargs = {
                "loc": "upper left",
                "bbox_to_anchor": (1.05, 1),
                "frameon": False,
            }

        ax.legend(**legend_kwargs)

    return fig, ax

def save_figure(fig, output_path, dpi=300):
    """
    Save a matplotlib figure, creating the output directory if necessary.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_path,
        bbox_inches="tight",
        dpi=dpi,
    )