import numpy as np
from scipy.interpolate import RegularGridInterpolator
from matplotlib.ticker import MultipleLocator, NullFormatter, FixedLocator
### load data 
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.interpolate import interp1d

CONCENTRATIONS = np.array([10, 50, 100, 500])


def standard_error(values):
    """Calculate SEM across flies/trials."""
    values = np.asarray(values)
    return np.nanstd(values, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(values), axis=0))


def summarize_per_dataframe(df):
    """Return mean, SEM, and n for each sucrose concentration."""
    values = df.to_numpy(dtype=float)

    return {
        "mean": np.nanmean(values, axis=0),
        "sem": standard_error(values),
        "n": np.sum(~np.isnan(values), axis=0),
    }


def load_per_pair(excel_path, sugar_sheet="sugar", geosmin_sheet="sugar+geosmin"):
    """Load sugar and sugar+geosmin PER data from one Excel file."""
    excel_path = Path(excel_path)

    sugar_df = pd.read_excel(excel_path, sheet_name=sugar_sheet, index_col=False)
    geosmin_df = pd.read_excel(excel_path, sheet_name=geosmin_sheet, index_col=False)

    return {
        "sugar": summarize_per_dataframe(sugar_df),
        "sugar_geosmin": summarize_per_dataframe(geosmin_df),
        "raw": {
            "sugar": sugar_df,
            "sugar_geosmin": geosmin_df,
        },
    }



### functions 

def hill(c, V, K, n):
    return V * (c**n) / (K**n + c**n)


def sigmoid(rate, k, r0, h):
    return h / (1 + np.exp(-k * (rate - r0)))


def inv_hill(f, V, K, n):
    f = np.clip(f, 1e-6, V-1e-6)  
    return K * (f / (V - f))**(1/n)


def make_interpolator(rate_matrix, stim_rates=None):
    """Create a RegularGridInterpolator from a 2D stimulation matrix."""
    if stim_rates is None:
        stim_rates = np.arange(0, 201, 20)

    return RegularGridInterpolator(
        (stim_rates, stim_rates),
        rate_matrix,
        bounds_error=False,
        fill_value=None,   # extrapolation 허용
    )



### fiting functions 

def model_prediction_global(params, c,  cond,interp_sugar,interp_geosmin):
    V_s1, K_s1, n_s1, V_s2, K_s2, n_s2,k_act,r0,h = params
    c1 = np.array(c)[:,0]
    c2 = np.array(c)[:,1]
    
    f_c_s1 = V_s1 * (c1**n_s1) / (K_s1**n_s1 + c1**n_s1)
    f_c_s2 = V_s2 * (c2**n_s2) / (K_s2**n_s2 + c2**n_s2)

    # r_val 초기화
    r_val = np.zeros_like(cond,dtype=float)

    # 각 조건별 mask
    mask_suc = (cond == 0)
    mask_geo = (cond == 1)
    
    pts_all = np.column_stack((f_c_s1[mask_suc], f_c_s2[mask_suc]))
    r_val[mask_suc] = interp_sugar(pts_all).ravel()
    pts_all = np.column_stack((f_c_s1[mask_geo], f_c_s2[mask_geo]))
    r_val[mask_geo] = interp_geosmin(pts_all).ravel()
    

    per_pred = h / (1 + np.exp(-k_act * (r_val - r0)))

    return per_pred


def objective_global_wo_constraint(
    params,
    c_all_,
    cond_all,
    per_all,
    interp_sugar,
    interp_geosmin,
):
    pred = model_prediction_global(
        params,
        c_all_,
        cond_all,
        interp_sugar,
        interp_geosmin,
    )
    data_loss = np.sum((pred - per_all)**2)

    V_s1, K_s1, n_s1, V_s2, K_s2, n_s2,k_act,r0,h = params


    return data_loss 


### plot functions 

def add_firing_rate_axis(
    ax,
    label,
    hill_params,
    y_text,
    outward,
    major_ticks,
    minor_ticks,
    xlabel=False,
):
    V, K, n = hill_params

    secax = ax.secondary_xaxis("top")

    secax.set_xticks(inv_hill(major_ticks, V, K, n))
    secax.set_xticklabels(major_ticks)

    secax.set_xticks(inv_hill(minor_ticks, V, K, n), minor=True)
    secax.xaxis.set_minor_formatter(NullFormatter())

    secax.spines["top"].set_position(("outward", outward))
    secax.spines["top"].set_linewidth(.7)
    secax.tick_params(axis="both", width=.7)

    ax.text(
        -0.05,
        y_text,
        label,
        transform=ax.transAxes,
        ha="right",
        va="center",
    )

    if xlabel:
        secax.set_xlabel("Activation firing rate (Hz)")

    return secax

def plot_condition(
    ax,
    mode,
    interp_sugar,
    interp_geosmin,
    experiment,
    hill_params_first,
    hill_params_second,
    activation_params,
    labels,
    colors,
    sucrose_range=None,
    ylim=(0, 0.8),
    axis_tick_config=None
):
    """
    Plot predicted and experimental PER for a two-input stimulation model.

    Parameters
    ----------
    mode : {"both", "first", "second"}
        "both"   : stimulate both input channels
        "first"  : stimulate first input only, second input = 0
        "second" : stimulate second input only, first input = 0

    labels : tuple of str
        Names of the two input channels.
        Example:
            ("atGRN", "TPN1")
            ("L1+L2", "L3")
    """

    if sucrose_range is None:
        sucrose_range = np.logspace(0.9, np.log10(1000), 200)
    if axis_tick_config is None:
        axis_tick_config = {
            "both": {
                "first": {
                    "major_ticks": np.arange(40, 141, 40),
                    "minor_ticks": np.arange(0, 141, 4),
                },
                "second": {
                    "major_ticks": np.arange(40, 141, 10),
                    "minor_ticks": np.arange(0, 141, 1),
                },
            },
            "first": {
                "major_ticks": np.arange(40, 141, 40),
                "minor_ticks": np.arange(0, 141, 4),
            },
            "second": {
                "major_ticks": np.arange(20, 141, 10),
                "minor_ticks": np.arange(0, 141, 1),
            },
        }
    V1, K1, n1 = hill_params_first
    V2, K2, n2 = hill_params_second
    k_act, r0, h = activation_params

    if mode == "both":
        first_rate = hill(sucrose_range, V1, K1, n1)
        second_rate = hill(sucrose_range, V2, K2, n2)

    elif mode == "first":
        first_rate = hill(sucrose_range, V1, K1, n1)
        second_rate = np.zeros_like(sucrose_range)

    elif mode == "second":
        first_rate = np.zeros_like(sucrose_range)
        second_rate = hill(sucrose_range, V2, K2, n2)

    else:
        raise ValueError("mode must be one of: 'both', 'first', 'second'")

    first_rate = np.round(first_rate, 1)
    second_rate = np.round(second_rate, 1)

    points = np.column_stack([first_rate, second_rate])

    mn9_sugar = interp_sugar(points).ravel()
    mn9_geo = interp_geosmin(points).ravel()

    per_sugar = sigmoid(mn9_sugar, k_act, r0, h)
    per_geo = sigmoid(mn9_geo, k_act, r0, h)

    ax.semilogx(sucrose_range, per_sugar, color=colors[0])
    ax.semilogx(sucrose_range, per_geo, color=colors[1])

    c_exp = np.array([10, 50, 100, 500])

    ax.scatter(
        c_exp,
        experiment["sugar"]["mean"],
        s=20,
        color=colors[0],
        edgecolors="black",
        zorder=3,
        linewidth=.5
    )
    ax.scatter(
        c_exp,
        experiment["sugar_geosmin"]["mean"],
        s=20,
        color=colors[1],
        edgecolors="black",
        zorder=3,
        linewidth=.5
    )

    if mode == "both":
        add_firing_rate_axis(
            ax=ax,
            label=labels[1],
            hill_params=hill_params_second,
            y_text=1.125,
            outward=5,
            major_ticks=axis_tick_config["both"]["second"]["major_ticks"],
            minor_ticks=axis_tick_config["both"]["second"]["minor_ticks"],
            xlabel=False,
        )

        add_firing_rate_axis(
            ax=ax,
            label=labels[0],
            hill_params=hill_params_first,
            y_text=1.3,
            outward=25,
            major_ticks=axis_tick_config["both"]["first"]["major_ticks"],
            minor_ticks=axis_tick_config["both"]["first"]["minor_ticks"],
            xlabel=True,
        )

    elif mode == "first":
        add_firing_rate_axis(
            ax=ax,
            label=labels[0],
            hill_params=hill_params_first,
            y_text=1.125,
            outward=5,
            major_ticks=axis_tick_config["first"]["major_ticks"],
            minor_ticks=axis_tick_config["first"]["minor_ticks"],
            xlabel=True,
        )

    elif mode == "second":
        add_firing_rate_axis(
            ax=ax,
            label=labels[1],
            hill_params=hill_params_second,
            y_text=1.125,
            outward=5,
            major_ticks=axis_tick_config["second"]["major_ticks"],
            minor_ticks=axis_tick_config["second"]["minor_ticks"],
            xlabel=True,
        )

    ax.spines[["right", "top"]].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))
    ax.spines[["bottom", "left"]].set_linewidth(.7)
    ax.tick_params(axis="both", width=.7)

    ax.set_ylabel("PER")
    ax.set_ylim(ylim)
    ax.set_xlabel("[Sucrose] (mM)")

    return ax