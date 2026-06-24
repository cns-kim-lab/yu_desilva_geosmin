import matplotlib.pyplot as plt
import distinctipy
import numpy as np 
import pandas as pd 
import os 
import pickle
from scipy.stats import mannwhitneyu,ranksums,wilcoxon,shapiro
import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['font.family'] = 'Arial'


fig_params = {
'cmap' : ['#13739E', '#952B7E', '#E68613', '#4CAF50'],
'fontsize':10,
'lw' : 1,
'elw' : .75,
'ms' : 0,
'cs':1,
'lp' : 2,
'ct' : .5,
'ecolor' : 'black'}



def std_err(arr):
    err = np.nanstd(arr,axis=0)/np.sqrt(np.sum(~np.isnan(arr),axis=0))
    return err

def pval_to_asterisks(p):
    if p < 0.0001:
        return "****"
    elif p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"



def plot_dose_dp_PER(ax,data,cidx=0,alpha=1,linestyle='solid',shape='o'):
    # fig,ax = plt.subplots(1,1,figsize=figsize)
    lw = fig_params['lw'] 
    elw = fig_params['elw']
    ms = fig_params['ms']
    cmap_all = fig_params['cmap']
    cmap = cmap_all[cidx]
    cs = fig_params['cs']
    ecolor = fig_params['ecolor']
    ct = fig_params['ct']
    p1 = ax.errorbar(list(range(len(data.columns))),data.values.mean(axis=0),yerr=std_err(data.values),linewidth=lw,elinewidth=elw,markersize=ms,fmt='-o',capsize=cs,color=cmap,ecolor=ecolor,capthick=ct,alpha=alpha,linestyle=linestyle)  

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_statistics_result(ax,data_compare):
    d1,d2 = data_compare
    mean_d1 = d1.values.mean(axis=0)
    stat_p_all = []
    for x,concent in enumerate(d1.columns):
        d_x = np.asarray(d1[concent])
        d_y = np.asarray(d2[concent])

        mask = ~np.isnan(d_x) & ~np.isnan(d_y)
        d_x, d_y = d_x[mask], d_y[mask]
        diff = d_x - d_y
        diff = diff[diff != 0]

        stat,p = wilcoxon(diff,alternative='greater',method='auto')
        ast = pval_to_asterisks(p)
        ax.text(x,mean_d1[x]+.05,ast, ha='center', va='bottom')
        stat_p_all.append((stat,p))
    return stat_p_all


def make_pivot_optogenetics_data(df):
    mean_df = (
        df.groupby(['retinal', 'light'])['PER']
        .mean()
        .reset_index()
        .rename(columns={'PER': 'mean PER'})
    )

    sem_df = (
        df.groupby(['retinal', 'light'])['PER']
        .sem()
        .reset_index()
        .rename(columns={'PER': 'ste PER'})
    )
    df_pivot = mean_df.pivot(index='light', columns='retinal', values='mean PER')
    df_pivot.columns = ['retinal (-)', 'retinal (+)']

    err_pivot = sem_df.pivot(index='light', columns='retinal', values='ste PER')
    err_pivot.columns = ['retinal (-)', 'retinal (+)']

    data_df = {}
    for i,d in df.groupby(['light', 'retinal'])['PER']:
        data_df[i] = d 
    return data_df,df_pivot,err_pivot



def jitter_x_by_value_counts(values, x_base, bar_width, side_sign, sort_values=True):
    """
    values: 1D array (NaN 제외된 raw y)
    x_base: 해당 condition의 중심 x
    bar_width: bar 폭
    side_sign: -1 (왼쪽 bar), +1 (오른쪽 bar)
    """
    v = np.asarray(values)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return np.array([]), np.array([])

    if sort_values:
        v_unique, v_counts = np.unique(v, return_counts=True)
    else:
        v_unique, v_counts = np.unique(v, return_counts=True)

    x_jittered = []
    for val, n in zip(v_unique, v_counts):
        step = bar_width / (n + 1)
        for s in range(1, n + 1):
            x_jittered.append(x_base + side_sign * step * s)

    y_resorted = np.concatenate([np.repeat(val, n) for val, n in zip(v_unique, v_counts)])
    return np.array(x_jittered), y_resorted

def add_sig_bracket(ax, x1, x2, y, h, text, fontsize=8, color='k', lw=1):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], c=color, lw=lw)
    ax.text((x1 + x2) * 0.5, y + h, text, ha='center', va='bottom', color=color, fontsize=fontsize)


def draw_grouped_bar_with_points(
    ax,
    conditions,                 # list of condition labels (x tick labels)
    groups,                     # list of group names (len=2, recommened) ex ['sugar','sugar+geosmin']
    get_values,                 # function(cond, group)-> 1D array-like raw values
    means=None,                 # dict-like: means[cond][group] or means[(cond,group)]
    errs=None,                  # dict-like: errs[cond][group] or errs[(cond,group)]
    get_mean=None,              # function(cond, group)-> float (means 대신 사용)
    get_err=None,               # function(cond, group)-> float (errs 대신 사용)
    bar_width=0.2,
    bar_spacing=0.7,
    colors=None,                # list len=2
    edgecolor='black',
    linewidth=0.5,
    capsize=3,
    point_facecolor='white',
    point_edgecolor='black',
    point_lw=0.3,
    point_size=15,
    point_alpha=0.9,
    zorder_points=10,
):
    if colors is None:
        colors = ['#13739E', '#952B7E']
    assert len(groups) == 2, "현재 구현은 2-group(좌/우 bar) 구조를 기본으로 합니다."

    # x locations
    num_bars = len(conditions)
    x_loc = np.arange(0, num_bars * bar_spacing, bar_spacing)

    # bar centers offset: left(-) right(+)
    group_offsets = [-bar_width/2, +bar_width/2]
    group_sides   = [-1, +1]  # jitter 방향

    # bars
    for gi, g in enumerate(groups):
        heights = []
        yerrs = []
        for cond in conditions:
            if get_mean is not None:
                heights.append(get_mean(cond, g))
            else:
                heights.append(means[cond][g] if isinstance(means[cond], dict) else means[(cond, g)])

            if get_err is not None:
                yerrs.append(get_err(cond, g))
            else:
                yerrs.append(errs[cond][g] if isinstance(errs[cond], dict) else errs[(cond, g)])

        ax.bar(
            x_loc + group_offsets[gi],
            heights,
            yerr=yerrs,
            width=bar_width,
            color=colors[gi],
            edgecolor=edgecolor,
            linewidth=linewidth,
            capsize=capsize,
            label=str(g),
        )

    # raw points
    for i, cond in enumerate(conditions):
        x_base = x_loc[i]
        for gi, g in enumerate(groups):
            vals = np.asarray(get_values(cond, g))
            vals = vals[~np.isnan(vals)]

            xj, yj = jitter_x_by_value_counts(
                vals,
                x_base=x_base,
                bar_width=bar_width,
                side_sign=group_sides[gi]
            )

            ax.scatter(
                xj, yj,
                color=point_facecolor,
                edgecolor=point_edgecolor,
                linewidth=point_lw,
                s=point_size,
                zorder=zorder_points,
                alpha=point_alpha
            )

    return x_loc, group_offsets





def draw_paired_bar_figure(
    data,
    mean_df,
    err_df,
    ax,
    conds,
    groups,
    get_values,
    stat_pairs=None,  # [((cond1, group1), (cond2, group2)), ...]
    colors=('#AEACAB', 'black'),
    ylabel='Tarsal PER',
    xtick_func=lambda x: str(x),
    ylim=(0, 0.8),
    yticks=None,
    alternative='greater',
    fontsize=10,
    bar_width=0.15,
    bar_spacing=0.7,
    dot_size=15,
    stat_fontsize=8,
):
    x_loc = np.arange(0, len(conds) * bar_spacing, bar_spacing)

    def bar_x(cond, group):
        ci = conds.index(cond)
        gi = groups.index(group)
        offset = -bar_width / 2 if gi == 0 else bar_width / 2
        return x_loc[ci] + offset

    for group_idx, group in enumerate(groups):
        offset = -bar_width / 2 if group_idx == 0 else bar_width / 2

        ax.bar(
            x_loc + offset,
            mean_df.loc[conds, group].values,
            yerr=err_df.loc[conds, group].values,
            width=bar_width,
            label=group,
            color=colors[group_idx],
            edgecolor='black',
            linewidth=.5,
            capsize=3
        )

    for i, cond in enumerate(conds):
        x_base = x_loc[i]

        for group_idx, group in enumerate(groups):
            values = np.array(get_values(data, cond, group), dtype=float)
            values = values[~np.isnan(values)]

            x_jittered = []
            v_unique, v_counts = np.unique(values, return_counts=True)
            direction = (-1) ** (group_idx + 1)

            for v, n in zip(v_unique, v_counts):
                step = bar_width / (n + 1)
                for s in range(1, n + 1):
                    x_jittered.append(x_base + direction * step * s)

            values_resorted = np.concatenate([
                np.repeat(v, c) for v, c in zip(v_unique, v_counts)
            ])

            ax.scatter(
                x_jittered,
                values_resorted,
                color='white',
                edgecolor='black',
                linewidth=0.3,
                s=dot_size,
                zorder=10,
                alpha=0.9
            )

    if stat_pairs is not None:
        stat_all = []
        for pair_idx, ((cond1, group1), (cond2, group2)) in enumerate(stat_pairs):
            v1 = np.array(get_values(data, cond1, group1), dtype=float)
            v2 = np.array(get_values(data, cond2, group2), dtype=float)

            valid = ~np.isnan(v1) & ~np.isnan(v2)
            v1 = v1[valid]
            v2 = v2[valid]

            stat, pval = wilcoxon(v1, v2, alternative=alternative,method='auto')
            stat_all.append((stat,pval))
            x1 = bar_x(cond1, group1)
            x2 = bar_x(cond2, group2)

            
            h = 0.025
            y_offset = 0.015
            y = max(np.nanmax(v1), np.nanmax(v2)) + y_offset
            ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], c='k')
            ax.text(
                (x1 + x2) / 2,
                y + h,
                pval_to_asterisks(pval),
                ha='center',
                va='bottom',
                fontsize=stat_fontsize
            )

    ax.set_xticks(
        x_loc,
        [xtick_func(c) for c in conds],
        rotation=0,
        fontsize=fontsize
    )

    if yticks is None:
        yticks = np.round(np.arange(ylim[0], ylim[1] + 0.001, 0.2), 2)

    ax.set_yticks(yticks, yticks, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=2)
    ax.set_xlim([x_loc[0] - 0.4, x_loc[-1] + 0.4])
    ax.set_ylim(ylim)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    if ax.get_legend() is not None:
        ax.get_legend().remove()

    return ax,stat_all