import navis 
import os 
import numpy as np 
import pandas as pd 
import pickle 
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, set_link_color_palette
import distinctipy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns 
from color_utils import rgb_to_hex


path_syn = '/volume_4/research/seongbong/flywire/geosmin_project_version_update/data/synapse/v783'
path_comp = '/volume_4/research/seongbong/flywire/geosmin_project_version_update/data/Completeness_783.csv'
with open('/volume_4/research/seongbong/flywire/geosmin_project_version_update/data/interneuron_group_info/target_ids_all_v783.pkl','rb') as f:
    target_ids_valid_all = pickle.load(f)
with open('/volume_4/research/seongbong/flywire/geosmin_project_version_update/data/interneuron_group_info/g_info_all_v783.pkl','rb') as f:
    g_info_all = pickle.load(f)
fid2g = dict(zip(list(target_ids_valid_all),list(g_info_all)))



def make_adjacency_matrix_GRN_to_2ndtype(interest_grns):
    second = []
    for c in interest_grns:
        c_df = pd.read_csv(f'{path_syn}/{c}.csv')
        second.append(c_df)
        
    
    second_df = pd.concat(second)
    second_df = second_df[second_df['cleft_score']>50]  # cleft score thresholed synapses

    
    df_comp = pd.read_csv(path_comp,index_col=False)
    completed = df_comp['Unnamed: 0'].values
    second_list = np.unique(second_df['post'].values) 
    second_valid = second_list[np.isin(second_list,completed)] # use only postsynaptic neurons which were completed 
    second_valid = np.setdiff1d(second_valid,interest_grns) # remove intersynapse between GRNs 
    second_valid = second_valid[np.isin(second_valid,target_ids_valid_all)] # use only neurons which were pre-clustered neurons


    valid_df = second_df[(np.isin(second_df['pre'],interest_grns))&(np.isin(second_df['post'],second_valid))]
    edges,edge_w = np.unique(valid_df[['pre','post']],axis=0,return_counts=True)

    second_fid2id = {j: i for i, j in enumerate(second_valid)}
    id2second_fid = {j: i for i, j in second_fid2id.items()}

    first_fid2id = {j: i for i, j in enumerate(interest_grns)}
    id2first_fid = {j: i for i, j in first_fid2id.items()}

    # adjacency matrix GRN to 2nd 
    mat = np.zeros((len(interest_grns),len(second_valid)))
    for i in list(range(len(edges))):
        mat[first_fid2id[edges[i,0]],second_fid2id[edges[i,1]]] = edge_w[i]
        
    # apply 2nd type information on 
    g_info = [fid2g[c] for c in second_valid]
    g_info_unique = np.unique(g_info)
    mat_g = np.zeros((len(interest_grns),len(g_info_unique)))
    num_g = 0
    for g in g_info_unique:
        idx = g_info==g
        mat_g[:,num_g] = np.sum(mat[:,idx],axis=1)
        num_g += 1 
    mat_g = pd.DataFrame(mat_g,columns=g_info_unique,index=interest_grns)
    return mat_g


def cluster_GRNs(mat_g):
    
    conn_sim_mat = navis.connectivity_similarity(mat_g,metric='cosine')
    conn_dist_mat = 1 - conn_sim_mat



    aba_vec = squareform(conn_dist_mat, checks=False)
    Z = linkage(aba_vec, method='ward')

    return Z


def draw_dendrogram_with_adjmat(fig,mat_g,interest_grns,colors='default',fontsize=10,threshold=.3):

    if colors == 'default':
        colors_rgb = distinctipy.get_colors(20) # arbitrary number of clustered neurons 
        colors = [rgb_to_hex(x) for x in colors_rgb]    
    set_link_color_palette(colors)
    

    outer = fig.add_gridspec(1, 2, width_ratios=[0.15, 0.85], wspace=0)
    axD1 = fig.add_subplot(outer[0, 0])
    inner = gridspec.GridSpecFromSubplotSpec(
        3, 2, subplot_spec=outer[0, 1], width_ratios=[0.96, 0.04],height_ratios=[1,1,1], wspace=0.05 
    )
    axD2 = fig.add_subplot(inner[:, 0])  # heatmap
    axD3 = fig.add_subplot(inner[-1, 1])  # colorbar
    
    Z = cluster_GRNs(mat_g)
    
    Z2 = np.copy(Z)
    alpha = 1
    Z2[:, 2] = [np.log10(x/np.min(Z[:,2])+alpha)for x in Z2[:,2]]
    dn = dendrogram(Z2, labels=interest_grns, orientation='left',
                    color_threshold=np.log10(threshold/np.min(Z[:,2])+alpha),ax=axD1)#,no_plot=True)

    # for xs, ys, color in zip(dn['icoord'], dn['dcoord'], dn['color_list']):
    #     axD1.plot(ys, xs, color=color, linewidth=.75)  
    # axD1.invert_xaxis()

    # heatmap
    wanted_order = np.array(dn['ivl'])[::-1]
    idx = [list(interest_grns).index(x) for x in wanted_order]
    mat_g_arr = np.array(mat_g)
    mat_g_sorted = mat_g_arr[idx, :]

    mat_data = np.log10(mat_g_sorted[:, np.argsort(np.sum(mat_g_arr,axis=0))[::-1][:50]])
    mat_data[np.isneginf(mat_data)] = 0
    im = sns.heatmap(mat_data, ax=axD2, cmap='jet', cbar=False)


    # colorbar beside D2
    cbar = fig.colorbar(im.collections[0], cax=axD3,
                        ticks=np.log10([1,10,100]))
    cbar.set_ticklabels([1,10,100])
    cbar.ax.tick_params(labelsize=fontsize*.5)

    axD1.set_yticks([])
    axD1.set_xticks([])
    axD2.set_yticks([])
    axD2.set_xticks([])
    axD1.axis('off')

    return dn

def draw_dendrogram_with_adjmat_vertical(
    fig, mat_g, interest_grns, colors='default', fontsize=10, threshold=.3
):
    if colors == 'default':
        colors_rgb = distinctipy.get_colors(20)
        colors = [rgb_to_hex(x) for x in colors_rgb]
    set_link_color_palette(colors)

    # top: dendrogram / bottom: heatmap + colorbar
    outer = fig.add_gridspec(
        2, 1,
        height_ratios=[0.15, 0.85],
        hspace=0
    )

    top = gridspec.GridSpecFromSubplotSpec(
        1, 2,
        subplot_spec=outer[0, 0],
        width_ratios=[0.96, 0.04],
        wspace=0.05
    )

    bottom = gridspec.GridSpecFromSubplotSpec(
        1, 2,
        subplot_spec=outer[1, 0],
        width_ratios=[0.96, 0.04],
        wspace=0.05
    )

    axD1 = fig.add_subplot(top[0, 0])      # dendrogram
    ax_empty = fig.add_subplot(top[0, 1])  # dummy for colorbar width
    axD2 = fig.add_subplot(bottom[0, 0])   # heatmap
    axD3 = fig.add_subplot(bottom[0, 1])   # colorbar

    ax_empty.axis('off')


    # clustering
    Z = cluster_GRNs(mat_g)

    Z2 = np.copy(Z)
    alpha = 1
    Z2[:, 2] = [
        np.log10(x / np.min(Z[:, 2]) + alpha)
        for x in Z2[:, 2]
    ]

    dn = dendrogram(
        Z2,
        labels=interest_grns,
        orientation='top',
        color_threshold=np.log10(threshold / np.min(Z[:, 2]) + alpha),
        ax=axD1
    )

    # dendrogram order: left to right
    wanted_order = np.array(dn['ivl'])
    idx = [list(interest_grns).index(x) for x in wanted_order]

    mat_g_arr = np.array(mat_g)

    # rows: top 50 downstream neurons
    # columns: clustered GRNs
    top_cols = np.argsort(np.sum(mat_g_arr, axis=0))[::-1][:50]

    mat_data = np.log10(mat_g_arr[idx, :][:, top_cols].T)
    mat_data[np.isneginf(mat_data)] = 0

    im = sns.heatmap(
        mat_data,
        ax=axD2,
        cmap='jet',
        cbar=False
    )

    # colorbar
    cbar = fig.colorbar(
        im.collections[0],
        cax=axD3,
        ticks=np.log10([1, 10, 100])
    )
    cbar.set_ticklabels([1, 10, 100])
    cbar.ax.tick_params(labelsize=fontsize *2)

    # clean axes
    axD1.set_xticks([])
    axD1.set_yticks([])
    axD2.set_xticks([])
    axD2.set_yticks([])

    axD1.axis('off')

    return dn


def draw_dendrogram(ax,mat_g,interest_grns,colors='default',fontsize=10,threshold=.3):

    if colors == 'default':
        colors_rgb = distinctipy.get_colors(20) # arbitrary number of clustered neurons 
        colors = [rgb_to_hex(x) for x in colors_rgb]    
    set_link_color_palette(colors)
    

    Z = cluster_GRNs(mat_g)
    
    Z2 = np.copy(Z)
    alpha = 1
    Z2[:, 2] = [np.log10(x/np.min(Z[:,2])+alpha)for x in Z2[:,2]]
    dn = dendrogram(Z2, labels=interest_grns, 
                    color_threshold=np.log10(threshold/np.min(Z[:,2])+alpha),ax=ax)#,no_plot=True)
    
    return dn 