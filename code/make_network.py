import networkx as nx 
import numpy as np 
import pandas as pd 
import os 
import pickle
from color_utils import rgb_to_hex

with open('/volume_4/research/seongbong/flywire/geosmin_project_version_update/data/sez_neurons.pickle','rb') as f:
    sez = pickle.load(f)
# sez['MN9'] = [720575940660219265,720575940618238523]
# sez['FDG'] = sez.pop('Fdg')
# sez['G2N-1'] = sez.pop('G2N_1')
# sez['sink and sync'] = sez.pop('sink_sync')
# sez['rounddown'] = [720575940627847752,720575940609407939]



sez_ids = np.concatenate(list(sez.values()))    
sez_id2g = {f:g for g,fid in sez.items() for f in fid}
sez_label = list(sez_id2g.values())




# sensory neuron data
labial_cluster = pd.read_parquet('/volume_4/research/seongbong/flywire/geosmin_project_version_update/data/labial_cluster_info_v783.parquet')
labial_c2g = {}
for t in np.unique(labial_cluster.type):
    labial_c2g[t] = list(labial_cluster.flyid.values[labial_cluster.type==t])
labial_sweet = labial_c2g['L1']+labial_c2g['L2']+labial_c2g['L3']
labial_cluster = labial_cluster.set_index('flyid')

atGRN_cluster = pd.read_parquet('/volume_4/research/seongbong/flywire/geosmin_project_version_update/data/atGRN_cluster_info_v783.parquet')
atGRNs = atGRN_cluster[np.isin(atGRN_cluster.type,['a6','a7'])].flyid.values


TPN1 = [720575940623118029, 720575940624967561]
av1a1 = [720575940623041549,720575940622894616,720575940626958878,720575940633984924,720575940611137742,720575940627192337]
mn9 = [720575940660219265,720575940618238523]
with open('/volume_4/research/seongbong/flywire/geosmin_project_version_update/data/DA2_PN.pkl','rb') as f:
    da2pn = pickle.load(f)
with open('/volume_4/research/seongbong/flywire/geosmin_project_version_update/data/or56a.pkl','rb') as f:
    or56a = pickle.load(f)

syn_df = pd.read_parquet('/volume_4/research/seongbong/flywire/geosmin_project_version_update/data/synapse/simulation_data/Connectivity_783.parquet')

target_ids_valid_all = pickle.load(open('/volume_4/research/seongbong/flywire/geosmin_project_version_update/data/interneuron_group_info/target_ids_all_v783.pkl','rb'))
g_info_all = pickle.load(open('/volume_4/research/seongbong/flywire/geosmin_project_version_update/data/interneuron_group_info/g_info_all_v783.pkl','rb'))
fid2g = dict(zip(target_ids_valid_all,list(g_info_all)))


# add FlyWire type id 
class_df = pd.read_csv('/volume_4/research/seongbong/flywire/geosmin_project_version_update/data/consolidated_cell_types.csv',index_col=False)
fid2flyCid = dict(zip(class_df.root_id,class_df['primary_type']))
# fid2flyCid[720575940619853515] = fid2flyCid[720575940627099338]
# fid2flyCid[720575940620761187] = fid2flyCid[720575940634672610]
# fid2flyCid[720575940645521262] = fid2flyCid[720575940618238523]
# fid2flyCid[720575940610506594] = fid2flyCid[720575940614492397]
# fid2flyCid[720575940621002083] = fid2flyCid[720575940641701328]
# fid2flyCid[720575940640407923] = fid2flyCid[720575940625019911]
# fid2flyCid[720575940619498968] = fid2flyCid[720575940625673196]
# fid2flyCid[720575940632645805] = fid2flyCid[720575940625656980]
# fid2flyCid[720575940624012105] = fid2flyCid[720575940613039795]
# fid2flyCid[720575940625102094] = fid2flyCid[720575940634646753]
# fid2flyCid[720575940625926500] = fid2flyCid[720575940625978867]
# fid2flyCid[720575940632722636] = fid2flyCid[720575940633529548]
# fid2flyCid[720575940644995950] = fid2flyCid[720575940622942196]
# fid2flyCid[720575940626038714] = fid2flyCid[720575940615023535]
# fid2flyCid[720575940627750672] = fid2flyCid[720575940628787016]
# fid2flyCid[720575940612611301] = fid2flyCid[720575940626835146]
# fid2flyCid[720575940633041619] = fid2flyCid[720575940618252249]
# fid2flyCid[720575940643786184] = fid2flyCid[720575940624537284]
# fid2flyCid[720575940609935172] = fid2flyCid[720575940621777390]
# fid2flyCid[720575940621998460] = fid2flyCid[720575940617684107]
# fid2flyCid[720575940622805030] = fid2flyCid[720575940634660833]
# fid2flyCid[720575940623013887] = fid2flyCid[720575940629431814]
# fid2flyCid[720575940626741891] = fid2flyCid[720575940618594022]
# fid2flyCid[720575940621002083] = fid2flyCid[720575940641701328]
# fid2flyCid[720575940640407923] = fid2flyCid[720575940611958994]
# fid2flyCid[720575940619498968] = fid2flyCid[720575940625673196]
# fid2flyCid[720575940624012105] = fid2flyCid[720575940613039795]
# fid2flyCid[720575940625926500] = fid2flyCid[720575940625978867]




# flyCid2typeid = {fid2flyCid[c]:int(fid2g[c]) for c in cells_in_network_merged}




# make network of connectome neurons 
def make_network(cells_in_network,syn_df=syn_df):
    G = nx.DiGraph()
    edges = np.array(syn_df[np.isin(syn_df['Presynaptic_ID'],cells_in_network)&np.isin(syn_df['Postsynaptic_ID'],cells_in_network)][['Presynaptic_ID','Postsynaptic_ID','Connectivity']])
    excitatory = np.array(syn_df[np.isin(syn_df['Presynaptic_ID'],cells_in_network)&np.isin(syn_df['Postsynaptic_ID'],cells_in_network)][['Excitatory']])
    excitatory = np.concatenate(excitatory)
    G.add_nodes_from(cells_in_network)
    G.add_weighted_edges_from(edges)
    
    return G,edges,excitatory


# make network of type of neurons (metaNetwork)
def make_meta_network(cells_in_network,add=False,syn_df=syn_df):

    sugar_network,edges,excitatory = make_network(cells_in_network,syn_df=syn_df)
    for c in cells_in_network:
        if c in labial_cluster.index:
            sugar_network.nodes[c]['name'] = labial_cluster.loc[c]['type']
        elif c in atGRNs:
            sugar_network.nodes[c]['name'] = 'atGRN'
        elif c in TPN1:
            sugar_network.nodes[c]['name'] = 'TPN1'
        elif c in av1a1:
            sugar_network.nodes[c]['name'] = 'av1a1'
        elif c in or56a:
            sugar_network.nodes[c]['name'] = 'Or56a'
        elif c in da2pn:
            sugar_network.nodes[c]['name'] = 'DA2_PN'
        elif c not in sez_ids:
            # sugar_network.nodes[c]['name'] = fid2flyCid[c]
            sugar_network.nodes[c]['name'] = str(int(fid2g[c]))
        else:
            sugar_network.nodes[c]['name'] = str(sez_id2g[c])

    edges_copy = edges.copy()

    meta_edges = []
    for e in edges_copy:
        temp = []
        if e[0] in labial_cluster.index:
            temp.append(labial_cluster.loc[e[0]]['type'])
        elif e[0] in atGRNs:
            temp.append('atGRN')
        elif e[0] in TPN1:
            temp.append('TPN1')
        elif e[0] in av1a1:
            temp.append('av1a1')
        elif e[0] in or56a:
            temp.append('Or56a')
        elif e[0] in da2pn:
            temp.append('DA2_PN')
        elif e[0] in sez_ids:
            temp.append(str(sez_id2g[e[0]]))
        elif e[0] in target_ids_valid_all:
            temp.append(str(int(fid2g[e[0]])))
        # elif e[0] in target_ids_valid_all:
        #     temp.append(fid2flyCid[e[0]])


        if e[1] in labial_cluster.index:
            temp.append(labial_cluster.loc[e[1]]['type'])
        elif e[1] in atGRNs:
            temp.append('atGRN')
        elif e[1] in TPN1:
            temp.append('TPN1')
        elif e[1] in av1a1:
            temp.append('av1a1')
        elif e[1] in or56a:
            temp.append('Or56a')
        elif e[1] in da2pn:
            temp.append('DA2_PN')
        elif e[1] in sez_ids:
            temp.append(str(sez_id2g[e[1]]))#temp.append(str(sez_label[sez_ids==e[1]][0]))
        elif e[1] in target_ids_valid_all:
            temp.append(str(int(fid2g[e[1]])))
        # elif e[1] in target_ids_valid_all:
        #     temp.append(fid2flyCid[e[1]])


        temp.append(e[-1])

        meta_edges.append(temp)

        
        
    a = np.unique(np.array(meta_edges)[:,:-1],axis=0)
    meta_counts = []
    meta_excit = []

    for e in a:
        meta_counts.append(np.sum([int(x) for x in np.array(meta_edges)[np.all(np.array(meta_edges)[:,:-1]==e,axis=1),-1]]))
        meta_excit.append(np.sum(np.array(excitatory)[np.all(np.array(meta_edges)[:,:-1]==e,axis=1)]))
    meta_excit = np.array(meta_excit)
    # print(a[meta_excit==0])
    meta_excit[meta_excit==0] = 1 
    meta_excit_ = np.array(meta_excit)/np.abs(meta_excit)
    
    meta_w_e_sugar =[]
    for e in enumerate(a):
        x = list(e[1])
        x.append(meta_counts[e[0]])
        meta_w_e_sugar.append(tuple(x))

    metaG = nx.DiGraph()
    metaG.add_weighted_edges_from(meta_w_e_sugar)
    for i,e in enumerate(a):
        u,v = e
        metaG[u][v]['excit'] = meta_excit_[i]
    if add == False:
        return metaG
    else:
        return metaG,edges_copy,meta_edges


def set_node_location(metaG,pos_info):
    # pos_info = {node1:{'x':x_pos,'y':y_pos,'z':0},...}
    for n in metaG.nodes():
        metaG.nodes[n]['viz']= {}
        metaG.nodes[n]['viz']['position'] = pos_info[n]
    return metaG

def set_node_color(metaG,color_info):
    # color_info = {node1:{'a':a,'r':r,'g':g,'b':b}}
    for n in metaG.nodes():
        metaG.nodes[n]['viz']['color'] = color_info[n]
    return metaG

def set_edge_color(G):
    for u,v in G.edges():
        if G[u][v]['excit']==1:
            G[u][v]['viz'] = {"color": {"r": 0, "g": 255, "b": 0, "a": 1.0}}
        else:
            G[u][v]['viz'] = {"color": {"r": 255, "g": 0, "b": 0, "a": 1.0}}
    return G



def get_cell2canonical(cells_in_network):
    cell2canonical = {}
    for c in cells_in_network:
        if c in labial_cluster.index:
            cell2canonical[c] = labial_cluster.loc[c]['type']
        elif c in atGRNs:
            cell2canonical[c] = 'atGRN'
        elif c in TPN1:
            cell2canonical[c] = 'TPN1'
        elif c in av1a1:
            cell2canonical[c] = 'av1a1'
        elif c not in sez_ids:
            # sugar_network.nodes[c]['name'] = fid2flyCid[c]
            cell2canonical[c] = str(int(fid2g[c]))
        else:
            cell2canonical[c] = str(sez_id2g[c])

    return cell2canonical