import os 
import numpy as np
import pandas as pd 
import navis 
from fafbseg import flywire
from caveclient import CAVEclient 
import datetime
save_path  = '../data/synapse/v783_cave'
datastack_name = "flywire_fafb_production"
client = CAVEclient(datastack_name)
client.version = 783
# q_time = datetime.datetime(2023, 4, 2)

# def fetch_synapse_from_server(cell):
#     clist = os.listdir(save_path)
#     clist = [int(x[:-4]) for x in clist]
#     if cell not in clist:
#         try:
#             df = client.materialize.synapse_query(pre_ids=cell)#,timestamp=q_time
#             df.to_csv(f'{save_path}/{cell}.csv',index=False)
#         except:
#             return cell

def query_synapses_from_server(fid,pre=True,post=True):
    df = flywire.synapses.get_synapses(fid,pre=pre,post=post,filtered=False,materialization=783)
    df.to_csv(f'/volume_4/research/seongbong/flywire/geosmin_project_version_update/data/synapse/v783/{fid}.csv')

def query_synapses_from_local(fid):
    df = pd.read_csv(f'/volume_4/research/seongbong/flywire/geosmin_project_version_update/data/synapse/v783/{fid}.csv',index_col=False)
    return df 

def query_synapses(fid):
    flist = os.listdir(f'/volume_4/research/seongbong/flywire/geosmin_project_version_update/data/synapse/v783')
    if f"{fid}.csv" not in flist:
        query_synapses_from_server(fid)
    else:
        df = query_synapses_from_local(fid)
        return df 



def check_latest_ids(cell):
    id_info = flywire.update_ids(cell)#,timestamp=q_time
    if ~(id_info['changed'].values[0]):
        fetch_synapse_from_server(cell)
    else:
        return id_info 

if __name__ == '__main__':
    import sys
    cell = int(sys.argv[1])
    check_latest_ids(cell)

