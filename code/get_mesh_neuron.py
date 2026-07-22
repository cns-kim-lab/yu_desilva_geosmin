import flybrains
import navis
import os 
import numpy as np 
from fafbseg import flywire
import pickle 

def read_pickle(flyid,what):
    path = '/volume_4/research/seongbong/flywire/geosmin_project_version_update/data'
    path = f'{path}/{what}'
    with open(f'{path}/{flyid}.pkl','rb') as f:
        data = pickle.load(f)
    return data

def save_pickle(save_path,file):
    with open(save_path,'wb') as f: 
        pickle.dump(file,f)

def get_mesh_from_server(flyid):
    mlist = os.listdir('/volume_4/research/seongbong/flywire/geosmin_project_version_update/data/mesh')
    mlist = [int(x[:-4]) for x in mlist]
    if flyid not in mlist:
        try:
            m = flywire.get_mesh_neuron(flyid)
            save_pickle(f'/volume_4/research/seongbong/flywire/geosmin_project_version_update/data/mesh/{flyid}.pkl',m)
            return m
        except:
            m = [] 
            print(f'{flyid} failed')
            return m 



def load_mesh(flyid):
    mlist = os.listdir('/volume_4/research/seongbong/flywire/geosmin_project_version_update/data/mesh')
    mlist = [int(x[:-4]) for x in mlist]
    if flyid in mlist:
        # print(flyid)
        m = read_pickle(flyid,'mesh')
    else:
        m = get_mesh_from_server(flyid)
    return m 


def get_dps_from_server(mesh):
    dpslist = os.listdir('/volume_3/research/seongbong/flywire/data/dps')
    dpslist = [int(x[:-4]) for x in dpslist]
    if mesh.id not in dpslist:
        try:
            dps = navis.make_dotprops(mesh,k=5,resample=1000)
            dps2018 = navis.xform_brain(dps,source='FLYWIRE',target='JRC2018F')
            save_pickle(f'/volume_3/research/seongbong/flywire/data/dps/{dps2018.id}.pkl',dps2018)
            return dps2018
        except:
            dps2018 = [] 
            print(f'{dps.id} failed')
            return dps2018


def load_dps(mesh):
    dpslist = os.listdir('/volume_3/research/seongbong/flywire/data/dps')
    dpslist = [int(x[:-4]) for x in dpslist]
    if mesh.id in dpslist:
        dps2018 = read_pickle(mesh.id,'dps')
    else:
        dps2018 = get_dps_from_server(mesh)
    return dps2018


def get_dps_mir_from_server(dps):
    dpsmirlist = os.listdir('/volume_3/research/seongbong/flywire/data/dps_mirr')
    dpsmirlist = [int(x[:-4]) for x in dpsmirlist]
    if dps.id not in dpsmirlist:
        try:
            dps_mir = navis.mirror_brain(dps,template='JRC2018F')
            save_pickle(f'/volume_3/research/seongbong/flywire/data/dps_mirr/{dps_mir.id}.pkl',dps_mir)
            return dps_mir
        except:
            dps_mir = [] 
            print(f'{dps.id} failed')
            return dps_mir    
        
def load_dps_mir(dps):
    dpsmirlist = os.listdir('/volume_3/research/seongbong/flywire/data/dps_mirr')
    dpsmirlist = [int(x[:-4]) for x in dpsmirlist]
    if dps.id in dpsmirlist:
        dps_mir = read_pickle(dps.id,'dps_mirr')
    else:
        dps_mir = get_dps_mir_from_server(dps)
    return dps_mir