from multiprocessing import Pool
from tqdm import tqdm
import pickle 
import os 
import utils as utl 
import pandas as pd 
import numpy as np 
mn9 = [720575940660219265,720575940618238523] # 783 Ver. 
# sensory neuron data
df_comp = pd.read_csv('/volume_4/research/seongbong/flywire/geosmin_project_code/data/2023_03_23_completeness_630_final.csv')
labial_cluster = pd.read_parquet('/volume_4/research/seongbong/flywire/geosmin_project_version_update/data/labial_cluster_info_v783.parquet')
labial_sweet_df = labial_cluster[np.isin(labial_cluster.type,['L1','L2','L3'])]
labial_sweet_df.set_index('flyid',inplace=True)
labial_sweet = labial_cluster[np.isin(labial_cluster.type,['L1','L2','L3'])].flyid.values
labial_sweet = labial_sweet[np.isin(labial_sweet,df_comp['Unnamed: 0'])]

atGRN_cluster = pd.read_parquet('/volume_4/research/seongbong/flywire/geosmin_project_version_update/data/atGRN_cluster_info_v783.parquet')
atGRNs = atGRN_cluster[np.isin(atGRN_cluster.type,['a6','a7'])].flyid.values

TPN1 = [720575940623118029, 720575940624967561]
av1a1 = [720575940623041549,720575940622894616,720575940626958878,720575940633984924,720575940611137742,720575940627192337]




def read_simul_result(df_path):
    path = '/'.join(df_path.split('/')[:-1])
    params = pickle.load(open(os.path.join(path,'params','params.pkl'),'rb'))
    n_run=params['n_run']
    ps = [f'{df_path}']
#     print(ps)
    df_spike = utl.load_exps(ps)
    df_rate, df_std = utl.get_rate_include_zero(df_spike, duration=params['t_run'],n_run=n_run)
    return df_rate, df_spike, df_std

def read_simul_result_interest(df_path,interest=mn9):  
    path = '/'.join(df_path.split('/')[:-1])
    params = pickle.load(open(os.path.join(path,'params','params.pkl'),'rb'))  
    n_run=params['n_run']
    ps = [f'{df_path}']
#     print(ps)
    df_spike = utl.load_exps_interest(ps,interest)
    df_rate, df_std = utl.get_rate_include_zero(df_spike, duration=params['t_run'],n_run=n_run)
    return df_rate, df_spike, df_std

def read_simul_result_groups(df_path,interest=mn9,interest_id='MN9'):
    path = '/'.join(df_path.split('/')[:-1])
    params = pickle.load(open(os.path.join(path,'params','params.pkl'),'rb'))  
    n_run=params['n_run']
    ps = [f'{df_path}']
    condition = ps[0].split('/')[-1]
    condition = condition.split('.parquet')[0]
    df_spike = utl.load_exps_interest(ps,interest)
    df_rate, df_rate_std = utl.get_rate_group(df_spike,condition,interest,interest_id, duration=params['t_run'],n_run=n_run)
    return df_rate, df_spike,df_rate_std


def read_result_with_multi_thread(total_paths,*add_inputs):
    if len(add_inputs) == 1:
        interest = add_inputs[0]
        interest_group_id = None
    elif len(add_inputs) == 2:
        interest = add_inputs[0]
        interest_group_id = add_inputs[1]
    else:
        interest = None 
        interest_group_id = None
        
    if interest != None:
        if  interest_group_id == None:
            func = read_simul_result_interest
            arg_list = [(t, interest) for t in total_paths]
        else:
            func = read_simul_result_groups
            arg_list = [(t, interest,interest_group_id) for t in total_paths]
    else:
        func = read_simul_result
        arg_list = [(t,) for t in total_paths]

    with Pool(os.cpu_count()) as pool:
        results = list(tqdm(
            pool.starmap(func, arg_list),
            total=len(total_paths)
        ))

    spikes = [x[1] for x in results]
    rates  = [x[0] for x in results]
    std    = [x[2] for x in results]

    return rates,spikes,std


def make_data_frame_general(df, *ranges, index=None):
    all_df = []
    for combo in zip(*ranges):
        # ('10', '20', '30') -> '10Hz_20Hz_30Hz'
        label = '_'.join(f'{v}Hz' for v in combo)
        key   = f'{label}.parquet'
        this_df = df.get(key, pd.DataFrame())

        if len(this_df) == 0:
            this_df = pd.DataFrame(index=index, columns=[label])
        else:
            this_df = this_df.copy()
            this_df.columns = [label]
        all_df.append(this_df)

    all_df = pd.concat(all_df, axis=1)
    all_df.fillna(0, inplace=True)
    return all_df


def make_df(raw_data,group_id,exp_name):
    mean_per_trial = np.mean(raw_data)
    std_per_trial = np.std(raw_data)
    d = {
        'r' : [mean_per_trial],
        'std': [std_per_trial],
        'group_id' : [group_id],
        'exp_name' : [exp_name],
    }
    df = pd.DataFrame(d)

    df_rate = df.pivot_table(columns='exp_name', index='group_id', values='r')
    df_std = df.pivot_table(columns='exp_name', index='group_id', values='std')
    return df_rate,df_std