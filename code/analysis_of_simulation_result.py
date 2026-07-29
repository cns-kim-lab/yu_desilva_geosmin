from multiprocessing import Pool
import re
from pathlib import Path
import sys
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


def load_mn9_results(
    result_dir,
    target_neuron,
    target_name="MN9",
    n_simulations=100,
):
    """
    Load MN9 simulation results from each group directory.

    Expected directory structure
    ----------------------------
    result_dir/
        group_1/
            *.parquet
        group_2/
            *.parquet

    Returns
    -------
    mean_df : pandas.DataFrame
        Mean MN9 firing rates. Rows correspond to groups.
    sem_df : pandas.DataFrame
        SEM of MN9 firing rates. Rows correspond to groups.
    """
    result_dir = Path(result_dir)

    group_dirs = sorted(
        directory
        for directory in result_dir.iterdir()
        if directory.is_dir() and not directory.name.startswith(".")
    )

    mean_by_group = {}
    sem_by_group = {}

    for group_dir in group_dirs:
        parquet_files = sorted(group_dir.glob("*.parquet"))

        if not parquet_files:
            continue

        rate, _, std = read_result_with_multi_thread(
            [str(file) for file in parquet_files],
            target_neuron,
            target_name,
        )

        mean_by_group[group_dir.name] = pd.concat(rate, axis=1)
        sem_by_group[group_dir.name] = (
            pd.concat(std, axis=1) / np.sqrt(n_simulations)
        )

    if not mean_by_group:
        raise FileNotFoundError(
            f"No parquet result files were found in: {result_dir}"
        )

    mean_df = pd.concat(mean_by_group, names=["group"])
    sem_df = pd.concat(sem_by_group, names=["group"])

    # Each group is expected to contain one target-neuron row.
    mean_df = mean_df.droplevel(-1)
    sem_df = sem_df.droplevel(-1)

    return mean_df, sem_df

def make_frequency_grid(
    df,
    sweet_frequencies,
    geosmin_frequencies,
):
    """
    Reshape columns such as '20Hz_100Hz' into a frequency grid.

    Rows correspond to geosmin stimulation frequencies.
    Columns correspond to sweet stimulation frequencies.

    A separate grid is produced for each original row, using a MultiIndex.
    """
    grids = []

    for group_name, row in df.iterrows():
        group_grid = pd.DataFrame(
            {
                f"{sweet_freq}Hz": [
                    row[f"{sweet_freq}Hz_{geosmin_freq}Hz"]
                    for geosmin_freq in geosmin_frequencies
                ]
                for sweet_freq in sweet_frequencies
            },
            index=[f"{freq}Hz" for freq in geosmin_frequencies],
        )

        group_grid.index.name = "geosmin_frequency"
        group_grid["group"] = group_name
        grids.append(group_grid.reset_index())

    frequency_grid = pd.concat(grids, ignore_index=True)

    frequency_grid = frequency_grid.set_index(
        ["group", "geosmin_frequency"]
    )

    return frequency_grid


def frequency_sort_key(column_name):
    """Extract frequency values from names such as '20Hz' or '20Hz_100Hz'."""
    return tuple(
        int(value)
        for value in re.findall(r"(\d+)Hz", str(column_name))
    )