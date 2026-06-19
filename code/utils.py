import pandas as pd
import numpy as np

##########
# analysis
def load_exps(l_pqt):
    '''Load simulation results from disk

    Parameters
    ----------
    l_pkl : list
        List of parquet files with simulation results

    Returns
    -------
    exps : df
        data for all experiments 'path_res'
    '''
    # cycle through all experiments
    dfs = []
    for p in l_pqt:
        # load metadata from pickle
        with open(p, 'rb') as f:
            df = pd.read_parquet(p)
            df.loc[:, 't'] = df.loc[:, 't'].astype(float)
            dfs.append(df)

    df = pd.concat(dfs)

    return df


def load_exps_interest(l_pqt,interest):
    '''Load simulation results from disk

    Parameters
    ----------
    l_pkl : list
        List of parquet files with simulation results

    Returns
    -------
    exps : df
        data for all experiments 'path_res'
    '''
    # cycle through all experiments
    dfs = []
    for p in l_pqt:
        # load metadata from pickle
        with open(p, 'rb') as f:
            df = pd.read_parquet(p)
            df = df[np.isin(df['flywire_id'],interest)]
            df.loc[:, 't'] = df.loc[:, 't'].astype(float)
            dfs.append(df)

    df = pd.concat(dfs)

    return df

def get_rate(df, duration, flyid2name=dict()):
    '''Calculate rate and standard deviation for all experiments
    in df

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe generated with `load_exps` containing spike times
    duration : float
        Trial duration in seconds
    flyid2name : dict (optional)
        Mapping between flywire IDs and custom names

    Returns
    -------
    df_rate : pd.DataFrame
        Dataframe with average firing rates
    df_std : pd.DataFrame
        Dataframe with standard deviation of firing rates
    '''

    rate, std, flyid, exp_name = [], [], [], []

    for e, df_e in df.groupby('exp_name', sort=False):
        for f, df_f in df_e.groupby('flywire_id'):

            r = []
            for _, df_t in df_f.groupby('trial'): # 한 번도 fire하지 않은 경우는 고려되지 않았음
                r.append(len(df_t) / duration)    # rate 평균을 구할 것이 아니라, 분포를 봐야할 듯 함 
            r = np.array(r)

            rate.append(r.mean())
            std.append(r.std())
            flyid.append(f)
            exp_name.append(e)

    d = {
        'r' : rate,
        'std': std,
        'flyid' : flyid,
        'exp_name' : exp_name,
    }
    df = pd.DataFrame(d)
    
    df_rate = df.pivot_table(columns='exp_name', index='flyid', values='r')
    df_std = df.pivot_table(columns='exp_name', index='flyid', values='std')
    
    if flyid2name:
        df_rate.insert(loc=0, column='name', value=df_rate.index.map(flyid2name).fillna(''))
        df_std.insert(loc=0, column='name', value=df_rate.index.map(flyid2name).fillna(''))

    return df_rate, df_std


def rm_outlier_using_zscore(data):
    m = np.mean(data)
    std = np.std(data)
    z = (data-m)/std
    return data[np.abs(z)<3]

def get_rate_rm_outlier(df, duration, n_run=30,flyid2name=dict()):
    '''Calculate rate and standard deviation for all experiments
    in df

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe generated with `load_exps` containing spike times
    duration : float
        Trial duration in seconds
    flyid2name : dict (optional)
        Mapping between flywire IDs and custom names

    Returns
    -------
    df_rate : pd.DataFrame
        Dataframe with average firing rates
    df_std : pd.DataFrame
        Dataframe with standard deviation of firing rates
    '''

    rate, std, flyid, exp_name = [], [], [], []

    for e, df_e in df.groupby('exp_name', sort=False):
        for f, df_f in df_e.groupby('flywire_id'):

            r = np.zeros(n_run) # 현재는 30으로 고정 -> 실험 상황에 따라 바꿀 수 있도록 수정할 필요성이 있음 
            for _, df_t in df_f.groupby('trial'): # 한 번도 fire하지 않은 경우는 고려되지 않았음
                r[df_t.trial.values[0]] = len(df_t) / duration    # rate 평균을 구할 것이 아니라, 분포를 봐야할 듯 함 
            # r = np.array(r)

            normal_r = rm_outlier_using_zscore(r)

            rate.append(normal_r.mean())
            std.append(normal_r.std())
            flyid.append(f)
            exp_name.append(e)

    d = {
        'r' : rate,
        'std': std,
        'flyid' : flyid,
        'exp_name' : exp_name,
    }
    df = pd.DataFrame(d)
    
    df_rate = df.pivot_table(columns='exp_name', index='flyid', values='r')
    df_std = df.pivot_table(columns='exp_name', index='flyid', values='std')
    
    if flyid2name:
        df_rate.insert(loc=0, column='name', value=df_rate.index.map(flyid2name).fillna(''))
        df_std.insert(loc=0, column='name', value=df_rate.index.map(flyid2name).fillna(''))

    return df_rate, df_std



def get_rate_include_zero(df, duration,n_run=30, flyid2name=dict()):
    '''Calculate rate and standard deviation for all experiments
    in df

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe generated with `load_exps` containing spike times
    duration : float
        Trial duration in seconds
    flyid2name : dict (optional)
        Mapping between flywire IDs and custom names

    Returns
    -------
    df_rate : pd.DataFrame
        Dataframe with average firing rates
    df_std : pd.DataFrame
        Dataframe with standard deviation of firing rates
    '''

    rate, std, flyid, exp_name = [], [], [], []

    for e, df_e in df.groupby('exp_name', sort=False):
        for f, df_f in df_e.groupby('flywire_id'):

            r = np.zeros(n_run)
            for _, df_t in df_f.groupby('trial'): # 한 번도 fire하지 않은 경우는 고려되지 않았음
                r[df_t.trial.values[0]] = len(df_t) / duration 
            

            rate.append(r.mean())
            std.append(r.std())
            flyid.append(f)
            exp_name.append(e)

    d = {
        'r' : rate,
        'std': std,
        'flyid' : flyid,
        'exp_name' : exp_name,
    }
    df = pd.DataFrame(d)
    
    df_rate = df.pivot_table(columns='exp_name', index='flyid', values='r')
    df_std = df.pivot_table(columns='exp_name', index='flyid', values='std')
    
    if flyid2name:
        df_rate.insert(loc=0, column='name', value=df_rate.index.map(flyid2name).fillna(''))
        df_std.insert(loc=0, column='name', value=df_rate.index.map(flyid2name).fillna(''))

    return df_rate, df_std




def get_rate_group(df,exp_name, group_cells,group_id,duration, flyid2name=dict(),n_run=30):

    rate, std, flyid = [], [], []
    df = df[np.isin(df.flywire_id.values,group_cells)]
    if len(df)!=0:
        r_per_cell = []
        for f, df_f in df.groupby('flywire_id'):
            r = np.zeros(n_run)
            for _, df_t in df_f.groupby('trial'): 
                r[df_t.trial.values[0]] = len(df_t) / duration   
            r_per_cell.append(r)
        mean_in_group = np.mean(r_per_cell,axis=0)
        std_in_group = np.std(r_per_cell,axis=0)

        mean_per_trial = np.mean(mean_in_group) 
        std_per_trial = np.std(mean_in_group)
#         exp_name = df.exp_name.values[0]
    else: 
        mean_per_trial = 0
        std_per_trial = 0 
        
        
            
    d = {
        'r' : [mean_per_trial],
        'std': [std_per_trial],
        'group_id' : [group_id],
        'exp_name' : [exp_name],
    }
    df = pd.DataFrame(d)
    
    df_rate = df.pivot_table(columns='exp_name', index='group_id', values='r')
    df_std = df.pivot_table(columns='exp_name', index='group_id', values='std')
    
    if flyid2name:
        df_rate.insert(loc=0, column='name', value=df_rate.index.map(flyid2name).fillna(''))
        df_std.insert(loc=0, column='name', value=df_rate.index.map(flyid2name).fillna(''))

    return df_rate, df_std
    
    
def get_rate_group_raw(df,exp_name, group_cells,group_id,duration, flyid2name=dict(),n_run=30):

    rate, std, flyid = [], [], []
    df = df[np.isin(df.flywire_id.values,group_cells)]
    if len(df)!=0:
        r_per_cell = []
        for f, df_f in df.groupby('flywire_id'):
            r = np.zeros(n_run)
            for _, df_t in df_f.groupby('trial'): # 한 번도 fire하지 않은 경우는 고려되지 않았음
                r[df_t.trial.values[0]] = len(df_t) / duration    # rate 평균을 구할 것이 아니라, 분포를 봐야할 듯 함 
            r_per_cell.append(r)
        mean_in_group = np.mean(r_per_cell,axis=0)
        std_in_group = np.std(r_per_cell,axis=0)

        mean_per_trial = np.mean(mean_in_group) 
        std_per_trial = np.std(mean_in_group)
#         exp_name = df.exp_name.values[0]
    else: 
        mean_in_group = np.array([0 for x in range(n_run)])
        std_in_group = np.array([0 for x in range(n_run)])
        mean_per_trial = 0
        std_per_trial = 0 
        
    return mean_in_group, std_in_group



def get_rate_group_return_raw_data(df,exp_name, group_cells,group_id,duration, flyid2name=dict(),n_run=30):

    rate, std, flyid = [], [], []
    df = df[np.isin(df.flywire_id.values,group_cells)]
    if len(df)!=0:
        r_per_cell = []
        for f, df_f in df.groupby('flywire_id'):
            r = np.zeros(n_run)
            for _, df_t in df_f.groupby('trial'): 
                r[df_t.trial.values[0]] = len(df_t) / duration   
            r_per_cell.append(r)
        mean_in_group = np.mean(r_per_cell,axis=0)
        std_in_group = np.std(r_per_cell,axis=0)

        mean_per_trial = np.mean(mean_in_group) 
        std_per_trial = np.std(mean_in_group)
#         exp_name = df.exp_name.values[0]
    else: 
        mean_in_group = np.zeros(n_run)
        mean_per_trial = 0
        std_per_trial = 0 
        
        

    return mean_in_group