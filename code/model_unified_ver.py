import pandas as pd
from textwrap import dedent
import numpy as np 
from math import exp

# brian 2
from brian2 import NeuronGroup, Synapses, PoissonInput, SpikeMonitor, Network, SpikeGeneratorGroup, StateMonitor
from brian2 import mV, ms, Hz, second, volt

# file handling
from pathlib import Path

# parallelization
from joblib import Parallel, delayed, parallel_backend
from time import time

default_params = {
    # trials
    't_run'     : 1000 * ms,              # duration of trial
    'n_run'     : 30,                     # number of runs

    # network constants
    # Kakaria and de Bivort 2017 https://doi.org/10.3389/fnbeh.2017.00008
    'v_0'       : -52 * mV,               # resting potential
    'v_rst'     : -52 * mV,               # reset potential after spike
    'v_th_base' : -45 * mV,               # threshold for spiking
    't_mbr'     :  20 * ms,               # membrane time scale (capacitance * resistance = .002 * uF * 10. * Mohm)

    # Jürgensen et al https://doi.org/10.1088/2634-4386/ac3ba6
    'tau'       : 5 * ms,                 # time constant 

    # Lazar et al https://doi.org/10.7554/eLife.62362
    't_rfc'     : 2.2 * ms,               # refractory period

    # Paul et al 2015 doi: 10.3389/fncel.2015.00029
    't_dly'     : 1.8*ms,                 # delay for changes in post-synaptic neuron

    # Free parameter 
    'w_syn'     : .55 * mV,              # weight per synapse (note: modulated by exponential decay)
    # Default activation rates 
    'r_poi1'     : 150*Hz,                 # default rate of the Poisson inputs
    'r_poi2'    :   0*Hz,                 # default rate of a 2nd class of Poisson inputs
    'r_poi3'    :   0*Hz,                 # default rate of a 3rd class of Poisson inputs
    'r_poi4'    :   0*Hz,                 # default rate of a 4th class of Poisson inputs
    'f_poi'     : 250,                    # scaling factor for Poisson synapse; 250 is sufficient to cause spiking


    # equations for neurons               # alpha synapse https://doi.org/10.1017/CBO9780511815706; See https://brian2.readthedocs.io/en/stable/user/converting_from_integrated_form.html
    'eqs'       : dedent(''' 
                    dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
                    dg/dt = -g / tau               : volt (unless refractory) 
                    rfc                            : second
                    v_th                           : volt
                    '''),
    # condition for spike
    'eq_th'     : 'v > v_th', 
    # rules for spike        
    'eq_rst'    : 'v = v_rst; w = 0; g = 0 * mV', 
}


#######################
# brian2 model setup

def inhomogeneous_poisson(rate, bin_size):
    n_bins = len(rate)
    spikes = np.random.rand(n_bins) < rate * bin_size
    spike_times = np.nonzero(spikes)[0] * bin_size 
    return spike_times

def inhomogeneous_poisson_generator(n_trials, rate, bin_size):
    for i in range(n_trials):
        yield inhomogeneous_poisson(rate, bin_size)

def integrand(x, k):
    return -k * (np.exp(-20*x) - np.exp(-3*x))
def calculate_constant(target_integral,tmin=0,tmax=1):
    import numpy as np
    from scipy.integrate import quad
    initial_k = 10 
    k = initial_k
    tolerance = 1e-5
    max_iterations = 10000
    iteration = 0 
    while True:
        result, _ = quad(integrand, tmin, tmax, args=(k,))
        
        if np.abs(result - target_integral) < tolerance:

            break
        
        # 적분 값이 목표 값보다 크면 k를 줄이고, 작으면 k를 늘립니다.
        if result < target_integral:
            k += 0.01 * initial_k
        else:
            k -= 0.01 * initial_k
        
        iteration += 1
        
        if iteration >= max_iterations:
            # print("최대 반복 횟수를 초과하여 종료합니다.")
            break
    # print(f'target_integral: {result}')
    return k 

def make_input_perdetermined(neu,exc_,r_poi,t_run,spike_times):
    # spike_times = predetermined_rate[r_poi/Hz][:len(exc_)]

    num_spikes_per_cells = [len(x) for x in spike_times]
    indices = np.repeat(list(range(len(exc_))),num_spikes_per_cells)
    times = np.concatenate(spike_times)*second

    times_ = np.sort(times/second)*second
    indices_ = indices[np.argsort(times/second)]

    # purple_i = list(range(len(exc)))

    indices_ = [exc_[x] for x in indices_]

    inp = SpikeGeneratorGroup(len(neu),indices_,times_)
    return inp




def make_input(neu,exc_,r_poi,t_run,method='inhomogeneous'):
    n_trials = len(exc_) 
    tmax = float(t_run/second)
    # max_rate = 120
    bin_size = 0.0023
    time = np.arange(0,tmax,bin_size)
    k = calculate_constant(int(r_poi/Hz),tmax=tmax)
    if method == 'inhomogeneous':
        rate = np.array([-k*(exp(-20*t)-exp(-3*t)) for t in time]) # tmax should be fixed -> exp. duration 
    elif method == 'homogeneous':
        rate = np.full_like(time,r_poi/Hz)
    spike_times = list(inhomogeneous_poisson_generator(n_trials, rate, bin_size))
   
    num_spikes_per_cells = [len(x) for x in spike_times]
    indices = np.repeat(list(range(len(exc_))),num_spikes_per_cells)
    times = np.concatenate(spike_times)*second

    times_ = np.sort(times/second)*second
    indices_ = indices[np.argsort(times/second)]

    # purple_i = list(range(len(exc)))

    indices_ = [exc_[x] for x in indices_]
    
    inp = SpikeGeneratorGroup(len(neu),indices_,times_)

    return inp,spike_times





def silence(slnc, syn):
    '''Silence neuron by setting weights of all synapses from it to 0

    Parameters
    ----------
    slnc : list
        List of neuron indices to silence
    syn : brian2.Synapses
        Defined synapses object

    Returns
    -------
    syn : brian2.Synapses
        Synapses with modified weights
    '''

    for k in slnc:
        syn.w[' {} == i'.format(k)] = 0*mV
    
    return syn

def create_model(path_comp, path_con, params,exc_all,Custom_threshold_Ids=None,Custom_threshold_th=None, Except_output_Ids = None,
                 Except_output_w_syn = None,
                 Except_input_Ids = None,
                 Except_input_w_syn = None,
                 predetermined_or_not=None,
                 spike_times = None):

    # load neuron connectivity dataframes
    df_comp = pd.read_csv(path_comp, index_col=0)
    df_con = pd.read_parquet(path_con)


    if Custom_threshold_Ids is None:
        neu = NeuronGroup( # create neurons
            N=len(df_comp),
            model=params['eqs'],
            method='linear',
            threshold=params['eq_th'],
            reset=params['eq_rst'],
            refractory='rfc',
            name='default_neurons',
            namespace=params,
        )
        neu.v = params['v_0'] # initialize values
        neu.g = 0
        neu.rfc = params['t_rfc']
        neu.v_th=params['v_th_base']    
    else:
        neu = NeuronGroup( # create neurons
            N=len(df_comp),
            model=params['eqs'],
            method='linear',
            threshold=params['eq_th'],
            reset=params['eq_rst'],
            refractory='rfc',
            name='default_neurons',
            namespace=params,
        )
        neu.v = params['v_0'] # initialize values
        neu.g = 0
        neu.rfc = params['t_rfc']
        neu.v_th[~df_comp.index.isin(Custom_threshold_Ids)]=params['v_th_base']
        for ci in enumerate(Custom_threshold_Ids):
            neu.v_th[df_comp.index.isin([ci[1]])]=Custom_threshold_th[ci[0]]
            # print(neu.v_th[df_comp.index.isin([ci[1]])])

    t_run = params['t_run']

    
    if predetermined_or_not == None:
        inps = []
        feedforwards = []
        spike_times_per_exc = {}
        for e in enumerate(exc_all):
            i = e[0]+1 
            exc = e[1]
            inp,spike_times = make_input(neu,exc,params[f'r_poi{i}'],t_run)
            spike_times_per_exc[i] = spike_times
            inps.append(inp)
            # create feedforward synapses
            feedforward = Synapses(inp,neu,'w : volt',on_pre='g+=w',delay=0*ms,name=f'input_synapses{i}')
            feedforward.connect(i=exc,j=exc)
            feedforward.w = [6000*params['w_syn'] for x in range(len(exc))]
            feedforwards.append(feedforward)
    else:
        inps = []
        feedforwards = []
        spike_times_per_exc = {}
        for i,exc in enumerate(exc_all):
            if predetermined_or_not[i]==0:
                inp,spike_times = make_input(neu,exc,params[f'r_poi{i+1}'],t_run)
                spike_times_per_exc[i+1] = spike_times
            else:
                inp = make_input_perdetermined(neu,exc,params[f'r_poi{i+1}'],t_run,spike_times[i+1])
                spike_times_per_exc[i+1] = spike_times
            inps.append(inp)
            # create feedforward synapses
            feedforward = Synapses(inp,neu,'w : volt',on_pre='g+=w',delay=0*ms,name=f'input_synapses{i}')
            feedforward.connect(i=exc,j=exc)
            feedforward.w = [6000*params['w_syn'] for x in range(len(exc))]
            feedforwards.append(feedforward)



    # create synapses
    syn = Synapses(neu, neu, 'w : volt', on_pre='g += w', delay=params['t_dly'], name='default_synapses')

    # connect synapses
    i_pre = df_con.loc[:, 'Presynaptic_Index'].values
    i_post = df_con.loc[:, 'Postsynaptic_Index'].values
    syn.connect(i=i_pre, j=i_post)

    df_con_2 = df_con.reset_index(drop=True, inplace=False)
    base_w = df_con_2['Excitatory x Connectivity'].values * params['w_syn']
    Syn_array = base_w.copy()

    if Except_output_Ids is not None and len(Except_output_Ids) > 0:
        for ids, w_ex in zip(Except_output_Ids, Except_output_w_syn):
            if ids is None or len(ids) == 0:
                continue
            mask = df_con_2['Presynaptic_ID'].isin(ids)
            Syn_array[mask] = df_con_2.loc[mask, 'Excitatory x Connectivity'].values * w_ex
            
    if Except_input_Ids is not None and len(Except_input_Ids) > 0:
        for ids, w_ex in zip(Except_input_Ids, Except_input_w_syn):
            if ids is None or len(ids) == 0:
                continue
            mask = df_con_2['Postsynaptic_ID'].isin(ids)
            Syn_array[mask] = df_con_2.loc[mask, 'Excitatory x Connectivity'].values * w_ex

    # 4) Brian2에 할당
    syn.w = Syn_array

    # object to record spikes
    spk_mon = SpikeMonitor(neu) 
    # M = StateMonitor(neu,'v',record=True)
 
    return neu, syn, spk_mon, inps, feedforwards,spike_times_per_exc

#####################
# running simulations
def get_spk_trn(spk_mon):
    '''Extracts spike times from 'spk_mon'

    The spike times recorded in the SpikeMonitor object during 
    simulation are converted to a list of times for each neurons.
    Returns dict with "brian ID": "list of spike times".

    Parameters
    ----------
    spk_mon : SpikeMonitor
        Contains recorded spike times

    Returns
    -------
    spk_trn : dict
        Mapping between brian neuron IDs and spike times
    '''

    spk_trn = {k: v for k, v in spk_mon.spike_trains().items() if len(v)}
    
    return spk_trn

def construct_dataframe(res, exp_name, i2flyid):
    '''Take spike time dict and collects spikes in pandas dataframe

    Parameters
    ----------
    res : list
        List with spike time dicts for each trial
    exp_name : str
        Name of the experiment
    i2flyid : dict
        Mapping between Brian IDs and flywire IDs

    Returns
    -------
    df : pandas.DataFrame
        Dataframe where each row is one spike
    '''
    
    ids, ts, nrun = [], [], []

    for n, i in enumerate(res):
        for j, k  in i.items():
            ids.extend([j for _ in k])
            nrun.extend([n for _ in k])
            ts.extend([float(l) for l in k])

    d = {
        't': ts,
        'trial': nrun,
        'flywire_id': [i2flyid[i] for i in ids],
        'exp_name': exp_name
    }
    df = pd.DataFrame(d)

    return df

def run_trial(exc_all, slnc, path_comp, path_con, params, Custom_threshold_Ids=None,
              Custom_threshold_th=None,
                Except_output_Ids = None,
                Except_output_w_syn = None,
                Except_input_Ids = None,
                Except_input_w_syn = None,
                predetermined_or_not = None,
                spike_times = None):


    # get default network
    neu, syn, spk_mon,inps,feedforwards,spike_times_per_exc = create_model(path_comp, path_con, params,exc_all, Custom_threshold_Ids, Custom_threshold_th,
                Except_output_Ids,
                Except_output_w_syn,
                Except_input_Ids,
                Except_input_w_syn,
                predetermined_or_not,
                spike_times)

    # silence neurons
    syn = silence(slnc, syn)

    net = Network(neu,syn,spk_mon,*inps,*feedforwards)

    # run simulation
    net.run(duration=params['t_run'])

    # spike times 
    spk_trn = get_spk_trn(spk_mon)


    # v = np.array(M.v / mV)     # shape: (n_recorded, n_time)
    # t = np.array(M.t / ms)


    return spk_trn,spike_times_per_exc


def run_exp(exp_name, neu_exc, path_res, path_comp, path_con,
            params=default_params, neu_slnc=[],neu_exc_add=[], 
            n_proc=-1, force_overwrite=False, Custom_threshold_Ids=None,
            Custom_threshold_th=None, Except_output_Ids = None,
            Except_output_w_syn = None,
            Except_input_Ids = None,
            Except_input_w_syn = None,
            predetermined_input_or_not = None,
            spike_path=None):

    # convert to Path objects
    path_res, path_comp, path_con = [ Path(i) for i in [path_res, path_comp, path_con] ]

    # define output files
    path_save = path_res / '{}.parquet'.format(exp_name)
    if path_save.is_file() and not force_overwrite:
        print('>>> Skipping experiment {} because {} exists and force_overwrite = {}'.format(exp_name, path_save, force_overwrite))
        return 

    # load name/id mappings
    df_comp = pd.read_csv(path_comp, index_col=0) # load completeness dataframe run_trial
    flyid2i = {j: i for i, j in enumerate(df_comp.index)}  # flywire id: biran ID
    i2flyid = {j: i for i, j in flyid2i.items()} # brian ID: flywire ID
    
    # print info
    print('>>> Experiment:     {}'.format(exp_name))
    print('    Output file:    {}'.format(path_save))
    num_neu_exc_add = sum([len(x) for x in neu_exc_add])
    print('    Excited neurons: {}'.format(len(neu_exc)+num_neu_exc_add))
    if neu_slnc:
        print('    Silenced neurons: {}'.format(len(neu_slnc)))


    # read predetermined spike patterns
    import pickle
    if spike_path != None:
        with open(spike_path,'rb') as f:
            spike_times_all = pickle.load(f)
    else:
        spike_times_all = [None for x in range(params['n_run'])]     
    
    # start parallel calculation
    n_run = params['n_run']
    start = time() 
    exc_all = [[flyid2i[n] for n in neu_exc]]
    with parallel_backend('loky', n_jobs=n_proc):
        for neu in neu_exc_add:
            exc_all.append([flyid2i[n] for n in neu])
        slnc = [ flyid2i[n] for n in neu_slnc ]
        res = Parallel()(
            delayed(
                run_trial)(exc_all, slnc, path_comp, path_con, params, Custom_threshold_Ids, Custom_threshold_th, Except_output_Ids, Except_output_w_syn,Except_input_Ids, Except_input_w_syn,predetermined_input_or_not,spike_times_all[iii]) for iii in range(n_run))
    

    spike_times_per_exc_all_trial = [x[1] for x in res]
    # mp_monitor = [x[2] for x in res]
    res = [x[0] for x in res]
    # print simulation time
    walltime = time() - start 
    print('    Elapsed time:   {} s'.format(int(walltime)))

    # dataframe with spike times
    df = construct_dataframe(res, exp_name, i2flyid)

    # store spike data
    df.to_parquet(path_save, compression='brotli')

    # store spike pattern of input neurons 
    import os 
    import pickle 
    if 'input_spike_pattern' not in os.listdir(path_res):
        os.mkdir(f'{path_res}/input_spike_pattern')
    with open(f'{path_res}/input_spike_pattern/{exp_name}.pkl','wb') as f:
        pickle.dump(spike_times_per_exc_all_trial,f)
