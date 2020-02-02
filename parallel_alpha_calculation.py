# %load ../../loaders/imports.py
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import time
import pdb
import pandas as pd
import sqlalchemy
import pickle

from mpi4py.MPI import MPI_COMM_WORLD
from schwimmbad import MPIPool

from postprocess_utils import *
from utils import gen_data, gen_covariance, sparsify_beta, gen_beta2

class Agrregator():

    def __init__(self):

        self.result_list = []

    def __call__(self, result):

        self.result_list.append(result)

    def save(self, name):

        result_df = pd.DataFrame(self.result_list)
        with open(name, 'wb') as f:
            f.write(pickle.dumps(result_df))

def do_task(task):
    
    df, selection_method, betawidth, sparsity = task

    cov_indices = np.unique(df['cov_idx'].values)

    alphas, sa = calc_alpha_sa(cov_indices, df)

    result_dict = {'alphas' : alphas, 'sa' : sa, 'selection_method' : selection_method,
                   'betawidth': betawidth, 'sparsity' : sparsity}

    return result_dict

# Eigenvalue constant
def calc_alpha_sa(cov_indices, df, flag, threshold=1):
    t0 = time.time()
    alphas = np.zeros(len(cov_indices))    
    sa = np.zeros((len(cov_indices)))
    for i, cov_idx in enumerate(cov_indices):

        df_ = apply_df_filters(df, cov_idx=cov_idx)

        # Calculate the alpha associated with this unique combination of model parameters
        sigma, cov_param = load_covariance(cov_idx)
        rho = 1/bound_eigenvalue(np.linalg.inv(sigma), int(df_.iloc[0]['sparsity'] * df_.iloc[0]['n_features']))            

        # take the minimum non-zero beta value
        beta = gen_beta2(df_.iloc[0]['n_features'], df_.iloc[0]['n_features'], 
                         1, df_.iloc[0]['betawidth'], seed=1234, distribution='normal')        
        
        # Sparsify beta
        beta = sparsify_beta(beta[np.newaxis, :], cov_param['block_size'], df_.iloc[0]['sparsity'],
                             seed = cov_param['block_size'])

        beta=beta.ravel()
        
        # generate the data under the assumption that the SNR will be very similar across reps
        _, _, _, _, ss = gen_data(df_.iloc[0]['n_samples'], df_.iloc[0]['n_features'],
                                  df_.iloc[0]['kappa'], sigma, beta, 
                                  seed=df_.iloc[0]['seed'])

        alphas[i] = np.mean(rho * np.min(np.abs(beta[np.nonzero(beta)[0]]))/ss)

        # Just return the average selection accuracy
        if flag is None:
            sa[i] = np.mean(df_['sa'].values)

        if flag == 'threshold':
            sa[i] = np.count_nonzero(1 * df_.iloc[cov_indices[i]]['sa'].values > threshold)/len(cov_indices[i])
           
    return alphas, sa


if __name__ == '__main__':

    # Plot either the average selection accuracy or the percent of runs that exceed a certain threshold as a 
    # function of the parameter alpha. alpha = rho(Omega) sum(beta_min^2)/sigma^2
    comm = MPI_COMM_WORLD

    # Calculate across signal to noise ratios, n/p ratio = 4
    np_ratio = 4
    kappa = 5
    betawidth = np.unique(lasso['betawidth'].values)
    selection_method = 'CV'

    if comm.rank == 0: 

        root_dir = os.environ['CFS']
        lasso = pd.read_pickle('%s/finalfinal/lasso_concat_df.dat' % root_dir)
        mcp = pd.read_pickle('%s/finalfinal/mcp_concat_df.dat' % root_dir)
        scad = pd.read_pickle('%s/finalfinal/scad_concat_df.dat' % root_dir)
        en = pd.read_pickle('%s/finalfinal/en_concat_df.dat' % root_dir)

        dframes = [lasso, mcp, scad, en]
        dframe_names = ['Lasso', 'MCP', 'SCAD', 'EN']
        param_combos = itertools.product(betawidth, sparsity)
        tasks = []
        for i, dframe in enumerate(dframes): 
            for param_combo in param_combs:
                df_ = apply_df_filters(dframe, kappa=kappa, np_ratio=np_ratio,
                                       selection_method=selection_method, betawidth=param_combo[0],
                                       sparsity = param_combo[1])
                tasks.append((dframe, dframe_names[i], selection_method, 
                              param_combo[0], param_combo[1]))
    else:

        tasks = None

    aggregator = Aggregator()
    pool = MPIPool(comm)
    pool.map(do_task, tasks, callback=aggregator)
    ## SAVE TO PICKLE ##
    aggregator.save()