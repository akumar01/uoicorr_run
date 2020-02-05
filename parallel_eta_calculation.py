# %load ../../loaders/imports.py
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import time
import pdb
import itertools 
# Add the uoicorr directory to the path
sys.path.append('/global/homes/a/akumar25/repos/uoicorr_analysis')

from postprocess_utils import *
import pandas as pd
import sqlalchemy

from utils import gen_data, gen_covariance, sparsify_beta, gen_beta2
from expanded_ensemble import load_covariance
from plotting_utils import *
import h5py
import pickle

class Worker():

    def __init__(self):

        # On initialization, load all the covariance parameters and matrices to 
        # save time
        self.cov_params = []
        for cov_idx in np.arange(120):
            sigma, cp = load_covariance(cov_idx)
            self.cov_params.append((sigma, cp))

    def calc_eta_sa(self, cov_indices, df, flag, threshold=1):
        t0 = time.time()
        eta = np.zeros(len(cov_indices))    
        sa = np.zeros(len(cov_indices))
        for i, cov_idx in enumerate(cov_indices):
            sigma, cov_param = self.cov_params[cov_idx]
            df_ = apply_df_filters(df, cov_idx=cov_idx)                
            # take the minimum non-zero beta value - magnitudes don't matter
            beta = np.ones(df_.iloc[0]['n_features'])               
            # Sparsify beta
            beta = sparsify_beta(beta[:, np.newaxis], cov_param['block_size'], df_.iloc[0]['sparsity'],
                                 seed = cov_param['block_size'])
            beta = beta.ravel()
            k = np.nonzero(beta)[0]
            if len(k) > 0 and len(k) < beta.size:                    
                # For each seed in df_, generate data accordingly and use the empirical covariance to calculate
                # eta
                eta_ = np.zeros(df_.shape[0])
                for j in range(df_.shape[0]):
                    _df_ = df_.iloc[j]
                    np.random.seed(int(_df_['seed']))
                    X = np.random.multivariate_normal(mean = np.zeros(_df_['n_features']), cov=sigma, 
                                                      size=_df_['np_ratio'] * _df_['n_features'])
                    sigma_hat = X.T @ X
                    eta_[j] = calc_irrep_const(sigma_hat, k)
                    
                eta[i] = np.nanmean(eta_)
            else:
                eta[i] = np.nan

            # Just return the average selection accuracy
            if flag is None:
                sa[i] = np.mean(df_['sa'].values)

            if flag == 'threshold':
                sa[i] = np.count_nonzero(1 * df_['sa'].values > threshold)/len(cov_indices[i])
        
        return eta, sa


if __name__ == '__main__':

comm = MPI.COMM_WORLD

# Load dataframes and narrow down their scope
if comm.rank == 0:

    selection_methods = ['BIC', 'AIC', 'CV']
    lasso = pd.read_pickle('%s/finalfinal/lasso_concat_df.dat' % root_dir)
    mcp = pd.read_pickle('%s/finalfinal/mcp_concat_df.dat' % root_dir)
    scad = pd.read_pickle('%s/finalfinal/scad_concat_df.dat' % root_dir)
    en = pd.read_pickle('%s/finalfinal/en_concat_df.dat' % root_dir)

    # Narrow down the size
    lasso = apply_df_filters(lasso, kappa=kappa, np_ratio=np_ratio, betawidth=np.inf)
    mcp = apply_df_filters(mcp, kappa=kappa, np_ratio=np_ratio, betawidth=np.inf)    
    scad = apply_df_filters(scad, kappa=kappa, np_ratio=np_ratio, betawidth=np.inf)    
    en = apply_df_filters(en, kappa=kappa, np_ratio=np_ratio, betawidth=np.inf)

    lasso = lasso[lasso['selection_method'].isin(selection_methods)]
    mcp = mcp[mcp['selection_method'].isin(selection_methods)]
    scad = scad[scad['selection_method'].isin(selection_methods)]
    en = en[en['selection_method'].isin(selection_methods)]

    sparsity = np.unique(lasso['sparsity'].values)

    # Sort dataframes hierarchally
    lasso.sort_values(inplace=True, by=['selection_method', 'sparsity', 'cov_idx'])
    mcp.sort_values(inplace=True, by=['selection_method', 'sparsity', 'cov_idx'])
    scad.sort_values(inplace=True, by=['selection_method', 'sparsity', 'cov_idx'])
    en.sort_values(inplace=True, by=['selection_method', 'sparsity', 'cov_idx'])
    dframes = [lasso, mcp, scad, en]

    assert(np.all([df.shape[0] % 20 == 0 for df in dframes]))

else:

    dframes = None
    sparsity = None

dframes = comm.bcast(dframes)
sparsity = comm.bcast(sparsity)
dframe_names = ['Lasso', 'MCP', 'SCAD', 'EN']

tasklist = []
for i, df in enumerate(dframes):
    for j in np.arange(int(dframes[0].shape[0]/20)):
        tasklist.append((i, j))

# Initialize worker classes
worker = Worker()

task_chunks = np.array_split(tasklist, comm.size)
eta_datalist = []
for task in task_chunks[comm.rank]:
    t0 = time.time()
    i, j = task
    df = dframes[i].iloc[20*j:20*(j + 1)]
    cov_indices = np.unique(df['cov_idx'].values)
    assert(cov_indices.size == 1)

    eta, sa = worker.calc_eta_sa(cov_indices, df, flag=None)
    bw = df.iloc[0]['betawidth']
    sparsity = df.iloc[0]['sparsity']
    sm = df.iloc[0]['selection_method']
    eta_datalist.append({'df_name' : dframe_names[i], 'betawidth': bw, 'sparsity' : s,
                       'eta': eta_, 'sa': sa_, 'selection_method': sm})       

    print('Task exec time: %f' % (time.time() - t0))