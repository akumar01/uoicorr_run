# %load ../../loaders/imports.py
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import time
import pdb
import pandas as pd
import sqlalchemy
from mpi4py import MPI
from schwimmbad import MPIPool

from utils import gen_data, gen_covariance, sparsify_beta, gen_beta2

sys.path.append('/global/homes/a/akumar25/repos/uoicorr_analysis')

from postprocess_utils import *

class Agrregator():

    def __init__(self):

        self.result_list = []

    def __call__(self, result):

        self.result_list.append(result)

    def save(self):
        result_df = pd.DataFrame(self.result_list)
        with open('bias_var_df.dat', 'wb') as f:
            f.write(pickle.dumps(result_df))

def calc_bias_var(task_tuple):

    df, df_name, beta_file = task_tuple
    
    indices = list(df.index)
    # Take the indices
    beta = beta_file['beta'][indices, :]
    # Ensure all the betas are the same
    assert(np.isclose(beta, beta[0]).all())
    
    beta_hats = beta_file['beta_hat'][indices, :]
    
    # Total bias
    total_bias = np.mean(np.linalg.norm(beta - beta_hats, axis=1))
    
    common_support_bias = 0
    
    # Common support bias
    for i in range(len(indices)):
        common_support = list(set(np.nonzero(beta[i, :])[0]).intersection(set(np.nonzero(beta_hats[i, :])[0])))
        common_support_bias += 1/len(indices) * np.linalg.norm(beta[i, common_support] - beta_hats[i, common_support])
    variance = np.mean(np.var(beta_hats, axis = 0))

    s = df['sparsity']
    bw = df['betawidth']
    sm = df['selection_method']
    cidx = df['cov_idx']
    kappa = df['kappa']
    np_ratio = df['np_ratio']

    # Return the result as a properly formatted dictionary
    bias_variance_results = {'df_name': df_name, 'sparsity': s, 'betawidth' : bw,
                              'selection_method' : sm, 'cov_idx': cidx, 'kappa': kappa,
                              'np_ratio': np_ratio, 'total_bias': total_bias, 'common_bias': common_support_bias,
                              'variance': variance}

    return bias_variance_results


if __name__ == '__main__':

    root_dir = 

    comm = MPI.COMM_WORLD
    pool = MPIPool(comm)

    if comm.rank == 0:
        # Read the non-concatenated dataframes to ensure indices are properly preserved
        lasso = pd.read_pickle('%s/finalfinal/lasso_df.dat' % root_dir)
        mcp = pd.read_pickle('%s/finalfinal/mcp_df.dat' % root_dir)
        scad = pd.read_pickle('%s/finalfinal/scad_df.dat' % root_dir)
        en = pd.read_pickle('%s/finalfinal/en_df.dat' % root_dir)

        # Remove the parasitic index field
        lasso = lasso.drop('index', axis=1)
        mcp = mcp.drop('index', axis=1)
        scad = scad.drop('index', axis=1)
        en = en.drop('index', axis=1)

        # Replace with a robust index
        lasso.set_index(np.arange(lasso.shape[0]), inplace=True)
        mcp.set_index(np.arange(mcp.shape[0]), inplace=True)
        scad.set_index(np.arange(scad.shape[0]), inplace=True)
        en.set_index(np.arange(en.shape[0]), inplace=True)

        dframes = [lasso, mcp, scad, en]
        dframe_names = ['Lasso', 'MCP', 'SCAD', 'EN']
        sparsity = np.unique(lasso['sparsity'].values)
        betawidth = np.unique(lasso['betawidth'].values)
        selection_methods = np.unique(lasso['selection_method'].values)
        kappa = 5
        np_ratio = 4
        cov_idxs = np.arange(80)

        beta_fnames = ['%s/finalfinal/%s_pp_beta.h5' % (root_dir, dfname) for dfname in ['lasso', 'mcp', 'scad', 'en']]
        beta_files = [h5py.File(beta_fname, 'r') for beta_fname in beta_fnames]

        param_combos = list(itertools.product(sparsity, betawidth, selection_methods, cov_idxs))

        # Arrange tasks from param combos 
        task_list = []
        for i, dframe in enumerate(dframes):
            for param_comb in param_combos:
                s, bw, sm, cidx = param_combo
                df = apply_df_filters(dframe, sparsity=s, betawidth=bw, 
                                          selection_method=sm, cov_idx=cidx, kappa=kappa, np_ratio=np_ratio)
                if df.shape[0] == 0:
                    continue
                else:
                    assert(df.shape[0] == 20)
                task_list.append((df, dframe_names[i], beta_files[i]))
    else:

        task_list = None

    aggregator = Aggregator()
    pool.map(task_list, calc_bias_var, callback=aggregator)
    pool.close()