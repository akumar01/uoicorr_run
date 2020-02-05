# %load ../../loaders/imports.py
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import time
import pdb
import pandas as pd
import sqlalchemy
import h5py

from mpi4py import MPI
from schwimmbad import MPIPool

from utils import gen_data, gen_covariance, sparsify_beta, gen_beta2

sys.path.append('/global/homes/a/akumar25/repos/uoicorr_analysis')

from postprocess_utils import *

class Aggregator():

    def __init__(self, beta_fnames):
        self.result_list = []
        self.task_counter = 0

        self.beta_files = [h5py.File(beta_fname, 'r') for beta_fname in beta_fnames]

    def __call__(self, result):

        self.task_counter += 1
        print('Task counter: %d' % self.task_counter)
        self.result_list.append(result)

    def map(self, task):
        # Replace the index with a sliced dataframe
        group_idx, dframe_name, beta_idx = task
        dframe = dframes[dframe_names.index(dframe_name)]
        df = dframe.iloc[20 * group_idx:20 * (group_idx + 1)]
        assert(np.unique(df['cov_idx'].values).size == 1)
        df_dict = df.to_dict(orient='index')
        return df_dict, dframe_name, beta_idx

    def save(self):
        result_df = pd.DataFrame(self.result_list)
        with open('bias_var_df.dat', 'wb') as f:
            f.write(pickle.dumps(result_df))

    def calc_bias_var(self, task_tuple):
        print('Started task!')
        t0 = time.time()
        df, df_name, beta_idx = task_tuple
        beta_files = self.beta_files[beta_idx]
        pritn('Hooray!')
        print(indices)
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

        print(time.time() - t0)

        return bias_variance_results


if __name__ == '__main__':

    root_dir = os.environ['SCRATCH'] + '/finalall'

    comm = MPI.COMM_WORLD
    pool = MPIPool(comm)

    if comm.rank == 0:
        # Read the non-concatenated dataframes to ensure indices are properly preserved
        lasso = pd.read_pickle('%s/lasso_df.dat' % root_dir)
        print('read lasso') 
        mcp = pd.read_pickle('%s/mcp_df.dat' % root_dir)
        print('read mcp')
        scad = pd.read_pickle('%s/scad_df.dat' % root_dir)
        print('read scad')
        en = pd.read_pickle('%s/en_df.dat' % root_dir)
        print('read en')

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

        beta_fnames = ['%s/%s_pp_beta.h5' % (root_dir, dfname) for dfname in ['lasso', 'mcp', 'scad', 'en']]
        # Arrange tasks from param combos 
        task_list = []
        print('Arranging task list!')
        for i, dframe in enumerate(dframes):
            dframe = apply_df_filters(dframe, kappa=kappa, np_ratio=np_ratio)
            dframe.sort_values(inplace=True, by=['selection_method', 'betawidth', 'sparsity', 'cov_idx'])
            assert(dframe.shape[0] % 20 == 0)
            for j in np.arange(dframe.shape[0]/20).astype(int):
                task_list.append((j, dframe_names[i], i))
    else:
        task_list = None

    aggregator = Aggregator(beta_fnames=beta_fnames)
    pool.map(aggregator.calc_bias_var, task_list, callback=aggregator, map_fn=aggregator.map)
    pool.close()

