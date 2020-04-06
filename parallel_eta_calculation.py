# %load ../../loaders/imports.py
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import time
import pdb
import itertools
from sklearn.preprocessing import StandardScaler
 
# Add the uoicorr directory to the path
sys.path.append('/global/homes/a/akumar25/repos/uoicorr_analysis')

from postprocess_utils import *
import pandas as pd
import sqlalchemy

from mpi4py import MPI

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

    def calc_eta_sa(self, cov_indices, df):
        t0 = time.time()
        
        eta = np.zeros(len(cov_indices))    
        sa = np.zeros(len(cov_indices))
        FNR = np.zeros(len(cov_indices))
        FPR = np.zeros(len(cov_indices))
        sa_perfect = np.zeros(len(cov_indices))
        sa_thresh = np.zeros(len(cov_indices))

        for i, cov_idx in enumerate(cov_indices):
            t0 = time.time()
            sigma, cov_param = self.cov_params[cov_idx]
            df_ = apply_df_filters(df, cov_idx=cov_idx)         
            if df_.shape[0] == 0:
                eta[i] = np.nan
                sa[i] = np.nan
                FNR[i] = np.nan
                FPR[i] = np.nan 
                sa_perfect[i] = np.nan
                sa_thresh[i] = np.nan       
            
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
                    X = StandardScaler().fit_transform(X)
                    # Columns have mean squared one
                    X *= 1/np.linalg.norm(X, axis = 0)
                    sigma_hat = 1/X.shape[0] * X.T @ X
                    eta_[j] = calc_irrep_const(sigma_hat, k)
                    
                eta[i] = np.nanmean(eta_)
            else:
                eta[i] = np.nan

            # Just return the average selection accuracy
            sa[i] = np.mean(df_['sa'].values)
            FNR[i] = np.mean(df_['FNR'].values)
            FPR[i] = np.mean(df_['FPR'].values)

            # TODO: Count the number out of twenty that do perfectly
            sa_perfect[i] = np.count_nonzero(1 * np.isclose(df_['sa'].values, np.ones(df_['sa'].size)))/df_['sa'].size
            sa_thresh[i] = np.count_nonzero(1 * (df_['sa'].values > 0.8))/df_['sa'].size
            # print('Per cov_idx iteration time: %f' % (time.time() - t0))
        return eta, sa, FNR, FPR, sa_perfect, sa_thresh


if __name__ == '__main__':

    comm = MPI.COMM_WORLD

    root_dir = os.environ['SCRATCH'] + '/finalall'
    save_dir = os.environ['SCRATCH'] + '/etascaling'

    # caseno = sys.argv[1]
 
    #if caseno == 4:
    kappa = 10
    np_ratio = 16
    # elif caseno == 3:
    #    kappa = 10
    #    np_ratio = 16

    # Load dataframes and narrow down their scope
    if comm.rank == 0:
    
        selection_methods = ['BIC', 'AIC', 'CV']
        lasso = pd.read_pickle('%s/lasso_concat_df.dat' % root_dir)
        mcp = pd.read_pickle('%s/mcp_concat_df.dat' % root_dir)
        scad = pd.read_pickle('%s/scad_concat_df.dat' % root_dir)
        en = pd.read_pickle('%s/en_concat_df.dat' % root_dir)
        uoi = pd.read_pickle('%s/uoi_concat_df.dat' % root_dir)

        sparsity = np.unique(uoi['sparsity'].values)
        # Only need to do a single betawidth
        selection_methods = np.unique(lasso['selection_method'])
        dframes = [uoi, lasso, mcp, scad, en]

        # Narrow to cases
        dframes = [apply_df_filters(df, kappa=kappa, np_ratio=np_ratio, betawidth=np.inf) for df in dframes]

    else:

        dframes = None
        sparsity = None
        selection_methods = None

    dframes = comm.bcast(dframes)
    sparsity = comm.bcast(sparsity)
    selection_methods = comm.bcast(selection_methods)

    betawidth = 0.1
    dframe_names = ['Lasso', 'MCP', 'SCAD', 'EN', 'UoI Lasso']

    tasklist = list(itertools.product(np.arange(len(dframes)), selection_methods, sparsity))

    # Initialize worker classes
    worker = Worker()

    print(len(tasklist))
    task_chunks = np.array_split(tasklist, comm.size)
    eta_datalist = []
    for j, task in enumerate(task_chunks[comm.rank]):
        t0 = time.time()
        i, sm, s = task 
        i = int(i)
        s = float(s) 
        df = apply_df_filters(dframes[i], selection_method=sm,  sparsity=s)
        cov_indices = np.unique(df['cov_idx'].values)
        cov_indices = cov_indices[cov_indices < 80]
        eta, sa, FNR, FPR, sa_perf, sa_thresh = worker.calc_eta_sa(cov_indices, df)
        eta_datalist.append({'df_name' : dframe_names[i], 'betawidth': 0.1, 'sparsity' : s,
                             'eta': eta, 'sa': sa, 'selection_method': sm, 'cov_indices' : cov_indices,
                             'sa_perf': sa_perf, 'sa_thresh': sa_thresh, 'FNR': FNR, 'FPR': FPR})       
        print('Task %d/%d exec time: %f' % (j + 1, len(task_chunks[comm.rank]), time.time() - t0))
    
    with open('%s/eta_datalist%d.dat' % (save_dir, comm.rank), 'wb') as f:
        f.write(pickle.dumps(eta_datalist))
