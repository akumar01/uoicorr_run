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

root_dir = os.environ['SCRATCH']

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

# Save the task list away
with open('bias_var_tasklist.dat', 'wb') as f:
    f.write(pickle.dumps(task_list))