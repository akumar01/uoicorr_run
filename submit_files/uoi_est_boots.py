import numpy as np
import itertools
import pdb
from misc import get_cov_list
from utils import gen_beta2

script_dir = '/global/homes/a/akumar25/repos/uoicorr_run'

###### Master list of parameters to be iterated over #######

exp_types =  ['UoILasso']

# Estimated worst case run-time for a single repitition for each algorithm in exp_types 
algorithm_times = ['2:00:00', '00:30:00', '00:30:00']

n_features = 100

# Block sizes
block_sizes = [5, 10, 20]

# Block correlation
correlation = [0, 0.08891397, 0.15811388, 0.28117066, 0.5]

# Exponential length scales
L = [2, 5, 10, 20]

cov_list, _ = get_cov_list(n_features, 60, correlation, block_sizes, L, n_supplement = 20)

cov_params = [{'correlation' : t[0], 'block_size' : t[1], 'L' : t[2], 't': t[3]} for t in cov_list]

cov_params = cov_params[::4]

sparsity = np.array([0.25, 0.5])

iter_params = {
    'cov_params' : cov_params,
}

#############################################################

##### DO NOT CHANGE THIS UNLESS YOU ARE SURE!!!! ############
betaseed = 1234

# For each desired betawidth, we generate a fixed beta vector
# that is held fixed across all paramter combinations. For a fixed
# sparsity, the exact imposed sparsity profile may vary depending 
# on the block size of the covariance matrix. However, we use the
# blocks as a seed for the shuffling that is done, so that for a fixed
# sparsity and block size, all beta vectors should be identical

betawidth = [-1, np.inf]

beta_dict = []
for i, bw in enumerate(betawidth):
    # NOTE THE DISTRIBUTION
	beta_dict.append({'betawidth' : bw, 'beta': gen_beta2(n_features, n_features, 1, bw, seed = betaseed, distribution='normal')})

##### Common parameters held fixed across all jobs ##########
comm_params = {
'sparsity': sparsity,
'cov_type' : 'interpolation',
'n_features' : n_features,
# n/p ratio #
'np_ratio': [4],
'est_score' : 'BIC',
'reps' : 20,
'stability_selection' : [0.75],
'n_boots_sel': 25,
'n_boots_est' : [5, 10, 25, 50, 100],
'betadict' : beta_dict,
# Inverse Signal to noise ratio
'kappa' : [5],
'estimation_frac' : np.linspace(0.5, 0.9, 4),
'sub_iter_params': ['betadict', 'sparsity', 'kappa', 'np_ratio', 'n_boots_est', 'estimation_frac'],
}

# Parameters for ElasticNet
comm_params['l1_ratios'] = [0.1,  0.5, 0.75, 0.9, 0.95]
comm_params['n_alphas'] = 100

# Parameters for SCAD/MCP
comm_params['gamma'] = [3]

###############################################################

# Which selection methods should we apply to the algorithms?
comm_params['selection_methods'] = ['BIC', 'AIC', 'CV', 'gMDL', 'empirical_bayes', 'oracle']
# Which fields should we record for each selection method? 
comm_params['fields'] = {'BIC' : ['beta', 'beta_hats', 'FNR', 'FPR', 'sa', 'ee', 'r2', 'MSE', 'reg_param', 'bias'], 
						 'AIC' : ['beta', 'beta_hats', 'FNR', 'FPR', 'sa', 'ee', 'r2', 'MSE', 'reg_param', 'bias'], 
						 'CV' : ['beta', 'beta_hats', 'FNR', 'FPR', 'sa', 'ee', 'r2', 'MSE', 'reg_param', 'bias'],
                         'gMDL' : ['beta', 'beta_hats', 'FNR', 'FPR', 'sa', 'ee', 'r2', 'MSE', 'reg_param', 'bias'],
                         'empirical_bayes' : ['beta', 'beta_hats', 'FNR', 'FPR', 'sa', 'ee', 'r2', 'MSE', 'reg_param', 'bias'
                                              ], 
                         'oracle' : ['beta', 'beta_hats', 'FNR', 'FPR', 'sa', 'ee', 'r2', 'MSE', 'reg_param', 'bias']}

