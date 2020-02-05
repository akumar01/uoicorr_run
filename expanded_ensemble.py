import pdb
import os
import numpy as np 
from numpy.random import RandomState
from scipy.stats import wishart
import pickle 
from utils import gen_covariance
import glob

PATH = os.path.split(os.path.realpath(__file__))[0]

def load_covariance(index):

    p = 500
    
    # Load the orignal set of covariance parameters
    with open('%s/unique_cov_param.dat' % PATH, 'rb') as f:
        cov_params = pickle.load(f)

    if index < 80:
        sigma = gen_covariance(500, **cov_params[index])
        cov_param = cov_params[index]
    else:

        # Load the indicies that constrain the random perturbations
        with open('%s/ensemble_expansion_idxs.dat' % PATH, 'rb') as f:
            expansion_idxs = pickle.load(f)

        index = index - 80
        ensemble_index = expansion_idxs[index]

        # How many Wishart matrices to generate
        nreps = 20

        # Tune the strength of perturbation
        n = np.linspace(500, 8000 * 8, 10)

        # Do not include fully dense model
        sparsity = np.logspace(np.log10(0.02), 0, 15)[:-1]

       	# Unravel the index into a 3-tuple
       	cidx, nidx, rep = np.unravel_index(int(ensemble_index), (len(cov_params), n.size, nreps))
        cov_param = cov_params[cidx]
        sigma0 = gen_covariance(500, **cov_param)

        random_state = RandomState(rep)
        sigma = wishart.rvs(df=n[nidx], scale=sigma0, random_state=random_state)
        # Normalize the matrix 
        D = np.diag(np.sqrt(np.diag(sigma)))
        sigma = np.linalg.inv(D) @ sigma @ np.linalg.inv(D)            

    return sigma, cov_param
