import pdb
import numpy as np 
from numpy.random import RandomState
from scipy.stats import wishart
import pickle 
from utils import gen_covariance
import glob

def gen_from_expanded_ensemble(index, random=True):

    p = 500
    
    # Tune the strength of perturbation
    n = np.linspace(500, 8000 * 8, 10)

    # Load the covariance parameters
    with open('unique_cov_param.dat', 'rb') as f:
        cov_params = pickle.load(f)

    # How many Wishart matrices to generate
    nreps = 20

    # Do not include fully dense model
    sparsity = np.logspace(np.log10(0.02), 0, 15)[:-1]

    if random:
       	# Unravel the index into a 3-tuple
       	cidx, nidx, rep = np.unravel_index(int(index), (len(cov_params), n.size, nreps))

        sigma0 = gen_covariance(500, **cov_params[cidx])

        random_state = RandomState(rep)
        sigma = wishart.rvs(df=n[nidx], scale=sigma0, random_state=random_state)

        # Normalize the matrix 
        D = np.sqrt(np.diag(np.diag(sigma)))
        sigma = np.linalg.inv(D) @ sigma @ np.linalg.inv(D)            

    else:

        sigma = gen_covariance(500, **cov_params[index])

    return sigma