# %load ../../loaders/imports.py
import sys, os
import numpy as np
import time
import pdb
from utils import sparsify_beta
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from pyuoi import UoI_Lasso
from pyc_based.pycasso_cv import PycassoCV
from utils import gen_covariance, gen_beta2, gen_data, calc_avg_cov

from mpi4py import MPI
from schwimmbad import MPIPool
import itertools

class Worker(object):

    def __init__(self, n_features, n_samples, cov_params):


        self.n_features = n_features
        self.n_samples = n_samples
        self.cov_params = cov_params

    def __call__(self, cov_param_idx, rep, algorithm):

        n_features = self.n_features
        n_samples = self.n_samples

        beta = gen_beta2(n_features, n_features, 
                         sparsity = 1, betawidth = -1, seed=1234)    

        cov_param = self.cov_params[cov_param_idx]

        sigma = gen_covariance(n_features, cov_param['correlation'],
                               cov_param['block_size'], cov_param['L'],
                               cov_param['t'])

        beta_ = sparsify_beta(beta, cov_param['block_size'], sparsity = 0.25,
                              seed = cov_param['block_size'])
        
        # Follow the procedure of generating beta with a fixed betaseed at the getgo
        # and then sparsifying as one goes on. Is this reproducible subsequently?
            
        t0 = time.time()
        X, X_test, y, y_test, ss = gen_data(n_samples, n_features, kappa = 5, 
                                        covariance = sigma, beta = beta_)
    
        # Standardize
        X = StandardScaler().fit_transform(X)
        y -= np.mean(y)

        if algorithm == 0:
            lasso = LassoCV(fit_intercept=False, cv=5)
            lasso.fit(X, y.ravel())
            beta_hat = lasso.coef_

        elif algorithm == 1:

            uoi = UoI_Lasso(fit_intercept=False, estimation_score='r2')
            uoi.fit(X, y)
            beta_hat = uoi.coef_        

        elif algorithm == 2:        
            scad = PycassoCV(penalty='scad', fit_intercept=False, nfolds=5, 
                             n_alphas=100)
            scad.fit(X, y)
            beta_hat = scad.coef_
        
        elif algorithm == 3:

            mcp = PycassoCV(penalty='mcp', fit_intercept=False, nfolds=5,
                            n_alphas=100)

            mcp.fit(X, y)
            beta_hat = mcp.coef_        

        self.beta.extend(beta_)
        self.beta_hat.extend(beta_hat)
        self.task_signature.append((cov_param_idx, rep, algorithm))

if __name__ == '__main__':

    # Block sizes
    block_sizes = [5, 10, 20]

    # Block correlation
    correlation = [0, 0.08891397, 0.15811388, 0.28117066, 0.5]

    # Exponential length scales
    L = [2, 5, 10, 20]

    cov_list, _ = get_cov_list(n_features, 60, correlation, block_sizes, L, n_supplement = 20)

    cov_params = [{'correlation' : t[0], 'block_size' : t[1], 'L' : t[2], 't': t[3]} for t in cov_list]

    nreps = 10
    n_features = 100
    n_samples = 400

    # Divide up the tasks
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    pool = MPIPool(comm)

    # Take the outer product over cov_params, reps, and algorithms
    # 0: Lasso
    # 1: UoI
    # 2: SCAD
    # 3: MCP

    tasks = itertools.product(np.arange(len(cov_params)), np.arange(nreps), (0, 1, 2, 3))
    worker = Worker(n_features=n_features, n_samples=n_samples, cov_params=cov_params)
    pool.map(worker, tasks)

    pool.close()

    # Gather across ranks and save
    betas = comm.gather(worker.beta, root = 0)
    beta_hats = comm.gather(worker.beta_hat, root = 0)
    task_signatures = comm.gather(worker.task-signature, root = 0)

    # Just save:
    with open('bias_exp_results.dat', 'wb') as f:
        f.write(pickle.dumps(betas))
        f.write(pickle.dumps(beta_hats))
        f.write(pickle.dumps(task_signatures))
