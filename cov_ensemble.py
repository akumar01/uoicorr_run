import sys, os
import time
import pdb
import itertools
import pickle
from mpi4py import MPI

import numpy as np
from numpy.random import RandomState

from scipy.sparse.linalg import eigsh
from scipy import optimize
from scipy.stats import wishart, invwishart

from sklearn.preprocessing import StandardScaler

from mpi_utils.ndarray import Gatherv_rows

from utils import gen_covariance, gen_data, sparsify_beta, get_cov_list

def bound_eigenvalue(matrix, k):

    # Will need the matrix to be symmetric
    assert(np.allclose(matrix, matrix.T))
    
    t1 = time.time()
    # Sort each row
    ordering = np.argsort(np.abs(matrix), axis = 1)

    # Change to descending order
    ordering = np.fliplr(ordering)
    
    sorted_matrix = np.take_along_axis(np.abs(matrix), ordering, 1)

    # Find the diagonal and move it first    
    diagonal_locs = np.array([np.where(ordering[i, :] == i)[0][0] 
                              for i in range(ordering.shape[0])])
    for (row, column) in zip(range(ordering.shape[0]), diagonal_locs):
        sorted_matrix[row][:column+1] = np.roll(sorted_matrix[row][:column+1], 1)
        
    # Sum the first (k - 1) elements after the diagonal
    row_sums = np.sum(sorted_matrix[:, 1:k], axis = 1)
    diag = np.diagonal(matrix)
    
    # Evaluate all Bauer Cassini ovals
    pairs = list(itertools.combinations(np.arange(matrix.shape[0]), 2))
    # This takes a little bit of algebra
    oval_edges = [(np.sqrt(row_sums[idx[0]] * row_sums[idx[1]] + 1/4 * (diag[idx[0]] - diag[idx[1]])**2) \
                 + 1/2 * (row_sums[idx[1]] + row_sums[idx[0]])) for idx in pairs]
    
    # Take the max. This is a bound for any conceivable eigenvalue
    eig_bound1 = np.max(oval_edges)
    t1 = time.time() - t1
    
    return eig_bound1

# In contrast to bounding the eigenvalue, we explicitly fix the non-zero indices in advance, so we can just
# explicitly calculate the eigenvalue bound
def calc_eigenvalue(matrix, idxs):
    
    # Assemble the submatrix
    submatrix = matrix[np.ix_(idxs, idxs)]
    eig = eigsh(submatrix, k = 1, return_eigenvectors=False)[0]
    
    return eig


def calc_irrep_const(matrix, idxs):
    
    p = matrix.shape[0]
    k = len(idxs)
    idxs_complement = np.setdiff1d(np.arange(p), idxs)
    
    C11 = matrix[np.ix_(idxs, idxs)]
    C21 = matrix[np.ix_(idxs_complement, idxs)]
    
    # Calculate the resulting irrep. constant
    eta = np.max(C21 @ np.linalg.inv(C11) @ np.ones(k))

    return eta

if __name__ == '__main__':

    # Parallelize
    comm = MPI.COMM_WORLD

    # generate the standard covariance ensemble
    p = 500
    
    # Tune the strength of perturbation
    n = np.linspace(500, 8000 * 8, 10)

    # Load the covariance parameters
    with open('unique_cov_params.dat', 'rb') as f:
        cov_params = pickle.load(f)

    # How many Wishart matrices to generate
    nreps = 20

    # How many instantiations of data from that matrix to take (needed to calculate the irrep constant)
    nreps2 = 20

    # Do not include fully dense model
    sparsity = np.logspace(np.log10(0.02), 0, 15)[:-1]

    # Chunk the cov params
    cov_params_chunk = np.array_split(cov_params, comm.size)

    rho = np.zeros((len(cov_params_chunk[comm.rank]), len(n), nreps))
    eta = np.zeros((len(cov_params_chunk[comm.rank]), len(n), nreps, sparsity.size, nreps2))        
    eta2 = np.zeros(eta.size)
    norm_diff = np.zeros((len(cov_params_chunk[comm.rank]), len(n), nreps))

    for i1, cov_param in enumerate(cov_params_chunk[comm.rank]):
        print('Rank %d starting task %d' % (comm.rank, i1))
        # Generate Wishart matrices seeded by this particular sigma
        sigma = gen_covariance(p, **cov_param)
        for nidx, n_ in enumerate(n):         
            # t0 = time.time()
       
            for rep in range(nreps):
                t0 = time.time()  
                # Seed to be able to reconstruct the matrix later
                random_state = RandomState(rep)
                sigma_rep = wishart.rvs(df=n_, scale=sigma, random_state=random_state)
                # Normalize the matrix 
                D = np.sqrt(np.diag(np.diag(sigma_rep)))
                sigma_rep = np.linalg.inv(D) @ sigma_rep @ np.linalg.inv(D)            
                # sigma_rep = sigma
                norm_diff[i1, nidx, rep] = np.linalg.norm(sigma_rep - sigma, 'fro')            
                try:
                    rho[i1, nidx, rep] = 1/eigsh(np.linalg.inv(sigma_rep, 1, which='LM', return_eigenvectors=False, maxiter=10000))
                except:
                    rho[i1, nidx, rep] = 1/eigsh(np.linalg.inv(sigma_rep, 1, which='LM', return_eigenvectors=False, maxiter=10000, tol=1e-2))
                for i3, s in enumerate(sparsity):
#                    t0 = time.time()
                    # Keep the nonzero components of beta fixed for each sparsity
                    # Here, we ensure that blocks are treated equally
                
                    subset = sparsify_beta(np.ones((p, 1), dtype=int), block_size=cov_param['block_size'], 
                                       sparsity = s, seed = s).ravel()
                    if len(np.nonzero(subset)[0]) == 0:
                        eta[i1, nidx, rep, i3, :] = np.nan
                        eta2[i1, nidx, rep, i3] = np.nan
                        continue

                    else:
                        eta2[i1, nidx, rep, i3] = calc_irrep_const(sigma_rep, np.nonzero(subset)[0])               
                        
                        for rep2 in range(nreps2):         
                
                            X, _, _, _, _ = gen_data(2000, p, covariance = sigma_rep, beta=subset.ravel())
                            # Normalize X
                            X = StandardScaler().fit_transform(X)
                            C = 1/n_ * X.T @ X
                            eta[i1, nidx, rep, i3, rep2] = calc_irrep_const(C, np.nonzero(subset)[0])                            
                    
                if comm.rank == 0:    
                    print(time.time() - t0)        
        
        print('%d/%d' % (i1 + 1, len(cov_params_chunk[comm.rank])))
    

    # Gather and save results
    rho = Gatherv_rows(rho, comm, root = 0)
    eta = Gatherv_rows(eta, comm, root = 0)
    eta2 = Gatherv_rows(eta2, comm, root = 0)
    norm_diff = Gatherv_rows(norm_diff, comm, root = 0)


    if comm.rank == 0:
        with open('cov_ensemble.dat', 'wb') as f:
            f.write(pickle.dumps(rho))
            f.write(pickle.dumps(eta))
            f.write(pickle.dumps(eta2))
            f.write(pickle.dumps(norm_diff))
