import sys, os
import numpy as np
import matplotlib.pyplot as plt
import time
import pdb
import argparse
import itertools
import pickle
from mpi4py import MPI

from dchisq import DChiSq
from mpi_utils.ndarray import Gatherv_rows

# No scaling

# Provide n, S, and the savepath through command line arguments

###### Command line arguments #######
parser = argparse.ArgumentParser()

parser.add_argument('n', type=int)
parser.add_argument('p', type=int)
parser.add_argument('savepath')

args = parser.parse_args()
n = args.n
p = args.p
S_ = int(np.log(p))
savepath = args.savepath

comm = MPI.COMM_WORLD
rank = comm.rank
numproc = comm.Get_size()

# Keep sigma and gamma squared equal to each other
sigma_sq = 1
gamma_sq = 0.1

F = np.linspace(0, 2 * np.log(n), 25)

F_chunk = np.array_split(F, numproc)

# Storage
cdf_vals = np.zeros((len(F_chunk[rank]), np.arange(1, p/2).size))

for i, F_ in enumerate(F_chunk[rank]):

    for i3, T in enumerate(np.linspace(1, p/2, 50, dtype=int)): 

        t0 = time.time()
        dx2 = DChiSq(gamma_sq, sigma_sq, n - T, T)
        
        DeltaF = F_ * (S_ - T)
        
        # Calculate the CDF        
        p = dx2.nCDF(DeltaF)
        cdf_vals[i, i3] = p
        print('Rank %d: %d/%d, %f s' % (rank, i3, len(np.arange(1, 20, 2)), time.time() - t0))
            

# Gather
cdf_vals = Gatherv_rows(cdf_vals, comm, root = 0)

# Save
if rank == 0:
    with open(savepath, 'wb') as f:
        f.write(pickle.dumps(cdf_vals))
        f.write(pickle.dumps(n))
        f.write(pickle.dumps(p))
        f.write(pickle.dumps(F))
