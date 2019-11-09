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
parser.add_argument('S', type=int)
parser.add_argument('savepath')

args = parser.parse_args()
n = args.n
S = args.S
savepath = args.savepath

p = 250

comm = MPI.COMM_WORLD
rank = comm.rank
numproc = comm.Get_size()

# Keep sigma and gamma squared equal to each other
sigma_sq = 1
gamma_sq = 0.1

S = np.linspace(5, 125, 25)
F = np.linspace(0, 2 * np.log(n), 25)

# Create tuples of unique combinations of S and F to distribute across ranks
S_F = list(itertools.combinations([S, F]))
S_F_idx = list(itertools.combinations([np.arange(S.size), np.arange(F.size)]))

S_F_chunk = np.array_split(S_F, numproc)
S_F_idx_chunk = np.array_split(S_F_chunk, numproc)

# Storage
cdf_vals = np.zeros((len(S_F_chunk[rank]), np.arange(1, p/2).size))

for i, S_F_tuple in enumerate(S_F_chunk[rank]):

    S_ = S_F_tuple[0]
    F_ = S_F_tuple[1]

    for i3, T in enumerate(np.arange(1, p/2)): 

        t0 = time.time()
        dx2 = DChiSq(gamma_sq, sigma_sq, n - T, T)
        
        DeltaF = F_ * (S_ - T)
        
        # Calculate the CDF        
        p = dx2.nCDF(DeltaF)
        cdf_vals[i, i3] = p
        print(time.time() - t0)
            

# Gather
cdf_vals = Gatherv_rows(cdf_vals, comm, root = 0)

# Save
if rank == 0:
    with open(savepath, 'wb') as f:
        f.write(pickle.dumps(cdf_vals))
        f.write(pickle.dumps(S_F_idx_chunk))
        f.write(pickle.dumps(S))
        f.write(pickle.dumps(F))
        f.write(pickle.dumps(T))