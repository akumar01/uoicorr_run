import sys, os
import numpy as np
import matplotlib.pyplot as plt
import time
import pdb
import argparse
import itertools
import pickle
from mpi4py import MPI
from schwimmbad import MPIPool

from dchisq import DChiSq
from mpi_utils.ndarray import Gatherv_rows

class Worker(object):

    def __init__(self, savepath, gamma_sq, sigma_sq, n, S):

        self.savepath = savepath
        self.gamma_sq = gamma_sq
        self.sigma_sq = sigma_sq
        self.n = n
        self.S = S

    def CDFcalc(self, T, F):

        t0 = time.time()
        dx2 = DChiSq(self.gamma_sq, self.sigma_sq, n - T, T)
        DeltaF = F * (self.S - T)
        # Calculate the CDF        
        p = dx2.nCDF(DeltaF)

        # Log what process is working on which task
        print('Rank %d: T: %d, F: %d, %f s' % (rank, T, F, time.time() - t0))

        return p, T, F

    def save(self, result, T, F)

        # Save each individual task away as a separate pickle file
        with open('%s/%d_%d.dat' % (self.savepath, T, F), 'wb') as f:

            f.write(pickle.dumps(result))
            f.write(pickle.dumps(T))
            f.write(pickle.dumps(F))
            f.write(pickle.dumps(self.S))
            f.write(pickle.dumps(self.n))
            f.write(pickle.dumps(self.sigma_sq))
            f.write(pickle.dumps(self.gamma_sq))


    def __call__(self, task):

        T, F = task
        return self.CDFcalc(T, F)


if __name__ == '__main__':

    # Log scaling

    # Provide n, S, and the savepath through command line arguments

    ###### Command line arguments #######
    parser = argparse.ArgumentParser()

    parser.add_argument('n', type=int)
    parser.add_argument('S', type=int)
    parser.add_argument('savepath')

    args = parser.parse_args()
    n = args.n
    S_ = args.S
    savepath = args.savepath

    p = 250

    comm = MPI.COMM_WORLD
    rank = comm.rank
    numproc = comm.Get_size()

    # Keep sigma and gamma squared equal to each other
    sigma_sq = 1
    gamma_sq = 0.1

    F = np.linspace(0, 2 * np.log(n), 25)
    T = np.linspace(1, p/2, 25, dtype=int)

    pool = MPIPool(comm)

    worker = Worker(savepath, gamma_sq, sigma_sq, n, S_)

    # Split F and T into tasks
    tasks = zip(F, T)

    for r in pool.map(worker, tasks, callback = worker.save):
        pass

    pool.close()