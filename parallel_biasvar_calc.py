import sys, os
import time
import pdb
import pandas as pd
import h5py, itertools, pickle
from itertools import islice
import numpy as np
from mpi4py import MPI

# Chunk up a dictionary:
def dict_chunker(dict_, max_size):

    it= iter(dict_)
    for i in range(0, len(dict_), max_size):
        yield {k:dict_[k] for k in islice(it, max_size)}

def calc_bias_var(beta_file, bids):
    
    try:
        beta = beta_file['beta'][bids, :]
    except:
        pdb.set_trace()

    # Ensure all the betas are the same
    try:
        assert(np.isclose(beta, beta[0]).all())
    except:
        pdb.set_trace()

    beta_hats = beta_file['beta_hat'][bids, :]
    
    # Total bias
    total_bias = np.mean(np.linalg.norm(beta - beta_hats, axis=1))
    
    common_support_bias = 0
    
    # Common support bias
    for i in range(bids.size):
        common_support = list(set(np.nonzero(beta[i, :])[0]).intersection(set(np.nonzero(beta_hats[i, :])[0])))
        # Nothing correctly selected
        if len(common_support) == 0:
            common_support_bias = np.nan
        else:
            common_support_bias += 1/len(common_support) * np.linalg.norm(beta[i, common_support] - beta_hats[i, common_support])
    
    common_support_bias *= 1/bids.size
    variance = np.mean(np.var(beta_hats, axis = 0))
    
    return total_bias, common_support_bias, variance


if __name__ == '__main__':

    root_dir = os.environ['SCRATCH'] + '/finalfinal'
    # Create communicator objects
    comm = MPI.COMM_WORLD

    df_name = 'SCAD'

    if comm.rank == 0:

       # Load the uid groups
        with open('uid_groups1.dat', 'rb') as f:
            uid_groups = pickle.load(f)

        split_size = int(np.ceil(len(uid_groups[df_name])/comm.size))

        # Assemble tasks and then distribute:
        tasklist = [chunk for chunk in dict_chunker(uid_groups[df_name], split_size)]
        for i, chunk in enumerate(tasklist[1:]):
            comm.send(chunk, dest=i + 1, tag=11)
        tasks = tasklist[0]
    else:
        tasks = comm.recv(source=0, tag=11)

    beta_file = h5py.File('%s/scad_pp_beta.h5' % root_dir , 'r')

    bias_var_datalist = []

    print('Rank %d has %d tasks' % (comm.rank, len(tasks)))

    # Tasks are a dictionary. Iterate over the keys:
    task_counter = 0
    for uid, bids in tasks.items():
        t0 = time.time()
        total_bias, common_support_bias, variance = calc_bias_var(beta_file, bids)
        bias_var_datalist.append({'uid': uid, 'total_bias': total_bias, 'common_bias': common_support_bias,
                                  'variance': variance})
        print('Rank %d completes task %d in %f s' % (comm.rank, task_counter, time.time() - t0))
        task_counter += 1   

    bias_var_datalist = comm.gather(bias_var_datalist)

    if comm.rank == 0:
        with open('biasvar_calc_scad.dat', 'wb') as f:
            f.write(pickle.dumps(bias_var_datalist))

