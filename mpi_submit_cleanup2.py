import time
t0 = time.time()
import sys, os
import pdb
import itertools
import glob
import argparse
import pickle
import struct
import numpy as np
from mpi4py import MPI
import h5py_wrapper
from pydoc import locate
from sklearn.preprocessing import StandardScaler

from mpi_utils.ndarray import Bcast_from_root, Gatherv_rows, Gather_ndlist

from job_utils.results import  ResultsManager
from job_utils.idxpckl import Indexed_Pickle

from utils import sparsify_beta, gen_data
from expanded_ensemble import load_covariance
from results_manager import init_results_container, calc_result
# from loguru import logger

print('Import time: %f' % (time.time() - t0))

# Reformat mpi_submit to take an entire directory of jobs that have been partially
# completed and use schwimmbad to deal out fit responsibilities to each one

# Prefer this to using the --resume flag
 
def mpi_main(task_tuple, subcomm):
    # Unpack args
    rmanager, arg_file, idx = task_tuple
    exp_type = args.exp_type
    results_dir = rmanager.directory

    argnumber = int(results_dir.split('_')[-1]) 
    #dir_logger = logger.bind(logid = argnumber)
    #dir_logger.debug('Starting task %d of %s' % (idx, arg_file))
    start = time.time()
    # hard-code n_reg_params because why not
    if exp_type in ['EN', 'scad', 'mcp']:
        n_reg_params = 2
    else:
        n_reg_params = 1

    # Open the arg file and read out metadata
    f = Indexed_Pickle(arg_file)
    f.init_read()

    n_features = f.header['n_features']
    selection_methods = f.header['selection_methods']
    fields = f.header['fields']

    params = f.read(idx)
    params['comm'] = subcomm
    X, X_test, y, y_test, params = gen_data_(params, 
                                         subcomm, subrank)
    #dir_logger.debug('Generated data!')
    # Hard-coded convenience for SCAD/MCP
    if exp_type in ['scad', 'mcp']:
        exp = locate('exp_types.%s' % 'PYC')
        params['penalty'] = exp_type
    else:
        exp = locate('exp_types.%s' % exp_type)
    
    exp_results = exp.run(X, y, params, selection_methods)
    if subrank == 0:
        # print('checkpoint 2: %f' % (time.time() - start))

        results_dict = init_results_container(selection_methods)

        #### Calculate and log results for each selection method

        for selection_method in selection_methods:

            for field in fields[selection_method]:
                results_dict[selection_method][field] = calc_result(X, X_test, y, y_test,
                                                                   params['betas'].ravel(), field,
                                                                   exp_results[selection_method])
        
        rmanager.add_child(results_dict, idx = idx)

    f.close_read()
    if subcomm.rank == 0:
        print('Total time: %f' % (time.time() - start))
    #dir_logger.debug('Task completed!')
        
def manage_comm():

    '''Create a comm object and split it into the appropriate number of subcomms'''

    comm = MPI.COMM_WORLD
    rank = comm.rank
    numproc = comm.Get_size()

    if args.comm_splits is None:
        if args.exp_type in ['UoILasso', 'UoIElasticNet']:
            comm_splits = 2
        else:
            comm_splits = numproc
    else:
        comm_splits = args.comm_splits

    # Use array split to do comm.split

    # Take the root node and set it aside - this is what schwimmbad will use to coordinate
    # the other groups

    ranks = np.arange(numproc)
    split_ranks = np.array_split(ranks, comm_splits)
    # if rank == 0:
    #     color = 0
    # else:
    color = [i for i in np.arange(comm_splits) if rank in split_ranks[i]][0]
    subcomm_roots = [split_ranks[i][0] for i in np.arange(comm_splits)]
    subcomm = comm.Split(color, rank)

    nchunks = comm_splits
    subrank = subcomm.rank
    numproc = subcomm.Get_size()

    # Create a group including the root of each subcomm (unused at the moment)
    global_group = comm.Get_group()
    root_group = MPI.Group.Incl(global_group, subcomm_roots)
    root_comm = comm.Create(root_group)
    return comm, rank, color, subcomm, subrank, root_comm

def gen_data_(params, subcomm, subrank):
    ''' Use the seeds provided from the arg file to generate regression design and data'''

    seed = params['seed']

    if subrank == 0:
        # Generate covariance according to index
        sigma, cov_params = load_covariance(params['cov_idx'])

        # Sparsify the beta - seed with the block size
        beta = sparsify_beta(params['betadict']['beta'], cov_params['block_size'],
                             params['sparsity'], seed=cov_params['block_size'])
    else:

        sigma = None
        beta = None

    sigma = Bcast_from_root(sigma, subcomm)
    beta = Bcast_from_root(beta, subcomm)

    params['sigma'] = sigma
    params['betas'] = beta

    # If all betas end up zero for this sparsity level, output a warning and skip
    # this iteration (Make sure all ranks are instructed to continue!)
    if np.count_nonzero(beta) == 0:
        print('Warning, all betas were 0!')
        print(params)

    if subrank == 0:

        # Generate data
        X, X_test, y, y_test, ss = gen_data(params['n_samples'], params['n_features'],
                                        params['kappa'], sigma, beta, seed)

        # Standardize
        X = StandardScaler().fit_transform(X)
        X_test = StandardScaler().fit_transform(X_test)
        y -= np.mean(y)
        y_test -= np.mean(y_test)

    else:
        X = None
        X_test = None
        y = None
        y_test = None
        ss = None

    X = Bcast_from_root(X, subcomm)
    X_test = Bcast_from_root(X_test, subcomm)
    y = Bcast_from_root(y, subcomm)
    y_test = Bcast_from_root(y_test, subcomm)
    # ss = Bcast_from_root(ss, subcomm)

    params['ss'] = ss

    return X, X_test, y, y_test, params

if __name__ == '__main__':

    total_start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('tasklist')
    parser.add_argument('exp_type', default = 'UoILasso')
    parser.add_argument('--comm_splits', type=int, default = None)
    parser.add_argument('-t', '--test', action = 'store_true')
    # Number of reps to break after if we are just testing
    parser.add_argument('--ntest', type = int, default = 1)

    args = parser.parse_args()

    # Create subcommunicators as needed
    comm, rank, color, subcomm, subrank, root_comm = manage_comm()
    print('rank: %d, color %d: subrank %d' % (rank, color, subrank))
    if rank == 0:
        with open(args.tasklist, 'rb') as f:
            tasklist = pickle.load(f)
    else:
        tasklist = None

    tasklist = comm.bcast(tasklist, root = 0)
    
    # Split across comm_splits
    tasklist_chunked = np.array_split(tasklist, args.comm_splits)

    for task in tasklist_chunked[color]:
        if subrank == 0:
            print('%s, %s' % (task[1], task[2]))
        mpi_main(task, subcomm)
