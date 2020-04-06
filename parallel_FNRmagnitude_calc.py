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

def calc_FNR_magnitude(beta_file, bids):
    
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
    

    false_negative_magnitudes = []
    false_positive_magnitudes = []
    true_coeff_magnitudes = []

    # Identify False Negatives and False Positives
    for i in range(bids.size):
        S = set(np.nonzero(beta[i, :])[0])
        Shat = set(np.nonzero(beta_hats[i, :])[0])

        false_negatives = list(S.difference(Shat))
        false_positives = list(Shat.difference(S))

        false_negative_magnitudes.extend(np.abs(beta[i, false_negatives]))
        false_positive_magntiudes.extend(np.abs(beta_hats[i, false_positives]))

        true_coeff_magnitudes.extend(np.abs(beta[i, S]))


    # Reduce the false negatives and the true coefficient magnitudes to a distribution 
    # on the same histogram scale
    range_min = 0
    range_max = np.max([np.max(true_coeff_magnitudes)])

    FNR_hist, bin_edges = np.histogram(false_negative_magnitudes, range=(range_min, range_max), bins=20)
    true_coef_hist, _ = np.histogram(true_coeff_magnitudes, range=(range_min, range_max), bins=20)

    # Separate histogram for the false positives, as the scale might be skewed here
    FPR_hist, bin_edges2 = np.histogram(false_positive_magntiudes, range=(0, np.max(false_positive_magntiudes)), 
                                        bins=20)

    # Keep track of a few summary statistics of the true support magnitudes
    coeff_mean = np.mean(true_coeff_magnitudes)
    coeff_min = np.min(true_coeff_magnitudes)
    coeff_quantiles = np.quantile(true_coeff_magnitudes, q=[0.25, 0.5, 0.75])

    # And the FNR and FPR (as not to have to recalculate things)    
    FNR_mean = np.mean(false_negative_magnitudes)
    FNR_min = np.min(false_negative_magnitudes)
    FNR_quantiles = np.quantile(false_negative_magnitudes, q=[0.25, 0.5, 0.75])

    FPR_mean = np.mean(false_positive_magnitudes)
    FPR_min = np.min(false_positive_magnitudes)
    FPR_quantiles = np.quantile(false_positive_magnitudes, q=[0.25, 0.5, 0.75])

    results_dict = {'true_hist': true_coef_hist, 'FNR_hist': FNR_hist, 'FPR_hist': FPR_hist, 
                    'bin_edges1': bin_edges, 'bin_edges2': bin_edges2, 'true_mean': true_mean,
                    'true_min': true_min, 'true_quantiles': coeff_quantiles,
                    'FNR_mean': FNR_mean, 'FNR_min': FNR_min, 'FNR_quantiles': FNR_quantiles,
                    'FPR_mean': FPR_mean, 'FPR_min': FPR_min, 'FPR_quantiles': FPR_quantiles}

    return results_dict


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
        true_coef_hist, FNR_hist, bin_edges, FPR_hist, bin_edges2,
        results_dict = calc_FNR_magnitude(beta_file, bids)
        results_dict.update({'uid': uid})
        bias_var_datalist.append(results_dict)
        print('Rank %d completes task %d in %f s' % (comm.rank, task_counter, time.time() - t0))
        task_counter += 1   

    bias_var_datalist = comm.gather(bias_var_datalist)

    if comm.rank == 0:
        with open('biasvar_calc_scad.dat', 'wb') as f:
            f.write(pickle.dumps(bias_var_datalist))

