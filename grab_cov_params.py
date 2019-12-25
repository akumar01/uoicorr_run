import sys
import pdb
import pickle 
import glob
import numpy as np
from job_utils.idxpckl import Indexed_Pickle

from misc import group_dictionaries

if __name__ == '__main__':
    jobdir = sys.argv[1]
    savename = sys.argv[2]

    # Go through the master directory, open each param file
    # and grab the cov params from the first entry
    param_files = glob.glob('%s/master/*.dat' % jobdir)
    cov_params = []

    for param_file in param_files:

        ip = Indexed_Pickle(param_file)
        ip.init_read()
        try:
            params0 = ip.read(0)
        except:
            continue
        ip.close_read()
        cov_params.append(params0['cov_params'])


    cov_params = group_dictionaries(cov_params, None)[0]
    

    with open(savename, 'wb') as f:
        f.write(pickle.dumps(cov_params))
