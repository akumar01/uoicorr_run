import sys
import pickle 
import glob
import numpy as np
from job_utils.idxpckl import Indexed_Pickle

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

		params0 = ip.read(0)

		ip.close_read()

		cov_params.append([params0['correlation'],
						   params0['block_size'],
						   params0['L'],
						   params0['t']])


	with open(savename, 'wb') as f:
		f.write(pickle.dumps(cov_params))
