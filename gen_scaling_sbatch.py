import sys, os
import numpy as np
import matplotlib.pyplot as plt
import time
import pdb
import argparse
import itertools
import pickle
import copy

# Write sbatch files to submit linear scaling, log scaling, no sclaing

# No scaling

n = np.linspace(125, 1000, 10, dtype=int)
S = np.linspace(5, 125, 25, dtype=int)

n_S_combo = intertools.product(n, S)

sbatch_dir = os.environ['SCRATCH'] + '/scaling/no_scaling'
if not os.path.exists(sbatch_dir):
    os.makedirs(sbatch_dir)
 
with open('no_scaling.sh', 'w') as sb:

	# For each combination of n and S, create a line in the sbatch file
	sb.write('#!/bin/bash\n')
	sb.write('#SBATCH --qos=regular\n')
	sb.write('#SBATCH --constraint=knl\n')            
	sb.write('#SBATCH -N %d\n' % n_nodes)
	sb.write('#SBATCH -t 01:30:00\n')
	sb.write('#SBATCH --job-name=%s\n' % 'noscaling')
	sb.write('#SBATCH --out=%s/out.o\n' % sbatch_dir)
	sb.write('#SBATCH --error=%s/error.e\n' % sbatch_dir)
	sb.write('#SBATCH --mail-user=ankit_kumar@berkeley.edu\n')
	sb.write('#SBATCH --mail-type=FAIL\n')
	# Work with out own Anaconda environment
	# To make this work, we had to add some paths to our .bash_profile.ext
	sb.write('source ~/anaconda3/bin/activate\n')
	sb.write('source activate nse\n')

	# Critical to prevent threads competing for resources
	sb.write('export OMP_NUM_THREADS=1\n')
	sb.write('export KMP_AFFINITY=disabled\n')

	for node, n_S_tuple in enumerate(n_S_combo):
	    savepath = '%s/node%d.dat' % (sbatch_dir, node)
	    sb.write('srun -N 1 -n 25 python3 -u noscaling.py %d %d %s &\n' % (n_S_tuple[0], n_S_tuple[1], savepath))
	sb.write('wait')


# Log scaling

p = np.linspace(125, 1000, 10, dtype=int)
n = copy.deepcopy(p)

n_p_combo = intertools.product(n, p)

sbatch_dir = os.environ['SCRATCH'] + '/scaling/log_scaling'
if not os.path.exists(sbatch_dir):
    os.makedirs(sbatch_dir)
 
with open('log_scaling.sh', 'w') as sb:

	# For each combination of n and S, create a line in the sbatch file
	sb.write('#!/bin/bash\n')
	sb.write('#SBATCH --qos=regular\n')
	sb.write('#SBATCH --constraint=knl\n')            
	sb.write('#SBATCH -N %d\n' % n_nodes)
	sb.write('#SBATCH -t 01:30:00\n')
	sb.write('#SBATCH --job-name=%s\n' % 'logscaling')
	sb.write('#SBATCH --out=%s/out.o\n' % sbatch_dir)
	sb.write('#SBATCH --error=%s/error.e\n' % sbatch_dir)
	sb.write('#SBATCH --mail-user=ankit_kumar@berkeley.edu\n')
	sb.write('#SBATCH --mail-type=FAIL\n')
	# Work with out own Anaconda environment
	# To make this work, we had to add some paths to our .bash_profile.ext
	sb.write('source ~/anaconda3/bin/activate\n')
	sb.write('source activate nse\n')

	# Critical to prevent threads competing for resources
	sb.write('export OMP_NUM_THREADS=1\n')
	sb.write('export KMP_AFFINITY=disabled\n')

	for node, n_p_tuple in enumerate(n_p_combo):
	    savepath = '%s/node%d.dat' % (sbatch_dir, node)
	    sb.write('srun -N 1 -n 25 python3 -u logscaling.py %d %d %s &\n' % (n_p_tuple[0], n_p_tuple[1], savepath))
	sb.write('wait')

# Linear scaling

p = np.linspace(125, 1000, 10, dtype=int)
n = copy.deepcopy(p)

n_p_combo = intertools.product(n, p)

sbatch_dir = os.environ['SCRATCH'] + '/scaling/linear_scaling'
if not os.path.exists(sbatch_dir):
    os.makedirs(sbatch_dir)
 
with open('linear_scaling.sh', 'w') as sb:

	# For each combination of n and S, create a line in the sbatch file
	sb.write('#!/bin/bash\n')
	sb.write('#SBATCH --qos=regular\n')
	sb.write('#SBATCH --constraint=knl\n')            
	sb.write('#SBATCH -N %d\n' % n_nodes)
	sb.write('#SBATCH -t 01:30:00\n')
	sb.write('#SBATCH --job-name=%s\n' % 'linearscaling')
	sb.write('#SBATCH --out=%s/out.o\n' % sbatch_dir)
	sb.write('#SBATCH --error=%s/error.e\n' % sbatch_dir)
	sb.write('#SBATCH --mail-user=ankit_kumar@berkeley.edu\n')
	sb.write('#SBATCH --mail-type=FAIL\n')
	# Work with out own Anaconda environment
	# To make this work, we had to add some paths to our .bash_profile.ext
	sb.write('source ~/anaconda3/bin/activate\n')
	sb.write('source activate nse\n')

	# Critical to prevent threads competing for resources
	sb.write('export OMP_NUM_THREADS=1\n')
	sb.write('export KMP_AFFINITY=disabled\n')

	for node, n_p_tuple in enumerate(n_p_combo):
	    savepath = '%s/node%d.dat' % (sbatch_dir, node)
	    sb.write('srun -N 1 -n 25 python3 -u linearscaling.py %d %d %s &\n' % (n_p_tuple[0], n_p_tuple[1], savepath))
	sb.write('wait')

