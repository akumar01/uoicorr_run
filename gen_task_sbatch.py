import sys, os
import numpy as np 
import pickle 

task_path = sys.argv[1]
n_nodes = int(sys.arv[2])
job_time = sys.argv[3]
exp_type = 'UoILasso'

tasklists = glob.glob('%s/*' % task_path)

sbname = 'cleanup_tasks.sh'
jobname = 'uoi_cleanup'


# Each param file has access to this many nodes
nodes_per_file = max(1, np.floor(n_nodes/len(tasklist)).astype(int))

# We will take the metadata from the first element in the
# sbatch array chunk
qos = 'regular'

script_dir = os.environ['SCRATCH'] + '/run/uoicorr_run'
script = 'mpi_submit_cleanup2.py'

with open(sbname, 'w') as sb:
    # Arguments common across jobs
    sb.write('#!/bin/bash\n')
    # USE SHIFTER IMAGE
    # sb.write('#SBATCH --image=docker:akumar25/nersc_uoicorr:latest\n')
    sb.write('#SBATCH --qos=%s\n' % qos)
    sb.write('#SBATCH --constraint=knl\n')            
    sb.write('#SBATCH -N %d\n' % n_nodes)
    sb.write('#SBATCH -t %s\n' % job_time)
    sb.write('#SBATCH --job-name=%s\n' % jobname)
    sb.write('#SBATCH --out=%s/%s.o\n' % (os.environ['SCRATCH'] + '/run', jobname))
    sb.write('#SBATCH --error=%s/%s.e\n' % (os.environ['SCRATCH'] + '/run', jobname))
    sb.write('#SBATCH --mail-user=ankit_kumar@berkeley.edu\n')
    sb.write('#SBATCH --mail-type=FAIL\n')
    # Work with out own Anaconda environment
    # To make this work, we had to add some paths to our .bash_profile.ext
    ## UNNECESSARY WITH SHIFTER
    sb.write('source ~/anaconda3/bin/activate\n')
    sb.write('source activate nse\n')

    # Critical to prevent threads competing for resources
    sb.write('export OMP_NUM_THREADS=1\n')
    sb.write('export KMP_AFFINITY=disabled\n')

    for i, tasklist in enumerate(tasklists):

        comm_splits = 4
        sb.write('srun -N %d -n %d -c 4 --cpu-bind=threads python -u %s/%s %s %s %s --comm_splits=%d &\n' % \
                (nodes_per_file, 64 * nodes_per_file,
                script_dir, script, tasklist, exp_type,
                comm_splits))

    sb.write('wait')
