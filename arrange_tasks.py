import numpy as np
import sys, os
import glob
import pickle
import pdb
import time
from job_utils.results import ResultsManager
from job_utils.idxpckl import Indexed_Pickle

path = sys.argv[1]
exp_type = sys.argv[2]
savedir = sys.argv[3]
nsplits = int(sys.argv[4])
skip_cov = int(sys.argv[5])

dirs = glob.glob('%s/%s_job0_*/' % (path, exp_type))

# If no dirs (i.e. jobs have not been run yet, then iterate through the param files

# For each directory in the list of directories, 
# assemble the unifnished jobs as a tuple of results_manager object
# and index
task_list = []

if exp_type == 'CV_Lasso':
    root_dir = '/'.join(dirs[0].split('/')[:-3])
else:
    root_dir = '/'.join(dirs[0].split('/')[:-3])


for i, dir_ in enumerate(dirs):
    t0 = time.time()
    print(i)

    # the last numbers correspond to the index of the arg file
    argnumber = int(dir_.split('_')[-1].split('/')[0])
    arg_file = '%s/master/params%d.dat' % (root_dir, argnumber)

    # Open the arg file and read out metadata
    f = Indexed_Pickle(arg_file)
    f.init_read()
    total_tasks = len(f.index)
    try:
        rmanager = ResultsManager.restore_from_directory(dir_)
    except:
        rmanager = ResultsManager(total_tasks = total_tasks, directory = dir_)

    # Take the difference between what has been done and what needs to be done
    todo = np.array(list(set(np.arange(total_tasks)).difference(set(rmanager.inserted_idxs()))))
    
    for idx in todo:

        # Check whether the the todo has cov_idx >= 80 (discard if so)
        params = f.read(idx)
        if skip_cov:
            if params['cov_idx'] >= 80:
                continue
            else:
                task_list.append((rmanager, arg_file, idx))
        else:
            task_list.append((rmanager, arg_file, idx))
    
    print(time.time() - t0)


print(len(task_list))

# Top level splits - done across the 
task_list = np.array_split(task_list, nsplits)

if not os.path.exists(savedir):
    os.makedirs(savedir)

for split in range(nsplits):
    with open('%s/%s_tasks%d.dat' % (savedir, exp_type, split), 'wb') as f:
        f.write(pickle.dumps(task_list[split]))
