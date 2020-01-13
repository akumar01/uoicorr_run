import numpy as np
import sys, os
import glob
import pickle
import pdb
from job_utils.results import ResultsManager

path = sys.argv[1]
exp_type = sys.argv[2]
dirs = glob.glob('%s/%s_job0_*')

# For each directory in the list of directories, 
# assemble the unifnished jobs as a tuple of results_manager object
# and index
task_list = []
root_dir = '/'.join(dirs[0].split('/')[:-2])
for i, dir_ in enumerate(dirs):
    # the last numbers correspond to the index of the arg file
    argnumber = int(dir_.split('_')[-1])
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
        task_list.append((rmanager, arg_file, idx))

with open('%s_tasks.dat', 'wb') as f:
    f.write(pickle.dumps(task_list))