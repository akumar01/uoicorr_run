import numpy as np
import os
import glob
import argparse
import pickle
import pdb

from job_utils.results import ResultsManager

def arrange_tasks(bad_children, root_dir, exp_type):

    task_list = []
    for i, child in enumerate(bad_children):
        try:
            argnumber = int(child['path'].split('/child')[0].split('_')[2])
        except:
             argnumber = int(child['path'].split('//')[0].split('_')[3])
             
        childir = '%s/%s/%s' % (root_dir, exp_type, child['path'].split('/child')[0])
        arg_file = '%s/master/params%d.dat' % (root_dir, argnumber)
        rmanager = ResultsManager(total_tasks = 2880, directory = childir)
        idx = child['idx']        
        task_list.append((rmanager, arg_file, idx))

    return task_list

if __name__ == '__main__':
 
    parser = argparse.ArgumentParser()
    parser.add_argument('source_dir')
    parser.add_argument('exp_type', default = 'UoILasso')

    args = parser.parse_args()
    # Grab the bad children
    bad_children_files = glob.glob('%s/bad_children*' % args.source_dir)

    bad_children = []
    for bad_children_file in bad_children_files:
        with open(bad_children_file, 'rb') as f:
            bad_children.extend(pickle.load(f))

    root_dir = '/'.join(args.source_dir.split('/')[0:6])
    task_list = arrange_tasks(bad_children, root_dir, args.exp_type)
    # Save the task_list to file
    with open('%s_badchildren_tasks.dat' % args.exp_type, 'wb') as f:
        f.write(pickle.dumps(task_list))
