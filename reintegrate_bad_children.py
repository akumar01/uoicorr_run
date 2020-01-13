import sys, os
import pdb
import glob
import argparse
import pickle

import h5py_wrapper 

from job_utils.results import ResultsManager
from job_utils.results import insert_data

# Run this script after running discipline_bad_children.py, in turn after running parallel_concatenate
# and identifying the children that could not be integrated into the concatenated data files
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('source_dir')
    parser.add_argument('exp_type')

    args = parser.parse_args()

    # Grab the bad children

    # Grab the bad children
    bad_children_files = glob.glob('%s/%s/bad_children*' % (args.source_dir, args.exp_type))  

    bad_children = []
    for bad_children_file in bad_children_files:
        with open(bad_children_file, 'rb') as f:
            bad_children.extend(pickle.load(f))

    # Iterate through the corresponding data files, load the child data, and insert it into the 
    # right place
    for bad_child in bad_children:

        bad_child['path'] = bad_child['path'].replace('//', '/')

        # Assemble full path to bad_child path
        child_path = args.source_dir + '/' + args.exp_type + '/' + bad_child['path']
        child_data = h5py_wrapper.load(child_path)
        # Load master file
        child_dir = bad_child['path'].split('/child')[0]
        master_path = args.source_dir + '/' + args.exp_type + '/' + child_dir + '.dat'
        master_data = h5py_wrapper.load(master_path)
        master_data = insert_data(master_data, child_data, bad_child['idx'])
        h5py_wrapper.save(master_path, master_data, write_mode='w')
