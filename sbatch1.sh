#!/bin/bash
#SBATCH --qos=premium
#SBATCH --constraint=knl
#SBATCH -N 25
#SBATCH -t 02:00:00
#SBATCH --job-name=lasso_cleanup
#SBATCH --out=/global/cscratch1/sd/akumar25/finalfinal/CV_Lasso/cleanup.o
#SBATCH --error=/global/cscratch1/sd/akumar25/finalfinal/CV_Lasso/cleanup.e
#SBATCH --mail-user=ankit_kumar@berkeley.edu
#SBATCH --mail-type=FAIL
source ~/anaconda3/bin/activate
source activate nse
export OMP_NUM_THREADS=1
srun -n 1700 python -u mpi_submit_cleanup.py /global/cscratch1/sd/akumar25/finalfinal/lasso_cleanup_dirs.dat CV_Lasso --comm_splits=1700