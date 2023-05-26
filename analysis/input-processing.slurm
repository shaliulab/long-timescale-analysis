#!/bin/bash
#SBATCH --job-name=input  # Name of the job
#SBATCH --output=logs/input_%A_%a.out  # STDOUT file
#SBATCH --error=logs/input_%A_%a.err   # STDERR file
#SBATCH --nodes=1               # Node count
#SBATCH --ntasks=1          # Number of tasks across all nodes
#SBATCH --cpus-per-task=2      # Cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=64G           # total memory per node
#SBATCH --array=1-141 # 1-78        # Array range
#SBATCH --time=06:00:00          # Run time limit (HH:MM:SS)        # Job array
#SBATCH --mail-type=all          # Email on job start, end, and fault
#SBATCH --mail-user=swwolf@princeton.edu

conda activate sleap_dev

python lts-tsne-input-1-singlefly.py --number $((${SLURM_ARRAY_TASK_ID} - 1))
