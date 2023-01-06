#!/bin/bash
#SBATCH --job-name=conv-h5  # Name of the job
#SBATCH --output=logs/conv-h5-%A-%a.out  # STDOUT file
#SBATCH --error=logs/conv-h5-%A-%a.err   # STDERR file
#SBATCH --nodes=1               # Node count
#SBATCH --ntasks=1          # Number of tasks across all nodes
#SBATCH --cpus-per-task=6      # Cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=32G           # total memory per node
#SBATCH --array=1-141          # Array range
#SBATCH --time=01:00:00          # Run time limit (HH:MM:SS)        # Job array
#SBATCH --mail-type=all          # Email on job start, end, and fault
#SBATCH --mail-user=swwolf@princeton.edu

conda activate sleap_dev

python lts-convert-h5s.py --number $((${SLURM_ARRAY_TASK_ID} - 1))
