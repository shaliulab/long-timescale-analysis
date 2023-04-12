#!/bin/bash
#SBATCH --job-name=embed  # Name of the job
#SBATCH --output=logs/batch-%A-%a.out  # STDOUT file
#SBATCH --error=logs/batch-%A-%a.err   # STDERR file
#SBATCH --nodes=1               # Node count
#SBATCH --ntasks=1          # Number of tasks across all nodes
#SBATCH --cpus-per-task=20     # Cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=350G           # total memory per node
#SBATCH --array=157-312# 1-156
#SBATCH --time=1-00:00:00          # Run time limit (HH:MM:SS)        # Job array
#SBATCH --mail-type=all          # Email on job start, end, and fault
#SBATCH --mail-user=swwolf@princeton.edu

conda activate sleap_dev

python lts-tsne-embed-4.py --number $((${SLURM_ARRAY_TASK_ID} - 1))

# ls -lta /Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/20230210-mmpy-lts-day1-headprobinterp-missingness-pchionfinite0-removegt1missing/TSNE