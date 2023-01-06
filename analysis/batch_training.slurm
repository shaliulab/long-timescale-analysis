#!/bin/bash
#BATCH --job-name=bat_inf                              # Name of the job
#SBATCH --output=logs/batch_naps_%j_%A.out          # STDOUT file
#SBATCH --error=logs/batch_naps_%j_%A.err           # STDERR file
#SBATCH --nodes=1                                       # Node count
#SBATCH --ntasks=1                                      # Number of tasks across all nodes
#SBATCH --cpus-per-task=16                               # Cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=64G                                       # total memory per node
#SBATCH --array=1-3                                                 # Number of jobs % Max number of jobs to consume
#SBATCH --time=1-00:00:00                               # Run time limit (HH:MM:SS)
#SBATCH --mail-type=all                                 # Email on job start, end, and fault
#SBATCH --mail-user=swwolf@princeton.edu

module load conda

conda activate sleap_dev

FILE=$(sed "${SLURM_ARRAY_TASK_ID}q;d" list_of_files_centroid_configs.txt)

sleap-train -m ${FILE} BASE_PACKAGE
