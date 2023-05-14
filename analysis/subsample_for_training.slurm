#!/bin/bash
#SBATCH --job-name=subsamp  # Name of the job
#SBATCH --output=logs/subsamp-%A-%a.out  # STDOUT file
#SBATCH --error=logs/subsamp-%A-%a.err   # STDERR file
#SBATCH --nodes=1               # Node count
#SBATCH --ntasks=1          # Number of tasks across all nodes
#SBATCH --cpus-per-task=1      # Cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=64G           # total memory per node
#SBATCH --array=1-141
#SBATCH --time=06:00:00          # Run time limit (HH:MM:SS)        # Job array
#SBATCH --mail-type=all          # Email on job start, end, and fault
#SBATCH --mail-user=swwolf@princeton.edu

conda activate sleap_dev

python lts-tsne-subsample-batch.py --number $((${SLURM_ARRAY_TASK_ID} - 1))
# mkdir 20230210-mmpy-lts-all-headprobinterp-missingness-pchip5-fillnanmedian-setnonfinite0-removegt1missing/old-wavelets/
# mkdir 20230210-mmpy-lts-all-headprobinterp-missingness-pchip5-fillnanmedian-setnonfinite0-removegt1missing/old-projections/
# mv 20230210-mmpy-lts-all-headprobinterp-missingness-pchip5-fillnanmedian-setnonfinite0-removegt1missing/Wavelets/ 20230210-mmpy-lts-all-headprobinterp-missingness-pchip5-fillnanmedian-setnonfinite0-removegt1missing/old-wavelets/
# mv 20230210-mmpy-lts-all-headprobinterp-missingness-pchip5-fillnanmedian-setnonfinite0-removegt1missing/Projections/ 20230210-mmpy-lts-all-headprobinterp-missingness-pchip5-fillnanmedian-setnonfinite0-removegt1missing/old-projections/

# mkdir 20230210-mmpy-lts-all-headprobinterp-missingness-pchip5-fillnanmedian-setnonfinite0-removegt1missing/Wavelets/
# mkdir 20230210-mmpy-lts-all-headprobinterp-missingness-pchip5-fillnanmedian-setnonfinite0-removegt1missing/Projections/

# cp 20230221-mmpy-lts-problematic-subset-headprobinterp-missingness-pchip5-fillnanmedian-setnonfinite0-removegt1missing/Projections/* 20230210-mmpy-lts-all-headprobinterp-missingness-pchip5-fillnanmedian-setnonfinite0-removegt1missing/Projections/
# rclone sync -iP 20230221-mmpy-lts-problematic-subset-headprobinterp-missingness-pchip5-fillnanmedian-setnonfinite0-removegt1missing/Wavelets/ 20230210-mmpy-lts-all-headprobinterp-missingness-pchip5-fillnanmedian-setnonfinite0-removegt1missing/Wavelets/

# mv -f /Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/20230210-mmpy-lts-all-headprobinterp-missingness-setnonfinite0-removegt1missing/Wavelets/* /Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/20230210-mmpy-lts-all-headprobinterp-missingness-pchip5-fillnanmedian-setnonfinite0-removegt1missing/old-wavelets/Wavelets/

# mv -f  /Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/20230210-mmpy-lts-all-headprobinterp-missingness-pchip5-fillnanmedian-setnonfinite0-removegt1missing/old-wavelets/Wavelets/* /Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/20230210-mmpy-lts-all-headprobinterp-missingness-pchip5-fillnanmedian-setnonfinite0-removegt1missing/Wavelets/
# mv -f  /Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/20230210-mmpy-lts-all-headprobinterp-missingness-pchip5-fillnanmedian-setnonfinite0-removegt1missing/old-projections/Projections/* /Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/20230210-mmpy-lts-all-headprobinterp-missingness-pchip5-fillnanmedian-setnonfinite0-removegt1missing/Projections/