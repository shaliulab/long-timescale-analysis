#!/bin/bash
#SBATCH --job-name=bat_inf  # Name of the job
#SBATCH --output=logs/batch_inf_%A_%a.out  # STDOUT file
#SBATCH --error=logs/batch_inf_%A_%a.err   # STDERR file
#SBATCH --nodes=1               # Node count
#SBATCH --ntasks=1          # Number of tasks across all nodes
#SBATCH --cpus-per-task=2      # Cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=32G           # total memory per node
#SBATCH --array=1-120
#SBATCH --time=1-00:00:00          # Run time limit (HH:MM:SS)        # Job array
#SBATCH --gres=gpu:1
#SBATCH --mail-type=all          # Email on job start, end, and fault
#SBATCH --mail-user=swwolf@princeton.edu

module load anaconda3
module load cudatoolkit/11.3
module load cudnn/cuda-11.x/8.2.0

conda activate sleap_dev
TARGET_DIR="/projects/SHAEVITZ/swwolf/lts/videos/experimental_data/exp3/ayroles-gpu/Camera1"
TARGET_VIDEO="0through228"
OUTPUT_NAME="exp3_cam1"

#python chunk_for_tracking.py ${TARGET_DIR}/${TARGET_VIDEO}.mp4 24

LIST=${TARGET_VIDEO}.frame_list

LINE_NUMBER=${SLURM_ARRAY_TASK_ID}
sleap-track ${TARGET_DIR}/${TARGET_VIDEO}.mkv --frames $(sed "${LINE_NUMBER}q;d" ${LIST}) -o "${OUTPUT_NAME}_${TARGET_VIDEO}_${LINE_NUMBER}.slp" -m models/centered_instance -m models/centroid --verbosity json
# --verbosity json --tracking.tracker simple --tracking.similarity iou --tracker.track_window 3 --tracking.target_instance_count 4 --tracking.clean_instance_count 4 --tracking.post_connect_single_breaks 1
#python merge_raw.py $TARGET_VIDEO

# --tracking.tracker simple --tracking.similarity centroid --tracker.track_window 2 --tracking.target_instance_count 4 --tracking.post_connect_single_breaks 1

#python ../sleap/sleap/nn/tracking.py ${TARGET_VIDEO}.slp --tracker simple --tracker simple --match greedy --track_window 3 --post_connect_single_breaks 1 --target_instance_count 4 --clean_instance_count 4 --output ${TARGET_VIDEO}.tracked.slp
# sleap-convert --format analysis "${OUTPUT_NAME}_${TARGET_VIDEO}_${LINE_NUMBER}.slp"
