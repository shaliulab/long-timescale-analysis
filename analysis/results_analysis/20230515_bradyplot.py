import multiprocessing
import os
import re

import cv2
import h5py
import imageio
import numpy as np
from natsort import natsorted

imageio.plugins.ffmpeg.download()

# Load the data
z_vals_file = "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/20230522-mmpy-lts-all-pchip5-headprobinterpy0xhead-medianwin5-gaussian-lombscargle-dynamicwinomega020-singleflysampledtracks/UMAP/20230523_minregions100_zVals_wShed_groups_prevregions.mat"
with h5py.File(z_vals_file, "r") as f:
    z_val_names_dset = f["zValNames"]
    references = [
        f[z_val_names_dset[dset_idx][0]]
        for dset_idx in range(z_val_names_dset.shape[0])
    ]
    z_val_names = ["".join(chr(i) for i in obj[:]) for obj in references]
    z_lens = [l[0] for l in f["zValLens"][:]]

wshedfile = h5py.File(z_vals_file, "r")
wregs = wshedfile["watershedRegions"][:].flatten()
ethogram = np.zeros((wregs.max() + 1, len(wregs)))
ethogram_split = np.split(wregs, np.cumsum(wshedfile["zValLens"][:].flatten())[:-1])
ethogram_dict = {k: v for k, v in zip(z_val_names, ethogram_split)}


def run_length_encoding(sequence):
    """
    Identify runs in the sequence
    """
    values = []
    starts = []
    lengths = []

    current_value = None
    current_start = None
    current_length = 0

    for i, value in enumerate(sequence):
        if current_value is None:
            current_value = value
            current_start = i
            current_length = 1
        elif current_value == value:
            current_length += 1
        else:
            values.append(current_value)
            starts.append(current_start)
            lengths.append(current_length)
            current_value = value
            current_start = i
            current_length = 1

    values.append(current_value)
    starts.append(current_start)
    lengths.append(current_length)

    return np.array(values), np.array(starts), np.array(lengths)


def get_video_file(key):
    """
    Returns the path to the video file corresponding to the given key.
    """
    # Extract the flid number and sample number from the key
    flid_number, sample_number = re.match(r"flid_(\d+)samp_(\d+)", key).groups()

    # Get the base directory
    base_dir = "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/data/brady/brady_sample_vids"

    # Get a list of all folders in the base directory
    folders = os.listdir(base_dir)

    # Filter folders that end with the flid number
    folders = [folder for folder in folders if folder.endswith(f"flid{flid_number}")]

    if not folders:
        raise ValueError(f"No folders found for flid number {flid_number}")

    # For simplicity, we assume there's only one such folder. If there can be more,
    # you might need additional logic to select the correct one.
    folder_path = os.path.join(base_dir, folders[0])

    # Get a list of all files in the folder
    filenames = os.listdir(folder_path)

    # Filter out any non-.mp4 files
    filenames = [fn for fn in filenames if fn.endswith(".mp4")]
    filenames = natsorted(filenames)

    # Get the filename for the specified sample
    video_file = filenames[int(sample_number) - 1]

    # Construct the full path to the video file
    video_file_path = os.path.join(folder_path, video_file)
    print(f"key: {key} maps to video file: {video_file_path}")
    return video_file_path


def get_track_file(key):
    """
    Returns the path to the track file corresponding to the given key.
    """
    file_prefix = key.split("-")[0]
    print(f"key: {key} maps to track file: {file_prefix}.h5")
    return f"/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/data/brady/single_sample_tracks/{file_prefix}.h5"


for key in ethogram_dict.keys():
    print(f"Processing key: {key}")
    values, starts, lengths = run_length_encoding(ethogram_dict[key])
    ethogram_dict[key] = {"starts": starts, "lengths": lengths, "values": values}


def extract_frames(video_file, frame_indices, track_file, crop_size=(128, 128)):
    frames = []

    with h5py.File(track_file, "r") as track_f:
        thorax_tracks = track_f["tracks"][:].T[
            :, 3, :
        ]  # index 3 corresponds to the thorax

    # Read the video using imageio
    vid = imageio.get_reader(video_file, "ffmpeg")

    for frame_idx in frame_indices:
        try:
            image = vid.get_data(frame_idx)

            # Convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Get thorax position and crop around it
            thorax_pos = thorax_tracks[frame_idx].astype(int)
            start_x = max(0, thorax_pos[0] - crop_size[0] // 2)
            start_y = max(0, thorax_pos[1] - crop_size[1] // 2)
            image = image[
                start_y : start_y + crop_size[1], start_x : start_x + crop_size[0]
            ]

            frames.append(image)
        except IndexError:
            print(f"Frame index {frame_idx} is out of range.")

    return frames


# Identifying runs for each behavior (key in the dictionary)


def process_behavior(
    key, data, output_folder, min_frame_length, max_total_frames, target_frames
):
    starts, lengths, values = data["starts"], data["lengths"], data["values"]
    # Sort all by lengths
    sort_idx = np.argsort(lengths)[::-1]
    starts = starts[sort_idx]
    lengths = lengths[sort_idx]
    values = values[sort_idx]

    unique_values = np.unique(values[~np.isnan(values)])
    nan_mask = np.isnan(values)
    behaviors = (
        np.concatenate((unique_values, [np.nan])) if np.any(nan_mask) else unique_values
    )

    video_file = get_video_file(key)
    track_file = get_track_file(key)
    print("Matched video and track files.")
    print(f"Video file: {video_file}")
    print(f"Track file: {track_file}")
    print(f"Key: {key}")

    for behavior in behaviors:
        total_frames_seen = 0
        videos_created = 0  # Counter for the number of videos created for the behavior
        print(f"Length of values: {len(values)}")

        if np.isnan(behavior):
            behavior_indices = np.where(np.isnan(values))[0]
        else:
            behavior_indices = np.where(values == behavior)[0]
        # np.random.shuffle(behavior_indices)

        for i in behavior_indices:
            if total_frames_seen >= target_frames or videos_created >= 10:
                break

            length = lengths[i]
            if length > min_frame_length:
                print(f"Processing key: {key}, behavior: {behavior}, run: {i}")
                print(f"Start: {starts[i]}, length: {length}, value: {values[i]}")

                # Extract the frame indices for this run
                frame_indices = range(starts[i], starts[i] + length)

                # Limit the total number of frames
                if i == 0 and length > max_total_frames:
                    total_frames = max_total_frames
                else:
                    total_frames = min(max_total_frames, length)

                # Adjust the number of frames to reach the target
                remaining_frames = target_frames - total_frames_seen
                total_frames = min(total_frames, remaining_frames)

                frame_indices = np.linspace(
                    starts[i], starts[i] + total_frames - 1, total_frames, dtype=int
                )

                # Extract and crop the frames
                frames = extract_frames(video_file, frame_indices, track_file)

                # Determine the size of the output video based on the size of the first frame
                frame_height, frame_width, _ = frames[0].shape
                output_size = (frame_width, frame_height)

                # Define the codec and create a VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                behavior_id = str(values[i]).replace("nan", "NaN")
                output_file = os.path.join(
                    output_folder,
                    f"behavior_{behavior_id}",
                    f"behavior{behavior_id}_start{starts[i]}_end{starts[i]+total_frames}_source{key}.mp4",
                )

                print(f"Saving video to: {output_file}")
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                video_writer = cv2.VideoWriter(output_file, fourcc, 100.0, output_size)

                # Write each frame to the video
                for frame in frames:
                    # OpenCV expects images in BGR format
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    video_writer.write(frame_bgr)

                video_writer.release()

                # Update the total frames seen and videos created
                total_frames_seen += total_frames
                videos_created += 1


def save_matching_videos_parallel(
    output_folder="20230522_awkde_noidle_bradies_25",
    min_frame_length=25,
    max_total_frames=1000,
    target_frames=1000,
):
    os.makedirs(output_folder, exist_ok=True)

    # Get the number of available cores or limit to 24
    num_cores = min(multiprocessing.cpu_count(), 8)

    pool = multiprocessing.Pool(processes=num_cores)
    results = []

    for key, data in ethogram_dict.items():
        result = pool.apply_async(
            process_behavior,
            args=(
                key,
                data,
                output_folder,
                min_frame_length,
                max_total_frames,
                target_frames,
            ),
        )
        results.append(result)

    pool.close()
    pool.join()

    # Check if any exceptions occurred during processing
    for result in results:
        result.get()  # This will raise an exception if one occurred


# Call the function


import numpy as np
import pandas as pd


def compute_behavior_fractions(ethogram_dict, max_behavior_code):
    """
    Computes the fraction of time spent in each behavior for each key in ethogram_dict.
    """
    behavior_fractions = {}

    for key, data in ethogram_dict.items():
        total_frames = np.sum(data["lengths"])

        behavior_counts = np.zeros(int(max_behavior_code) + 1, dtype=int)
        nan_count = 0
        for value, length in zip(data["values"], data["lengths"]):
            if np.isnan(value):
                nan_count += length
            else:
                behavior_counts[int(value)] += length

        behavior_fractions[key] = behavior_counts / total_frames
        behavior_fractions[key] = np.append(
            behavior_fractions[key], nan_count / total_frames
        )

    return behavior_fractions


def write_behavior_fractions_to_csv(behavior_fractions, filename):
    """
    Writes the behavior fractions data to a CSV file.
    """
    df = pd.DataFrame(behavior_fractions)
    df.to_csv(filename, index=True)


import numpy as np


def compute_overall_behavior_fractions(ethogram_dict, max_behavior_code):
    """
    Computes the fraction of time spent in each behavior across all keys in ethogram_dict.
    """
    total_frames_all_flies = 0
    overall_behavior_counts = np.zeros(int(max_behavior_code) + 1, dtype=int)
    nan_count = 0

    for key, data in ethogram_dict.items():
        total_frames = np.sum(data["lengths"])
        total_frames_all_flies += total_frames

        for value, length in zip(data["values"], data["lengths"]):
            if np.isnan(value):
                nan_count += length
            else:
                overall_behavior_counts[int(value)] += length

    overall_behavior_fractions = overall_behavior_counts / total_frames_all_flies
    overall_behavior_fractions = np.append(
        overall_behavior_fractions, nan_count / total_frames_all_flies
    )

    return overall_behavior_fractions


save_matching_videos_parallel()

max_behavior_code = max(max(data["values"]) for data in ethogram_dict.values())
behavior_fractions = compute_behavior_fractions(ethogram_dict, max_behavior_code)

write_behavior_fractions_to_csv(behavior_fractions, "behavior_fractions.csv")

overall_behavior_fractions = compute_overall_behavior_fractions(
    ethogram_dict, max_behavior_code
)

# Convert the fractions array to a pandas DataFrame with appropriate indices and column names
df_overall = pd.DataFrame(
    overall_behavior_fractions, columns=["Overall behavior fractions"]
)
df_overall.to_csv("overall_behavior_fractions.csv", index=True)

ethogram_dict.keys()

output_folder = "20230522_awkde_noidle_bradies_25"
min_frame_length = 25
max_total_frames = 1000
target_frames = 1000

for key, data in ethogram_dict.items():
    process_behavior(
        key,
        data,
        output_folder,
        min_frame_length,
        max_total_frames,
        target_frames,
    )
    break

process_behavior(
    key,
    data,
    output_folder,
    min_frame_length,
    max_total_frames,
    target_frames,
)
