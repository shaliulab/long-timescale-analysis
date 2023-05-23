import os
from moviepy.editor import VideoFileClip, clips_array
import moviepy.video.fx.all as vfx


def make_grid(videofiles, outputfile):
    # Load the videos
    clips = [VideoFileClip(videofile) for videofile in videofiles]

    # Get the max duration
    duration = max(clip.duration for clip in clips)

    # Loop shorter clips until they match the duration of the longest clip
    clips = [
        clip.fx(vfx.loop, duration=duration) if clip.duration < duration else clip
        for clip in clips
    ]

    # Arrange clips into a grid
    cols = 5  # desired number of columns
    rows = (len(clips) + cols - 1) // cols  # calculate number of rows based on columns
    grid = [
        clips[n * cols : (n + 1) * cols] for n in range(rows)
    ]  # split clips into sublists
    grid += [[]] * (
        cols - len(grid[-1])
    )  # pad last sublist with empty lists if necessary

    final_clip = clips_array(grid)

    # Write the result to a file
    final_clip.write_videofile(outputfile, codec="libx264")


import random


def make_grids(folder):
    for subfolder in os.listdir(folder):
        try:
            subfolder_path = os.path.join(folder, subfolder)
            if os.path.isdir(subfolder_path):
                print(f"Processing subfolder: {subfolder}")
                # Get the list of video files in the subfolder
                videofiles = [
                    os.path.join(subfolder_path, filename)
                    for filename in os.listdir(subfolder_path)
                    if filename.endswith(".mp4")
                ]

                # Select the 25 longest videos
                videofiles = sorted(
                    videofiles, key=lambda x: VideoFileClip(x).duration, reverse=True
                )[:25]

                # Make a grid for the videos in the subfolder
                outputfile = os.path.join(folder, f"{subfolder}_grid.mp4")
                make_grid(videofiles, outputfile)
        except Exception as e:
            print(f"Error processing subfolder: {subfolder}")
            print(e)


# Call the function
make_grids("bradies")

import random


from tqdm import tqdm


def make_grids(folder):
    output_path = os.path.join(folder, "grids")
    os.mkdir(output_path)
    for subfolder in tqdm(os.listdir(folder)):
        try:
            subfolder_path = os.path.join(folder, subfolder)
            if os.path.isdir(subfolder_path):
                print(f"Processing subfolder: {subfolder}")
                # Get the list of video files in the subfolder
                videofiles = [
                    os.path.join(subfolder_path, filename)
                    for filename in os.listdir(subfolder_path)
                    if filename.endswith(".mp4")
                ]

                # Select the 25 longest videos
                videofiles = sorted(
                    videofiles, key=lambda x: VideoFileClip(x).duration, reverse=True
                )[:25]

                # Make a grid for the videos in the subfolder
                outputfile = os.path.join(output_path, f"{subfolder}_grid.mp4")
                make_grid(videofiles, outputfile)
        except Exception as e:
            print(f"Error processing subfolder: {subfolder}")
            print(e)


# Call the function
make_grids("20230522_awkde_noidle_bradies_25")
