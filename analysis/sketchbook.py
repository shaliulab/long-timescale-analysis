import pathlib

import pandas as pd

metadata = pd.read_csv(
    "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/data_index.csv"
)
metadata = metadata[metadata["Fly id"].notnull()]
metadata["date"] = pd.to_datetime(metadata["Date"], format="%m/%d/%Y").dt.strftime(
    "%Y%m%d"
)
metadata["within_arena_id"] = (metadata["Fly id"].astype(int) - 1) % 4
metadata["death_day"] = (metadata["Collapse (hours into video)"] // 24).astype(int)
metadata["death_frame_in_death_day"] = (
    (metadata["Collapse (hours into video)"] % 24) * 60 * 60 * 99.96
).astype(int)
metadata["cam_num"] = metadata["Camera"].str.replace("Camera ", "").astype(int)

metadata[
    [
        "date",
        "Fly id",
        "Collapse (hours into video)",
        "death_day",
        "death_frame_in_death_day",
    ]
]

filename = "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/20220312-lts-cam4_day4_24hourvars.h5"

# get day from filename
day = int(pathlib.Path(filename).name.split("_")[1].replace("day", ""))
cam_number = int(
    pathlib.Path(filename).name.split("_")[0].split("-")[2].replace("cam", "")
)
date = pathlib.Path(filename).name.split("-")[0]

matching_rows = metadata[
    (metadata["date"].astype(str) == date) & (metadata["cam_num"] == cam_number)
]

id = 0
matching_row = matching_rows[matching_rows["within_arena_id"] == id]
if day > matching_row["death_day"].values[0]:
    print("skipping!")
elif day == matching_row["death_day"].values[0]:
    cutoff = matching_row["death_frame_in_death_day"].values[0]
    print(f"Cutting off at {cutoff} for individual {id} in file {filename}!")
else:
    pass
