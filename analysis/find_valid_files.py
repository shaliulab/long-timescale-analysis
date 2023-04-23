import pandas as pd

working_files = pd.read_csv(
    "analysis/list_of_working_files.txt", names=["file"], index_col=None, header=None
)
working_file_lengths = pd.read_csv(
    "analysis/list_of_working_files_length.txt",
    names=["length"],
    index_col=None,
    header=None,
)
result = pd.concat([working_files, working_file_lengths], axis=1)
result = result[result.length == 32000]
result["file"].to_csv(
    "analysis/list_of_working_files_32000.txt", index=False, header=False
)
