import glob
import h5py
import natsort

def list_files(base_path):
    filenames = glob.glob(base_path + "/*.h5")
    if len(filenames) == 0:
        filenames = glob.glob(base_path + "/*/*.h5")

    assert filenames, f"No files found in {base_path}"

    filenames = natsort.natsorted(filenames)
    return filenames

def load_node_names(file):
    with h5py.File(file, "r") as f:
        node_names=[n.decode() for n in f["node_names"][:]]

    return node_names